import dotenv from "dotenv";
import { spawn } from "child_process";
import readline from "readline";
import axios from "axios";
import { parseStringPromise } from "xml2js";
import { Database } from "bun:sqlite";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { ChatAnthropic } from "@langchain/anthropic";
import { CohereEmbeddings } from "@langchain/cohere";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate } from "@langchain/core/prompts";

// Load environment
dotenv.config();

interface CaptionCue {
  label: string;
  url: string;
  languageCode: string;
}

async function fetchYoutubeTitleOEmbed(videoId: string): Promise<string> {
  const url = `https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v=${videoId}&format=json`;
  const { data } = await axios.get<{ title: string }>(url);
  return data.title;
}

async function getYouTubeTranscript({
  videoId,
}: {
  videoId: string;
}): Promise<any> {
  const instance = "https://yewtu.be";
  const url = `${instance}/api/v1/captions/${videoId}?format=json3`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} from ${url}`);
  const cues: { captions: CaptionCue[] } = await res.json();
  for (const cue of cues.captions) {
    if (cue.languageCode === "en") {
      const transcript = await fetch(instance + cue.url).then((r) => r.text());
      return transcript;
    }
  }
}

// Paths for our new C++ transcriber
const TRANSCRIBE_BIN = process.env.TRANSCRIBE_BIN || "./src/transcribe";
const WHISPER_MODEL_PATH =
  process.env.WHISPER_MODEL_PATH || "./whisper.cpp/models/ggml-base.en.bin";

// --- CLI Setup ---
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
const ask = (q: string) =>
  new Promise<string>((res) => rl.question(q, (ans) => res(ans.trim())));

// --- Configurable providers & models ---
const PROVIDERS = {
  LLM: process.env.LLM_PROVIDER?.toLowerCase() || "google-genai",
  SUMMARY_MODEL: process.env.SUMMARY_MODEL || "gemini-2.5-pro",
  QA_MODEL: process.env.QA_MODEL || "gemini-2.5-pro",
  EMBEDDING: process.env.EMBEDDING_PROVIDER?.toLowerCase() || "openai",
  EMBEDDING_MODEL: process.env.EMBEDDING_MODEL || "text-embedding-ada-002",
};

// --- Factory functions ---
function createChatModel(provider: string, model: string) {
  switch (provider) {
    case "openai":
      return new ChatOpenAI({
        openAIApiKey: process.env.OPENAI_API_KEY,
        modelName: model,
      });
    case "anthropic":
      return new ChatAnthropic({
        anthropicApiKey: process.env.ANTHROPIC_API_KEY,
        modelName: model,
      });
    case "google-genai":
      return new ChatGoogleGenerativeAI({
        apiKey: process.env.GEMINI_API_KEY,
        model,
      });
    default:
      throw new Error(`Unsupported LLM provider: ${provider}`);
  }
}

function createEmbeddingModel(provider: string, model: string) {
  switch (provider) {
    case "openai":
      return new OpenAIEmbeddings({
        openAIApiKey: process.env.OPENAI_API_KEY,
        modelName: model,
      });
    case "cohere":
      return new CohereEmbeddings({
        apiKey: process.env.COHERE_API_KEY,
        model,
      });
    default:
      throw new Error(`Unsupported embedding provider: ${provider}`);
  }
}

// --- Initialize models ---
const summaryModel = createChatModel(PROVIDERS.LLM, PROVIDERS.SUMMARY_MODEL);
const qaModel = createChatModel(PROVIDERS.LLM, PROVIDERS.QA_MODEL);
const embeddingModel = createEmbeddingModel(
  PROVIDERS.EMBEDDING,
  PROVIDERS.EMBEDDING_MODEL
);

// --- Database setup ---
async function initDb(): Promise<Database> {
  const db = new Database("transcripts.sqlite");
  db.exec(`
    CREATE TABLE IF NOT EXISTS content (
      content_id TEXT PRIMARY KEY,
      content_type TEXT NOT NULL,
      title TEXT,
      author TEXT,
      audio_url TEXT,
      transcript TEXT,
      summary TEXT,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
  `);
  return db;
}

interface ContentData {
  content_id: string;
  content_type: "youtube" | "apple_podcast";
  title: string;
  author: string;
  audio_url?: string;
  transcript: string;
  summary: string;
}

// --- Utility functions ---
const getVideoId = (url: string) => {
  const match = url.match(/(?:v=|\/)([A-Za-z0-9_-]{11})/);
  return match?.[1] ?? "";
};

async function parsePodcastUrl(url: string): Promise<{
  podcastId: string;
  episodeId: string;
  title: string;
  author: string;
  audioUrl: string;
} | null> {
  const podcastMatch = url.match(/\/id(\d+)/);
  if (!podcastMatch) return null;
  const podcastId = podcastMatch[1];
  const episodeId = url.match(/[?&]i=(\d+)/)?.[1] ?? "";

  const lookup = await axios.get(
    `https://itunes.apple.com/lookup?id=${podcastId}&entity=podcast`
  );
  const info = lookup.data.results?.[0];
  if (!info?.feedUrl) return null;

  const feed = await axios.get(info.feedUrl);
  const channel = (await parseStringPromise(feed.data)).rss.channel[0];
  const author = channel.author?.[0] ?? channel["itunes:author"]?.[0] ?? "";

  const findEpisode = (items: any[]) =>
    items.find((it) => it.guid?.[0]?._?.includes(episodeId)) || items[0];

  const episode = findEpisode(channel.item || []);
  const title = episode.title[0];
  const enclosure = episode.enclosure?.[0]?.$?.url || "";

  return {
    podcastId,
    episodeId: episodeId || title,
    title,
    author,
    audioUrl: enclosure,
  };
}

// --- UPDATED: spawn the C++ transcriber instead of Python ---
async function fetchOrTranscribe(audioUrl: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const proc = spawn(TRANSCRIBE_BIN, [audioUrl, WHISPER_MODEL_PATH]);
    let out = "";
    proc.stdout.on("data", (d) => (out += d.toString()));
    proc.stderr.on("data", (d) => console.error(d.toString()));
    proc.on("close", (code) =>
      code === 0
        ? resolve(out.trim())
        : reject(new Error(`Transcriber exited ${code}`))
    );
  });
}

async function summarize(transcript: string): Promise<string> {
  console.log("summarizing transcript");
  const prompt = ChatPromptTemplate.fromTemplate(
    `You are an expert analyst. Read the following transcript and produce a clear, concise summary that:

    • Highlights the 3–5 most important points or themes.  
    • Uses bullet points, each no more than two sentences.  
    • Identifies any key takeaways or action items at the end, under an “Action Items” heading.

    Do not include any other text or explanations.

    Transcript:
    {transcript}`
  );
  const chain = prompt.pipe(summaryModel);
  const res = await chain.invoke({ transcript });
  return res.content as string;
}

async function answer(transcript: string, question: string): Promise<string> {
  const prompt = ChatPromptTemplate.fromTemplate(
    `Answer based on the transcript:\n{transcript}\n\nQuestion:\n{question}`
  );
  const chain = prompt.pipe(qaModel);
  const res = await chain.invoke({ transcript, question });
  return res.content as string;
}

function cosineSimilarity(a: number[], b: number[]): number {
  const dot = a.reduce((s, v, i) => s + v * b[i], 0);
  const magA = Math.hypot(...a);
  const magB = Math.hypot(...b);
  return magA && magB ? dot / (magA * magB) : 0;
}

// --- Core workflows ---
async function getOrCreateTranscript(
  db: Database,
  url: string
): Promise<ContentData> {
  const videoId = getVideoId(url);
  console.log("videoId", videoId);
  if (videoId) {
    const existing = db
      .query(
        "SELECT * FROM content WHERE content_id = ? AND content_type = 'youtube'"
      )
      .get(videoId) as ContentData;
    if (existing) return existing;

    let transcript: string;
    try {
      transcript = await getYouTubeTranscript({ videoId });
    } catch (err: any) {
      console.log("fetching transcript failed", err);
      transcript = await fetchOrTranscribe(url);
    }

    const title = await fetchYoutubeTitleOEmbed(videoId);
    const summary = await summarize(transcript);
    db.run(
      "INSERT INTO content (content_id, content_type, title, author, transcript, summary) VALUES (?, ?, ?, ?, ?, ?)",
      [videoId, "youtube", title, "", transcript, summary]
    );
    return {
      content_id: videoId,
      content_type: "youtube",
      title,
      author: "",
      transcript,
      summary,
    };
  }

  const getSubtitlesFromRSS = async (podcastId: string, episodeId: string) => {
    try {
      // First get the feed URL
      const lookupResponse = await axios.get(
        `https://itunes.apple.com/lookup?id=${podcastId}&entity=podcast`
      );
      if (!lookupResponse.data?.results?.[0]?.feedUrl) {
        return null;
      }

      const feedUrl = lookupResponse.data.results[0].feedUrl;
      const feedResponse = await axios.get(feedUrl);
      const parsedFeed = await parseStringPromise(feedResponse.data);

      const channel = parsedFeed.rss.channel[0];
      const episode = channel.item?.find((item: any) => {
        const guid = item.guid?.[0]?._;
        return guid && guid.includes(episodeId);
      });

      if (episode && episode["podcast:transcript"]) {
        // If the podcast has transcript metadata, try to fetch it
        const transcriptUrl = episode["podcast:transcript"][0].$.url;
        if (transcriptUrl) {
          const transcriptResponse = await axios.get(transcriptUrl);
          return transcriptResponse.data;
        }
      }

      return null;
    } catch (error) {
      console.error("Error fetching subtitles from RSS:", error);
      return null;
    }
  };

  const info = await parsePodcastUrl(url);
  if (!info) throw new Error("Invalid podcast URL");

  const contentId = `podcast_${info.podcastId}_episode_${info.episodeId}`;
  const existing = db
    .query(
      "SELECT * FROM content WHERE content_id = ? AND content_type = 'apple_podcast'"
    )
    .get(contentId) as ContentData;
  if (existing) return existing;

  let transcript = await getSubtitlesFromRSS(info.podcastId, info.episodeId);
  if (!transcript) transcript = await fetchOrTranscribe(info.audioUrl);

  const summary = await summarize(transcript);
  db.run(
    "INSERT INTO content (content_id, content_type, title, author, audio_url, transcript, summary) VALUES (?, ?, ?, ?, ?, ?, ?)",
    [
      contentId,
      "apple_podcast",
      info.title,
      info.author,
      info.audioUrl,
      transcript,
      summary,
    ]
  );
  return {
    content_id: contentId,
    content_type: "apple_podcast",
    ...info,
    transcript,
    summary,
  };
}

async function restoreSession(
  db: Database,
  narrative: string
): Promise<ContentData> {
  const rows = db.query("SELECT * FROM content").all() as ContentData[];
  if (rows.length === 0) throw new Error("No sessions saved");

  const navEmb = await embeddingModel.embedQuery(narrative);
  const sims = await Promise.all(
    rows.map((r) => embeddingModel.embedQuery(r.summary))
  );

  let best = { idx: -1, score: -Infinity };
  sims.forEach((emb, i) => {
    const sim = cosineSimilarity(navEmb, emb);
    if (sim > best.score) best = { idx: i, score: sim };
  });

  if (best.score < 0.8) throw new Error("No close match found");
  return rows[best.idx];
}

// --- Interaction loops ---
async function main() {
  const db = await initDb();

  try {
    const choice = await ask('Enter "1" for new session or "2" to restore: ');
    let data: ContentData;

    if (choice === "1") {
      const url = await ask("Enter YouTube or Apple Podcast URL: ");
      data = await getOrCreateTranscript(db, url);
    } else {
      const narrative = await ask("Enter narrative to restore: ");
      data = await restoreSession(db, narrative);
    }

    console.log(`\nTitle: ${data.title}`);
    if (data.author) console.log(`Author: ${data.author}`);
    console.log(`\nSummary: ${data.summary}\n`);

    // Q&A loop
    while (true) {
      const q = await ask('Question (or type "exit"): ');
      if (q.toLowerCase() === "exit") break;
      console.log("\nAnswer:", await answer(data.transcript, q), "\n");
    }
  } catch (err: any) {
    console.error("Error:", err.message);
  } finally {
    await db.close();
    rl.close();
  }
}

main();
