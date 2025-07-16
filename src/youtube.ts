import dotenv from "dotenv";
import { spawn } from "child_process";
import readline from "readline";
import axios from "axios";
import { parseStringPromise } from "xml2js";
import { Database } from "bun:sqlite";
import fs from "fs";
import path from "path";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { ChatAnthropic } from "@langchain/anthropic";
import { CohereEmbeddings } from "@langchain/cohere";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import inquirer from "inquirer";
import { gzipSync, gunzipSync } from "zlib";
import { Buffer } from "node:buffer";
import { $ } from "bun";

/* helper ───────────────────────────────*/
const gz = (txt: string) =>
  gzipSync(Buffer.from(txt, "utf8").toBase64()).toBase64();
const gunz = (buf: Buffer | string) =>
  typeof buf === "string"
    ? buf // already plain text
    : gunzipSync(buf.toBase64()).toString("utf8");

// ────────────────────────────────────────────────────────────────
// 1. Environment & util setup
// ────────────────────────────────────────────────────────────────

dotenv.config();

const YTDLP_BIN = process.env.YTDLP_BIN || "yt-dlp"; // path to yt-dlp binary
const TRANSCRIBE_BIN = process.env.TRANSCRIBE_BIN || "./src/transcribe";
const WHISPER_MODEL_PATH =
  process.env.WHISPER_MODEL_PATH || "./whisper.cpp/models/ggml-base.en.bin";

interface CaptionCue {
  start: number;
  end: number;
  text: string;
}

// ────────────────────────────────────────────────────────────────
// 2. Caption retrieval (uses yt-dlp directly – no Invidious needed)
// ────────────────────────────────────────────────────────────────

async function fetchCaptionsWithYtDlp(
  videoUrl: string,
  videoId: string
): Promise<string | null> {
  const tmpDir = path.join("./tmp");
  await fs.promises.mkdir(tmpDir, { recursive: true });

  // yt-dlp writes captions to `<id>.en.vtt` when using these flags
  const proc = spawn(YTDLP_BIN, [
    "--write-auto-sub",
    "--sub-lang",
    "en",
    "--sub-format",
    "vtt",
    "--skip-download",
    "-o",
    path.join(tmpDir, "%(id)s.%(ext)s"),
    videoUrl,
  ]);

  let stderr = "";
  proc.stderr.on("data", (d) => (stderr += d.toString()));

  return new Promise((resolve) => {
    proc.on("close", async (code) => {
      if (code !== 0) {
        console.warn(`yt-dlp exited ${code}:\n${stderr}`);
        return resolve(null);
      }
      const vttPath = path.join(tmpDir, `${videoId}.en.vtt`);
      try {
        const raw = await fs.promises.readFile(vttPath, "utf8");
        // simple VTT → plain-text converter
        const transcript = raw
          .split(/\r?\n/)
          .filter(
            (line) =>
              line &&
              !/^WEBVTT/.test(line) &&
              !/^\d+$/.test(line) && // cue index lines
              !/-->/.test(line) // timestamp lines
          )
          .join(" ")
          .replace(/\s+/g, " ")
          .trim();
        return resolve(transcript || null);
      } catch (err) {
        console.warn("Failed to read VTT:", err);
        return resolve(null);
      }
    });
  });
}

// ────────────────────────────────────────────────────────────────
// 3. YouTube helpers
// ────────────────────────────────────────────────────────────────

async function fetchYoutubeTitleOEmbed(videoId: string): Promise<string> {
  const url = `https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v=${videoId}&format=json`;
  const { data } = await axios.get<{ title: string }>(url);
  return data.title;
}

const getVideoId = (url: string) => {
  const match = url.match(/(?:v=|\/)([A-Za-z0-9_-]{11})/);
  return match?.[1] ?? "";
};

// ────────────────────────────────────────────────────────────────
// 4. Whisper fallback (same as before)
// ────────────────────────────────────────────────────────────────

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

// ────────────────────────────────────────────────────────────────
// 5. LLM & embedding factory helpers (unchanged)
// ────────────────────────────────────────────────────────────────

const defaultModel = "gemini-2.5-pro";
const PROVIDERS = {
  LLM: process.env.LLM_PROVIDER?.toLowerCase() || "google-genai",
  SUMMARY_MODEL: process.env.SUMMARY_MODEL || defaultModel,
  QA_MODEL: process.env.QA_MODEL || defaultModel,
  EMBEDDING: process.env.EMBEDDING_PROVIDER?.toLowerCase() || "openai",
  EMBEDDING_MODEL: process.env.EMBEDDING_MODEL || "text-embedding-ada-002",
};

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

const summaryModel = createChatModel(PROVIDERS.LLM, PROVIDERS.SUMMARY_MODEL);
const qaModel = createChatModel(PROVIDERS.LLM, PROVIDERS.QA_MODEL);
const embeddingModel = createEmbeddingModel(
  PROVIDERS.EMBEDDING,
  PROVIDERS.EMBEDDING_MODEL
);

// ────────────────────────────────────────────────────────────────
// 6. Database setup (unchanged)
// ────────────────────────────────────────────────────────────────

async function initDb(): Promise<Database> {
  const db = new Database("transcripts.sqlite");
  db.exec(`
    CREATE TABLE IF NOT EXISTS content (
      content_id   TEXT PRIMARY KEY,
      content_type TEXT NOT NULL,
      title        TEXT,
      author       TEXT,
      audio_url    TEXT,
      transcript   BLOB,
      summary      TEXT,
      created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    /* new table */
    CREATE TABLE IF NOT EXISTS qa (
      id          INTEGER PRIMARY KEY AUTOINCREMENT,
      content_id  TEXT NOT NULL,
      question    TEXT NOT NULL,
      answer      TEXT NOT NULL,
      created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (content_id) REFERENCES content(content_id)
    );
  `);
  /* turn on incremental auto-vacuum (first run triggers a VACUUM) */
  db.exec(`
    PRAGMA auto_vacuum = INCREMENTAL;
    PRAGMA journal_mode = WAL;
    VACUUM;
  `);

  return db;
}

/* helper to persist Q&A */
function storeQA(
  db: Database,
  contentId: string,
  question: string,
  answer: string
) {
  db.run(`INSERT INTO qa (content_id, question, answer) VALUES (?, ?, ?)`, [
    contentId,
    question,
    answer,
  ]);
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

// ────────────────────────────────────────────────────────────────
// 7. Transcript acquisition (now uses yt-dlp first)
// ────────────────────────────────────────────────────────────────

async function getOrCreateTranscript(
  db: Database,
  url: string
): Promise<ContentData> {
  const videoId = getVideoId(url);
  if (videoId) {
    const existing = db
      .query(
        "SELECT * FROM content WHERE content_id = ? AND content_type = 'youtube'"
      )
      .get(videoId) as ContentData;
    if (existing) {
      existing.transcript = gunz(existing.transcript); // ← inflate
      return existing as ContentData;
    }

    // 1️⃣ Try yt-dlp captions
    let transcript = await fetchCaptionsWithYtDlp(url, videoId);

    // 2️⃣ Fallback to Whisper transcription
    if (!transcript) transcript = await fetchOrTranscribe(url);

    const title = await fetchYoutubeTitleOEmbed(videoId);
    const summary = await summarize(transcript);
    db.run(
      `INSERT INTO content
   (content_id, content_type, title, author, transcript, summary)
   VALUES (?, ?, ?, ?, ?, ?)`,
      [videoId, "youtube", title, "", gz(transcript), summary] // ← gzip here
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

  // ────────────────────────────────────────────────────────────
  //  Apple Podcasts path (unchanged)
  // ────────────────────────────────────────────────────────────

  const parsePodcastUrl = async (
    url: string
  ): Promise<{
    podcastId: string;
    episodeId: string;
    title: string;
    author: string;
    audioUrl: string;
  } | null> => {
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

  // Try RSS transcript first
  const transcriptFromRSS = async (
    podcastId: string,
    episodeId: string
  ): Promise<string | null> => {
    try {
      const lookupResponse = await axios.get(
        `https://itunes.apple.com/lookup?id=${podcastId}&entity=podcast`
      );
      if (!lookupResponse.data?.results?.[0]?.feedUrl) return null;

      const feedUrl = lookupResponse.data.results[0].feedUrl;
      const feedResponse = await axios.get(feedUrl);
      const parsedFeed = await parseStringPromise(feedResponse.data);

      const channel = parsedFeed.rss.channel[0];
      const episode = channel.item?.find((item: any) => {
        const guid = item.guid?.[0]?._;
        return guid && guid.includes(episodeId);
      });

      if (episode && episode["podcast:transcript"]) {
        const transcriptUrl = episode["podcast:transcript"][0].$.url;
        if (transcriptUrl) {
          const transcriptResponse = await axios.get(transcriptUrl);
          return transcriptResponse.data;
        }
      }
      return null;
    } catch {
      return null;
    }
  };

  let transcript = await transcriptFromRSS(info.podcastId, info.episodeId);
  if (!transcript) transcript = await fetchOrTranscribe(info.audioUrl);

  const summary = await summarize(transcript);
  db.run(
    `INSERT INTO content
   (content_id, content_type, title, author, audio_url, transcript, summary)
   VALUES (?, ?, ?, ?, ?, ?, ?)`,
    [
      contentId,
      "apple_podcast",
      info.title,
      info.author,
      info.audioUrl,
      gz(transcript), // gzip
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

// ────────────────────────────────────────────────────────────────
// 8. Summarisation & QA chains (unchanged)
// ────────────────────────────────────────────────────────────────

async function summarize(transcript: string): Promise<string> {
  console.log(
    `summarizing ${
      transcript.split(" ").length
    } word transcript with ${defaultModel}`
  );
  const prompt = ChatPromptTemplate.fromTemplate(
    `You are an expert analyst. Read the following transcript and produce a clear, concise summary that:\n\n• Highlights the 3–5 most important points or themes.\n• Uses bullet points, each no more than two sentences.\n• Identifies any key takeaways or action items at the end, under an “Action Items” heading.\n\nDo not include any other text or explanations.\n\nTranscript:\n{transcript}`
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

// ────────────────────────────────────────────────────────────────
// 9. CLI interaction loop (unchanged except log tweaks)
// ────────────────────────────────────────────────────────────────

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
const ask = (q: string) =>
  new Promise<string>((res) => rl.question(q, (ans) => res(ans.trim())));

async function main() {
  const db = await initDb();
  try {
    /* ① fetch saved sessions, newest first */
    const sessions = db
      .query(
        `SELECT content_id, title, content_type, DATE(created_at) AS created
         FROM content ORDER BY created_at DESC`
      )
      .all() as {
      content_id: string;
      title: string;
      content_type: string;
      created: string;
    }[];

    /* ② build choices list */
    const menuChoices = [
      { name: "Start new session", value: "__new" },
      ...sessions.map((s) => ({
        name: `${s.title}  – ${s.content_type.toUpperCase()}  [${s.created}]`,
        value: s.content_id,
      })),
    ];

    const { pick } = await inquirer.prompt<{ pick: string }>({
      name: "pick",
      type: "list",
      message: "Select a saved transcript or start a new one:",
      choices: menuChoices,
      default: 0, // so <Enter> = new session
      pageSize: 12,
    });

    let data: ContentData;

    if (pick === "__new") {
      const { url } = await inquirer.prompt<{ url: string }>({
        name: "url",
        type: "input",
        message: "Enter YouTube or Apple Podcast URL:",
      });
      data = await getOrCreateTranscript(db, url.trim());
    } else {
      /* restore by content_id directly */
      data = db
        .query("SELECT * FROM content WHERE content_id = ?")
        .get(pick) as ContentData;
      if (!data) throw new Error("Session not found in DB");
      data.transcript = gunz(data.transcript);
    }

    console.log(`\nTitle: ${data.title}`);
    if (data.author) console.log(`Author: ${data.author}`);
    console.log(`\nSummary: ${data.summary}\n`);

    /* QA selector loop (same as before) */
    while (true) {
      const qaRows = db
        .query(
          "SELECT id, question, answer FROM qa WHERE content_id = ? ORDER BY created_at DESC"
        )
        .all(data.content_id) as {
        id: number;
        question: string;
        answer: string;
      }[];

      const { sel } = await inquirer.prompt<{ sel: string | number }>({
        name: "sel",
        type: "list",
        message: "Questions:",
        choices: [
          { name: "🆕  New question", value: "__new" },
          ...qaRows.map((r) => ({ name: r.question, value: r.id })),
          { name: "Exit", value: "__exit" },
        ],
        pageSize: 10,
        default: 0,
      });

      if (sel === "__exit") break;

      if (sel === "__new") {
        const { newQ } = await inquirer.prompt<{ newQ: string }>({
          name: "newQ",
          type: "input",
          message: "question:",
        });
        if (!newQ.trim()) continue;

        const a = await answer(data.transcript, newQ);
        console.log(`\nAnswer: ${a}\n`);
        storeQA(db, data.content_id, newQ, a);
      } else {
        const qa = qaRows.find((r) => r.id === sel)!;
        console.log(`\nAnswer: ${qa.answer}\n`);
      }
    }
  } catch (err: any) {
    console.error("Error:", err.message);
  } finally {
    await $`rm -rf transcripts.sqlite-wal transcripts.sqlite-shm`;
    await new Promise((res) => setTimeout(res, 1000)); // give it a moment to finish
    db.close();
  }
}

main();
