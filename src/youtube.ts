import dotenv from "dotenv";
import OpenAI from "openai";
import { Database } from "bun:sqlite";
import readline from "readline";
import { spawn } from "child_process";
import axios from "axios";
import { parseStringPromise } from "xml2js";
import { fetchTranscript } from "./utils/youtube-transcript-fetcher"; // Keep for YouTube fallback

dotenv.config();

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

// Configurable models via environment variables
const summaryModel = process.env.SUMMARY_MODEL || "o3-mini";
const qaModel = process.env.QA_MODEL || "o3-mini";
const embeddingModel = "text-embedding-ada-002";

interface ContentData {
  content_id: string;
  content_type: "youtube" | "apple_podcast";
  title: string;
  author: string;
  audio_url?: string;
  transcript: string;
  summary: string;
}

/** Initialize SQLite database */
const initDb = async (): Promise<Database> => {
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
};

/** Extract video ID from YouTube URL */
const getVideoId = (url: string): string => {
  const match = url.match(/(?:v=|\/)([a-zA-Z0-9_-]{11})/);
  return match ? match[1] : "";
};

/** Parse Apple Podcast URL and extract podcast ID and episode ID */
const parsePodcastUrl = async (
  url: string
): Promise<{
  podcastId: string;
  episodeId: string;
  title: string;
  author: string;
  audioUrl: string;
} | null> => {
  // Handle Apple Podcast URL formats
  // Example: https://podcasts.apple.com/us/podcast/podcast-name/id1234567890?i=1000000000001
  try {
    // Extract podcast ID
    const podcastIdMatch = url.match(/\/id(\d+)/);
    if (!podcastIdMatch) return null;
    const podcastId = podcastIdMatch[1];

    // Extract episode ID if present in the URL
    let episodeId = "";
    const episodeIdMatch = url.match(/[?&]i=(\d+)/);
    if (episodeIdMatch) {
      episodeId = episodeIdMatch[1];
    }

    // If we have both IDs, we can proceed
    if (podcastId) {
      // Fetch the podcast RSS feed
      const rssUrl = `https://itunes.apple.com/lookup?id=${podcastId}&entity=podcast`;
      const response = await axios.get(rssUrl);

      if (
        response.data &&
        response.data.results &&
        response.data.results.length > 0
      ) {
        const podcastInfo = response.data.results[0];
        const feedUrl = podcastInfo.feedUrl;

        if (feedUrl) {
          // Fetch and parse the RSS feed
          const feedResponse = await axios.get(feedUrl);
          const parsedFeed = await parseStringPromise(feedResponse.data);

          const channel = parsedFeed.rss.channel[0];
          const author =
            channel.author?.[0] || channel["itunes:author"]?.[0] || "";

          // If we have a specific episode ID, find that episode
          if (episodeId) {
            const episode = channel.item?.find((item: any) => {
              const guid = item.guid?.[0]?._;
              return guid && guid.includes(episodeId);
            });

            if (episode) {
              const title = episode.title[0];
              const audioUrl = episode.enclosure?.[0]?.$?.url || "";
              return {
                podcastId,
                episodeId,
                title,
                author,
                audioUrl,
              };
            }
          } else if (channel.item && channel.item.length > 0) {
            // If no specific episode, use the latest one
            const latestEpisode = channel.item[0];
            const title = latestEpisode.title[0];
            const guidParts = latestEpisode.guid[0]._.split("/");
            episodeId = guidParts[guidParts.length - 1];
            const audioUrl = latestEpisode.enclosure?.[0]?.$?.url || "";

            return {
              podcastId,
              episodeId,
              title,
              author,
              audioUrl,
            };
          }
        }
      }
    }
  } catch (error) {
    console.error("Error parsing podcast URL:", error);
  }

  return null;
};

/** Summarize a transcript using OpenAI */
const summarizeTranscript = async (transcript: string): Promise<string> => {
  const response = await openai.chat.completions.create({
    model: summaryModel,
    messages: [
      { role: "user", content: `Summarize this transcript:\n\n${transcript}` },
    ],
    temperature: 1,
  });
  return response.choices[0].message.content || "";
};

/** Transcribe audio using WhisperX via Python script */
const transcribeWithWhisperX = async (audioUrl: string): Promise<string> => {
  return new Promise((resolve, reject) => {
    const pythonInterpreter = process.env.PYTHON_INTERPRETER || "python";
    const pythonProcess = spawn(pythonInterpreter, [
      "src/transcribe.py",
      audioUrl,
    ]);
    let transcription = "";

    pythonProcess.stdout.on("data", (data) => {
      const output = data.toString().trim();
      if (output.startsWith("Transcribing chunk")) {
        console.log(output);
      } else {
        transcription += output + " ";
      }
    });

    pythonProcess.stderr.on("data", (data) =>
      console.error(`WhisperX error: ${data}`)
    );
    pythonProcess.on("close", (code) => {
      code === 0
        ? resolve(transcription.trim())
        : reject(new Error(`WhisperX exited with code ${code}`));
    });
  });
};

/** Get subtitles from podcast RSS if available */
const getSubtitlesFromRSS = async (
  podcastId: string,
  episodeId: string
): Promise<string | null> => {
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

/** Fetch or create transcript for content */
const getOrCreateTranscript = async (
  db: Database,
  contentUrl: string
): Promise<ContentData> => {
  // Try to parse as YouTube URL first
  const videoId = getVideoId(contentUrl);

  if (videoId) {
    // This is a YouTube URL
    const existing = db
      .query(
        "SELECT * FROM content WHERE content_id = $contentId AND content_type = 'youtube'"
      )
      .get({ $contentId: videoId }) as ContentData | null;

    if (existing) return existing;

    let transcript: string | undefined = undefined;
    try {
      const transcriptList = await fetchTranscript(videoId);
      transcript = transcriptList.map((entry) => entry.text).join(" ");
    } catch (error) {
      if (
        error instanceof Error &&
        error.message.includes("Transcript is disabled")
      ) {
        console.log("Transcript disabled. Falling back to WhisperX...");
        transcript = await transcribeWithWhisperX(contentUrl);
      } else {
        throw error;
      }
    }

    const summary = await summarizeTranscript(transcript);
    db.run(
      "INSERT INTO content (content_id, content_type, title, author, transcript, summary) VALUES (?, ?, ?, ?, ?, ?)",
      [videoId, "youtube", "YouTube Video", "", transcript, summary]
    );

    return {
      content_id: videoId,
      content_type: "youtube",
      title: "YouTube Video",
      author: "",
      transcript,
      summary,
    };
  } else {
    // Try parsing as Apple Podcast URL
    const podcastInfo = await parsePodcastUrl(contentUrl);
    if (!podcastInfo) {
      throw new Error("Invalid URL. Could not extract content information.");
    }

    const contentId = `podcast_${podcastInfo.podcastId}_episode_${podcastInfo.episodeId}`;

    // Check if we already have this podcast episode
    const existing = db
      .query(
        "SELECT * FROM content WHERE content_id = $contentId AND content_type = 'apple_podcast'"
      )
      .get({ $contentId: contentId }) as ContentData | null;

    if (existing) return existing;

    // Try to get transcript from the RSS feed first
    let transcript = await getSubtitlesFromRSS(
      podcastInfo.podcastId,
      podcastInfo.episodeId
    );

    // If no transcript is available in RSS, use WhisperX to transcribe the audio
    if (!transcript) {
      console.log("No transcript found in RSS. Falling back to WhisperX...");
      if (!podcastInfo.audioUrl) {
        throw new Error("No audio URL found for this podcast episode.");
      }
      transcript = await transcribeWithWhisperX(podcastInfo.audioUrl);
    }

    const summary = await summarizeTranscript(transcript);

    db.run(
      "INSERT INTO content (content_id, content_type, title, author, audio_url, transcript, summary) VALUES (?, ?, ?, ?, ?, ?, ?)",
      [
        contentId,
        "apple_podcast",
        podcastInfo.title,
        podcastInfo.author,
        podcastInfo.audioUrl,
        transcript,
        summary,
      ]
    );

    return {
      content_id: contentId,
      content_type: "apple_podcast",
      title: podcastInfo.title,
      author: podcastInfo.author,
      audio_url: podcastInfo.audioUrl,
      transcript,
      summary,
    };
  }
};

/** Answer a question based on the transcript */
const answerQuestion = async (
  transcript: string,
  question: string
): Promise<string> => {
  const response = await openai.chat.completions.create({
    model: qaModel,
    messages: [
      {
        role: "user",
        content: `Answer based on the transcript:\n${transcript}\n\nQuestion:\n${question}`,
      },
    ],
  });
  return response.choices[0].message.content || "";
};

/** Calculate cosine similarity between two vectors */
const cosineSimilarity = (vecA: number[], vecB: number[]): number => {
  const dotProduct = vecA.reduce((sum, a, idx) => sum + a * vecB[idx], 0);
  const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  return magnitudeA === 0 || magnitudeB === 0
    ? 0
    : dotProduct / (magnitudeA * magnitudeB);
};

/** Restore a session based on narrative similarity */
const restoreSession = async (
  db: Database,
  narrative: string
): Promise<ContentData> => {
  const rows = db
    .query(
      "SELECT content_id, content_type, title, author, audio_url, transcript, summary FROM content"
    )
    .all() as ContentData[];
  if (!rows.length) throw new Error("No saved sessions available.");

  const inputEmbedding = (
    await openai.embeddings.create({
      model: embeddingModel,
      input: narrative,
    })
  ).data[0].embedding;

  const summaries = rows.map((row) => row.summary);
  const summariesEmbeddings = (
    await openai.embeddings.create({
      model: embeddingModel,
      input: summaries,
    })
  ).data.map((item) => item.embedding);

  let bestScore = -Infinity;
  let bestIndex = -1;
  for (let i = 0; i < summariesEmbeddings.length; i++) {
    const similarity = cosineSimilarity(inputEmbedding, summariesEmbeddings[i]);
    if (similarity > bestScore) {
      bestScore = similarity;
      bestIndex = i;
    }
  }

  const SIMILARITY_THRESHOLD = 0.8;
  if (bestScore < SIMILARITY_THRESHOLD) {
    throw new Error("No matching session found with sufficient similarity.");
  }

  return rows[bestIndex];
};

/** Prompt user for input */
const askQuestion = (query: string): Promise<string> => {
  return new Promise((resolve) =>
    rl.question(query, (answer) => resolve(answer.trim()))
  );
};

/** Get session type from user */
const getSessionType = async (): Promise<string> => {
  while (true) {
    const input = await askQuestion(
      'Enter "1" for new session or "2" to restore a session: '
    );
    if (input === "1" || input === "2") return input;
    console.log('Invalid input. Please enter "1" or "2".');
  }
};

/** Get a valid content URL from user */
const getValidContentUrl = async (): Promise<string> => {
  while (true) {
    const url = await askQuestion(
      "Please enter the YouTube or Apple Podcast URL: "
    );
    if (getVideoId(url) || url.includes("podcasts.apple.com")) return url;
    console.log("Invalid URL. Please enter a YouTube or Apple Podcast URL.");
  }
};

/** Get narrative for session restoration */
const getNarrative = async (): Promise<string> => {
  return askQuestion("Enter text to restore session: ");
};

/** Handle question-answering loop */
const handleQuestions = async (transcript: string) => {
  while (true) {
    const question = await askQuestion(
      'Question about transcript (or "exit"): '
    );
    if (question.toLowerCase() === "exit") break;
    const answer = await answerQuestion(transcript, question);
    console.log("\nAnswer:", answer);
  }
};

/** Main execution function */
const main = async () => {
  const db = await initDb();
  try {
    const sessionType = await getSessionType();
    let contentData: ContentData;
    if (sessionType === "1") {
      const contentUrl = await getValidContentUrl();
      contentData = await getOrCreateTranscript(db, contentUrl);
      console.log(`\nTitle: ${contentData.title}`);
      if (contentData.author) console.log(`Author: ${contentData.author}`);
    } else {
      const narrative = await getNarrative();
      contentData = await restoreSession(db, narrative);
      console.log(`\nTitle: ${contentData.title}`);
      if (contentData.author) console.log(`Author: ${contentData.author}`);
    }
    console.log("\nSummary:", contentData.summary);
    await handleQuestions(contentData.transcript);
  } catch (error) {
    console.error(
      "Error:",
      error instanceof Error ? error.message : String(error)
    );
  } finally {
    await db.close();
    rl.close();
  }
};

main();
