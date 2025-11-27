import dotenv from "dotenv";
import axios from "axios";
import { parseStringPromise } from "xml2js";
import { Database } from "bun:sqlite";
import path from "path";
import { google } from "@ai-sdk/google";
import { generateText, embed } from "ai";
import { $ } from "bun";

const zip = (txt: string): Uint8Array => {
  const encoded = new TextEncoder().encode(txt);
  return Bun.gzipSync(encoded);
};

const unzip = (txt: string) => {
  const dec = new TextDecoder();
  const uncompressed = Bun.gunzipSync(txt);
  return dec.decode(uncompressed);
};

// ────────────────────────────────────────────────────────────────
// 1. Environment & util setup
// ────────────────────────────────────────────────────────────────

dotenv.config();

const YTDLP_BIN = process.env.YTDLP_BIN || "yt-dlp"; // path to yt-dlp binary
const TRANSCRIBE_BIN = process.env.TRANSCRIBE_BIN || "./src/transcribe";
const WHISPER_MODEL_PATH =
  process.env.WHISPER_MODEL_PATH || "./whisper.cpp/models/ggml-base.en.bin";

// ────────────────────────────────────────────────────────────────
// 2. Caption retrieval (uses yt-dlp directly – no Invidious needed)
// ────────────────────────────────────────────────────────────────

async function fetchCaptionsWithYtDlp(
  videoUrl: string,
  videoId: string
): Promise<string | null> {
  const tmpDir = "./tmp";
  const vttPath = path.join(tmpDir, `${videoId}.en.vtt`);

  // Create tmp directory
  await $`mkdir -p ${tmpDir}`.quiet();

  try {
    // Run yt-dlp to download captions
    await $`${YTDLP_BIN} \
      --write-auto-sub \
      --sub-lang en \
      --sub-format vtt \
      --skip-download \
      -o ${path.join(tmpDir, "%(id)s.%(ext)s")} \
      ${videoUrl}`.quiet();

    // Read and process the VTT file
    const raw = await Bun.file(vttPath).text();

    // Convert VTT to plain text
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

    return transcript || null;
  } catch (err: any) {
    // Check for specific YouTube errors
    if (err?.stderr) {
      const stderr = err.stderr.toString();
      if (stderr.includes("live event will begin")) {
        throw new Error(
          "Cannot process upcoming live streams. Please wait until the stream has started and finished."
        );
      }
      if (stderr.includes("This video is unavailable")) {
        throw new Error("Video is unavailable or private.");
      }
    }
    console.warn("Failed to fetch captions, will try transcription fallback");
    return null;
  }
}

// ────────────────────────────────────────────────────────────────
// 3. Cleanup & YouTube helpers
// ────────────────────────────────────────────────────────────────

async function cleanupTmpFiles(videoId: string): Promise<void> {
  const tmpDir = "./tmp";
  try {
    const files = await Array.fromAsync(new Bun.Glob(`${videoId}*`).scan(tmpDir));
    for (const file of files) {
      await Bun.file(path.join(tmpDir, file)).delete();
    }
  } catch {
    // Ignore cleanup errors
  }
}

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
  try {
    const { stdout } =
      await $`${TRANSCRIBE_BIN} ${audioUrl} ${WHISPER_MODEL_PATH}`;
    return stdout.toString().trim();
  } catch (err) {
    console.error("Transcription failed:", err);
    throw new Error(
      `Transcriber failed: ${err instanceof Error ? err.message : String(err)}`
    );
  }
}

// ────────────────────────────────────────────────────────────────
// 5. LLM & embedding setup (using Gemini)
// ────────────────────────────────────────────────────────────────

const defaultModel = "gemini-3-pro-preview";
const PROVIDERS = {
  SUMMARY_MODEL: process.env.SUMMARY_MODEL || defaultModel,
  QA_MODEL: process.env.QA_MODEL || defaultModel,
  EMBEDDING_MODEL: process.env.EMBEDDING_MODEL || "text-embedding-004",
};

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
  // db.exec(`
  //   PRAGMA auto_vacuum = INCREMENTAL;
  //   PRAGMA journal_mode = WAL;
  //   VACUUM;
  // `);

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
      existing.transcript = unzip(existing.transcript); // ← inflate
      return existing as ContentData;
    }

    // Fetch title in parallel with transcript
    const titlePromise = fetchYoutubeTitleOEmbed(videoId);

    // 1️⃣ Try yt-dlp captions
    let transcript = await fetchCaptionsWithYtDlp(url, videoId);

    // 2️⃣ Fallback to Whisper transcription
    if (!transcript) transcript = await fetchOrTranscribe(url);

    // Cleanup tmp VTT file
    await cleanupTmpFiles(videoId);

    const title = await titlePromise;
    const summary = await summarize(transcript);
    const zipped = zip(transcript);
    db.run(
      `INSERT INTO content (content_id, content_type, title, author, transcript, summary) VALUES (?, ?, ?, ?, ?, ?)`,
      [videoId, "youtube", title, "", zipped, summary]
    );
    console.log("Transcript stored for", videoId);
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
      zip(transcript), // gzip
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
// 8. Summarisation & QA using Gemini native API
// ────────────────────────────────────────────────────────────────

// Retry helper with exponential backoff
async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  maxRetries = 6,
  initialDelay = 5000
): Promise<T> {
  let lastError: Error | undefined;
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await fn();
    } catch (err: any) {
      lastError = err;

      // Check if this is a rate limit error
      const errorMessage = err?.message || "";
      const isRateLimitError =
        errorMessage.includes("Too Many Requests") ||
        errorMessage.includes("429") ||
        errorMessage.includes("Rate limit") ||
        err?.status === 429 ||
        err?.statusCode === 429;

      if (!isRateLimitError || attempt === maxRetries - 1) {
        // Not a rate limit error or final attempt - throw it
        const errorDetails = err?.cause?.message || err?.message || String(err);
        throw new Error(`Gemini API Error: ${errorDetails}`);
      }

      // Calculate delay with jitter to avoid thundering herd
      const baseDelay = initialDelay * Math.pow(2, attempt);
      const jitter = Math.random() * 1000;
      const delay = baseDelay + jitter;

      console.log(
        `Rate limited. Waiting ${Math.round(delay / 1000)}s before retry ${attempt + 1}/${maxRetries}...`
      );
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }
  throw lastError || new Error("Retry failed");
}

const MAX_WORDS_PER_CHUNK = 15000;
const CHUNK_OVERLAP_WORDS = 200;

function chunkTranscript(transcript: string): string[] {
  const words = transcript.split(/\s+/);
  if (words.length <= MAX_WORDS_PER_CHUNK) {
    return [transcript];
  }

  const chunks: string[] = [];
  let start = 0;
  while (start < words.length) {
    const end = Math.min(start + MAX_WORDS_PER_CHUNK, words.length);
    chunks.push(words.slice(start, end).join(" "));
    start = end - CHUNK_OVERLAP_WORDS;
    if (start >= words.length) break;
  }
  return chunks;
}

async function summarizeChunk(chunk: string, chunkNum?: number, totalChunks?: number): Promise<string> {
  const chunkInfo = totalChunks && totalChunks > 1 ? ` (part ${chunkNum} of ${totalChunks})` : "";
  const { text } = await retryWithBackoff(() =>
    generateText({
      model: google(PROVIDERS.SUMMARY_MODEL),
      prompt: `You are an expert content analyst specializing in extracting insights from video and podcast transcripts${chunkInfo}.

Analyze this transcript and provide:

## Key Points
- List 3-5 main ideas, arguments, or themes discussed
- For each point, include specific details, examples, or data mentioned
- Note any contrarian or surprising perspectives

## Notable Quotes
- Include 1-2 memorable or impactful direct quotes (if any stand out)

## People & References
- List any people, books, papers, companies, or tools mentioned

## Action Items / Takeaways
- Practical advice or recommendations given
- Things the viewer/listener should consider doing

Be specific and factual. Preserve technical terms and proper nouns exactly as used.

Transcript:
${chunk}`,
    })
  );
  return text;
}

async function summarize(transcript: string): Promise<string> {
  const wordCount = transcript.split(/\s+/).length;
  console.log(`summarizing ${wordCount} word transcript with ${defaultModel}`);

  const chunks = chunkTranscript(transcript);

  if (chunks.length === 1) {
    return summarizeChunk(transcript);
  }

  console.log(`Transcript too long, splitting into ${chunks.length} chunks...`);

  const chunkSummaries: string[] = [];
  for (let i = 0; i < chunks.length; i++) {
    console.log(`Summarizing chunk ${i + 1}/${chunks.length}...`);
    const summary = await summarizeChunk(chunks[i], i + 1, chunks.length);
    chunkSummaries.push(summary);
  }

  console.log("Combining chunk summaries into final summary...");
  const { text } = await retryWithBackoff(() =>
    generateText({
      model: google(PROVIDERS.SUMMARY_MODEL),
      prompt: `You are an expert content analyst. Below are analyses from different parts of a long transcript.

Synthesize them into a single cohesive summary following this structure:

## Key Points
- Combine and deduplicate the main ideas across all parts
- Prioritize the most significant and recurring themes
- Include specific details and examples

## Notable Quotes
- Select the 2-3 best quotes from across all parts

## People & References
- Consolidated list of all people, books, papers, companies, and tools mentioned

## Action Items / Takeaways
- Combined practical advice and recommendations
- Remove duplicates, keep the most actionable items

Remove redundancy from overlapping sections. Preserve specificity and technical accuracy.

Part analyses:
${chunkSummaries.map((s, i) => `=== Part ${i + 1} ===\n${s}`).join("\n\n")}`,
    })
  );
  return text;
}

async function answer(transcript: string, question: string): Promise<string> {
  const { text } = await retryWithBackoff(() =>
    generateText({
      model: google(PROVIDERS.QA_MODEL),
      prompt: `You are answering questions about a video/podcast transcript. Your role is to be a helpful assistant that has "watched" this content.

Guidelines:
- Answer based ONLY on information in the transcript
- If the answer isn't in the transcript, say so clearly
- Quote relevant parts when helpful (use quotation marks)
- Be specific: include names, numbers, timestamps context if mentioned
- For opinion questions, present what the speaker(s) said, not your own views
- If multiple speakers discuss the topic, note different perspectives

Transcript:
${transcript}

Question: ${question}

Answer:`,
    })
  );
  return text;
}

function cosineSimilarity(a: number[], b: number[]): number {
  const dot = a.reduce((s, v, i) => s + v * b[i], 0);
  const magA = Math.hypot(...a);
  const magB = Math.hypot(...b);
  return magA && magB ? dot / (magA * magB) : 0;
}

async function getEmbedding(text: string): Promise<number[]> {
  const { embedding } = await embed({
    model: google.textEmbeddingModel(PROVIDERS.EMBEDDING_MODEL),
    value: text,
  });
  return embedding;
}

async function restoreSession(
  db: Database,
  narrative: string
): Promise<ContentData> {
  const rows = db.query("SELECT * FROM content").all() as ContentData[];
  if (rows.length === 0) throw new Error("No sessions saved");

  const navEmb = await getEmbedding(narrative);
  const sims = await Promise.all(rows.map((r) => getEmbedding(r.summary)));

  let best = { idx: -1, score: -Infinity };
  sims.forEach((emb: number[], i: number) => {
    const sim = cosineSimilarity(navEmb, emb);
    if (sim > best.score) best = { idx: i, score: sim };
  });

  if (best.score < 0.8) throw new Error("No close match found");
  return rows[best.idx];
}

// ────────────────────────────────────────────────────────────────
// 9. Export functions
// ────────────────────────────────────────────────────────────────

interface QARow {
  id: number;
  question: string;
  answer: string;
  created_at?: string;
}

async function exportQA(
  db: Database,
  contentId: string,
  title: string,
  format: "markdown" | "json"
): Promise<string> {
  const qaRows = db
    .query(
      "SELECT id, question, answer, created_at FROM qa WHERE content_id = ? ORDER BY created_at ASC"
    )
    .all(contentId) as QARow[];

  if (qaRows.length === 0) {
    throw new Error("No Q&A to export");
  }

  const safeTitle = title.replace(/[^a-zA-Z0-9]/g, "_").slice(0, 50);
  const timestamp = new Date().toISOString().split("T")[0];

  if (format === "json") {
    const exportData = {
      title,
      content_id: contentId,
      exported_at: new Date().toISOString(),
      qa: qaRows.map((r) => ({
        question: r.question,
        answer: r.answer,
        created_at: r.created_at,
      })),
    };
    const filename = `${safeTitle}_qa_${timestamp}.json`;
    await Bun.write(filename, JSON.stringify(exportData, null, 2));
    return filename;
  } else {
    const markdown = [
      `# Q&A Export: ${title}`,
      ``,
      `*Exported: ${new Date().toISOString()}*`,
      ``,
      ...qaRows.flatMap((r, i) => [
        `## Question ${i + 1}`,
        ``,
        `**Q:** ${r.question}`,
        ``,
        `**A:** ${r.answer}`,
        ``,
        `---`,
        ``,
      ]),
    ].join("\n");
    const filename = `${safeTitle}_qa_${timestamp}.md`;
    await Bun.write(filename, markdown);
    return filename;
  }
}

// ────────────────────────────────────────────────────────────────
// 10. CLI interaction loop
// ────────────────────────────────────────────────────────────────

import { search, select, input } from "@inquirer/prompts";
import { parseArgs } from "util";

async function main() {
  // Parse CLI arguments
  const { values } = parseArgs({
    args: Bun.argv.slice(2),
    options: {
      url: { type: "string", short: "u" },
      find: { type: "string", short: "f" },
      delete: { type: "string", short: "d" },
      help: { type: "boolean", short: "h" },
    },
    allowPositionals: true,
  });

  if (values.help) {
    console.log(`
Usage: bun run src/youtube.ts [options]

Options:
  -u, --url <url>      Process a YouTube or podcast URL directly (non-interactive)
  -f, --find <text>    Find a session by fuzzy matching description
  -d, --delete <url>   Delete a video/podcast and its Q&A from the database
  -h, --help           Show this help message

Interactive mode (default):
  Run without options to use the interactive session picker.
`);
    return;
  }

  const db = await initDb();
  try {
    let data: ContentData;

    // Non-interactive: --delete flag
    if (values.delete) {
      const urlToDelete = values.delete.trim();
      const videoId = getVideoId(urlToDelete);
      let contentId = videoId;

      // If not a YouTube URL, try podcast format
      if (!videoId) {
        const podcastMatch = urlToDelete.match(/\/id(\d+)/);
        const episodeId = urlToDelete.match(/[?&]i=(\d+)/)?.[1] ?? "";
        if (podcastMatch) {
          contentId = `podcast_${podcastMatch[1]}_episode_${episodeId}`;
        }
      }

      if (!contentId) {
        console.error("Could not parse URL. Please provide a valid YouTube or podcast URL.");
        return;
      }

      const existing = db
        .query("SELECT title FROM content WHERE content_id = ?")
        .get(contentId) as { title: string } | null;

      if (!existing) {
        console.log(`No entry found for: ${contentId}`);
        return;
      }

      // Delete Q&A first (foreign key), then content
      const qaResult = db.run("DELETE FROM qa WHERE content_id = ?", [contentId]);
      const contentResult = db.run("DELETE FROM content WHERE content_id = ?", [contentId]);

      console.log(`Deleted "${existing.title}"`);
      console.log(`  - Removed ${qaResult.changes} Q&A entries`);
      console.log(`  - Removed ${contentResult.changes} content entry`);
      return;
    }

    // Non-interactive: --url flag
    if (values.url) {
      data = await getOrCreateTranscript(db, values.url.trim());
      console.log(`\nTitle: ${data.title}`);
      if (data.author) console.log(`Author: ${data.author}`);
      console.log(`\nSummary:\n${data.summary}\n`);
      return;
    }

    // Non-interactive: --find flag (fuzzy search using embeddings)
    if (values.find) {
      console.log(`Searching for: "${values.find}"...`);
      data = await restoreSession(db, values.find);
      data.transcript = unzip(data.transcript);
      console.log(`\nFound: ${data.title}`);
      if (data.author) console.log(`Author: ${data.author}`);
      console.log(`\nSummary:\n${data.summary}\n`);
      return;
    }

    // Interactive mode
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

    console.log(`Loaded ${sessions.length} sessions from DB.`);

    let lastTerm = "";
    const pick = await search({
      message: "Select a saved transcript or start a new one:",
      source: async (term) => {
        lastTerm = term || "";
        if (!term) {
          return [
            { name: "Start new session", value: "__new" },
            ...sessions.map((s) => ({
              name: `${s.title}  – ${s.content_type.toUpperCase()}  [${s.created}]`,
              value: s.content_id,
            })),
          ];
        }

        const isUrl = term.startsWith("http") || term.startsWith("www");
        if (isUrl) {
          return [
            {
              name: `Start new session from: ${term}`,
              value: `__url:${term}`,
            },
          ];
        }

        // Filter sessions
        const lowerTerm = term.toLowerCase();
        return sessions
          .filter((s) => s.title.toLowerCase().includes(lowerTerm))
          .map((s) => ({
            name: `${s.title}  – ${s.content_type.toUpperCase()}  [${s.created}]`,
            value: s.content_id,
          }));
      },
      pageSize: 12,
    });

    if (pick === "__new") {
      if (lastTerm.startsWith("http") || lastTerm.startsWith("www")) {
        data = await getOrCreateTranscript(db, lastTerm.trim());
      } else {
        const url = await input({
          message: "Enter YouTube or Apple Podcast URL:",
        });
        data = await getOrCreateTranscript(db, url.trim());
      }
    } else if (pick.startsWith("__url:")) {
      const url = pick.substring(6);
      data = await getOrCreateTranscript(db, url.trim());
    } else {
      data = db
        .query("SELECT * FROM content WHERE content_id = ?")
        .get(pick) as ContentData;

      if (!data) throw new Error("Session not found in DB");
      data.transcript = unzip(data.transcript);
    }

    console.log(`\nTitle: ${data.title}`);
    if (data.author) console.log(`Author: ${data.author}`);
    console.log(`\nSummary: ${data.summary}\n`);

    /* QA selector loop */
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

      const sel = await select({
        message: `Questions (${PROVIDERS.QA_MODEL}):`,
        choices: [
          { name: "🆕  New question", value: "__new" },
          ...qaRows.map((r) => ({ name: r.question, value: String(r.id) })),
          { name: "📤  Export Q&A", value: "__export" },
          { name: "Exit", value: "__exit" },
        ],
        pageSize: 10,
        default: "0",
      });

      if (sel === "__exit") break;

      if (sel === "__export") {
        try {
          const format = await select({
            message: "Export format:",
            choices: [
              { name: "Markdown (.md)", value: "markdown" as const },
              { name: "JSON (.json)", value: "json" as const },
            ],
          });
          const filename = await exportQA(db, data.content_id, data.title, format);
          console.log(`\nExported to: ${filename}\n`);
        } catch (err: any) {
          console.error(`Export failed: ${err.message}`);
        }
        continue;
      }

      if (sel === "__new") {
        const newQ = await input({
          message: "question:",
        });
        if (!newQ.trim()) continue;

        const a = await answer(data.transcript, newQ);
        console.log(`\nAnswer: ${a}\n`);
        storeQA(db, data.content_id, newQ, a);
      } else {
        const qa = qaRows.find((r) => String(r.id) === sel)!;
        console.log(`\nAnswer: ${qa.answer}\n`);
      }
    }
  } catch (err: any) {
    console.error("Error:", err.message);
  } finally {
    db.close(true);
    await new Promise((res) => setTimeout(res, 1000));
  }
}

main();
