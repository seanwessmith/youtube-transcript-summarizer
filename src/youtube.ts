import { Database } from "bun:sqlite";
import { $ } from "bun";
import { confirm, input, search, select } from "@inquirer/prompts";
import axios from "axios";
import * as dotenv from "dotenv";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { parseArgs } from "node:util";
import OpenAI from "openai";

import {
  buildRagIndex,
  chunkText,
  countWords,
  formatContext,
  selectRelevantChunks,
  type RagIndex,
} from "./rag.ts";

dotenv.config();

const YTDLP_BIN = process.env.YTDLP_BIN || "yt-dlp";
const FFMPEG_BIN = process.env.FFMPEG_BIN || "ffmpeg";
const WHISPER_CLI_BIN =
  process.env.WHISPER_CLI_BIN || "./whisper.cpp/build/bin/whisper-cli";
const WHISPER_MODEL_PATH =
  process.env.WHISPER_MODEL_PATH || "./whisper.cpp/models/ggml-base.en.bin";

const defaultModel = "gpt-5-mini";
const PROVIDERS = {
  SUMMARY_MODEL: process.env.SUMMARY_MODEL || defaultModel,
  QA_MODEL: process.env.QA_MODEL || defaultModel,
  EMBEDDING_MODEL: process.env.EMBEDDING_MODEL || "text-embedding-3-small",
};

const SUMMARY_CHUNK_CONFIG = { maxWords: 15000, overlapWords: 200 };
const QA_CHUNK_CONFIG = { maxWords: 1200, overlapWords: 150 };
const QA_CONTEXT_CHUNKS = 4;
const FIND_MATCH_THRESHOLD = 0.8;

if (!process.env.OPENAI_API_KEY) {
  throw new Error("OPENAI_API_KEY is required.");
}

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const zip = (text: string): Uint8Array =>
  Bun.gzipSync(new TextEncoder().encode(text));

const toUint8Array = (value: Uint8Array | ArrayBuffer): Uint8Array =>
  value instanceof Uint8Array
    ? Uint8Array.from(value)
    : new Uint8Array(value.slice(0));

const unzip = (value: Uint8Array | ArrayBuffer): string => {
  const uncompressed = Bun.gunzipSync(
    toUint8Array(value) as unknown as Uint8Array<ArrayBuffer>
  );
  return new TextDecoder().decode(Uint8Array.from(uncompressed));
};

const isGzip = (bytes: Uint8Array) =>
  bytes.length >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b;

const decodeBinaryTranscript = (bytes: Uint8Array): string => {
  const normalized = new Uint8Array(bytes);
  if (isGzip(normalized)) {
    try {
      return unzip(normalized);
    } catch {
      // Fall back to plain decoding if the stored bytes are malformed.
    }
  }
  return new TextDecoder().decode(normalized);
};

const decodeTranscript = (raw: unknown): string => {
  if (!raw) return "";
  if (typeof raw === "string") return raw;
  if (raw instanceof Uint8Array) return decodeBinaryTranscript(raw);
  if (raw instanceof ArrayBuffer) return decodeBinaryTranscript(new Uint8Array(raw));
  if (ArrayBuffer.isView(raw)) {
    return decodeBinaryTranscript(
      new Uint8Array(raw.buffer, raw.byteOffset, raw.byteLength)
    );
  }
  return String(raw);
};

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const createTempDir = (prefix: string): string =>
  fs.mkdtempSync(path.join(os.tmpdir(), prefix));

const removeDir = (dirPath: string) => {
  fs.rmSync(dirPath, { recursive: true, force: true });
};

const listMatchingFiles = (
  dirPath: string,
  predicate: (fileName: string) => boolean
): string[] =>
  fs
    .readdirSync(dirPath)
    .filter(predicate)
    .sort((a, b) => a.localeCompare(b))
    .map((fileName) => path.join(dirPath, fileName));

const readProcessOutput = (value: unknown): string => {
  if (!value) return "";
  if (typeof value === "string") return value;
  if (value instanceof Uint8Array) return new TextDecoder().decode(value);
  if (value instanceof ArrayBuffer) {
    return new TextDecoder().decode(new Uint8Array(value));
  }
  if (ArrayBuffer.isView(value)) {
    return new TextDecoder().decode(
      new Uint8Array(value.buffer, value.byteOffset, value.byteLength)
    );
  }
  return String(value);
};

const ensureLocalFile = (filePath: string, description: string) => {
  if (!filePath.includes(path.sep)) return;
  if (!fs.existsSync(filePath)) {
    throw new Error(`${description} not found: ${filePath}`);
  }
};

const normalizeTranscriptText = (text: string): string =>
  text.replace(/\r/g, "").replace(/\n{3,}/g, "\n\n").trim();

const extractVttText = (raw: string): string =>
  raw
    .split(/\r?\n/)
    .filter(
      (line) =>
        line &&
        !/^WEBVTT/.test(line) &&
        !/^NOTE/.test(line) &&
        !/^\d+$/.test(line) &&
        !/-->/.test(line)
    )
    .join(" ")
    .replace(/<[^>]+>/g, " ")
    .replace(/\s+/g, " ")
    .trim();

const getVideoId = (input: string): string => {
  const trimmed = input.trim();
  if (/^[A-Za-z0-9_-]{11}$/.test(trimmed)) return trimmed;

  const withProtocol = /^https?:\/\//i.test(trimmed) ? trimmed : `https://${trimmed}`;

  try {
    const url = new URL(withProtocol);

    if (url.hostname === "youtu.be") {
      const candidate = url.pathname.split("/").filter(Boolean)[0] ?? "";
      return /^[A-Za-z0-9_-]{11}$/.test(candidate) ? candidate : "";
    }

    if (
      url.hostname === "youtube.com" ||
      url.hostname === "www.youtube.com" ||
      url.hostname === "m.youtube.com"
    ) {
      const watchId = url.searchParams.get("v");
      if (watchId && /^[A-Za-z0-9_-]{11}$/.test(watchId)) return watchId;

      const parts = url.pathname.split("/").filter(Boolean);
      const marker = parts.findIndex((part) =>
        ["embed", "shorts", "live"].includes(part)
      );
      if (marker >= 0) {
        const candidate = parts[marker + 1] ?? "";
        return /^[A-Za-z0-9_-]{11}$/.test(candidate) ? candidate : "";
      }
    }
  } catch {
    // Fall back to the regex below.
  }

  const fallbackMatch = trimmed.match(
    /(?:v=|youtu\.be\/|\/embed\/|\/shorts\/|\/live\/)([A-Za-z0-9_-]{11})/
  );
  return fallbackMatch?.[1] ?? "";
};

async function fetchYoutubeTitle(videoId: string): Promise<string> {
  try {
    const url = `https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v=${videoId}&format=json`;
    const { data } = await axios.get<{ title?: string }>(url);
    return data.title?.trim() || `YouTube video ${videoId}`;
  } catch {
    return `YouTube video ${videoId}`;
  }
}

async function fetchCaptionsWithYtDlp(
  videoUrl: string,
  videoId: string
): Promise<string | null> {
  const tmpDir = createTempDir("yt-captions-");
  const outputTemplate = path.join(tmpDir, "%(id)s.%(ext)s");

  try {
    await $`${YTDLP_BIN} --no-warnings --skip-download --write-sub --write-auto-sub --sub-lang en --sub-format vtt -o ${outputTemplate} ${videoUrl}`.quiet();

    const candidates = listMatchingFiles(
      tmpDir,
      (fileName) => fileName.startsWith(videoId) && fileName.endsWith(".vtt")
    ).sort((a, b) => {
      const rank = (filePath: string) => {
        const base = path.basename(filePath);
        if (base.includes(".en.")) return 0;
        if (base.includes(".en-")) return 1;
        return 2;
      };
      return rank(a) - rank(b) || a.localeCompare(b);
    });

    if (candidates.length === 0) return null;

    const raw = fs.readFileSync(candidates[0], "utf8");
    const transcript = extractVttText(raw);
    return transcript || null;
  } catch (error) {
    const stderr = readProcessOutput((error as { stderr?: unknown })?.stderr);
    if (stderr.includes("live event will begin")) {
      throw new Error(
        "Cannot process upcoming live streams. Wait until the stream has finished."
      );
    }
    if (stderr.includes("This video is unavailable")) {
      throw new Error("Video is unavailable or private.");
    }
    console.warn("Captions unavailable, falling back to Whisper.");
    return null;
  } finally {
    removeDir(tmpDir);
  }
}

async function transcribeYoutubeWithWhisper(videoUrl: string): Promise<string> {
  ensureLocalFile(WHISPER_CLI_BIN, "Whisper CLI binary");
  ensureLocalFile(WHISPER_MODEL_PATH, "Whisper model");

  const tmpDir = createTempDir("yt-transcribe-");
  const audioTemplate = path.join(tmpDir, "audio.%(ext)s");
  const chunkTemplate = path.join(tmpDir, "chunk_%03d.mp3");

  try {
    await $`${YTDLP_BIN} --no-warnings -x --audio-format mp3 --audio-quality 128 -o ${audioTemplate} ${videoUrl}`.quiet();

    const audioFiles = listMatchingFiles(
      tmpDir,
      (fileName) => fileName.startsWith("audio.") && fileName.endsWith(".mp3")
    );
    const audioFile = audioFiles[0];

    if (!audioFile) {
      throw new Error("yt-dlp did not produce an MP3 file for transcription.");
    }

    await $`${FFMPEG_BIN} -hide_banner -loglevel error -i ${audioFile} -f segment -segment_time 600 -c copy ${chunkTemplate}`.quiet();

    const chunkFiles = listMatchingFiles(
      tmpDir,
      (fileName) => fileName.startsWith("chunk_") && fileName.endsWith(".mp3")
    );

    if (chunkFiles.length === 0) {
      throw new Error("ffmpeg did not produce any audio chunks.");
    }

    const transcriptParts: string[] = [];

    for (let index = 0; index < chunkFiles.length; index += 1) {
      const chunkFile = chunkFiles[index];
      const baseName = path.parse(chunkFile).name;
      const wavPath = path.join(tmpDir, `${baseName}.wav`);
      const outputBase = path.join(tmpDir, baseName);

      console.log(`Transcribing chunk ${index + 1}/${chunkFiles.length}...`);

      await $`${FFMPEG_BIN} -hide_banner -loglevel error -i ${chunkFile} -ar 16000 -ac 1 ${wavPath}`.quiet();
      await $`${WHISPER_CLI_BIN} --no-prints --no-timestamps --output-txt --output-file ${outputBase} --language en --model ${WHISPER_MODEL_PATH} --file ${wavPath}`.quiet();

      const transcriptPath = `${outputBase}.txt`;
      if (!fs.existsSync(transcriptPath)) {
        throw new Error(`Whisper did not produce ${path.basename(transcriptPath)}.`);
      }

      const chunkTranscript = normalizeTranscriptText(
        fs.readFileSync(transcriptPath, "utf8")
      );
      if (chunkTranscript) transcriptParts.push(chunkTranscript);
    }

    const transcript = transcriptParts.join("\n\n").trim();
    if (!transcript) {
      throw new Error("Whisper returned an empty transcript.");
    }

    return transcript;
  } finally {
    removeDir(tmpDir);
  }
}

interface ContentData {
  content_id: string;
  content_type: "youtube";
  title: string;
  transcript: string;
  summary: string;
  created_at?: string;
}

interface StoredContentRow {
  content_id: string;
  content_type: string;
  title: string;
  transcript: unknown;
  summary: string;
  created_at?: string;
}

interface QARow {
  id: number;
  question: string;
  answer: string;
  created_at?: string;
}

interface TranscriptOptions {
  forceRefresh?: boolean;
}

const toContentData = (row: StoredContentRow | null): ContentData | null => {
  if (!row) return null;
  return {
    content_id: row.content_id,
    content_type: "youtube",
    title: row.title || `YouTube video ${row.content_id}`,
    transcript: decodeTranscript(row.transcript),
    summary: row.summary || "",
    created_at: row.created_at,
  };
};

async function initDb(): Promise<Database> {
  const resolveDbPath = (): string => {
    const envPath = process.env.TRANSCRIPTS_DB?.trim();
    if (envPath) return envPath;

    const home = process.env.HOME || "";
    const candidates = [
      path.resolve(process.cwd(), "transcripts.sqlite"),
      path.resolve(import.meta.dir, "..", "transcripts.sqlite"),
      home ? path.resolve(home, "transcripts.sqlite") : "",
      home ? path.resolve(home, "Documents", "transcripts.sqlite") : "",
    ].filter(Boolean);

    const seen = new Set<string>();
    const existing = candidates.filter((candidate) => {
      if (seen.has(candidate)) return false;
      seen.add(candidate);
      return fs.existsSync(candidate);
    });

    for (const dbPath of existing) {
      try {
        const probe = new Database(dbPath);
        const row = probe
          .query("SELECT count(*) AS c FROM content")
          .get() as { c: number } | null;
        probe.close(true);
        if ((row?.c ?? 0) > 0) return dbPath;
      } catch {
        // Ignore non-SQLite files and continue.
      }
    }

    return existing[0] || path.resolve(import.meta.dir, "..", "transcripts.sqlite");
  };

  const dbPath = resolveDbPath();

  if (fs.existsSync(dbPath) && fs.statSync(dbPath).size > 0) {
    const backupDir = path.resolve(path.dirname(dbPath), "db_backups");
    if (!fs.existsSync(backupDir)) {
      fs.mkdirSync(backupDir, { recursive: true });
    }
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    fs.copyFileSync(
      dbPath,
      path.join(backupDir, `transcripts.${timestamp}.sqlite`)
    );
  }

  const db = new Database(dbPath);
  db.exec(`
    PRAGMA foreign_keys = ON;

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

  return db;
}

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

async function getOrCreateTranscript(
  db: Database,
  url: string,
  options: TranscriptOptions = {}
): Promise<ContentData> {
  const videoId = getVideoId(url);
  if (!videoId) {
    throw new Error("Only YouTube URLs are supported.");
  }

  const existing = toContentData(
    db
      .query(
        "SELECT * FROM content WHERE content_id = ? AND content_type = 'youtube'"
      )
      .get(videoId) as StoredContentRow | null
  );

  if (existing && !options.forceRefresh) return existing;

  const titlePromise = fetchYoutubeTitle(videoId);

  let transcript = await fetchCaptionsWithYtDlp(url, videoId);
  if (!transcript) {
    transcript = await transcribeYoutubeWithWhisper(url);
  }

  const title = await titlePromise;
  const summary = await summarizeTranscript(transcript);

  if (existing) {
    const qaResult = db.run("DELETE FROM qa WHERE content_id = ?", [videoId]);
    db.run(
      `UPDATE content
       SET title = ?, author = ?, audio_url = ?, transcript = ?, summary = ?, created_at = CURRENT_TIMESTAMP
       WHERE content_id = ? AND content_type = 'youtube'`,
      [title, "", "", zip(transcript), summary, videoId]
    );
    console.log(
      `Re-ran ${videoId} and cleared ${qaResult.changes} cached Q&A entr${qaResult.changes === 1 ? "y" : "ies"}.`
    );
  } else {
    db.run(
      `INSERT INTO content (content_id, content_type, title, author, audio_url, transcript, summary)
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
      [videoId, "youtube", title, "", "", zip(transcript), summary]
    );
    console.log("Transcript stored for", videoId);
  }

  return {
    content_id: videoId,
    content_type: "youtube",
    title,
    transcript,
    summary,
  };
}

async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  maxRetries = 6,
  initialDelayMs = 5000
): Promise<T> {
  let lastError: unknown;

  for (let attempt = 0; attempt < maxRetries; attempt += 1) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      const message = error instanceof Error ? error.message : String(error);
      const isRateLimit =
        message.includes("Too Many Requests") ||
        message.includes("429") ||
        (error as { status?: number; statusCode?: number })?.status === 429 ||
        (error as { status?: number; statusCode?: number })?.statusCode === 429;
      const isTimeout =
        message.toLowerCase().includes("timeout") ||
        message.includes("AbortError");

      if ((!isRateLimit && !isTimeout) || attempt === maxRetries - 1) {
        throw new Error(`OpenAI API Error: ${message}`);
      }

      const backoffMs = initialDelayMs * 2 ** attempt + Math.random() * 1000;
      console.log(
        `${isTimeout ? "Timeout" : "Rate limited"}. Waiting ${Math.round(
          backoffMs / 1000
        )}s before retry ${attempt + 1}/${maxRetries}...`
      );
      await sleep(backoffMs);
    }
  }

  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

async function completeChat(
  model: string,
  system: string,
  prompt: string
): Promise<string> {
  const response = await retryWithBackoff(() =>
    openai.chat.completions.create({
      model,
      messages: [
        { role: "system", content: system },
        { role: "user", content: prompt },
      ],
    })
  );

  return response.choices[0]?.message?.content?.trim() || "";
}

async function summarizeChunk(
  chunk: string,
  chunkNum?: number,
  totalChunks?: number
): Promise<string> {
  const chunkInfo =
    totalChunks && totalChunks > 1 ? ` (part ${chunkNum} of ${totalChunks})` : "";
  console.log(`Sending ${countWords(chunk)} words to ${PROVIDERS.SUMMARY_MODEL}...`);

  return completeChat(
    PROVIDERS.SUMMARY_MODEL,
    `You are an expert content analyst specializing in extracting insights from YouTube transcripts${chunkInfo}.

Analyze the transcript and provide:

## Key Points
- List 3-5 main ideas, arguments, or themes discussed
- For each point, include specific details, examples, or data mentioned
- Note any contrarian or surprising perspectives

## Notable Quotes
- Include 1-2 memorable or impactful direct quotes if any stand out

## People & References
- List any people, books, papers, companies, or tools mentioned

## Action Items / Takeaways
- Practical advice or recommendations given
- Things the viewer should consider doing

Be specific and factual. Preserve technical terms and proper nouns exactly as used.`,
    `Analyze this transcript:\n\n${chunk}`
  );
}

async function summarizeTranscript(transcript: string): Promise<string> {
  const chunks = chunkText(transcript, SUMMARY_CHUNK_CONFIG);
  console.log(
    `Summarizing ${countWords(transcript)} words with ${PROVIDERS.SUMMARY_MODEL}...`
  );

  if (chunks.length <= 1) {
    return summarizeChunk(transcript);
  }

  console.log(`Long transcript, splitting into ${chunks.length} chunks...`);
  const chunkSummaries: string[] = [];

  for (const chunk of chunks) {
    console.log(`Summarizing chunk ${chunk.index + 1}/${chunks.length}...`);
    chunkSummaries.push(
      await summarizeChunk(chunk.text, chunk.index + 1, chunks.length)
    );
  }

  return completeChat(
    PROVIDERS.SUMMARY_MODEL,
    `You are an expert content analyst. You will receive analyses from different parts of a long YouTube transcript.

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
- Remove duplicates and keep the most actionable items

Remove redundancy from overlapping sections. Preserve specificity and technical accuracy.`,
    `Synthesize these part analyses into a final summary:\n\n${chunkSummaries
      .map((summary, index) => `=== Part ${index + 1} ===\n${summary}`)
      .join("\n\n")}`
  );
}

async function getEmbedding(text: string): Promise<number[]> {
  const result = await retryWithBackoff(() =>
    openai.embeddings.create({
      model: PROVIDERS.EMBEDDING_MODEL,
      input: text,
    })
  );
  return result.data[0]?.embedding ?? [];
}

async function buildQaIndex(transcript: string): Promise<RagIndex> {
  return buildRagIndex(transcript, getEmbedding, QA_CHUNK_CONFIG);
}

async function answerQuestion(index: RagIndex, question: string): Promise<string> {
  const contextChunks = await selectRelevantChunks(
    index,
    question,
    getEmbedding,
    QA_CONTEXT_CHUNKS
  );
  const context = formatContext(contextChunks);

  return completeChat(
    PROVIDERS.QA_MODEL,
    `You answer questions about a YouTube transcript.

Rules:
- Use only the supplied transcript excerpts
- If the answer is not supported by the excerpts, say so clearly
- Quote or reference the most relevant excerpt when helpful
- Be concise but specific`,
    `Question: ${question}\n\nTranscript excerpts:\n${context}`
  );
}

async function restoreSession(
  db: Database,
  narrative: string
): Promise<ContentData> {
  const rows = (
    db
      .query("SELECT * FROM content WHERE content_type = 'youtube'")
      .all() as StoredContentRow[]
  )
    .map((row) => toContentData(row))
    .filter((row): row is ContentData => Boolean(row));

  if (rows.length === 0) throw new Error("No YouTube sessions saved.");

  const queryEmbedding = await getEmbedding(narrative);
  const summaryEmbeddings = await Promise.all(
    rows.map((row) => getEmbedding(row.summary || row.title))
  );

  let bestIndex = -1;
  let bestScore = -Infinity;

  summaryEmbeddings.forEach((embedding, index) => {
    const score = embedding.length === 0 ? 0 : cosineSimilarity(queryEmbedding, embedding);
    if (score > bestScore) {
      bestScore = score;
      bestIndex = index;
    }
  });

  if (bestIndex === -1 || bestScore < FIND_MATCH_THRESHOLD) {
    throw new Error("No close match found.");
  }

  return rows[bestIndex];
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length === 0 || b.length === 0 || a.length !== b.length) return 0;

  let dot = 0;
  let magA = 0;
  let magB = 0;

  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }

  const denominator = Math.sqrt(magA) * Math.sqrt(magB);
  return denominator ? dot / denominator : 0;
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
    throw new Error("No Q&A to export.");
  }

  const safeTitle = title.replace(/[^a-zA-Z0-9]/g, "_").slice(0, 50) || "youtube";
  const timestamp = new Date().toISOString().split("T")[0];

  if (format === "json") {
    const exportData = {
      title,
      content_id: contentId,
      exported_at: new Date().toISOString(),
      qa: qaRows.map((row) => ({
        question: row.question,
        answer: row.answer,
        created_at: row.created_at,
      })),
    };

    const filename = `${safeTitle}_qa_${timestamp}.json`;
    await Bun.write(filename, JSON.stringify(exportData, null, 2));
    return filename;
  }

  const markdown = [
    `# Q&A Export: ${title}`,
    "",
    `*Exported: ${new Date().toISOString()}*`,
    "",
    ...qaRows.flatMap((row, index) => [
      `## Question ${index + 1}`,
      "",
      `**Q:** ${row.question}`,
      "",
      `**A:** ${row.answer}`,
      "",
      "---",
      "",
    ]),
  ].join("\n");

  const filename = `${safeTitle}_qa_${timestamp}.md`;
  await Bun.write(filename, markdown);
  return filename;
}

async function main() {
  const { values } = parseArgs({
    args: Bun.argv.slice(2),
    options: {
      url: { type: "string", short: "u" },
      rerun: { type: "string", short: "r" },
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
  -u, --url <url>      Process a YouTube URL directly (non-interactive)
  -r, --rerun <url>    Re-fetch and re-summarize a saved YouTube URL
  -f, --find <text>    Find a saved YouTube session by semantic match
  -d, --delete <url>   Delete a saved YouTube session and its Q&A
  -h, --help           Show this help message

Interactive mode:
  Run without options to pick a saved transcript or start a new one.
`);
    return;
  }

  const db = await initDb();

  try {
    let data: ContentData;

    if (values.delete) {
      const videoId = getVideoId(values.delete.trim());
      if (!videoId) {
        throw new Error("Please provide a valid YouTube URL to delete.");
      }

      const existing = db
        .query(
          "SELECT title FROM content WHERE content_id = ? AND content_type = 'youtube'"
        )
        .get(videoId) as { title: string } | null;

      if (!existing) {
        console.log(`No YouTube entry found for: ${videoId}`);
        return;
      }

      const qaResult = db.run("DELETE FROM qa WHERE content_id = ?", [videoId]);
      const contentResult = db.run(
        "DELETE FROM content WHERE content_id = ? AND content_type = 'youtube'",
        [videoId]
      );

      console.log(`Deleted "${existing.title}"`);
      console.log(`  - Removed ${qaResult.changes} Q&A entries`);
      console.log(`  - Removed ${contentResult.changes} content entry`);
      return;
    }

    if (values.url) {
      data = await getOrCreateTranscript(db, values.url.trim());
      console.log(`\nTitle: ${data.title}`);
      console.log(`\nSummary:\n${data.summary}\n`);
      return;
    }

    if (values.rerun) {
      data = await getOrCreateTranscript(db, values.rerun.trim(), {
        forceRefresh: true,
      });
      console.log(`\nTitle: ${data.title}`);
      console.log(`\nSummary:\n${data.summary}\n`);
      return;
    }

    if (values.find) {
      console.log(`Searching for: "${values.find}"...`);
      data = await restoreSession(db, values.find);
      console.log(`\nFound: ${data.title}`);
      console.log(`\nSummary:\n${data.summary}\n`);
      return;
    }

    const sessions = db
      .query(
        `SELECT content_id, title, DATE(created_at) AS created
         FROM content
         WHERE content_type = 'youtube'
         ORDER BY created_at DESC`
      )
      .all() as {
        content_id: string;
        title: string;
        created: string;
      }[];

    console.log(`Loaded ${sessions.length} YouTube sessions from DB.`);

    let lastTerm = "";
    const pick = await search({
      message: "Select a saved transcript or start a new one:",
      source: async (term) => {
        lastTerm = term || "";

        if (!term) {
          return [
            { name: "Start new session", value: "__new" },
            ...sessions.map((session) => ({
              name: `${session.title} [${session.created}]`,
              value: session.content_id,
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

        const lowerTerm = term.toLowerCase();
        return sessions
          .filter((session) => session.title.toLowerCase().includes(lowerTerm))
          .map((session) => ({
            name: `${session.title} [${session.created}]`,
            value: session.content_id,
          }));
      },
      pageSize: 12,
    });

    if (pick === "__new") {
      if (lastTerm.startsWith("http") || lastTerm.startsWith("www")) {
        data = await getOrCreateTranscript(db, lastTerm.trim());
      } else {
        const url = await input({ message: "Enter YouTube URL:" });
        data = await getOrCreateTranscript(db, url.trim());
      }
    } else if (pick.startsWith("__url:")) {
      data = await getOrCreateTranscript(db, pick.slice(6).trim());
    } else {
      const row = db
        .query("SELECT * FROM content WHERE content_id = ? AND content_type = 'youtube'")
        .get(pick) as StoredContentRow | null;
      const loaded = toContentData(row);
      if (!loaded) throw new Error("Session not found in DB.");
      data = loaded;
    }

    console.log(`\nTitle: ${data.title}`);
    console.log(`\nSummary: ${data.summary}\n`);

    let qaIndexPromise: Promise<RagIndex> | undefined;

    while (true) {
      const qaRows = db
        .query(
          "SELECT id, question, answer FROM qa WHERE content_id = ? ORDER BY created_at DESC"
        )
        .all(data.content_id) as QARow[];

      const selection = await select({
        message: `Questions (${PROVIDERS.QA_MODEL}):`,
        choices: [
          { name: "New question", value: "__new" },
          ...qaRows.map((row) => ({ name: row.question, value: String(row.id) })),
          { name: "Re-run this video", value: "__rerun" },
          { name: "Export Q&A", value: "__export" },
          { name: "Delete this session", value: "__delete" },
          { name: "Exit", value: "__exit" },
        ],
        pageSize: 12,
      });

      if (selection === "__exit") break;

      if (selection === "__delete") {
        const confirmed = await confirm({
          message: `Delete "${data.title}" and all its Q&A? This cannot be undone.`,
          default: false,
        });

        if (!confirmed) continue;

        const qaResult = db.run("DELETE FROM qa WHERE content_id = ?", [data.content_id]);
        const contentResult = db.run(
          "DELETE FROM content WHERE content_id = ? AND content_type = 'youtube'",
          [data.content_id]
        );

        console.log(`\nDeleted "${data.title}"`);
        console.log(`  - Removed ${qaResult.changes} Q&A entries`);
        console.log(`  - Removed ${contentResult.changes} content entry\n`);
        break;
      }

      if (selection === "__rerun") {
        const confirmed = await confirm({
          message:
            "Re-run this YouTube video from source and replace the saved transcript/summary? Cached Q&A will be cleared.",
          default: false,
        });

        if (!confirmed) continue;

        data = await getOrCreateTranscript(
          db,
          `https://www.youtube.com/watch?v=${data.content_id}`,
          { forceRefresh: true }
        );
        qaIndexPromise = undefined;
        console.log(`\nTitle: ${data.title}`);
        console.log(`\nSummary: ${data.summary}\n`);
        continue;
      }

      if (selection === "__export") {
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
        } catch (error) {
          console.error(
            `Export failed: ${error instanceof Error ? error.message : String(error)}`
          );
        }
        continue;
      }

      if (selection === "__new") {
        const question = await input({ message: "Question:" });
        if (!question.trim()) continue;

        if (!qaIndexPromise) {
          console.log("Indexing transcript for Q&A...");
          qaIndexPromise = buildQaIndex(data.transcript);
        }

        const answer = await answerQuestion(await qaIndexPromise, question.trim());
        console.log(`\nAnswer: ${answer}\n`);
        storeQA(db, data.content_id, question.trim(), answer);
        continue;
      }

      const qa = qaRows.find((row) => String(row.id) === selection);
      if (qa) {
        console.log(`\nAnswer: ${qa.answer}\n`);
      }
    }
  } catch (error) {
    console.error("Error:", error instanceof Error ? error.message : String(error));
  } finally {
    db.close(true);
  }
}

main();
