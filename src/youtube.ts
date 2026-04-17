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
  cosineSimilarity,
  countWords,
  formatContext,
  selectRelevantChunks,
  type RagIndex,
  type TextChunk,
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
  SUMMARY_MODEL: process.env.SUMMARY_MODEL?.trim() || defaultModel,
  QA_MODEL: process.env.QA_MODEL?.trim() || defaultModel,
  EMBEDDING_MODEL:
    process.env.EMBEDDING_MODEL?.trim() || "text-embedding-3-small",
};

const SUMMARY_CHUNK_CONFIG = { maxWords: 6000, overlapWords: 250 };
const QA_CHUNK_CONFIG = { maxWords: 1200, overlapWords: 150 };
const QA_CONTEXT_CHUNKS = 4;
const QA_MIN_RELEVANCE_SCORE = 0.2;
const FIND_MATCH_THRESHOLD = 0.8;
const FIND_EMBEDDING_KIND = "find_summary";
const EMBEDDING_BATCH_SIZE = 16;
const MAX_DB_BACKUPS = 10;
const UNSUPPORTED_ANSWER =
  "I couldn't find support for that in the retrieved transcript excerpts.";
const NO_CAPTION_PATTERNS = [
  /there are no subtitles/i,
  /requested subtitles.*not available/i,
  /requested languages?.*not available/i,
  /no automatic captions/i,
  /subtitles are not available/i,
  /video doesn't have subtitles/i,
];

let openaiClient: OpenAI | null = null;
const backedUpDatabases = new Set<string>();

interface ContentData {
  contentId: string;
  title: string;
  transcript: string;
  summary: string;
  createdAt?: string;
}

interface StoredContentRow {
  content_id: string;
  content_type: string;
  title: string | null;
  transcript: unknown;
  summary: string | null;
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

interface DatabaseHandle {
  db: Database;
  path: string;
  hadExistingData: boolean;
}

interface StoredContentEmbeddingRow {
  content_id: string;
  source_text: string;
  embedding: string;
}

interface StoredChunkEmbeddingRow {
  chunk_index: number;
  start_word: number;
  end_word: number;
  text: string;
  embedding: string;
}

interface PersistedChunkEmbedding {
  chunk: TextChunk;
  embedding: number[];
}

const getOpenAIClient = (): OpenAI => {
  const apiKey = process.env.OPENAI_API_KEY?.trim();
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY is required for this action.");
  }

  if (!openaiClient) {
    openaiClient = new OpenAI({ apiKey });
  }

  return openaiClient;
};

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

const encodeEmbedding = (embedding: number[]): string => JSON.stringify(embedding);

const decodeEmbedding = (raw: string): number[] => {
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) && parsed.every((value) => typeof value === "number")
      ? parsed
      : [];
  } catch {
    return [];
  }
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

const collapseWhitespace = (text: string): string =>
  text.replace(/\s+/g, " ").trim();

const getProcessFailureOutput = (error: unknown): string => {
  const failure = error as { stderr?: unknown; stdout?: unknown };
  return [
    readProcessOutput(failure.stderr),
    readProcessOutput(failure.stdout),
    error instanceof Error ? error.message : String(error),
  ]
    .filter(Boolean)
    .join("\n")
    .trim();
};

const describeProcessError = (error: unknown): string => {
  const output = getProcessFailureOutput(error);
  if (!output) return "unknown process failure";

  const bestLine = output
    .split(/\r?\n/)
    .map((line) => collapseWhitespace(line))
    .find(Boolean);

  return bestLine || collapseWhitespace(output);
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

const isNoCaptionFailure = (output: string): boolean =>
  NO_CAPTION_PATTERNS.some((pattern) => pattern.test(output));

const getSearchText = (row: ContentData): string =>
  row.summary.trim() || row.title.trim() || row.contentId;

const toContentData = (row: StoredContentRow | null): ContentData | null => {
  if (!row) return null;
  return {
    contentId: row.content_id,
    title: row.title?.trim() || `YouTube video ${row.content_id}`,
    transcript: decodeTranscript(row.transcript),
    summary: row.summary?.trim() || "",
    createdAt: row.created_at,
  };
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
  ensureLocalFile(YTDLP_BIN, "yt-dlp binary");

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
    const output = getProcessFailureOutput(error);
    if (/live event will begin/i.test(output)) {
      throw new Error(
        "Cannot process upcoming live streams. Wait until the stream has finished."
      );
    }
    if (/this video is unavailable/i.test(output)) {
      throw new Error("Video is unavailable or private.");
    }
    if (isNoCaptionFailure(output)) {
      console.warn("Captions unavailable, falling back to Whisper.");
      return null;
    }

    throw new Error(`yt-dlp caption fetch failed: ${describeProcessError(error)}`);
  } finally {
    removeDir(tmpDir);
  }
}

async function transcribeYoutubeWithWhisper(videoUrl: string): Promise<string> {
  ensureLocalFile(YTDLP_BIN, "yt-dlp binary");
  ensureLocalFile(FFMPEG_BIN, "ffmpeg binary");
  ensureLocalFile(WHISPER_CLI_BIN, "Whisper CLI binary");
  ensureLocalFile(WHISPER_MODEL_PATH, "Whisper model");

  const tmpDir = createTempDir("yt-transcribe-");
  const audioTemplate = path.join(tmpDir, "audio.%(ext)s");
  const chunkTemplate = path.join(tmpDir, "chunk_%03d.mp3");

  try {
    try {
      await $`${YTDLP_BIN} --no-warnings -x --audio-format mp3 --audio-quality 128 -o ${audioTemplate} ${videoUrl}`.quiet();
    } catch (error) {
      throw new Error(`yt-dlp audio download failed: ${describeProcessError(error)}`);
    }

    const audioFiles = listMatchingFiles(
      tmpDir,
      (fileName) => fileName.startsWith("audio.") && fileName.endsWith(".mp3")
    );
    const audioFile = audioFiles[0];

    if (!audioFile) {
      throw new Error("yt-dlp did not produce an MP3 file for transcription.");
    }

    try {
      await $`${FFMPEG_BIN} -hide_banner -loglevel error -i ${audioFile} -f segment -segment_time 600 -c copy ${chunkTemplate}`.quiet();
    } catch (error) {
      throw new Error(`ffmpeg audio segmentation failed: ${describeProcessError(error)}`);
    }

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

      try {
        await $`${FFMPEG_BIN} -hide_banner -loglevel error -i ${chunkFile} -ar 16000 -ac 1 ${wavPath}`.quiet();
      } catch (error) {
        throw new Error(
          `ffmpeg WAV conversion failed for chunk ${index + 1}: ${describeProcessError(error)}`
        );
      }

      try {
        await $`${WHISPER_CLI_BIN} --no-prints --no-timestamps --output-txt --output-file ${outputBase} --language en --model ${WHISPER_MODEL_PATH} --file ${wavPath}`.quiet();
      } catch (error) {
        throw new Error(
          `Whisper transcription failed for chunk ${index + 1}: ${describeProcessError(error)}`
        );
      }

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

const pruneBackups = (dbPath: string) => {
  const backupDir = path.resolve(path.dirname(dbPath), "db_backups");
  if (!fs.existsSync(backupDir)) return;

  const extension = path.extname(dbPath) || ".sqlite";
  const baseName = path.basename(dbPath, extension);

  const backups = fs
    .readdirSync(backupDir)
    .filter(
      (fileName) => fileName.startsWith(`${baseName}.`) && fileName.endsWith(extension)
    )
    .sort((a, b) => b.localeCompare(a));

  for (const oldBackup of backups.slice(MAX_DB_BACKUPS)) {
    fs.rmSync(path.join(backupDir, oldBackup), { force: true });
  }
};

const backupDatabaseIfNeeded = (database: DatabaseHandle) => {
  if (!database.hadExistingData || backedUpDatabases.has(database.path)) {
    return;
  }
  if (!fs.existsSync(database.path) || fs.statSync(database.path).size === 0) {
    backedUpDatabases.add(database.path);
    return;
  }

  const backupDir = path.resolve(path.dirname(database.path), "db_backups");
  fs.mkdirSync(backupDir, { recursive: true });

  const extension = path.extname(database.path) || ".sqlite";
  const baseName = path.basename(database.path, extension);
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const backupPath = path.join(backupDir, `${baseName}.${timestamp}${extension}`);

  fs.copyFileSync(database.path, backupPath);
  backedUpDatabases.add(database.path);
  pruneBackups(database.path);
};

async function initDb(): Promise<DatabaseHandle> {
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
  const hadExistingData = fs.existsSync(dbPath) && fs.statSync(dbPath).size > 0;
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

    CREATE TABLE IF NOT EXISTS content_embeddings (
      content_id      TEXT NOT NULL,
      embedding_kind  TEXT NOT NULL,
      model           TEXT NOT NULL,
      source_text     TEXT NOT NULL,
      embedding       TEXT NOT NULL,
      updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
      PRIMARY KEY (content_id, embedding_kind, model),
      FOREIGN KEY (content_id) REFERENCES content(content_id)
    );

    CREATE TABLE IF NOT EXISTS transcript_chunk_embeddings (
      content_id   TEXT NOT NULL,
      model        TEXT NOT NULL,
      chunk_index  INTEGER NOT NULL,
      start_word   INTEGER NOT NULL,
      end_word     INTEGER NOT NULL,
      text         TEXT NOT NULL,
      embedding    TEXT NOT NULL,
      updated_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
      PRIMARY KEY (content_id, model, chunk_index),
      FOREIGN KEY (content_id) REFERENCES content(content_id)
    );

    CREATE INDEX IF NOT EXISTS idx_content_type_created_at
      ON content (content_type, created_at DESC);

    CREATE INDEX IF NOT EXISTS idx_qa_content_created_at
      ON qa (content_id, created_at DESC);
  `);

  return { db, path: dbPath, hadExistingData };
}

const clearCachedEmbeddings = (db: Database, contentId: string) => {
  db.run("DELETE FROM content_embeddings WHERE content_id = ?", [contentId]);
  db.run("DELETE FROM transcript_chunk_embeddings WHERE content_id = ?", [
    contentId,
  ]);
};

const storeQA = (
  database: DatabaseHandle,
  contentId: string,
  question: string,
  answer: string
) => {
  backupDatabaseIfNeeded(database);
  database.db.run(
    "INSERT INTO qa (content_id, question, answer) VALUES (?, ?, ?)",
    [contentId, question, answer]
  );
};

const deleteSession = (
  database: DatabaseHandle,
  contentId: string
): { qaChanges: number; contentChanges: number } => {
  let qaChanges = 0;
  let contentChanges = 0;

  const tx = database.db.transaction(() => {
    qaChanges = database.db.run("DELETE FROM qa WHERE content_id = ?", [
      contentId,
    ]).changes;
    clearCachedEmbeddings(database.db, contentId);
    contentChanges = database.db.run(
      "DELETE FROM content WHERE content_id = ? AND content_type = 'youtube'",
      [contentId]
    ).changes;
  });

  backupDatabaseIfNeeded(database);
  tx();

  return { qaChanges, contentChanges };
};

async function getOrCreateTranscript(
  database: DatabaseHandle,
  url: string,
  options: TranscriptOptions = {}
): Promise<ContentData> {
  const videoId = getVideoId(url);
  if (!videoId) {
    throw new Error("Only YouTube URLs are supported.");
  }

  const existing = toContentData(
    database.db
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

  backupDatabaseIfNeeded(database);

  if (existing) {
    let clearedQaCount = 0;
    const tx = database.db.transaction(() => {
      clearedQaCount = database.db.run("DELETE FROM qa WHERE content_id = ?", [
        videoId,
      ]).changes;
      clearCachedEmbeddings(database.db, videoId);
      database.db.run(
        `UPDATE content
         SET title = ?, author = ?, audio_url = ?, transcript = ?, summary = ?, created_at = CURRENT_TIMESTAMP
         WHERE content_id = ? AND content_type = 'youtube'`,
        [title, "", "", zip(transcript), summary, videoId]
      );
    });
    tx();
    console.log(
      `Re-ran ${videoId} and cleared ${clearedQaCount} cached Q&A entr${clearedQaCount === 1 ? "y" : "ies"}.`
    );
  } else {
    const tx = database.db.transaction(() => {
      database.db.run(
        `INSERT INTO content (content_id, content_type, title, author, audio_url, transcript, summary)
         VALUES (?, ?, ?, ?, ?, ?, ?)`,
        [videoId, "youtube", title, "", "", zip(transcript), summary]
      );
      clearCachedEmbeddings(database.db, videoId);
    });
    tx();
    console.log("Transcript stored for", videoId);
  }

  return {
    contentId: videoId,
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
    getOpenAIClient().chat.completions.create({
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
    `You are a careful transcript summarizer${chunkInfo}. Your job is faithful compression, not creative interpretation.

Rules:
- Use only information explicitly supported by the transcript.
- Do not invent quotes, names, titles, books, papers, tools, companies, or recommendations.
- Preserve technical terms and proper nouns exactly as written when they are clear.
- If the transcript is ambiguous, noisy, or incomplete, say so briefly instead of guessing.
- Quotes must be exact text from the transcript. If no short exact quote stands out, write "None".
- Recommendations belong in the final section only if the speaker clearly gives advice, steps, or actions.
- Distinguish speaker opinions or claims from established facts when the wording makes that distinction clear.

Return Markdown with exactly these sections:

## Main Points
- 3-5 bullets covering the most important ideas in this part
- Each bullet should include concrete supporting detail, examples, or data when present

## Evidence & Examples
- 2-4 bullets with notable supporting details, examples, caveats, disagreements, or non-obvious claims that matter for understanding this part
- Write "None" if this part is too thin or repetitive to support the section

## Exact Quotes
- 0-2 bullets with short exact quotes copied from the transcript
- Write "None" if there is no strong quote

## People & References
- Bulleted list of clearly mentioned people, books, papers, companies, products, or tools
- Write "None" if nothing specific is clearly named

## Explicit Recommendations
- Bulleted list of advice, steps, or actions clearly stated by the speaker
- Write "None" if the speaker does not give explicit recommendations`,
    `Analyze this transcript excerpt. Keep the summary faithful to this excerpt only.\n\n${chunk}`
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
    `You are synthesizing analyses from different parts of one long YouTube transcript into a final faithful summary.

Rules:
- Use only information present in the part analyses.
- Deduplicate overlap without flattening away important one-off details.
- Preserve important caveats, disagreements, and conditions when they materially change interpretation.
- Do not elevate speculation into fact.
- Keep exact quotes exactly as provided. If the quote quality is weak or unsupported, omit it.
- Only include recommendations that are clearly stated by the speaker.
- Prefer accuracy and signal over completeness. If a section is unsupported, write "None".

Return Markdown with exactly these sections:

## Overall Summary
- 1 short paragraph capturing the central thesis or purpose of the video

## Main Points
- 4-8 bullets covering the most important ideas across the full transcript
- Include specific supporting detail where available

## Important Details
- 3-6 bullets with notable examples, evidence, caveats, disagreements, or rare but important details that should not be lost
- Write "None" only if the transcript is unusually repetitive and adds no material details

## Exact Quotes
- 0-3 bullets with the strongest exact quotes from the transcript
- Write "None" if there are no strong quotes

## People & References
- Consolidated bulleted list of clearly mentioned people, books, papers, companies, products, or tools
- Write "None" if nothing specific is clearly named

## Explicit Recommendations
- Consolidated bulleted list of advice, steps, or actions clearly stated by the speaker
- Write "None" if the speaker does not give explicit recommendations`,
    `Synthesize these part analyses into one final summary:\n\n${chunkSummaries
      .map((summary, index) => `=== Part ${index + 1} ===\n${summary}`)
      .join("\n\n")}`
  );
}

async function getEmbeddings(texts: string[]): Promise<number[][]> {
  if (texts.length === 0) return [];

  const embeddings = texts.map((): number[] => []);
  const nonEmptyInputs = texts
    .map((text, index) => ({ index, text: text.trim() }))
    .filter((item) => item.text.length > 0);

  for (let index = 0; index < nonEmptyInputs.length; index += EMBEDDING_BATCH_SIZE) {
    const batch = nonEmptyInputs.slice(index, index + EMBEDDING_BATCH_SIZE);
    const response = await retryWithBackoff(() =>
      getOpenAIClient().embeddings.create({
        model: PROVIDERS.EMBEDDING_MODEL,
        input: batch.map((item) => item.text),
      })
    );

    if (response.data.length !== batch.length) {
      throw new Error(
        `Expected ${batch.length} embeddings, received ${response.data.length}.`
      );
    }

    response.data.forEach((item, batchIndex) => {
      embeddings[batch[batchIndex].index] = item.embedding ?? [];
    });
  }

  return embeddings;
}

async function getEmbedding(text: string): Promise<number[]> {
  const [embedding] = await getEmbeddings([text]);
  return embedding ?? [];
}

async function getOrCreateFindEmbeddings(
  database: DatabaseHandle,
  rows: ContentData[]
): Promise<number[][]> {
  if (rows.length === 0) return [];

  const storedRows = database.db
    .query(
      `SELECT content_id, source_text, embedding
       FROM content_embeddings
       WHERE embedding_kind = ? AND model = ?`
    )
    .all(FIND_EMBEDDING_KIND, PROVIDERS.EMBEDDING_MODEL) as StoredContentEmbeddingRow[];

  const storedById = new Map(storedRows.map((row) => [row.content_id, row]));
  const embeddingsById = new Map<string, number[]>();
  const pending = rows
    .map((row) => ({
      contentId: row.contentId,
      sourceText: getSearchText(row),
      stored: storedById.get(row.contentId),
    }))
    .filter((entry) => {
      const cached = entry.stored ? decodeEmbedding(entry.stored.embedding) : [];
      if (
        entry.stored &&
        entry.stored.source_text === entry.sourceText &&
        cached.length > 0
      ) {
        embeddingsById.set(entry.contentId, cached);
        return false;
      }
      return true;
    });

  if (pending.length > 0) {
    const freshEmbeddings = await getEmbeddings(
      pending.map((entry) => entry.sourceText)
    );

    const tx = database.db.transaction(
      (entries: Array<{ contentId: string; sourceText: string; embedding: number[] }>) => {
        for (const entry of entries) {
          database.db.run(
            `INSERT INTO content_embeddings (
               content_id,
               embedding_kind,
               model,
               source_text,
               embedding,
               updated_at
             )
             VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
             ON CONFLICT(content_id, embedding_kind, model)
             DO UPDATE SET
               source_text = excluded.source_text,
               embedding = excluded.embedding,
               updated_at = CURRENT_TIMESTAMP`,
            [
              entry.contentId,
              FIND_EMBEDDING_KIND,
              PROVIDERS.EMBEDDING_MODEL,
              entry.sourceText,
              encodeEmbedding(entry.embedding),
            ]
          );
        }
      }
    );

    backupDatabaseIfNeeded(database);
    tx(
      pending.map((entry, index) => ({
        contentId: entry.contentId,
        sourceText: entry.sourceText,
        embedding: freshEmbeddings[index] ?? [],
      }))
    );

    pending.forEach((entry, index) => {
      embeddingsById.set(entry.contentId, freshEmbeddings[index] ?? []);
    });
  }

  return rows.map((row) => embeddingsById.get(row.contentId) ?? []);
}

async function buildQaIndex(
  database: DatabaseHandle,
  contentId: string,
  transcript: string
): Promise<RagIndex> {
  const chunks = chunkText(transcript, QA_CHUNK_CONFIG);
  if (chunks.length === 0) {
    return { chunks: [], embeddings: [] };
  }

  const storedRows = database.db
    .query(
      `SELECT chunk_index, start_word, end_word, text, embedding
       FROM transcript_chunk_embeddings
       WHERE content_id = ? AND model = ?
       ORDER BY chunk_index ASC`
    )
    .all(contentId, PROVIDERS.EMBEDDING_MODEL) as StoredChunkEmbeddingRow[];

  const storedEmbeddings = storedRows.map((row) => decodeEmbedding(row.embedding));
  const canReuse =
    storedRows.length === chunks.length &&
    storedRows.every((row, index) => {
      const chunk = chunks[index];
      return (
        row.chunk_index === chunk.index &&
        row.start_word === chunk.startWord &&
        row.end_word === chunk.endWord &&
        row.text === chunk.text &&
        storedEmbeddings[index].length > 0
      );
    });

  if (canReuse) {
    return { chunks, embeddings: storedEmbeddings };
  }

  const index = await buildRagIndex(transcript, getEmbeddings, QA_CHUNK_CONFIG);
  const tx = database.db.transaction((entries: PersistedChunkEmbedding[]) => {
    database.db.run(
      "DELETE FROM transcript_chunk_embeddings WHERE content_id = ? AND model = ?",
      [contentId, PROVIDERS.EMBEDDING_MODEL]
    );

    for (const entry of entries) {
      database.db.run(
        `INSERT INTO transcript_chunk_embeddings (
           content_id,
           model,
           chunk_index,
           start_word,
           end_word,
           text,
           embedding,
           updated_at
         )
         VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)`,
        [
          contentId,
          PROVIDERS.EMBEDDING_MODEL,
          entry.chunk.index,
          entry.chunk.startWord,
          entry.chunk.endWord,
          entry.chunk.text,
          encodeEmbedding(entry.embedding),
        ]
      );
    }
  });

  backupDatabaseIfNeeded(database);
  tx(
    index.chunks.map((chunk, indexPosition) => ({
      chunk,
      embedding: index.embeddings[indexPosition] ?? [],
    }))
  );

  return index;
}

async function answerQuestion(index: RagIndex, question: string): Promise<string> {
  const contextChunks = await selectRelevantChunks(
    index,
    question,
    getEmbedding,
    QA_CONTEXT_CHUNKS,
    QA_MIN_RELEVANCE_SCORE
  );

  if (contextChunks.length === 0) {
    return UNSUPPORTED_ANSWER;
  }

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
  database: DatabaseHandle,
  narrative: string
): Promise<ContentData> {
  const rows = (
    database.db
      .query("SELECT * FROM content WHERE content_type = 'youtube'")
      .all() as StoredContentRow[]
  )
    .map((row) => toContentData(row))
    .filter((row): row is ContentData => Boolean(row));

  if (rows.length === 0) throw new Error("No YouTube sessions saved.");

  const queryEmbedding = await getEmbedding(narrative);
  if (queryEmbedding.length === 0) {
    throw new Error("Could not embed the search query.");
  }

  const summaryEmbeddings = await getOrCreateFindEmbeddings(database, rows);

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

  const database = await initDb();
  const { db } = database;

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
        .get(videoId) as { title: string | null } | null;

      if (!existing) {
        console.log(`No YouTube entry found for: ${videoId}`);
        return;
      }

      const result = deleteSession(database, videoId);
      console.log(`Deleted "${existing.title || videoId}"`);
      console.log(`  - Removed ${result.qaChanges} Q&A entries`);
      console.log(`  - Removed ${result.contentChanges} content entry`);
      return;
    }

    if (values.url) {
      data = await getOrCreateTranscript(database, values.url.trim());
      console.log(`\nTitle: ${data.title}`);
      console.log(`\nSummary:\n${data.summary}\n`);
      return;
    }

    if (values.rerun) {
      data = await getOrCreateTranscript(database, values.rerun.trim(), {
        forceRefresh: true,
      });
      console.log(`\nTitle: ${data.title}`);
      console.log(`\nSummary:\n${data.summary}\n`);
      return;
    }

    if (values.find) {
      console.log(`Searching for: "${values.find}"...`);
      data = await restoreSession(database, values.find);
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
        title: string | null;
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
              name: `${session.title || session.content_id} [${session.created}]`,
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
          .filter((session) =>
            (session.title || session.content_id).toLowerCase().includes(lowerTerm)
          )
          .map((session) => ({
            name: `${session.title || session.content_id} [${session.created}]`,
            value: session.content_id,
          }));
      },
      pageSize: 12,
    });

    if (pick === "__new") {
      if (lastTerm.startsWith("http") || lastTerm.startsWith("www")) {
        data = await getOrCreateTranscript(database, lastTerm.trim());
      } else {
        const url = await input({ message: "Enter YouTube URL:" });
        data = await getOrCreateTranscript(database, url.trim());
      }
    } else if (pick.startsWith("__url:")) {
      data = await getOrCreateTranscript(database, pick.slice(6).trim());
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
        .all(data.contentId) as QARow[];

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

        const result = deleteSession(database, data.contentId);
        console.log(`\nDeleted "${data.title}"`);
        console.log(`  - Removed ${result.qaChanges} Q&A entries`);
        console.log(`  - Removed ${result.contentChanges} content entry\n`);
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
          database,
          `https://www.youtube.com/watch?v=${data.contentId}`,
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
          const filename = await exportQA(db, data.contentId, data.title, format);
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
        const trimmedQuestion = question.trim();
        if (!trimmedQuestion) continue;

        if (!qaIndexPromise) {
          console.log("Indexing transcript for Q&A...");
          qaIndexPromise = buildQaIndex(database, data.contentId, data.transcript);
        }

        const answer = await answerQuestion(await qaIndexPromise, trimmedQuestion);
        console.log(`\nAnswer: ${answer}\n`);
        storeQA(database, data.contentId, trimmedQuestion, answer);
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
