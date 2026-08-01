export const sanitizeColorEnv = (): void => {
  if (process.env.NO_COLOR !== undefined && process.env.FORCE_COLOR !== undefined) {
    delete process.env.FORCE_COLOR;
  }
};

sanitizeColorEnv();

import type { Database } from "bun:sqlite";
import { $ } from "bun";
import { confirm, expand, input, search, select } from "@inquirer/prompts";
import axios from "axios";
import * as dotenv from "dotenv";
import { spawnSync } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { parseArgs } from "node:util";
import OpenAI from "openai";

import { getAppConfig } from "./config.ts";
import {
  backupDatabaseIfNeeded,
  clearCachedEmbeddings,
  decodeTranscript,
  encodeTranscript,
  openDatabase,
  type DatabaseHandle,
} from "./database.ts";

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
import {
  exportSummaryHtml,
  formatSummaryForTerminal,
  openSummaryInBrowser,
  type SummaryDocument,
} from "./summary-output.ts";

dotenv.config({ quiet: true });

const isUsableExecutable = (candidate: string, validationArgs: string[]): boolean => {
  try {
    fs.accessSync(candidate, fs.constants.X_OK);
  } catch {
    return false;
  }

  if (validationArgs.length === 0) return true;

  const result = spawnSync(candidate, validationArgs, {
    stdio: "ignore",
    timeout: 5000,
  });
  return result.status === 0;
};

const findExecutable = (name: string, validationArgs: string[] = []): string | null => {
  const candidates = [
    ...new Set([
      ...((process.env.PATH || "")
        .split(path.delimiter)
        .filter(Boolean)
        .map((dirPath) => path.join(dirPath, name))),
      path.join(process.env.HOME || "", ".darkbloom", "bin", name),
      `/opt/homebrew/bin/${name}`,
      `/usr/local/bin/${name}`,
    ]),
  ];

  for (const candidate of candidates) {
    if (isUsableExecutable(candidate, validationArgs)) {
      return candidate;
    }
  }

  return null;
};

const resolveConfiguredExecutable = (
  configuredValue: string | undefined,
  defaultName: string,
  validationArgs: string[] = []
): string => {
  const configured = configuredValue?.trim();
  if (configured) {
    return configured.includes(path.sep)
      ? configured
      : findExecutable(configured, validationArgs) || configured;
  }

  return findExecutable(defaultName, validationArgs) || defaultName;
};

const YTDLP_BIN = resolveConfiguredExecutable(process.env.YTDLP_BIN, "yt-dlp", [
  "--version",
]);
const FFMPEG_BIN = resolveConfiguredExecutable(process.env.FFMPEG_BIN, "ffmpeg", [
  "-version",
]);
const WHISPER_CLI_BIN =
  process.env.WHISPER_CLI_BIN || "./whisper.cpp/build/bin/whisper-cli";
const WHISPER_MODEL_PATH =
  process.env.WHISPER_MODEL_PATH || "./whisper.cpp/models/ggml-base.en.bin";

const DEFAULT_SUMMARY_CHUNK_CONCURRENCY = 3;
const MAX_SUMMARY_CHUNK_CONCURRENCY = 8;

export const getSummaryChunkConcurrency = (
  rawValue = process.env.SUMMARY_CHUNK_CONCURRENCY
): number => {
  const parsed = Number.parseInt(rawValue?.trim() ?? "", 10);
  if (!Number.isFinite(parsed) || parsed < 1) {
    return DEFAULT_SUMMARY_CHUNK_CONCURRENCY;
  }

  return Math.min(parsed, MAX_SUMMARY_CHUNK_CONCURRENCY);
};

const appConfig = getAppConfig();
const PROVIDERS = {
  SUMMARY_MODEL: appConfig.summaryModel,
  QA_MODEL: appConfig.qaModel,
  EMBEDDING_MODEL: appConfig.embeddingModel,
};
const TRANSCRIPT_LANGUAGE = appConfig.transcriptLanguage;

const SUMMARY_CHUNK_CONFIG = { maxWords: 4000, overlapWords: 180 };
const SUMMARY_MAX_CHARS = 22000;
const SUMMARY_RETRY_SPLIT_MIN_WORDS = 800;
const SUMMARY_RETRY_SPLIT_OVERLAP_WORDS = 80;
const SUMMARY_CHUNK_CONCURRENCY = getSummaryChunkConcurrency();
const QA_CHUNK_CONFIG = { maxWords: 1200, overlapWords: 150 };
const QA_CONTEXT_CHUNKS = 4;
const QA_MIN_RELEVANCE_SCORE = 0.2;
const FIND_MATCH_THRESHOLD = 0.8;
const FIND_EMBEDDING_KIND = "find_summary";
const EMBEDDING_BATCH_SIZE = 16;
const UNSUPPORTED_ANSWER =
  "I couldn't find support for that in the retrieved transcript excerpts.";
const NO_CAPTION_PATTERNS = [
  /there are no subtitles/i,
  /has no subtitles/i,
  /requested subtitles.*not available/i,
  /requested languages?.*not available/i,
  /no automatic captions/i,
  /subtitles are not available/i,
  /video doesn't have subtitles/i,
];

const SUPPORTED_CONTENT_TYPES = ["youtube", "vimeo", "x"] as const;
type ContentType = (typeof SUPPORTED_CONTENT_TYPES)[number];


interface ContentData {
  contentId: string;
  contentType: ContentType;
  sourceId: string;
  sourceUrl: string;
  title: string;
  transcript: string;
  summary: string;
  createdAt?: string;
}

interface StoredContentRow {
  content_id: string;
  content_type: string;
  title: string | null;
  audio_url: string | null;
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


const clearTerminal = () => {
  if (!process.stdout.isTTY) return;
  process.stdout.write("\x1b[2J\x1b[3J\x1b[H");
};

const printSummary = (data: ContentData, options: { clearBefore?: boolean } = {}) => {
  if (options.clearBefore) clearTerminal();
  const terminalWidth = process.stdout.columns
    ? Math.max(40, process.stdout.columns - 2)
    : 88;
  const useColor = Boolean(process.stdout.isTTY && process.env.NO_COLOR === undefined);
  console.log(
    `\n${formatSummaryForTerminal(data.title, data.summary, {
      width: terminalWidth,
      useColor,
    })}\n`
  );
};

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

interface OpenAIErrorDetails {
  status?: number;
  code?: string;
  type?: string;
  message: string;
  requestId?: string;
}

class OpenAIRequestError extends Error {
  details: OpenAIErrorDetails;

  constructor(details: OpenAIErrorDetails) {
    super(`OpenAI API Error: ${formatOpenAIError(details)}`);
    this.name = "OpenAIRequestError";
    this.details = details;
  }
}

const getOpenAIClient = (): OpenAI => {
  const apiKey = process.env.OPENAI_API_KEY?.trim();
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY is required for this action.");
  }

  return new OpenAI({
    apiKey,
    maxRetries: 0,
    organization: null,
    project: null,
  });
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

async function mapWithConcurrency<T, R>(
  items: T[],
  concurrency: number,
  mapper: (item: T, index: number) => Promise<R>
): Promise<R[]> {
  if (items.length === 0) return [];

  const results = new Array<R>(items.length);
  const workerCount = Math.min(Math.max(1, Math.floor(concurrency)), items.length);
  let nextIndex = 0;

  const workers = Array.from({ length: workerCount }, async () => {
    while (true) {
      const currentIndex = nextIndex;
      nextIndex += 1;

      if (currentIndex >= items.length) return;

      results[currentIndex] = await mapper(items[currentIndex], currentIndex);
    }
  });

  await Promise.all(workers);
  return results;
}

const getHeaderValue = (
  headers: Headers | Record<string, string> | undefined,
  key: string
): string | undefined => {
  if (!headers) return undefined;
  if (headers instanceof Headers) {
    return headers.get(key) ?? undefined;
  }

  const match = Object.entries(headers).find(
    ([headerKey]) => headerKey.toLowerCase() === key.toLowerCase()
  );
  return match?.[1];
};

const getOpenAIErrorDetails = (error: unknown): OpenAIErrorDetails => {
  const candidate = error as {
    status?: number;
    statusCode?: number;
    code?: string;
    type?: string;
    message?: string;
    request_id?: string;
    headers?: Headers | Record<string, string>;
    error?: {
      code?: string;
      type?: string;
      message?: string;
    };
    response?: {
      headers?: Headers | Record<string, string>;
      data?: {
        error?: {
          code?: string;
          type?: string;
          message?: string;
        };
      };
    };
  };

  const responseError = candidate.response?.data?.error;
  const nestedError = candidate.error;
  const status = candidate.status ?? candidate.statusCode;
  const message =
    responseError?.message ||
    nestedError?.message ||
    candidate.message ||
    (error instanceof Error ? error.message : String(error));
  const code = responseError?.code || nestedError?.code || candidate.code;
  const type = responseError?.type || nestedError?.type || candidate.type;
  const requestId =
    candidate.request_id ||
    getHeaderValue(candidate.headers, "x-request-id") ||
    getHeaderValue(candidate.response?.headers, "x-request-id");

  return {
    status,
    code,
    type,
    message,
    requestId,
  };
};

const formatOpenAIError = (details: OpenAIErrorDetails): string =>
  [
    details.message.trim(),
    details.code ? `code=${details.code}` : "",
    details.type ? `type=${details.type}` : "",
    details.status ? `status=${details.status}` : "",
    details.requestId ? `request_id=${details.requestId}` : "",
  ]
    .filter(Boolean)
    .join(" | ");

const isOpenAIRequestTooLarge = (details: OpenAIErrorDetails): boolean => {
  const message = details.message.toLowerCase();
  return (
    details.status === 431 ||
    details.code === "request_headers_too_large" ||
    message.includes("too large") ||
    message.includes("context length") ||
    message.includes("too many tokens") ||
    message.includes("maximum context")
  );
};

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

const getYtDlpFfmpegLocation = (): string =>
  FFMPEG_BIN.includes(path.sep) ? path.dirname(FFMPEG_BIN) : FFMPEG_BIN;

const normalizeTranscriptText = (text: string): string =>
  text.replace(/\r/g, "").replace(/\n{3,}/g, "\n\n").trim();

interface VttCue {
  text: string;
  words: string[];
}

const cleanVttCueText = (lines: string[]): string =>
  lines
    .join(" ")
    .replace(/<[^>]+>/g, " ")
    .replace(/\s+/g, " ")
    .trim();

const parseVttCues = (raw: string): VttCue[] =>
  raw
    .replace(/\r/g, "")
    .split(/\n{2,}/)
    .flatMap((block) => {
      const lines = block.split("\n");
      const timingLineIndex = lines.findIndex((line) => line.includes("-->"));
      if (timingLineIndex < 0) return [];

      const text = cleanVttCueText(lines.slice(timingLineIndex + 1));
      return text ? [{ text, words: text.split(/\s+/) }] : [];
    });

const findRollingCueOverlap = (
  transcriptWords: string[],
  cueWords: string[]
): number => {
  const maxOverlap = Math.min(transcriptWords.length, cueWords.length);

  for (let overlap = maxOverlap; overlap > 0; overlap -= 1) {
    const transcriptStart = transcriptWords.length - overlap;
    let matches = true;

    for (let index = 0; index < overlap; index += 1) {
      if (transcriptWords[transcriptStart + index] !== cueWords[index]) {
        matches = false;
        break;
      }
    }

    if (matches) return overlap;
  }

  return 0;
};

export const extractVttText = (raw: string): string => {
  const cues = parseVttCues(raw);
  const usesRollingCues = /<\d{2}:\d{2}(?::\d{2})?\.\d{3}>/.test(raw);

  if (!usesRollingCues) {
    return cues.map((cue) => cue.text).join(" ").trim();
  }

  const transcriptWords: string[] = [];
  for (const cue of cues) {
    const overlap = findRollingCueOverlap(transcriptWords, cue.words);
    transcriptWords.push(...cue.words.slice(overlap));
  }

  return transcriptWords.join(" ").trim();
};

export const getVideoId = (input: string): string => {
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
    /(?:youtu\.be\/|youtube\.com\/(?:embed|shorts|live)\/|\/(?:embed|shorts|live)\/)([A-Za-z0-9_-]{11})/
  );
  return fallbackMatch?.[1] ?? "";
};

const getVimeoId = (input: string): string => {
  const trimmed = input.trim();
  if (/^\d{6,}$/.test(trimmed)) return trimmed;

  const withProtocol = /^https?:\/\//i.test(trimmed) ? trimmed : `https://${trimmed}`;

  try {
    const url = new URL(withProtocol);
    const hostname = url.hostname.replace(/^www\./, "");
    const parts = url.pathname.split("/").filter(Boolean);

    if (hostname === "vimeo.com") {
      const numericPart = parts.find((part) => /^\d+$/.test(part));
      return numericPart ?? "";
    }

    if (hostname === "player.vimeo.com") {
      const marker = parts.findIndex((part) => part === "video");
      const candidate = marker >= 0 ? parts[marker + 1] ?? "" : "";
      return /^\d+$/.test(candidate) ? candidate : "";
    }
  } catch {
    // Fall back to the regex below.
  }

  const fallbackMatch = trimmed.match(
    /(?:vimeo\.com\/(?:.*\/)?|player\.vimeo\.com\/video\/)(\d+)/
  );
  return fallbackMatch?.[1] ?? "";
};

const getXPostId = (input: string): string => {
  const trimmed = input.trim();
  const withProtocol = /^https?:\/\//i.test(trimmed) ? trimmed : `https://${trimmed}`;

  try {
    const url = new URL(withProtocol);
    const hostname = url.hostname.replace(/^www\./, "");
    const parts = url.pathname.split("/").filter(Boolean);

    if (
      hostname === "x.com" ||
      hostname === "twitter.com" ||
      hostname === "mobile.twitter.com"
    ) {
      const marker = parts.findIndex((part) =>
        ["status", "statuses"].includes(part)
      );
      const candidate = marker >= 0 ? parts[marker + 1] ?? "" : "";
      return /^\d{10,}$/.test(candidate) ? candidate : "";
    }
  } catch {
    // Fall back to the regex below.
  }

  const fallbackMatch = trimmed.match(
    /(?:x\.com|twitter\.com|mobile\.twitter\.com)\/[^/]+\/status(?:es)?\/(\d{10,})/
  );
  return fallbackMatch?.[1] ?? "";
};

const isContentType = (value: string): value is ContentType =>
  SUPPORTED_CONTENT_TYPES.includes(value as ContentType);

const getContentTypeLabel = (contentType: ContentType): string => {
  switch (contentType) {
    case "youtube":
      return "YouTube";
    case "vimeo":
      return "Vimeo";
    case "x":
      return "X";
  }
};

const getStorageContentId = (contentType: ContentType, sourceId: string): string =>
  contentType === "youtube" ? sourceId : `${contentType}:${sourceId}`;

const getSourceIdFromStorage = (
  contentId: string,
  contentType: ContentType
): string => {
  const prefix = `${contentType}:`;
  return contentId.startsWith(prefix) ? contentId.slice(prefix.length) : contentId;
};

interface ParsedContentInput {
  contentType: ContentType;
  sourceId: string;
  contentId: string;
  canonicalUrl: string;
}

export const parseContentInput = (input: string): ParsedContentInput | null => {
  const youtubeId = getVideoId(input);
  if (youtubeId) {
    return {
      contentType: "youtube",
      sourceId: youtubeId,
      contentId: getStorageContentId("youtube", youtubeId),
      canonicalUrl: `https://www.youtube.com/watch?v=${youtubeId}`,
    };
  }

  const vimeoId = getVimeoId(input);
  if (vimeoId) {
    return {
      contentType: "vimeo",
      sourceId: vimeoId,
      contentId: getStorageContentId("vimeo", vimeoId),
      canonicalUrl: `https://vimeo.com/${vimeoId}`,
    };
  }

  const xPostId = getXPostId(input);
  if (xPostId) {
    return {
      contentType: "x",
      sourceId: xPostId,
      contentId: getStorageContentId("x", xPostId),
      canonicalUrl: `https://x.com/i/status/${xPostId}`,
    };
  }

  return null;
};

const getContentUrl = (data: ContentData): string => {
  if (data.sourceUrl) return data.sourceUrl;

  if (data.contentType === "youtube") {
    return `https://www.youtube.com/watch?v=${data.sourceId}`;
  }
  if (data.contentType === "vimeo") {
    return `https://vimeo.com/${data.sourceId}`;
  }
  return `https://x.com/i/status/${data.sourceId}`;
};

const getSummaryDocument = (data: ContentData): SummaryDocument => ({
  contentId: data.contentId,
  title: data.title,
  summary: data.summary,
  sourceUrl: getContentUrl(data),
});

const getFetchableUrl = (input: string, content: ParsedContentInput): string => {
  const trimmed = input.trim();
  return /^(https?:\/\/|www\.)/i.test(trimmed) ? trimmed : content.canonicalUrl;
};

const formatProviderLabel = (contentType: string): string =>
  isContentType(contentType) ? getContentTypeLabel(contentType) : "Video";

const isNoCaptionFailure = (output: string): boolean =>
  NO_CAPTION_PATTERNS.some((pattern) => pattern.test(output));

const getSearchText = (row: ContentData): string =>
  row.summary.trim() || row.title.trim() || row.contentId;

const toContentData = (row: StoredContentRow | null): ContentData | null => {
  if (!row) return null;
  const contentType = isContentType(row.content_type) ? row.content_type : "youtube";
  const sourceId = getSourceIdFromStorage(row.content_id, contentType);
  const providerLabel = getContentTypeLabel(contentType);

  return {
    contentId: row.content_id,
    contentType,
    sourceId,
    sourceUrl: row.audio_url?.trim() || "",
    title: row.title?.trim() || `${providerLabel} video ${sourceId}`,
    transcript: decodeTranscript(row.transcript),
    summary: row.summary?.trim() || "",
    createdAt: row.created_at,
  };
};

async function fetchContentTitle(content: ParsedContentInput): Promise<string> {
  const providerLabel = getContentTypeLabel(content.contentType);
  try {
    const url = (() => {
      switch (content.contentType) {
        case "youtube":
          return `https://www.youtube.com/oembed?url=${encodeURIComponent(
            content.canonicalUrl
          )}&format=json`;
        case "vimeo":
          return `https://vimeo.com/api/oembed.json?url=${encodeURIComponent(
            content.canonicalUrl
          )}`;
        case "x":
          return `https://publish.twitter.com/oembed?url=${encodeURIComponent(
            content.canonicalUrl
          )}`;
      }
    })();
    const { data } = await axios.get<{ title?: string; author_name?: string }>(url);
    return (
      data.title?.trim() ||
      data.author_name?.trim() ||
      `${providerLabel} video ${content.sourceId}`
    );
  } catch {
    return `${providerLabel} video ${content.sourceId}`;
  }
}

async function fetchCaptionsWithYtDlp(
  videoUrl: string,
  sourceId: string
): Promise<string | null> {
  ensureLocalFile(YTDLP_BIN, "yt-dlp binary");

  const tmpDir = createTempDir("yt-captions-");
  const outputTemplate = path.join(tmpDir, "%(id)s.%(ext)s");

  try {
    await $`${YTDLP_BIN} --no-warnings --skip-download --write-sub --write-auto-sub --sub-lang ${TRANSCRIPT_LANGUAGE} --sub-format vtt -o ${outputTemplate} ${videoUrl}`.quiet();

    const allSubtitleFiles = listMatchingFiles(tmpDir, (fileName) =>
      fileName.endsWith(".vtt")
    );
    const sourceSubtitleFiles = allSubtitleFiles.filter((filePath) =>
      path.basename(filePath).startsWith(sourceId)
    );
    const candidates = (
      sourceSubtitleFiles.length > 0 ? sourceSubtitleFiles : allSubtitleFiles
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

async function transcribeVideoWithWhisper(videoUrl: string): Promise<string> {
  ensureLocalFile(YTDLP_BIN, "yt-dlp binary");
  ensureLocalFile(FFMPEG_BIN, "ffmpeg binary");
  ensureLocalFile(WHISPER_CLI_BIN, "Whisper CLI binary");
  ensureLocalFile(WHISPER_MODEL_PATH, "Whisper model");

  const tmpDir = createTempDir("video-transcribe-");
  const audioTemplate = path.join(tmpDir, "audio.%(ext)s");
  const chunkTemplate = path.join(tmpDir, "chunk_%03d.mp3");

  try {
    try {
      await $`${YTDLP_BIN} --no-warnings --ffmpeg-location ${getYtDlpFfmpegLocation()} -x --audio-format mp3 --audio-quality 128 -o ${audioTemplate} ${videoUrl}`.quiet();
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
        await $`${WHISPER_CLI_BIN} --no-prints --no-timestamps --output-txt --output-file ${outputBase} --language ${TRANSCRIPT_LANGUAGE} --model ${WHISPER_MODEL_PATH} --file ${wavPath}`.quiet();
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
      "DELETE FROM content WHERE content_id = ?",
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
  const content = parseContentInput(url);
  if (!content) {
    throw new Error("Only YouTube, Vimeo, and X URLs are supported.");
  }
  const fetchableUrl = getFetchableUrl(url, content);

  const existing = toContentData(
    database.db
      .query(
        "SELECT * FROM content WHERE content_id = ? AND content_type = ?"
      )
      .get(content.contentId, content.contentType) as StoredContentRow | null
  );

  if (existing && !options.forceRefresh) return existing;

  const titlePromise = fetchContentTitle(content);

  let transcript = await fetchCaptionsWithYtDlp(fetchableUrl, content.sourceId);
  if (!transcript) {
    transcript = await transcribeVideoWithWhisper(fetchableUrl);
  }

  const title = await titlePromise;
  const summary = await summarizeTranscript(transcript);

  backupDatabaseIfNeeded(database);

  if (existing) {
    let clearedQaCount = 0;
    const tx = database.db.transaction(() => {
      clearedQaCount = database.db.run("DELETE FROM qa WHERE content_id = ?", [
        content.contentId,
      ]).changes;
      clearCachedEmbeddings(database.db, content.contentId);
      database.db.run(
        `UPDATE content
         SET title = ?, author = ?, audio_url = ?, transcript = ?, summary = ?, created_at = CURRENT_TIMESTAMP
         WHERE content_id = ? AND content_type = ?`,
        [
          title,
          "",
          fetchableUrl,
          encodeTranscript(transcript),
          summary,
          content.contentId,
          content.contentType,
        ]
      );
    });
    tx();
    console.log(
      `Re-ran ${content.contentId} and cleared ${clearedQaCount} cached Q&A entr${clearedQaCount === 1 ? "y" : "ies"}.`
    );
  } else {
    const tx = database.db.transaction(() => {
      database.db.run(
        `INSERT INTO content (content_id, content_type, title, author, audio_url, transcript, summary)
         VALUES (?, ?, ?, ?, ?, ?, ?)`,
        [
          content.contentId,
          content.contentType,
          title,
          "",
          fetchableUrl,
          encodeTranscript(transcript),
          summary,
        ]
      );
      clearCachedEmbeddings(database.db, content.contentId);
    });
    tx();
    console.log("Transcript stored for", content.contentId);
  }

  return {
    contentId: content.contentId,
    contentType: content.contentType,
    sourceId: content.sourceId,
    sourceUrl: fetchableUrl,
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
      const details = getOpenAIErrorDetails(error);
      const message = details.message;
      const isStatus429 = details.status === 429;
      const isInsufficientQuota =
        details.code === "insufficient_quota" ||
        details.type === "insufficient_quota" ||
        /insufficient_quota/i.test(message);
      const isRateLimit =
        !isInsufficientQuota &&
        (details.code === "rate_limit_exceeded" ||
          /rate limit/i.test(message) ||
          /too many requests/i.test(message) ||
          isStatus429);
      const isTimeout =
        message.toLowerCase().includes("timeout") ||
        message.includes("AbortError");

      if ((!isRateLimit && !isTimeout) || attempt === maxRetries - 1) {
        throw new OpenAIRequestError(details);
      }

      const backoffMs = initialDelayMs * 2 ** attempt + Math.random() * 1000;
      console.log(
        `${isTimeout ? "Timeout" : "Rate limited"} (${formatOpenAIError(details)}). Waiting ${Math.round(
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
    getOpenAIClient().responses.create({
      model,
      instructions: system,
      input: prompt,
      store: false,
    })
  );

  return response.output_text?.trim() || "";
}

const splitByCharacterBudget = (
  chunk: TextChunk,
  maxChars: number
): TextChunk[] => {
  if (chunk.text.length <= maxChars) return [chunk];

  const words = chunk.text.split(/\s+/).filter(Boolean);
  if (words.length === 0) return [];

  const safeMaxChars = Math.max(1, maxChars);
  const wordLengthPrefix = [0];
  for (const word of words) {
    wordLengthPrefix.push(wordLengthPrefix[wordLengthPrefix.length - 1] + word.length);
  }

  const rangeCharLength = (start: number, end: number): number =>
    wordLengthPrefix[end] - wordLengthPrefix[start] + Math.max(0, end - start - 1);

  const partCount = Math.max(1, Math.ceil(rangeCharLength(0, words.length) / safeMaxChars));
  const chunks: TextChunk[] = [];
  let startOffset = 0;

  while (startOffset < words.length) {
    const remainingParts = Math.max(1, partCount - chunks.length);
    const targetChars = Math.ceil(
      rangeCharLength(startOffset, words.length) / remainingParts
    );
    let endOffset = startOffset;
    let currentLength = 0;

    while (endOffset < words.length) {
      const nextLength =
        currentLength +
        (endOffset > startOffset ? 1 : 0) +
        words[endOffset].length;

      if (currentLength > 0 && nextLength > safeMaxChars) break;

      currentLength = nextLength;
      endOffset += 1;

      const remainingWords = words.length - endOffset;
      const remainingSlots = remainingParts - 1;
      if (
        remainingSlots > 0 &&
        remainingWords >= remainingSlots &&
        currentLength >= targetChars
      ) {
        break;
      }
    }

    if (endOffset === startOffset) {
      endOffset += 1;
    }

    chunks.push({
      index: chunks.length,
      startWord: chunk.startWord + startOffset,
      endWord: chunk.startWord + endOffset,
      text: words.slice(startOffset, endOffset).join(" "),
    });
    startOffset = endOffset;
  }

  return chunks;
};

export const chunkTranscriptForSummary = (
  transcript: string,
  wordOptions = SUMMARY_CHUNK_CONFIG,
  maxChars = SUMMARY_MAX_CHARS
): TextChunk[] =>
  chunkText(transcript, wordOptions)
    .flatMap((chunk) => splitByCharacterBudget(chunk, maxChars))
    .map((chunk, index) => ({ ...chunk, index }));

async function summarizeOversizedChunk(
  chunk: string,
  chunkNum?: number,
  totalChunks?: number
): Promise<string> {
  const wordCount = countWords(chunk);
  const splitWordCount = Math.max(
    SUMMARY_RETRY_SPLIT_MIN_WORDS,
    Math.ceil(wordCount / 2)
  );
  const parts = chunkTranscriptForSummary(
    chunk,
    {
      maxWords: splitWordCount,
      overlapWords: Math.min(SUMMARY_RETRY_SPLIT_OVERLAP_WORDS, splitWordCount - 1),
    },
    Math.max(4000, Math.floor(SUMMARY_MAX_CHARS / 2))
  );

  if (parts.length <= 1) {
    throw new Error("Unable to split oversized transcript chunk any further.");
  }

  console.log(
    `OpenAI rejected part ${chunkNum ?? "?"}${
      totalChunks ? `/${totalChunks}` : ""
    } as too large; retrying as ${parts.length} smaller parts...`
  );

  const summaries: string[] = [];
  for (const part of parts) {
    summaries.push(await summarizeChunk(part.text, part.index + 1, parts.length));
  }

  return completeChat(
    PROVIDERS.SUMMARY_MODEL,
    `You are combining summaries from smaller pieces of one transcript part. Preserve only claims supported by those piece summaries and keep the same section format.`,
    `Combine these smaller summaries back into one summary for original part ${
      chunkNum ?? "?"
    }${totalChunks ? ` of ${totalChunks}` : ""}:\n\n${summaries
      .map((summary, index) => `=== Subpart ${index + 1} ===\n${summary}`)
      .join("\n\n")}`
  );
}

async function summarizeChunk(
  chunk: string,
  chunkNum?: number,
  totalChunks?: number
): Promise<string> {
  const chunkInfo =
    totalChunks && totalChunks > 1 ? ` (part ${chunkNum} of ${totalChunks})` : "";
  console.log(`Sending ${countWords(chunk)} words to ${PROVIDERS.SUMMARY_MODEL}...`);

  try {
    return await completeChat(
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
- Preserve temporal relationships. Distinguish events occurring before, during, and after the main narrative, and label flashbacks or flash-forwards instead of grouping them into the surrounding period.
- Do not merge distinct events or transfer actions, causes, or consequences between them. Preserve who or what did what, and in what sequence, when compressing related material.

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
  } catch (error) {
    if (
      error instanceof OpenAIRequestError &&
      isOpenAIRequestTooLarge(error.details) &&
      countWords(chunk) > SUMMARY_RETRY_SPLIT_MIN_WORDS
    ) {
      return summarizeOversizedChunk(chunk, chunkNum, totalChunks);
    }

    throw error;
  }
}

async function summarizeTranscript(transcript: string): Promise<string> {
  const chunks = chunkTranscriptForSummary(transcript);
  console.log(
    `Summarizing ${countWords(transcript)} words with ${PROVIDERS.SUMMARY_MODEL}...`
  );

  if (chunks.length <= 1) {
    return summarizeChunk(transcript);
  }

  console.log(
    `Long transcript, splitting into ${chunks.length} chunks with up to ${SUMMARY_CHUNK_CONCURRENCY} concurrent summaries...`
  );
  const chunkSummaries = await mapWithConcurrency(
    chunks,
    SUMMARY_CHUNK_CONCURRENCY,
    async (chunk) => {
      console.log(`Summarizing chunk ${chunk.index + 1}/${chunks.length}...`);
      return summarizeChunk(chunk.text, chunk.index + 1, chunks.length);
    }
  );

  return completeChat(
    PROVIDERS.SUMMARY_MODEL,
    `You are synthesizing analyses from different parts of one long video transcript into a final faithful summary.

Rules:
- Use only information present in the part analyses.
- Deduplicate overlap without flattening away important one-off details.
- Preserve important caveats, disagreements, and conditions when they materially change interpretation.
- Do not elevate speculation into fact.
- Keep exact quotes exactly as provided. If the quote quality is weak or unsupported, omit it.
- Only include recommendations that are clearly stated by the speaker.
- Preserve temporal relationships across parts. Distinguish events occurring before, during, and after the main narrative, and label flashbacks or flash-forwards instead of grouping them into the surrounding period.
- Do not merge distinct events or transfer actions, causes, or consequences between them. Preserve who or what did what, and in what sequence, when compressing related material.
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
    `You answer questions about a video transcript.

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
      .query("SELECT * FROM content WHERE content_type IN ('youtube', 'vimeo', 'x')")
      .all() as StoredContentRow[]
  )
    .map((row) => toContentData(row))
    .filter((row): row is ContentData => Boolean(row));

  if (rows.length === 0) throw new Error("No video sessions saved.");

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

  const safeTitle = title.replace(/[^a-zA-Z0-9]/g, "_").slice(0, 50) || "video";
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

interface DoctorCheck {
  label: string;
  ok: boolean;
  detail: string;
}

export const runDoctor = async (): Promise<boolean> => {
  const checks: DoctorCheck[] = [
    { label: "yt-dlp", ok: isUsableExecutable(YTDLP_BIN, ["--version"]), detail: YTDLP_BIN },
    { label: "ffmpeg", ok: isUsableExecutable(FFMPEG_BIN, ["-version"]), detail: FFMPEG_BIN },
    { label: "Whisper CLI", ok: isUsableExecutable(WHISPER_CLI_BIN, []), detail: WHISPER_CLI_BIN },
    { label: "Whisper model", ok: fs.existsSync(WHISPER_MODEL_PATH), detail: WHISPER_MODEL_PATH },
    {
      label: "OpenAI key",
      ok: Boolean(process.env.OPENAI_API_KEY?.trim()),
      detail: process.env.OPENAI_API_KEY?.trim() ? "configured" : "missing",
    },
  ];

  try {
    const database = openDatabase();
    database.db.query("SELECT 1 AS ok").get();
    database.db.close(true);
    checks.push({ label: "SQLite", ok: true, detail: database.path });
  } catch (error) {
    checks.push({
      label: "SQLite",
      ok: false,
      detail: error instanceof Error ? error.message : String(error),
    });
  }

  if (process.env.OPENAI_API_KEY?.trim()) {
    try {
      await getOpenAIClient().models.list();
      checks.push({ label: "OpenAI access", ok: true, detail: "API reachable" });
    } catch (error) {
      checks.push({ label: "OpenAI access", ok: false, detail: getOpenAIErrorDetails(error).message });
    }
  }

  console.log("Video Transcript Summarizer doctor\n");
  for (const check of checks) {
    console.log(`${check.ok ? "✓" : "✗"} ${check.label}: ${check.detail}`);
  }
  console.log(`\nLanguage: ${TRANSCRIPT_LANGUAGE}`);
  console.log(`Models: ${PROVIDERS.SUMMARY_MODEL} summary, ${PROVIDERS.QA_MODEL} Q&A`);
  return checks.every((check) => check.ok);
};

async function main() {
  const { values, positionals } = parseArgs({
    args: Bun.argv.slice(2),
    options: {
      url: { type: "string", short: "u" },
      rerun: { type: "string", short: "r" },
      find: { type: "string", short: "f" },
      delete: { type: "string", short: "d" },
      doctor: { type: "boolean" },
      help: { type: "boolean", short: "h" },
    },
    allowPositionals: true,
  });

  if (values.help) {
    console.log(`
Usage: bun run src/youtube.ts [url] [options]

Options:
  -u, --url <url>      Process a YouTube, Vimeo, or X URL directly (non-interactive)
  -r, --rerun <url>    Re-fetch and re-summarize a saved YouTube, Vimeo, or X URL
  -f, --find <text>    Find a saved video session by semantic match
  -d, --delete <url>   Delete a saved video session and its Q&A
      --doctor         Validate local tools, storage, and OpenAI access
  -h, --help           Show this help message

Interactive mode:
  Run without options to paste a video URL or search saved transcripts.
`);
    return;
  }

  if (values.doctor) {
    if (!(await runDoctor())) process.exitCode = 1;
    return;
  }

  const database = openDatabase();
  const { db } = database;

  try {
    let data: ContentData;
    const positionalUrl = positionals[0]?.trim();
    const directUrl =
      values.url?.trim() ||
      (positionalUrl && parseContentInput(positionalUrl) ? positionalUrl : "");

    if (values.delete) {
      const content = parseContentInput(values.delete.trim());
      if (!content) {
        throw new Error("Please provide a valid YouTube, Vimeo, or X URL to delete.");
      }

      const existing = db
        .query(
          "SELECT title FROM content WHERE content_id = ? AND content_type = ?"
        )
        .get(content.contentId, content.contentType) as { title: string | null } | null;

      if (!existing) {
        console.log(`No ${getContentTypeLabel(content.contentType)} entry found for: ${content.sourceId}`);
        return;
      }

      const result = deleteSession(database, content.contentId);
      console.log(`Deleted "${existing.title || content.contentId}"`);
      console.log(`  - Removed ${result.qaChanges} Q&A entries`);
      console.log(`  - Removed ${result.contentChanges} content entry`);
      return;
    }

    if (directUrl) {
      data = await getOrCreateTranscript(database, directUrl);
      printSummary(data, { clearBefore: true });
    } else if (values.rerun) {
      data = await getOrCreateTranscript(database, values.rerun.trim(), {
        forceRefresh: true,
      });
      printSummary(data, { clearBefore: true });
    } else if (values.find) {
      console.log(`Searching for: "${values.find}"...`);
      data = await restoreSession(database, values.find);
      printSummary(data);
    } else {
      const sessions = db
        .query(
          `SELECT content_id, content_type, title, DATE(created_at) AS created
           FROM content
           WHERE content_type IN ('youtube', 'vimeo', 'x')
           ORDER BY created_at DESC`
        )
        .all() as {
          content_id: string;
          content_type: string;
          title: string | null;
          created: string;
        }[];

      let lastTerm = "";
      const pick = await search({
        message: "Paste a video URL or search saved transcripts:",
        source: async (term) => {
          lastTerm = term || "";

          if (!term) {
            return [
              { name: "Start new session", value: "__new" },
              ...sessions.map((session) => ({
                name: `${
                  session.title || session.content_id
                } (${formatProviderLabel(session.content_type)}) [${session.created}]`,
                value: session.content_id,
              })),
            ];
          }

          const isUrl = Boolean(parseContentInput(term));
          if (isUrl) {
            return [{ name: `Process URL: ${term}`, value: `__url:${term}` }];
          }

          const lowerTerm = term.toLowerCase();
          return sessions
            .filter((session) =>
              (session.title || session.content_id).toLowerCase().includes(lowerTerm)
            )
            .map((session) => ({
              name: `${
                session.title || session.content_id
              } (${formatProviderLabel(session.content_type)}) [${session.created}]`,
              value: session.content_id,
            }));
        },
        pageSize: 12,
      });

      if (pick === "__new") {
        if (parseContentInput(lastTerm)) {
          data = await getOrCreateTranscript(database, lastTerm.trim());
        } else {
          const url = await input({ message: "Enter YouTube, Vimeo, or X URL:" });
          data = await getOrCreateTranscript(database, url.trim());
        }
        printSummary(data, { clearBefore: true });
      } else if (pick.startsWith("__url:")) {
        data = await getOrCreateTranscript(database, pick.slice(6).trim());
        printSummary(data, { clearBefore: true });
      } else {
        const row = db
          .query("SELECT * FROM content WHERE content_id = ?")
          .get(pick) as StoredContentRow | null;
        const loaded = toContentData(row);
        if (!loaded) throw new Error("Session not found in DB.");
        data = loaded;
        printSummary(data);
      }
    }

    if (!process.stdin.isTTY || !process.stdout.isTTY) return;

    let qaIndexPromise: Promise<RagIndex> | undefined;

    while (true) {
      const qaRows = db
        .query(
          "SELECT id, question, answer FROM qa WHERE content_id = ? ORDER BY created_at DESC"
        )
        .all(data.contentId) as QARow[];

      const selection = await expand({
        message: `Questions (${PROVIDERS.QA_MODEL}):`,
        expanded: true,
        choices: [
          { key: "n", name: "New question", value: "__new" },
          ...(qaRows.length > 0
            ? [{ key: "p" as const, name: "Previous questions", value: "__previous" }]
            : []),
          { key: "o", name: "Open formatted summary", value: "__open" },
          { key: "r", name: "Re-run this video", value: "__rerun" },
          { key: "e", name: "Export", value: "__export" },
          { key: "d", name: "Delete this session", value: "__delete" },
          { key: "x", name: "Exit", value: "__exit" },
        ],
      });

      if (selection === "__exit") break;

      if (selection === "__open") {
        try {
          const filename = openSummaryInBrowser(getSummaryDocument(data));
          console.log(`\nOpened formatted summary: ${filename}\n`);
        } catch (error) {
          console.error(
            `\nOpen failed: ${error instanceof Error ? error.message : String(error)}\n`
          );
        }
        continue;
      }

      if (selection === "__previous") {
        const questionId = await select({
          message: "Previous questions:",
          choices: qaRows.map((row) => ({ name: row.question, value: String(row.id) })),
          pageSize: 12,
        });
        const qa = qaRows.find((row) => String(row.id) === questionId);
        if (qa) console.log(`\nAnswer: ${qa.answer}\n`);
        continue;
      }

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
            "Re-run this video from source and replace the saved transcript/summary? Cached Q&A will be cleared.",
          default: false,
        });

        if (!confirmed) continue;

        data = await getOrCreateTranscript(
          database,
          getContentUrl(data),
          { forceRefresh: true }
        );
        qaIndexPromise = undefined;
        printSummary(data, { clearBefore: true });
        continue;
      }

      if (selection === "__export") {
        try {
          const format = await select({
            message: "Export format:",
            choices: [
              { name: "Formatted summary (.html)", value: "summary-html" as const },
              { name: "Q&A Markdown (.md)", value: "markdown" as const },
              { name: "Q&A JSON (.json)", value: "json" as const },
            ],
          });
          const filename =
            format === "summary-html"
              ? exportSummaryHtml(getSummaryDocument(data))
              : await exportQA(db, data.contentId, data.title, format);
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

    }
  } catch (error) {
    console.error("Error:", error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  } finally {
    db.close(true);
  }
}

if (import.meta.main) {
  main();
}
