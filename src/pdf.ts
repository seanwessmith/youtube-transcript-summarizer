import { Database } from "bun:sqlite";
import * as dotenv from "dotenv";
import * as fs from "node:fs";
import * as path from "node:path";
import * as readline from "node:readline";
import OpenAI from "openai";
import { PDFParse } from "pdf-parse";

import {
  buildRagIndex,
  chunkText,
  countWords,
  formatContext,
  selectRelevantChunks,
  type RagIndex,
} from "./rag.ts";

dotenv.config();

if (!process.env.OPENAI_API_KEY) {
  throw new Error("OPENAI_API_KEY is required.");
}

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const PDF_MODELS = {
  SUMMARY_MODEL: process.env.PDF_SUMMARY_MODEL || "gpt-4",
  QA_MODEL: process.env.PDF_QA_MODEL || "gpt-4",
  EMBEDDING_MODEL: process.env.PDF_EMBEDDING_MODEL || "text-embedding-3-small",
};

const SUMMARY_CHUNK_CONFIG = { maxWords: 12000, overlapWords: 200 };
const QA_CHUNK_CONFIG = { maxWords: 1200, overlapWords: 150 };
const QA_CONTEXT_CHUNKS = 4;

interface PdfData {
  pdf_id: string;
  transcript: string;
  summary: string;
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const ask = (rl: readline.Interface, prompt: string): Promise<string> =>
  new Promise((resolve) => rl.question(prompt, (answer) => resolve(answer.trim())));

async function retryOpenAI<T>(
  fn: () => Promise<T>,
  maxRetries = 5,
  initialDelayMs = 2000
): Promise<T> {
  let lastError: unknown;

  for (let attempt = 0; attempt < maxRetries; attempt += 1) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      const message = error instanceof Error ? error.message : String(error);
      const isRetryable =
        message.includes("429") ||
        message.includes("Rate limit") ||
        message.toLowerCase().includes("timeout");

      if (!isRetryable || attempt === maxRetries - 1) {
        throw error;
      }

      const delayMs = initialDelayMs * 2 ** attempt + Math.random() * 500;
      console.log(
        `OpenAI request failed, retrying in ${Math.round(delayMs / 1000)}s...`
      );
      await sleep(delayMs);
    }
  }

  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

async function initDb(): Promise<Database> {
  const db = new Database("pdfs.sqlite");

  db.exec(`
    CREATE TABLE IF NOT EXISTS pdfs (
      pdf_id TEXT PRIMARY KEY,
      transcript TEXT,
      summary TEXT,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
  `);

  return db;
}

async function completeChat(
  model: string,
  system: string,
  prompt: string
): Promise<string> {
  const response = await retryOpenAI(() =>
    openai.chat.completions.create({
      model,
      messages: [
        { role: "system", content: system },
        { role: "user", content: prompt },
      ],
      temperature: 0.2,
    })
  );

  return response.choices[0]?.message?.content?.trim() || "";
}

async function summarizeChunk(
  transcriptChunk: string,
  chunkNumber?: number,
  totalChunks?: number
): Promise<string> {
  const chunkLabel =
    totalChunks && totalChunks > 1 ? ` (part ${chunkNumber} of ${totalChunks})` : "";

  return completeChat(
    PDF_MODELS.SUMMARY_MODEL,
    `You summarize PDF document text${chunkLabel}. Be precise and concise.`,
    `Summarize this document excerpt with the following sections:

## Key Points
- 3-5 key ideas or findings

## Important Details
- notable names, metrics, references, or claims

## Takeaways
- practical conclusions or next steps

Document excerpt:
${transcriptChunk}`
  );
}

async function summarizeTranscript(transcript: string): Promise<string> {
  const chunks = chunkText(transcript, SUMMARY_CHUNK_CONFIG);
  console.log(
    `Summarizing ${countWords(transcript)} words with ${PDF_MODELS.SUMMARY_MODEL}...`
  );

  if (chunks.length <= 1) {
    return summarizeChunk(transcript);
  }

  const chunkSummaries: string[] = [];
  for (const chunk of chunks) {
    console.log(`Summarizing chunk ${chunk.index + 1}/${chunks.length}...`);
    chunkSummaries.push(
      await summarizeChunk(chunk.text, chunk.index + 1, chunks.length)
    );
  }

  return completeChat(
    PDF_MODELS.SUMMARY_MODEL,
    "You consolidate document summaries into one final summary.",
    `Synthesize these partial summaries into one cohesive final summary.

Keep this structure:

## Key Points
## Important Details
## Takeaways

Remove redundancy and preserve factual specificity.

Partial summaries:
${chunkSummaries.map((summary, index) => `=== Part ${index + 1} ===\n${summary}`).join("\n\n")}`
  );
}

async function getPdfEmbedding(text: string): Promise<number[]> {
  const response = await retryOpenAI(() =>
    openai.embeddings.create({
      model: PDF_MODELS.EMBEDDING_MODEL,
      input: text,
    })
  );

  return response.data[0]?.embedding ?? [];
}

async function buildPdfQaIndex(transcript: string): Promise<RagIndex> {
  return buildRagIndex(transcript, getPdfEmbedding, QA_CHUNK_CONFIG);
}

async function answerQuestion(index: RagIndex, question: string): Promise<string> {
  const contextChunks = await selectRelevantChunks(
    index,
    question,
    getPdfEmbedding,
    QA_CONTEXT_CHUNKS
  );
  const context = formatContext(contextChunks);

  return completeChat(
    PDF_MODELS.QA_MODEL,
    `You answer questions about a PDF using only the supplied excerpts.

If the answer is not supported by the excerpts, say so clearly.`,
    `Question: ${question}

Document excerpts:
${context}`
  );
}

async function extractPdfText(pdfPath: string): Promise<string> {
  const absolutePath = path.resolve(pdfPath);
  if (!fs.existsSync(absolutePath)) {
    throw new Error(`File not found: ${absolutePath}`);
  }
  if (!fs.statSync(absolutePath).isFile()) {
    throw new Error(`PDF path must point to a file: ${absolutePath}`);
  }

  const parser = new PDFParse({
    data: new Uint8Array(fs.readFileSync(absolutePath)),
  });

  try {
    const result = await parser.getText();
    const transcript = result.text.trim();
    if (!transcript) {
      throw new Error("No text could be extracted from the PDF.");
    }
    return transcript;
  } finally {
    await parser.destroy();
  }
}

async function getOrCreateTranscript(
  db: Database,
  pdfPath: string
): Promise<PdfData> {
  const pdfId = path.resolve(pdfPath);
  const existing = db
    .query("SELECT * FROM pdfs WHERE pdf_id = ?")
    .get(pdfId) as PdfData | null;

  if (existing) {
    return existing;
  }

  const transcript = await extractPdfText(pdfId);
  const summary = await summarizeTranscript(transcript);

  db.run("INSERT INTO pdfs (pdf_id, transcript, summary) VALUES (?, ?, ?)", [
    pdfId,
    transcript,
    summary,
  ]);

  return { pdf_id: pdfId, transcript, summary };
}

async function main() {
  const db = await initDb();
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  try {
    const pdfPath = await ask(rl, "Please enter the PDF file path: ");
    if (!pdfPath) {
      throw new Error("PDF path is required.");
    }
    const { transcript, summary } = await getOrCreateTranscript(db, pdfPath);

    console.log("\nSummary:", summary);

    const countQuery = db.query("SELECT COUNT(*) as count FROM pdfs;");
    const { count } = countQuery.get() as { count: number };
    console.log(`\nThere are currently ${count} PDF transcript(s) in the database.`);

    let qaIndexPromise: Promise<RagIndex> | undefined;

    while (true) {
      const question = await ask(
        rl,
        '\nQuestion about the PDF text (or "exit"): '
      );

      if (!question) continue;
      if (question.toLowerCase() === "exit") break;

      try {
        if (!qaIndexPromise) {
          console.log("Indexing PDF for Q&A...");
          qaIndexPromise = buildPdfQaIndex(transcript);
        }

        const answer = await answerQuestion(await qaIndexPromise, question);
        console.log("\nAnswer:", answer);
      } catch (error) {
        console.error(
          "Question failed:",
          error instanceof Error ? error.message : String(error)
        );
      }
    }
  } catch (error) {
    console.error("Error:", error instanceof Error ? error.message : String(error));
  } finally {
    db.close(true);
    rl.close();
  }
}

main();
