import dotenv from "dotenv";
// import { YoutubeTranscript } from "youtube-transcript";
import OpenAI from "openai";
import { Database } from "bun:sqlite";
import readline from "readline";
import { spawn } from "child_process";
import { fetchTranscript } from "./utils/youtube-transcript-fetcher";

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

interface VideoData {
  video_id: string;
  transcript: string;
  summary: string;
}

/** Initialize SQLite database */
const initDb = async (): Promise<Database> => {
  const db = new Database("transcripts.sqlite");
  db.exec(`
    CREATE TABLE IF NOT EXISTS videos (
      video_id TEXT PRIMARY KEY,
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

/** Transcribe video using WhisperX via Python script */
const transcribeWithWhisperX = async (videoUrl: string): Promise<string> => {
  return new Promise((resolve, reject) => {
    const pythonInterpreter = process.env.PYTHON_INTERPRETER || "python";
    const pythonProcess = spawn(pythonInterpreter, [
      "src/transcribe.py",
      videoUrl,
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

/** Fetch or create transcript for a video */
const getOrCreateTranscript = async (
  db: Database,
  videoUrl: string
): Promise<VideoData> => {
  const videoId = getVideoId(videoUrl);
  if (!videoId)
    throw new Error("Invalid YouTube URL. Could not extract video ID.");

  const existing = db
    .query("SELECT * FROM videos WHERE video_id = $videoId")
    .get({ $videoId: videoId }) as VideoData | null;
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
      transcript = await transcribeWithWhisperX(videoUrl);
    } else {
      throw error;
    }
  }

  const summary = await summarizeTranscript(transcript);
  db.run(
    "INSERT INTO videos (video_id, transcript, summary) VALUES (?, ?, ?)",
    [videoId, transcript, summary]
  );

  return { video_id: videoId, transcript, summary };
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
): Promise<VideoData> => {
  const rows = db
    .query("SELECT video_id, transcript, summary FROM videos")
    .all() as VideoData[];
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

/** Get a valid YouTube URL from user */
const getValidVideoUrl = async (): Promise<string> => {
  while (true) {
    const url = await askQuestion("Please enter the YouTube URL: ");
    if (getVideoId(url)) return url;
    console.log("Invalid YouTube URL. Please try again.");
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
    let videoData: VideoData;
    if (sessionType === "1") {
      const videoUrl = await getValidVideoUrl();
      videoData = await getOrCreateTranscript(db, videoUrl);
    } else {
      const narrative = await getNarrative();
      videoData = await restoreSession(db, narrative);
    }
    console.log("\nSummary:", videoData.summary);
    await handleQuestions(videoData.transcript);
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
