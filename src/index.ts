import dotenv from "dotenv";
import { YoutubeTranscript } from "youtube-transcript";
import OpenAI from "openai";
import { Database } from "bun:sqlite";
import readline from "readline";

interface Transcript {
  text: string;
  duration: number;
  offset: number;
}

interface VideoData {
  transcript: string;
  summary: string;
}

dotenv.config();
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const initDb = async () => {
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

const getVideoId = (url: string): string => {
  const match = url.match(/(?:v=|\/)([a-zA-Z0-9_-]{11})/);
  return match ? match[1] : "";
};

const summarizeTranscript = async (transcript: string): Promise<string> => {
  const response = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [
      { role: "system", content: "Summarize transcripts concisely." },
      { role: "user", content: `Summarize this transcript:\n\n${transcript}` },
    ],
    temperature: 0.5,
  });
  return response.choices[0].message.content || "";
};

const getOrCreateTranscript = async (
  db: Database,
  videoUrl: string
): Promise<VideoData> => {
  const videoId = getVideoId(videoUrl);
  if (!videoId) {
    throw new Error("Invalid YouTube URL. Could not extract video ID.");
  }

  // Check if we already have this video
  const query = db.query("SELECT * FROM videos WHERE video_id = $videoId;");
  const existing = query.get({ $videoId: videoId }) as VideoData | undefined;
  if (existing) {
    return {
      transcript: existing.transcript,
      summary: existing.summary,
    };
  }

  // Get new transcript and summary
  const transcriptList = await YoutubeTranscript.fetchTranscript(videoId);
  const transcript = transcriptList.map((entry) => entry.text).join(" ");
  const summary = await summarizeTranscript(transcript);

  // Store in database
  await db.run(
    "INSERT INTO videos (video_id, transcript, summary) VALUES (?, ?, ?)",
    [videoId, transcript, summary]
  );

  return { transcript, summary };
};

const answerQuestion = async (
  transcript: string,
  question: string
): Promise<string> => {
  const response = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [
      { role: "system", content: "Answer questions based on the transcript." },
      {
        role: "user",
        content: `Transcript:\n${transcript}\n\nQuestion:\n${question}`,
      },
    ],
    temperature: 0.5,
  });
  return response.choices[0].message.content || "";
};

const main = async () => {
  const db = await initDb();

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  // Prompt user for YouTube URL
  const videoUrl = await new Promise<string>((resolve) => {
    rl.question("Please enter the YouTube URL: ", (answer) => {
      resolve(answer.trim());
    });
  });

  try {
    const { transcript, summary } = await getOrCreateTranscript(db, videoUrl);
    console.log("\nSummary:", summary);

    // Count how many transcripts have been stored
    const countQuery = db.query("SELECT COUNT(*) as count FROM videos;");
    const { count } = countQuery.get() as { count: number };
    console.log(
      `\nThere are currently ${count} transcript(s) stored in the database.`
    );

    const askQuestion = () => {
      rl.question(
        '\nQuestion about transcript (or "exit"): ',
        async (question: string) => {
          if (question.toLowerCase() === "exit") {
            await db.close();
            rl.close();
            return;
          }

          const answer = await answerQuestion(transcript, question);
          console.log("\nAnswer:", answer);
          askQuestion();
        }
      );
    };

    askQuestion();
  } catch (error) {
    console.error("Error in main:", error);
    await db.close();
    rl.close();
  }
};

main();
