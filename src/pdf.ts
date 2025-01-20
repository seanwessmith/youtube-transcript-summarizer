import dotenv from "dotenv";
import OpenAI from "openai";
import { Database } from "bun:sqlite";
import readline from "readline";
import fs from "fs";
import pdf from "pdf-parse";

interface PdfData {
  transcript: string;
  summary: string;
}

dotenv.config();

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const initDb = async () => {
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
};

const summarizeTranscript = async (transcript: string): Promise<string> => {
  const response = await openai.chat.completions.create({
    model: "gpt-4",
    messages: [
      {
        role: "system",
        content: "Summarize the following document text concisely.",
      },
      { role: "user", content: `Summarize this text:\n\n${transcript}` },
    ],
    temperature: 0.5,
  });
  return response.choices[0].message.content || "";
};

const getOrCreateTranscript = async (
  db: Database,
  pdfPath: string
): Promise<PdfData> => {
  // Use the PDF filename as a unique ID
  const pdfId = pdfPath;

  // Check if we already have this PDF in the database
  const query = db.query("SELECT * FROM pdfs WHERE pdf_id = $pdfId;");
  const existing = query.get({ $pdfId: pdfId }) as PdfData | undefined;
  if (existing) {
    return {
      transcript: existing.transcript,
      summary: existing.summary,
    };
  }

  // Extract transcript (all text) from PDF
  if (!fs.existsSync(pdfPath)) {
    throw new Error(`File not found: ${pdfPath}`);
  }
  const dataBuffer = fs.readFileSync(pdfPath);
  const pdfData = await pdf(dataBuffer);
  const transcript = pdfData.text;

  // Summarize the transcript
  const summary = await summarizeTranscript(transcript);

  // Store in database
  await db.run(
    "INSERT INTO pdfs (pdf_id, transcript, summary) VALUES (?, ?, ?)",
    [pdfId, transcript, summary]
  );

  return { transcript, summary };
};

const answerQuestion = async (
  transcript: string,
  question: string
): Promise<string> => {
  const response = await openai.chat.completions.create({
    model: "gpt-4",
    messages: [
      {
        role: "system",
        content: "Answer questions based on the provided text.",
      },
      {
        role: "user",
        content: `Document text:\n${transcript}\n\nQuestion:\n${question}`,
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

  // Prompt user for PDF file path
  const pdfPath = await new Promise<string>((resolve) => {
    rl.question("Please enter the PDF file path: ", (answer) => {
      resolve(answer.trim());
    });
  });

  try {
    const { transcript, summary } = await getOrCreateTranscript(db, pdfPath);
    console.log("\nSummary:", summary);

    // Count how many PDFs have been stored
    const countQuery = db.query("SELECT COUNT(*) as count FROM pdfs;");
    const { count } = countQuery.get() as { count: number };
    console.log(
      `\nThere are currently ${count} transcript(s) stored in the database.`
    );

    const askQuestion = () => {
      rl.question(
        '\nQuestion about the PDF text (or "exit"): ',
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
