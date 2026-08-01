import { Database } from "bun:sqlite";
import * as dotenv from "dotenv";
import * as fs from "node:fs";
import * as path from "node:path";
import { parseArgs } from "node:util";

import { MODEL_PROFILES, type ModelProfile } from "./config.ts";
import { decodeTranscript, resolveDatabasePath } from "./database.ts";
import { formatRequestMetrics } from "./openai-metrics.ts";
import { summarizeTranscript } from "./youtube.ts";

dotenv.config({ quiet: true });

const REQUIRED_HEADINGS = [
  "## Overall Summary",
  "## Main Points",
  "## Important Details",
  "## Exact Quotes",
  "## People & References",
  "## Explicit Recommendations",
];

interface EvalFixture {
  required?: string[];
  forbidden?: string[];
}

const parseProfiles = (raw: string): ModelProfile[] => {
  const profiles = [...new Set(raw.split(",").map((value) => value.trim()))];
  if (
    profiles.length === 0 ||
    profiles.some((value) => !(value in MODEL_PROFILES))
  ) {
    throw new Error("--profiles must contain cheap, balanced, and/or quality.");
  }
  return profiles as ModelProfile[];
};

const loadFixture = (filename?: string): EvalFixture => {
  if (!filename) return {};
  const parsed = JSON.parse(fs.readFileSync(path.resolve(filename), "utf8"));
  return {
    required: Array.isArray(parsed.required) ? parsed.required.map(String) : [],
    forbidden: Array.isArray(parsed.forbidden) ? parsed.forbidden.map(String) : [],
  };
};

const main = async (): Promise<void> => {
  const { values } = parseArgs({
    args: Bun.argv.slice(2),
    options: {
      "content-id": { type: "string" },
      profiles: { type: "string", default: "cheap,balanced,quality" },
      fixture: { type: "string" },
      help: { type: "boolean", short: "h" },
    },
  });

  if (values.help) {
    console.log(`Usage: bun run eval --content-id <id> [--profiles cheap,balanced,quality] [--fixture checks.json]

This command makes paid OpenAI API requests. It never selects content automatically.`);
    return;
  }
  const contentId = values["content-id"]?.trim();
  if (!contentId) throw new Error("--content-id is required; no content is selected automatically.");
  if (!process.env.OPENAI_API_KEY?.trim()) throw new Error("OPENAI_API_KEY is required.");

  const database = new Database(resolveDatabasePath(), { readonly: true });
  const row = database
    .query("SELECT title, transcript FROM content WHERE content_id = ?")
    .get(contentId) as { title: string | null; transcript: unknown } | null;
  database.close(true);
  if (!row) throw new Error(`No saved content found for ID: ${contentId}`);
  const transcript = decodeTranscript(row.transcript);
  if (!transcript.trim()) throw new Error("The selected content has no transcript.");

  const profiles = parseProfiles(values.profiles ?? "cheap,balanced,quality");
  const fixture = loadFixture(values.fixture);
  const results = [];
  for (const profileName of profiles) {
    const profile = MODEL_PROFILES[profileName].summary;
    console.log(`\nEvaluating ${profileName} (${profile.model})...`);
    const result = await summarizeTranscript(transcript, undefined, profile);
    const lower = result.text.toLowerCase();
    results.push({
      profile: profileName,
      ...result,
      checks: {
        headings: REQUIRED_HEADINGS.map((heading) => ({
          value: heading,
          passed: result.text.includes(heading),
        })),
        required: (fixture.required ?? []).map((value) => ({
          value,
          passed: lower.includes(value.toLowerCase()),
        })),
        forbidden: (fixture.forbidden ?? []).map((value) => ({
          value,
          passed: !lower.includes(value.toLowerCase()),
        })),
      },
    });
  }

  const outputDir = path.resolve("eval-results");
  fs.mkdirSync(outputDir, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const safeId = contentId.replace(/[^a-zA-Z0-9_-]/g, "_");
  const base = path.join(outputDir, `${safeId}-${stamp}`);
  await Bun.write(`${base}.json`, JSON.stringify({ contentId, title: row.title, results }, null, 2));
  await Bun.write(
    `${base}.md`,
    [
      `# Model evaluation: ${row.title || contentId}`,
      "",
      ...results.flatMap((result) => [
        `## ${result.profile}`,
        "",
        formatRequestMetrics(result.metrics),
        "",
        result.text,
        "",
      ]),
    ].join("\n")
  );
  console.log(`\nSaved ${base}.md and ${base}.json`);
};

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
