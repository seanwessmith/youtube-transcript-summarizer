import { afterEach, describe, expect, test } from "bun:test";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

import {
  DATABASE_SCHEMA_VERSION,
  backupDatabaseIfNeeded,
  clearCachedEmbeddings,
  decodeTranscript,
  encodeTranscript,
  openDatabase,
  resolveDatabasePath,
} from "./database.ts";

const tempDirs: string[] = [];
const makeTempDir = (): string => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "video-db-test-"));
  tempDirs.push(dir);
  return dir;
};

afterEach(() => {
  for (const dir of tempDirs.splice(0)) fs.rmSync(dir, { recursive: true, force: true });
});

describe("database", () => {
  test("creates the current schema and enforces foreign keys", () => {
    const database = openDatabase(path.join(makeTempDir(), "nested", "transcripts.sqlite"));
    const version = database.db.query("PRAGMA user_version").get() as { user_version: number };
    expect(version.user_version).toBe(DATABASE_SCHEMA_VERSION);
    expect(
      database.db.query("SELECT name FROM sqlite_master WHERE type = 'table'").all()
        .map((row) => (row as { name: string }).name)
    ).toContain("transcript_chunk_embeddings");
    expect(
      database.db.query("PRAGMA table_info(content)").all()
        .map((row) => (row as { name: string }).name)
    ).toContain("summary_cost_microusd");
    expect(
      database.db.query("PRAGMA table_info(qa)").all()
        .map((row) => (row as { name: string }).name)
    ).toContain("cached_input_tokens");
    expect(() => database.db.run(
      "INSERT INTO qa (content_id, question, answer) VALUES ('missing', 'q', 'a')"
    )).toThrow();
    database.db.close(true);
  });

  test("round-trips compressed and legacy plain transcripts", () => {
    const transcript = "A transcript with unicode: hej världen";
    expect(decodeTranscript(encodeTranscript(transcript))).toBe(transcript);
    expect(decodeTranscript(new TextEncoder().encode(transcript))).toBe(transcript);
  });

  test("clears both embedding caches for one content item", () => {
    const database = openDatabase(path.join(makeTempDir(), "transcripts.sqlite"));
    database.db.run("INSERT INTO content (content_id, content_type) VALUES ('video', 'youtube')");
    database.db.run("INSERT INTO content_embeddings VALUES ('video', 'find_summary', 'model', 'text', '[]', CURRENT_TIMESTAMP)");
    database.db.run("INSERT INTO transcript_chunk_embeddings VALUES ('video', 'model', 0, 0, 1, 'text', '[]', CURRENT_TIMESTAMP)");
    clearCachedEmbeddings(database.db, "video");
    expect(database.db.query("SELECT count(*) AS count FROM content_embeddings").get()).toEqual({ count: 0 });
    expect(database.db.query("SELECT count(*) AS count FROM transcript_chunk_embeddings").get()).toEqual({ count: 0 });
    database.db.close(true);
  });

  test("backs up an existing database before its first mutation", () => {
    const dir = makeTempDir();
    const dbPath = path.join(dir, "transcripts.sqlite");
    const first = openDatabase(dbPath);
    first.db.run("INSERT INTO content (content_id, content_type) VALUES ('video', 'youtube')");
    first.db.close(true);
    const reopened = openDatabase(dbPath);
    backupDatabaseIfNeeded(reopened);
    expect(fs.readdirSync(path.join(dir, "db_backups"))).toHaveLength(1);
    reopened.db.close(true);
  });

  test("honors an explicit database path", () => {
    const requested = path.join(makeTempDir(), "custom.sqlite");
    expect(resolveDatabasePath(requested)).toBe(path.resolve(requested));
  });

  test("migrates a version 1 database without backfilling metrics", () => {
    const dbPath = path.join(makeTempDir(), "legacy.sqlite");
    const legacy = openDatabase(dbPath);
    legacy.db.run(
      "INSERT INTO content (content_id, content_type, summary) VALUES ('legacy', 'youtube', 'saved')"
    );
    legacy.db.exec("PRAGMA user_version = 1");
    legacy.db.exec(`
      CREATE TABLE content_v1 AS SELECT content_id, content_type, title, author, audio_url, transcript, summary, created_at FROM content;
      DROP TABLE content;
      ALTER TABLE content_v1 RENAME TO content;
      CREATE TABLE qa_v1 AS SELECT id, content_id, question, answer, created_at FROM qa;
      DROP TABLE qa;
      ALTER TABLE qa_v1 RENAME TO qa;
    `);
    legacy.db.close(true);

    const migrated = openDatabase(dbPath);
    expect(
      migrated.db.query("SELECT summary, summary_model, summary_cost_microusd FROM content WHERE content_id = 'legacy'").get()
    ).toEqual({ summary: "saved", summary_model: null, summary_cost_microusd: null });
    expect(
      (migrated.db.query("PRAGMA user_version").get() as { user_version: number }).user_version
    ).toBe(2);
    migrated.db.close(true);
  });
});
