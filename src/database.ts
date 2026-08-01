import { Database } from "bun:sqlite";
import * as fs from "node:fs";
import * as path from "node:path";

export const DATABASE_SCHEMA_VERSION = 1;
const MAX_DB_BACKUPS = 10;
const backedUpDatabases = new Set<string>();

export interface DatabaseHandle {
  db: Database;
  path: string;
  hadExistingData: boolean;
}

export const resolveDatabasePath = (
  configuredPath = process.env.TRANSCRIPTS_DB,
  cwd = process.cwd(),
  projectDir = path.resolve(import.meta.dir, ".."),
  home = process.env.HOME || ""
): string => {
  const envPath = configuredPath?.trim();
  if (envPath) return path.resolve(envPath);

  const candidates = [
    path.resolve(cwd, "transcripts.sqlite"),
    path.resolve(projectDir, "transcripts.sqlite"),
    home ? path.resolve(home, "transcripts.sqlite") : "",
    home ? path.resolve(home, "Documents", "transcripts.sqlite") : "",
  ].filter(Boolean);

  const existing = [...new Set(candidates)].filter((candidate) =>
    fs.existsSync(candidate)
  );

  for (const dbPath of existing) {
    try {
      const probe = new Database(dbPath, { readonly: true });
      const row = probe
        .query("SELECT count(*) AS c FROM content")
        .get() as { c: number } | null;
      probe.close(true);
      if ((row?.c ?? 0) > 0) return dbPath;
    } catch {
      // Ignore non-SQLite and pre-schema files.
    }
  }

  return existing[0] || path.resolve(projectDir, "transcripts.sqlite");
};

const migrateDatabase = (db: Database): void => {
  const versionRow = db.query("PRAGMA user_version").get() as {
    user_version: number;
  } | null;
  const currentVersion = Number(versionRow?.user_version ?? 0);
  if (currentVersion > DATABASE_SCHEMA_VERSION) {
    throw new Error(
      `Database schema version ${currentVersion} is newer than supported version ${DATABASE_SCHEMA_VERSION}.`
    );
  }

  if (currentVersion < 1) {
    db.transaction(() => {
      db.exec(`
        CREATE TABLE IF NOT EXISTS content (
          content_id TEXT PRIMARY KEY,
          content_type TEXT NOT NULL,
          title TEXT,
          author TEXT,
          audio_url TEXT,
          transcript BLOB,
          summary TEXT,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS qa (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          content_id TEXT NOT NULL,
          question TEXT NOT NULL,
          answer TEXT NOT NULL,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY (content_id) REFERENCES content(content_id)
        );
        CREATE TABLE IF NOT EXISTS content_embeddings (
          content_id TEXT NOT NULL,
          embedding_kind TEXT NOT NULL,
          model TEXT NOT NULL,
          source_text TEXT NOT NULL,
          embedding TEXT NOT NULL,
          updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY (content_id, embedding_kind, model),
          FOREIGN KEY (content_id) REFERENCES content(content_id)
        );
        CREATE TABLE IF NOT EXISTS transcript_chunk_embeddings (
          content_id TEXT NOT NULL,
          model TEXT NOT NULL,
          chunk_index INTEGER NOT NULL,
          start_word INTEGER NOT NULL,
          end_word INTEGER NOT NULL,
          text TEXT NOT NULL,
          embedding TEXT NOT NULL,
          updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY (content_id, model, chunk_index),
          FOREIGN KEY (content_id) REFERENCES content(content_id)
        );
        CREATE INDEX IF NOT EXISTS idx_content_type_created_at
          ON content (content_type, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_qa_content_created_at
          ON qa (content_id, created_at DESC);
        PRAGMA user_version = 1;
      `);
    })();
  }
};

export const openDatabase = (dbPath = resolveDatabasePath()): DatabaseHandle => {
  const resolvedPath = path.resolve(dbPath);
  fs.mkdirSync(path.dirname(resolvedPath), { recursive: true });
  const hadExistingData =
    fs.existsSync(resolvedPath) && fs.statSync(resolvedPath).size > 0;
  const db = new Database(resolvedPath);
  db.exec("PRAGMA foreign_keys = ON;");
  const database = { db, path: resolvedPath, hadExistingData };
  backupDatabaseIfNeeded(database);
  migrateDatabase(db);
  return database;
};

const pruneBackups = (dbPath: string): void => {
  const backupDir = path.resolve(path.dirname(dbPath), "db_backups");
  if (!fs.existsSync(backupDir)) return;
  const extension = path.extname(dbPath) || ".sqlite";
  const baseName = path.basename(dbPath, extension);
  const backups = fs
    .readdirSync(backupDir)
    .filter((name) => name.startsWith(`${baseName}.`) && name.endsWith(extension))
    .sort((a, b) => b.localeCompare(a));
  for (const oldBackup of backups.slice(MAX_DB_BACKUPS)) {
    fs.rmSync(path.join(backupDir, oldBackup), { force: true });
  }
};

export const backupDatabaseIfNeeded = (database: DatabaseHandle): void => {
  if (!database.hadExistingData || backedUpDatabases.has(database.path)) return;
  if (!fs.existsSync(database.path) || fs.statSync(database.path).size === 0) {
    backedUpDatabases.add(database.path);
    return;
  }

  const backupDir = path.resolve(path.dirname(database.path), "db_backups");
  fs.mkdirSync(backupDir, { recursive: true });
  const extension = path.extname(database.path) || ".sqlite";
  const baseName = path.basename(database.path, extension);
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  database.db.exec("PRAGMA wal_checkpoint(FULL);");
  fs.copyFileSync(
    database.path,
    path.join(backupDir, `${baseName}.${timestamp}${extension}`)
  );
  backedUpDatabases.add(database.path);
  pruneBackups(database.path);
};

export const clearCachedEmbeddings = (db: Database, contentId: string): void => {
  db.run("DELETE FROM content_embeddings WHERE content_id = ?", [contentId]);
  db.run("DELETE FROM transcript_chunk_embeddings WHERE content_id = ?", [contentId]);
};

export const encodeTranscript = (text: string): Uint8Array =>
  Bun.gzipSync(new TextEncoder().encode(text));

const decodeBytes = (bytes: Uint8Array): string => {
  if (bytes.length >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b) {
    try {
      return new TextDecoder().decode(
        Bun.gunzipSync(Uint8Array.from(bytes) as Uint8Array<ArrayBuffer>)
      );
    } catch {
      // Preserve compatibility with malformed legacy blobs.
    }
  }
  return new TextDecoder().decode(bytes);
};

export const decodeTranscript = (raw: unknown): string => {
  if (!raw) return "";
  if (typeof raw === "string") return raw;
  if (raw instanceof Uint8Array) return decodeBytes(Uint8Array.from(raw));
  if (raw instanceof ArrayBuffer) return decodeBytes(new Uint8Array(raw));
  if (ArrayBuffer.isView(raw)) {
    return decodeBytes(new Uint8Array(raw.buffer, raw.byteOffset, raw.byteLength));
  }
  return String(raw);
};
