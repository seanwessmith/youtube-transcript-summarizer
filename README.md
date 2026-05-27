# YouTube Transcript Summarizer

This project stores YouTube transcripts in SQLite, generates summaries with OpenAI, and supports transcript Q&A. It prefers downloaded captions and falls back to local Whisper transcription only when captions are unavailable.

## Requirements

- Bun
- `yt-dlp`
- `ffmpeg`
- `whisper-cli` from `whisper.cpp`
- `OPENAI_API_KEY` for summarization, semantic search, and Q&A

## Install

```bash
bun install
```

## Environment

Create `.env` in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
WHISPER_CLI_BIN=./whisper.cpp/build/bin/whisper-cli
WHISPER_MODEL_PATH=./whisper.cpp/models/ggml-base.en.bin
```

Optional overrides:

- `MODEL_PROFILE` (`cheap`, `balanced`, or `quality`; defaults to `balanced`)
- `SUMMARY_MODEL`
- `QA_MODEL`
- `EMBEDDING_MODEL`
- `YTDLP_BIN`
- `FFMPEG_BIN`
- `TRANSCRIPTS_DB`

Model profiles:

- `cheap`: `gpt-5.4-nano` for summaries and Q&A
- `balanced`: `gpt-5.4-mini` for summaries, `gpt-5.4-nano` for Q&A
- `quality`: `gpt-5.5` for summaries, `gpt-5.4-mini` for Q&A

Explicit `SUMMARY_MODEL` and `QA_MODEL` values override the selected profile.

## Commands

```bash
bun run youtube
npm run start
bun run typecheck
```

## YouTube Flow

```bash
bun run youtube
```

The CLI can:

- start a new YouTube session from a URL
- reopen saved sessions from SQLite
- find a saved session semantically with `--find`
- export stored Q&A to Markdown or JSON
- reuse persisted embeddings for faster semantic find and transcript Q&A

Non-interactive usage:

```bash
bun run youtube --url "https://www.youtube.com/watch?v=<VIDEO_ID>"
bun run youtube --rerun "https://www.youtube.com/watch?v=<VIDEO_ID>"
bun run youtube --find "video about vector databases"
bun run youtube --delete "https://www.youtube.com/watch?v=<VIDEO_ID>"
```

The `youtube` and `start` scripts both run the same CLI entrypoint: `src/youtube.ts`.
