# YouTube Transcript Summarizer

This project stores YouTube transcripts in SQLite, generates summaries with OpenAI, and supports transcript Q&A. It prefers downloaded captions and falls back to local Whisper transcription only when captions are unavailable.

Podcast support has been removed.

## Requirements

- Bun
- `yt-dlp`
- `ffmpeg`
- `whisper-cli` from `whisper.cpp`
- `OPENAI_API_KEY` for both the YouTube and PDF flows

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

- `SUMMARY_MODEL`
- `QA_MODEL`
- `EMBEDDING_MODEL`
- `PDF_SUMMARY_MODEL`
- `PDF_QA_MODEL`
- `PDF_EMBEDDING_MODEL`
- `YTDLP_BIN`
- `FFMPEG_BIN`
- `TRANSCRIPTS_DB`

## Commands

```bash
bun run youtube
bun run pdf
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

Non-interactive usage:

```bash
bun run src/youtube.ts --url "https://www.youtube.com/watch?v=<VIDEO_ID>"
bun run src/youtube.ts --rerun "https://www.youtube.com/watch?v=<VIDEO_ID>"
bun run src/youtube.ts --find "video about vector databases"
bun run src/youtube.ts --delete "https://www.youtube.com/watch?v=<VIDEO_ID>"
```

## PDF Flow

```bash
bun run pdf
```

The PDF tool extracts text, stores it in `pdfs.sqlite`, creates a chunked summary, and answers questions using retrieved excerpts instead of sending the whole document every time.
