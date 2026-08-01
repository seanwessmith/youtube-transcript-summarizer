# Video Transcript Summarizer Guidelines

## Source of truth

- This is a Bun + TypeScript CLI using SQLite, the OpenAI Responses API, `yt-dlp`, FFmpeg, and local `whisper.cpp`.
- Preserve the three model-profile roles in `src/config.ts`. Check current official OpenAI pricing and model guidance before changing model IDs, reasoning, verbosity, or pricing metadata.
- Preserve saved databases. Schema changes must use `PRAGMA user_version`, nullable additions where possible, and the existing pre-migration backup path.
- Do not run the paid evaluator unless the user explicitly authorizes a content ID and API spend.

## Commands

- Run: `bun run youtube` or `bun run video`
- Preflight: `bun run doctor`
- Typecheck: `bun run typecheck`
- Tests: `bun run test` (scoped to `src` so vendored `whisper.cpp` tests are excluded)
- Opt-in paid evaluation: `bun run eval --content-id <id> --profiles cheap,balanced,quality`

## Change expectations

- Keep transcript claims, quotations, references, recommendations, and chronology grounded in source text.
- Retain direct-context and chunk/RAG fallbacks unless a measured replacement covers both normal and oversized transcripts.
- Keep API tests mocked and deterministic; CI must never make paid OpenAI calls.
- For code changes, run `bun run typecheck`, `bun run test`, and `git diff --check`.
- For runtime changes, also run `bun run doctor` and distinguish local preflight from a paid end-to-end model evaluation.
- Use conventional commit messages and do not push or publish unless explicitly requested.
