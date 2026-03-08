# YouTube Rag Project Guidelines

## Commands
- Build/Run: `bun run youtube`
- Typecheck: `bun run typecheck`
- TypeScript: Strict mode enabled in `tsconfig.json`

## Coding Style
- Imports: Clean imports at top, grouped by source
- Types: Use TypeScript interfaces for data structures
- Async: Use async/await consistently
- Error handling: Prefer specific, recoverable errors over broad failures
- Environment: Use dotenv for environment variables
- Database: Bun SQLite with consistent query patterns
- Functions: Prefer small, descriptive helpers
- Documentation: Keep README and scripts aligned with actual runtime behavior
