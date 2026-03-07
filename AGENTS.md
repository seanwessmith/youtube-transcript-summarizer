# YouTube Rag Project Guidelines

## Commands
- Build/Run: `bun run src/index.ts` or `npm run start`
- Test: No specific test command found (consider adding Jest/Vitest)
- TypeScript: Strict mode enabled in tsconfig.json

## Coding Style
- Imports: Clean imports at top, grouped by source
- Types: Use TypeScript interfaces for data structures
- Async: Use async/await pattern consistently
- Error handling: Try-catch with specific error type checking
- Environment: Use dotenv for environment variables
- Database: Bun's SQLite integration with consistent query patterns
- Functions: Pure functions with descriptive names
- API: Consistent OpenAI API call patterns
- CLI: Readline for user interaction
- Documentation: Add comments for key operations
- Variables: Descriptive names in camelCase
- Git: Commit messages follow conventional commits format