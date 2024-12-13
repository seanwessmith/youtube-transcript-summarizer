# YouTube Transcript Summarizer & Q&A Tool

This project retrieves transcripts from YouTube videos, stores them in a local SQLite database, generates a concise summary, and then allows you to ask questions about the transcript. It leverages OpenAI's GPT model for summarization and question-answering.

## Features

- **Transcript Retrieval:** Automatically fetches transcripts for given YouTube videos.
- **Database Storage:** Stores transcripts and their summaries in a local SQLite database, avoiding re-fetching for previously processed videos.
- **Summarization:** Generates a concise summary of the transcript using OpenAI's GPT model.
- **Q&A Session:** Interactively asks the user for questions related to the transcript and provides answers derived from the transcript context.
- **CLI Interaction:** Prompts for YouTube URL and interactive Q&A directly in the command-line interface.

## Requirements

- **Node.js 16+** and **Bun** for running the code.
- A valid **OpenAI API Key** for using the GPT models.
- **SQLite** included via `bun:sqlite` (installed by Bun).

## Dependencies

- [dotenv](https://www.npmjs.com/package/dotenv) for managing environment variables.
- [youtube-transcript](https://www.npmjs.com/package/youtube-transcript) for fetching YouTube transcripts.
- [openai](https://www.npmjs.com/package/openai) for accessing OpenAI's GPT models.
- [bun:sqlite](https://bun.sh/docs/api/sqlite) for local SQLite database interactions.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/yt-transcript-summarizer.git
cd yt-transcript-summarizer
```

### 2. Install Dependencies

Install Node.js dependencies using `npm` or `yarn`:

```bash
npm install
# or
yarn install
```

### 3. Set Up Environment Variables

Create a `.env` file in the project root and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Run the Program

Use Bun to run the script (assuming Bun is installed):

```bash
bun run index.ts
```

When prompted, enter a YouTube URL:

```none
Please enter the YouTube URL: https://www.youtube.com/watch?v=<VIDEO_ID>
```

The script will:

1. Fetch the transcript.
2. Summarize it.
3. Display the summary.
4. Inform how many transcripts are currently stored in the database.

You can then ask questions about the transcript:

```none
Question about transcript (or "exit"): What is the main topic of the video?
Answer: ...
```

Type `exit` when you're done to close the session.

## Tips & Notes

- **Re-using Transcripts:** If you run the script multiple times with the same YouTube URL, it will use the stored transcript and summary, saving time and API calls.
- **Model Choice:** The code references a model named `gpt-4o`. Make sure you have access to the intended OpenAI model and update the code if necessary.
- **Custom Questions:** Ask specific questions related to the content of the transcript to get the most value out of the Q&A feature.
- **Error Handling:** If you encounter errors (e.g., no transcript available, invalid URL), the script will print an error message and exit.

## License

This project is distributed under the MIT License. See [LICENSE](./LICENSE) for details.