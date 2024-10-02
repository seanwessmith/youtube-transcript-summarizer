# YouTube Transcript Summarizer

A Python tool that transcribes a YouTube video and generates a concise summary using OpenAI's GPT-4 model. This script leverages the `crewai_tools` library for video transcription and the OpenAI API for summarization.

## Table of Contents

- [YouTube Transcript Summarizer](#youtube-transcript-summarizer)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)

## Features

- **Video Transcription:** Automatically transcribes the audio content of a specified YouTube video.
- **Transcript Caching:** Saves the transcript locally to avoid repeated API calls, enhancing efficiency.
- **Summarization:** Generates a concise summary of the transcript using OpenAI's GPT-4 model.
- **Easy Configuration:** Simple setup with environment variables for API keys.

## Prerequisites

- Python 3.7 or higher
- An OpenAI API key
- Access to the `crewai_tools` library

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/youtube-transcript-summarizer.git
   cd youtube-transcript-summarizer