import os
from crewai_tools import YoutubeVideoSearchTool
import openai

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # this is for crewai
openai.api_key = os.getenv("OPENAI_API_KEY")  # this is for openai

# this first time you run it, it will get a new transcript. then you can reuse it instead of calling crewai multiple times
GET_NEW_TRANSCRIPT = True

TRANSCRIPT = ""

if GET_NEW_TRANSCRIPT:
    VIDEO_URL = "https://www.youtube.com/watch?v=Q4RkavtviYU"

    # Targeted search within a specific Youtube video's content
    tool = YoutubeVideoSearchTool(youtube_video_url=VIDEO_URL)

    # Transcribe the video
    TRANSCRIPT = tool.run("transcribe")

    # Save or view the transcript
    with open("transcript.txt", "w", encoding="utf-8") as f:
        f.write(TRANSCRIPT)

with open("transcript.txt", "r", encoding="utf-8") as f:
    transcript = f.read()


def summarize_transcript(max_tokens=150):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes transcripts.",
            },
            {
                "role": "user",
                "content": f"Please provide a concise summary of the following transcript:\n\n{TRANSCRIPT}",
            },
        ],
        max_tokens=max_tokens,
        temperature=0.5,
    )
    summary = response.choices[0].message
    return summary


SUMMARY = summarize_transcript(transcript)
print("Summary:")
print(SUMMARY.content)
