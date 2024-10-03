import os
from dotenv import dotenv_values
from crewai_tools import YoutubeVideoSearchTool
import openai


def load_api_key():
    config = dotenv_values(".env")
    OPENAI_API_KEY = config.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    openai.api_key = OPENAI_API_KEY


def get_transcript(video_url, transcript_file):
    # Targeted search within a specific YouTube video's content
    tool = YoutubeVideoSearchTool(youtube_video_url=video_url)

    # Transcribe the video
    transcript = tool.run("transcribe")

    # Save the transcript
    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(transcript)
    return transcript


def read_transcript(transcript_file):
    with open(transcript_file, "r", encoding="utf-8") as f:
        transcript = f.read()
    return transcript


def summarize_transcript(transcript, max_tokens=150):
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes transcripts.",
                },
                {
                    "role": "user",
                    "content": f"Please provide a concise summary of the following transcript:\n\n{transcript}",
                },
            ],
            max_tokens=max_tokens,
            temperature=0.5,
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        print(f"Error during summarization: {e}")
        return None


def answer_question(transcript, question, max_tokens=150):
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on a transcript.",
                },
                {
                    "role": "user",
                    "content": f"Based on the following transcript, please answer the question.\n\nTranscript:\n{transcript}\n\nQuestion:\n{question}",
                },
            ],
            max_tokens=max_tokens,
            temperature=0.5,
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"Error during question answering: {e}")
        return None


def main():
    load_api_key()
    GET_NEW_TRANSCRIPT = True
    TRANSCRIPT_FILE = "transcript.txt"
    VIDEO_URL = "https://www.youtube.com/watch?v=4NI2P-R6EQU"

    if GET_NEW_TRANSCRIPT or not os.path.exists(TRANSCRIPT_FILE):
        TRANSCRIPT = get_transcript(VIDEO_URL, TRANSCRIPT_FILE)
    else:
        TRANSCRIPT = read_transcript(TRANSCRIPT_FILE)

    # Summarize the transcript
    SUMMARY = summarize_transcript(TRANSCRIPT)
    if SUMMARY:
        print("Summary:")
        print(SUMMARY)
    else:
        print("Failed to generate summary.")

    # Allow user to ask multiple questions
    while True:
        question = input(
            "\nEnter a question about the transcript (or 'exit' to quit): "
        )
        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        answer = answer_question(TRANSCRIPT, question)
        if answer:
            print("\nAnswer:")
            print(answer)
        else:
            print("Failed to generate answer.")


if __name__ == "__main__":
    main()
