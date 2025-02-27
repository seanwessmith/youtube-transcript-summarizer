import sys
import yt_dlp
import whisperx
from pydub import AudioSegment
import logging
import os

def download_audio(youtube_url):
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "outtmpl": "audio.%(ext)s",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return "audio.mp3"

def split_audio(audio_file, chunk_length_ms=600000):  # 10-minute chunks
    audio = AudioSegment.from_mp3(audio_file)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk_file = f"chunk_{i // chunk_length_ms}.mp3"
        chunk.export(chunk_file, format="mp3")
        chunks.append(chunk_file)
    return chunks

def transcribe_chunks(chunks):
    try:
        model = whisperx.load_model("base", device="cpu", compute_type="float32", language="en")
        logging.info("WhisperX model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load WhisperX model: {e}")
        return "Model loading failed"

    transcription = ""
    for i, chunk in enumerate(chunks):
        logging.info(f"Transcribing chunk {i + 1}/{len(chunks)}: {chunk}")
        try:
            result = model.transcribe(chunk, language="en")
            logging.info(f"Raw result: {result}")
            if "segments" in result:
                text = " ".join([seg["text"] for seg in result["segments"]])
            else:
                text = result.get("text", "[No text detected]")
            transcription += text + " "
            os.remove(chunk)  # Clean up
        except Exception as e:
            logging.error(f"Error transcribing chunk {i + 1}: {e}")
            transcription += "[Transcription failed] "
            if os.path.exists(chunk):
                os.remove(chunk)
    return transcription.strip()


if __name__ == "__main__":
    youtube_url = sys.argv[1]
    audio_file = download_audio(youtube_url)
    chunks = split_audio(audio_file)
    transcription = transcribe_chunks(chunks)
    os.remove(audio_file)  # Clean up
    print(transcription)