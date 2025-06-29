// transcribe.cpp
// A C++ program to download YouTube audio, split it into chunks, and transcribe using whisper.cpp CLI

#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

// Helper to run system commands
void run_command(const std::string &cmd)
{
    std::cout << "Running: " << cmd << std::endl;
    int ret = std::system(cmd.c_str());
    if (ret != 0)
    {
        std::cerr << "Command failed with code " << ret << ": " << cmd << std::endl;
        std::exit(ret);            // <— bail out immediately on download or split failure
    }
}

// Download audio from YouTube as MP3
std::string download_audio(const std::string &url)
{
    // force the name to audio.mp3, so no weird shell-parsing errors
    std::string cmd =
        "yt-dlp --no-warnings -x --audio-format mp3 --audio-quality 192 "
        "-o audio.mp3 \"" +
        url + "\"";
    run_command(cmd);
    return "audio.mp3";
}

// Split audio into fixed-length segments (in seconds)
std::vector<std::string> split_audio(const std::string &audio_file, int chunk_length_sec = 600)
{
    std::vector<std::string> chunks;
    std::string cmd = "ffmpeg -hide_banner -loglevel error -i \"" + audio_file +
                      "\" -f segment -segment_time " + std::to_string(chunk_length_sec) +
                      " -c copy chunk_%03d.mp3";
    run_command(cmd);

    for (auto &p : fs::directory_iterator(fs::current_path()))
    {
        auto name = p.path().filename().string();
        if (p.path().extension() == ".mp3" && name.rfind("chunk_", 0) == 0)
            chunks.push_back(p.path().string());
    }
    std::sort(chunks.begin(), chunks.end());
    return chunks;
}

// Transcribe each chunk using whisper.cpp CLI and assemble
std::string transcribe_chunks(const std::vector<std::string> &chunks, const std::string &model_path)
{
    std::string transcription;
    for (const auto &chunk : chunks)
    {
        std::cout << "Processing: " << chunk << std::endl;
        // Convert MP3 chunk to WAV (required by whisper.cpp)
        std::string wav_file = fs::path(chunk).stem().string() + ".wav";
        run_command("ffmpeg -hide_banner -loglevel error -i " + chunk + " -ar 16000 -ac 1 " + wav_file);

        // Transcribe with whisper.cpp CLI, redirect output to text file
        std::string out_txt = wav_file + ".txt";
        std::string cmd = "./whisper.cpp/build/bin/whisper-cli -m " + model_path + " -f " + wav_file + " > " + out_txt + " 2>&1";
        run_command(cmd);

        // Read transcription text
        std::ifstream ifs(out_txt);
        if (ifs)
        {
            transcription += std::string(fs::path(out_txt).stem().string()) + ":\n";
            std::string line;
            while (std::getline(ifs, line))
            {
                transcription += line + "\n";
            }
            transcription += "\n";
        }
        // Clean up chunk files
        fs::remove(chunk);
        fs::remove(wav_file);
        fs::remove(out_txt);
    }
    return transcription;
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <YouTube URL> <path-to-whisper-model>" << std::endl;
        return 1;
    }
    std::string url = argv[1];
    std::string model_path = argv[2];

    // Step 1: Download audio
    std::string audio_file = download_audio(url);

    // Step 2: Split into chunks
    auto chunks = split_audio(audio_file);

    // Step 3: Transcribe each chunk
    std::string result = transcribe_chunks(chunks, model_path);

    // Clean up original audio
    fs::remove(audio_file);

    // Output full transcription
    std::cout << "----- Transcription Start -----\n";
    std::cout << result << std::endl;
    std::cout << "----- Transcription End -----" << std::endl;

    return 0;
}
