// transcribe.cpp – download YouTube audio, split, transcribe with whisper.cpp and show a live progress bar
// Build: g++ -std=c++17 -O2 transcribe.cpp -o transcribe
// Usage:   ./transcribe <YouTube URL> <path-to-whisper-model>
// Example: ./transcribe https://youtu.be/dQw4w9WgXcQ ./whisper.cpp/models/ggml-base.en.bin

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

/*───────────────────────────────────────────────────────────────
  Utility – ASCII progress bar (single-line, 50 chars wide)
──────────────────────────────────────────────────────────────*/
static void print_progress(std::size_t current, std::size_t total)
{
    constexpr int bar_width = 50;
    const float ratio = static_cast<float>(current) / total;
    const int filled = static_cast<int>(ratio * bar_width);

    std::cout << '\r' << '[';
    for (int i = 0; i < bar_width; ++i)
        std::cout << (i < filled ? '=' : (i == filled ? '>' : ' '));

    std::cout << "] " << std::setw(3) << static_cast<int>(ratio * 100)
              << "% (" << current << '/' << total << ")" << std::flush;

    if (current == total)
        std::cout << std::endl;
}

/*───────────────────────────────────────────────────────────────
  Helper – run external command, abort if it fails
──────────────────────────────────────────────────────────────*/
static void run_cmd(const std::string &cmd, bool echo = false)
{
    if (echo)
        std::cout << "\n> " << cmd << std::endl;
    const int ret = std::system(cmd.c_str());
    if (ret)
    {
        std::cerr << "\nCommand failed (" << ret << "): " << cmd << std::endl;
        std::exit(ret);
    }
}

/*───────────────────────────────────────────────────────────────
  1. Download audio from YouTube (mp3 128 kbps)
──────────────────────────────────────────────────────────────*/
static std::string download_audio(const std::string &url)
{
    const std::string cmd =
        "yt-dlp --no-warnings -x --audio-format mp3 --audio-quality 128 "
        "-o audio.mp3 \"" +
        url + "\"";
    run_cmd(cmd, /*echo=*/true);
    return "audio.mp3";
}

/*───────────────────────────────────────────────────────────────
  2. Split audio into ≤10-min chunks (mp3 copy)
──────────────────────────────────────────────────────────────*/
static std::vector<std::string> split_audio(const std::string &audio_file,
                                            int chunk_len_sec = 600)
{
    const std::string cmd =
        "ffmpeg -hide_banner -loglevel error -i \"" + audio_file +
        "\" -f segment -segment_time " + std::to_string(chunk_len_sec) +
        " -c copy chunk_%03d.mp3";
    run_cmd(cmd, /*echo=*/true);

    std::vector<std::string> chunks;
    for (auto &p : fs::directory_iterator(fs::current_path()))
    {
        const auto name = p.path().filename().string();
        if (p.path().extension() == ".mp3" && name.rfind("chunk_", 0) == 0)
            chunks.push_back(p.path().string());
    }
    std::sort(chunks.begin(), chunks.end());
    if (chunks.empty())
    {
        std::cerr << "No chunks produced – check ffmpeg output." << std::endl;
        std::exit(1);
    }
    return chunks;
}

/*───────────────────────────────────────────────────────────────
  3. Transcribe each chunk and concatenate results
──────────────────────────────────────────────────────────────*/
static std::string transcribe_chunks(const std::vector<std::string> &chunks,
                                     const std::string &model_path)
{
    std::string transcription;
    const std::size_t total = chunks.size();
    std::size_t done = 0;
    print_progress(0, total);

    for (const auto &chunk : chunks)
    {
        /* Convert MP3 → WAV (16 kHz mono) */
        const std::string wav = fs::path(chunk).stem().string() + ".wav";
        run_cmd("ffmpeg -hide_banner -loglevel error -i \"" + chunk +
                "\" -ar 16000 -ac 1 \"" + wav + "\"");

        /* Whisper.cpp CLI */
        const std::string txt = wav + ".txt";
        run_cmd("./whisper.cpp/build/bin/whisper-cli -m \"" + model_path +
                "\" -f \"" + wav + "\" > \"" + txt + "\" 2>&1");

        /* Append transcript */
        if (std::ifstream ifs{txt})
        {
            transcription += fs::path(txt).stem().string() + ":\n";
            transcription.append(std::istreambuf_iterator<char>(ifs), {});
            transcription += '\n';
        }

        /* Clean-up per-chunk artefacts */
        fs::remove(chunk);
        fs::remove(wav);
        fs::remove(txt);

        print_progress(++done, total);
    }
    return transcription;
}

/*───────────────────────────────────────────────────────────────*/
int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <YouTube URL> <path-to-whisper-model>" << std::endl;
        return 1;
    }

    const std::string url = argv[1];
    const std::string model_path = argv[2];

    const std::string audio_file = download_audio(url);
    const auto chunks = split_audio(audio_file);
    const auto script = transcribe_chunks(chunks, model_path);

    fs::remove(audio_file);

    std::cout << "\n----- Transcription Start -----\n"
              << script
              << "----- Transcription End -----" << std::endl;
    return 0;
}