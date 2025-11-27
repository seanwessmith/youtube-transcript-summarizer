// transcribe-mp4.cpp – extract audio from an MP4, split, transcribe with whisper.cpp
// Build: g++ -std=c++17 -O2 transcribe-mp4.cpp -o transcribe-mp4
// Usage:   ./transcribe-mp4 <path-to-video.mp4> <path-to-whisper-model>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static std::string trim(const std::string &s)
{
    std::size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start])))
        ++start;

    std::size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1])))
        --end;

    return s.substr(start, end - start);
}

static bool starts_with(const std::string &txt, const std::string &prefix)
{
    return txt.rfind(prefix, 0) == 0;
}

static std::string clean_transcript_line(const std::string &raw_line)
{
    std::string line = trim(raw_line);
    if (line.empty())
        return {};

    static const std::vector<std::string> metadata_prefixes = {
        "whisper_",
        "ggml_",
        "system_info:",
        "main:",
        "whisper_print_timings:",
        "chunk_"
    };

    for (const auto &prefix : metadata_prefixes)
    {
        if (starts_with(line, prefix))
            return {};
    }

    if (!line.empty() && line.front() == '[')
    {
        const auto pos = line.find(']');
        if (pos != std::string::npos)
            line = trim(line.substr(pos + 1));
        else
            return {};
    }

    if (!line.empty() && line.front() == '-')
        line = trim(line.substr(1));

    return line;
}

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
  1. Extract audio from MP4 (stereo mp3)
──────────────────────────────────────────────────────────────*/
static std::string extract_audio_mp3(const std::string &video_file)
{
    const std::string audio_file = "audio.mp3";
    const std::string cmd =
        "ffmpeg -hide_banner -loglevel error -i \"" + video_file +
        "\" -vn -acodec libmp3lame -ar 44100 -ac 2 \"" + audio_file + "\"";
    run_cmd(cmd, /*echo=*/true);
    return audio_file;
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
                "\" -f \"" + wav + "\" > \"" + txt + "\"");

        /* Append transcript */
        if (std::ifstream ifs{txt})
        {
            std::string line;
            bool wrote_line = false;
            while (std::getline(ifs, line))
            {
                const auto cleaned = clean_transcript_line(line);
                if (cleaned.empty())
                    continue;

                transcription += cleaned;
                transcription += '\n';
                wrote_line = true;
            }

            if (wrote_line)
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
                  << " <path-to-video.mp4> <path-to-whisper-model>" << std::endl;
        return 1;
    }

    const std::string video_path = argv[1];
    const std::string model_path = argv[2];

    if (!fs::exists(video_path) || !fs::is_regular_file(video_path))
    {
        std::cerr << "Video file not found: " << video_path << std::endl;
        return 1;
    }

    const std::string audio_file = extract_audio_mp3(video_path);
    const auto chunks = split_audio(audio_file);
    const auto script = transcribe_chunks(chunks, model_path);

    fs::remove(audio_file);

    const std::string transcript_filename =
        fs::path(video_path).stem().string() + "_transcript.txt";

    if (std::ofstream ofs{transcript_filename})
    {
        ofs << "----- Transcription Start -----\n"
            << script
            << "----- Transcription End -----\n";
        std::cout << "Transcript saved to: " << transcript_filename << std::endl;
    }
    else
    {
        std::cerr << "Warning: failed to write transcript file '"
                  << transcript_filename << "'." << std::endl;
    }

    std::cout << "\n----- Transcription Start -----\n"
              << script
              << "----- Transcription End -----" << std::endl;
    return 0;
}

