import re
import sys
import argparse
import subprocess
import warnings
import torch
import requests
from datetime import timedelta
import whisper_timestamped
from pathlib import Path
import os
os.environ['SILERO_CACHE_DIR'] = os.path.expanduser('~/.cache/silero')
os.makedirs(os.environ['SILERO_CACHE_DIR'], exist_ok=True)

# Скачать модели при первом запуске
if not Path(os.path.expanduser("~/.cache/whisper/small.pt")).exists():
    os.system("whisper --model small --download-model")

warnings.filterwarnings("ignore", category=UserWarning)


def parse_timestamps(text):
    segments = []
    pattern = r'(\d+:\d+:\d+|\d+:\d+)\s+(.+)'

    for line in text.split('\n'):
        line = line.strip()
        if line:
            match = re.match(pattern, line)
            if not match:
                raise ValueError(f"Invalid line format: {line}")

            time_str, title = match.groups()
            try:
                parts = list(map(int, time_str.replace(':', ' ').split()))
                if len(parts) == 3:
                    seconds = parts[0] * 3600 + parts[1] * 60 + parts[2]
                else:
                    seconds = parts[0] * 60 + parts[1]
                segments.append({'time': seconds, 'title': title})
            except:
                raise ValueError(f"Invalid time format: {time_str}")

    if len(segments) < 1:
        raise ValueError("At least one timestamp required")

    segments.sort(key=lambda x: x['time'])
    return segments


def convert_to_audio(input_file):
    audio_file = "temp_audio.wav"
    try:
        subprocess.run([
            'ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-i', input_file,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y', audio_file
        ], check=True)
        return audio_file
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
        sys.exit(1)


def transcribe_audio(audio_file):
    def get_device_info():
        """Get detailed device information and handle CUDA initialization"""
        try:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                # Check CUDA version compatibility
                cuda_version = torch.version.cuda
                print(f"CUDA version: {cuda_version}")
                print(f"GPU: {torch.cuda.get_device_name(0)}")
                print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
                return device
            else:
                print("CUDA not available, falling back to CPU")
                return torch.device("cpu")
        except Exception as e:
            print(f"Error initializing CUDA: {e}")
            return torch.device("cpu")

    def load_model_with_fallback(device):
        """Load model with error handling and fallback"""
        try:
            model = whisper_timestamped.load_model("small", device=device)
            return model
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("GPU out of memory, attempting to clear cache and retry...")
                torch.cuda.empty_cache()
                try:
                    model = whisper_timestamped.load_model("small", device=device)
                    return model
                except RuntimeError:
                    print("Still out of memory, falling back to CPU")
                    return whisper_timestamped.load_model("small", device="cpu")
            else:
                print(f"GPU error: {e}\nFalling back to CPU")
                return whisper_timestamped.load_model("small", device="cpu")

    # Initialize device with error handling
    device = get_device_info()
    print(f"Using {device.type.upper()} for transcription")

    # Load model with fallback mechanism
    model = load_model_with_fallback(device)
    
    try:
        # Load audio and perform transcription
        audio = whisper_timestamped.load_audio(audio_file)
        
        # Create a simple progress indicator
        print("Starting transcription...")
        result = model.transcribe(
            audio,
            language="ru",
            beam_size=5
        )
        print("✓ Transcription complete")
        return result
    except Exception as e:
        print(f"Error during transcription: {e}")
        if device.type == "cuda":
            print("Attempting CPU fallback...")
            model = whisper_timestamped.load_model("small", device="cpu")
            audio = whisper_timestamped.load_audio(audio_file)
            print("Starting CPU transcription...")
            result = model.transcribe(
                audio,
                language="ru",
                beam_size=5
            )
            print("✓ CPU transcription complete")
            return result
        else:
            raise
    finally:
        # Clean up GPU memory
        if device.type == "cuda":
            torch.cuda.empty_cache()


def split_media(input_file, segments, output_dir):
    file_ext = os.path.splitext(input_file)[1][1:]
    clips = []

    for i in range(len(segments)):
        start = segments[i]['time']
        end = segments[i + 1]['time'] if i + 1 < len(segments) else None
        title = re.sub(r'[\\/*?:"<>|]', "", segments[i]['title'])

        output_file = os.path.join(
            output_dir,
            f"{i + 1:03d}_{title.replace(' ', '_')}.{file_ext}"
        )

        print(f"Processing segment {i + 1}/{len(segments)}: {title}")
        print(f"Time range: {format_time(timedelta(seconds=start))} - {format_time(timedelta(seconds=end)) if end else 'end'}")

        cmd = ['ffmpeg', '-hide_banner', '-y',
               '-ss', str(start), '-i', input_file]

        if end:
            cmd += ['-t', str(end - start)]

        cmd += ['-c', 'copy', output_file]

        try:
            subprocess.run(cmd, check=True,
                           stderr=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL)
            print(f"✓ Segment {i + 1} saved: {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error processing segment {i + 1}: {e}")
            continue

        clips.append({
            'start': start,
            'end': end,
            'file': output_file,
            'title': title
        })

    return clips


def generate_analysis(text, output_path):
    API_KEY_FILE = "deepseek_api_key.txt"

    if not os.path.exists(API_KEY_FILE):
        raise FileNotFoundError(f"API key file {API_KEY_FILE} not found")

    with open(API_KEY_FILE, 'r', encoding='utf-8') as f:
        API_KEY = f.read().strip()

    if not API_KEY:
        raise ValueError("API key is empty in the key file")

    API_URL = "https://api.deepseek.com/v1/chat/completions"

    prompt = f"""
    Проанализируй текст для контент-мейкера:
    1. Краткая выжимка (до 1000 символов) 
    2. Ключевые моменты с таймкодами [ЧЧ:ММ:СС]
    3. Ключевые слова
    4. Анализ для YouTube
    5. Анализ для TikTok

    Текст с таймкодами:
    {text}
    """

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "temperature": 0.7}

    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        analysis = response.json()['choices'][0]['message']['content']

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Анализ контента:\n{analysis}")

    except Exception as e:
        print(f"Analysis error: {str(e)}")


def create_subtitles(transcription, clips, output_dir):
    print("\nGenerating subtitles and analysis for each clip...")
    for i, clip in enumerate(clips, 1):
        print(f"\nProcessing clip {i}/{len(clips)}: {clip['title']}")
        start = clip['start']
        end = clip['end'] or float('inf')
        clip_text = []

        segments = [s for s in transcription['segments'] if s['start'] >= start and s['end'] <= end]

        if segments:
            base_name = os.path.splitext(clip['file'])[0]
            srt_path = f"{base_name}.srt"
            txt_path = f"{base_name}_analysis.txt"

            print(f"Creating subtitle file: {srt_path}")
            with open(srt_path, 'w', encoding='utf-8') as f:
                for idx, seg in enumerate(segments, 1):
                    start_time = timedelta(seconds=seg['start'] - start)
                    end_time = timedelta(seconds=seg['end'] - start)
                    text_line = seg['text'].strip()
                    time_code = f"[{format_time(start_time)}] "
                    clip_text.append(time_code + text_line)

                    f.write(f"{idx}\n"
                            f"{format_time(start_time)} --> {format_time(end_time)}\n"
                            f"{text_line}\n\n")

            if clip_text:
                print(f"Generating analysis file: {txt_path}")
                full_text = ' '.join(clip_text)
                generate_analysis(full_text, txt_path)
                print(f"✓ Analysis complete for clip {i}")
            else:
                print(f"✗ No text segments found for clip {i}")
        else:
            print(f"✗ No segments found for clip {i}")


def format_time(td):
    total_seconds = td.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:06.3f}".replace('.', ',')


def process_media(input_file, segments):
    try:
        output_dir = os.path.join(os.getcwd(), "output_clips")
        os.makedirs(output_dir, exist_ok=True)

        print("\n=== Video Analysis Process ===")
        print("[1/4] Converting to audio...")
        print(f"Input file: {input_file}")
        audio_file = convert_to_audio(input_file)
        print(f"✓ Audio conversion complete: {audio_file}")

        print("\n[2/4] Transcribing audio...")
        print("Initializing transcription model...")
        transcription = transcribe_audio(audio_file)
        print("✓ Transcription complete")

        print("\n[3/4] Splitting media file...")
        print(f"Processing {len(segments)} segments...")
        clips = split_media(input_file, segments, output_dir)
        print(f"✓ Successfully split into {len(clips)} clips")

        print("\n[4/4] Generating subtitles and analysis...")
        print("Creating subtitles and analysis files...")
        create_subtitles(transcription, clips, output_dir)
        print("✓ Subtitles and analysis generated")

        os.remove(audio_file)
        print("\n=== Processing Complete ===")
        print(f"Results saved to: {output_dir}")
        print(f"Total segments processed: {len(clips)}")
        print("===========================")

    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Video Analyzer')
    parser.add_argument('--input', required=True, help='Input media file path')
    parser.add_argument('--timestamps', required=True,
                        help='Path to timestamps file (format: HH:MM:SS Title)')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        sys.exit(1)

    try:
        with open(args.timestamps, 'r', encoding='utf-8') as f:
            segments = parse_timestamps(f.read())
    except Exception as e:
        print(f"Timestamp error: {str(e)}")
        sys.exit(1)

    process_media(args.input, segments)


if __name__ == "__main__":
    main()