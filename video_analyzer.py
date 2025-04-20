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

if not Path(os.path.expanduser("~/.cache/torch/hub/checkpoints/silero_vad.jit")).exists():
    os.system("python -c 'from silero import vad; vad()'")
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device.upper()} for transcription")

    model = whisper_timestamped.load_model("small", device=device)
    audio = whisper_timestamped.load_audio(audio_file)

    return whisper_timestamped.transcribe(
        model,
        audio,
        language="ru",
        vad=True,
        beam_size=5
    )


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

        cmd = ['ffmpeg', '-hide_banner', '-y',
               '-ss', str(start), '-i', input_file]

        if end:
            cmd += ['-t', str(end - start)]

        cmd += ['-c', 'copy', output_file]

        try:
            subprocess.run(cmd, check=True,
                           stderr=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"Error splitting segment {i + 1}: {e}")
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
    1. Краткая выжимка (до 500 символов) 
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
    for clip in clips:
        start = clip['start']
        end = clip['end'] or float('inf')
        clip_text = []

        segments = [s for s in transcription['segments'] if s['start'] >= start and s['end'] <= end]

        if segments:
            base_name = os.path.splitext(clip['file'])[0]
            srt_path = f"{base_name}.srt"
            txt_path = f"{base_name}_analysis.txt"

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
                full_text = ' '.join(clip_text)
                generate_analysis(full_text, txt_path)


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

        print("[1/4] Converting to audio...")
        audio_file = convert_to_audio(input_file)

        print("[2/4] Transcribing audio...")
        transcription = transcribe_audio(audio_file)

        print("[3/4] Splitting media file...")
        clips = split_media(input_file, segments, output_dir)

        print("[4/4] Generating subtitles and analysis...")
        create_subtitles(transcription, clips, output_dir)

        os.remove(audio_file)
        print(f"\nProcessing complete! Results saved to: {output_dir}")

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