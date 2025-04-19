import os
import re
import subprocess
import threading
import warnings
import torch  # Добавить эту строку в секцию импортов
import requests
from datetime import timedelta
import whisper_timestamped
from tkinter import *
from tkinter import ttk, filedialog, messagebox

warnings.filterwarnings("ignore", category=UserWarning)


class VideoSplitterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Splitter with Subtitles")
        self.root.geometry("800x600")

        self.media_file = StringVar()
        self.progress_value = DoubleVar()
        self.status_text = StringVar()
        self.is_processing = False

        self.create_widgets()

    def create_widgets(self):
        file_frame = ttk.LabelFrame(self.root, text="Медиафайл")
        file_frame.pack(padx=10, pady=5, fill=X)

        ttk.Button(file_frame, text="Выбрать файл",
                   command=self.select_media_file).pack(side=LEFT, padx=5)

        self.file_entry = ttk.Entry(file_frame,
                                    textvariable=self.media_file,
                                    width=70)
        self.file_entry.pack(side=LEFT, fill=X, expand=True, padx=5)

        timestamp_frame = ttk.LabelFrame(self.root, text="Таймкоды")
        timestamp_frame.pack(padx=10, pady=5, fill=BOTH, expand=True)

        self.text_area = Text(timestamp_frame, wrap=WORD, height=15)
        self.text_area.pack(padx=5, pady=5, fill=BOTH, expand=True)

        example_label = ttk.Label(timestamp_frame,
                                  text="Формат: ВРЕМЯ ТЕМА (например: 3:20 Начало урока)")
        example_label.pack(side=BOTTOM, pady=5)

        control_frame = ttk.Frame(self.root)
        control_frame.pack(padx=10, pady=10, fill=X)

        ttk.Button(control_frame, text="Очистить",
                   command=self.clear_fields).pack(side=LEFT, padx=5)

        self.process_btn = ttk.Button(control_frame, text="Запустить обработку",
                                      command=self.toggle_processing)
        self.process_btn.pack(side=RIGHT, padx=5)

        self.progress = ttk.Progressbar(self.root,
                                        variable=self.progress_value,
                                        maximum=100,
                                        orient=HORIZONTAL)
        self.progress.pack(pady=10, padx=10, fill=X)

        status_label = ttk.Label(self.root, textvariable=self.status_text)
        status_label.pack(pady=5)

    def select_media_file(self):
        filetypes = (
            ('Медиафайлы', '*.mp4 *.avi *.mkv *.mov *.mp3 *.wav *.ogg'),
            ('Все файлы', '*.*')
        )
        filename = filedialog.askopenfilename(title="Выберите медиафайл",
                                              filetypes=filetypes)
        if filename:
            self.media_file.set(filename)

    def clear_fields(self):
        self.media_file.set('')
        self.text_area.delete('1.0', END)
        self.progress_value.set(0)
        self.status_text.set('')

    def parse_timestamps(self):
        text = self.text_area.get("1.0", "end-1c")
        segments = []
        pattern = r'(\d+:\d+:\d+|\d+:\d+)\s+(.+)'

        for line in text.split('\n'):
            line = line.strip()
            if line:
                match = re.match(pattern, line)
                if match:
                    time_str, title = match.groups()
                    try:
                        parts = list(map(int, time_str.replace(':', ' ').split()))
                        if len(parts) == 3:
                            seconds = parts[0] * 3600 + parts[1] * 60 + parts[2]
                        else:
                            seconds = parts[0] * 60 + parts[1]
                        segments.append({'time': seconds, 'title': title})
                    except:
                        messagebox.showerror("Ошибка", f"Неверный формат времени: {time_str}")
                        return None
                else:
                    messagebox.showerror("Ошибка", f"Неверный формат строки: {line}")
                    return None

        if len(segments) < 1:
            messagebox.showerror("Ошибка", "Должен быть хотя бы один таймкод")
            return None

        segments.sort(key=lambda x: x['time'])
        return segments

    def toggle_processing(self):
        if self.is_processing:
            self.stop_processing()
        else:
            self.start_processing()

    def start_processing(self):
        if not self.media_file.get():
            messagebox.showerror("Ошибка", "Выберите медиафайл!")
            return

        segments = self.parse_timestamps()
        if not segments:
            return

        self.is_processing = True
        self.process_btn.config(text="Остановить")
        self.status_text.set("Подготовка...")

        threading.Thread(target=self.process_media,
                         args=(self.media_file.get(), segments),
                         daemon=True).start()

    def stop_processing(self):
        self.is_processing = False
        self.process_btn.config(text="Запустить обработку")
        self.status_text.set("Обработка остановлена")

    def process_media(self, input_file, segments):
        try:
            output_dir = os.path.join(os.getcwd(), "output_clips")
            os.makedirs(output_dir, exist_ok=True)

            self.update_status("Конвертация в аудио...", 10)
            audio_file = self.convert_to_audio(input_file)

            self.update_status("Транскрибация...", 30)
            transcription = self.transcribe_audio(audio_file)

            self.update_status("Разделение медиафайла...", 60)
            clips = self.split_media(input_file, segments, output_dir)

            self.update_status("Генерация субтитров...", 80)
            self.create_subtitles(transcription, clips, output_dir)

            os.remove(audio_file)
            self.update_status("Обработка завершена!", 100)
            messagebox.showinfo("Готово", f"Результаты сохранены в:\n{output_dir}")

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
        finally:
            self.is_processing = False
            self.root.after(0, lambda: self.process_btn.config(text="Запустить обработку"))

    def update_status(self, message, progress):
        self.root.after(0, lambda: self.status_text.set(message))
        self.root.after(0, lambda: self.progress_value.set(progress))

    def convert_to_audio(self, input_file):
        audio_file = "temp_audio.wav"
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

    def transcribe_audio(self, audio_file):
        # Автовыбор устройства с fallback на CPU
        if torch.cuda.is_available():
            device = "cuda"
            print("[Система] Используется GPU для ускорения транскрибации")
        else:
            device = "cpu"
            print("[Система] GPU не обнаружен, используется CPU")

        model = whisper_timestamped.load_model(
            "small",
            device=device  # Используем выбранное устройство
        )

        audio = whisper_timestamped.load_audio(audio_file)
        return whisper_timestamped.transcribe(
            model,
            audio,
            language="ru",
            vad=True,
            beam_size=5
        )

    def split_media(self, input_file, segments, output_dir):
        file_ext = os.path.splitext(input_file)[1][1:]
        clips = []

        for i in range(len(segments)):
            if not self.is_processing:
                break

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

            subprocess.run(cmd, check=True,
                           stderr=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL)

            clips.append({
                'start': start,
                'end': end,
                'file': output_file,
                'title': title
            })

        return clips

    def generate_analysis(self, text, output_path):
        API_KEY = "sk-34032fc2894c45b2870fefa6d882860e"
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

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }

        try:
            response = requests.post(API_URL, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            analysis = result['choices'][0]['message']['content']

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Анализ контента:\n{analysis}")

        except Exception as e:
            self.log_error(f"Ошибка анализа: {str(e)}")

    def log_error(self, message):
        self.root.after(0, lambda: self.status_text.set(message))
        self.progress_value.set(0)

    def create_subtitles(self, transcription, clips, output_dir):
        for clip in clips:
            if not self.is_processing:
                break

            start = clip['start']
            end = clip['end'] or float('inf')
            clip_text = []

            segments = [
                s for s in transcription['segments']
                if s['start'] >= start and s['end'] <= end
            ]

            if segments:
                base_name = os.path.splitext(clip['file'])[0]
                srt_path = f"{base_name}.srt"
                txt_path = f"{base_name}_analysis.txt"

                with open(srt_path, 'w', encoding='utf-8') as f:
                    for idx, seg in enumerate(segments, 1):
                        start_time = timedelta(seconds=seg['start'] - start)
                        end_time = timedelta(seconds=seg['end'] - start)
                        text_line = seg['text'].strip()
                        time_code = f"[{self.format_time(start_time)}] "
                        clip_text.append(time_code + text_line)

                        f.write(
                            f"{idx}\n"
                            f"{self.format_time(start_time)} --> {self.format_time(end_time)}\n"
                            f"{text_line}\n\n"
                        )

                if clip_text:
                    full_text = ' '.join(clip_text)
                    self.generate_analysis(full_text, txt_path)

    def format_time(self, td):
        total_seconds = td.total_seconds()
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        return f"{hours:02}:{minutes:02}:{seconds:06.3f}".replace('.', ',')


if __name__ == "__main__":
    root = Tk()
    app = VideoSplitterApp(root)
    root.mainloop()