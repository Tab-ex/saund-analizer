"""
RTSP аудио рекордер
===================
Подключается к RTSP потоку, извлекает аудио и записывает отрезки по 30 секунд

================================================================================
РУКОВОДСТВО ПО ИСПОЛЬЗОВАНИЮ
================================================================================

1. БАЗОВЫЙ ЗАПУСК (запись с одной камеры):
   python rtsp_audio_recorder.py rtsp://192.168.1.100:554/stream

2. ЗАПИСЬ С НЕСКОЛЬКИХ КАМЕР (параллельно в разных терминалах):
   Терминал 1: python rtsp_audio_recorder.py rtsp://camera1:554/stream -o rec_cam1
   Терминал 2: python rtsp_audio_recorder.py rtsp://camera2:554/stream -o rec_cam2
   Терминал 3: python rtsp_audio_recorder.py rtsp://camera3:554/stream -o rec_cam3
   
   ВАЖНО: Используйте разные папки (-o) для каждой камеры!

3. ПАКЕТНЫЙ ЗАПУСК ВСЕХ КАМЕР (создайте файл start_all.bat):
   @echo off
   start cmd /k python rtsp_audio_recorder.py rtsp://camera1:554/stream -o rec_cam1
   start cmd /k python rtsp_audio_recorder.py rtsp://camera2:554/stream -o rec_cam2
   start cmd /k python rtsp_audio_recorder.py rtsp://camera3:554/stream -o rec_cam3

================================================================================
ПАРАМЕТРЫ КОМАНДНОЙ СТРОКИ
================================================================================

Обязательный параметр:
  rtsp_url                    RTSP ссылка потока
                              Пример: rtsp://192.168.1.100:554/stream
                              С авторизацией: rtsp://user:pass@192.168.1.100:554/stream

Опциональные параметры:
  --output, -o                Папка для записи (по умолчанию: rtsp_recordings)
                              Рекомендуется: отдельная папка для каждой камеры
                              
  --duration, -d              Длительность одного сегмента в секундах (по умолчанию: 30)
                              Пример: -d 60 (запись минутными отрезками)
                              
  --segments, -n              Максимальное количество сегментов (по умолчанию: бесконечно)
                              Пример: -n 120 (запишет 120 сегментов и остановится)
                              
  --ffmpeg, -f                Путь к ffmpeg.exe (по умолчанию: ffmpeg.exe в корне проекта)
                              Обычно не нужно указывать, если ffmpeg.exe лежит рядом

================================================================================
ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ
================================================================================

# Запись с камеры в текущую папку, сегменты по 30 секунд (базовый)
python rtsp_audio_recorder.py rtsp://192.168.1.100:554/stream

# Запись с указанием папки и длительности сегмента 60 секунд
python rtsp_audio_recorder.py rtsp://192.168.1.100:554/stream -o my_recordings -d 60

# Запись только 10 сегментов (5 минут при d=30) и остановка
python rtsp_audio_recorder.py rtsp://192.168.1.100:554/stream -n 10

# Запись с камерой с авторизацией (логин/пароль в URL)
python rtsp_audio_recorder.py rtsp://admin:password123@192.168.1.100:554/stream

# Параллельная запись с двух камер в разные папки
# Терминал 1:
python rtsp_audio_recorder.py rtsp://camera1:554/stream -o recordings_cam1
# Терминал 2:
python rtsp_audio_recorder.py rtsp://camera2:554/stream -o recordings_cam2

================================================================================
ФОРМАТ ЗАПИСИ
================================================================================
  Формат: WAV (PCM 16-bit, 44.1 kHz, моно)
  Размер: ~5 MB за 1 минуту, ~315 MB за 1 час
  Остановка: Ctrl+C в терминале

================================================================================
"""

import subprocess
import threading
import time
import os
from pathlib import Path
from datetime import datetime
import signal
import sys


class RTSPAudioRecorder:
    """Запись аудио с RTSP потока"""

    def __init__(self, rtsp_url, output_dir="rtsp_recordings", segment_duration=30, ffmpeg_path="ffmpeg.exe"):
        """
        Args:
            rtsp_url: RTSP ссылка (например, rtsp://192.168.1.100:554/stream)
            output_dir: Папка для сохранения записей
            segment_duration: Длительность одного отрезка в секундах
            ffmpeg_path: Путь к ffmpeg.exe (по умолчанию: ffmpeg.exe в корне проекта)
        """
        self.rtsp_url = rtsp_url
        self.output_dir = Path(output_dir)
        self.segment_duration = segment_duration
        self.ffmpeg_path = Path(ffmpeg_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.is_recording = False
        self.process = None
        self.record_thread = None
        self.stop_event = threading.Event()

    def _generate_filename(self, index=0):
        """Генерация имени файла с таймстампом"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if index > 0:
            return f"audio_{timestamp}_part{index:03d}.wav"
        return f"audio_{timestamp}.wav"

    def _record_segment(self, segment_index):
        """Запись одного сегмента аудио"""
        output_file = self.output_dir / self._generate_filename(segment_index)

        # FFmpeg команда для извлечения аудио из RTSP
        ffmpeg_cmd = [
            str(self.ffmpeg_path),  # Путь к ffmpeg.exe
            '-y',                    # Перезаписывать файл
            '-rtsp_transport', 'tcp', # TCP транспорт
            '-i', self.rtsp_url,     # RTSP URL
            '-t', str(self.segment_duration), # Длительность
            '-vn',                   # Без видео
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', '44100',          # Sample rate 44.1kHz
            '-ac', '1',              # Моно
            '-loglevel', 'error',    # Минимум логов
            str(output_file)
        ]

        print(f"🎙 Запись сегмента {segment_index + 1}: {output_file.name}")
        print(f"   Длительность: {self.segment_duration} сек")

        try:
            self.process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Ждём завершения или остановки
            while self.process.poll() is None:
                if self.stop_event.is_set():
                    self.process.terminate()
                    break
                time.sleep(0.5)

            if self.process.returncode == 0:
                print(f"✅ Сохранён: {output_file.name}")
                return True
            else:
                stderr = self.process.stderr.read().decode('utf-8', errors='ignore')
                print(f"❌ Ошибка записи: {stderr}")
                return False

        except FileNotFoundError:
            print(f"❌ FFmpeg не найден: {self.ffmpeg_path}")
            print("   Положите ffmpeg.exe в корень проекта или укажите путь через --ffmpeg")
            return False
        except Exception as e:
            print(f"❌ Исключение: {e}")
            return False

    def start_continuous_recording(self, max_segments=None):
        """
        Непрерывная запись сегментов

        Args:
            max_segments: Максимальное количество сегментов (None = бесконечно)
        """
        if self.is_recording:
            print("⚠ Запись уже запущена")
            return

        self.is_recording = True
        self.stop_event.clear()

        print(f"📡 Подключение к RTSP: {self.rtsp_url}")
        print(f"📁 Папка сохранения: {self.output_dir}")
        print(f"⏱ Длительность сегмента: {self.segment_duration} сек")
        print(f"🔢 Макс сегментов: {max_segments if max_segments else '∞'}")
        print("Нажмите Ctrl+C для остановки\n")

        segment_index = 0

        try:
            while not self.stop_event.is_set():
                if max_segments and segment_index >= max_segments:
                    print(f"\n✅ Достигнут лимит: {max_segments} сегментов")
                    break

                success = self._record_segment(segment_index)
                segment_index += 1

                if not success and not self.stop_event.is_set():
                    print("⚠ Повторная попытка через 5 секунд...")
                    time.sleep(5)

        except KeyboardInterrupt:
            print("\n⏹ Остановка по команде пользователя")
        finally:
            self.stop()

    def start_background_recording(self, max_segments=None):
        """Запись в фоновом потоке"""
        self.record_thread = threading.Thread(
            target=self.start_continuous_recording,
            args=(max_segments,),
            daemon=True
        )
        self.record_thread.start()
        return self.record_thread

    def stop(self):
        """Остановка записи"""
        if not self.is_recording:
            return

        print("\n⏹ Остановка записи...")
        self.stop_event.set()
        self.is_recording = False

        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

        if self.record_thread and self.record_thread.is_alive():
            self.record_thread.join(timeout=10)

        print("✅ Запись остановлена")

    def get_recorded_files(self):
        """Получить список записанных файлов"""
        return sorted(self.output_dir.glob("audio_*.wav"))


# ==================== CLI интерфейс ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Запись аудио с RTSP потока",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python rtsp_audio_recorder.py rtsp://192.168.1.100:554/stream
  python rtsp_audio_recorder.py rtsp://user:pass@192.168.1.100/stream -d 60
  python rtsp_audio_recorder.py rtsp://192.168.1.100/stream -o ./recordings -n 10
        """
    )

    parser.add_argument("rtsp_url", help="RTSP ссылка потока")
    parser.add_argument("--output", "-o", default="rtsp_recordings",
                        help="Папка для записей (по умолчанию: rtsp_recordings)")
    parser.add_argument("--duration", "-d", type=int, default=30,
                        help="Длительность сегмента в секундах (по умолчанию: 30)")
    parser.add_argument("--segments", "-n", type=int, default=None,
                        help="Максимальное количество сегментов (по умолчанию: бесконечно)")
    parser.add_argument("--ffmpeg", "-f", default="ffmpeg.exe",
                        help="Путь к ffmpeg.exe (по умолчанию: ffmpeg.exe в корне проекта)")

    args = parser.parse_args()

    # Обработка Ctrl+C
    def signal_handler(sig, frame):
        print("\n\n⏹ Получен сигнал остановки")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Запуск рекордера
    recorder = RTSPAudioRecorder(
        rtsp_url=args.rtsp_url,
        output_dir=args.output,
        segment_duration=args.duration,
        ffmpeg_path=args.ffmpeg
    )

    try:
        recorder.start_continuous_recording(max_segments=args.segments)
    except KeyboardInterrupt:
        recorder.stop()
