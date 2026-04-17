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
НАДЁЖНОСТЬ И ВОССТАНОВЛЕНИЕ
================================================================================

Механизмы надёжности (включены по умолчанию):

  1. TIMEOUT — автоматическое прерывание зависшего ffmpeg
     По умолчанию: 2x от duration (для 60 сек = 120 сек timeout)

  2. ВАЛИДАЦИЯ ФАЙЛОВ — проверка каждого записанного файла
     - Проверка WAV заголовка
     - Проверка длительности (>= 90% от ожидаемой)
     - Повреждённые файлы → папка invalid/

  3. АВТОПЕРЕПОДКЛЮЧЕНИЕ — при 3 ошибках подряд
     - Проверка связи с камерой
     - Пауза для стабилизации
     - Продолжение записи

  4. HEALTH CHECK — каждые 10 минут
     - Статистика записи
     - Проверка места на диске
     - Логирование в файл

  5. ЛОГИРОВАНИЕ — все события с таймстампами
     - Файл: output_dir/recorder.log
     - Статистика: output_dir/recorder_stats.json

Примеры надёжной записи:
  # Запись на выходные с проверкой файлов:
  python rtsp_audio_recorder.py rtsp://камера/stream -d 60 -r 5

  # Запись с увеличенным timeout (для нестабильной сети):
  python rtsp_audio_recorder.py rtsp://камера/stream -T 180

  # Запись без валидации (быстрее, но менее надёжно):
  python rtsp_audio_recorder.py rtsp://камера/stream --no-validate

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
import wave
import json
import shutil


class RTSPAudioRecorder:
    """Запись аудио с RTSP потока с механизмами надёжности"""

    def __init__(self, rtsp_url, output_dir="rtsp_recordings", segment_duration=30,
                 ffmpeg_path="ffmpeg.exe", max_retries=3, timeout_multiplier=2.0,
                 validate_files=True, log_file=None):
        """
        Args:
            rtsp_url: RTSP ссылка (например, rtsp://192.168.1.100:554/stream)
            output_dir: Папка для сохранения записей
            segment_duration: Длительность одного отрезка в секундах
            ffmpeg_path: Путь к ffmpeg.exe (по умолчанию: ffmpeg.exe в корне проекта)
            max_retries: Максимум повторных попыток при ошибке
            timeout_multiplier: Множитель timeout (2.0 = 2x от duration)
            validate_files: Проверять целостность WAV файлов
            log_file: Путь к файлу лога (None = авто)
        """
        self.rtsp_url = rtsp_url
        self.output_dir = Path(output_dir)
        self.segment_duration = segment_duration
        self.ffmpeg_path = Path(ffmpeg_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Надёжность
        self.max_retries = max_retries
        self.timeout = segment_duration * timeout_multiplier  # Timeout на процесс
        self.validate_files = validate_files

        # Логирование
        if log_file is None:
            log_file = self.output_dir / "recorder.log"
        self.log_file = Path(log_file)
        self.log_entries = []

        self.is_recording = False
        self.process = None
        self.record_thread = None
        self.stop_event = threading.Event()

        # Статистика
        self.stats = {
            'started_at': datetime.now().isoformat(),
            'total_segments': 0,
            'successful_segments': 0,
            'failed_segments': 0,
            'reconnected': 0,
            'invalid_files': 0,
            'last_success': None,
            'errors': []
        }

        self._log("=" * 60)
        self._log(f"📼 RECORDER ЗАПУЩЕН")
        self._log(f"   RTSP: {rtsp_url}")
        self._log(f"   Duration: {segment_duration} сек")
        self._log(f"   Timeout: {self.timeout} сек")
        self._log(f"   Max retries: {max_retries}")
        self._log(f"   Validate: {validate_files}")
        self._log("=" * 60)

    def _generate_filename(self, index=0):
        """Генерация имени файла с таймстампом"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if index > 0:
            return f"audio_{timestamp}_part{index:03d}.wav"
        return f"audio_{timestamp}.wav"

    def _log(self, message, level="INFO"):
        """Логирование с таймстампом"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        self.log_entries.append(log_entry)

        # Печать в консоль
        print(message)

        # Запись в файл (периодически)
        if len(self.log_entries) >= 50:
            self._flush_log()

    def _flush_log(self):
        """Сохранение лога на диск"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                for entry in self.log_entries:
                    f.write(entry + '\n')
            self.log_entries = []
        except Exception as e:
            print(f"⚠ Ошибка записи лога: {e}")

    def _validate_wav_file(self, file_path):
        """
        Проверка целостности WAV файла

        Returns:
            (is_valid, duration_sec, file_size_bytes)
        """
        try:
            if not file_path.exists():
                return False, 0, 0

            file_size = file_path.stat().st_size

            # Минимальный размер WAV файла (44 байта заголовок + данные)
            if file_size < 1000:
                self._log(f"⚠ Файл слишком маленький: {file_size} байт", "ERROR")
                return False, 0, file_size

            # Проверка WAV заголовка
            with wave.open(str(file_path), 'rb') as wf:
                n_frames = wf.getnframes()
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                duration = n_frames / sample_rate

            # Проверка длительности (должно быть >= 90% от ожидаемой)
            expected_duration = self.segment_duration * 0.9

            if duration < expected_duration:
                self._log(
                    f"⚠ Файл повреждён: {duration:.1f} сек < {expected_duration:.1f} сек",
                    "WARNING"
                )
                return False, duration, file_size

            return True, duration, file_size

        except Exception as e:
            self._log(f"⚠ Ошибка проверки файла: {e}", "ERROR")
            return False, 0, 0

    def _record_segment(self, segment_index):
        """Запись одного сегмента аудио с проверками надёжности"""
        output_file = self.output_dir / self._generate_filename(segment_index)

        # Пишем сразу в .wav — потом проверяем и перемещаем если повреждён
        temp_file = self.output_dir / f"_temp_{segment_index:06d}.wav"

        # FFmpeg команда для извлечения аудио из RTSP
        ffmpeg_cmd = [
            str(self.ffmpeg_path),  # Путь к ffmpeg.exe
            '-y',                    # Перезаписывать файл
            '-rtsp_transport', 'tcp', # TCP транспорт
            '-timeout', '10000000',   # Timeout 10 секунд на подключение (в микросекундах)
            '-i', self.rtsp_url,     # RTSP URL
            '-t', str(self.segment_duration), # Длительность
            '-vn',                   # Без видео
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', '44100',          # Sample rate 44.1kHz
            '-ac', '1',              # Моно
            '-loglevel', 'warning',  # Логи для отладки
            str(temp_file)
        ]

        self._log(f"🎙 [{segment_index + 1}] Запись: {output_file.name}")

        try:
            self.process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Ждём завершения С TIMEOUT
            start_time = time.time()
            while self.process.poll() is None:
                elapsed = time.time() - start_time

                # Timeout проверка
                if elapsed > self.timeout:
                    self._log(
                        f"❌ TIMEOUT: ffmpeg работает {elapsed:.0f} сек > {self.timeout:.0f} сек",
                        "ERROR"
                    )
                    self._kill_process()
                    return False

                if self.stop_event.is_set():
                    self._kill_process()
                    return False

                time.sleep(0.5)

            # Проверка кода возврата
            if self.process.returncode == 0:
                # Проверка файла
                if not temp_file.exists():
                    self._log(f"❌ Файл не создан после успешного завершения", "ERROR")
                    return False

                # Валидация WAV
                if self.validate_files:
                    is_valid, duration, size = self._validate_wav_file(temp_file)

                    if not is_valid:
                        self._log(f"❌ Файл повреждён ({duration:.1f} сек, {size} байт)", "WARNING")
                        # Перемещаем в папку повреждённых
                        invalid_dir = self.output_dir / "invalid"
                        invalid_dir.mkdir(exist_ok=True)
                        invalid_file = invalid_dir / temp_file.name
                        shutil.move(str(temp_file), str(invalid_file))
                        self.stats['invalid_files'] += 1
                        return False

                    self._log(f"✅ OK: {duration:.1f} сек, {size / 1024 / 1024:.1f} MB")
                else:
                    self._log(f"✅ Сохранён: {temp_file.name}")

                # Переименование в финальный файл
                shutil.move(str(temp_file), str(output_file))

                # Статистика
                self.stats['successful_segments'] += 1
                self.stats['last_success'] = datetime.now().isoformat()

                return True
            else:
                # Ошибка ffmpeg
                stderr = ""
                try:
                    stderr = self.process.stderr.read().decode('utf-8', errors='ignore')
                except:
                    pass

                self._log(f"❌ ffmpeg error (code={self.process.returncode}): {stderr[:200]}", "ERROR")
                self.stats['errors'].append({
                    'segment': segment_index,
                    'error': stderr[:200],
                    'time': datetime.now().isoformat()
                })

                # Очистка временного файла
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except:
                        pass

                return False

        except FileNotFoundError:
            self._log(f"❌ FFmpeg не найден: {self.ffmpeg_path}", "ERROR")
            return False
        except Exception as e:
            self._log(f"❌ Исключение: {e}", "ERROR")
            self.stats['errors'].append({
                'segment': segment_index,
                'error': str(e),
                'time': datetime.now().isoformat()
            })
            return False

    def _kill_process(self):
        """Безопасное завершение процесса ffmpeg"""
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
            except:
                pass

    def start_continuous_recording(self, max_segments=None):
        """
        Непрерывная запись сегментов с механизмами надёжности

        Args:
            max_segments: Максимальное количество сегментов (None = бесконечно)
        """
        if self.is_recording:
            self._log("⚠ Запись уже запущена", "WARNING")
            return

        self.is_recording = True
        self.stop_event.clear()

        self._log(f"📡 Подключение к RTSP: {self.rtsp_url}")
        self._log(f"📁 Папка сохранения: {self.output_dir}")
        self._log(f"⏱ Длительность сегмента: {self.segment_duration} сек")
        self._log(f"🔢 Макс сегментов: {max_segments if max_segments else '∞'}")
        self._log("Нажмите Ctrl+C для остановки\n")

        segment_index = 0
        consecutive_failures = 0
        last_health_check = time.time()

        try:
            while not self.stop_event.is_set():
                if max_segments and segment_index >= max_segments:
                    self._log(f"\n✅ Достигнут лимит: {max_segments} сегментов")
                    break

                # Запись сегмента
                success = self._record_segment(segment_index)
                self.stats['total_segments'] += 1

                if success:
                    segment_index += 1
                    consecutive_failures = 0  # Сброс счётчика ошибок
                else:
                    consecutive_failures += 1
                    self.stats['failed_segments'] += 1

                    # Health check
                    if consecutive_failures >= self.max_retries:
                        self._log(
                            f"🔄 {consecutive_failures} ошибок подряд — переподключение...",
                            "WARNING"
                        )
                        self._reconnect()
                        consecutive_failures = 0

                    # Пауза перед повторной попыткой
                    if not self.stop_event.is_set():
                        pause = min(5 * consecutive_failures, 30)  # 5, 10, 15... сек
                        self._log(f"⏳ Пауза {pause} сек перед повтором...")
                        time.sleep(pause)

                # Периодический health check (каждые 10 минут)
                current_time = time.time()
                if current_time - last_health_check > 600:  # 10 минут
                    last_health_check = current_time
                    self._health_check()
                    self._flush_log()

        except KeyboardInterrupt:
            self._log("\n⏹ Остановка по команде пользователя")
        finally:
            self._flush_log()
            self._save_stats()
            self.stop()

    def _reconnect(self):
        """Переподключение к RTSP потоку"""
        self._log("🔄 Переподключение к RTSP...", "INFO")
        self.stats['reconnected'] += 1

        # Пауза для стабилизации
        time.sleep(3)

        # Проверка связи с камерой (попытка быстрого подключения)
        test_cmd = [
            str(self.ffmpeg_path),
            '-rtsp_transport', 'tcp',
            '-timeout', '5000000',  # 5 секунд
            '-i', self.rtsp_url,
            '-t', '1',
            '-f', 'null',
            '-'
        ]

        try:
            result = subprocess.run(
                test_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10
            )

            if result.returncode == 0:
                self._log("✅ Связь с камерой восстановлена")
            else:
                self._log("⚠ Камера не отвечает — продолжаем попытки", "WARNING")

        except subprocess.TimeoutExpired:
            self._log("❌ Таймаут проверки связи", "ERROR")
        except Exception as e:
            self._log(f"❌ Ошибка проверки: {e}", "ERROR")

    def _health_check(self):
        """Периодическая проверка здоровья рекордера"""
        self._log("\n" + "=" * 50)
        self._log("📊 HEALTH CHECK")
        self._log("=" * 50)
        self._log(f"   Всего сегментов: {self.stats['total_segments']}")
        self._log(f"   Успешных: {self.stats['successful_segments']}")
        self._log(f"   Ошибок: {self.stats['failed_segments']}")
        self._log(f"   Переподключений: {self.stats['reconnected']}")
        self._log(f"   Повреждённых файлов: {self.stats['invalid_files']}")

        if self.stats['last_success']:
            last = datetime.fromisoformat(self.stats['last_success'])
            ago = (datetime.now() - last).total_seconds()
            self._log(f"   Последний успех: {ago:.0f} сек назад")
        else:
            self._log(f"   Последний успех: никогда")

        # Проверка места на диске
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.output_dir)
            free_gb = free / (1024 ** 3)
            self._log(f"   Свободно на диске: {free_gb:.1f} GB")

            if free_gb < 1.0:
                self._log(f"⚠ МАЛО МЕСТА! Очистите диск!", "CRITICAL")

        except Exception as e:
            self._log(f"   Ошибка проверки диска: {e}")

        self._log("=" * 50 + "\n")

    def _save_stats(self):
        """Сохранение статистики в файл"""
        stats_file = self.output_dir / "recorder_stats.json"

        stats_data = {
            'updated_at': datetime.now().isoformat(),
            'stats': self.stats
        }

        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self._log(f"⚠ Ошибка сохранения статистики: {e}")

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
    parser.add_argument("--max-retries", "-r", type=int, default=3,
                        help="Максимум повторных попыток (по умолчанию: 3)")
    parser.add_argument("--timeout", "-T", type=float, default=None,
                        help="Timeout записи в секундах (по умолчанию: duration * 2)")
    parser.add_argument("--no-validate", action="store_true",
                        help="Отключить проверку целостности WAV файлов")
    parser.add_argument("--log", "-l", default=None,
                        help="Путь к файлу лога (по умолчанию: output_dir/recorder.log)")

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
        ffmpeg_path=args.ffmpeg,
        max_retries=args.max_retries,
        timeout_multiplier=args.timeout / args.duration if args.timeout else 2.0,
        validate_files=not args.no_validate,
        log_file=args.log
    )

    try:
        recorder.start_continuous_recording(max_segments=args.segments)
    except KeyboardInterrupt:
        recorder.stop()
