"""
Быстрый детектор двигателя мопеда
===================================
Анализирует характерные частоты напрямую через FFT — без YAMNet и предобработки

Работает в 50-100 раз быстрее нейросети, прозрачная логика, настройка порогов.

================================================================================
КАК ЭТО РАБОТАЕТ
================================================================================

Двигатель мопеда имеет характерный спектр:
  - Основная частота: 50-200 Hz (обороты двигателя)
  - Гармоники: 200-2000 Hz (жужжание, выхлоп)
  - Широкополосный шум с пиками на кратных частотах

Алгоритм:
  1. Разбиваем аудио на окна (по умолчанию 1 секунда)
  2. Для каждого окна: FFT → спектр частот
  3. Считаем энергию в характерных диапазонах
  4. Если энергия выше порога → двигатель обнаружен!

================================================================================
ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ
================================================================================

# Анализ всех файлов в папке rtsp_recordings:
python moped_frequency_detector.py

# Анализ конкретной папки:
python moped_frequency_detector.py -i my_recordings

# Анализ одного файла:
python moped_frequency_detector.py -f audio_20260410_123456.wav

# Настройка чувствительности (0.1-1.0, по умолчанию 0.3):
python moped_frequency_detector.py --threshold 0.25

# Экспорт результатов в JSON:
python moped_frequency_detector.py --export results.json

================================================================================
НАСТРОЙКА ЧАСТОТ ДЛЯ ДРУГИХ ТИПОВ ДВИГАТЕЛЕЙ
================================================================================

Измените в коде (класс MopedDetector.__init__):

  # Мопед/мотоцикл (по умолчанию)
  self.fundamental_freq = (50, 200)
  self.harmonic_freq = (200, 2000)

  # Легковой автомобиль
  self.fundamental_freq = (30, 150)
  self.harmonic_freq = (150, 1500)

  # Грузовик/автобус
  self.fundamental_freq = (20, 100)
  self.harmonic_freq = (100, 1000)

  # Квадрокоптер/дрон
  self.fundamental_freq = (100, 400)
  self.harmonic_freq = (400, 4000)

================================================================================
"""

import numpy as np
import soundfile as sf
from scipy.fft import fft
from pathlib import Path
from datetime import datetime
import json
import argparse


class MopedDetector:
    """
    Детектор двигателя по характерным частотам
    
    Анализирует энергию в частотных окнах, характерных для двигателя мопеда.
    Работает напрямую с FFT — без нейросетей и предобработки.
    """

    def __init__(self, fundamental_freq=(50, 200), harmonic_freq=(200, 2000),
                 threshold=0.3):
        """
        Args:
            fundamental_freq: Диапазон основной частоты (обороты двигателя)
            harmonic_freq: Диапазон гармоник (жужжание, выхлоп)
            threshold: Порог обнаружения (0.1-1.0, чем меньше — тем чувствительнее)
        """
        self.fundamental_freq = fundamental_freq
        self.harmonic_freq = harmonic_freq
        self.threshold = threshold

    def detect(self, audio, sample_rate=44100, window_size=1.0):
        """
        Анализ аудио по окнам

        Args:
            audio: аудиоданные (numpy array)
            sample_rate: частота дискретизации
            window_size: размер окна в секундах

        Returns:
            list of dict с результатами для каждого окна
        """
        results = []
        window_samples = int(window_size * sample_rate)

        for i in range(0, len(audio) - window_samples, window_samples):
            chunk = audio[i:i + window_samples]
            time_sec = i / sample_rate

            # FFT
            spectrum = np.abs(fft(chunk))
            freqs = np.fft.fftfreq(len(chunk), 1/sample_rate)

            # Только положительные частоты
            pos_mask = freqs > 0
            freqs = freqs[pos_mask]
            spectrum = spectrum[pos_mask]

            # Энергия в диапазонах
            fund_mask = (freqs >= self.fundamental_freq[0]) & (freqs <= self.fundamental_freq[1])
            harm_mask = (freqs >= self.harmonic_freq[0]) & (freqs <= self.harmonic_freq[1])

            fund_energy = np.sum(spectrum[fund_mask] ** 2)
            harm_energy = np.sum(spectrum[harm_mask] ** 2)
            total_energy = np.sum(spectrum ** 2)

            # Нормализация
            if total_energy > 0:
                fund_ratio = fund_energy / total_energy
                harm_ratio = harm_energy / total_energy
            else:
                fund_ratio = 0
                harm_ratio = 0

            # Решение: двигатель?
            is_moped = (fund_ratio > self.threshold and
                       harm_ratio > self.threshold)

            confidence = (fund_ratio + harm_ratio) / 2

            results.append({
                'time': round(time_sec, 2),
                'fund_ratio': round(fund_ratio, 4),
                'harm_ratio': round(harm_ratio, 4),
                'confidence': round(confidence, 4),
                'is_moped': is_moped
            })

        return results

    def analyze_file(self, file_path, window_size=1.0):
        """
        Анализ одного файла

        Args:
            file_path: путь к аудиофайлу
            window_size: размер окна в секундах

        Returns:
            dict с результатами анализа
        """
        print(f"\n🔍 Анализ: {Path(file_path).name}")

        # Загрузка
        audio, sr = sf.read(file_path)

        # Конвертация в моно
        if len(audio.shape) > 1:
            audio = audio[:, 0]

        print(f"   📊 Длительность: {len(audio)/sr:.1f}с | Sample rate: {sr} Hz")
        print(f"   🎯 Частоты: основа={self.fundamental_freq} Hz, гармоники={self.harmonic_freq} Hz")
        print(f"   ⚡ Порог: {self.threshold}\n")

        # Анализ
        results = self.detect(audio, sr, window_size)

        # Статистика
        moped_count = sum(1 for r in results if r['is_moped'])
        total_count = len(results)
        percent = moped_count / total_count * 100 if total_count > 0 else 0

        print(f"📈 РЕЗУЛЬТАТ:")
        print(f"   Обнаружен двигатель: {moped_count}/{total_count} окон ({percent:.1f}%)")
        print(f"   Длительность работы двигателя: {moped_count * window_size:.1f} сек\n")

        # Вывод по окнам
        print("📋 ДЕТАЛИЗАЦИЯ:")
        for r in results:
            if r['is_moped']:
                marker = "🔊"
            else:
                marker = "  "
            print(f"   {marker} {r['time']:6.1f}с | conf={r['confidence']:.3f} "
                  f"(fund={r['fund_ratio']:.3f}, harm={r['harm_ratio']:.3f})")

        return {
            'file': str(file_path),
            'total_windows': total_count,
            'moped_windows': moped_count,
            'percent': percent,
            'results': results
        }

    def analyze_directory(self, input_dir, pattern="*.wav", window_size=1.0, export=None):
        """
        Анализ всех файлов в папке

        Args:
            input_dir: папка с аудиофайлами
            pattern: шаблон файлов
            window_size: размер окна в секундах
            export: путь для экспорта результатов в JSON
        """
        input_path = Path(input_dir)
        audio_files = list(input_path.glob(pattern))

        if not audio_files:
            print(f"❌ В папке {input_path} нет файлов {pattern}")
            return []

        print(f"📁 Найдено файлов: {len(audio_files)}")
        print(f"⚙  Настройки: порог={self.threshold}, окно={window_size}с")
        print("=" * 70)

        all_results = []

        for file_path in audio_files:
            result = self.analyze_file(file_path, window_size)
            all_results.append(result)

        # Итоговая статистика
        total_files = len(all_results)
        files_with_moped = sum(1 for r in all_results if r['moped_windows'] > 0)
        total_moped_time = sum(r['moped_windows'] for r in all_results) * window_size

        print("\n" + "=" * 70)
        print(f"📊 ИТОГО:")
        print(f"   Файлов: {total_files}")
        print(f"   С двигателем: {files_with_moped}")
        print(f"   Общее время работы двигателя: {total_moped_time:.1f} сек")

        # Экспорт
        if export:
            export_data = {
                'created': datetime.now().isoformat(),
                'settings': {
                    'threshold': self.threshold,
                    'fundamental_freq': self.fundamental_freq,
                    'harmonic_freq': self.harmonic_freq,
                    'window_size': window_size
                },
                'summary': {
                    'total_files': total_files,
                    'files_with_moped': files_with_moped,
                    'total_moped_time_sec': total_moped_time
                },
                'files': all_results
            }

            with open(export, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            print(f"💾 Экспорт: {export}")

        return all_results


# ==================== CLI ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Быстрый детектор двигателя мопеда по характерным частотам",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python moped_frequency_detector.py
  python moped_frequency_detector.py -i my_recordings --threshold 0.25
  python moped_frequency_detector.py -f audio_20260410_123456.wav
  python moped_frequency_detector.py --export results.json
        """
    )

    parser.add_argument("--input", "-i", default="rtsp_recordings",
                        help="Папка с аудиофайлами (по умолчанию: rtsp_recordings)")
    parser.add_argument("--file", "-f", default=None,
                        help="Анализ одного файла")
    parser.add_argument("--threshold", "-t", type=float, default=0.3,
                        help="Порог обнаружения 0.1-1.0 (по умолчанию: 0.3)")
    parser.add_argument("--window", "-w", type=float, default=1.0,
                        help="Размер окна в секундах (по умолчанию: 1.0)")
    parser.add_argument("--pattern", default="*.wav",
                        help="Шаблон файлов (по умолчанию: *.wav)")
    parser.add_argument("--export", "-e", default=None,
                        help="Экспорт результатов в JSON")

    args = parser.parse_args()

    # Создание детектора
    detector = MopedDetector(threshold=args.threshold)

    # Анализ одного файла или папки
    if args.file:
        detector.analyze_file(args.file, window_size=args.window)
    else:
        detector.analyze_directory(
            args.input,
            pattern=args.pattern,
            window_size=args.window,
            export=args.export
        )
