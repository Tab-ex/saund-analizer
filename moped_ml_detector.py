"""
ML-детектор двигателя мопеда на основе RandomForest
=====================================================
Классификатор на основе извлечения частотных признаков (как в проекте SkufUp)

Алгоритм:
  1. Извлечение признаков из аудио (энергия, спектр, центроид и др.)
  2. Классификация через RandomForest (обученная модель)
  3. Быстрое предсказание в реальном времени

================================================================================
КАК ЭТО РАБОТАЕТ
================================================================================

Вместо нейросети используем классический ML:

  1. Из аудио извлекаем числовые признаки:
     - Энергия в разных частотных полосах
     - Спектральный центроид (центр тяжести спектра)
     - Соотношение высоких/низких частот
     - RMS громкость
     - Пиковое значение
     - Спектральный спред (разброс)
     - Зеро-кроссинг (частота пересечения нуля)

  2. Обучаем RandomForest на этих признаках
     - Позитивные примеры: запись двигателя
     - Негативные примеры: тишина, шум, речь, музыка

  3. Модель предсказывает: двигатель / не двигатель

================================================================================
ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ
================================================================================

# 1. Обучение модели (нужны примеры):
python train_moped_detector.py --moped moped_samples/ --noise noise_samples/

# 2. Анализ файлов обученной моделью:
python moped_ml_detector.py -i rtsp_recordings

# 3. Анализ одного файла:
python moped_ml_detector.py -f audio_20260410_123456.wav

# 4. Проверка точности модели:
python moped_ml_detector.py --test

# 5. Визуализация признаков:
python moped_ml_detector.py --visualize

================================================================================
СТРУКТУРА ФАЙЛОВ
================================================================================

  train_moped_detector.py      - Скрипт для обучения модели
  moped_ml_detector.py         - Этот файл (использование модели)
  moped_model.pkl              - Обученная модель (создаётся после обучения)
  moped_features.json          - Информация о признаках (авто)

================================================================================
ТРЕБОВАНИЯ К ДАННЫМ ДЛЯ ОБУЧЕНИЯ
================================================================================

Минимум:
  - 10 файлов с двигателем (по 5-30 сек каждый)
  - 20 файлов без двигателя (разные звуки)

Идеально:
  - 50+ файлов с двигателем
  - 100+ файлов без двигателя

Где взять данные:
  - Записать с камеры (rtsp_audio_recorder.py)
  - Скачать с YouTube (видео с мопедами)
  - Датасет ESC-50 (содержит класс "Engine")
  - Записать фоновые шумы самостоятельно

================================================================================
"""

import numpy as np
import soundfile as sf
from scipy.fft import fft
from scipy.signal import spectral
from pathlib import Path
import pickle
import json
from datetime import datetime
import argparse
import os

# ML библиотеки
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠ sklearn не установлен. Установите: pip install scikit-learn")


class MopedFeatureExtractor:
    """
    Извлечение признаков из аудио для ML классификации

    Вдохновлено проектом SkufUp (github.com/rudnstudent/SkufUp)
    """

    def __init__(self, sample_rate=44100, window_size=1.0):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_samples = int(window_size * sample_rate)

    def extract_features(self, audio_chunk):
        """
        Извлечение признаков из одного окна аудио

        Возвращает dict с числовыми признаками:
        """
        if len(audio_chunk) < self.window_samples:
            return None

        chunk = audio_chunk[:self.window_samples]
        sr = self.sample_rate

        # === 1. Временные признаки ===
        rms = np.sqrt(np.mean(chunk ** 2))
        peak = np.max(np.abs(chunk))
        zero_crossings = np.sum(np.abs(np.diff(np.sign(chunk)))) / len(chunk)

        # === 2. Частотные признаки ===
        spectrum = np.abs(fft(chunk))
        freqs = np.fft.fftfreq(len(chunk), 1/sr)

        # Только положительные частоты
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        spectrum = spectrum[pos_mask]

        # Энергия в частотных полосах
        bands = {
            'sub_bass': (20, 60),
            'bass': (60, 250),
            'low_mid': (250, 500),
            'mid': (500, 2000),
            'high_mid': (2000, 4000),
            'high': (4000, 8000),
            'very_high': (8000, 20000)
        }

        total_energy = np.sum(spectrum ** 2) + 1e-10
        band_energies = {}
        for band_name, (low, high) in bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            band_energy = np.sum(spectrum[band_mask] ** 2)
            band_energies[band_name] = band_energy / total_energy

        # Спектральный центроид (центр тяжести спектра)
        spectral_centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10)

        # Спектральный спред (разброс)
        spectral_spread = np.sqrt(
            np.sum(((freqs - spectral_centroid) ** 2) * spectrum) / (np.sum(spectrum) + 1e-10)
        )

        # Соотношение высоких/низких частот
        low_energy = band_energies['bass'] + band_energies['low_mid']
        high_energy = band_energies['high_mid'] + band_energies['high']
        hl_ratio = high_energy / (low_energy + 1e-10)

        # Спектральный флэкс (неровность спектра)
        diff_spectrum = np.diff(spectrum)
        spectral_flux = np.mean(diff_spectrum ** 2)

        # === 3. Сбор признаков ===
        features = {
            # Временные
            'rms': rms,
            'peak': peak,
            'zero_crossing_rate': zero_crossings,

            # Частотные полосы
            'energy_sub_bass': band_energies['sub_bass'],
            'energy_bass': band_energies['bass'],
            'energy_low_mid': band_energies['low_mid'],
            'energy_mid': band_energies['mid'],
            'energy_high_mid': band_energies['high_mid'],
            'energy_high': band_energies['high'],
            'energy_very_high': band_energies['very_high'],

            # Спектральные
            'spectral_centroid': spectral_centroid,
            'spectral_spread': spectral_spread,
            'spectral_flux': spectral_flux,
            'hl_ratio': hl_ratio,

            # Отношение сигнал/шум (приближённое)
            'snr_estimate': 20 * np.log10(peak / rms) if rms > 1e-10 else 0,
        }

        return features

    def extract_all_features(self, audio):
        """
        Извлечение признаков из всего аудио (по окнам)

        Returns:
            features: list of dict
            times: list of float
        """
        features = []
        times = []

        for i in range(0, len(audio) - self.window_samples, self.window_samples):
            chunk = audio[i:i + self.window_samples]
            feat = self.extract_features(chunk)
            if feat is not None:
                features.append(feat)
                times.append(i / self.sample_rate)

        return features, times


class MopedMLDetector:
    """
    ML-детектор двигателя мопеда на основе RandomForest

    Алгоритм извлечения признаков вдохновлён проектом SkufUp:
    https://github.com/rudnstudent/SkufUp
    """

    FEATURE_NAMES = [
        'rms', 'peak', 'zero_crossing_rate',
        'energy_sub_bass', 'energy_bass', 'energy_low_mid',
        'energy_mid', 'energy_high_mid', 'energy_high',
        'energy_very_high',
        'spectral_centroid', 'spectral_spread', 'spectral_flux',
        'hl_ratio', 'snr_estimate'
    ]

    def __init__(self, model_path="moped_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.feature_extractor = MopedFeatureExtractor()
        self.is_loaded = False

    def load_model(self):
        """Загрузка обученной модели"""
        if not Path(self.model_path).exists():
            print(f"❌ Модель не найдена: {self.model_path}")
            print("   Сначала обучите модель: python train_moped_detector.py")
            return False

        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)

        # Модель может быть сохранена как dict или как прямой объект
        if isinstance(model_data, dict):
            self.model = model_data['model']
            print(f"✅ Модель загружена: {self.model_path}")
            print(f"   Точность: {model_data.get('metadata', {}).get('accuracy', 'N/A')}")
        else:
            self.model = model_data
            print(f"✅ Модель загружена: {self.model_path}")

        self.is_loaded = True

        return True

    def predict(self, features, threshold=0.9):
        """
        Предсказание для одного набора признаков
        
        Args:
            features: dict с признаками
            threshold: порог уверенности для класса "moped" (по умолчанию 0.5)
            
        Returns:
            (is_moped: bool, confidence: float)
        """
        if not self.is_loaded:
            print("❌ Модель не загружена!")
            return False, 0.0

        # 1. Вектор признаков
        feature_vector = [features[name] for name in self.FEATURE_NAMES]
        X = np.array([feature_vector])

        # 2. Вероятности классов [prob_class_0, prob_class_1]
        probabilities = self.model.predict_proba(X)[0]
        
        # 3. Безопасное получение вероятности класса "1" (moped)
        if hasattr(self.model, 'classes_'):
            classes = list(self.model.classes_)
            moped_idx = classes.index(1) if 1 in classes else 1  # fallback
        else:
            moped_idx = 1  # для большинства бинарных моделей
            
        confidence = probabilities[moped_idx]
        
        # 4. Применяем ВАШ порог
        is_moped = confidence >= threshold
        
        return is_moped, confidence

    def predict_chunk(self, audio_chunk):
        """
        Предсказание для сырого аудио (автоматическое извлечение признаков)

        Args:
            audio_chunk: numpy array (1 секунда аудио)

        Returns:
            (is_moped, confidence)
        """
        features = self.feature_extractor.extract_features(audio_chunk)
        if features is None:
            return False, 0.0
        return self.predict(features)

    def analyze_file(self, file_path):
        """
        Анализ аудиофайла

        Returns:
            dict с результатами
        """
        if not self.is_loaded:
            if not self.load_model():
                return None

        print(f"\n🔍 ML-анализ: {Path(file_path).name}")

        # Загрузка
        audio, sr = sf.read(file_path)
        if len(audio.shape) > 1:
            audio = audio[:, 0]  # Моно

        # Ресемплинг если нужно
        if sr != self.feature_extractor.sample_rate:
            from scipy.signal import resample
            audio = resample(audio, int(len(audio) * self.feature_extractor.sample_rate / sr))

        print(f"   📊 Длительность: {len(audio)/self.feature_extractor.sample_rate:.1f}с")
        print(f"   📈 Sample rate: {self.feature_extractor.sample_rate} Hz\n")

        # Извлечение признаков
        features, times = self.feature_extractor.extract_all_features(audio)

        # Предсказание для каждого окна
        results = []
        moped_count = 0

        print("📋 РЕЗУЛЬТАТЫ:")
        for feat, time_sec in zip(features, times):
            is_moped, confidence = self.predict(feat)

            if is_moped:
                marker = "🔊"
                moped_count += 1
            else:
                marker = "  "

            print(f"   {marker} {time_sec:6.1f}с | conf={confidence:.3f} | {'ДВИГАТЕЛЬ' if is_moped else 'нет'}")
            results.append({
                'time': round(time_sec, 2),
                'is_moped': is_moped,
                'confidence': round(confidence, 4)
            })

        # Статистика
        total = len(results)
        percent = moped_count / total * 100 if total > 0 else 0

        # Определяем факт обнаружения (если найдено хотя бы одно окно)
        detected = moped_count > 0
        
        if detected:
            print(f"\n✅ ОБНАРУЖЕНО: Двигатель мопеда найден!")
            print(f"   Количество сегментов: {moped_count}")
            print(f"   Примерная длительность: {moped_count * self.feature_extractor.window_size:.1f} сек")
            print(f"   Процент нахождения: {percent} %")
        else:
            print(f"\n❌ НЕ ОБНАРУЖЕНО: Звуков мопеда нет.")

        return {
            'file': str(file_path),
            'detected': detected,
            'moped_windows': moped_count,
            'total_windows': total,
            'results': results
        }

    def analyze_directory(self, input_dir, pattern="*.wav"):
        """Анализ всех файлов в папке"""
        if not self.is_loaded:
            if not self.load_model():
                return []

        input_path = Path(input_dir)
        audio_files = list(input_path.glob(pattern))

        if not audio_files:
            print(f"❌ В папке {input_path} нет файлов {pattern}")
            return []

        print(f"📁 Найдено файлов: {len(audio_files)}")
        print("=" * 70)

        all_results = []
        detected_files = []
        total_moped_time = 0

        for file_path in audio_files:
            result = self.analyze_file(file_path)
            if result:
                all_results.append(result)
                if result['detected']:
                    detected_files.append(result['file'])
                total_moped_time += result['moped_windows'] * self.feature_extractor.window_size

        print("\n" + "=" * 70)
        print(f"📊 ИТОГО:")
        print(f"   Файлов: {len(all_results)}")
        print(f"   Общее время работы двигателя: {total_moped_time:.1f} сек")
        
        if detected_files:
            print(f"\n✅ ФАЙЛЫ С ОБНАРУЖЕНИЕМ ({len(detected_files)}):")
            for f in detected_files:
                print(f"   - {Path(f).name}")
        else:
            print(f"\n❌ В файлах ничего не обнаружено.")

        return all_results


# ==================== CLI ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ML-детектор двигателя мопеда (RandomForest)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python moped_ml_detector.py -i rtsp_recordings
  python moped_ml_detector.py -f audio_20260410_123456.wav
  python moped_ml_detector.py --test
  python moped_ml_detector.py --visualize
        """
    )

    parser.add_argument("--input", "-i", default="rtsp_recordings",
                        help="Папка с аудиофайлами")
    parser.add_argument("--file", "-f", default=None,
                        help="Анализ одного файла")
    parser.add_argument("--model", "-m", default="moped_model.pkl",
                        help="Путь к модели (по умолчанию: moped_model.pkl)")
    parser.add_argument("--test", action="store_true",
                        help="Тестирование модели")
    parser.add_argument("--visualize", action="store_true",
                        help="Визуализация признаков")

    args = parser.parse_args()

    detector = MopedMLDetector(model_path=args.model)

    if args.test:
        # Тестирование модели
        if detector.load_model():
            print("\n🧪 Тестирование модели...")
            print("   (Нужны тестовые данные)")
            print("   Запустите: python train_moped_detector.py --test")
    elif args.visualize:
        # Визуализация
        print("\n📊 Визуализация признаков...")
        print("   (Функция в разработке)")
    elif args.file:
        # Один файл
        detector.analyze_file(args.file)
    else:
        # Папка
        detector.analyze_directory(args.input)
