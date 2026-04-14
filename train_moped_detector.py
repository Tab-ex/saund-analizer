"""
Обучение ML-детектора двигателя мопеда
========================================
Создаёт модель RandomForest для классификации звуков

Алгоритм извлечения признаков вдохновлён проектом SkufUp:
https://github.com/rudnstudent/SkufUp

================================================================================
БЫСТРЫЙ СТАРТ
================================================================================

1. Подготовьте данные:
   moped_samples/     - Записи с двигателем (10+ файлов)
   noise_samples/     - Записи без двигателя (20+ файлов)

2. Обучите модель:
   python train_moped_detector.py --moped moped_samples/ --noise noise_samples/

3. Используйте:
   python moped_ml_detector.py -i rtsp_recordings

================================================================================
СТРУКТУРА ДАННЫХ
================================================================================

moped_samples/
  ├── moped_1.wav          # Запись двигателя (5-30 сек)
  ├── moped_2.wav
  ├── moped_3.wav
  └── ...

noise_samples/
  ├── silence_1.wav        # Тишина/фоновый шум
  ├── speech_1.wav         # Человеческая речь
  ├── music_1.wav          # Музыка
  ├── traffic_1.wav        # Уличный шум
  └── ...

Чем больше данных — тем лучше!

================================================================================
ГДЕ ВЗЯТЬ ДАННЫЕ
================================================================================

1. Записать с камеры:
   python rtsp_audio_recorder.py rtsp://камера/stream -d 30

2. Записать фоновые шумы:
   - Тишина на улице
   - Разговоры людей
   - Шум ветра
   - Птицы
   - Машины (не мопед)

3. Скачать датасеты:
   - ESC-50: https://github.com/karolpiczak/ESC-50
     (класс "Engine" = 100 записей двигателя)
   
   - FSD50K: https://zenodo.org/record/4060432
     (50K звуков, есть транспорт)

4. Записать с телефона:
   - Подойдите к мопеду
   - Запишите 10-20 разных звуков
   - Разные обороты, расстояние, условия

================================================================================
ПРИМЕРЫ КОМАНД
================================================================================

# Обучение с указанием папок:
python train_moped_detector.py --moped moped_samples/ --noise noise_samples/

# Сохранение модели в другое место:
python train_moped_detector.py --moped moped_samples/ --noise noise_samples/ --model my_model.pkl

# Тестирование модели:
python train_moped_detector.py --test --model my_model.pkl

# Автоматическая генерация синтетических данных:
python train_moped_detector.py --generate --moped moped_samples/

# Полная информация (отчёт):
python train_moped_detector.py --moped moped_samples/ --noise noise_samples/ --report

================================================================================
"""

import numpy as np
import soundfile as sf
from scipy.fft import fft
from pathlib import Path
import pickle
import json
import argparse
from datetime import datetime

# ML библиотеки
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("❌ sklearn не установлен!")
    print("   Установите: pip install scikit-learn")
    exit(1)


class MopedFeatureExtractor:
    """Извлечение признаков из аудио"""

    def __init__(self, sample_rate=44100, window_size=1.0):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_samples = int(window_size * sample_rate)

    def extract_features(self, audio_chunk):
        """Извлечение признаков из одного окна"""
        if len(audio_chunk) < self.window_samples:
            return None

        chunk = audio_chunk[:self.window_samples]
        sr = self.sample_rate

        # === Временные признаки ===
        rms = np.sqrt(np.mean(chunk ** 2))
        peak = np.max(np.abs(chunk))
        zero_crossings = np.sum(np.abs(np.diff(np.sign(chunk)))) / len(chunk)

        # === Частотные признаки ===
        spectrum = np.abs(fft(chunk))
        freqs = np.fft.fftfreq(len(chunk), 1/sr)

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

        # Спектральный центроид
        spectral_centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10)

        # Спектральный спред
        spectral_spread = np.sqrt(
            np.sum(((freqs - spectral_centroid) ** 2) * spectrum) / (np.sum(spectrum) + 1e-10)
        )

        # Соотношение высоких/низких частот
        low_energy = band_energies['bass'] + band_energies['low_mid']
        high_energy = band_energies['high_mid'] + band_energies['high']
        hl_ratio = high_energy / (low_energy + 1e-10)

        # Спектральный флэкс
        diff_spectrum = np.diff(spectrum)
        spectral_flux = np.mean(diff_spectrum ** 2)

        # === Сбор признаков ===
        features = [
            rms, peak, zero_crossings,
            band_energies['sub_bass'], band_energies['bass'],
            band_energies['low_mid'], band_energies['mid'],
            band_energies['high_mid'], band_energies['high'],
            band_energies['very_high'],
            spectral_centroid, spectral_spread, spectral_flux,
            hl_ratio,
            20 * np.log10(peak / rms) if rms > 1e-10 else 0
        ]

        return features

    def extract_from_file(self, file_path):
        """Извлечение признаков из всего файла"""
        try:
            audio, sr = sf.read(file_path)
            if len(audio.shape) > 1:
                audio = audio[:, 0]  # Моно

            # Ресемплинг если нужно
            if sr != self.sample_rate:
                from scipy.signal import resample
                audio = resample(audio, int(len(audio) * self.sample_rate / sr))

            features = []
            for i in range(0, len(audio) - self.window_samples, self.window_samples):
                chunk = audio[i:i + self.window_samples]
                feat = self.extract_features(chunk)
                if feat is not None:
                    features.append(feat)

            return features

        except Exception as e:
            print(f"   ⚠ Ошибка загрузки {Path(file_path).name}: {e}")
            return []


class MopedModelTrainer:
    """Обучение модели для детектора двигателя"""

    FEATURE_NAMES = [
        'rms', 'peak', 'zero_crossing_rate',
        'energy_sub_bass', 'energy_bass', 'energy_low_mid',
        'energy_mid', 'energy_high_mid', 'energy_high',
        'energy_very_high',
        'spectral_centroid', 'spectral_spread', 'spectral_flux',
        'hl_ratio', 'snr_estimate'
    ]

    def __init__(self):
        self.feature_extractor = MopedFeatureExtractor()
        self.model = None

    def collect_data(self, moped_dir, noise_dir):
        """
        Сбор данных из папок

        Args:
            moped_dir: папка с записями двигателя
            noise_dir: папка с записями без двигателя

        Returns:
            X, y (features, labels)
        """
        print("=" * 70)
        print("📊 СБОР ДАННЫХ")
        print("=" * 70)

        X = []  # Признаки
        y = []  # Метки (1 = двигатель, 0 = нет)

        # === 1. Данные с двигателем (класс 1) ===
        moped_path = Path(moped_dir)
        moped_files = list(moped_path.glob("*.wav")) + list(moped_path.glob("*.mp3"))

        print(f"\n🔊 Двигатель: {moped_dir}")
        print(f"   Найдено файлов: {len(moped_files)}")

        if len(moped_files) == 0:
            print("   ❌ Нет файлов!")
            return None, None

        for file_path in moped_files:
            features = self.feature_extractor.extract_from_file(file_path)
            X.extend(features)
            y.extend([1] * len(features))  # Метка: двигатель
            print(f"   ✅ {file_path.name}: {len(features)} окон")

        # === 2. Данные без двигателя (класс 0) ===
        noise_path = Path(noise_dir)
        noise_files = list(noise_path.glob("*.wav")) + list(noise_path.glob("*.mp3"))

        print(f"\n🔇 Шум: {noise_dir}")
        print(f"   Найдено файлов: {len(noise_files)}")

        if len(noise_files) == 0:
            print("   ❌ Нет файлов!")
            return None, None

        for file_path in noise_files:
            features = self.feature_extractor.extract_from_file(file_path)
            X.extend(features)
            y.extend([0] * len(features))  # Метка: нет двигателя
            print(f"   ✅ {file_path.name}: {len(features)} окон")

        # === 3. Итоговая статистика ===
        X = np.array(X)
        y = np.array(y)

        print(f"\n📈 ИТОГО:")
        print(f"   Примеров: {len(X)}")
        print(f"   Двигатель: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
        print(f"   Без двигателя: {len(y) - sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")
        print(f"   Признаков: {X.shape[1]}")

        return X, y

    def train(self, X, y, model_path="moped_model.pkl", n_estimators=100):
        """
        Обучение модели RandomForest

        Args:
            X: признаки
            y: метки
            model_path: путь для сохранения модели
            n_estimators: количество деревьев (больше = точнее, но медленнее)
        """
        print("\n" + "=" * 70)
        print("🎓 ОБУЧЕНИЕ МОДЕЛИ")
        print("=" * 70)

        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"\n📋 Данные:")
        print(f"   Train: {len(X_train)} примеров")
        print(f"   Test: {len(X_test)} примеров")

        # Создание модели
        print(f"\n⚙  Параметры:")
        print(f"   Алгоритм: RandomForest")
        print(f"   Деревьев: {n_estimators}")
        print(f"   Признаков: {X.shape[1]}")

        # Обучение
        print("\n🔄 Обучение...")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1  # Все ядра
        )

        self.model.fit(X_train, y_train)

        # Оценка на test set
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\n📊 РЕЗУЛЬТАТЫ НА TEST SET:")
        print(f"   Точность: {accuracy:.4f} ({accuracy*100:.1f}%)")

        # Классификационный отчёт
        print(f"\n📋 ОТЧЁТ:")
        target_names = ['нет двигателя', 'двигатель']
        print(classification_report(y_test, y_pred, target_names=target_names))

        # Матрица ошибок
        print(f"🔍 МАТРИЦА ОШИБОК:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"                Предсказано")
        print(f"                нет     да")
        print(f"   Факт нет   [{cm[0,0]:4d}  {cm[0,1]:4d}]")
        print(f"        да    [{cm[1,0]:4d}  {cm[1,1]:4d}]")

        # Кросс-валидация
        print(f"\n🔄 Кросс-валидация (5 fold):")
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        print(f"   Среднее: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        # Сохранение модели
        model_data = {
            'model': self.model,
            'feature_names': self.FEATURE_NAMES,
            'metadata': {
                'trained_at': datetime.now().isoformat(),
                'n_samples': len(X),
                'n_features': X.shape[1],
                'accuracy': accuracy,
                'cv_accuracy': cv_scores.mean(),
                'n_estimators': n_estimators
            }
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\n💾 Модель сохранена: {model_path}")

        # Важность признаков
        print(f"\n📊 ВАЖНОСТЬ ПРИЗНАКОВ:")
        importances = self.model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]

        for i, idx in enumerate(sorted_idx):
            bar = "█" * int(importances[idx] * 50)
            print(f"   {self.FEATURE_NAMES[idx]:25s} {importances[idx]:.4f} {bar}")

        return accuracy

    def test_model(self, model_path="moped_model.pkl", test_dir=None):
        """Тестирование модели"""
        print("=" * 70)
        print("🧪 ТЕСТИРОВАНИЕ МОДЕЛИ")
        print("=" * 70)

        # Загрузка модели
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']

        print(f"\n✅ Модель загружена: {model_path}")
        print(f"   Точность при обучении: {model_data['metadata']['accuracy']:.4f}")

        if test_dir:
            print(f"\n📂 Тестирование на данных: {test_dir}")
            test_path = Path(test_dir)
            test_files = list(test_path.glob("*.wav")) + list(test_path.glob("*.mp3"))

            if test_files:
                print(f"   Найдено файлов: {len(test_files)}")
                # TODO: Добавить анализ тестовых файлов
            else:
                print(f"   ❌ Нет файлов в {test_dir}")


# ==================== CLI ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Обучение ML-детектора двигателя мопеда",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python train_moped_detector.py --moped moped_samples/ --noise noise_samples/
  python train_moped_detector.py --moped moped/ --noise noise/ --model my_model.pkl
  python train_moped_detector.py --test --model moped_model.pkl
        """
    )

    parser.add_argument("--moped", "-m", default=None,
                        help="Папка с записями двигателя")
    parser.add_argument("--noise", "-n", default=None,
                        help="Папка с записями без двигателя")
    parser.add_argument("--model", default="moped_model.pkl",
                        help="Путь к модели (по умолчанию: moped_model.pkl)")
    parser.add_argument("--trees", "-t", type=int, default=100,
                        help="Количество деревьев в RandomForest (по умолчанию: 100)")
    parser.add_argument("--test", action="store_true",
                        help="Тестирование модели")
    parser.add_argument("--report", action="store_true",
                        help="Полный отчёт")

    args = parser.parse_args()

    trainer = MopedModelTrainer()

    if args.test:
        # Тестирование
        trainer.test_model(args.model)
    elif args.moped and args.noise:
        # Обучение
        X, y = trainer.collect_data(args.moped, args.noise)

        if X is not None and len(X) > 0:
            trainer.train(X, y, model_path=args.model, n_estimators=args.trees)

            print("\n" + "=" * 70)
            print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
            print("=" * 70)
            print(f"\n📌 Используйте модель:")
            print(f"   python moped_ml_detector.py -i rtsp_recordings --model {args.model}")
        else:
            print("\n❌ Ошибка сбора данных!")
    else:
        print("❌ Укажите --moped и --noise папки!")
        print("\nПример:")
        print("  python train_moped_detector.py --moped moped_samples/ --noise noise_samples/")
