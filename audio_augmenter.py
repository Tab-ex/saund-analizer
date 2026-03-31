"""
Инструменты аугментации аудио для Few-Shot Learning
Создаёт вариации аудиофайлов для расширения датасета
"""

import numpy as np
import soundfile as sf
import os
import json
from datetime import datetime
from pathlib import Path


class AudioAugmenter:
    """Набор инструментов для аугментации аудио"""
    
    def __init__(self, input_dir="data/raw", output_dir="data/augmented"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.augmentation_log = []
        
    # ==================== Базовые трансформации ====================
    
    def change_speed(self, audio, sample_rate, speed_factor):
        """Изменение скорости воспроизведения"""
        from scipy.signal import resample
        n_samples = int(len(audio) / speed_factor)
        resampled = resample(audio, n_samples)
        
        if speed_factor > 1:
            # Ускорение - обрезаем
            return resampled[:min(len(resampled), len(audio))]
        else:
            # Замедление - дополняем нулями
            result = np.zeros(len(audio))
            result[:len(resampled)] = resampled
            return result
    
    def pitch_shift(self, audio, sample_rate, semitones):
        """Сдвиг высоты тона (питч-шифт)"""
        # Простая реализация через изменение скорости с компенсацией
        factor = 2 ** (semitones / 12)
        from scipy.signal import resample
        n_samples = int(len(audio) / factor)
        resampled = resample(audio, n_samples)
        
        if len(resampled) < len(audio):
            result = np.zeros(len(audio))
            result[:len(resampled)] = resampled
            return result
        else:
            return resampled[:len(audio)]
    
    def add_noise(self, audio, noise_level=0.01):
        """Добавление белого шума"""
        noise = np.random.randn(len(audio)) * noise_level
        return audio + noise
    
    def change_volume(self, audio, volume_factor):
        """Изменение громкости"""
        return audio * volume_factor
    
    def apply_bandpass_filter(self, audio, sample_rate, low_freq=100, high_freq=5000):
        """Полосовой фильтр (оставляем только нужный диапазон частот)"""
        from scipy.signal import butter, filtfilt
        
        nyquist = sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Ограничиваем значения
        low = max(0.01, min(low, 0.99))
        high = max(low + 0.01, min(high, 0.99))
        
        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, audio)
    
    def time_stretch(self, audio, rate):
        """Растягивание/сжатие во времени без изменения питча"""
        from librosa import time_stretch as librosa_stretch
        return librosa_stretch(audio, rate=rate)
    
    # ==================== Комбинированные трансформации ====================
    
    def apply_random_augmentation(self, audio, sample_rate, augmentation_level='medium'):
        """
        Случайная комбинация трансформаций
        augmentation_level: 'light', 'medium', 'heavy'
        """
        import random
        
        result = audio.copy()
        
        # Параметры в зависимости от уровня
        if augmentation_level == 'light':
            ranges = {
                'speed': (0.95, 1.05),
                'pitch': (-2, 2),
                'volume': (0.8, 1.2),
                'noise': (0.001, 0.005),
            }
            n_transforms = random.randint(1, 2)
        elif augmentation_level == 'medium':
            ranges = {
                'speed': (0.9, 1.1),
                'pitch': (-3, 3),
                'volume': (0.7, 1.3),
                'noise': (0.005, 0.01),
            }
            n_transforms = random.randint(2, 3)
        else:  # heavy
            ranges = {
                'speed': (0.85, 1.15),
                'pitch': (-5, 5),
                'volume': (0.5, 1.5),
                'noise': (0.01, 0.02),
            }
            n_transforms = random.randint(3, 4)
        
        # Случайный выбор трансформаций
        transforms = random.sample(
            ['speed', 'pitch', 'volume', 'noise', 'filter'],
            min(n_transforms, 5)
        )
        
        applied = []
        
        for transform in transforms:
            if transform == 'speed':
                factor = random.uniform(*ranges['speed'])
                if factor != 1.0:
                    result = self.change_speed(result, sample_rate, factor)
                    applied.append(f'speed_{factor:.2f}')
                    
            elif transform == 'pitch':
                semitones = random.uniform(*ranges['pitch'])
                if semitones != 0:
                    result = self.pitch_shift(result, sample_rate, semitones)
                    applied.append(f'pitch_{semitones:+.1f}')
                    
            elif transform == 'volume':
                factor = random.uniform(*ranges['volume'])
                result = self.change_volume(result, factor)
                applied.append(f'volume_{factor:.2f}')
                
            elif transform == 'noise':
                level = random.uniform(*ranges['noise'])
                result = self.add_noise(result, level)
                applied.append(f'noise_{level:.3f}')
                
            elif transform == 'filter':
                low = random.randint(50, 200)
                high = random.randint(3000, 8000)
                result = self.apply_bandpass_filter(result, sample_rate, low, high)
                applied.append(f'filter_{low}-{high}Hz')
        
        # Нормализация
        max_val = np.max(np.abs(result))
        if max_val > 0:
            result = result / max_val * 0.9
        
        return result, applied
    
    # ==================== Основные методы ====================
    
    def augment_file(self, file_path, n_variants=5, augmentation_levels=None):
        """
        Создание нескольких вариантов одного файла
        
        Args:
            file_path: путь к исходному файлу
            n_variants: количество вариантов для создания
            augmentation_levels: список уровней ['light', 'medium', 'heavy']
        """
        if augmentation_levels is None:
            augmentation_levels = ['light', 'medium', 'heavy']
        
        # Загрузка аудио
        audio, sample_rate = sf.read(file_path)
        
        # Конвертация в моно
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        
        # Нормализация
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.9
        
        file_name = Path(file_path).stem
        results = []
        
        for i in range(n_variants):
            level = augmentation_levels[i % len(augmentation_levels)]
            
            # Применяем аугментацию
            augmented, applied_transforms = self.apply_random_augmentation(
                audio, sample_rate, level
            )
            
            # Сохранение
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"{file_name}_aug{i+1}_{level}_{timestamp}.wav"
            output_path = self.output_dir / output_name
            
            sf.write(output_path, augmented, sample_rate)
            
            # Логирование
            log_entry = {
                'source': str(file_path),
                'output': str(output_path),
                'level': level,
                'transforms': applied_transforms,
                'timestamp': timestamp
            }
            self.augmentation_log.append(log_entry)
            results.append(log_entry)
            
            print(f"✅ Создан: {output_name}")
            print(f"   Трансформации: {', '.join(applied_transforms)}")
        
        return results
    
    def augment_directory(self, n_variants_per_file=5):
        """Аугментация всех файлов в input_dir"""
        audio_files = list(self.input_dir.glob("*.wav")) + \
                      list(self.input_dir.glob("*.mp3"))
        
        if not audio_files:
            print(f"❌ В папке {self.input_dir} нет аудиофайлов")
            return []
        
        print(f"📁 Найдено файлов: {len(audio_files)}")
        print(f"📈 Вариантов на файл: {n_variants_per_file}")
        print("-" * 50)
        
        all_results = []
        
        for file_path in audio_files:
            print(f"\n🎵 Обработка: {file_path.name}")
            results = self.augment_file(file_path, n_variants_per_file)
            all_results.extend(results)
        
        # Сохранение лога
        self.save_log()
        
        print("\n" + "=" * 50)
        print(f"✅ Всего создано файлов: {len(all_results)}")
        print(f"📄 Лог сохранён: {self.output_dir / 'augmentation_log.json'}")
        
        return all_results
    
    def save_log(self):
        """Сохранение лога аугментации"""
        log_path = self.output_dir / 'augmentation_log.json'
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump({
                'created': datetime.now().isoformat(),
                'total_files': len(self.augmentation_log),
                'entries': self.augmentation_log
            }, f, ensure_ascii=False, indent=2)
    
    def load_log(self):
        """Загрузка лога аугментации"""
        log_path = self.output_dir / 'augmentation_log.json'
        
        if not log_path.exists():
            return None
        
        with open(log_path, 'r', encoding='utf-8') as f:
            return json.load(f)


# ==================== CLI интерфейс ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Аугментация аудио для Few-Shot Learning")
    parser.add_argument("--input", "-i", default="data/raw", help="Папка с исходниками")
    parser.add_argument("--output", "-o", default="data/augmented", help="Папка для результатов")
    parser.add_argument("--variants", "-n", type=int, default=5, help="Вариантов на файл")
    
    args = parser.parse_args()
    
    augmenter = AudioAugmenter(args.input, args.output)
    augmenter.augment_directory(args.variants)
