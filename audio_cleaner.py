"""
Инструмент очистки аудио от шумов и нормализации сигнала
Поддерживает пакетную обработку файлов

Основано на методах ЦОС:
- БИХ/КИХ фильтры (Butterworth)
- Спектральное вычитание
- Фильтр Винера
- Компенсация затухания в воздухе
- Нормализация LUFS-подобная
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from scipy.signal import butter, filtfilt, lfilter, firwin
from scipy.fft import fft, fftfreq
import json
from datetime import datetime


class AudioCleaner:
    """Очистка аудио от шумов и нормализация"""

    def __init__(self, input_dir="rtsp_recordings", output_dir="cleaned_audio"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processing_log = []

    # ==================== Фильтрация ====================

    def highpass_filter(self, audio, sample_rate, cutoff=80, order=4):
        """БИХ фильтр высоких частот (убирает низкочастотный гул)"""
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff / nyquist
        b, a = butter(order, normalized_cutoff, btype='high', analog=False)
        return filtfilt(b, a, audio)

    def lowpass_filter(self, audio, sample_rate, cutoff=8000, order=4):
        """БИХ фильтр низких частот (убирает высокочастотный шум)"""
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff / nyquist
        b, a = butter(order, normalized_cutoff, btype='low', analog=False)
        return filtfilt(b, a, audio)

    def bandpass_filter(self, audio, sample_rate, low_cutoff=80, high_cutoff=5000, order=4):
        """БИХ полосовой фильтр (оставляем только полезный диапазон)"""
        nyquist = sample_rate / 2
        low = low_cutoff / nyquist
        high = high_cutoff / nyquist
        b, a = butter(order, [low, high], btype='band', analog=False)
        return filtfilt(b, a, audio)

    def notch_filter(self, audio, sample_rate, freq=50, Q=30):
        """Режекторный фильтр (убирает конкретную частоту, например 50Hz)"""
        nyquist = sample_rate / 2
        w0 = freq / nyquist
        b, a = butter(Q, w0, btype='bandstop', analog=False)
        return filtfilt(b, a, audio)

    def fir_bandpass(self, audio, sample_rate, low_cutoff=80, high_cutoff=5000, numtaps=101):
        """КИХ полосовой фильтр (линейная фаза, без искажений)"""
        nyquist = sample_rate / 2
        taps = firwin(numtaps, [low_cutoff, high_cutoff], pass_zero=False, fs=sample_rate)
        # firwin возвращает чётное количество, нужно нечётное для filtfilt
        if len(taps) % 2 == 0:
            taps = taps[:-1]
        return filtfilt(taps, [1.0], audio)

    def compensate_air_absorption(self, audio, sample_rate, distance_meters=10):
        """
        Компенсация затухания высоких частот в воздухе

        Воздух поглощает ВЧ: ~0.001*f² дБ/м при 20°C, 50% влажности
        Чем дальше источник — тем больше ВЧ потеряно

        Args:
            audio: аудиоданные
            sample_rate: частота дискретизации
            distance_meters: расстояние до источника (м)
        """
        # Коэффициент затухания (упрощённая модель)
        # Реальные значения зависят от температуры и влажности
        freqs = fftfreq(len(audio), 1/sample_rate)
        positive_mask = freqs > 0

        # Затухание в дБ/м для каждой частоты (приближённая модель)
        # f в кГц, затухание в дБ/м
        f_khz = np.abs(freqs) / 1000
        attenuation_db_per_m = 0.0001 * f_khz**2 + 0.001 * f_khz + 0.005

        # Компенсация
        gain_db = attenuation_db_per_m * distance_meters
        gain_linear = 10 ** (gain_db / 20)

        # Ограничиваем усиление (не более 20 дБ)
        gain_linear = np.clip(gain_linear, 1, 10)

        # Применяем в частотной области
        spectrum = fft(audio)
        spectrum[positive_mask] *= gain_linear[positive_mask]
        spectrum[~positive_mask] *= gain_linear[~positive_mask]

        return np.real(np.fft.ifft(spectrum))

    # ==================== Шумоподавление ====================

    def spectral_subtraction(self, audio, sample_rate, noise_duration=0.5):
        """
        Подавление шума методом спектрального вычитания

        Args:
            audio: аудиоданные
            sample_rate: частота дискретизации
            noise_duration: длительность шума в начале (сек)
        """
        # Параметры STFT
        n_fft = 2048
        hop_length = n_fft // 4

        # Оценка шума из первого сегмента
        noise_samples = int(noise_duration * sample_rate)
        noise_segment = audio[:min(noise_samples, len(audio))]

        # STFT
        from scipy.signal import stft, istft
        f, t, Zxx = stft(audio, fs=sample_rate, nperseg=n_fft, noverlap=hop_length)
        _, _, Zxx_noise = stft(noise_segment, fs=sample_rate, nperseg=n_fft, noverlap=hop_length)

        # Оценка мощности шума
        noise_power = np.mean(np.abs(Zxx_noise) ** 2, axis=1, keepdims=True)

        # Спектральное вычитание
        signal_power = np.abs(Zxx) ** 2
        clean_power = np.maximum(signal_power - 2 * noise_power, 0)
        clean_mag = np.sqrt(clean_power)

        # Восстановление фазы
        phase = np.angle(Zxx)
        Zxx_clean = clean_mag * np.exp(1j * phase)

        # Обратное STFT
        _, clean_audio = istft(Zxx_clean, fs=sample_rate, nperseg=n_fft, noverlap=hop_length)

        return clean_audio[:len(audio)]

    def wiener_filter(self, audio, sample_rate, noise_duration=0.5):
        """
        Фильтр Винера — оптимальное шумоподавление (минимизация СКО)

        В отличие от спектрального вычитания, даёт меньше артефактов
        """
        n_fft = 2048
        hop_length = n_fft // 4

        from scipy.signal import stft, istft

        noise_samples = int(noise_duration * sample_rate)
        noise_segment = audio[:min(noise_samples, len(audio))]

        f, t, Zxx = stft(audio, fs=sample_rate, nperseg=n_fft, noverlap=hop_length)
        _, _, Zxx_noise = stft(noise_segment, fs=sample_rate, nperseg=n_fft, noverlap=hop_length)

        # Оценка мощности сигнала и шума
        signal_power = np.abs(Zxx) ** 2
        noise_power = np.mean(np.abs(Zxx_noise) ** 2, axis=1, keepdims=True)

        # Фильтр Винера: H = S / (S + N)
        wiener_gain = signal_power / (signal_power + noise_power + 1e-10)

        # Применяем
        Zxx_clean = Zxx * wiener_gain

        _, clean_audio = istft(Zxx_clean, fs=sample_rate, nperseg=n_fft, noverlap=hop_length)

        return clean_audio[:len(audio)]

    def simple_noise_gate(self, audio, threshold=0.02):
        """
        Простой шумовой гейт (обнуляет тихие участки)

        Args:
            audio: аудиоданные
            threshold: порог (0-1)
        """
        result = audio.copy()
        result[np.abs(result) < threshold] = 0
        return result

    def adaptive_noise_gate(self, audio, sample_rate, window_size=0.1, multiplier=2.0):
        """
        Адаптивный шумовой гейт

        Args:
            audio: аудиоданные
            sample_rate: частота дискретизации
            window_size: размер окна в секундах
            multiplier: множитель для порога
        """
        window_samples = int(window_size * sample_rate)
        result = audio.copy()

        for i in range(0, len(audio), window_samples):
            chunk = audio[i:i + window_samples]
            noise_level = np.std(chunk)
            threshold = noise_level * multiplier
            result[i:i + window_samples][np.abs(chunk) < threshold] = 0

        return result

    # ==================== Нормализация ====================

    def normalize_peak(self, audio, target_peak=0.9):
        """Нормализация по пиковому значению"""
        peak = np.max(np.abs(audio))
        if peak > 0:
            return audio * (target_peak / peak)
        return audio

    def normalize_rms(self, audio, target_rms=0.1):
        """Нормализация по среднеквадратичному значению (громкости)"""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            return audio * (target_rms / rms)
        return audio

    def normalize_loudness(self, audio, target_loudness=-20):
        """
        Нормализация по громкости (dB) — упрощённый аналог LUFS

        Args:
            audio: аудиоданные
            target_loudness: целевая громкость в dBFS
        """
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            current_loudness = 20 * np.log10(rms)
            gain_db = target_loudness - current_loudness
            gain_linear = 10 ** (gain_db / 20)
            return audio * gain_linear
        return audio

    def normalize_lufs_like(self, audio, sample_rate, target_lufs=-19):
        """
        LUFS-подобная нормализация (ближе к стандарту EBU R128)

        Учитывает восприятие громкости человеком (K-взвешивание)

        Args:
            audio: аудиоданные
            sample_rate: частота дискретизации
            target_lufs: целевая громкость в LUFS (моно: -19, стерео: -16)
        """
        # K-взвешивание (упрощённое)
        # Фильтр, имитирующий чувствительность слуха
        from scipy.signal import butter, filtfilt

        # High-shelf фильтр для имитации восприятия
        b, a = butter(2, 0.001, btype='high')  # Очень низкий cutoff
        weighted = filtfilt(b, a, audio)

        # RMS взвешенного сигнала
        rms_weighted = np.sqrt(np.mean(weighted ** 2))

        if rms_weighted > 0:
            current_lufs = 20 * np.log10(rms_weighted)
            gain_db = target_lufs - current_lufs
            gain_linear = 10 ** (gain_db / 20)
            return audio * gain_linear

        return audio

    # ==================== Комбинированная обработка ====================

    def clean_audio(self, audio, sample_rate, preset='voice', distance_meters=None):
        """
        Полная очистка аудио по пресету

        Presets:
            'voice' - для голоса (фильтр 80-5000Hz + Винер + LUFS)
            'engine' - для двигателя (фильтр 20-8000Hz + режекторный + нормализация)
            'music' - для музыки (мягкий фильтр + LUFS)
            'outdoor' - для записи на улице (компенсация воздуха + Винер)
            'custom' - минимальная обработка
            'aggressive' - максимальная очистка (Винер + гейт + фильтр)

        Args:
            audio: аудиоданные
            sample_rate: частота дискретизации
            preset: пресет обработки
            distance_meters: расстояние до источника (для компенсации воздуха)
        """
        result = audio.copy()
        applied = []

        if preset == 'voice':
            # Для голоса — чистый диапазон + Винер
            result = self.bandpass_filter(result, sample_rate, 80, 5000)
            applied.append('bandpass_80-5000Hz')

            result = self.wiener_filter(result, sample_rate)
            applied.append('wiener_filter')

            result = self.normalize_lufs_like(result, sample_rate, -19)
            applied.append('normalize_lufs_-19')

        elif preset == 'engine':
            # Для двигателя — широкий диапазон + компенсация 50Hz
            result = self.bandpass_filter(result, sample_rate, 20, 8000)
            applied.append('bandpass_20-8000Hz')

            result = self.notch_filter(result, sample_rate, 50)
            applied.append('notch_50Hz')

            result = self.notch_filter(result, sample_rate, 100)
            applied.append('notch_100Hz')

            result = self.normalize_peak(result, 0.9)
            applied.append('normalize_peak_0.9')

        elif preset == 'music':
            # Для музыки — мягкая обработка
            result = self.bandpass_filter(result, sample_rate, 20, 18000)
            applied.append('bandpass_20-18000Hz')

            result = self.wiener_filter(result, sample_rate, noise_duration=1.0)
            applied.append('wiener_filter')

            result = self.normalize_lufs_like(result, sample_rate, -16)
            applied.append('normalize_lufs_-16')

        elif preset == 'outdoor':
            # Для уличной записи — компенсация воздуха + Винер
            if distance_meters:
                result = self.compensate_air_absorption(result, sample_rate, distance_meters)
                applied.append(f'air_compensation_{distance_meters}m')

            result = self.bandpass_filter(result, sample_rate, 50, 10000)
            applied.append('bandpass_50-10000Hz')

            result = self.wiener_filter(result, sample_rate)
            applied.append('wiener_filter')

            result = self.normalize_lufs_like(result, sample_rate, -19)
            applied.append('normalize_lufs_-19')

        elif preset == 'aggressive':
            # Максимальная очистка
            result = self.bandpass_filter(result, sample_rate, 80, 5000)
            applied.append('bandpass_80-5000Hz')

            result = self.wiener_filter(result, sample_rate)
            applied.append('wiener_filter')

            result = self.adaptive_noise_gate(result, sample_rate, multiplier=1.5)
            applied.append('adaptive_gate_1.5')

            result = self.normalize_lufs_like(result, sample_rate, -19)
            applied.append('normalize_lufs_-19')

        elif preset == 'custom':
            # Только нормализация
            result = self.normalize_peak(result, 0.9)
            applied.append('normalize_peak_0.9')

        else:
            raise ValueError(f"Неизвестный пресет: {preset}")

        # Клиппинг защита
        max_val = np.max(np.abs(result))
        if max_val > 1.0:
            result = result / max_val * 0.95
            applied.append('clipping_protection')

        return result, applied

    # ==================== Анализ ====================

    def analyze_noise(self, audio, sample_rate):
        """Анализ уровня шума"""
        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.max(np.abs(audio))
        snr_estimate = 20 * np.log10(peak / rms) if rms > 0 else 0

        return {
            'rms': float(rms),
            'peak': float(peak),
            'snr_estimate_db': float(snr_estimate),
            'dynamic_range_db': float(20 * np.log10(peak / rms)) if rms > 0 else 0
        }

    # ==================== Обработка файлов ====================

    def process_file(self, file_path, preset='voice', distance_meters=None):
        """Обработка одного файла"""
        print(f"\n🎵 Обработка: {Path(file_path).name}")

        # Загрузка
        audio, sample_rate = sf.read(file_path)

        # Конвертация в моно
        if len(audio.shape) > 1:
            audio = audio[:, 0]

        # Анализ до
        analysis_before = self.analyze_noise(audio, sample_rate)
        print(f"   До: RMS={analysis_before['rms']:.4f}, SNR={analysis_before['snr_estimate_db']:.1f}dB")

        # Очистка
        cleaned, applied = self.clean_audio(audio, sample_rate, preset, distance_meters)

        # Анализ после
        analysis_after = self.analyze_noise(cleaned, sample_rate)
        print(f"   После: RMS={analysis_after['rms']:.4f}, SNR={analysis_after['snr_estimate_db']:.1f}dB")

        # Сохранение
        output_name = Path(file_path).stem + "_cleaned.wav"
        output_path = self.output_dir / output_name
        sf.write(output_path, cleaned, sample_rate)

        # Лог
        log_entry = {
            'source': str(file_path),
            'output': str(output_path),
            'preset': preset,
            'distance_meters': distance_meters,
            'transforms': applied,
            'analysis_before': analysis_before,
            'analysis_after': analysis_after,
            'timestamp': datetime.now().isoformat()
        }
        self.processing_log.append(log_entry)

        print(f"   ✅ Сохранён: {output_name}")
        print(f"   Применено: {', '.join(applied)}")

        return log_entry

    def process_directory(self, preset='voice', pattern="*.wav", distance_meters=None):
        """Обработка всех файлов в папке"""
        audio_files = list(self.input_dir.glob(pattern))

        if not audio_files:
            print(f"❌ В папке {self.input_dir} нет файлов {pattern}")
            return []

        print(f"📁 Найдено файлов: {len(audio_files)}")
        print(f"🔧 Пресет: {preset}")
        if distance_meters:
            print(f"📏 Компенсация воздуха: {distance_meters}м")
        print("=" * 50)

        all_results = []

        for file_path in audio_files:
            result = self.process_file(file_path, preset, distance_meters)
            all_results.append(result)

        # Сохранение лога
        self.save_log()

        print("\n" + "=" * 50)
        print(f"✅ Обработано файлов: {len(all_results)}")
        print(f"📄 Лог: {self.output_dir / 'cleaning_log.json'}")

        return all_results

    def save_log(self):
        """Сохранение лога"""
        log_path = self.output_dir / 'cleaning_log.json'

        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump({
                'created': datetime.now().isoformat(),
                'total_files': len(self.processing_log),
                'entries': self.processing_log
            }, f, ensure_ascii=False, indent=2)


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Очистка аудио от шумов и нормализация")
    parser.add_argument("--input", "-i", default="rtsp_recordings", help="Папка с исходниками")
    parser.add_argument("--output", "-o", default="cleaned_audio", help="Папка для результатов")
    parser.add_argument("--preset", "-p", default="voice",
                        choices=['voice', 'engine', 'music', 'outdoor', 'aggressive', 'custom'],
                        help="Пресет обработки")
    parser.add_argument("--pattern", default="*.wav", help="Шаблон файлов")
    parser.add_argument("--distance", "-d", type=float, default=None,
                        help="Расстояние до источника (м), для компенсации воздуха")

    args = parser.parse_args()

    cleaner = AudioCleaner(args.input, args.output)

    # Для outdoor пресета используем расстояние
    if args.preset == 'outdoor' and args.distance is None:
        print("⚠ Для пресета 'outdoor' рекомендуется указать --distance")

    cleaner.process_directory(preset=args.preset, pattern=args.pattern, distance_meters=args.distance)
