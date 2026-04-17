import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from matplotlib.widgets import Button
import sys
import os

def load_wav_mono(filepath):
    """Загружает WAV, приводит к float и моно (если стерео)."""
    fs, data = wavfile.read(filepath)
    data = data.astype(np.float64)
    if data.ndim > 1:
        data = data.mean(axis=1)  # Усредняем каналы
    return fs, data

def compute_spectrum(signal, fs):
    """Вычисляет АЧХ через БПФ с применением окна Ханна."""
    N = len(signal)
    # Для длинных файлов можно обрезать до первых 10-20 секунд для ускорения
    # signal = signal[:int(fs * 10)]
    
    window = np.hanning(N)  # В новых numpy: np.windows.hann(N)
    windowed = signal * window
    
    fft_vals = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(N, 1/fs)
    magnitude = np.abs(fft_vals)
    
    # Перевод в дБ с защитой от log(0) и нормализацией (пик = 0 дБ)
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    magnitude_db -= np.max(magnitude_db)
    
    return freqs, magnitude_db

class SpectrumViewer:
    def __init__(self, filepath):
        fs, data = load_wav_mono(filepath)
        self.freqs, self.magnitude_db = compute_spectrum(data, fs)
        
        self.fig, self.ax = plt.subplots(figsize=(11, 6))
        self.line, = self.ax.plot(self.freqs, self.magnitude_db, linewidth=1.2, color='#1f77b4')
        self.ax.set_xlabel('Частота (Гц)')
        self.ax.set_ylabel('Амплитуда (дБ)')
        self.ax.set_title('АЧХ аудиосигнала')
        self.ax.grid(True, which='both', linestyle='--', alpha=0.6)
        
        # Настройки осей
        self.max_freq = fs / 2
        self.min_freq = 20.0
        self.ax.set_xscale('log')  # По умолчанию логарифмическая
        self.ax.set_xlim(self.min_freq, self.max_freq)
        self.ax.set_ylim(-100, 5)  # Диапазон дБ
        self.is_log = True
        
        # Кнопка переключения шкалы
        ax_btn = plt.axes([0.7, 0.92, 0.25, 0.06])
        self.btn = Button(ax_btn, 'Линейная шкала')
        self.btn.on_clicked(self.toggle_scale)
        
        plt.subplots_adjust(top=0.85, bottom=0.12)

    def toggle_scale(self, event):
        """Переключает шкалу частот и обновляет пределы оси X."""
        if self.is_log:
            self.ax.set_xscale('linear')
            self.ax.set_xlim(0, self.max_freq)
            self.btn.label.set_text('Логарифмическая шкала')
        else:
            self.ax.set_xscale('log')
            self.ax.set_xlim(self.min_freq, self.max_freq)
            self.btn.label.set_text('Линейная шкала')
        self.is_log = not self.is_log
        self.fig.canvas.draw_idle()

def main():
    if len(sys.argv) < 2:
        print("Использование: python wav_spectrum.py <файл.wav>")
        sys.exit(1)
        
    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"Ошибка: файл '{filepath}' не найден.")
        sys.exit(1)
        
    try:
        print(f"Загрузка: {filepath}")
        viewer = SpectrumViewer(filepath)
        # Matplotlib автоматически показывает панель инструментов (Zoom, Pan, Save)
        plt.show()
    except Exception as e:
        print(f"Ошибка при обработке: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()