"""
Анализатор звука двигателя автомобиля
Отображает АЧХ, спектрограмму и другие характеристики аудиофайла
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import librosa
import librosa.display
import soundfile as sf
from pathlib import Path


class SoundAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализатор звука двигателя")
        self.root.geometry("1200x800")
        
        self.audio_data = None
        self.sample_rate = None
        self.file_path = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # Верхняя панель с кнопками
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.btn_load = ttk.Button(top_frame, text="Загрузить MP3/WAV", command=self.load_file)
        self.btn_load.pack(side=tk.LEFT, padx=5)
        
        self.btn_analyze = ttk.Button(top_frame, text="Анализировать", command=self.analyze, state=tk.DISABLED)
        self.btn_analyze.pack(side=tk.LEFT, padx=5)
        
        self.lbl_status = ttk.Label(top_frame, text="Файл не загружен", foreground="gray")
        self.lbl_status.pack(side=tk.LEFT, padx=20)
        
        # Панель с информацией
        info_frame = ttk.LabelFrame(self.root, text="Информация о файле")
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.info_text = tk.Text(info_frame, height=6, wrap=tk.WORD)
        self.info_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Контейнер для графиков
        self.graph_frame = ttk.Frame(self.root)
        self.graph_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
    def load_file(self):
        file_path = filedialog.askopenfilename(
            title="Выберите аудиофайл",
            filetypes=[
                ("Аудиофайлы", "*.mp3 *.wav *.flac *.ogg *.m4a"),
                ("MP3 файлы", "*.mp3"),
                ("WAV файлы", "*.wav"),
                ("Все файлы", "*.*")
            ]
        )
        
        if file_path:
            self.file_path = file_path
            self.lbl_status.config(text=f"Загружен: {Path(file_path).name}", foreground="green")
            self.btn_analyze.config(state=tk.NORMAL)
            
            # Загрузка аудио
            try:
                self.audio_data, self.sample_rate = librosa.load(file_path, sr=None)
                self.info_text.insert(tk.END, f"Файл загружен успешно!\n")
                self.info_text.insert(tk.END, f"Путь: {file_path}\n")
                self.info_text.insert(tk.END, f"Длительность: {len(self.audio_data)/self.sample_rate:.2f} сек\n")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить файл:\n{str(e)}")
                self.file_path = None
                self.btn_analyze.config(state=tk.DISABLED)
                
    def analyze(self):
        if self.audio_data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите аудиофайл")
            return
            
        # Очистка предыдущих графиков
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
            
        # Создание фигуры с подграфиками
        fig = plt.Figure(figsize=(10, 8), dpi=100)
        
        # 1. Временной сигнал
        ax1 = fig.add_subplot(2, 2, 1)
        time = np.arange(0, len(self.audio_data)) / self.sample_rate
        ax1.plot(time, self.audio_data, linewidth=0.5, color='blue')
        ax1.set_title('Временной сигнал')
        ax1.set_xlabel('Время (сек)')
        ax1.set_ylabel('Амплитуда')
        ax1.grid(True, alpha=0.3)
        
        # 2. АЧХ (FFT)
        ax2 = fig.add_subplot(2, 2, 2)
        fft_data = np.fft.fft(self.audio_data)
        fft_data = np.abs(fft_data[:len(fft_data)//2])
        freqs = np.fft.fftfreq(len(self.audio_data), 1/self.sample_rate)[:len(fft_data)]
        
        # Ограничиваем до 20 кГц (слышимый диапазон)
        max_freq_idx = np.searchsorted(freqs, 20000)
        ax2.plot(freqs[:max_freq_idx], fft_data[:max_freq_idx], linewidth=0.5, color='red')
        ax2.set_title('АЧХ (Амплитудно-частотная характеристика)')
        ax2.set_xlabel('Частота (Гц)')
        ax2.set_ylabel('Амплитуда')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 5000])  # Для двигателя актуально до 5 кГц
        
        # 3. Спектрограмма
        ax3 = fig.add_subplot(2, 2, 3)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(self.audio_data)), ref=np.max)
        img = librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=self.sample_rate, ax=ax3)
        ax3.set_title('Спектрограмма')
        ax3.set_ylim([0, 5000])
        plt.colorbar(img, ax=ax3, format='%+2.0f dB')
        
        # 4. MFCC (для анализа тембра)
        ax4 = fig.add_subplot(2, 2, 4)
        mfccs = librosa.feature.mfcc(y=self.audio_data, sr=self.sample_rate, n_mfcc=13)
        img2 = librosa.display.specshow(mfccs, sr=self.sample_rate, x_axis='time', ax=ax4)
        ax4.set_title('MFCC коэффициенты')
        plt.colorbar(img2, ax=ax4, format='%+2.0f dB')
        
        plt.tight_layout()
        
        # Вставка графиков в Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Обновление информации
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, f"Файл: {Path(self.file_path).name}\n")
        self.info_text.insert(tk.END, f"Частота дискретизации: {self.sample_rate} Гц\n")
        self.info_text.insert(tk.END, f"Длительность: {len(self.audio_data)/self.sample_rate:.2f} сек\n")
        self.info_text.insert(tk.END, f"Каналы: 1 (моно)\n")
        self.info_text.insert(tk.END, f"Макс. амплитуда: {np.max(np.abs(self.audio_data)):.4f}\n")
        self.info_text.insert(tk.END, f"Средняя амплитуда: {np.mean(np.abs(self.audio_data)):.4f}\n")
        
        # Анализ доминирующих частот
        fft_magnitude = np.abs(np.fft.fft(self.audio_data))
        fft_freq = np.fft.fftfreq(len(self.audio_data), 1/self.sample_rate)
        
        # Находим топ-5 частот
        positive_freq_idx = fft_freq > 0
        top_indices = np.argsort(fft_magnitude[positive_freq_idx])[-10:][::-1]
        top_freqs = fft_freq[positive_freq_idx][top_indices]
        top_magnitudes = fft_magnitude[positive_freq_idx][top_indices]
        
        self.info_text.insert(tk.END, f"\nДоминирующие частоты (Гц):\n")
        for i, (freq, mag) in enumerate(zip(top_freqs[:5], top_magnitudes[:5])):
            self.info_text.insert(tk.END, f"  {i+1}. {freq:.1f} Гц (амплитуда: {mag:.1f})\n")
            
        # Рекомендации для анализа двигателя
        self.info_text.insert(tk.END, f"\n=== Анализ двигателя ===\n")
        
        # Проверка низких частот (0-100 Гц) - холостые обороты
        low_freq_idx = (fft_freq > 0) & (fft_freq < 100)
        low_freq_energy = np.sum(fft_magnitude[low_freq_idx])
        
        # Средние частоты (100-1000 Гц) - рабочие обороты
        mid_freq_idx = (fft_freq >= 100) & (fft_freq < 1000)
        mid_freq_energy = np.sum(fft_magnitude[mid_freq_idx])
        
        # Высокие частоты (1000-5000 Гц) - шумы, дефекты
        high_freq_idx = (fft_freq >= 1000) & (fft_freq < 5000)
        high_freq_energy = np.sum(fft_magnitude[high_freq_idx])
        
        total_energy = low_freq_energy + mid_freq_energy + high_freq_energy
        
        if total_energy > 0:
            self.info_text.insert(tk.END, f"Энергия НЧ (0-100 Гц): {low_freq_energy/total_energy*100:.1f}%\n")
            self.info_text.insert(tk.END, f"Энергия СЧ (100-1000 Гц): {mid_freq_energy/total_energy*100:.1f}%\n")
            self.info_text.insert(tk.END, f"Энергия ВЧ (1000-5000 Гц): {high_freq_energy/total_energy*100:.1f}%\n")
            
            if high_freq_energy/total_energy > 0.3:
                self.info_text.insert(tk.END, f"\n⚠ Внимание: Высокий уровень ВЧ шумов!\n")
                self.info_text.insert(tk.END, f"Возможны механические дефекты.\n")


def main():
    root = tk.Tk()
    app = SoundAnalyzerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
