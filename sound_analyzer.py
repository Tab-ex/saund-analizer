"""
Анализатор звука двигателя автомобиля
Отображает АЧХ, АФЧХ, спектрограмму и другие характеристики аудиофайла
Поддерживает сравнение нескольких файлов одновременно
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import librosa
import librosa.display
from pathlib import Path
from scipy import signal
from scipy.fft import fft, fftfreq


# Цвета для разных дорожек
TRACK_COLORS = [
    '#1f77b4',  # синий
    '#ff7f0e',  # оранжевый
    '#2ca02c',  # зелёный
    '#d62728',  # красный
    '#9467bd',  # фиолетовый
    '#8c564b',  # коричневый
    '#e377c2',  # розовый
    '#7f7f7f',  # серый
    '#bcbd22',  # оливковый
    '#17becf',  # голубой
]


class SoundAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализатор звука двигателя")
        self.root.geometry("1400x900")
        
        # Хранилище данных для нескольких дорожек
        self.tracks = {}  # {name: {'audio': data, 'sr': sample_rate, 'color': color}}
        self.selected_track = None
        self.color_index = 0
        
        self.create_widgets()
        
    def create_widgets(self):
        # Верхняя панель с кнопками
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.btn_load = ttk.Button(top_frame, text="Загрузить файлы", command=self.load_files)
        self.btn_load.pack(side=tk.LEFT, padx=5)
        
        self.btn_analyze = ttk.Button(top_frame, text="Анализировать", command=self.analyze, state=tk.DISABLED)
        self.btn_analyze.pack(side=tk.LEFT, padx=5)
        
        self.btn_clear = ttk.Button(top_frame, text="Очистить всё", command=self.clear_all)
        self.btn_clear.pack(side=tk.LEFT, padx=5)
        
        self.lbl_status = ttk.Label(top_frame, text="Файлы не загружены", foreground="gray")
        self.lbl_status.pack(side=tk.LEFT, padx=20)
        
        # Панель управления треками
        track_frame = ttk.LabelFrame(self.root, text="Дорожки")
        track_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.track_listbox = tk.Listbox(track_frame, height=4, selectmode=tk.SINGLE)
        self.track_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        self.track_listbox.bind('<<ListboxSelect>>', self.on_track_select)
        
        track_btn_frame = ttk.Frame(track_frame)
        track_btn_frame.pack(side=tk.RIGHT, padx=5, pady=5)
        
        self.btn_remove_track = ttk.Button(track_btn_frame, text="Удалить", command=self.remove_track, state=tk.DISABLED)
        self.btn_remove_track.pack(pady=2)
        
        # Панель с информацией
        info_frame = ttk.LabelFrame(self.root, text="Информация")
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.info_text = tk.Text(info_frame, height=8, wrap=tk.WORD)
        self.info_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Контейнер для графиков с тулбаром
        self.graph_frame = ttk.Frame(self.root)
        self.graph_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
    def load_files(self):
        file_paths = filedialog.askopenfilenames(
            title="Выберите аудиофайлы",
            filetypes=[
                ("Аудиофайлы", "*.mp3 *.wav *.flac *.ogg *.m4a"),
                ("MP3 файлы", "*.mp3"),
                ("WAV файлы", "*.wav"),
                ("Все файлы", "*.*")
            ]
        )
        
        for file_path in file_paths:
            try:
                name = Path(file_path).name
                audio_data, sample_rate = librosa.load(file_path, sr=None)
                
                color = TRACK_COLORS[self.color_index % len(TRACK_COLORS)]
                self.color_index += 1
                
                self.tracks[name] = {
                    'audio': audio_data,
                    'sr': sample_rate,
                    'color': color,
                    'path': file_path
                }
                
                self.track_listbox.insert(tk.END, name)
                self.selected_track = name
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить {file_path}:\n{str(e)}")
        
        if self.tracks:
            count = len(self.tracks)
            self.lbl_status.config(text=f"Загружено файлов: {count}", foreground="green")
            self.btn_analyze.config(state=tk.NORMAL)
            self.btn_remove_track.config(state=tk.NORMAL)
            
    def on_track_select(self, event):
        selection = self.track_listbox.curselection()
        if selection:
            index = selection[0]
            self.selected_track = self.track_listbox.get(index)
            
    def remove_track(self):
        if self.selected_track and self.selected_track in self.tracks:
            del self.tracks[self.selected_track]
            self.track_listbox.delete(self.track_listbox.get(0, tk.END).index(self.selected_track))
            
            if not self.tracks:
                self.selected_track = None
                self.lbl_status.config(text="Файлы не загружены", foreground="gray")
                self.btn_analyze.config(state=tk.DISABLED)
                self.btn_remove_track.config(state=tk.DISABLED)
                self.info_text.delete(1.0, tk.END)
                for widget in self.graph_frame.winfo_children():
                    widget.destroy()
            else:
                self.selected_track = list(self.tracks.keys())[0]
                
    def clear_all(self):
        self.tracks.clear()
        self.selected_track = None
        self.color_index = 0
        self.track_listbox.delete(0, tk.END)
        self.lbl_status.config(text="Файлы не загружены", foreground="gray")
        self.btn_analyze.config(state=tk.DISABLED)
        self.btn_remove_track.config(state=tk.DISABLED)
        self.info_text.delete(1.0, tk.END)
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
            
    def analyze(self):
        if not self.tracks:
            messagebox.showwarning("Предупреждение", "Сначала загрузите аудиофайлы")
            return
            
        # Очистка предыдущих графиков
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        
        # Создание фигуры с подграфиками
        fig = plt.Figure(figsize=(12, 10), dpi=100)
        
        # 1. Временной сигнал (все дорожки)
        ax1 = fig.add_subplot(2, 3, 1)
        max_len = 0
        for name, track in self.tracks.items():
            audio = track['audio']
            sr = track['sr']
            color = track['color']
            time = np.arange(0, len(audio)) / sr
            ax1.plot(time, audio, linewidth=0.5, color=color, label=name, alpha=0.8)
            max_len = max(max_len, len(audio))
        
        ax1.set_title('Временной сигнал')
        ax1.set_xlabel('Время (сек)')
        ax1.set_ylabel('Амплитуда')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=8)
        
        # 2. АЧХ (FFT) - все дорожки
        ax2 = fig.add_subplot(2, 3, 2)
        for name, track in self.tracks.items():
            audio = track['audio']
            sr = track['sr']
            color = track['color']
            
            fft_data = np.abs(fft(audio))
            fft_data = fft_data[:len(fft_data)//2]
            freqs = fftfreq(len(audio), 1/sr)[:len(fft_data)]
            
            max_freq_idx = np.searchsorted(freqs, 20000)
            ax2.plot(freqs[:max_freq_idx], fft_data[:max_freq_idx], 
                    linewidth=0.5, color=color, label=name, alpha=0.8)
        
        ax2.set_title('АЧХ (Амплитудно-частотная характеристика)')
        ax2.set_xlabel('Частота (Гц)')
        ax2.set_ylabel('Амплитуда')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 5000])
        ax2.legend(loc='upper right', fontsize=8)
        
        # 3. АФЧХ (Амплитудно-фазовая характеристика)
        ax3 = fig.add_subplot(2, 3, 3)
        for name, track in self.tracks.items():
            audio = track['audio']
            sr = track['sr']
            color = track['color']
            
            fft_complex = fft(audio)
            fft_magnitude = np.abs(fft_complex[:len(fft_complex)//2])
            fft_phase = np.angle(fft_complex[:len(fft_complex)//2])
            freqs = fftfreq(len(audio), 1/sr)[:len(fft_magnitude)]
            
            max_freq_idx = np.searchsorted(freqs, 5000)
            ax3.plot(freqs[:max_freq_idx], fft_phase[:max_freq_idx], 
                    linewidth=0.5, color=color, label=name, alpha=0.8)
        
        ax3.set_title('АФЧХ (Фазовая характеристика)')
        ax3.set_xlabel('Частота (Гц)')
        ax3.set_ylabel('Фаза (рад)')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([0, 5000])
        ax3.legend(loc='upper right', fontsize=8)
        
        # 4. Спектрограмма (выбранный трек или первый)
        display_track = self.selected_track if self.selected_track else list(self.tracks.keys())[0]
        track = self.tracks[display_track]
        audio = track['audio']
        sr = track['sr']
        
        ax4 = fig.add_subplot(2, 3, 4)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        img = librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=ax4)
        ax4.set_title(f'Спектрограмма ({display_track})')
        ax4.set_ylim([0, 5000])
        plt.colorbar(img, ax=ax4, format='%+2.0f dB')
        
        # 5. MFCC (выбранный трек)
        ax5 = fig.add_subplot(2, 3, 5)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        img2 = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax5)
        ax5.set_title(f'MFCC ({display_track})')
        plt.colorbar(img2, ax=ax5, format='%+2.0f dB')
        
        # 6. Сравнительная гистограмма энергии по диапазонам
        ax6 = fig.add_subplot(2, 3, 6)
        ranges = ['НЧ\n(0-100 Гц)', 'СЧ\n(100-1000 Гц)', 'ВЧ\n(1000-5000 Гц)']
        x_pos = np.arange(len(ranges))
        width = 0.8 / len(self.tracks) if len(self.tracks) > 1 else 0.6
        
        for i, (name, track) in enumerate(self.tracks.items()):
            audio = track['audio']
            color = track['color']
            
            fft_magnitude = np.abs(fft(audio))
            fft_freq = fftfreq(len(audio), 1/track['sr'])
            
            low_idx = (fft_freq > 0) & (fft_freq < 100)
            mid_idx = (fft_freq >= 100) & (fft_freq < 1000)
            high_idx = (fft_freq >= 1000) & (fft_freq < 5000)
            
            low_energy = np.sum(fft_magnitude[low_idx])
            mid_energy = np.sum(fft_magnitude[mid_idx])
            high_energy = np.sum(fft_magnitude[high_idx])
            total = low_energy + mid_energy + high_energy
            
            if total > 0:
                energies = [low_energy/total*100, mid_energy/total*100, high_energy/total*100]
            else:
                energies = [0, 0, 0]
            
            offset = (i - (len(self.tracks)-1)/2) * width
            ax6.bar(x_pos + offset, energies, width, label=name, color=color, alpha=0.8)
        
        ax6.set_title('Распределение энергии по диапазонам')
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(ranges)
        ax6.set_ylabel('Энергия (%)')
        ax6.legend(loc='upper right', fontsize=8)
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Вставка графиков в Tkinter с тулбаром для навигации
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Тулбар с кнопками (зум, панорамирование, сохранение)
        toolbar = NavigationToolbar2Tk(canvas, self.graph_frame)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)
        
        # Обновление информации
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, f"=== Анализ дорожек: {len(self.tracks)} ===\n\n")
        
        for name, track in self.tracks.items():
            audio = track['audio']
            sr = track['sr']
            color = track['color']
            
            self.info_text.insert(tk.END, f"📁 {name}\n")
            self.info_text.insert(tk.END, f"   Цвет: {color}\n")
            self.info_text.insert(tk.END, f"   Частота дискретизации: {sr} Гц\n")
            self.info_text.insert(tk.END, f"   Длительность: {len(audio)/sr:.2f} сек\n")
            self.info_text.insert(tk.END, f"   Макс. амплитуда: {np.max(np.abs(audio)):.4f}\n")
            self.info_text.insert(tk.END, f"   Средняя амплитуда: {np.mean(np.abs(audio)):.4f}\n")
            
            # Анализ доминирующих частот
            fft_magnitude = np.abs(fft(audio))
            fft_freq_vals = fftfreq(len(audio), 1/sr)
            
            positive_freq_idx = fft_freq_vals > 0
            top_indices = np.argsort(fft_magnitude[positive_freq_idx])[-5:][::-1]
            top_freqs = fft_freq_vals[positive_freq_idx][top_indices]
            top_magnitudes = fft_magnitude[positive_freq_idx][top_indices]
            
            self.info_text.insert(tk.END, f"   Доминирующие частоты:\n")
            for j, (freq, mag) in enumerate(zip(top_freqs[:3], top_magnitudes[:3])):
                self.info_text.insert(tk.END, f"      {j+1}. {freq:.1f} Гц\n")
            
            # Анализ двигателя
            low_idx = (fft_freq_vals > 0) & (fft_freq_vals < 100)
            mid_idx = (fft_freq_vals >= 100) & (fft_freq_vals < 1000)
            high_idx = (fft_freq_vals >= 1000) & (fft_freq_vals < 5000)
            
            low_energy = np.sum(fft_magnitude[low_idx])
            mid_energy = np.sum(fft_magnitude[mid_idx])
            high_energy = np.sum(fft_magnitude[high_idx])
            total_energy = low_energy + mid_energy + high_energy
            
            if total_energy > 0:
                high_pct = high_energy/total_energy*100
                self.info_text.insert(tk.END, f"   Энергия ВЧ: {high_pct:.1f}%")
                if high_pct > 30:
                    self.info_text.insert(tk.END, f" ⚠ Высокий уровень!\n")
                else:
                    self.info_text.insert(tk.END, f" ✓ Норма\n")
            
            self.info_text.insert(tk.END, "\n")


def main():
    root = tk.Tk()
    app = SoundAnalyzerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
