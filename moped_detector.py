"""
Детектор двигателя мопеда на основе YAMNet
Использует микрофон для обнаружения звуков мотоцикла/двигателя

Перед запуском скачайте модель:
    powershell Invoke-WebRequest -Uri "https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite" -OutFile "yamnet.tflite"
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import tensorflow as tf
import resampy
import time
import os
import csv

# Проверка наличия модели
MODEL_PATH = "yamnet.tflite"
if not os.path.exists(MODEL_PATH):
    print(f"❌ Модель {MODEL_PATH} не найдена!")
    print("Скачайте её:")
    print('   powershell Invoke-WebRequest -Uri "https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite" -OutFile "yamnet.tflite"')
    exit(1)

# Загрузка карты классов
CLASS_MAP_PATH = "yamnet_class_map.csv"
class_names = {}
if os.path.exists(CLASS_MAP_PATH):
    with open(CLASS_MAP_PATH, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # пропускаем заголовок
        for row in reader:
            if len(row) >= 3:
                try:
                    idx = int(row[0])
                    class_names[idx] = row[2]  # название класса
                except ValueError:
                    pass
    print(f"✅ Загружено {len(class_names)} классов")
else:
    print(f"⚠️ Файл {CLASS_MAP_PATH} не найден (скачайте с GitHub tensorflow/models)")

# Используем новый API LiteRT
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Проверка входных данных модели
print(f"📥 Вход: {input_details[0]['shape']} dtype={input_details[0]['dtype']}")
print(f"📤 Выход: {output_details[0]['shape']}")

# YAMNet ожидает спектрограмму, поэтому используем tf.signal для обработки
def extract_spectrogram(audio, sample_rate=16000, window_size=0.96):
    """Преобразование аудио в спектрограмму для YAMNet"""
    # YAMNet требует спектрограмму с определёнными параметрами
    # Используем встроенную предобработку через tf.signal
    spectrogram = tf.signal.stft(
        audio,
        frame_length=int(0.025 * sample_rate),  # 25ms окно
        frame_step=int(0.010 * sample_rate),    # 10ms шаг
        fft_length=int(0.025 * sample_rate)
    )
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.pow(spectrogram, 0.5)  # Корень из амплитуды
    return spectrogram.numpy()


def is_moped_engine(scores, threshold=0.4):
    """Проверка наличия звука двигателя мопеда"""
    motorcycle_idx = 288  # Motorcycle
    engine_idx = 289      # Engine
    return scores[motorcycle_idx] > threshold or scores[engine_idx] > threshold


def get_top_sounds(scores, top_n=5):
    """Возвращает топ-N звуков с наибольшими вероятностями"""
    top_indices = np.argsort(scores)[-top_n:][::-1]
    results = []
    for idx in top_indices:
        name = class_names.get(idx, f"Unknown-{idx}")
        results.append((idx, name, scores[idx]))
    return results


def analyze_audio_file(file_path, threshold=0.4, debug=False):
    """Анализ аудиофайла вместо микрофона"""
    audio, sr = sf.read(file_path, dtype='float32')
    
    # Конвертация в моно если стерео
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    
    # Ресемплинг до 16kHz если нужно
    if sr != 16000:
        audio = resampy.resample(audio, sr, 16000)
        sr = 16000
    
    print(f"🔍 Анализ файла: {file_path}")
    print(f"Длительность: {len(audio)/sr:.2f} сек")
    
    # YAMNet требует ровно 15600 сэмплов (0.975 сек при 16kHz)
    expected_samples = input_details[0]['shape'][0]  # 15600
    print(f"📊 Требуется сэмплов: {expected_samples}")
    
    for i in range(0, len(audio) - expected_samples, expected_samples):
        window = audio[i:i + expected_samples]
        
        # Передаём как 1D массив [15600], а не 2D [1, 15600]
        input_data = window.astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        scores = interpreter.get_tensor(output_details[0]['index'])[0]
        
        if debug and i == 0:
            print("\n📊 Топ-5 звуков в первом окне:")
            for idx, name, score in get_top_sounds(scores):
                print(f"   Класс {idx}: {name} ({score:.3f})")
        
        if is_moped_engine(scores, threshold):
            time_sec = i / sr
            print(f"⏱ {time_sec:.1f} сек - 🔊 ОБНАРУЖЕН ДВИГАТЕЛЬ МОПЕДА!")
    
    print("✅ Анализ завершён")


if __name__ == "__main__":
    import sys
    
    # Если передан файл - анализируем его, иначе - режим микрофона
    if len(sys.argv) > 1:
        analyze_audio_file(sys.argv[1], debug=True)
    else:
        print("🚀 Детектор двигателя мопеда запущен...")
        print("💡 Для анализа файла: python moped_detector.py <путь_к_файлу.wav>")
        
        # YAMNet требует 15600 сэмплов
        expected_samples = input_details[0]['shape'][0]
        
        with sd.InputStream(samplerate=16000, channels=1, dtype='float32', blocksize=expected_samples):
            while True:
                audio = sd.read(expected_samples)[0][:, 0]  # mono, 1D массив
                
                interpreter.set_tensor(input_details[0]['index'], audio.astype(np.float32))
                interpreter.invoke()
                scores = interpreter.get_tensor(output_details[0]['index'])[0]
                
                if is_moped_engine(scores):
                    print("🔊 ОБНАРУЖЕН ДВИГАТЕЛЬ МОПЕДА!")
                    # Здесь можно добавить GPIO, уведомление и т.д.
                
                time.sleep(0.5)  # пауза между проверками
