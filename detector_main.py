# detector_main.py
import time
import numpy as np
import sys
import os

# Добавляем текущую директорию в путь импортов
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from recorder_ram import record_5s_to_ram
from sender_json import send_to_server
from moped_ml_detector import MopedFeatureExtractor, MopedMLDetector

# === НАСТРОЙКИ ===
MODEL_PATH = "my_model_1.pkl"
SERVER_URL = "http://192.168.1.100:5000/api/detect"  # <-- ЗАМЕНИТЕ НА ВАШ ENDPOINT
CONFIDENCE_THRESHOLD = 0.75
LOOP_DELAY = 1  # Пауза между циклами (сек)

def main():
    print("🔧 Инициализация компонентов...")

    # 1. Загрузка ML-модели
    detector = MopedMLDetector(model_path=MODEL_PATH)
    if not detector.load_model():
        print("❌ Не удалось загрузить модель. Проверьте наличие moped_model.pkl")
        sys.exit(1)

    # 2. Настройка экстрактора под ваше железо (48kHz, окна по 1 сек)
    extractor = MopedFeatureExtractor(sample_rate=48000, window_size=1.0)

    print("✅ Компоненты готовы. Запуск цикла...")
    print("🛑 Для остановки нажмите Ctrl+C\n")

    iteration = 0
    try:
        while True:
            iteration += 1
            print(f"🔄 Цикл #{iteration} | Запись 5 сек в RAM...")

            # --- ШАГ 1: Запись ---
            audio_data, sr = record_5s_to_ram()
            if audio_data is None:
                print("⚠️ Ошибка записи, пропуск...")
                time.sleep(LOOP_DELAY)
                continue

            # --- ШАГ 2: Анализ в памяти (без сохранения на диск) ---
            window_samples = int(1.0 * sr)
            moped_windows = 0
            max_conf = 0.0
            predictions = []

            # Проходим по аудио окнами по 1 секунде
            for i in range(0, len(audio_data) - window_samples, window_samples):
                chunk = audio_data[i:i+window_samples]
                features = extractor.extract_features(chunk)
                
                if features:
                    is_moped, conf = detector.predict(features, threshold=CONFIDENCE_THRESHOLD)
                    predictions.append(conf)
                    if is_moped:
                        moped_windows += 1
                    if conf > max_conf:
                        max_conf = conf

            # Агрегация результатов
            avg_conf = float(np.mean(predictions)) if predictions else 0.0
            is_detected = moped_windows > 0
            status = "DETECTED" if is_detected else "CLEAN"

            print(f"📊 Результат: {status} | MaxConf: {max_conf:.3f} | AvgConf: {avg_conf:.3f} | OK: {moped_windows}/5")

            # --- ШАГ 3: Отправка на сервер ---
            payload = {
                "iteration": iteration,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "status": status,
                "moped_detected": is_detected,
                "confidence_avg": round(avg_conf, 4),
                "confidence_max": round(max_conf, 4),
                "positive_windows": moped_windows,
                "total_windows": len(predictions)
            }
            send_to_server(payload, server_url=SERVER_URL)

            # Освобождаем память явно (numpy сам соберёт мусор, но для Pi3 полезно)
            del audio_data, predictions
            time.sleep(LOOP_DELAY)

    except KeyboardInterrupt:
        print("\n✅ Детектор остановлен пользователем.")
        sys.exit(0)

if __name__ == "__main__":
    main()