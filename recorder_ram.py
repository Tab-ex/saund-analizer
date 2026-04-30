# recorder_ram.py
import subprocess
import numpy as np

def record_5s_to_ram():
    """
    Записывает 5 секунд аудио с микрофона напрямую в оперативную память.
    Возвращает: (numpy_array, sample_rate) или (None, None) при ошибке.
    """
    cmd = [
        "arecord", "-q", "-D", "dmic_sv", "-c2", "-r", "48000",
        "-f", "S32_LE", "-t", "raw", "-d", "5", "-"
    ]
    try:
        # timeout=6 гарантирует выход, если arecord зависнет
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=6)
        raw_data = proc.stdout
    except subprocess.TimeoutExpired:
        print("⏱  Таймаут записи")
        return None, None
    except Exception as e:
        print(f"❌ Ошибка arecord: {e}")
        return None, None

    if not raw_data or len(raw_data) == 0:
        return None, None

    # Декодирование S32_LE -> float32 в диапазоне [-1.0, 1.0]
    samples = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / (2**31)
    
    # Stereo -> Mono (усреднение каналов для совместимости с ML-моделью)
    mono_audio = samples.reshape(-1, 2).mean(axis=1)
    
    return mono_audio, 48000