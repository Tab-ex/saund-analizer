#!/usr/bin/env python3
import subprocess
import struct
import math
import time
import sys

# === НАСТРОЙКИ ===
DURATION = 5
RATE = 48000
CHANNELS = 2
BYTES_PER_SAMPLE = 4  # S32_LE
# Точный размер буфера на 5 сек: 5 * 48000 Гц * 2 канала * 4 байта = 1 920 000 байт
TOTAL_BYTES = DURATION * RATE * CHANNELS * BYTES_PER_SAMPLE
MAX_VAL = (1 << (BYTES_PER_SAMPLE * 8 - 1)) - 1  # 2^31 - 1 = 2147483647

def record_and_analyze(iteration):
    print(f"\n{'='*40}")
    print(f"🎙️  Итерация #{iteration} | Запись 5 сек в RAM...")

    # -t raw: без WAV-заголовка, чистый PCM
    # -      : вывод в stdout вместо файла
    cmd = [
        "arecord", "-q", "-D", "dmic_sv", "-c2", "-r", "48000",
        "-f", "S32_LE", "-t", "raw", "-d", "5", "-"
    ]

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        raw_data = proc.stdout.read(TOTAL_BYTES)
        proc.wait()
    except FileNotFoundError:
        print("❌ 'arecord' не найден. Установите: sudo apt install alsa-utils")
        sys.exit(1)

    if len(raw_data) != TOTAL_BYTES:
        print(f"⚠️  Получено {len(raw_data)} байт вместо {TOTAL_BYTES}. Пропуск.")
        return

    # Распаковка 32-битных чисел (Little Endian) прямо в памяти
    fmt = f"<{len(raw_data)//BYTES_PER_SAMPLE}i"
    samples = struct.unpack(fmt, raw_data)
    
    # Сразу удаляем сырые байты, освобождая ~1.9 МБ
    del raw_data

    # === ОПТИМИЗИРОВАННЫЙ РАСЧЁТ БЕЗ СОЗДАНИЯ ДОПОЛНИТЕЛЬНЫХ СПИСКОВ ===
    peak = 0
    sum_sq = 0
    count = len(samples)
    
    for s in samples:
        a = abs(s)
        if a > peak:
            peak = a
        sum_sq += s * s
        
    rms = math.sqrt(sum_sq / count)
    level_pct = (rms / MAX_VAL) * 100
    duration = count / (RATE * CHANNELS)

    # Удаляем кортеж сэмплов, освобождая ~3.8 МБ
    del samples

    # === ВЫВОД СВОДКИ ===
    print(f"⏱  Длительность: {duration:.2f} сек")
    print(f"📊 Формат      : {CHANNELS} кан., {RATE} Гц, 32 бит")
    print(f"📈 RMS        : {rms:,.0f} ({level_pct:.1f}%)")
    print(f"🔺 Пик        : {peak:,.0f} ({(peak/MAX_VAL)*100:.1f}%)")

    if level_pct < 0.5:
        status = "🤫 ТИХО / ФОН"
    elif level_pct < 5:
        status = "🗣️  УМЕРЕННО (речь)"
    else:
        status = "🔊 ГРОМКИЙ"
    print(f"🏷  Статус      : {status}")

def main():
    print("🔄 Циклическая запись в RAM (без обращения к SD-карте).")
    print("🛑 Для остановки нажмите Ctrl+C\n")

    iteration = 0
    try:
        while True:
            iteration += 1
            record_and_analyze(iteration)
            time.sleep(1)  # Пауза для стабилизации ALSA-буфера
    except KeyboardInterrupt:
        print("\n✅ Остановка по запросу пользователя.")
        sys.exit(0)

if __name__ == "__main__":
    main()