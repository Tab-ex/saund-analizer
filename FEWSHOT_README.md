# Few-Shot Learning для детекции звуков

## Структура проекта

```
saund-analizer/
├── data/
│   ├── raw/           # Исходные аудиофайлы (запишите сюда)
│   └── augmented/     # Результаты аугментации
│       ├── *.wav      # Аугментированные файлы
│       └── augmentation_log.json  # Лог трансформаций
├── audio_augmenter.py    # Инструменты аугментации
├── train_fewshot.py      # Обучение модели
├── moped_detector.py     # Детектор (готовая модель)
└── FEWSHOT_README.md     # Этот файл
```

## Быстрый старт

### Шаг 1: Подготовка данных

1. Поместите исходные аудиофайлы в `data/raw/`
   - Формат: WAV, MP3
   - Рекомендуется: 5-10 записей целевого звука

2. Запустите аугментацию:
```bash
.\venv\Scripts\activate
python audio_augmenter.py --variants 10
```

Это создаст 10 вариантов каждого файла с разными трансформациями.

### Шаг 2: Обучение модели

```bash
python train_fewshot.py train --epochs 50 --batch 16
```

Модель сохранится как `drone_detector.h5`

### Шаг 3: Использование

```bash
python train_fewshot.py predict --model drone_detector.h5 --audio ваш_звук.wav
```

## Инструменты аугментации

### Базовое использование

```bash
python audio_augmenter.py --input data/raw --output data/augmented --variants 5
```

### Параметры

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--input`, `-i` | Папка с исходниками | `data/raw` |
| `--output`, `-o` | Папка для результатов | `data/augmented` |
| `--variants`, `-n` | Вариантов на файл | `5` |

### Типы трансформаций

- **Speed** — изменение скорости (0.85x - 1.15x)
- **Pitch Shift** — сдвиг высоты тона (±5 полутонов)
- **Volume** — изменение громкости
- **Noise** — добавление белого шума
- **Bandpass Filter** — фильтрация частот
- **Time Stretch** — растягивание времени

### Уровни аугментации

- `light` — 1-2 трансформации, мягкие параметры
- `medium` — 2-3 трансформации, средние параметры
- `heavy` — 3-4 трансформации, агрессивные параметры

## Обучение модели

### Базовое обучение

```bash
python train_fewshot.py train --data data/augmented --epochs 50 --batch 16
```

### Параметры обучения

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--data`, `-d` | Папка с данными | `data/augmented` |
| `--model`, `-m` | Имя модели | `drone_detector.h5` |
| `--epochs`, `-e` | Количество эпох | `50` |
| `--batch`, `-b` | Размер батча | `16` |

### Предсказание

```bash
python train_fewshot.py predict --model drone_detector.h5 --audio тест.wav
```

## Рекомендации по данным

### Минимальные требования

| Метод | Минимум записей | После аугментации |
|-------|-----------------|-------------------|
| Few-Shot | 3-5 | 30-50 |
| Transfer Learning | 10-20 | 100-200 |

### Советы

1. **Качество записей**
   - Записывайте в тишине
   - Избегайте фоновых шумов
   - Длительность: 5-30 секунд

2. **Разнообразие**
   - Разные расстояния до источника
   - Разные ракурсы
   - Разное время суток

3. **Аугментация**
   - Начните с `--variants 5`
   - Для малых данных: `--variants 10-15`
   - Проверяйте качество визуально

## Примеры использования

### Пример 1: Детектор дрона

```bash
# 1. Запишите звуки дрона в data/raw/
# 2. Аугментация
python audio_augmenter.py -n 10

# 3. Обучение
python train_fewshot.py train -e 100 -b 8

# 4. Тест
python train_fewshot.py predict -m drone_detector.h5 -a тест.wav
```

### Пример 2: Детектор сигнала

```bash
# Для другого типа звука
python audio_augmenter.py -i data/raw_signals -o data/aug_signals -n 8
python train_fewshot.py train -d data/aug_signals -m signal_detector.h5
```

## Логирование

После аугментации создаётся `augmentation_log.json`:

```json
{
  "created": "2026-03-31T12:00:00",
  "total_files": 50,
  "entries": [
    {
      "source": "drone_sound.wav",
      "output": "data/augmented/drone_sound_aug1_medium_20260331_120000.wav",
      "level": "medium",
      "transforms": ["speed_1.05", "pitch_+2.0", "noise_0.008"],
      "timestamp": "20260331_120000"
    }
  ]
}
```

## Требования

Установите зависимости:

```bash
pip install scikit-learn librosa
```

Основные зависимости уже есть в `requirements.txt`.

## Структура выходных данных

```
data/augmented/
├── drone_01_aug1_light_20260331_120000.wav
├── drone_01_aug2_medium_20260331_120001.wav
├── drone_01_aug3_heavy_20260331_120002.wav
├── drone_02_aug1_light_20260331_120003.wav
└── augmentation_log.json
```

Формат имён: `{исходник}_aug{номер}_{уровень}_{дата_время}.wav`

## Возможные проблемы

### Мало данных
- Увеличьте `--variants` до 15-20
- Добавьте больше исходных записей
- Используйте `heavy` аугментацию

### Переобучение
- Уменьшите количество эпох
- Увеличьте `validation_split`
- Добавьте Dropout в модель

### Низкая точность
- Проверьте качество записей
- Увеличьте датасет
- Попробуйте разные параметры аугментации
