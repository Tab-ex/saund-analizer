# Детектор двигателя мопеда

## Установка

### 1. Активируйте виртуальное окружение

```bash
.\venv\Scripts\activate
```

### 2. Установите основные зависимости

```bash
pip install -r requirements.txt
```

### 3. Установите tflite-runtime

**Важно:** `tflite-runtime` недоступен в PyPI для Windows. Установите одним из способов:

#### Способ А: Через tensorflow (рекомендуется)
```bash
pip install tensorflow
```

Затем измените импорт в `moped_detector.py`:
```python
from tensorflow.lite.python import interpreter as tflite
```

#### Способ Б: Скачать готовый wheel
Найдите подходящий wheel для вашей версии Python на [GitHub Releases](https://github.com/amaiya/tflite-runtime-win/releases) или [Google Coral](https://github.com/google-coral/pycoral/releases)

```bash
pip install tflite_runtime-2.x.x-cp3xx-cp312-win_amd64.whl
```

### 4. Скачайте модель YAMNet

```bash
# PowerShell
Invoke-WebRequest -Uri "https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite" -OutFile "yamnet.tflite"

# или через wget (если установлен)
wget https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite -O yamnet.tflite
```

## Запуск

```bash
python moped_detector.py
```

## Примечания

- Для работы требуется микрофон
- Модель YAMNet распознаёт звуки двигателя и мотоцикла
- Порог чувствительности можно изменить в параметре `threshold` функции `is_moped_engine()`
