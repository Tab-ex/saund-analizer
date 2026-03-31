"""
Few-Shot Learning для детекции звуков
Дообучение YAMNet на небольшом датасете (Transfer Learning)
"""

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import soundfile as sf
import os
import json
from pathlib import Path
from datetime import datetime
import resampy


class FewShotTrainer:
    """Обучение методом Few-Shot Learning на базе YAMNet"""
    
    def __init__(self, model_path="yamnet.tflite", class_map_path="yamnet_class_map.csv"):
        self.model_path = model_path
        self.class_map_path = class_map_path
        
        # Параметры YAMNet
        self.sample_rate = 16000
        self.window_size = 15600  # 0.975 сек
        
        # Классы для детекции (можно изменить)
        self.target_classes = {
            'moped': [288, 289],  # Motorcycle, Engine
            'drone': [],          # Заполняется пользователем
        }
        
        self.model = None
        self.class_names = {}
        self._load_class_map()
        
    def _load_class_map(self):
        """Загрузка карты классов"""
        if os.path.exists(self.class_map_path):
            import csv
            with open(self.class_map_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) >= 3:
                        try:
                            idx = int(row[0])
                            self.class_names[idx] = row[2]
                        except ValueError:
                            pass
            print(f"✅ Загружено {len(self.class_names)} классов YAMNet")
    
    def prepare_data(self, data_dir="data/augmented", target_class_name="drone"):
        """
        Подготовка данных для обучения
        
        Args:
            data_dir: папка с аугментированными данными
            target_class_name: имя целевого класса (например, 'drone')
        
        Returns:
            spectrograms, labels
        """
        data_path = Path(data_dir)
        
        # Загрузка лога аугментации
        log_path = data_path / 'augmentation_log.json'
        if not log_path.exists():
            raise FileNotFoundError(f"Лог аугментации не найден: {log_path}")
        
        with open(log_path, 'r', encoding='utf-8') as f:
            log = json.load(f)
        
        print(f"📊 Загрузка данных из {data_dir}")
        print(f"   Всего файлов: {log['total_files']}")
        
        # Группировка по исходным файлам
        source_files = {}
        for entry in log['entries']:
            source = Path(entry['source']).stem
            if source not in source_files:
                source_files[source] = []
            source_files[source].append(entry)
        
        print(f"   Уникальных источников: {len(source_files)}")
        
        # Создание датасета
        spectrograms = []
        labels = []  # 0 = фон, 1 = целевой класс
        
        # Загрузка модели для экстракции признаков
        print("\n🔧 Загрузка модели...")
        self._load_model()
        
        for source_name, variants in source_files.items():
            print(f"\n🎵 Обработка: {source_name}")
            
            for entry in variants:
                audio_path = entry['output']
                
                if not os.path.exists(audio_path):
                    print(f"   ⚠️ Файл не найден: {audio_path}")
                    continue
                
                # Извлечение спектрограммы
                spec = self._extract_features(audio_path)
                if spec is not None:
                    spectrograms.append(spec)
                    labels.append(1)  # Целевой класс
        
        return np.array(spectrograms), np.array(labels)
    
    def _extract_features(self, audio_path):
        """Извлечение признаков из аудиофайла"""
        try:
            audio, sr = sf.read(audio_path, dtype='float32')
            
            # Конвертация в моно
            if len(audio.shape) > 1:
                audio = audio[:, 0]
            
            # Ресемплинг до 16kHz
            if sr != self.sample_rate:
                audio = resampy.resample(audio, sr, self.sample_rate)
            
            # Обрезка до нужного размера
            if len(audio) < self.window_size:
                audio = np.pad(audio, (0, self.window_size - len(audio)))
            else:
                audio = audio[:self.window_size]
            
            # YAMNet принимает raw audio
            return audio.astype(np.float32)
            
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            return None
    
    def _load_model(self):
        """Загрузка модели YAMNet"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Модель не найдена: {self.model_path}\n"
                "Скачайте: powershell Invoke-WebRequest -Uri \"https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite\" -OutFile \"yamnet.tflite\""
            )
        
        # Используем TFLite модель
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print("✅ Модель загружена")
    
    def create_classifier_model(self, input_shape=(15600,), num_classes=2):
        """
        Создание классификатора поверх YAMNet
        
        Args:
            input_shape: форма входных данных
            num_classes: количество классов (фон + целевые)
        """
        from tensorflow import keras
        from tensorflow.keras import layers
        
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Reshape((input_shape[0], 1)),
            
            # Свёрточные слои
            layers.Conv1D(32, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            
            # Полносвязные слои
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            
            # Выходной слой
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def train(self, data_dir="data/augmented", epochs=50, batch_size=16, 
              validation_split=0.2, output_model="drone_detector.h5"):
        """
        Обучение модели
        
        Args:
            data_dir: папка с данными
            epochs: количество эпох
            batch_size: размер батча
            validation_split: доля валидации
            output_model: путь для сохранения модели
        """
        from tensorflow import keras
        
        print("=" * 60)
        print("🚀 Few-Shot Learning - Обучение детектора")
        print("=" * 60)
        
        # Подготовка данных
        X, y = self.prepare_data(data_dir)
        
        print(f"\n📊 Данные:")
        print(f"   Примеров: {len(X)}")
        print(f"   Положительных: {np.sum(y)}")
        print(f"   Форма: {X.shape}")
        
        if len(X) < 10:
            print("\n⚠️ ПРЕДУПРЕЖДЕНИЕ: Мало данных для обучения!")
            print("   Рекомендуется хотя бы 20-30 примеров после аугментации")
        
        # Разделение на train/val
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        print(f"\n📈 Train: {len(X_train)}, Validation: {len(X_val)}")
        
        # Создание модели
        print("\n🏗️ Создание модели...")
        model = self.create_classifier_model()
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(name='precision'), 
                     keras.metrics.Recall(name='recall')]
        )
        
        print(f"   Параметры: {model.count_params():,}")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
            ),
            keras.callbacks.ModelCheckpoint(
                output_model, monitor='val_accuracy', save_best_only=True
            )
        ]
        
        # Обучение
        print("\n📚 Обучение...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Оценка
        print("\n📊 Результаты на валидации:")
        val_results = model.evaluate(X_val, y_val, verbose=0)
        print(f"   Accuracy: {val_results[1]:.3f}")
        print(f"   Precision: {val_results[2]:.3f}")
        print(f"   Recall: {val_results[3]:.3f}")
        
        # Сохранение метрик
        metrics = {
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'epochs_trained': len(history.history['loss']),
            'final_accuracy': float(val_results[1]),
            'final_precision': float(val_results[2]),
            'final_recall': float(val_results[3]),
            'history': {k: [float(v) for v in values] 
                       for k, values in history.history.items()}
        }
        
        metrics_path = Path(output_model).with_suffix('.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✅ Модель сохранена: {output_model}")
        print(f"📄 Метрики: {metrics_path}")
        
        return model, history
    
    def predict(self, audio_path, model_path="drone_detector.h5", threshold=0.5):
        """Предсказание для аудиофайла"""
        from tensorflow import keras
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        
        model = keras.models.load_model(model_path)
        
        # Извлечение признаков
        features = self._extract_features(audio_path)
        if features is None:
            return None
        
        # Предсказание
        features = features.reshape(1, -1)
        prediction = model.predict(features, verbose=0)[0]
        
        return {
            'class': int(np.argmax(prediction)),
            'confidence': float(np.max(prediction)),
            'probabilities': prediction.tolist()
        }


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Few-Shot Learning для детекции звуков")
    parser.add_argument("command", choices=["train", "predict"], help="Команда")
    parser.add_argument("--data", "-d", default="data/augmented", help="Папка с данными")
    parser.add_argument("--model", "-m", default="drone_detector.h5", help="Модель")
    parser.add_argument("--audio", "-a", help="Аудиофайл для предсказания")
    parser.add_argument("--epochs", "-e", type=int, default=50, help="Эпох")
    parser.add_argument("--batch", "-b", type=int, default=16, help="Размер батча")
    
    args = parser.parse_args()
    
    trainer = FewShotTrainer()
    
    if args.command == "train":
        trainer.train(
            data_dir=args.data,
            epochs=args.epochs,
            batch_size=args.batch,
            output_model=args.model
        )
    elif args.command == "predict":
        if not args.audio:
            print("❌ Укажите аудиофайл: --audio файл.wav")
        else:
            result = trainer.predict(args.audio, args.model)
            print(f"Класс: {result['class']}")
            print(f"Уверенность: {result['confidence']:.3f}")
