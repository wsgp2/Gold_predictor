#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль с моделями машинного обучения для прогнозирования цены золота.
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# Для XGBoost
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix

# Для LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Настройка TensorFlow
tf.random.set_seed(42)  # Для воспроизводимости результатов


class XGBoostModel:
    """Класс для работы с моделью XGBoost."""
    
    def __init__(self, model_dir="../models", target_type='binary'):
        """
        Инициализация модели XGBoost.
        
        Args:
            model_dir (str): Директория для сохранения моделей
            target_type (str): Тип задачи ('binary', 'regression', 'classification')
        """
        self.model_dir = model_dir
        self.target_type = target_type
        self.model = None
        
        # Создание директории для моделей, если она не существует
        os.makedirs(model_dir, exist_ok=True)
        
        # Задаем начальные параметры в зависимости от типа задачи
        if target_type == 'binary':
            self.params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'eta': 0.1,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0,
                'seed': 42
            }
        elif target_type == 'regression':
            self.params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'eta': 0.1,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0,
                'seed': 42
            }
        elif target_type == 'classification':
            self.params = {
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'num_class': 5,  # 5 классов: сильно вниз, вниз, боковик, вверх, сильно вверх
                'eta': 0.1,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0,
                'seed': 42
            }
        else:
            raise ValueError(f"Неизвестный тип задачи: {target_type}")
    
    def train(self, X_train, y_train, X_val=None, y_val=None, params=None, num_rounds=1000, early_stopping_rounds=50):
        """
        Обучение модели XGBoost.
        
        Args:
            X_train (pandas.DataFrame): Обучающие признаки
            y_train (pandas.Series): Обучающие метки
            X_val (pandas.DataFrame, optional): Валидационные признаки
            y_val (pandas.Series, optional): Валидационные метки
            params (dict, optional): Гиперпараметры модели
            num_rounds (int): Максимальное число раундов обучения
            early_stopping_rounds (int): Число раундов для раннего останова
            
        Returns:
            xgboost.Booster: Обученная модель
        """
        logger.info("Начало обучения модели XGBoost")
        
        # Обновляем параметры, если предоставлены
        if params is not None:
            self.params.update(params)
        
        # Преобразуем данные в формат DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        # Если есть валидационные данные, создаем DMatrix для них
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            watchlist = [(dtrain, 'train'), (dval, 'eval')]
        else:
            watchlist = [(dtrain, 'train')]
        
        # Обучение модели
        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=num_rounds,
            evals=watchlist,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=100
        )
        
        logger.info(f"Модель XGBoost обучена за {self.model.best_iteration} итераций")
        
        return self.model
    
    def predict(self, X):
        """
        Предсказание с помощью обученной модели.
        
        Args:
            X (pandas.DataFrame): Данные для предсказания
            
        Returns:
            numpy.ndarray: Предсказания
        """
        if self.model is None:
            logger.error("Модель не обучена")
            return None
        
        # Преобразуем в DMatrix
        dtest = xgb.DMatrix(X)
        
        # Получаем предсказания
        if self.target_type == 'binary':
            # Вероятности принадлежности к положительному классу
            y_pred_proba = self.model.predict(dtest)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
        elif self.target_type == 'regression':
            # Прямые предсказания для регрессии
            y_pred = self.model.predict(dtest)
            
        elif self.target_type == 'classification':
            # Вероятности для каждого класса, берем класс с максимальной вероятностью
            y_pred_proba = self.model.predict(dtest)
            y_pred = np.argmax(y_pred_proba, axis=1)
        
        return y_pred
    
    def predict_proba(self, X):
        """
        Предсказание вероятностей принадлежности к классам.
        
        Args:
            X (pandas.DataFrame): Данные для предсказания
            
        Returns:
            numpy.ndarray: Предсказанные вероятности
        """
        if self.model is None:
            logger.error("Модель не обучена")
            return None
        
        # Преобразуем в DMatrix
        dtest = xgb.DMatrix(X)
        
        # Получаем вероятности
        return self.model.predict(dtest)
    
    def evaluate(self, X, y_true):
        """
        Оценка качества модели.
        
        Args:
            X (pandas.DataFrame): Данные для предсказания
            y_true (pandas.Series): Истинные метки
            
        Returns:
            dict: Метрики качества
        """
        if self.model is None:
            logger.error("Модель не обучена")
            return None
        
        # Получаем предсказания
        y_pred = self.predict(X)
        
        metrics = {}
        
        if self.target_type == 'binary':
            # Метрики для бинарной классификации
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred)
            metrics['recall'] = recall_score(y_true, y_pred)
            metrics['f1'] = f1_score(y_true, y_pred)
            
            # Процент правильного угадывания направления
            metrics['direction_accuracy'] = accuracy_score(y_true, y_pred)
            
            logger.info(f"Метрики XGBoost (бинарная классификация):")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Precision: {metrics['precision']:.4f}")
            logger.info(f"Recall: {metrics['recall']:.4f}")
            logger.info(f"F1: {metrics['f1']:.4f}")
            
        elif self.target_type == 'regression':
            # Метрики для регрессии
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
            
            # Процент правильного угадывания направления
            y_direction_true = np.sign(np.diff(np.append([y_true.iloc[0]], y_true)))
            y_direction_pred = np.sign(np.diff(np.append([y_true.iloc[0]], y_pred)))
            metrics['direction_accuracy'] = np.mean(y_direction_true == y_direction_pred)
            
            logger.info(f"Метрики XGBoost (регрессия):")
            logger.info(f"RMSE: {metrics['rmse']:.4f}")
            logger.info(f"MAE: {metrics['mae']:.4f}")
            logger.info(f"R²: {metrics['r2']:.4f}")
            logger.info(f"Direction Accuracy: {metrics['direction_accuracy']:.4f}")
            
        elif self.target_type == 'classification':
            # Метрики для мультиклассовой классификации
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            
            logger.info(f"Метрики XGBoost (мультиклассовая классификация):")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info("\nКлассификационный отчет:")
            logger.info(classification_report(y_true, y_pred))
        
        return metrics
    
    def save_model(self, filename=None):
        """
        Сохранение модели.
        
        Args:
            filename (str, optional): Имя файла для сохранения модели
            
        Returns:
            str: Путь к сохраненной модели
        """
        if self.model is None:
            logger.error("Нет модели для сохранения")
            return None
        
        if filename is None:
            # Создаем имя файла с текущей датой и типом модели
            today = datetime.now().strftime('%Y%m%d')
            filename = f"xgboost_{self.target_type}_{today}.json"
        
        file_path = os.path.join(self.model_dir, filename)
        
        # Сохраняем модель в JSON формате
        self.model.save_model(file_path)
        logger.info(f"Модель XGBoost сохранена в {file_path}")
        
        # Сохраняем параметры модели
        params_file = os.path.join(self.model_dir, f"{os.path.splitext(filename)[0]}_params.joblib")
        joblib.dump(self.params, params_file)
        
        return file_path
    
    def load_model(self, filename):
        """
        Загрузка модели из файла.
        
        Args:
            filename (str): Имя файла с сохраненной моделью
            
        Returns:
            xgboost.Booster: Загруженная модель
        """
        file_path = os.path.join(self.model_dir, filename)
        
        if not os.path.exists(file_path):
            logger.error(f"Файл модели {file_path} не найден")
            return None
        
        try:
            self.model = xgb.Booster()
            self.model.load_model(file_path)
            
            # Пытаемся загрузить параметры модели
            params_file = os.path.join(self.model_dir, f"{os.path.splitext(filename)[0]}_params.joblib")
            if os.path.exists(params_file):
                self.params = joblib.load(params_file)
            
            logger.info(f"Модель XGBoost загружена из {file_path}")
            return self.model
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            return None


class LSTMModel:
    """Класс для работы с моделью LSTM для прогнозирования временных рядов."""
    
    def __init__(self, model_dir="../models", target_type='binary', sequence_length=10):
        """
        Инициализация модели LSTM.
        
        Args:
            model_dir (str): Директория для сохранения моделей
            target_type (str): Тип задачи ('binary', 'regression', 'classification')
            sequence_length (int): Длина последовательности (окна)
        """
        self.model_dir = model_dir
        self.target_type = target_type
        self.sequence_length = sequence_length
        self.model = None
        
        # Создание директории для моделей, если она не существует
        os.makedirs(model_dir, exist_ok=True)
    
    def build_model(self, input_shape, lstm_units=64, dropout_rate=0.2):
        """
        Создание архитектуры LSTM модели.
        
        Args:
            input_shape (tuple): Форма входных данных (sequence_length, n_features)
            lstm_units (int): Количество LSTM-блоков
            dropout_rate (float): Вероятность отключения нейронов для Dropout
            
        Returns:
            tensorflow.keras.models.Sequential: Созданная модель
        """
        logger.info(f"Создание LSTM модели для {self.target_type} с input_shape={input_shape}")
        
        model = Sequential()
        
        # Входной LSTM слой с возвратом последовательностей
        model.add(LSTM(
            units=lstm_units,
            return_sequences=True,
            input_shape=input_shape
        ))
        model.add(Dropout(dropout_rate))
        model.add(BatchNormalization())
        
        # Средний LSTM слой
        model.add(LSTM(
            units=lstm_units,
            return_sequences=True
        ))
        model.add(Dropout(dropout_rate))
        model.add(BatchNormalization())
        
        # Последний LSTM слой
        model.add(LSTM(
            units=lstm_units,
            return_sequences=False
        ))
        model.add(Dropout(dropout_rate))
        model.add(BatchNormalization())
        
        # Выходной слой в зависимости от типа задачи
        if self.target_type == 'binary':
            model.add(Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        elif self.target_type == 'regression':
            model.add(Dense(1, activation='linear'))
            loss = 'mse'
            metrics = ['mae']
        elif self.target_type == 'classification':
            model.add(Dense(5, activation='softmax'))  # 5 классов
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        else:
            raise ValueError(f"Неизвестный тип задачи: {self.target_type}")
        
        # Компиляция модели
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=loss,
            metrics=metrics
        )
        
        # Отображение архитектуры модели
        model.summary()
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """
        Обучение LSTM модели.
        
        Args:
            X_train (numpy.ndarray): Обучающие последовательности формы (n_samples, sequence_length, n_features)
            y_train (numpy.ndarray): Обучающие метки
            X_val (numpy.ndarray, optional): Валидационные последовательности
            y_val (numpy.ndarray, optional): Валидационные метки
            epochs (int): Количество эпох обучения
            batch_size (int): Размер батча
            
        Returns:
            tensorflow.keras.models.Sequential: Обученная модель
        """
        if self.model is None:
            # Создаем модель, если она еще не создана
            input_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, n_features)
            self.build_model(input_shape)
        
        # Создаем колбэки для обучения
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Если указан путь для сохранения чекпоинтов
        # Keras >=3 требует расширение .keras для чекпоинтов
        checkpoint_path = os.path.join(self.model_dir, 'lstm_checkpoint.keras')
        callbacks.append(
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                save_best_only=True
            )
        )
        
        # Подготавливаем валидационные данные
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Обучение модели
        logger.info(f"Начало обучения LSTM модели на {len(X_train)} примерах")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        # Загружаем лучшие веса, если они были сохранены
        if os.path.exists(checkpoint_path):
            self.model.load_weights(checkpoint_path)
        
        logger.info("Обучение LSTM модели завершено")
        
        return self.model, history
    
    def predict(self, X):
        """
        Предсказание с помощью обученной LSTM модели.

        Args:
            X (numpy.ndarray): Данные для предсказания формы (n_samples, sequence_length, n_features)

        Returns:
            numpy.ndarray: Предсказания
        """
        logger = logging.getLogger("models")
        logger.info(f"LSTMModel.predict: X shape = {X.shape}")
        if self.model is None:
            logger.error("Модель не обучена")
            return None
        predictions = self.model.predict(X)
        logger.info(f"LSTMModel.predict: predictions shape = {predictions.shape}")
        if self.target_type == 'binary':
            result = (predictions > 0.5).astype(int).flatten()
            logger.info(f"LSTMModel.predict: result shape = {result.shape}")
            return result
        elif self.target_type == 'regression':
            result = predictions.flatten()
            logger.info(f"LSTMModel.predict: result shape = {result.shape}")
            return result
        elif self.target_type == 'classification':
            result = np.argmax(predictions, axis=1)
            logger.info(f"LSTMModel.predict: result shape = {result.shape}")
            return result
        logger.info(f"LSTMModel.predict: predictions shape = {predictions.shape}")
        return predictions
    
    def predict_proba(self, X):
        """
        Предсказание вероятностей принадлежности к классам.
        
        Args:
            X (numpy.ndarray): Данные для предсказания
            
        Returns:
            numpy.ndarray: Предсказанные вероятности
        """
        if self.model is None:
            logger.error("Модель не обучена")
            return None
        
        # Получаем вероятности
        return self.model.predict(X)
    
    def evaluate(self, X, y_true):
        """
        Оценка качества LSTM модели.
        
        Args:
            X (numpy.ndarray): Данные для предсказания
            y_true (numpy.ndarray): Истинные метки
            
        Returns:
            dict: Метрики качества
        """
        if self.model is None:
            logger.error("Модель не обучена")
            return None
        
        # Получаем предсказания
        y_pred = self.predict(X)
        
        metrics = {}
        
        if self.target_type == 'binary':
            # Метрики для бинарной классификации
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred)
            metrics['recall'] = recall_score(y_true, y_pred)
            metrics['f1'] = f1_score(y_true, y_pred)
            
            # Процент правильного угадывания направления
            metrics['direction_accuracy'] = accuracy_score(y_true, y_pred)
            
            logger.info(f"Метрики LSTM (бинарная классификация):")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Precision: {metrics['precision']:.4f}")
            logger.info(f"Recall: {metrics['recall']:.4f}")
            logger.info(f"F1: {metrics['f1']:.4f}")
            
        elif self.target_type == 'regression':
            # Метрики для регрессии
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
            
            # Процент правильного угадывания направления
            y_direction_true = np.sign(np.diff(np.append([y_true[0]], y_true)))
            y_direction_pred = np.sign(np.diff(np.append([y_true[0]], y_pred)))
            metrics['direction_accuracy'] = np.mean(y_direction_true == y_direction_pred)
            
            logger.info(f"Метрики LSTM (регрессия):")
            logger.info(f"RMSE: {metrics['rmse']:.4f}")
            logger.info(f"MAE: {metrics['mae']:.4f}")
            logger.info(f"R²: {metrics['r2']:.4f}")
            logger.info(f"Direction Accuracy: {metrics['direction_accuracy']:.4f}")
            
        elif self.target_type == 'classification':
            # Метрики для мультиклассовой классификации
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            
            logger.info(f"Метрики LSTM (мультиклассовая классификация):")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info("\nКлассификационный отчет:")
            logger.info(classification_report(y_true, y_pred))
        
        return metrics
    
    def save_model(self, filename=None):
        """
        Сохранение LSTM модели.
        
        Args:
            filename (str, optional): Имя файла для сохранения модели
            
        Returns:
            str: Путь к сохраненной модели
        """
        if self.model is None:
            logger.error("Нет модели для сохранения")
            return None
        
        if filename is None:
            # Создаем имя файла с текущей датой и типом модели
            today = datetime.now().strftime('%Y%m%d')
            filename = f"lstm_{self.target_type}_{today}.h5"
        
        file_path = os.path.join(self.model_dir, filename)
        
        # Сохраняем модель в h5 формате
        self.model.save(file_path)
        logger.info(f"Модель LSTM сохранена в {file_path}")
        
        # Сохраняем параметры модели
        params = {
            'target_type': self.target_type,
            'sequence_length': self.sequence_length
        }
        params_file = os.path.join(self.model_dir, f"{os.path.splitext(filename)[0]}_params.joblib")
        joblib.dump(params, params_file)
        
        return file_path
    
    def load_model(self, filename):
        """
        Загрузка LSTM модели из файла.
        
        Args:
            filename (str): Имя файла с сохраненной моделью
            
        Returns:
            tensorflow.keras.models.Sequential: Загруженная модель
        """
        file_path = os.path.join(self.model_dir, filename)
        
        if not os.path.exists(file_path):
            logger.error(f"Файл модели {file_path} не найден")
            return None
        
        try:
            # Загружаем модель
            self.model = load_model(file_path)
            
            # Пытаемся загрузить параметры модели
            params_file = os.path.join(self.model_dir, f"{os.path.splitext(filename)[0]}_params.joblib")
            if os.path.exists(params_file):
                params = joblib.load(params_file)
                self.target_type = params.get('target_type', self.target_type)
                self.sequence_length = params.get('sequence_length', self.sequence_length)
            
            logger.info(f"Модель LSTM загружена из {file_path}")
            return self.model
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            return None


class EnsembleModel:
    """Класс для ансамблирования моделей XGBoost и LSTM."""
    
    def __init__(self, model_dir="../models", target_type='binary'):
        """
        Инициализация ансамбля моделей.
        
        Args:
            model_dir (str): Директория для сохранения моделей
            target_type (str): Тип задачи ('binary', 'regression', 'classification')
        """
        self.model_dir = model_dir
        self.target_type = target_type
        self.models = {}
        self.weights = {}
        
        # Создание директории для моделей, если она не существует
        os.makedirs(model_dir, exist_ok=True)
    
    def add_model(self, model_name, model, weight=1.0):
        """
        Добавление модели в ансамбль.
        
        Args:
            model_name (str): Уникальное имя модели
            model: Модель с методом predict (и predict_proba для классификации)
            weight (float): Вес модели в ансамбле
        """
        self.models[model_name] = model
        self.weights[model_name] = weight
        logger.info(f"Модель {model_name} добавлена в ансамбль с весом {weight}")
    
    def predict(self, X, X_sequences=None):
        """
        Предсказание с помощью ансамбля моделей.
        """
        logger = logging.getLogger("models")
        logger.info(f"EnsembleModel.predict: X shape = {getattr(X, 'shape', None)}, X_sequences shape = {getattr(X_sequences, 'shape', None)}")
        if not self.models:
            logger.error("Ансамбль не содержит моделей")
            return None
        predictions = {}
        for name, model in self.models.items():
            if name == 'lstm' and X_sequences is not None:
                pred = model.predict(X_sequences)
            else:
                pred = model.predict(X)
            logger.info(f"EnsembleModel.predict: {name} pred shape = {getattr(pred, 'shape', None)}")
            predictions[name] = pred * self.weights[name]
        # --- ВЫРАВНИВАНИЕ длин предсказаний ---
        min_len = min(pred.shape[0] for pred in predictions.values())
        for name in predictions:
            if predictions[name].shape[0] > min_len:
                logger.warning(f"EnsembleModel.predict: {name} pred trimmed from {predictions[name].shape[0]} to {min_len}")
                predictions[name] = predictions[name][:min_len]
        # --------------------------------------
        weights_sum = sum(self.weights.values())
        if self.target_type == 'binary' or self.target_type == 'regression':
            ensemble_pred = sum(pred for pred in predictions.values()) / weights_sum
            if isinstance(ensemble_pred, np.ndarray) and ensemble_pred.ndim > 1:
                ensemble_pred = ensemble_pred.flatten()
            logger.info(f"EnsembleModel.predict: ensemble_pred shape = {ensemble_pred.shape}")
        elif self.target_type == 'classification':
            ensemble_pred_proba = sum(pred for pred in predictions.values()) / weights_sum
            ensemble_pred = np.argmax(ensemble_pred_proba, axis=1)
            logger.info(f"EnsembleModel.predict: ensemble_pred shape = {ensemble_pred.shape}")
        if self.target_type == 'binary':
            # Взвешенная сумма вероятностей
            ensemble_pred_proba = sum(pred for pred in predictions.values()) / weights_sum
            ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)
            
            if isinstance(ensemble_pred, np.ndarray) and ensemble_pred.ndim > 1:
                ensemble_pred = ensemble_pred.flatten()
            
        elif self.target_type == 'regression':
            # Взвешенная сумма предсказаний
            ensemble_pred = sum(pred for pred in predictions.values()) / weights_sum
            
            if isinstance(ensemble_pred, np.ndarray) and ensemble_pred.ndim > 1:
                ensemble_pred = ensemble_pred.flatten()
            
        elif self.target_type == 'classification':
            # Взвешенная сумма вероятностей по классам
            ensemble_pred_proba = sum(pred for pred in predictions.values()) / weights_sum
            ensemble_pred = np.argmax(ensemble_pred_proba, axis=1)
        
        return ensemble_pred
    
    def predict_proba(self, X, X_sequences=None):
        """
        Предсказание вероятностей ансамблем моделей.
        
        Args:
            X (pandas.DataFrame): Данные для предсказания табличных моделей
            X_sequences (numpy.ndarray, optional): Последовательности для LSTM моделей
            
        Returns:
            numpy.ndarray: Предсказанные вероятности
        """
        if not self.models:
            logger.error("Ансамбль не содержит моделей")
            return None
        
        predictions = {}
        weights_sum = 0.0
        
        for model_name, model in self.models.items():
            weight = self.weights[model_name]
            weights_sum += weight
            
            # Выбираем соответствующие данные в зависимости от типа модели
            if isinstance(model, LSTMModel) and X_sequences is not None:
                predictions[model_name] = model.predict_proba(X_sequences) * weight
            elif isinstance(model, XGBoostModel):
                predictions[model_name] = model.predict_proba(X) * weight
            else:
                # Предполагаем, что модель поддерживает метод predict_proba
                try:
                    predictions[model_name] = model.predict_proba(X) * weight
                except Exception as e:
                    logger.error(f"Ошибка предсказания вероятностей модели {model_name}: {e}")
                    # Уменьшаем сумму весов, если модель не смогла сделать предсказание
                    weights_sum -= weight
        
        if weights_sum == 0:
            logger.error("Не удалось получить предсказания ни от одной модели")
            return None
        
        # Объединяем предсказания вероятностей
        ensemble_pred_proba = sum(pred for pred in predictions.values()) / weights_sum
        
        return ensemble_pred_proba
    
    def evaluate(self, X, y_true, X_sequences=None):
        """
        Оценка качества ансамбля моделей.

        Args:
            X (pandas.DataFrame): Данные для предсказания табличных моделей
            y_true (pandas.Series or numpy.ndarray): Истинные метки
            X_sequences (numpy.ndarray, optional): Последовательности для LSTM моделей

        Returns:
            dict: Метрики качества
        """
        logger = logging.getLogger("models")
        y_pred = self.predict(X, X_sequences)
        logger.info(f"EnsembleModel.evaluate: y_true shape = {getattr(y_true, 'shape', None)}, y_pred shape = {getattr(y_pred, 'shape', None)}")
        # --- ВЫРАВНИВАНИЕ длин y_true и y_pred ---
        if len(y_true) > len(y_pred):
            logger.warning(f"EnsembleModel.evaluate: y_true trimmed from {len(y_true)} to {len(y_pred)}")
            y_true = y_true[:len(y_pred)]
        elif len(y_pred) > len(y_true):
            logger.warning(f"EnsembleModel.evaluate: y_pred trimmed from {len(y_pred)} to {len(y_true)}")
            y_pred = y_pred[:len(y_true)]
        assert len(y_true) == len(y_pred), f"Размерности не совпадают: y_true={len(y_true)}, y_pred={len(y_pred)}"
        
        if not self.models:
            logger.error("Ансамбль не содержит моделей")
            return None
        
        # Получаем предсказания ансамбля
        y_pred = self.predict(X, X_sequences)
        
        if y_pred is None:
            return None
        
        metrics = {}
        
        if self.target_type == 'binary':
            # Метрики для бинарной классификации
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred)
            metrics['recall'] = recall_score(y_true, y_pred)
            metrics['f1'] = f1_score(y_true, y_pred)
            
            # Процент правильного угадывания направления
            metrics['direction_accuracy'] = accuracy_score(y_true, y_pred)
            
            logger.info(f"Метрики ансамбля (бинарная классификация):")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Precision: {metrics['precision']:.4f}")
            logger.info(f"Recall: {metrics['recall']:.4f}")
            logger.info(f"F1: {metrics['f1']:.4f}")
            
        elif self.target_type == 'regression':
            # Метрики для регрессии
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
            
            # Процент правильного угадывания направления
            if isinstance(y_true, pd.Series):
                y_true = y_true.values
            
            y_direction_true = np.sign(np.diff(np.append([y_true[0]], y_true)))
            y_direction_pred = np.sign(np.diff(np.append([y_true[0]], y_pred)))
            metrics['direction_accuracy'] = np.mean(y_direction_true == y_direction_pred)
            
            logger.info(f"Метрики ансамбля (регрессия):")
            logger.info(f"RMSE: {metrics['rmse']:.4f}")
            logger.info(f"MAE: {metrics['mae']:.4f}")
            logger.info(f"R²: {metrics['r2']:.4f}")
            logger.info(f"Direction Accuracy: {metrics['direction_accuracy']:.4f}")
            
        elif self.target_type == 'classification':
            # Метрики для мультиклассовой классификации
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            
            logger.info(f"Метрики ансамбля (мультиклассовая классификация):")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info("\nКлассификационный отчет:")
            logger.info(classification_report(y_true, y_pred))
        
        return metrics
    
    def save_model(self, filename=None):
        """
        Сохранение информации об ансамбле моделей.
        
        Args:
            filename (str, optional): Имя файла для сохранения
            
        Returns:
            str: Путь к сохраненному файлу
        """
        if not self.models:
            logger.error("Ансамбль не содержит моделей для сохранения")
            return None
        
        if filename is None:
            # Создаем имя файла с текущей датой
            today = datetime.now().strftime('%Y%m%d')
            filename = f"ensemble_{self.target_type}_{today}.joblib"
        
        file_path = os.path.join(self.model_dir, filename)
        
        # Сохраняем только веса, т.к. сами модели должны быть сохранены отдельно
        ensemble_info = {
            'weights': self.weights,
            'target_type': self.target_type,
            'model_names': list(self.models.keys())
        }
        
        joblib.dump(ensemble_info, file_path)
        logger.info(f"Информация об ансамбле сохранена в {file_path}")
        
        return file_path
    
    def load_ensemble_info(self, filename):
        """
        Загрузка информации об ансамбле моделей.
        
        Args:
            filename (str): Имя файла с сохраненной информацией
            
        Returns:
            dict: Информация об ансамбле
        """
        file_path = os.path.join(self.model_dir, filename)
        
        if not os.path.exists(file_path):
            logger.error(f"Файл информации об ансамбле {file_path} не найден")
            return None
        
        try:
            ensemble_info = joblib.load(file_path)
            
            self.weights = ensemble_info.get('weights', {})
            self.target_type = ensemble_info.get('target_type', self.target_type)
            
            logger.info(f"Информация об ансамбле загружена из {file_path}")
            logger.info(f"Требуемые модели: {ensemble_info.get('model_names', [])}")
            
            return ensemble_info
        except Exception as e:
            logger.error(f"Ошибка при загрузке информации об ансамбле: {e}")
            return None


if __name__ == "__main__":
    # Пример использования
    import numpy as np
    
    # Создаем синтетические данные для теста
    np.random.seed(42)
    X_train = np.random.rand(100, 10)  # 100 примеров, 10 признаков
    y_train = np.random.randint(0, 2, 100)  # Бинарные метки
    
    # Тестируем XGBoost
    print("\nТестирование XGBoost...")
    xgb_model = XGBoostModel(target_type='binary')
    dtrain = xgb.DMatrix(X_train, label=y_train)
    xgb_model.model = xgb.train(xgb_model.params, dtrain, num_boost_round=10)
    
    y_pred_xgb = xgb_model.predict(X_train)
    print(f"Точность XGBoost: {accuracy_score(y_train, y_pred_xgb):.4f}")
    
    # Тестируем LSTM
    print("\nТестирование LSTM...")
    # Преобразуем данные в формат последовательностей
    X_sequences = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    
    lstm_model = LSTMModel(target_type='binary')
    lstm_model.build_model((1, X_train.shape[1]))
    
    # Обучение модели можно закомментировать, т.к. оно может занять время
    # lstm_model.model.fit(X_sequences, y_train, epochs=5, batch_size=32, verbose=0)
    
    # Тестируем ансамбль
    print("\nТестирование ансамбля моделей...")
    ensemble = EnsembleModel(target_type='binary')
    ensemble.add_model('xgboost', xgb_model, weight=0.7)
    # ensemble.add_model('lstm', lstm_model, weight=0.3)
    
    # Получаем предсказания ансамбля
    # y_pred_ensemble = ensemble.predict(X_train, X_sequences)
    # print(f"Точность ансамбля: {accuracy_score(y_train, y_pred_ensemble):.4f}")
