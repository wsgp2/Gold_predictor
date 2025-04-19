#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для генерации прогнозов цены золота и отправки их в Telegram.
"""
import os
import logging
import numpy as np
import pandas as pd
import argparse
import json
import asyncio
from datetime import datetime, timedelta
import joblib

# Для Telegram бота
import telegram
from telegram.ext import Updater, CommandHandler

# Импорт необходимых модулей из проекта
from data_loader import GoldDataLoader
from features import FeatureGenerator
from models import XGBoostModel, LSTMModel, EnsembleModel

# Импорт трекера предсказаний (опционально)
try:
    from prediction_tracker import PredictionTracker
    HAS_PREDICTION_TRACKER = True
except ImportError:
    HAS_PREDICTION_TRACKER = False

# Настройка логирования
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'predict.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GoldPredictor:
    """Класс для генерации прогнозов цены золота."""
    
    def __init__(self, model_dir="../models", data_dir="../data", config_path="../config/predictor_config.json", use_tracker=True):
        """
        Инициализация предсказателя.
        Args:
            model_dir (str): Директория с сохраненными моделями
            data_dir (str): Директория для сохранения данных
            config_path (str): Путь к конфигурационному файлу
            use_tracker (bool): Использовать ли трекер предсказаний
        """
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.config_path = config_path
        self.args = None  # Для хранения аргументов CLI
        
        # Создаем необходимые директории
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Загружаем конфигурацию
        self.config = self.load_config()
        
        # Инициализируем компоненты
        self.data_loader = GoldDataLoader(data_dir=data_dir)
        self.feature_generator = FeatureGenerator(scaling_method='standard')
        self.xgb_model = None
        self.lstm_model = None
        self.ensemble = None
        
        # Инициализируем трекер предсказаний, если он доступен
        self.tracker = None
        if use_tracker and HAS_PREDICTION_TRACKER:
            try:
                self.tracker = PredictionTracker()
                logger.info("Трекер предсказаний инициализирован успешно")
            except Exception as e:
                logger.error(f"Ошибка при инициализации трекера предсказаний: {e}")
        
        # Загружаем модели
        self.load_models()

    def load_config(self):
        """
        Загрузка конфигурации из файла.
        Returns:
            dict: Конфигурация
        """
        default_config = {
            "target_type": "binary",
            "horizon": 1,
            "sequence_length": 10,
            "xgb_model_path": "",
            "lstm_model_path": "",
            "ensemble_info_path": "",
            "telegram_token": "",
            "telegram_chat_id": "",
            "prediction_time": "10:00",
            "verification_time": "10:00"
        }
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    default_config.update(config)
                logger.info(f"Конфигурация загружена из {self.config_path}")
            except Exception as e:
                logger.error(f"Ошибка при загрузке конфигурации: {e}")
        else:
            self.save_config(default_config)
            logger.info(f"Создана конфигурация по умолчанию в {self.config_path}")
        return default_config

    def save_config(self, config=None):
        """
        Сохранение конфигурации в файл.
        Args:
            config (dict, optional): Конфигурация для сохранения
        """
        if config is None:
            config = self.config
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)
            logger.info(f"Конфигурация сохранена в {self.config_path}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении конфигурации: {e}")

    def update_config(self, **kwargs):
        """
        Обновление конфигурации.
        Args:
            **kwargs: Параметры для обновления
            
        Returns:
            bool: True если конфигурация успешно обновлена
        """
        try:
            # Обрабатываем специфические параметры
            if 'model_type' in kwargs:
                model_type = kwargs['model_type'].lower()
                if model_type not in ['xgboost', 'lstm', 'ensemble']:
                    logger.warning(f"Неверный тип модели: {model_type}. Используем 'ensemble'.")
                    kwargs['model_type'] = 'ensemble'
            
            if 'horizon' in kwargs:
                horizon = int(kwargs['horizon'])
                if horizon < 1:
                    logger.warning(f"Неверный горизонт: {horizon}. Используем 1.")
                    kwargs['horizon'] = 1
            
            if 'target_type' in kwargs:
                target_type = kwargs['target_type'].lower()
                if target_type not in ['binary', 'classification']:
                    logger.warning(f"Неверный тип цели: {target_type}. Используем 'binary'.")
                    kwargs['target_type'] = 'binary'
            
            # Обновляем конфигурацию
            self.config.update(kwargs)
            self.save_config()
            logger.info(f"Конфигурация обновлена: {kwargs}")
            return True
        except Exception as e:
            logger.error(f"Ошибка при обновлении конфигурации: {e}")
            return False

    def load_models(self):
        """
        Загрузка моделей.
        Returns:
            bool: True, если хотя бы одна модель загружена успешно
        """
        success = False
        if self.config["xgb_model_path"]:
            try:
                # Проверяем существует ли файл по указанному пути
                xgb_file_path = self.config["xgb_model_path"]
                if not os.path.isabs(xgb_file_path):
                    # Если путь относительный, добавляем путь к моделям
                    xgb_file_path = os.path.join(self.model_dir, os.path.basename(xgb_file_path))
                
                if os.path.exists(xgb_file_path):
                    self.xgb_model = XGBoostModel(target_type=self.config["target_type"])
                    self.xgb_model.load_model(xgb_file_path)
                    logger.info(f"XGBoost модель загружена из {xgb_file_path}")
                    success = True
                else:
                    logger.error(f"Файл модели {xgb_file_path} не найден")
            except Exception as e:
                logger.error(f"Ошибка при загрузке XGBoost модели: {e}")
        
        if self.config["lstm_model_path"]:
            try:
                # Проверяем существует ли файл по указанному пути
                lstm_file_path = self.config["lstm_model_path"]
                if not os.path.isabs(lstm_file_path):
                    # Если путь относительный, добавляем путь к моделям
                    lstm_file_path = os.path.join(self.model_dir, os.path.basename(lstm_file_path))
                
                if os.path.exists(lstm_file_path):
                    self.lstm_model = LSTMModel(
                        target_type=self.config["target_type"],
                        sequence_length=self.config["sequence_length"]
                    )
                    self.lstm_model.load_model(lstm_file_path)
                    logger.info(f"LSTM модель загружена из {lstm_file_path}")
                    success = True
                else:
                    logger.error(f"Файл модели {lstm_file_path} не найден")
            except Exception as e:
                logger.error(f"Ошибка при загрузке LSTM модели: {e}")
        
        if self.config["ensemble_info_path"]:
            try:
                # Проверяем существует ли файл по указанному пути
                ensemble_file_path = self.config["ensemble_info_path"]
                if not os.path.isabs(ensemble_file_path):
                    # Если путь относительный, добавляем путь к моделям
                    ensemble_file_path = os.path.join(self.model_dir, os.path.basename(ensemble_file_path))
                
                if os.path.exists(ensemble_file_path):
                    self.ensemble = EnsembleModel(target_type=self.config["target_type"])
                    ensemble_info = self.ensemble.load_ensemble_info(ensemble_file_path)
                    
                    if ensemble_info and self.xgb_model and "xgboost" in ensemble_info["model_names"]:
                        self.ensemble.add_model("xgboost", self.xgb_model, weight=ensemble_info["weights"].get("xgboost", 1.0))
                    
                    if ensemble_info and self.lstm_model and "lstm" in ensemble_info["model_names"]:
                        self.ensemble.add_model("lstm", self.lstm_model, weight=ensemble_info["weights"].get("lstm", 1.0))
                    
                    logger.info(f"Информация об ансамбле загружена из {ensemble_file_path}")
                    success = True
                else:
                    logger.error(f"Файл информации об ансамбле {ensemble_file_path} не найден")
            except Exception as e:
                logger.error(f"Ошибка при загрузке информации об ансамбле: {e}")
        
        return success
        
    def prepare_latest_data(self):
        """
        Подготовка последних данных для прогнозирования.
        Returns:
            dict: Словарь с последними данными или None в случае ошибки
        """
        try:
            # Загружаем последние данные, берем больше данных для надежного расчета индикаторов
            latest_data = self.data_loader.get_latest_data(days=300)  # Увеличиваем исторический период
            
            if latest_data is None or len(latest_data) < 100:
                logger.error("Недостаточно данных для прогнозирования")
                return None
            
            # Получаем последнюю известную цену закрытия
            last_close = latest_data['Close'].iloc[-1]
            last_date = latest_data.index[-1]
            
            logger.info(f"Получены данные с {latest_data.index[0]} по {last_date}, всего {len(latest_data)} записей")
            
            # Создаем индикаторы, не удаляя строки с пропущенными значениями
            data = latest_data.copy()
            data_with_indicators = self.feature_generator.create_technical_indicators(data)
            
            # Заполняем пропуски прямо здесь, вместо удаления строк
            data_with_indicators = data_with_indicators.replace([np.inf, -np.inf], np.nan)
            data_with_indicators = data_with_indicators.ffill()  # Используем ffill() вместо fillna(method='ffill')
            data_with_indicators = data_with_indicators.bfill()  # Используем bfill() вместо fillna(method='bfill')
            data_with_indicators = data_with_indicators.fillna(0)  # Если всё ещё есть пропуски
            
            # Очищаем названия столбцов для совместимости с моделями
            logger.info(f"Оригинальные столбцы: {data_with_indicators.columns.tolist()[:5]}")
            
            # Проверяем, что столбцы представлены в виде кортежей (tuples)
            if isinstance(data_with_indicators.columns[0], tuple):
                # Преобразуем MultiIndex в обычные столбцы, используя только первый элемент кортежа
                logger.info("Обнаружен MultiIndex, преобразуем в обычные столбцы")
                
                # Создаем новый DataFrame с простыми именами столбцов
                new_columns = {}
                for col in data_with_indicators.columns:
                    # Берем только первый элемент кортежа (например, 'Close' из ('Close', 'GC=F'))
                    new_columns[col] = col[0]
                
                # Переименовываем столбцы
                data_with_indicators = data_with_indicators.rename(columns=new_columns)
                
                # Убираем MultiIndex и делаем обычный Index
                data_with_indicators.columns = list(new_columns.values())
            else:
                # Если столбцы уже в формате строк, очищаем суффиксы и пробелы
                renamed_columns = {}
                for col in data_with_indicators.columns:
                    # Удаляем суффиксы вида 'GC=F'
                    new_col = col.split(' ')[0] if ' ' in col else col
                    # Удаляем пробелы в конце имени
                    new_col = new_col.strip()
                    renamed_columns[col] = new_col
                
                # Переименовываем столбцы
                data_with_indicators = data_with_indicators.rename(columns=renamed_columns)
            
            logger.info(f"Очищенные столбцы: {data_with_indicators.columns.tolist()[:5]}")
            
            # Берем только последние 50 записей для создания признаков
            last_n_rows = min(50, len(data_with_indicators))
            data_for_features = data_with_indicators.iloc[-last_n_rows:].copy()
            
            # Добавляем недостающие признаки, которые ожидают модели
            expected_features = [
                'Close', 'High', 'Low', 'Open', 'Volume', 'MA_5', 'MA_ratio_5', 'MA_10', 'MA_ratio_10',
                'MA_20', 'MA_ratio_20', 'MA_50', 'MA_ratio_50', 'MA_100', 'MA_ratio_100', 'EMA_5', 'EMA_ratio_5',
                'EMA_10', 'EMA_ratio_10', 'EMA_20', 'EMA_ratio_20', 'EMA_50', 'EMA_ratio_50', 'EMA_100',
                'EMA_ratio_100', 'RSI_7', 'RSI_14', 'RSI_21', 'MACD_line', 'MACD_signal', 'MACD_histogram',
                'BB_upper_20', 'BB_lower_20', 'BB_width_20', 'BB_position_20', 'Stoch_%K_14', 'Stoch_%D_14',
                'ATR_14', 'CCI_20', 'Price_Change', 'Return', 'Volatility_5', 'Volatility_10', 'Volatility_21',
                'High_Low_Range', 'High_Low_Range_Pct', 'Volume_MA_5', 'Volume_ratio_5', 'Volume_MA_10',
                'Volume_ratio_10', 'Volume_MA_20', 'Volume_ratio_20', 'Volume_Price'
            ]
            
            # Добавляем мэппинг имен признаков
            # Подготавливаем последнюю строку для XGBoost
            last_features = numeric_features.iloc[-1:].drop(['Target', 'Future_Close'], axis=1, errors='ignore')
            
            # Для LSTM нам нужна последовательность
            sequence_length = self.config["sequence_length"]
            
            # Берем последние sequence_length строк для создания последовательности
            if len(numeric_features) >= sequence_length:
                # Берем последние sequence_length строк
                seq_features = numeric_features.iloc[-sequence_length:].drop(['Target'], axis=1, errors='ignore')
                
                # Убедимся, что Future_Close присутствует (необходим для обеих моделей)
                if 'Future_Close' not in seq_features.columns:
                    seq_features['Future_Close'] = seq_features['Close']
                
                # Создаем последовательность для LSTM
                last_sequence = seq_features.values.reshape(1, sequence_length, seq_features.shape[1])
                logger.info(f"Создана последовательность для LSTM размером {last_sequence.shape}")
            else:
                logger.warning(f"Недостаточно данных для создания последовательности длиной {sequence_length}")
                last_sequence = None
            
            return {
                'last_close': last_close,
                'last_date': last_date,
                'last_features': last_features,
                'last_sequence': last_sequence
            }
        except Exception as e:
            logger.error(f"Ошибка при подготовке данных: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
            
    def _reorder_features_for_xgboost(self, features_df):
        """
        Переупорядочивает признаки в соответствии с ожиданиями модели XGBoost
        
        Args:
            features_df (pd.DataFrame): Датафрейм с признаками
            
        Returns:
            pd.DataFrame: Датафрейм с признаками в нужном порядке
        """
        # Создаем копию, чтобы не изменять оригинальный датафрейм
        features_copy = features_df.copy()
        
        # Полный список всех признаков, которые ожидает модель XGBoost
        # Необходимо сохранить порядок признаков точно как при обучении
        expected_features = [
            'Close', 'High', 'Low', 'Open', 'Volume', 'MA_5', 'MA_ratio_5', 'MA_10', 'MA_ratio_10',
            'MA_20', 'MA_ratio_20', 'MA_50', 'MA_ratio_50', 'MA_100', 'MA_ratio_100', 'EMA_5', 'EMA_ratio_5',
            'EMA_10', 'EMA_ratio_10', 'EMA_20', 'EMA_ratio_20', 'EMA_50', 'EMA_ratio_50', 'EMA_100',
            'EMA_ratio_100', 'RSI_7', 'RSI_14', 'RSI_21', 'MACD_line', 'MACD_signal', 'MACD_histogram',
            'BB_upper_20', 'BB_lower_20', 'BB_width_20', 'BB_position_20', 'Stoch_%K_14', 'Stoch_%D_14',
            'ATR_14', 'CCI_20', 'Price_Change', 'Return', 'Volatility_5', 'Volatility_10', 'Volatility_21',
            'High_Low_Range', 'High_Low_Range_Pct', 'Volume_MA_5', 'Volume_ratio_5', 'Volume_MA_10',
            'Volume_ratio_10', 'Volume_MA_20', 'Volume_ratio_20', 'Volume_Price'
        ]
        
        # Проверяем наличие всех необходимых признаков и добавляем отсутствующие
        for feature in expected_features:
            if feature not in features_copy.columns:
                logger.warning(f"Признак {feature} отсутствует, добавляем с нулевыми значениями")
                features_copy[feature] = 0.0
        
        # Выводим диагностическую информацию о готовом наборе признаков
        logger.info(f"Признаки для XGBoost (всего {len(expected_features)}): {expected_features[:5]}... и еще {len(expected_features)-5}")
        
        # Возвращаем датафрейм с признаками в правильном порядке
        return features_copy[expected_features]
        
    def _diagnostic_features(self, features_df):
        """
        Вывод информации о признаках для диагностики
        
        Args:
            features_df (pd.DataFrame): Датафрейм с признаками
        """
        logger.info(f"Всего признаков: {len(features_df.columns)}")
        logger.info(f"Список признаков: {features_df.columns.tolist()}")
        
        # Пример значений для нескольких признаков
        if not features_df.empty:
            sample = {k: v for k, v in features_df.iloc[-1].to_dict().items() 
                     if k in ['Close', 'RSI_14', 'MA_5', 'MACD_line']}
            logger.info(f"Пример значений: {sample}")
        
        # Проверка наличия NaN/inf
        nan_count = features_df.isna().sum().sum()
        inf_count = np.isinf(features_df.select_dtypes(include=[np.number])).sum().sum()
        logger.info(f"NaN значений: {nan_count}, Inf значений: {inf_count}")
    
    def predict(self):
        """
        Генерация прогноза цены золота на horizon дней вперед.
        Returns:
            dict: Прогноз или None в случае ошибки
        """
        # Проверяем наличие моделей
        if self.xgb_model is None and self.lstm_model is None and self.ensemble is None:
            logger.error("Ни одна модель не загружена")
            return None
        
        # Получаем последние данные
        logger.info(f"Аргументы запуска: {getattr(self, 'args', None)}")
        logger.info(f"Конфиг: {self.config}")
        
        data = self.prepare_latest_data()
        if data is None:
            logger.error("prepare_latest_data вернул None — нет данных для прогноза")
            print("[ERROR] Нет данных для прогноза (prepare_latest_data вернул None)")
            return None
            
        last_close = data['last_close']
        # Если last_close - Series или DataFrame, преобразуем в скаляр
        if isinstance(last_close, pd.Series):
            last_close = last_close.iloc[-1] if not last_close.empty else 0.0
        elif isinstance(last_close, pd.DataFrame):
            last_close = last_close.iloc[-1, 0] if not last_close.empty else 0.0
        elif isinstance(last_close, np.ndarray):
            last_close = float(last_close[-1]) if len(last_close) > 0 else 0.0
        
        last_date = data['last_date']
        last_features = data['last_features']
        last_sequence = data['last_sequence']
        
        # Обрабатываем дату, независимо от формата (с временем или без)
        try:
            if isinstance(last_date, str):
                # Если дата уже строка, извлекаем только дату
                date_part = last_date.split()[0] if ' ' in last_date else last_date
                prediction_date = (datetime.strptime(date_part, "%Y-%m-%d") + timedelta(days=self.config["horizon"])).strftime("%Y-%m-%d")
            else:
                # Если это объект datetime или Timestamp
                prediction_date = (last_date + timedelta(days=self.config["horizon"])).strftime("%Y-%m-%d")
        except Exception as e:
            logger.error(f"Ошибка при обработке даты: {e}")
            prediction_date = datetime.now().strftime("%Y-%m-%d")
            
        predictions = {}

        # Проводим диагностику признаков перед прогнозированием
        self._diagnostic_features(last_features)
        
        # Прогноз XGBoost
        if self.xgb_model is not None:
            try:
                # Преобразуем признаки в формат, ожидаемый XGBoost
                xgb_features = self._reorder_features_for_xgboost(last_features)
                logger.info(f"Признаки для XGBoost подготовлены, количество: {len(xgb_features.columns)}")
                
                xgb_pred = self.xgb_model.predict(xgb_features)
                # Проверяем, что модель вернула результат
                if xgb_pred is not None:
                    # Получаем вероятности
                    try:
                        xgb_proba = self.xgb_model.predict_proba(xgb_features)
                        if xgb_proba is not None:
                            # Бинарная классификация: направление и уверенность
                            if self.config["target_type"] == 'binary':
                                # Для бинарной классификации - просто вероятность положительного класса
                                xgb_direction = "UP" if xgb_pred[0] == 1 else "DOWN"
                                # Берем вероятность в зависимости от формата результата (массив или скаляр)
                                if hasattr(xgb_proba, 'ndim') and xgb_proba.ndim > 1 and xgb_proba.shape[1] > 1:
                                    xgb_confidence = float(xgb_proba[0, 1])
                                else:
                                    # Если возвращается просто вероятность положительного класса
                                    xgb_confidence = float(xgb_proba[0])
                                    
                                predictions['xgboost'] = {
                                    'direction': xgb_direction,
                                    'confidence': xgb_confidence
                                }
                                if hasattr(self, 'args') and getattr(self.args, 'print_proba', False):
                                    print(f"[XGBoost] Прогноз: {xgb_direction}, вероятность: {xgb_confidence:.3f}")
                    except Exception as e:
                        logger.error(f"Ошибка при получении вероятностей XGBoost: {e}")
                else:
                    logger.error("XGBoost модель вернула None вместо предсказаний")
            except Exception as e:
                logger.error(f"Ошибка при прогнозировании с XGBoost: {e}")
        
        # Прогноз LSTM
        if self.lstm_model is not None and last_sequence is not None:
            try:
                lstm_pred = self.lstm_model.predict(last_sequence)
                # Проверяем, что модель вернула результат
                if lstm_pred is not None:
                    # Бинарная классификация
                    if self.config["target_type"] == 'binary':
                        # Проверяем формат предсказания (его размерность)
                        if isinstance(lstm_pred, np.ndarray):
                            if lstm_pred.ndim == 1:  # Уже сплющенный массив
                                lstm_direction = "UP" if lstm_pred[0] > 0.5 else "DOWN"
                                lstm_confidence = float(lstm_pred[0])
                            else:  # Многомерный массив
                                lstm_direction = "UP" if lstm_pred.flatten()[0] > 0.5 else "DOWN"
                                lstm_confidence = float(lstm_pred.flatten()[0])
                                
                            predictions['lstm'] = {
                                'direction': lstm_direction,
                                'confidence': lstm_confidence
                            }
                            if hasattr(self, 'args') and getattr(self.args, 'print_proba', False):
                                print(f"[LSTM] Прогноз: {lstm_direction}, вероятность: {lstm_confidence:.3f}")
                else:
                    logger.error("LSTM модель вернула None вместо предсказаний")
            except Exception as e:
                logger.error(f"Ошибка при прогнозировании с LSTM: {e}")
        
        # Прогноз ансамбля
        if self.ensemble is not None:
            try:
                # Проверяем наличие достаточного количества моделей для ансамбля
                available_models = {model: info for model, info in predictions.items()}
                
                if len(available_models) >= 1:  # Достаточно хотя бы одной модели
                    weights = {}
                    if hasattr(self.ensemble, 'get'):
                        weights = self.ensemble.get('weights', {})
                    elif hasattr(self.ensemble, 'weights'):
                        weights = self.ensemble.weights
                    
                    # По умолчанию все модели имеют одинаковый вес
                    default_weight = 1.0 / len(available_models)
                    
                    # Вычисляем взвешенную вероятность
                    total_weight = 0.0
                    weighted_confidence = 0.0
                    
                    for model_name, model_info in available_models.items():
                        model_weight = weights.get(model_name, default_weight)
                        model_confidence = model_info['confidence']
                        weighted_confidence += model_weight * model_confidence
                        total_weight += model_weight
                    
                    # Нормализуем по общему весу
                    if total_weight > 0:
                        ensemble_confidence = weighted_confidence / total_weight
                    else:
                        # Если веса не заданы, используем среднее арифметическое
                        ensemble_confidence = weighted_confidence / len(available_models)
                    
                    ensemble_direction = "UP" if ensemble_confidence > 0.5 else "DOWN"
                    
                    predictions['ensemble'] = {
                        'direction': ensemble_direction,
                        'confidence': ensemble_confidence
                    }
                    
                    if hasattr(self, 'args') and getattr(self.args, 'print_proba', False):
                        print(f"[Ensemble] Прогноз: {ensemble_direction}, вероятность: {ensemble_confidence:.3f}")
                else:
                    logger.warning("Недостаточно моделей для ансамбля")
            except Exception as e:
                logger.error(f"Ошибка при прогнозировании с ансамблем: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        if not predictions:
            logger.error("Не удалось получить ни одного прогноза")
            print("[ERROR] Прогноз не получен.")
            return None
        
        # Определяем основной прогноз в зависимости от заданной модели
        model_type = getattr(self.args, 'model', 'xgboost')
        if model_type in predictions:
            main_prediction = predictions[model_type]
        else:
            available_models = list(predictions.keys())
            if available_models:
                main_prediction = predictions[available_models[0]]
                logger.warning(f"Модель {model_type} не найдена, используем {available_models[0]}")
            else:
                logger.error("Не удалось получить прогноз ни от одной модели")
                print("[ERROR] Прогноз не получен.")
                return None
        
        # Формируем результат
        direction = main_prediction['direction']
        confidence = main_prediction['confidence']
        
        # Формируем сообщение с использованием Markdown
        # Используем эмодзи и форматирование для наглядности
        emoji_direction = "🔼" if direction == "UP" else "🔽"
        emoji_confidence = "🎯" if confidence > 0.7 else "🔍"
        
        # Определяем модель с эмодзи
        model_emoji = {
            "xgboost": "🌲", # Дерево
            "lstm": "🧠",    # Нейросеть
            "ensemble": "⚖️"  # Весы/ансамбль
        }.get(model_type.lower(), "🔮")
        
        # Красивое форматирование с Markdown
        message = f"*📈 Прогноз цены золота*\n\n"
        message += f"📅 *Дата:* {prediction_date}\n\n"
        message += f"{emoji_direction} *Направление:* {direction}\n"
        message += f"💰 *Текущая цена:* ${last_close:.2f}\n"
        message += f"{emoji_confidence} *Вероятность:* {confidence:.2f}\n\n"
        message += f"{model_emoji} *Модель:* {model_type.upper()}\n\n"
        
        # Добавляем прогнозы отдельных моделей
        message += "*Прогнозы моделей:*\n"
        for model_name, pred in predictions.items():
            model_icon = {
                "xgboost": "🌲",
                "lstm": "🧠",
                "ensemble": "⚖️"
            }.get(model_name.lower(), "🔮")
            direction_icon = "🔼" if pred['direction'] == "UP" else "🔽"
            confidence_value = pred['confidence']
            # Добавляем визуализацию уверенности в виде бара
            confidence_bar = ""
            bar_length = int(confidence_value * 10)
            if pred['direction'] == "UP":
                confidence_bar = "🟩" * bar_length + "⬜️" * (10 - bar_length)
            else:  # DOWN
                confidence_bar = "🟥" * bar_length + "⬜️" * (10 - bar_length)
                
            message += f"{model_icon} *{model_name.capitalize()}:* {direction_icon} {pred['direction']} ({confidence_value:.2f})\n"
            message += f"{confidence_bar}\n"
        
        # Добавляем время прогноза
        message += f"\n🕒 *Дата прогноза:* {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        # Отправляем в Telegram, если требуется
        if hasattr(self.args, 'send_telegram') and self.args.send_telegram:
            self.send_telegram_message(message)
        
        # Формируем итоговый прогноз в расширенном формате
        result = {
            'date': str(last_date),
            'prediction_date': prediction_date,
            'current_price': float(last_close),
            'last_close': float(last_close),   # Добавляем для совместимости
            'last_date': str(last_date),       # Добавляем для совместимости
            'direction': direction,
            'confidence': float(confidence),
            'model': model_type,
            'target_type': self.config.get("target_type", "binary"),
            'horizon': self.config.get("horizon", 1),
            'predictions': predictions,        # Для совместимости со старым кодом
            'all_predictions': predictions,    
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Время формирования прогноза
        }
        
        # Сохраняем предсказание в трекере, если он доступен
        if self.tracker is not None:
            try:
                self.tracker.save_prediction(result)
                logger.info(f"Предсказание сохранено в трекере для даты {prediction_date}")
            except Exception as e:
                logger.error(f"Ошибка при сохранении предсказания в трекере: {e}")
        
        # Отправляем в Telegram, если требуется
        if hasattr(self, 'args') and hasattr(self.args, 'send_telegram') and self.args.send_telegram:
            self._send_prediction_to_telegram(result)
        
        # Выводим сообщение в консоль
        print(f"\n[RESULT] Прогноз на {prediction_date}:")
        print(f"Направление: {direction} (уверенность: {confidence:.3f})")
        print(f"Текущая цена: ${float(last_close):.2f}")
        print(f"Модель: {model_type.upper()}")
        
        return result
        
    def _send_prediction_to_telegram(self, prediction):
        """
        Форматирование и отправка прогноза в Telegram.
        Args:
            prediction (dict): Данные прогноза
        Returns:
            bool: True, если сообщение отправлено успешно
        """
        try:
            # Определяем эмодзи на основе направления и уверенности
            direction = prediction.get('direction', 'UNKNOWN')
            confidence = prediction.get('confidence', 0.0)
            price = prediction.get('current_price', 0.0)
            prediction_date = prediction.get('prediction_date', '')
            model_type = prediction.get('model', 'ensemble')
            
            # Эмодзи для направления движения цены
            direction_emoji = "🔼" if direction == "UP" else "🔽" if direction == "DOWN" else "⏹️"
            
            # Эмодзи для уверенности в прогнозе
            confidence_emoji = "🎯" if confidence > 0.8 else "🔍" if confidence > 0.6 else "❓"
            
            # Эмодзи для модели
            model_emoji = {
                "xgboost": "🌲",  # Дерево для XGBoost
                "lstm": "🧠",     # Мозг для нейросети
                "ensemble": "⚖️"  # Весы для ансамбля
            }.get(model_type.lower(), "🔮")
            
            # Создаем сообщение с Markdown форматированием
            message = f"*📈 Прогноз цены золота*\n\n"
            message += f"📅 *Дата:* {prediction_date}\n"
            message += f"💰 *Текущая цена:* ${price:.2f}\n\n"
            message += f"{direction_emoji} *Направление:* {direction}\n"
            message += f"{confidence_emoji} *Уверенность:* {confidence:.2f}\n\n"
            message += f"{model_emoji} *Модель:* {model_type.upper()}\n\n"
            
            # Добавляем прогнозы отдельных моделей
            all_predictions = prediction.get('all_predictions', {})
            if all_predictions:
                message += "*Прогнозы моделей:*\n"
                for model_name, pred in all_predictions.items():
                    model_icon = {
                        "xgboost": "🌲",
                        "lstm": "🧠",
                        "ensemble": "⚖️"
                    }.get(model_name.lower(), "🔮")
                    pred_direction = pred.get('direction', 'UNKNOWN')
                    pred_emoji = "🔼" if pred_direction == "UP" else "🔽" if pred_direction == "DOWN" else "⏹️"
                    pred_confidence = pred.get('confidence', 0.0)
                    
                    # Визуализация уверенности
                    bar_length = int(pred_confidence * 10)
                    confidence_bar = ""
                    if pred_direction == "UP":
                        confidence_bar = "🟩" * bar_length + "⬜️" * (10 - bar_length)
                    else:  # DOWN
                        confidence_bar = "🟥" * bar_length + "⬜️" * (10 - bar_length)
                    
                    message += f"{model_icon} *{model_name.capitalize()}:* {pred_emoji} {pred_direction} ({pred_confidence:.2f})\n"
                    message += f"{confidence_bar}\n"
            
            # Добавляем время генерации прогноза
            message += f"\n🕒 *Прогноз сгенерирован:* {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            
            # Отправляем сообщение
            return self.send_telegram_message(message)
        except Exception as e:
            logger.error(f"Ошибка при форматировании и отправке прогноза: {e}")
            return False

    def send_telegram_message(self, message):
        """
        Отправка сообщения в Telegram.
        Args:
            message (str): Сообщение для отправки
        Returns:
            bool: True, если сообщение отправлено успешно
        """
        if not self.config["telegram_token"] or not self.config["telegram_chat_id"]:
            logger.error("Не настроен Telegram бот (отсутствует токен или chat_id)")
            return False
        
        try:
            # Используем старую версию API для совместимости с python-telegram-bot 13.x
            bot = telegram.Bot(token=self.config["telegram_token"])
            
            # Несмотря на то, что в новых версиях методы асинхронные, в старых версиях они синхронные
            # Проверяем версию библиотеки
            if hasattr(telegram, '__version__'):
                v = telegram.__version__.split('.')
                if int(v[0]) >= 20:  # В версии 20+ API асинхронное
                    import asyncio
                    # Создаем асинхронную функцию
                    async def send_async():
                        await bot.send_message(chat_id=self.config["telegram_chat_id"], text=message, parse_mode='Markdown')
                    
                    # Запускаем асинхронную функцию в синхронном контексте
                    try:
                        asyncio.run(send_async())
                    except RuntimeError as e:  # Если event loop уже запущен
                        logger.warning(f"RuntimeError при запуске asyncio: {e}")
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            loop.create_task(send_async()) 
                        else:
                            loop.run_until_complete(send_async())
                else:
                    # Для старых версий используем синхронный API
                    bot.send_message(chat_id=self.config["telegram_chat_id"], text=message)
            else:
                # Если версия неизвестна, пробуем синхронный вариант
                bot.send_message(chat_id=self.config["telegram_chat_id"], text=message)
            
            logger.info("Сообщение отправлено в Telegram")
            return True
        except Exception as e:
            logger.error(f"Ошибка при отправке сообщения в Telegram: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False


if __name__ == "__main__":
    # CLI-интерфейс
    parser = argparse.ArgumentParser(description="Gold Price Predictor CLI")
    parser.add_argument('--model', type=str, choices=['xgboost', 'lstm', 'ensemble'], 
                       default='ensemble', help='Модель для прогноза')
    parser.add_argument('--target_type', type=str, choices=['binary', 'classification'], 
                       default='binary', help='Тип целевой переменной')
    parser.add_argument('--horizon', type=int, default=1, help='Горизонт прогноза в днях')
    parser.add_argument('--print_proba', action='store_true', help='Печатать вероятности')
    parser.add_argument('--send_telegram', action='store_true', help='Отправлять результат в Telegram')
    parser.add_argument('--config', type=str, default="../config/predictor_config.json", 
                       help='Путь к конфигу')
    args = parser.parse_args()

    # Создаем предсказатель
    predictor = GoldPredictor(config_path=args.config)
    predictor.args = args
    predictor.config['target_type'] = args.target_type
    predictor.config['horizon'] = args.horizon
    predictor.save_config()

    # Генерируем прогноз
    result = predictor.predict()
    if result is not None:
        print("\n[RESULT] Прогноз:")
        print(f"Последняя цена: {result['last_close']}")
        print(f"Последняя дата: {result['last_date']}")
        print(f"Дата прогноза: {result['prediction_date']}")
        print(f"Горизонт: {result['horizon']} дней")
        print("\nПрогнозы моделей:")
        for model_name, pred in result['predictions'].items():
            print(f"  {model_name}: {pred['direction']} (уверенность: {pred['confidence']:.3f})")
        
        if args.send_telegram:
            msg = f"*Gold prediction* ({args.model}, {args.target_type}, horizon={args.horizon}):\n"
            msg += f"Last close: {result['last_close']}\n"
            msg += f"Date: {result['last_date']}\n"
            msg += f"Prediction for: {result['prediction_date']}\n\n"
            
            for model_name, pred in result['predictions'].items():
                msg += f"*{model_name}*: {pred['direction']} (conf: {pred['confidence']:.3f})\n"
            
            predictor.send_telegram_message(msg)
    else:
        print("[ERROR] Прогноз не получен.")