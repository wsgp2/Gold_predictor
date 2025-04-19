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
        Загрузка конфигурации из файла и переменных окружения.
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
        
        # Загружаем конфигурацию из файла
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
        
        # Загружаем переменные окружения и обновляем конфигурацию
        from config_loader import load_environment_variables
        load_environment_variables()
        
        # Проверяем наличие Telegram токена и chat_id в переменных окружения
        telegram_token = os.environ.get('TELEGRAM_TOKEN')
        telegram_chat_id = os.environ.get('TELEGRAM_CHAT_ID')
        
        # Обновляем конфигурацию, если переменные найдены
        if telegram_token and telegram_token.strip():
            default_config['telegram_token'] = telegram_token.strip()
            logger.info("Токен Telegram загружен из переменных окружения")
            
        if telegram_chat_id and telegram_chat_id.strip():
            default_config['telegram_chat_id'] = telegram_chat_id.strip()
            logger.info("Chat ID Telegram загружен из переменных окружения")
        
        # Сохраняем обновленную конфигурацию в файл
        self.save_config(default_config)
            
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
                # Возьмем путь из конфига и проверим, существует ли файл
                xgb_file_path = self.config["xgb_model_path"]
                
                # Сначала проверим чистый путь
                if os.path.exists(xgb_file_path):
                    pass  # Файл найден, ничего не меняем
                # Проверим, есть ли файл в ../models
                elif os.path.exists(os.path.join(self.model_dir, os.path.basename(xgb_file_path))):
                    xgb_file_path = os.path.join(self.model_dir, os.path.basename(xgb_file_path))
                
                if os.path.exists(xgb_file_path):
                    # Передаём правильную директорию с моделями
                    model_directory = os.path.dirname(os.path.abspath(xgb_file_path)) if os.path.dirname(xgb_file_path) else self.model_dir
                    self.xgb_model = XGBoostModel(model_dir=model_directory, target_type=self.config["target_type"])
                    self.xgb_model.load_model(os.path.basename(xgb_file_path))
                    logger.info(f"XGBoost модель загружена из {xgb_file_path}")
                    success = True
            except Exception as e:
                logger.error(f"Ошибка при загрузке XGBoost модели: {e}")
        
        if self.config["lstm_model_path"]:
            try:
                # Проверяем существует ли файл по указанному пути
                # Возьмем путь из конфига и проверим, существует ли файл
                lstm_file_path = self.config["lstm_model_path"]
                
                # Сначала проверим чистый путь
                if os.path.exists(lstm_file_path):
                    pass  # Файл найден, ничего не меняем
                # Проверим, есть ли файл в ../models
                elif os.path.exists(os.path.join(self.model_dir, os.path.basename(lstm_file_path))):
                    lstm_file_path = os.path.join(self.model_dir, os.path.basename(lstm_file_path))
                
                if os.path.exists(lstm_file_path):
                    # Передаём правильную директорию с моделями
                    model_directory = os.path.dirname(os.path.abspath(lstm_file_path)) if os.path.dirname(lstm_file_path) else self.model_dir
                    self.lstm_model = LSTMModel(
                        model_dir=model_directory,
                        target_type=self.config["target_type"],
                        sequence_length=self.config["sequence_length"]
                    )
                    self.lstm_model.load_model(os.path.basename(lstm_file_path))
                    logger.info(f"LSTM модель загружена из {lstm_file_path}")
                    success = True
            except Exception as e:
                logger.error(f"Ошибка при загрузке LSTM модели: {e}")
        
        if self.config["ensemble_info_path"]:
            try:
                # Проверяем существует ли файл по указанному пути
                # Возьмем путь из конфига и проверим, существует ли файл
                ensemble_file_path = self.config["ensemble_info_path"]
                
                # Сначала проверим чистый путь
                if not ensemble_file_path or os.path.exists(ensemble_file_path):
                    pass  # Файл найден или путь пустой, ничего не меняем
                # Проверим, есть ли файл в ../models
                elif os.path.exists(os.path.join(self.model_dir, os.path.basename(ensemble_file_path))):
                    ensemble_file_path = os.path.join(self.model_dir, os.path.basename(ensemble_file_path))
                
                if os.path.exists(ensemble_file_path):
                    # Передаём правильную директорию с моделями
                    model_directory = os.path.dirname(os.path.abspath(ensemble_file_path)) if os.path.dirname(ensemble_file_path) else self.model_dir
                    self.ensemble = EnsembleModel(model_dir=model_directory, target_type=self.config["target_type"])
                    ensemble_info = self.ensemble.load_ensemble_info(os.path.basename(ensemble_file_path))
                    
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
            # Сначала обновляем данные через Bybit API, чтобы гарантировать актуальность
            update_success = self.update_data()
            logger.info(f"Bybit API обновление данных: {update_success}")
            
            # Загружаем напрямую из обновленного файла, вместо использования get_latest_data
            import pandas as pd
            import os
            
            csv_path = os.path.join(self.data_dir, 'GC_F_latest.csv')
            if os.path.exists(csv_path):
                try:
                    # Загружаем данные напрямую из файла, включая свежие обновления от Bybit
                    latest_data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                    logger.info(f"Загружены данные с обновлениями от Bybit, последняя дата: {latest_data.index[-1]}")
                except Exception as e:
                    logger.error(f"Ошибка при загрузке данных из {csv_path}: {str(e)}")
                    # Фолбэк на стандартный метод
                    latest_data = self.data_loader.get_latest_data(days=300)  # Увеличиваем исторический период
            else:
                # Файл не найден, используем стандартный метод загрузки
                logger.warning(f"Файл {csv_path} не найден, используем стандартный метод загрузки")
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
            
            # Добавляем Future_Close для совместимости с моделями
            if 'Future_Close' not in data_for_features.columns:
                data_for_features['Future_Close'] = data_for_features['Close']
                logger.info("Добавлен признак Future_Close для совместимости с моделями")

            # Получаем только числовые признаки
            numeric_features = data_for_features.select_dtypes(include=[np.number])
            
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
        
        # Загружаем точный список признаков из метаданных модели
        import joblib
        try:
            model_path = os.path.join(self.model_dir, self.config["xgb_model_path"])
            # Извлекаем только имя файла без пути для корректного поиска метаданных
            base_filename = os.path.basename(self.config["xgb_model_path"])
            metadata_path = os.path.join(self.model_dir, base_filename.replace('.json', '_metadata.joblib'))
            
            logger.info(f"Путь к метаданным: {metadata_path}")
            if not os.path.exists(metadata_path):
                # Поддержка для старой версии имени файла
                metadata_path = os.path.join(self.model_dir, base_filename.replace('.json', '_metadata.joblib'))
                if not os.path.exists(metadata_path):
                    raise FileNotFoundError(f"Файл метаданных не найден: {metadata_path}")
            
            metadata = joblib.load(metadata_path)
            expected_features = metadata.get('feature_names', [])
            logger.info(f"Загружено {len(expected_features)} признаков из метаданных XGBoost модели")
            logger.info(f"Первые 5 признаков из метаданных: {expected_features[:5]}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке метаданных модели: {e}")
            # Запасной список - точный список из 62 признаков для модели версии 20250419
            expected_features = [
                'Open', 'High', 'Low', 'Close', 'Volume', 
                'MA_5', 'MA_ratio_5', 'MA_10', 'MA_ratio_10', 'MA_20', 'MA_ratio_20', 'MA_50', 'MA_ratio_50', 
                'MA_100', 'MA_ratio_100', 'EMA_5', 'EMA_ratio_5', 'EMA_10', 'EMA_ratio_10', 'EMA_20', 'EMA_ratio_20', 
                'EMA_50', 'EMA_ratio_50', 'EMA_100', 'EMA_ratio_100', 'RSI_7', 'RSI_14', 'RSI_21', 
                'MACD', 'MACD_Signal', 'MACD_Hist', 'MACD_Hist_Change',
                'BB_Upper_20', 'BB_Lower_20', 'BB_Width_20', 'BB_Position_20',
                'Stoch_%K_14', 'Stoch_%D_14', 'ATR_14', 'CCI_20', 'Price_Change', 'Return', 
                'Volatility_5', 'Volatility_10', 'Volatility_21', 'High_Low_Range', 'High_Low_Range_Pct', 
                'Volume_MA_5', 'Volume_ratio_5', 'Volume_MA_10', 'Volume_ratio_10', 'Volume_MA_20', 'Volume_ratio_20', 
                'Future_Close'
            ]
            logger.info(f"Используем запасной список из {len(expected_features)} признаков")
        
        # Проверяем, какие признаки есть в features_df но отсутствуют в expected_features
        extra_features = [f for f in features_copy.columns if f not in expected_features]
        if extra_features:
            logger.info(f"Обнаружены дополнительные признаки в данных: {extra_features[:5]}{'...' if len(extra_features) > 5 else ''}")
        
        # Проверяем, каких признаков не хватает
        missing_features = [f for f in expected_features if f not in features_copy.columns]
        if missing_features:
            logger.warning(f"Отсутствуют необходимые признаки: {missing_features}")
            
            # Мэппинг альтернативных имен для быстрой замены
            alt_mappings = {
                'MACD_line': 'MACD', 
                'MACD_signal': 'MACD_Signal',
                'MACD_histogram': 'MACD_Hist',
                'BB_upper_20': 'BB_Upper_20',
                'BB_lower_20': 'BB_Lower_20',
                'BB_width_20': 'BB_Width_20',
                'BB_position_20': 'BB_Position_20'
            }
            
            # Добавляем отсутствующие признаки с умными значениями по умолчанию
            for feature in missing_features:
                # Проверяем сначала альтернативные имена
                alt_found = False
                
                # Проверка на кортежные альтернативы ('Column', 'GC=F')
                base_feature = None
                if feature.startswith("('") and feature.endswith("', 'GC=F')"):
                    base_feature = feature.split("'")[1]  # Извлекаем имя столбца из кортежа
                    if base_feature in features_copy:
                        features_copy[feature] = features_copy[base_feature]
                        alt_found = True
                        logger.info(f"Признак {feature} заменен на {base_feature}")
                
                # Проверка на известные альтернативные имена
                if not alt_found and feature in alt_mappings and alt_mappings[feature] in features_copy:
                    features_copy[feature] = features_copy[alt_mappings[feature]]
                    alt_found = True
                    logger.info(f"Признак {feature} заменен на {alt_mappings[feature]}")
                
                # Особая обработка для Future_Close
                if not alt_found and feature == 'Future_Close' and 'Close' in features_copy:
                    features_copy[feature] = features_copy['Close']
                    alt_found = True
                    logger.info(f"Признак Future_Close заменен на текущий Close")
                
                # Если ничего не найдено, добавляем нули
                if not alt_found:
                    logger.warning(f"Признак {feature} отсутствует, добавляем нулевые значения")
                    features_copy[feature] = 0.0
            
            # Выводим диагностическую информацию о готовом наборе признаков
            logger.info(f"Признаки для XGBoost (всего {len(expected_features)}): {expected_features[:5]}... и еще {len(expected_features)-5 if len(expected_features) > 5 else 0}")
            
        # Возвращаем датафрейм с признаками в строгом порядке, как требует модель
        ordered_df = features_copy[expected_features]
        
        # Финальная проверка - все ли признаки на месте
        if list(ordered_df.columns) != expected_features:
            logger.error(f"Порядок столбцов не соответствует требуемому!")
            # Если не совпадают, принудительно задаем порядок снова
            ordered_df = ordered_df.reindex(columns=expected_features)
        
        # Проверим на наличие NaN и заменим их на 0
        if ordered_df.isna().any().any():
            nan_columns = ordered_df.columns[ordered_df.isna().any()].tolist()
            logger.warning(f"Обнаружены NaN значения в признаках: {nan_columns}, заменяем на 0")
            ordered_df = ordered_df.fillna(0)
        
        # Проверяем размерность данных
        logger.info(f"Финальная размерность данных для XGBoost: {ordered_df.shape}, ожидаемое число признаков: {len(expected_features)}")
        
        return ordered_df
        
    def get_current_price(self):
        """
        Получение актуальной цены золота в реальном времени
        
        Returns:
            dict: Словарь с актуальной ценой, временем и источником данных
        """
        try:
            # Импортируем модуль price_fetchers
            from price_fetchers import get_latest_gold_price
            
            # Получаем актуальную цену
            price, timestamp, source = get_latest_gold_price()
            
            if price is not None:
                logger.info(f"Получена актуальная цена золота: ${price:.2f} (источник: {source}, время: {timestamp})")
                return {
                    "price": price,
                    "timestamp": timestamp,
                    "source": source
                }
            else:
                logger.warning("Не удалось получить актуальную цену из внешних источников")
                return None
        except Exception as e:
            logger.error(f"Ошибка при получении актуальной цены: {e}")
            return None
    
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
    
    def update_data(self):
        """
        Автоматическое обновление данных через Bybit для получения самых свежих данных.
        
        Returns:
            bool: True если данные обновлены успешно, иначе False.
        """
        try:
            # Загружаем переменные окружения и модуль обновления данных
            from config_loader import load_environment_variables
            load_environment_variables()
            import os
            from data_updater import update_gold_history_from_bybit
            
            # Получаем API ключи из переменных окружения
            api_key = os.getenv('BYBIT_API_KEY', '')
            api_secret = os.getenv('BYBIT_API_SECRET', '')
            
            # Проверяем наличие ключей
            if not api_key or not api_secret:
                logger.warning("Отсутствуют API ключи Bybit. Используем кэшированные данные.")
                return False
            
            # Обновляем данные
            csv_path = os.path.join(self.data_dir, 'GC_F_latest.csv')
            update_result = update_gold_history_from_bybit(csv_path, api_key, api_secret)
            
            if update_result:
                logger.info("Данные успешно обновлены через Bybit API")
                return True
            else:
                logger.warning("Не удалось обновить данные через Bybit API")
                return False
        except Exception as e:
            logger.error(f"Ошибка при обновлении данных: {str(e)}")
            return False
    
    def predict(self):
        """
        Генерация прогноза цены золота на horizon дней вперед.
        Returns:
            dict: Прогноз или None в случае ошибки
        """
        try:
            # Проверяем наличие моделей
            if self.xgb_model is None and self.lstm_model is None and self.ensemble is None:
                logger.error("Ни одна модель не загружена")
                return None
            
            # Автоматически обновляем данные
            self.update_data()
            
            # Получаем актуальную цену золота в реальном времени
            current_market_price = self.get_current_price()
            
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
            
            # Определяем дату прогноза - всегда следующий день от текущей даты
            current_date = datetime.now()
            prediction_date = (current_date + timedelta(days=self.config["horizon"])).strftime("%Y-%m-%d")
            logger.info(f"Прогноз на дату: {prediction_date}")
            
            # Проводим диагностику признаков
            self._diagnostic_features(last_features)
            
            # Создаем список предсказаний
            predictions = {}
            
            # Прогноз XGBoost
            if self.xgb_model is not None:
                try:
                    # Преобразуем признаки в формат, ожидаемый XGBoost
                    xgb_features = self._reorder_features_for_xgboost(last_features)
                    # Логируем размерность данных для проверки
                    logger.info(f"Размерность данных для XGBoost: {xgb_features.shape}")
                    
                    # Прогнозируем
                    xgb_pred = self.xgb_model.predict(xgb_features)
                    
                    # Получаем вероятности
                    xgb_proba = self.xgb_model.predict_proba(xgb_features)
                    
                    # Формируем предсказание
                    if self.config["target_type"] == 'binary':
                        xgb_direction = "UP" if xgb_pred[0] == 1 else "DOWN"
                        xgb_confidence = float(xgb_proba[0, 1]) if xgb_proba.ndim > 1 else float(xgb_proba[0])
                        
                        predictions['xgboost'] = {
                            'direction': xgb_direction,
                            'confidence': xgb_confidence
                        }
                        logger.info(f"XGBoost предсказание: {xgb_direction} (уверенность: {xgb_confidence:.3f})")
                except Exception as e:
                    logger.error(f"Ошибка при прогнозировании с XGBoost: {e}")
                    
            # Прогноз LSTM
            if self.lstm_model is not None and last_sequence is not None:
                try:
                    lstm_pred = self.lstm_model.predict(last_sequence)
                    
                    # Формируем предсказание
                    if self.config["target_type"] == 'binary':
                        if isinstance(lstm_pred, np.ndarray):
                            lstm_confidence = float(lstm_pred.flatten()[0])
                            lstm_direction = "UP" if lstm_confidence > 0.5 else "DOWN"
                            
                            predictions['lstm'] = {
                                'direction': lstm_direction,
                                'confidence': lstm_confidence
                            }
                            logger.info(f"LSTM предсказание: {lstm_direction} (уверенность: {lstm_confidence:.3f})")
                except Exception as e:
                    logger.error(f"Ошибка при прогнозировании с LSTM: {e}")
    
            # Прогноз ансамбля
            if self.ensemble is not None and predictions:
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
                        
                        logger.info(f"[Ensemble] Прогноз: {ensemble_direction}, вероятность: {ensemble_confidence:.3f}")
                except Exception as e:
                    logger.error(f"Ошибка при прогнозировании с ансамблем: {e}")
            
            if not predictions:
                logger.error("Не удалось получить ни одного прогноза")
                return None
            
            # Определяем основной прогноз в зависимости от заданной модели
            model_type = getattr(self.args, 'model', 'ensemble')
            main_model = model_type
            
            # Если указанная модель недоступна, создаем лучшее предсказание
            if model_type not in predictions:
                available_models = list(predictions.keys())
                
                # Если доступны обе модели XGBoost и LSTM, создаем взвешенное предсказание
                if 'xgboost' in available_models and 'lstm' in available_models:
                    # Используем веса на основе F1-метрик: LSTM=0.64, XGBoost=0.36
                    weights = {'lstm': 0.64, 'xgboost': 0.36}
                    
                    # Вычисляем взвешенную вероятность
                    weighted_confidence = (predictions['lstm']['confidence'] * weights['lstm'] + 
                                          predictions['xgboost']['confidence'] * weights['xgboost'])
                    
                    # Определяем направление на основе взвешенной вероятности
                    ensemble_direction = "UP" if weighted_confidence > 0.5 else "DOWN"
                    
                    # Создаем виртуальный ансамбль
                    predictions['weighted_ensemble'] = {
                        'direction': ensemble_direction,
                        'confidence': weighted_confidence
                    }
                    
                    main_model = 'weighted_ensemble'
                    logger.info(f"Создано взвешенное предсказание: {ensemble_direction} (уверенность: {weighted_confidence:.3f})")
                    logger.info(f"Веса моделей: LSTM={weights['lstm']}, XGBoost={weights['xgboost']}")
                
                # Если доступна только одна модель, используем LSTM при наличии (как более точную)
                elif len(available_models) > 0:
                    if 'lstm' in available_models:
                        main_model = 'lstm'
                        logger.warning(f"Модель {model_type} не найдена, используем LSTM (лучшая метрика F1=0.8)")
                    else:
                        main_model = available_models[0]
                        logger.warning(f"Модель {model_type} не найдена, используем {main_model}")
            
            # Собираем результаты предсказаний
            result = {
                'predictions': predictions,
                'last_close': float(last_close),
                'last_date': str(last_date) if last_date is not None else None,
                'date': prediction_date,
                'model': main_model,
                'direction': predictions[main_model]['direction'],
                'confidence': float(predictions[main_model]['confidence']),
                'models_used': list(predictions.keys()),
                'target_type': self.config['target_type'],
                'horizon': self.config['horizon']
            }
            
            # Добавляем актуальную рыночную цену
            if current_market_price is not None:
                result['current_market_price'] = current_market_price
                logger.info(f"Добавлена актуальная цена ${current_market_price['price']:.2f} в предсказание")
            
            # Отправляем предсказание в Telegram, если нужно
            if hasattr(self, 'args') and hasattr(self.args, 'send_telegram') and self.args.send_telegram:
                self._send_prediction_to_telegram(result)
            
            # Сохраняем предсказание в трекер, если он доступен
            if self.tracker:
                self.tracker.save_prediction(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Непредвиденная ошибка при генерации прогноза: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
        print(f"Цена закрытия: ${float(last_close):.2f} (данные на {last_date})")
        
        # Выводим актуальную рыночную цену, если она доступна
        if current_market_price is not None:
            actual_price = current_market_price.get('price', 0.0)
            source = current_market_price.get('source', 'unknown')
            timestamp = current_market_price.get('timestamp', '')
            price_diff = actual_price - float(last_close)
            diff_pct = (price_diff / float(last_close)) * 100 if last_close != 0 else 0
            diff_sign = '+' if price_diff >= 0 else ''
            print(f"Актуальная цена: ${actual_price:.2f} ({diff_sign}{diff_pct:.2f}%)")
            print(f"Источник: {source}, {timestamp}")
            
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
            last_close = prediction.get('last_close', 0.0)
            prediction_date = prediction.get('date', '')
            model_type = prediction.get('model', 'ensemble')
            
            # Получаем текущую цену рынка из prediction, если она доступна
            current_market_price = prediction.get('current_market_price', None)
            
            # Эмодзи для направления движения цены
            direction_emoji = "🔼" if direction == "UP" else "🔽" if direction == "DOWN" else "⏹️"
            
            # Эмодзи для уверенности в прогнозе
            confidence_emoji = "🎥" if confidence > 0.8 else "🔍" if confidence > 0.6 else "❓"
            
            # Эмодзи для модели
            model_emoji = {
                "xgboost": "🌲",  # Дерево для XGBoost
                "lstm": "🧠",     # Мозг для нейросети
                "ensemble": "⚖️"  # Весы для ансамбля
            }.get(model_type.lower(), "🔮")
            
            # Создаем сообщение с Markdown форматированием
            message = f"*📈 Прогноз цены золота*\n\n"
            message += f"📅 *Дата прогноза:* {prediction_date}\n\n"
            
            # Добавляем информацию о ценах
            last_close = float(prediction.get('last_close', 0.0))
            message += f"💰 *Цена в момент прогноза:* ${last_close:.2f}\n"
            
            # Если доступна актуальная рыночная цена
            current_market_price = prediction.get('current_market_price', None)
            if current_market_price is not None:
                actual_price = current_market_price.get('price', 0.0)
                source = current_market_price.get('source', 'unknown')
                timestamp = current_market_price.get('timestamp', '')
                
                # Вычисляем разницу между актуальной ценой и ценой в момент прогноза
                price_diff = actual_price - last_close
                diff_pct = (price_diff / last_close) * 100 if last_close != 0 else 0
                
                # Добавляем эмодзи для разницы цен
                diff_emoji = "🔼" if price_diff > 0 else "🔽" if price_diff < 0 else "↔️"
                
                # Добавляем актуальную цену в сообщение с экранированием спецсимволов
                message += f"⚡ *Актуальная цена:* ${actual_price:.2f} {diff_emoji} ({diff_pct:+.2f}\%)\n"
                message += f"   \_Источник: {source}, {timestamp}\_\n"
            
            message += f"\n{direction_emoji} *Направление:* {direction}\n"
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
            message += f"\n🕒 *Прогноз сгенерирован:* {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
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
        print(f"Последняя цена закрытия: ${result['last_close']:.2f} (данные на {result['last_date']})")
        print(f"Дата прогноза: {result['date']}")
        print(f"Горизонт: {result['horizon']} дней")
        
        # Выводим актуальную рыночную цену, если она доступна
        if 'current_market_price' in result:
            market_price = result['current_market_price']
            actual_price = market_price.get('price', 0.0)
            source = market_price.get('source', 'unknown')
            timestamp = market_price.get('timestamp', '')
            price_diff = actual_price - float(result['last_close'])
            diff_pct = (price_diff / float(result['last_close'])) * 100 if result['last_close'] != 0 else 0
            diff_sign = '+' if price_diff >= 0 else ''
            print(f"Актуальная цена: ${actual_price:.2f} ({diff_sign}{diff_pct:.2f}%)")
            print(f"Источник: {source}, {timestamp}")
        
        # Направление прогноза
        if 'direction' in result and 'confidence' in result:
            print(f"Прогноз направления: {result['direction']} (уверенность: {result['confidence']:.3f})")
        
        print(f"Модель: {result.get('model', args.model).upper()}")
        
        print("\nПрогнозы отдельных моделей:")
        for model_name, pred in result['predictions'].items():
            print(f"  {model_name}: {pred['direction']} (уверенность: {pred['confidence']:.3f})")
        
        if args.send_telegram:
            msg = f"*Gold prediction* ({args.model}, {args.target_type}, horizon={args.horizon}):\n"
            msg += f"Last close: {result['last_close']}\n"
            msg += f"Date: {result['last_date']}\n"
            msg += f"Prediction for: {result['date']}\n\n"
            
            for model_name, pred in result['predictions'].items():
                msg += f"*{model_name}*: {pred['direction']} (conf: {pred['confidence']:.3f})\n"
            
            predictor.send_telegram_message(msg)
    else:
        print("[ERROR] Прогноз не получен.")