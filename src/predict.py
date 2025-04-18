#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для генерации прогнозов цены золота и отправки их в Telegram.
"""

import os
import logging
import os  # Для создания директории логов

import numpy as np
import pandas as pd
import argparse
import json
from datetime import datetime, timedelta
import joblib

# Для Telegram бота
import telegram
from telegram.ext import Updater, CommandHandler

# Наши модули
from data_loader import GoldDataLoader
from features import FeatureGenerator
from models import XGBoostModel, LSTMModel, EnsembleModel

# Настройка логирования
# --- Создание директории logs, если не существует ---
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
    
    def __init__(self, model_dir="../models", data_dir="../data", config_path="../config/predictor_config.json"):
        """
        Инициализация предсказателя.
        
        Args:
            model_dir (str): Директория с сохраненными моделями
            data_dir (str): Директория для сохранения данных
            config_path (str): Путь к конфигурационному файлу
        """
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.config_path = config_path
        
        # Проверяем наличие директорий
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Загружаем конфигурацию
        self.config = self.load_config()
        
        # Инициализируем объекты
        self.data_loader = GoldDataLoader(data_dir=data_dir)
        self.feature_generator = FeatureGenerator(scaling_method='standard')
        
        # Модели
        self.xgb_model = None
        self.lstm_model = None
        self.ensemble = None
        
        # Загружаем модели, если путь указан в конфигурации
        self.load_models()
        
    def load_config(self):
        """
        Загрузка конфигурации из файла.
        
        Returns:
            dict: Конфигурация
        """
        # Значения по умолчанию
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
        
        # Проверяем существование файла конфигурации
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    # Обновляем значения по умолчанию
                    default_config.update(config)
                logger.info(f"Конфигурация загружена из {self.config_path}")
            except Exception as e:
                logger.error(f"Ошибка при загрузке конфигурации: {e}")
        else:
            # Создаем конфигурацию по умолчанию
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
        """
        self.config.update(kwargs)
        self.save_config()
    
    def load_models(self):
        """
        Загрузка моделей.
        
        Returns:
            bool: True, если хотя бы одна модель загружена успешно
        """
        success = False
        
        # Загружаем XGBoost модель
        if self.config["xgb_model_path"]:
            try:
                self.xgb_model = XGBoostModel(target_type=self.config["target_type"])
                self.xgb_model.load_model(os.path.basename(self.config["xgb_model_path"]))
                logger.info(f"XGBoost модель загружена из {self.config['xgb_model_path']}")
                success = True
            except Exception as e:
                logger.error(f"Ошибка при загрузке XGBoost модели: {e}")
        
        # Загружаем LSTM модель
        if self.config["lstm_model_path"]:
            try:
                self.lstm_model = LSTMModel(
                    target_type=self.config["target_type"],
                    sequence_length=self.config["sequence_length"]
                )
                self.lstm_model.load_model(os.path.basename(self.config["lstm_model_path"]))
                logger.info(f"LSTM модель загружена из {self.config['lstm_model_path']}")
                success = True
            except Exception as e:
                logger.error(f"Ошибка при загрузке LSTM модели: {e}")
        
        # Загружаем информацию об ансамбле
        if self.config["ensemble_info_path"]:
            try:
                self.ensemble = EnsembleModel(target_type=self.config["target_type"])
                ensemble_info = self.ensemble.load_ensemble_info(os.path.basename(self.config["ensemble_info_path"]))
                
                # Добавляем модели в ансамбль
                if ensemble_info and self.xgb_model and "xgboost" in ensemble_info["model_names"]:
                    self.ensemble.add_model("xgboost", self.xgb_model, weight=ensemble_info["weights"].get("xgboost", 1.0))
                
                if ensemble_info and self.lstm_model and "lstm" in ensemble_info["model_names"]:
                    self.ensemble.add_model("lstm", self.lstm_model, weight=ensemble_info["weights"].get("lstm", 1.0))
                
                logger.info(f"Информация об ансамбле загружена из {self.config['ensemble_info_path']}")
                success = True
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
            # Загружаем последние данные
            latest_data = self.data_loader.get_latest_data(days=100)  # Берем достаточно исторических данных
            
            if latest_data is None or len(latest_data) < 50:
                logger.error("Недостаточно данных для прогнозирования")
                return None
            
            # Получаем последнюю известную цену закрытия
            last_close = latest_data['Close'].iloc[-1]
            last_date = latest_data.index[-1]
            
            # Подготавливаем признаки для XGBoost
            features_df = self.feature_generator.prepare_features(
                latest_data, 
                horizon=self.config["horizon"], 
                target_type=self.config["target_type"]
            )
            
            # Получаем последнюю строку с признаками (без целевой переменной)
            last_features = features_df.iloc[-1:].drop(['Target', 'Future_Close'], axis=1, errors='ignore')
            
            # Для LSTM нам нужна последовательность
            sequence_length = self.config["sequence_length"]
            
            # Берем последние sequence_length строк для создания последовательности
            if len(latest_data) >= sequence_length:
                # Подготавливаем признаки для последних sequence_length дней
                seq_df = self.feature_generator.prepare_features(
                    latest_data.iloc[-(sequence_length+1):], 
                    horizon=self.config["horizon"], 
                    target_type=self.config["target_type"],
                    add_technical=True,
                    scale=True
                )
                
                # Удаляем целевую переменную
                seq_features = seq_df.drop(['Target', 'Future_Close'], axis=1, errors='ignore')
                
                # Создаем последовательность для LSTM
                last_sequence = seq_features.values.reshape(1, sequence_length, seq_features.shape[1])
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
            return None
    
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
        data = self.prepare_latest_data()
        if data is None:
            return None
        
        last_close = data['last_close']
        last_date = data['last_date']
        last_features = data['last_features']
        last_sequence = data['last_sequence']
        
        predictions = {}
        
        # Прогноз с помощью XGBoost
        if self.xgb_model is not None:
            try:
                xgb_pred = self.xgb_model.predict(last_features)
                xgb_pred_proba = self.xgb_model.predict_proba(last_features)
                
                if self.config["target_type"] == 'binary':
                    # 1 = рост, 0 = падение
                    xgb_direction = "UP" if xgb_pred[0] == 1 else "DOWN"
                    xgb_confidence = float(xgb_pred_proba[0]) if xgb_pred[0] == 1 else 1.0 - float(xgb_pred_proba[0])
                    predictions['xgboost'] = {
                        'direction': xgb_direction,
                        'confidence': xgb_confidence
                    }
                elif self.config["target_type"] == 'regression':
                    xgb_price = float(xgb_pred[0])
                    xgb_direction = "UP" if xgb_price > last_close else "DOWN"
                    predictions['xgboost'] = {
                        'predicted_price': xgb_price,
                        'direction': xgb_direction,
                        'change': float(xgb_price - last_close),
                        'change_percent': float((xgb_price - last_close) / last_close * 100)
                    }
                elif self.config["target_type"] == 'classification':
                    # Классы: 0 (сильно вниз), 1 (вниз), 2 (боковик), 3 (вверх), 4 (сильно вверх)
                    class_names = ["STRONG DOWN", "DOWN", "SIDEWAYS", "UP", "STRONG UP"]
                    xgb_class = int(xgb_pred[0])
                    xgb_direction = class_names[xgb_class]
                    xgb_confidence = float(xgb_pred_proba[0][xgb_class])
                    predictions['xgboost'] = {
                        'direction': xgb_direction,
                        'confidence': xgb_confidence
                    }
            except Exception as e:
                logger.error(f"Ошибка при прогнозировании с XGBoost: {e}")
        
        # Прогноз с помощью LSTM
        if self.lstm_model is not None and last_sequence is not None:
            try:
                lstm_pred = self.lstm_model.predict(last_sequence)
                lstm_pred_proba = self.lstm_model.predict_proba(last_sequence)
                
                if self.config["target_type"] == 'binary':
                    # 1 = рост, 0 = падение
                    lstm_direction = "UP" if lstm_pred[0] == 1 else "DOWN"
                    lstm_confidence = float(lstm_pred_proba[0][0]) if lstm_pred[0] == 1 else 1.0 - float(lstm_pred_proba[0][0])
                    predictions['lstm'] = {
                        'direction': lstm_direction,
                        'confidence': lstm_confidence
                    }
                elif self.config["target_type"] == 'regression':
                    lstm_price = float(lstm_pred[0])
                    lstm_direction = "UP" if lstm_price > last_close else "DOWN"
                    predictions['lstm'] = {
                        'predicted_price': lstm_price,
                        'direction': lstm_direction,
                        'change': float(lstm_price - last_close),
                        'change_percent': float((lstm_price - last_close) / last_close * 100)
                    }
                elif self.config["target_type"] == 'classification':
                    # Классы: 0 (сильно вниз), 1 (вниз), 2 (боковик), 3 (вверх), 4 (сильно вверх)
                    class_names = ["STRONG DOWN", "DOWN", "SIDEWAYS", "UP", "STRONG UP"]
                    lstm_class = int(lstm_pred[0])
                    lstm_direction = class_names[lstm_class]
                    lstm_confidence = float(lstm_pred_proba[0][lstm_class])
                    predictions['lstm'] = {
                        'direction': lstm_direction,
                        'confidence': lstm_confidence
                    }
            except Exception as e:
                logger.error(f"Ошибка при прогнозировании с LSTM: {e}")
        
        # Прогноз с помощью ансамбля
        if self.ensemble is not None:
            try:
                ensemble_pred = self.ensemble.predict(last_features, last_sequence)
                ensemble_pred_proba = self.ensemble.predict_proba(last_features, last_sequence)
                
                if self.config["target_type"] == 'binary':
                    # 1 = рост, 0 = падение
                    ensemble_direction = "UP" if ensemble_pred[0] == 1 else "DOWN"
                    ensemble_confidence = float(ensemble_pred_proba[0]) if ensemble_pred[0] == 1 else 1.0 - float(ensemble_pred_proba[0])
                    predictions['ensemble'] = {
                        'direction': ensemble_direction,
                        'confidence': ensemble_confidence
                    }
                elif self.config["target_type"] == 'regression':
                    ensemble_price = float(ensemble_pred[0])
                    ensemble_direction = "UP" if ensemble_price > last_close else "DOWN"
                    predictions['ensemble'] = {
                        'predicted_price': ensemble_price,
                        'direction': ensemble_direction,
                        'change': float(ensemble_price - last_close),
                        'change_percent': float((ensemble_price - last_close) / last_close * 100)
                    }
                elif self.config["target_type"] == 'classification':
                    # Классы: 0 (сильно вниз), 1 (вниз), 2 (боковик), 3 (вверх), 4 (сильно вверх)
                    class_names = ["STRONG DOWN", "DOWN", "SIDEWAYS", "UP", "STRONG UP"]
                    ensemble_class = int(ensemble_pred[0])
                    ensemble_direction = class_names[ensemble_class]
                    ensemble_confidence = float(ensemble_pred_proba[0][ensemble_class])
                    predictions['ensemble'] = {
                        'direction': ensemble_direction,
                        'confidence': ensemble_confidence
                    }
            except Exception as e:
                logger.error(f"Ошибка при прогнозировании с ансамблем: {e}")
        
        if not predictions:
            logger.error("Не удалось получить прогнозы")
            return None
        
        # Формируем итоговый прогноз
        prediction_date = last_date + timedelta(days=self.config["horizon"])
        
        return {
            'last_close': last_close,
            'last_date': last_date.strftime('%Y-%m-%d'),
            'prediction_date': prediction_date.strftime('%Y-%m-%d'),
            'horizon': self.config["horizon"],
            'predictions': predictions,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def format_prediction_message(self, prediction):
        """
        Форматирование прогноза для отправки в Telegram.
        
        Args:
            prediction (dict): Прогноз
            
        Returns:
            str: Отформатированное сообщение
        """
        if prediction is None:
            return "Не удалось сформировать прогноз."
        
        last_close = prediction['last_close']
        last_date = prediction['last_date']
        prediction_date = prediction['prediction_date']
        predictions = prediction['predictions']
        
        message = f"🔮 *Прогноз цены золота*\n\n"
        message += f"📊 Последняя цена: ${last_close:.2f} ({last_date})\n"
        message += f"🎯 Прогноз на: {prediction_date} (горизонт: {prediction['horizon']} дней)\n\n"
        
        # Добавляем прогнозы от разных моделей
        if 'ensemble' in predictions:
            message += f"*🤖 Ансамбль моделей:*\n"
            ensemble_pred = predictions['ensemble']
            
            if 'predicted_price' in ensemble_pred:  # Регрессия
                pred_price = ensemble_pred['predicted_price']
                change = ensemble_pred['change']
                change_percent = ensemble_pred['change_percent']
                direction = ensemble_pred['direction']
                
                emoji = "🟢" if direction == "UP" else "🔴"
                sign = "+" if change > 0 else ""
                
                message += f"{emoji} Прогноз: ${pred_price:.2f} ({sign}{change:.2f}, {sign}{change_percent:.2f}%)\n"
            else:  # Классификация
                direction = ensemble_pred['direction']
                confidence = ensemble_pred['confidence'] * 100
                
                emoji = "🟢" if "UP" in direction else "🔴" if "DOWN" in direction else "⚪️"
                message += f"{emoji} Направление: {direction} (уверенность: {confidence:.1f}%)\n"
        
        # XGBoost прогноз
        if 'xgboost' in predictions:
            message += f"\n*📈 XGBoost:*\n"
            xgb_pred = predictions['xgboost']
            
            if 'predicted_price' in xgb_pred:  # Регрессия
                pred_price = xgb_pred['predicted_price']
                change = xgb_pred['change']
                change_percent = xgb_pred['change_percent']
                direction = xgb_pred['direction']
                
                emoji = "🟢" if direction == "UP" else "🔴"
                sign = "+" if change > 0 else ""
                
                message += f"{emoji} Прогноз: ${pred_price:.2f} ({sign}{change:.2f}, {sign}{change_percent:.2f}%)\n"
            else:  # Классификация
                direction = xgb_pred['direction']
                confidence = xgb_pred['confidence'] * 100
                
                emoji = "🟢" if "UP" in direction else "🔴" if "DOWN" in direction else "⚪️"
                message += f"{emoji} Направление: {direction} (уверенность: {confidence:.1f}%)\n"
        
        # LSTM прогноз
        if 'lstm' in predictions:
            message += f"\n*🧠 LSTM:*\n"
            lstm_pred = predictions['lstm']
            
            if 'predicted_price' in lstm_pred:  # Регрессия
                pred_price = lstm_pred['predicted_price']
                change = lstm_pred['change']
                change_percent = lstm_pred['change_percent']
                direction = lstm_pred['direction']
                
                emoji = "🟢" if direction == "UP" else "🔴"
                sign = "+" if change > 0 else ""
                
                message += f"{emoji} Прогноз: ${pred_price:.2f} ({sign}{change:.2f}, {sign}{change_percent:.2f}%)\n"
            else:  # Классификация
                direction = lstm_pred['direction']
                confidence = lstm_pred['confidence'] * 100
                
                emoji = "🟢" if "UP" in direction else "🔴" if "DOWN" in direction else "⚪️"
                message += f"{emoji} Направление: {direction} (уверенность: {confidence:.1f}%)\n"
        
        # Добавляем время прогноза
        message += f"\n⏱ Прогноз сформирован: {prediction['timestamp']}"
        
        return message
    
    def save_prediction(self, prediction):
        """
        Сохранение прогноза в файл.
        
        Args:
            prediction (dict): Прогноз
            
        Returns:
            str: Путь к сохраненному файлу или None в случае ошибки
        """
        if prediction is None:
            return None
        
        # Создаем директорию для прогнозов, если она не существует
        predictions_dir = os.path.join(self.data_dir, 'predictions')
        os.makedirs(predictions_dir, exist_ok=True)
        
        # Формируем имя файла с датой прогноза
        prediction_date = prediction['prediction_date']
        file_name = f"prediction_{prediction_date}.json"
        file_path = os.path.join(predictions_dir, file_name)
        
        try:
            with open(file_path, 'w') as f:
                json.dump(prediction, f, indent=4)
            logger.info(f"Прогноз сохранен в {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Ошибка при сохранении прогноза: {e}")
            return None
    
    def load_prediction(self, prediction_date):
        """
        Загрузка прогноза из файла.
        
        Args:
            prediction_date (str): Дата прогноза в формате 'YYYY-MM-DD'
            
        Returns:
            dict: Прогноз или None в случае ошибки
        """
        predictions_dir = os.path.join(self.data_dir, 'predictions')
        file_name = f"prediction_{prediction_date}.json"
        file_path = os.path.join(predictions_dir, file_name)
        
        if not os.path.exists(file_path):
            logger.error(f"Файл прогноза {file_path} не найден")
            return None
        
        try:
            with open(file_path, 'r') as f:
                prediction = json.load(f)
            logger.info(f"Прогноз загружен из {file_path}")
            return prediction
        except Exception as e:
            logger.error(f"Ошибка при загрузке прогноза: {e}")
            return None
    
    def verify_prediction(self, prediction_date):
        """
        Проверка точности прогноза путем сравнения с фактическими данными.
        
        Args:
            prediction_date (str): Дата прогноза в формате 'YYYY-MM-DD'
            
        Returns:
            dict: Результаты проверки или None в случае ошибки
        """
        # Загружаем прогноз
        prediction = self.load_prediction(prediction_date)
        if prediction is None:
            return None
        
        # Загружаем актуальные данные
        latest_data = self.data_loader.get_latest_data(days=30)
        if latest_data is None:
            logger.error("Не удалось загрузить актуальные данные")
            return None
        
        # Проверяем наличие данных для даты прогноза
        try:
            # Преобразуем строковую дату в datetime
            pred_date = datetime.strptime(prediction_date, '%Y-%m-%d')
            
            # Находим ближайшую доступную дату (для случаев, когда прогноз на выходные)
            closest_date = None
            min_days_diff = float('inf')
            
            for date in latest_data.index:
                days_diff = abs((date - pred_date).days)
                if days_diff < min_days_diff:
                    min_days_diff = days_diff
                    closest_date = date
            
            if closest_date is None or min_days_diff > 3:  # Допускаем максимальное отклонение в 3 дня
                logger.error(f"Не найдены данные для даты прогноза {prediction_date}")
                return None
            
            # Получаем фактическую цену закрытия
            actual_close = latest_data.loc[closest_date, 'Close']
            actual_date = closest_date.strftime('%Y-%m-%d')
            
            # Получаем последнюю известную цену из прогноза
            last_close = prediction['last_close']
            
            # Определяем фактическое направление движения
            actual_direction = "UP" if actual_close > last_close else "DOWN"
            actual_change = actual_close - last_close
            actual_change_percent = (actual_change / last_close) * 100
            
            # Формируем результаты проверки
            verification = {
                'prediction_date': prediction_date,
                'actual_date': actual_date,
                'last_close': last_close,
                'actual_close': float(actual_close),
                'actual_direction': actual_direction,
                'actual_change': float(actual_change),
                'actual_change_percent': float(actual_change_percent),
                'predictions': {}
            }
            
            # Проверяем точность прогнозов от разных моделей
            for model_name, model_pred in prediction['predictions'].items():
                model_verification = {}
                
                if 'predicted_price' in model_pred:  # Регрессия
                    predicted_price = model_pred['predicted_price']
                    predicted_direction = model_pred['direction']
                    
                    # Ошибка прогноза
                    error = actual_close - predicted_price
                    error_percent = (error / predicted_price) * 100
                    
                    # Правильно ли угадано направление
                    direction_correct = predicted_direction == actual_direction
                    
                    model_verification.update({
                        'predicted_price': predicted_price,
                        'predicted_direction': predicted_direction,
                        'error': float(error),
                        'error_percent': float(error_percent),
                        'direction_correct': direction_correct
                    })
                else:  # Классификация
                    predicted_direction = model_pred['direction']
                    confidence = model_pred['confidence']
                    
                    # Определяем, соответствует ли прогноз фактическому направлению
                    direction_match = False
                    
                    if ("UP" in predicted_direction and actual_direction == "UP") or \
                       ("DOWN" in predicted_direction and actual_direction == "DOWN") or \
                       (predicted_direction == "SIDEWAYS" and abs(actual_change_percent) < 1.0):  # Боковик, если изменение менее 1%
                        direction_match = True
                    
                    model_verification.update({
                        'predicted_direction': predicted_direction,
                        'confidence': confidence,
                        'direction_correct': direction_match
                    })
                
                verification['predictions'][model_name] = model_verification
            
            # Сохраняем результаты проверки
            verifications_dir = os.path.join(self.data_dir, 'verifications')
            os.makedirs(verifications_dir, exist_ok=True)
            
            file_name = f"verification_{prediction_date}.json"
            file_path = os.path.join(verifications_dir, file_name)
            
            with open(file_path, 'w') as f:
                json.dump(verification, f, indent=4)
            
            logger.info(f"Проверка прогноза на {prediction_date} сохранена в {file_path}")
            
            return verification
            
        except Exception as e:
            logger.error(f"Ошибка при проверке прогноза: {e}")
            return None
    
    def format_verification_message(self, verification):
        """
        Форматирование результатов проверки прогноза для отправки в Telegram.
        
        Args:
            verification (dict): Результаты проверки
            
        Returns:
            str: Отформатированное сообщение
        """
        if verification is None:
            return "Не удалось проверить прогноз."
        
        last_close = verification['last_close']
        actual_close = verification['actual_close']
        actual_direction = verification['actual_direction']
        actual_change = verification['actual_change']
        actual_change_percent = verification['actual_change_percent']
        
        emoji = "🟢" if actual_direction == "UP" else "🔴"
        sign = "+" if actual_change > 0 else ""
        
        message = f"📊 *Проверка прогноза*\n\n"
        message += f"📅 Прогноз на: {verification['prediction_date']}\n"
        message += f"📅 Фактическая дата: {verification['actual_date']}\n\n"
        message += f"💰 Последняя цена: ${last_close:.2f}\n"
        message += f"{emoji} Фактическая цена: ${actual_close:.2f} ({sign}{actual_change:.2f}, {sign}{actual_change_percent:.2f}%)\n\n"
        
        # Добавляем результаты проверки для разных моделей
        message += f"*Точность моделей:*\n"
        
        for model_name, model_verif in verification['predictions'].items():
            model_emoji = "🤖" if model_name == "ensemble" else "📈" if model_name == "xgboost" else "🧠" if model_name == "lstm" else "⚙️"
            
            message += f"\n{model_emoji} *{model_name.capitalize()}:*\n"
            
            if 'predicted_price' in model_verif:  # Регрессия
                pred_price = model_verif['predicted_price']
                error = model_verif['error']
                error_percent = model_verif['error_percent']
                direction_correct = model_verif['direction_correct']
                
                dir_emoji = "✅" if direction_correct else "❌"
                error_sign = "-" if error < 0 else "+"
                
                message += f"Прогноз: ${pred_price:.2f}\n"
                message += f"Ошибка: {error_sign}${abs(error):.2f} ({error_sign}{abs(error_percent):.2f}%)\n"
                message += f"Направление: {dir_emoji}\n"
            else:  # Классификация
                pred_direction = model_verif['predicted_direction']
                confidence = model_verif['confidence'] * 100
                direction_correct = model_verif['direction_correct']
                
                dir_emoji = "✅" if direction_correct else "❌"
                
                message += f"Прогноз: {pred_direction} (уверенность: {confidence:.1f}%)\n"
                message += f"Направление: {dir_emoji}\n"
        
        return message
    
    def send_telegram_message(self, message):
        """
        Отправка сообщения в Telegram.
        
        Args:
            message (str): Сообщение для отправки
            
        Returns:
            bool: True, если сообщение отправлено успешно
        """
        if not self.config["telegram_token"] or not self.config["telegram_chat_id"]:
            logger.error("Не настроены параметры Telegram (токен или chat_id)")
            return False
        
        try:
            bot = telegram.Bot(token=self.config["telegram_token"])
            bot.send_message(
                chat_id=self.config["telegram_chat_id"],
                text=message,
                parse_mode=telegram.ParseMode.MARKDOWN
            )
            logger.info("Сообщение отправлено в Telegram")
            return True
        except Exception as e:
            logger.error(f"Ошибка при отправке сообщения в Telegram: {e}")
            return False
    
    def run_prediction(self):
        """
        Запуск процесса прогнозирования и отправка результатов в Telegram.
        
        Returns:
            bool: True, если прогноз выполнен и отправлен успешно
        """
        # Формируем прогноз
        prediction = self.predict()
        
        if prediction is None:
            logger.error("Не удалось сформировать прогноз")
            return False
        
        # Сохраняем прогноз
        self.save_prediction(prediction)
        
        # Форматируем сообщение
        message = self.format_prediction_message(prediction)
        
        # Отправляем сообщение в Telegram
        return self.send_telegram_message(message)
    
    def run_verification(self, prediction_date=None):
        """
        Запуск процесса проверки прогноза и отправка результатов в Telegram.
        
        Args:
            prediction_date (str, optional): Дата прогноза в формате 'YYYY-MM-DD'.
                                           Если не указана, используется вчерашняя дата.
        
        Returns:
            bool: True, если проверка выполнена и отправлена успешно
        """
        # Если дата не указана, используем вчерашнюю дату
        if prediction_date is None:
            yesterday = datetime.now() - timedelta(days=1)
            prediction_date = yesterday.strftime('%Y-%m-%d')
        
        # Проверяем прогноз
        verification = self.verify_prediction(prediction_date)
        
        if verification is None:
            logger.error(f"Не удалось проверить прогноз на {prediction_date}")
            return False
        
        # Форматируем сообщение
        message = self.format_verification_message(verification)
        
        # Отправляем сообщение в Telegram
        return self.send_telegram_message(message)
