#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для обучения моделей машинного обучения для прогнозирования цены золота.
"""

import os
import logging
import os  # Для создания директории логов
import numpy as np
import pandas as pd
import argparse
from datetime import datetime, timedelta

# Наши модули
from data_loader import GoldDataLoader
from features import FeatureGenerator
from models import XGBoostModel, LSTMModel, EnsembleModel

# Настройка логирования
# --- Создание директории logs, если не существует ---
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'train.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def prepare_data(period='5y', target_type='binary', horizon=1, sequence_length=10):
    """
    Подготовка данных для обучения моделей.
    
    Args:
        period (str): Период для загрузки данных
        target_type (str): Тип целевой переменной ('binary', 'regression', 'classification')
        horizon (int): Горизонт прогнозирования (в днях)
        sequence_length (int): Длина последовательности для LSTM
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, X_train_seq, X_val_seq, X_test_seq)
    """
    logger.info(f"Подготовка данных для {target_type} с горизонтом {horizon} дней")
    
    # Загружаем данные
    loader = GoldDataLoader()
    data = loader.update_dataset()  # Обновляем и загружаем данные
    
    if data is None or len(data) < 100:
        logger.warning("Недостаточно данных. Загружаем исторические данные за последние 5 лет")
        data = loader.download_data(period=period)
        if data is not None:
            loader.save_data(data)
    
    if data is None:
        logger.error("Не удалось загрузить данные")
        return None
    
    logger.info(f"Загружено {len(data)} строк данных с {data.index[0]} по {data.index[-1]}")
    
    # Создаем генератор признаков
    feature_gen = FeatureGenerator(scaling_method='standard')
    
    # Подготавливаем признаки для табличных моделей (XGBoost)
    features_df = feature_gen.prepare_features(data, horizon=horizon, target_type=target_type)
    
    # Разделяем на обучающую, валидационную и тестовую выборки
    X_train, X_val, X_test, y_train, y_val, y_test = feature_gen.split_train_test(features_df)
    
    # Подготавливаем данные для LSTM
    X_sequences, y_targets = feature_gen.prepare_sequence_data(data, sequence_length, horizon, target_type)
    
    # Разделяем последовательности на обучающую, валидационную и тестовую выборки
    seq_test_size = int(len(X_sequences) * 0.2)
    seq_val_size = int(len(X_sequences) * 0.1)
    
    X_train_seq = X_sequences[:-seq_test_size-seq_val_size]
    y_train_seq = y_targets[:-seq_test_size-seq_val_size]
    
    X_val_seq = X_sequences[-seq_test_size-seq_val_size:-seq_test_size]
    y_val_seq = y_targets[-seq_test_size-seq_val_size:-seq_test_size]
    
    X_test_seq = X_sequences[-seq_test_size:]
    y_test_seq = y_targets[-seq_test_size:]
    
    logger.info(f"Данные подготовлены:")
    logger.info(f"XGBoost - обучающая: {X_train.shape}, валидационная: {X_val.shape}, тестовая: {X_test.shape}")
    logger.info(f"LSTM - обучающая: {X_train_seq.shape}, валидационная: {X_val_seq.shape}, тестовая: {X_test_seq.shape}")
    
    return {
        'tabular': (X_train, X_val, X_test, y_train, y_val, y_test),
        'sequence': (X_train_seq, X_val_seq, X_test_seq, y_train_seq, y_val_seq, y_test_seq),
        'feature_names': X_train.columns.tolist()
    }


def train_xgboost_model(data, target_type='binary', params=None):
    """
    Обучение модели XGBoost.
    
    Args:
        data (dict): Данные для обучения и тестирования
        target_type (str): Тип целевой переменной
        params (dict, optional): Параметры модели
        
    Returns:
        tuple: (xgb_model, metrics)
    """
    logger.info("Обучение модели XGBoost")
    
    X_train, X_val, X_test, y_train, y_val, y_test = data['tabular']
    
    # Создаем модель XGBoost
    xgb_model = XGBoostModel(target_type=target_type)
    
    # Обновляем параметры, если предоставлены
    if params is not None:
        xgb_model.params.update(params)
    
    # Обучаем модель
    xgb_model.train(X_train, y_train, X_val, y_val)
    
    # Оцениваем качество на тестовой выборке
    metrics = xgb_model.evaluate(X_test, y_test)
    
    # Сохраняем модель
    model_path = xgb_model.save_model()
    logger.info(f"Модель XGBoost сохранена в {model_path}")
    
    return xgb_model, metrics


def train_lstm_model(data, target_type='binary', epochs=100, batch_size=32):
    """
    Обучение модели LSTM.
    
    Args:
        data (dict): Данные для обучения и тестирования
        target_type (str): Тип целевой переменной
        epochs (int): Количество эпох обучения
        batch_size (int): Размер батча
        
    Returns:
        tuple: (lstm_model, metrics)
    """
    logger.info("Обучение модели LSTM")
    
    X_train_seq, X_val_seq, X_test_seq, y_train_seq, y_val_seq, y_test_seq = data['sequence']
    
    # Создаем модель LSTM
    lstm_model = LSTMModel(target_type=target_type, sequence_length=X_train_seq.shape[1])
    
    # Создаем архитектуру модели
    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    lstm_model.build_model(input_shape)
    
    # Обучаем модель
    model, history = lstm_model.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=epochs, batch_size=batch_size)
    
    # Оцениваем качество на тестовой выборке
    metrics = lstm_model.evaluate(X_test_seq, y_test_seq)
    
    # Сохраняем модель
    model_path = lstm_model.save_model()
    logger.info(f"Модель LSTM сохранена в {model_path}")
    
    return lstm_model, metrics


def create_ensemble(data, xgb_model, lstm_model=None, xgb_weight=0.7, lstm_weight=0.3):
    """
    Создание ансамбля моделей.
    
    Args:
        data (dict): Данные для обучения и тестирования
        xgb_model: Обученная модель XGBoost
        lstm_model: Обученная модель LSTM
        xgb_weight (float): Вес модели XGBoost в ансамбле
        lstm_weight (float): Вес модели LSTM в ансамбле
        
    Returns:
        tuple: (ensemble_model, metrics)
    """
    logger.info("Создание ансамбля моделей")
    
    X_test, y_test = data['tabular'][2], data['tabular'][5]
    X_test_seq = data['sequence'][2]
    
    # Создаем ансамбль
    ensemble = EnsembleModel(target_type=xgb_model.target_type)
    
    # Добавляем модели в ансамбль
    ensemble.add_model('xgboost', xgb_model, weight=xgb_weight)
    
    if lstm_model is not None:
        ensemble.add_model('lstm', lstm_model, weight=lstm_weight)
    
    # Оцениваем качество на тестовой выборке
    metrics = ensemble.evaluate(X_test, y_test, X_test_seq)
    
    # Сохраняем информацию об ансамбле
    ensemble_path = ensemble.save_model()
    logger.info(f"Информация об ансамбле сохранена в {ensemble_path}")
    
    return ensemble, metrics


def main(args):
    """
    Основная функция для обучения моделей.
    
    Args:
        args: Аргументы командной строки
    """
    # Создаем директории, если они не существуют
    os.makedirs("../models", exist_ok=True)
    os.makedirs("../logs", exist_ok=True)
    
    # Подготавливаем данные
    data = prepare_data(
        period=args.period,
        target_type=args.target_type,
        horizon=args.horizon,
        sequence_length=args.sequence_length
    )
    
    if data is None:
        logger.error("Не удалось подготовить данные")
        return
    
    # Обучаем модель XGBoost
    if args.train_xgboost:
        xgb_model, xgb_metrics = train_xgboost_model(data, target_type=args.target_type)
    else:
        logger.info("Пропускаем обучение XGBoost")
        xgb_model = None
    
    # Обучаем модель LSTM
    if args.train_lstm:
        lstm_model, lstm_metrics = train_lstm_model(
            data, 
            target_type=args.target_type, 
            epochs=args.epochs, 
            batch_size=args.batch_size
        )
    else:
        logger.info("Пропускаем обучение LSTM")
        lstm_model = None
    
    # Создаем ансамбль, если обе модели обучены
    if xgb_model is not None and lstm_model is not None and args.create_ensemble:
        ensemble, ensemble_metrics = create_ensemble(
            data, 
            xgb_model, 
            lstm_model, 
            xgb_weight=args.xgb_weight, 
            lstm_weight=args.lstm_weight
        )
    elif xgb_model is not None and args.create_ensemble:
        ensemble, ensemble_metrics = create_ensemble(
            data, 
            xgb_model, 
            None, 
            xgb_weight=1.0, 
            lstm_weight=0.0
        )
    
    logger.info("Обучение моделей завершено")


if __name__ == "__main__":
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Обучение моделей для прогнозирования цены золота")
    
    parser.add_argument("--period", type=str, default="5y", help="Период для загрузки данных")
    parser.add_argument("--target_type", type=str, default="binary", choices=["binary", "regression", "classification"], 
                        help="Тип целевой переменной")
    parser.add_argument("--horizon", type=int, default=1, help="Горизонт прогнозирования (в днях)")
    parser.add_argument("--sequence_length", type=int, default=10, help="Длина последовательности для LSTM")
    
    parser.add_argument("--train_xgboost", action="store_true", help="Обучать XGBoost модель")
    parser.add_argument("--train_lstm", action="store_true", help="Обучать LSTM модель")
    parser.add_argument("--create_ensemble", action="store_true", help="Создать ансамбль моделей")
    
    parser.add_argument("--epochs", type=int, default=100, help="Количество эпох обучения LSTM")
    parser.add_argument("--batch_size", type=int, default=32, help="Размер батча для LSTM")
    
    parser.add_argument("--xgb_weight", type=float, default=0.7, help="Вес XGBoost в ансамбле")
    parser.add_argument("--lstm_weight", type=float, default=0.3, help="Вес LSTM в ансамбле")
    
    args = parser.parse_args()
    
    # Включаем обучение моделей по умолчанию, если не указаны явно
    if not args.train_xgboost and not args.train_lstm:
        args.train_xgboost = True
        args.train_lstm = True
        args.create_ensemble = True
    
    main(args)
