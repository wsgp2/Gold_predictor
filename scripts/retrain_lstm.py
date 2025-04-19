#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для переобучения LSTM модели на новом наборе признаков (61 фича)
Цель: исправить несоответствие размерностей между данными и весами модели
"""

import os
import sys
import logging
from datetime import datetime

# Добавляем каталог src в путь для импорта
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Импортируем наши модули
from data_loader import GoldDataLoader
from features import FeatureGenerator
from models import LSTMModel
from predict import GoldPredictor

# Настройка логирования - исправленная версия
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'retrain_lstm.log')

# Удаляем существующий лог-файл, если он есть (может быть проблема с правами доступа)
if os.path.exists(log_file_path):
    try:
        os.remove(log_file_path)
    except Exception as e:
        print(f"Не удалось удалить старый лог-файл: {str(e)}")

# Сбрасываем конфигурацию логгера для всей библиотеки logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Создаем форматирование логов
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Создаем файловый обработчик с явным указанием мода записи
file_handler = logging.FileHandler(log_file_path, mode='w')
file_handler.setFormatter(formatter)

# Создаем консольный обработчик
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Конфигурируем корневой логгер
logging.root.setLevel(logging.INFO)
logging.root.addHandler(file_handler)
logging.root.addHandler(console_handler)

# Создаем специфичный логгер для данного модуля
logger = logging.getLogger(__name__)
logger.info("Логирование настроено. Начинаем переобучение LSTM...")

def retrain_lstm(period='10y', target_type='binary', horizon=1, sequence_length=10, epochs=100, batch_size=32, use_bybit=True):
    """
    Переобучение LSTM модели на новом наборе из 61 признака.
    
    Args:
        period (str): Период для загрузки данных
        target_type (str): Тип целевой переменной ('binary', 'regression', 'classification')
        horizon (int): Горизонт прогнозирования (в днях)
        sequence_length (int): Длина последовательности для LSTM
        epochs (int): Количество эпох обучения
        batch_size (int): Размер батча для обучения
    
    Returns:
        str: Путь к сохраненной модели
    """
    logger.info(f"Запуск переобучения LSTM на новом наборе из 61 признака (вместо 53)")
    
    # Загружаем данные
    logger.info(f"Загрузка исторических данных за {period}")
    loader = GoldDataLoader()
    data = loader.download_data(period=period)
    
    # Обновляем данные с Bybit до текущего дня (если нужно)
    if use_bybit:
        logger.info("Обновление данных через Bybit API для получения самых свежих цен")
        try:
            data_path = os.path.join(loader.data_dir, 'GC_F_latest.csv')
            
            # Сначала сохраняем текущие данные, если их нет
            if not os.path.exists(data_path):
                loader.save_data(data, 'GC_F_latest.csv')
                
            # Импортируем update_gold_history_from_bybit из модуля data_updater
            from data_updater import update_gold_history_from_bybit
            
            # Загружаем переменные окружения для API ключей
            import os
            from config_loader import load_environment_variables
            load_environment_variables()
            
            api_key = os.getenv('BYBIT_API_KEY', '')
            api_secret = os.getenv('BYBIT_API_SECRET', '')
            
            if not api_key or not api_secret:
                logger.error("Отсутствуют API ключи Bybit, проверьте файл .env")
                logger.info("Продолжаем без обновления данных Bybit...")
            else:
                # Обновляем данные через Bybit
                update_success = update_gold_history_from_bybit(data_path, api_key, api_secret)
            
            if update_success:
                # Загружаем обновленные данные
                updated_data = loader.load_data('GC_F_latest.csv')
                if updated_data is not None and not updated_data.empty:
                    logger.info(f"Данные успешно обновлены с Bybit до {updated_data.index[-1]}")
                    data = updated_data
        except Exception as e:
            logger.warning(f"Не удалось обновить данные через Bybit: {str(e)}")
            logger.info("Продолжаем с имеющимися данными из Yahoo Finance")
    
    if data is None or len(data) < 100:
        logger.error("Недостаточно данных для обучения")
        return None
    
    logger.info(f"Загружено {len(data)} строк данных с {data.index[0]} по {data.index[-1]}")
    
    # Генерируем признаки
    feature_gen = FeatureGenerator(scaling_method='standard')
    
    # Подготавливаем признаки для модели
    features_df = feature_gen.prepare_features(data, horizon=horizon, target_type=target_type)
    
    # Подготавливаем данные для LSTM (последовательности) - модифицированный подход
    logger.info(f"Генерация последовательностей для LSTM длиной {sequence_length}")
    
    # Вместо вызова prepare_sequence_data, который создает пустые последовательности,
    # мы вручную создадим последовательности из уже обработанных данных
    # Это позволит избежать повторной обработки и потери данных
    
    # Проверяем наличие пропущенных значений и заполняем их
    logger.info("Проверка и заполнение пропущенных значений...")
    features_df = features_df.ffill().bfill()  # Используем методы вместо fillna
    
    # Получаем числовые колонки (признаки)
    feature_columns = features_df.drop(['Target'], axis=1).columns
    X_data = features_df[feature_columns].values
    y_data = features_df['Target'].values
    
    # Создаем последовательности вручную
    import numpy as np
    sequences = []
    targets = []
    
    for i in range(len(X_data) - sequence_length):
        # Создаем последовательность длиной sequence_length
        seq = X_data[i:i+sequence_length]
        # Целевое значение - следующий после последовательности
        target = y_data[i+sequence_length]
        sequences.append(seq)
        targets.append(target)
    
    # Преобразуем в numpy массивы
    X_sequences = np.array(sequences)
    y_targets = np.array(targets)
    
    # Проверка на пустые последовательности
    if len(sequences) == 0:
        logger.error("Ни одной последовательности не создано. Проверьте данные и параметры.")
        return None
    
    # Проверяем размерность данных
    n_features = X_sequences.shape[2]
    logger.info(f"Размерность данных для LSTM: {X_sequences.shape}, количество признаков: {n_features}")
    
    # Делим данные на обучающую, валидационную и тестовую выборки
    logger.info("Разделение данных на обучающую, валидационную и тестовую выборки")
    seq_test_size = int(len(X_sequences) * 0.2)  # 20% данных для теста
    seq_val_size = int(len(X_sequences) * 0.1)   # 10% данных для валидации
    
    X_train = X_sequences[:-seq_test_size-seq_val_size]
    y_train = y_targets[:-seq_test_size-seq_val_size]
    
    X_val = X_sequences[-seq_test_size-seq_val_size:-seq_test_size]
    y_val = y_targets[-seq_test_size-seq_val_size:-seq_test_size]
    
    X_test = X_sequences[-seq_test_size:]
    y_test = y_targets[-seq_test_size:]
    
    logger.info(f"Обучающая выборка: {X_train.shape}, {y_train.shape}")
    logger.info(f"Валидационная выборка: {X_val.shape}, {y_val.shape}")
    logger.info(f"Тестовая выборка: {X_test.shape}, {y_test.shape}")
    
    # Создаем и обучаем модель
    logger.info(f"Создание и обучение LSTM модели...")
    lstm_model = LSTMModel(target_type=target_type, sequence_length=sequence_length)
    
    # Строим модель с правильной размерностью признаков
    input_shape = (sequence_length, n_features)
    lstm_model.build_model(input_shape)
    
    # Обучаем модель
    lstm_model.train(
        X_train, y_train, 
        X_val=X_val, y_val=y_val,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Оцениваем модель на тестовой выборке
    logger.info("Оценка модели на тестовой выборке")
    metrics = lstm_model.evaluate(X_test, y_test)
    logger.info(f"Метрики на тестовой выборке: {metrics}")
    
    # Сохраняем модель
    today = datetime.now().strftime('%Y%m%d')
    model_path = lstm_model.save_model(f"lstm_{target_type}_{today}.h5")
    logger.info(f"Модель сохранена в {model_path}")
    
    # Обновляем конфигурацию
    predictor = GoldPredictor()
    predictor.config['lstm_model_path'] = os.path.basename(model_path)
    predictor.save_config()
    logger.info(f"Конфигурация обновлена: lstm_model_path = {os.path.basename(model_path)}")
    
    return model_path

if __name__ == "__main__":
    # Параметры по умолчанию
    PARAMS = {
        'period': '10y',           # Период для загрузки данных
        'target_type': 'binary',  # Тип целевой переменной
        'horizon': 1,             # Горизонт прогнозирования (в днях)
        'sequence_length': 10,    # Длина последовательности для LSTM
        'epochs': 100,            # Количество эпох обучения
        'batch_size': 32          # Размер батча для обучения
    }
    
    # Переопределение параметров из аргументов командной строки
    import argparse
    parser = argparse.ArgumentParser(description="Переобучение LSTM модели на 61 признаке")
    parser.add_argument('--period', type=str, default=PARAMS['period'], help='Период для загрузки данных')
    parser.add_argument('--use_bybit', action='store_true', default=True, help='Использовать Bybit для обновления данных')
    parser.add_argument('--target_type', type=str, default=PARAMS['target_type'], 
                       choices=['binary', 'regression', 'classification'], help='Тип целевой переменной')
    parser.add_argument('--horizon', type=int, default=PARAMS['horizon'], help='Горизонт прогнозирования (в днях)')
    parser.add_argument('--sequence_length', type=int, default=PARAMS['sequence_length'], 
                       help='Длина последовательности для LSTM')
    parser.add_argument('--epochs', type=int, default=PARAMS['epochs'], help='Количество эпох обучения')
    parser.add_argument('--batch_size', type=int, default=PARAMS['batch_size'], help='Размер батча для обучения')
    
    args = parser.parse_args()
    
    # Переобучаем модель
    model_path = retrain_lstm(
        period=args.period,
        target_type=args.target_type,
        horizon=args.horizon,
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_bybit=args.use_bybit
    )
    
    if model_path:
        print(f"\n✅ Успешно переобучена LSTM модель на новых данных (61 признак)")
        print(f"📄 Модель сохранена в: {model_path}")
        print(f"📊 Конфигурация обновлена для использования новой модели")
    else:
        print("\n❌ Не удалось переобучить модель. Проверьте логи: logs/retrain_lstm.log")
