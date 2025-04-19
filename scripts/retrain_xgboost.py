#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для переобучения XGBoost модели на новом наборе признаков (61 фича)
Цель: обеспечить совместимость с LSTM моделью, которая уже обучена на 61 признаке
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
from models import XGBoostModel
from predict import GoldPredictor

# Настройка логирования
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'retrain_xgboost.log')

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
logger.info("Логирование настроено. Начинаем переобучение XGBoost...")

def retrain_xgboost(period='10y', target_type='binary', horizon=1, use_bybit=True, n_estimators=200, max_depth=5, learning_rate=0.1):
    """
    Переобучение XGBoost модели на новом наборе из 61 признака.
    
    Args:
        period (str): Период для загрузки данных
        target_type (str): Тип целевой переменной ('binary', 'regression', 'classification')
        horizon (int): Горизонт прогнозирования (в днях)
        use_bybit (bool): Использовать ли Bybit для обновления данных
        n_estimators (int): Количество деревьев в модели XGBoost
        max_depth (int): Максимальная глубина деревьев
        learning_rate (float): Скорость обучения
    
    Returns:
        str: Путь к сохраненной модели
    """
    logger.info(f"Запуск переобучения XGBoost на новом наборе из 61 признака (вместо 53)")
    
    # Загружаем данные
    logger.info(f"Загрузка исторических данных за {period}")
    loader = GoldDataLoader()
    data = loader.download_data(period=period)
    
    # Импортируем os на уровне функции для обеспечения видимости
    import os
    
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
            
            # API ключи тестовые (из data_updater.py), при необходимости заменить
            api_key = 'vcpsoaLUBwfj1jPfCz'
            api_secret = 'xf4WxWufuFleJuAjVWXdxRe6WHugoKPbCqQE'
            
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
    
    # Заполняем пропущенные значения
    logger.info("Проверка и заполнение пропущенных значений...")
    features_df = features_df.ffill().bfill()
    
    # Записываем список всех доступных признаков
    all_features = list(features_df.drop(['Target'], axis=1).columns)
    logger.info(f"Всего признаков для обучения: {len(all_features)}")
    
    # Разделяем данные на обучающую, валидационную и тестовую выборки
    test_size = int(len(features_df) * 0.2)  # 20% для тестирования
    val_size = int(len(features_df) * 0.1)   # 10% для валидации
    
    # Разделение данных
    train_data = features_df.iloc[:-test_size-val_size]
    val_data = features_df.iloc[-test_size-val_size:-test_size]
    test_data = features_df.iloc[-test_size:]
    
    logger.info(f"Обучающая выборка: {train_data.shape}")
    logger.info(f"Валидационная выборка: {val_data.shape}")
    logger.info(f"Тестовая выборка: {test_data.shape}")
    
    # Создаем и обучаем модель XGBoost
    logger.info(f"Создание и обучение XGBoost модели...")
    xgb_model = XGBoostModel(target_type=target_type)
    
    # Настраиваем параметры модели напрямую
    logger.info(f"Параметры XGBoost: n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}")
    
    # Обновляем параметры модели
    xgb_model.params['eta'] = learning_rate
    xgb_model.params['max_depth'] = max_depth
    
    # Подготавливаем данные для обучения XGBoost
    X_train = train_data.drop(['Target'], axis=1)
    y_train = train_data['Target']
    
    X_val = val_data.drop(['Target'], axis=1)
    y_val = val_data['Target']
    
    X_test = test_data.drop(['Target'], axis=1)
    y_test = test_data['Target']
    
    # Обучаем модель с заданным числом деревьев
    xgb_model.train(X_train, y_train, X_val=X_val, y_val=y_val, num_rounds=n_estimators)
    
    # Оцениваем модель на тестовой выборке
    logger.info("Оценка модели на тестовой выборке")
    metrics = xgb_model.evaluate(X_test, y_test)
    logger.info(f"Метрики на тестовой выборке: {metrics}")
    
    # Информация о важности признаков (если модель и список признаков доступны)
    if xgb_model.model is not None and xgb_model.feature_names is not None:
        try:
            # Получаем важность признаков через стандартный метод XGBoost
            importance_dict = {name: score for name, score in 
                             zip(xgb_model.feature_names, xgb_model.model.get_score(importance_type='weight').values())}
            # Сортируем по убыванию важности
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            # Выводим топ-10 важных признаков
            logger.info(f"Топ-10 важных признаков: {sorted_importance[:10]}")
        except Exception as e:
            logger.warning(f"Не удалось получить важность признаков: {str(e)}")
    
    # Импортируем os еще раз перед использованием, чтобы избежать ошибок области видимости
    import os
    
    # Сохраняем модель
    today = datetime.now().strftime('%Y%m%d')
    model_path = xgb_model.save_model(f"xgb_{target_type}_{today}.json")
    logger.info(f"Модель сохранена в {model_path}")
    
    # Обновляем конфигурацию
    predictor = GoldPredictor()
    predictor.config['xgb_model_path'] = os.path.basename(model_path)
    predictor.save_config()
    logger.info(f"Конфигурация обновлена: xgb_model_path = {os.path.basename(model_path)}")
    
    return model_path

if __name__ == "__main__":
    # Параметры по умолчанию
    import argparse
    parser = argparse.ArgumentParser(description="Переобучение XGBoost модели на 61 признаке")
    parser.add_argument('--period', type=str, default='10y', help='Период для загрузки данных')
    parser.add_argument('--use_bybit', action='store_true', default=True, help='Использовать Bybit для обновления данных')
    parser.add_argument('--target_type', type=str, default='binary', 
                       choices=['binary', 'regression', 'classification'], help='Тип целевой переменной')
    parser.add_argument('--horizon', type=int, default=1, help='Горизонт прогнозирования (в днях)')
    parser.add_argument('--n_estimators', type=int, default=200, help='Количество деревьев')
    parser.add_argument('--max_depth', type=int, default=5, help='Максимальная глубина деревьев')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Скорость обучения')
    
    args = parser.parse_args()
    
    # Переобучаем модель
    model_path = retrain_xgboost(
        period=args.period,
        target_type=args.target_type,
        horizon=args.horizon,
        use_bybit=args.use_bybit,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate
    )
    
    if model_path:
        print(f"\n✅ Успешно переобучена XGBoost модель на новых данных (61 признак)")
        print(f"📄 Модель сохранена в: {model_path}")
        print(f"📊 Конфигурация обновлена для использования новой модели")
    else:
        print("\n❌ Не удалось переобучить модель. Проверьте логи: logs/retrain_xgboost.log")
