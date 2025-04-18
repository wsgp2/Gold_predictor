#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Главный скрипт для запуска прогнозирования цены золота.
Запускает прогнозирование и проверку, а также настраивает регулярные запуски по расписанию.
"""

import os
import sys
import logging
import argparse
import json
import schedule
import time
from datetime import datetime, timedelta

# Настраиваем корневую директорию проекта
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

# Добавляем src/ в PYTHONPATH для корректного импорта модулей
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# Настройка логирования
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/main.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Импортируем после настройки директорий
from predict import GoldPredictor

def setup_predictor(config_path=None):
    """
    Настройка и инициализация предсказателя цены золота.
    
    Args:
        config_path (str, optional): Путь к файлу конфигурации
        
    Returns:
        GoldPredictor: Настроенный объект предсказателя
    """
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "config", "predictor_config.json")
    
    # Создаем директорию для конфигурации, если она не существует
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Создаем предсказатель
    predictor = GoldPredictor(
        model_dir=os.path.join(PROJECT_ROOT, "models"),
        data_dir=os.path.join(PROJECT_ROOT, "data"),
        config_path=config_path
    )
    
    return predictor

def run_prediction(predictor=None):
    """
    Запуск прогнозирования цены золота.
    
    Args:
        predictor (GoldPredictor, optional): Предсказатель
        
    Returns:
        bool: True, если прогноз выполнен успешно
    """
    if predictor is None:
        predictor = setup_predictor()
    
    logger.info("Запуск процесса прогнозирования цены золота")
    
    # Выполняем прогнозирование
    success = predictor.run_prediction()
    
    if success:
        logger.info("Прогнозирование выполнено успешно")
    else:
        logger.error("Ошибка при прогнозировании")
    
    return success

def run_verification(prediction_date=None, predictor=None):
    """
    Запуск проверки предыдущего прогноза цены золота.
    
    Args:
        prediction_date (str, optional): Дата прогноза для проверки
        predictor (GoldPredictor, optional): Предсказатель
        
    Returns:
        bool: True, если проверка выполнена успешно
    """
    if predictor is None:
        predictor = setup_predictor()
    
    if prediction_date is None:
        # Проверяем прогноз за вчерашний день
        yesterday = datetime.now() - timedelta(days=1)
        prediction_date = yesterday.strftime('%Y-%m-%d')
    
    logger.info(f"Запуск проверки прогноза цены золота на {prediction_date}")
    
    # Выполняем проверку
    success = predictor.run_verification(prediction_date)
    
    if success:
        logger.info(f"Проверка прогноза на {prediction_date} выполнена успешно")
    else:
        logger.error(f"Ошибка при проверке прогноза на {prediction_date}")
    
    return success

def configure_telegram(token=None, chat_id=None, predictor=None):
    """
    Настройка параметров Telegram бота.
    
    Args:
        token (str, optional): Токен Telegram бота
        chat_id (str, optional): ID чата или канала
        predictor (GoldPredictor, optional): Предсказатель
        
    Returns:
        bool: True, если настройка выполнена успешно
    """
    if predictor is None:
        predictor = setup_predictor()
    
    if token:
        predictor.update_config(telegram_token=token)
    
    if chat_id:
        predictor.update_config(telegram_chat_id=chat_id)
    
    # Проверяем наличие необходимых параметров
    if not predictor.config["telegram_token"] or not predictor.config["telegram_chat_id"]:
        logger.error("Не настроены параметры Telegram (требуются токен и chat_id)")
        return False
    
    logger.info("Параметры Telegram настроены успешно")
    return True

def configure_schedule(prediction_time="10:00", verification_time="10:30", predictor=None):
    """
    Настройка расписания для прогнозирования и проверки.
    
    Args:
        prediction_time (str): Время для ежедневного прогнозирования (HH:MM)
        verification_time (str): Время для ежедневной проверки прогноза (HH:MM)
        predictor (GoldPredictor, optional): Предсказатель
        
    Returns:
        bool: True, если расписание настроено успешно
    """
    if predictor is None:
        predictor = setup_predictor()
    
    try:
        # Обновляем конфигурацию
        predictor.update_config(
            prediction_time=prediction_time,
            verification_time=verification_time
        )
        
        # Настраиваем расписание для прогнозирования
        schedule.every().day.at(prediction_time).do(run_prediction, predictor=predictor)
        logger.info(f"Ежедневное прогнозирование запланировано на {prediction_time}")
        
        # Настраиваем расписание для проверки
        schedule.every().day.at(verification_time).do(run_verification, predictor=predictor)
        logger.info(f"Ежедневная проверка запланирована на {verification_time}")
        
        return True
    except Exception as e:
        logger.error(f"Ошибка при настройке расписания: {e}")
        return False

def run_scheduler():
    """
    Запуск планировщика для выполнения задач по расписанию.
    
    Returns:
        None
    """
    logger.info("Запуск планировщика задач")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Проверка каждую минуту
    except KeyboardInterrupt:
        logger.info("Планировщик остановлен пользователем")
    except Exception as e:
        logger.error(f"Ошибка в планировщике: {e}")

def main():
    """
    Главная функция для запуска прогнозирования цены золота.
    
    Returns:
        None
    """
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Прогнозирование цены золота и отправка результатов в Telegram")
    
    parser.add_argument("--config", type=str, help="Путь к файлу конфигурации")
    parser.add_argument("--telegram-token", type=str, help="Токен Telegram бота")
    parser.add_argument("--telegram-chat", type=str, help="ID чата или канала Telegram")
    
    parser.add_argument("--predict", action="store_true", help="Запустить прогнозирование")
    parser.add_argument("--verify", action="store_true", help="Запустить проверку прогноза")
    parser.add_argument("--date", type=str, help="Дата прогноза для проверки (YYYY-MM-DD)")
    
    parser.add_argument("--schedule", action="store_true", help="Запустить планировщик")
    parser.add_argument("--prediction-time", type=str, default="10:00", help="Время для ежедневного прогнозирования (HH:MM)")
    parser.add_argument("--verification-time", type=str, default="10:30", help="Время для ежедневной проверки (HH:MM)")
    
    args = parser.parse_args()
    
    # Настраиваем предсказатель
    predictor = setup_predictor(args.config)
    
    # Настраиваем Telegram
    if args.telegram_token or args.telegram_chat:
        configure_telegram(args.telegram_token, args.telegram_chat, predictor)
    
    # Выполняем действия в зависимости от указанных аргументов
    if args.predict:
        run_prediction(predictor)
    
    if args.verify:
        run_verification(args.date, predictor)
    
    # Настраиваем расписание
    if args.schedule:
        configure_schedule(args.prediction_time, args.verification_time, predictor)
        run_scheduler()

if __name__ == "__main__":
    main()
