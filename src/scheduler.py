#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Планировщик для автоматического запуска предсказаний Gold Price Predictor
каждый день в заданное время.
"""

import os
import time
import logging
import json
import traceback
import argparse
from datetime import datetime, timedelta
import schedule
import threading

from predict import GoldPredictor
from prediction_tracker import PredictionTracker

# Настройка логирования
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'scheduler.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GoldPredictorScheduler:
    """Планировщик для автоматического запуска предсказаний Gold Price Predictor."""
    
    def __init__(self, config_path="../config/predictor_config.json"):
        """
        Инициализация планировщика.
        
        Args:
            config_path (str): Путь к конфигурационному файлу
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Инициализация предиктора
        self.predictor = GoldPredictor(config_path=config_path)
        
        # Трекер для статистики предсказаний
        self.tracker = PredictionTracker()
        
        # Планировщик
        self.scheduler = schedule
        
        # Параметры автоматической работы
        self.running = False
        self.thread = None
    
    def _load_config(self):
        """
        Загрузка конфигурации из файла.
        
        Returns:
            dict: Конфигурация
        """
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Ошибка при загрузке конфигурации: {e}")
                return {}
        else:
            logger.error(f"Файл конфигурации не найден: {self.config_path}")
            return {}
    
    def update_config(self):
        """Обновление конфигурации из файла."""
        self.config = self._load_config()
        
        # Перенастраиваем планировщик с новыми параметрами
        self.scheduler.clear()
        self._schedule_tasks()
    
    def _schedule_tasks(self):
        """Настройка расписания задач."""
        # Время предсказания
        prediction_time = self.config.get("prediction_time", "10:00")
        logger.info(f"Настроено расписание предсказаний на {prediction_time} ежедневно")
        self.scheduler.every().day.at(prediction_time).do(self.generate_prediction)
        
        # Время верификации предсказаний
        verification_time = self.config.get("verification_time", "10:00")
        logger.info(f"Настроено расписание верификации на {verification_time} ежедневно")
        self.scheduler.every().day.at(verification_time).do(self.verify_prediction)
    
    def generate_prediction(self):
        """
        Генерация прогноза и отправка в Telegram.
        
        Returns:
            bool: Результат выполнения
        """
        try:
            logger.info("🔄 Проверка актуальности исторических данных через Bybit...")
            from data_updater import update_gold_history_from_bybit
            # Загружаем переменные окружения
            import os
            from config_loader import load_environment_variables
            load_environment_variables()
            
            api_key = os.getenv('BYBIT_API_KEY', '')
            api_secret = os.getenv('BYBIT_API_SECRET', '')
            
            if not api_key or not api_secret:
                logger.error("Отсутствуют API ключи Bybit, проверьте файл .env")
                return False
            
            update_result = update_gold_history_from_bybit(
                os.path.join(self.predictor.data_dir, 'GC_F_latest.csv'),
                api_key,  # Bybit API KEY из переменной окружения
                api_secret  # Bybit API SECRET из переменной окружения
            )
            if update_result:
                logger.info("✅ Исторические данные успешно обновлены!")
            else:
                logger.warning("⚠️ Не удалось обновить исторические данные!")

            logger.info("🔮 Запуск генерации прогноза...")
            
            # Создаем аргументы для предиктора
            class Args:
                def __init__(self):
                    self.model = 'ensemble'
                    self.target_type = 'binary'
                    self.horizon = 1
                    self.print_proba = False
                    self.send_telegram = False  # Отключаем отправку в предикторе, будем отправлять сами
                    self.config = '../config/predictor_config.json'
            
            self.predictor.args = Args()
            
            # Генерируем прогноз
            result = self.predictor.predict()
            
            if result:
                logger.info(f"✅ Прогноз успешно сгенерирован на {result.get('date', result.get('prediction_date', ''))}")
                logger.info(f"Направление: {result.get('direction')} с уверенностью {result.get('confidence'):.2f}")
                
                # Отправляем сообщение в Telegram
                try:
                    self.predictor._send_prediction_to_telegram(result)
                    logger.info(f"✉️ Прогноз отправлен в Telegram")
                except Exception as e:
                    logger.error(f"❌ Ошибка при отправке в Telegram: {e}")
                    
                return True
            else:
                logger.error("❌ Не удалось сгенерировать прогноз")
                return False
        except Exception as e:
            logger.error(f"❌ Ошибка при генерации прогноза: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def verify_prediction(self):
        """
        Верификация предыдущего прогноза на основе текущих данных.
        
        Returns:
            bool: Результат выполнения
        """
        try:
            # Получаем вчерашнюю дату
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            logger.info(f"🧪 Верификация прогноза для даты {yesterday}")
            
            # Получаем последние данные
            data = self.predictor.prepare_latest_data()
            if data is None:
                logger.error("❌ Не удалось получить последние данные для верификации")
                return False
            
            # Получаем актуальную информацию о ценах
            last_close = data['last_close']
            prev_close = data.get('prev_close', None)
            
            if prev_close is None:
                logger.error("❌ Нет информации о предыдущей цене для верификации")
                return False
            
            # Определяем фактическое направление
            actual_direction = "UP" if last_close > prev_close else "DOWN"
            logger.info(f"Фактическое направление: {actual_direction} (last: {last_close}, prev: {prev_close})")
            
            # Верифицируем прогноз в трекере
            verified = self.tracker.verify_prediction(yesterday, actual_direction)
            
            if verified:
                logger.info(f"✅ Прогноз для {yesterday} успешно верифицирован")
                
                # Получаем обновленную статистику и отправляем отчет, если есть успешные/неуспешные серии
                stats = self.tracker.get_statistics()
                if stats.get("recent_streak", 0) >= 3:
                    # Отправляем отчет о серии успешных прогнозов
                    streak_message = f"*🔥 Серия успешных прогнозов: {stats['recent_streak']}*\n\n"
                    streak_message += f"Текущая серия успешных прогнозов достигла {stats['recent_streak']} подряд!\n"
                    streak_message += f"Общая точность: {stats['accuracy'] * 100:.1f}%"
                    
                    self.predictor.send_telegram_message(streak_message)
                
                # Еженедельный отчет по воскресеньям
                if datetime.now().weekday() == 6:  # Воскресенье
                    weekly_report = self.tracker.generate_weekly_report()
                    self.predictor.send_telegram_message(weekly_report)
                
                return True
            else:
                logger.error(f"❌ Не удалось верифицировать прогноз для {yesterday}")
                return False
        except Exception as e:
            logger.error(f"❌ Ошибка при верификации прогноза: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def start(self):
        """Запуск планировщика."""
        self._schedule_tasks()
        
        self.running = True
        self.thread = threading.Thread(target=self._run_scheduler)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info("🚀 Планировщик Gold Price Predictor запущен")
        
        # Проверяем, есть ли запланированные задачи и когда они выполнятся
        next_runs = []
        for job in self.scheduler.get_jobs():
            next_run = job.next_run
            next_runs.append((job.job_func.__name__, next_run))
            logger.info(f"📅 Задача {job.job_func.__name__} запланирована на {next_run}")
        
        return True
    
    def _run_scheduler(self):
        """Фоновый запуск планировщика."""
        while self.running:
            self.scheduler.run_pending()
            time.sleep(1)
    
    def stop(self):
        """Остановка планировщика."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        self.scheduler.clear()
        logger.info("🛑 Планировщик Gold Price Predictor остановлен")
        return True


def run_as_service():
    """Запуск планировщика как сервиса."""
    scheduler = GoldPredictorScheduler()
    scheduler.start()
    
    # Запускаем первичное предсказание
    scheduler.generate_prediction()
    
    try:
        while True:
            time.sleep(60)  # Проверка каждую минуту
    except KeyboardInterrupt:
        logger.info("Остановка планировщика по запросу пользователя...")
        scheduler.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gold Price Predictor Scheduler")
    parser.add_argument("--run_now", action="store_true", help="Запустить прогноз прямо сейчас")
    parser.add_argument("--verify_now", action="store_true", help="Запустить верификацию прямо сейчас")
    parser.add_argument("--service", action="store_true", help="Запустить как сервис")
    parser.add_argument("--config", type=str, default="../config/predictor_config.json", help="Путь к конфигу")
    
    args = parser.parse_args()
    
    scheduler = GoldPredictorScheduler(config_path=args.config)
    
    if args.run_now:
        logger.info("Запуск прогноза вручную...")
        scheduler.generate_prediction()
    elif args.verify_now:
        logger.info("Запуск верификации вручную...")
        scheduler.verify_prediction()
    elif args.service:
        logger.info("Запуск планировщика как сервиса...")
        run_as_service()
    else:
        scheduler.start()
        print("Планировщик запущен. Нажмите Ctrl+C для остановки.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            scheduler.stop()
