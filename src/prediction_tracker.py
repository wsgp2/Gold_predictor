#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для отслеживания и анализа истории предсказаний цены золота.
"""

import os
import pandas as pd
import json
from datetime import datetime, timedelta
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PredictionTracker:
    """Класс для отслеживания истории предсказаний и расчета статистики точности."""
    
    def __init__(self, db_path="../db"):
        """
        Инициализация трекера предсказаний.
        
        Args:
            db_path (str): Путь к директории для хранения данных
        """
        self.db_path = db_path
        self.predictions_file = os.path.join(db_path, "predictions_history.json")
        self.verified_file = os.path.join(db_path, "verified_predictions.json")
        self.statistics_file = os.path.join(db_path, "prediction_statistics.json")
        
        # Создаем директорию, если она не существует
        os.makedirs(db_path, exist_ok=True)
        
        # Загружаем существующие данные или инициализируем новые
        self.predictions = self._load_json_data(self.predictions_file, [])
        self.verified = self._load_json_data(self.verified_file, [])
        self.statistics = self._load_json_data(self.statistics_file, {
            "total": 0,
            "correct": 0,
            "accuracy": 0.0,
            "by_model": {},
            "by_week": {},
            "by_month": {},
            "recent_streak": 0,
            "best_streak": 0
        })
    
    def _load_json_data(self, file_path, default_data):
        """
        Загрузка данных из JSON файла.
        
        Args:
            file_path (str): Путь к файлу
            default_data: Данные по умолчанию, если файл не существует
            
        Returns:
            Загруженные данные или default_data
        """
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Ошибка при загрузке данных из {file_path}: {e}")
                return default_data
        return default_data
    
    def _save_json_data(self, file_path, data):
        """
        Сохранение данных в JSON файл.
        
        Args:
            file_path (str): Путь к файлу
            data: Данные для сохранения
            
        Returns:
            bool: True в случае успеха, False в случае ошибки
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
            return True
        except IOError as e:
            logger.error(f"Ошибка при сохранении данных в {file_path}: {e}")
            return False
    
    def save_prediction(self, prediction_data):
        """
        Сохранение нового предсказания.
        
        Args:
            prediction_data (dict): Данные предсказания
            
        Returns:
            bool: True в случае успеха, False в случае ошибки
        """
        # Добавляем идентификатор и дату сохранения
        prediction_data["prediction_id"] = f"pred_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        prediction_data["saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Добавляем в начало списка, чтобы последние предсказания были первыми
        self.predictions.insert(0, prediction_data)
        
        # Сохраняем обновленные данные
        return self._save_json_data(self.predictions_file, self.predictions)
    
    def verify_prediction(self, prediction_date, actual_direction):
        """
        Проверка точности предсказания на основе фактического результата.
        
        Args:
            prediction_date (str): Дата предсказания (YYYY-MM-DD)
            actual_direction (str): Фактическое направление движения цены ('UP' или 'DOWN')
            
        Returns:
            bool: True в случае успеха, False в случае ошибки
        """
        # Поиск предсказания для указанной даты
        found = False
        for pred in self.predictions:
            # Проверяем оба варианта ключа даты
            pred_date = pred.get("date", pred.get("prediction_date", ""))
            if pred_date == prediction_date:
                # Копируем предсказание и добавляем фактический результат
                verified_pred = pred.copy()
                verified_pred["actual_direction"] = actual_direction
                verified_pred["is_correct"] = verified_pred.get("direction") == actual_direction
                verified_pred["verified_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Добавляем в проверенные предсказания
                self.verified.insert(0, verified_pred)
                found = True
                
                # Обновляем статистику
                self._update_statistics(verified_pred)
                break
        
        if not found:
            logger.warning(f"Предсказание для даты {prediction_date} не найдено")
            return False
        
        # Сохраняем обновленные данные
        success1 = self._save_json_data(self.verified_file, self.verified)
        success2 = self._save_json_data(self.statistics_file, self.statistics)
        
        return success1 and success2
    
    def _update_statistics(self, verified_pred):
        """
        Обновление статистики на основе нового проверенного предсказания.
        
        Args:
            verified_pred (dict): Проверенное предсказание
        """
        # Общая статистика
        self.statistics["total"] += 1
        
        if verified_pred.get("is_correct", False):
            self.statistics["correct"] += 1
            self.statistics["recent_streak"] += 1
            
            # Обновляем лучшую серию
            if self.statistics["recent_streak"] > self.statistics["best_streak"]:
                self.statistics["best_streak"] = self.statistics["recent_streak"]
        else:
            self.statistics["recent_streak"] = 0
        
        # Обновляем общую точность
        self.statistics["accuracy"] = round(self.statistics["correct"] / self.statistics["total"], 4)
        
        # Статистика по моделям
        model = verified_pred.get("model", "unknown")
        if model not in self.statistics["by_model"]:
            self.statistics["by_model"][model] = {"total": 0, "correct": 0, "accuracy": 0.0}
        
        self.statistics["by_model"][model]["total"] += 1
        if verified_pred.get("is_correct", False):
            self.statistics["by_model"][model]["correct"] += 1
        
        self.statistics["by_model"][model]["accuracy"] = round(
            self.statistics["by_model"][model]["correct"] / self.statistics["by_model"][model]["total"], 
            4
        )
        
        # Статистика по неделям
        pred_date = verified_pred.get("date", verified_pred.get("prediction_date", ""))
        if pred_date:
            try:
                dt = datetime.strptime(pred_date, "%Y-%m-%d")
                year_week = f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"
                
                if year_week not in self.statistics["by_week"]:
                    self.statistics["by_week"][year_week] = {"total": 0, "correct": 0, "accuracy": 0.0}
                
                self.statistics["by_week"][year_week]["total"] += 1
                if verified_pred.get("is_correct", False):
                    self.statistics["by_week"][year_week]["correct"] += 1
                
                self.statistics["by_week"][year_week]["accuracy"] = round(
                    self.statistics["by_week"][year_week]["correct"] / self.statistics["by_week"][year_week]["total"], 
                    4
                )
            except ValueError:
                logger.error(f"Некорректный формат даты: {pred_date}")
        
        # Статистика по месяцам
        if pred_date:
            try:
                dt = datetime.strptime(pred_date, "%Y-%m-%d")
                year_month = f"{dt.year}-{dt.month:02d}"
                
                if year_month not in self.statistics["by_month"]:
                    self.statistics["by_month"][year_month] = {"total": 0, "correct": 0, "accuracy": 0.0}
                
                self.statistics["by_month"][year_month]["total"] += 1
                if verified_pred.get("is_correct", False):
                    self.statistics["by_month"][year_month]["correct"] += 1
                
                self.statistics["by_month"][year_month]["accuracy"] = round(
                    self.statistics["by_month"][year_month]["correct"] / self.statistics["by_month"][year_month]["total"], 
                    4
                )
            except ValueError:
                # Ошибка уже залогирована в статистике по неделям
                pass
    
    def get_statistics(self):
        """
        Получение статистики предсказаний.
        
        Returns:
            dict: Статистика предсказаний
        """
        return self.statistics
    
    def get_weekly_statistics(self):
        """
        Получение еженедельной статистики предсказаний.
        
        Returns:
            dict: Еженедельная статистика
        """
        # Определяем текущую неделю
        now = datetime.now()
        current_week = f"{now.isocalendar()[0]}-W{now.isocalendar()[1]:02d}"
        
        # Получаем статистику за текущую неделю
        current_week_stats = self.statistics["by_week"].get(current_week, {"total": 0, "correct": 0, "accuracy": 0.0})
        
        return {
            "current_week": current_week,
            "statistics": current_week_stats,
            "recent_predictions": self._get_recent_verified_predictions(7)  # За последние 7 дней
        }
    
    def get_monthly_statistics(self):
        """
        Получение ежемесячной статистики предсказаний.
        
        Returns:
            dict: Ежемесячная статистика
        """
        # Определяем текущий месяц
        now = datetime.now()
        current_month = f"{now.year}-{now.month:02d}"
        
        # Получаем статистику за текущий месяц
        current_month_stats = self.statistics["by_month"].get(current_month, {"total": 0, "correct": 0, "accuracy": 0.0})
        
        return {
            "current_month": current_month,
            "statistics": current_month_stats,
            "recent_predictions": self._get_recent_verified_predictions(30)  # За последние 30 дней
        }
    
    def _get_recent_verified_predictions(self, days=7):
        """
        Получение последних проверенных предсказаний.
        
        Args:
            days (int): Количество дней для выборки
            
        Returns:
            list: Список последних проверенных предсказаний
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        recent_predictions = []
        for pred in self.verified:
            pred_date = pred.get("date", pred.get("prediction_date", ""))
            if pred_date >= cutoff_date:
                recent_predictions.append(pred)
            
            # Ограничиваем количество предсказаний для производительности
            if len(recent_predictions) >= 30:
                break
        
        return recent_predictions
    
    def get_prediction_by_date(self, prediction_date):
        """
        Получение предсказания для указанной даты.
        
        Args:
            prediction_date (str): Дата предсказания (YYYY-MM-DD)
            
        Returns:
            dict: Предсказание или None, если не найдено
        """
        for pred in self.predictions:
            # Проверяем оба варианта ключа даты
            pred_date = pred.get("date", pred.get("prediction_date", ""))
            if pred_date == prediction_date:
                return pred
        
        return None
    
    def get_verified_prediction_by_date(self, prediction_date):
        """
        Получение проверенного предсказания для указанной даты.
        
        Args:
            prediction_date (str): Дата предсказания (YYYY-MM-DD)
            
        Returns:
            dict: Проверенное предсказание или None, если не найдено
        """
        for pred in self.verified:
            # Проверяем оба варианта ключа даты
            pred_date = pred.get("date", pred.get("prediction_date", ""))
            if pred_date == prediction_date:
                return pred
        
        return None
    
    def generate_weekly_report(self):
        """
        Генерация еженедельного отчета по предсказаниям.
        
        Returns:
            str: Текст отчета
        """
        weekly_stats = self.get_weekly_statistics()
        stats = weekly_stats["statistics"]
        
        # Формируем отчет
        report = f"*📊 Еженедельный отчет за {weekly_stats['current_week']}*\n\n"
        
        if stats["total"] > 0:
            report += f"Всего предсказаний: {stats['total']}\n"
            report += f"Верных предсказаний: {stats['correct']} ({stats['accuracy'] * 100:.1f}%)\n\n"
            
            # Добавляем статистику по моделям
            report += "*По моделям:*\n"
            for model, model_stats in self.statistics["by_model"].items():
                if model_stats["total"] > 0:
                    model_icon = {
                        "xgboost": "🌲",
                        "lstm": "🧠",
                        "ensemble": "⚖️"
                    }.get(model.lower(), "🔮")
                    report += f"{model_icon} *{model.capitalize()}*: {model_stats['correct']}/{model_stats['total']} ({model_stats['accuracy'] * 100:.1f}%)\n"
            
            # Добавляем серии успешных предсказаний
            report += f"\n*🔥 Текущая серия:* {self.statistics['recent_streak']} предсказаний\n"
            report += f"*🏆 Лучшая серия:* {self.statistics['best_streak']} предсказаний\n"
            
            # Добавляем последние предсказания
            recent_preds = weekly_stats["recent_predictions"]
            if recent_preds:
                report += "\n*Последние предсказания:*\n"
                for pred in recent_preds[:5]:  # Показываем только 5 последних
                    date = pred.get("date", pred.get("prediction_date", ""))
                    is_correct = pred.get("is_correct", False)
                    direction = pred.get("direction", "")
                    
                    icon = "✅" if is_correct else "❌"
                    direction_icon = "🔼" if direction == "UP" else "🔽"
                    
                    report += f"{icon} {date}: {direction_icon} {direction}\n"
        else:
            report += "На этой неделе еще нет проверенных предсказаний."
        
        return report


# Запуск для тестирования
if __name__ == "__main__":
    tracker = PredictionTracker()
    
    # Добавляем тестовое предсказание
    test_prediction = {
        "date": "2025-04-17",
        "prediction_date": "2025-04-18",
        "current_price": 3308.70,
        "direction": "UP",
        "confidence": 0.6,
        "model": "ensemble",
        "predictions": {
            "xgboost": {"direction": "DOWN", "confidence": 0.3},
            "lstm": {"direction": "UP", "confidence": 0.9},
            "ensemble": {"direction": "UP", "confidence": 0.6}
        }
    }
    
    tracker.save_prediction(test_prediction)
    
    # Проверяем предсказание (имитируем фактический результат)
    tracker.verify_prediction("2025-04-18", "UP")
    
    # Выводим статистику
    stats = tracker.get_statistics()
    print(f"Общая статистика: {stats['correct']}/{stats['total']} ({stats['accuracy'] * 100:.1f}%)")
    
    # Генерируем еженедельный отчет
    report = tracker.generate_weekly_report()
    print(report)
