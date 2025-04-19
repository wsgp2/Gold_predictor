#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для работы с Telegram-ботом Gold Price Predictor.
Реализует интерактивное взаимодействие через кнопки и команды.
"""

import os
import logging
import json
from datetime import datetime, timedelta
import asyncio
import asyncio
import schedule
import time
import threading
from datetime import datetime, timedelta

from data_updater import update_gold_history_from_bybit

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

from predict import GoldPredictor
from prediction_tracker import PredictionTracker

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GoldPredictorBot:
    """Класс для управления Telegram-ботом прогнозирования цены золота."""
    
    def __init__(self, config_path="config/predictor_config.json"):
        """
        Инициализация бота.
        
        Args:
            config_path (str): Путь к конфигурационному файлу
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        if not self.config.get("telegram_token"):
            raise ValueError("Не указан токен Telegram-бота в конфигурации")
        
        # Инициализация компонентов
        self.predictor = GoldPredictor(config_path=config_path)
        self.tracker = PredictionTracker()
        
        # Инициализация бота
        self.application = Application.builder().token(self.config["telegram_token"]).build()
        
        # Регистрация обработчиков
        self._register_handlers()
        
        # Планировщик для отправки прогнозов по расписанию
        self.scheduler = schedule
        self.scheduler_thread = None
        self.running = False
    
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
            except json.JSONDecodeError as e:
                logger.error(f"Ошибка при загрузке конфигурации: {e}")
                return {}
        else:
            logger.error(f"Файл конфигурации не найден: {self.config_path}")
            return {}
    
    def _register_handlers(self):
        """Регистрация обработчиков команд и callback-запросов."""
        # Команды
        self.application.add_handler(CommandHandler("start", self.cmd_start))
        self.application.add_handler(CommandHandler("help", self.cmd_help))
        self.application.add_handler(CommandHandler("predict", self.cmd_predict))
        self.application.add_handler(CommandHandler("stats", self.cmd_stats))
        self.application.add_handler(CommandHandler("settings", self.cmd_settings))
        
        # Обработчики callback-запросов
        self.application.add_handler(CallbackQueryHandler(self.cb_predict, pattern="^predict"))
        self.application.add_handler(CallbackQueryHandler(self.cb_stats, pattern="^stats$"))
        self.application.add_handler(CallbackQueryHandler(self.cb_settings, pattern="^settings"))
        self.application.add_handler(CallbackQueryHandler(self.cb_model, pattern="^model"))
        self.application.add_handler(CallbackQueryHandler(self.cb_horizon, pattern="^horizon"))
        
        # Добавляем обработчики для кнопок возврата в меню и статистики
        self.application.add_handler(CallbackQueryHandler(self.cb_start, pattern="^start"))
        self.application.add_handler(CallbackQueryHandler(self.cb_help, pattern="^help"))
        self.application.add_handler(CallbackQueryHandler(self.cb_stats_weekly, pattern="^stats_weekly"))
        self.application.add_handler(CallbackQueryHandler(self.cb_stats_monthly, pattern="^stats_monthly"))
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start."""
        keyboard = [
            [
                InlineKeyboardButton("🔮 Получить прогноз", callback_data="predict"),
                InlineKeyboardButton("📊 Статистика", callback_data="stats")
            ],
            [
                InlineKeyboardButton("⚙️ Настройки", callback_data="settings"),
                InlineKeyboardButton("❓ Помощь", callback_data="help")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message = (
            "*🌟 Gold Price Predictor Bot*\n\n"
            "Добро пожаловать! Этот бот предсказывает движение цены золота с использованием "
            "моделей машинного обучения.\n\n"
            "Выберите действие из меню ниже:"
        )
        
        await update.message.reply_text(message, reply_markup=reply_markup, parse_mode="Markdown")
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help."""
        message = (
            "*📚 Справка по Gold Price Predictor Bot*\n\n"
            "*Доступные команды:*\n"
            "• /start - Запуск бота и главное меню\n"
            "• /predict - Получить прогноз цены золота\n"
            "• /stats - Просмотр статистики прогнозов\n"
            "• /settings - Настройки прогнозирования\n"
            "• /help - Показать эту справку\n\n"
            "*О боте:*\n"
            "Бот анализирует исторические данные о цене золота и "
            "использует модели машинного обучения для предсказания "
            "направления движения цены (вверх или вниз).\n\n"
            "*Доступные модели:*\n"
            "• 🌲 XGBoost - Модель на основе деревьев решений\n"
            "• 🧠 LSTM - Нейронная сеть с длинной краткосрочной памятью\n"
            "• ⚖️ Ensemble - Ансамбль из нескольких моделей\n\n"
            "*Контакты:*\n"
            "По всем вопросам обращайтесь к разработчику бота."
        )
        
        keyboard = [
            [InlineKeyboardButton("🔮 Получить прогноз", callback_data="predict")],
            [InlineKeyboardButton("« Назад в меню", callback_data="start")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(message, reply_markup=reply_markup, parse_mode="Markdown")
    
    async def cmd_predict(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /predict."""
        await self._generate_prediction(update, context)
    
    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /stats."""
        await self._show_statistics(update, context)
    
    async def cmd_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /settings."""
        await self._show_settings(update, context)
    
    async def cb_predict(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик callback-запроса для получения прогноза."""
        query = update.callback_query
        await query.answer()
        
        if query.data == "predict":
            await self._generate_prediction(update, context, is_callback=True)
        
    async def cb_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик callback-запроса для просмотра статистики."""
        query = update.callback_query
        await query.answer()
        
        if query.data == "stats":
            await self._show_statistics(update, context, is_callback=True)
        elif query.data == "stats_weekly":
            await self._show_weekly_statistics(update, context, is_callback=True)
        elif query.data == "stats_monthly":
            await self._show_monthly_statistics(update, context, is_callback=True)
    
    async def cb_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик callback-запроса для настроек."""
        query = update.callback_query
        await query.answer()
        
        if query.data == "settings":
            await self._show_settings(update, context, is_callback=True)
    
    async def cb_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик callback-запроса для выбора модели."""
        query = update.callback_query
        await query.answer()
        
        model = query.data.split("_")[1]
        self.predictor.update_config(model_type=model)
        
        message = f"*⚙️ Модель изменена на:* {model.upper()}"
        
        keyboard = [
            [InlineKeyboardButton("« Назад к настройкам", callback_data="settings")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(message, reply_markup=reply_markup, parse_mode="Markdown")
    
    async def cb_horizon(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик callback-запроса для выбора горизонта прогнозирования."""
        query = update.callback_query
        await query.answer()
        
        horizon = int(query.data.split("_")[1])
        self.predictor.update_config(horizon=horizon)
        
        message = f"*⚙️ Горизонт прогнозирования изменен на:* {horizon} {'день' if horizon == 1 else 'дня' if 1 < horizon < 5 else 'дней'}"
        
        keyboard = [
            [InlineKeyboardButton("« Назад к настройкам", callback_data="settings")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(message, reply_markup=reply_markup, parse_mode="Markdown")
    
    async def cb_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик callback-запроса для возврата в главное меню."""
        query = update.callback_query
        await query.answer()
        
        # Дублируем функционал из cmd_start, но с использованием edit_message_text
        keyboard = [
            [
                InlineKeyboardButton("🔮 Получить прогноз", callback_data="predict"),
                InlineKeyboardButton("📊 Статистика", callback_data="stats")
            ],
            [
                InlineKeyboardButton("⚙️ Настройки", callback_data="settings"),
                InlineKeyboardButton("❓ Помощь", callback_data="help")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message = f"*🔮 Gold Price Predictor • Главное меню*\n\n"
        message += f"Добро пожаловать в Gold Price Predictor!\n\n"
        message += f"Я помогу предсказать движение цены на золото, используя современные модели машинного обучения.\n\n"
        message += f"Выберите действие из меню ниже:"
        
        await query.edit_message_text(message, reply_markup=reply_markup, parse_mode="Markdown")
        
    async def cb_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик callback-запроса для отображения помощи."""
        query = update.callback_query
        await query.answer()
        
        # Дублируем код из cmd_help, но используем edit_message_text
        help_text = """*❓ Помощь по Gold Price Predictor*

Доступные команды:

/start - Запуск бота и главное меню
/predict - Получить прогноз цены золота
/stats - Просмотр статистики прогнозов
/settings - Настройки прогнозирования
/help - Показать эту справку

О прогнозах:

• Прогнозы создаются на основе исторических данных и моделей машинного обучения
• Вы можете выбрать разные модели и горизонты прогнозирования
• По умолчанию бот отправляет прогнозы ежедневно в 10:00

Если у вас есть вопросы или предложения, пожалуйста, свяжитесь с разработчиком."""
        
        keyboard = [
            [InlineKeyboardButton("🔮 Получить прогноз", callback_data="predict")],
            [InlineKeyboardButton("« Назад в меню", callback_data="start")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(help_text, reply_markup=reply_markup, parse_mode="Markdown")
        
    async def cb_stats_weekly(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик callback-запроса для просмотра еженедельной статистики."""
        # Перенаправляем команду на _show_weekly_statistics
        await self._show_weekly_statistics(update, context, is_callback=True)
        
    async def cb_stats_monthly(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик callback-запроса для просмотра ежемесячной статистики."""
        # Перенаправляем команду на _show_monthly_statistics
        await self._show_monthly_statistics(update, context, is_callback=True)
    
    async def _generate_prediction(self, update: Update, context: ContextTypes.DEFAULT_TYPE, is_callback=False):
        """
        Генерация прогноза цены золота.
        
        Args:
            update (Update): Обновление Telegram
            context (ContextTypes.DEFAULT_TYPE): Контекст
            is_callback (bool): True, если вызвано из callback-запроса
        """
        # Отправляем сообщение о начале прогнозирования
        message = "*🔄 Генерирую прогноз... Пожалуйста, подождите.*"
        
        if is_callback:
            await update.callback_query.edit_message_text(message, parse_mode="Markdown")
            chat_id = update.callback_query.message.chat_id
            message_id = update.callback_query.message.message_id
        else:
            sent_message = await update.message.reply_text(message, parse_mode="Markdown")
            chat_id = update.message.chat_id
            message_id = sent_message.message_id
        
        # Запускаем процесс прогнозирования асинхронно
        asyncio.create_task(self._async_generate_prediction(chat_id, message_id))
    
    async def _async_generate_prediction(self, chat_id, message_id):
        """
        Асинхронная генерация прогноза.
        
        Args:
            chat_id (int): ID чата
            message_id (int): ID сообщения для обновления
        """
        try:
            # 1. Автообновление истории золота через Bybit
            try:
                from data_updater import update_gold_history_from_bybit
                from config_loader import load_environment_variables
                
                # Загружаем переменные окружения
                load_environment_variables()
                
                # Сначала проверяем переменные окружения
                import os
                api_key = os.getenv('BYBIT_API_KEY')
                api_secret = os.getenv('BYBIT_API_SECRET')
                
                # Если в переменных окружения нет, пробуем взять из конфига
                if not api_key or not api_secret:
                    api_key = self.config.get('bybit_api_key', '')
                    api_secret = self.config.get('bybit_api_secret', '')
                    
                if not api_key or not api_secret:
                    logger.error('Отсутствуют API ключи Bybit в .env и конфиге')
                    await self.application.bot.send_message(
                        chat_id=chat_id,
                        text="⚠️ Невозможно обновить данные: отсутствуют API ключи Bybit"
                    )
                else:
                    csv_path = os.path.join(self.predictor.data_dir, 'GC_F_latest.csv')
                    update_gold_history_from_bybit(csv_path, api_key, api_secret)
                    logger.info('Исторические данные золота обновлены через Bybit')
            except Exception as update_exc:
                logger.error(f'Ошибка при обновлении истории через Bybit: {update_exc}')
            # Симулируем задержку, т.к. генерация прогноза может занять время
            await asyncio.sleep(1)
            
            # Генерируем прогноз
            self.predictor.args = type('Args', (), {
                'model': 'ensemble',
                'target_type': 'binary',
                'horizon': 1,
                'print_proba': False,
                'send_telegram': False
            })
            
            prediction = self.predictor.predict()
            
            if prediction:
                # Сохраняем предсказание
                self.tracker.save_prediction(prediction)
                
                # Формируем сообщение с прогнозом
                emoji_direction = "🔼" if prediction["direction"] == "UP" else "🔽"
                emoji_confidence = "🎯" if prediction["confidence"] > 0.7 else "🔍"
                
                # Определяем модель с эмодзи
                model_type = prediction.get("model", "ensemble")
                model_emoji = {
                    "xgboost": "🌲",  # Дерево
                    "lstm": "🧠",     # Нейросеть
                    "ensemble": "⚖️"  # Весы/ансамбль
                }.get(model_type.lower(), "🔮")
                
                # Красивое форматирование с Markdown
                message = f"*📈 Прогноз цены золота*\n\n"
                message += f"📅 *Дата:* {prediction['prediction_date']}\n\n"
                message += f"{emoji_direction} *Направление:* {prediction['direction']}\n"
                message += f"💰 *Текущая цена:* ${prediction['current_price']:.2f}\n"
                message += f"{emoji_confidence} *Вероятность:* {prediction['confidence']:.2f}\n\n"
                message += f"{model_emoji} *Модель:* {model_type.upper()}\n\n"
                
                # Добавляем прогнозы отдельных моделей
                message += "*Прогнозы моделей:*\n"
                for model_name, pred in prediction.get("all_predictions", {}).items():
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
                
                # Кнопки для дальнейших действий
                keyboard = [
                    [
                        InlineKeyboardButton("🔮 Новый прогноз", callback_data="predict"),
                        InlineKeyboardButton("📊 Статистика", callback_data="stats")
                    ],
                    [InlineKeyboardButton("« Главное меню", callback_data="start")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                # Обновляем сообщение с прогнозом
                await self.application.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text=message,
                    reply_markup=reply_markup,
                    parse_mode="Markdown"
                )
            else:
                # В случае ошибки прогнозирования
                message = (
                    "*❌ Не удалось получить прогноз*\n\n"
                    "Произошла ошибка при генерации прогноза. "
                    "Пожалуйста, попробуйте позже или обратитесь к разработчику."
                )
                
                keyboard = [
                    [InlineKeyboardButton("🔄 Попробовать снова", callback_data="predict")],
                    [InlineKeyboardButton("« Главное меню", callback_data="start")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await self.application.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text=message,
                    reply_markup=reply_markup,
                    parse_mode="Markdown"
                )
        except Exception as e:
            logger.error(f"Ошибка при генерации прогноза: {e}")
            
            message = (
                "*❌ Ошибка*\n\n"
                f"Произошла ошибка при генерации прогноза: {str(e)}\n"
                "Пожалуйста, попробуйте позже или обратитесь к разработчику."
            )
            
            keyboard = [
                [InlineKeyboardButton("🔄 Попробовать снова", callback_data="predict")],
                [InlineKeyboardButton("« Главное меню", callback_data="start")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            try:
                await self.application.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text=message,
                    reply_markup=reply_markup,
                    parse_mode="Markdown"
                )
            except Exception as e2:
                logger.error(f"Ошибка при обновлении сообщения: {e2}")
    
    async def _show_statistics(self, update: Update, context: ContextTypes.DEFAULT_TYPE, is_callback=False):
        """
        Показ статистики прогнозов.
        
        Args:
            update (Update): Обновление Telegram
            context (ContextTypes.DEFAULT_TYPE): Контекст
            is_callback (bool): True, если вызвано из callback-запроса
        """
        stats = self.tracker.get_statistics()
        
        if stats["total"] > 0:
            message = (
                "*📊 Статистика прогнозов*\n\n"
                f"Всего прогнозов: {stats['total']}\n"
                f"Верных прогнозов: {stats['correct']} ({stats['accuracy'] * 100:.1f}%)\n\n"
                f"🔥 Текущая серия: {stats['recent_streak']} прогнозов\n"
                f"🏆 Лучшая серия: {stats['best_streak']} прогнозов\n\n"
                "*Статистика по моделям:*\n"
            )
            
            # Добавляем статистику по моделям
            for model, model_stats in stats.get("by_model", {}).items():
                if model_stats["total"] > 0:
                    model_icon = {
                        "xgboost": "🌲",
                        "lstm": "🧠",
                        "ensemble": "⚖️"
                    }.get(model.lower(), "🔮")
                    message += f"{model_icon} *{model.capitalize()}*: {model_stats['correct']}/{model_stats['total']} ({model_stats['accuracy'] * 100:.1f}%)\n"
        else:
            message = (
                "*📊 Статистика прогнозов*\n\n"
                "Пока нет проверенных прогнозов. Статистика будет доступна после верификации "
                "предсказаний на основе фактических движений цены."
            )
        
        keyboard = [
            [
                InlineKeyboardButton("📊 Недельная", callback_data="stats_weekly"),
                InlineKeyboardButton("📊 Месячная", callback_data="stats_monthly")
            ],
            [InlineKeyboardButton("« Назад в меню", callback_data="start")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if is_callback:
            await update.callback_query.edit_message_text(message, reply_markup=reply_markup, parse_mode="Markdown")
        else:
            await update.message.reply_text(message, reply_markup=reply_markup, parse_mode="Markdown")
    
    async def _show_weekly_statistics(self, update: Update, context: ContextTypes.DEFAULT_TYPE, is_callback=False):
        """
        Показ еженедельной статистики прогнозов.
        
        Args:
            update (Update): Обновление Telegram
            context (ContextTypes.DEFAULT_TYPE): Контекст
            is_callback (bool): True, если вызвано из callback-запроса
        """
        report = self.tracker.generate_weekly_report()
        
        keyboard = [
            [
                InlineKeyboardButton("📊 Общая", callback_data="stats"),
                InlineKeyboardButton("📊 Месячная", callback_data="stats_monthly")
            ],
            [InlineKeyboardButton("« Назад в меню", callback_data="start")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if is_callback:
            await update.callback_query.edit_message_text(report, reply_markup=reply_markup, parse_mode="Markdown")
        else:
            await update.message.reply_text(report, reply_markup=reply_markup, parse_mode="Markdown")
    
    async def _show_monthly_statistics(self, update: Update, context: ContextTypes.DEFAULT_TYPE, is_callback=False):
        """
        Показ ежемесячной статистики прогнозов.
        
        Args:
            update (Update): Обновление Telegram
            context (ContextTypes.DEFAULT_TYPE): Контекст
            is_callback (bool): True, если вызвано из callback-запроса
        """
        monthly_stats = self.tracker.get_monthly_statistics()
        stats = monthly_stats["statistics"]
        
        # Формируем сообщение
        message = f"*📊 Статистика за {monthly_stats['current_month']}*\n\n"
        
        if stats["total"] > 0:
            message += f"Всего прогнозов: {stats['total']}\n"
            message += f"Верных прогнозов: {stats['correct']} ({stats['accuracy'] * 100:.1f}%)\n\n"
            
            # Добавляем статистику по моделям
            general_stats = self.tracker.get_statistics()
            message += "*За всё время:*\n"
            message += f"Общая точность: {general_stats['accuracy'] * 100:.1f}%\n\n"
            
            # Добавляем последние предсказания
            recent_preds = monthly_stats["recent_predictions"]
            if recent_preds:
                message += "*Последние проверенные прогнозы:*\n"
                for pred in recent_preds[:5]:  # Показываем только 5 последних
                    date = pred.get("prediction_date", "")
                    is_correct = pred.get("is_correct", False)
                    direction = pred.get("direction", "")
                    
                    icon = "✅" if is_correct else "❌"
                    direction_icon = "🔼" if direction == "UP" else "🔽"
                    
                    message += f"{icon} {date}: {direction_icon} {direction}\n"
        else:
            message += "В этом месяце еще нет проверенных прогнозов."
        
        keyboard = [
            [
                InlineKeyboardButton("📊 Общая", callback_data="stats"),
                InlineKeyboardButton("📊 Недельная", callback_data="stats_weekly")
            ],
            [InlineKeyboardButton("« Назад в меню", callback_data="start")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if is_callback:
            await update.callback_query.edit_message_text(message, reply_markup=reply_markup, parse_mode="Markdown")
        else:
            await update.message.reply_text(message, reply_markup=reply_markup, parse_mode="Markdown")
    
    async def _show_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE, is_callback=False):
        """
        Показ настроек прогнозирования.
        
        Args:
            update (Update): Обновление Telegram
            context (ContextTypes.DEFAULT_TYPE): Контекст
            is_callback (bool): True, если вызвано из callback-запроса
        """
        # Текущие настройки
        config = self.predictor.config
        model_type = config.get("model_type", "ensemble")
        horizon = config.get("horizon", 1)
        target_type = config.get("target_type", "binary")
        
        # Формируем сообщение
        message = (
            "*⚙️ Настройки прогнозирования*\n\n"
            "*Текущие параметры:*\n"
            f"• Модель: {model_type.upper()}\n"
            f"• Горизонт: {horizon} {'день' if horizon == 1 else 'дня' if 1 < horizon < 5 else 'дней'}\n"
            f"• Тип цели: {target_type.upper()}\n\n"
            "Выберите параметр для изменения:"
        )
        
        # Кнопки для изменения настроек
        keyboard = [
            [
                InlineKeyboardButton("🌲 XGBoost", callback_data="model_xgboost"),
                InlineKeyboardButton("🧠 LSTM", callback_data="model_lstm"),
                InlineKeyboardButton("⚖️ Ensemble", callback_data="model_ensemble")
            ],
            [
                InlineKeyboardButton("🕒 1 день", callback_data="horizon_1"),
                InlineKeyboardButton("🕒 3 дня", callback_data="horizon_3"),
                InlineKeyboardButton("🕒 7 дней", callback_data="horizon_7")
            ],
            [InlineKeyboardButton("« Назад в меню", callback_data="start")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if is_callback:
            await update.callback_query.edit_message_text(message, reply_markup=reply_markup, parse_mode="Markdown")
        else:
            await update.message.reply_text(message, reply_markup=reply_markup, parse_mode="Markdown")
    
    def _schedule_tasks(self):
        """Настройка расписания задач."""
        # Время предсказания
        prediction_time = self.config.get("prediction_time", "10:00")
        logger.info(f"Настроено расписание предсказаний на {prediction_time} ежедневно")
        self.scheduler.every().day.at(prediction_time).do(self.scheduled_prediction)
        
        # Время верификации предсказаний
        verification_time = self.config.get("verification_time", "10:00")
        logger.info(f"Настроено расписание верификации на {verification_time} ежедневно")
        self.scheduler.every().day.at(verification_time).do(self.scheduled_verification)
        
        # Отправляем сообщение о настройке расписания, если указан chat_id
        chat_id = self.config.get("telegram_chat_id")
        if chat_id:
            msg = f"*⏰ Планировщик Gold Predictor активирован*\n\n"
            msg += f"• Ежедневное прогнозирование: {prediction_time}\n"
            msg += f"• Ежедневная верификация: {verification_time}\n\n"
            msg += f"Бот будет автоматически отправлять прогнозы и статистику в это время."
            
            try:
                # Используем обычный Telegram API без asyncio
                import requests
                # Синхронный запрос к Telegram API
                url = f"https://api.telegram.org/bot{self.config['telegram_token']}/sendMessage"
                payload = {
                    'chat_id': chat_id,
                    'text': msg,
                    'parse_mode': 'Markdown'
                }
                response = requests.post(url, json=payload)
                if not response.ok:
                    logger.warning(f"Неудачный ответ от Telegram API: {response.status_code} - {response.text}")
            except Exception as e:
                logger.error(f"Ошибка при отправке сообщения о расписании: {e}")
    
    def _run_scheduler(self):
        """Фоновый запуск планировщика."""
        while self.running:
            self.scheduler.run_pending()
            time.sleep(1)
    
    def scheduled_prediction(self):
        """Отправка запланированного прогноза."""
        try:
            logger.info("🔮 Выполнение запланированного прогноза...")
            
            # Создаем аргументы для предиктора
            class Args:
                def __init__(self):
                    self.model = 'ensemble'
                    self.target_type = 'binary'
                    self.horizon = 1
                    self.print_proba = False
                    self.send_telegram = False
                    self.config = '../config/predictor_config.json'
            
            self.predictor.args = Args()
            
            # Генерируем прогноз
            result = self.predictor.predict()
            
            if result:
                logger.info(f"✅ Запланированный прогноз успешно сгенерирован на {result.get('prediction_date')}")
                
                # Отправляем сообщение в Telegram
                chat_id = self.config.get("telegram_chat_id")
                if chat_id:
                    prediction_message = self._format_prediction_message(result)
                    try:
                        # Используем синхронный запрос к Telegram API
                        import requests
                        url = f"https://api.telegram.org/bot{self.config['telegram_token']}/sendMessage"
                        payload = {
                            'chat_id': chat_id,
                            'text': prediction_message,
                            'parse_mode': 'Markdown'
                        }
                        response = requests.post(url, json=payload)
                        if response.ok:
                            logger.info(f"✉️ Запланированный прогноз отправлен в Telegram")
                        else:
                            logger.warning(f"Неудачный ответ от Telegram API: {response.status_code} - {response.text}")
                    except Exception as e:
                        logger.error(f"Ошибка при отправке запланированного прогноза: {e}")
                
                return True
            else:
                logger.error("❌ Не удалось сгенерировать запланированный прогноз")
                return False
        except Exception as e:
            logger.error(f"❌ Ошибка при генерации запланированного прогноза: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def scheduled_verification(self):
        """Верификация предыдущего прогноза на основе текущих данных."""
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
                chat_id = self.config.get("telegram_chat_id")
                
                if chat_id:
                    if stats.get("recent_streak", 0) >= 3:
                        try:
                            # Отправляем отчет о серии успешных прогнозов
                            streak_message = f"*🔥 Серия успешных прогнозов: {stats['recent_streak']}*\n\n"
                            streak_message += f"Текущая серия успешных прогнозов достигла {stats['recent_streak']} подряд!\n"
                            streak_message += f"Общая точность: {stats['accuracy'] * 100:.1f}%"
                            
                            # Используем синхронный запрос к Telegram API
                            import requests
                            url = f"https://api.telegram.org/bot{self.config['telegram_token']}/sendMessage"
                            payload = {
                                'chat_id': chat_id,
                                'text': streak_message,
                                'parse_mode': 'Markdown'
                            }
                            response = requests.post(url, json=payload)
                            if not response.ok:
                                logger.warning(f"Неудачный ответ от Telegram API: {response.status_code} - {response.text}")
                        except Exception as e:
                            logger.error(f"Ошибка при отправке уведомления о серии прогнозов: {e}")
                    
                    # Еженедельный отчет по воскресеньям
                    if datetime.now().weekday() == 6:  # Воскресенье
                        try:
                            weekly_report = self.tracker.generate_weekly_report()
                            # Используем синхронный запрос к Telegram API
                            import requests
                            url = f"https://api.telegram.org/bot{self.config['telegram_token']}/sendMessage"
                            payload = {
                                'chat_id': chat_id,
                                'text': weekly_report,
                                'parse_mode': 'Markdown'
                            }
                            response = requests.post(url, json=payload)
                            if not response.ok:
                                logger.warning(f"Неудачный ответ от Telegram API: {response.status_code} - {response.text}")
                        except Exception as e:
                            logger.error(f"Ошибка при отправке еженедельного отчета: {e}")
                
                return True
            else:
                logger.error(f"❌ Не удалось верифицировать прогноз для {yesterday}")
                return False
        except Exception as e:
            logger.error(f"❌ Ошибка при верификации прогноза: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _format_prediction_message(self, prediction):
        """Форматирование сообщения с прогнозом для Telegram."""
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
        message = f"*📈 Ежедневный прогноз цены золота*\n\n"
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
        message += f"\n_Используйте команду /predict для получения нового прогноза_"
        
        return message
    
    def start_scheduler(self):
        """Запуск планировщика для отправки прогнозов по расписанию."""
        self._schedule_tasks()
        
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        logger.info("🚀 Планировщик Gold Price Predictor запущен")
        
        # Проверяем, есть ли запланированные задачи и когда они выполнятся
        next_runs = []
        for job in self.scheduler.get_jobs():
            next_run = job.next_run
            next_runs.append((job.job_func.__name__, next_run))
            logger.info(f"📅 Задача {job.job_func.__name__} запланирована на {next_run}")
        
        return True
    
    def stop_scheduler(self):
        """Остановка планировщика."""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=1)
        self.scheduler.clear()
        logger.info("🛑 Планировщик Gold Price Predictor остановлен")
        return True
        
    async def run(self):
        """Запуск бота."""
        logger.info("Запуск Telegram-бота Gold Predictor...")
        # Запускаем планировщик
        self.start_scheduler()
        # Запускаем бота
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling(allowed_updates=Update.ALL_TYPES)
        
        # Запуск цикла обработки событий
        try:
            logger.info("Бот запущен и обрабатывает события. Нажмите Ctrl+C для остановки.")
            # Просто ждем бесконечно
            await asyncio.Event().wait()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Получен сигнал остановки, завершение работы бота...")
        finally:
            # Останавливаем планировщик и бота
            self.stop_scheduler()
            await self.application.stop()
    
    def stop(self):
        """Остановка бота."""
        logger.info("Остановка Telegram-бота Gold Predictor...")
        # Останавливаем планировщик
        self.stop_scheduler()
        # Останавливаем бота
        self.application.stop()


# Запуск бота
if __name__ == "__main__":
    try:
        bot = GoldPredictorBot()
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("Бот остановлен пользователем")
    except Exception as e:
        logger.error(f"Ошибка при запуске бота: {e}")
