
(base) wsgp@Sergeys-MacBook-Air Gold_predictor % python src/telegram_bot.py
2025-04-19 23:37:25,308 - config_loader - INFO - Переменные окружения загружены из /Users/wsgp/CascadeProjects/Gold_predictor/.env
2025-04-19 23:37:25,309 - config_loader - INFO - Переменные окружения загружены из /Users/wsgp/CascadeProjects/Gold_predictor/.env
2025-04-19 23:37:25,309 - config_loader - INFO - Конфигурация загружена из config/predictor_config.json
2025-04-19 23:37:25,309 - config_loader - INFO - Переменные окружения загружены из /Users/wsgp/CascadeProjects/Gold_predictor/.env
2025-04-19 23:37:25,309 - predict - INFO - Конфигурация загружена из config/predictor_config.json
2025-04-19 23:37:25,309 - config_loader - INFO - Переменные окружения загружены из /Users/wsgp/CascadeProjects/Gold_predictor/.env
2025-04-19 23:37:25,309 - predict - INFO - Токен Telegram загружен из переменных окружения
2025-04-19 23:37:25,309 - predict - INFO - Chat ID Telegram загружен из переменных окружения
2025-04-19 23:37:25,310 - predict - INFO - Конфигурация сохранена в config/predictor_config.json
2025-04-19 23:37:25,311 - predict - INFO - Трекер предсказаний инициализирован успешно
2025-04-19 23:37:25,311 - models - ERROR - Файл модели ../models/models/xgboost_binary_20250419.json не найден
2025-04-19 23:37:25,311 - predict - INFO - XGBoost модель загружена из models/xgboost_binary_20250419.json
2025-04-19 23:37:25,311 - models - ERROR - Файл модели ../models/models/lstm_binary_20250419.h5 не найден
2025-04-19 23:37:25,311 - predict - INFO - LSTM модель загружена из models/lstm_binary_20250419.h5
2025-04-19 23:37:25,311 - models - ERROR - Файл информации об ансамбле ../models/models/ensemble_binary_20250419.joblib не найден
2025-04-19 23:37:25,311 - predict - INFO - Информация об ансамбле загружена из models/ensemble_binary_20250419.joblib
2025-04-19 23:37:25,438 - __main__ - INFO - Запуск Telegram-бота Gold Predictor...
2025-04-19 23:37:25,438 - __main__ - INFO - Настроено расписание предсказаний на 10:00 ежедневно
2025-04-19 23:37:25,438 - __main__ - INFO - Настроено расписание верификации на 10:00 ежедневно
2025-04-19 23:37:26,143 - __main__ - INFO - 🚀 Планировщик Gold Price Predictor запущен
2025-04-19 23:37:26,143 - __main__ - INFO - 📅 Задача scheduled_prediction запланирована на 2025-04-20 10:00:00
2025-04-19 23:37:26,143 - __main__ - INFO - 📅 Задача scheduled_verification запланирована на 2025-04-20 10:00:00
2025-04-19 23:37:26,726 - httpx - INFO - HTTP Request: POST https://api.telegram.org/bot7827555043:AAFScXM8OLSkO2xfhiqrhFIy0NN2Tu1udUU/getMe "HTTP/1.1 200 OK"
2025-04-19 23:37:26,727 - apscheduler.scheduler - INFO - Scheduler started
2025-04-19 23:37:26,728 - telegram.ext.Application - INFO - Application started
2025-04-19 23:37:26,916 - httpx - INFO - HTTP Request: POST https://api.telegram.org/bot7827555043:AAFScXM8OLSkO2xfhiqrhFIy0NN2Tu1udUU/deleteWebhook "HTTP/1.1 200 OK"
2025-04-19 23:37:26,916 - __main__ - INFO - Бот запущен и обрабатывает события. Нажмите Ctrl+C для остановки.
2025-04-19 23:37:30,530 - httpx - INFO - HTTP Request: POST https://api.telegram.org/bot7827555043:AAFScXM8OLSkO2xfhiqrhFIy0NN2Tu1udUU/getUpdates "HTTP/1.1 200 OK"
2025-04-19 23:37:30,752 - httpx - INFO - HTTP Request: POST https://api.telegram.org/bot7827555043:AAFScXM8OLSkO2xfhiqrhFIy0NN2Tu1udUU/sendMessage "HTTP/1.1 200 OK"
2025-04-19 23:37:30,755 - config_loader - INFO - Переменные окружения загружены из /Users/wsgp/CascadeProjects/Gold_predictor/.env
История уже актуальна!
2025-04-19 23:37:30,757 - __main__ - INFO - Исторические данные золота обновлены через Bybit
2025-04-19 23:37:31,759 - config_loader - INFO - Переменные окружения загружены из /Users/wsgp/CascadeProjects/Gold_predictor/.env
История уже актуальна!
2025-04-19 23:37:31,759 - predict - INFO - Данные успешно обновлены через Bybit API
2025-04-19 23:37:31,762 - config_loader - INFO - Переменные окружения загружены из /Users/wsgp/CascadeProjects/Gold_predictor/.env
2025-04-19 23:37:31,934 - predict - INFO - Получена актуальная цена золота: $3345.30 (источник: Bybit XAUTUSDT, время: 2025-04-19 23:37:31)
2025-04-19 23:37:31,935 - predict - INFO - Аргументы запуска: <class '__main__.Args'>
2025-04-19 23:37:31,935 - predict - INFO - Конфиг: {'target_type': 'binary', 'horizon': 1, 'sequence_length': 10, 'xgb_model_path': 'models/xgboost_binary_20250419.json', 'lstm_model_path': 'models/lstm_binary_20250419.h5', 'ensemble_info_path': 'models/ensemble_binary_20250419.joblib', 'telegram_token': '7827555043:AAFScXM8OLSkO2xfhiqrhFIy0NN2Tu1udUU', 'telegram_chat_id': '-1002564245552', 'prediction_time': '10:00', 'verification_time': '10:00'}
2025-04-19 23:37:31,935 - config_loader - INFO - Переменные окружения загружены из /Users/wsgp/CascadeProjects/Gold_predictor/.env
История уже актуальна!
2025-04-19 23:37:31,935 - predict - INFO - Данные успешно обновлены через Bybit API
2025-04-19 23:37:31,935 - predict - INFO - Bybit API обновление данных: True
/Users/wsgp/CascadeProjects/Gold_predictor/src/predict.py:303: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
  latest_data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
2025-04-19 23:37:31,989 - predict - INFO - Загружены данные с обновлениями от Bybit, последняя дата: 2025-04-19
2025-04-19 23:37:31,992 - predict - INFO - Получены данные с Ticker по 2025-04-19, всего 211 записей
2025-04-19 23:37:31,993 - features - INFO - Создание технических индикаторов
2025-04-19 23:37:31,998 - features - ERROR - Ошибка при удалении NaN: ['Open', 'High', 'Low', 'Close', 'Volume']
2025-04-19 23:37:32,005 - features - INFO - Удалено 2 строк с некорректными значениями OHLCV
2025-04-19 23:37:32,012 - features - INFO - EMA_10 type: <class 'pandas.core.series.Series'>
2025-04-19 23:37:32,012 - features - INFO - EMA_ratio_10 type: <class 'pandas.core.series.Series'>
2025-04-19 23:37:32,016 - features - INFO - RSI_7 type: <class 'pandas.core.series.Series'>
2025-04-19 23:37:32,017 - features - INFO - RSI_14 type: <class 'pandas.core.series.Series'>
2025-04-19 23:37:32,025 - features - INFO - Stoch_%K_14 type: <class 'pandas.core.series.Series'>
2025-04-19 23:37:32,025 - features - INFO - Stoch_%D_14 type: <class 'pandas.core.series.Series'>
2025-04-19 23:37:32,026 - features - INFO - ATR_14 type: <class 'pandas.core.series.Series'>
2025-04-19 23:37:32,031 - features - INFO - Создано технических индикаторов: 61 столбцов
2025-04-19 23:37:32,032 - predict - INFO - Оригинальные столбцы: ['Open', 'High', 'Low', 'Close', 'Volume']
2025-04-19 23:37:32,033 - predict - INFO - Очищенные столбцы: ['Open', 'High', 'Low', 'Close', 'Volume']
2025-04-19 23:37:32,033 - predict - INFO - Добавлен признак Future_Close для совместимости с моделями
2025-04-19 23:37:32,035 - predict - INFO - Создана последовательность для LSTM размером (1, 10, 62)
2025-04-19 23:37:32,035 - predict - INFO - Прогноз на дату: 2025-04-20
2025-04-19 23:37:32,035 - predict - INFO - Всего признаков: 61
2025-04-19 23:37:32,035 - predict - INFO - Список признаков: ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_5', 'MA_ratio_5', 'MA_10', 'MA_ratio_10', 'MA_20', 'MA_ratio_20', 'MA_50', 'MA_ratio_50', 'MA_100', 'MA_ratio_100', 'EMA_5', 'EMA_ratio_5', 'EMA_10', 'EMA_ratio_10', 'EMA_20', 'EMA_ratio_20', 'EMA_50', 'EMA_ratio_50', 'EMA_100', 'EMA_ratio_100', 'BB_upper_20', 'BB_lower_20', 'BB_width_20', 'BB_position_20', 'BB_Upper_20', 'BB_Lower_20', 'BB_Width_20', 'BB_Position_20', 'RSI_7', 'RSI_14', 'RSI_21', 'MACD', 'MACD_Signal', 'MACD_Hist', 'MACD_Hist_Change', 'MACD_line', 'MACD_signal', 'MACD_histogram', 'Stoch_%K_14', 'Stoch_%D_14', 'ATR_14', 'CCI_20', 'Price_Change', 'Return', 'Volatility_5', 'Volatility_10', 'Volatility_21', 'High_Low_Range', 'High_Low_Range_Pct', 'Volume_MA_5', 'Volume_ratio_5', 'Volume_MA_10', 'Volume_ratio_10', 'Volume_MA_20', 'Volume_ratio_20', 'Volume_Price']
2025-04-19 23:37:32,035 - predict - INFO - Пример значений: {'Close': 3346.0, 'MA_5': 3307.56, 'RSI_14': 66.82472444192028, 'MACD_line': 90.47476907763757}
2025-04-19 23:37:32,038 - predict - INFO - NaN значений: 0, Inf значений: 0
2025-04-19 23:37:32,039 - predict - ERROR - Ошибка при загрузке метаданных модели: [Errno 2] No such file or directory: '../models/xgboost_binary_20250419_metadata.joblib'
2025-04-19 23:37:32,040 - predict - INFO - Признаки для XGBoost (всего 67): ['Close', 'High', 'Low', 'Open', 'Volume']... и еще 62
2025-04-19 23:37:32,040 - models - ERROR - Модель не обучена
2025-04-19 23:37:32,040 - models - ERROR - Модель не обучена
2025-04-19 23:37:32,040 - predict - ERROR - Ошибка при прогнозировании с XGBoost: 'NoneType' object is not subscriptable
2025-04-19 23:37:32,040 - models - INFO - LSTMModel.predict: X shape = (1, 10, 62)
2025-04-19 23:37:32,040 - models - ERROR - Модель не обучена
2025-04-19 23:37:32,040 - predict - ERROR - Не удалось получить ни одного прогноза
2025-04-19 23:37:32,257 - httpx - INFO - HTTP Request: POST https://api.telegram.org/bot7827555043:AAFScXM8OLSkO2xfhiqrhFIy0NN2Tu1udUU/editMessageText "HTTP/1.1 200 OK"
2025-04-19 23:37:40,736 - httpx - INFO - HTTP Request: POST https://api.telegram.org/bot7827555043:AAFScXM8OLSkO2xfhiqrhFIy0NN2Tu1udUU/getUpdates "HTTP/1.1 200 OK"
