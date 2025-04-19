
2025-04-19 21:56:04,611 - httpx - INFO - HTTP Request: POST https://api.telegram.org/bot7827555043:AAFScXM8OLSkO2xfhiqrhFIy0NN2Tu1udUU/getMe "HTTP/1.1 200 OK"
2025-04-19 21:56:04,612 - apscheduler.scheduler - INFO - Scheduler started
2025-04-19 21:56:04,612 - telegram.ext.Application - INFO - Application started
2025-04-19 21:56:04,801 - httpx - INFO - HTTP Request: POST https://api.telegram.org/bot7827555043:AAFScXM8OLSkO2xfhiqrhFIy0NN2Tu1udUU/deleteWebhook "HTTP/1.1 200 OK"
2025-04-19 21:56:04,802 - __main__ - INFO - Бот запущен и обрабатывает события. Нажмите Ctrl+C для остановки.
2025-04-19 21:56:11,841 - httpx - INFO - HTTP Request: POST https://api.telegram.org/bot7827555043:AAFScXM8OLSkO2xfhiqrhFIy0NN2Tu1udUU/getUpdates "HTTP/1.1 200 OK"
2025-04-19 21:56:12,489 - httpx - INFO - HTTP Request: POST https://api.telegram.org/bot7827555043:AAFScXM8OLSkO2xfhiqrhFIy0NN2Tu1udUU/sendMessage "HTTP/1.1 200 OK"
Обновление истории с 2025-04-18 по 2025-04-19
Добавлено 2 новых строк.
2025-04-19 21:56:12,765 - __main__ - INFO - Исторические данные золота обновлены через Bybit
2025-04-19 21:56:13,766 - predict - INFO - Аргументы запуска: <class '__main__.Args'>
2025-04-19 21:56:13,767 - predict - INFO - Конфиг: {'target_type': 'binary', 'horizon': 1, 'sequence_length': 10, 'xgb_model_path': 'models/xgboost_binary_20250419.json', 'lstm_model_path': 'models/lstm_binary_20250419.h5', 'ensemble_info_path': 'models/ensemble_binary_20250419.joblib', 'telegram_token': '7827555043:AAFScXM8OLSkO2xfhiqrhFIy0NN2Tu1udUU', 'telegram_chat_id': '-1002564245552', 'prediction_time': '10:00', 'verification_time': '10:00'}
2025-04-19 21:56:13,767 - data_loader - INFO - Загрузка данных для GC=F с периодом 5y и интервалом 1d
2025-04-19 21:56:13,794 - data_loader - INFO - Загружены кэшированные данные от 2025-04-19
2025-04-19 21:56:15,389 - data_loader - INFO - Данные успешно загружены с попытки 1
2025-04-19 21:56:15,394 - data_loader - INFO - Данные сохранены в ../data/GC_F_latest.csv
2025-04-19 21:56:15,395 - predict - INFO - Получены данные с 2024-06-24 00:00:00 по 2025-04-17 00:00:00, всего 207 записей
2025-04-19 21:56:15,395 - features - INFO - Создание технических индикаторов
2025-04-19 21:56:15,400 - features - INFO - Удалено 0 строк с некорректными значениями OHLCV
2025-04-19 21:56:15,407 - features - INFO - EMA_10 type: <class 'pandas.core.series.Series'>
2025-04-19 21:56:15,407 - features - INFO - EMA_ratio_10 type: <class 'pandas.core.series.Series'>
2025-04-19 21:56:15,415 - features - INFO - RSI_7 type: <class 'pandas.core.series.Series'>
2025-04-19 21:56:15,416 - features - INFO - RSI_14 type: <class 'pandas.core.series.Series'>
2025-04-19 21:56:15,422 - features - INFO - Stoch_%K_14 type: <class 'pandas.core.series.Series'>
2025-04-19 21:56:15,422 - features - INFO - Stoch_%D_14 type: <class 'pandas.core.series.Series'>
2025-04-19 21:56:15,424 - features - INFO - ATR_14 type: <class 'pandas.core.series.Series'>
2025-04-19 21:56:15,429 - features - INFO - Создано технических индикаторов: 61 столбцов
2025-04-19 21:56:15,430 - predict - INFO - Оригинальные столбцы: ['Open', 'High', 'Low', 'Close', 'Volume']
2025-04-19 21:56:15,430 - predict - INFO - Очищенные столбцы: ['Open', 'High', 'Low', 'Close', 'Volume']
2025-04-19 21:56:15,431 - predict - INFO - Добавлен признак Future_Close для совместимости с моделями
2025-04-19 21:56:15,431 - predict - INFO - Создана последовательность для LSTM размером (1, 10, 62)
2025-04-19 21:56:15,431 - predict - INFO - Всего признаков: 61
2025-04-19 21:56:15,431 - predict - INFO - Список признаков: ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_5', 'MA_ratio_5', 'MA_10', 'MA_ratio_10', 'MA_20', 'MA_ratio_20', 'MA_50', 'MA_ratio_50', 'MA_100', 'MA_ratio_100', 'EMA_5', 'EMA_ratio_5', 'EMA_10', 'EMA_ratio_10', 'EMA_20', 'EMA_ratio_20', 'EMA_50', 'EMA_ratio_50', 'EMA_100', 'EMA_ratio_100', 'BB_upper_20', 'BB_lower_20', 'BB_width_20', 'BB_position_20', 'BB_Upper_20', 'BB_Lower_20', 'BB_Width_20', 'BB_Position_20', 'RSI_7', 'RSI_14', 'RSI_21', 'MACD', 'MACD_Signal', 'MACD_Hist', 'MACD_Hist_Change', 'MACD_line', 'MACD_signal', 'MACD_histogram', 'Stoch_%K_14', 'Stoch_%D_14', 'ATR_14', 'CCI_20', 'Price_Change', 'Return', 'Volatility_5', 'Volatility_10', 'Volatility_21', 'High_Low_Range', 'High_Low_Range_Pct', 'Volume_MA_5', 'Volume_ratio_5', 'Volume_MA_10', 'Volume_ratio_10', 'Volume_MA_20', 'Volume_ratio_20', 'Volume_Price']
2025-04-19 21:56:15,431 - predict - INFO - Пример значений: {'Close': 3308.699951171875, 'MA_5': 3256.2, 'RSI_14': 66.39126107736678, 'MACD_line': 77.75925023767422}
2025-04-19 21:56:15,432 - predict - INFO - NaN значений: 0, Inf значений: 0
2025-04-19 21:56:15,432 - models - INFO - LSTMModel.predict: X shape = (1, 10, 62)
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 263ms/step
2025-04-19 21:56:15,760 - models - INFO - LSTMModel.predict: predictions shape = (1, 1)
2025-04-19 21:56:15,760 - models - INFO - LSTMModel.predict: Вероятности = [0.47972226]
2025-04-19 21:56:15,760 - predict - WARNING - Модель ensemble не найдена, используем lstm
2025-04-19 21:56:15,761 - predict - INFO - Предсказание сохранено в трекере для даты 2025-04-18

[RESULT] Прогноз на 2025-04-18:
Направление: DOWN (уверенность: 0.480)
Текущая цена: $3308.70
Модель: ENSEMBLE