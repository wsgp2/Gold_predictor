#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для создания и обработки признаков (feature engineering) для прогнозирования цены золота.
"""

import numpy as np
import pandas as pd
import logging
import traceback

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureGenerator:
    """Класс для генерации признаков на основе OHLCV данных золота."""
    
    def __init__(self, scaling_method='standard'):
        """
        Инициализация генератора признаков.
        
        Args:
            scaling_method (str): Метод масштабирования признаков ('standard', 'minmax', None)
        """
        self.scaling_method = scaling_method
        self.scaler = None
        
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
    
    def _ensure_all_columns_are_series(self, data):
        """
        Гарантирует, что все столбцы в DataFrame являются Series, а не DataFrame или ndarray.
        
        Args:
            data (pandas.DataFrame): Исходный DataFrame
            
        Returns:
            pandas.DataFrame: Очищенный DataFrame
        """
        if data is None or not isinstance(data, pd.DataFrame):
            logger.error(f"Некорректный тип данных для нормализации: {type(data)}")
            return data
            
        for col in data.columns:
            try:
                if isinstance(data[col], pd.DataFrame):
                    # Если столбец - DataFrame, берем первый столбец из него
                    logger.info(f"Конвертируем столбец {col} из DataFrame в Series")
                    # Просто сплющиваем первый столбец DataFrame с формой (n, 1)
                    if data[col].shape == (len(data.index), 1):
                        # Извлекаем первый столбец и сплющиваем в одномерный массив
                        values = data[col].iloc[:, 0].values
                        data[col] = pd.Series(values, index=data.index)
                    else:
                        # Если размер отличается, используем первый доступный столбец
                        if len(data[col].columns) > 0:
                            col_name = data[col].columns[0]
                            values = data[col][col_name].values
                            data[col] = pd.Series(values, index=data.index)
                        else:
                            # Если нет столбцов, создаем пустую Series
                            data[col] = pd.Series(index=data.index)
                elif isinstance(data[col], np.ndarray):
                    # Если это ndarray с размерностью > 1
                    if len(data[col].shape) > 1:
                        logger.info(f"Сплющиваем ndarray размера {data[col].shape} для колонки {col}")
                        # Сплющиваем в одномерный массив
                        values = np.ravel(data[col])
                        data[col] = pd.Series(values, index=data.index)
                    else:
                        # Одномерный массив просто конвертируем в Series
                        data[col] = pd.Series(data[col], index=data.index)
                elif not isinstance(data[col], pd.Series):
                    # Преобразуем в Series другие типы данных
                    data[col] = pd.Series(data[col], index=data.index)
            except Exception as e:
                logger.error(f"Не удалось конвертировать колонку {col} в Series: {e}")
                logger.error(f"Тип данных: {type(data[col])}")
                if hasattr(data[col], 'shape'):
                    logger.error(f"Форма: {data[col].shape}")
        
        return data
        
    def create_technical_indicators(self, df):
        """
        Создание технических индикаторов для финансовых данных.
        
        Args:
            df (pandas.DataFrame): DataFrame с данными о цене золота
            
        Returns:
            pandas.DataFrame: DataFrame с добавленными техническими индикаторами
        """
        logger.info("Создание технических индикаторов")
        
        try:
            # Создаем новый пустой DataFrame с тем же индексом
            # Это безопасный способ избежать проблем с многомерными данными
            new_data = pd.DataFrame(index=df.index)
            
            # Преобразуем все базовые колонки из исходного датафрейма
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            for col in required_columns:
                if col in df.columns:
                    try:
                        # Если это DataFrame с формой (n, 1), извлекаем значения первого столбца
                        if isinstance(df[col], pd.DataFrame) and df[col].shape[1] == 1:
                            new_data[col] = df[col].iloc[:, 0]
                        # Если это Series или другой тип, просто копируем
                        else:
                            new_data[col] = df[col]
                    except Exception as e:
                        logger.error(f"Ошибка при извлечении {col}: {str(e)}")
                else:
                    logger.error(f"Отсутствует обязательная колонка: {col}")
                    return None
            
            # Используем новый DataFrame для дальнейшей обработки
            data = new_data.copy()
            
            # Применяем дополнительную очистку данных и преобразование типов
            data = self._ensure_all_columns_are_series(data)
            
            # Удаляем строки с некорректными значениями OHLCV
            try:
                rows_before = len(data)
                # Фильтруем строки с корректными значениями
                data = data[(
                    (data['Open'] > 0) &
                    (data['High'] > 0) &
                    (data['Low'] > 0) &
                    (data['Close'] > 0) &
                    (data['Volume'] > 0) &
                    (data['High'] >= data['Low'])
                )]
                rows_after = len(data)
                logger.info(f"Удалено {rows_before - rows_after} строк с некорректными значениями OHLCV")
            except Exception as e:
                logger.error(f"Ошибка при удалении NaN: {data.columns.tolist()}")
            
            # Проверяем и нормализуем форматы всех столбцов
            data = self._ensure_all_columns_are_series(data)
            
            # Проверяем корректность базовых столбцов
            data = self._ensure_correct_column_format(data)
            if data is None:
                logger.error("Не удалось правильно отформатировать данные")
                return None
                
            # Убедимся, что все столбцы в правильном формате перед созданием индикаторов
            data = self._ensure_all_columns_are_series(data)
            
            # Убедимся, что генерируем все необходимые окна, которые ожидают модели
            windows = [5, 10, 20, 50, 100]  # Полный набор окон для всех моделей
            
            for window in windows:
                try:
                    # Ещё раз проверяем, что данные в правильном формате
                    data = self._ensure_all_columns_are_series(data)
                    
                    # Проверяем, что данные точно являются Series
                    if not isinstance(data['Close'], pd.Series):
                        logger.warning(f"'Close' всё ещё не Series, а {type(data['Close'])}. Преобразуем явно.")
                        data.loc[:, 'Close'] = pd.Series(data['Close'], index=data.index)
                    
                    # Рассчитываем показатели
                    ma = data['Close'].rolling(window=window).mean()
                    
                    # Всегда используем консистентные имена и .loc для избежания SettingWithCopyWarning
                    data.loc[:, f'MA_{window}'] = ma
                    
                    # Рассчитываем только если обе серии в правильном формате
                    if isinstance(ma, pd.Series):
                        data.loc[:, f'MA_ratio_{window}'] = data['Close'] / ma
                    else:
                        logger.error(f"MA_{window} не является Series, а {type(ma)}")
                except Exception as e:
                    logger.error(f"Ошибка при создании скользящих средних: {str(e)}")
                    logger.error(f"Close column type: {type(data['Close'])}")
                    logger.error(f"Close column shape (if ndarray): {getattr(data['Close'], 'shape', 'Not ndarray')}")
                    logger.error(f"Трассировка: {traceback.format_exc()}")
            
            # 2. Экспоненциальные скользящие средние (EMA) 
            # Используем те же окна, что и для MA, чтобы обеспечить генерацию всех необходимых признаков
            for window in windows:
                try:
                    # Ещё раз проверяем, что данные в правильном формате
                    data = self._ensure_all_columns_are_series(data)
                    
                    if not isinstance(data['Close'], pd.Series):
                        data.loc[:, 'Close'] = pd.Series(data['Close'], index=data.index)
                    
                    # Рассчитываем EMA
                    ema = data['Close'].ewm(span=window, adjust=False).mean()
                    data.loc[:, f'EMA_{window}'] = ema
                    
                    # Рассчитываем соотношение
                    if isinstance(ema, pd.Series):
                        ratio = data['Close'] / ema
                        data.loc[:, f'EMA_ratio_{window}'] = ratio
                    else:
                        logger.error(f"EMA_{window} не является Series, а {type(ema)}")
                        
                    # Проверяем, что все данные генерируются правильно
                    if window == 10:
                        logger.info(f"EMA_{window} type: {type(data[f'EMA_{window}'])}")
                        logger.info(f"EMA_ratio_{window} type: {type(data[f'EMA_ratio_{window}'])}")
                except Exception as e:
                    logger.error(f"Ошибка при создании EMA: {str(e)}")
                    logger.error(f"Трассировка: {traceback.format_exc()}")
            
            # 3. Полосы Боллинджера (Bollinger Bands)
            for window in [20]:
                try:
                    # Ещё раз проверяем, что данные в правильном формате
                    data = self._ensure_all_columns_are_series(data)
                    
                    if not isinstance(data['Close'], pd.Series):
                        data.loc[:, 'Close'] = pd.Series(data['Close'], index=data.index)
                    
                    rolling_mean = data['Close'].rolling(window=window).mean()
                    rolling_std = data['Close'].rolling(window=window).std()
                    
                    # Сохраняем старые имена для обратной совместимости с использованием .loc
                    data.loc[:, f'BB_upper_{window}'] = rolling_mean + (rolling_std * 2)
                    data.loc[:, f'BB_lower_{window}'] = rolling_mean - (rolling_std * 2)
                    data.loc[:, f'BB_width_{window}'] = (data[f'BB_upper_{window}'] - data[f'BB_lower_{window}']) / rolling_mean
                    data.loc[:, f'BB_position_{window}'] = (data['Close'] - data[f'BB_lower_{window}']) / (data[f'BB_upper_{window}'] - data[f'BB_lower_{window}'])
                    
                    # Добавляем также и новые имена для совместимости с использованием .loc
                    data.loc[:, f'BB_Upper_{window}'] = data[f'BB_upper_{window}']
                    data.loc[:, f'BB_Lower_{window}'] = data[f'BB_lower_{window}']
                    data.loc[:, f'BB_Width_{window}'] = data[f'BB_width_{window}']
                    data.loc[:, f'BB_Position_{window}'] = data[f'BB_position_{window}']
                except Exception as e:
                    logger.error(f"Ошибка при создании полос Боллинджера: {str(e)}")
                    logger.error(f"Трассировка: {traceback.format_exc()}")
            
            # 4. Индекс относительной силы (RSI)
            # Добавляем RSI_7, который ожидает модель XGBoost
            for window in [7, 14, 21]:
                try:
                    # Ещё раз проверяем, что данные в правильном формате
                    data = self._ensure_all_columns_are_series(data)
                    
                    if not isinstance(data['Close'], pd.Series):
                        data.loc[:, 'Close'] = pd.Series(data['Close'], index=data.index)
                    
                    delta = data['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                    rs = gain / loss
                    data.loc[:, f'RSI_{window}'] = 100 - (100 / (1 + rs))
                    
                    # Проверяем генерацию RSI для ключевых окон
                    if window == 7 or window == 14:
                        logger.info(f"RSI_{window} type: {type(data[f'RSI_{window}'])}")
                except Exception as e:
                    logger.error(f"Ошибка при создании RSI: {str(e)}")
                    logger.error(f"Трассировка: {traceback.format_exc()}")
            
            # 5. MACD (Moving Average Convergence Divergence)
            try:
                # Ещё раз проверяем, что данные в правильном формате
                data = self._ensure_all_columns_are_series(data)
                
                if not isinstance(data['Close'], pd.Series):
                    data['Close'] = pd.Series(data['Close'], index=data.index)
                
                # Создаем с новыми именами и использованием .loc
                data.loc[:, 'MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
                data.loc[:, 'MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
                data.loc[:, 'MACD_Hist'] = data['MACD'] - data['MACD_Signal']
                data.loc[:, 'MACD_Hist_Change'] = data['MACD_Hist'].diff()
                
                # Добавляем также старые имена для совместимости с моделями, используя .loc
                data.loc[:, 'MACD_line'] = data['MACD']
                data.loc[:, 'MACD_signal'] = data['MACD_Signal']
                data.loc[:, 'MACD_histogram'] = data['MACD_Hist']
            except Exception as e:
                logger.error(f"Ошибка при создании MACD: {str(e)}")
                logger.error(f"Трассировка: {traceback.format_exc()}")
            
            # 6. Замена бесконечности и NaN значениями
            data = data.replace([np.inf, -np.inf], np.nan)
                # 6. Стохастический осциллятор (Stochastic Oscillator)
            for window in [14]:
                try:
                    # Ещё раз проверяем, что данные в правильном формате
                    data = self._ensure_all_columns_are_series(data)
                    
                    if not isinstance(data['Close'], pd.Series) or not isinstance(data['High'], pd.Series) or not isinstance(data['Low'], pd.Series):
                        data.loc[:, 'Close'] = pd.Series(data['Close'], index=data.index)
                        data.loc[:, 'High'] = pd.Series(data['High'], index=data.index)
                        data.loc[:, 'Low'] = pd.Series(data['Low'], index=data.index)
                    
                    # Расчет стохастического осциллятора
                    low_min = data['Low'].rolling(window=window).min()
                    high_max = data['High'].rolling(window=window).max()
                    
                    # %K - текущая цена закрытия относительно диапазона (последние n дней), используем .loc
                    data.loc[:, f'Stoch_%K_{window}'] = 100 * ((data['Close'] - low_min) / (high_max - low_min))
                    # %D - скользящая средняя %K, используем .loc
                    data.loc[:, f'Stoch_%D_{window}'] = data[f'Stoch_%K_{window}'].rolling(window=3).mean()
                    
                    logger.info(f"Stoch_%K_{window} type: {type(data[f'Stoch_%K_{window}'])}")
                    logger.info(f"Stoch_%D_{window} type: {type(data[f'Stoch_%D_{window}'])}")
                except Exception as e:
                    logger.error(f"Ошибка при создании стохастического осциллятора: {str(e)}")
                    logger.error(f"Трассировка: {traceback.format_exc()}")
            
            # 7. ATR (Average True Range)
            for window in [14]:
                try:
                    # Ещё раз проверяем, что данные в правильном формате
                    data = self._ensure_all_columns_are_series(data)
                    
                    if not isinstance(data['Close'], pd.Series) or not isinstance(data['High'], pd.Series) or not isinstance(data['Low'], pd.Series):
                        data.loc[:, 'Close'] = pd.Series(data['Close'], index=data.index)
                        data.loc[:, 'High'] = pd.Series(data['High'], index=data.index)
                        data.loc[:, 'Low'] = pd.Series(data['Low'], index=data.index)
                    
                    # Расчет ATR
                    high_low = data['High'] - data['Low']
                    high_close = (data['High'] - data['Close'].shift(1)).abs()
                    low_close = (data['Low'] - data['Close'].shift(1)).abs()
                    
                    ranges = pd.concat([high_low, high_close, low_close], axis=1)
                    true_range = ranges.max(axis=1)
                    
                    data.loc[:, f'ATR_{window}'] = true_range.rolling(window=window).mean()
                    logger.info(f"ATR_{window} type: {type(data[f'ATR_{window}'])}")
                except Exception as e:
                    logger.error(f"Ошибка при создании ATR: {str(e)}")
                    logger.error(f"Трассировка: {traceback.format_exc()}")
            
            # 8. Commodity Channel Index (CCI)
            for window in [20]:
                tp = (data['High'] + data['Low'] + data['Close']) / 3
                tp_ma = tp.rolling(window=window).mean()
                tp_md = (tp - tp_ma).abs().rolling(window=window).mean()
                
                data.loc[:, f'CCI_{window}'] = (tp - tp_ma) / (0.015 * tp_md)
        
            # 9. Изменения и доходности
            data.loc[:, 'Price_Change'] = data['Close'].diff()
            data.loc[:, 'Return'] = data['Close'].pct_change()
            
            # 10. Волатильность
            for window in [5, 10, 21]:
                data.loc[:, f'Volatility_{window}'] = data['Return'].rolling(window=window).std()
        
            # 11. High-Low диапазоны
            data.loc[:, 'High_Low_Range'] = data['High'] - data['Low']
            data.loc[:, 'High_Low_Range_Pct'] = (data['High'] - data['Low']) / data['Close']
            
            # 12. Volume Features
            for window in [5, 10, 20]:
                try:
                    # Проверяем и преобразуем данные объема
                    data = self._ensure_all_columns_are_series(data)
                    
                    if not isinstance(data['Volume'], pd.Series):
                        data.loc[:, 'Volume'] = pd.Series(data['Volume'], index=data.index)
                    
                    # Рассчитываем скользящую среднюю объема
                    vol_ma = data['Volume'].rolling(window=window).mean()
                    data.loc[:, f'Volume_MA_{window}'] = vol_ma
                    
                    # Рассчитываем отношение текущего объема к скользящей средней
                    if isinstance(vol_ma, pd.Series):
                        data.loc[:, f'Volume_ratio_{window}'] = data['Volume'] / vol_ma
                    else:
                        logger.error(f"Volume_MA_{window} не является Series, а {type(vol_ma)}")
                except Exception as e:
                    logger.error(f"Ошибка при создании признаков объема: {str(e)}")
                    logger.error(f"Трассировка: {traceback.format_exc()}")
        
            # 13. Объем, умноженный на цену
            try:
                data = self._ensure_all_columns_are_series(data)
                
                if not isinstance(data['Volume'], pd.Series) or not isinstance(data['Close'], pd.Series):
                    data.loc[:, 'Volume'] = pd.Series(data['Volume'], index=data.index)
                    data.loc[:, 'Close'] = pd.Series(data['Close'], index=data.index)
                    
                data.loc[:, 'Volume_Price'] = data['Volume'] * data['Close']
            except Exception as e:
                logger.error(f"Ошибка при расчете Volume_Price: {str(e)}")
        
            logger.info(f"Создано технических индикаторов: {len(data.columns)} столбцов")
        
        except Exception as e:
            logger.error(f"Ошибка при создании индикаторов: {str(e)}")
            logger.error(f"Трассировка: {traceback.format_exc()}")
            return None
        
        # Возвращаем данные с индикаторами
        return data
    
    def _ensure_correct_column_format(self, data):
        """
        Проверяет и исправляет формат столбцов в DataFrame.
        
        Args:
            data (pandas.DataFrame): DataFrame для проверки
            
        Returns:
            pandas.DataFrame: Исправленный DataFrame
        """
        if data is None or not isinstance(data, pd.DataFrame):
            logger.error(f"Некорректный тип данных: {type(data)}")
            return data
            
        # Проверка наличия минимально необходимых столбцов
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in cols if col not in data.columns]
        if missing_cols:
            logger.error(f"Отсутствуют обязательные столбцы: {missing_cols}")
            # Пытаемся восстановить данные из столбцов в разных регистрах
            alternative_cols = {}
            for col in missing_cols:
                # Проверяем разные варианты написания столбцов
                alternatives = [col.lower(), col.upper(), col.capitalize()]
                for alt in alternatives:
                    if alt in data.columns:
                        alternative_cols[col] = alt
                        break
            
            # Если нашли альтернативы
            if alternative_cols:
                logger.info(f"Найдены альтернативные столбцы: {alternative_cols}")
                # Переименовываем столбцы к ожидаемым названиям
                for original, alt in alternative_cols.items():
                    data[original] = data[alt]
            else:
                # Если не нашли альтернатив, возвращаем None
                logger.error(f"Не удалось найти альтернативы для столбцов: {missing_cols}")
                return None
        
        # Сначала приведем все столбцы к правильному формату
        for col in cols:
            try:
                if col in data.columns:
                    # Обработка различных типов данных столбца
                    if isinstance(data[col], pd.DataFrame):
                        # Если столбец - DataFrame, берем первый столбец из него
                        logger.info(f"Преобразуем DataFrame в Series для столбца {col}")
                        if len(data[col].columns) > 0:
                            # Берем первый столбец
                            data[col] = data[col].iloc[:, 0]
                        else:
                            # Создаем пустую Series с тем же индексом
                            data[col] = pd.Series(index=data.index)
                    elif isinstance(data[col], np.ndarray):
                        # Если столбец - многомерный массив, делаем его плоским
                        if len(data[col].shape) > 1:
                            logger.info(f"Сглаживаем многомерный массив для столбца {col}")
                            # Сгладить массив в 1D
                            data[col] = data[col].flatten() if data[col].size > 0 else np.array([])
                    
                    # Теперь преобразуем в Series
                    if not isinstance(data[col], pd.Series):
                        data[col] = pd.Series(data[col], index=data.index)
                    
                    # Преобразуем в числовой тип
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            except Exception as e:
                logger.error(f"Ошибка при обработке столбца {col}: {str(e)}")
                logger.error(f"Тип данных столбца {col}: {type(data[col])}")
                if hasattr(data[col], 'shape'):
                    logger.error(f"Форма столбца {col}: {data[col].shape}")
        
        # Проверяем наличие всех необходимых столбцов перед удалением NaN
        available_cols = [col for col in cols if col in data.columns]
        if available_cols:
            try:
                # Удаляем строки с NaN во всех доступных ключевых столбцах
                before = len(data)
                data = data.dropna(subset=available_cols, how='all')
                after = len(data)
                if before != after:
                    logger.info(f"Удалено {before - after} строк с некорректными значениями OHLCV")
            except Exception as e:
                logger.error(f"Ошибка при удалении NaN: {str(e)}")
        else:
            logger.warning(f"Не найдены необходимые столбцы для удаления NaN: {cols}")

        
        # Убедимся, что названия столбцов соответствуют ожидаемым
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in expected_columns if col not in data.columns]
        if missing:
            logger.warning(f"DataFrame не содержит все ожидаемые столбцы. Отсутствуют: {missing}")
            
            # Попытка восстановить отсутствующие столбцы из альтернативных названий
            for col in missing:
                # Проверяем разные варианты написания
                for alt in [col.lower(), col.upper(), col.capitalize()]:
                    if alt in data.columns and alt != col:
                        data[col] = data[alt]
                        logger.info(f"Столбец {col} восстановлен из {alt}")
                        break
        
        # Проверяем наличие обязательных столбцов после восстановления
        still_missing = [col for col in expected_columns if col not in data.columns]
        if still_missing:
            logger.error(f"Не удалось восстановить обязательные столбцы: {still_missing}")
            return None
        
        return data
    
    def create_target_variable(self, df, horizon=1, target_type='binary'):
        """
        Создание целевой переменной для прогнозирования.
        
        Args:
            df (pandas.DataFrame): DataFrame с данными
            horizon (int): Горизонт прогнозирования (в днях)
            target_type (str): Тип целевой переменной ('binary', 'regression', 'classification')
            
        Returns:
            pandas.DataFrame: DataFrame с добавленной целевой переменной
        """
        logger.info(f"Создание целевой переменной с горизонтом {horizon} дней, тип: {target_type}")
        
        data = df.copy()
        
        # Создаем будущую цену
        data['Future_Close'] = data['Close'].shift(-horizon)
        
        if target_type == 'binary':
            # Бинарная классификация: вверх(1) или вниз(0)
            data['Target'] = (data['Future_Close'] > data['Close']).astype(int)
            
        elif target_type == 'regression':
            # Регрессия: предсказываем абсолютную цену
            data['Target'] = data['Future_Close']
            
        elif target_type == 'classification':
            # Мультиклассовая классификация: сильно вверх, вверх, боковик, вниз, сильно вниз
            # Вычисляем относительное изменение
            data['Price_Change_Pct'] = (data['Future_Close'] - data['Close']) / data['Close'] * 100
            
            # Определяем пороги для классификации
            thresholds = [-2, -0.5, 0.5, 2]  # Пороги в процентах
            
            # Создаем категории: 0 (сильно вниз), 1 (вниз), 2 (боковик), 3 (вверх), 4 (сильно вверх)
            data['Target'] = pd.cut(
                data['Price_Change_Pct'], 
                bins=[-float('inf')] + thresholds + [float('inf')], 
                labels=[0, 1, 2, 3, 4]
            ).astype(int)
            
            # Удаляем промежуточный столбец
            data.drop('Price_Change_Pct', axis=1, inplace=True)
        
        else:
            raise ValueError(f"Неизвестный тип целевой переменной: {target_type}")
        
        return data
    
    def scale_features(self, df, fit=True):
        """
        Масштабирование признаков.
        
        Args:
            df (pandas.DataFrame): DataFrame с признаками
            fit (bool): True для обучения скейлера, False только для трансформации
            
        Returns:
            pandas.DataFrame: DataFrame с масштабированными признаками
        """
        if self.scaler is None:
            logger.info("Масштабирование не применяется")
            return df
        
        # Копируем данные и проверяем типы имен колонок
        data = df.copy()
        
        # Проверяем, есть ли MultiIndex в колонках и преобразуем все имена в строки
        if isinstance(data.columns, pd.MultiIndex):
            logger.info("Обнаружен MultiIndex в колонках, преобразуем в строки")
            # Преобразуем MultiIndex в обычные строковые колонки
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
        else:
            # Проверяем на другие нестроковые типы имен колонок
            column_types = set(type(col) for col in data.columns)
            if len(column_types) > 1 or str not in column_types:
                logger.info(f"Обнаружены разные типы имен колонок: {column_types}, преобразуем все в строки")
                data.columns = data.columns.astype(str)
        
        # Получаем список числовых столбцов после приведения имен к единому формату
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Исключаем целевую переменную из масштабирования
        if 'Target' in numeric_columns:
            numeric_columns.remove('Target')
        
        if 'Future_Close' in numeric_columns:
            numeric_columns.remove('Future_Close')
        
        if not numeric_columns:
            logger.warning("Нет числовых признаков для масштабирования")
            return data
        
        logger.info(f"Масштабирование {len(numeric_columns)} признаков с методом {self.scaling_method}")
        
        # Получаем подмножество данных для масштабирования
        X = data[numeric_columns]
        
        # Заменяем бесконечности на NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Заполняем NaN медианными значениями по каждому столбцу отдельно
        for col in X.columns:
            # Используем медиану вместо среднего для устойчивости к выбросам
            if X[col].isna().any():
                median_val = X[col].median()
                # Если и медиана NaN, используем 0
                if pd.isna(median_val):
                    X[col] = X[col].fillna(0)
                else:
                    X[col] = X[col].fillna(median_val)
        
        # Проверяем, что не осталось NaN
        if X.isna().any().any():
            logger.warning("После заполнения остались NaN значения. Заполняем нулями.")
            X = X.fillna(0)
        
        # Масштабируем данные
        if fit:
            scaled_features = self.scaler.fit_transform(X)
        else:
            scaled_features = self.scaler.transform(X)
        
        # Преобразуем обратно в DataFrame
        scaled_df = pd.DataFrame(scaled_features, index=data.index, columns=numeric_columns)
        
        # Заменяем исходные столбцы на масштабированные
        for col in numeric_columns:
            data[col] = scaled_df[col]
        
        return data
    
    def prepare_features(self, df, horizon=1, target_type='binary', add_technical=True, scale=True):
        """
        Комплексная подготовка признаков для модели.
        
        Args:
            df (pandas.DataFrame): Исходный DataFrame с OHLCV данными
            horizon (int): Горизонт прогнозирования (в днях)
            target_type (str): Тип целевой переменной
            add_technical (bool): Добавлять ли технические индикаторы
            scale (bool): Масштабировать ли признаки
            
        Returns:
            pandas.DataFrame: Подготовленный DataFrame с признаками и целевой переменной
        """
        logger.info("Подготовка признаков для модели")
        
        data = df.copy()
        
        # Добавляем технические индикаторы
        if add_technical:
            data = self.create_technical_indicators(data)
        
        # Создаем целевую переменную
        data = self.create_target_variable(data, horizon, target_type)
        
        # Масштабируем признаки, если требуется
        if scale and self.scaler is not None:
            data = self.scale_features(data)
        
        # Удаляем строки с NaN (в начале из-за скользящих окон)
        data_clean = data.dropna()
        
        rows_dropped = len(data) - len(data_clean)
        logger.info(f"Удалено {rows_dropped} строк с пропущенными значениями ({rows_dropped/len(data)*100:.2f}%)")
        
        return data_clean
    
    def split_train_test(self, df, test_size=0.2, validation_size=0.1):
        """
        Разделение данных на обучающий, валидационный и тестовый наборы.
        
        Args:
            df (pandas.DataFrame): DataFrame с признаками и целевой переменной
            test_size (float): Доля тестового набора
            validation_size (float): Доля валидационного набора
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info(f"Разделение данных: тестовый={test_size}, валидационный={validation_size}")
        
        # Разделяем признаки и целевую переменную
        X = df.drop(['Target', 'Future_Close'], axis=1, errors='ignore')
        y = df['Target']
        
        # Определяем границы тестового и валидационного наборов
        test_border = int(len(df) * (1 - test_size))
        val_border = int(len(df) * (1 - test_size - validation_size))
        
        # Разделяем данные без перемешивания (важно для временных рядов)
        X_train = X.iloc[:val_border]
        X_val = X.iloc[val_border:test_border]
        X_test = X.iloc[test_border:]
        
        y_train = y.iloc[:val_border]
        y_val = y.iloc[val_border:test_border]
        y_test = y.iloc[test_border:]
        
        logger.info(f"Размеры наборов: обучающий={len(X_train)}, валидационный={len(X_val)}, тестовый={len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def prepare_sequence_data(self, df, sequence_length=10, horizon=1, target_type='binary'):
        """
        Подготовка последовательностей для рекуррентных нейронных сетей.
        
        Args:
            df (pandas.DataFrame): DataFrame с признаками
            sequence_length (int): Длина последовательности (окна)
            horizon (int): Горизонт прогнозирования
            target_type (str): Тип целевой переменной
            
        Returns:
            tuple: (X_sequences, y_targets)
        """
        logger.info(f"Подготовка последовательностей длиной {sequence_length} для нейросети")
        
        # Подготавливаем данные с признаками и целевой переменной
        prepared_data = self.prepare_features(df, horizon, target_type)
        
        # Получаем только числовые признаки
        numeric_data = prepared_data.select_dtypes(include=[np.number])
        
        # Исключаем целевую переменную из признаков
        if 'Target' in numeric_data.columns:
            y = numeric_data['Target'].values
            X = numeric_data.drop(['Target', 'Future_Close'], axis=1, errors='ignore').values
        else:
            raise ValueError("Целевая переменная 'Target' не найдена в данных")
        
        # Создаем последовательности
        X_sequences = []
        y_targets = []
        
        for i in range(len(X) - sequence_length - horizon + 1):
            X_sequences.append(X[i:(i + sequence_length)])
            y_targets.append(y[i + sequence_length + horizon - 1])
        
        # Преобразуем в numpy массивы
        X_sequences = np.array(X_sequences)
        y_targets = np.array(y_targets)
        
        logger.info(f"Создано {len(X_sequences)} последовательностей формы {X_sequences.shape}")
        
        return X_sequences, y_targets
    
    def get_feature_importance(self, model, feature_names):
        """
        Получение важности признаков из обученной модели.
        
        Args:
            model: Обученная модель (XGBoost, LightGBM, RandomForest и т.д.)
            feature_names (list): Список названий признаков
            
        Returns:
            pandas.DataFrame: DataFrame с важностью признаков
        """
        try:
            # Получаем важность признаков из модели
            importance = model.feature_importances_
            
            # Создаем DataFrame
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            })
            
            # Сортируем по важности
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            
            return feature_importance
        except:
            logger.warning("Не удалось получить важность признаков из модели")
            return None


if __name__ == "__main__":
    # Пример использования
    from data_loader import GoldDataLoader
    
    # Загружаем данные
    loader = GoldDataLoader()
    gold_data = loader.download_data(period="2y")
    
    if gold_data is not None:
        # Создаем генератор признаков
        feature_gen = FeatureGenerator(scaling_method='standard')
        
        # Подготавливаем признаки
        features_df = feature_gen.prepare_features(gold_data, horizon=1, target_type='binary')
        
        # Вывод информации
        print(f"Подготовлено {len(features_df)} строк с {len(features_df.columns)} признаками")
        print("\nПример данных с признаками:")
        print(features_df.tail())
