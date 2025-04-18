#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для создания и обработки признаков (feature engineering) для прогнозирования цены золота.
"""

import numpy as np
import pandas as pd
import logging
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
    
    def create_technical_indicators(self, df):
        """
        Создание технических индикаторов на основе OHLCV данных.
        
        Args:
            df (pandas.DataFrame): DataFrame с OHLCV данными
            
        Returns:
            pandas.DataFrame: DataFrame с добавленными техническими индикаторами
        """
        logger.info("Создание технических индикаторов")
        
        # Создаем копию DataFrame
        data = df.copy()

        # Приведение ключевых столбцов к float и очистка NaN
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        before = len(data)
        data = data.dropna(subset=cols, how='all')
        after = len(data)
        if before != after:
            logger.info(f"Удалено {before - after} строк с некорректными значениями OHLCV")

        
        # Убедимся, что названия столбцов соответствуют ожидаемым
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in expected_columns):
            logger.warning(f"DataFrame не содержит все ожидаемые столбцы: {expected_columns}")
        
        # 1. Скользящие средние (Moving Averages)
        windows = [5, 10, 20, 50, 100]
        for window in windows:
            data[f'MA_{window}'] = data['Close'].rolling(window=window).mean()
            data[f'MA_ratio_{window}'] = data['Close'] / data[f'MA_{window}']
        
        # 2. Экспоненциальные скользящие средние (EMA)
        for window in windows:
            data[f'EMA_{window}'] = data['Close'].ewm(span=window, adjust=False).mean()
            data[f'EMA_ratio_{window}'] = data['Close'] / data[f'EMA_{window}']
        
        # 3. Индекс относительной силы (RSI)
        for window in [7, 14, 21]:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            
            rs = gain / loss
            data[f'RSI_{window}'] = 100 - (100 / (1 + rs))
        
        # 4. Схождение/расхождение скользящих средних (MACD)
        data['MACD_line'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD_signal'] = data['MACD_line'].ewm(span=9, adjust=False).mean()
        data['MACD_histogram'] = data['MACD_line'] - data['MACD_signal']
        
        # 5. Полосы Боллинджера (Bollinger Bands)
        for window in [20]:
            rolling_mean = data['Close'].rolling(window=window).mean()
            rolling_std = data['Close'].rolling(window=window).std()
            
            data[f'BB_upper_{window}'] = rolling_mean + (rolling_std * 2)
            data[f'BB_lower_{window}'] = rolling_mean - (rolling_std * 2)
            data[f'BB_width_{window}'] = (data[f'BB_upper_{window}'] - data[f'BB_lower_{window}']) / rolling_mean
            data[f'BB_position_{window}'] = (data['Close'] - data[f'BB_lower_{window}']) / (data[f'BB_upper_{window}'] - data[f'BB_lower_{window}'])
        
        # 6. Стохастический осциллятор (Stochastic Oscillator)
        for window in [14]:
            low_min = data['Low'].rolling(window=window).min()
            high_max = data['High'].rolling(window=window).max()
            
            data[f'Stoch_%K_{window}'] = 100 * ((data['Close'] - low_min) / (high_max - low_min))
            data[f'Stoch_%D_{window}'] = data[f'Stoch_%K_{window}'].rolling(window=3).mean()
        
        # 7. Среднее истинное отклонение (ATR - Average True Range)
        for window in [14]:
            high_low = data['High'] - data['Low']
            high_close = (data['High'] - data['Close'].shift(1)).abs()
            low_close = (data['Low'] - data['Close'].shift(1)).abs()
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            
            data[f'ATR_{window}'] = true_range.rolling(window=window).mean()
        
        # 8. Commodity Channel Index (CCI)
        for window in [20]:
            tp = (data['High'] + data['Low'] + data['Close']) / 3
            tp_ma = tp.rolling(window=window).mean()
            tp_md = (tp - tp_ma).abs().rolling(window=window).mean()
            
            data[f'CCI_{window}'] = (tp - tp_ma) / (0.015 * tp_md)
        
        # 9. Изменения и доходности
        data['Price_Change'] = data['Close'].diff()
        data['Return'] = data['Close'].pct_change()
        
        # 10. Волатильность
        for window in [5, 10, 21]:
            data[f'Volatility_{window}'] = data['Return'].rolling(window=window).std()
        
        # 11. High-Low диапазоны
        data['High_Low_Range'] = data['High'] - data['Low']
        data['High_Low_Range_Pct'] = (data['High'] - data['Low']) / data['Close']
        
        # 12. Volume Features
        for window in [5, 10, 20]:
            data[f'Volume_MA_{window}'] = data['Volume'].rolling(window=window).mean()
            data[f'Volume_ratio_{window}'] = data['Volume'] / data[f'Volume_MA_{window}']
        
        # 13. Объем, умноженный на цену
        data['Volume_Price'] = data['Volume'] * data['Close']
        
        logger.info(f"Создано {len(data.columns) - len(df.columns)} новых признаков")
        
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
        
        # Копируем данные и получаем список числовых столбцов
        data = df.copy()
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
        
        # Заменяем бесконечности на NaN, затем NaN на среднее значение
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(X.mean(), inplace=True)
        
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
