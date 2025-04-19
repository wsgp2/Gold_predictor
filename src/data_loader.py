#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для загрузки исторических OHLCV данных по золоту через Yahoo Finance API.
"""

import os
import logging
import pandas as pd
import yfinance as yf
import time
import random
from datetime import datetime, timedelta

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GoldDataLoader:
    """Класс для загрузки и обработки данных по золоту."""
    
    def __init__(self, ticker="GC=F", data_dir="../data"):
        """
        Инициализация загрузчика данных.
        
        Args:
            ticker (str): Тикер золота на Yahoo Finance (GC=F для фьючерсов на золото)
            data_dir (str): Директория для сохранения данных
        """
        self.ticker = ticker
        self.data_dir = data_dir
        
        # Создание директории для данных, если она не существует
        os.makedirs(data_dir, exist_ok=True)
        
    def download_data(self, start_date=None, end_date=None, period="5y", interval="1d", max_retries=3):
        """
        Загрузка исторических данных в указанном диапазоне дат.
        
        Args:
            start_date (str, optional): Начальная дата в формате 'YYYY-MM-DD'
            end_date (str, optional): Конечная дата в формате 'YYYY-MM-DD'
            period (str, optional): Период для загрузки, если даты не указаны 
                                    (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max)
            interval (str, optional): Интервал данных (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            max_retries (int): Максимальное количество попыток загрузки при ошибке
        
        Returns:
            pandas.DataFrame: DataFrame с OHLCV данными или None при ошибке
        """
        logger.info(f"Загрузка данных для {self.ticker} с периодом {period} и интервалом {interval}")
        
        # Проверяем, есть ли кэшированные данные для использования в случае ошибки
        cached_data = self._get_cached_data_fallback()
        
        for attempt in range(max_retries):
            try:
                # Если даты не указаны, используем period
                if start_date is None and end_date is None:
                    data = yf.download(
                        self.ticker, 
                        period=period, 
                        interval=interval, 
                        auto_adjust=True,
                        progress=False  # Отключаем индикатор прогресса для уменьшения вывода
                    )
                else:
                    data = yf.download(
                        self.ticker, 
                        start=start_date, 
                        end=end_date, 
                        interval=interval, 
                        auto_adjust=True,
                        progress=False
                    )
                
                # Если данные успешно загружены
                if not data.empty:
                    logger.info(f"Данные успешно загружены с попытки {attempt+1}")
                    return data
                
                logger.warning(f"Попытка {attempt+1}/{max_retries} не удалась, данные пусты")
            except Exception as e:
                logger.warning(f"Попытка {attempt+1}/{max_retries} не удалась: {str(e)}")
                
                # Экспоненциальная задержка между попытками
                if attempt < max_retries - 1:  # Если это не последняя попытка
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"Повторная попытка через {delay:.2f} секунд...")
                    time.sleep(delay)
        
        # Если все попытки не удались, используем кэшированные данные
        if cached_data is not None and not cached_data.empty:
            logger.warning(f"Не удалось загрузить свежие данные. Используем кэшированные данные от {cached_data.index[-1]}")
            return cached_data
            
        logger.error(f"Все {max_retries} попыток загрузки данных для {self.ticker} не удались")
        return None
        
    def _get_cached_data_fallback(self):
        """
        Получает последние кэшированные данные для использования при ошибках загрузки.
        
        Returns:
            pandas.DataFrame: Кэшированные данные или None, если их нет
        """
        try:
            # Проверяем наличие последнего сохраненного файла данных
            cache_file = os.path.join(self.data_dir, f"{self.ticker.replace('=', '_')}_latest.csv")
            if os.path.exists(cache_file):
                # Загружаем данные с явным указанием формата даты
                data = pd.read_csv(cache_file, index_col=0, parse_dates=True, date_format='%Y-%m-%d')
                
                # Проверяем и исправляем структуру данных
                data = self._ensure_correct_column_format(data)
                
                logger.info(f"Загружены кэшированные данные от {data.index[-1]}")
                return data
                
            # Если нет последнего файла, ищем любой файл с данными
            for file in os.listdir(self.data_dir):
                if file.endswith('.csv') and self.ticker.replace('=', '_') in file:
                    full_path = os.path.join(self.data_dir, file)
                    data = pd.read_csv(full_path, index_col=0, parse_dates=True, date_format='%Y-%m-%d')
                    
                    # Проверяем и исправляем структуру данных
                    data = self._ensure_correct_column_format(data)
                    
                    logger.info(f"Загружены альтернативные кэшированные данные из {file}")
                    return data
                    
        except Exception as e:
            logger.error(f"Ошибка при загрузке кэшированных данных: {str(e)}")
        
        return None

        # --- Приведение MultiIndex к обычному индексу для совместимости ---
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            logger.info("Столбцы с MultiIndex преобразованы к обычному индексу")
        
        logger.info(f"Загружено {len(data)} строк данных")
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
            
        # Проверяем типы столбцов
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in expected_columns:
            # Если столбца нет - ничего не делаем
            if col not in data.columns:
                continue
                
            # Если столбец - DataFrame вместо Series
            if isinstance(data[col], pd.DataFrame):
                logger.info(f"Конвертируем столбец {col} из DataFrame в Series")
                # Берем первый столбец внутреннего DataFrame
                if len(data[col].columns) > 0:
                    first_col = data[col].columns[0]
                    data[col] = data[col][first_col]
                else:
                    # Если нет столбцов, создаем пустую Series
                    data[col] = pd.Series(index=data.index)
                    
            # Проверяем, что тип числовой
            if not pd.api.types.is_numeric_dtype(data[col]):
                # Преобразовываем в числовой тип
                data[col] = pd.to_numeric(data[col], errors='coerce')
                
        # Удаляем строки, где все значения в ожидаемых столбцах - NaN
        existing_cols = [col for col in expected_columns if col in data.columns]
        if existing_cols:
            data = data.dropna(subset=existing_cols, how='all')
            
        return data
        
    def save_data(self, data, filename=None):
        """
        Сохранение данных в CSV файл.
        
        Args:
            data (pandas.DataFrame): DataFrame с данными для сохранения
            filename (str, optional): Имя файла для сохранения
        
        Returns:
            str: Путь к сохраненному файлу
        """
        if data is None or data.empty:
            logger.warning("Нет данных для сохранения")
            return None
        
        if filename is None:
            # Создаем имя файла с текущей датой
            today = datetime.now().strftime('%Y%m%d')
            filename = f"{self.ticker.replace('=', '_')}_history_{today}.csv"
        
        file_path = os.path.join(self.data_dir, filename)
        data.to_csv(file_path)
        logger.info(f"Данные сохранены в {file_path}")
        
        return file_path
    
    def load_data(self, filename=None):
        """
        Загрузка данных из CSV файла.
        
        Args:
            filename (str, optional): Имя файла для загрузки. Если не указано,
                                      загружается самый свежий файл.
        
        Returns:
            pandas.DataFrame: DataFrame с загруженными данными
        """
        if filename is None:
            # Ищем самый свежий файл
            files = [f for f in os.listdir(self.data_dir) if f.startswith(self.ticker.replace('=', '_'))]
            if not files:
                logger.warning(f"В директории {self.data_dir} не найдено файлов с данными")
                return None
            
            # Сортировка файлов по дате в имени
            files.sort(reverse=True)
            filename = files[0]
        
        file_path = os.path.join(self.data_dir, filename)
        
        try:
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            logger.info(f"Данные загружены из {file_path}: {len(data)} строк")
            return data
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных из {file_path}: {e}")
            return None
    
    def get_latest_data(self, days=30, save=True):
        """
        Получение самых свежих данных за указанное количество дней.
        
        Args:
            days (int): Количество дней для загрузки
            save (bool): Сохранять ли данные в файл
        
        Returns:
            pandas.DataFrame: DataFrame с последними данными
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = self.download_data(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            interval='1d'
        )
        
        if save and data is not None:
            self.save_data(data, f"{self.ticker.replace('=', '_')}_latest.csv")
        
        return data
    
    def update_dataset(self):
        """
        Обновление набора данных до текущей даты.
        
        Returns:
            pandas.DataFrame: Обновленный DataFrame с данными
        """
        try:
            # Загружаем существующие данные
            existing_data = self.load_data()
            
            if existing_data is None:
                # Если данных нет, загружаем за последние 5 лет
                logger.info("Существующие данные не найдены, загрузка исторических данных за 5 лет")
                data = self.download_data(period="5y")
                if data is not None:
                    self.save_data(data)
                return data
            
            # Определяем последнюю дату в существующих данных
            last_date = existing_data.index[-1]
            # Приведение last_date к типу pandas.Timestamp (datetime)
            import pandas as pd
            if isinstance(last_date, str):
                last_date = pd.to_datetime(last_date)
            
            # Получаем текущую дату
            current_date = datetime.now()
            
            # Если прошло меньше дня, не обновляем
            if (current_date - last_date).days < 1:
                logger.info("Данные уже актуальны")
                return existing_data
            
            # Загружаем новые данные с последней даты до текущей
            try:
                new_data = self.download_data(
                    start_date=(last_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                    end_date=current_date.strftime('%Y-%m-%d')
                )
                
                if new_data is None or new_data.empty:
                    logger.info("Новых данных нет или возникла ошибка при загрузке")
                    return existing_data
                
                # Объединяем существующие и новые данные
                updated_data = pd.concat([existing_data, new_data])
                
                # Удаляем возможные дубликаты
                updated_data = updated_data[~updated_data.index.duplicated(keep='last')]
                
                # Сохраняем обновленные данные
                self.save_data(updated_data)
                
                logger.info(f"Данные обновлены до {updated_data.index[-1].strftime('%Y-%m-%d')}")
                return updated_data
                
            except Exception as e:
                logger.error(f"Ошибка при загрузке новых данных: {str(e)}")
                logger.info("Используем существующие данные без обновления")
                return existing_data
                
        except Exception as e:
            logger.error(f"Ошибка при обновлении набора данных: {str(e)}")
            
            # Пытаемся получить любые доступные данные как fallback
            fallback_data = self._get_cached_data_fallback()
            if fallback_data is not None:
                logger.info("Используем резервные кэшированные данные")
                return fallback_data
                
            # Если все способы не удались, пытаемся загрузить данные за последний месяц
            logger.info("Попытка загрузить данные за последний месяц как последнее средство")
            try:
                return self.download_data(period="1mo")
            except Exception as e2:
                logger.error(f"Не удалось загрузить данные даже за последний месяц: {str(e2)}")
                return None


if __name__ == "__main__":
    # Пример использования
    loader = GoldDataLoader()
    
    # Загрузка исторических данных
    historical_data = loader.download_data(period="5y")
    
    if historical_data is not None:
        # Сохранение данных
        loader.save_data(historical_data)
        
        # Вывод информации о данных
        print(f"Загружено {len(historical_data)} строк данных с {historical_data.index[0]} по {historical_data.index[-1]}")
        print("\nПример данных:")
        print(historical_data.tail())
