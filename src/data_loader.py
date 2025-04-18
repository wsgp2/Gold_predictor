#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для загрузки исторических OHLCV данных по золоту через Yahoo Finance API.
"""

import os
import logging
import pandas as pd
import yfinance as yf
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
        
    def download_data(self, start_date=None, end_date=None, period="5y", interval="1d"):
        """
        Загрузка исторических данных в указанном диапазоне дат.
        
        Args:
            start_date (str, optional): Начальная дата в формате 'YYYY-MM-DD'
            end_date (str, optional): Конечная дата в формате 'YYYY-MM-DD'
            period (str, optional): Период для загрузки, если даты не указаны 
                                    (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max)
            interval (str, optional): Интервал данных (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            pandas.DataFrame: DataFrame с OHLCV данными
        """
        logger.info(f"Загрузка данных для {self.ticker} с периодом {period} и интервалом {interval}")
        
        # Если даты не указаны, используем period
        if start_date is None and end_date is None:
            data = yf.download(
                self.ticker, 
                period=period, 
                interval=interval, 
                auto_adjust=True
            )
        else:
            data = yf.download(
                self.ticker, 
                start=start_date, 
                end=end_date, 
                interval=interval, 
                auto_adjust=True
            )
        
        # Проверка успешности загрузки
        if data.empty:
            logger.error(f"Ошибка загрузки данных для {self.ticker}")
            return None

        # --- Приведение MultiIndex к обычному индексу для совместимости ---
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            logger.info("Столбцы с MultiIndex преобразованы к обычному индексу")
        
        logger.info(f"Загружено {len(data)} строк данных")
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
        new_data = self.download_data(
            start_date=(last_date + timedelta(days=1)).strftime('%Y-%m-%d'),
            end_date=current_date.strftime('%Y-%m-%d')
        )
        
        if new_data is None or new_data.empty:
            logger.info("Новых данных нет")
            return existing_data
        
        # Объединяем существующие и новые данные
        updated_data = pd.concat([existing_data, new_data])
        
        # Удаляем возможные дубликаты
        updated_data = updated_data[~updated_data.index.duplicated(keep='last')]
        
        # Сохраняем обновленные данные
        self.save_data(updated_data)
        
        logger.info(f"Данные обновлены до {updated_data.index[-1].strftime('%Y-%m-%d')}")
        return updated_data


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
