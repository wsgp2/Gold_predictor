import csv
import os
from datetime import datetime, timedelta
import requests

def get_last_date_from_csv(csv_path):
    if not os.path.exists(csv_path):
        return None
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        if not lines:
            return None
        last_line = lines[-1].strip()
        if not last_line or not last_line[0].isdigit():
            last_line = lines[-2].strip()  # skip empty or header
        last_date = last_line.split(',')[0]
        return datetime.strptime(last_date, '%Y-%m-%d').date()

def fetch_bybit_daily_ohlcv(start_date, end_date, api_key, api_secret):
    """
    Получить дневные свечи XAUTUSDT с Bybit за диапазон дат (включительно)
    """
    API_URL = "https://api.bybit.com/v5/market/kline"
    CATEGORY = "linear"
    SYMBOL = "XAUTUSDT"
    interval = 'D'
    candles = []
    # Bybit: макс 200 свечей за раз
    params = {
        'category': CATEGORY,
        'symbol': SYMBOL,
        'interval': interval,
        'limit': 200,
    }
    resp = requests.get(API_URL, params=params, timeout=10)
    data = resp.json()
    if data.get('retCode') != 0 or not data.get('result') or not data['result'].get('list'):
        raise RuntimeError(f"Ошибка Bybit: {data}")
    for candle in data['result']['list']:
        ts = int(candle[0]) // 1000
        dt = datetime.utcfromtimestamp(ts).date()
        if start_date <= dt <= end_date:
            candles.append({
                'date': dt.strftime('%Y-%m-%d'),
                'open': candle[1],
                'high': candle[2],
                'low': candle[3],
                'close': candle[4],
                'volume': candle[5]
            })
    return sorted(candles, key=lambda x: x['date'])

def append_to_csv(csv_path, candles):
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        for c in candles:
            writer.writerow([c['date'], c['open'], c['high'], c['low'], c['close'], c['volume']])

def update_gold_history_from_bybit(csv_path, api_key, api_secret):
    today = datetime.utcnow().date()
    last_date = get_last_date_from_csv(csv_path)
    if last_date is None:
        print('Ошибка: не удалось определить последнюю дату в CSV')
        return False
    if last_date >= today:
        print('История уже актуальна!')
        return True
    # Получаем недостающие даты
    start_date = last_date + timedelta(days=1)
    print(f'Обновление истории с {start_date} по {today}')
    candles = fetch_bybit_daily_ohlcv(start_date, today, api_key, api_secret)
    if not candles:
        print('Нет новых данных для добавления.')
        return True
    append_to_csv(csv_path, candles)
    print(f'Добавлено {len(candles)} новых строк.')
    return True

if __name__ == '__main__':
    # Пример запуска вручную
    update_gold_history_from_bybit(
        '../data/GC_F_latest.csv',
        'vcpsoaLUBwfj1jPfCz',
        'xf4WxWufuFleJuAjVWXdxRe6WHugoKPbCqQE'
    )
