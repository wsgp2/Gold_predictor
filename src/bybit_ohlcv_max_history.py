import requests
from datetime import datetime

API_URL = "https://api.bybit.com/v5/market/kline"
SYMBOL = "XAUTUSDT"
CATEGORY = "linear"

params = {
    'category': CATEGORY,
    'symbol': SYMBOL,
    'interval': 'D',
    'limit': 200,
}

resp = requests.get(API_URL, params=params, timeout=10)
data = resp.json()

if data.get('retCode') != 0 or not data.get('result') or not data['result'].get('list'):
    print(f"Нет данных: {data}")
else:
    candles = data['result']['list']
    first = candles[-1]
    last = candles[0]
    ts_first = int(first[0]) // 1000
    ts_last = int(last[0]) // 1000
    print(f"Самая ранняя дневная свеча: {datetime.utcfromtimestamp(ts_first)}")
    print(f"Самая последняя дневная свеча: {datetime.utcfromtimestamp(ts_last)}")
    print(f"Всего дневных свечей в одном запросе: {len(candles)}")
