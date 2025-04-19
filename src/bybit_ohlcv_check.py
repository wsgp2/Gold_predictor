import requests
from datetime import datetime

API_URL = "https://api.bybit.com/v5/market/kline"
SYMBOL = "XAUTUSDT"
CATEGORY = "linear"

intervals = {
    '1m': '1',
    '1h': '60',
    '1d': 'D'
}

def fetch_ohlcv(interval, limit=5):
    params = {
        'category': CATEGORY,
        'symbol': SYMBOL,
        'interval': interval,
        'limit': limit
    }
    resp = requests.get(API_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data.get('retCode') != 0 or not data.get('result') or not data['result'].get('list'):
        print(f"Нет данных для {interval}: {data}")
        return
    print(f"--- {interval} ---")
    for candle in data['result']['list']:
        ts = int(candle[0]) // 1000
        dt = datetime.utcfromtimestamp(ts)
        print(f"{dt} | open={candle[1]} high={candle[2]} low={candle[3]} close={candle[4]} volume={candle[5]}")

if __name__ == "__main__":
    for label, interval in intervals.items():
        fetch_ohlcv(interval)
