import requests

url = "https://api.bybit.com/v5/market/tickers?category=linear"
resp = requests.get(url, timeout=15)
data = resp.json()

symbols = []
if data.get("retCode") == 0 and data.get("result") and data["result"].get("list"):
    for ticker in data["result"]["list"]:
        symbol = ticker.get("symbol", "")
        if any(x in symbol.upper() for x in ["GOLD", "XAU"]):
            print(symbol)
        symbols.append(symbol)
else:
    print(f"Ошибка запроса: {data}")

print(f"Всего найдено {len(symbols)} тикеров. Примеры: {symbols[:10]}")
