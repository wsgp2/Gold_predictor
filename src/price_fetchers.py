import requests
from bs4 import BeautifulSoup
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

INVESTING_URL = "https://www.investing.com/commodities/gold"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
}

def get_gold_price_investing():
    """
    Получить актуальную цену золота (Gold Futures) с Investing.com.
    Returns:
        tuple: (price: float, date: str) или (None, None) при ошибке
    """
    try:
        resp = requests.get(INVESTING_URL, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Находим цену (основной блок)
        price_tag = soup.find("span", attrs={"data-test": "instrument-price-last"})
        if not price_tag:
            price_tag = soup.find("span", class_="text-2xl")  # fallback
        price = float(price_tag.text.replace(",", "").strip())
        # Дата (берём текущую, т.к. сайт не всегда показывает дату)
        date_str = datetime.now().strftime("%Y-%m-%d")
        logger.info(f"[Investing] Gold price: {price} ({date_str})")
        return price, date_str
    except Exception as e:
        logger.error(f"[Investing] Ошибка при получении цены: {e}")
        return None, None

def get_gold_price_yahoo():
    """
    Получить цену золота через yfinance (Yahoo Finance API).
    Returns:
        tuple: (price: float, date: str) или (None, None) при ошибке
    """
    try:
        import yfinance as yf
        gold = yf.Ticker("GC=F")
        data = gold.history(period="1d")
        if data.empty:
            return None, None
        # Берём цену закрытия (Close)
        price = float(data["Close"].iloc[-1])
        date_str = data.index[-1].strftime("%Y-%m-%d")
        logger.info(f"[Yahoo] Gold price: {price} ({date_str})")
        return price, date_str
    except Exception as e:
        logger.error(f"[Yahoo] Ошибка при получении цены: {e}")
        return None, None

def get_latest_gold_price():
    """
    Получить актуальную цену золота: сначала Bybit XAUTUSDT, затем Yahoo и Investing.com как резерв.
    Returns:
        tuple: (price: float, date: str, source: str)
    """
    # Bybit как основной источник
    price, date_str = get_gold_price_bybit_auth(
        'vcpsoaLUBwfj1jPfCz',
        'xf4WxWufuFleJuAjVWXdxRe6WHugoKPbCqQE',
        'XAUTUSDT'
    )
    if price is not None:
        return price, date_str, "Bybit XAUTUSDT"
    # fallback: Yahoo
    price, date_str = get_gold_price_yahoo()
    if price is not None:
        return price, date_str, "Yahoo Finance"
    # fallback: Investing
    price, date_str = get_gold_price_investing()
    if price is not None:
        return price, date_str, "Investing.com"
    return None, None, None

def get_gold_price_bybit_auth(api_key, api_secret, symbol="XAUUSDT"):
    """
    Получить цену золота с Bybit (с авторизацией, если потребуется).
    Args:
        api_key (str): API Key Bybit
        api_secret (str): API Secret Bybit
        symbol (str): тикер, по умолчанию XAUUSDT
    Returns:
        tuple: (price: float, date: str) или (None, None) при ошибке
    """
    import requests
    import time
    import hmac
    import hashlib
    from datetime import datetime

    url = f"https://api.bybit.com/v5/market/tickers?category=linear&symbol={symbol}"
    timestamp = str(int(time.time() * 1000))
    recv_window = "5000"
    sign_payload = timestamp + api_key + recv_window + ""
    sign = hmac.new(api_secret.encode(), sign_payload.encode(), hashlib.sha256).hexdigest()
    headers = {
        "X-BAPI-API-KEY": api_key,
        "X-BAPI-SIGN": sign,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": recv_window
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("retCode") != 0 or not data.get("result") or not data["result"].get("list"):
            print(f"[Bybit Auth] Нет данных: {data}")
            return None, None
        ticker = data["result"]["list"][0]
        price = float(ticker["lastPrice"])
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return price, date_str
    except Exception as e:
        print(f"[Bybit Auth] Ошибка: {e}")
        return None, None

if __name__ == "__main__":
    price, date_str, source = get_latest_gold_price()
    print(f"Gold price: {price} ({date_str}) [source: {source}]")
