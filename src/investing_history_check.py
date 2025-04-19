import requests
from bs4 import BeautifulSoup
from datetime import datetime
import re

INVESTING_HISTORY_URL = "https://www.investing.com/commodities/gold-historical-data"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
}

def fetch_investing_history():
    resp = requests.get(INVESTING_HISTORY_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table")
    if not table:
        raise ValueError("Не найдена таблица с историей цен!")
    rows = table.find_all("tr")
    results = []
    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 6:
            continue
        date_str = cols[0].text.strip()
        # Приводим дату к ISO формату
        try:
            dt = datetime.strptime(date_str, "%b %d, %Y")
            iso_date = dt.strftime("%Y-%m-%d")
        except Exception:
            continue
        close = cols[1].text.strip().replace(",", "")
        results.append((iso_date, close))
    return results

def check_dates(dates_to_check):
    data = fetch_investing_history()
    for d in dates_to_check:
        found = False
        for row in data:
            if row[0] == d:
                print(f"{d}: Close = {row[1]}")
                found = True
                break
        if not found:
            print(f"{d}: Нет данных на Investing.com")

if __name__ == "__main__":
    check_dates(["2025-04-18", "2025-04-19"])
