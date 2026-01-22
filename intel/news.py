# intel/news.py
from __future__ import annotations
from typing import List, Dict
import requests
import xml.etree.ElementTree as ET
from config import NEWS_TIMEOUT


def get_google_news_rss(query: str, n: int = 8) -> List[Dict]:
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    r = requests.get(url, timeout=NEWS_TIMEOUT)
    r.raise_for_status()

    root = ET.fromstring(r.content)
    items = []
    for item in root.findall(".//item")[:n]:
        title = item.findtext("title") or ""
        link = item.findtext("link") or ""
        if title:
            items.append({"title": title, "url": link})
    return items


def get_news_headlines(ticker: str, n: int = 8) -> List[Dict]:
    try:
        return get_google_news_rss(f"{ticker} stock", n=n)
    except Exception:
        return []
