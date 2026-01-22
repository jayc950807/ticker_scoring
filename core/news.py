# core/news.py
from __future__ import annotations
import requests, xml.etree.ElementTree as ET
from typing import List, Dict

def google_news_rss(ticker: str, limit: int = 7) -> List[Dict]:
    url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.content)
        out = []
        for item in root.findall(".//item")[:limit]:
            title = item.find("title")
            link = item.find("link")
            if title is None:
                continue
            out.append({"title": title.text or "", "url": (link.text if link is not None else "")})
        return out
    except Exception:
        return []

def try_translate_titles(items: List[Dict]) -> List[Dict]:
    try:
        from deep_translator import GoogleTranslator
        tr = GoogleTranslator(source="auto", target="ko")
    except Exception:
        tr = None

    out = []
    for it in items:
        t = it.get("title", "")
        if tr:
            try:
                t_ko = tr.translate(t)
            except Exception:
                t_ko = t
        else:
            t_ko = t
        out.append({"title_ko": t_ko, "title": t, "url": it.get("url", "")})
    return out
