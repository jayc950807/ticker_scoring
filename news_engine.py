import requests
import xml.etree.ElementTree as ET

def get_google_news_rss(ticker: str, limit: int = 8) -> list[dict]:
    url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    try:
        resp = requests.get(url, timeout=6)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        out = []
        for item in root.findall(".//item")[:limit]:
            title = item.find("title")
            link = item.find("link")
            if title is not None:
                out.append({"title": title.text, "url": link.text if link is not None else "#"})
        return out
    except Exception:
        return []

def translate_ko(items: list[dict]) -> list[dict]:
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
                t = tr.translate(t)
            except Exception:
                pass
        out.append({"title": t, "url": it.get("url", "#")})
    return out
