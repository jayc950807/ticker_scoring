# core/sec.py
from __future__ import annotations
import requests
import pandas as pd
from typing import Optional, Dict

SEC_UA = "AIStockSniper/1.0 (contact: example@example.com)"  # 필요 시 네 이메일로 바꿔도 됨

def _headers() -> Dict[str, str]:
    return {"User-Agent": SEC_UA, "Accept-Encoding": "gzip, deflate", "Host": "data.sec.gov"}

def get_cik_by_ticker(ticker: str) -> Optional[str]:
    """
    SEC의 company_tickers.json을 받아 ticker->CIK 매핑.
    """
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        r = requests.get(url, headers=_headers(), timeout=10)
        if r.status_code != 200:
            return None
        j = r.json()
        t = ticker.upper()
        for _, row in j.items():
            if str(row.get("ticker", "")).upper() == t:
                cik_int = int(row.get("cik_str"))
                return str(cik_int).zfill(10)
        return None
    except Exception:
        return None

def get_recent_filings(ticker: str, limit: int = 15) -> Optional[pd.DataFrame]:
    cik = get_cik_by_ticker(ticker)
    if not cik:
        return None
    try:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        r = requests.get(url, headers=_headers(), timeout=10)
        if r.status_code != 200:
            return None
        j = r.json()
        recent = j.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        acc = recent.get("accessionNumber", [])
        prim = recent.get("primaryDocument", [])

        rows = []
        for i in range(min(len(forms), len(dates), len(acc), len(prim), limit)):
            accession = str(acc[i]).replace("-", "")
            doc = str(prim[i])
            link = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
            rows.append({"공시일": dates[i], "서식": forms[i], "링크": link})
        df = pd.DataFrame(rows)
        return df if not df.empty else None
    except Exception:
        return None
