# intel/sec_edgar.py
from __future__ import annotations
from typing import List, Dict, Optional
import requests
import re

from config import SEC_USER_AGENT, SEC_TIMEOUT


def cik_from_ticker(ticker: str) -> Optional[str]:
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=SEC_TIMEOUT)
    r.raise_for_status()
    data = r.json()

    t = ticker.upper().strip()
    for _, v in data.items():
        if (v.get("ticker") or "").upper() == t:
            cik = str(v.get("cik_str"))
            return cik.zfill(10)
    return None


def get_recent_filings(ticker: str, n: int = 25) -> List[Dict]:
    cik = cik_from_ticker(ticker)
    if not cik:
        return []

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = requests.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=SEC_TIMEOUT)
    r.raise_for_status()
    j = r.json()

    recent = j.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primdocs = recent.get("primaryDocument", [])

    out = []
    for form, date, acc, doc in list(zip(forms, dates, accessions, primdocs))[:n]:
        acc_nodash = re.sub(r"-", "", acc)
        link = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/{doc}"
        out.append({"date": date, "form": form, "link": link, "accession": acc})
    return out
