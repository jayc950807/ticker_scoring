# intel/insider.py
from __future__ import annotations
from typing import Optional
import pandas as pd
import yfinance as yf


def get_insider_transactions(ticker: str, limit: int = 30) -> Optional[pd.DataFrame]:
    try:
        t = yf.Ticker(ticker)
        df = getattr(t, "insider_transactions", None)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return None

        if "Start Date" in df.columns:
            df = df.sort_values("Start Date", ascending=False)

        return df.head(limit).reset_index(drop=True)
    except Exception:
        return None
