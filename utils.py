# utils.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pytz
import pandas as pd


def safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x, default=None):
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def format_kst_time(ts) -> str:
    if ts is None:
        return "N/A"
    try:
        if isinstance(ts, pd.Timestamp):
            if ts.tzinfo is None:
                ts = pytz.utc.localize(ts)
            return ts.astimezone(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M")
        return str(ts)
    except Exception:
        return "N/A"


@dataclass
class Snapshot:
    ticker: str
    name: str
    close: float
    atr: float
    ma20: float
    ma120: float
    rsi: float
    vol_spike: float
    squeeze_on: bool
    squeeze_streak: int
    updated_at: str


def clamp(x, lo, hi):
    return max(lo, min(hi, x))
