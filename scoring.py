# scoring.py
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from config import Q_WINDOW, Q_LOW, Q_HIGH
from utils import clamp


FEATURES = [
    "RET_5", "RET_20",
    "VOL_SPIKE", "RANGE_PCT",
    "RSI", "MACD_H",
    "OBV_SLP", "AD_SLP",
]

def _q_signal(series: pd.Series, window: int, low_q: float, high_q: float) -> pd.Series:
    """
    Quantile signal:
      +1 if value >= rolling_q(high)
      -1 if value <= rolling_q(low)
       0 otherwise
    """
    ql = series.rolling(window).quantile(low_q)
    qh = series.rolling(window).quantile(high_q)
    out = pd.Series(0, index=series.index, dtype=float)
    out = out.where(~(series <= ql), -1)
    out = out.where(~(series >= qh), +1)
    return out


def build_quant_signals(df: pd.DataFrame, window: int = Q_WINDOW) -> pd.DataFrame:
    """
    Precompute quantile signals for backtest use.
    """
    out = df.copy()
    for f in FEATURES:
        if f in out.columns:
            out[f"Q_{f}"] = _q_signal(out[f], window, Q_LOW, Q_HIGH)

    # Trend conditions (not quantiles — deterministic)
    out["TREND_MA20"] = np.where(out["Close"] >= out["MA20"], 1, -1)
    out["TREND_MA120"] = np.where(out["Close"] >= out["MA120"], 1, -1)

    # Squeeze (binary)
    out["SIG_SQZ"] = np.where(out["SQZ_ON"], 1, 0)

    # RSI shape constraint (anti-overfit guard)
    # - for MOMO: RSI too high is risk
    out["RSI_OVERHEAT"] = np.where(out["RSI"] >= 75, 1, 0)
    out["RSI_OVERSOLD"] = np.where(out["RSI"] <= 25, 1, 0)

    return out


# -------------------------
# Mode scorecards
# -------------------------
SCORECARDS: Dict[str, Dict[str, float]] = {
    # MOMO: prioritize returns/volume/trend, penalize overheated RSI
    "MOMO": {
        "Q_RET_5": 1.8,
        "Q_RET_20": 1.2,
        "Q_VOL_SPIKE": 1.6,
        "Q_MACD_H": 1.0,
        "TREND_MA20": 1.2,
        "TREND_MA120": 0.6,
        "Q_RANGE_PCT": 0.6,     # momo likes movement
        "Q_OBV_SLP": 0.8,
        "Q_AD_SLP": 0.8,
        "RSI_OVERHEAT": -1.2,   # risk control
    },
    # SQUEEZE: prioritize squeeze state + volume/flow; trend less
    "SQUEEZE": {
        "SIG_SQZ": 2.5,
        "Q_VOL_SPIKE": 1.6,
        "Q_RANGE_PCT": -0.6,    # squeeze wants compression (range low)
        "Q_OBV_SLP": 1.0,
        "Q_AD_SLP": 1.0,
        "Q_MACD_H": 0.8,
        "TREND_MA20": 0.8,
        "RSI_OVERHEAT": -0.8,
    },
    # QUALITY: trend stability + lower range + healthy MACD/flow
    "QUALITY": {
        "TREND_MA120": 2.0,
        "TREND_MA20": 1.0,
        "Q_RANGE_PCT": -1.2,    # lower is better
        "Q_MACD_H": 1.0,
        "Q_OBV_SLP": 0.8,
        "Q_AD_SLP": 0.8,
        "Q_VOL_SPIKE": 0.3,     # not decisive
        "RSI_OVERHEAT": -0.6,
        "RSI_OVERSOLD": 0.4,
    },
}


def score_row(row: pd.Series, mode: str) -> Tuple[float, Dict[str, float]]:
    """
    Returns: score in [0,1], and contributions for explainability.
    We intentionally keep this conservative (avoid 0.99 nonsense).
    """
    card = SCORECARDS.get(mode, SCORECARDS["QUALITY"])
    contrib: Dict[str, float] = {}
    raw = 0.0
    wsum = 0.0

    for k, w in card.items():
        v = row.get(k, np.nan)
        if pd.isna(v):
            continue
        # signals are mostly -1/0/+1; squeeze is 0/1; overheats are 0/1
        raw += w * float(v)
        wsum += abs(w)
        contrib[k] = w * float(v)

    if wsum <= 0:
        return 0.0, contrib

    # map raw -> 0..1 using tanh; clamp to avoid overconfidence
    s = 0.5 + 0.5 * np.tanh(raw / max(1.0, (0.9 * wsum)))
    s = float(clamp(s, 0.05, 0.95))
    return s, contrib


def classify_mode(df: pd.DataFrame) -> str:
    """
    A simple mode classifier for UI (not used to trade).
    Conservative by design.
    """
    last = df.iloc[-1]
    price = float(last["Close"])
    vol_spike = float(last.get("VOL_SPIKE", 1.0) or 1.0)
    ret_5 = float(last.get("RET_5", 0.0) or 0.0)
    sqz = bool(last.get("SQZ_ON", False))

    if sqz and vol_spike >= 2.0:
        return "SQUEEZE"
    if ret_5 >= 0.15 and vol_spike >= 2.0 and price < 50:
        return "MOMO"
    return "QUALITY"
