# core/regime.py
from __future__ import annotations
import pandas as pd

def detect_regime(df: pd.DataFrame) -> dict:
    """
    레짐을 단순하지만 실전적으로:
    - Risk-On: Close > MA200 and MA20 > MA60
    - Neutral: 그 외
    - Risk-Off: Close < MA200 and MA20 < MA60
    """
    last = df.iloc[-1]
    close = float(last["Close"])
    ma200 = float(last.get("MA200", float("nan")))
    ma20 = float(last.get("MA20", float("nan")))
    ma60 = float(last.get("MA60", float("nan")))

    regime = "중립"
    gate_block_buy = False

    if close > ma200 and ma20 > ma60:
        regime = "리스크온(상승 우위)"
    elif close < ma200 and ma20 < ma60:
        regime = "리스크오프(방어 필요)"
        gate_block_buy = True

    return {
        "레짐": regime,
        "BUY_차단": gate_block_buy
    }
