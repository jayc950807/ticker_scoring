# core/strategy.py
from __future__ import annotations
import pandas as pd

def compute_entry_signal(df: pd.DataFrame) -> pd.Series:
    """
    스윙용 고정 진입 룰(파라미터 최소화, 감사 가능성 높게):
    - 추세 필터: Close > MA200 (장기 상승장만)
    - 눌림: Close가 MA20 근처(±1.0*ATR) 안에 들어옴
    - 모멘텀 확인: RSI14가 40~65 (과열/과매도 극단 피함)
    - 거래대금: DollarVol20 확보(유동성 필터는 scoring에서 강하게)
    """
    close = df["Close"]
    ma20 = df["MA20"]
    ma200 = df["MA200"]
    atr = df["ATR14"]
    rsi = df["RSI14"]

    pullback = (close - ma20).abs() <= (atr * 1.0)
    trend_ok = close > ma200
    rsi_ok = (rsi >= 40) & (rsi <= 65)

    entry = trend_ok & pullback & rsi_ok
    return entry.fillna(False)

def squeeze_state(df: pd.DataFrame, min_on_days: int = 5) -> dict:
    raw = df["SQUEEZE_RAW_ON"].fillna(False)
    streak = 0
    for v in raw.iloc[::-1]:
        if bool(v):
            streak += 1
        else:
            break
    on = streak >= int(min_on_days)
    prev_raw = bool(raw.iloc[-2]) if len(raw) >= 2 else False
    curr_raw = bool(raw.iloc[-1])
    return {
        "ON": on,
        "연속일": streak,
        "오늘ON전환": (not prev_raw) and curr_raw and on,
        "오늘OFF전환": prev_raw and (not curr_raw),
    }
