# features.py
from __future__ import annotations
import numpy as np
import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Moving averages
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df["MA120"] = df["Close"].rolling(120).mean()

    # ATR(14)
    tr = np.maximum(
        df["High"] - df["Low"],
        np.maximum((df["High"] - df["Close"].shift(1)).abs(),
                   (df["Low"] - df["Close"].shift(1)).abs())
    )
    df["ATR"] = tr.rolling(14).mean()

    # RSI(14)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().replace(0, 1e-9)
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD histogram
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    df["MACD_H"] = macd - sig

    # Bollinger(20,2)
    ma = df["MA20"]
    sd = df["Close"].rolling(20).std()
    df["BB_U"] = ma + 2 * sd
    df["BB_L"] = ma - 2 * sd
    df["BB_W"] = (df["BB_U"] - df["BB_L"]).replace(0, np.nan)

    # Keltner(20, 1.5*ATR)
    df["KC_U"] = df["MA20"] + 1.5 * df["ATR"]
    df["KC_L"] = df["MA20"] - 1.5 * df["ATR"]
    df["KC_W"] = (df["KC_U"] - df["KC_L"]).replace(0, np.nan)

    # Squeeze raw
    df["SQZ_RAW"] = (df["BB_W"] < df["KC_W"]).fillna(False)

    # Squeeze streak (consecutive raw ON)
    streak = np.zeros(len(df), dtype=int)
    on = df["SQZ_RAW"].to_numpy()
    for i in range(len(df)):
        streak[i] = (streak[i-1] + 1) if (i > 0 and on[i]) else (1 if on[i] else 0)
    df["SQZ_STREAK"] = streak
    df["SQZ_ON"] = df["SQZ_STREAK"] >= 5  # default (UI can override in strategy params if needed)

    # Returns
    df["RET_5"] = df["Close"].pct_change(5)
    df["RET_20"] = df["Close"].pct_change(20)

    # Range % (volatility proxy)
    df["RANGE_PCT"] = ((df["High"] - df["Low"]) / df["Close"]).replace([np.inf, -np.inf], np.nan)

    # Volume spike
    vol_ma20 = df["Volume"].rolling(20).mean().replace(0, np.nan)
    df["VOL_SPIKE"] = (df["Volume"] / vol_ma20).replace([np.inf, -np.inf], np.nan)

    # OBV and slope
    obv = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
    df["OBV"] = obv
    df["OBV_SLP"] = df["OBV"].diff(20)

    # A/D line and slope
    mfm = (((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) /
           (df["High"] - df["Low"]).replace(0, np.nan)).fillna(0)
    df["AD"] = (mfm * df["Volume"]).fillna(0).cumsum()
    df["AD_SLP"] = df["AD"].diff(20)

    return df
