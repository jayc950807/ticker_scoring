# core/indicators.py
from __future__ import annotations
import numpy as np
import pandas as pd

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # MA
    out["MA5"] = out["Close"].rolling(5).mean()
    out["MA20"] = out["Close"].rolling(20).mean()
    out["MA60"] = out["Close"].rolling(60).mean()
    out["MA120"] = out["Close"].rolling(120).mean()
    out["MA200"] = out["Close"].rolling(200).mean()

    # MACD
    out["EMA12"] = out["Close"].ewm(span=12, adjust=False).mean()
    out["EMA26"] = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = out["EMA12"] - out["EMA26"]
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()

    # RSI
    delta = out["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    out["RSI14"] = 100 - (100 / (1 + rs))

    # Stoch
    low14 = out["Low"].rolling(14).min()
    high14 = out["High"].rolling(14).max()
    out["StochK"] = ((out["Close"] - low14) / (high14 - low14).replace(0, 1)) * 100
    out["StochD"] = out["StochK"].rolling(3).mean()

    # Bollinger
    std20 = out["Close"].rolling(20).std()
    out["BB_Upper"] = out["MA20"] + 2 * std20
    out["BB_Lower"] = out["MA20"] - 2 * std20

    # ATR
    tr = np.maximum(out["High"] - out["Low"],
                    np.maximum((out["High"] - out["Close"].shift(1)).abs(),
                               (out["Low"] - out["Close"].shift(1)).abs()))
    out["TR"] = tr
    out["ATR14"] = out["TR"].rolling(14).mean()

    # Keltner(간이)
    out["KC_Upper"] = out["MA20"] + out["ATR14"] * 1.5
    out["KC_Lower"] = out["MA20"] - out["ATR14"] * 1.5

    # TTM Squeeze raw
    bb_width = (out["BB_Upper"] - out["BB_Lower"])
    kc_width = (out["KC_Upper"] - out["KC_Lower"])
    out["SQUEEZE_RAW_ON"] = (bb_width < kc_width).fillna(False)

    # OBV
    out["OBV"] = (np.sign(out["Close"].diff()) * out["Volume"]).fillna(0).cumsum()

    # A/D Line
    ad_factor = ((out["Close"] - out["Low"]) - (out["High"] - out["Close"])) / (out["High"] - out["Low"]).replace(0, 1)
    out["AD_Line"] = (ad_factor * out["Volume"]).fillna(0).cumsum()

    # MFI
    typical = (out["High"] + out["Low"] + out["Close"]) / 3
    mf = typical * out["Volume"]
    pos = mf.where(typical > typical.shift(1), 0).rolling(14).sum()
    neg = mf.where(typical < typical.shift(1), 0).rolling(14).sum().replace(0, 1)
    out["MFI14"] = 100 - (100 / (1 + (pos / neg)))

    # 거래대금
    out["DollarVol"] = out["Close"] * out["Volume"]
    out["DollarVol20"] = out["DollarVol"].rolling(20).mean()

    # 수익률/변동성
    out["LogRet"] = np.log(out["Close"] / out["Close"].shift(1))
    out["RangePct"] = (out["High"] - out["Low"]) / out["Close"] * 100

    return out
