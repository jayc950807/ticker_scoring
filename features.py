# features.py
import numpy as np
import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    백테스트/스코어링에 필요한 핵심 feature.
    NOTE: df에는 MA20/MA120/ATR/BB/KC 등이 이미 존재한다고 가정(data_engine에서 생성).
    """
    out = df.copy()

    # returns
    out["ret_5"] = out["Close"].pct_change(5)
    out["ret_20"] = out["Close"].pct_change(20)

    # volume spike (today vs 20d mean)
    out["vol_spike"] = out["Volume"] / out["Volume"].rolling(20).mean()

    # volatility proxy
    out["range_pct_60"] = ((out["High"] - out["Low"]) / out["Close"]).rolling(60).mean()

    # trend proximity
    out["close_vs_MA20"] = out["Close"] / out["MA20"] - 1.0
    out["close_vs_MA120"] = out["Close"] / out["MA120"] - 1.0

    # drawdown
    roll_max_120 = out["Close"].rolling(120).max()
    out["drawdown_120"] = out["Close"] / roll_max_120 - 1.0  # <=0

    # squeeze proxies
    out["bb_width"] = (out["BB_Upper"] - out["BB_Lower"])
    out["kc_width"] = (out["KC_Upper"] - out["KC_Lower"])
    out["bb_over_kc"] = out["bb_width"] / out["kc_width"].replace(0, np.nan)

    # ATR normalization (breakout strength에 쓰기)
    out["range_over_atr"] = (out["High"] - out["Low"]) / out["ATR"].replace(0, np.nan)

    return out
