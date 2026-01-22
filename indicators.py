import numpy as np
import pandas as pd

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # MA
    d["MA20"] = d["Close"].rolling(20).mean()
    d["MA50"] = d["Close"].rolling(50).mean()
    d["MA200"] = d["Close"].rolling(200).mean()

    # ATR(14)
    tr = np.maximum(
        d["High"] - d["Low"],
        np.maximum((d["High"] - d["Close"].shift(1)).abs(), (d["Low"] - d["Close"].shift(1)).abs()),
    )
    d["ATR"] = tr.rolling(14).mean()

    # RSI(14)
    delta = d["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-12)
    d["RSI"] = 100 - (100 / (1 + rs))

    # Volume ratio vs 20d avg
    vavg = d["Volume"].rolling(20).mean()
    d["VOL_RATIO"] = d["Volume"] / vavg.replace(0, 1e-12)

    # Returns
    d["LOG_RET"] = np.log(d["Close"] / d["Close"].shift(1))

    return d

def add_quantile_thresholds(df: pd.DataFrame, lookback: int = 504) -> pd.DataFrame:
    """
    고정 수치(예: VOL_RATIO>3) 같은 직감 기준을 줄이기 위해 '분위수 기반'을 사용.
    rolling(lookback)은 과거 정보만 쓰므로 미래누수(leak) 위험이 상대적으로 낮음.
    """
    d = df.copy()
    lb = max(252, int(lookback))

    volq = d["VOL_RATIO"].rolling(lb)
    d["VOL_Q80"] = volq.quantile(0.80)
    d["VOL_Q90"] = volq.quantile(0.90)

    rsiq = d["RSI"].rolling(lb)
    d["RSI_Q30"] = rsiq.quantile(0.30)
    d["RSI_Q40"] = rsiq.quantile(0.40)

    atr_pct = (d["ATR"] / d["Close"]).rolling(lb)
    d["ATR_PCT_Q50"] = atr_pct.quantile(0.50)
    d["ATR_PCT_Q80"] = atr_pct.quantile(0.80)

    return d
