# core/data_engine.py
from __future__ import annotations
import pandas as pd
import yfinance as yf
from typing import Optional, Tuple
from .config import AppConfig

def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

def load_price_data(ticker: str, include_extended: bool, cfg: AppConfig) -> Tuple[Optional[pd.DataFrame], Optional[pd.Timestamp]]:
    """
    - 3y 일봉 + 5d 1m 분봉(pre/post 포함 가능)으로 마지막 일봉 OHLCV 보정
    - auto_adjust=True
    """
    try:
        df_d = yf.download(ticker, period=cfg.DAILY_PERIOD, interval="1d",
                           auto_adjust=cfg.AUTO_ADJUST, progress=False)
        df_d = _flatten_cols(df_d)
        if df_d is None or df_d.empty:
            return None, None

        df_i = yf.download(ticker, period=cfg.INTRADAY_PERIOD, interval=cfg.INTRADAY_INTERVAL,
                           auto_adjust=cfg.AUTO_ADJUST, prepost=include_extended, progress=False)
        df_i = _flatten_cols(df_i)

        data_time = df_d.index[-1]

        # intraday가 있으면 마지막 일봉 보정
        if df_i is not None and not df_i.empty:
            last_day = df_d.index[-1]
            real_open = float(df_i["Open"].iloc[0])
            real_high = float(df_i["High"].max())
            real_low = float(df_i["Low"].min())
            real_close = float(df_i["Close"].iloc[-1])
            real_vol = float(df_i["Volume"].sum())

            df_d.loc[last_day, "Open"] = real_open
            df_d.loc[last_day, "High"] = max(float(df_d.loc[last_day, "High"]), real_high)
            df_d.loc[last_day, "Low"] = min(float(df_d.loc[last_day, "Low"]), real_low)
            df_d.loc[last_day, "Close"] = real_close
            df_d.loc[last_day, "Volume"] = real_vol
            data_time = df_i.index[-1]

        df_d = df_d.dropna(subset=["Close"]).copy()
        if len(df_d) < 400:
            return None, None

        return df_d, data_time
    except Exception:
        return None, None

def load_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}
