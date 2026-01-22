# data_engine.py
from __future__ import annotations
from typing import Tuple, Optional, Dict, Any
import pandas as pd
import yfinance as yf

from config import WINDOW_DAILY, INTRADAY_PERIOD, INTRADAY_INTERVAL, INDICATOR_WARMUP
from utils import safe_float


def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def get_ticker_info(ticker: str) -> Dict[str, Any]:
    try:
        t = yf.Ticker(ticker)
        return t.info or {}
    except Exception:
        return {}


def get_price_history(
    ticker: str,
    include_extended: bool = True,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Timestamp]]:
    """
    - 2y daily auto_adjust=True
    - optional 5d 1m (prepost) to patch last daily candle with real OHLCV (intraday aggregated)
    """
    try:
        df_daily = yf.download(
            ticker,
            period=WINDOW_DAILY,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        df_daily = _flatten_cols(df_daily)

        if df_daily is None or df_daily.empty:
            return None, None

        data_time_utc = df_daily.index[-1]

        # Patch last candle with intraday aggregation (better "latest")
        df_intra = None
        if include_extended:
            try:
                df_intra = yf.download(
                    ticker,
                    period=INTRADAY_PERIOD,
                    interval=INTRADAY_INTERVAL,
                    auto_adjust=True,
                    prepost=True,
                    progress=False,
                    threads=True,
                )
                df_intra = _flatten_cols(df_intra)
            except Exception:
                df_intra = None

        if df_intra is not None and not df_intra.empty:
            real_open = float(df_intra["Open"].iloc[0])
            real_high = float(df_intra["High"].max())
            real_low = float(df_intra["Low"].min())
            real_close = float(df_intra["Close"].iloc[-1])
            real_vol = float(df_intra["Volume"].sum())

            last_idx = df_daily.index[-1]
            df_daily.loc[last_idx, "Open"] = real_open
            df_daily.loc[last_idx, "High"] = max(float(df_daily.loc[last_idx, "High"]), real_high)
            df_daily.loc[last_idx, "Low"] = min(float(df_daily.loc[last_idx, "Low"]), real_low)
            df_daily.loc[last_idx, "Close"] = real_close
            df_daily.loc[last_idx, "Volume"] = real_vol
            data_time_utc = df_intra.index[-1]

        # Warmup: not enough rows = not analyzable
        if len(df_daily) < INDICATOR_WARMUP + 60:
            return None, data_time_utc

        df_daily = df_daily.copy()
        df_daily = df_daily.dropna(subset=["Close"])

        return df_daily, data_time_utc

    except Exception:
        return None, None
