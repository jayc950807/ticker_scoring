# ui_api.py
from __future__ import annotations
from typing import Dict, Any, Optional
import pandas as pd
import streamlit as st

from data_engine import get_price_history, get_ticker_info
from features import add_indicators
from scoring import build_quant_signals, classify_mode, score_row
from strategies import default_params
from backtest import backtest_single
from utils import Snapshot, format_kst_time, safe_float, safe_int

from intel.news import get_news_headlines
from intel.insider import get_insider_transactions
from intel.sec_edgar import get_recent_filings


@st.cache_data(show_spinner=False, ttl=60 * 10)
def cached_price(ticker: str, include_extended: bool):
    return get_price_history(ticker, include_extended=include_extended)


@st.cache_data(show_spinner=False, ttl=60 * 30)
def cached_info(ticker: str):
    return get_ticker_info(ticker)


@st.cache_data(show_spinner=False, ttl=60 * 10)
def cached_news(ticker: str):
    return get_news_headlines(ticker, n=8)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def cached_filings(ticker: str):
    return get_recent_filings(ticker, n=25)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def cached_insider(ticker: str):
    df = get_insider_transactions(ticker, limit=30)
    # DataFrame is cacheable, but keep it lightweight
    return df


def run_app_payload(
    ticker: str,
    include_extended: bool,
    fee_bps: float,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Returns a single payload dict that UI can render without knowing internals.
    """
    overrides = overrides or {}

    raw_df, data_time = cached_price(ticker, include_extended)
    if raw_df is None or raw_df.empty:
        return {"ok": False, "error": "가격 데이터를 불러올 수 없습니다."}

    info = cached_info(ticker)

    df = add_indicators(raw_df)
    df = build_quant_signals(df)

    # Drop early NaNs
    df = df.dropna(subset=["Close", "ATR", "MA20", "MA120", "RSI"]).copy()
    if len(df) < 350:
        return {"ok": False, "error": "지표 계산에 필요한 데이터가 부족합니다(기간/상장 이력 확인)."}

    auto_mode = classify_mode(df)

    # Snapshot (last bar)
    last = df.iloc[-1]
    snap = Snapshot(
        ticker=ticker,
        name=(info.get("longName") or info.get("shortName") or ticker),
        close=float(last["Close"]),
        atr=float(last["ATR"]),
        ma20=float(last["MA20"]),
        ma120=float(last["MA120"]),
        rsi=float(last["RSI"]),
        vol_spike=float(last.get("VOL_SPIKE", 1.0) or 1.0),
        squeeze_on=bool(last.get("SQZ_ON", False)),
        squeeze_streak=int(last.get("SQZ_STREAK", 0) or 0),
        updated_at=format_kst_time(data_time),
    )

    # UI only: short metrics are informational (unreliable; do not trade on it)
    short_pct = safe_float(info.get("shortPercentOfFloat"), None)
    short_ratio = safe_float(info.get("shortRatio"), None)
    float_shares = safe_int(info.get("floatShares"), None)
    short_pack = {
        "short_pct": (None if short_pct is None else float(short_pct) * 100.0),
        "days_to_cover": short_ratio,
        "float_m": (None if float_shares is None else float(float_shares) / 1e6),
        "quality": "OK" if (short_pct is not None and short_ratio is not None) else "MISSING/STALE",
    }

    # Backtests for all modes
    modes = ["MOMO", "SQUEEZE", "QUALITY"]
    bt = {}
    for mode in modes:
        p = default_params(mode)
        # allow override per-mode
        if overrides.get("mode") == mode:
            p = dict(p)
            for k in ("entry_score", "max_hold", "atr_stop", "atr_target"):
                if k in overrides:
                    p[k] = overrides[k]

        curve, trades, summary = backtest_single(
            df=df,
            mode=mode,
            entry_score=float(p["entry_score"]),
            max_hold=int(p["max_hold"]),
            atr_stop=float(p["atr_stop"]),
            atr_target=float(p["atr_target"]),
            fee_bps=float(fee_bps),
        )

        bt[mode] = {
            "params": p,
            "curve": curve,
            "trades": trades,
            "summary": summary,
        }

    # Explainability for current day score
    current_scores = {}
    for mode in modes:
        s, contrib = score_row(df.iloc[-1], mode)
        current_scores[mode] = {"score": s, "contrib": contrib}

    # Intel (news/filings/insider)
    news = cached_news(ticker)
    filings = cached_filings(ticker)
    insider_df = cached_insider(ticker)

    payload = {
        "ok": True,
        "ticker": ticker,
        "snapshot": snap.__dict__,
        "auto_mode": auto_mode,
        "short_pack": short_pack,
        "df_tail": df.tail(60),
        "backtests": bt,
        "current_scores": current_scores,
        "intel": {
            "news": news,
            "filings": filings,
            "insider": insider_df,
        }
    }
    return payload
