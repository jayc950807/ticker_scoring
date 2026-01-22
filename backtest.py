# backtest.py
from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

from config import START_EQUITY
from scoring import score_row


def backtest_single(
    df: pd.DataFrame,
    mode: str,
    entry_score: float,
    max_hold: int,
    atr_stop: float,
    atr_target: float,
    fee_bps: float = 20.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Rules:
      - Signal evaluated at close (t)
      - Entry at next open (t+1)
      - One position at a time (long-only)
      - Exit: hit stop/target intraday (using High/Low), else time-based exit at close on max_hold
      - Fees: applied on entry & exit (bps of price)
    """
    df = df.copy().dropna(subset=["Open", "High", "Low", "Close", "ATR"])
    if len(df) < 300:
        empty_curve = pd.DataFrame({"equity": [START_EQUITY]}, index=[df.index[-1] if len(df) else pd.Timestamp.utcnow()])
        return empty_curve, pd.DataFrame(), {"trades": 0, "cagr": None, "mdd": None, "winrate": None,
                                             "expectancy": None, "profit_factor": None, "final_equity": float(START_EQUITY)}

    fee = fee_bps / 10000.0

    in_pos = False
    entry_i = None
    entry_px = None
    stop_px = None
    target_px = None

    equity = START_EQUITY
    equity_curve = []
    peak = START_EQUITY
    drawdowns = []

    trades = []

    # precompute scores for all bars
    scores = []
    contribs = []
    for _, row in df.iterrows():
        s, c = score_row(row, mode)
        scores.append(s)
        contribs.append(c)
    df["SCORE"] = scores

    for i in range(1, len(df) - 1):  # i uses signal at i; entry at i+1 open
        ts = df.index[i]
        row = df.iloc[i]
        nxt = df.iloc[i + 1]

        # mark-to-market equity curve (close-based)
        equity_curve.append((ts, equity))
        peak = max(peak, equity)
        dd = (equity / peak) - 1.0
        drawdowns.append(dd)

        if not in_pos:
            # entry condition
            if float(row["SCORE"]) >= float(entry_score):
                in_pos = True
                entry_i = i + 1
                entry_px = float(nxt["Open"]) * (1 + fee)  # pay fee on buy
                atr = float(row["ATR"])
                stop_px = entry_px - atr_stop * atr
                target_px = entry_px + atr_target * atr
        else:
            # manage position using today's bar (i) since we are holding from entry_i
            hold_days = i - int(entry_i)

            low = float(row["Low"])
            high = float(row["High"])
            close = float(row["Close"])

            exit_reason = None
            exit_px = None

            # stop/target intraday
            if low <= float(stop_px):
                exit_reason = "STOP"
                exit_px = float(stop_px) * (1 - fee)
            elif high >= float(target_px):
                exit_reason = "TARGET"
                exit_px = float(target_px) * (1 - fee)
            elif hold_days >= int(max_hold):
                exit_reason = "TIME"
                exit_px = close * (1 - fee)

            if exit_reason is not None:
                # realize trade
                ret = (exit_px / entry_px) - 1.0
                equity *= (1.0 + ret)

                trades.append({
                    "mode": mode,
                    "entry_date": df.index[entry_i],
                    "exit_date": ts,
                    "entry_px": entry_px,
                    "exit_px": exit_px,
                    "ret": ret,
                    "hold_days": hold_days,
                    "reason": exit_reason,
                    "entry_score": float(df.iloc[entry_i - 1]["SCORE"]),  # signal day score
                })

                in_pos = False
                entry_i = entry_px = stop_px = target_px = None

    # finalize curve
    if not equity_curve:
        equity_curve = [(df.index[-1], equity)]
    curve = pd.DataFrame({"equity": [v for _, v in equity_curve]}, index=[t for t, _ in equity_curve])

    trades_df = pd.DataFrame(trades)
    summary = _summarize(curve, trades_df)

    return curve, trades_df, summary


def _summarize(curve: pd.DataFrame, trades: pd.DataFrame) -> Dict[str, Any]:
    final_equity = float(curve["equity"].iloc[-1])
    n = int(len(trades)) if trades is not None else 0

    # MDD
    eq = curve["equity"].astype(float)
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    mdd = float(dd.min()) if len(dd) else None

    # CAGR (approx)
    if len(curve) >= 2:
        days = (curve.index[-1] - curve.index[0]).days
        if days > 0:
            cagr = float(final_equity ** (365.0 / days) - 1.0)
        else:
            cagr = None
    else:
        cagr = None

    if n == 0:
        return {
            "trades": 0,
            "cagr": cagr,
            "mdd": mdd,
            "winrate": None,
            "expectancy": None,
            "profit_factor": None,
            "final_equity": final_equity,
        }

    rets = trades["ret"].astype(float)
    winrate = float((rets > 0).mean())
    expectancy = float(rets.mean())

    gains = rets[rets > 0].sum()
    losses = -rets[rets < 0].sum()
    profit_factor = float(gains / losses) if losses > 0 else float("inf")

    return {
        "trades": n,
        "cagr": cagr,
        "mdd": mdd,
        "winrate": winrate,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "final_equity": final_equity,
    }
