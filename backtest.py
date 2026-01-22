# backtest.py
import numpy as np
import pandas as pd
from typing import Tuple, Dict

from features import build_features
from scoring import compute_mode_score
from strategies import breakout_20d, squeeze_release, quality_entry, exit_quality_trend_break

def backtest_single(
    df: pd.DataFrame,
    mode: str,
    entry_score: float,
    max_hold: int,
    atr_stop: float,
    atr_target: float,
    fee_bps: float = 20.0
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    - 룩어헤드 방지: 오늘 신호 -> 내일 시가 진입
    - 단일 포지션(롱)
    - 청산: stop/target/time + (QUALITY는 MA120 이탈 추가)
    """
    d = build_features(df.copy())
    d["score"] = compute_mode_score(d, mode)

    if mode == "MOMO":
        d["entry_signal"] = (d["score"] >= entry_score) & breakout_20d(d)
        extra_exit = None
    elif mode == "SQUEEZE":
        d["entry_signal"] = (d["score"] >= entry_score) & squeeze_release(d)
        extra_exit = None
    elif mode == "QUALITY":
        d["entry_signal"] = (d["score"] >= entry_score) & quality_entry(d)
        extra_exit = exit_quality_trend_break(d)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    fee = float(fee_bps) / 10000.0

    in_pos = False
    entry_px = 0.0
    entry_i = None
    stop = None
    target = None

    equity = 1.0
    eq_curve = []
    trades = []

    for i in range(len(d) - 1):  # next open 필요
        row = d.iloc[i]
        nxt = d.iloc[i + 1]

        if not in_pos:
            if bool(row["entry_signal"]) and np.isfinite(row["ATR"]) and float(row["ATR"]) > 0:
                entry_px = float(nxt["Open"])
                entry_i = i + 1
                stop = entry_px - atr_stop * float(row["ATR"])
                target = entry_px + atr_target * float(row["ATR"])
                in_pos = True
        else:
            low = float(row["Low"])
            high = float(row["High"])
            exit_px = None
            reason = None

            if low <= stop:
                exit_px = float(stop)
                reason = "STOP"
            elif high >= target:
                exit_px = float(target)
                reason = "TARGET"
            elif (i - entry_i) >= max_hold:
                exit_px = float(row["Close"])
                reason = "TIME"
            elif extra_exit is not None and bool(extra_exit.iloc[i]):
                exit_px = float(row["Close"])
                reason = "TREND_BREAK"

            if exit_px is not None:
                ret = (exit_px / entry_px - 1.0) - fee * 2.0
                equity *= (1.0 + ret)

                trades.append({
                    "mode": mode,
                    "entry_date": d.index[entry_i],
                    "exit_date": d.index[i],
                    "entry_px": entry_px,
                    "exit_px": exit_px,
                    "ret": ret,
                    "reason": reason,
                    "hold_days": int(i - entry_i),
                })

                in_pos = False
                entry_px = 0.0
                entry_i = None
                stop = None
                target = None

        eq_curve.append(equity)

    curve = d.iloc[:len(eq_curve)].copy()
    curve["equity"] = eq_curve

    trades_df = pd.DataFrame(trades)
    summary = summarize(trades_df, curve)
    return curve, trades_df, summary

def summarize(trades: pd.DataFrame, curve: pd.DataFrame) -> Dict:
    if trades is None or trades.empty or curve is None or curve.empty:
        return {"trades": 0, "cagr": None, "mdd": None, "winrate": None}

    eq = curve["equity"].astype(float)
    peak = eq.cummax()
    dd = (eq / peak - 1.0)
    mdd = float(dd.min())

    winrate = float((trades["ret"] > 0).mean())
    avg_win = float(trades.loc[trades["ret"] > 0, "ret"].mean()) if (trades["ret"] > 0).any() else 0.0
    avg_loss = float(trades.loc[trades["ret"] <= 0, "ret"].mean()) if (trades["ret"] <= 0).any() else 0.0
    expectancy = float(trades["ret"].mean())

    days = len(curve)
    cagr = float(eq.iloc[-1] ** (252 / max(1, days)) - 1.0)

    profit_factor = np.inf
    loss_sum = float(trades.loc[trades["ret"] <= 0, "ret"].sum())
    if loss_sum < 0:
        profit_factor = float(trades.loc[trades["ret"] > 0, "ret"].sum() / abs(loss_sum))

    return {
        "trades": int(len(trades)),
        "cagr": cagr,
        "mdd": mdd,
        "winrate": winrate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "final_equity": float(eq.iloc[-1]),
    }
