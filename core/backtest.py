# core/backtest.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict

def run_backtest(
    df: pd.DataFrame,
    entry_signal: pd.Series,
    stop_atr: float,
    take_atr: float,
    max_hold: int,
    cost_bps: float
) -> Dict:
    """
    단일 포지션(롱-only), 다음날 시가 진입 가정(보수적).
    - 진입: entry_signal[t] == True 이면 t+1 Open에 진입
    - 청산: (1) stop, (2) take, (3) max_hold, (4) 추세 붕괴(종가<MA200) 중 먼저
    - 비용: round-trip으로 cost_bps*2 적용(진입+청산)
    """
    df = df.copy()
    n = len(df)
    if n < 260:
        return {"trades": pd.DataFrame(), "equity": pd.Series(dtype=float), "stats": {"trades": 0}}

    cost = (cost_bps / 10000.0) * 2.0

    in_pos = False
    entry_idx = None
    entry_px = None
    stop_px = None
    take_px = None
    hold = 0

    trades = []

    equity = []
    eq = 1.0

    for i in range(n - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]

        close = float(row["Close"])
        ma200 = float(row.get("MA200", np.nan))
        atr = float(row.get("ATR14", np.nan))

        # 포지션 없을 때: 진입
        if not in_pos:
            equity.append(eq)

            if bool(entry_signal.iloc[i]) and np.isfinite(atr):
                # t+1 Open 진입
                entry_idx = i + 1
                entry_px = float(next_row["Open"])
                stop_px = entry_px - atr * stop_atr
                take_px = entry_px + atr * take_atr
                hold = 0
                in_pos = True
            continue

        # 포지션 있을 때: 당일 high/low로 stop/take 체크
        hold += 1
        day_high = float(next_row["High"])
        day_low = float(next_row["Low"])
        exit_reason = None
        exit_px = None

        # 1) stop 우선(보수적)
        if day_low <= stop_px:
            exit_px = stop_px
            exit_reason = "손절(ATR)"
        # 2) take
        elif day_high >= take_px:
            exit_px = take_px
            exit_reason = "익절(ATR)"
        # 3) 추세 붕괴(전일 기준)
        elif np.isfinite(ma200) and close < ma200:
            exit_px = float(next_row["Open"])
            exit_reason = "추세붕괴(MA200)"
        # 4) 최대 보유
        elif hold >= max_hold:
            exit_px = float(next_row["Close"])
            exit_reason = "시간청산"

        if exit_reason is None:
            # 마크투마켓(간이): 전일 close 대비 다음 close로 equity 업데이트
            cur_close = float(next_row["Close"])
            ret = (cur_close / float(df.iloc[entry_idx]["Open"]))  # 상대값(단순)
            # equity는 실제 트레이드 종료 시에만 갱신(보수적으로 유지)
            equity.append(eq)
            continue

        # 트레이드 확정 수익률
        gross = (exit_px / entry_px) - 1.0
        net = gross - cost
        eq *= (1.0 + net)
        equity.append(eq)

        trades.append({
            "진입일": df.index[entry_idx],
            "진입가": entry_px,
            "청산일": df.index[i + 1],
            "청산가": exit_px,
            "보유일": hold,
            "수익률(%)": net * 100,
            "사유": exit_reason,
            "STOP": stop_px,
            "TAKE": take_px
        })

        in_pos = False
        entry_idx = entry_px = stop_px = take_px = None
        hold = 0

    equity_s = pd.Series(equity, index=df.index[:len(equity)], name="equity")
    trades_df = pd.DataFrame(trades)

    stats = compute_stats(trades_df, equity_s)
    return {"trades": trades_df, "equity": equity_s, "stats": stats}

def compute_stats(trades: pd.DataFrame, equity: pd.Series) -> dict:
    if trades is None or trades.empty:
        mdd = max_drawdown(equity)
        return {"trades": 0, "win_rate": 0.0, "pf": 0.0, "mdd": mdd, "total_return": (equity.iloc[-1]-1)*100 if len(equity) else 0.0}

    rets = trades["수익률(%)"] / 100.0
    wins = rets[rets > 0]
    losses = rets[rets < 0]

    win_rate = float((rets > 0).mean() * 100)
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    pf = float(gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    mdd = max_drawdown(equity)
    total_return = float((equity.iloc[-1] - 1.0) * 100) if len(equity) else 0.0

    return {
        "trades": int(len(trades)),
        "win_rate": win_rate,
        "pf": pf,
        "mdd": mdd,
        "total_return": total_return
    }

def max_drawdown(equity: pd.Series) -> float:
    if equity is None or equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min() * 100)  # 음수(%)
