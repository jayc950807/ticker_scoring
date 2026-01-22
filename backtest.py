import numpy as np
import pandas as pd

def backtest(
    df: pd.DataFrame,
    fee_bps: float = 5,
    slippage_bps: float = 10,
    stop_atr: float = 2.0,
    take_atr: float = 4.0,
    max_hold_days: int = 30,
) -> tuple[pd.DataFrame, list[dict], dict]:
    """
    ENTRY 시그널 다음날 시가 진입.
    청산 우선순위(보수적):
      1) 손절(일봉 저가가 stop 이하)
      2) 익절(일봉 고가가 take 이상)
      3) 추세훼손(EXIT_TREND_BREAK)
      4) 타임스탑
    비용: entry/exit에 bps 차감
    """
    d = df.copy().dropna(subset=["Open","High","Low","Close","ATR"])
    if d.empty or len(d) < 60:
        return d, [], _empty_stats()

    cost = (fee_bps + slippage_bps) / 10_000.0

    in_pos = False
    entry_px = stop_px = take_px = 0.0
    entry_dt = None

    equity = 1.0
    eq_curve = []
    trades: list[dict] = []

    for i in range(len(d)):
        row = d.iloc[i]
        dt = d.index[i]

        if not in_pos:
            eq_curve.append(equity)

            # 진입: 시그널 발생일 다음날 시가
            if bool(row.get("ENTRY", False)) and i + 1 < len(d):
                nxt = d.iloc[i + 1]
                atr = float(row["ATR"])
                entry_px = float(nxt["Open"]) * (1 + cost)
                stop_px = entry_px - stop_atr * atr
                take_px = entry_px + take_atr * atr
                entry_dt = d.index[i + 1]
                in_pos = True
            continue

        # 포지션 보유
        hold_days = (dt - entry_dt).days if entry_dt is not None else 0

        hi = float(row["High"])
        lo = float(row["Low"])
        cl = float(row["Close"])

        exit_reason = None
        exit_px = None

        if lo <= stop_px:
            exit_px = stop_px * (1 - cost)
            exit_reason = "손절(ATR)"
        elif hi >= take_px:
            exit_px = take_px * (1 - cost)
            exit_reason = "익절(ATR)"
        elif bool(row.get("EXIT_TREND_BREAK", False)):
            exit_px = cl * (1 - cost)
            exit_reason = "추세훼손(MA50)"
        elif hold_days >= max_hold_days:
            exit_px = cl * (1 - cost)
            exit_reason = "타임스탑"

        if exit_reason:
            ret = (exit_px / entry_px) - 1.0
            equity *= (1.0 + ret)
            trades.append({
                "진입일": entry_dt,
                "청산일": dt,
                "진입가": entry_px,
                "청산가": exit_px,
                "수익률": ret,
                "사유": exit_reason,
                "보유일": hold_days
            })
            in_pos = False
            entry_px = stop_px = take_px = 0.0
            entry_dt = None

        eq_curve.append(equity)

    d["EQUITY"] = pd.Series(eq_curve, index=d.index[:len(eq_curve)])
    stats = compute_stats(d["EQUITY"].dropna(), trades)

    return d, trades, stats

def compute_stats(equity: pd.Series, trades: list[dict]) -> dict:
    if equity is None or equity.empty:
        return _empty_stats()

    rets = equity.pct_change().dropna()
    n = max(1, len(rets))

    cagr = (equity.iloc[-1] ** (252 / n) - 1) if n else 0.0
    peak = equity.cummax()
    dd = (equity / peak - 1.0)
    mdd = float(dd.min()) * -1.0

    wins = [t for t in trades if t["수익률"] > 0]
    losses = [t for t in trades if t["수익률"] <= 0]

    gross_win = sum(t["수익률"] for t in wins)
    gross_loss = -sum(t["수익률"] for t in losses) if losses else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)

    winrate = (len(wins) / len(trades) * 100) if trades else 0.0
    avg_ret = (sum(t["수익률"] for t in trades) / len(trades)) if trades else 0.0

    sharpe = 0.0
    if rets.std(ddof=1) > 1e-12:
        sharpe = float(rets.mean() / rets.std(ddof=1) * np.sqrt(252))

    return {
        "트레이드수": len(trades),
        "승률": float(winrate),
        "평균수익률": float(avg_ret),
        "PF": float(pf),
        "CAGR(근사)": float(cagr),
        "MDD": float(mdd),
        "샤프(근사)": float(sharpe),
        "최종에퀴티": float(equity.iloc[-1]),
    }

def _empty_stats() -> dict:
    return {
        "트레이드수": 0,
        "승률": 0.0,
        "평균수익률": 0.0,
        "PF": 0.0,
        "CAGR(근사)": 0.0,
        "MDD": 1.0,
        "샤프(근사)": 0.0,
        "최종에퀴티": 1.0,
    }

def walk_forward_oos(
    df: pd.DataFrame,
    train_days: int = 252*3,
    test_days: int = 252//2,
    step_days: int = 63,
    **bt_kwargs
) -> tuple[pd.DataFrame, dict]:
    """
    워크포워드(OOS) 검증:
    - train 구간의 과거를 포함해 test를 진행(미래 누수 최소화)
    - 결과는 test 구간 stats를 누적해서 'OOS 집계' + '최악 구간'을 같이 본다
    """
    d = df.copy().dropna(subset=["Open","High","Low","Close","ATR"])
    if len(d) < (train_days + test_days + 50):
        return pd.DataFrame(), {"ok": False, "reason": "워크포워드에 필요한 데이터 길이 부족"}

    results = []
    start = 0
    while True:
        train_start = start
        train_end = train_start + train_days
        test_end = train_end + test_days
        if test_end > len(d):
            break

        # 테스트 구간은 train 이후만 성과를 평가
        window = d.iloc[:test_end].copy()
        _, _, stats = backtest(window, **bt_kwargs)

        # test-only 성과를 얻기 위해: test 구간만 에퀴티 변화를 다시 계산하는 게 정확하지만,
        # 여기서는 보수적으로 "전체 window" stats가 아닌, test 구간 트레이드만 추출해서 평가한다.
        # backtest를 전체 window에 돌리면 train 구간 트레이드가 섞이므로, test 기간 트레이드만 필터링.
        _, trades, _ = backtest(window, **bt_kwargs)
        test_start_dt = d.index[train_end]
        test_end_dt = d.index[test_end-1]

        test_trades = [t for t in trades if (t["진입일"] >= test_start_dt and t["진입일"] <= test_end_dt)]
        # test_trades만으로 stats 재계산(간단 버전)
        if len(test_trades) == 0:
            test_stats = {
                "트레이드수": 0, "승률": 0.0, "평균수익률": 0.0, "PF": 0.0,
                "MDD": 0.0, "샤프(근사)": 0.0, "CAGR(근사)": 0.0
            }
        else:
            win = [t for t in test_trades if t["수익률"] > 0]
            loss = [t for t in test_trades if t["수익률"] <= 0]
            gross_win = sum(t["수익률"] for t in win)
            gross_loss = -sum(t["수익률"] for t in loss) if loss else 0.0
            pf = (gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
            winrate = len(win) / len(test_trades) * 100
            avg_ret = sum(t["수익률"] for t in test_trades) / len(test_trades)
            test_stats = {
                "트레이드수": len(test_trades),
                "승률": float(winrate),
                "평균수익률": float(avg_ret),
                "PF": float(pf),
                "MDD": float("nan"),
                "샤프(근사)": float("nan"),
                "CAGR(근사)": float("nan"),
            }

        results.append({
            "구간(테스트)": f"{test_start_dt.date()} ~ {test_end_dt.date()}",
            "트레이드수": test_stats["트레이드수"],
            "승률(%)": round(test_stats["승률"], 1),
            "PF": round(test_stats["PF"], 2),
            "평균수익률(%)": round(test_stats["평균수익률"]*100, 2),
        })

        start += step_days

    res_df = pd.DataFrame(results)
    if res_df.empty:
        return res_df, {"ok": False, "reason": "워크포워드 결과 없음"}

    # 집계
    total_trades = int(res_df["트레이드수"].sum())
    w = (res_df["승률(%)"] * res_df["트레이드수"]).sum() / max(1, total_trades)
    pf_med = float(res_df["PF"].replace([np.inf, -np.inf], np.nan).dropna().median()) if "PF" in res_df else 0.0

    worst_pf = float(res_df["PF"].min())
    worst_win = float(res_df["승률(%)"].min())
    zero_windows = int((res_df["트레이드수"] == 0).sum())

    oos = {
        "ok": True,
        "총트레이드수": total_trades,
        "가중승률(%)": float(round(w, 1)),
        "PF(중앙값)": float(round(pf_med, 2)),
        "최악PF": float(round(worst_pf, 2)),
        "최악승률(%)": float(round(worst_win, 1)),
        "무거래구간수": zero_windows,
        "구간수": int(len(res_df)),
    }
    return res_df, oos
