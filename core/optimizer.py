# core/optimizer.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from .backtest import run_backtest
from .config import AppConfig

def walkforward_optimize(df: pd.DataFrame, entry_signal: pd.Series, cfg: AppConfig) -> Dict:
    """
    - 마지막 3y 중 최근 구간을 워크포워드로 여러 조각으로 쪼개 OOS 성능으로 파라미터 선택
    - 목적: "우연히 좋은 조합" 방지
    """
    # 거래일 기준 window 계산
    train = cfg.WF_TRAIN_DAYS
    test = cfg.WF_TEST_DAYS

    if len(df) < (train + test + 50):
        return {"ok": False, "reason": "데이터가 워크포워드에 부족합니다.", "best": None, "wf_table": pd.DataFrame()}

    grid = []
    for s in cfg.GRID_STOP_ATR:
        for t in cfg.GRID_TAKE_ATR:
            for h in cfg.GRID_MAX_HOLD:
                grid.append((s, t, h))

    segments = []
    start = 0
    while True:
        tr_start = start
        tr_end = tr_start + train
        te_end = tr_end + test
        if te_end >= len(df):
            break
        segments.append((tr_start, tr_end, tr_end, te_end))
        start += test  # 앞으로 test만큼 이동(워크포워드)

    if len(segments) < 2:
        return {"ok": False, "reason": "워크포워드 세그먼트가 부족합니다.", "best": None, "wf_table": pd.DataFrame()}

    rows = []
    for (stop_atr, take_atr, max_hold) in grid:
        oos_pfs = []
        oos_wins = []
        oos_mdds = []
        oos_trades = []

        for (_, _, te_s, te_e) in segments:
            df_te = df.iloc[te_s:te_e].copy()
            sig_te = entry_signal.iloc[te_s:te_e].copy()

            res = run_backtest(df_te, sig_te, stop_atr, take_atr, max_hold, cfg.COST_BPS)
            st = res["stats"]
            oos_pfs.append(st.get("pf", 0.0))
            oos_wins.append(st.get("win_rate", 0.0))
            oos_mdds.append(st.get("mdd", 0.0))
            oos_trades.append(st.get("trades", 0))

        # 요약
        pf_med = float(np.median(oos_pfs))
        pf_worst = float(np.min(oos_pfs))
        win_med = float(np.median(oos_wins))
        mdd_worst = float(np.min(oos_mdds))  # 더 음수인 값이 더 나쁨
        trades_sum = int(np.sum(oos_trades))

        # 목적함수(보수적): PF 중앙값 + 최악 PF 패널티 + 거래수 패널티
        score = pf_med
        if pf_worst < cfg.GATE_WORST_SEGMENT_PF:
            score -= 0.5
        if trades_sum < cfg.WF_MIN_TRADES_OOS:
            score -= 0.5
        if mdd_worst < -cfg.GATE_MAX_MDD_OOS:
            score -= 0.3

        rows.append({
            "stop_atr": stop_atr,
            "take_atr": take_atr,
            "max_hold": max_hold,
            "pf_med": pf_med,
            "pf_worst": pf_worst,
            "win_med": win_med,
            "mdd_worst": mdd_worst,
            "trades_sum": trades_sum,
            "wf_score": score
        })

    wf_table = pd.DataFrame(rows).sort_values("wf_score", ascending=False).reset_index(drop=True)

    best = wf_table.iloc[0].to_dict()
    best["ok_gate_hint"] = (best["pf_med"] >= cfg.GATE_MIN_PF_OOS and best["pf_worst"] >= cfg.GATE_WORST_SEGMENT_PF)

    return {"ok": True, "reason": "OK", "best": best, "wf_table": wf_table}
