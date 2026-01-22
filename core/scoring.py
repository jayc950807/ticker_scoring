# core/scoring.py
from __future__ import annotations
from typing import Dict

def score_all(
    oos_stats: Dict,
    regime: Dict,
    liquidity_ok: bool,
    short_quality: str
) -> Dict:
    """
    점수(0~100):
    - 검증 60점: PF/승률/거래수/MDD
    - 레짐 20점: 리스크오프면 상한 제한
    - 실행가능성 20점: 유동성/데이터 신뢰(Short 등)
    """
    pf = float(oos_stats.get("pf", 0.0))
    win = float(oos_stats.get("win_rate", 0.0))
    trades = int(oos_stats.get("trades", 0))
    mdd = float(oos_stats.get("mdd", 0.0))  # 음수

    score = 0

    # (1) 검증 점수(60)
    # PF
    if pf >= 1.5: score += 25
    elif pf >= 1.2: score += 18
    elif pf >= 1.0: score += 10
    else: score += 2

    # 승률
    if win >= 55: score += 15
    elif win >= 50: score += 12
    elif win >= 45: score += 8
    else: score += 3

    # 거래 수(너무 적으면 과최적화 가능성)
    if trades >= 15: score += 10
    elif trades >= 8: score += 7
    elif trades >= 5: score += 3
    else: score += 1

    # MDD (절대값 작을수록 좋음)
    if mdd >= -15: score += 10
    elif mdd >= -25: score += 7
    elif mdd >= -35: score += 4
    else: score += 1

    # (2) 레짐 점수(20)
    if regime.get("레짐", "").startswith("리스크온"):
        score += 18
    elif regime.get("레짐", "").startswith("리스크오프"):
        score += 4
    else:
        score += 10

    # (3) 실행가능성(20)
    score += 15 if liquidity_ok else 3
    # short 품질은 "정보 신뢰"로만 반영 (방향성 가점 X)
    score += 5 if short_quality == "OK" else 2

    # 과신 방지: 리스크오프면 상한
    if regime.get("BUY_차단", False):
        score = min(score, 60)

    return {"score": int(max(0, min(100, score)))}
