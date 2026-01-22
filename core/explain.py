# core/explain.py
from __future__ import annotations
from typing import Dict, List

def decide_and_explain(
    score: int,
    gate: Dict,
    regime: Dict,
    latest_signal: bool,
    oos_stats: Dict,
    liquidity_ok: bool
) -> Dict:
    """
    BUY/WAIT/SELL은 "규칙 기반"으로 고정.
    - 게이트 불통과 or 유동성 부족 or 리스크오프 + 신호 없음 => SELL/관망
    - 게이트 통과 + 최신 신호 True + 레짐 OK => BUY
    - 게이트 통과지만 신호 False => WAIT
    """
    reasons: List[str] = []

    pf = float(oos_stats.get("pf", 0.0))
    win = float(oos_stats.get("win_rate", 0.0))
    mdd = float(oos_stats.get("mdd", 0.0))
    trades = int(oos_stats.get("trades", 0))

    # 기본 근거(감사 가능하게 수치 포함)
    reasons.append(f"OOS 성과: PF {pf:.2f}, 승률 {win:.1f}%, 거래 {trades}회, MDD {mdd:.1f}%")
    reasons.append(f"레짐: {regime.get('레짐','중립')}")

    if not liquidity_ok:
        reasons.append("유동성/실행가능성 리스크: 20일 평균 거래대금이 기준 미달")
    if not gate.get("pass", False):
        reasons.append(f"게이트 미통과: {gate.get('fail_reason','')}")
    if regime.get("BUY_차단", False):
        reasons.append("리스크오프 레짐: BUY 차단(과신 방지)")

    # 결정
    if (not gate.get("pass", False)) or (not liquidity_ok):
        action = "SELL(회피)"
        summary = "검증(OOS) 또는 실행가능성 기준 미달이라 회피가 합리적입니다."
    else:
        if regime.get("BUY_차단", False) and not latest_signal:
            action = "SELL(회피)"
            summary = "방어 레짐이며 트리거도 약해 회피가 합리적입니다."
        else:
            if latest_signal and (not regime.get("BUY_차단", False)):
                action = "BUY(진입)"
                summary = "검증(OOS) 통과 + 레짐 우호 + 최신 진입 신호 충족."
            elif latest_signal and regime.get("BUY_차단", False):
                action = "WAIT(대기)"
                summary = "신호는 있지만 레짐이 방어적이라 대기가 합리적입니다."
            else:
                action = "WAIT(대기)"
                summary = "검증은 통과했지만 오늘 진입 트리거가 부족합니다."

    return {
        "action": action,
        "headline": summary,
        "reasons": reasons,
        "score": score
    }
