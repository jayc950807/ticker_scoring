def trust_gate(
    liquidity_pass: bool,
    oos: dict,
    min_total_trades: int = 40,
    min_pf_median: float = 1.2,
    min_winrate: float = 45.0,
    max_zero_windows: int = 2,
) -> dict:
    """
    '확실'에 가까워지려면 인샘플이 아니라 OOS(워크포워드)로 게이트를 걸어야 함.
    """
    if not liquidity_pass:
        return {"allow_buy": False, "reason": "유동성 미달(20일 평균 거래대금 부족): 스윙에서 특히 위험"}

    if not oos or not oos.get("ok", False):
        return {"allow_buy": False, "reason": f"OOS 검증 실패/부족: {oos.get('reason','') if isinstance(oos, dict) else ''}".strip()}

    if oos.get("총트레이드수", 0) < min_total_trades:
        return {"allow_buy": False, "reason": f"OOS 트레이드 수 부족({oos.get('총트레이드수',0)} < {min_total_trades})"}

    if oos.get("PF(중앙값)", 0.0) < min_pf_median:
        return {"allow_buy": False, "reason": f"OOS PF 부족(PF 중앙값 {oos.get('PF(중앙값)',0)} < {min_pf_median})"}

    if oos.get("가중승률(%)", 0.0) < min_winrate:
        return {"allow_buy": False, "reason": f"OOS 승률 부족({oos.get('가중승률(%)',0)}% < {min_winrate}%)"}

    if oos.get("무거래구간수", 0) > max_zero_windows:
        return {"allow_buy": False, "reason": "OOS 구간 중 무거래(신호 부재) 비중 큼 → 재현성 의심"}

    # 최악 구간이 너무 처참하면 차단(장세 바뀔 때 무너지는 전략 방지)
    if oos.get("최악PF", 0.0) < 0.9:
        return {"allow_buy": False, "reason": f"최악 구간 PF가 너무 낮음({oos.get('최악PF')}) → 레짐 내성 부족"}

    return {"allow_buy": True, "reason": "OOS(워크포워드) 신뢰도 게이트 통과"}
