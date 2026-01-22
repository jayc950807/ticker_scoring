import pandas as pd

def swing_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    스윙 룰(고정, 백테스트 가능한 스펙):
    ENTRY:
      1) 추세필터: Close > MA200
      2) 눌림목: RSI <= RSI_Q40 (분위수 기반)
      3) 거래량 확인: VOL_RATIO >= VOL_Q80
      4) 회복 확인: Close > MA20
      5) 레짐 필터: RISK_OFF=False (시장 위험구간 BUY 금지)

    EXIT (후보 플래그, 실제 청산은 backtest에서 손절/익절/타임스탑과 함께 처리):
      - 추세 훼손: Close < MA50
      - 과열: RSI >= 70
    """
    d = df.copy()

    trend_ok = d["Close"] > d["MA200"]
    pullback = d["RSI"] <= d["RSI_Q40"]
    vol_confirm = d["VOL_RATIO"] >= d["VOL_Q80"]
    regain = d["Close"] > d["MA20"]
    regime_ok = (~d["RISK_OFF"]).fillna(False)

    d["ENTRY"] = (trend_ok & pullback & vol_confirm & regain & regime_ok).fillna(False)

    d["EXIT_TREND_BREAK"] = (d["Close"] < d["MA50"]).fillna(False)
    d["EXIT_OVERHEAT"] = (d["RSI"] >= 70).fillna(False)

    return d

def rule_reco(latest_row: pd.Series, gate: dict) -> tuple[str, str]:
    """
    규칙 기반 추천 (고정):
    - gate(신뢰도/유동성/OOS/레짐) 통과 못하면 BUY 금지
    """
    if not gate.get("allow_buy", False):
        return "대기", gate.get("reason", "신뢰도 기준 미충족")

    if bool(latest_row.get("ENTRY", False)):
        return "매수", "추세(MA200) + 눌림(RSI 분위수) + 거래량 확인 + MA20 회복 + 레짐 OK"

    if bool(latest_row.get("EXIT_TREND_BREAK", False)) or bool(latest_row.get("EXIT_OVERHEAT", False)):
        return "매도/정리", "추세 훼손(MA50 하회) 또는 과열(RSI≥70)"

    return "관망", "진입 조건 미충족(조건 대기)"
