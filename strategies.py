# strategies.py
import numpy as np
import pandas as pd

def breakout_20d(df: pd.DataFrame) -> pd.Series:
    prev_high = df["High"].rolling(20).max().shift(1)
    return (df["Close"] > prev_high).fillna(False)

def squeeze_release(df: pd.DataFrame, tight_q: float = 0.20) -> pd.Series:
    """
    단순/안정형 squeeze release:
    - bb_over_kc가 과거 분포에서 낮았던(압축) 상태였고(어제 기준)
    - 오늘 bb_over_kc가 증가(확장) + range_over_atr가 커짐(실제 움직임)
    NOTE: tight_q는 '압축 판단'을 위한 분위수 컷오프. scoring에서 percentile을 따로 쓰면 더 깔끔.
    """
    # 룩어헤드 방지 위해 "압축"은 어제 기준으로만 쓰는 게 안전
    bb = df["bb_over_kc"].astype(float)
    bb_prev = bb.shift(1)

    # 압축 상태: 절대값으로 때리면 위험해서, 여기선 최소한의 안정 규칙만 둠
    # (실전은 scoring의 분위수로 tight 판정 권장)
    tight = (bb_prev < 1.0)  # BB<KC 근처면 압축 가정

    expand = bb.diff().fillna(0) > 0
    move = df["range_over_atr"].fillna(0) > 0.9  # 그날 움직임이 ATR 수준이면 의미 있는 해제

    return (tight & expand & move).fillna(False)

def quality_entry(df: pd.DataFrame) -> pd.Series:
    # 장기 추세 우위 + 단기 추세 무너짐 방지
    return ((df["Close"] > df["MA120"]) & (df["Close"] > df["MA20"])).fillna(False)

def exit_quality_trend_break(df: pd.DataFrame) -> pd.Series:
    # QUALITY는 느린 청산: MA120 이탈을 핵심으로
    return (df["Close"] < df["MA120"]).fillna(False)

def default_params(mode: str) -> dict:
    """
    모드별 '기본' 파라미터.
    (백테스트로 민감도 분석해서 튜닝해야 함. 지금은 안전한 보수값 위주)
    """
    if mode == "MOMO":
        return dict(entry_score=0.65, max_hold=15, atr_stop=2.0, atr_target=4.0)
    if mode == "SQUEEZE":
        return dict(entry_score=0.65, max_hold=10, atr_stop=1.8, atr_target=4.5)
    if mode == "QUALITY":
        return dict(entry_score=0.70, max_hold=60, atr_stop=2.5, atr_target=6.0)
    return dict(entry_score=0.65, max_hold=15, atr_stop=2.0, atr_target=4.0)
