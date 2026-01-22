# scoring.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# 스코어카드: (feature_col, direction, weight)
# direction: '+' 높을수록 좋음, '-' 낮을수록 좋음
SCORECARDS: Dict[str, List[Tuple[str, str, float]]] = {
    "MOMO": [
        ("ret_20", "+", 0.30),
        ("ret_5", "+", 0.20),
        ("vol_spike", "+", 0.20),
        ("range_pct_60", "+", 0.15),
        ("close_vs_MA20", "+", 0.15),
    ],
    "SQUEEZE": [
        ("bb_over_kc", "-", 0.35),      # 작을수록 압축 강함
        ("vol_spike", "-", 0.20),       # 거래량 마름(평균 대비 낮음)
        ("range_pct_60", "-", 0.15),    # 압축기간 변동성 낮음
        ("close_vs_MA20", "+", 0.15),   # 위로 붙어있을수록 좋음(너무 하단 압축은 실패 확률↑)
        ("ret_5", "+", 0.15),           # 해제 직전 약한 가속
    ],
    "QUALITY": [
        ("close_vs_MA120", "+", 0.25),  # 장기 추세 우위
        ("range_pct_60", "-", 0.25),    # 낮은 변동성 선호
        ("drawdown_120", "+", 0.25),    # 덜 빠질수록 좋음(덜 음수)
        ("ret_20", "+", 0.15),
        ("vol_spike", "-", 0.10),       # 갑작스러운 과열은 감점 성격
    ],
}

def rolling_percentile_rank(series: pd.Series, lookback: int = 252, min_hist: int = 60) -> pd.Series:
    """
    룩어헤드 방지:
    - 오늘의 분위수는 '어제까지' 히스토리로만 계산
    - (<= val).mean 방식 percentile rank
    """
    s = series.astype(float)
    base = s.shift(1)

    ranks = []
    for i in range(len(s)):
        start = max(0, i - lookback)
        hist = base.iloc[start:i].dropna()
        if len(hist) < min_hist or not np.isfinite(s.iloc[i]):
            ranks.append(np.nan)
            continue
        val = float(s.iloc[i])
        ranks.append(float((hist <= val).mean()))
    return pd.Series(ranks, index=s.index)

def pct_to_score(p: pd.Series, curve: float = 1.6) -> pd.Series:
    """
    분위수(0~1) -> -1~+1
    - 중간(0.5)은 0
    - 극단일수록 비선형으로 강해짐(잡음 구간 억제)
    """
    x = (p - 0.5) * 2
    x = np.sign(x) * (np.abs(x) ** curve)
    return x.clip(-1, 1)

def feature_score(df: pd.DataFrame, col: str, direction: str, lookback: int = 252) -> pd.Series:
    p = rolling_percentile_rank(df[col], lookback=lookback)
    s = pct_to_score(p)
    return s if direction == "+" else -s

def compute_mode_score(df: pd.DataFrame, mode: str, lookback: int = 252) -> pd.Series:
    if mode not in SCORECARDS:
        raise ValueError(f"Unknown mode: {mode}")

    parts = []
    for col, direction, w in SCORECARDS[mode]:
        fs = feature_score(df, col, direction, lookback=lookback)
        parts.append(fs * float(w))

    raw = pd.concat(parts, axis=1).sum(axis=1)  # -1~+1 근처
    score01 = (raw + 1.0) / 2.0                 # 0~1
    return score01.clip(0, 1)

def classify_mode(df: pd.DataFrame) -> str:
    """
    '절대' 모드 분류는 위험해서 최소 규칙만 둠.
    - QUALITY: 장기 추세 우위(종가>MA120) & 변동성 낮음
    - SQUEEZE: bb_over_kc가 낮은 구간(압축) 빈번
    - MOMO: 최근 수익률/거래량 가속
    """
    if len(df) < 200:
        return "MOMO"

    last = df.iloc[-1]

    # 간단한 상태 기반 후보: (백테스트에선 모드 고정이 더 낫다)
    cond_quality = bool(last["Close"] > last["MA120"]) and bool(last.get("range_pct_60", 1) < 0.04)
    cond_squeeze = bool(last.get("bb_over_kc", 9) < 1.0)  # BB폭 < KC폭 근처
    cond_momo = bool(last.get("ret_5", 0) > 0.05) and bool(last.get("vol_spike", 1) > 1.5)

    if cond_squeeze:
        return "SQUEEZE"
    if cond_quality:
        return "QUALITY"
    if cond_momo:
        return "MOMO"
    return "MOMO"
