# core/config.py
from dataclasses import dataclass

@dataclass(frozen=True)
class AppConfig:
    TZ: str = "Asia/Seoul"

    # 데이터
    DAILY_PERIOD: str = "3y"
    INTRADAY_PERIOD: str = "5d"
    INTRADAY_INTERVAL: str = "1m"
    AUTO_ADJUST: bool = True

    # 워크포워드(자동 최적화)
    WF_TRAIN_DAYS: int = 504      # 약 2년(거래일)
    WF_TEST_DAYS: int = 126       # 약 6개월
    WF_MIN_TRADES_OOS: int = 6
    COST_BPS: float = 10.0        # 0.10% (보수적)

    # 최적화 탐색 범위(좁게! 데이터 스누핑 방지)
    GRID_STOP_ATR = (1.5, 2.0, 2.5)
    GRID_TAKE_ATR = (3.0, 4.0, 5.0)
    GRID_MAX_HOLD = (20, 30, 40)

    # BUY/WAIT/SELL 게이트(투자 품질 중심)
    GATE_MIN_PF_OOS: float = 1.20
    GATE_MIN_WIN_OOS: float = 45.0
    GATE_MAX_MDD_OOS: float = 35.0
    GATE_MIN_TRADES_OOS: int = 6
    GATE_WORST_SEGMENT_PF: float = 0.95

    # 유동성/실행가능성(보수적으로)
    MIN_DOLLAR_VOL_20D: float = 5_000_000  # 20일 평균 거래대금 최소
