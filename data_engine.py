# data_engine.py
import pandas as pd
import numpy as np
import yfinance as yf
import pytz
from typing import Optional, Tuple, Dict, Any

WINDOW_WARMUP = 160  # indicator 워밍업 컷

def get_market_macro() -> Dict[str, Any]:
    """
    ^VIX, ^TNX 기반 단순 리스크 조정.
    (절대값 컷오프는 참고만 하고, 실제 전략에선 사용 최소화 권장)
    """
    try:
        df = yf.download(['^VIX', '^TNX'], period='10d', progress=False)
        if df.empty:
            return {'vix': np.nan, 'tnx': np.nan, 'status': 'Unknown', 'score_adj': 0}

        close = df["Close"]
        if isinstance(close.columns, pd.MultiIndex):
            close.columns = [c[0] for c in close.columns]

        vix = float(close['^VIX'].dropna().iloc[-1]) if '^VIX' in close.columns else np.nan
        tnx = float(close['^TNX'].dropna().iloc[-1]) if '^TNX' in close.columns else np.nan

        status = "Normal"
        score_adj = 0
        if np.isfinite(vix):
            if vix > 25:
                status = "FEAR"
                score_adj = -10
            elif vix < 14:
                status = "CALM"
                score_adj = +3

        return {'vix': vix, 'tnx': tnx, 'status': status, 'score_adj': score_adj}
    except Exception:
        return {'vix': np.nan, 'tnx': np.nan, 'status': 'Unknown', 'score_adj': 0}

def _ensure_ohlcv_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    네 기존 지표 계산을 최소/안정적으로 재구성.
    백테스트에 꼭 필요한 것(ATR/MA/BB/KC) + 몇 개만 유지.
    """
    out = df.copy()

    # MAs
    out['MA20'] = out['Close'].rolling(20).mean()
    out['MA60'] = out['Close'].rolling(60).mean()
    out['MA120'] = out['Close'].rolling(120).mean()

    # Bollinger
    std_20 = out['Close'].rolling(20).std()
    out['BB_Upper'] = out['MA20'] + (std_20 * 2)
    out['BB_Lower'] = out['MA20'] - (std_20 * 2)

    # TR/ATR
    tr = np.maximum(
        out['High'] - out['Low'],
        np.maximum((out['High'] - out['Close'].shift(1)).abs(),
                   (out['Low'] - out['Close'].shift(1)).abs())
    )
    out['TR'] = tr
    out['ATR'] = out['TR'].rolling(14).mean()

    # Keltner (간이)
    out['KC_Upper'] = out['MA20'] + (out['ATR'] * 1.5)
    out['KC_Lower'] = out['MA20'] - (out['ATR'] * 1.5)

    # returns
    out['Log_Ret'] = np.log(out['Close'] / out['Close'].shift(1))

    return out

def get_realtime_synced_data(
    ticker: str,
    include_extended: bool = True,
    daily_period: str = "2y",
    intraday_period: str = "5d",
    intraday_interval: str = "1m",
    auto_adjust: bool = True
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Timestamp]]:
    """
    - 일봉 + 분봉으로 당일 OHLCV 보정
    - include_extended True면 pre/post 포함
    """
    try:
        df_daily = yf.download(
            ticker, period=daily_period, interval="1d",
            progress=False, auto_adjust=auto_adjust
        )
        df_daily = _ensure_ohlcv_cols(df_daily)

        df_intraday = yf.download(
            ticker, period=intraday_period, interval=intraday_interval,
            progress=False, auto_adjust=auto_adjust, prepost=include_extended
        )
        df_intraday = _ensure_ohlcv_cols(df_intraday)

        if df_daily is None or df_daily.empty:
            return None, None

        data_time_utc = df_daily.index[-1]

        # intraday로 당일 보정
        if df_intraday is not None and not df_intraday.empty:
            real_open = float(df_intraday['Open'].iloc[0])
            real_high = float(df_intraday['High'].max())
            real_low = float(df_intraday['Low'].min())
            real_close = float(df_intraday['Close'].iloc[-1])
            real_volume = float(df_intraday['Volume'].sum())

            last_idx = df_daily.index[-1]
            df_daily.loc[last_idx, 'Open'] = real_open
            df_daily.loc[last_idx, 'High'] = max(float(df_daily.loc[last_idx, 'High']), real_high)
            df_daily.loc[last_idx, 'Low'] = min(float(df_daily.loc[last_idx, 'Low']), real_low)
            df_daily.loc[last_idx, 'Close'] = real_close
            df_daily.loc[last_idx, 'Volume'] = real_volume

            data_time_utc = df_intraday.index[-1]

        # 지표 계산
        df = _add_indicators(df_daily)

        # 워밍업 컷 + 결측 제거
        if len(df) <= WINDOW_WARMUP + 50:
            return None, None

        df = df.iloc[WINDOW_WARMUP:].copy()
        df = df.dropna(subset=['Close', 'Open', 'High', 'Low', 'Volume', 'ATR', 'MA20', 'BB_Upper', 'KC_Upper'])

        return df, data_time_utc

    except Exception:
        return None, None

def format_kst_time(ts: Optional[pd.Timestamp]) -> str:
    if ts is None:
        return "N/A"
    try:
        if isinstance(ts, pd.Timestamp):
            if ts.tzinfo is None:
                ts = pytz.utc.localize(ts)
            kst = ts.astimezone(pytz.timezone("Asia/Seoul"))
            return kst.strftime("%Y-%m-%d %H:%M KST")
        return str(ts)
    except Exception:
        return str(ts)
