import pandas as pd
import yfinance as yf

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

def get_daily_ohlcv(ticker: str, period: str = "5y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    df = _flatten_columns(df)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    df = df[df["Volume"] >= 0]
    return df

def get_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}

def liquidity_ok(df: pd.DataFrame, min_dollar_vol_20d: float = 2_000_000) -> bool:
    """20일 평균 거래대금(달러) 기준. 스윙에서 이거 미달이면 체결/슬리피지로 성과 붕괴 가능성 큼."""
    if df is None or df.empty or len(df) < 30:
        return False
    dv = (df["Close"] * df["Volume"]).rolling(20).mean().iloc[-1]
    return bool(dv >= min_dollar_vol_20d)
