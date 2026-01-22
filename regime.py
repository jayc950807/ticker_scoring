import pandas as pd

def add_market_regime(stock_df: pd.DataFrame, spy_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """
    레짐(장세) 필터:
    - SPY가 MA200 아래면 리스크오프(스윙 매수 금지)
    - VIX가 높으면(기본 25) 리스크오프(스윙 매수 금지)
    """
    d = stock_df.copy()

    # Align dates
    spy = spy_df[["Close"]].rename(columns={"Close": "SPY_Close"}).copy()
    spy["SPY_MA200"] = spy["SPY_Close"].rolling(200).mean()

    vix = vix_df[["Close"]].rename(columns={"Close": "VIX_Close"}).copy()

    merged = d.join(spy, how="left").join(vix, how="left")
    merged["SPY_Close"] = merged["SPY_Close"].ffill()
    merged["SPY_MA200"] = merged["SPY_MA200"].ffill()
    merged["VIX_Close"] = merged["VIX_Close"].ffill()

    # Regime flags
    merged["RISK_OFF_SPY"] = merged["SPY_Close"] < merged["SPY_MA200"]
    merged["RISK_OFF_VIX"] = merged["VIX_Close"] >= 25

    # Conservative: either risk-off => risk-off
    merged["RISK_OFF"] = (merged["RISK_OFF_SPY"] | merged["RISK_OFF_VIX"]).fillna(True)

    return merged

def regime_summary(latest_row: pd.Series) -> tuple[str, str]:
    if bool(latest_row.get("RISK_OFF", True)):
        parts = []
        if bool(latest_row.get("RISK_OFF_SPY", False)):
            parts.append("지수 하락 레짐(SPY<MA200)")
        if bool(latest_row.get("RISK_OFF_VIX", False)):
            parts.append("변동성 레짐(VIX≥25)")
        msg = " / ".join(parts) if parts else "레짐 불명확"
        return "리스크오프", msg
    return "리스크온", "지수 추세 유지 + 변동성 과열 아님"
