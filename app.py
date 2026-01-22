import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import pytz
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

# ============================================================
# Streamlit 기본 설정
# ============================================================
st.set_page_config(
    page_title="AI Stock Sniper",
    page_icon="🦄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 다크모드 스타일(기존 유지)
st.markdown("""
    <style>
    .stApp { background-color: #121212; color: #e0e0e0; }
    .stTextInput > div > div > input { background-color: #262626; color: white; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
    a { color: #00B0FF !important; text-decoration: none; }
    a:hover { text-decoration: underline; }
    [data-testid="stDataFrame"] { border: 1px solid #333; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# 로거 차단
logger = logging.getLogger("yfinance")
logger.setLevel(logging.CRITICAL)
plt.style.use("dark_background")

# ============================================================
# Optional libs (있으면 사용, 없으면 graceful fallback)
# ============================================================
try:
    from duckduckgo_search import DDGS
except Exception:
    class DDGS:
        def news(self, keywords, max_results=5):
            return []

try:
    from deep_translator import GoogleTranslator
except Exception:
    class GoogleTranslator:
        def __init__(self, source="auto", target="ko"):
            pass
        def translate(self, text):
            return text

# ============================================================
# Constants
# ============================================================
REF_DATA = {
    'AAPL': 'Apple', 'MSFT': 'Microsoft', 'NVDA': 'NVIDIA', 'TSLA': 'Tesla',
    'GOOGL': 'Google', 'AMZN': 'Amazon', 'META': 'Meta', 'AMD': 'AMD',
    'NFLX': 'Netflix', 'INTC': 'Intel', 'QCOM': 'Qualcomm', 'AVGO': 'Broadcom',
    'JPM': 'JPMorgan', 'BAC': 'BoA', 'GS': 'GoldmanSachs', 'V': 'Visa',
    'JNJ': 'Johnson&Johnson', 'LLY': 'EliLilly', 'PFE': 'Pfizer', 'UNH': 'UnitedHealth',
    'KO': 'CocaCola', 'PEP': 'Pepsi', 'MCD': 'McDonalds', 'WMT': 'Walmart',
    'PLTR': 'Palantir', 'SOFI': 'SoFi', 'COIN': 'Coinbase', 'AMC': 'AMC', 'GME': 'GameStop',
    'XOM': 'Exxon', 'CVX': 'Chevron',
    'IWM': 'Russell2000', 'SPY': 'S&P500', 'QQQ': 'Nasdaq', 'SOXX': 'Semiconductor'
}
REFERENCE_TICKERS = list(REF_DATA.keys())

WINDOW_SIZE = 60
FORECAST_DAYS = 30
DEFAULT_SIM_DAYS = 120
DEFAULT_SIM_N = 5000

# Color palette (기존 유지)
C_BULL = "#00E676"
C_BEAR = "#FF5252"
C_NEUT = "#B0BEC5"
C_WARN = "#FFD740"
C_CYAN = "#00B0FF"
C_PURP = "#E040FB"

# ============================================================
# Caching
# ============================================================
@st.cache_resource
def get_global_ref_cache():
    return {}

GLOBAL_REF_CACHE = get_global_ref_cache()

# ============================================================
# Helpers
# ============================================================
def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def safe_int(x, default=0):
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default

# ============================================================
# Market / Info
# ============================================================
def get_stock_info_basic(ticker: str) -> Dict[str, Any]:
    """
    가벼운 표시용 요약(기존 get_stock_info 대체).
    분류/공매도/float 등은 t.info 원본을 별도로 사용.
    """
    try:
        info = yf.Ticker(ticker).info or {}
        return {
            "name": info.get("longName", ticker),
            "mkt_cap": info.get("marketCap", 0),
            "per": info.get("trailingPE", None),
            "pbr": info.get("priceToBook", None),
            "roe": info.get("returnOnEquity", None),
            "insider_own": info.get("heldPercentInsiders", 0),
        }
    except Exception:
        return {"name": ticker, "mkt_cap": 0, "per": None, "pbr": None, "roe": None, "insider_own": 0}

def get_market_macro() -> Dict[str, Any]:
    """
    ^VIX, ^TNX 기반 단순 리스크 조정(기존 유지 + 안정성 강화)
    """
    try:
        df = yf.download(['^VIX', '^TNX'], period='10d', progress=False)
        if df.empty:
            return {'vix': 0, 'tnx': 0, 'status': 'Unknown', 'score_adj': 0}
        close = df["Close"]
        if isinstance(close.columns, pd.MultiIndex):
            close.columns = [c[0] for c in close.columns]
        vix = safe_float(close['^VIX'].dropna().iloc[-1], 0)
        tnx = safe_float(close['^TNX'].dropna().iloc[-1], 0)

        status = "Normal"
        score_adj = 0
        if vix > 25:
            status = "FEAR (위험)"
            score_adj = -15
        elif vix < 14:
            status = "GREED (안정)"
            score_adj = +5
        return {'vix': vix, 'tnx': tnx, 'status': status, 'score_adj': score_adj}
    except Exception:
        return {'vix': 0, 'tnx': 0, 'status': 'Unknown', 'score_adj': 0}

# ============================================================
# Data Engine (prepost 옵션 포함)
# ============================================================
def get_realtime_synced_data(ticker: str, include_extended: bool = True) -> Tuple[Optional[pd.DataFrame], Optional[pd.Timestamp]]:
    """
    - 일봉 2년 + 분봉(최근 5일)으로 당일 OHLCV 보정
    - include_extended=True면 pre/post 포함 가능(prepost=True)
    """
    try:
        df_daily = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
        if isinstance(df_daily.columns, pd.MultiIndex):
            df_daily.columns = [col[0] for col in df_daily.columns]

        df_intraday = yf.download(
            ticker, period="5d", interval="1m",
            progress=False, auto_adjust=True,
            prepost=include_extended
        )
        if isinstance(df_intraday.columns, pd.MultiIndex):
            df_intraday.columns = [col[0] for col in df_intraday.columns]

        if df_daily is None or df_daily.empty:
            return None, None

        # intraday 기반 당일 보정(가능한 경우)
        if df_intraday is not None and not df_intraday.empty:
            real_open = df_intraday['Open'].iloc[0]
            real_high = df_intraday['High'].max()
            real_low = df_intraday['Low'].min()
            real_close = df_intraday['Close'].iloc[-1]
            real_volume = df_intraday['Volume'].sum()

            last_idx = df_daily.index[-1]
            df_daily.loc[last_idx, 'Open'] = real_open
            df_daily.loc[last_idx, 'High'] = max(df_daily.loc[last_idx, 'High'], real_high)
            df_daily.loc[last_idx, 'Low'] = min(df_daily.loc[last_idx, 'Low'], real_low)
            df_daily.loc[last_idx, 'Close'] = real_close
            df_daily.loc[last_idx, 'Volume'] = real_volume
            data_time_utc = df_intraday.index[-1]
        else:
            data_time_utc = df_daily.index[-1]

        if len(df_daily) < WINDOW_SIZE + FORECAST_DAYS + 150:
            return None, None

        df = df_daily.copy()

        # Moving averages
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()
        df['MA120'] = df['Close'].rolling(120).mean()

        # MACD
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-6)
        df['RSI'] = 100 - (100 / (1 + rs))

        # Stoch
        low_14 = df['Low'].rolling(14).min()
        high_14 = df['High'].rolling(14).max()
        df['Stoch_K'] = ((df['Close'] - low_14) / (high_14 - low_14).replace(0, 1)) * 100
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

        # CCI
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        df['CCI'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std()).replace(0, 1e-6)

        # WillR
        df['WillR'] = ((high_14 - df['Close']) / (high_14 - low_14).replace(0, 1)) * -100

        # Bollinger
        std_20 = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['MA20'] + (std_20 * 2)
        df['BB_Lower'] = df['MA20'] - (std_20 * 2)

        # ATR / Keltner(간이)
        df['TR'] = np.maximum(df['High'] - df['Low'],
                              np.maximum(abs(df['High'] - df['Close'].shift(1)),
                                         abs(df['Low'] - df['Close'].shift(1))))
        df['ATR'] = df['TR'].rolling(14).mean()
        df['KC_Upper'] = df['MA20'] + (df['ATR'] * 1.5)
        df['KC_Lower'] = df['MA20'] - (df['ATR'] * 1.5)

        # OBV
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

        # Acc/Dist line
        ad_factor = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']).replace(0, 1)
        df['AD_Line'] = (ad_factor * df['Volume']).fillna(0).cumsum()

        # MFI
        typical = (df['High'] + df['Low'] + df['Close']) / 3
        mf = typical * df['Volume']
        pos_mf = mf.where(typical > typical.shift(1), 0).rolling(14).sum()
        neg_mf = mf.where(typical < typical.shift(1), 0).rolling(14).sum().replace(0, 1)
        df['MFI'] = 100 - (100 / (1 + (pos_mf / neg_mf)))

        # VWAP(20)
        df['VWAP'] = (df['Volume'] * typical).rolling(20).sum() / df['Volume'].rolling(20).sum().replace(0, 1)

        # ROC(12)
        df['ROC'] = ((df['Close'] - df['Close'].shift(12)) / df['Close'].shift(12).replace(0, 1)) * 100

        # Ichimoku
        nine_high = df['High'].rolling(window=9).max()
        nine_low = df['Low'].rolling(window=9).min()
        df['Tenkan'] = (nine_high + nine_low) / 2
        twenty_six_high = df['High'].rolling(window=26).max()
        twenty_six_low = df['Low'].rolling(window=26).min()
        df['Kijun'] = (twenty_six_high + twenty_six_low) / 2
        df['Senkou_Span_A'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
        fifty_two_high = df['High'].rolling(window=52).max()
        fifty_two_low = df['Low'].rolling(window=52).min()
        df['Senkou_Span_B'] = ((fifty_two_high + fifty_two_low) / 2).shift(26)

        # Log returns + intraday-ish volatility proxy
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = (df['High'] - df['Low']) / df['Close'] * 100

        # 워밍업 제거
        df = df.iloc[160:].copy()
        df = df.dropna(subset=['Close'])

        return df, data_time_utc

    except Exception:
        return None, None

# ============================================================
# Squeeze (연속 상태 + 트리거)
# ============================================================
def get_ttm_squeeze_state(df: pd.DataFrame, min_on_days: int = 5) -> Dict[str, Any]:
    """
    raw_on: BB폭 < KC폭
    on: raw_on이 min_on_days 이상 연속일 때만 True
    trigger_on_today: 오늘 raw_on 전환 + on 충족
    """
    if df is None or len(df) < 60:
        return {"on": False, "on_streak": 0, "trigger_on_today": False, "trigger_off_today": False}

    bb_width = (df['BB_Upper'] - df['BB_Lower'])
    kc_width = (df['KC_Upper'] - df['KC_Lower'])
    raw_on = (bb_width < kc_width).fillna(False)

    streak = 0
    for v in raw_on.iloc[::-1]:
        if bool(v):
            streak += 1
        else:
            break

    on = streak >= int(min_on_days)
    prev_raw = bool(raw_on.iloc[-2]) if len(raw_on) >= 2 else False
    curr_raw = bool(raw_on.iloc[-1])

    trigger_on_today = (not prev_raw) and curr_raw and on
    trigger_off_today = prev_raw and (not curr_raw)

    return {"on": on, "on_streak": streak, "trigger_on_today": trigger_on_today, "trigger_off_today": trigger_off_today}

def check_candle_pattern(df: pd.DataFrame) -> Optional[str]:
    last = df.iloc[-1]
    open_p, close_p = last['Open'], last['Close']
    high_p, low_p = last['High'], last['Low']
    body = abs(close_p - open_p)
    upper_shadow = high_p - max(open_p, close_p)
    lower_shadow = min(open_p, close_p) - low_p
    total_range = high_p - low_p
    if total_range == 0:
        return None
    if (lower_shadow > body * 2) and (upper_shadow < body * 0.5) and (lower_shadow > upper_shadow * 2):
        return "Hammer"
    if body <= (total_range * 0.1):
        return "Doji"
    return None

def check_rsi_divergence(df: pd.DataFrame, window=10) -> Optional[str]:
    if len(df) < window * 2:
        return None
    current = df.iloc[-window:]
    prev = df.iloc[-window*2:-window]

    curr_low_price = current['Close'].min()
    prev_low_price = prev['Close'].min()
    curr_low_rsi = current.loc[current['Close'].idxmin()]['RSI']
    prev_low_rsi = prev.loc[prev['Close'].idxmin()]['RSI']

    curr_high_price = current['Close'].max()
    prev_high_price = prev['Close'].max()
    curr_high_rsi = current.loc[current['Close'].idxmax()]['RSI']
    prev_high_rsi = prev.loc[prev['Close'].idxmax()]['RSI']

    if curr_low_price < prev_low_price and curr_low_rsi > prev_low_rsi:
        return "REG_BULL"
    if curr_high_price > prev_high_price and curr_high_rsi < prev_high_rsi:
        return "REG_BEAR"
    if curr_low_price > prev_low_price and curr_low_rsi < prev_low_rsi:
        return "HID_BULL"
    if curr_high_price < prev_high_price and curr_high_rsi > prev_high_rsi:
        return "HID_BEAR"
    return None

# ============================================================
# Monte Carlo (오류 수정: GBM/lognormal로 일관)
# ============================================================
def run_monte_carlo(df: pd.DataFrame, num_simulations: int = DEFAULT_SIM_N, days: int = DEFAULT_SIM_DAYS):
    close = df['Close'].dropna()
    if len(close) < 60:
        # 기존 인터페이스 유지
        return None, None, None, None, 0, "데이터 부족", 0, {}, 0, []

    last_price = float(close.iloc[-1])
    log_ret = np.log(close / close.shift(1)).dropna()

    window = min(60, len(log_ret))
    r = log_ret.tail(window)

    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    drift = mu - 0.5 * sigma**2

    sims = int(num_simulations)
    z = np.random.normal(0, 1, size=(days, sims))
    log_path = np.log(last_price) + np.cumsum(drift + sigma * z, axis=0)
    price_path = np.exp(log_path)
    sim_df = pd.DataFrame(price_path)

    target_percents = [0.3, 0.5, 0.7, 1.0, 1.5]
    max_peaks = sim_df.max(axis=0)

    main_target = last_price * 1.30
    win_prob = float((max_peaks >= main_target).mean() * 100)

    target_peak_price = float(np.median(max_peaks))
    peak_yield = float((target_peak_price - last_price) / last_price * 100)

    ending_values = sim_df.iloc[-1, :]
    min_yield = float((np.percentile(ending_values, 10) - last_price) / last_price * 100)

    extra_scenarios = []
    for pct in target_percents:
        tgt_price = last_price * (1 + pct)
        prob = float((max_peaks >= tgt_price).mean() * 100)
        extra_scenarios.append({'pct': int(pct*100), 'prob': prob, 'date': "-"})

    expected_date_str = "확률 기반(경로 의존)"
    forecast_data = {}
    return sim_df, None, None, None, win_prob, expected_date_str, peak_yield, forecast_data, min_yield, extra_scenarios

# ============================================================
# Short data reliability (없음=0 금지)
# ============================================================
def safe_get_short_metrics(info: Dict[str, Any]):
    spof = info.get('shortPercentOfFloat', None)  # 비율(0~1)
    dtc = info.get('shortRatio', None)            # days to cover
    float_shares = info.get('floatShares', None)

    if spof is None and dtc is None:
        return None, None, float_shares, "MISSING"

    short_pct = None if spof is None else float(spof) * 100
    days_to_cover = None if dtc is None else float(dtc)

    flag = "OK"
    if short_pct is None or days_to_cover is None:
        flag = "STALE/UNSURE"

    return short_pct, days_to_cover, float_shares, flag

# ============================================================
# News (기존 유지)
# ============================================================
def get_google_news_rss(ticker: str):
    try:
        url = f"https://news.google.com/rss/search?q={ticker}+stock+news&hl=en-US&gl=US&ceid=US:en"
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            news_items = []
            for item in root.findall('.//item')[:3]:
                title = item.find('title')
                link = item.find('link')
                if title is not None:
                    url_txt = link.text if link is not None else "#"
                    news_items.append({'title': title.text, 'url': url_txt})
            return news_items
    except Exception:
        return []
    return []

def get_sentiment_and_short_data(ticker: str, df: pd.DataFrame, info: Dict[str, Any], squeeze_min_on_days: int):
    data = {
        'short_pct': None,
        'days_to_cover': None,
        'short_signal': 'Unknown',
        'short_data_quality': 'MISSING',
        'upside_pot': 0,
        'analyst_signal': 'N/A',
        'news_score': 0,
        'news_signal': 'Neutral',
        'headlines': [],
        'squeeze_state': get_ttm_squeeze_state(df, min_on_days=squeeze_min_on_days)
    }

    short_pct, dtc, float_shares, qflag = safe_get_short_metrics(info)
    data['short_pct'] = short_pct
    data['days_to_cover'] = dtc
    data['short_data_quality'] = qflag

    # short signal은 OK일 때만 의미 있게
    if qflag == "OK" and short_pct is not None and dtc is not None:
        if short_pct >= 30 and dtc >= 5:
            data['short_signal'] = "Squeeze Setup"
        elif short_pct >= 20:
            data['short_signal'] = "High Short"
        else:
            data['short_signal'] = "Normal"
    else:
        data['short_signal'] = "Unknown"

    # Analyst upside (있으면 참고)
    try:
        current_price = float(df['Close'].iloc[-1])
        target_mean = info.get('targetMeanPrice', None)
        if target_mean is None:
            target_mean = current_price
        target_mean = float(target_mean)
        upside_pot = ((target_mean - current_price) / current_price) * 100
        data['upside_pot'] = upside_pot
        data['analyst_signal'] = "Bull" if upside_pot > 10 else ("Bear" if upside_pot < -10 else "Neutral")
    except Exception:
        pass

    # 뉴스 수집 (yfinance -> ddg -> google rss)
    raw_news_items = []
    try:
        t = yf.Ticker(ticker)
        yf_news = getattr(t, "news", None)
        if yf_news:
            for item in yf_news[:3]:
                title = item.get('title', '')
                link = item.get('link', '#')
                if title:
                    raw_news_items.append({'title': title, 'url': link})
    except Exception:
        pass

    if len(raw_news_items) < 3:
        try:
            ddgs = DDGS()
            ddg_res = ddgs.news(keywords=f"{ticker} stock", max_results=3)
            if ddg_res:
                for item in ddg_res:
                    title = item.get('title', '')
                    link = item.get('url', '#')
                    if title:
                        raw_news_items.append({'title': title, 'url': link})
        except Exception:
            pass

    if len(raw_news_items) < 3:
        try:
            g_news = get_google_news_rss(ticker)
            if g_news:
                raw_news_items.extend(g_news)
        except Exception:
            pass

    # 중복 제거
    unique_news = []
    seen = set()
    for item in raw_news_items:
        if item['title'] not in seen:
            seen.add(item['title'])
            unique_news.append(item)
    unique_news = unique_news[:5]

    sentiment_score = 0
    bull_words = ['up','surge','jump','beat','growth','gain','buy','strong','profit','partnership','merger','record','soar','bull','upgrade']
    bear_words = ['down','drop','fall','miss','loss','sell','weak','lawsuit','investigation','inflation','cut','crash','plunge','bear','downgrade']

    try:
        translator = GoogleTranslator(source='auto', target='ko')
    except Exception:
        translator = None

    final_headlines = []
    for item in unique_news:
        title = item['title']
        url = item['url']
        lower = title.lower()
        for w in bull_words:
            if w in lower:
                sentiment_score += 1
        for w in bear_words:
            if w in lower:
                sentiment_score -= 1

        translated = title
        if translator:
            try:
                translated = translator.translate(title)
            except Exception:
                pass

        final_headlines.append({'title': translated, 'url': url})

    data['news_score'] = sentiment_score
    data['news_signal'] = "Positive" if sentiment_score > 0 else ("Negative" if sentiment_score < 0 else "Neutral")
    data['headlines'] = final_headlines
    return data

# ============================================================
# Technical signals (18) - squeeze state 반영
# ============================================================
def get_18_tech_signals(df: pd.DataFrame, squeeze_min_on_days: int):
    last = df.iloc[-1]
    signals = []

    signals.append(("SMA 20", f"{last['MA20']:.2f}", "Bull" if last['Close'] > last['MA20'] else "Bear"))
    signals.append(("SMA 60", f"{last['MA60']:.2f}", "Bull" if last['Close'] > last['MA60'] else "Bear"))
    signals.append(("SMA 120", f"{last['MA120']:.2f}", "Bull" if last['Close'] > last['MA120'] else "Bear"))

    rsi = last['RSI']
    bias = "Bear" if rsi > 70 else ("Bull" if rsi < 30 else "Neutral")
    signals.append(("RSI (14)", f"{rsi:.1f}", bias))

    macd = last['MACD']
    sig = last['MACD_Signal']
    signals.append(("MACD", f"{macd:.2f}", "Bull" if macd > sig else "Bear"))

    k = last['Stoch_K']
    d = last['Stoch_D']
    signals.append(("Stoch", f"{k:.0f}/{d:.0f}", "Bull" if k > d else "Bear"))

    cci = last['CCI']
    bias = "Bear" if cci > 100 else ("Bull" if cci < -100 else "Neutral")
    signals.append(("CCI", f"{cci:.0f}", bias))

    wr = last['WillR']
    bias = "Bull" if wr < -80 else ("Bear" if wr > -20 else "Neutral")
    signals.append(("Will%R", f"{wr:.0f}", bias))

    pos, bias = ("Mid", "Neutral")
    if last['Close'] > last['BB_Upper']:
        pos, bias = "High", "Bear"
    elif last['Close'] < last['BB_Lower']:
        pos, bias = "Low", "Bull"
    signals.append(("Bollinger", pos, bias))

    signals.append(("ATR", f"{last['ATR']:.2f}", "Neutral"))

    obv_ma = df['OBV'].rolling(20).mean().iloc[-1]
    signals.append(("OBV", "Up" if last['OBV'] > obv_ma else "Down", "Bull" if last['OBV'] > obv_ma else "Bear"))

    mfi = last['MFI']
    bias = "Bear" if mfi > 80 else ("Bull" if mfi < 20 else "Neutral")
    signals.append(("MFI", f"{mfi:.0f}", bias))

    signals.append(("VWAP", f"{last['VWAP']:.2f}", "Bull" if last['Close'] > last['VWAP'] else "Bear"))

    roc = last['ROC']
    signals.append(("ROC", f"{roc:.2f}%", "Bull" if roc > 0 else "Bear"))

    cloud_top = max(last['Senkou_Span_A'], last['Senkou_Span_B'])
    cloud_bot = min(last['Senkou_Span_A'], last['Senkou_Span_B'])
    ichi, bias = "In", "Neutral"
    if last['Close'] > cloud_top:
        ichi, bias = "Above", "Bull"
    elif last['Close'] < cloud_bot:
        ichi, bias = "Below", "Bear"
    signals.append(("Ichimoku", ichi, bias))

    sqz = get_ttm_squeeze_state(df, min_on_days=squeeze_min_on_days)
    sqz_val = f"ON({sqz['on_streak']})" if sqz['on'] else "OFF"
    sqz_bias = "Bull" if sqz['trigger_on_today'] else ("Bear" if sqz['trigger_off_today'] else "Neutral")
    signals.append(("Squeeze", sqz_val, sqz_bias))

    pat = check_candle_pattern(df)
    signals.append(("Candle", pat if pat else "-", "Bull" if pat == "Hammer" else "Neutral"))

    vol = last['Volatility']
    signals.append(("Vol Ratio", f"{vol:.2f}%", "Neutral"))

    return signals

# ============================================================
# Insider / filings (기존 유지 + 안정성)
# ============================================================
def get_insider_trading(ticker: str) -> Optional[pd.DataFrame]:
    try:
        t = yf.Ticker(ticker)
        insider = t.insider_transactions
        if insider is None or insider.empty:
            return None

        if 'Start Date' in insider.columns:
            insider = insider.sort_values(by='Start Date', ascending=False)

        cols = [c for c in ['Start Date', 'Insider', 'Position', 'Text', 'Shares', 'Value'] if c in insider.columns]
        insider = insider[cols].copy()

        if 'Value' in insider.columns:
            insider['Value'] = pd.to_numeric(insider['Value'], errors='coerce').fillna(0)

        insider = insider.reset_index(drop=True).head(30)
        return insider
    except Exception:
        return None

def get_sec_filings(ticker: str) -> Optional[pd.DataFrame]:
    """
    yfinance의 sec_filings는 종종 구조가 달라지거나 비어있을 수 있음.
    가능한 형태만 안전하게 파싱.
    """
    try:
        t = yf.Ticker(ticker)
        filings = getattr(t, "sec_filings", None)
        if not filings:
            return None

        # filings가 list[dict] 형태를 기대
        if isinstance(filings, list):
            rows = []
            for f in filings:
                if not isinstance(f, dict):
                    continue
                rows.append({
                    'Date': f.get('date'),
                    'Type': f.get('type'),
                    'Title': f.get('title'),
                    'Link': f.get('edgarUrl')
                })
            out = pd.DataFrame(rows)
            return out if not out.empty else None

        # filings가 DataFrame 형태인 경우도 방어
        if isinstance(filings, pd.DataFrame):
            return filings

        return None
    except Exception:
        return None

# ============================================================
# Automatic classification: QUALITY vs MOMO vs SQUEEZE
# ============================================================
def classify_mode(ticker: str, df: pd.DataFrame, info: Dict[str, Any]) -> Dict[str, Any]:
    mkt_cap = info.get("marketCap") or 0
    price = float(df["Close"].iloc[-1])

    last20 = df.tail(20)
    last60 = df.tail(60)

    range_pct = float(((last60["High"] - last60["Low"]) / last60["Close"]).mean() * 100) if len(last60) else 0
    vol_avg20 = float(last20["Volume"].mean()) if len(last20) else 0
    vol_spike = float(df["Volume"].iloc[-1] / vol_avg20) if vol_avg20 else 0

    ret_5 = float((df["Close"].iloc[-1] / df["Close"].iloc[-6] - 1) * 100) if len(df) > 6 else 0
    ret_20 = float((df["Close"].iloc[-1] / df["Close"].iloc[-21] - 1) * 100) if len(df) > 21 else 0

    sp, dtc, flt, qflag = safe_get_short_metrics(info)

    # Squeeze score: short 데이터 OK일 때만 의미 있게
    squeeze_score = 0
    if qflag == "OK" and sp is not None and dtc is not None:
        if sp >= 30: squeeze_score += 45
        elif sp >= 20: squeeze_score += 30
        if dtc >= 5: squeeze_score += 25
        elif dtc >= 3: squeeze_score += 15
        if flt and flt < 30_000_000: squeeze_score += 15
        if vol_spike >= 3: squeeze_score += 10
        if ret_5 >= 15: squeeze_score += 5

    momo_score = 0
    if ret_5 >= 30: momo_score += 45
    elif ret_20 >= 50: momo_score += 35
    if vol_spike >= 3: momo_score += 25
    if range_pct >= 5: momo_score += 20
    if price < 10: momo_score += 10

    quality_score = 0
    if mkt_cap >= 10_000_000_000: quality_score += 45
    if range_pct <= 3.5: quality_score += 25
    roe = info.get("returnOnEquity")
    if roe is not None and roe > 0.10: quality_score += 15

    if squeeze_score >= 60:
        mode, conf = "SQUEEZE", min(95, squeeze_score)
    elif momo_score >= 60:
        mode, conf = "MOMO", min(90, momo_score)
    elif quality_score >= 60:
        mode, conf = "QUALITY", min(90, quality_score)
    else:
        mode, conf = "UNKNOWN", max(squeeze_score, momo_score, quality_score)

    label = {"SQUEEZE":"숏스퀴즈 후보", "MOMO":"급등주(모멘텀)", "QUALITY":"우량주(안정형)", "UNKNOWN":"판단불가"}[mode]
    return {
        "mode": mode,
        "label": label,
        "confidence": int(conf),
        "short_quality": qflag,
        "metrics": {
            "ret_5": ret_5, "ret_20": ret_20, "vol_spike": vol_spike,
            "range_pct": range_pct, "short_pct": sp, "dtc": dtc,
            "float_m": (flt/1e6 if flt else None),
            "mktcap_b": (mkt_cap/1e9 if mkt_cap else None),
        }
    }

# ============================================================
# Score engine (중복 가산 억제: 축별 캡 + 디미니싱)
# ============================================================
class ScoreEngine:
    def __init__(self, caps: Dict[str, int]):
        self.caps = caps
        self.pos = {k: 0 for k in caps}
        self.neg = {k: 0 for k in caps}
        self.dir_count = {k: {"Bull": 0, "Bear": 0} for k in caps}
        self.notes = []

    def add(self, bucket: str, pts: int, direction: str = "Neutral", note: str = ""):
        if bucket not in self.caps:
            return
        if note:
            self.notes.append(note)

        mult = 1.0
        if direction in ("Bull", "Bear"):
            n = self.dir_count[bucket][direction]
            mult = 1.0 if n == 0 else (0.5 if n == 1 else 0.0)
            self.dir_count[bucket][direction] += 1

        pts = int(pts * mult)

        if pts >= 0:
            self.pos[bucket] = min(self.caps[bucket], self.pos[bucket] + pts)
        else:
            self.neg[bucket] = min(self.caps[bucket], self.neg[bucket] + abs(pts))

    def total(self, base=50) -> int:
        score = base + sum(self.pos.values()) - sum(self.neg.values())
        return int(max(0, min(100, score)))

# ============================================================
# Kelly
# ============================================================
def calculate_kelly(win_rate: float, reward_risk_ratio: float) -> float:
    p = win_rate / 100
    q = 1 - p
    b = reward_risk_ratio
    if b <= 0:
        return 0.0
    k = p - (q / b)
    safe_k = max(0.0, k * 0.5)
    return safe_k * 100

# ============================================================
# Main analysis (점수 중복 줄이고, short Unknown 안전 처리)
# ============================================================
def analyze_whale_mode(
    ticker: str,
    df: pd.DataFrame,
    info: Dict[str, Any],
    stock_basic: Dict[str, Any],
    monte_prob: float,
    macro_data: Dict[str, Any],
    data_time_utc,
    squeeze_min_on_days: int,
    mode_pack: Dict[str, Any]
) -> Dict[str, Any]:

    last = df.iloc[-1]
    close = float(last['Close'])
    atr = float(last['ATR'])
    volatility = float(last['Volatility'])
    mkt_cap = safe_int(info.get('marketCap', 0), 0)

    # time formatting
    try:
        if isinstance(data_time_utc, pd.Timestamp):
            if data_time_utc.tzinfo is None:
                utc_time = pytz.utc.localize(data_time_utc)
            else:
                utc_time = data_time_utc
            kst_time = utc_time.astimezone(pytz.timezone('Asia/Seoul'))
            last_date_str = kst_time.strftime('%m월 %d일 %H시 %M분')
        else:
            last_date_str = str(data_time_utc)
    except Exception:
        last_date_str = "시간 확인 불가"

    # Sentiment / Short / Squeeze state
    sent_data = get_sentiment_and_short_data(ticker, df, info, squeeze_min_on_days)
    sqz_state = sent_data['squeeze_state']

    # POC 계산(볼륨 프로파일)
    vp_window = df.iloc[-60:].copy()
    hist, bins = np.histogram(vp_window['Close'], bins=30, weights=vp_window['Volume'])
    poc_idx = int(np.argmax(hist)) if len(hist) else 0
    poc_price = float((bins[poc_idx] + bins[poc_idx+1]) / 2) if len(bins) > poc_idx+1 else close

    # AD signal
    ad_trend = float(df['AD_Line'].diff(20).iloc[-1])
    price_trend_val = float(df['Close'].diff(20).iloc[-1])
    ad_signal = "Neut"
    if price_trend_val < 0 and ad_trend > 0:
        ad_signal = "Bull"
    elif price_trend_val > 0 and ad_trend < 0:
        ad_signal = "Bear"

    # OBV/price rank -> whale gap
    recent_20 = df.iloc[-20:]
    price_rank = float((close - recent_20['Close'].min()) / (recent_20['Close'].max() - recent_20['Close'].min() + 1e-9) * 100)
    obv_rank = float((last['OBV'] - recent_20['OBV'].min()) / (recent_20['OBV'].max() - recent_20['OBV'].min() + 1e-9) * 100)
    whale_gap = obv_rank - price_rank

    # Mode-based caps (핵심: 모드별로 가중치 방향이 달라야 과신이 줄어듦)
    if mode_pack['mode'] == "QUALITY":
        caps = {"MACRO": 15, "FUND": 20, "TREND": 20, "MOMO": 10, "FLOW": 15, "VOL": 10, "SHORT": 5, "NEWS": 5}
    elif mode_pack['mode'] == "SQUEEZE":
        caps = {"MACRO": 10, "FUND": 5, "TREND": 15, "MOMO": 10, "FLOW": 15, "VOL": 15, "SHORT": 20, "NEWS": 10}
    elif mode_pack['mode'] == "MOMO":
        caps = {"MACRO": 10, "FUND": 5, "TREND": 20, "MOMO": 20, "FLOW": 15, "VOL": 15, "SHORT": 10, "NEWS": 5}
    else:
        caps = {"MACRO": 15, "FUND": 15, "TREND": 20, "MOMO": 15, "FLOW": 20, "VOL": 15, "SHORT": 15, "NEWS": 10}

    sc = ScoreEngine(caps)
    cards = []
    red_flags = 0

    # 0) macro
    sc.add("MACRO", macro_data.get('score_adj', 0), direction="Neutral",
           note=f"Macro status={macro_data.get('status')}")
    if macro_data['status'] == 'FEAR (위험)':
        cards.append({'title': '0. 시장 상황', 'stat': '공포(VIX↑)', 'desc': '변동성 주의', 'col': C_BEAR})
    elif macro_data['status'] == 'GREED (안정)':
        cards.append({'title': '0. 시장 상황', 'stat': '안정(VIX↓)', 'desc': '리스크 완화', 'col': C_BULL})
    else:
        cards.append({'title': '0. 시장 상황', 'stat': '보통', 'desc': '특이사항 없음', 'col': C_NEUT})

    # 1) fundamentals (QUALITY에서만 의미 있게 가중)
    per = stock_basic.get('per', None)
    roe = stock_basic.get('roe', None)
    if per is not None and roe is not None:
        if per < 25 and roe > 0.10:
            sc.add("FUND", +15, direction="Bull", note="Value+ROE good")
            cards.append({'title': '1. 펀더멘털', 'stat': '저평가/수익성', 'desc': f'PER {per:.1f}, ROE {roe*100:.1f}%', 'col': C_CYAN})
        elif roe > 0.15:
            sc.add("FUND", +10, direction="Bull", note="High ROE")
            cards.append({'title': '1. 펀더멘털', 'stat': '수익성 양호', 'desc': f'ROE {roe*100:.1f}%', 'col': C_BULL})
        elif per > 80:
            sc.add("FUND", -10, direction="Bear", note="Overvalued")
            red_flags += 1
            cards.append({'title': '1. 펀더멘털', 'stat': '고평가 주의', 'desc': f'PER {per:.1f}', 'col': C_WARN})
        else:
            cards.append({'title': '1. 펀더멘털', 'stat': '중립', 'desc': '특이사항 없음', 'col': C_NEUT})
    else:
        cards.append({'title': '1. 펀더멘털', 'stat': '정보 부족', 'desc': '지표 미확인', 'col': C_NEUT})

    # 2) flow: whale gap
    if whale_gap > 30:
        sc.add("FLOW", +15, direction="Bull", note="Whale accumulation strong")
        cards.append({'title': '2. 수급(OBV)', 'stat': '강력 매집', 'desc': 'OBV가 가격보다 강함', 'col': C_BULL})
    elif whale_gap > 10:
        sc.add("FLOW", +8, direction="Bull", note="Whale accumulation mild")
        cards.append({'title': '2. 수급(OBV)', 'stat': '매집 의심', 'desc': '수급 우위', 'col': C_CYAN})
    elif whale_gap < -10:
        sc.add("FLOW", -12, direction="Bear", note="Distribution risk")
        red_flags += 1
        cards.append({'title': '2. 수급(OBV)', 'stat': '이탈 징후', 'desc': '수급 약화', 'col': C_BEAR})
    else:
        cards.append({'title': '2. 수급(OBV)', 'stat': '중립', 'desc': '특이점 없음', 'col': C_NEUT})

    # 3) volatility bucket: squeeze state
    if sqz_state['on']:
        pts = 10 + (5 if sqz_state['trigger_on_today'] else 0)
        sc.add("VOL", +pts, direction="Bull", note=f"Squeeze ON streak={sqz_state['on_streak']}")
        cards.append({'title': '3. 변동성', 'stat': f"스퀴즈 ON({sqz_state['on_streak']})", 'desc': '압축 상태', 'col': C_PURP})
    else:
        cards.append({'title': '3. 변동성', 'stat': '일반', 'desc': '압축 약함', 'col': C_NEUT})

    # 4) divergence
    div_status = check_rsi_divergence(df)
    if div_status == "REG_BULL":
        sc.add("MOMO", +12, direction="Bull", note="RSI bullish divergence")
        cards.append({'title': '4. 다이버전스', 'stat': '상승 반전', 'desc': 'REG_BULL', 'col': C_BULL})
    elif div_status == "REG_BEAR":
        sc.add("MOMO", -12, direction="Bear", note="RSI bearish divergence")
        red_flags += 1
        cards.append({'title': '4. 다이버전스', 'stat': '하락 반전', 'desc': 'REG_BEAR', 'col': C_BEAR})
    else:
        cards.append({'title': '4. 다이버전스', 'stat': '없음', 'desc': '특이 신호 없음', 'col': C_NEUT})

    # 5) candle pattern
    pat = check_candle_pattern(df)
    if pat == "Hammer":
        sc.add("MOMO", +6, direction="Bull", note="Hammer")
        cards.append({'title': '5. 캔들 패턴', 'stat': 'Hammer', 'desc': '단기 반등 시사', 'col': C_WARN})
    elif pat == "Doji":
        cards.append({'title': '5. 캔들 패턴', 'stat': 'Doji', 'desc': '방향성 탐색', 'col': C_NEUT})
    else:
        cards.append({'title': '5. 캔들 패턴', 'stat': '일반', 'desc': '특이 패턴 없음', 'col': C_NEUT})

    # 6) Ichimoku / trend bucket (중복 가산 억제는 ScoreEngine가 담당)
    c_top = max(last['Senkou_Span_A'], last['Senkou_Span_B'])
    c_bot = min(last['Senkou_Span_A'], last['Senkou_Span_B'])
    if close > c_top:
        sc.add("TREND", +10, direction="Bull", note="Above cloud")
        cards.append({'title': '6. 일목균형표', 'stat': '구름대 위', 'desc': '상승 추세', 'col': C_CYAN})
    elif close < c_bot:
        sc.add("TREND", -10, direction="Bear", note="Below cloud")
        red_flags += 1
        cards.append({'title': '6. 일목균형표', 'stat': '구름대 아래', 'desc': '하방 압력', 'col': C_BEAR})
    else:
        cards.append({'title': '6. 일목균형표', 'stat': '구름대 안', 'desc': '혼조', 'col': C_NEUT})

    if close > float(last['MA20']):
        sc.add("TREND", +10, direction="Bull", note="Above MA20")
        cards.append({'title': '7. 추세 (MA)', 'stat': '단기 상승', 'desc': '20일선 위', 'col': C_BULL})
    else:
        sc.add("TREND", -12, direction="Bear", note="Below MA20")
        red_flags += 1
        cards.append({'title': '7. 추세 (MA)', 'stat': '단기 약세', 'desc': '20일선 아래', 'col': C_BEAR})

    # 8) Monte Carlo bucket (확률은 참고치, 과신 방지 위해 캡 작게)
    if monte_prob >= 50:
        sc.add("MOMO", +6, direction="Bull", note="MC win_prob high")
        cards.append({'title': '8. 시뮬레이션', 'stat': f'{monte_prob:.0f}%', 'desc': '상방 경로 우위', 'col': C_BULL})
    elif monte_prob <= 10:
        sc.add("MOMO", -6, direction="Bear", note="MC win_prob low")
        cards.append({'title': '8. 시뮬레이션', 'stat': f'{monte_prob:.0f}%', 'desc': '상방 경로 빈약', 'col': C_BEAR})
    else:
        cards.append({'title': '8. 시뮬레이션', 'stat': f'{monte_prob:.0f}%', 'desc': '중립', 'col': C_NEUT})

    # 9) Short bucket (신뢰도 OK일 때만 반영)
    sp = sent_data['short_pct']
    dtc = sent_data['days_to_cover']
    qflag = sent_data['short_data_quality']
    if qflag == "OK" and sp is not None and dtc is not None:
        if sent_data['short_signal'] == "Squeeze Setup":
            sc.add("SHORT", +15, direction="Bull", note="Short squeeze setup")
            cards.append({'title': '9. 공매도(Short)', 'stat': f'{sp:.1f}%', 'desc': f'DTC {dtc:.1f} (setup)', 'col': C_PURP})
        elif sent_data['short_signal'] == "High Short":
            sc.add("SHORT", -8, direction="Bear", note="High short (pressure)")
            cards.append({'title': '9. 공매도(Short)', 'stat': f'{sp:.1f}%', 'desc': f'DTC {dtc:.1f} (high short)', 'col': C_BEAR})
        else:
            cards.append({'title': '9. 공매도(Short)', 'stat': f'{sp:.1f}%', 'desc': f'DTC {dtc:.1f} (normal)', 'col': C_NEUT})
    else:
        # ✅ 핵심: Unknown이면 점수 가산/감산하지 않음 + 경고 카드
        red_flags += 1  # 신뢰도 리스크는 리스크로 본다(과신 방지)
        cards.append({'title': '9. 공매도(Short)', 'stat': 'Unknown', 'desc': '데이터 신뢰 낮음(가산/감산 제외)', 'col': C_WARN})

    # 10) News bucket (과대평가 방지 위해 캡 낮게)
    if sent_data['news_signal'] == "Positive":
        sc.add("NEWS", +5, direction="Bull", note="News positive")
        cards.append({'title': '10. 뉴스 심리', 'stat': '긍정', 'desc': '호재성 키워드', 'col': C_BULL})
    elif sent_data['news_signal'] == "Negative":
        sc.add("NEWS", -5, direction="Bear", note="News negative")
        red_flags += 1
        cards.append({'title': '10. 뉴스 심리', 'stat': '부정', 'desc': '악재성 키워드', 'col': C_BEAR})
    else:
        cards.append({'title': '10. 뉴스 심리', 'stat': '중립', 'desc': '특이점 없음', 'col': C_NEUT})

    # 추가 flow signals: AD / POC / MFI
    # AD
    if ad_signal == "Bull":
        sc.add("FLOW", +8, direction="Bull", note="A/D bullish")
    elif ad_signal == "Bear":
        sc.add("FLOW", -8, direction="Bear", note="A/D bearish")
        red_flags += 1

    # POC
    if close > poc_price * 1.02:
        sc.add("FLOW", +6, direction="Bull", note="Above POC")
        poc_signal = "Bull"
    elif close < poc_price * 0.98:
        sc.add("FLOW", -6, direction="Bear", note="Below POC")
        red_flags += 1
        poc_signal = "Bear"
    else:
        poc_signal = "Supp"

    # MFI
    mfi_val = float(last['MFI'])
    if mfi_val < 20:
        sc.add("FLOW", +5, direction="Bull", note="MFI oversold")
        mfi_signal = "Oversold"
    elif mfi_val > 80:
        sc.add("FLOW", -5, direction="Bear", note="MFI overbought")
        mfi_signal = "Overbot"
    else:
        mfi_signal = "Neut"

    # Score & mode theme
    score = sc.total(base=50)

    # risk cap: red_flags가 많으면 상단을 제한(과신 방지)
    if red_flags >= 2:
        score = min(score, 70)
    if red_flags >= 4:
        score = min(score, 60)

    # mode decision
    if mkt_cap < 10_000_000_000 or volatility > 3.0:
        mode_txt, theme_col = "🦄 야수 (고위험)", C_PURP
        stop_mult, target_mult = 2.5, 5.0
    else:
        mode_txt, theme_col = "🛡️ 우량 (안전형)", C_CYAN
        stop_mult, target_mult = 2.0, 3.0

    stop = close - (atr * stop_mult)
    target = close + (atr * target_mult)

    # title
    if score >= 80:
        t, c = "강력 매수", C_BULL
    elif score >= 60:
        t, c = ("주의 (혼조세)" if red_flags > 0 else "매수"), (C_WARN if red_flags > 0 else C_CYAN)
    elif score <= 30:
        t, c = "매도 / 관망", C_BEAR
    else:
        t, c = "관망 / 중립", C_NEUT

    vol_avg = df['Volume'].rolling(20).mean().iloc[-1]
    vol_ratio = (float(last['Volume']) / float(vol_avg) * 100) if vol_avg and vol_avg > 0 else 0

    rr = (target - close) / (close - stop) if close > stop else 1.0
    kelly = calculate_kelly(monte_prob, rr)

    return {
        'mode': mode_txt,
        'theme': theme_col,
        'score': score,
        'title': t,
        'color': c,
        'cards': cards,
        'tech_signals': get_18_tech_signals(df, squeeze_min_on_days),
        'stop': stop,
        'target': target,
        'close': close,
        'kelly': kelly,
        'vol_data': {'last': float(last['Volume']), 'avg': float(vol_avg), 'ratio': vol_ratio},
        'adv_features': {
            'whale_gap': whale_gap,
            'ad_signal': ad_signal,
            'poc_signal': poc_signal,
            'mfi_signal': mfi_signal,
            'poc_price': poc_price
        },
        'monte_prob': monte_prob,
        'entry_date': last_date_str,
        'sent_data': sent_data,
        'mode_pack': mode_pack,
        'red_flags': red_flags
    }

# ============================================================
# Action Strategy HTML (기존 유지 + 표현만 안전)
# ============================================================
def get_action_strategy_html(ticker: str, analysis: Dict[str, Any], monte_res):
    score = analysis['score']
    win_prob = monte_res[4] if monte_res else 0
    peak_yield = monte_res[6] if monte_res else 0
    min_yield = monte_res[8] if monte_res else 0
    kelly = analysis['kelly']

    downside = abs(min_yield) if min_yield < 0 else 1.0
    if downside == 0:
        downside = 1.0
    rr_ratio = peak_yield / downside if downside else 0

    whale_gap = analysis['adv_features']['whale_gap']
    sqz = analysis['sent_data'].get('squeeze_state', {})
    is_squeeze = bool(sqz.get('on', False))

    decision = "HOLD"
    reason = "판단 보류"
    color = "#aaa"

    # 과신 방지: short 데이터 Unknown이면 squeeze를 판단 근거에 넣지 않음
    short_quality = analysis['sent_data'].get('short_data_quality', 'MISSING')

    if score < 60:
        decision = "DROP (관심 삭제)"
        reason = "점수가 60점 미만입니다. 신호 대비 리스크가 큽니다."
        color = C_BEAR
    elif win_prob < 50:
        decision = "WAIT (관망)"
        reason = "시뮬레이션 상 상방 경로 우위가 명확하지 않습니다."
        color = C_WARN
    elif rr_ratio < 2.0:
        decision = "WAIT (관망)"
        reason = f"손익비가 {rr_ratio:.1f}배로 낮습니다. (목표수익 대비 리스크 큼)"
        color = C_WARN
    else:
        if whale_gap > 10 or (is_squeeze and short_quality == "OK"):
            decision = "BUY (진입 검토)"
            reason = "점수/손익비 조건 충족 + 수급/압축 신호가 유의미합니다."
            color = C_BULL
        else:
            decision = "WATCH (타이밍 대기)"
            reason = "조건은 괜찮지만 결정적 트리거(수급/전환)가 부족합니다."
            color = C_CYAN

    html = f"""
    <div style="background:#1E1E1E; border:1px solid #333; border-radius:8px; padding:15px; margin-bottom:10px;">
        <div style="font-size:1.6rem; font-weight:900; color:{color}; white-space:nowrap;">{decision}</div>
        <div style="background:#252525; padding:8px 12px; margin-top:5px; border-radius:6px; font-size:0.9rem; color:#ccc;">
            <b>💡 판단 근거:</b> {reason}
        </div>
        <div style="display:flex; justify-content:space-between; margin-top:10px;">
            <div style="text-align:center;">
                <div style="font-size:0.75rem; color:#999;">진입 비중</div>
                <div style="font-size:1rem; font-weight:700; color:{C_CYAN};">{kelly:.1f}%</div>
            </div>
             <div style="text-align:center;">
                <div style="font-size:0.75rem; color:#999;">손익비 (R/R)</div>
                <div style="font-size:1rem; font-weight:700; color:{C_WARN if rr_ratio < 2 else C_BULL};">{rr_ratio:.1f}배</div>
            </div>
             <div style="text-align:center;">
                <div style="font-size:0.75rem; color:#999;">손절가</div>
                <div style="font-size:1rem; font-weight:700; color:{C_BEAR};">${analysis['stop']:.2f}</div>
            </div>
        </div>
    </div>
    """
    return html

# ============================================================
# Main
# ============================================================
def main():
    with st.sidebar:
        st.header("🔍 종목 검색")
        input_ticker = st.text_input("Ticker", value="").upper().strip()

        st.markdown("---")
        include_extended = st.toggle("프리/애프터 포함(prepost)", value=True)
        squeeze_min_on_days = st.slider("Squeeze 연속 ON 최소 일수", 2, 10, 5, 1)

        st.markdown("---")
        sim_days = st.slider("몬테카를로 기간(일)", 30, 200, DEFAULT_SIM_DAYS, 10)
        sim_n = st.selectbox("몬테카를로 시뮬 수", [2000, 5000, 10000], index=1)

        run_btn = st.button("AI 분석 실행")

        st.markdown("---")
        st.caption("⚠️ 공매도/short 데이터는 소스 지연/누락이 잦아 Unknown일 수 있습니다. Unknown이면 점수에 반영하지 않습니다.")

        if run_btn:
            st.rerun()

    if not input_ticker:
        st.info("좌측 사이드바에 종목코드(예: NVDA)를 입력하고 실행 버튼을 누르세요.")
        return

    with st.spinner(f"📡 {input_ticker} 데이터 정밀 분석 중..."):
        # 원본 info(분류/short/float 등에 필요)
        try:
            t = yf.Ticker(input_ticker)
            info = t.info or {}
        except Exception:
            info = {}

        stock_basic = get_stock_info_basic(input_ticker)
        df, data_time_utc = get_realtime_synced_data(input_ticker, include_extended=include_extended)
        macro_data = get_market_macro()

        if df is None or df.empty:
            st.error("데이터를 불러올 수 없습니다.")
            return

        # 모드 자동 분류
        mode_pack = classify_mode(input_ticker, df, info)

        # Monte Carlo(수정된 GBM)
        monte_res = run_monte_carlo(df, num_simulations=sim_n, days=sim_days)
        win_prob = monte_res[4] if monte_res else 0

        analysis = analyze_whale_mode(
            input_ticker, df, info, stock_basic, win_prob, macro_data, data_time_utc,
            squeeze_min_on_days=squeeze_min_on_days,
            mode_pack=mode_pack
        )

        insider_data = get_insider_trading(input_ticker)
        filings_df = get_sec_filings(input_ticker)

    # ============================================================
    # Header
    # ============================================================
    st.markdown(
        f"<h1 style='color:white;'>{input_ticker} "
        f"<span style='font-size:0.5em; color:#888;'>{stock_basic['name']}</span></h1>",
        unsafe_allow_html=True
    )

    # Mode line (자동 분류 결과)
    short_quality = mode_pack.get("short_quality", "MISSING")
    short_warn = ""
    if mode_pack["mode"] == "SQUEEZE" and short_quality != "OK":
        short_warn = " <span style='color:#FF5252;'>[Short data unreliable → SQUEEZE 판단 신뢰 낮음]</span>"

    st.markdown(
        f"<div style='color:#ddd; font-size:1.1rem;'>"
        f"분류: <b>{mode_pack['label']}</b> "
        f"<span style='color:#888'>(confidence {mode_pack['confidence']}/100)</span>"
        f"{short_warn}"
        f"</div>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.caption(f"기준: {analysis['entry_date']} | Mode: {analysis['mode']} | ExtendedHours: {'ON' if include_extended else 'OFF'}")
    with col2:
        st.markdown(f"""
            <div style='text-align:right;'>
                <span style='font-size:1rem; color:#888;'>AI Score</span>
                <span style='font-size:2rem; font-weight:bold; color:{analysis['color']};'>{analysis['score']}</span>
                <br>
                <span style='font-size:0.8rem; color:#888;'>Monte Win</span>
                <span style='font-size:1.2rem; font-weight:bold; color:{C_BULL if win_prob >= 50 else '#666'};'>{win_prob:.0f}%</span>
            </div>
        """, unsafe_allow_html=True)

    # ============================================================
    # Tabs
    # ============================================================
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 대시보드", "📑 핵심 요인", "🎛 기술지표(18)", "🎲 시뮬레이션", "👥 내부자 거래", "📢 공시/서류"
    ])

    with tab1:
        st.markdown(get_action_strategy_html(input_ticker, analysis, monte_res), unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("현재가", f"${analysis['close']:.2f}")
        c2.metric("목표가(ATR 기반)", f"${analysis['target']:.2f}")
        c3.metric("Win Rate(+30% 도달)", f"{win_prob:.1f}%")

        st.markdown("---")
        st.subheader("🧭 핵심 리스크/주의")
        rf = analysis.get("red_flags", 0)
        if rf >= 4:
            st.error("리스크 플래그가 다수입니다. 신호 대비 불확실성이 큽니다(과신 금지).")
        elif rf >= 2:
            st.warning("리스크 플래그가 있습니다. 신호 해석을 보수적으로 하세요.")
        else:
            st.info("치명적 플래그는 크지 않습니다. 다만 단일 지표로 결론내지 마세요.")

        st.markdown("---")
        st.subheader("📰 AI 뉴스 감지 (한국어 번역)")
        for news in analysis['sent_data']['headlines']:
            st.markdown(f"- [{news['title']}]({news['url']})")

    with tab2:
        st.markdown("### 🧬 AI 진단 카드")
        for card in analysis['cards']:
            st.markdown(f"""
            <div style="background:#262626; padding:10px; margin-bottom:8px; border-radius:5px; border-left: 4px solid {card['col']}; display:flex; justify-content:space-between; align-items:center;">
                <div style="color:#ddd; font-weight:bold;">{card['title']}</div>
                <div style="text-align:right;">
                    <div style="color:{card['col']}; font-weight:bold;">{card['stat']}</div>
                    <div style="font-size:0.8em; color:#888;">{card['desc']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🔎 분류 근거(요약)")
        m = mode_pack["metrics"]
        st.write({
            "ret_5(%)": round(m["ret_5"], 2),
            "ret_20(%)": round(m["ret_20"], 2),
            "vol_spike(x)": round(m["vol_spike"], 2),
            "range_pct(%)": round(m["range_pct"], 2),
            "short_pct(%)": (None if m["short_pct"] is None else round(m["short_pct"], 2)),
            "DTC(days)": (None if m["dtc"] is None else round(m["dtc"], 2)),
            "float(M)": m["float_m"],
            "mktcap(B)": m["mktcap_b"]
        })

    with tab3:
        st.markdown("### 🎛 18개 기술적 지표 (중복 과신 방지: Squeeze는 연속상태/트리거 반영)")
        signals = analysis['tech_signals']
        t_col1, t_col2 = st.columns(2)
        mid = (len(signals) + 1) // 2

        with t_col1:
            for name, val, bias in signals[:mid]:
                color = C_BULL if bias == "Bull" else (C_BEAR if bias == "Bear" else "#888")
                st.markdown(
                    f"<div style='display:flex; justify-content:space-between; border-bottom:1px solid #333; padding:5px;'>"
                    f"<span style='color:#ccc;'>{name}</span>"
                    f"<span style='color:{color}; font-weight:bold;'>{val}</span></div>",
                    unsafe_allow_html=True
                )
        with t_col2:
            for name, val, bias in signals[mid:]:
                color = C_BULL if bias == "Bull" else (C_BEAR if bias == "Bear" else "#888")
                st.markdown(
                    f"<div style='display:flex; justify-content:space-between; border-bottom:1px solid #333; padding:5px;'>"
                    f"<span style='color:#ccc;'>{name}</span>"
                    f"<span style='color:{color}; font-weight:bold;'>{val}</span></div>",
                    unsafe_allow_html=True
                )

    with tab4:
        st.markdown("### 🎲 120일(설정값) 미래 시뮬레이션 (GBM/lognormal)")
        if monte_res and monte_res[0] is not None:
            peak_yield = monte_res[6]
            min_yield = monte_res[8]

            sc1, sc2 = st.columns(2)
            sc1.metric("예상 최고 수익(중앙값)", f"+{peak_yield:.1f}%")
            sc2.metric("최악의 하락폭(10%ile)", f"{min_yield:.1f}%")
            st.metric("승률 (+30% 목표)", f"{win_prob:.1f}%")

            st.markdown("#### 목표별 도달 확률")
            st.table(pd.DataFrame(monte_res[9]).set_index('pct'))
        else:
            st.warning("시뮬레이션 결과를 생성하지 못했습니다(데이터 부족).")

    with tab5:
        st.markdown("### 👥 최근 내부자 거래 (Insider Trading)")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("내부자 보유율", f"{stock_basic.get('insider_own', 0)*100:.2f}%")
        with c2:
            st.info("내부자 데이터는 종목/기간에 따라 공백이 많습니다. '없음'은 '없다'가 아니라 '잡히지 않았다'일 수 있습니다.")

        if insider_data is not None and not insider_data.empty:
            def highlight_insider(row):
                text = str(row.get('Text', '')).lower()
                if 'sale' in text:
                    return ['background-color: #4a1414'] * len(row)
                elif 'purchase' in text or 'grant' in text:
                    return ['background-color: #144a20'] * len(row)
                else:
                    return [''] * len(row)

            st.dataframe(
                insider_data.style.apply(highlight_insider, axis=1).format({
                    "Shares": "{:,}",
                    "Value": "${:,.0f}"
                }),
                use_container_width=True
            )
        else:
            st.warning("최근 내부자 거래 내역이 없거나 데이터를 불러올 수 없습니다.")

    with tab6:
        st.markdown("### 📢 SEC 기업 공시 (Filings)")
        st.info("10-K/10-Q/8-K 등 주요 공시 링크(가능한 범위 내). yfinance 소스 특성상 누락될 수 있습니다.")
        if filings_df is not None and not filings_df.empty:
            st.dataframe(
                filings_df,
                use_container_width=True,
                column_config={
                    "Link": st.column_config.LinkColumn(
                        "원문 보기",
                        help="클릭시 SEC Edgar 사이트로 이동합니다",
                        validate="^https://.*",
                        display_text="View Report"
                    ),
                    "Date": st.column_config.DateColumn("공시 날짜", format="YYYY-MM-DD")
                }
            )
        else:
            st.warning("최근 공시 내역을 불러올 수 없습니다.")

if __name__ == "__main__":
    main()
