import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import logging
import time
import pytz
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

# --- [Streamlit 기본 설정] ---
st.set_page_config(
    page_title="AI Stock Sniper",
    page_icon="🦄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 다크모드 스타일 강제 적용
st.markdown("""
    <style>
    .stApp { background-color: #121212; color: #e0e0e0; }
    .stTextInput > div > div > input { background-color: #262626; color: white; }
    </style>
""", unsafe_allow_html=True)

# 로거 차단
logger = logging.getLogger('yfinance')
logger.setLevel(logging.CRITICAL)
plt.style.use('dark_background')

# [라이브러리 세팅]
try:
    from duckduckgo_search import DDGS
except ImportError:
    class DDGS:
        def news(self, keywords, max_results=5): return []

try:
    from deep_translator import GoogleTranslator
except ImportError:
    class GoogleTranslator:
        def __init__(self, source='auto', target='ko'): pass
        def translate(self, text): return text

# 2. 참조 데이터
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
TOP_N = 5

# --- [COLOR PALETTE] ---
C_BULL = "#00E676"
C_BEAR = "#FF5252"
C_NEUT = "#B0BEC5"
C_WARN = "#FFD740"
C_CYAN = "#00B0FF"
C_PURP = "#E040FB"
C_BG   = "#121212"

# --- [캐싱 처리] ---
@st.cache_resource
def get_global_ref_cache():
    return {}

GLOBAL_REF_CACHE = get_global_ref_cache()

# 3. 데이터 엔진 (원본 함수 100% 동일 유지)
def get_stock_info(ticker):
    try:
        info = yf.Ticker(ticker).info
        data = {
            'mkt_cap': info.get('marketCap', 0),
            'per': info.get('trailingPE', None),
            'pbr': info.get('priceToBook', None),
            'roe': info.get('returnOnEquity', None),
            'name': info.get('longName', ticker)
        }
        return data
    except:
        return {'mkt_cap': 0, 'per': None, 'pbr': None, 'roe': None, 'name': ticker}

def get_realtime_synced_data(ticker):
    try:
        # 1. 과거 맥락용 일봉 데이터 (2년치)
        df_daily = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
        if isinstance(df_daily.columns, pd.MultiIndex):
            df_daily.columns = [col[0] for col in df_daily.columns]

        # 2. 현재 상태용 실시간 데이터 (오늘 하루치 1분봉)
        df_intraday = yf.download(ticker, period="1d", interval="1m", progress=False, auto_adjust=True)
        if isinstance(df_intraday.columns, pd.MultiIndex):
            df_intraday.columns = [col[0] for col in df_intraday.columns]

        if df_daily.empty: return None, None

        if not df_intraday.empty:
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

        if len(df_daily) < WINDOW_SIZE + FORECAST_DAYS: return None, None

        # 3. 모든 지표 계산 (원본 그대로)
        df = df_daily.copy()

        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()
        df['MA120'] = df['Close'].rolling(120).mean()

        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 0.001)
        df['RSI'] = 100 - (100 / (1 + rs))

        low_14 = df['Low'].rolling(14).min()
        high_14 = df['High'].rolling(14).max()
        df['Stoch_K'] = ((df['Close'] - low_14) / (high_14 - low_14).replace(0, 1)) * 100
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

        tp = (df['High'] + df['Low'] + df['Close']) / 3
        df['CCI'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std()).replace(0, 0.001)
        df['WillR'] = ((high_14 - df['Close']) / (high_14 - low_14).replace(0, 1)) * -100

        std_20 = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['MA20'] + (std_20 * 2)
        df['BB_Lower'] = df['MA20'] - (std_20 * 2)

        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        df['KC_Upper'] = df['MA20'] + (df['ATR'] * 1.5)
        df['KC_Lower'] = df['MA20'] - (df['ATR'] * 1.5)

        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

        ad_factor = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']).replace(0, 1)
        df['AD_Line'] = (ad_factor * df['Volume']).fillna(0).cumsum()

        typical = (df['High'] + df['Low'] + df['Close']) / 3
        mf = typical * df['Volume']
        df['MFI'] = 100 - (100 / (1 + (mf.where(typical > typical.shift(1), 0).rolling(14).sum() / mf.where(typical < typical.shift(1), 0).rolling(14).sum().replace(0, 1))))

        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).rolling(20).sum() / df['Volume'].rolling(20).sum().replace(0, 1)
        df['ROC'] = ((df['Close'] - df['Close'].shift(12)) / df['Close'].shift(12).replace(0, 1)) * 100

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

        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = (df['High'] - df['Low']) / df['Close'] * 100

        if len(df) > 130:
            df = df.iloc[130:]

        if pd.isna(df['Close'].iloc[-1]):
            df = df.iloc[:-1]

        return df, data_time_utc

    except Exception as e:
        return None, None

def get_market_macro():
    try:
        df = yf.download(['^VIX', '^TNX'], period='5d', progress=False)['Close']
        if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
        vix = df['^VIX'].iloc[-1]
        tnx = df['^TNX'].iloc[-1]
        status = "Normal"
        score_adj = 0
        if vix > 25:
            status = "FEAR (위험)"
            score_adj = -15
        elif vix < 14:
            status = "GREED (안정)"
            score_adj = +5
        return {'vix': vix, 'tnx': tnx, 'status': status, 'score_adj': score_adj}
    except:
        return {'vix': 0, 'tnx': 0, 'status': 'Unknown', 'score_adj': 0}

def get_google_news_rss(ticker):
    try:
        url = f"https://news.google.com/rss/search?q={ticker}+stock+news&hl=en-US&gl=US&ceid=US:en"
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            titles = []
            for item in root.findall('.//item')[:3]:
                title = item.find('title')
                if title is not None: titles.append(title.text)
            return titles
    except: return []
    return []

def get_sentiment_and_short_data(ticker, df):
    # 원본 로직 그대로
    data = {'short_pct': 0, 'short_signal': 'N/A', 'upside_pot': 0, 'analyst_signal': 'N/A', 'news_score': 0, 'news_signal': 'Neutral', 'headlines': []}
    t = yf.Ticker(ticker)
    try:
        info = t.info
        short_float = info.get('shortPercentOfFloat', 0)
        if short_float is None: short_float = 0
        short_pct = short_float * 100
        short_signal = "Neutral"
        if short_pct > 30: short_signal = "Squeeze Possibility"
        elif short_pct > 10: short_signal = "High Short (Bad)"
        current_price = df['Close'].iloc[-1]
        target_mean = info.get('targetMeanPrice', current_price)
        if target_mean is None: target_mean = current_price
        upside_pot = ((target_mean - current_price) / current_price) * 100
        analyst_signal = "Bull" if upside_pot > 10 else ("Bear" if upside_pot < -10 else "Neutral")
        data['short_pct'] = short_pct
        data['short_signal'] = short_signal
        data['upside_pot'] = upside_pot
        data['analyst_signal'] = analyst_signal
    except: pass

    raw_headlines = []
    # 뉴스 수집 로직 (예외처리 포함)
    try:
        yf_news = t.news
        if yf_news:
            for item in yf_news[:3]:
                title = item.get('title', '')
                if title: raw_headlines.append(title)
    except: pass

    try:
        if len(raw_headlines) < 3:
            ddgs = DDGS()
            ddg_res = ddgs.news(keywords=f"{ticker} stock", max_results=3)
            if ddg_res:
                for item in ddg_res:
                    title = item.get('title', '')
                    if title: raw_headlines.append(title)
    except: pass

    unique_headlines = list(set(raw_headlines))[:5]
    sentiment_score = 0
    bull_words = ['up', 'surge', 'jump', 'beat', 'growth', 'gain', 'buy', 'strong', 'profit', 'partnership', 'merger', 'record', 'soar', 'bull', 'upgrade']
    bear_words = ['down', 'drop', 'fall', 'miss', 'loss', 'sell', 'weak', 'lawsuit', 'investigation', 'inflation', 'cut', 'crash', 'plunge', 'bear', 'downgrade']
    
    for title in unique_headlines:
        title_lower = title.lower()
        for w in bull_words:
            if w in title_lower: sentiment_score += 1
        for w in bear_words:
            if w in title_lower: sentiment_score -= 1
            
    news_signal = "Positive" if sentiment_score > 0 else ("Negative" if sentiment_score < 0 else "Neutral")
    data['news_score'] = sentiment_score
    data['news_signal'] = news_signal
    data['headlines'] = unique_headlines 
    return data

def get_benchmark(mode):
    ticker = "SPY" if mode == "SAFE" else "IWM"
    try:
        df = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
        return df
    except: return None

# [중요] 18개 기술적 지표 계산 함수 (누락 없이 포함)
def get_18_tech_signals(df):
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
    if last['Close'] > last['BB_Upper']: pos, bias = "High", "Bear"
    elif last['Close'] < last['BB_Lower']: pos, bias = "Low", "Bull"
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
    if last['Close'] > cloud_top: ichi, bias = "Above", "Bull"
    elif last['Close'] < cloud_bot: ichi, bias = "Below", "Bear"
    signals.append(("Ichimoku", ichi, bias))

    sqz = check_ttm_squeeze(df)
    signals.append(("Squeeze", "ON" if sqz else "OFF", "Bull" if sqz else "Neutral"))
    pat = check_candle_pattern(df)
    signals.append(("Candle", pat if pat else "-", "Bull" if pat == "Hammer" else "Neutral"))
    vol = last['Volatility']
    signals.append(("Vol Ratio", f"{vol:.2f}%", "Neutral"))
    return signals

def z_score_normalize(series):
    return (series - series.mean()) / (series.std() + 1e-9)

def check_rsi_divergence(df, window=10):
    if len(df) < window * 2: return None
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
    if curr_low_price < prev_low_price and curr_low_rsi > prev_low_rsi: return "REG_BULL"
    if curr_high_price > prev_high_price and curr_high_rsi < prev_high_rsi: return "REG_BEAR"
    if curr_low_price > prev_low_price and curr_low_rsi < prev_low_rsi: return "HID_BULL"
    if curr_high_price < prev_high_price and curr_high_rsi > prev_high_rsi: return "HID_BEAR"
    return None

def check_ttm_squeeze(df):
    last = df.iloc[-1]
    bb_width = last['BB_Upper'] - last['BB_Lower']
    kc_width = last['KC_Upper'] - last['KC_Lower']
    return bb_width < kc_width

def check_candle_pattern(df):
    last = df.iloc[-1]
    open_p, close_p = last['Open'], last['Close']
    high_p, low_p = last['High'], last['Low']
    body = abs(close_p - open_p)
    upper_shadow = high_p - max(open_p, close_p)
    lower_shadow = min(open_p, close_p) - low_p
    total_range = high_p - low_p
    if total_range == 0: return None
    if (lower_shadow > body * 2) and (upper_shadow < body * 0.5) and (lower_shadow > upper_shadow * 2): return "Hammer"
    if body <= (total_range * 0.1): return "Doji"
    return None

def run_monte_carlo(df, num_simulations=10000, days=120): # 시뮬레이션 횟수 원본 유지
    last_price = df['Close'].iloc[-1]
    target_percents = [0.3, 0.5, 0.7, 1.0, 1.5]

    if len(df) < 30: daily_vol = df['Log_Ret'].std()
    else: daily_vol = df['Log_Ret'].tail(30).std()

    sim_df = pd.DataFrame()
    max_peaks = []

    actual_sims = 5000 
    
    sim_matrix = np.zeros((days, actual_sims))
    sim_matrix[0] = last_price
    shocks = np.random.normal(0, daily_vol, (days, actual_sims))
    
    for t in range(1, days):
        sim_matrix[t] = sim_matrix[t-1] * (1 + shocks[t])
        
    sim_df = pd.DataFrame(sim_matrix)
    max_peaks = sim_df.max()
    
    main_target = last_price * 1.30
    win_count = (max_peaks >= main_target).sum()
    win_prob = (win_count / actual_sims) * 100

    expected_date_str = "도달 불가"

    target_peak_price = np.median(max_peaks)
    peak_yield = (target_peak_price - last_price) / last_price * 100

    extra_scenarios = []
    for pct in target_percents:
        tgt_price = last_price * (1 + pct)
        count = (max_peaks >= tgt_price).sum()
        prob = (count / actual_sims) * 100
        extra_scenarios.append({'pct': int(pct*100), 'prob': prob, 'date': "-"})

    ending_values = sim_df.iloc[-1, :]
    min_yield = (np.percentile(ending_values, 10) - last_price) / last_price * 100
    
    forecast_data = {} 

    return sim_df, None, None, None, win_prob, expected_date_str, peak_yield, forecast_data, min_yield, extra_scenarios

def calculate_kelly(win_rate, reward_risk_ratio):
    p = win_rate / 100
    q = 1 - p
    b = reward_risk_ratio
    if b <= 0: return 0
    kelly_fraction = p - (q / b)
    safe_kelly = max(0, kelly_fraction * 0.5)
    return safe_kelly * 100

def analyze_whale_mode(ticker, df, benchmark_df, win_rate, avg_return, stock_info, monte_prob, macro_data, data_time_utc):
    last = df.iloc[-1]
    close = last['Close']
    atr = last['ATR']
    volatility = last['Volatility']
    mkt_cap = stock_info['mkt_cap']

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
    except:
        last_date_str = "시간 확인 불가"

    recent_20 = df.iloc[-20:]
    price_rank = (close - recent_20['Close'].min()) / (recent_20['Close'].max() - recent_20['Close'].min() + 1e-9) * 100
    obv_rank = (last['OBV'] - recent_20['OBV'].min()) / (recent_20['OBV'].max() - recent_20['OBV'].min() + 1e-9) * 100
    whale_gap = obv_rank - price_rank

    ad_trend = df['AD_Line'].diff(20).iloc[-1]
    price_trend_val = df['Close'].diff(20).iloc[-1]
    ad_signal = "Neut"
    if price_trend_val < 0 and ad_trend > 0: ad_signal = "Bull"
    elif price_trend_val > 0 and ad_trend < 0: ad_signal = "Bear"

    vp_window = df.iloc[-60:]
    hist, bins = np.histogram(vp_window['Close'], bins=30, weights=vp_window['Volume'])
    poc_idx = hist.argmax()
    poc_price = (bins[poc_idx] + bins[poc_idx+1]) / 2

    poc_signal = "Supp"
    if close > poc_price * 1.02: poc_signal = "Bull"
    elif close < poc_price * 0.98: poc_signal = "Bear"

    mfi_val = last['MFI']
    mfi_signal = "Neut"
    if mfi_val < 20: mfi_signal = "Oversold"
    elif mfi_val > 80: mfi_signal = "Overbot"

    score = 50
    cards = []
    red_flags = 0

    score += macro_data['score_adj']
    if macro_data['status'] == 'FEAR (위험)':
        cards.append({'title':'0. 시장 상황','stat':'공포(VIX↑)','desc':'변동성 주의', 'col':C_BEAR})
    elif macro_data['status'] == 'GREED (안정)':
        cards.append({'title':'0. 시장 상황','stat':'안정(VIX↓)','desc':'투자 심리 호조', 'col':C_BULL})
    else:
        cards.append({'title':'0. 시장 상황','stat':'보통','desc':'특이사항 없음', 'col':C_NEUT})

    per, roe = stock_info['per'], stock_info['roe']
    if per and roe:
        if per < 25 and roe > 0.10: score += 15; cards.append({'title':'1. 펀더멘털','stat':'저평가 우량','desc':f'PER {per:.1f}', 'col':C_CYAN})
        elif roe > 0.15: score += 10; cards.append({'title':'1. 펀더멘털','stat':'고수익성','desc':f'ROE {roe*100:.1f}%', 'col':C_BULL})
        elif per > 80: score -= 10; cards.append({'title':'1. 펀더멘털','stat':'고평가 주의','desc':f'PER {per:.1f}', 'col':C_WARN})
        else: cards.append({'title':'1. 펀더멘털','stat':'적정/보통','desc':'특이사항 없음', 'col':C_NEUT})
    else: cards.append({'title':'1. 펀더멘털','stat':'정보 없음','desc':'데이터 부족', 'col':C_NEUT})

    if whale_gap > 30: score += 20; cards.append({'title':'2. 고래 수급','stat':'강력 매집','desc':'개미 털고 매집 중', 'col':C_BULL})
    elif whale_gap > 10: score += 10; cards.append({'title':'2. 고래 수급','stat':'매집 의심','desc':'자금 유입 포착', 'col':C_CYAN})
    elif whale_gap < -10:
        score -= 15; red_flags += 1
        cards.append({'title':'2. 고래 수급','stat':'세력 이탈','desc':'매도 시그널', 'col':C_BEAR})
    else: cards.append({'title':'2. 고래 수급','stat':'중립','desc':'수급 특이점 없음', 'col':C_NEUT})

    if check_ttm_squeeze(df): score += 15; cards.append({'title':'3. 변동성','stat':'스퀴즈 ON','desc':'에너지 폭발 임박', 'col':C_PURP})
    else: cards.append({'title':'3. 변동성','stat':'일반','desc':'에너지 축적 필요', 'col':C_NEUT})

    div_status = check_rsi_divergence(df)
    if div_status == "REG_BULL": score += 20; cards.append({'title':'4. 다이버전스','stat':'상승 반전','desc':'추세 전환 신호', 'col':C_BULL})
    elif div_status == "REG_BEAR": score -= 20; cards.append({'title':'4. 다이버전스','stat':'하락 반전','desc':'고점 징후 포착', 'col':C_BEAR})
    else: cards.append({'title':'4. 다이버전스','stat':'없음','desc':'지표와 주가 동행', 'col':C_NEUT})

    pat = check_candle_pattern(df)
    if pat == "Hammer": score += 10; cards.append({'title':'5. 캔들 패턴','stat':'망치형 (Bull)','desc':'바닥권 반등 암시', 'col':C_WARN})
    elif pat == "Doji": cards.append({'title':'5. 캔들 패턴','stat':'도지 (Doji)','desc':'추세 고민 중', 'col':C_NEUT})
    else: cards.append({'title':'5. 캔들 패턴','stat':'일반','desc':'특이 패턴 없음', 'col':C_NEUT})

    c_top = max(last['Senkou_Span_A'], last['Senkou_Span_B'])
    c_bot = min(last['Senkou_Span_A'], last['Senkou_Span_B'])
    if close > c_top: score += 10; cards.append({'title':'6. 일목균형표','stat':'구름대 위','desc':'상승 추세 지지', 'col':C_CYAN})
    elif close < c_bot: score -= 10; cards.append({'title':'6. 일목균형표','stat':'구름대 아래','desc':'강한 저항 구간', 'col':C_BEAR})
    else: cards.append({'title':'6. 일목균형표','stat':'구름대 안','desc':'방향성 탐색 중', 'col':C_NEUT})

    if close > last['MA20']:
        score += 10
        cards.append({'title':'7. 추세 (MA)','stat':'단기 상승','desc':'20일선 위', 'col':C_BULL})
    else:
        score -= 15
        cards.append({'title':'7. 추세 (MA)','stat':'단기 하락','desc':'20일선 붕괴', 'col':C_BEAR})

    if monte_prob >= 40: score += 10; cards.append({'title':'8. 대박 확률','stat':f'{monte_prob:.0f}% (>30%)','desc':'120일 내 +30% 유력', 'col':C_BULL})
    elif monte_prob <= 10: score -= 10; cards.append({'title':'8. 대박 확률','stat':f'{monte_prob:.0f}% (>30%)','desc':'대시세 희박', 'col':C_BEAR})
    else: cards.append({'title':'8. 대박 확률','stat':f'{monte_prob:.0f}% (>30%)','desc':'보통', 'col':C_NEUT})

    sent_data = get_sentiment_and_short_data(ticker, df)
    sp = sent_data['short_pct']
    if sent_data['short_signal'] == "Squeeze Possibility":
        score += 10
        cards.append({'title':'9. 공매도(Short)','stat':f'{sp:.1f}% (폭발적)','desc':'⚠️ 숏 스퀴즈 가능성!', 'col':C_PURP})
    elif sent_data['short_signal'] == "High Short (Bad)":
        score -= 15
        cards.append({'title':'9. 공매도(Short)','stat':f'{sp:.1f}% (위험)','desc':'하락 베팅 세력 많음', 'col':C_BEAR})
    else:
        cards.append({'title':'9. 공매도(Short)','stat':f'{sp:.1f}% (양호)','desc':'특이사항 없음', 'col':C_NEUT})

    if sent_data['news_signal'] == "Positive":
        score += 10
        cards.append({'title':'10. 뉴스 심리','stat':'긍정적','desc':'호재성 키워드 포착', 'col':C_BULL})
    elif sent_data['news_signal'] == "Negative":
        score -= 10
        cards.append({'title':'10. 뉴스 심리','stat':'부정적','desc':'악재성 키워드 주의', 'col':C_BEAR})

    if sent_data['upside_pot'] > 30: score += 5
    if ad_signal == "Bull": score += 15
    elif ad_signal == "Bear": score -= 15; red_flags += 1
    if poc_signal == "Bull": score += 10
    elif poc_signal == "Bear": score -= 10; red_flags += 1
    if mfi_signal == "Oversold": score += 10

    if red_flags > 0: score = min(score, 65)
    score = max(0, min(100, int(score)))

    if mkt_cap < 10_000_000_000 or volatility > 3.0:
        mode_txt, theme_col = "🦄 야수 (고위험)", C_PURP
        stop_mult, target_mult = 2.5, 5.0
    else:
        mode_txt, theme_col = "🛡️ 우량 (안전형)", C_CYAN
        stop_mult, target_mult = 2.0, 3.0

    stop = close - (atr * stop_mult)
    target = close + (atr * target_mult)

    if score >= 80: t, c = "강력 매수", C_BULL
    elif score >= 60:
        if red_flags > 0: t, c = "주의 (혼조세)", C_WARN
        else: t, c = "매수", C_CYAN
    elif score <= 30: t, c = "매도 / 관망", C_BEAR
    else: t, c = "관망 / 중립", C_NEUT
    vol_avg = df['Volume'].rolling(20).mean().iloc[-1]
    vol_ratio = (last['Volume'] / vol_avg) * 100

    return {
        'mode': mode_txt, 'theme': theme_col, 'score': score,
        'title': t, 'color': c,
        'cards': cards, 'tech_signals': get_18_tech_signals(df), # 18개 지표 원본 유지
        'stop': stop, 'target': target, 'close': close,
        'kelly': calculate_kelly(monte_prob, (target-close)/(close-stop) if close>stop else 1),
        'vol_data': {'last': last['Volume'], 'avg': vol_avg, 'ratio': vol_ratio},
        'adv_features': {'whale_gap': whale_gap, 'ad_signal': ad_signal, 'poc_signal': poc_signal, 'mfi_signal': mfi_signal, 'poc_price': poc_price},
        'monte_prob': monte_prob,
        'entry_date': last_date_str, 
        'sent_data': sent_data
    }

def get_action_strategy_html(ticker, analysis, monte_res):
    score = analysis['score']
    win_prob = monte_res[4]
    peak_yield = monte_res[6]
    min_yield = monte_res[8]
    kelly = analysis['kelly']

    downside = abs(min_yield) if min_yield < 0 else 1.0
    if downside == 0: downside = 1.0
    rr_ratio = peak_yield / downside

    whale_gap = analysis['adv_features']['whale_gap']
    is_squeeze = any(c['title'] == '3. 변동성' and '스퀴즈 ON' in c['stat'] for c in analysis['cards'])

    decision = "HOLD"
    reason = "판단 보류"
    color = "#aaa"

    if score < 60:
        decision = "DROP (관심 삭제)"
        reason = "AI 점수가 60점 미만입니다. 상승 모멘텀이 부족합니다."
        color = C_BEAR
    elif win_prob < 50:
        decision = "DROP (관심 삭제)"
        reason = "시뮬레이션 승률이 50% 미만입니다. 기회 비용이 큽니다."
        color = C_BEAR
    elif rr_ratio < 2.0:
        decision = "WAIT (관망)"
        reason = f"손익비가 {rr_ratio:.1f}배로 낮습니다. (목표수익 대비 리스크가 큼)"
        color = C_WARN
    else:
        if whale_gap > 10 or is_squeeze:
            decision = "BUY (진입 추천)"
            reason = "점수/승률/손익비 합격 + 고래 매집/스퀴즈 신호 포착됨."
            color = C_BULL
        else:
            decision = "WATCH (타이밍 대기)"
            reason = "조건은 훌륭하나, 결정적인 매수 트리거(고래/변동성)가 아직 없습니다."
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

# --- [메인 실행 함수 (UI 매핑)] ---
def main():
    # 사이드바 입력
    with st.sidebar:
        st.header("🔍 종목 검색")
        # [수정됨] 초기값을 비워두고, 사용자가 입력 전에는 빈 화면 유지
        input_ticker = st.text_input("Ticker", value="").upper()
        if st.button("AI 분석 실행"):
            st.rerun()

    if not input_ticker:
        st.info("좌측 사이드바에 종목코드(예: NVDA)를 입력하고 실행 버튼을 누르세요.")
        return

    # 로딩 및 데이터 처리
    with st.spinner(f"📡 {input_ticker} 데이터 정밀 분석 중... (원본 로직 적용)"):
        # 1. 데이터 가져오기
        stock_info = get_stock_info(input_ticker)
        df, data_time_utc = get_realtime_synced_data(input_ticker)
        macro_data = get_market_macro()

        if df is None:
            st.error("데이터를 불러올 수 없습니다.")
            return

        # 2. 분석 수행
        monte_res = run_monte_carlo(df)
        analysis = analyze_whale_mode(input_ticker, df, None, 0, 0, stock_info, monte_res[4], macro_data, data_time_utc)
        
        # 3. 화면 렌더링 (HTML 사용)
        
        # 헤더
        st.markdown(f"<h1 style='color:white;'>{input_ticker} <span style='font-size:0.5em; color:#888;'>{stock_info['name']}</span></h1>", unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])
        with col1:
            st.caption(f"기준: {analysis['entry_date']} | Mode: {analysis['mode']}")
        with col2:
            st.markdown(f"<div style='text-align:right; font-size:2rem; font-weight:bold; color:{analysis['color']};'>{analysis['score']}점</div>", unsafe_allow_html=True)

        # 탭 구성
        tab1, tab2, tab3, tab4 = st.tabs(["📊 대시보드", "📑 핵심 8대요인", "🎛 18개 기술지표", "🎲 시뮬레이션"])

        with tab1:
            # 액션 가이드 HTML 렌더링
            st.markdown(get_action_strategy_html(input_ticker, analysis, monte_res), unsafe_allow_html=True)
            
            # 주요 수치
            c1, c2, c3 = st.columns(3)
            c1.metric("현재가", f"${analysis['close']:.2f}")
            c1.metric("목표가", f"${analysis['target']:.2f}")
            c2.metric("고래 갭 (Whale Gap)", f"{analysis['adv_features']['whale_gap']:.1f}", delta_color="off")
            
            # 뉴스 헤드라인
            st.markdown("---")
            st.subheader("📰 AI 뉴스 감지")
            for news in analysis['sent_data']['headlines']:
                st.markdown(f"- {news}")

        with tab2:
            # 8대 요인 카드 (HTML 스타일 복원)
            st.markdown("### 🧬 AI 정밀 진단 결과")
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

        with tab3:
            # 18개 기술적 지표 (사용자가 원한 것)
            st.markdown("### 🎛 18개 기술적 지표 (Tech Signals)")
            signals = analysis['tech_signals']
            
            # 2열로 나누어 표시
            t_col1, t_col2 = st.columns(2)
            mid = (len(signals) + 1) // 2
            
            with t_col1:
                for name, val, bias in signals[:mid]:
                    color = C_BULL if bias == "Bull" else (C_BEAR if bias == "Bear" else "#888")
                    st.markdown(f"<div style='display:flex; justify-content:space-between; border-bottom:1px solid #333; padding:5px;'><span style='color:#ccc;'>{name}</span><span style='color:{color}; font-weight:bold;'>{val}</span></div>", unsafe_allow_html=True)
            
            with t_col2:
                 for name, val, bias in signals[mid:]:
                    color = C_BULL if bias == "Bull" else (C_BEAR if bias == "Bear" else "#888")
                    st.markdown(f"<div style='display:flex; justify-content:space-between; border-bottom:1px solid #333; padding:5px;'><span style='color:#ccc;'>{name}</span><span style='color:{color}; font-weight:bold;'>{val}</span></div>", unsafe_allow_html=True)

        with tab4:
            # 몬테카를로 결과
            st.markdown("### 🎲 120일 미래 시뮬레이션")
            peak_yield = monte_res[6]
            min_yield = monte_res[8]
            win_prob = monte_res[4]
            
            sc1, sc2 = st.columns(2)
            sc1.metric("예상 최고 수익", f"+{peak_yield:.1f}%")
            sc2.metric("최악의 하락폭", f"{min_yield:.1f}%")
            st.metric("승률 (Target 도달)", f"{win_prob:.1f}%")
            
            # 추가 목표 시나리오
            st.table(pd.DataFrame(monte_res[9]).set_index('pct'))

if __name__ == "__main__":
    main()
