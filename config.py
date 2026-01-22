# config.py
import os

WINDOW_DAILY = "2y"
INTRADAY_PERIOD = "5d"
INTRADAY_INTERVAL = "1m"

INDICATOR_WARMUP = 200

# Quantile-based signals
Q_WINDOW = 252        # about 1y trading days
Q_LOW = 0.10
Q_HIGH = 0.90

# Backtest defaults
START_EQUITY = 1.0

# SEC requires a real User-Agent. Set env var if you want.
SEC_USER_AGENT = os.getenv(
    "SEC_USER_AGENT",
    "AIStockSniper/1.0 (contact: youremail@example.com)"
)

NEWS_TIMEOUT = 6
SEC_TIMEOUT = 10
YF_TIMEOUT = 20
