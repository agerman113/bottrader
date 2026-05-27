# config.py — Все настройки и константы бота
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# ВЕРСИЯ И МЕТАДАННЫЕ
# ============================================================
BOT_VERSION = "11.1 Ultimate Pro Fixed"
RELEASE_DATE = "28.05.2026"

# ============================================================
# API КЛЮЧИ (проверка при импорте)
# ============================================================
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")

if not API_KEY or len(API_KEY) < 10:
    raise ValueError("BYBIT_API_KEY не задан или слишком короткий! Проверь .env файл")
if not API_SECRET or len(API_SECRET) < 10:
    raise ValueError("BYBIT_API_SECRET не задан или слишком короткий! Проверь .env файл")

# ============================================================
# ТОРГОВЫЕ ПАРЫ
# ============================================================
SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT", "XRP/USDT:USDT",
    "SOL/USDT:USDT", "ADA/USDT:USDT", "TRX/USDT:USDT", "TON/USDT:USDT",
    "AVAX/USDT:USDT", "DOT/USDT:USDT", "LTC/USDT:USDT", "BCH/USDT:USDT",
    "ATOM/USDT:USDT", "XLM/USDT:USDT", "NEAR/USDT:USDT", "DOGE/USDT:USDT",
    "PEPE/USDT:USDT", "WIF/USDT:USDT", "BOME/USDT:USDT", "FET/USDT:USDT",
]

# ============================================================
# ОСНОВНЫЕ ПАРАМЕТРЫ
# ============================================================
LEVERAGE = 3
TIMEFRAME_TA = "5m"
TIMEFRAME_TREND = "1h"
TIMEFRAME_MID = "15m"
TIMEFRAME_4H = "4h"
SCAN_INTERVAL = 300

MIN_SCORE = 65

# ============================================================
# ТП / СЛ
# ============================================================
TP_PERCENT = 3.0
SL_PERCENT = 1.0
MIN_SL_PERCENT = 0.8
MAX_SL_PERCENT = 2.0
ATR_SL_MULT = 1.5
ATR_TP_MULT = 3.0
MIN_RR_RATIO = 2.0

# ============================================================
# РИСК-МЕНЕДЖМЕНТ
# ============================================================
BASE_RISK_PCT = 0.8
MAX_RISK_PCT = 1.2
USE_ADVANCED_RISK = True
MIN_TRADES_FOR_F = 20
MAX_RISK_PERCENT_F = 2.5
MAX_RISK_PER_TRADE_PCT = 2.0
MIN_EXPECTED_PROFIT_USDT = 0.5
MAX_PORTFOLIO_RISK = 0.05

# ============================================================
# АНАЛИЗ ПРИБЫЛЬНОСТИ
# ============================================================
DEPOSIT_ANALYSIS_ENABLED = True
SLIPPAGE_PCT = 0.05
FUNDING_RATE_CHECK = True
BYBIT_FEE = 0.00055

# ============================================================
# ЧАСТИЧНЫЙ БЕЗУБЫТОК
# ============================================================
PARTIAL_BE_ENABLED = True
PARTIAL_BE_CLOSE_PCT = 50.0
PARTIAL_BE_PROFIT = 0.2

# ============================================================
# ТРЕЙЛИНГ
# ============================================================
TRAILING_ATR_PERIOD = 14
TRAILING_ATR_MULT = 2.0
TRAILING_OFFSET_MULT = 1.5
MIN_TRAILING_STEP = 0.4
MIN_TRAILING_OFFSET = 0.6
MIN_PROFIT_FOR_TRAIL = 1.0
RR_EXIT_TRIGGER = 0.6

# ============================================================
# MA КРОССОВЕР
# ============================================================
MA_CROSSOVER_ENABLED = True
MA1_TYPE = "EMA"
MA2_TYPE = "EMA"
MA1_LENGTH = 21
MA2_LENGTH = 50
MA_TIMEFRAME = "5m"

# ============================================================
# ФИЛЬТРЫ
# ============================================================
SESSION_FILTER_ENABLED = False
SESSION_BLOCK_START = 0
SESSION_BLOCK_END = 4

DAILY_LOSS_LIMIT_PCT = 3.0
DAILY_LOSS_PAUSE_SEC = 10800

VOLUME_SPIKE_MULT = 3.5
VOLUME_AVG_PERIOD = 20

SIGNAL_EXIT_ENABLED = True
ENTRY_CONFIRM_BARS = 0
ENTRY_CONFIRM_MIN_SCORE = 60

SYMBOL_BLOCK_AFTER_TP = 90
SYMBOL_BLOCK_AFTER_SL = 180
SL_STREAK_LIMIT = 2
SL_STREAK_PAUSE = 3600
SL_STREAK_EXTRA_PAUSE = 300

MIN_BALANCE = 5.0
MAX_DRAWDOWN_PCT = 15.0

TRADE_MAX_LIFETIME = 7200
REPORT_INTERVAL = 1800

# ============================================================
# S/R УРОВНИ
# ============================================================
SR_PERIOD = 100
SR_PROXIMITY_PCT = 0.5
SR_MIN_TOUCHES = 3
SR_CLUSTER_TOL = 0.005
SR_BLOCK_DIST_PCT = 0.5

# ============================================================
# КВАНТОВЫЙ АНАЛИЗ
# ============================================================
QUANT_ENABLED = True
COINTEGRATION_PAIRS = [
    ("BTC/USDT:USDT", "ETH/USDT:USDT"),
    ("BTC/USDT:USDT", "BNB/USDT:USDT"),
    ("ETH/USDT:USDT", "SOL/USDT:USDT"),
]
COINTEGRATION_WINDOW = 100
MEAN_REVERSION_THRESHOLD = 2.0
MOMENTUM_WINDOW = 20

# ============================================================
# ORDER FLOW
# ============================================================
ORDER_FLOW_ENABLED = True
ORDER_BOOK_DEPTH = 20
VOLUME_PROFILE_ENABLED = True
VOLUME_PROFILE_BARS = 50
CLUSTER_TOLERANCE = 0.005

# ============================================================
# МАШИННОЕ ОБУЧЕНИЕ
# ============================================================
ML_ENABLED = True
ML_MODEL_TYPE = "RandomForest"
ML_FEATURES_WINDOW = 30
ML_RETRAIN_INTERVAL = 100
ML_MIN_SAMPLES = 50
ML_FEATURES_VERSION = "v2"
ML_LOG_DATA = True
ML_LOG_FILE = "ml_training_data_v11.json"
ML_MODEL_FILE = "ml_model_v11.pkl"

# ============================================================
# ПОРТФЕЛЬ
# ============================================================
PORTFOLIO_OPTIMIZATION = True
CORRELATION_THRESHOLD = 0.8

# ============================================================
# МОНТЕ-КАРЛО
# ============================================================
MONTE_CARLO_ENABLED = True
MONTE_CARLO_SIMULATIONS = 1000
MONTE_CARLO_DAYS = 30

# ============================================================
# ФАЙЛЫ
# ============================================================
STATE_FILE = "state_bot_v11.json"
TRADES_FILE = "trades_bot_v11.json"
INDICATOR_STATS_FILE = "indicator_stats_v11.json"
METRICS_FILE = "strategy_metrics_v11.json"
PORTFOLIO_STATE_FILE = "portfolio_state_v11.json"
