# config.py — Все настройки и константы бота
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# ВЕРСИЯ И МЕТАДАННЫЕ
# ============================================================
BOT_VERSION = "12.0 Ultimate Pro Fixed"
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
# ТЕЛЕГРАМ (для уведомлений)
# ============================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ============================================================
# ТОРГОВЫЕ ПАРЫ
# ============================================================
SYMBOLS = [
    "PEPE/USDT", "WIF/USDT", "BOME/USDT", "FET/USDT",
    "BONK/USDT", "FLOKI/USDT", "DOGE/USDT", "SHIB/USDT",
    "ONG/USDT", "1000SATS/USDT", "BABYDOGE/USDT", "MONGCOIN/USDT",
]

# ============================================================
# ОСНОВНЫЕ ПАРАМЕТРЫ
# ============================================================
LEVERAGE = 3
TIMEFRAME_TA = "5m"
TIMEFRAME_TREND = "1h"
TIMEFRAME_MID = "15m"
TIMEFRAME_4H = "4h"
SCAN_INTERVAL = 300  # Секунды между сканированиями

MIN_SCORE = 65  # Минимальный скор для входа

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
MAX_SLIPPAGE_PCT = 0.1  # Максимальное допустимое проскальзывание (%)

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
MAX_PORTFOLIO_RISK = 0.05  # 5% от портфеля
MAX_OPEN_POSITIONS = 3  # Максимум открытых позиций
MIN_BALANCE = 5.0  # Минимальный баланс для торговли (USDT)
MAX_DRAWDOWN_PCT = 15.0  # Максимальная просадка (%)
CORRELATION_THRESHOLD = 0.8  # Порог корреляции между парами

# ============================================================
# АНАЛИЗ ПРИБЫЛЬНОСТИ
# ============================================================
DEPOSIT_ANALYSIS_ENABLED = True
SLIPPAGE_PCT = 0.05
FUNDING_RATE_CHECK = True
BYBIT_FEE = 0.00055  # Комиссия Bybit

# ============================================================
# ЧАСТИЧНЫЙ БЕЗУБЫТОК
# ============================================================
PARTIAL_BE_ENABLED = True
PARTIAL_BE_CLOSE_PCT = 50.0  # % позиции для частичного закрытия
PARTIAL_BE_PROFIT = 0.2  # % прибыли для активации частичного БУ

# ============================================================
# ТРЕЙЛИНГ
# ============================================================
TRAILING_ATR_PERIOD = 14
TRAILING_ATR_MULT = 2.0
TRAILING_OFFSET_MULT = 1.5
MIN_TRAILING_STEP = 0.4
MIN_TRAILING_OFFSET = 0.6
MIN_PROFIT_FOR_TRAIL = 1.0  # % прибыли для активации трейлинга
RR_EXIT_TRIGGER = 0.6  # Триггер для выхода по RR

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
SESSION_BLOCK_START = 0  # Час начала блокировки (UTC)
SESSION_BLOCK_END = 4    # Час конца блокировки (UTC)

DAILY_LOSS_LIMIT_PCT = 3.0  # % от дневного депозита
DAILY_LOSS_PAUSE_SEC = 10800  # Пауза при превышении дневного лимита (секунды)

VOLUME_SPIKE_MULT = 3.5
VOLUME_AVG_PERIOD = 20

SIGNAL_EXIT_ENABLED = True
ENTRY_CONFIRM_BARS = 1  # Количество свечей для подтверждения входа
ENTRY_CONFIRM_MIN_SCORE = 60  # Минимальный скор для подтверждения

SYMBOL_BLOCK_AFTER_TP = 90  # Минуты блокировки после TP
SYMBOL_BLOCK_AFTER_SL = 180  # Минуты блокировки после SL
SL_STREAK_LIMIT = 2  # Количество SL подряд для активации cooldown
SL_STREAK_PAUSE = 3600  # Пауза при SL-стрике (секунды)
SL_STREAK_EXTRA_PAUSE = 300  # Дополнительная пауза (секунды)

TRADE_MAX_LIFETIME = 7200  # Максимальное время жизни сделки (секунды)
REPORT_INTERVAL = 1800  # Интервал отчётов (секунды)

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
ML_MODEL_TYPE = "RandomForest"  # RandomForest, GradientBoosting, XGBoost
ML_FEATURES_WINDOW = 30
ML_RETRAIN_INTERVAL = 100  # Количество сделок между переобучением
ML_MIN_SAMPLES = 50  # Минимальное количество образцов для обучения
ML_FEATURES_VERSION = "v2"
ML_LOG_DATA = True
ML_LOG_FILE = "ml_training_data_v12.json"
ML_MODEL_FILE = "ml_model_v12.pkl"

# ============================================================
# ПОРТФЕЛЬ
# ============================================================
PORTFOLIO_OPTIMIZATION = True

# ============================================================
# МОНТЕ-КАРЛО
# ============================================================
MONTE_CARLO_ENABLED = True
MONTE_CARLO_SIMULATIONS = 1000
MONTE_CARLO_DAYS = 30

# ============================================================
# ФАЙЛЫ
# ============================================================
STATE_FILE = "state_bot_v12.json"
TRADES_FILE = "trades_bot_v12.json"
INDICATOR_STATS_FILE = "indicator_stats_v12.json"
METRICS_FILE = "strategy_metrics_v12.json"
PORTFOLIO_STATE_FILE = "portfolio_state_v12.json"
