# config.py — Настройки для торговли мем-коинами (Bybit v5 API)
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# ВЕРСИЯ И МЕТАДАННЫЕ
# ============================================================
BOT_VERSION = "12.3 MemeCoin Turbo Fixed"
RELEASE_DATE = "28.05.2026"

# ============================================================
# API КЛЮЧИ
# ============================================================
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")

if not API_KEY or len(API_KEY) < 10:
    raise ValueError("❌ BYBIT_API_KEY не задан или слишком короткий! Проверьте .env файл")
if not API_SECRET or len(API_SECRET) < 10:
    raise ValueError("❌ BYBIT_API_SECRET не задан или слишком короткий! Проверьте .env файл")

# ============================================================
# TELEGRAM (для уведомлений)
# ============================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ============================================================
# ТОРГОВЫЕ ПАРЫ (20 мем-коинов для Bybit v5 API)
# ============================================================
SYMBOLS = [
    "PEPE/USDT", "WIF/USDT", "BONK/USDT", "FLOKI/USDT", "DOGE/USDT",
    "SHIB/USDT", "BOME/USDT", "ONG/USDT", "1000SATS/USDT", "BABYDOGE/USDT",
    "MONGCOIN/USDT", "TOSHI/USDT", "BRETT/USDT", "POPCAT/USDT", "WEN/USDT",
    "SLERF/USDT", "DYM/USDT", "PUNDX/USDT", "GALA/USDT", "AIDOGE/USDT",
]

# ============================================================
# ОСНОВНЫЕ ПАРАМЕТРЫ
# ============================================================
LEVERAGE = 10  # Высокое плечо для мем-коинов
TIMEFRAME_TA = "5m"  # Таймфрейм для анализа
TIMEFRAME_TREND = "15m"
TIMEFRAME_4H = "4h"
SCAN_INTERVAL = 30  # Сканирование каждые 30 секунд
MIN_SCORE = 55  # Минимальный скор для входа

# ============================================================
# ТП / СЛ
# ============================================================
TP_PERCENT = 8.0  # Take Profit (%)
SL_PERCENT = 3.0  # Stop Loss (%)
MIN_SL_PERCENT = 1.5
MAX_SL_PERCENT = 5.0
ATR_SL_MULT = 2.5  # Множитель ATR для SL
ATR_TP_MULT = 3.5  # Множитель ATR для TP
MIN_RR_RATIO = 1.2  # Минимальное соотношение риска/прибыли
MAX_SLIPPAGE_PCT = 0.2  # Максимальное проскальзывание (%)

# ============================================================
# РИСК-МЕНЕДЖМЕНТ
# ============================================================
BASE_RISK_PCT = 1.5  # Базовый риск на сделку (% от баланса)
MAX_RISK_PCT = 3.0   # Максимальный риск на сделку
USE_ADVANCED_RISK = True
MIN_TRADES_FOR_F = 10
MAX_RISK_PERCENT_F = 4.0
MAX_RISK_PER_TRADE_PCT = 4.0
MIN_EXPECTED_PROFIT_USDT = 0.1
MAX_PORTFOLIO_RISK = 0.15  # 15% от портфеля
MAX_OPEN_POSITIONS = 5  # Максимум открытых позиций
MIN_BALANCE = 5.0  # Минимальный баланс для торговли (USDT)
MAX_DRAWDOWN_PCT = 25.0  # Максимальная просадка (%)

# ============================================================
# АНАЛИЗ ПРИБЫЛЬНОСТИ
# ============================================================
DEPOSIT_ANALYSIS_ENABLED = True
SLIPPAGE_PCT = 0.1  # Проскальзывание (%)
FUNDING_RATE_CHECK = True
BYBIT_FEE = 0.00055  # Комиссия Bybit

# ============================================================
# ЧАСТИЧНЫЙ БЕЗУБЫТОК
# ============================================================
PARTIAL_BE_ENABLED = True
PARTIAL_BE_CLOSE_PCT = 50.0  # % позиции для частичного закрытия
PARTIAL_BE_PROFIT = 2.0  # % прибыли для активации частичного БУ

# ============================================================
# ТРЕЙЛИНГ-СТОП
# ============================================================
TRAILING_ATR_PERIOD = 14
TRAILING_ATR_MULT = 3.0
TRAILING_OFFSET_MULT = 1.5
MIN_TRAILING_STEP = 0.5
MIN_TRAILING_OFFSET = 0.7
MIN_PROFIT_FOR_TRAIL = 2.0
RR_EXIT_TRIGGER = 0.5

# ============================================================
# MA КРОССОВЕР
# ============================================================
MA_CROSSOVER_ENABLED = True
MA1_TYPE = "EMA"
MA2_TYPE = "EMA"
MA1_LENGTH = 9
MA2_LENGTH = 21
MA_TIMEFRAME = "5m"

# ============================================================
# ФИЛЬТРЫ
# ============================================================
SESSION_FILTER_ENABLED = False
SESSION_BLOCK_START = 0
SESSION_BLOCK_END = 4

DAILY_LOSS_LIMIT_PCT = 5.0
DAILY_LOSS_PAUSE_SEC = 10800  # 3 часа

VOLUME_SPIKE_MULT = 5.0
VOLUME_AVG_PERIOD = 10

SIGNAL_EXIT_ENABLED = True
ENTRY_CONFIRM_BARS = 0  # Отключено подтверждение входа
ENTRY_CONFIRM_MIN_SCORE = 60

SYMBOL_BLOCK_AFTER_TP = 15  # Минуты блокировки после TP
SYMBOL_BLOCK_AFTER_SL = 30  # Минуты блокировки после SL

SL_STREAK_LIMIT = 3
SL_STREAK_PAUSE = 1800  # 30 минут
SL_STREAK_EXTRA_PAUSE = 300  # 5 минут

TRADE_MAX_LIFETIME = 3600  # 1 час
REPORT_INTERVAL = 300  # 5 минут

# ============================================================
# S/R УРОВНИ
# ============================================================
SR_PERIOD = 50
SR_PROXIMITY_PCT = 0.8
SR_MIN_TOUCHES = 2
SR_CLUSTER_TOL = 0.01
SR_BLOCK_DIST_PCT = 0.8

# ============================================================
# КВАНТОВЫЙ АНАЛИЗ (Отключён)
# ============================================================
QUANT_ENABLED = False

# ============================================================
# ORDER FLOW
# ============================================================
ORDER_FLOW_ENABLED = True
ORDER_BOOK_DEPTH = 10
VOLUME_PROFILE_ENABLED = False
CLUSTER_TOLERANCE = 0.01

# ============================================================
# МАШИННОЕ ОБУЧЕНИЕ (Отключено)
# ============================================================
ML_ENABLED = False

# ============================================================
# ПОРТФЕЛЬ
# ============================================================
PORTFOLIO_OPTIMIZATION = False
CORRELATION_THRESHOLD = 0.8

# ============================================================
# МОНТЕ-КАРЛО (Отключено)
# ============================================================
MONTE_CARLO_ENABLED = False

# ============================================================
# ФАЙЛЫ
# ============================================================
STATE_FILE = "state_bot_meme.json"
TRADES_FILE = "trades_bot_meme.json"
INDICATOR_STATS_FILE = "indicator_stats_meme.json"
METRICS_FILE = "strategy_metrics_meme.json"
PORTFOLIO_STATE_FILE = "portfolio_state_meme.json"
