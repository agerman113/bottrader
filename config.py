# config.py — Настройки для торговли мем-коинами (Bybit v5 API)
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# ВЕРСИЯ И МЕТАДАННЫЕ
# ============================================================
BOT_VERSION = "12.2 MemeCoin Turbo"
RELEASE_DATE = "28.05.2026"

# ============================================================
# API КЛЮЧИ (проверка при импорте)
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
    # Топ мем-коины
    "PEPE/USDT",    # Pepe
    "WIF/USDT",     # Dogwifhat
    "BONK/USDT",    # Bonk
    "FLOKI/USDT",   # Floki
    "DOGE/USDT",    # Dogecoin
    "SHIB/USDT",    # Shiba Inu
    "BOME/USDT",    # Book of Meme
    "ONG/USDT",     # Ongoing
    "1000SATS/USDT", # 1000 Satoshi
    "BABYDOGE/USDT", # Baby Doge Coin

    # Дополнительные мем-коины
    "MONGCOIN/USDT", # MongCoin
    "TOSHI/USDT",    # Toshi
    "BRETT/USDT",    # Brett
    "POPCAT/USDT",   # Popcat
    "WEN/USDT",      # Wen
    "SLERF/USDT",    # Slerf
    "BONK/USDT",     # Bonk (дубль для надёжности)
    "DYM/USDT",      # Dymension
    "PUNDX/USDT",    # Pundi X (иногда относится к мем-коинам)
    "GALA/USDT",     # Gala
    "AIDOGE/USDT",   # AI Doge
]

# ============================================================
# ОСНОВНЫЕ ПАРАМЕТРЫ (Оптимизированы для мем-коинов)
# ============================================================
LEVERAGE = 10  # Высокое плечо для мем-коинов (Bybit поддерживает до 100x)
TIMEFRAME_TA = "5m"  # Частые сделки
TIMEFRAME_TREND = "15m"
TIMEFRAME_4H = "4h"
SCAN_INTERVAL = 30  # Сканирование каждые 30 секунд (для мем-коинов)
MIN_SCORE = 55  # Уменьшен минимальный скор для частых сделок

# ============================================================
# ТП / СЛ (Агрессивная стратегия для мем-коинов)
# ============================================================
TP_PERCENT = 8.0   # Высокий TP для мем-коинов
SL_PERCENT = 3.0   # Широкий SL
MIN_SL_PERCENT = 1.5
MAX_SL_PERCENT = 5.0
ATR_SL_MULT = 2.5  # Множитель ATR для SL
ATR_TP_MULT = 3.5  # Множитель ATR для TP
MIN_RR_RATIO = 1.2  # Минимальное соотношение риска/прибыли
MAX_SLIPPAGE_PCT = 0.2  # Допустимое проскальзывание (20% для мем-коинов)

# ============================================================
# РИСК-МЕНЕДЖМЕНТ (Агрессивный для мем-коинов)
# ============================================================
BASE_RISK_PCT = 1.5   # Базовый риск на сделку (% от баланса)
MAX_RISK_PCT = 3.0    # Максимальный риск на сделку
USE_ADVANCED_RISK = True
MIN_TRADES_FOR_F = 10
MAX_RISK_PERCENT_F = 4.0  # Максимальный риск по Келли
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
BYBIT_FEE = 0.00055  # Комиссия Bybit (0.055%)

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
TRAILING_ATR_MULT = 3.0  # Множитель ATR для трейлинга
TRAILING_OFFSET_MULT = 1.5
MIN_TRAILING_STEP = 0.5
MIN_TRAILING_OFFSET = 0.7
MIN_PROFIT_FOR_TRAIL = 2.0  # % прибыли для активации трейлинга
RR_EXIT_TRIGGER = 0.5  # Триггер для выхода по RR

# ============================================================
# MA КРОССОВЕР (Быстрые индикаторы для мем-коинов)
# ============================================================
MA_CROSSOVER_ENABLED = True
MA1_TYPE = "EMA"
MA2_TYPE = "EMA"
MA1_LENGTH = 9   # Быстрая MA
MA2_LENGTH = 21  # Медленная MA
MA_TIMEFRAME = "5m"

# ============================================================
# ФИЛЬТРЫ
# ============================================================
SESSION_FILTER_ENABLED = False  # Время торговли не ограничено
SESSION_BLOCK_START = 0   # Час начала блокировки (UTC)
SESSION_BLOCK_END = 4     # Час конца блокировки (UTC)

DAILY_LOSS_LIMIT_PCT = 5.0  # % от дневного депозита
DAILY_LOSS_PAUSE_SEC = 10800  # Пауза при превышении дневного лимита (3 часа)

VOLUME_SPIKE_MULT = 5.0  # Множитель для фильтра объёма
VOLUME_AVG_PERIOD = 10   # Период для расчёта среднего объёма

SIGNAL_EXIT_ENABLED = True  # Выход по сигналу разворота
ENTRY_CONFIRM_BARS = 0   # Отключено подтверждение входа (для частых сделок)
ENTRY_CONFIRM_MIN_SCORE = 60

# Блокировка символов после TP/SL (минуты)
SYMBOL_BLOCK_AFTER_TP = 15  # Уменьшено для мем-коинов
SYMBOL_BLOCK_AFTER_SL = 30

# Лимит SL подряд
SL_STREAK_LIMIT = 3
SL_STREAK_PAUSE = 1800  # Пауза 30 минут
SL_STREAK_EXTRA_PAUSE = 300  # Дополнительная пауза 5 минут

TRADE_MAX_LIFETIME = 3600  # Максимальное время жизни сделки (1 час)
REPORT_INTERVAL = 300  # Интервал отчётов (5 минут)

# ============================================================
# S/R УРОВНИ (Уровни поддержки/сопротивления)
# ============================================================
SR_PERIOD = 50  # Период для анализа S/R
SR_PROXIMITY_PCT = 0.8  # Процент близости к уровню
SR_MIN_TOUCHES = 2     # Минимальное количество касаний уровня
SR_CLUSTER_TOL = 0.01  # Толерантность для кластеризации уровней
SR_BLOCK_DIST_PCT = 0.8  # Процент блокировки при приближении к уровню

# ============================================================
# КВАНТОВЫЙ АНАЛИЗ (Отключён для мем-коинов)
# ============================================================
QUANT_ENABLED = False

# ============================================================
# ORDER FLOW (Анализ стакана)
# ============================================================
ORDER_FLOW_ENABLED = True
ORDER_BOOK_DEPTH = 10  # Глубина стакана для анализа
VOLUME_PROFILE_ENABLED = False
CLUSTER_TOLERANCE = 0.01

# ============================================================
# МАШИННОЕ ОБУЧЕНИЕ (Отключено для мем-коинов)
# ============================================================
ML_ENABLED = False
ML_MODEL_TYPE = "RandomForest"
ML_FEATURES_WINDOW = 30
ML_RETRAIN_INTERVAL = 100
ML_MIN_SAMPLES = 50
ML_FEATURES_VERSION = "v2"
ML_LOG_DATA = False
ML_LOG_FILE = "ml_training_data_meme.json"
ML_MODEL_FILE = "ml_model_meme.pkl"

# ============================================================
# ПОРТФЕЛЬ
# ============================================================
PORTFOLIO_OPTIMIZATION = False  # Отключено для мем-коинов
CORRELATION_THRESHOLD = 0.8  # Порог корреляции между парами

# ============================================================
# МОНТЕ-КАРЛО (Отключено для мем-коинов)
# ============================================================
MONTE_CARLO_ENABLED = False
MONTE_CARLO_SIMULATIONS = 1000
MONTE_CARLO_DAYS = 30

# ============================================================
# ФАЙЛЫ (для сохранения состояния и истории)
# ============================================================
STATE_FILE = "state_bot_meme_v12.json"
TRADES_FILE = "trades_bot_meme_v12.json"
INDICATOR_STATS_FILE = "indicator_stats_meme_v12.json"
METRICS_FILE = "strategy_metrics_meme_v12.json"
PORTFOLIO_STATE_FILE = "portfolio_state_meme_v12.json"
