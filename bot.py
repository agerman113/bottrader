"""
Bybit ГИБРИДНЫЙ ФЬЮЧЕРСНЫЙ БОТ — v6 FIXED
============================================
ИСПРАВЛЕНИЯ v6:
  [FIX-1] set_trading_stop → только v5 fallback API (убран битый метод)
  [FIX-2] JSON serializable: default=str в json.dump
  [FIX-3] Блокировка монеты 90 мин после ЛЮБОГО закрытия (TP и SL)
  [FIX-4] SL минимум 2×ATR, не менее 1.5% — конец тесным стопам
  [FIX-5] TP минимум 4.5% (соотношение TP:SL = 3:1)
  [FIX-6] Rate limit guard между запросами к бирже
  [FIX-7] Предстартовая проверка (5 этапов) — бот не стартует при ошибках
  [FIX-8] Реальный PnL считается по балансу биржи до/после сделки
  [FIX-9] Пауза между сканированиями увеличена — меньше сделок, выше качество
  [FIX-10] Убрана логика Partial TP (слишком часто нарушала RR)
  [FIX-11] После SL — монета блокируется на SYMBOL_BLOCK_MINUTES
  [FIX-12] Скор входа должен держаться ≥60 при перепроверке (был слишком мягкий)
"""

import os
import time
import json
import logging
import requests
import ccxt
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from datetime import datetime, timezone

load_dotenv()

# ================== ТОРГОВЫЕ ПАРЫ ==================
SYMBOLS = [
    "BTC/USDT:USDT",   "ETH/USDT:USDT",   "BNB/USDT:USDT",
    "XRP/USDT:USDT",   "SOL/USDT:USDT",   "ADA/USDT:USDT",
    "TRX/USDT:USDT",   "TON/USDT:USDT",   "AVAX/USDT:USDT",
    "DOT/USDT:USDT",   "LTC/USDT:USDT",   "BCH/USDT:USDT",
    "ATOM/USDT:USDT",  "XLM/USDT:USDT",   "NEAR/USDT:USDT",
    "DOGE/USDT:USDT",  "PEPE/USDT:USDT",
    "WIF/USDT:USDT",   "BOME/USDT:USDT",
    "FET/USDT:USDT",   "RENDER/USDT:USDT","TAO/USDT:USDT",
    "WLD/USDT:USDT",   "ARKM/USDT:USDT",
    "IO/USDT:USDT",    "ONDO/USDT:USDT",  "VIRTUAL/USDT:USDT",
    "UNI/USDT:USDT",   "AAVE/USDT:USDT",
    "ARB/USDT:USDT",   "OP/USDT:USDT",
    "LINK/USDT:USDT",  "GRT/USDT:USDT",
    "INJ/USDT:USDT",   "SUI/USDT:USDT",   "APT/USDT:USDT",
    "TIA/USDT:USDT",   "JTO/USDT:USDT",   "EIGEN/USDT:USDT",
    "HBAR/USDT:USDT",  "VET/USDT:USDT",
    "NOT/USDT:USDT",   "CATI/USDT:USDT",
]

# ================== ОСНОВНЫЕ НАСТРОЙКИ ==================
LEVERAGE              = 3
TIMEFRAME_TA          = "15m"        # [FIX-9] 5m→15m: меньше шума, качественнее сигналы
TIMEFRAME_TREND       = "1h"
TIMEFRAME_MID         = "1h"
TIMEFRAME_4H          = "4h"
SCAN_INTERVAL         = 600          # [FIX-9] 5 мин → 10 мин между сканами

MIN_SCORE             = 70           # [FIX] Порог повышен с 65 до 70

# === ТЕЙКПРОФИТ / СТОПЛОСС [FIX-4, FIX-5] ===
TP_PERCENT            = 4.5          # [FIX-5] Было 2.5% → теперь 4.5%
SL_PERCENT            = 1.5          # [FIX-4] Было 0.8% → теперь 1.5%
MIN_SL_PERCENT        = 1.5          # Минимальный SL в %
MAX_SL_PERCENT        = 3.0          # Максимальный SL в %
ATR_SL_MULT           = 2.5          # [FIX-4] SL = 2.5 × ATR (было 1.0)
ATR_TP_MULT           = 5.0          # TP = 5.0 × ATR (соотношение 2:1 мин)

# === РИСК НА СДЕЛКУ ===
BASE_RISK_PCT         = 1.0          # % от баланса на сделку
MAX_RISK_PCT          = 1.5          # максимум при скоре 100

# === АДАПТИВНЫЙ ТРЕЙЛИНГ ===
TRAILING_ATR_PERIOD   = 14
TRAILING_ATR_MULT     = 2.0          # [FIX] Увеличен отступ трейлинга
TRAILING_OFFSET_MULT  = 1.5
MIN_TRAILING_STEP     = 0.5
MIN_TRAILING_OFFSET   = 0.8          # [FIX] Увеличен минимальный отступ
MIN_PROFIT_FOR_TRAIL  = 1.5          # [FIX] Трейлинг только при прибыли ≥1.5%

# === RREXIT ===
RR_EXIT_TRIGGER       = 0.6          # [FIX] Трейлинг при 60% пути к TP (было 50%)

# === MA КРОССОВЕР ===
MA_CROSSOVER_ENABLED  = True
MA1_TYPE              = "EMA"
MA2_TYPE              = "EMA"
MA1_LENGTH            = 21
MA2_LENGTH            = 50
MA_TIMEFRAME          = "15m"

# === SESSION FILTER ===
SESSION_FILTER_ENABLED = True
SESSION_BLOCK_START    = 0
SESSION_BLOCK_END      = 4           # [FIX] Расширен ночной блок

# === DAILY LOSS LIMIT ===
DAILY_LOSS_LIMIT_PCT  = 2.5          # [FIX] Снижен с 3% до 2.5%
DAILY_LOSS_PAUSE_SEC  = 18000        # 5 часов

# === VOLUME SPIKE GUARD ===
VOLUME_SPIKE_MULT     = 2.5          # [FIX] Снижен с 3.0x до 2.5x
VOLUME_AVG_PERIOD     = 20

# === SIGNAL EXIT ===
SIGNAL_EXIT_ENABLED   = True

# === ПОДТВЕРЖДЕНИЕ ВХОДА [FIX-6] ===
ENTRY_CONFIRM_BARS    = 1            # [FIX] Было 2 → 1 свеча (быстрее)
ENTRY_CONFIRM_MIN_SCORE = 65         # [FIX] Минимальный скор при перепроверке

# === БЛОКИРОВКА ПОСЛЕ ЗАКРЫТИЯ [FIX-3] ===
SYMBOL_BLOCK_AFTER_TP  = 90          # [FIX] 90 мин блок после TP
SYMBOL_BLOCK_AFTER_SL  = 180         # [FIX] 180 мин блок после SL
SL_STREAK_LIMIT        = 2           # [FIX] Снижен с 3 до 2
SL_STREAK_PAUSE        = 3600        # 1 час

# === ПРОСАДКА И БАЛАНС ===
MIN_BALANCE           = 5.0
MAX_DRAWDOWN_PCT      = 15.0         # [FIX] Снижен с 20% до 15%

# === ПРОЧИЕ ===
TRADE_MAX_LIFETIME    = 14400        # [FIX] 2ч → 4ч: дать сделке время
REPORT_INTERVAL       = 1800
STATE_FILE            = "state_hybrid_v6.json"
TRADES_FILE           = "trades_hybrid_v6.json"
BYBIT_FEE             = 0.00055
REQUEST_DELAY         = 0.3          # [FIX-6] Пауза между запросами к бирже (сек)

# === S/R УРОВНИ ===
SR_PERIOD             = 100
SR_PROXIMITY_PCT      = 0.5          # [FIX] Расширена зона S/R
SR_MIN_TOUCHES        = 3
SR_CLUSTER_TOL        = 0.005
SR_BLOCK_DIST_PCT     = 0.3

# ================== ЛОГИРОВАНИЕ ==================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]  %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot_hybrid_v6.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ================== БИРЖА ==================
exchange = ccxt.bybit({
    "apiKey":          os.getenv("BYBIT_API_KEY"),
    "secret":          os.getenv("BYBIT_API_SECRET"),
    "enableRateLimit": True,
    "options": {"defaultType": "linear"},
})

# ================== СТАТИСТИКА ==================
stats = {
    "запусков":          0,
    "сделок_всего":      0,
    "тейкпрофит":        0,
    "стоплосс":          0,
    "таймаут":           0,
    "прибыль_usdt":      0.0,
    "убыток_usdt":       0.0,
    "депозит_старт":     0.0,
    "баланс_начало_дня": 0.0,
    "дата_дня":          "",
    "старт_время":       "",
    "последний_отчёт":   0.0,
    "sl_streak":         0,
}

# ============================================================
# ██████  ПРЕДСТАРТОВАЯ ПРОВЕРКА (5 ЭТАПОВ) [FIX-7]
# ============================================================

def этап_1_проверка_окружения() -> tuple[bool, list]:
    """Проверяет API ключи и переменные окружения."""
    errors = []
    warnings = []
    api_key    = os.getenv("BYBIT_API_KEY",    "")
    api_secret = os.getenv("BYBIT_API_SECRET", "")
    if not api_key:
        errors.append("BYBIT_API_KEY не задан в .env")
    elif len(api_key) < 10:
        errors.append(f"BYBIT_API_KEY подозрительно короткий: {len(api_key)} символов")
    if not api_secret:
        errors.append("BYBIT_API_SECRET не задан в .env")
    elif len(api_secret) < 10:
        errors.append(f"BYBIT_API_SECRET подозрительно короткий: {len(api_secret)} символов")
    if not os.path.exists(".env"):
        warnings.append(".env файл не найден — переменные берутся из окружения")
    return len(errors) == 0, errors + warnings


def этап_2_проверка_подключения() -> tuple[bool, list]:
    """Проверяет соединение с биржей и права доступа."""
    errors = []
    try:
        баланс = exchange.fetch_balance({"type": "linear"})
        usdt = float(баланс.get("USDT", {}).get("free", 0))
        if usdt < MIN_BALANCE:
            errors.append(f"Баланс {usdt:.2f} USDT меньше минимума {MIN_BALANCE} USDT")
        else:
            log.info(f"  ✅ Подключение OK | Баланс: {usdt:.4f} USDT")
    except ccxt.AuthenticationError as e:
        errors.append(f"Ошибка аутентификации: {e}")
    except ccxt.NetworkError as e:
        errors.append(f"Сетевая ошибка: {e}")
    except Exception as e:
        errors.append(f"Неизвестная ошибка подключения: {e}")
    return len(errors) == 0, errors


def этап_3_проверка_конфигурации() -> tuple[bool, list]:
    """Проверяет параметры бота на логичность."""
    errors = []
    warnings = []
    rr = TP_PERCENT / SL_PERCENT
    if rr < 2.0:
        errors.append(f"RR ratio {rr:.1f}:1 слишком низкий (минимум 2:1). TP={TP_PERCENT}% SL={SL_PERCENT}%")
    elif rr < 2.5:
        warnings.append(f"RR ratio {rr:.1f}:1 — рекомендуется ≥2.5:1")
    if LEVERAGE > 5:
        warnings.append(f"Плечо {LEVERAGE}x высокое — риск ликвидации")
    if MIN_SCORE < 65:
        errors.append(f"MIN_SCORE={MIN_SCORE} слишком низкий (минимум 65)")
    if BASE_RISK_PCT > 3.0:
        errors.append(f"BASE_RISK_PCT={BASE_RISK_PCT}% слишком высокий (максимум 3%)")
    if DAILY_LOSS_LIMIT_PCT > 5.0:
        warnings.append(f"DAILY_LOSS_LIMIT_PCT={DAILY_LOSS_LIMIT_PCT}% высокий")
    if ATR_SL_MULT < 1.5:
        errors.append(f"ATR_SL_MULT={ATR_SL_MULT} слишком мал — SL будет слишком тесным")
    if len(SYMBOLS) < 5:
        warnings.append(f"Мало пар для торговли: {len(SYMBOLS)}")
    log.info(f"  ✅ Конфигурация: TP={TP_PERCENT}% | SL={SL_PERCENT}% | RR={rr:.1f}:1 | Плечо={LEVERAGE}x")
    return len(errors) == 0, errors + warnings


def этап_4_проверка_рынка() -> tuple[bool, list]:
    """Проверяет доступность первых 5 торговых пар."""
    errors = []
    warnings = []
    test_symbols = SYMBOLS[:5]
    доступные = 0
    for sym in test_symbols:
        try:
            time.sleep(REQUEST_DELAY)
            ticker = exchange.fetch_ticker(sym)
            if float(ticker["last"]) > 0:
                доступные += 1
        except Exception as e:
            warnings.append(f"Пара {sym} недоступна: {e}")
    if доступные == 0:
        errors.append("Ни одна из тестовых пар не доступна")
    elif доступные < len(test_symbols) // 2:
        errors.append(f"Доступно только {доступные}/{len(test_symbols)} пар")
    else:
        log.info(f"  ✅ Рынок: {доступные}/{len(test_symbols)} тестовых пар доступны")
    return len(errors) == 0, errors + warnings


def этап_5_проверка_существующих_позиций() -> tuple[bool, list]:
    """Проверяет открытые позиции и состояние системы."""
    errors = []
    warnings = []
    try:
        positions = exchange.fetch_positions()
        открытые = [p for p in positions if float(p.get("contracts", 0) or 0) > 0]
        if открытые:
            for p in открытые:
                sym = p.get("symbol", "?")
                side = p.get("side", "?")
                qty  = p.get("contracts", 0)
                warnings.append(f"Уже открыта позиция: {sym} {side} qty={qty} — бот будет ждать закрытия")
        else:
            log.info("  ✅ Открытых позиций нет — готов к торговле")
    except Exception as e:
        warnings.append(f"Не удалось проверить позиции: {e}")

    # Проверка старых файлов состояния
    if os.path.exists(TRADES_FILE):
        try:
            with open(TRADES_FILE, "r", encoding="utf-8") as f:
                история = json.load(f)
            log.info(f"  ℹ️  История: {len(история)} сделок найдено")
        except Exception:
            warnings.append(f"Файл {TRADES_FILE} повреждён — будет создан новый")
    return len(errors) == 0, errors + warnings


def запустить_предстартовую_проверку() -> bool:
    """
    Запускает все 5 этапов проверки.
    Возвращает True если бот может запуститься.
    """
    log.info("")
    log.info("=" * 65)
    log.info("  🔍 ПРЕДСТАРТОВАЯ ПРОВЕРКА (5 ЭТАПОВ)")
    log.info("=" * 65)

    этапы = [
        ("Этап 1: Окружение и API ключи",       этап_1_проверка_окружения),
        ("Этап 2: Подключение к бирже",          этап_2_проверка_подключения),
        ("Этап 3: Конфигурация бота",            этап_3_проверка_конфигурации),
        ("Этап 4: Доступность рынка",            этап_4_проверка_рынка),
        ("Этап 5: Существующие позиции",         этап_5_проверка_существующих_позиций),
    ]

    все_ок    = True
    все_ошибки = []

    for название, функция in этапы:
        log.info(f"\n  ▶ {название}...")
        try:
            ок, сообщения = функция()
            for msg in сообщения:
                if "⚠" in msg or "рекомендуется" in msg or "Уже" in msg or "Мало" in msg:
                    log.warning(f"    ⚠️  {msg}")
                elif ок:
                    log.warning(f"    ⚠️  {msg}")
                else:
                    log.error(f"    ❌  {msg}")
                    все_ошибки.append(f"[{название}] {msg}")
            if ок:
                log.info(f"    ✅ {название} — ПРОЙДЕН")
            else:
                log.error(f"    ❌ {название} — ПРОВАЛЕН")
                все_ок = False
        except Exception as e:
            log.error(f"    💥 Исключение в {название}: {e}")
            все_ошибки.append(f"[{название}] Исключение: {e}")
            все_ок = False

    log.info("")
    log.info("=" * 65)
    if все_ок:
        log.info("  ✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ — БОТ ЗАПУСКАЕТСЯ")
    else:
        log.error("  ❌ ЕСТЬ КРИТИЧЕСКИЕ ОШИБКИ — БОТ НЕ ЗАПУСТИТСЯ")
        log.error("")
        log.error("  СПИСОК ОШИБОК:")
        for err in все_ошибки:
            log.error(f"    • {err}")
        log.error("")
        log.error("  ВАРИАНТЫ УСТРАНЕНИЯ:")
        log.error("  1. Проверьте .env файл: BYBIT_API_KEY и BYBIT_API_SECRET")
        log.error("  2. Убедитесь что API ключ имеет права на торговлю фьючерсами")
        log.error("  3. Проверьте баланс: минимум 5 USDT на фьючерсном аккаунте")
        log.error("  4. Измените параметры: TP/SL ratio должен быть ≥2:1")
        log.error("  5. Проверьте интернет-соединение и доступность api.bybit.com")
    log.info("=" * 65)
    log.info("")
    return все_ок


# ================== ИСТОРИЯ СДЕЛОК ==================
def загрузить_историю() -> list:
    if not os.path.exists(TRADES_FILE):
        return []
    try:
        with open(TRADES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def сохранить_сделку(запись: dict):
    история = загрузить_историю()
    история.append(запись)
    try:
        with open(TRADES_FILE, "w", encoding="utf-8") as f:
            # [FIX-2] default=str решает проблему bool/datetime/etc
            json.dump(история, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        log.warning(f"Не удалось сохранить сделку: {e}")


# ================== СОСТОЯНИЕ ==================
def сохранить_состояние():
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        log.warning(f"Не удалось сохранить состояние: {e}")


def загрузить_состояние():
    global stats
    if not os.path.exists(STATE_FILE):
        return False
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        for key in stats:
            if key in saved:
                stats[key] = saved[key]
        log.info(f"Состояние восстановлено из {STATE_FILE}")
        return True
    except Exception as e:
        log.warning(f"Не удалось загрузить состояние: {e}")
        return False


# ================== БАЛАНС И ПОЗИЦИИ ==================
def баланс_usdt() -> float:
    try:
        time.sleep(REQUEST_DELAY)
        b = exchange.fetch_balance({"type": "linear"})
        return float(b.get("USDT", {}).get("free", 0.0))
    except Exception as e:
        log.warning(f"Ошибка получения баланса: {e}")
        return 0.0


def полный_баланс_usdt() -> float:
    try:
        time.sleep(REQUEST_DELAY)
        b = exchange.fetch_balance({"type": "linear"})
        total = float(b.get("USDT", {}).get("total", 0.0))
        return total if total > 0 else баланс_usdt()
    except Exception as e:
        log.warning(f"Ошибка получения полного баланса: {e}")
        return баланс_usdt()


def получить_позиции() -> list:
    try:
        time.sleep(REQUEST_DELAY)
        positions = exchange.fetch_positions()
        return [p for p in positions if float(p.get("contracts", 0) or 0) > 0]
    except Exception as e:
        log.warning(f"Ошибка получения позиций: {e}")
        return []


# ================== УСТАНОВКА ПЛЕЧА ==================
def установить_плечо(symbol: str, leverage: int) -> bool:
    # Метод 1: стандартный ccxt
    try:
        time.sleep(REQUEST_DELAY)
        exchange.set_leverage(
            leverage, symbol,
            params={"buyLeverage": leverage, "sellLeverage": leverage}
        )
        log.info(f"  ⚙️  Плечо {leverage}x установлено для {symbol}")
        return True
    except Exception as e1:
        log.warning(f"  Метод 1 плеча не сработал: {e1}")

    # Метод 2: прямой v5 API
    try:
        time.sleep(REQUEST_DELAY)
        coin_sym = symbol.replace("/", "").replace(":USDT", "")
        exchange.private_post_v5_position_set_leverage({
            "category":     "linear",
            "symbol":       coin_sym,
            "buyLeverage":  str(leverage),
            "sellLeverage": str(leverage),
        })
        log.info(f"  ⚙️  Плечо {leverage}x установлено (v5) для {symbol}")
        return True
    except Exception as e2:
        log.warning(f"  Метод 2 плеча не сработал: {e2}")

    # Метод 3: проверить текущее плечо
    try:
        time.sleep(REQUEST_DELAY)
        positions = exchange.fetch_positions([symbol])
        for p in positions:
            curr_lev = int(float(p.get("leverage", 0) or 0))
            if curr_lev == leverage:
                log.info(f"  ⚙️  Плечо уже = {leverage}x для {symbol} — OK")
                return True
    except Exception as e3:
        log.warning(f"  Метод 3 проверки плеча не сработал: {e3}")

    log.error(f"  ❌ Не удалось установить плечо {leverage}x для {symbol} — сделка отменена")
    return False


# ================== SESSION FILTER ==================
def торговля_разрешена_по_времени() -> bool:
    if not SESSION_FILTER_ENABLED:
        return True
    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour
    if SESSION_BLOCK_START < SESSION_BLOCK_END:
        blocked = SESSION_BLOCK_START <= hour < SESSION_BLOCK_END
    else:
        blocked = hour >= SESSION_BLOCK_START or hour < SESSION_BLOCK_END
    if blocked:
        log.info(f"  🕐 Session Filter: час {hour} UTC заблокирован")
    return not blocked


# ================== DAILY LOSS LIMIT ==================
def обновить_начало_дня(баланс: float):
    сегодня = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if stats["дата_дня"] != сегодня:
        stats["дата_дня"]          = сегодня
        stats["баланс_начало_дня"] = баланс
        log.info(f"  📅 Новый торговый день. Баланс: {баланс:.2f} USDT")
        сохранить_состояние()


def превышен_дневной_лимит() -> bool:
    нач = stats.get("баланс_начало_дня", 0.0)
    if нач <= 0:
        return False
    текущий = полный_баланс_usdt()
    потеря_pct = (нач - текущий) / нач * 100
    if потеря_pct >= DAILY_LOSS_LIMIT_PCT:
        log.warning(f"  ⛔ Дневной лимит убытков: -{потеря_pct:.1f}% (лимит {DAILY_LOSS_LIMIT_PCT}%)")
        return True
    return False


# ================== БАЗОВЫЕ ИНДИКАТОРЫ ==================
def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _rma(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(alpha=1/span, adjust=False).mean()

def _sma(s: pd.Series, span: int) -> pd.Series:
    return s.rolling(span).mean()

def _wma(s: pd.Series, span: int) -> pd.Series:
    weights = np.arange(1, span + 1, dtype=float)
    return s.rolling(span).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def _hma(s: pd.Series, span: int) -> pd.Series:
    half = max(1, span // 2)
    sqrt_p = max(1, int(np.sqrt(span)))
    return _wma(2 * _wma(s, half) - _wma(s, span), sqrt_p)

def calc_ma(df: pd.DataFrame, ma_type: str, length: int) -> pd.Series:
    s = df["c"]
    ma_type = ma_type.upper()
    if   ma_type == "EMA":  return _ema(s, length)
    elif ma_type == "SMA":  return _sma(s, length)
    elif ma_type == "WMA":  return _wma(s, length)
    elif ma_type == "HMA":  return _hma(s, length)
    else:                   return _ema(s, length)


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0)
    loss = (-d).clip(lower=0)
    avg_g = _rma(gain, period)
    avg_l = _rma(loss, period)
    rs = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ml = _ema(close, fast) - _ema(close, slow)
    sl = _ema(ml, signal)
    return ml, sl, ml - sl


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hi, lo, pc = df["h"], df["l"], df["c"].shift(1)
    tr = pd.concat([hi - lo, (hi - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    return _rma(tr, period)


def calc_supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0):
    atr = calc_atr(df, period)
    hl2 = (df["h"] + df["l"]) / 2
    ub  = (hl2 + mult * atr).copy()
    lb  = (hl2 - mult * atr).copy()
    trend = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        c, pc = df["c"].iloc[i], df["c"].iloc[i-1]
        pu, pl, pt = ub.iloc[i-1], lb.iloc[i-1], trend.iloc[i-1]
        ub.iloc[i] = ub.iloc[i] if ub.iloc[i] < pu or pc > pu else pu
        lb.iloc[i] = lb.iloc[i] if lb.iloc[i] > pl or pc < pl else pl
        if   pt ==  1 and c < lb.iloc[i]: trend.iloc[i] = -1
        elif pt == -1 and c > ub.iloc[i]: trend.iloc[i] =  1
        else:                              trend.iloc[i] = pt
    return trend == 1, trend == -1


def calc_stochastic(df: pd.DataFrame, k: int = 14, d: int = 3, smooth: int = 3):
    lo = df["l"].rolling(k).min()
    hi = df["h"].rolling(k).max()
    ks = (100 * (df["c"] - lo) / (hi - lo + 1e-10)).rolling(smooth).mean()
    return ks, ks.rolling(d).mean()


def calc_hull(close: pd.Series, period: int = 55):
    hma = _ema(2 * _ema(close, period//2) - _ema(close, period), int(np.sqrt(period)))
    return hma > hma.shift(2), hma < hma.shift(2)


def calc_adx(df: pd.DataFrame, period: int = 14):
    atr = calc_atr(df, period)
    pdm = (df["h"] - df["h"].shift(1)).clip(lower=0)
    mdm = (df["l"].shift(1) - df["l"]).clip(lower=0)
    pdm = pdm.where(pdm >= mdm, 0)
    mdm = mdm.where(mdm >= pdm, 0)
    pdi = 100 * _rma(pdm, period) / atr.replace(0, np.nan)
    mdi = 100 * _rma(mdm, period) / atr.replace(0, np.nan)
    adx = _rma(100 * (pdi - mdi).abs() / (pdi + mdi + 1e-10), period)
    return adx, pdi, mdi


def calc_range_filter(df: pd.DataFrame, period: int = 100, qty: float = 2.5):
    close = df["c"]
    rng   = qty * calc_atr(df, period)
    filt  = close.copy()
    for i in range(1, len(close)):
        c, r, pf = close.iloc[i], rng.iloc[i], filt.iloc[i-1]
        if   c - r > pf: filt.iloc[i] = c - r
        elif c + r < pf: filt.iloc[i] = c + r
        else:            filt.iloc[i] = pf
    up   = (filt > filt.shift(1)) & (close > filt)
    down = (filt < filt.shift(1)) & (close < filt)
    return filt, filt + rng, filt - rng, up, down


def calc_support_resistance(df: pd.DataFrame, period: int = SR_PERIOD) -> dict:
    df_sr = df.tail(period).reset_index(drop=True)
    highs = df_sr["h"].values
    lows  = df_sr["l"].values
    close = float(df["c"].iloc[-1])

    raw_res, raw_sup = [], []
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            raw_res.append(highs[i])
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            raw_sup.append(lows[i])

    def кластеризовать(levels):
        if not levels:
            return []
        levels = sorted(levels)
        out = []
        cur = [levels[0]]
        for lvl in levels[1:]:
            if (lvl - cur[0]) / (cur[0] + 1e-10) < SR_CLUSTER_TOL:
                cur.append(lvl)
            else:
                out.append((float(np.mean(cur)), len(cur)))
                cur = [lvl]
        out.append((float(np.mean(cur)), len(cur)))
        return out

    res_cl = кластеризовать(raw_res)
    sup_cl = кластеризовать(raw_sup)

    res_above = sorted([(p, n) for p, n in res_cl if p > close], key=lambda x: x[0])
    sup_below = sorted([(p, n) for p, n in sup_cl if p < close], key=lambda x: x[0], reverse=True)

    nearest_res, res_n = res_above[0] if res_above else (close * 1.05, 0)
    nearest_sup, sup_n = sup_below[0] if sup_below else (close * 0.95, 0)

    dist_res = (nearest_res - close) / close * 100
    dist_sup = (close - nearest_sup) / close * 100
    near_sup = dist_sup < SR_PROXIMITY_PCT and sup_n >= SR_MIN_TOUCHES
    near_res = dist_res < SR_PROXIMITY_PCT and res_n >= SR_MIN_TOUCHES

    return {
        "support":         round(nearest_sup, 10),
        "resistance":      round(nearest_res, 10),
        "dist_to_sup_pct": round(dist_sup, 2),
        "dist_to_res_pct": round(dist_res, 2),
        "sup_cluster":     sup_n,
        "res_cluster":     res_n,
        "near_support":    near_sup,
        "near_resistance": near_res,
    }


# ================== MA КРОССОВЕР ==================
def проверить_ma_кроссовер(df: pd.DataFrame, side: str = "long") -> bool:
    if not MA_CROSSOVER_ENABLED:
        return True
    try:
        min_len = max(MA1_LENGTH, MA2_LENGTH) * 2 + 5
        if len(df) < min_len:
            return True
        ma1 = calc_ma(df, MA1_TYPE, MA1_LENGTH)
        ma2 = calc_ma(df, MA2_TYPE, MA2_LENGTH)
        ma_aligned_long  = ma1.iloc[-1] > ma2.iloc[-1]
        ma_aligned_short = ma1.iloc[-1] < ma2.iloc[-1]
        if side == "long":
            return ma_aligned_long
        else:
            return ma_aligned_short
    except Exception as e:
        log.warning(f"  Ошибка расчёта MA кроссовера: {e}")
        return True


# ================== VOLUME SPIKE GUARD ==================
def volume_spike_guard(df: pd.DataFrame) -> bool:
    try:
        vol_avg = df["v"].rolling(VOLUME_AVG_PERIOD).mean().iloc[-1]
        vol_now = df["v"].iloc[-1]
        ratio   = vol_now / (vol_avg + 1e-10)
        if ratio > VOLUME_SPIKE_MULT:
            log.info(f"  🔊 Volume Spike Guard: объём {ratio:.1f}x > {VOLUME_SPIKE_MULT}x — вход отменён")
            return False
        return True
    except Exception:
        return True


# ================== BYBIT AI СИГНАЛ ==================
def получить_bybit_ai(symbol: str) -> dict:
    result = {"signal": "neutral", "long_ratio": 0.5, "short_ratio": 0.5, "available": False}
    try:
        coin = symbol.split("/")[0]
        url  = (f"https://api.bybit.com/v5/market/account-ratio"
                f"?category=linear&symbol={coin}USDT&period=1h&limit=1")
        resp = requests.get(url, timeout=5)
        data = resp.json()
        if data.get("retCode") == 0:
            items = data.get("result", {}).get("list", [])
            if items:
                buy_r  = float(items[0].get("buyRatio",  0.5))
                sell_r = float(items[0].get("sellRatio", 0.5))
                result.update({"long_ratio": buy_r, "short_ratio": sell_r, "available": True})
                if   buy_r > 0.6: result["signal"] = "bullish"
                elif buy_r < 0.4: result["signal"] = "bearish"
    except Exception as e:
        log.debug(f"  Bybit ratio недоступен для {symbol}: {e}")
    return result


# ================== ТРЕНД 4H ==================
def тренд_4h_бычий(symbol: str) -> bool:
    try:
        time.sleep(REQUEST_DELAY)
        raw = exchange.fetch_ohlcv(symbol, TIMEFRAME_4H, limit=60)
        if len(raw) < 55:
            return False
        df    = pd.DataFrame(raw, columns=["ts","o","h","l","c","v"])
        ema20 = _ema(df["c"], 20).iloc[-1]
        ema50 = _ema(df["c"], 50).iloc[-1]
        return bool(ema20 > ema50)
    except Exception:
        return False


def тренд_4h_медвежий(symbol: str) -> bool:
    try:
        time.sleep(REQUEST_DELAY)
        raw = exchange.fetch_ohlcv(symbol, TIMEFRAME_4H, limit=60)
        if len(raw) < 55:
            return False
        df    = pd.DataFrame(raw, columns=["ts","o","h","l","c","v"])
        ema20 = _ema(df["c"], 20).iloc[-1]
        ema50 = _ema(df["c"], 50).iloc[-1]
        return bool(ema20 < ema50)
    except Exception:
        return False


# ================== ТЕХНИЧЕСКИЙ СКОР ==================
def получить_скор(symbol: str) -> dict:
    details = {}
    score   = 0
    price   = 0.0
    sr      = {}
    df_ta   = None

    try:
        time.sleep(REQUEST_DELAY)
        raw_ta = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA,    limit=300)
        time.sleep(REQUEST_DELAY)
        raw_1h = exchange.fetch_ohlcv(symbol, TIMEFRAME_TREND, limit=300)

        if len(raw_ta) < 100 or len(raw_1h) < 100:
            return {"score": 0, "details": {}, "price": 0, "sr": {}, "df_ta": None}

        cols   = ["ts","o","h","l","c","v"]
        df_ta  = pd.DataFrame(raw_ta, columns=cols).reset_index(drop=True)
        df_1h  = pd.DataFrame(raw_1h, columns=cols).reset_index(drop=True)
        c_ta   = df_ta["c"]
        c_1h   = df_1h["c"]
        price  = float(c_ta.iloc[-1])

        # --- RSI на таймфрейме TA ---
        rsi_val = calc_rsi(c_ta).iloc[-1]
        details["rsi"] = round(rsi_val, 1)
        if   25 <= rsi_val <= 40: score += 20
        elif 40 <  rsi_val <= 50: score += 12
        elif rsi_val < 25:        score += 10   # перепроданность — осторожно
        elif 50 <  rsi_val <= 60: score +=  5

        # --- RSI 1h ---
        rsi_1h = calc_rsi(c_1h).iloc[-1]
        details["rsi_1h"] = round(rsi_1h, 1)
        if   rsi_1h < 50: score += 10
        elif rsi_1h < 60: score +=  5

        # --- MACD ---
        ml, sl_macd, _ = calc_macd(c_ta)
        macd_bull  = ml.iloc[-1] > sl_macd.iloc[-1]
        macd_cross = macd_bull and ml.iloc[-2] <= sl_macd.iloc[-2]
        details["macd"] = "бычий" if macd_bull else "медвежий"
        if   macd_cross: score += 18
        elif macd_bull:  score +=  8

        # --- Range Filter ---
        _, _, _, rf_up, rf_down = calc_range_filter(df_ta)
        rf_up_now = rf_up.iloc[-1]
        details["range_filter"] = "вверх" if rf_up_now else ("вниз" if rf_down.iloc[-1] else "бок")
        if rf_up_now:
            score += 15

        # --- Supertrend ---
        st_up, _ = calc_supertrend(df_ta)
        details["supertrend"] = "вверх" if st_up.iloc[-1] else "вниз"
        if st_up.iloc[-1]:
            score += 12

        # --- Hull MA ---
        hu_up, _ = calc_hull(c_ta)
        details["hull"] = "вверх" if hu_up.iloc[-1] else "вниз"
        if hu_up.iloc[-1]:
            score += 8

        # --- Тренд 1h EMA ---
        ema50_1h  = _ema(c_1h, 50).iloc[-1]
        ema200_1h = _ema(c_1h, 200).iloc[-1]
        details["тренд_1h"] = "бычий" if ema50_1h > ema200_1h else "медвежий"
        if ema50_1h > ema200_1h:
            score += 10

        # --- ADX ---
        adx, pdi, mdi = calc_adx(df_ta)
        adx_val = adx.iloc[-1]
        details["adx"] = round(adx_val, 1)
        # [FIX] Требуем ADX>25 для подтверждения тренда
        if   adx_val > 25 and pdi.iloc[-1] > mdi.iloc[-1]: score += 10
        elif adx_val > 20 and pdi.iloc[-1] > mdi.iloc[-1]: score +=  4

        # --- Stochastic ---
        k_ser, _ = calc_stochastic(df_ta)
        k_val = k_ser.iloc[-1]
        details["stoch_k"] = round(k_val, 1)
        if   k_val < 20: score += 10
        elif k_val < 40: score +=  5

        # --- Volume ---
        vol_avg   = df_ta["v"].rolling(VOLUME_AVG_PERIOD).mean().iloc[-1]
        vol_ratio = df_ta["v"].iloc[-1] / (vol_avg + 1e-10)
        details["объём_ratio"] = round(vol_ratio, 2)
        if   vol_ratio > 1.5: score += 8
        elif vol_ratio > 1.2: score += 4

        # --- S/R уровни ---
        sr = calc_support_resistance(df_ta)
        details.update({
            "support":    sr["support"],
            "resistance": sr["resistance"],
            "dist_sup":   sr["dist_to_sup_pct"],
            "dist_res":   sr["dist_to_res_pct"],
        })
        if sr["near_support"]:
            score += 15
            details["sr_signal"] = f"у поддержки ✅ ({sr['sup_cluster']} касаний)"
        elif sr["near_resistance"]:
            score -= 25  # [FIX] Сильнее штрафуем за сопротивление
            details["sr_signal"] = f"у сопротивления ❌ ({sr['res_cluster']} касаний)"
        else:
            details["sr_signal"] = f"нейтр (sup={sr['dist_to_sup_pct']:.2f}% res={sr['dist_to_res_pct']:.2f}%)"

        # --- Несколько красных свечей подряд ---
        last3_bearish = all(df_ta["c"].iloc[-i] < df_ta["o"].iloc[-i] for i in range(1, 4))
        if last3_bearish:
            score -= 20
            details["свечи_3red"] = True

        # --- MA Кроссовер (флаг) ---
        details["ma_cross"] = проверить_ma_кроссовер(df_ta, side="long")

        # --- Volume Spike Guard (флаг) ---
        details["vol_spike_ok"] = volume_spike_guard(df_ta)

        score = max(0, min(100, score))

    except Exception as e:
        log.warning(f"Ошибка анализа {symbol}: {e}")

    return {"score": score, "details": details, "price": price, "sr": sr, "df_ta": df_ta}


def получить_скор_шорта(symbol: str) -> dict:
    res = получить_скор(symbol)
    if res["score"] == 0:
        return res
    inverted = max(0, 100 - res["score"] - 10)
    res["score"] = inverted
    if res.get("df_ta") is not None:
        res["details"]["ma_cross"] = проверить_ma_кроссовер(res["df_ta"], side="short")
    return res


# ================== AI КОРРЕКТИРОВКА ==================
def применить_ai_корректировку(score: int, symbol: str) -> int:
    ai = получить_bybit_ai(symbol)
    if not ai["available"]:
        return score
    long_r = ai["long_ratio"]
    signal = ai["signal"]
    log.info(f"  🤖 Bybit ratio: long={long_r:.1%} short={ai['short_ratio']:.1%} сигнал={signal}")
    if signal == "bullish":
        return min(100, score + 5)
    elif signal == "bearish":
        return max(0, score - 15)
    return score


# ================== РАЗМЕР ПОЗИЦИИ ==================
def рассчитать_размер_позиции(score: int, баланс: float, sl_dist_pct: float) -> float:
    factor   = max(0, (score - MIN_SCORE)) / (100 - MIN_SCORE)
    risk_pct = BASE_RISK_PCT + (MAX_RISK_PCT - BASE_RISK_PCT) * factor
    risk_pct = min(risk_pct, MAX_RISK_PCT)

    max_loss_usdt = баланс * risk_pct / 100
    margin_usdt   = max_loss_usdt / (sl_dist_pct / 100)

    log.info(
        f"  📐 Скор={score} → риск={risk_pct:.1f}%  SL_dist={sl_dist_pct:.2f}%  "
        f"макс.убыток={max_loss_usdt:.2f}U → маржа={margin_usdt:.2f}U"
    )
    return round(max(1.0, margin_usdt), 2)


# ================== ПОДТВЕРЖДЕНИЕ ВХОДА ==================
def подтвердить_вход(symbol: str, исходный_скор: int, side: str = "long") -> bool:
    if ENTRY_CONFIRM_BARS <= 0:
        return True

    tf_seconds = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600}
    wait = tf_seconds.get(TIMEFRAME_TA, 900) * ENTRY_CONFIRM_BARS

    log.info(f"  ⏳ Подтверждение входа: ждём {wait}с ({ENTRY_CONFIRM_BARS} свеча)...")
    time.sleep(wait)

    if side == "long":
        новый = получить_скор(symbol)
    else:
        новый = получить_скор_шорта(symbol)

    новый_скор = новый["score"]
    log.info(f"  🔍 Перепроверка скора: {исходный_скор} → {новый_скор} (мин={ENTRY_CONFIRM_MIN_SCORE})")

    if новый_скор < ENTRY_CONFIRM_MIN_SCORE:
        log.info(f"  ❌ Подтверждение не прошло: скор упал до {новый_скор}")
        return False

    det = новый.get("details", {})
    if not det.get("vol_spike_ok", True):
        log.info("  ❌ Подтверждение не прошло: volume spike при перепроверке")
        return False

    log.info(f"  ✅ Вход подтверждён. Скор {новый_скор}/100")
    return True


# ================== ОБНОВЛЕНИЕ SL [FIX-1] ==================
def обновить_sl_на_бирже(symbol: str, new_sl: float, side: str = "long") -> bool:
    """
    [FIX-1] Использует только рабочий v5 fallback метод.
    Убран битый set_trading_stop.
    """
    try:
        sl_str   = exchange.price_to_precision(symbol, new_sl)
        coin_sym = symbol.replace("/", "").replace(":USDT", "")
        time.sleep(REQUEST_DELAY)
        exchange.private_post_v5_position_trading_stop({
            "category":    "linear",
            "symbol":      coin_sym,
            "stopLoss":    sl_str,
            "slTriggerBy": "MarkPrice",
            "positionIdx": "0",
        })
        log.info(f"  🔧 SL обновлён (v5) → {sl_str}")
        return True
    except Exception as e:
        log.warning(f"  ⚠️ Не удалось обновить SL: {e}")
        return False


# ================== ОТКРЫТИЕ ПОЗИЦИИ ==================
def открыть_позицию(symbol: str, margin_usdt: float, tp_price: float,
                    sl_price: float, side: str = "long"):
    try:
        if not установить_плечо(symbol, LEVERAGE):
            log.error(f"  ❌ Плечо не установлено — сделка отменена для {symbol}")
            return None, None

        time.sleep(REQUEST_DELAY)
        ticker       = exchange.fetch_ticker(symbol)
        price        = float(ticker["last"])
        pos_size_usdt= margin_usdt * LEVERAGE
        qty_raw      = pos_size_usdt / price
        qty          = float(exchange.amount_to_precision(symbol, qty_raw))

        if qty <= 0:
            log.error(f"  Нулевое количество {symbol}")
            return None, None

        tp_str = exchange.price_to_precision(symbol, tp_price)
        sl_str = exchange.price_to_precision(symbol, sl_price)

        buy_sell = "buy" if side == "long" else "sell"
        log.info(
            f"  Открываем {side} {symbol}: qty={qty}, маржа≈{margin_usdt:.2f}U, "
            f"плечо={LEVERAGE}x, TP={tp_str}, SL={sl_str}"
        )

        time.sleep(REQUEST_DELAY)
        order = exchange.create_market_order(
            symbol, buy_sell, qty,
            params={"takeProfit": float(tp_str), "stopLoss": float(sl_str)}
        )

        entry_price = price
        try:
            if order.get("average") and float(order["average"]) > 0:
                entry_price = float(order["average"])
        except Exception:
            pass

        log.info(f"  📈 {side.upper()} открыт: {qty} {symbol} @ ~{entry_price:.8f}")
        return entry_price, qty

    except Exception as e:
        log.error(f"  ❌ Ошибка открытия {side}: {e}")
        return None, None


# ================== ПРОВЕРКА SIGNAL EXIT ==================
def проверить_signal_exit(symbol: str, side: str) -> bool:
    if not SIGNAL_EXIT_ENABLED:
        return False
    try:
        time.sleep(REQUEST_DELAY)
        raw = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA, limit=50)
        if len(raw) < 30:
            return False
        df = pd.DataFrame(raw, columns=["ts","o","h","l","c","v"])
        st_up, st_down = calc_supertrend(df)
        _, _, _, rf_up, rf_down = calc_range_filter(df)
        if side == "long":
            return bool(st_down.iloc[-1] and rf_down.iloc[-1])
        else:
            return bool(st_up.iloc[-1] and rf_up.iloc[-1])
    except Exception:
        return False


# ================== МОНИТОРИНГ ПОЗИЦИИ ==================
def мониторить_позицию(symbol: str, entry_price: float, qty: float,
                        открыта_в: float, sl_цена: float,
                        tp_цена: float, side: str = "long") -> str:
    deadline     = открыта_в + TRADE_MAX_LIFETIME
    coin         = symbol.split("/")[0]
    breakeven_price = (
        entry_price * (1 + BYBIT_FEE * 2 + 0.0005) if side == "long"
        else entry_price * (1 - BYBIT_FEE * 2 - 0.0005)
    )

    # Рассчитываем адаптивный трейлинг по ATR
    trailing_step   = MIN_TRAILING_STEP   / 100
    trailing_offset = MIN_TRAILING_OFFSET / 100
    try:
        time.sleep(REQUEST_DELAY)
        raw = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA, limit=50)
        if len(raw) >= 30:
            df      = pd.DataFrame(raw, columns=["ts","o","h","l","c","v"])
            atr_val = calc_atr(df, TRAILING_ATR_PERIOD).iloc[-1]
            atr_pct = (atr_val / entry_price) * 100
            trailing_step   = max(MIN_TRAILING_STEP,   atr_pct * TRAILING_ATR_MULT)   / 100
            trailing_offset = max(MIN_TRAILING_OFFSET, atr_pct * TRAILING_OFFSET_MULT) / 100
            log.info(
                f"  📊 ATR={atr_pct:.2f}% → шаг {trailing_step*100:.2f}%  "
                f"отступ {trailing_offset*100:.2f}%"
            )
    except Exception as e:
        log.warning(f"  Не удалось рассчитать ATR для трейлинга: {e}")

    # rrExit триггер
    if side == "long":
        rr_trigger_price = entry_price + (tp_цена - entry_price) * RR_EXIT_TRIGGER
    else:
        rr_trigger_price = entry_price - (entry_price - tp_цена) * RR_EXIT_TRIGGER
    log.info(f"  🎯 rrExit триггер={rr_trigger_price:.8f}  (RR_EXIT={RR_EXIT_TRIGGER})")

    фаза             = 1
    текущий_sl       = sl_цена
    пиковая_цена     = entry_price
    trailing_активен = (RR_EXIT_TRIGGER == 0.0)
    оставшийся_qty   = qty

    log.info(
        f"  🚦 Мониторинг {coin} {side}  вход={entry_price:.8f}  "
        f"SL={sl_цена:.8f}  TP={tp_цена:.8f}  breakeven={breakeven_price:.8f}"
    )

    while True:
        сейчас = time.time()

        if сейчас >= deadline:
            log.warning("  ⏰ Дедлайн — принудительное закрытие")
            try:
                close_side = "sell" if side == "long" else "buy"
                time.sleep(REQUEST_DELAY)
                exchange.create_market_order(
                    symbol, close_side, оставшийся_qty,
                    params={"reduceOnly": True}
                )
            except Exception as e:
                log.warning(f"  Ошибка закрытия по дедлайну: {e}")
            return "таймаут"

        time.sleep(15)  # [FIX] Увеличен интервал проверки с 10 до 15 сек

        try:
            time.sleep(REQUEST_DELAY)
            positions = exchange.fetch_positions([symbol])
            active = [
                p for p in positions
                if float(p.get("contracts", 0) or 0) > 0 and p.get("side") == side
            ]

            if not active:
                time.sleep(REQUEST_DELAY)
                cur_price = float(exchange.fetch_ticker(symbol)["last"])
                if side == "long":
                    hit_tp = cur_price >= entry_price * (1 + TP_PERCENT / 100 * 0.7)
                else:
                    hit_tp = cur_price <= entry_price * (1 - TP_PERCENT / 100 * 0.7)
                if hit_tp or фаза >= 2:
                    log.info("  ✅ Позиция закрыта: TP или трейлинг")
                    return "tp"
                else:
                    log.info("  ❌ Позиция закрыта по SL")
                    return "sl"

            pos       = active[0]
            time.sleep(REQUEST_DELAY)
            cur_price = float(exchange.fetch_ticker(symbol)["last"])
            оставшийся_qty = abs(float(pos.get("contracts", 0) or 0))

            if side == "long":
                pnl_pct = (cur_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - cur_price) / entry_price * 100

            до_дед = int(deadline - сейчас)

            # === SIGNAL EXIT (только в прибыльной фазе) ===
            if SIGNAL_EXIT_ENABLED and фаза >= 2 and проверить_signal_exit(symbol, side):
                log.info(f"  🔄 Signal Exit: Supertrend+RangeFilter развернулись — закрываем")
                try:
                    close_side = "sell" if side == "long" else "buy"
                    time.sleep(REQUEST_DELAY)
                    exchange.create_market_order(
                        symbol, close_side, оставшийся_qty,
                        params={"reduceOnly": True}
                    )
                except Exception as e:
                    log.warning(f"  Ошибка signal exit: {e}")
                return "tp" if pnl_pct > 0 else "sl"

            # === БЕЗУБЫТОК ===
            if фаза == 1 and pnl_pct >= 0.3:
                if side == "long":
                    new_sl_be = entry_price * (1 + BYBIT_FEE * 2 + 0.0003)
                else:
                    new_sl_be = entry_price * (1 - BYBIT_FEE * 2 - 0.0003)
                if обновить_sl_на_бирже(symbol, new_sl_be, side):
                    фаза       = 2
                    текущий_sl = new_sl_be
                    пиковая_цена = cur_price
                    log.info(f"  🔒 БЕЗУБЫТОК! SL → {new_sl_be:.8f}")

            # === АКТИВАЦИЯ ТРЕЙЛИНГА ===
            if not trailing_активен and фаза >= 2:
                if side == "long":
                    trailing_активен = cur_price >= rr_trigger_price
                else:
                    trailing_активен = cur_price <= rr_trigger_price
                if trailing_активен:
                    log.info(f"  📈 Трейлинг активирован @ {cur_price:.8f}")

            # === ТРЕЙЛИНГ ===
            if trailing_активен and фаза >= 2 and pnl_pct >= MIN_PROFIT_FOR_TRAIL:
                if side == "long":
                    if cur_price > пиковая_цена:
                        пиковая_цена = cur_price
                    new_sl_trail = пиковая_цена * (1 - trailing_offset)
                    if new_sl_trail > текущий_sl:
                        if обновить_sl_на_бирже(symbol, new_sl_trail, side):
                            текущий_sl = new_sl_trail
                            log.info(f"  📈 ТРЕЙЛИНГ: пик={пиковая_цена:.8f} → SL={new_sl_trail:.8f}")
                else:
                    if cur_price < пиковая_цена:
                        пиковая_цена = cur_price
                    new_sl_trail = пиковая_цена * (1 + trailing_offset)
                    if new_sl_trail < текущий_sl:
                        if обновить_sl_на_бирже(symbol, new_sl_trail, side):
                            текущий_sl = new_sl_trail
                            log.info(f"  📈 ТРЕЙЛИНГ: пик={пиковая_цена:.8f} → SL={new_sl_trail:.8f}")

            фаза_лейбл  = {1: "обычная", 2: "безубыток 🔒", 3: "трейлинг 📈"}.get(фаза, "?")
            trail_лейбл = "🟢" if trailing_активен else "⏸"
            log.info(
                f"  [{coin}] {cur_price:.8f}  P&L={pnl_pct:+.2f}%"
                f"  SL={текущий_sl:.8f}  фаза={фаза_лейбл}  trail={trail_лейбл}  дед={до_дед}с"
                f"  qty={оставшийся_qty:.4f}"
            )

        except Exception as e:
            log.warning(f"  Ошибка в цикле мониторинга: {e}")


# ================== ПОСТ-ТРЕЙД АНАЛИЗ ==================
def пост_трейд_анализ(запись: dict):
    r   = запись["результат"]
    sym = запись["symbol"]
    pnl = запись.get("pnl_usdt", 0)
    dur = запись.get("duration_min", 0)
    знак = "✅" if r == "tp" else ("❌" if r == "sl" else "⏰")
    log.info("")
    log.info("━" * 60)
    log.info(f"  📋 ПОСТ-ТРЕЙД: {sym.split(':')[0]}  {знак} {r.upper()}"
             f"  P&L: {pnl:+.4f} USDT  Длит: {dur:.1f} мин")
    log.info(f"  Скор входа: {запись.get('score', '?')}/100")
    log.info("━" * 60)
    log.info("")


# ================== ОТЧЁТ ==================
def печатать_отчёт():
    баланс = полный_баланс_usdt()
    старт  = stats["депозит_старт"]
    дельта = баланс - старт
    чистый = stats["прибыль_usdt"] - stats["убыток_usdt"]
    пct    = (дельта / старт * 100) if старт > 0 else 0
    всего  = stats["сделок_всего"]
    tp_    = stats["тейкпрофит"]
    sl_    = stats["стоплосс"]
    wr     = (tp_ / всего * 100) if всего > 0 else 0.0

    log.info("")
    log.info("=" * 65)
    log.info("  📊 ОТЧЁТ ГИБРИДНОГО БОТА v6")
    log.info(f"  Баланс:            {баланс:.2f} USDT  ({дельта:+.2f} USDT / {пct:+.2f}%)")
    log.info(f"  Сделок:            {всего}  TP={tp_}  SL={sl_}  Таймаут={stats['таймаут']}")
    log.info(f"  WinRate:           {wr:.1f}%")
    log.info(f"  Прибыль/Убыток:    {stats['прибыль_usdt']:.4f} / {stats['убыток_usdt']:.4f} USDT")
    log.info(f"  Чистый P&L:        {чистый:+.4f} USDT")
    log.info("=" * 65)
    log.info("")
    stats["последний_отчёт"] = time.time()
    сохранить_состояние()


# ================== ГЛАВНЫЙ ЦИКЛ ==================
def main():
    global stats

    # [FIX-7] Предстартовая проверка — бот не запустится при ошибках
    if not запустить_предстартовую_проверку():
        log.error("  🛑 Бот остановлен из-за ошибок предстартовой проверки.")
        return

    загрузить_состояние()
    stats["запусков"] += 1

    баланс_сейчас = полный_баланс_usdt()
    if stats["депозит_старт"] <= 0:
        stats["депозит_старт"] = баланс_сейчас
    if not stats["старт_время"]:
        stats["старт_время"] = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

    обновить_начало_дня(баланс_сейчас)
    сохранить_состояние()

    log.info("")
    log.info("=" * 65)
    log.info("  🤖  ГИБРИДНЫЙ ФЬЮЧЕРСНЫЙ БОТ v6 (FIXED)")
    log.info(f"  Плечо:       {LEVERAGE}x  |  RR: {TP_PERCENT}/{SL_PERCENT} ({TP_PERCENT/SL_PERCENT:.1f}:1)")
    log.info(f"  Баланс:      {баланс_сейчас:.4f} USDT")
    log.info(f"  MIN_SCORE:   {MIN_SCORE}  |  Пар: {len(SYMBOLS)}")
    log.info(f"  SL ATR:      {ATR_SL_MULT}x ATR  |  TP ATR: {ATR_TP_MULT}x ATR")
    log.info(f"  Блок SL:     {SYMBOL_BLOCK_AFTER_SL} мин  |  Блок TP: {SYMBOL_BLOCK_AFTER_TP} мин")
    log.info("=" * 65)
    log.info("")

    # Словарь блокировок: symbol → время_разблокировки (unix timestamp)
    заблокированные: dict = {}

    while True:
        try:
            # --- Отчёт ---
            if time.time() - stats["последний_отчёт"] >= REPORT_INTERVAL:
                печатать_отчёт()

            баланс    = полный_баланс_usdt()
            свободный = баланс_usdt()
            обновить_начало_дня(баланс)

            # --- Минимальный баланс ---
            if свободный < MIN_BALANCE:
                log.warning(f"  🛑 Свободный баланс {свободный:.2f} < {MIN_BALANCE}. Пауза 10 мин.")
                time.sleep(600)
                continue

            # --- Максимальная просадка ---
            if stats["депозит_старт"] > 0:
                просадка = (stats["депозит_старт"] - баланс) / stats["депозит_старт"] * 100
                if просадка > MAX_DRAWDOWN_PCT:
                    log.warning(f"  ⛔ Просадка {просадка:.1f}% > {MAX_DRAWDOWN_PCT}%. Пауза 2ч.")
                    time.sleep(7200)
                    continue

            # --- Дневной лимит ---
            if превышен_дневной_лимит():
                log.warning(f"  ⛔ Дневной лимит. Пауза {DAILY_LOSS_PAUSE_SEC//60} мин.")
                time.sleep(DAILY_LOSS_PAUSE_SEC)
                continue

            # --- Session Filter ---
            if not торговля_разрешена_по_времени():
                log.info("  🕐 Заблокировано по времени. Пауза 5 мин.")
                time.sleep(300)
                continue

            # --- SL Streak cooldown ---
            if stats.get("sl_streak", 0) >= SL_STREAK_LIMIT:
                log.warning(f"  🧊 {SL_STREAK_LIMIT} SL подряд — cooldown {SL_STREAK_PAUSE//60} мин.")
                stats["sl_streak"] = 0
                сохранить_состояние()
                time.sleep(SL_STREAK_PAUSE)
                continue

            # --- Активные позиции ---
            активные = получить_позиции()
            if активные:
                log.info(f"  ⏳ Открытые позиции: {[p['symbol'] for p in активные]} — ждём")
                time.sleep(60)
                continue

            log.info(
                f"── Сканирование {len(SYMBOLS)} пар "
                f"(баланс={свободный:.2f}U | порог={MIN_SCORE}) ──"
            )

            # ==================== СКАНИРОВАНИЕ ====================
            scores = {}
            for sym in SYMBOLS:
                try:
                    # [FIX-3] Проверка блокировки
                    if sym in заблокированные:
                        ост = заблокированные[sym] - time.time()
                        if ост > 0:
                            log.info(f"  🚫 {sym.split(':')[0]} заблокирован ещё {ост/60:.0f} мин")
                            continue
                        else:
                            del заблокированные[sym]

                    # Тренд 4h
                    if not тренд_4h_бычий(sym):
                        continue

                    res      = получить_скор(sym)
                    ai_score = применить_ai_корректировку(res["score"], sym)
                    res["score_final"] = ai_score
                    scores[sym] = res

                    det = res.get("details", {})
                    sr  = res.get("sr", {})
                    log.info(
                        f"  {sym.split(':')[0]:12s}  скор={ai_score:3d}/100"
                        f"  rsi={det.get('rsi', '?'):5}"
                        f"  rf={det.get('range_filter', '?'):4}"
                        f"  st={det.get('supertrend', '?'):4}"
                        f"  ma_cross={'✅' if det.get('ma_cross', False) else '❌'}"
                        f"  vol={'✅' if det.get('vol_spike_ok', True) else '🔊'}"
                        f"  SR={det.get('sr_signal', '')}"
                    )
                except Exception as e:
                    log.warning(f"  Ошибка скора {sym}: {e}")
                    time.sleep(1)  # [FIX-6] пауза после ошибки rate limit

            if not scores:
                log.info(f"  Нет кандидатов — ждём {SCAN_INTERVAL} сек")
                time.sleep(SCAN_INTERVAL)
                continue

            # ==================== ВЫБОР ЛОНГ-КАНДИДАТА ====================
            кандидаты = sorted(
                [(s, d) for s, d in scores.items() if d.get("score_final", 0) >= MIN_SCORE],
                key=lambda x: x[1]["score_final"],
                reverse=True
            )[:5]

            выбрана  = None
            фин_скор = 0
            цена     = 0.0
            sr_info  = {}
            side     = "long"
            df_выб   = None

            for лучшая, данные in кандидаты:
                фин_скор = данные["score_final"]
                цена     = данные["price"]
                sr_info  = данные.get("sr", {})
                det      = данные.get("details", {})
                df_выб   = данные.get("df_ta")

                # S/R фильтр
                dist_res_now = sr_info.get("dist_to_res_pct", 99)
                if sr_info.get("near_resistance") and dist_res_now < SR_BLOCK_DIST_PCT:
                    log.info(f"  ⛔ {лучшая.split(':')[0]}: resistance {dist_res_now:.3f}% — пропуск")
                    continue

                # RSI фильтр
                rsi_val = float(det.get("rsi", 50) or 50)
                if rsi_val > 65 and not sr_info.get("near_support"):
                    log.info(f"  ⚠️ {лучшая.split(':')[0]}: RSI={rsi_val:.1f} перекуплен — пропуск")
                    continue

                # MA Кроссовер
                if MA_CROSSOVER_ENABLED and not det.get("ma_cross", True):
                    log.info(f"  ⚠️ {лучшая.split(':')[0]}: MA кроссовер не подтверждён — пропуск")
                    continue

                # Volume Spike
                if not det.get("vol_spike_ok", True):
                    continue

                выбрана = лучшая
                log.info(
                    f"  ► Выбрана {лучшая.split(':')[0]} (лонг)  скор={фин_скор}  "
                    f"цена={цена:.8f}  dist_res={dist_res_now:.3f}%"
                )
                break

            # ==================== ШОРТ-КАНДИДАТ ====================
            if выбрана is None:
                for sym in SYMBOLS:
                    if sym in заблокированные and time.time() < заблокированные[sym]:
                        continue
                    if тренд_4h_медвежий(sym):
                        short_res = получить_скор_шорта(sym)
                        if short_res["score"] >= MIN_SCORE:
                            det_sh = short_res.get("details", {})
                            if MA_CROSSOVER_ENABLED and not det_sh.get("ma_cross", True):
                                continue
                            if not det_sh.get("vol_spike_ok", True):
                                continue
                            log.info(f"  🐻 Шорт-кандидат: {sym.split(':')[0]} скор={short_res['score']}")
                            выбрана = sym
                            фин_скор = short_res["score"]
                            цена     = short_res["price"]
                            sr_info  = short_res.get("sr", {})
                            df_выб   = short_res.get("df_ta")
                            side     = "short"
                            break

            if выбрана is None:
                log.info(f"  Нет кандидатов — ждём {SCAN_INTERVAL} сек")
                time.sleep(SCAN_INTERVAL)
                continue

            # ==================== РАСЧЁТ SL / TP [FIX-4, FIX-5] ====================
            atr_пт  = 0.0
            atr_pct = 1.5
            try:
                time.sleep(REQUEST_DELAY)
                raw_atr = exchange.fetch_ohlcv(выбрана, TIMEFRAME_TA, limit=50)
                if len(raw_atr) >= 20:
                    df_atr  = pd.DataFrame(raw_atr, columns=["ts","o","h","l","c","v"])
                    atr_пт  = float(calc_atr(df_atr, 14).iloc[-1])
                    atr_pct = (atr_пт / цена) * 100
            except Exception:
                pass

            if side == "long":
                # SL: ATR * множитель, но не менее MIN_SL и не более MAX_SL
                sl_atr_dist = atr_пт * ATR_SL_MULT if atr_пт > 0 else цена * SL_PERCENT / 100
                sl_pct_dist = (sl_atr_dist / цена) * 100
                sl_pct_dist = max(MIN_SL_PERCENT, min(MAX_SL_PERCENT, sl_pct_dist))
                sl_цена     = цена * (1 - sl_pct_dist / 100)

                # TP: минимум TP_PERCENT, или ATR * TP множитель
                tp_atr_dist = atr_пт * ATR_TP_MULT if atr_пт > 0 else цена * TP_PERCENT / 100
                tp_pct_dist = max(TP_PERCENT, (tp_atr_dist / цена) * 100)
                tp_цена     = цена * (1 + tp_pct_dist / 100)

                # Поддержка как ориентир SL
                support = sr_info.get("support", sl_цена)
                if support < sl_цена and support > цена * 0.97:
                    sl_цена = support * 0.998

            else:  # short
                sl_atr_dist = atr_пт * ATR_SL_MULT if atr_пт > 0 else цена * SL_PERCENT / 100
                sl_pct_dist = (sl_atr_dist / цена) * 100
                sl_pct_dist = max(MIN_SL_PERCENT, min(MAX_SL_PERCENT, sl_pct_dist))
                sl_цена     = цена * (1 + sl_pct_dist / 100)

                tp_atr_dist = atr_пт * ATR_TP_MULT if atr_пт > 0 else цена * TP_PERCENT / 100
                tp_pct_dist = max(TP_PERCENT, (tp_atr_dist / цена) * 100)
                tp_цена     = цена * (1 - tp_pct_dist / 100)

                resistance = sr_info.get("resistance", sl_цена)
                if resistance > sl_цена and resistance < цена * 1.03:
                    sl_цена = resistance * 1.002

            # Реальное расстояние до SL
            sl_dist_pct = abs(цена - sl_цена) / цена * 100

            # Реальное RR
            real_rr = abs(tp_цена - цена) / abs(цена - sl_цена)
            log.info(f"  📐 ATR={atr_pct:.2f}%  SL={sl_dist_pct:.2f}%  RR={real_rr:.1f}:1")

            # [FIX] Не входить если RR < 2:1
            if real_rr < 2.0:
                log.warning(f"  ⛔ RR={real_rr:.1f}:1 < 2:1 — пропуск {выбрана.split(':')[0]}")
                time.sleep(SCAN_INTERVAL)
                continue

            margin = рассчитать_размер_позиции(фин_скор, свободный, sl_dist_pct)
            if свободный < margin * 1.1:
                log.warning(f"  ⚠️ Баланс {свободный:.2f} < маржа {margin:.2f} — уменьшаем")
                margin = свободный * 0.8

            log.info(
                f"  ✅ ВХОД {side.upper()}: скор={фин_скор} | SL={sl_цена:.8f} | "
                f"TP={tp_цена:.8f} | sl_dist={sl_dist_pct:.2f}% | маржа={margin:.2f}U"
            )

            # ==================== ПОДТВЕРЖДЕНИЕ ВХОДА ====================
            if ENTRY_CONFIRM_BARS > 0:
                if not подтвердить_вход(выбрана, фин_скор, side):
                    log.info(f"  ⛔ Вход в {выбрана} отменён по подтверждению")
                    time.sleep(30)
                    continue

            # ==================== ОТКРЫТИЕ ПОЗИЦИИ ====================
            баланс_до = полный_баланс_usdt()  # [FIX-8] Реальный PnL
            время_входа = time.time()
            вход_цена, кол_во = открыть_позицию(выбрана, margin, tp_цена, sl_цена, side)

            if вход_цена is None or кол_во is None:
                log.warning("  Не удалось открыть позицию — пауза 30 сек")
                time.sleep(30)
                continue

            stats["сделок_всего"] += 1
            сохранить_состояние()

            # ==================== МОНИТОРИНГ ====================
            результат = "sl"
            try:
                результат = мониторить_позицию(
                    выбрана, вход_цена, кол_во,
                    время_входа, sl_цена, tp_цена, side
                )
            except Exception as monitor_err:
                log.error(f"  💥 Краш мониторинга: {monitor_err}", exc_info=True)
                try:
                    close_side = "sell" if side == "long" else "buy"
                    time.sleep(REQUEST_DELAY)
                    exchange.create_market_order(
                        выбрана, close_side, кол_во,
                        params={"reduceOnly": True}
                    )
                    log.info(f"  ✅ {выбрана} закрыт принудительно")
                except Exception as close_err:
                    log.error(f"  ❌ Не удалось закрыть после краша: {close_err}")
                результат = "sl"

            # ==================== РЕЗУЛЬТАТ [FIX-8] ====================
            time.sleep(3)
            баланс_после = полный_баланс_usdt()
            pnl_реальный = баланс_после - баланс_до  # Реальный PnL с биржи

            длит_мин = (time.time() - время_входа) / 60

            if результат == "tp":
                stats["тейкпрофит"]   += 1
                stats["прибыль_usdt"] += max(0, pnl_реальный)
                stats["sl_streak"]     = 0
                log.info(f"  ✅ TP: прибыль ≈{pnl_реальный:+.4f} USDT (реальный)")
                # [FIX-3] Блок после TP
                заблокированные[выбрана] = time.time() + SYMBOL_BLOCK_AFTER_TP * 60
                log.info(f"  🔒 {выбрана.split(':')[0]} заблокирован на {SYMBOL_BLOCK_AFTER_TP} мин после TP")

            elif результат == "sl":
                stats["стоплосс"]    += 1
                stats["убыток_usdt"] += abs(min(0, pnl_реальный))
                stats["sl_streak"]    = stats.get("sl_streak", 0) + 1
                # [FIX-3] Блок после SL — дольше чем после TP
                заблокированные[выбрана] = time.time() + SYMBOL_BLOCK_AFTER_SL * 60
                log.warning(
                    f"  ❌ SL: убыток ≈{pnl_реальный:+.4f} USDT (реальный)  "
                    f"streak: {stats['sl_streak']}/{SL_STREAK_LIMIT}  "
                    f"блок {SYMBOL_BLOCK_AFTER_SL} мин"
                )

            elif результат == "таймаут":
                stats["таймаут"]     += 1
                stats["убыток_usdt"] += abs(min(0, pnl_реальный))
                stats["sl_streak"]    = 0
                заблокированные[выбрана] = time.time() + SYMBOL_BLOCK_AFTER_TP * 60
                log.warning(f"  ⏰ Таймаут: P&L ≈{pnl_реальный:+.4f} USDT")

            запись = {
                "id":           stats["сделок_всего"],
                "время_входа":  datetime.fromtimestamp(время_входа).strftime("%d.%m.%Y %H:%M:%S"),
                "время_выхода": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                "duration_min": round(длит_мин, 1),
                "symbol":       выбрана,
                "side":         side,
                "score":        фин_скор,
                "entry_price":  вход_цена,
                "sl_price":     sl_цена,
                "tp_price":     tp_цена,
                "sl_dist_pct":  round(sl_dist_pct, 3),
                "margin_usdt":  margin,
                "leverage":     LEVERAGE,
                "результат":    результат,
                "pnl_usdt":     round(pnl_реальный, 4),
                "rr_ratio":     round(real_rr, 2),
                "sr":           {k: str(v) for k, v in (sr_info or {}).items()},
            }
            сохранить_сделку(запись)
            пост_трейд_анализ(запись)
            сохранить_состояние()
            log.info("  Сделка завершена — пауза 30 сек")
            time.sleep(30)

        except Exception as e:
            log.error(f"Глобальная ошибка главного цикла: {e}", exc_info=True)
            time.sleep(60)


if __name__ == "__main__":
    main()
