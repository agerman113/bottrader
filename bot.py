"""
Bybit ГИБРИДНЫЙ ФЬЮЧЕРСНЫЙ БОТ — v5 HYBRID
============================================
Объединяет:
  1. Скоринговую систему v4.1 (RSI, MACD, Supertrend, Hull MA, Range Filter,
     ADX, Stochastic, VWAP, Volume, S/R)
  2. MA Кроссовер из TradingView "3Commas Bot" (9 типов MA)
  3. Стопы по swing low/high + ATR (вместо фиксированного %)
  4. Трейлинг с триггером по % достижения цели (rrExit)
  5. Session Filter (не торговать в указанные часы UTC)
  6. Daily Loss Limit (не более X% потерь в день)
  7. Partial TP (закрыть 50% при +TP1%, вести остаток с трейлингом)
  8. Volume Spike Guard (не входить если объём >3x среднего)
  9. Signal Exit (ранний выход если Supertrend+RangeFilter развернулись)
 10. Swing Low SL (SL под локальный минимум последних N свечей)
 11. Подтверждение входа (ждать 2 свечи и перепроверить скор)

КРИТИЧЕСКИЕ ФИКСЫ:
 - Все константы определены ДО использования
 - Ошибка установки плеча → позиция НЕ открывается
 - Краш мониторинга → позиция принудительно закрывается
 - Просадка по реальному балансу
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
    "DOGE/USDT:USDT",  "SHIB/USDT:USDT",  "PEPE/USDT:USDT",
    "FLOKI/USDT:USDT", "BONK/USDT:USDT",  "WIF/USDT:USDT",
    "MEME/USDT:USDT",  "BOME/USDT:USDT",  "DOGS/USDT:USDT",
    "NEIRO/USDT:USDT", "PNUT/USDT:USDT",  "ACT/USDT:USDT",
    "POPCAT/USDT:USDT","TURBO/USDT:USDT", "BRETT/USDT:USDT",
    "FET/USDT:USDT",   "RENDER/USDT:USDT","TAO/USDT:USDT",
    "WLD/USDT:USDT",   "ARKM/USDT:USDT",  "AGIX/USDT:USDT",
    "IO/USDT:USDT",    "ONDO/USDT:USDT",  "VIRTUAL/USDT:USDT",
    "AI16Z/USDT:USDT",
    "UNI/USDT:USDT",   "AAVE/USDT:USDT",  "CRV/USDT:USDT",
    "DYDX/USDT:USDT",  "JUP/USDT:USDT",   "PENDLE/USDT:USDT",
    "GMX/USDT:USDT",   "LDO/USDT:USDT",
    "ARB/USDT:USDT",   "OP/USDT:USDT",    "MATIC/USDT:USDT",
    "STX/USDT:USDT",   "IMX/USDT:USDT",   "STRK/USDT:USDT",
    "ZK/USDT:USDT",    "MANTA/USDT:USDT",
    "AXS/USDT:USDT",   "SAND/USDT:USDT",  "MANA/USDT:USDT",
    "GALA/USDT:USDT",  "ENJ/USDT:USDT",   "ILV/USDT:USDT",
    "PIXEL/USDT:USDT", "PORTAL/USDT:USDT",
    "LINK/USDT:USDT",  "GRT/USDT:USDT",   "FIL/USDT:USDT",
    "ICP/USDT:USDT",   "RUNE/USDT:USDT",  "INJ/USDT:USDT",
    "SUI/USDT:USDT",   "APT/USDT:USDT",   "SEI/USDT:USDT",
    "TIA/USDT:USDT",   "PYTH/USDT:USDT",  "JTO/USDT:USDT",
    "W/USDT:USDT",     "ENA/USDT:USDT",   "EIGEN/USDT:USDT",
    "HBAR/USDT:USDT",  "VET/USDT:USDT",   "ALGO/USDT:USDT",
    "IOTA/USDT:USDT",  "EOS/USDT:USDT",   "XTZ/USDT:USDT",
    "THETA/USDT:USDT", "FLOW/USDT:USDT",  "KSM/USDT:USDT",
    "CHZ/USDT:USDT",   "MASK/USDT:USDT",  "1INCH/USDT:USDT",
    "COMP/USDT:USDT",  "ZRO/USDT:USDT",   "NOT/USDT:USDT",
    "HMSTR/USDT:USDT", "CATI/USDT:USDT",
]

# ================== ОСНОВНЫЕ НАСТРОЙКИ ==================
LEVERAGE             = 3
TIMEFRAME_TA         = "5m"
TIMEFRAME_TREND      = "1h"
TIMEFRAME_MID        = "15m"
TIMEFRAME_4H         = "4h"
SCAN_INTERVAL        = 300        # секунд между сканированиями

MIN_SCORE            = 65
BASE_RISK_PCT        = 0.8
MAX_RISK_PCT         = 1.5

# === ТЕЙКПРОФИТ / СТОПЛОСС ===
TP_PERCENT           = 2.5        # % базовый TP
SL_PERCENT           = 0.8        # % максимальный SL
MIN_SL_PERCENT       = 0.5        # % минимальный SL (защита от слишком тесного стопа)

# === PARTIAL TAKE PROFIT ===
PARTIAL_TP_ENABLED   = True
PARTIAL_TP_PCT       = 1.0        # % от входа для частичного TP
PARTIAL_TP_CLOSE     = 0.5        # доля позиции для закрытия (0.5 = 50%)

# === SWING STOP ===
SWING_LOOKBACK       = 10         # свечей для поиска swing low/high
SWING_ATR_MULT       = 1.0        # ATR множитель для отступа от swing

# === АДАПТИВНЫЙ ТРЕЙЛИНГ ===
TRAILING_ATR_PERIOD  = 14
TRAILING_ATR_MULT    = 1.5
TRAILING_OFFSET_MULT = 1.0
MIN_TRAILING_STEP    = 0.25       # % минимальный шаг трейлинга
MIN_TRAILING_OFFSET  = 0.35       # % минимальный отступ трейлинга
MIN_PROFIT_FOR_TRAIL = 0.5        # % прибыли для активации трейлинга

# === RREXIT — триггер трейлинга по % достижения цели ===
# 0.0 = сразу при входе; 0.5 = когда пройдено 50% пути до TP; 1.0 = только достигнув TP
RR_EXIT_TRIGGER      = 0.5

# === MA КРОССОВЕР (дополнительный фильтр входа) ===
MA_CROSSOVER_ENABLED = True
MA1_TYPE             = "EMA"      # EMA HEMA SMA HMA WMA DEMA VWMA VWAP T3
MA2_TYPE             = "EMA"
MA1_LENGTH           = 21
MA2_LENGTH           = 50
MA_TIMEFRAME         = "5m"       # таймфрейм для MA кроссовера

# === SESSION FILTER ===
SESSION_FILTER_ENABLED = True
SESSION_BLOCK_START    = 0        # UTC час начала блокировки
SESSION_BLOCK_END      = 3        # UTC час конца блокировки (не торгуем 00:00-03:00 UTC)

# === DAILY LOSS LIMIT ===
DAILY_LOSS_LIMIT_PCT = 3.0        # % от баланса — максимум потерь за день
DAILY_LOSS_PAUSE_SEC = 14400      # пауза при достижении лимита (4 часа)

# === VOLUME SPIKE GUARD ===
VOLUME_SPIKE_MULT    = 3.0        # не входить если объём > X * среднее
VOLUME_AVG_PERIOD    = 20         # период для расчёта среднего объёма

# === SIGNAL EXIT (ранний выход) ===
SIGNAL_EXIT_ENABLED  = True       # Supertrend + RangeFilter разворот → выход

# === ПОДТВЕРЖДЕНИЕ ВХОДА ===
ENTRY_CONFIRM_BARS   = 2          # ждать N свечей и перепроверить скор
ENTRY_CONFIRM_MIN_SCORE = 60      # минимальный скор при перепроверке

# === БЛОКИРОВКА ПОСЛЕ SL ===
SYMBOL_BLOCK_MINUTES = 30
SL_STREAK_LIMIT      = 3
SL_STREAK_PAUSE      = 3600

# === ПРОСАДКА И БАЛАНС ===
MIN_BALANCE          = 5.0
MAX_DRAWDOWN_PCT     = 20.0

# === ПРОЧИЕ НАСТРОЙКИ ===
TRADE_MAX_LIFETIME   = 7200       # секунд макс длительность сделки
REPORT_INTERVAL      = 1800       # секунд между отчётами
STATE_FILE           = "state_hybrid.json"
TRADES_FILE          = "trades_hybrid.json"
BYBIT_FEE            = 0.00055

# === S/R УРОВНИ ===
SR_PERIOD            = 100
SR_PROXIMITY_PCT     = 0.3
SR_MIN_TOUCHES       = 3
SR_CLUSTER_TOL       = 0.005
SR_BLOCK_DIST_PCT    = 0.15

# ================== ЛОГИРОВАНИЕ ==================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot_hybrid.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ================== БИРЖА ==================
exchange = ccxt.bybit({
    "apiKey":          os.getenv("BYBIT_API_KEY"),
    "secret":          os.getenv("BYBIT_API_SECRET"),
    "enableRateLimit": True,
    "options": {
        "defaultType": "linear",
    },
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
            json.dump(история, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning(f"Не удалось сохранить сделку: {e}")


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


# ================== АНАЛИТИКА ==================
def аналитика_по_инструментам():
    история = загрузить_историю()
    if len(история) < 2:
        log.info("  📊 Аналитика: мало сделок")
        return
    по_символам: dict = {}
    for сд in история:
        sym = сд.get("symbol", "?").split(":")[0]
        if sym not in по_символам:
            по_символам[sym] = {"tp": 0, "sl": 0, "timeout": 0, "pnl": 0.0}
        r = сд.get("результат", "")
        if   r == "tp":   по_символам[sym]["tp"]      += 1
        elif r == "sl":   по_символам[sym]["sl"]      += 1
        else:             по_символам[sym]["timeout"]  += 1
        по_символам[sym]["pnl"] += сд.get("pnl_usdt", 0)
    log.info("")
    log.info("=" * 60)
    log.info(f"  📊 АНАЛИТИКА — сделок: {len(история)}")
    log.info(f"  {'Символ':<16} {'Всего':>6}  {'TP':>4}  {'SL':>4}  {'WR%':>6}  {'P&L':>8}")
    log.info("  " + "─" * 50)
    for sym, d in sorted(по_символам.items(), key=lambda x: x[1]["pnl"], reverse=True):
        всего = d["tp"] + d["sl"] + d["timeout"]
        wr    = d["tp"] / всего * 100 if всего > 0 else 0
        знак  = "+" if d["pnl"] >= 0 else ""
        log.info(f"  {sym:<16} {всего:>6}  {d['tp']:>4}  {d['sl']:>4}  {wr:>5.1f}%  {знак}{d['pnl']:>7.4f}U")
    log.info("=" * 60)
    log.info("")


# ================== СОСТОЯНИЕ ==================
def сохранить_состояние():
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
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
    """Реальный свободный баланс USDT."""
    try:
        b = exchange.fetch_balance({"type": "linear"})
        return float(b.get("USDT", {}).get("free", 0.0))
    except Exception as e:
        log.warning(f"Ошибка получения баланса: {e}")
        return 0.0


def полный_баланс_usdt() -> float:
    """Полный баланс включая unrealized PnL (для расчёта просадки)."""
    try:
        b = exchange.fetch_balance({"type": "linear"})
        total = float(b.get("USDT", {}).get("total", 0.0))
        return total if total > 0 else баланс_usdt()
    except Exception as e:
        log.warning(f"Ошибка получения полного баланса: {e}")
        return баланс_usdt()


def получить_позиции() -> list:
    try:
        positions = exchange.fetch_positions()
        return [p for p in positions if float(p.get("contracts", 0) or 0) > 0]
    except Exception as e:
        log.warning(f"Ошибка получения позиций: {e}")
        return []


def закрыть_позицию_рынком(symbol: str, qty: float, side: str):
    """Принудительное закрытие позиции рыночным ордером."""
    close_side = "sell" if side == "long" else "buy"
    try:
        exchange.create_market_order(symbol, close_side, qty, params={"reduceOnly": True})
        log.info(f"  🔒 Позиция {symbol} {side} qty={qty} закрыта рыночным ордером")
    except Exception as e:
        log.error(f"  ❌ Не удалось закрыть {symbol}: {e}")


# ================== УСТАНОВКА ПЛЕЧА ==================
def установить_плечо(symbol: str, leverage: int) -> bool:
    """
    Устанавливает плечо через 3 метода.
    Возвращает True если успешно, False если все провалились.
    Позиция НЕ открывается если False.
    """
    # Метод 1: стандартный ccxt
    try:
        exchange.set_leverage(
            leverage, symbol,
            params={"buyLeverage": leverage, "sellLeverage": leverage}
        )
        log.info(f"  ⚙️  Плечо {leverage}x установлено для {symbol}")
        return True
    except Exception as e1:
        log.warning(f"  Метод 1 плеча не сработал: {e1}")

    # Метод 2: прямой Bybit v5 API
    try:
        coin_sym = symbol.replace("/", "").replace(":USDT", "")
        exchange.private_post_v5_position_set_leverage({
            "category":     "linear",
            "symbol":       coin_sym,
            "buyLeverage":  str(leverage),
            "sellLeverage": str(leverage),
        })
        log.info(f"  ⚙️  Плечо {leverage}x установлено (v5 fallback) для {symbol}")
        return True
    except Exception as e2:
        log.warning(f"  Метод 2 плеча не сработал: {e2}")

    # Метод 3: проверить текущее плечо
    try:
        positions = exchange.fetch_positions([symbol])
        for p in positions:
            curr_lev = int(float(p.get("leverage", 0) or 0))
            if curr_lev == leverage:
                log.info(f"  ⚙️  Плечо уже = {leverage}x для {symbol} — OK")
                return True
            elif curr_lev > 0:
                log.warning(f"  ⚠️  Текущее плечо {curr_lev}x ≠ {leverage}x для {symbol}")
                return False
    except Exception as e3:
        log.warning(f"  Метод 3 проверки плеча не сработал: {e3}")

    log.error(f"  ❌ Не удалось установить плечо {leverage}x для {symbol} — сделка отменена")
    return False


# ================== SESSION FILTER ==================
def торговля_разрешена_по_времени() -> bool:
    """Проверяет, не попадаем ли мы в заблокированный сессионный диапазон."""
    if not SESSION_FILTER_ENABLED:
        return True
    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour
    if SESSION_BLOCK_START < SESSION_BLOCK_END:
        blocked = SESSION_BLOCK_START <= hour < SESSION_BLOCK_END
    else:  # переходит через полночь: 22-03
        blocked = hour >= SESSION_BLOCK_START or hour < SESSION_BLOCK_END
    if blocked:
        log.info(f"  🕐 Session Filter: час {hour} UTC заблокирован ({SESSION_BLOCK_START}-{SESSION_BLOCK_END})")
    return not blocked


# ================== DAILY LOSS LIMIT ==================
def обновить_начало_дня(баланс: float):
    """Обновляет баланс начала дня если наступил новый день."""
    сегодня = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if stats["дата_дня"] != сегодня:
        stats["дата_дня"]          = сегодня
        stats["баланс_начало_дня"] = баланс
        log.info(f"  📅 Новый торговый день. Баланс: {баланс:.2f} USDT")
        сохранить_состояние()


def превышен_дневной_лимит() -> bool:
    """Проверяет дневной лимит убытков."""
    нач = stats.get("баланс_начало_дня", 0.0)
    if нач <= 0:
        return False
    текущий = полный_баланс_usdt()
    потеря_пct = (нач - текущий) / нач * 100
    if потеря_пct >= DAILY_LOSS_LIMIT_PCT:
        log.warning(
            f"  ⛔ Дневной лимит убытков: -{потеря_пct:.1f}% (лимит {DAILY_LOSS_LIMIT_PCT}%)"
        )
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


def _dema(s: pd.Series, span: int) -> pd.Series:
    e = _ema(s, span)
    return 2 * e - _ema(e, span)


def _vwma(df: pd.DataFrame, span: int) -> pd.Series:
    return (df["c"] * df["v"]).rolling(span).sum() / df["v"].rolling(span).sum()


def _vwap_series(df: pd.DataFrame, span: int = 20) -> pd.Series:
    typical = (df["h"] + df["l"] + df["c"]) / 3
    return (typical * df["v"]).rolling(span).sum() / df["v"].rolling(span).sum()


def _t3(s: pd.Series, span: int, vfactor: float = 0.7) -> pd.Series:
    c1 = -(vfactor ** 3)
    c2 = 3 * vfactor ** 2 + 3 * vfactor ** 3
    c3 = -6 * vfactor ** 2 - 3 * vfactor - 3 * vfactor ** 3
    c4 = 1 + 3 * vfactor + vfactor ** 3 + 3 * vfactor ** 2
    e1 = _ema(s, span)
    e2 = _ema(e1, span)
    e3 = _ema(e2, span)
    e4 = _ema(e3, span)
    e5 = _ema(e4, span)
    e6 = _ema(e5, span)
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


def _hema(s: pd.Series, span: int) -> pd.Series:
    """Hull EMA = 2*EMA(n/2) - EMA(n), smoothed with EMA(sqrt(n))."""
    half   = max(1, span // 2)
    sqrt_p = max(1, int(np.sqrt(span)))
    return _ema(2 * _ema(s, half) - _ema(s, span), sqrt_p)


def calc_ma(df: pd.DataFrame, ma_type: str, length: int) -> pd.Series:
    """Рассчитывает MA заданного типа. Принимает датафрейм с колонками o/h/l/c/v."""
    s = df["c"]
    ma_type = ma_type.upper()
    if   ma_type == "EMA":  return _ema(s, length)
    elif ma_type == "SMA":  return _sma(s, length)
    elif ma_type == "WMA":  return _wma(s, length)
    elif ma_type == "HMA":  return _hma(s, length)
    elif ma_type == "HEMA": return _hema(s, length)
    elif ma_type == "DEMA": return _dema(s, length)
    elif ma_type == "VWMA": return _vwma(df, length)
    elif ma_type == "VWAP": return _vwap_series(df, length)
    elif ma_type == "T3":   return _t3(s, length)
    else:
        log.warning(f"Неизвестный тип MA '{ma_type}', используем EMA")
        return _ema(s, length)


# ================== MA КРОССОВЕР ==================
def проверить_ma_кроссовер(df: pd.DataFrame, side: str = "long") -> bool:
    """
    Проверяет кроссовер MA1 / MA2.
    Long: MA1 пересекает MA2 снизу вверх (или MA1 > MA2 последние 2 свечи, кросс произошёл)
    Short: MA1 пересекает MA2 сверху вниз
    Возвращает True если кроссовер в текущей или предыдущей свече.
    """
    if not MA_CROSSOVER_ENABLED:
        return True
    try:
        min_len = max(MA1_LENGTH, MA2_LENGTH) * 2 + 5
        if len(df) < min_len:
            return True  # недостаточно данных — не блокируем
        ma1 = calc_ma(df, MA1_TYPE, MA1_LENGTH)
        ma2 = calc_ma(df, MA2_TYPE, MA2_LENGTH)
        # Кроссовер: предыдущая свеча ma1 <= ma2, текущая ma1 > ma2 (для лонга)
        crossover  = ma1.iloc[-2] <= ma2.iloc[-2] and ma1.iloc[-1] >  ma2.iloc[-1]
        crossunder = ma1.iloc[-2] >= ma2.iloc[-2] and ma1.iloc[-1] <  ma2.iloc[-1]
        # Или текущий тренд MA (менее строгий вариант)
        ma_aligned_long  = ma1.iloc[-1] > ma2.iloc[-1]
        ma_aligned_short = ma1.iloc[-1] < ma2.iloc[-1]

        if side == "long":
            result = crossover or ma_aligned_long
        else:
            result = crossunder or ma_aligned_short
        return result
    except Exception as e:
        log.warning(f"  Ошибка расчёта MA кроссовера: {e}")
        return True  # при ошибке не блокируем


# ================== VOLUME SPIKE GUARD ==================
def volume_spike_guard(df: pd.DataFrame) -> bool:
    """
    Возвращает True если объём НОРМАЛЬНЫЙ (можно входить).
    Возвращает False если объём спайк (не входить).
    """
    try:
        vol_avg  = df["v"].rolling(VOLUME_AVG_PERIOD).mean().iloc[-1]
        vol_now  = df["v"].iloc[-1]
        ratio    = vol_now / (vol_avg + 1e-10)
        if ratio > VOLUME_SPIKE_MULT:
            log.info(f"  🔊 Volume Spike Guard: объём {ratio:.1f}x > {VOLUME_SPIKE_MULT}x — вход отменён")
            return False
        return True
    except Exception:
        return True


# ================== ИНДИКАТОРЫ ==================
def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d     = close.diff()
    gain  = d.clip(lower=0)
    loss  = (-d).clip(lower=0)
    avg_g = _rma(gain, period)
    avg_l = _rma(loss, period)
    rs    = avg_g / avg_l.replace(0, np.nan)
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
    atr  = calc_atr(df, period)
    hl2  = (df["h"] + df["l"]) / 2
    ub   = (hl2 + mult * atr).copy()
    lb   = (hl2 - mult * atr).copy()
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


def calc_vwap_deviation(df: pd.DataFrame, period: int = 20) -> pd.Series:
    typical = (df["h"] + df["l"] + df["c"]) / 3
    vwap    = (typical * df["v"]).rolling(period).sum() / df["v"].rolling(period).sum()
    return (df["c"] - vwap) / vwap * 100


def calc_range_filter(df: pd.DataFrame, period: int = 200, qty: float = 3.0):
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


# ================== SWING LOW/HIGH SL ==================
def calc_swing_sl(df: pd.DataFrame, side: str, atr: float) -> float:
    """
    Рассчитывает SL на основе swing low/high.
    Long: swing low последних SWING_LOOKBACK свечей - ATR*SWING_ATR_MULT
    Short: swing high последних SWING_LOOKBACK свечей + ATR*SWING_ATR_MULT
    """
    n = SWING_LOOKBACK
    if len(df) < n + 1:
        return 0.0
    recent = df.tail(n + 1)
    if side == "long":
        swing_level = float(recent["l"].min())
        sl = swing_level - atr * SWING_ATR_MULT
    else:
        swing_level = float(recent["h"].max())
        sl = swing_level + atr * SWING_ATR_MULT
    return sl


# ================== S/R УРОВНИ ==================
def _кластеризовать_уровни(levels: list, tolerance: float = SR_CLUSTER_TOL) -> list:
    if not levels:
        return []
    levels = sorted(levels)
    кластеры = []
    текущий  = [levels[0]]
    for lvl in levels[1:]:
        if (lvl - текущий[0]) / (текущий[0] + 1e-10) < tolerance:
            текущий.append(lvl)
        else:
            кластеры.append((float(np.mean(текущий)), len(текущий)))
            текущий = [lvl]
    кластеры.append((float(np.mean(текущий)), len(текущий)))
    return кластеры


def calc_support_resistance(df: pd.DataFrame, period: int = SR_PERIOD) -> dict:
    df_sr  = df.tail(period).reset_index(drop=True)
    highs  = df_sr["h"].values
    lows   = df_sr["l"].values
    close  = float(df["c"].iloc[-1])

    raw_resistances, raw_supports = [], []
    for i in range(2, len(highs) - 2):
        if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
            raw_resistances.append(highs[i])
        if (lows[i]  < lows[i-1]  and lows[i]  < lows[i-2]  and
                lows[i]  < lows[i+1]  and lows[i]  < lows[i+2]):
            raw_supports.append(lows[i])

    res_clusters = _кластеризовать_уровни(raw_resistances, SR_CLUSTER_TOL)
    sup_clusters = _кластеризовать_уровни(raw_supports,    SR_CLUSTER_TOL)

    res_above        = [(p, n) for p, n in res_clusters if p > close]
    sup_below        = [(p, n) for p, n in sup_clusters if p < close]
    res_above_sorted = sorted(res_above, key=lambda x: x[0])
    sup_below_sorted = sorted(sup_below, key=lambda x: x[0], reverse=True)

    nearest_resistance, res_cluster = res_above_sorted[0] if res_above_sorted else (close * 1.05, 0)
    nearest_support,    sup_cluster = sup_below_sorted[0] if sup_below_sorted else (close * 0.95, 0)

    dist_to_res = (nearest_resistance - close) / close * 100
    dist_to_sup = (close - nearest_support)    / close * 100

    near_support    = dist_to_sup < SR_PROXIMITY_PCT and sup_cluster >= SR_MIN_TOUCHES
    near_resistance = dist_to_res < SR_PROXIMITY_PCT and res_cluster >= SR_MIN_TOUCHES

    return {
        "support":         round(nearest_support,    10),
        "resistance":      round(nearest_resistance, 10),
        "dist_to_sup_pct": round(dist_to_sup, 2),
        "dist_to_res_pct": round(dist_to_res, 2),
        "sup_cluster":     sup_cluster,
        "res_cluster":     res_cluster,
        "near_support":    near_support,
        "near_resistance": near_resistance,
    }


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


# ================== ГЛОБАЛЬНЫЙ ТРЕНД 4H ==================
def тренд_4h_бычий(symbol: str) -> bool:
    try:
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
    """
    Полный технический скор 0-100.
    Теперь включает MA кроссовер и Volume Spike Guard как дополнительные флаги.
    """
    details = {}
    score   = 0
    price   = 0.0
    sr      = {}
    df5     = None

    try:
        raw5  = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA,    limit=300)
        raw1h = exchange.fetch_ohlcv(symbol, TIMEFRAME_TREND, limit=300)
        raw15 = exchange.fetch_ohlcv(symbol, TIMEFRAME_MID,   limit=100)

        if len(raw5) < 100 or len(raw1h) < 100:
            return {"score": 0, "details": {}, "price": 0, "sr": {}, "df5": None}

        cols  = ["ts","o","h","l","c","v"]
        df5   = pd.DataFrame(raw5,  columns=cols).reset_index(drop=True)
        df1h  = pd.DataFrame(raw1h, columns=cols).reset_index(drop=True)
        df15  = pd.DataFrame(raw15, columns=cols).reset_index(drop=True)
        c5, c1h, c15 = df5["c"], df1h["c"], df15["c"]

        price = float(c5.iloc[-1])

        # --- RSI 5m ---
        rsi_val = calc_rsi(c5).iloc[-1]
        details["rsi"] = round(rsi_val, 1)
        if   25 <= rsi_val <= 42: score += 20
        elif 42 <  rsi_val <= 52: score += 10
        elif rsi_val < 25:        score += 12
        elif 52 <  rsi_val <= 65: score +=  5

        # --- RSI 1h ---
        rsi_1h = calc_rsi(c1h).iloc[-1]
        details["rsi_1h"] = round(rsi_1h, 1)
        if   rsi_1h < 55: score += 8
        elif rsi_1h < 65: score += 4

        # --- MACD 5m ---
        ml, sl_macd, _ = calc_macd(c5)
        macd_bull  = ml.iloc[-1] > sl_macd.iloc[-1]
        macd_cross = macd_bull and ml.iloc[-2] <= sl_macd.iloc[-2]
        details["macd"] = "бычий" if macd_bull else "медвежий"
        if   macd_cross: score += 18
        elif macd_bull:  score +=  8

        # --- Range Filter ---
        _, _, _, rf_up, rf_down = calc_range_filter(df5)
        rf_up_now = rf_up.iloc[-1]
        details["range_filter"] = "вверх" if rf_up_now else ("вниз" if rf_down.iloc[-1] else "бок")
        if rf_up_now:
            score += 15

        # --- Supertrend 5m ---
        st_up, _ = calc_supertrend(df5)
        details["supertrend"] = "вверх" if st_up.iloc[-1] else "вниз"
        if st_up.iloc[-1]:
            score += 12

        # --- Supertrend 15m ---
        st_up_15, _ = calc_supertrend(df15)
        details["supertrend_15m"] = "вверх" if st_up_15.iloc[-1] else "вниз"
        if st_up_15.iloc[-1]:
            score += 8

        # --- Hull MA ---
        hu_up, _ = calc_hull(c5)
        details["hull"] = "вверх" if hu_up.iloc[-1] else "вниз"
        if hu_up.iloc[-1]:
            score += 8

        # --- Тренд 1h ---
        ema50_1h  = _ema(c1h, 50).iloc[-1]
        ema200_1h = _ema(c1h, 200).iloc[-1]
        details["тренд_1h"] = "бычий" if ema50_1h > ema200_1h else "медвежий"
        if ema50_1h > ema200_1h:
            score += 10

        # --- Тренд 15m ---
        ema20_15 = _ema(c15, 20).iloc[-1]
        ema50_15 = _ema(c15, 50).iloc[-1]
        details["тренд_15m"] = "бычий" if ema20_15 > ema50_15 else "медвежий"
        if ema20_15 > ema50_15:
            score += 5

        # --- ADX ---
        adx, pdi, mdi = calc_adx(df5)
        adx_val = adx.iloc[-1]
        details["adx"] = round(adx_val, 1)
        if   adx_val > 25 and pdi.iloc[-1] > mdi.iloc[-1]: score += 8
        elif adx_val > 20:                                   score += 3

        # --- Stochastic ---
        k_ser, _ = calc_stochastic(df5)
        k_val = k_ser.iloc[-1]
        details["stoch_k"] = round(k_val, 1)
        if   k_val < 25: score += 8
        elif k_val < 50: score += 4

        # --- Volume ---
        vol_avg   = df5["v"].rolling(VOLUME_AVG_PERIOD).mean().iloc[-1]
        vol_ratio = df5["v"].iloc[-1] / (vol_avg + 1e-10)
        details["объём_ratio"] = round(vol_ratio, 2)
        if   vol_ratio > 1.5: score += 8
        elif vol_ratio > 1.2: score += 4

        # --- VWAP deviation ---
        vwap_dev = calc_vwap_deviation(df5).iloc[-1]
        details["vwap_dev"] = round(vwap_dev, 2)
        if   -3 <= vwap_dev <= -0.3: score += 8
        elif vwap_dev < -3:          score += 4
        elif vwap_dev <= 1:          score += 2

        # --- S/R уровни ---
        sr = calc_support_resistance(df5)
        details.update({
            "support":   sr["support"],
            "resistance": sr["resistance"],
            "dist_sup":  sr["dist_to_sup_pct"],
            "dist_res":  sr["dist_to_res_pct"],
        })
        if sr["near_support"]:
            score += 12
            details["sr_signal"] = f"у поддержки ✅ ({sr['sup_cluster']} касаний)"
        elif sr["near_resistance"]:
            score -= 20
            details["sr_signal"] = f"у сопротивления ❌ ({sr['res_cluster']} касаний)"
        else:
            details["sr_signal"] = f"нейтр (sup={sr['dist_to_sup_pct']:.2f}% res={sr['dist_to_res_pct']:.2f}%)"

        # --- 3 красных свечи подряд ---
        last3_bearish = all(df5["c"].iloc[-i] < df5["o"].iloc[-i] for i in range(1, 4))
        if last3_bearish:
            score -= 15
            details["свечи_3red"] = True

        # --- MA Кроссовер как ДОПОЛНИТЕЛЬНЫЙ флаг (не добавляет очков, блокирует вход) ---
        details["ma_cross"] = проверить_ma_кроссовер(df5, side="long")

        # --- Volume Spike Guard как флаг ---
        details["vol_spike_ok"] = volume_spike_guard(df5)

        score = max(0, min(100, score))

    except Exception as e:
        log.warning(f"Ошибка анализа {symbol}: {e}")

    return {"score": score, "details": details, "price": price, "sr": sr, "df5": df5}


def получить_скор_шорта(symbol: str) -> dict:
    """Инвертированный скор для шорта (с MA кроссовером в обратную сторону)."""
    res = получить_скор(symbol)
    if res["score"] == 0:
        return res
    inverted = max(0, 100 - res["score"] - 10)
    res["score"] = inverted
    # Перепроверяем MA кроссовер для шорта
    if res.get("df5") is not None:
        res["details"]["ma_cross"] = проверить_ma_кроссовер(res["df5"], side="short")
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
        if long_r > 0.75:
            return score
        return min(100, score + 5)
    elif signal == "bearish":
        return max(0, score - 15)
    return score


# ================== РАЗМЕР ПОЗИЦИИ ==================
def рассчитать_размер_позиции(score: int, баланс: float, atr_pct: float = 1.0,
                               sl_dist_pct: float = 0.0) -> float:
    """
    Рассчитывает маржу в USDT на основе риска.
    sl_dist_pct: реальное расстояние до SL в % (для swing SL)
    """
    if score <= MIN_SCORE:
        risk_pct = BASE_RISK_PCT
    else:
        factor   = (score - MIN_SCORE) / (100 - MIN_SCORE)
        risk_pct = BASE_RISK_PCT + (MAX_RISK_PCT - BASE_RISK_PCT) * factor

    if atr_pct > 1.5:
        risk_pct *= (1.5 / atr_pct)
        risk_pct  = max(BASE_RISK_PCT * 0.5, risk_pct)
    risk_pct = min(risk_pct, MAX_RISK_PCT)

    max_loss_usdt = баланс * risk_pct / 100
    # Используем реальный SL если он задан (swing), иначе стандартный SL_PERCENT
    actual_sl_pct = sl_dist_pct if sl_dist_pct > 0 else SL_PERCENT
    margin_usdt   = max_loss_usdt / (actual_sl_pct / 100)

    log.info(
        f"  📐 Скор={score} → риск={risk_pct:.1f}%  SL_dist={actual_sl_pct:.2f}%  "
        f"макс.убыток={max_loss_usdt:.2f}U → маржа={margin_usdt:.2f}U"
    )
    return round(max(1.0, margin_usdt), 2)


# ================== ПОДТВЕРЖДЕНИЕ ВХОДА ==================
def подтвердить_вход(symbol: str, исходный_скор: int, side: str = "long") -> bool:
    """
    Ждёт ENTRY_CONFIRM_BARS свечей (примерно N * длительность таймфрейма)
    и перепроверяет скор. Возвращает True если вход подтверждён.
    """
    if ENTRY_CONFIRM_BARS <= 0:
        return True

    tf_seconds = {
        "1m": 60, "3m": 180, "5m": 300, "15m": 900,
        "30m": 1800, "1h": 3600, "4h": 14400
    }
    wait = tf_seconds.get(TIMEFRAME_TA, 300) * ENTRY_CONFIRM_BARS

    log.info(f"  ⏳ Подтверждение входа: ждём {wait}с ({ENTRY_CONFIRM_BARS} свечи)...")
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


# ================== ОТКРЫТИЕ ПОЗИЦИИ ==================
def открыть_позицию(symbol: str, margin_usdt: float, tp_price: float,
                    sl_price: float, side: str = "long"):
    """
    Открывает позицию рыночным ордером с TP/SL.
    КРИТИЧНО: не открывает если плечо не установлено.
    """
    try:
        if not установить_плечо(symbol, LEVERAGE):
            log.error(f"  ❌ Плечо не установлено — сделка отменена для {symbol}")
            return None, None

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

        order = exchange.create_market_order(
            symbol, buy_sell, qty,
            params={
                "takeProfit": float(tp_str),
                "stopLoss":   float(sl_str),
            }
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


# ================== ОБНОВЛЕНИЕ SL ==================
def обновить_sl_на_бирже(symbol: str, new_sl: float, side: str = "long") -> bool:
    try:
        sl_str = exchange.price_to_precision(symbol, new_sl)
        exchange.set_trading_stop(
            symbol,
            params={
                "category":    "linear",
                "stopLoss":    float(sl_str),
                "slTriggerBy": "MarkPrice",
                "positionIdx": 0,
            }
        )
        log.info(f"  🔧 SL обновлён → {sl_str}")
        return True
    except Exception as e:
        try:
            sl_str   = exchange.price_to_precision(symbol, new_sl)
            coin_sym = symbol.replace("/", "").replace(":USDT", "")
            exchange.private_post_v5_position_trading_stop({
                "category":    "linear",
                "symbol":      coin_sym,
                "stopLoss":    sl_str,
                "slTriggerBy": "MarkPrice",
                "positionIdx": "0",
            })
            log.info(f"  🔧 SL обновлён (fallback) → {sl_str}")
            return True
        except Exception as e2:
            log.warning(f"  ⚠️ Не удалось обновить SL: {e} | {e2}")
            return False


# ================== ЧАСТИЧНОЕ ЗАКРЫТИЕ ==================
def закрыть_часть_позиции(symbol: str, qty: float, доля: float, side: str) -> float:
    """
    Закрывает долю позиции рыночным ордером.
    Возвращает фактически закрытый объём.
    """
    try:
        qty_close  = float(exchange.amount_to_precision(symbol, qty * доля))
        close_side = "sell" if side == "long" else "buy"
        if qty_close <= 0:
            return 0.0
        exchange.create_market_order(
            symbol, close_side, qty_close,
            params={"reduceOnly": True}
        )
        log.info(f"  💰 Partial TP: закрыто {qty_close} ({доля*100:.0f}%) {symbol}")
        return qty_close
    except Exception as e:
        log.warning(f"  ⚠️ Не удалось частично закрыть {symbol}: {e}")
        return 0.0


# ================== ПРОВЕРКА SIGNAL EXIT ==================
def проверить_signal_exit(symbol: str, side: str) -> bool:
    """
    Проверяет, развернулись ли Supertrend и RangeFilter.
    Возвращает True если нужно досрочно выйти.
    """
    if not SIGNAL_EXIT_ENABLED:
        return False
    try:
        raw = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA, limit=50)
        if len(raw) < 30:
            return False
        df = pd.DataFrame(raw, columns=["ts","o","h","l","c","v"])
        st_up, st_down = calc_supertrend(df)
        _, _, _, rf_up, rf_down = calc_range_filter(df)

        if side == "long":
            # Разворот вниз: оба развернулись
            return bool(st_down.iloc[-1] and rf_down.iloc[-1])
        else:
            return bool(st_up.iloc[-1] and rf_up.iloc[-1])
    except Exception:
        return False


# ================== МОНИТОРИНГ ПОЗИЦИИ ==================
def мониторить_позицию(symbol: str, entry_price: float, qty: float,
                        открыта_в: float, sl_цена: float,
                        tp_цена: float, side: str = "long") -> str:
    """
    Мониторит открытую позицию до закрытия.
    Включает:
    - Безубыток
    - Adaptive trailing stop
    - Partial TP (закрывает часть позиции)
    - rrExit trigger (трейлинг активируется по % достижения цели)
    - Signal Exit (Supertrend + RangeFilter разворот)
    - Deadline (таймаут)

    КРИТИЧНО: при краше этой функции — позиция закрывается снаружи.
    """
    deadline         = открыта_в + TRADE_MAX_LIFETIME
    coin             = symbol.split("/")[0]
    breakeven_price  = (entry_price * (1 + BYBIT_FEE * 2 + 0.0005) if side == "long"
                        else entry_price * (1 - BYBIT_FEE * 2 - 0.0005))

    # Адаптивный трейлинг на основе ATR
    trailing_step   = MIN_TRAILING_STEP   / 100
    trailing_offset = MIN_TRAILING_OFFSET / 100
    try:
        raw = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA, limit=50)
        if len(raw) >= 30:
            df       = pd.DataFrame(raw, columns=["ts","o","h","l","c","v"])
            atr_val  = calc_atr(df, TRAILING_ATR_PERIOD).iloc[-1]
            atr_pct  = (atr_val / entry_price) * 100
            trailing_step   = max(MIN_TRAILING_STEP,   atr_pct * TRAILING_ATR_MULT)   / 100
            trailing_offset = max(MIN_TRAILING_OFFSET, atr_pct * TRAILING_OFFSET_MULT) / 100
            log.info(
                f"  📊 ATR={atr_pct:.2f}% → шаг {trailing_step*100:.2f}%  "
                f"отступ {trailing_offset*100:.2f}%"
            )
    except Exception as e:
        log.warning(f"  Не удалось рассчитать ATR для трейлинга: {e}")

    # rrExit: уровень цены, при достижении которого активируется трейлинг
    if side == "long":
        rr_trigger_price = entry_price + (tp_цена - entry_price) * RR_EXIT_TRIGGER
    else:
        rr_trigger_price = entry_price - (entry_price - tp_цена) * RR_EXIT_TRIGGER
    log.info(f"  🎯 rrExit триггер={rr_trigger_price:.8f}  (RR_EXIT={RR_EXIT_TRIGGER})")

    фаза              = 1          # 1=обычная 2=безубыток 3=трейлинг
    текущий_sl        = sl_цена
    пиковая_цена      = entry_price
    trailing_активен  = RR_EXIT_TRIGGER == 0.0  # если 0 — трейлинг сразу
    partial_tp_сделан = False
    оставшийся_qty    = qty

    log.info(
        f"  🚦 Мониторинг {coin} {side}  вход={entry_price:.8f}  "
        f"SL={sl_цена:.8f}  TP={tp_цена:.8f}  breakeven={breakeven_price:.8f}"
    )

    while True:
        сейчас = time.time()

        # Таймаут
        if сейчас >= deadline:
            log.warning("  ⏰ Дедлайн — принудительное закрытие")
            try:
                close_side = "sell" if side == "long" else "buy"
                exchange.create_market_order(
                    symbol, close_side, оставшийся_qty,
                    params={"reduceOnly": True}
                )
            except Exception as e:
                log.warning(f"  Ошибка закрытия по дедлайну: {e}")
            return "таймаут"

        time.sleep(10)

        try:
            positions = exchange.fetch_positions([symbol])
            active    = [p for p in positions
                         if float(p.get("contracts", 0) or 0) > 0 and p.get("side") == side]

            if not active:
                # Позиция закрыта биржей (TP/SL сработал)
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
            cur_price = float(exchange.fetch_ticker(symbol)["last"])
            оставшийся_qty = abs(float(pos.get("contracts", 0) or 0))

            if side == "long":
                pnl_pct = (cur_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - cur_price) / entry_price * 100

            до_дед = int(deadline - сейчас)

            # === SIGNAL EXIT ===
            if SIGNAL_EXIT_ENABLED and фаза >= 2 and проверить_signal_exit(symbol, side):
                log.info(f"  🔄 Signal Exit: Supertrend+RangeFilter развернулись — закрываем {symbol}")
                try:
                    close_side = "sell" if side == "long" else "buy"
                    exchange.create_market_order(
                        symbol, close_side, оставшийся_qty,
                        params={"reduceOnly": True}
                    )
                except Exception as e:
                    log.warning(f"  Ошибка signal exit: {e}")
                return "tp" if pnl_pct > 0 else "sl"

            # === PARTIAL TP ===
            if (PARTIAL_TP_ENABLED and not partial_tp_сделан and
                    оставшийся_qty > 0):
                if side == "long":
                    partial_hit = cur_price >= entry_price * (1 + PARTIAL_TP_PCT / 100)
                else:
                    partial_hit = cur_price <= entry_price * (1 - PARTIAL_TP_PCT / 100)
                if partial_hit:
                    закрыто = закрыть_часть_позиции(symbol, оставшийся_qty, PARTIAL_TP_CLOSE, side)
                    if закрыто > 0:
                        partial_tp_сделан = True
                        оставшийся_qty   -= закрыто
                        log.info(f"  💰 Partial TP исполнен: +{PARTIAL_TP_PCT}%  "
                                 f"остаток qty={оставшийся_qty:.4f}")

            # === БЕЗУБЫТОК ===
            if фаза == 1 and pnl_pct >= 0:
                if side == "long":
                    new_sl_be = entry_price * (1 + BYBIT_FEE * 2 + 0.0003)
                else:
                    new_sl_be = entry_price * (1 - BYBIT_FEE * 2 - 0.0003)
                if обновить_sl_на_бирже(symbol, new_sl_be, side):
                    фаза         = 2
                    текущий_sl   = new_sl_be
                    пиковая_цена = cur_price
                    log.info(f"  🔒 БЕЗУБЫТОК! SL → {new_sl_be:.8f}")

            # === АКТИВАЦИЯ ТРЕЙЛИНГА ===
            if not trailing_активен and фаза >= 2:
                if side == "long":
                    trailing_активен = cur_price >= rr_trigger_price
                else:
                    trailing_активен = cur_price <= rr_trigger_price
                if trailing_активен:
                    log.info(f"  📈 Трейлинг активирован по rrExit @ {cur_price:.8f}")

            # === ТРЕЙЛИНГ ===
            if trailing_активен and фаза >= 2 and pnl_pct >= MIN_PROFIT_FOR_TRAIL:
                if side == "long":
                    if cur_price > пиковая_цена:
                        пиковая_цена = cur_price
                    new_sl_trail = пиковая_цена * (1 - trailing_offset)
                    if new_sl_trail > текущий_sl:
                        if обновить_sl_на_бирже(symbol, new_sl_trail, side):
                            текущий_sl = new_sl_trail
                            log.info(
                                f"  📈 ТРЕЙЛИНГ: пик={пиковая_цена:.8f} → SL={new_sl_trail:.8f} "
                                f"(зафиксировано {(new_sl_trail-entry_price)/entry_price*100:+.2f}%)"
                            )
                else:
                    if cur_price < пиковая_цена:
                        пиковая_цена = cur_price
                    new_sl_trail = пиковая_цена * (1 + trailing_offset)
                    if new_sl_trail < текущий_sl:
                        if обновить_sl_на_бирже(symbol, new_sl_trail, side):
                            текущий_sl = new_sl_trail
                            log.info(
                                f"  📈 ТРЕЙЛИНГ: пик={пиковая_цена:.8f} → SL={new_sl_trail:.8f} "
                                f"(зафиксировано {(entry_price-new_sl_trail)/entry_price*100:+.2f}%)"
                            )

            фаза_лейбл = {1: "обычная", 2: "безубыток 🔒", 3: "трейлинг 📈"}.get(фаза, "?")
            trail_лейбл = "🟢" if trailing_активен else "⏸"
            log.info(
                f"  [{coin}] {cur_price:.8f}  P&L={pnl_pct:+.2f}%"
                f"  SL={текущий_sl:.8f}  фаза={фаза_лейбл}  trail={trail_лейбл}  дед={до_дед}с"
                f"  qty={оставшийся_qty:.4f}"
            )

        except Exception as e:
            log.warning(f"  Ошибка в цикле мониторинга: {e}")


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
    log.info("  📊 ОТЧЁТ ГИБРИДНОГО БОТА")
    log.info(f"  Баланс:            {баланс:.2f} USDT  ({дельта:+.2f} USDT / {пct:+.2f}%)")
    log.info(f"  Сделок:            {всего}  TP={tp_}  SL={sl_}  Таймаут={stats['таймаут']}")
    log.info(f"  WinRate:           {wr:.1f}%")
    log.info(f"  Прибыль/Убыток:    {stats['прибыль_usdt']:.4f} / {stats['убыток_usdt']:.4f} USDT")
    log.info(f"  Чистый P&L:        {чистый:+.4f} USDT")
    log.info("=" * 65)
    log.info("")
    stats["последний_отчёт"] = time.time()
    сохранить_состояние()
    if всего > 0 and всего % 10 == 0:
        аналитика_по_инструментам()


# ================== ГЛАВНЫЙ ЦИКЛ ==================
def main():
    global stats

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
    log.info("  🤖  ГИБРИДНЫЙ ФЬЮЧЕРСНЫЙ БОТ v5")
    log.info(f"  Запуск №:              {stats['запусков']}")
    log.info(f"  Плечо:                 {LEVERAGE}x")
    log.info(f"  Баланс:                {баланс_сейчас:.2f} USDT")
    log.info(f"  Пар для торговли:      {len(SYMBOLS)}")
    log.info(f"  MIN_SCORE:             {MIN_SCORE}")
    log.info(f"  TP / SL:               {TP_PERCENT}% / {SL_PERCENT}%")
    log.info(f"  Partial TP:            {'ВКЛ' if PARTIAL_TP_ENABLED else 'ВЫКЛ'} @ +{PARTIAL_TP_PCT}% ({PARTIAL_TP_CLOSE*100:.0f}%)")
    log.info(f"  MA Кроссовер:          {MA1_TYPE}({MA1_LENGTH}) x {MA2_TYPE}({MA2_LENGTH})")
    log.info(f"  Session Filter:        {SESSION_BLOCK_START}:00-{SESSION_BLOCK_END}:00 UTC")
    log.info(f"  Daily Loss Limit:      {DAILY_LOSS_LIMIT_PCT}%")
    log.info(f"  Volume Spike Guard:    >{VOLUME_SPIKE_MULT}x среднего")
    log.info(f"  rrExit Trigger:        {RR_EXIT_TRIGGER*100:.0f}% до цели")
    log.info(f"  Signal Exit:           {'ВКЛ' if SIGNAL_EXIT_ENABLED else 'ВЫКЛ'}")
    log.info(f"  Swing SL:              {SWING_LOOKBACK} свечей + ATR*{SWING_ATR_MULT}")
    log.info(f"  Подтверждение входа:   {ENTRY_CONFIRM_BARS} свечи")
    log.info("=" * 65)
    log.info("")

    заблокированные_символы: dict = {}

    while True:
        try:
            # --- Отчёт ---
            if time.time() - stats["последний_отчёт"] >= REPORT_INTERVAL:
                печатать_отчёт()

            баланс = полный_баланс_usdt()
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
                    log.warning(f"  ⛔ Просадка {просадка:.1f}% > {MAX_DRAWDOWN_PCT}%. Пауза 2 ч.")
                    time.sleep(7200)
                    continue

            # --- Дневной лимит убытков ---
            if превышен_дневной_лимит():
                log.warning(f"  ⛔ Дневной лимит убытков. Пауза {DAILY_LOSS_PAUSE_SEC//60} мин.")
                time.sleep(DAILY_LOSS_PAUSE_SEC)
                continue

            # --- Session Filter ---
            if not торговля_разрешена_по_времени():
                log.info("  🕐 Торговля заблокирована по сессии. Пауза 5 мин.")
                time.sleep(300)
                continue

            # --- SL Streak cooldown ---
            if stats.get("sl_streak", 0) >= SL_STREAK_LIMIT:
                log.warning(
                    f"  🧊 {SL_STREAK_LIMIT} SL подряд — cooldown {SL_STREAK_PAUSE//60} мин."
                )
                stats["sl_streak"] = 0
                сохранить_состояние()
                time.sleep(SL_STREAK_PAUSE)
                continue

            # --- Активные позиции ---
            активные = получить_позиции()
            if активные:
                log.info(f"  ⏳ Открытые позиции: {[p['symbol'] for p in активные]} — ждём")
                time.sleep(30)
                continue

            log.info(
                f"── Сканирование {len(SYMBOLS)} пар "
                f"(баланс={свободный:.2f}U / полный={баланс:.2f}U, порог={MIN_SCORE}) ──"
            )

            # ==================== СКАНИРОВАНИЕ ====================
            scores = {}
            for sym in SYMBOLS:
                try:
                    # Проверка блокировки
                    if sym in заблокированные_символы and time.time() < заблокированные_символы[sym]:
                        scores[sym] = {"score": 0, "score_final": 0, "details": {}, "price": 0, "sr": {}, "df5": None}
                        continue

                    # Фильтр 4h тренда (лонг)
                    if not тренд_4h_бычий(sym):
                        scores[sym] = {"score": 0, "score_final": 0, "details": {"тренд_4h": "медвежий"}, "price": 0, "sr": {}, "df5": None}
                        continue

                    res      = получить_скор(sym)
                    ai_score = применить_ai_корректировку(res["score"], sym)
                    res["score_final"] = ai_score
                    scores[sym] = res

                    sr = res.get("sr", {})
                    log.info(
                        f"  {sym.split(':')[0]:12s}  скор={ai_score:3d}/100"
                        f"  rsi={res['details'].get('rsi', '?'):5}"
                        f"  rf={res['details'].get('range_filter', '?'):5}"
                        f"  st={res['details'].get('supertrend', '?'):5}"
                        f"  ma_cross={'✅' if res['details'].get('ma_cross', False) else '❌'}"
                        f"  vol={'✅' if res['details'].get('vol_spike_ok', True) else '🔊'}"
                        f"  SR={res['details'].get('sr_signal', '')}"
                    )
                except Exception as e:
                    log.warning(f"  Ошибка скора {sym}: {e}")
                    scores[sym] = {"score": 0, "score_final": 0, "details": {}, "price": 0, "sr": {}, "df5": None}

            if not scores:
                time.sleep(SCAN_INTERVAL)
                continue

            # ==================== ВЫБОР КАНДИДАТОВ ЛОНГ ====================
            кандидаты = sorted(
                [(s, d) for s, d in scores.items() if d["score_final"] >= MIN_SCORE],
                key=lambda x: x[1]["score_final"],
                reverse=True
            )[:5]

            выбрана  = None
            фин_скор = 0
            цена     = 0.0
            sr_info  = {}
            side     = "long"
            df5_выб  = None

            # Выбираем лучшего кандидата с учётом фильтров
            for лучшая, данные in кандидаты:
                фин_скор = данные["score_final"]
                цена     = данные["price"]
                sr_info  = данные.get("sr", {})
                det      = данные.get("details", {})
                df5_выб  = данные.get("df5")

                # Фильтр S/R
                dist_res_now = sr_info.get("dist_to_res_pct", 99)
                if sr_info.get("near_resistance") and dist_res_now < SR_BLOCK_DIST_PCT:
                    log.info(f"  ⛔ {лучшая.split(':')[0]}: resistance в {dist_res_now:.3f}% — пропуск")
                    continue

                # Фильтр перекупленности RSI
                rsi_val = float(det.get("rsi", 50) or 50)
                if rsi_val > 68 and not sr_info.get("near_support"):
                    log.info(f"  ⚠️ {лучшая.split(':')[0]}: RSI={rsi_val:.1f} перекуплен без поддержки — пропуск")
                    continue

                # MA Кроссовер фильтр
                if MA_CROSSOVER_ENABLED and not det.get("ma_cross", True):
                    log.info(f"  ⚠️ {лучшая.split(':')[0]}: MA кроссовер не подтверждён — пропуск")
                    continue

                # Volume Spike Guard
                if not det.get("vol_spike_ok", True):
                    log.info(f"  🔊 {лучшая.split(':')[0]}: volume spike — пропуск")
                    continue

                выбрана = лучшая
                log.info(
                    f"  ► Выбрана {лучшая.split(':')[0]} (лонг)  скор={фин_скор}  "
                    f"цена={цена:.8f}  dist_res={dist_res_now:.3f}%"
                )
                break

            # ==================== ШОРТ КАНДИДАТ ====================
            if выбрана is None:
                for sym, data in scores.items():
                    if sym in заблокированные_символы and time.time() < заблокированные_символы[sym]:
                        continue
                    if тренд_4h_медвежий(sym):
                        short_res = получить_скор_шорта(sym)
                        if short_res["score"] >= MIN_SCORE - 5:
                            det_sh = short_res.get("details", {})
                            # MA кроссовер для шорта
                            if MA_CROSSOVER_ENABLED and not det_sh.get("ma_cross", True):
                                continue
                            # Volume spike
                            if not det_sh.get("vol_spike_ok", True):
                                continue
                            log.info(f"  🐻 Шорт-кандидат: {sym.split(':')[0]} скор={short_res['score']}")
                            выбрана = sym
                            фин_скор = short_res["score"]
                            цена     = short_res["price"]
                            sr_info  = short_res.get("sr", {})
                            df5_выб  = short_res.get("df5")
                            side     = "short"
                            break

            if выбрана is None:
                log.info(f"  Нет кандидатов — ждём {SCAN_INTERVAL} сек")
                time.sleep(SCAN_INTERVAL)
                continue

            # ==================== РАСЧЁТ SL / TP ====================
            support    = sr_info.get("support",    цена * (1 - SL_PERCENT / 100))
            resistance = sr_info.get("resistance", цена * (1 + TP_PERCENT / 100))

            # Получаем ATR для swing SL
            atr_пт  = 0.0
            atr_pct = 1.0
            try:
                raw_atr = exchange.fetch_ohlcv(выбрана, TIMEFRAME_TA, limit=50)
                if len(raw_atr) >= 20:
                    df_atr  = pd.DataFrame(raw_atr, columns=["ts","o","h","l","c","v"])
                    atr_пт  = float(calc_atr(df_atr, 14).iloc[-1])
                    atr_pct = (atr_пт / цена) * 100
            except Exception:
                pass

            if side == "long":
                # Swing Low SL
                sl_swing = 0.0
                if df5_выб is not None and atr_пт > 0:
                    sl_swing = calc_swing_sl(df5_выб, "long", atr_пт)

                sl_базовый = цена * (1 - SL_PERCENT / 100)
                sl_от_sup  = float(support) * 0.998
                if sl_swing > 0:
                    # Берём наиболее консервативный (ближайший к цене)
                    sl_цена = max(sl_базовый, sl_от_sup, sl_swing)
                else:
                    sl_цена = max(sl_базовый, sl_от_sup)
                sl_цена = max(sl_цена, цена * (1 - MIN_SL_PERCENT / 100))

                tp_базовый = цена * (1 + TP_PERCENT / 100)
                dist_res   = sr_info.get("dist_to_res_pct", 99)
                tp_цена    = (цена + (float(resistance) - цена) * 0.90
                              if dist_res > TP_PERCENT * 1.2 else tp_базовый)

            else:  # short
                # Swing High SL
                sl_swing = 0.0
                if df5_выб is not None and atr_пт > 0:
                    sl_swing = calc_swing_sl(df5_выб, "short", atr_пт)

                sl_базовый = цена * (1 + SL_PERCENT / 100)
                sl_от_res  = float(resistance) * 1.002
                if sl_swing > 0:
                    sl_цена = min(sl_базовый, sl_от_res, sl_swing)
                else:
                    sl_цена = min(sl_базовый, sl_от_res)
                sl_цена = min(sl_цена, цена * (1 + MIN_SL_PERCENT / 100))

                tp_базовый = цена * (1 - TP_PERCENT / 100)
                dist_sup   = sr_info.get("dist_to_sup_pct", 99)
                tp_цена    = (цена - (цена - float(support)) * 0.90
                              if dist_sup > TP_PERCENT * 1.2 else tp_базовый)

            # Реальное расстояние до SL в %
            sl_dist_pct = abs(цена - sl_цена) / цена * 100

            margin = рассчитать_размер_позиции(фин_скор, свободный, atr_pct, sl_dist_pct)
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
            время_входа = time.time()
            вход_цена, кол_во = открыть_позицию(выбрана, margin, tp_цена, sl_цена, side)

            if вход_цена is None or кол_во is None:
                log.warning("  Не удалось открыть позицию — пауза 30 сек")
                time.sleep(30)
                continue

            stats["сделок_всего"] += 1
            сохранить_состояние()

            # ==================== МОНИТОРИНГ (с защитой от краша) ====================
            результат = "sl"  # пессимистичный дефолт
            try:
                результат = мониторить_позицию(
                    выбрана, вход_цена, кол_во,
                    время_входа, sl_цена, tp_цена, side
                )
            except Exception as monitor_err:
                log.error(f"  💥 Краш мониторинга: {monitor_err}", exc_info=True)
                log.warning("  🔒 Принудительное закрытие после краша мониторинга...")
                try:
                    close_side = "sell" if side == "long" else "buy"
                    exchange.create_market_order(
                        выбрана, close_side, кол_во,
                        params={"reduceOnly": True}
                    )
                    log.info(f"  ✅ {выбрана} закрыт принудительно")
                except Exception as close_err:
                    log.error(f"  ❌ Не удалось закрыть после краша: {close_err}")
                результат = "sl"

            # ==================== РЕЗУЛЬТАТ СДЕЛКИ ====================
            объём     = margin * LEVERAGE
            комиссии  = объём * BYBIT_FEE * 2
            длит_мин  = (time.time() - время_входа) / 60
            pnl_сд    = 0.0

            if результат == "tp":
                pnl_сд = объём * TP_PERCENT / 100 - комиссии
                stats["тейкпрофит"]   += 1
                stats["прибыль_usdt"] += max(0, pnl_сд)
                stats["sl_streak"]     = 0
                log.info(f"  ✅ TP: прибыль ≈{pnl_сд:.4f} USDT")

            elif результат == "sl":
                pnl_сд = -(объём * SL_PERCENT / 100 + комиссии)
                stats["стоплосс"]    += 1
                stats["убыток_usdt"] += abs(pnl_сд)
                stats["sl_streak"]    = stats.get("sl_streak", 0) + 1
                заблокированные_символы[выбрана] = time.time() + SYMBOL_BLOCK_MINUTES * 60
                log.warning(
                    f"  ❌ SL: убыток ≈{pnl_сд:.4f} USDT  "
                    f"(streak: {stats['sl_streak']}/{SL_STREAK_LIMIT}, "
                    f"блок {SYMBOL_BLOCK_MINUTES} мин)"
                )

            elif результат == "таймаут":
                pnl_сд = -комиссии
                stats["таймаут"]     += 1
                stats["убыток_usdt"] += комиссии
                stats["sl_streak"]    = 0
                log.warning(f"  ⏰ Таймаут: потери на комиссиях ≈{комиссии:.4f} USDT")

            запись_сделки = {
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
                "pnl_usdt":     round(pnl_сд, 4),
                "details":      scores.get(выбрана, {}).get("details", {}),
                "sr":           {k: str(v) for k, v in (sr_info or {}).items()},
            }
            сохранить_сделку(запись_сделки)
            пост_трейд_анализ(запись_сделки)
            сохранить_состояние()
            log.info("  Сделка завершена — пауза 30 сек")
            time.sleep(30)

        except Exception as e:
            log.error(f"Глобальная ошибка главного цикла: {e}", exc_info=True)
            time.sleep(60)


if __name__ == "__main__":
    main()
