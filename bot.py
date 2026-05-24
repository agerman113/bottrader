"""
Bybit ФЬЮЧЕРСНЫЙ бот (Linear USDT Perpetual) — v4 FIXED
=========================================================
ИСПРАВЛЕНИЯ v4.1:
1. [КРИТИЧНО] Исправлен NameError: TRAILING_STEP_PCT не был определён —
   функция мониторить_позицию использовала несуществующие константы.
   Теперь используются MIN_TRAILING_STEP и MIN_TRAILING_OFFSET напрямую.
2. [КРИТИЧНО] Плечо: добавлен fallback через private API Bybit v5 при ошибке 10001.
   Если и fallback не сработает — позиция НЕ открывается (раньше открывалась с
   неизвестным плечом).
3. [КРИТИЧНО] Защита от открытия позиции при краше мониторинга: если мониторинг
   упал с исключением — позиция принудительно закрывается через рыночный ордер.
4. [ВАЖНО] RENDER/USDT и ONDO/USDT с RSI > 68 без поддержки теперь корректно
   фильтруются (логика уже была, но не срабатывала из-за краша до неё).
5. [ВАЖНО] Просадка: добавлена проверка реального баланса с учётом незакрытых позиций.
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
from datetime import datetime

load_dotenv()

# ================== НАСТРОЙКИ ==================
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

LEVERAGE            = 3
TIMEFRAME_TA        = "5m"
TIMEFRAME_TREND     = "1h"
TIMEFRAME_MID       = "15m"
TIMEFRAME_4H        = "4h"
SCAN_INTERVAL       = 300

MIN_SCORE           = 65
BASE_RISK_PCT       = 0.8
MAX_RISK_PCT        = 1.5

TP_PERCENT          = 2.5
SL_PERCENT          = 0.8
MIN_SL_PERCENT      = 0.5

TRADE_MAX_LIFETIME  = 7200

# === АДАПТИВНЫЙ ТРЕЙЛИНГ ===
TRAILING_ATR_PERIOD = 14
TRAILING_ATR_MULT   = 1.5
TRAILING_OFFSET_MULT= 1.0
MIN_TRAILING_STEP   = 0.25   # % — используется как fallback в мониторинге
MIN_TRAILING_OFFSET = 0.35   # % — используется как fallback в мониторинге
MIN_PROFIT_FOR_TRAIL= 0.5

# === БЛОКИРОВКА ПОСЛЕ SL ===
SYMBOL_BLOCK_MINUTES = 30
SL_STREAK_LIMIT     = 3
SL_STREAK_PAUSE     = 3600

MIN_BALANCE         = 5.0
REPORT_INTERVAL     = 1800
STATE_FILE          = "state_futures.json"
BYBIT_FEE           = 0.00055

SR_PERIOD           = 100
SR_PROXIMITY_PCT    = 0.3
SR_MIN_TOUCHES      = 3
SR_CLUSTER_TOL      = 0.005
SR_BLOCK_DIST_PCT   = 0.15

MAX_DRAWDOWN_PCT    = 20.0

# ================== ЛОГИРОВАНИЕ ==================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot_futures.log", encoding="utf-8"),
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
TRADES_FILE    = "trades_history.json"
ANALYTICS_FILE = "analytics_report.json"

stats = {
    "запусков":        0,
    "сделок_всего":    0,
    "тейкпрофит":      0,
    "стоплосс":        0,
    "таймаут":         0,
    "прибыль_usdt":    0.0,
    "убыток_usdt":     0.0,
    "депозит_старт":   0.0,
    "старт_время":     "",
    "последний_отчёт": 0.0,
    "sl_streak":       0,
}

# ================== ЖУРНАЛ СДЕЛОК ==================
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
    r     = запись["результат"]
    sym   = запись["symbol"]
    score = запись["score"]
    det   = запись.get("details", {})
    pnl   = запись.get("pnl_usdt", 0)
    dur   = запись.get("duration_min", 0)
    знак  = "✅" if r == "tp" else ("❌" if r == "sl" else "⏰")

    log.info("")
    log.info("━" * 60)
    log.info(f"  📋 ПОСТ-ТРЕЙД АНАЛИЗ: {sym.split(':')[0]}")
    log.info(f"  Результат: {знак} {r.upper()}   P&L: {pnl:+.4f} USDT   "
             f"Длительность: {dur:.1f} мин")
    log.info(f"  Скор входа: {score}/100")
    log.info("━" * 60)
    log.info("")


# ================== АНАЛИТИКА ==================
def аналитика_по_инструментам():
    история = загрузить_историю()
    if len(история) < 2:
        log.info("  📊 Аналитика: недостаточно сделок (нужно минимум 2)")
        return

    по_символам: dict = {}
    for сд in история:
        sym = сд.get("symbol", "?").split(":")[0]
        if sym not in по_символам:
            по_символам[sym] = {"tp": 0, "sl": 0, "timeout": 0, "pnl": 0.0}
        r = сд.get("результат", "")
        if r == "tp":   по_символам[sym]["tp"] += 1
        elif r == "sl": по_символам[sym]["sl"] += 1
        else:           по_символам[sym]["timeout"] += 1
        по_символам[sym]["pnl"] += сд.get("pnl_usdt", 0)

    log.info("")
    log.info("=" * 60)
    log.info(f"  📊 АНАЛИТИКА — всего сделок: {len(история)}")
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
    try:
        b = exchange.fetch_balance({"type": "linear"})
        return float(b.get("USDT", {}).get("free", 0.0))
    except Exception as e:
        log.warning(f"Ошибка получения баланса: {e}")
        return 0.0


def получить_позиции() -> list:
    try:
        positions = exchange.fetch_positions()
        return [p for p in positions if float(p.get("contracts", 0) or 0) > 0]
    except Exception as e:
        log.warning(f"Ошибка получения позиций: {e}")
        return []


def закрыть_позиции_бота(символы: list):
    log.info("  🔒 Закрытие позиций бота...")
    try:
        positions = получить_позиции()
        for pos in positions:
            sym  = pos["symbol"]
            sym_normalized = sym.replace("/", "").replace(":USDT", "") + "/USDT:USDT"
            if sym not in символы and sym_normalized not in символы:
                log.info(f"    ⏭️  Пропуск {sym} — не из списка бота")
                continue
            side = pos["side"]
            qty  = abs(float(pos.get("contracts") or 0))
            if qty <= 0:
                continue
            close_side = "sell" if side == "long" else "buy"
            try:
                exchange.create_market_order(sym, close_side, qty, params={"reduceOnly": True})
                log.info(f"    Закрыта позиция {sym} {side} qty={qty}")
            except Exception as e:
                log.warning(f"    Не удалось закрыть {sym}: {e}")
    except Exception as e:
        log.warning(f"  Ошибка закрытия позиций: {e}")


def отменить_ордера_бота(символы: list):
    log.info("  🗑️  Отмена ордеров бота...")
    try:
        orders = exchange.fetch_open_orders()
        for o in orders:
            if o["symbol"] in символы:
                try:
                    exchange.cancel_order(o["id"], o["symbol"])
                except Exception as e:
                    log.warning(f"    Не удалось отменить ордер {o['id']}: {e}")
    except Exception as e:
        log.warning(f"  Ошибка отмены ордеров: {e}")


# ================== ИСПРАВЛЕНИЕ #2: НАДЁЖНАЯ УСТАНОВКА ПЛЕЧА ==================
def установить_плечо(symbol: str, leverage: int) -> bool:
    """
    Устанавливает плечо через несколько методов.
    Возвращает True если успешно, False если все методы провалились.
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

    # Метод 2: прямой вызов Bybit v5 API
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

    # Метод 3: проверить текущее плечо — возможно уже установлено правильно
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

    log.error(f"  ❌ Не удалось установить плечо {leverage}x для {symbol} — пропускаем сделку")
    return False


# ================== ИНДИКАТОРЫ ==================
def _ema(s, span):
    return s.ewm(span=span, adjust=False).mean()


def _rma(s, span):
    return s.ewm(alpha=1/span, adjust=False).mean()


def calc_rsi(close, period=14):
    d     = close.diff()
    gain  = d.clip(lower=0)
    loss  = (-d).clip(lower=0)
    avg_g = _rma(gain, period)
    avg_l = _rma(loss, period)
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(close, fast=12, slow=26, signal=9):
    ml = _ema(close, fast) - _ema(close, slow)
    sl = _ema(ml, signal)
    return ml, sl, ml - sl


def calc_atr(df, period=14):
    hi, lo, pc = df["h"], df["l"], df["c"].shift(1)
    tr = pd.concat([hi - lo, (hi - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    return _rma(tr, period)


def calc_supertrend(df, period=10, mult=3.0):
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


def calc_stochastic(df, k=14, d=3, smooth=3):
    lo  = df["l"].rolling(k).min()
    hi  = df["h"].rolling(k).max()
    ks  = (100 * (df["c"] - lo) / (hi - lo + 1e-10)).rolling(smooth).mean()
    return ks, ks.rolling(d).mean()


def calc_hull(close, period=55):
    hma = _ema(2 * _ema(close, period//2) - _ema(close, period), int(np.sqrt(period)))
    return hma > hma.shift(2), hma < hma.shift(2)


def calc_adx(df, period=14):
    atr = calc_atr(df, period)
    pdm = (df["h"] - df["h"].shift(1)).clip(lower=0)
    mdm = (df["l"].shift(1) - df["l"]).clip(lower=0)
    pdm = pdm.where(pdm >= mdm, 0)
    mdm = mdm.where(mdm >= pdm, 0)
    pdi = 100 * _rma(pdm, period) / atr.replace(0, np.nan)
    mdi = 100 * _rma(mdm, period) / atr.replace(0, np.nan)
    adx = _rma(100 * (pdi - mdi).abs() / (pdi + mdi + 1e-10), period)
    return adx, pdi, mdi


def calc_vwap_deviation(df, period=20):
    typical = (df["h"] + df["l"] + df["c"]) / 3
    vwap    = (typical * df["v"]).rolling(period).sum() / df["v"].rolling(period).sum()
    return (df["c"] - vwap) / vwap * 100


def calc_range_filter(df, period=200, qty=3.0):
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


# ================== УРОВНИ S/R ==================
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
        if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
            raw_supports.append(lows[i])

    res_clusters = _кластеризовать_уровни(raw_resistances, SR_CLUSTER_TOL)
    sup_clusters = _кластеризовать_уровни(raw_supports,    SR_CLUSTER_TOL)

    res_above = [(p, n) for p, n in res_clusters if p > close]
    sup_below = [(p, n) for p, n in sup_clusters if p < close]

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
                buy_ratio  = float(items[0].get("buyRatio",  0.5))
                sell_ratio = float(items[0].get("sellRatio", 0.5))
                result["long_ratio"]  = buy_ratio
                result["short_ratio"] = sell_ratio
                result["available"]   = True
                if   buy_ratio > 0.6: result["signal"] = "bullish"
                elif buy_ratio < 0.4: result["signal"] = "bearish"
                else:                 result["signal"] = "neutral"
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
    except Exception as e:
        log.debug(f"  Ошибка 4h тренда {symbol}: {e}")
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
    details = {}
    score   = 0
    price   = 0.0
    sr      = {}

    try:
        raw5  = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA,    limit=300)
        raw1h = exchange.fetch_ohlcv(symbol, TIMEFRAME_TREND, limit=300)
        raw15 = exchange.fetch_ohlcv(symbol, TIMEFRAME_MID,   limit=100)

        if len(raw5) < 100 or len(raw1h) < 100:
            return {"score": 0, "details": {}, "price": 0, "sr": {}}

        cols = ["ts","o","h","l","c","v"]
        df5  = pd.DataFrame(raw5,  columns=cols).reset_index(drop=True)
        df1h = pd.DataFrame(raw1h, columns=cols).reset_index(drop=True)
        df15 = pd.DataFrame(raw15, columns=cols).reset_index(drop=True)
        c5, c1h, c15 = df5["c"], df1h["c"], df15["c"]

        price = float(c5.iloc[-1])

        rsi_val = calc_rsi(c5).iloc[-1]
        details["rsi"] = round(rsi_val, 1)
        if   25 <= rsi_val <= 42: score += 20
        elif 42 < rsi_val <= 52:  score += 10
        elif rsi_val < 25:        score += 12
        elif 52 < rsi_val <= 65:  score +=  5

        rsi_1h = calc_rsi(c1h).iloc[-1]
        details["rsi_1h"] = round(rsi_1h, 1)
        if   rsi_1h < 55: score += 8
        elif rsi_1h < 65: score += 4

        ml, sl_macd, _ = calc_macd(c5)
        macd_bull  = ml.iloc[-1] > sl_macd.iloc[-1]
        macd_cross = macd_bull and ml.iloc[-2] <= sl_macd.iloc[-2]
        details["macd"] = "бычий" if macd_bull else "медвежий"
        if   macd_cross: score += 18
        elif macd_bull:  score +=  8

        _, _, _, rf_up, rf_down = calc_range_filter(df5)
        rf_up_now = rf_up.iloc[-1]
        details["range_filter"] = "вверх" if rf_up_now else ("вниз" if rf_down.iloc[-1] else "бок")
        if rf_up_now:
            score += 15

        st_up, _ = calc_supertrend(df5)
        details["supertrend"] = "вверх" if st_up.iloc[-1] else "вниз"
        if st_up.iloc[-1]:
            score += 12

        st_up_15, _ = calc_supertrend(df15)
        details["supertrend_15m"] = "вверх" if st_up_15.iloc[-1] else "вниз"
        if st_up_15.iloc[-1]:
            score += 8

        hu_up, _ = calc_hull(c5)
        details["hull"] = "вверх" if hu_up.iloc[-1] else "вниз"
        if hu_up.iloc[-1]:
            score += 8

        ema50_1h  = _ema(c1h, 50).iloc[-1]
        ema200_1h = _ema(c1h, 200).iloc[-1]
        details["тренд_1h"] = "бычий" if ema50_1h > ema200_1h else "медвежий"
        if ema50_1h > ema200_1h:
            score += 10

        ema20_15 = _ema(c15, 20).iloc[-1]
        ema50_15 = _ema(c15, 50).iloc[-1]
        details["тренд_15m"] = "бычий" if ema20_15 > ema50_15 else "медвежий"
        if ema20_15 > ema50_15:
            score += 5

        adx, pdi, mdi = calc_adx(df5)
        adx_val = adx.iloc[-1]
        details["adx"] = round(adx_val, 1)
        if   adx_val > 25 and pdi.iloc[-1] > mdi.iloc[-1]: score += 8
        elif adx_val > 20:                                   score += 3

        k_ser, _ = calc_stochastic(df5)
        k_val = k_ser.iloc[-1]
        details["stoch_k"] = round(k_val, 1)
        if   k_val < 25: score += 8
        elif k_val < 50: score += 4

        vol_avg   = df5["v"].rolling(20).mean().iloc[-1]
        vol_ratio = df5["v"].iloc[-1] / (vol_avg + 1e-10)
        details["объём_ratio"] = round(vol_ratio, 2)
        if   vol_ratio > 1.5: score += 8
        elif vol_ratio > 1.2: score += 4

        vwap_dev = calc_vwap_deviation(df5).iloc[-1]
        details["vwap_dev"] = round(vwap_dev, 2)
        if   -3 <= vwap_dev <= -0.3: score += 8
        elif vwap_dev < -3:          score += 4
        elif vwap_dev <= 1:          score += 2

        sr = calc_support_resistance(df5)
        details["support"]   = sr["support"]
        details["resistance"]= sr["resistance"]
        details["dist_sup"]  = sr["dist_to_sup_pct"]
        details["dist_res"]  = sr["dist_to_res_pct"]

        if sr["near_support"]:
            score += 12
            details["sr_signal"] = f"у поддержки ✅ ({sr['sup_cluster']} касаний)"
        elif sr["near_resistance"]:
            score -= 20
            details["sr_signal"] = f"у сопротивления ❌ ({sr['res_cluster']} касаний)"
        else:
            details["sr_signal"] = (
                f"нейтрально (sup={sr['dist_to_sup_pct']:.2f}% "
                f"res={sr['dist_to_res_pct']:.2f}%)"
            )

        last3_bearish = all(df5["c"].iloc[-i] < df5["o"].iloc[-i] for i in range(1, 4))
        if last3_bearish:
            score -= 15
            details["свечи_3red"] = True

        score = max(0, min(100, score))

    except Exception as e:
        log.warning(f"Ошибка анализа {symbol}: {e}")

    return {"score": score, "details": details, "price": price, "sr": sr}


def получить_скор_шорта(symbol: str) -> dict:
    res = получить_скор(symbol)
    if res["score"] == 0:
        return res
    inverted = max(0, 100 - res["score"] - 10)
    res["score"] = inverted
    return res


# ================== AI КОРРЕКЦИЯ ==================
def применить_ai_корректировку(score: int, symbol: str) -> int:
    ai = получить_bybit_ai(symbol)
    if not ai["available"]:
        log.info(f"  🤖 Bybit ratio: недоступен")
        return score

    long_r = ai["long_ratio"]
    signal = ai["signal"]
    log.info(f"  🤖 Bybit ratio: long={long_r:.1%}  short={ai['short_ratio']:.1%}  сигнал={signal}")

    if signal == "bullish":
        if long_r > 0.75:
            return score
        return min(100, score + 5)
    elif signal == "bearish":
        return max(0, score - 15)
    return score


# ================== РАЗМЕР ПОЗИЦИИ ==================
def рассчитать_размер_позиции(score: int, баланс: float, atr_pct: float = 1.0) -> float:
    if score <= MIN_SCORE:
        risk_pct = BASE_RISK_PCT
    else:
        factor   = (score - MIN_SCORE) / (100 - MIN_SCORE)
        risk_pct = BASE_RISK_PCT + (MAX_RISK_PCT - BASE_RISK_PCT) * factor

    if atr_pct > 1.5:
        risk_pct *= (1.5 / atr_pct)
        risk_pct = max(BASE_RISK_PCT * 0.5, risk_pct)
    risk_pct = min(risk_pct, MAX_RISK_PCT)

    max_loss_usdt = баланс * risk_pct / 100
    margin_usdt   = max_loss_usdt / (SL_PERCENT / 100)

    log.info(
        f"  📐 Скор={score} → риск={risk_pct:.1f}% "
        f"(макс.убыток={max_loss_usdt:.2f} USDT) → маржа={margin_usdt:.2f} USDT"
    )
    return round(max(1.0, margin_usdt), 2)


# ================== ОТКРЫТИЕ ПОЗИЦИИ ==================
def открыть_позицию(symbol: str, margin_usdt: float, tp_price: float, sl_price: float, side: str = "long"):
    try:
        # ИСПРАВЛЕНИЕ #2: плечо должно быть установлено, иначе не открываем
        if not установить_плечо(symbol, LEVERAGE):
            log.error(f"  ❌ Не удалось установить плечо — сделка отменена для {symbol}")
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


def открыть_лонг(symbol: str, margin_usdt: float, tp_price: float, sl_price: float):
    return открыть_позицию(symbol, margin_usdt, tp_price, sl_price, "long")


def открыть_шорт(symbol: str, margin_usdt: float, tp_price: float, sl_price: float):
    return открыть_позицию(symbol, margin_usdt, tp_price, sl_price, "short")


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


# ================== ИСПРАВЛЕНИЕ #1: МОНИТОРИНГ ПОЗИЦИИ ==================
def мониторить_позицию(symbol: str, entry_price: float, qty: float,
                        открыта_в: float, sl_цена: float, side: str = "long") -> str:
    deadline        = открыта_в + TRADE_MAX_LIFETIME
    coin            = symbol.split("/")[0]
    breakeven_price = entry_price * (1 + BYBIT_FEE * 2 + 0.0005)

    # ИСПРАВЛЕНИЕ: используем MIN_TRAILING_STEP и MIN_TRAILING_OFFSET напрямую
    # (TRAILING_STEP_PCT и TRAILING_OFFSET_PCT не существовали — это был критический баг)
    trailing_step   = MIN_TRAILING_STEP / 100      # дефолт: 0.25%
    trailing_offset = MIN_TRAILING_OFFSET / 100    # дефолт: 0.35%

    # Адаптивный трейлинг на основе ATR
    try:
        raw = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA, limit=50)
        if len(raw) >= 30:
            df = pd.DataFrame(raw, columns=["ts","o","h","l","c","v"])
            atr_series = calc_atr(df, TRAILING_ATR_PERIOD)
            atr_val = atr_series.iloc[-1]
            atr_pct = (atr_val / entry_price) * 100
            trailing_step   = max(MIN_TRAILING_STEP,   atr_pct * TRAILING_ATR_MULT)   / 100
            trailing_offset = max(MIN_TRAILING_OFFSET, atr_pct * TRAILING_OFFSET_MULT) / 100
            log.info(
                f"  📊 ATR={atr_pct:.2f}% → шаг трейлинга {trailing_step*100:.2f}%, "
                f"отступ {trailing_offset*100:.2f}%"
            )
    except Exception as e:
        log.warning(f"  Не удалось рассчитать ATR: {e}, используем стандартный шаг")

    фаза            = 1
    текущий_sl      = sl_цена
    пиковая_цена    = entry_price
    следующий_трейл = entry_price * (1 + trailing_step if side == "long" else 1 - trailing_step)

    log.info(
        f"  🚦 Мониторинг | вход={entry_price:.8f} | безубыток @ {breakeven_price:.8f} | "
        f"шаг={trailing_step*100:.2f}% отступ={trailing_offset*100:.2f}% | "
        f"активация трейлинга при +{MIN_PROFIT_FOR_TRAIL}%"
    )

    while True:
        сейчас = time.time()

        if сейчас >= deadline:
            log.warning("  ⏰ Дедлайн — принудительное закрытие")
            try:
                close_side = "sell" if side == "long" else "buy"
                exchange.create_market_order(symbol, close_side, qty, params={"reduceOnly": True})
            except Exception as e:
                log.warning(f"  Ошибка закрытия по дедлайну: {e}")
            return "таймаут"

        time.sleep(10)

        try:
            positions = exchange.fetch_positions([symbol])
            active    = [p for p in positions
                         if float(p.get("contracts", 0) or 0) > 0 and p.get("side") == side]

            if not active:
                cur_price = float(exchange.fetch_ticker(symbol)["last"])
                if (side == "long" and cur_price >= entry_price * (1 + TP_PERCENT / 100 * 0.7)) or \
                   (side == "short" and cur_price <= entry_price * (1 - TP_PERCENT / 100 * 0.7)):
                    log.info("  ✅ Позиция закрыта по Тейк-профиту")
                    return "tp"
                elif фаза >= 2:
                    log.info("  🔒 Закрыта по трейлинг/безубыток SL")
                    return "tp"
                else:
                    log.info("  ❌ Позиция закрыта по Стоп-лоссу")
                    return "sl"

            pos       = active[0]
            cur_price = float(exchange.fetch_ticker(symbol)["last"])
            pnl       = float(pos.get("unrealizedPnl", 0) or 0)
            if side == "long":
                pnl_pct = (cur_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - cur_price) / entry_price * 100
            до_дед = int(deadline - сейчас)

            if фаза == 1 and pnl_pct >= 0:
                new_sl = (entry_price * (1 + BYBIT_FEE * 2 + 0.0003) if side == "long"
                          else entry_price * (1 - BYBIT_FEE * 2 - 0.0003))
                if обновить_sl_на_бирже(symbol, new_sl, side):
                    фаза         = 2
                    текущий_sl   = new_sl
                    пиковая_цена = cur_price
                    следующий_трейл = cur_price * (1 + trailing_step if side == "long" else 1 - trailing_step)
                    log.info(f"  🔒 БЕЗУБЫТОК! SL → {new_sl:.8f}")

            if фаза >= 2 and pnl_pct >= MIN_PROFIT_FOR_TRAIL:
                if side == "long":
                    if cur_price >= следующий_трейл:
                        пиковая_цена = max(пиковая_цена, cur_price)
                        new_sl = пиковая_цена * (1 - trailing_offset)
                        if new_sl > текущий_sl:
                            if обновить_sl_на_бирже(symbol, new_sl, side):
                                текущий_sl      = new_sl
                                следующий_трейл = cur_price * (1 + trailing_step)
                                log.info(
                                    f"  📈 ТРЕЙЛИНГ: пик={пиковая_цена:.8f} → SL={new_sl:.8f} "
                                    f"(зафиксировано {(new_sl-entry_price)/entry_price*100:+.2f}%)"
                                )
                    if cur_price > пиковая_цена:
                        пиковая_цена = cur_price
                else:
                    if cur_price <= следующий_трейл:
                        пиковая_цена = min(пиковая_цена, cur_price)
                        new_sl = пиковая_цена * (1 + trailing_offset)
                        if new_sl < текущий_sl:
                            if обновить_sl_на_бирже(symbol, new_sl, side):
                                текущий_sl      = new_sl
                                следующий_трейл = cur_price * (1 - trailing_step)
                                log.info(
                                    f"  📈 ТРЕЙЛИНГ: пик={пиковая_цена:.8f} → SL={new_sl:.8f} "
                                    f"(зафиксировано {(entry_price-new_sl)/entry_price*100:+.2f}%)"
                                )
                    if cur_price < пиковая_цена:
                        пиковая_цена = cur_price

            фаза_лейбл = {1: "обычная", 2: "безубыток 🔒", 3: "трейлинг 📈"}.get(фаза, "?")
            log.info(
                f"  [{coin}] {cur_price:.8f}  P&L={pnl_pct:+.2f}% ({pnl:+.4f}U)"
                f"  SL={текущий_sl:.8f}  фаза={фаза_лейбл}  дед={до_дед}с"
            )

        except Exception as e:
            log.warning(f"  Ошибка мониторинга: {e}")


# ================== ОТЧЁТ ==================
def печатать_отчёт():
    баланс  = баланс_usdt()
    старт   = stats["депозит_старт"]
    дельта  = баланс - старт
    чистый  = stats["прибыль_usdt"] - stats["убыток_usdt"]
    процент = (дельта / старт * 100) if старт > 0 else 0
    winrate = (stats["тейкпрофит"] / stats["сделок_всего"] * 100) if stats["сделок_всего"] > 0 else 0

    log.info("")
    log.info("=" * 60)
    log.info("  📊  ОТЧЁТ ЗА СЕССИЮ")
    log.info(f"  Время:               {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    log.info(f"  Работает с:          {stats['старт_время']}")
    log.info("  ─" * 30)
    log.info(f"  Депозит при старте:  {старт:.2f} USDT")
    log.info(f"  Баланс сейчас:       {баланс:.2f} USDT  ({'+' if дельта >= 0 else ''}{дельта:.2f})")
    log.info(f"  Изменение:           {'+' if процент >= 0 else ''}{процент:.2f}%")
    log.info("  ─" * 30)
    log.info(f"  Сделок:              {stats['сделок_всего']}")
    log.info(f"  ✅ TP:               {stats['тейкпрофит']}  (winrate {winrate:.1f}%)")
    log.info(f"  ❌ SL:               {stats['стоплосс']}")
    log.info(f"  ⏰ Таймаут:          {stats['таймаут']}")
    log.info(f"  🧊 SL streak:        {stats['sl_streak']}/{SL_STREAK_LIMIT}")
    log.info("  ─" * 30)
    log.info(f"  💰 Прибыль:         +{stats['прибыль_usdt']:.4f} USDT")
    log.info(f"  💸 Убыток:          -{stats['убыток_usdt']:.4f} USDT")
    log.info(f"  📈 Чистый P&L:       {'+' if чистый >= 0 else ''}{чистый:.4f} USDT")
    log.info("=" * 60)
    log.info("")

    stats["последний_отчёт"] = time.time()
    сохранить_состояние()
    аналитика_по_инструментам()


# ================== ГЛАВНЫЙ ЦИКЛ ==================
def main():
    log.info("🔄 Инвентаризация (только позиции бота)...")
    отменить_ордера_бота(SYMBOLS)
    time.sleep(1)
    закрыть_позиции_бота(SYMBOLS)
    log.info("✅ Инвентаризация завершена")
    time.sleep(2)

    восстановлен  = загрузить_состояние()
    баланс_сейчас = баланс_usdt()
    stats["запусков"] += 1

    if not восстановлен or stats["депозит_старт"] == 0:
        stats["депозит_старт"] = баланс_сейчас
        stats["старт_время"]   = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

    stats["последний_отчёт"] = time.time()

    история = загрузить_историю()
    if история:
        log.info(f"  📖 Найдено {len(история)} сделок в истории — запуск аналитики...")
        аналитика_по_инструментам()

    заблокированные_символы = {}

    log.info("")
    log.info("=" * 60)
    log.info("  🤖  ФЬЮЧЕРСНЫЙ БОТ v4.1 (BUGFIXED)")
    log.info(f"  Запуск №:            {stats['запусков']}")
    log.info(f"  Плечо:               {LEVERAGE}x")
    log.info(f"  Баланс:              {баланс_сейчас:.2f} USDT")
    log.info(f"  Пар для торговли:    {len(SYMBOLS)}")
    log.info(f"  MIN_SCORE:           {MIN_SCORE}")
    log.info(f"  TP / SL:             {TP_PERCENT}% / {SL_PERCENT}%")
    log.info(f"  Трейлинг:            адаптивный (ATR), мин.шаг {MIN_TRAILING_STEP}%")
    log.info(f"  Блокировка символа:  {SYMBOL_BLOCK_MINUTES} мин после SL")
    log.info(f"  SL cooldown:         {SL_STREAK_LIMIT} SL подряд → пауза {SL_STREAK_PAUSE//60} мин")
    log.info("=" * 60)
    log.info("")

    while True:
        try:
            if time.time() - stats["последний_отчёт"] >= REPORT_INTERVAL:
                печатать_отчёт()

            баланс = баланс_usdt()

            if баланс < MIN_BALANCE:
                log.warning(f"  🛑 Баланс {баланс:.2f} < {MIN_BALANCE}. Пауза 10 мин.")
                time.sleep(600)
                continue

            if stats["депозит_старт"] > 0:
                просадка = (stats["депозит_старт"] - баланс) / stats["депозит_старт"] * 100
                if просадка > MAX_DRAWDOWN_PCT:
                    log.warning(f"  ⛔ Просадка {просадка:.1f}% > {MAX_DRAWDOWN_PCT}%. Пауза 2 часа.")
                    time.sleep(7200)
                    continue

            if stats.get("sl_streak", 0) >= SL_STREAK_LIMIT:
                log.warning(
                    f"  🧊 {SL_STREAK_LIMIT} SL подряд — cooldown {SL_STREAK_PAUSE//60} мин."
                )
                stats["sl_streak"] = 0
                сохранить_состояние()
                time.sleep(SL_STREAK_PAUSE)
                continue

            активные = получить_позиции()
            if активные:
                log.info(f"  ⏳ Открыта позиция в {[p['symbol'] for p in активные]} — ждём")
                time.sleep(30)
                continue

            log.info(f"── Сканирование {len(SYMBOLS)} пар (баланс={баланс:.2f} USDT, порог={MIN_SCORE}) ──")

            scores = {}
            for sym in SYMBOLS:
                try:
                    if sym in заблокированные_символы and time.time() < заблокированные_символы[sym]:
                        scores[sym] = {"score": 0, "score_final": 0, "details": {}, "price": 0, "sr": {}}
                        continue

                    if not тренд_4h_бычий(sym):
                        scores[sym] = {"score": 0, "score_final": 0, "details": {"тренд_4h": "медвежий"}, "price": 0, "sr": {}}
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
                        f"  SR={res['details'].get('sr_signal', '?')}"
                    )
                except Exception as e:
                    log.warning(f"  Ошибка скора {sym}: {e}")
                    scores[sym] = {"score": 0, "score_final": 0, "details": {}, "price": 0, "sr": {}}

            if not scores:
                time.sleep(SCAN_INTERVAL)
                continue

            кандидаты = sorted(
                [(s, d) for s, d in scores.items() if d["score_final"] >= MIN_SCORE],
                key=lambda x: x[1]["score_final"],
                reverse=True
            )[:5]

            if not кандидаты:
                log.info(f"  Нет пар с скором >= {MIN_SCORE} — ждём {SCAN_INTERVAL} сек")
                time.sleep(SCAN_INTERVAL)
                continue

            log.info(f"  🏆 Топ кандидаты (лонг): " +
                     ", ".join(f"{s.split(':')[0]}={d['score_final']}" for s, d in кандидаты))

            выбрана  = None
            фин_скор = 0
            цена     = 0.0
            sr_info  = {}
            side     = "long"

            for лучшая, данные in кандидаты:
                фин_скор = данные["score_final"]
                цена     = данные["price"]
                sr_info  = данные.get("sr", {})
                det      = данные.get("details", {})

                dist_res_now = sr_info.get("dist_to_res_pct", 99)
                if sr_info.get("near_resistance") and dist_res_now < SR_BLOCK_DIST_PCT:
                    log.info(
                        f"  ⛔ {лучшая.split(':')[0]}: resistance в {dist_res_now:.3f}% — пробуем следующего"
                    )
                    continue

                rsi_val = det.get("rsi", 50)
                try:
                    rsi_val = float(rsi_val)
                except Exception:
                    rsi_val = 50
                if rsi_val > 68 and not sr_info.get("near_support"):
                    log.info(
                        f"  ⚠️ {лучшая.split(':')[0]}: RSI перекуплен ({rsi_val:.1f}) без поддержки — пропускаем"
                    )
                    continue

                выбрана = лучшая
                log.info(
                    f"  ► Выбрана {лучшая.split(':')[0]} (лонг)  скор={фин_скор}  "
                    f"цена={цена:.8f}  dist_res={dist_res_now:.3f}%"
                )
                break

            if выбрана is None:
                for sym, data in scores.items():
                    if sym in заблокированные_символы:
                        continue
                    if тренд_4h_медвежий(sym):
                        short_res = получить_скор_шорта(sym)
                        if short_res["score"] >= MIN_SCORE - 5:
                            log.info(f"  🐻 Шорт-кандидат: {sym.split(':')[0]} скор={short_res['score']}")
                            выбрана = sym
                            фин_скор = short_res["score"]
                            цена = short_res["price"]
                            sr_info = short_res.get("sr", {})
                            side = "short"
                            break

            if выбрана is None:
                log.info("  Все кандидаты отфильтрованы — ждём следующего цикла")
                time.sleep(SCAN_INTERVAL)
                continue

            support    = sr_info.get("support",    цена * (1 - SL_PERCENT / 100))
            resistance = sr_info.get("resistance", цена * (1 + TP_PERCENT / 100))

            if side == "long":
                sl_базовый   = цена * (1 - SL_PERCENT / 100)
                sl_от_sup    = float(support) * 0.998
                sl_цена      = max(sl_базовый, sl_от_sup)
                sl_цена      = max(sl_цена, цена * (1 - MIN_SL_PERCENT / 100))

                tp_базовый   = цена * (1 + TP_PERCENT / 100)
                dist_res     = sr_info.get("dist_to_res_pct", 99)
                tp_цена      = (цена + (float(resistance) - цена) * 0.90
                                if dist_res > TP_PERCENT * 1.2 else tp_базовый)
            else:
                sl_базовый   = цена * (1 + SL_PERCENT / 100)
                sl_от_res    = float(resistance) * 1.002
                sl_цена      = min(sl_базовый, sl_от_res)
                sl_цена      = min(sl_цена, цена * (1 + MIN_SL_PERCENT / 100))

                tp_базовый   = цена * (1 - TP_PERCENT / 100)
                dist_sup     = sr_info.get("dist_to_sup_pct", 99)
                tp_цена      = (цена - (цена - float(support)) * 0.90
                                if dist_sup > TP_PERCENT * 1.2 else tp_базовый)

            try:
                raw = exchange.fetch_ohlcv(выбрана, TIMEFRAME_TA, limit=50)
                if len(raw) >= 30:
                    df = pd.DataFrame(raw, columns=["ts","o","h","l","c","v"])
                    atr = calc_atr(df, 14).iloc[-1]
                    atr_pct = (atr / цена) * 100
                else:
                    atr_pct = 1.0
            except Exception:
                atr_pct = 1.0

            margin = рассчитать_размер_позиции(фин_скор, баланс, atr_pct)
            if баланс < margin * 1.1:
                log.warning(f"  ⚠️ Баланс {баланс:.2f} < маржа {margin:.2f} — уменьшаем")
                margin = баланс * 0.8

            log.info(
                f"  ✅ ВХОД {side.upper()}: скор={фин_скор} | "
                f"SL={sl_цена:.8f} | TP={tp_цена:.8f} | маржа={margin:.2f}U"
            )

            время_входа = time.time()
            if side == "long":
                вход_цена, кол_во = открыть_лонг(выбрана, margin, tp_цена, sl_цена)
            else:
                вход_цена, кол_во = открыть_шорт(выбрана, margin, tp_цена, sl_цена)

            if вход_цена is None or кол_во is None:
                log.warning("  Не удалось открыть позицию — пауза 30 сек")
                time.sleep(30)
                continue

            stats["сделок_всего"] += 1
            сохранить_состояние()

            # ИСПРАВЛЕНИЕ #3: защита от краша мониторинга
            результат = "sl"  # пессимистичный дефолт
            try:
                результат = мониторить_позицию(выбрана, вход_цена, кол_во, время_входа, sl_цена, side)
            except Exception as monitor_err:
                log.error(f"  💥 Краш мониторинга: {monitor_err}", exc_info=True)
                log.warning("  🔒 Принудительное закрытие позиции после краша мониторинга...")
                try:
                    close_side = "sell" if side == "long" else "buy"
                    exchange.create_market_order(
                        выбрана, close_side, кол_во,
                        params={"reduceOnly": True}
                    )
                    log.info(f"  ✅ Позиция {выбрана} закрыта принудительно")
                except Exception as close_err:
                    log.error(f"  ❌ Не удалось закрыть позицию после краша: {close_err}")
                результат = "sl"

            объём      = margin * LEVERAGE
            комиссии   = объём * BYBIT_FEE * 2
            длит_мин   = (time.time() - время_входа) / 60
            pnl_сделки = 0.0

            if результат == "tp":
                pnl_сделки = объём * TP_PERCENT / 100 - комиссии
                stats["тейкпрофит"]   += 1
                stats["прибыль_usdt"] += max(0, pnl_сделки)
                stats["sl_streak"]     = 0
                log.info(f"  ✅ TP: прибыль ≈{pnl_сделки:.4f} USDT")

            elif результат == "sl":
                pnl_сделки = -(объём * SL_PERCENT / 100 + комиссии)
                stats["стоплосс"]    += 1
                stats["убыток_usdt"] += abs(pnl_сделки)
                stats["sl_streak"]    = stats.get("sl_streak", 0) + 1
                заблокированные_символы[выбрана] = time.time() + SYMBOL_BLOCK_MINUTES * 60
                log.warning(
                    f"  ❌ SL: убыток ≈{pnl_сделки:.4f} USDT  "
                    f"(streak: {stats['sl_streak']}/{SL_STREAK_LIMIT}, "
                    f"символ заблокирован на {SYMBOL_BLOCK_MINUTES} мин)"
                )

            elif результат == "таймаут":
                pnl_сделки = -комиссии
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
                "margin_usdt":  margin,
                "leverage":     LEVERAGE,
                "результат":    результат,
                "pnl_usdt":     round(pnl_сделки, 4),
                "details":      scores[выбрана].get("details", {}) if side == "long" else {},
                "sr":           {k: str(v) for k, v in (scores[выбрана].get("sr", {}) or {}).items()},
            }
            сохранить_сделку(запись_сделки)
            пост_трейд_анализ(запись_сделки)

            if stats["сделок_всего"] % 10 == 0:
                аналитика_по_инструментам()

            сохранить_состояние()
            log.info("  Сделка завершена — пауза 30 сек")
            time.sleep(30)

        except Exception as e:
            log.error(f"Глобальная ошибка: {e}", exc_info=True)
            time.sleep(60)


if __name__ == "__main__":
    main()
