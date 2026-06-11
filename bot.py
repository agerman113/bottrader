#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
Bybit ГИБРИДНЫЙ БОТ v12.0 — АНТИПОТЕРЬ
================================================================================
КЛЮЧЕВЫЕ ИСПРАВЛЕНИЯ (анализ 0% WinRate из логов v10.4):

ПРОБЛЕМА 1: Шорты на бычьем рынке
→ Полный запрет шортов при long_ratio > 55% на Bybit
→ При long_ratio > 65% вообще не торгуем шорты

ПРОБЛЕМА 2: Слабые фильтры входа
→ MIN_SCORE повышен до 82
→ Минимальный RR = 2.5:1
→ Требование: минимум 4 из 6 ключевых индикаторов совпадают

ПРОБЛЕМА 3: NEAR/USDT с RR 0.5:1 прорывался
→ Жёсткий RR-фильтр ПЕРЕД скорингом
→ Пересчёт SL/TP только через ATR (не фиксированный процент)

ПРОБЛЕМА 4: Выход слишком поздно
→ Exit signal: выход если 2 из 3 индикаторов развернулись
→ Трейлинг активируется раньше (RR_EXIT = 0.2)

ПРОБЛЕМА 5: Торговля против тренда 4h
→ Добавлен обязательный фильтр тренда на 4h
→ Лонг только если 4h бычий, шорт только если 4h медвежий

ПРОБЛЕМА 6: Подтверждение входа (1 бар)
→ Вход откладывается на 1 бар после сигнала (ENTRY_CONFIRM_BARS=1)
================================================================================
"""

import os
import sys
import time
import json
import logging
import requests
import ccxt
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

# ============================================================
#                   КОНСТАНТЫ И НАСТРОЙКИ
# ============================================================

SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT", "XRP/USDT:USDT",
    "SOL/USDT:USDT", "ADA/USDT:USDT", "TRX/USDT:USDT", "TON/USDT:USDT",
    "AVAX/USDT:USDT", "DOT/USDT:USDT", "LTC/USDT:USDT", "BCH/USDT:USDT",
    "XLM/USDT:USDT", "NEAR/USDT:USDT", "DOGE/USDT:USDT",
    "WIF/USDT:USDT", "RENDER/USDT:USDT", "WLD/USDT:USDT",
    "SOL/USDT:USDT", "LINK/USDT:USDT", "OP/USDT:USDT", "ARB/USDT:USDT",
]

# ============================================================
# КЛЮЧЕВЫЕ ПАРАМЕТРЫ (исправлены по анализу логов)
# ============================================================

LEVERAGE = 3
TIMEFRAME_TA   = "5m"
TIMEFRAME_TREND = "1h"
TIMEFRAME_4H   = "4h"
SCAN_INTERVAL  = 60

# --- ПОРОГИ СКОРА ---
MIN_SCORE = 82                # Было 60/75 — повышаем
MIN_INDICATORS_ALIGNED = 4   # Минимум совпадений из 6 ключевых индикаторов (НОВОЕ)

# --- TP / SL ---
ATR_SL_MULT    = 1.2          # SL = 1.2 × ATR
ATR_TP_MULT    = 3.0          # TP = 3.0 × ATR → RR ≥ 2.5
MIN_SL_PERCENT = 0.5
MAX_SL_PERCENT = 2.5
MIN_RR         = 2.5          # Минимальный Risk/Reward (было 1.5, потом 2.0)

# --- BYBIT RATIO ФИЛЬТРЫ (главное исправление!) ---
# Шорт запрещён если long_ratio выше этого порога
SHORT_MAX_LONG_RATIO = 0.55   # При >55% лонгов — шорты запрещены
# Лонг запрещён если long_ratio ниже этого порога
LONG_MIN_LONG_RATIO  = 0.45   # При <45% лонгов — лонги запрещены
# Усиление сигнала
RATIO_STRONG_LONG    = 0.65   # >65% — только лонги
RATIO_STRONG_SHORT   = 0.35   # <35% — только шорты

# --- ЧАСТИЧНЫЙ БЕЗУБЫТОК ---
PARTIAL_BE_ENABLED   = True
PARTIAL_BE_CLOSE_PCT = 50.0
PARTIAL_BE_PROFIT    = 0.08   # 0.08% прибыли для частичного закрытия

# --- РИСК ---
BASE_RISK_PCT = 0.7           # Уменьшено с 0.8
MAX_RISK_PCT  = 1.0           # Уменьшено с 1.2

# --- ТРЕЙЛИНГ ---
TRAILING_ATR_PERIOD    = 14
TRAILING_OFFSET_MULT   = 1.0
MIN_TRAILING_OFFSET    = 0.3
MIN_PROFIT_FOR_TRAIL   = 0.2
RR_EXIT_TRIGGER        = 0.2  # Трейлинг активируется при 20% пути к TP

# --- EXIT SIGNAL ---
SIGNAL_EXIT_ENABLED          = True
EXIT_SIGNAL_CHECK_INTERVAL   = 10   # секунд
EXIT_MIN_REVERSED_INDICATORS = 2    # Минимум индикаторов развернулись для выхода

# --- БЛОКИРОВКИ ---
SYMBOL_BLOCK_AFTER_TP    = 30
SYMBOL_BLOCK_AFTER_SL    = 180    # Увеличена пауза после SL
SYMBOL_MAX_FAIL_ATTEMPTS = 3
SYMBOL_BLOCK_AFTER_FAIL  = 60

SL_STREAK_LIMIT     = 2      # Пауза уже после 2 SL подряд (было 3)
SL_STREAK_PAUSE     = 3600
SL_STREAK_EXTRA_PAUSE = 300

MIN_BALANCE       = 10.0
MAX_DRAWDOWN_PCT  = 12.0     # Было 15%

DAILY_LOSS_LIMIT_PCT  = 3.0
DAILY_LOSS_PAUSE_SEC  = 7200

TRADE_MAX_LIFETIME = 1800
REPORT_INTERVAL    = 1800

# --- S/R ---
SR_PERIOD       = 100
SR_PROXIMITY_PCT = 0.5
SR_MIN_TOUCHES  = 2
SR_CLUSTER_TOL  = 0.005
SR_BLOCK_DIST_PCT = 0.4      # Уменьшено — строже фильтр

# --- ОБЪЁМ ---
VOLUME_AVG_PERIOD = 20
MIN_VOLUME_RATIO  = 0.6
VOLUME_SPIKE_MULT = 3.0

# --- API ---
API_CALL_DELAY         = 0.25
API_RATE_LIMIT_PAUSE   = 3

# --- ПОДТВЕРЖДЕНИЕ ВХОДА ---
ENTRY_CONFIRM_BARS = 1   # Подождать 1 бар после сигнала

# --- ФАЙЛЫ ---
STATE_FILE            = "state_bot_v12_0.json"
TRADES_FILE           = "trades_bot_v12_0.json"
INDICATOR_STATS_FILE  = "indicator_stats_v12_0.json"
METRICS_FILE          = "strategy_metrics_v12_0.json"

BYBIT_FEE = 0.00055

# ============================================================
#                       ЛОГИРОВАНИЕ
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot_v12_0.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ============================================================
#                         БИРЖА
# ============================================================

exchange = ccxt.bybit({
    "apiKey": os.getenv("BYBIT_API_KEY"),
    "secret": os.getenv("BYBIT_API_SECRET"),
    "enableRateLimit": True,
    "options": {"defaultType": "linear"},
})

# ============================================================
#                    СТАТИСТИКА БОТА
# ============================================================

stats = {
    "запусков": 0,
    "сделок_всего": 0,
    "тейкпрофит": 0,
    "стоплосс": 0,
    "таймаут": 0,
    "прибыль_usdt": 0.0,
    "убыток_usdt": 0.0,
    "депозит_старт": 0.0,
    "баланс_начало_дня": 0.0,
    "дата_дня": "",
    "старт_время": "",
    "последний_отчёт": 0.0,
    "sl_streak": 0,
}

# ============================================================
#           БЕЗОПАСНЫЕ API ВЫЗОВЫ
# ============================================================

def safe_api_call(func, *args, retries=3, delay=1.0, ignore_errors=None, **kwargs):
    ignore_errors = ignore_errors or []
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except ccxt.RateLimitExceeded:
            log.warning(f"Rate limit, пауза {API_RATE_LIMIT_PAUSE}с (попытка {attempt+1}/{retries})")
            time.sleep(API_RATE_LIMIT_PAUSE)
        except ccxt.NetworkError as e:
            log.warning(f"Сетевая ошибка: {e}, пауза {delay}с")
            time.sleep(delay)
            delay *= 2
        except Exception as e:
            err_str = str(e)
            for ignore in ignore_errors:
                if ignore.lower() in err_str.lower():
                    return None
            log.error(f"API ошибка: {e} (попытка {attempt+1}/{retries})")
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                raise
    return None

def safe_fetch_ohlcv(symbol: str, timeframe: str, limit: int = 300) -> List:
    try:
        r = safe_api_call(exchange.fetch_ohlcv, symbol, timeframe, limit=limit, retries=3)
        return r if r is not None else []
    except Exception:
        return []

def safe_fetch_ticker(symbol: str) -> Optional[dict]:
    try:
        return safe_api_call(exchange.fetch_ticker, symbol, retries=3)
    except Exception:
        return None

def safe_fetch_positions(symbols: Optional[List[str]] = None) -> List[dict]:
    try:
        if symbols:
            r = safe_api_call(exchange.fetch_positions, symbols, retries=3)
        else:
            r = safe_api_call(exchange.fetch_positions, retries=3)
        return r if r is not None else []
    except Exception as e:
        log.warning(f"safe_fetch_positions: {e}")
        return []

# ============================================================
#              БАЗОВЫЕ ИНДИКАТОРЫ
# ============================================================

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _rma(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(alpha=1/span, adjust=False).mean()

def _sma(s: pd.Series, span: int) -> pd.Series:
    return s.rolling(span).mean()

def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d    = close.diff()
    gain = d.clip(lower=0)
    loss = (-d).clip(lower=0)
    avg_g = _rma(gain, period)
    avg_l = _rma(loss, period)
    rs = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(close: pd.Series, fast=12, slow=26, signal=9):
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
        else:                              trend.iloc[i] =  pt
    return trend == 1, trend == -1

def calc_stochastic(df: pd.DataFrame, k=14, d=3, smooth=3):
    lo = df["l"].rolling(k).min()
    hi = df["h"].rolling(k).max()
    ks = (100 * (df["c"] - lo) / (hi - lo + 1e-10)).rolling(smooth).mean()
    return ks, ks.rolling(d).mean()

def calc_hull(close: pd.Series, period: int = 55):
    hma = _ema(2 * _ema(close, period//2) - _ema(close, period), int(np.sqrt(period)))
    return hma > hma.shift(2), hma < hma.shift(2)

def calc_adx(df: pd.DataFrame, period: int = 14):
    atr  = calc_atr(df, period)
    pdm  = (df["h"] - df["h"].shift(1)).clip(lower=0)
    mdm  = (df["l"].shift(1) - df["l"]).clip(lower=0)
    pdm  = pdm.where(pdm >= mdm, 0)
    mdm  = mdm.where(mdm >= pdm, 0)
    pdi  = 100 * _rma(pdm, period) / atr.replace(0, np.nan)
    mdi  = 100 * _rma(mdm, period) / atr.replace(0, np.nan)
    adx  = _rma(100 * (pdi - mdi).abs() / (pdi + mdi + 1e-10), period)
    return adx, pdi, mdi

def calc_range_filter(df: pd.DataFrame, period: int = 100, qty: float = 2.5):
    close = df["c"]
    rng   = qty * calc_atr(df, period)
    filt  = close.copy()
    for i in range(1, len(close)):
        c, r, pf = close.iloc[i], rng.iloc[i], filt.iloc[i-1]
        if   c - r > pf: filt.iloc[i] = c - r
        elif c + r < pf: filt.iloc[i] = c + r
        else:             filt.iloc[i] = pf
    up   = (filt > filt.shift(1)) & (close > filt)
    down = (filt < filt.shift(1)) & (close < filt)
    return filt, filt + rng, filt - rng, up, down

def calc_support_resistance(df: pd.DataFrame, period: int = SR_PERIOD) -> dict:
    df_sr  = df.tail(period).reset_index(drop=True)
    highs  = df_sr["h"].values
    lows   = df_sr["l"].values
    close  = float(df["c"].iloc[-1])
    raw_res, raw_sup = [], []

    for i in range(2, len(highs) - 2):
        if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
            highs[i] > highs[i+1] and highs[i] > highs[i+2]):
            raw_res.append(highs[i])
        if (lows[i]  <  lows[i-1] and lows[i]  <  lows[i-2] and
            lows[i]  <  lows[i+1] and lows[i]  <  lows[i+2]):
            raw_sup.append(lows[i])

    def кластер(levels):
        if not levels: return []
        levels = sorted(levels)
        out, cur = [], [levels[0]]
        for lvl in levels[1:]:
            if (lvl - cur[0]) / (cur[0] + 1e-10) < SR_CLUSTER_TOL:
                cur.append(lvl)
            else:
                out.append((float(np.mean(cur)), len(cur)))
                cur = [lvl]
        out.append((float(np.mean(cur)), len(cur)))
        return out

    res_cl = кластер(raw_res)
    sup_cl = кластер(raw_sup)
    res_above = sorted([(p, n) for p, n in res_cl if p > close], key=lambda x: x[0])
    sup_below = sorted([(p, n) for p, n in sup_cl if p < close], key=lambda x: x[0], reverse=True)

    nearest_res, res_n = res_above[0] if res_above else (close * 1.05, 0)
    nearest_sup, sup_n = sup_below[0] if sup_below else (close * 0.95, 0)
    dist_res = (nearest_res - close) / close * 100
    dist_sup = (close - nearest_sup) / close * 100
    near_sup = dist_sup < SR_PROXIMITY_PCT and sup_n >= SR_MIN_TOUCHES
    near_res = dist_res < SR_PROXIMITY_PCT and res_n >= SR_MIN_TOUCHES

    return {
        "support": round(nearest_sup, 10), "resistance": round(nearest_res, 10),
        "dist_to_sup_pct": round(dist_sup, 2), "dist_to_res_pct": round(dist_res, 2),
        "sup_cluster": sup_n, "res_cluster": res_n,
        "near_support": near_sup, "near_resistance": near_res,
    }

# ============================================================
#        BYBIT RATIO (ключевой фильтр)
# ============================================================

def получить_bybit_ratio(symbol: str) -> dict:
    result = {"signal": "neutral", "long_ratio": 0.5, "short_ratio": 0.5, "available": False}
    try:
        coin = symbol.split("/")[0]
        url  = f"https://api.bybit.com/v5/market/account-ratio?category=linear&symbol={coin}USDT&period=1h&limit=1"
        resp = requests.get(url, timeout=5)
        data = resp.json()
        if data.get("retCode") == 0:
            items = data.get("result", {}).get("list", [])
            if items:
                buy_r  = float(items[0].get("buyRatio", 0.5))
                sell_r = float(items[0].get("sellRatio", 0.5))
                result.update({"long_ratio": buy_r, "short_ratio": sell_r, "available": True})
                if buy_r > 0.55:   result["signal"] = "bullish"
                elif buy_r < 0.45: result["signal"] = "bearish"
    except Exception as e:
        log.debug(f"Bybit ratio недоступен: {e}")
    return result

# ============================================================
#    ТРЕНД 4H (обязательный фильтр направления)
# ============================================================

def получить_тренд_4h(symbol: str) -> str:
    """Возвращает 'bull', 'bear' или 'neutral'"""
    try:
        raw = safe_fetch_ohlcv(symbol, TIMEFRAME_4H, limit=60)
        if len(raw) < 50: return "neutral"
        df  = pd.DataFrame(raw, columns=["ts","o","h","l","c","v"])
        c   = df["c"]
        ema20 = _ema(c, 20).iloc[-1]
        ema50 = _ema(c, 50).iloc[-1]
        # Наклон EMA
        ema20_slope = (_ema(c, 20).iloc[-1] - _ema(c, 20).iloc[-4]) / _ema(c, 20).iloc[-4] * 100
        adx_val, pdi, mdi = calc_adx(df)
        adx = adx_val.iloc[-1]
        if ema20 > ema50 and ema20_slope > 0 and adx > 20:
            return "bull"
        elif ema20 < ema50 and ema20_slope < 0 and adx > 20:
            return "bear"
        return "neutral"
    except Exception as e:
        log.debug(f"тренд_4h {symbol}: {e}")
        return "neutral"

def получить_тренд_1h(symbol: str) -> str:
    try:
        raw = safe_fetch_ohlcv(symbol, TIMEFRAME_TREND, limit=60)
        if len(raw) < 55: return "neutral"
        df  = pd.DataFrame(raw, columns=["ts","o","h","l","c","v"])
        c   = df["c"]
        ema20 = _ema(c, 20).iloc[-1]
        ema50 = _ema(c, 50).iloc[-1]
        if ema20 > ema50: return "bull"
        elif ema20 < ema50: return "bear"
        return "neutral"
    except Exception:
        return "neutral"

# ============================================================
#         ПРОВЕРКА СОГЛАСОВАННОСТИ ИНДИКАТОРОВ
# ============================================================

def подсчитать_согласованность(details: dict, side: str) -> int:
    """
    Считает сколько из 6 ключевых индикаторов согласованы с направлением.
    Возвращает число от 0 до 6.
    """
    count = 0
    if side == "long":
        # 1. Supertrend вверх
        if details.get("supertrend") == "вверх": count += 1
        # 2. MACD бычий
        if details.get("macd") == "бычий": count += 1
        # 3. Range Filter вверх
        if details.get("range_filter") == "вверх": count += 1
        # 4. Hull вверх
        if details.get("hull") == "вверх": count += 1
        # 5. RSI в зоне роста (30-60)
        rsi = float(details.get("rsi", 50))
        if 30 <= rsi <= 60: count += 1
        # 6. Тренд 1h бычий
        if details.get("тренд_1h") == "бычий": count += 1
    else:  # short
        if details.get("supertrend") == "вниз": count += 1
        if details.get("macd") == "медвежий": count += 1
        if details.get("range_filter") == "вниз": count += 1
        if details.get("hull") == "вниз": count += 1
        rsi = float(details.get("rsi", 50))
        if 40 <= rsi <= 70: count += 1
        if details.get("тренд_1h") == "медвежий": count += 1
    return count

# ============================================================
#               РИСК-МЕНЕДЖМЕНТ
# ============================================================

def рассчитать_размер_позиции(score: int, баланс: float, sl_dist_pct: float) -> float:
    factor    = max(0, (score - MIN_SCORE)) / (100 - MIN_SCORE)
    risk_pct  = min(BASE_RISK_PCT + (MAX_RISK_PCT - BASE_RISK_PCT) * factor, MAX_RISK_PCT)
    max_loss  = баланс * risk_pct / 100
    margin    = min(max_loss / (sl_dist_pct / 100), баланс * 0.9)
    log.info(f"Скор={score} → риск={risk_pct:.1f}% SL={sl_dist_pct:.2f}% маржа={margin:.2f}U")
    return round(max(1.0, margin), 2)

# ============================================================
#           СКОРИНГ ЛОНГ
# ============================================================

def получить_скор(symbol: str) -> dict:
    details = {}
    score   = 0
    price   = 0.0
    sr      = {}
    try:
        raw_ta = safe_fetch_ohlcv(symbol, TIMEFRAME_TA, limit=300)
        time.sleep(API_CALL_DELAY)
        raw_1h = safe_fetch_ohlcv(symbol, TIMEFRAME_TREND, limit=300)
        if len(raw_ta) < 100 or len(raw_1h) < 100:
            return {"score": 0, "details": {}, "price": 0, "sr": {}}

        cols  = ["ts","o","h","l","c","v"]
        df_ta = pd.DataFrame(raw_ta, columns=cols).reset_index(drop=True)
        df_1h = pd.DataFrame(raw_1h, columns=cols).reset_index(drop=True)
        c_ta  = df_ta["c"]
        c_1h  = df_1h["c"]
        price = float(c_ta.iloc[-1])

        # RSI
        rsi_val = calc_rsi(c_ta).iloc[-1]
        details["rsi"] = round(rsi_val, 1)
        if 30 <= rsi_val <= 45:  score += 20
        elif 45 < rsi_val <= 55: score += 12
        elif rsi_val < 30:       score += 8   # перепродан — осторожно
        elif 55 < rsi_val <= 65: score += 5
        elif rsi_val > 70:       score -= 15  # перекуплен — штраф

        # RSI 1h
        rsi_1h = calc_rsi(c_1h).iloc[-1]
        details["rsi_1h"] = round(rsi_1h, 1)
        if rsi_1h < 50:   score += 10
        elif rsi_1h < 60: score += 5
        elif rsi_1h > 70: score -= 10

        # MACD
        ml, sl_macd, _ = calc_macd(c_ta)
        macd_bull  = ml.iloc[-1] > sl_macd.iloc[-1]
        macd_cross = macd_bull and ml.iloc[-2] <= sl_macd.iloc[-2]
        details["macd"] = "бычий" if macd_bull else "медвежий"
        if macd_cross: score += 18
        elif macd_bull: score += 8
        else:           score -= 5   # медвежий MACD — штраф при лонге

        # Range Filter
        _, _, _, rf_up, rf_down = calc_range_filter(df_ta)
        details["range_filter"] = "вверх" if rf_up.iloc[-1] else ("вниз" if rf_down.iloc[-1] else "бок")
        if rf_up.iloc[-1]:   score += 15
        elif rf_down.iloc[-1]: score -= 8

        # Supertrend
        st_up, st_down = calc_supertrend(df_ta)
        details["supertrend"] = "вверх" if st_up.iloc[-1] else "вниз"
        if st_up.iloc[-1]:   score += 15
        else:                 score -= 10  # штраф за медвежий Supertrend

        # Hull
        hu_up, hu_down = calc_hull(c_ta)
        details["hull"] = "вверх" if hu_up.iloc[-1] else "вниз"
        if hu_up.iloc[-1]: score += 10

        # Тренд 1h
        ema50_1h  = _ema(c_1h, 50).iloc[-1]
        ema200_1h = _ema(c_1h, 200).iloc[-1]
        details["тренд_1h"] = "бычий" if ema50_1h > ema200_1h else "медвежий"
        if ema50_1h > ema200_1h:  score += 12
        else:                      score -= 8

        # ADX
        adx, pdi, mdi = calc_adx(df_ta)
        adx_val = adx.iloc[-1]
        details["adx"] = round(adx_val, 1)
        if adx_val > 25 and pdi.iloc[-1] > mdi.iloc[-1]: score += 12
        elif adx_val > 20 and pdi.iloc[-1] > mdi.iloc[-1]: score += 5
        elif adx_val < 20: score -= 5  # слабый тренд

        # Stoch
        k_ser, _ = calc_stochastic(df_ta)
        k_val = k_ser.iloc[-1]
        details["stoch_k"] = round(k_val, 1)
        if k_val < 25:   score += 12
        elif k_val < 40: score += 6

        # Volume
        vol_avg   = df_ta["v"].rolling(VOLUME_AVG_PERIOD).mean().iloc[-1]
        vol_ratio = df_ta["v"].iloc[-1] / (vol_avg + 1e-10)
        details["объём_ratio"] = round(vol_ratio, 2)
        if vol_ratio > 1.5: score += 8
        elif vol_ratio > 1.0: score += 3
        elif vol_ratio < MIN_VOLUME_RATIO: score -= 5

        # S/R
        sr = calc_support_resistance(df_ta)
        details.update({
            "support": sr["support"], "resistance": sr["resistance"],
            "dist_sup": sr["dist_to_sup_pct"], "dist_res": sr["dist_to_res_pct"]
        })
        if sr["near_support"]:
            score += 15
            details["sr_signal"] = f"у поддержки ✅ ({sr['sup_cluster']} касаний)"
        elif sr["near_resistance"]:
            score -= 15  # усиленный штраф
            details["sr_signal"] = f"у сопротивления ⛔ ({sr['res_cluster']} касаний)"
        else:
            details["sr_signal"] = f"нейтр (sup={sr['dist_to_sup_pct']:.2f}% res={sr['dist_to_res_pct']:.2f}%)"

        # 3 зелёные свечи
        last3_bull = all(df_ta["c"].iloc[-i] > df_ta["o"].iloc[-i] for i in range(1, 4))
        if last3_bull:
            score += 8
            details["свечи_3green"] = True

        return {"score": max(0, min(100, int(score))), "details": details, "price": price, "sr": sr}
    except Exception as e:
        log.warning(f"Ошибка анализа лонг {symbol}: {e}")
        return {"score": 0, "details": {}, "price": 0, "sr": {}}

# ============================================================
#           СКОРИНГ ШОРТ
# ============================================================

def получить_скор_шорта(symbol: str) -> dict:
    details = {}
    score   = 0
    price   = 0.0
    sr      = {}
    try:
        raw_ta = safe_fetch_ohlcv(symbol, TIMEFRAME_TA, limit=300)
        raw_1h = safe_fetch_ohlcv(symbol, TIMEFRAME_TREND, limit=300)
        if len(raw_ta) < 100 or len(raw_1h) < 100:
            return {"score": 0, "details": {}, "price": 0, "sr": {}}

        cols  = ["ts","o","h","l","c","v"]
        df_ta = pd.DataFrame(raw_ta, columns=cols).reset_index(drop=True)
        df_1h = pd.DataFrame(raw_1h, columns=cols).reset_index(drop=True)
        c_ta  = df_ta["c"]
        c_1h  = df_1h["c"]
        price = float(c_ta.iloc[-1])

        # RSI — перекуплен = шорт-сигнал
        rsi_val = calc_rsi(c_ta).iloc[-1]
        details["rsi"] = round(rsi_val, 1)
        if rsi_val >= 70:   score += 25
        elif rsi_val >= 65: score += 18
        elif rsi_val >= 58: score += 10
        elif rsi_val < 40:  score -= 15  # перепродан — штраф для шорта

        # RSI 1h
        rsi_1h = calc_rsi(c_1h).iloc[-1]
        details["rsi_1h"] = round(rsi_1h, 1)
        if rsi_1h >= 60:   score += 12
        elif rsi_1h >= 55: score += 6
        elif rsi_1h < 45:  score -= 8

        # MACD — медвежий
        ml, sl_macd, _ = calc_macd(c_ta)
        macd_bear  = ml.iloc[-1] < sl_macd.iloc[-1]
        macd_cross = macd_bear and ml.iloc[-2] >= sl_macd.iloc[-2]
        details["macd"] = "медвежий" if macd_bear else "бычий"
        if macd_cross: score += 20
        elif macd_bear: score += 10
        else:           score -= 10  # бычий MACD — штраф при шорте

        # Supertrend вниз
        st_up, st_down = calc_supertrend(df_ta)
        details["supertrend"] = "вниз" if st_down.iloc[-1] else "вверх"
        if st_down.iloc[-1]:  score += 18
        else:                  score -= 15  # бычий Supertrend — большой штраф

        # Range Filter вниз
        _, _, _, rf_up, rf_down = calc_range_filter(df_ta)
        details["range_filter"] = "вниз" if rf_down.iloc[-1] else ("вверх" if rf_up.iloc[-1] else "бок")
        if rf_down.iloc[-1]:  score += 12
        elif rf_up.iloc[-1]:  score -= 10

        # Hull вниз
        hu_up, hu_down = calc_hull(c_ta)
        details["hull"] = "вниз" if hu_down.iloc[-1] else "вверх"
        if hu_down.iloc[-1]: score += 10

        # Тренд 1h медвежий
        ema50_1h  = _ema(c_1h, 50).iloc[-1]
        ema200_1h = _ema(c_1h, 200).iloc[-1]
        details["тренд_1h"] = "медвежий" if ema50_1h < ema200_1h else "бычий"
        if ema50_1h < ema200_1h:  score += 15
        else:                      score -= 12  # бычий тренд 1h — штраф

        # ADX
        adx, pdi, mdi = calc_adx(df_ta)
        adx_val = adx.iloc[-1]
        details["adx"] = round(adx_val, 1)
        if adx_val > 25 and mdi.iloc[-1] > pdi.iloc[-1]: score += 12
        elif adx_val > 20 and mdi.iloc[-1] > pdi.iloc[-1]: score += 5
        elif adx_val < 20: score -= 5

        # Stoch перекуплен
        k_ser, _ = calc_stochastic(df_ta)
        k_val = k_ser.iloc[-1]
        details["stoch_k"] = round(k_val, 1)
        if k_val >= 80:   score += 12
        elif k_val >= 65: score += 6
        elif k_val < 30:  score -= 8

        # Объём
        vol_avg   = df_ta["v"].rolling(VOLUME_AVG_PERIOD).mean().iloc[-1]
        vol_ratio = df_ta["v"].iloc[-1] / (vol_avg + 1e-10)
        details["объём_ratio"] = round(vol_ratio, 2)
        if vol_ratio > 1.5:  score += 8
        elif vol_ratio > 1.0: score += 3
        elif vol_ratio < MIN_VOLUME_RATIO: score -= 5

        # S/R — у сопротивления = шорт-сигнал
        sr = calc_support_resistance(df_ta)
        details.update({
            "support": sr["support"], "resistance": sr["resistance"],
            "dist_sup": sr["dist_to_sup_pct"], "dist_res": sr["dist_to_res_pct"]
        })
        if sr["near_resistance"]:
            score += 20
            details["sr_signal"] = f"у сопротивления ✅ ({sr['res_cluster']} касаний)"
        elif sr["near_support"]:
            score -= 15
            details["sr_signal"] = f"у поддержки ⛔ ({sr['sup_cluster']} касаний)"

        # 3 красные свечи
        last3_bear = all(df_ta["c"].iloc[-i] < df_ta["o"].iloc[-i] for i in range(1, 4))
        if last3_bear:
            score += 10
            details["свечи_3red"] = True

        return {"score": max(0, min(100, int(score))), "details": details, "price": price, "sr": sr}
    except Exception as e:
        log.warning(f"Ошибка шорт-анализа {symbol}: {e}")
        return {"score": 0, "details": {}, "price": 0, "sr": {}}

# ============================================================
#         ФИНАЛЬНЫЙ СКОРИНГ С BYBIT RATIO
# ============================================================

def применить_ratio_корректировку(score: int, side: str, long_ratio: float) -> int:
    """
    Ключевое исправление: штрафы/бонусы за направление рынка.
    При бычьем рынке шорты получают огромный штраф.
    """
    if side == "long":
        if long_ratio >= RATIO_STRONG_LONG:   return min(100, score + 15)
        elif long_ratio >= 0.55:               return min(100, score + 8)
        elif long_ratio < 0.40:               return max(0, score - 20)
        elif long_ratio < LONG_MIN_LONG_RATIO: return 0   # Блокируем лонг
    else:  # short
        if long_ratio > RATIO_STRONG_LONG:     return 0   # Полная блокировка шорта!
        elif long_ratio > SHORT_MAX_LONG_RATIO: return 0  # Блокируем шорт
        elif long_ratio <= RATIO_STRONG_SHORT:  return min(100, score + 15)
        elif long_ratio < 0.45:                 return min(100, score + 8)
    return score

# ============================================================
#         УПРАВЛЕНИЕ ПЛЕЧОМ
# ============================================================

def установить_плечо(symbol: str, leverage: int) -> bool:
    try:
        exchange.set_leverage(leverage, symbol,
                              params={"buyLeverage": leverage, "sellLeverage": leverage})
        log.info(f"Плечо {leverage}x установлено для {symbol}")
        return True
    except Exception as e1:
        if "leverage not modified" in str(e1).lower() or "110043" in str(e1):
            log.info(f"Плечо {leverage}x уже установлено для {symbol} — OK")
            return True
        log.warning(f"Метод 1 плеча не сработал: {e1}")
    try:
        coin_sym = symbol.replace("/", "").replace(":USDT", "")
        exchange.private_post_v5_position_set_leverage({
            "category": "linear", "symbol": coin_sym,
            "buyLeverage": str(leverage), "sellLeverage": str(leverage),
        })
        log.info(f"Плечо {leverage}x уже установлено (v5) для {symbol} — OK")
        return True
    except Exception as e2:
        if "leverage not modified" in str(e2).lower() or "110043" in str(e2):
            log.info(f"Плечо {leverage}x уже установлено (v5) для {symbol} — OK")
            return True
        log.warning(f"Метод 2 плеча не сработал: {e2}")
    log.warning(f"Плечо не установлено явно для {symbol}, продолжаем")
    return True

def обновить_sl_на_бирже(symbol: str, new_sl: float, side: str = "long") -> bool:
    try:
        sl_str   = exchange.price_to_precision(symbol, new_sl)
        coin_sym = symbol.replace("/", "").replace(":USDT", "")
        exchange.private_post_v5_position_trading_stop({
            "category": "linear", "symbol": coin_sym,
            "stopLoss": sl_str, "slTriggerBy": "MarkPrice", "positionIdx": "0",
        })
        log.info(f"SL обновлён → {sl_str}")
        return True
    except Exception as e:
        log.warning(f"Не удалось обновить SL: {e}")
        return False

# ============================================================
#         ОТКРЫТИЕ ПОЗИЦИИ
# ============================================================

def открыть_позицию(symbol: str, margin_usdt: float,
                    tp_price: float, sl_price: float,
                    side: str = "long") -> Tuple[Optional[float], Optional[float]]:
    try:
        установить_плечо(symbol, LEVERAGE)
        ticker = safe_fetch_ticker(symbol)
        if not ticker:
            log.error(f"Не удалось получить тикер {symbol}")
            return None, None

        price        = float(ticker["last"])
        pos_size_usdt = margin_usdt * LEVERAGE
        qty_raw      = pos_size_usdt / price
        qty          = float(exchange.amount_to_precision(symbol, qty_raw))

        if qty <= 0:
            log.error(f"Нулевое qty для {symbol}")
            return None, None

        # Корректировка SL/TP
        if side == "long":
            sl_price = min(sl_price, price * (1 - MIN_SL_PERCENT / 100))
            tp_price = max(tp_price, price * (1 + MIN_SL_PERCENT * MIN_RR / 100))
        else:
            sl_price = max(sl_price, price * (1 + MIN_SL_PERCENT / 100))
            tp_price = min(tp_price, price * (1 - MIN_SL_PERCENT * MIN_RR / 100))

        tp_str   = exchange.price_to_precision(symbol, tp_price)
        sl_str   = exchange.price_to_precision(symbol, sl_price)
        buy_sell = "buy" if side == "long" else "sell"

        log.info(f"Открываем {side} {symbol}: qty={qty}, маржа≈{margin_usdt:.2f}U, "
                 f"плечо={LEVERAGE}x, TP={tp_str}, SL={sl_str}")

        order = exchange.create_market_order(symbol, buy_sell, qty, params={
            "takeProfit": float(tp_str),
            "stopLoss":   float(sl_str)
        })

        entry_price = None
        for key in ("average", "price"):
            v = order.get(key)
            if v:
                try: entry_price = float(v); break
                except: pass
        if not entry_price or entry_price <= 0:
            time.sleep(2)
            try:
                positions = safe_fetch_positions([symbol])
                for pos in positions:
                    if float(pos.get("contracts", 0) or 0) > 0:
                        ep = pos.get("entryPrice") or pos.get("avgCost")
                        if ep: entry_price = float(ep); break
            except: pass
        if not entry_price or entry_price <= 0:
            entry_price = price
            log.warning(f"Entry price из тикера: {entry_price}")

        log.info(f"{side.upper()} открыт: {qty} {symbol} @ ~{entry_price:.8f}")
        return entry_price, qty
    except Exception as e:
        log.error(f"Ошибка открытия {side} {symbol}: {e}")
        return None, None

def закрыть_позицию_с_подтверждением(symbol: str, qty: float, side: str) -> bool:
    close_side = "sell" if side == "long" else "buy"
    for attempt in range(3):
        try:
            exchange.create_market_order(symbol, close_side, qty,
                                         params={"reduceOnly": True})
            time.sleep(3)
            positions = safe_fetch_positions([symbol])
            active = [p for p in positions
                      if float(p.get("contracts", 0) or 0) > 0 and p.get("side") == side]
            if not active:
                log.info(f"Позиция {symbol} закрыта")
                return True
            log.warning(f"Позиция {symbol} не закрылась, повтор...")
            time.sleep(2)
        except Exception as e:
            log.warning(f"Попытка {attempt+1} закрыть {symbol}: {e}")
            time.sleep(2)
    log.error(f"Не удалось закрыть {symbol} после 3 попыток")
    return False

# ============================================================
#        EXIT SIGNAL (2 из 3 развернулись — выходим)
# ============================================================

def проверить_exit_signal(symbol: str, side: str,
                           current_price: float, entry_price: float) -> bool:
    """
    Выходим если минимум EXIT_MIN_REVERSED_INDICATORS индикаторов
    развернулись против позиции. Требуем НЕСКОЛЬКО подтверждений,
    чтобы не выходить от ложных сигналов.
    """
    try:
        raw = safe_fetch_ohlcv(symbol, TIMEFRAME_TA, limit=60)
        if len(raw) < 30:
            return False

        df      = pd.DataFrame(raw, columns=["ts","o","h","l","c","v"])
        reversed_count = 0

        # 1. Supertrend
        st_up, st_down = calc_supertrend(df)
        if side == "long"  and st_down.iloc[-1]: reversed_count += 1
        if side == "short" and st_up.iloc[-1]:   reversed_count += 1

        # 2. Range Filter
        _, _, _, rf_up, rf_down = calc_range_filter(df)
        if side == "long"  and rf_down.iloc[-1]: reversed_count += 1
        if side == "short" and rf_up.iloc[-1]:   reversed_count += 1

        # 3. MACD
        ml, sl_m, _ = calc_macd(df["c"])
        if side == "long"  and ml.iloc[-1] < sl_m.iloc[-1]: reversed_count += 1
        if side == "short" and ml.iloc[-1] > sl_m.iloc[-1]: reversed_count += 1

        # 4. RSI экстремум
        rsi_val = calc_rsi(df["c"]).iloc[-1]
        if side == "long"  and rsi_val > 75:  reversed_count += 1
        if side == "short" and rsi_val < 25:  reversed_count += 1

        if reversed_count >= EXIT_MIN_REVERSED_INDICATORS:
            log.info(f"🚨 Exit Signal: {reversed_count} индикаторов развернулись → ВЫХОД")
            return True

        return False
    except Exception as e:
        log.debug(f"Ошибка exit signal: {e}")
        return False

# ============================================================
#            МОНИТОРИНГ ПОЗИЦИИ
# ============================================================

def мониторить_позицию(symbol: str, entry_price: float, qty: float,
                        открыта_в: float, sl_цена: float, tp_цена: float,
                        side: str = "long") -> Tuple[str, float]:
    deadline   = открыта_в + TRADE_MAX_LIFETIME
    coin       = symbol.split("/")[0]

    # ATR для трейлинга
    trailing_offset = MIN_TRAILING_OFFSET / 100
    try:
        raw = safe_fetch_ohlcv(symbol, TIMEFRAME_TA, limit=50)
        if len(raw) >= 30:
            df_atr      = pd.DataFrame(raw, columns=["ts","o","h","l","c","v"])
            atr_val     = calc_atr(df_atr, TRAILING_ATR_PERIOD).iloc[-1]
            atr_pct     = (atr_val / entry_price) * 100
            trailing_offset = max(MIN_TRAILING_OFFSET, atr_pct * TRAILING_OFFSET_MULT) / 100
    except: pass

    # Триггер трейлинга
    if side == "long":
        rr_trigger = entry_price + (tp_цена - entry_price) * RR_EXIT_TRIGGER
    else:
        rr_trigger = entry_price - (entry_price - tp_цена) * RR_EXIT_TRIGGER

    log.info(f"rrExit триггер={rr_trigger:.8f} (RR_EXIT={RR_EXIT_TRIGGER})")
    log.info(f"Мониторинг {coin} {side} вход={entry_price:.8f} SL={sl_цена:.8f} TP={tp_цена:.8f}")

    фаза             = 1
    текущий_sl       = sl_цена
    пиковая_цена     = entry_price
    trailing_активен = False
    partial_done     = False
    accumulated_pnl  = 0.0

    while True:
        сейчас = time.time()
        if сейчас >= deadline:
            log.warning("⏰ Дедлайн — принудительное закрытие")
            закрыть_позицию_с_подтверждением(symbol, qty, side)
            return "таймаут", accumulated_pnl

        time.sleep(EXIT_SIGNAL_CHECK_INTERVAL)

        try:
            positions = safe_fetch_positions([symbol])
            active = [p for p in positions
                      if float(p.get("contracts", 0) or 0) > 0 and p.get("side") == side]

            if not active:
                ticker = safe_fetch_ticker(symbol)
                cur    = float(ticker["last"]) if ticker else entry_price
                hit_tp = (cur >= tp_цена) if side == "long" else (cur <= tp_цена)
                return ("tp" if hit_tp else "sl"), accumulated_pnl

            pos      = active[0]
            ticker   = safe_fetch_ticker(symbol)
            cur      = float(ticker["last"]) if ticker else entry_price
            qty_act  = abs(float(pos.get("contracts", 0) or 0))
            pnl_real = float(pos.get("unrealizedPnl", 0) or 0)
            pnl_pct  = ((cur - entry_price) / entry_price * 100) if side == "long" \
                       else ((entry_price - cur) / entry_price * 100)
            до_дед   = int(deadline - сейчас)

            # --- EXIT SIGNAL ---
            if SIGNAL_EXIT_ENABLED and фаза >= 1:
                if проверить_exit_signal(symbol, side, cur, entry_price):
                    log.info(f"🚨 Закрываем {symbol} по exit signal")
                    закрыть_позицию_с_подтверждением(symbol, qty_act, side)
                    return ("tp" if pnl_pct > 0 else "sl"), accumulated_pnl + pnl_real

            # --- ЧАСТИЧНЫЙ БЕЗУБЫТОК ---
            if PARTIAL_BE_ENABLED and not partial_done and pnl_pct >= PARTIAL_BE_PROFIT:
                close_qty  = qty_act * (PARTIAL_BE_CLOSE_PCT / 100)
                if close_qty > 0:
                    close_side = "sell" if side == "long" else "buy"
                    try:
                        exchange.create_market_order(symbol, close_side, close_qty,
                                                     params={"reduceOnly": True})
                        partial_pnl      = (cur - entry_price) * close_qty if side == "long" \
                                           else (entry_price - cur) * close_qty
                        accumulated_pnl += partial_pnl
                        log.info(f"Частичный безубыток: закрыто {close_qty:.4f} "
                                 f"({PARTIAL_BE_CLOSE_PCT:.0f}%) @ ~{cur:.8f} PnL≈{partial_pnl:+.4f}U")

                        fee_buffer = entry_price * BYBIT_FEE * 4
                        if side == "long":
                            new_sl = entry_price + fee_buffer
                            if new_sl >= cur: new_sl = cur * 0.999
                        else:
                            new_sl = entry_price - fee_buffer
                            if new_sl <= cur: new_sl = cur * 1.001

                        if обновить_sl_на_бирже(symbol, new_sl, side):
                            текущий_sl = new_sl
                            log.info(f"SL обновлён → {new_sl:.8f}")
                        partial_done = True
                        фаза = 2
                    except Exception as e:
                        log.warning(f"Не удалось частично закрыть: {e}")

            # --- БЕЗУБЫТОК (без частичного) ---
            if not partial_done and фаза == 1 and pnl_pct >= 0.15:
                fee_buffer = entry_price * BYBIT_FEE * 4
                if side == "long":
                    new_sl_be = entry_price + fee_buffer
                    if new_sl_be >= cur: new_sl_be = cur * 0.999
                else:
                    new_sl_be = entry_price - fee_buffer
                    if new_sl_be <= cur: new_sl_be = cur * 1.001
                if обновить_sl_на_бирже(symbol, new_sl_be, side):
                    фаза       = 2
                    текущий_sl = new_sl_be
                    log.info(f"БЕЗУБЫТОК! SL → {new_sl_be:.8f}")

            # --- АКТИВАЦИЯ ТРЕЙЛИНГА ---
            if not trailing_активен and фаза >= 2:
                if side == "long":
                    trailing_активен = cur >= rr_trigger
                else:
                    trailing_активен = cur <= rr_trigger
                if trailing_активен:
                    log.info(f"🚀 ТРЕЙЛИНГ АКТИВИРОВАН @ {cur:.8f}")

            # --- ТРЕЙЛИНГ ---
            if trailing_активен and фаза >= 2 and pnl_pct >= MIN_PROFIT_FOR_TRAIL:
                if side == "long":
                    if cur > пиковая_цена: пиковая_цена = cur
                    new_sl_tr = пиковая_цена * (1 - trailing_offset)
                    if new_sl_tr > текущий_sl and new_sl_tr < cur:
                        if обновить_sl_на_бирже(symbol, new_sl_tr, side):
                            текущий_sl = new_sl_tr
                            log.info(f"ТРЕЙЛИНГ: пик={пиковая_цена:.8f} SL={new_sl_tr:.8f}")
                else:
                    if cur < пиковая_цена: пиковая_цена = cur
                    new_sl_tr = пиковая_цена * (1 + trailing_offset)
                    if new_sl_tr < текущий_sl and new_sl_tr > cur:
                        if обновить_sl_на_бирже(symbol, new_sl_tr, side):
                            текущий_sl = new_sl_tr
                            log.info(f"ТРЕЙЛИНГ: пик={пиковая_цена:.8f} SL={new_sl_tr:.8f}")

            log.info(f"[{coin}] {cur:.8f} P&L={pnl_pct:+.2f}% ({pnl_real:+.4f}U) "
                     f"SL={текущий_sl:.8f} фаза={фаза} дед={до_дед}с "
                     f"{'[частично]' if partial_done else ''}")

        except Exception as e:
            log.warning(f"Ошибка мониторинга: {e}")

    return "sl", accumulated_pnl

# ============================================================
#           ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def загрузить_историю() -> List[dict]:
    if not os.path.exists(TRADES_FILE): return []
    try:
        with open(TRADES_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except: return []

def сохранить_сделку(запись: dict):
    история = загрузить_историю()
    история.append(запись)
    try:
        with open(TRADES_FILE, "w", encoding="utf-8") as f:
            json.dump(история, f, ensure_ascii=False, indent=2, default=str)
        log.info(f"Сделка #{запись['id']} сохранена в {TRADES_FILE}")
    except Exception as e:
        log.warning(f"Не удалось сохранить сделку: {e}")

def сохранить_состояние():
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        log.warning(f"Не удалось сохранить состояние: {e}")

def загрузить_состояние():
    global stats
    if not os.path.exists(STATE_FILE): return False
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        for key in stats:
            if key in saved: stats[key] = saved[key]
        log.info(f"Состояние восстановлено из {STATE_FILE}")
        return True
    except Exception as e:
        log.warning(f"Не удалось загрузить состояние: {e}")
        return False

def баланс_usdt() -> float:
    try:
        b = exchange.fetch_balance({"type": "linear"})
        return float(b.get("USDT", {}).get("free", 0.0))
    except Exception as e:
        log.warning(f"Ошибка баланса: {e}")
        return 0.0

def полный_баланс_usdt() -> float:
    try:
        b     = exchange.fetch_balance({"type": "linear"})
        total = float(b.get("USDT", {}).get("total", 0.0))
        if total > 0: return total
        eq    = float(b.get("USDT", {}).get("equity", 0.0))
        if eq > 0:    return eq
        return баланс_usdt()
    except: return баланс_usdt()

def получить_позиции() -> List[dict]:
    return [p for p in safe_fetch_positions()
            if float(p.get("contracts", 0) or 0) > 0]

def обновить_начало_дня(баланс: float):
    сегодня = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if stats["дата_дня"] != сегодня:
        stats["дата_дня"]           = сегодня
        stats["баланс_начало_дня"]  = баланс
        log.info(f"Новый день. Баланс: {баланс:.2f} USDT")
        сохранить_состояние()

def превышен_дневной_лимит() -> bool:
    нач = stats.get("баланс_начало_дня", 0.0)
    if нач <= 0: return False
    текущий    = полный_баланс_usdt()
    потеря_pct = (нач - текущий) / нач * 100
    if потеря_pct >= DAILY_LOSS_LIMIT_PCT:
        log.warning(f"Дневной лимит убытков: -{потеря_pct:.1f}% (лимит {DAILY_LOSS_LIMIT_PCT}%)")
        return True
    return False

# ============================================================
#              СТАТИСТИКА ИНДИКАТОРОВ
# ============================================================

def загрузить_статистику_инд() -> dict:
    if not os.path.exists(INDICATOR_STATS_FILE): return {}
    try:
        with open(INDICATOR_STATS_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except: return {}

def сохранить_статистику_инд(data: dict):
    try:
        with open(INDICATOR_STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    except: pass

def обновить_статистику_инд(запись: dict):
    data     = загрузить_статистику_инд()
    details  = запись.get("details", {})
    результат = запись.get("результат", "")
    is_win   = (результат == "tp")

    индикаторы = {
        "rsi":          lambda v: 30 <= float(v) <= 55,
        "rsi_1h":       lambda v: float(v) < 55,
        "macd":         lambda v: v == "бычий",
        "range_filter": lambda v: v == "вверх",
        "supertrend":   lambda v: v == "вверх",
        "hull":         lambda v: v == "вверх",
        "тренд_1h":     lambda v: v == "бычий",
        "adx":          lambda v: float(v) > 25,
        "stoch_k":      lambda v: float(v) < 30,
        "объём_ratio":  lambda v: float(v) > 1.3,
        "sr_signal":    lambda v: "поддержки" in str(v) or "✅" in str(v),
    }
    for инд, условие in индикаторы.items():
        v = details.get(инд)
        if v is None: continue
        try:    is_bull = условие(v)
        except: continue
        if инд not in data:
            data[инд] = {"bullish": {"total": 0, "wins": 0}, "bearish": {"total": 0, "wins": 0}}
        key = "bullish" if is_bull else "bearish"
        data[инд][key]["total"] += 1
        if is_win: data[инд][key]["wins"] += 1
    сохранить_статистику_инд(data)

# ============================================================
#              ДЕТАЛЬНЫЙ ОТЧЁТ СДЕЛКИ
# ============================================================

def детальный_отчёт(запись: dict):
    r    = запись.get("результат", "?")
    sym  = запись.get("symbol", "?").split(":")[0]
    pnl  = запись.get("pnl_usdt", 0)
    dur  = запись.get("duration_min", 0)
    side = запись.get("side", "?")
    score = запись.get("score", "?")
    entry = запись.get("entry_price", 0)
    sl    = запись.get("sl_price", 0)
    tp    = запись.get("tp_price", 0)
    sl_p  = запись.get("sl_dist_pct", 0)
    rr    = запись.get("rr_ratio", 0)
    margin = запись.get("margin_usdt", 0)
    det    = запись.get("details", {})
    aligned = запись.get("aligned_indicators", 0)

    знак = ("✅ ТЕЙКПРОФИТ" if r == "tp"
            else "❌ СТОПЛОСС" if r == "sl" else "⏰ ТАЙМАУТ")

    log.info("")
    log.info("━" * 70)
    log.info(f"📋 ДЕТАЛЬНЫЙ ОТЧЁТ СДЕЛКИ #{запись.get('id', '?')}")
    log.info("━" * 70)
    log.info(f"  Символ:        {sym} ({side.upper()})")
    log.info(f"  Результат:     {знак}")
    log.info(f"  P&L:           {'+' if pnl >= 0 else ''}{pnl:.4f} USDT")
    log.info(f"  Скор входа:    {score}/100")
    log.info(f"  Согласованных: {aligned}/6 индикаторов")
    log.info(f"  Длительность:  {dur:.1f} мин")
    log.info(f"  Маржа:         {margin:.2f} USDT × {LEVERAGE}x = {margin*LEVERAGE:.2f} USDT")
    log.info(f"  Цены:          Вход={entry:.8f}  SL={sl:.8f}  TP={tp:.8f}")
    log.info(f"  SL расст.:     {sl_p:.2f}%  |  RR: {rr:.1f}:1")
    log.info(f"  Время входа:   {запись.get('время_входа', '?')}")
    log.info(f"  Время выхода:  {запись.get('время_выхода', '?')}")
    log.info("  ─── Индикаторы ───")
    log.info(f"  RSI 5m:        {det.get('rsi', '?')}  |  RSI 1h: {det.get('rsi_1h', '?')}")
    log.info(f"  MACD:          {det.get('macd', '?')}")
    log.info(f"  Supertrend:    {det.get('supertrend', '?')}")
    log.info(f"  Range Filter:  {det.get('range_filter', '?')}")
    log.info(f"  Hull:          {det.get('hull', '?')}")
    log.info(f"  ADX:           {det.get('adx', '?')}")
    log.info(f"  Stoch K:       {det.get('stoch_k', '?')}")
    log.info(f"  Тренд 1h:      {det.get('тренд_1h', '?')}")
    log.info(f"  Тренд 4h:      {запись.get('тренд_4h', '?')}")
    log.info(f"  Bybit ratio:   {запись.get('long_ratio', '?')}")
    log.info(f"  Объём ratio:   {det.get('объём_ratio', '?')}")
    log.info(f"  S/R сигнал:    {det.get('sr_signal', '?')}")
    log.info("━" * 70)
    log.info("")

# ============================================================
#              ПЕЧАТАТЬ ОТЧЁТ
# ============================================================

def печатать_отчёт():
    баланс   = полный_баланс_usdt()
    старт    = stats["депозит_старт"]
    дельта   = баланс - старт
    чистый   = stats["прибыль_usdt"] - stats["убыток_usdt"]
    пct      = (дельта / старт * 100) if старт > 0 else 0
    всего    = stats["сделок_всего"]
    tp_      = stats["тейкпрофит"]
    wr       = (tp_ / всего * 100) if всего > 0 else 0.0

    log.info("")
    log.info("=" * 70)
    log.info("📊 ОТЧЁТ ГИБРИДНОГО БОТА v12.0")
    log.info(f"  Баланс: {баланс:.2f} USDT ({дельта:+.2f} USDT / {пct:+.2f}%)")
    log.info(f"  Сделок: {всего} | TP={tp_} | SL={stats['стоплосс']} | Таймаут={stats['таймаут']}")
    log.info(f"  WinRate: {wr:.1f}%")
    log.info(f"  Прибыль: {stats['прибыль_usdt']:.4f} USDT")
    log.info(f"  Убыток:  {stats['убыток_usdt']:.4f} USDT")
    log.info(f"  Чистый P&L: {чистый:+.4f} USDT")
    log.info("=" * 70)
    log.info("")

    stats["последний_отчёт"] = time.time()
    сохранить_состояние()
    отчёт_по_индикаторам()

    история = загрузить_историю()
    if len(история) >= 5:
        pnls = [t["pnl_usdt"] for t in история]
        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p < 0)
        total  = len(pnls)
        cumul  = np.cumsum(pnls)
        r_max  = np.maximum.accumulate(cumul)
        max_dd = abs(min(cumul - r_max))
        sharpe = (np.mean(pnls) / np.std(pnls) * np.sqrt(252)
                  if len(pnls) > 1 and np.std(pnls) != 0 else 0)
        avg_win  = round(np.mean([p for p in pnls if p > 0]), 4) if wins else 0
        avg_loss = round(abs(np.mean([p for p in pnls if p < 0])), 4) if losses else 0
        pf = round(sum(p for p in pnls if p > 0) /
                   abs(sum(p for p in pnls if p < 0)), 2) if losses > 0 else 0

        log.info("📉 МЕТРИКИ:")
        log.info(f"  Sharpe: {sharpe:.2f}")
        log.info(f"  Max Drawdown: {max_dd:.2f} USDT")
        log.info(f"  Profit Factor: {pf:.2f}")
        log.info(f"  Avg Win: {avg_win:.4f} USDT | Avg Loss: {avg_loss:.4f} USDT")

def отчёт_по_индикаторам():
    data = загрузить_статистику_инд()
    if not data: return
    log.info("")
    log.info("=" * 75)
    log.info("📈 ЭФФЕКТИВНОСТЬ ИНДИКАТОРОВ")
    log.info(f"{'Индикатор':<18} {'🟢Бычий WR%':>11}  {'n':>4}  {'🔴Медвежий WR%':>14}  {'n':>4}  {'Разница':>8}")
    log.info(" " + "─" * 75)
    for инд, d in data.items():
        bt = d["bullish"]["total"]; bw = d["bullish"]["wins"]
        et = d["bearish"]["total"]; ew = d["bearish"]["wins"]
        b_wr = (bw / bt * 100) if bt > 0 else 0
        e_wr = (ew / et * 100) if et > 0 else 0
        diff = b_wr - e_wr
        знак = "▲" if diff > 5 else ("▼" if diff < -5 else "≈")
        log.info(f"{инд:<18}  {b_wr:>9.1f}%  {bt:>4}  {e_wr:>12.1f}%  {et:>4}  {знак}{diff:>+7.1f}%")
    log.info("=" * 75)
    log.info("")

# ============================================================
#             ПРЕДСТАРТОВАЯ ПРОВЕРКА
# ============================================================

def предстартовая_проверка() -> bool:
    log.info("=" * 70)
    log.info("🔍 ПРЕДСТАРТОВАЯ ПРОВЕРКА (v12.0)")
    log.info("=" * 70)
    все_ок = True

    # API ключи
    if not os.getenv("BYBIT_API_KEY") or not os.getenv("BYBIT_API_SECRET"):
        log.error("❌ API ключи не заданы")
        все_ок = False

    # Подключение
    try:
        b = exchange.fetch_balance({"type": "linear"})
        free  = float(b.get("USDT", {}).get("free", 0))
        total = float(b.get("USDT", {}).get("total", 0))
        log.info(f"✅ Подключение OK | Свободно: {free:.2f} USDT | Всего: {total:.2f} USDT")
        if free < MIN_BALANCE:
            log.warning(f"⚠️ Низкий свободный баланс {free:.2f} USDT")
    except Exception as e:
        log.error(f"❌ Ошибка подключения: {e}")
        все_ок = False

    # Конфигурация
    rr = ATR_TP_MULT / ATR_SL_MULT
    log.info(f"✅ ATR SL×{ATR_SL_MULT} / TP×{ATR_TP_MULT} → RR≈{rr:.1f}:1")
    log.info(f"   MIN_SCORE={MIN_SCORE} | MIN_INDICATORS={MIN_INDICATORS_ALIGNED}/6")
    log.info(f"   Запрет шортов при long_ratio > {SHORT_MAX_LONG_RATIO:.0%}")
    log.info(f"   Запрет лонгов при long_ratio < {LONG_MIN_LONG_RATIO:.0%}")

    # Открытые позиции
    try:
        positions = safe_fetch_positions()
        active    = [p for p in positions if float(p.get("contracts", 0) or 0) > 0]
        if active:
            log.warning(f"⚠️ Открытые позиции: {[p['symbol'] for p in active]}")
        else:
            log.info("✅ Открытых позиций нет")
    except Exception as e:
        log.warning(f"Не удалось проверить позиции: {e}")

    log.info("=" * 70)
    return все_ок

# ============================================================
#                     ГЛАВНЫЙ ЦИКЛ
# ============================================================

def main():
    global stats

    if not предстартовая_проверка():
        log.error("🛑 Бот остановлен из-за ошибок проверки.")
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

    log.info("=" * 70)
    log.info("🤖 ГИБРИДНЫЙ БОТ v12.0 (АНТИПОТЕРЬ)")
    log.info(f"  Плечо: {LEVERAGE}x | ATR SL×{ATR_SL_MULT} TP×{ATR_TP_MULT} | MIN_RR={MIN_RR}")
    log.info(f"  Баланс: {баланс_сейчас:.4f} USDT")
    log.info(f"  MIN_SCORE: {MIN_SCORE} | MIN_INDICATORS: {MIN_INDICATORS_ALIGNED}/6")
    log.info(f"  SHORT_MAX_LONG_RATIO: {SHORT_MAX_LONG_RATIO:.0%}")
    log.info("=" * 70)

    заблокированные: Dict[str, float] = {}
    fail_attempts:   Dict[str, int]   = {}

    while True:
        try:
            # Периодический отчёт
            if time.time() - stats["последний_отчёт"] >= REPORT_INTERVAL:
                печатать_отчёт()

            баланс   = полный_баланс_usdt()
            свободный = баланс_usdt()
            обновить_начало_дня(баланс)

            # Проверки
            if свободный < MIN_BALANCE:
                активные = получить_позиции()
                if активные:
                    log.info(f"⏳ Открытые позиции: {[p['symbol'] for p in активные]} — ждём")
                    time.sleep(30)
                    continue
                log.warning(f"🛑 Баланс {свободный:.2f} < {MIN_BALANCE}. Пауза 5 мин.")
                time.sleep(300)
                continue

            if stats["депозит_старт"] > 0:
                просадка = (stats["депозит_старт"] - баланс) / stats["депозит_старт"] * 100
                if просадка > MAX_DRAWDOWN_PCT:
                    log.warning(f"⛔ Просадка {просадка:.1f}% > {MAX_DRAWDOWN_PCT}%. Пауза 1ч.")
                    time.sleep(3600)
                    continue

            if превышен_дневной_лимит():
                log.warning(f"⛔ Дневной лимит. Пауза {DAILY_LOSS_PAUSE_SEC//60} мин.")
                time.sleep(DAILY_LOSS_PAUSE_SEC)
                continue

            if stats.get("sl_streak", 0) >= SL_STREAK_LIMIT:
                log.warning(f"🧊 {SL_STREAK_LIMIT} SL подряд — cooldown {SL_STREAK_PAUSE//60} мин.")
                stats["sl_streak"] = 0
                сохранить_состояние()
                time.sleep(SL_STREAK_PAUSE + SL_STREAK_EXTRA_PAUSE)
                continue

            # Ждём закрытия позиций
            активные = получить_позиции()
            if активные:
                log.info(f"⏳ Открытые позиции: {[p['symbol'] for p in активные]} — ждём")
                time.sleep(30)
                continue

            # ---- ГЛОБАЛЬНЫЙ ФИЛЬТР ПО BYBIT RATIO ----
            # Получаем ratio один раз для BTC (рыночный индикатор)
            market_ratio = получить_bybit_ratio("BTC/USDT:USDT")
            long_ratio   = market_ratio.get("long_ratio", 0.5)
            log.info(f"🌐 Рынок: long={long_ratio:.1%} short={market_ratio.get('short_ratio',0.5):.1%} "
                     f"→ {market_ratio.get('signal','?')}")

            # Определяем разрешённые направления
            allow_long  = long_ratio >= LONG_MIN_LONG_RATIO
            allow_short = long_ratio <= SHORT_MAX_LONG_RATIO

            if not allow_long and not allow_short:
                log.info(f"⏳ Рынок нейтрален (ratio={long_ratio:.1%}) — нет чёткого направления")
                time.sleep(SCAN_INTERVAL)
                continue

            # ---- СКАНИРОВАНИЕ ----
            log.info(f"── Сканирование {len(SYMBOLS)} пар "
                     f"(баланс={свободный:.2f}U | порог={MIN_SCORE}) ──")

            кандидаты: List[Tuple[str, dict, str]] = []  # (symbol, data, side)

            for sym in SYMBOLS:
                # Пропускаем заблокированные
                if sym in заблокированные:
                    if time.time() < заблокированные[sym]:
                        continue
                    else:
                        del заблокированные[sym]
                        fail_attempts.pop(sym, None)

                time.sleep(API_CALL_DELAY)

                # Тренд 4h — обязательный фильтр
                тренд4h = получить_тренд_4h(sym)
                time.sleep(API_CALL_DELAY)

                # --- ЛОНГ ---
                if allow_long and тренд4h != "bear":
                    res = получить_скор(sym)
                    if res["score"] >= MIN_SCORE:
                        det  = res.get("details", {})
                        sr   = res.get("sr", {})

                        # Проверка согласованности
                        aligned = подсчитать_согласованность(det, "long")
                        if aligned < MIN_INDICATORS_ALIGNED:
                            log.info(f"⚠️ {sym.split(':')[0]}: только {aligned}/6 индикаторов согласованы — пропуск")
                            continue

                        # Получаем ratio для конкретной монеты
                        r_data    = получить_bybit_ratio(sym)
                        l_ratio   = r_data.get("long_ratio", long_ratio)
                        log.info(f"Bybit ratio: long={l_ratio:.1%} short={r_data.get('short_ratio',0.5):.1%} "
                                 f"сигнал={r_data.get('signal','?')}")

                        fin_score = применить_ratio_корректировку(res["score"], "long", l_ratio)
                        if fin_score < MIN_SCORE:
                            log.info(f"⚠️ {sym.split(':')[0]}: после корректировки скор={fin_score} — пропуск")
                            continue

                        # S/R фильтр — не заходим у сопротивления для лонга
                        if sr.get("near_resistance") and sr.get("dist_to_res_pct", 99) < SR_BLOCK_DIST_PCT:
                            log.info(f"⛔ {sym.split(':')[0]}: сопротивление {sr.get('dist_to_res_pct',0):.2f}% — пропуск")
                            continue

                        # RSI перекуплен
                        rsi_v = float(det.get("rsi", 50))
                        if rsi_v > 70:
                            log.info(f"⚠️ {sym.split(':')[0]}: RSI={rsi_v:.1f} перекуплен — пропуск")
                            continue

                        # Объём
                        if float(det.get("объём_ratio", 0)) < MIN_VOLUME_RATIO:
                            log.info(f"⚠️ {sym.split(':')[0]}: низкий объём — пропуск")
                            continue

                        res["score_final"]   = fin_score
                        res["aligned"]       = aligned
                        res["long_ratio"]    = l_ratio
                        res["тренд_4h"]      = тренд4h
                        кандидаты.append((sym, res, "long"))
                        log.info(f"✅ ЛОНГ кандидат: {sym.split(':')[0]} скор={fin_score} align={aligned}/6 4h={тренд4h}")

                # --- ШОРТ ---
                if allow_short and тренд4h != "bull":
                    res = получить_скор_шорта(sym)
                    if res["score"] >= MIN_SCORE:
                        det  = res.get("details", {})
                        sr   = res.get("sr", {})

                        # Согласованность
                        aligned = подсчитать_согласованность(det, "short")
                        if aligned < MIN_INDICATORS_ALIGNED:
                            log.info(f"⚠️ {sym.split(':')[0]}: только {aligned}/6 индикаторов согласованы — пропуск")
                            continue

                        r_data  = получить_bybit_ratio(sym)
                        l_ratio = r_data.get("long_ratio", long_ratio)
                        log.info(f"Bybit ratio: long={l_ratio:.1%} short={r_data.get('short_ratio',0.5):.1%} "
                                 f"сигнал={r_data.get('signal','?')}")

                        fin_score = применить_ratio_корректировку(res["score"], "short", l_ratio)
                        if fin_score < MIN_SCORE:
                            log.info(f"⚠️ {sym.split(':')[0]}: после корректировки скор={fin_score} — пропуск")
                            continue

                        # S/R — не заходим у поддержки для шорта
                        if sr.get("near_support") and sr.get("dist_to_sup_pct", 99) < SR_BLOCK_DIST_PCT:
                            log.info(f"⛔ {sym.split(':')[0]}: поддержка {sr.get('dist_to_sup_pct',0):.2f}% — пропуск")
                            continue

                        # RSI перепродан — не шортим
                        rsi_v = float(det.get("rsi", 50))
                        if rsi_v < 30:
                            log.info(f"⚠️ {sym.split(':')[0]}: RSI={rsi_v:.1f} перепродан — пропуск")
                            continue

                        # Объём
                        if float(det.get("объём_ratio", 0)) < MIN_VOLUME_RATIO:
                            log.info(f"⚠️ {sym.split(':')[0]}: низкий объём — пропуск")
                            continue

                        res["score_final"] = fin_score
                        res["aligned"]     = aligned
                        res["long_ratio"]  = l_ratio
                        res["тренд_4h"]    = тренд4h
                        кандидаты.append((sym, res, "short"))
                        log.info(f"✅ ШОРТ кандидат: {sym.split(':')[0]} скор={fin_score} align={aligned}/6 4h={тренд4h}")

            # Сортируем по финальному скору
            кандидаты.sort(key=lambda x: x[1]["score_final"], reverse=True)

            if not кандидаты:
                log.info(f"Нет кандидатов — ждём {SCAN_INTERVAL} сек")
                time.sleep(SCAN_INTERVAL)
                continue

            # Берём лучшего
            выбрана, данные, side = кандидаты[0]
            фин_скор  = данные["score_final"]
            цена      = данные["price"]
            sr_info   = данные.get("sr", {})
            det       = данные.get("details", {})
            aligned   = данные.get("aligned", 0)
            l_ratio_v = данные.get("long_ratio", 0.5)
            тренд4h_v = данные.get("тренд_4h", "neutral")

            log.info(f"► Выбрана {выбрана.split(':')[0]} ({side}) скор={фин_скор} цена={цена:.8f}")

            # ---- ПОДТВЕРЖДЕНИЕ ВХОДА (ждём 1 бар) ----
            if ENTRY_CONFIRM_BARS > 0:
                log.info(f"⏳ Ждём подтверждение ({ENTRY_CONFIRM_BARS} бар × 5m = {ENTRY_CONFIRM_BARS*5} мин)...")
                time.sleep(ENTRY_CONFIRM_BARS * 5 * 60)

                # Перепроверяем скор после паузы
                if side == "long":
                    re_check = получить_скор(выбрана)
                else:
                    re_check = получить_скор_шорта(выбрана)
                re_score = применить_ratio_корректировку(
                    re_check["score"], side, получить_bybit_ratio(выбрана).get("long_ratio", 0.5))
                if re_score < MIN_SCORE - 10:  # Допускаем небольшое снижение скора
                    log.info(f"⚠️ Скор упал после подтверждения: {re_score} — пропуск {выбрана.split(':')[0]}")
                    time.sleep(10)
                    continue
                # Обновляем цену
                ticker_new = safe_fetch_ticker(выбрана)
                if ticker_new:
                    цена = float(ticker_new["last"])
                    log.info(f"📌 Подтверждение получено, обновлённая цена: {цена:.8f}")

            # ---- РАСЧЁТ SL/TP ЧЕРЕЗ ATR ----
            atr_val = 0.0
            try:
                raw_atr = safe_fetch_ohlcv(выбрана, TIMEFRAME_TA, limit=50)
                if len(raw_atr) >= 20:
                    df_atr  = pd.DataFrame(raw_atr, columns=["ts","o","h","l","c","v"])
                    atr_val = float(calc_atr(df_atr, 14).iloc[-1])
            except: pass

            if atr_val <= 0:
                atr_val = цена * 0.008  # Fallback 0.8%

            if side == "long":
                sl_dist   = atr_val * ATR_SL_MULT
                tp_dist   = atr_val * ATR_TP_MULT
                sl_цена   = цена - sl_dist
                tp_цена   = цена + tp_dist

                # Сдвигаем SL под поддержку если она рядом
                sup = sr_info.get("support", 0)
                if sup > 0 and sup < sl_цена and sup > цена * 0.96:
                    sl_цена = sup * 0.998

                # Ограничения
                sl_dist_pct = (цена - sl_цена) / цена * 100
                if sl_dist_pct < MIN_SL_PERCENT:
                    sl_цена      = цена * (1 - MIN_SL_PERCENT / 100)
                    sl_dist_pct  = MIN_SL_PERCENT
                elif sl_dist_pct > MAX_SL_PERCENT:
                    sl_цена      = цена * (1 - MAX_SL_PERCENT / 100)
                    sl_dist_pct  = MAX_SL_PERCENT
                    tp_цена      = цена + (цена - sl_цена) * MIN_RR

            else:  # short
                sl_dist  = atr_val * ATR_SL_MULT
                tp_dist  = atr_val * ATR_TP_MULT
                sl_цена  = цена + sl_dist
                tp_цена  = цена - tp_dist

                # Сдвигаем SL над сопротивлением
                res = sr_info.get("resistance", 0)
                if res > 0 and res > sl_цена and res < цена * 1.04:
                    sl_цена = res * 1.002

                sl_dist_pct = (sl_цена - цена) / цена * 100
                if sl_dist_pct < MIN_SL_PERCENT:
                    sl_цена      = цена * (1 + MIN_SL_PERCENT / 100)
                    sl_dist_pct  = MIN_SL_PERCENT
                elif sl_dist_pct > MAX_SL_PERCENT:
                    sl_цена      = цена * (1 + MAX_SL_PERCENT / 100)
                    sl_dist_pct  = MAX_SL_PERCENT
                    tp_цена      = цена - (sl_цена - цена) * MIN_RR

            real_rr = abs(tp_цена - цена) / abs(цена - sl_цена) if abs(цена - sl_цена) > 0 else 0
            log.info(f"📐 ATR={atr_val/цена*100:.2f}% SL={sl_dist_pct:.2f}% RR={real_rr:.1f}:1")

            # Жёсткий RR-фильтр
            if real_rr < MIN_RR:
                log.warning(f"⛔ RR={real_rr:.1f}:1 < {MIN_RR}:1 — пропуск {выбрана.split(':')[0]}")
                time.sleep(10)
                continue

            margin = рассчитать_размер_позиции(фин_скор, свободный, sl_dist_pct)
            if свободный < margin * 1.1:
                log.warning(f"⚠️ Баланс {свободный:.2f} < маржа {margin:.2f} — уменьшаем")
                margin = свободный * 0.8

            log.info(f"✅ ВХОД {side.upper()}: скор={фин_скор} align={aligned}/6 | "
                     f"SL={sl_цена:.8f} | TP={tp_цена:.8f} | маржа={margin:.2f}U")

            # ---- ОТКРЫТИЕ ----
            баланс_до   = полный_баланс_usdt()
            время_входа = time.time()
            вход_цена, кол_во = открыть_позицию(выбрана, margin, tp_цена, sl_цена, side)

            if вход_цена is None or кол_во is None:
                log.warning("❌ Не удалось открыть позицию — пауза 10 сек")
                fail_attempts[выбрана] = fail_attempts.get(выбрана, 0) + 1
                cnt = fail_attempts[выбрана]
                log.warning(f"  ► {выбрана.split(':')[0]}: попытка {cnt}/{SYMBOL_MAX_FAIL_ATTEMPTS}")
                if cnt >= SYMBOL_MAX_FAIL_ATTEMPTS:
                    заблокированные[выбрана] = time.time() + SYMBOL_BLOCK_AFTER_FAIL * 60
                    fail_attempts.pop(выбрана, None)
                    log.warning(f"  ⛔ {выбрана.split(':')[0]} заблокирован на {SYMBOL_BLOCK_AFTER_FAIL} мин")
                time.sleep(10)
                continue

            fail_attempts.pop(выбрана, None)
            stats["сделок_всего"] += 1
            сохранить_состояние()

            # ---- МОНИТОРИНГ ----
            результат   = "sl"
            monitor_pnl = 0.0
            try:
                результат, monitor_pnl = мониторить_позицию(
                    выбрана, вход_цена, кол_во, время_входа,
                    sl_цена, tp_цена, side
                )
            except Exception as me:
                log.error(f"💥 Краш мониторинга: {me}")
                закрыть_позицию_с_подтверждением(выбрана, кол_во, side)
                результат = "sl"

            time.sleep(3)
            баланс_после  = полный_баланс_usdt()
            pnl_реальный  = баланс_после - баланс_до
            длит_мин      = (time.time() - время_входа) / 60

            if результат == "tp":
                stats["тейкпрофит"]   += 1
                stats["прибыль_usdt"] += max(0, pnl_реальный)
                stats["sl_streak"]     = 0
                log.info(f"✅ TP: прибыль ≈{pnl_реальный:+.4f} USDT")
                заблокированные[выбрана] = time.time() + SYMBOL_BLOCK_AFTER_TP * 60
            elif результат == "sl":
                stats["стоплосс"]    += 1
                stats["убыток_usdt"] += abs(min(0, pnl_реальный))
                stats["sl_streak"]    = stats.get("sl_streak", 0) + 1
                log.warning(f"❌ SL: убыток ≈{pnl_реальный:+.4f} USDT "
                             f"streak={stats['sl_streak']}/{SL_STREAK_LIMIT}")
                заблокированные[выбрана] = time.time() + SYMBOL_BLOCK_AFTER_SL * 60
            else:  # таймаут
                stats["таймаут"] += 1
                if pnl_реальный >= 0: stats["прибыль_usdt"] += pnl_реальный
                else:                  stats["убыток_usdt"]  += abs(pnl_реальный)
                stats["sl_streak"] = 0
                log.warning(f"⏰ Таймаут: P&L ≈{pnl_реальный:+.4f} USDT")
                заблокированные[выбрана] = time.time() + SYMBOL_BLOCK_AFTER_TP * 60

            запись = {
                "id":           stats["сделок_всего"],
                "время_входа":  datetime.fromtimestamp(время_входа).strftime("%d.%m.%Y %H:%M:%S"),
                "время_выхода": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                "duration_min": round(длит_мин, 1),
                "symbol":       выбрана,
                "side":         side,
                "score":        фин_скор,
                "aligned_indicators": aligned,
                "тренд_4h":     тренд4h_v,
                "long_ratio":   round(l_ratio_v, 3),
                "entry_price":  вход_цена,
                "sl_price":     sl_цена,
                "tp_price":     tp_цена,
                "sl_dist_pct":  round(sl_dist_pct, 3),
                "margin_usdt":  margin,
                "leverage":     LEVERAGE,
                "результат":    результат,
                "pnl_usdt":     round(pnl_реальный, 4),
                "rr_ratio":     round(real_rr, 2),
                "details":      det,
            }
            сохранить_сделку(запись)
            обновить_статистику_инд(запись)
            детальный_отчёт(запись)
            сохранить_состояние()

            log.info("Сделка завершена — пауза 30 сек")
            time.sleep(30)

        except Exception as e:
            log.error(f"💥 Глобальная ошибка: {e}", exc_info=True)
            time.sleep(30)

if __name__ == "__main__":
    main()
