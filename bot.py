#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
Bybit ГИБРИДНЫЙ ПРОФЕССИОНАЛЬНЫЙ БОТ — v10.3
================================================================================
ИСПРАВЛЕНИЯ v10.3:
1. NEW: Независимый расчёт шорт-скора (не инверсия лонг-скора).
2. NEW: Блокировка шортов при Bybit ratio long > 62%.
3. NEW: Штраф за Bybit ratio для шортов (long_ratio > 70% → -20, 65% → -10).
4. FIX: MIN_SCORE = 75, ENTRY_CONFIRM_MIN_SCORE = 70.
5. FIX: TP_PERCENT = 2.0, SL_PERCENT = 0.8 (RR = 2.5:1).
6. FIX: TRADE_MAX_LIFETIME = 3600 (1 час).
7. FIX: PARTIAL_BE_PROFIT = 0.15.
8. FIX: Блокировка ATOM/USDT на 24 часа.
9. FIX: SL_STREAK_PAUSE = 5400 (1.5 часа).
10. FIX: Счётчик проваленных входов (SYMBOL_MAX_FAIL_ATTEMPTS = 3).
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
from scipy import stats as scipy_stats
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import coint, adfuller
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
    "1000PEPE/USDT:USDT", "WIF/USDT:USDT", "BOME/USDT:USDT",
    "RENDER/USDT:USDT", "TAO/USDT:USDT", "WLD/USDT:USDT", "ARKM/USDT:USDT",
    "IO/USDT:USDT", "ONDO/USDT:USDT", "VIRTUAL/USDT:USDT", "UNI/USDT:USDT",
    "AAVE/USDT:USDT", "ARB/USDT:USDT", "OP/USDT:USDT", "LINK/USDT:USDT",
    "GRT/USDT:USDT", "INJ/USDT:USDT", "SUI/USDT:USDT", "APT/USDT:USDT",
    "TIA/USDT:USDT", "JTO/USDT:USDT", "EIGEN/USDT:USDT", "HBAR/USDT:USDT",
    "VET/USDT:USDT", "NOT/USDT:USDT", "CATI/USDT:USDT",
]

# --- ПАРАМЕТРЫ КВАНТОВОГО АНАЛИЗА ---
QUANT_ENABLED = True
COINTEGRATION_PAIRS = [
    ("BTC/USDT:USDT", "ETH/USDT:USDT"),
    ("BTC/USDT:USDT", "BNB/USDT:USDT"),
    ("ETH/USDT:USDT", "SOL/USDT:USDT"),
]
COINTEGRATION_WINDOW = 100
MEAN_REVERSION_THRESHOLD = 2.0
MOMENTUM_WINDOW = 20

# --- ORDER FLOW ---
ORDER_FLOW_ENABLED = True
ORDER_BOOK_DEPTH = 20
VOLUME_PROFILE_BARS = 50
CLUSTER_TOLERANCE = 0.005

# --- ОСНОВНЫЕ ---
LEVERAGE = 3
TIMEFRAME_TA = "5m"
TIMEFRAME_TREND = "1h"
TIMEFRAME_4H = "4h"
SCAN_INTERVAL = 300

# --- ПОРОГИ СКОРА ---
MIN_SCORE = 70  # Повышен с 65
ENTRY_CONFIRM_MIN_SCORE = 65  # Повышен с 60/65

# --- TP / SL ---
TP_PERCENT = 2.0  # Уменьшен с 3.0
SL_PERCENT = 0.8
MIN_SL_PERCENT = 0.8
MAX_SL_PERCENT = 2.0
ATR_SL_MULT = 1.5
ATR_TP_MULT = 3.0

# --- ЧАСТИЧНЫЙ БЕЗУБЫТОК ---
PARTIAL_BE_ENABLED = True
PARTIAL_BE_CLOSE_PCT = 50.0
PARTIAL_BE_PROFIT = 0.15  # Уменьшен с 0.2

# --- РИСК ---
BASE_RISK_PCT = 0.8
MAX_RISK_PCT = 1.2

# --- ТРЕЙЛИНГ ---
TRAILING_ATR_PERIOD = 14
TRAILING_ATR_MULT = 2.0
TRAILING_OFFSET_MULT = 1.5
MIN_TRAILING_STEP = 0.4
MIN_TRAILING_OFFSET = 0.6
MIN_PROFIT_FOR_TRAIL = 1.0
RR_EXIT_TRIGGER = 0.6

# --- MA КРОССОВЕР ---
MA_CROSSOVER_ENABLED = True
MA1_TYPE = "EMA"
MA2_TYPE = "EMA"
MA1_LENGTH = 21
MA2_LENGTH = 50

# --- ФИЛЬТРЫ ---
SESSION_FILTER_ENABLED = False
SESSION_BLOCK_START = 0
SESSION_BLOCK_END = 4

DAILY_LOSS_LIMIT_PCT = 3.0
DAILY_LOSS_PAUSE_SEC = 10800

VOLUME_SPIKE_MULT = 2.5
VOLUME_AVG_PERIOD = 20

SIGNAL_EXIT_ENABLED = True
ENTRY_CONFIRM_BARS = 1

# --- БЛОКИРОВКИ ---
SYMBOL_BLOCK_AFTER_TP = 90  # минут
SYMBOL_BLOCK_AFTER_SL = 180  # минут
SYMBOL_MAX_FAIL_ATTEMPTS = 3  # Максимум отменённых входов подряд
SYMBOL_BLOCK_AFTER_FAIL = 120  # Минут блокировки после SYMBOL_MAX_FAIL_ATTEMPTS провалов

SL_STREAK_LIMIT = 2
SL_STREAK_PAUSE = 5400  # Увеличен с 3600 (1.5 часа)
SL_STREAK_EXTRA_PAUSE = 300

MIN_BALANCE = 20.0
MAX_DRAWDOWN_PCT = 15.0

TRADE_MAX_LIFETIME = 3600  # Уменьшен с 7200 (1 час)
REPORT_INTERVAL = 1800

# --- ФАЙЛЫ ---
STATE_FILE = "state_bot_v10_3.json"
TRADES_FILE = "trades_bot_v10_3.json"
INDICATOR_STATS_FILE = "indicator_stats_v10_3.json"
METRICS_FILE = "strategy_metrics_v10_3.json"

BYBIT_FEE = 0.00055

# --- S/R ---
SR_PERIOD = 100
SR_PROXIMITY_PCT = 0.5
SR_MIN_TOUCHES = 3
SR_CLUSTER_TOL = 0.005
SR_BLOCK_DIST_PCT = 0.3

# --- RATE LIMIT ЗАЩИТА ---
API_CALL_DELAY = 0.3  # секунды между запросами к одному символу
API_RATE_LIMIT_PAUSE = 5  # пауза при rate limit

# ============================================================
#                       ЛОГИРОВАНИЕ
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]  %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot_v10_3.log", encoding="utf-8"),
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
#        ДВУХФАКТОРНАЯ СИСТЕМА ОБХОДА ОШИБОК БИРЖИ
# ============================================================

def safe_api_call(func, *args, retries=3, delay=1.0, ignore_errors=None, **kwargs):
    ignore_errors = ignore_errors or []
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except ccxt.RateLimitExceeded as e:
            log.warning(f"Rate limit, пауза {API_RATE_LIMIT_PAUSE}с (попытка {attempt+1}/{retries})")
            time.sleep(API_RATE_LIMIT_PAUSE)
        except ccxt.NetworkError as e:
            log.warning(f"Сетевая ошибка: {e}, пауза {delay}с (попытка {attempt+1}/{retries})")
            time.sleep(delay)
            delay *= 2
        except Exception as e:
            err_str = str(e)
            for ignore in ignore_errors:
                if ignore.lower() in err_str.lower():
                    log.info(f"Игнорируем ожидаемый ответ: {ignore}")
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
        result = safe_api_call(exchange.fetch_ohlcv, symbol, timeframe, limit=limit, retries=3)
        return result if result is not None else []
    except Exception as e:
        log.debug(f"fetch_ohlcv {symbol} {timeframe}: {e}")
        return []

def safe_fetch_ticker(symbol: str) -> Optional[dict]:
    try:
        return safe_api_call(exchange.fetch_ticker, symbol, retries=3)
    except Exception as e:
        log.debug(f"fetch_ticker {symbol}: {e}")
        return None

def safe_fetch_positions(symbols: Optional[List[str]] = None) -> List[dict]:
    try:
        if symbols:
            result = safe_api_call(exchange.fetch_positions, symbols, retries=3)
        else:
            result = safe_api_call(exchange.fetch_positions, retries=3)
        return result if result is not None else []
    except Exception as e:
        log.warning(f"safe_fetch_positions ошибка: {e}")
        return []

# ============================================================
#              БАЗОВЫЕ ФУНКЦИИ ИНДИКАТОРОВ
# ============================================================

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
    if ma_type == "EMA": return _ema(s, length)
    elif ma_type == "SMA": return _sma(s, length)
    elif ma_type == "WMA": return _wma(s, length)
    elif ma_type == "HMA": return _hma(s, length)
    else: return _ema(s, length)

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
    ub = (hl2 + mult * atr).copy()
    lb = (hl2 - mult * atr).copy()
    trend = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        c, pc = df["c"].iloc[i], df["c"].iloc[i-1]
        pu, pl, pt = ub.iloc[i-1], lb.iloc[i-1], trend.iloc[i-1]
        ub.iloc[i] = ub.iloc[i] if ub.iloc[i] < pu or pc > pu else pu
        lb.iloc[i] = lb.iloc[i] if lb.iloc[i] > pl or pc < pl else pl
        if pt == 1 and c < lb.iloc[i]: trend.iloc[i] = -1
        elif pt == -1 and c > ub.iloc[i]: trend.iloc[i] = 1
        else: trend.iloc[i] = pt
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
    rng = qty * calc_atr(df, period)
    filt = close.copy()
    for i in range(1, len(close)):
        c, r, pf = close.iloc[i], rng.iloc[i], filt.iloc[i-1]
        if c - r > pf: filt.iloc[i] = c - r
        elif c + r < pf: filt.iloc[i] = c + r
        else: filt.iloc[i] = pf
    up = (filt > filt.shift(1)) & (close > filt)
    down = (filt < filt.shift(1)) & (close < filt)
    return filt, filt + rng, filt - rng, up, down

def calc_support_resistance(df: pd.DataFrame, period: int = SR_PERIOD) -> dict:
    df_sr = df.tail(period).reset_index(drop=True)
    highs = df_sr["h"].values
    lows = df_sr["l"].values
    close = float(df["c"].iloc[-1])
    raw_res, raw_sup = [], []

    for i in range(2, len(highs) - 2):
        if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
            highs[i] > highs[i+1] and highs[i] > highs[i+2]):
            raw_res.append(highs[i])
        if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
            lows[i] < lows[i+1] and lows[i] < lows[i+2]):
            raw_sup.append(lows[i])

    def кластеризовать(levels):
        if not levels: return []
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
        "support": round(nearest_sup, 10), "resistance": round(nearest_res, 10),
        "dist_to_sup_pct": round(dist_sup, 2), "dist_to_res_pct": round(dist_res, 2),
        "sup_cluster": sup_n, "res_cluster": res_n,
        "near_support": near_sup, "near_resistance": near_res,
    }

# ============================================================
#               ФИЛЬТРЫ И КРОССОВЕРЫ
# ============================================================

def проверить_ma_кроссовер(df: pd.DataFrame, side: str = "long") -> bool:
    if not MA_CROSSOVER_ENABLED: return True
    try:
        min_len = max(MA1_LENGTH, MA2_LENGTH) * 2 + 5
        if len(df) < min_len: return True
        ma1 = calc_ma(df, MA1_TYPE, MA1_LENGTH)
        ma2 = calc_ma(df, MA2_TYPE, MA2_LENGTH)
        return bool(ma1.iloc[-1] > ma2.iloc[-1]) if side == "long" else bool(ma1.iloc[-1] < ma2.iloc[-1])
    except Exception as e:
        log.warning(f"Ошибка MA кроссовера: {e}")
        return True

def volume_spike_guard(df: pd.DataFrame) -> bool:
    try:
        vol_avg = df["v"].rolling(VOLUME_AVG_PERIOD).mean().iloc[-1]
        vol_now = df["v"].iloc[-1]
        ratio = vol_now / (vol_avg + 1e-10)
        if ratio > VOLUME_SPIKE_MULT:
            log.info(f"Volume Spike Guard: объём {ratio:.1f}x > {VOLUME_SPIKE_MULT}x")
            return False
        return True
    except Exception: return True

def торговля_разрешена_по_времени() -> bool:
    if not SESSION_FILTER_ENABLED: return True
    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour
    if SESSION_BLOCK_START < SESSION_BLOCK_END:
        blocked = SESSION_BLOCK_START <= hour < SESSION_BLOCK_END
    else:
        blocked = hour >= SESSION_BLOCK_START or hour < SESSION_BLOCK_END
    if blocked: log.info(f"Session Filter: час {hour} UTC заблокирован")
    return not blocked

def получить_bybit_ai(symbol: str) -> dict:
    result = {"signal": "neutral", "long_ratio": 0.5, "short_ratio": 0.5, "available": False}
    try:
        coin = symbol.split("/")[0]
        url = f"https://api.bybit.com/v5/market/account-ratio?category=linear&symbol={coin}USDT&period=1h&limit=1"
        resp = requests.get(url, timeout=5)
        data = resp.json()
        if data.get("retCode") == 0:
            items = data.get("result", {}).get("list", [])
            if items:
                buy_r = float(items[0].get("buyRatio", 0.5))
                sell_r = float(items[0].get("sellRatio", 0.5))
                result.update({"long_ratio": buy_r, "short_ratio": sell_r, "available": True})
                if buy_r > 0.6: result["signal"] = "bullish"
                elif buy_r < 0.4: result["signal"] = "bearish"
    except Exception as e:
        log.debug(f"Bybit ratio недоступен: {e}")
    return result

def тренд_4h_бычий(symbol: str) -> bool:
    try:
        raw = safe_fetch_ohlcv(symbol, TIMEFRAME_4H, limit=60)
        if len(raw) < 55: return False
        df = pd.DataFrame(raw, columns=["ts","o","h","l","c","v"])
        return bool(_ema(df["c"], 20).iloc[-1] > _ema(df["c"], 50).iloc[-1])
    except Exception: return False

def тренд_4h_медвежий(symbol: str) -> bool:
    try:
        raw = safe_fetch_ohlcv(symbol, TIMEFRAME_4H, limit=60)
        if len(raw) < 55: return False
        df = pd.DataFrame(raw, columns=["ts","o","h","l","c","v"])
        return bool(_ema(df["c"], 20).iloc[-1] < _ema(df["c"], 50).iloc[-1])
    except Exception: return False

# ============================================================
#              БАЙЕСОВСКИЙ ТРЕНД
# ============================================================

def bayes_trend_probability(df: pd.DataFrame) -> float:
    try:
        close = df["c"]
        ema20 = _ema(close, 20).iloc[-1]
        ema50 = _ema(close, 50).iloc[-1]
        rsi = calc_rsi(close).iloc[-1]
        adx, _, _ = calc_adx(df)
        adx_val = adx.iloc[-1]
        z = ((ema20/ema50 - 1)*100 + (rsi - 50)/25 + (adx_val - 25)/10)
        return float(np.clip(1.0 / (1.0 + np.exp(-z)), 0.0, 1.0))
    except Exception: return 0.5

# ============================================================
#       КВАНТОВЫЙ МОДУЛЬ (Ernest Chan)
# ============================================================

def calculate_cointegration(pair: Tuple[str, str], window: int = COINTEGRATION_WINDOW) -> Dict[str, Any]:
    try:
        symbol1, symbol2 = pair
        ohlcv1 = safe_fetch_ohlcv(symbol1, TIMEFRAME_TA, limit=window)
        ohlcv2 = safe_fetch_ohlcv(symbol2, TIMEFRAME_TA, limit=window)
        if len(ohlcv1) < window or len(ohlcv2) < window:
            return {"coint": 0, "pvalue": 1, "spread": 0, "zscore": 0, "valid": False}

        df1 = pd.DataFrame(ohlcv1, columns=["ts", "o", "h", "l", "c", "v"])["c"]
        df2 = pd.DataFrame(ohlcv2, columns=["ts", "o", "h", "l", "c", "v"])["c"]

        min_len = min(len(df1), len(df2))
        df1 = df1.iloc[-min_len:].reset_index(drop=True)
        df2 = df2.iloc[-min_len:].reset_index(drop=True)

        adf1 = adfuller(df1)
        adf2 = adfuller(df2)
        if adf1[1] > 0.05 or adf2[1] > 0.05:
            df1 = df1.diff().dropna()
            df2 = df2.diff().dropna()
            min_len = min(len(df1), len(df2))
            df1 = df1.iloc[-min_len:].reset_index(drop=True)
            df2 = df2.iloc[-min_len:].reset_index(drop=True)

        coint_t, pvalue, crit_vals = coint(df1, df2)
        hedge_ratio, _ = np.polyfit(df2.values, df1.values, 1)

        spread = df1 - hedge_ratio * df2
        spread_mean = spread.mean()
        spread_std = spread.std()
        current_spread = float(spread.iloc[-1])
        zscore = (current_spread - spread_mean) / spread_std if spread_std > 0 else 0

        return {
            "coint": float(coint_t),
            "pvalue": float(pvalue),
            "spread": current_spread,
            "zscore": float(zscore),
            "hedge_ratio": float(hedge_ratio),
            "spread_mean": float(spread_mean),
            "spread_std": float(spread_std),
            "valid": True,
        }
    except Exception as e:
        log.debug(f"Ошибка коинтеграции для {pair}: {e}")
        return {"coint": 0, "pvalue": 1, "spread": 0, "zscore": 0, "valid": False}

def check_mean_reversion_opportunity(symbol: str) -> Dict[str, Any]:
    try:
        window = int(MEAN_REVERSION_THRESHOLD * 2) * 10
        ohlcv = safe_fetch_ohlcv(symbol, TIMEFRAME_TA, limit=window)
        if len(ohlcv) < window: return {"signal": "neutral", "zscore": 0, "valid": False}
        df = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c", "v"])
        close = df["c"]
        sma = close.rolling(window).mean()
        std = close.rolling(window).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        current_price = close.iloc[-1]
        sma_val = sma.iloc[-1]
        zscore = (current_price - sma_val) / std.iloc[-1] if std.iloc[-1] > 0 else 0
        if zscore > MEAN_REVERSION_THRESHOLD: signal = "sell"
        elif zscore < -MEAN_REVERSION_THRESHOLD: signal = "buy"
        else: signal = "neutral"
        return {
            "signal": signal, "zscore": float(zscore),
            "bollinger_upper": float(upper_band.iloc[-1]),
            "bollinger_lower": float(lower_band.iloc[-1]),
            "sma": float(sma_val), "current_price": float(current_price), "valid": True
        }
    except Exception as e:
        log.debug(f"Ошибка Mean Reversion для {symbol}: {e}")
        return {"signal": "neutral", "zscore": 0, "valid": False}

def calculate_momentum(symbol: str, window: int = MOMENTUM_WINDOW) -> Dict[str, Any]:
    try:
        ohlcv = safe_fetch_ohlcv(symbol, TIMEFRAME_TA, limit=window + 10)
        if len(ohlcv) < window: return {"momentum": 0, "signal": "neutral", "valid": False}
        df = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c", "v"])
        close = df["c"]
        momentum = (close.iloc[-1] - close.iloc[-window]) / close.iloc[-window] * 100
        ema_fast = _ema(close, window // 2)
        ema_slow = _ema(close, window)
        ema_momentum = (ema_fast.iloc[-1] - ema_slow.iloc[-1]) / ema_slow.iloc[-1] * 100
        if momentum > 2: signal = "bullish"
        elif momentum < -2: signal = "bearish"
        else: signal = "neutral"
        return {"momentum": float(momentum), "ema_momentum": float(ema_momentum), "signal": signal, "valid": True}
    except Exception as e:
        log.debug(f"Ошибка Momentum для {symbol}: {e}")
        return {"momentum": 0, "signal": "neutral", "valid": False}

def get_quant_signals(symbol: str) -> Dict[str, Any]:
    if not QUANT_ENABLED:
        return {"quant_score": 0, "details": {}}
    signals = {
        "cointegration": {},
        "mean_reversion": check_mean_reversion_opportunity(symbol),
        "momentum": calculate_momentum(symbol),
    }
    for pair in COINTEGRATION_PAIRS:
        if symbol in pair:
            other_symbol = pair[0] if pair[1] == symbol else pair[1]
            coint_result = calculate_cointegration(pair)
            if coint_result["valid"]:
                signals["cointegration"][other_symbol] = coint_result
    quant_score = 0
    if signals["mean_reversion"]["valid"]:
        zscore = abs(signals["mean_reversion"]["zscore"])
        if zscore > MEAN_REVERSION_THRESHOLD: quant_score += 25
        elif zscore > MEAN_REVERSION_THRESHOLD * 0.7: quant_score += 15
    if signals["momentum"]["valid"]:
        momentum = signals["momentum"]["momentum"]
        if abs(momentum) > 3: quant_score += 20
        elif abs(momentum) > 1.5: quant_score += 10
    for other_symbol, coint_data in signals["cointegration"].items():
        if coint_data["valid"] and coint_data["pvalue"] < 0.05:
            zscore = abs(coint_data["zscore"])
            if zscore > 2: quant_score += 20
            elif zscore > 1.5: quant_score += 10
    return {"quant_score": min(100, quant_score), "details": signals}

# ============================================================
#        ORDER FLOW МОДУЛЬ (Ed Ponsi)
# ============================================================

def get_order_book(symbol: str, depth: int = ORDER_BOOK_DEPTH) -> Dict[str, Any]:
    try:
        order_book = safe_api_call(exchange.fetch_order_book, symbol, depth, retries=2)
        if not order_book: return {"valid": False}
        bids = order_book["bids"]
        asks = order_book["asks"]
        total_bid_volume = sum([bid[1] for bid in bids])
        total_ask_volume = sum([ask[1] for ask in asks])
        bid_prices = [bid[0] for bid in bids]
        ask_prices = [ask[0] for ask in asks]
        avg_bid = np.mean(bid_prices) if bid_prices else 0
        avg_ask = np.mean(ask_prices) if ask_prices else 0
        best_bid = bids[0][0] if bids else 0
        best_ask = asks[0][0] if asks else 0
        spread = best_ask - best_bid
        spread_pct = (spread / best_bid * 100) if best_bid > 0 else 0
        imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume + 1e-10) * 100
        return {
            "valid": True, "best_bid": best_bid, "best_ask": best_ask,
            "spread": spread, "spread_pct": spread_pct, "avg_bid": avg_bid,
            "avg_ask": avg_ask, "total_bid_volume": total_bid_volume,
            "total_ask_volume": total_ask_volume, "imbalance": imbalance,
        }
    except Exception as e:
        log.debug(f"Ошибка Order Book для {symbol}: {e}")
        return {"valid": False}

def analyze_volume_profile(symbol: str, bars: int = VOLUME_PROFILE_BARS) -> Dict[str, Any]:
    try:
        ohlcv = safe_fetch_ohlcv(symbol, TIMEFRAME_TA, limit=bars)
        if len(ohlcv) < bars: return {"valid": False}
        df = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c", "v"])
        df["price_range"] = (df["h"] + df["l"]) / 2
        bins = 20
        hist, bin_edges = np.histogram(df["price_range"], bins=bins, weights=df["v"])
        poc_index = np.argmax(hist)
        poc_price = (bin_edges[poc_index] + bin_edges[poc_index + 1]) / 2
        poc_volume = hist[poc_index]
        current_price = df["c"].iloc[-1]
        support_levels = []
        resistance_levels = []
        volume_threshold = np.percentile(hist, 70)
        for i in range(len(hist)):
            if hist[i] >= volume_threshold:
                price_level = (bin_edges[i] + bin_edges[i + 1]) / 2
                if price_level < current_price:
                    support_levels.append(price_level)
                else:
                    resistance_levels.append(price_level)
        nearest_support = max(support_levels) if support_levels else current_price * 0.95
        nearest_resistance = min(resistance_levels) if resistance_levels else current_price * 1.05
        return {
            "valid": True, "poc_price": float(poc_price), "poc_volume": float(poc_volume),
            "nearest_support": float(nearest_support), "nearest_resistance": float(nearest_resistance),
            "current_price": float(current_price),
        }
    except Exception as e:
        log.debug(f"Ошибка Volume Profile для {symbol}: {e}")
        return {"valid": False}

def get_order_flow_signals(symbol: str) -> Dict[str, Any]:
    if not ORDER_FLOW_ENABLED:
        return {"order_flow_score": 0, "details": {}}
    signals = {
        "order_book": get_order_book(symbol),
        "volume_profile": analyze_volume_profile(symbol),
    }
    order_flow_score = 0
    if signals["order_book"]["valid"]:
        imbalance = signals["order_book"]["imbalance"]
        spread_pct = signals["order_book"]["spread_pct"]
        if imbalance > 20: order_flow_score += 15
        elif imbalance < -20: order_flow_score -= 15
        if spread_pct < 0.1: order_flow_score += 10
    if signals["volume_profile"]["valid"]:
        current_price = signals["volume_profile"]["current_price"]
        nearest_support = signals["volume_profile"]["nearest_support"]
        nearest_resistance = signals["volume_profile"]["nearest_resistance"]
        if current_price > nearest_support * 1.01: order_flow_score += 15
        elif current_price < nearest_resistance * 0.99: order_flow_score -= 15
    return {"order_flow_score": max(0, min(100, order_flow_score + 50)), "details": signals}

# ============================================================
#               РИСК-МЕНЕДЖМЕНТ
# ============================================================

def рассчитать_размер_позиции(score: int, баланс: float, sl_dist_pct: float) -> float:
    factor = max(0, (score - MIN_SCORE)) / (100 - MIN_SCORE)
    risk_pct = min(BASE_RISK_PCT + (MAX_RISK_PCT - BASE_RISK_PCT) * factor, MAX_RISK_PCT)
    max_loss_usdt = баланс * risk_pct / 100
    margin_usdt = min(max_loss_usdt / (sl_dist_pct / 100), баланс * 0.95)
    log.info(f"Скор={score} → риск={risk_pct:.1f}% SL_dist={sl_dist_pct:.2f}% маржа={margin_usdt:.2f}U")
    return round(max(1.0, margin_usdt), 2)

# ============================================================
#           РАСШИРЕННАЯ СКОРИНГОВАЯ СИСТЕМА
# ============================================================

def получить_скор(symbol: str) -> dict:
    details = {}
    score = 0
    price = 0.0
    sr = {}

    try:
        raw_ta = safe_fetch_ohlcv(symbol, TIMEFRAME_TA, limit=300)
        time.sleep(API_CALL_DELAY)
        raw_1h = safe_fetch_ohlcv(symbol, TIMEFRAME_TREND, limit=300)
        if len(raw_ta) < 100 or len(raw_1h) < 100:
            return {"score": 0, "details": {}, "price": 0, "sr": {}}

        cols = ["ts","o","h","l","c","v"]
        df_ta = pd.DataFrame(raw_ta, columns=cols).reset_index(drop=True)
        df_1h = pd.DataFrame(raw_1h, columns=cols).reset_index(drop=True)
        c_ta = df_ta["c"]
        c_1h = df_1h["c"]
        price = float(c_ta.iloc[-1])

        # RSI
        rsi_val = calc_rsi(c_ta).iloc[-1]
        details["rsi"] = round(rsi_val, 1)
        if 25 <= rsi_val <= 40: score += 20
        elif 40 < rsi_val <= 50: score += 12
        elif rsi_val < 25: score += 10
        elif 50 < rsi_val <= 60: score += 5

        rsi_1h = calc_rsi(c_1h).iloc[-1]
        details["rsi_1h"] = round(rsi_1h, 1)
        if rsi_1h < 50: score += 10
        elif rsi_1h < 60: score += 5

        # MACD
        ml, sl_macd, _ = calc_macd(c_ta)
        macd_bull = ml.iloc[-1] > sl_macd.iloc[-1]
        macd_cross = macd_bull and ml.iloc[-2] <= sl_macd.iloc[-2]
        details["macd"] = "бычий" if macd_bull else "медвежий"
        if macd_cross: score += 18
        elif macd_bull: score += 8

        # Range Filter
        _, _, _, rf_up, rf_down = calc_range_filter(df_ta)
        rf_up_now = rf_up.iloc[-1]
        details["range_filter"] = "вверх" if rf_up_now else ("вниз" if rf_down.iloc[-1] else "бок")
        if rf_up_now: score += 15

        # Supertrend
        st_up, _ = calc_supertrend(df_ta)
        details["supertrend"] = "вверх" if st_up.iloc[-1] else "вниз"
        if st_up.iloc[-1]: score += 12

        # Hull
        hu_up, _ = calc_hull(c_ta)
        details["hull"] = "вверх" if hu_up.iloc[-1] else "вниз"
        if hu_up.iloc[-1]: score += 8

        # EMA тренд 1h
        ema50_1h = _ema(c_1h, 50).iloc[-1]
        ema200_1h = _ema(c_1h, 200).iloc[-1]
        details["тренд_1h"] = "бычий" if ema50_1h > ema200_1h else "медвежий"
        if ema50_1h > ema200_1h: score += 10

        # ADX
        adx, pdi, mdi = calc_adx(df_ta)
        adx_val = adx.iloc[-1]
        details["adx"] = round(adx_val, 1)
        if adx_val > 25 and pdi.iloc[-1] > mdi.iloc[-1]: score += 10
        elif adx_val > 20 and pdi.iloc[-1] > mdi.iloc[-1]: score += 4

        # Stoch
        k_ser, _ = calc_stochastic(df_ta)
        k_val = k_ser.iloc[-1]
        details["stoch_k"] = round(k_val, 1)
        if k_val < 20: score += 10
        elif k_val < 40: score += 5

        # Volume
        vol_avg = df_ta["v"].rolling(VOLUME_AVG_PERIOD).mean().iloc[-1]
        vol_ratio = df_ta["v"].iloc[-1] / (vol_avg + 1e-10)
        details["объём_ratio"] = round(vol_ratio, 2)
        if vol_ratio > 1.5: score += 8
        elif vol_ratio > 1.2: score += 4

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
            score -= 25
            details["sr_signal"] = f"у сопротивления ❌ ({sr['res_cluster']} касаний)"
        else:
            details["sr_signal"] = f"нейтр (sup={sr['dist_to_sup_pct']:.2f}% res={sr['dist_to_res_pct']:.2f}%)"

        # 3 красных свечи
        last3_bearish = all(df_ta["c"].iloc[-i] < df_ta["o"].iloc[-i] for i in range(1, 4))
        if last3_bearish:
            score -= 20
            details["свечи_3red"] = True

        # Байес
        bayes_prob = bayes_trend_probability(df_ta)
        details["bayes_prob"] = round(bayes_prob, 2)
        score += int(bayes_prob * 10)

        # Квантовый анализ
        if QUANT_ENABLED:
            quant_data = get_quant_signals(symbol)
            quant_score = quant_data["quant_score"]
            details["quant_score"] = quant_score
            score += quant_score * 0.3

        # Order Flow
        if ORDER_FLOW_ENABLED:
            order_flow_data = get_order_flow_signals(symbol)
            of_score = order_flow_data["order_flow_score"]
            details["order_flow_score"] = of_score
            score += of_score * 0.2

        details["ma_cross"] = проверить_ma_кроссовер(df_ta, side="long")
        details["vol_spike_ok"] = volume_spike_guard(df_ta)

        return {"score": max(0, min(100, score)), "details": details, "price": price, "sr": sr}
    except Exception as e:
        log.warning(f"Ошибка анализа {symbol}: {e}")
        return {"score": 0, "details": {}, "price": 0, "sr": {}}


def получить_скор_шорта(symbol: str) -> dict:
    """
    НОВАЯ ЛОГИКА: Независимый расчёт медвежьего скора.
    """
    details = {}
    score = 0
    price = 0.0
    sr = {}

    try:
        raw_ta = safe_fetch_ohlcv(symbol, TIMEFRAME_TA, limit=300)
        raw_1h = safe_fetch_ohlcv(symbol, TIMEFRAME_TREND, limit=300)
        if len(raw_ta) < 100 or len(raw_1h) < 100:
            return {"score": 0, "details": {}, "price": 0, "sr": {}}

        cols = ["ts","o","h","l","c","v"]
        df_ta = pd.DataFrame(raw_ta, columns=cols).reset_index(drop=True)
        df_1h = pd.DataFrame(raw_1h, columns=cols).reset_index(drop=True)
        c_ta = df_ta["c"]
        c_1h = df_1h["c"]
        price = float(c_ta.iloc[-1])

        # RSI — перекуплен = шорт-сигнал
        rsi_val = calc_rsi(c_ta).iloc[-1]
        details["rsi"] = round(rsi_val, 1)
        if rsi_val >= 70: score += 25
        elif rsi_val >= 60: score += 15
        elif rsi_val >= 55: score += 8
        elif rsi_val <= 40: score -= 15

        # RSI 1h
        rsi_1h = calc_rsi(c_1h).iloc[-1]
        details["rsi_1h"] = round(rsi_1h, 1)
        if rsi_1h >= 60: score += 12
        elif rsi_1h >= 55: score += 6
        elif rsi_1h <= 45: score -= 10

        # MACD — медвежий = шорт-сигнал
        ml, sl_macd, _ = calc_macd(c_ta)
        macd_bear = ml.iloc[-1] < sl_macd.iloc[-1]
        macd_death_cross = macd_bear and ml.iloc[-2] >= sl_macd.iloc[-2]
        details["macd"] = "медвежий" if macd_bear else "бычий"
        if macd_death_cross: score += 20
        elif macd_bear: score += 10
        else: score -= 5

        # Supertrend вниз
        st_up, st_down = calc_supertrend(df_ta)
        details["supertrend"] = "вниз" if st_down.iloc[-1] else "вверх"
        if st_down.iloc[-1]: score += 15
        else: score -= 10

        # Range Filter вниз
        _, _, _, rf_up, rf_down = calc_range_filter(df_ta)
        details["range_filter"] = "вниз" if rf_down.iloc[-1] else ("вверх" if rf_up.iloc[-1] else "бок")
        if rf_down.iloc[-1]: score += 12
        elif rf_up.iloc[-1]: score -= 8

        # Hull вниз
        hu_up, hu_down = calc_hull(c_ta)
        details["hull"] = "вниз" if hu_down.iloc[-1] else "вверх"
        if hu_down.iloc[-1]: score += 10
        else: score -= 5

        # Тренд 1h — медвежий
        ema50_1h = _ema(c_1h, 50).iloc[-1]
        ema200_1h = _ema(c_1h, 200).iloc[-1]
        details["тренд_1h"] = "медвежий" if ema50_1h < ema200_1h else "бычий"
        if ema50_1h < ema200_1h: score += 12
        else: score -= 8

        # ADX — сила медвежьего тренда
        adx, pdi, mdi = calc_adx(df_ta)
        adx_val = adx.iloc[-1]
        details["adx"] = round(adx_val, 1)
        if adx_val > 25 and mdi.iloc[-1] > pdi.iloc[-1]: score += 12
        elif adx_val > 20 and mdi.iloc[-1] > pdi.iloc[-1]: score += 6
        elif adx_val > 25 and pdi.iloc[-1] > mdi.iloc[-1]: score -= 8

        # Stochastic — перекуплен
        k_ser, _ = calc_stochastic(df_ta)
        k_val = k_ser.iloc[-1]
        details["stoch_k"] = round(k_val, 1)
        if k_val >= 80: score += 12
        elif k_val >= 65: score += 6
        elif k_val <= 20: score -= 10

        # Объём
        vol_avg = df_ta["v"].rolling(VOLUME_AVG_PERIOD).mean().iloc[-1]
        vol_ratio = df_ta["v"].iloc[-1] / (vol_avg + 1e-10)
        details["объём_ratio"] = round(vol_ratio, 2)
        if vol_ratio > 1.5: score += 8
        elif vol_ratio > 1.2: score += 4

        # S/R — у сопротивления = шорт-сигнал, у поддержки = штраф
        sr = calc_support_resistance(df_ta)
        details.update({
            "support": sr["support"], "resistance": sr["resistance"],
            "dist_sup": sr["dist_to_sup_pct"], "dist_res": sr["dist_to_res_pct"]
        })
        if sr["near_resistance"]:
            score += 20
            details["sr_signal"] = f"у сопротивления ✅ ({sr['res_cluster']} касаний)"
        elif sr["near_support"]:
            score -= 20
            details["sr_signal"] = f"у поддержки ❌ ({sr['sup_cluster']} касаний)"
        else:
            details["sr_signal"] = f"нейтр (sup={sr['dist_to_sup_pct']:.2f}% res={sr['dist_to_res_pct']:.2f}%)"

        # 3 красные свечи подряд — подтверждение давления
        last3_bearish = all(
            df_ta["c"].iloc[-i] < df_ta["o"].iloc[-i] for i in range(1, 4)
        )
        if last3_bearish:
            score += 15
            details["свечи_3red"] = True

        # Байес
        bayes_prob = bayes_trend_probability(df_ta)
        details["bayes_prob"] = round(bayes_prob, 2)
        if bayes_prob < 0.4: score += 10
        elif bayes_prob > 0.6: score -= 10

        # MA кроссовер для шорта
        details["ma_cross"] = проверить_ma_кроссовер(df_ta, side="short")
        details["vol_spike_ok"] = volume_spike_guard(df_ta)

        # Quant и OrderFlow
        if QUANT_ENABLED:
            quant_data = get_quant_signals(symbol)
            quant_score = quant_data["quant_score"]
            details["quant_score"] = quant_score
            mr = quant_data["details"].get("mean_reversion", {})
            if mr.get("valid") and mr.get("zscore", 0) > MEAN_REVERSION_THRESHOLD:
                score += 15

        if ORDER_FLOW_ENABLED:
            order_flow_data = get_order_flow_signals(symbol)
            of_score = order_flow_data["order_flow_score"]
            details["order_flow_score"] = of_score
            if of_score < 40: score += 10

        final_score = max(0, min(100, score))
        return {
            "score": final_score,
            "details": details,
            "price": price,
            "sr": sr
        }

    except Exception as e:
        log.warning(f"Ошибка шорт-анализа {symbol}: {e}")
        return {"score": 0, "details": {}, "price": 0, "sr": {}}


def применить_ai_корректировку_шорт(score: int, symbol: str) -> int:
    """
    НОВАЯ ЛОГИКА: Штраф за Bybit ratio для шортов.
    """
    ai = получить_bybit_ai(symbol)
    if not ai["available"]: return score

    long_r = ai["long_ratio"]
    log.info(f"Bybit ratio: long={long_r:.1%} short={ai['short_ratio']:.1%} сигнал={ai['signal']}")

    # При сильном бычьем рынке штрафуем шорт
    if long_r > 0.70:
        return max(0, score - 20)
    elif long_r > 0.65:
        return max(0, score - 10)
    elif long_r < 0.45:
        return min(100, score + 10)  # Медвежий рынок — бонус

    return score


def применить_ai_корректировку(score: int, symbol: str) -> int:
    ai = получить_bybit_ai(symbol)
    if not ai["available"]: return score
    long_r = ai["long_ratio"]
    signal = ai["signal"]
    log.info(f"Bybit ratio: long={long_r:.1%} short={ai['short_ratio']:.1%} сигнал={signal}")
    if signal == "bullish": return min(100, score + 5)
    elif signal == "bearish": return max(0, score - 15)
    return score

# ============================================================
#         УПРАВЛЕНИЕ ПЛЕЧОМ
# ============================================================

def установить_плечо(symbol: str, leverage: int) -> bool:
    try:
        exchange.set_leverage(leverage, symbol, params={"buyLeverage": leverage, "sellLeverage": leverage})
        log.info(f"Плечо {leverage}x установлено для {symbol}")
        return True
    except Exception as e1:
        err1 = str(e1)
        if "leverage not modified" in err1.lower() or "110043" in err1:
            log.info(f"Плечо {leverage}x уже установлено для {symbol} — OK")
            return True
        log.warning(f"Метод 1 плеча не сработал: {e1}")

    try:
        coin_sym = symbol.replace("/", "").replace(":USDT", "")
        exchange.private_post_v5_position_set_leverage({
            "category": "linear", "symbol": coin_sym,
            "buyLeverage": str(leverage), "sellLeverage": str(leverage),
        })
        log.info(f"Плечо {leverage}x установлено (v5) для {symbol}")
        return True
    except Exception as e2:
        err2 = str(e2)
        if "leverage not modified" in err2.lower() or "110043" in err2:
            log.info(f"Плечо {leverage}x уже установлено (v5) для {symbol} — OK")
            return True
        log.warning(f"Метод 2 плеча не сработал: {e2}")

    try:
        positions = safe_fetch_positions([symbol])
        for pos in positions:
            cur_lev = pos.get("leverage")
            if cur_lev and abs(float(cur_lev) - leverage) < 0.1:
                log.info(f"Плечо {leverage}x подтверждено через позицию для {symbol}")
                return True
    except Exception as e3:
        log.debug(f"Проверка плеча через позицию не удалась: {e3}")

    log.warning(f"Плечо не удалось установить явно для {symbol}, продолжаем с текущим")
    return True


def обновить_sl_на_бирже(symbol: str, new_sl: float, side: str = "long") -> bool:
    try:
        sl_str = exchange.price_to_precision(symbol, new_sl)
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

def открыть_позицию(symbol: str, margin_usdt: float, tp_price: float,
                    sl_price: float, side: str = "long") -> Tuple[Optional[float], Optional[float]]:
    try:
        установить_плечо(symbol, LEVERAGE)

        ticker = safe_fetch_ticker(symbol)
        if not ticker:
            log.error(f"Не удалось получить тикер для {symbol}")
            return None, None

        price = float(ticker["last"])
        pos_size_usdt = margin_usdt * LEVERAGE
        qty_raw = pos_size_usdt / price
        qty = float(exchange.amount_to_precision(symbol, qty_raw))

        if qty <= 0:
            log.error(f"Нулевое количество {symbol}")
            return None, None

        if side == "long":
            sl_price = min(sl_price, price - max(price * MIN_SL_PERCENT/100, price * 0.001))
            tp_price = max(tp_price, price + price * TP_PERCENT/100)
        else:
            sl_price = max(sl_price, price + max(price * MIN_SL_PERCENT/100, price * 0.001))
            tp_price = min(tp_price, price - price * TP_PERCENT/100)

        tp_str = exchange.price_to_precision(symbol, tp_price)
        sl_str = exchange.price_to_precision(symbol, sl_price)
        buy_sell = "buy" if side == "long" else "sell"

        log.info(f"Открываем {side} {symbol}: qty={qty}, маржа≈{margin_usdt:.2f}U, плечо={LEVERAGE}x, TP={tp_str}, SL={sl_str}")

        order = exchange.create_market_order(symbol, buy_sell, qty, params={
            "takeProfit": float(tp_str),
            "stopLoss": float(sl_str)
        })

        entry_price = None

        if order.get("average") is not None:
            try:
                entry_price = float(order["average"])
            except (TypeError, ValueError):
                pass

        if entry_price is None or entry_price <= 0:
            if order.get("price") is not None:
                try:
                    entry_price = float(order["price"])
                except (TypeError, ValueError):
                    pass

        if entry_price is None or entry_price <= 0:
            time.sleep(2)
            try:
                positions = safe_fetch_positions([symbol])
                for pos in positions:
                    if float(pos.get("contracts", 0) or 0) > 0:
                        ep = pos.get("entryPrice") or pos.get("avgCost")
                        if ep:
                            entry_price = float(ep)
                            break
            except Exception as ep_err:
                log.debug(f"Не удалось получить entry price из позиции: {ep_err}")

        if entry_price is None or entry_price <= 0:
            entry_price = price
            log.warning(f"Entry price не получен из ордера, используем рыночную цену: {entry_price}")

        log.info(f"{side.upper()} открыт: {qty} {symbol} @ ~{entry_price:.8f}")
        return entry_price, qty

    except Exception as e:
        log.error(f"Ошибка открытия {side} {symbol}: {e}")
        return None, None


def закрыть_позицию_с_подтверждением(symbol: str, qty: float, side: str) -> bool:
    close_side = "sell" if side == "long" else "buy"
    for attempt in range(3):
        try:
            exchange.create_market_order(symbol, close_side, qty, params={"reduceOnly": True})
            time.sleep(3)
            positions = safe_fetch_positions([symbol])
            active = [p for p in positions if float(p.get("contracts", 0) or 0) > 0 and p.get("side") == side]
            if not active:
                log.info(f"Позиция {symbol} закрыта успешно")
                return True
            log.warning(f"Позиция {symbol} не закрылась, повтор через 2 сек...")
            time.sleep(2)
        except Exception as e:
            log.warning(f"Попытка {attempt+1} закрыть {symbol} не удалась: {e}")
            time.sleep(2)
    log.error(f"Не удалось закрыть {symbol} после 3 попыток")
    return False


def проверить_signal_exit(symbol: str, side: str) -> bool:
    if not SIGNAL_EXIT_ENABLED: return False
    try:
        raw = safe_fetch_ohlcv(symbol, TIMEFRAME_TA, limit=50)
        if len(raw) < 30: return False
        df = pd.DataFrame(raw, columns=["ts", "o", "h", "l", "c", "v"])
        st_up, st_down = calc_supertrend(df)
        _, _, _, rf_up, rf_down = calc_range_filter(df)
        return bool(st_down.iloc[-1] and rf_down.iloc[-1]) if side == "long" else bool(st_up.iloc[-1] and rf_up.iloc[-1])
    except Exception: return False

# ============================================================
#            МОНИТОРИНГ ПОЗИЦИЙ
# ============================================================

def мониторить_позицию(symbol: str, entry_price: float, qty: float,
                        открыта_в: float, sl_цена: float,
                        tp_цена: float, side: str = "long") -> Tuple[str, float]:
    deadline = открыта_в + TRADE_MAX_LIFETIME
    coin = symbol.split("/")[0]
    trailing_step = MIN_TRAILING_STEP / 100
    trailing_offset = MIN_TRAILING_OFFSET / 100

    try:
        raw = safe_fetch_ohlcv(symbol, TIMEFRAME_TA, limit=50)
        if len(raw) >= 30:
            df = pd.DataFrame(raw, columns=["ts", "o", "h", "l", "c", "v"])
            atr_val = calc_atr(df, TRAILING_ATR_PERIOD).iloc[-1]
            atr_pct = (atr_val / entry_price) * 100
            trailing_step = max(MIN_TRAILING_STEP, atr_pct * TRAILING_ATR_MULT) / 100
            trailing_offset = max(MIN_TRAILING_OFFSET, atr_pct * TRAILING_OFFSET_MULT) / 100
    except Exception: pass

    if side == "long":
        rr_trigger_price = entry_price + (tp_цена - entry_price) * RR_EXIT_TRIGGER
    else:
        rr_trigger_price = entry_price - (entry_price - tp_цена) * RR_EXIT_TRIGGER

    log.info(f"rrExit триггер={rr_trigger_price:.8f} (RR_EXIT={RR_EXIT_TRIGGER})")
    фаза, текущий_sl, пиковая_цена = 1, sl_цена, entry_price
    trailing_активен = (RR_EXIT_TRIGGER == 0.0)
    partial_done = False
    accumulated_pnl = 0.0

    log.info(f"Мониторинг {coin} {side} вход={entry_price:.8f} SL={sl_цена:.8f} TP={tp_цена:.8f}")

    while True:
        сейчас = time.time()
        if сейчас >= deadline:
            log.warning("Дедлайн — принудительное закрытие")
            закрыть_позицию_с_подтверждением(symbol, qty, side)
            return "таймаут", accumulated_pnl
        time.sleep(15)

        try:
            positions = safe_fetch_positions([symbol])
            active = [p for p in positions if float(p.get("contracts", 0) or 0) > 0 and p.get("side") == side]

            if not active:
                ticker = safe_fetch_ticker(symbol)
                cur_price = float(ticker["last"]) if ticker else entry_price

                hit_tp = (cur_price >= entry_price * (1 + TP_PERCENT / 100 * 0.7)) if side == "long" \
                         else (cur_price <= entry_price * (1 - TP_PERCENT / 100 * 0.7))
                result = "tp" if (hit_tp or фаза >= 2) else "sl"
                return result, accumulated_pnl

            pos = active[0]
            ticker = safe_fetch_ticker(symbol)
            cur_price = float(ticker["last"]) if ticker else entry_price
            qty_actual = abs(float(pos.get("contracts", 0) or 0))
            pnl_real = float(pos.get("unrealizedPnl", 0) or 0)
            pnl_pct = ((cur_price - entry_price) / entry_price * 100) if side == "long" \
                      else ((entry_price - cur_price) / entry_price * 100)
            до_дед = int(deadline - сейчас)

            # Частичный безубыток
            if PARTIAL_BE_ENABLED and not partial_done and pnl_pct >= PARTIAL_BE_PROFIT:
                close_qty = qty_actual * (PARTIAL_BE_CLOSE_PCT / 100)
                if close_qty > 0:
                    close_side = "sell" if side == "long" else "buy"
                    try:
                        exchange.create_market_order(symbol, close_side, close_qty, params={"reduceOnly": True})
                        partial_pnl = (cur_price - entry_price) * close_qty if side == "long" \
                                      else (entry_price - cur_price) * close_qty
                        accumulated_pnl += partial_pnl
                        log.info(f"Частичный безубыток: закрыто {close_qty:.4f} ({PARTIAL_BE_CLOSE_PCT:.0f}%) @ ~{cur_price:.8f} PnL≈{partial_pnl:+.4f}U")
                        qty_actual -= close_qty
                        new_sl = entry_price * (1 + BYBIT_FEE * 2 + 0.0003) if side == "long" \
                                 else entry_price * (1 - BYBIT_FEE * 2 - 0.0003)
                        if обновить_sl_на_бирже(symbol, new_sl, side):
                            текущий_sl = new_sl
                        partial_done = True
                    except Exception as e:
                        log.warning(f"Не удалось частично закрыть: {e}")

            # Signal Exit
            if SIGNAL_EXIT_ENABLED and фаза >= 2 and проверить_signal_exit(symbol, side):
                log.info("Signal Exit: разворот — закрываем")
                закрыть_позицию_с_подтверждением(symbol, qty_actual, side)
                result_type = "tp" if pnl_pct > 0 else "sl"
                return result_type, accumulated_pnl + pnl_real

            # Полный безубыток
            if not partial_done and фаза == 1 and pnl_pct >= 0.3:
                new_sl_be = entry_price * (1 + BYBIT_FEE * 2 + 0.0003) if side == "long" \
                            else entry_price * (1 - BYBIT_FEE * 2 - 0.0003)
                if обновить_sl_на_бирже(symbol, new_sl_be, side):
                    фаза, текущий_sl, пиковая_цена = 2, new_sl_be, cur_price
                    log.info(f"БЕЗУБЫТОК! SL → {new_sl_be:.8f}")

            # Активация трейлинга
            if not trailing_активен and фаза >= 2:
                trailing_активен = (cur_price >= rr_trigger_price) if side == "long" \
                                   else (cur_price <= rr_trigger_price)
                if trailing_активен:
                    log.info(f"Трейлинг активирован @ {cur_price:.8f}")

            # Трейлинг
            if trailing_активен and фаза >= 2 and pnl_pct >= MIN_PROFIT_FOR_TRAIL:
                if side == "long":
                    if cur_price > пиковая_цена: пиковая_цена = cur_price
                    new_sl_trail = пиковая_цена * (1 - trailing_offset)
                    if new_sl_trail > текущий_sl and обновить_sl_на_бирже(symbol, new_sl_trail, side):
                        текущий_sl = new_sl_trail
                        log.info(f"ТРЕЙЛИНГ: пик={пиковая_цена:.8f} → SL={new_sl_trail:.8f}")
                else:
                    if cur_price < пиковая_цена: пиковая_цена = cur_price
                    new_sl_trail = пиковая_цена * (1 + trailing_offset)
                    if new_sl_trail < текущий_sl and обновить_sl_на_бирже(symbol, new_sl_trail, side):
                        текущий_sl = new_sl_trail
                        log.info(f"ТРЕЙЛИНГ: пик={пиковая_цена:.8f} → SL={new_sl_trail:.8f}")

            log.info(f"[{coin}] {cur_price:.8f} P&L={pnl_pct:+.2f}% ({pnl_real:+.4f}U) "
                     f"SL={текущий_sl:.8f} фаза={фаза} дед={до_дед}с {'[частично]' if partial_done else ''}")

        except Exception as e:
            log.warning(f"Ошибка в цикле мониторинга: {e}")

    return "sl", accumulated_pnl

# ============================================================
#           ПОДХВАТ НЕЗАКРЫТЫХ ПОЗИЦИЙ ПРИ СТАРТЕ
# ============================================================

def проверить_и_подхватить_позиции() -> List[dict]:
    try:
        positions = safe_fetch_positions()
        active = [p for p in positions if float(p.get("contracts", 0) or 0) > 0]
        if not active:
            log.info("Открытых позиций нет — готов к торговле")
            return []

        log.info(f"⚠️ ОБНАРУЖЕНЫ ОТКРЫТЫЕ ПОЗИЦИИ ({len(active)} шт):")
        for pos in active:
            sym = pos.get("symbol", "?")
            s = pos.get("side", "?")
            qty = float(pos.get("contracts", 0) or 0)
            pnl = float(pos.get("unrealizedPnl", 0) or 0)
            entry = float(pos.get("entryPrice", 0) or pos.get("avgCost", 0) or 0)
            lev = pos.get("leverage", "?")
            log.info(f"  • {sym} {s} qty={qty} entry={entry:.6f} leverage={lev}x unrealizedPnL={pnl:+.4f}U")
        log.info("Бот будет ждать закрытия этих позиций перед открытием новых")
        return active
    except Exception as e:
        log.warning(f"Не удалось проверить позиции: {e}")
        return []

# ============================================================
#                СТАТИСТИКА ИНДИКАТОРОВ
# ============================================================

def загрузить_статистику_индикаторов() -> dict:
    if not os.path.exists(INDICATOR_STATS_FILE): return {}
    try:
        with open(INDICATOR_STATS_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except Exception: return {}

def сохранить_статистику_индикаторов(stats_data: dict):
    try:
        with open(INDICATOR_STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(stats_data, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        log.warning(f"Не удалось сохранить статистику индикаторов: {e}")

def обновить_статистику_индикаторов(запись_сделки: dict):
    stats_data = загрузить_статистику_индикаторов()
    details = запись_сделки.get("details", {})
    результат = запись_сделки.get("результат", "")
    is_win = (результат == "tp")
    индикаторы = {
        "rsi": lambda v: 25 <= float(v) <= 42,
        "rsi_1h": lambda v: float(v) < 55,
        "macd": lambda v: v == "бычий",
        "range_filter": lambda v: v == "вверх",
        "supertrend": lambda v: v == "вверх",
        "hull": lambda v: v == "вверх",
        "тренд_1h": lambda v: v == "бычий",
        "adx": lambda v: float(v) > 25,
        "stoch_k": lambda v: float(v) < 25,
        "объём_ratio": lambda v: float(v) > 1.5,
        "sr_signal": lambda v: "поддержки" in str(v),
        "bayes_prob": lambda v: float(v) > 0.6,
        "quant_score": lambda v: float(v) > 50,
        "order_flow_score": lambda v: float(v) > 50,
    }
    for инд, условие in индикаторы.items():
        значение = details.get(инд)
        if значение is None: continue
        try: is_bullish = условие(значение)
        except: continue
        if инд not in stats_data:
            stats_data[инд] = {"bullish": {"total": 0, "wins": 0}, "bearish": {"total": 0, "wins": 0}}
        if is_bullish:
            stats_data[инд]["bullish"]["total"] += 1
            if is_win: stats_data[инд]["bullish"]["wins"] += 1
        else:
            stats_data[инд]["bearish"]["total"] += 1
            if is_win: stats_data[инд]["bearish"]["wins"] += 1
    сохранить_статистику_индикаторов(stats_data)

def отчёт_по_индикаторам():
    stats_data = загрузить_статистику_индикаторов()
    if not stats_data:
        log.info("Статистика индикаторов пуста (нет завершённых сделок)")
        return
    log.info("")
    log.info("=" * 70)
    log.info("📈 ЭФФЕКТИВНОСТЬ ИНДИКАТОРОВ")
    log.info(f"{'Индикатор':<18} {'🟢Бычий WR%':>11}  {'n':>4}  {'🔴Медвежий WR%':>14}  {'n':>4}  {'Разница':>8}")
    log.info(" " + "─" * 70)
    for инд, данные in stats_data.items():
        b_total, b_wins = данные["bullish"]["total"], данные["bullish"]["wins"]
        be_total, be_wins = данные["bearish"]["total"], данные["bearish"]["wins"]
        b_wr = (b_wins / b_total * 100) if b_total > 0 else 0
        be_wr = (be_wins / be_total * 100) if be_total > 0 else 0
        diff = b_wr - be_wr
        знак = "▲" if diff > 5 else ("▼" if diff < -5 else "≈")
        log.info(f"{инд:<18}  {b_wr:>9.1f}%  {b_total:>4}  {be_wr:>12.1f}%  {be_total:>4}  {знак}{diff:>+7.1f}%")
    log.info("=" * 70)
    log.info("")

# ============================================================
#              ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def загрузить_историю() -> List[dict]:
    if not os.path.exists(TRADES_FILE): return []
    try:
        with open(TRADES_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except Exception: return []

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
        log.warning(f"Ошибка получения баланса: {e}")
        return 0.0

def полный_баланс_usdt() -> float:
    try:
        b = exchange.fetch_balance({"type": "linear"})
        total = float(b.get("USDT", {}).get("total", 0.0))
        if total > 0: return total
        equity = float(b.get("USDT", {}).get("equity", 0.0))
        if equity > 0: return equity
        return баланс_usdt()
    except Exception as e:
        log.warning(f"Ошибка получения полного баланса: {e}")
        return баланс_usdt()

def получить_позиции() -> List[dict]:
    positions = safe_fetch_positions()
    return [p for p in positions if float(p.get("contracts", 0) or 0) > 0]

def обновить_начало_дня(баланс: float):
    сегодня = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if stats["дата_дня"] != сегодня:
        stats["дата_дня"] = сегодня
        stats["баланс_начало_дня"] = баланс
        log.info(f"Новый торговый день. Баланс: {баланс:.2f} USDT")
        сохранить_состояние()

def превышен_дневной_лимит() -> bool:
    нач = stats.get("баланс_начало_дня", 0.0)
    if нач <= 0: return False
    текущий = полный_баланс_usdt()
    потеря_pct = (нач - текущий) / нач * 100
    if потеря_pct >= DAILY_LOSS_LIMIT_PCT:
        log.warning(f"Дневной лимит убытков: -{потеря_pct:.1f}% (лимит {DAILY_LOSS_LIMIT_PCT}%)")
        return True
    return False

# ============================================================
#            ДЕТАЛЬНЫЙ ОТЧЁТ ПО СДЕЛКЕ
# ============================================================

def детальный_отчёт_сделки(запись: dict):
    r = запись.get("результат", "?")
    sym = запись.get("symbol", "?").split(":")[0]
    pnl = запись.get("pnl_usdt", 0)
    dur = запись.get("duration_min", 0)
    side = запись.get("side", "?")
    score = запись.get("score", "?")
    entry = запись.get("entry_price", 0)
    sl = запись.get("sl_price", 0)
    tp = запись.get("tp_price", 0)
    sl_pct = запись.get("sl_dist_pct", 0)
    rr = запись.get("rr_ratio", 0)
    margin = запись.get("margin_usdt", 0)
    details = запись.get("details", {})

    знак = "✅ ТЕЙКПРОФИТ" if r == "tp" else ("❌ СТОПЛОСС" if r == "sl" else "⏰ ТАЙМАУТ")
    pnl_знак = "+" if pnl >= 0 else ""

    log.info("")
    log.info("━" * 65)
    log.info(f"📋 ДЕТАЛЬНЫЙ ОТЧЁТ СДЕЛКИ #{запись.get('id', '?')}")
    log.info("━" * 65)
    log.info(f"  Символ:      {sym} ({side.upper()})")
    log.info(f"  Результат:   {знак}")
    log.info(f"  P&L:         {pnl_знак}{pnl:.4f} USDT")
    log.info(f"  Скор входа:  {score}/100")
    log.info(f"  Длительность:{dur:.1f} мин")
    log.info(f"  Маржа:       {margin:.2f} USDT × {LEVERAGE}x = {margin*LEVERAGE:.2f} USDT")
    log.info(f"  Цены:        Вход={entry:.8f}  SL={sl:.8f}  TP={tp:.8f}")
    log.info(f"  SL расст.:   {sl_pct:.2f}%  |  RR: {rr:.1f}:1")
    log.info(f"  Время входа: {запись.get('время_входа', '?')}")
    log.info(f"  Время выхода:{запись.get('время_выхода', '?')}")
    log.info("  ─── Индикаторы ───")
    log.info(f"  RSI 5m:      {details.get('rsi', '?')}  |  RSI 1h: {details.get('rsi_1h', '?')}")
    log.info(f"  MACD:        {details.get('macd', '?')}")
    log.info(f"  Supertrend:  {details.get('supertrend', '?')}")
    log.info(f"  Range Filter:{details.get('range_filter', '?')}")
    log.info(f"  Hull:        {details.get('hull', '?')}")
    log.info(f"  ADX:         {details.get('adx', '?')}")
    log.info(f"  Stoch K:     {details.get('stoch_k', '?')}")
    log.info(f"  Тренд 1h:    {details.get('тренд_1h', '?')}")
    log.info(f"  Объём ratio: {details.get('объём_ratio', '?')}")
    log.info(f"  S/R сигнал:  {details.get('sr_signal', '?')}")
    log.info(f"  Байес:       {details.get('bayes_prob', '?')}")
    log.info(f"  Quant score: {details.get('quant_score', '?')}")
    log.info(f"  OrderFlow:   {details.get('order_flow_score', '?')}")
    log.info("━" * 65)
    log.info("")


def подтвердить_вход(symbol: str, исходный_скор: int, side: str = "long") -> bool:
    if ENTRY_CONFIRM_BARS <= 0: return True
    tf_seconds = {"1m": 60, "3m": 180, "5m": 300, "15m": 900}
    wait = tf_seconds.get(TIMEFRAME_TA, 300) * ENTRY_CONFIRM_BARS
    log.info(f"Подтверждение входа: ждём {wait}с ({ENTRY_CONFIRM_BARS} свеча)...")
    time.sleep(wait)
    новый = получить_скор(symbol) if side == "long" else получить_скор_шорта(symbol)
    новый_скор = новый["score"]
    log.info(f"Перепроверка скора: {исходный_скор} → {новый_скор} (мин={ENTRY_CONFIRM_MIN_SCORE})")
    if новый_скор < ENTRY_CONFIRM_MIN_SCORE:
        log.info(f"Подтверждение не прошло: скор упал до {новый_скор}")
        return False
    if not новый.get("details", {}).get("vol_spike_ok", True):
        log.info("Подтверждение не прошло: volume spike")
        return False
    log.info(f"Вход подтверждён. Скор {новый_скор}/100")
    return True

# ============================================================
#                       ОТЧЁТЫ
# ============================================================

def рассчитать_метрики(сделки: List[dict]) -> dict:
    if len(сделки) < 5: return {}
    pnls = [t['pnl_usdt'] for t in сделки]
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p < 0)
    total = len(pnls)
    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    max_dd = abs(min(drawdowns))
    sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252) if len(pnls) > 1 and np.std(pnls) != 0 else 0
    return {
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_usdt": round(max_dd, 2),
        "total_trades": total,
        "winrate": round(wins / total * 100, 1) if total > 0 else 0,
        "avg_win": round(np.mean([p for p in pnls if p > 0]), 2) if wins else 0,
        "avg_loss": round(abs(np.mean([p for p in pnls if p < 0])), 2) if losses else 0,
        "profit_factor": round(sum(p for p in pnls if p > 0) / abs(sum(p for p in pnls if p < 0)), 2)
                         if losses > 0 and sum(p for p in pnls if p > 0) > 0 else 0,
    }


def печатать_отчёт():
    баланс = полный_баланс_usdt()
    старт = stats["депозит_старт"]
    дельта = баланс - старт
    чистый = stats["прибыль_usdt"] - stats["убыток_usdt"]
    пct = (дельта / старт * 100) if старт > 0 else 0
    всего = stats["сделок_всего"]
    tp_ = stats["тейкпрофит"]
    sl_ = stats["стоплосс"]
    wr = (tp_ / всего * 100) if всего > 0 else 0.0

    log.info("")
    log.info("=" * 65)
    log.info("📊 ОТЧЁТ ГИБРИДНОГО БОТА v10.3")
    log.info(f"  Баланс: {баланс:.2f} USDT ({дельта:+.2f} USDT / {пct:+.2f}%)")
    log.info(f"  Сделок: {всего} | TP={tp_} | SL={sl_} | Таймаут={stats['таймаут']}")
    log.info(f"  WinRate: {wr:.1f}%")
    log.info(f"  Прибыль: {stats['прибыль_usdt']:.4f} USDT")
    log.info(f"  Убыток:  {stats['убыток_usdt']:.4f} USDT")
    log.info(f"  Чистый P&L: {чистый:+.4f} USDT")
    log.info("=" * 65)
    log.info("")

    stats["последний_отчёт"] = time.time()
    сохранить_состояние()
    отчёт_по_индикаторам()

    история = загрузить_историю()
    if len(история) >= 5:
        метрики = рассчитать_метрики(история)
        if метрики:
            log.info("📉 МЕТРИКИ:")
            log.info(f"  Sharpe: {метрики.get('sharpe_ratio',0)}")
            log.info(f"  Max Drawdown: {метрики.get('max_drawdown_usdt',0):.2f} USDT")
            log.info(f"  Profit Factor: {метрики.get('profit_factor',0)}")
            log.info(f"  Avg Win: {метрики.get('avg_win',0):.4f} USDT | Avg Loss: {метрики.get('avg_loss',0):.4f} USDT")
            try:
                with open(METRICS_FILE, "w", encoding="utf-8") as f:
                    json.dump(метрики, f, ensure_ascii=False, indent=2, default=str)
            except: pass

# ============================================================
#             ПРЕДСТАРТОВАЯ ПРОВЕРКА
# ============================================================

def запустить_предстартовую_проверку() -> bool:
    log.info("")
    log.info("=" * 65)
    log.info("🔍 ПРЕДСТАРТОВАЯ ПРОВЕРКА (5 ЭТАПОВ)")
    log.info("=" * 65)

    все_ок = True

    log.info("\n▶ Этап 1: Окружение и API ключи...")
    api_key = os.getenv("BYBIT_API_KEY", "")
    api_secret = os.getenv("BYBIT_API_SECRET", "")
    if not api_key or len(api_key) < 10:
        log.error("❌ BYBIT_API_KEY не задан или слишком короткий")
        все_ок = False
    if not api_secret or len(api_secret) < 10:
        log.error("❌ BYBIT_API_SECRET не задан или слишком короткий")
        все_ок = False
    if not os.path.exists(".env"):
        log.warning("⚠️ .env файл не найден (используем переменные окружения)")
    if все_ок: log.info("✅ Этап 1 — ПРОЙДЕН")

    log.info("\n▶ Этап 2: Подключение к бирже...")
    try:
        b = exchange.fetch_balance({"type": "linear"})
        usdt_free = float(b.get("USDT", {}).get("free", 0))
        usdt_total = float(b.get("USDT", {}).get("total", 0))
        log.info(f"Подключение OK | Свободно: {usdt_free:.4f} USDT | Всего: {usdt_total:.4f} USDT")
        if usdt_free < MIN_BALANCE:
            log.warning(f"⚠️ Свободный баланс {usdt_free:.2f} < {MIN_BALANCE} USDT (возможны открытые позиции)")
        log.info("✅ Этап 2 — ПРОЙДЕН")
    except Exception as e:
        log.error(f"❌ Ошибка подключения: {e}")
        все_ок = False

    log.info("\n▶ Этап 3: Конфигурация...")
    rr = TP_PERCENT / SL_PERCENT
    if rr < 2.0:
        log.error(f"❌ RR {rr:.1f}:1 < 2:1")
        все_ок = False
    if MIN_SCORE < 65:
        log.error(f"❌ MIN_SCORE={MIN_SCORE} < 65")
        все_ок = False
    log.info(f"Конфигурация: TP={TP_PERCENT}% | SL={SL_PERCENT}% | RR={rr:.1f}:1")
    log.info(f"  MIN_SCORE={MIN_SCORE} | ENTRY_CONFIRM_MIN_SCORE={ENTRY_CONFIRM_MIN_SCORE} ✅")
    log.info("✅ Этап 3 — ПРОЙДЕН")

    log.info("\n▶ Этап 4: Доступность рынка...")
    доступные = 0
    for sym in SYMBOLS[:5]:
        try:
            ticker = exchange.fetch_ticker(sym)
            if float(ticker["last"]) > 0: доступные += 1
        except: pass
    log.info(f"Рынок: {доступные}/5 тестовых пар доступны")
    if доступные == 0:
        log.error("❌ Ни одна тестовая пара не доступна")
        все_ок = False
    else:
        log.info("✅ Этап 4 — ПРОЙДЕН")

    log.info("\n▶ Этап 5: Открытые позиции...")
    проверить_и_подхватить_позиции()
    история = загрузить_историю()
    log.info(f"История: {len(история)} сделок в базе")
    log.info(f"  Лимит попыток на монету: {SYMBOL_MAX_FAIL_ATTEMPTS} | Блок: {SYMBOL_BLOCK_AFTER_FAIL} мин")
    log.info("✅ Этап 5 — ПРОЙДЕН")

    log.info("")
    log.info("=" * 65)
    if все_ок:
        log.info("✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ — БОТ ЗАПУСКАЕТСЯ")
    else:
        log.error("❌ КРИТИЧЕСКИЕ ОШИБКИ — ПРОВЕРЬТЕ .env И ПАРАМЕТРЫ")
    log.info("=" * 65)
    log.info("")
    return все_ок

# ============================================================
#                     ГЛАВНЫЙ ЦИКЛ
# ============================================================

def main():
    global stats

    if not запустить_предстартовую_проверку():
        log.error("🛑 Бот остановлен из-за ошибок предстартовой проверки.")
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
    log.info("🤖 ГИБРИДНЫЙ ФЬЮЧЕРСНЫЙ БОТ v10.3")
    log.info(f"  Плечо: {LEVERAGE}x | RR: {TP_PERCENT}/{SL_PERCENT} ({TP_PERCENT/SL_PERCENT:.1f}:1)")
    log.info(f"  Баланс: {баланс_сейчас:.4f} USDT (полный)")
    log.info(f"  Свободно: {баланс_usdt():.4f} USDT")
    log.info(f"  MIN_SCORE: {MIN_SCORE} | ENTRY_CONFIRM_MIN_SCORE: {ENTRY_CONFIRM_MIN_SCORE}")
    log.info(f"  Пар: {len(SYMBOLS)}")
    log.info(f"  Квантовый анализ: {'ВКЛ' if QUANT_ENABLED else 'ВЫКЛ'}")
    log.info(f"  Order Flow: {'ВКЛ' if ORDER_FLOW_ENABLED else 'ВЫКЛ'}")
    log.info(f"  Лимит попыток на монету: {SYMBOL_MAX_FAIL_ATTEMPTS} → блок {SYMBOL_BLOCK_AFTER_FAIL} мин")
    log.info("=" * 65)
    log.info("")

    # Заблокированные символы: {symbol: timestamp_разблокировки}
    заблокированные: Dict[str, float] = {"ATOM/USDT:USDT": time.time() + 24 * 3600}  # Блокировка ATOM на 24 часа

    # Счётчик проваленных входов: {symbol: количество_провалов}
    fail_attempts: Dict[str, int] = {}

    while True:
        try:
            if time.time() - stats["последний_отчёт"] >= REPORT_INTERVAL:
                печатать_отчёт()

            баланс = полный_баланс_usdt()
            свободный = баланс_usdt()
            обновить_начало_дня(баланс)

            if свободный < MIN_BALANCE:
                активные = получить_позиции()
                if активные:
                    log.info(f"⏳ Открытые позиции: {[p['symbol'] for p in активные]} — ждём")
                    time.sleep(60)
                    continue
                log.warning(f"🛑 Свободный баланс {свободный:.2f} < {MIN_BALANCE} USDT. Пауза 10 мин.")
                time.sleep(600)
                continue

            if stats["депозит_старт"] > 0:
                просадка = (stats["депозит_старт"] - баланс) / stats["депозит_старт"] * 100
                if просадка > MAX_DRAWDOWN_PCT:
                    log.warning(f"⛔ Просадка {просадка:.1f}% > {MAX_DRAWDOWN_PCT}%. Пауза 2ч.")
                    time.sleep(7200)
                    continue

            if превышен_дневной_лимит():
                log.warning(f"⛔ Дневной лимит. Пауза {DAILY_LOSS_PAUSE_SEC//60} мин.")
                time.sleep(DAILY_LOSS_PAUSE_SEC)
                continue

            if not торговля_разрешена_по_времени():
                log.info("🕐 Заблокировано по времени. Пауза 5 мин.")
                time.sleep(300)
                continue

            if stats.get("sl_streak", 0) >= SL_STREAK_LIMIT:
                log.warning(f"🧊 {SL_STREAK_LIMIT} SL подряд — cooldown {SL_STREAK_PAUSE//60} мин.")
                stats["sl_streak"] = 0
                сохранить_состояние()
                time.sleep(SL_STREAK_PAUSE + SL_STREAK_EXTRA_PAUSE)
                continue

            активные = получить_позиции()
            if активные:
                log.info(f"⏳ Открытые позиции: {[p['symbol'] for p in активные]} — ждём")
                time.sleep(60)
                continue

            # Проверка Bybit ratio для блокировки шортов
            ai_market = получить_bybit_ai("BTC/USDT:USDT")
            if ai_market.get("signal") == "bullish" and ai_market.get("long_ratio", 0) > 0.62:
                log.info("⛔ Рынок бычий — шорты заблокированы")
                time.sleep(SCAN_INTERVAL)
                continue

            log.info(f"── Сканирование {len(SYMBOLS)} пар (баланс={свободный:.2f}U | порог={MIN_SCORE}) ──")
            scores = {}

            # Сканирование лонгов
            for sym in SYMBOLS:
                if sym in заблокированные:
                    if time.time() < заблокированные[sym]: continue
                    else:
                        del заблокированные[sym]
                        fail_attempts.pop(sym, None)

                if not тренд_4h_бычий(sym): continue
                time.sleep(API_CALL_DELAY)

                res = получить_скор(sym)
                ai_score = применить_ai_корректировку(res["score"], sym)
                res["score_final"] = ai_score
                scores[sym] = res
                det = res.get("details", {})
                log.debug(f"{sym.split(':')[0]:12s} скор={ai_score:3.0f}/100 rsi={det.get('rsi', '?')} rf={det.get('range_filter', '?')} st={det.get('supertrend', '?')}")

            кандидаты = sorted(
                [(s, d) for s, d in scores.items() if d.get("score_final", 0) >= MIN_SCORE],
                key=lambda x: x[1]["score_final"], reverse=True
            )[:5]

            выбрана, фин_скор, цена, sr_info, side = None, 0, 0.0, {}, "long"

            for лучшая, данные in кандидаты:
                фин_скор = данные["score_final"]
                цена = данные["price"]
                sr_info = данные.get("sr", {})
                det = данные.get("details", {})

                if sr_info.get("near_resistance") and sr_info.get("dist_to_res_pct", 99) < SR_BLOCK_DIST_PCT:
                    log.info(f"⛔ {лучшая.split(':')[0]}: сопротивление {sr_info.get('dist_to_res_pct',0):.2f}% — пропуск")
                    continue

                rsi_val = float(det.get("rsi", 50) or 50)
                if rsi_val > 65 and not sr_info.get("near_support"):
                    log.info(f"⚠️ {лучшая.split(':')[0]}: RSI={rsi_val:.1f} перекуплен — пропуск")
                    continue

                if MA_CROSSOVER_ENABLED and not det.get("ma_cross", True): continue
                if not det.get("vol_spike_ok", True): continue

                выбрана = лучшая
                log.info(f"► Выбрана {лучшая.split(':')[0]} (лонг) скор={фин_скор} цена={цена:.8f}")
                break

            # Поиск шортов (если не найден лонг)
            if выбрана is None:
                for sym in SYMBOLS:
                    if sym in заблокированные: continue
                    if тренд_4h_медвежий(sym):
                        time.sleep(API_CALL_DELAY)
                        short_res = получить_скор_шорта(sym)
                        ai_score = применить_ai_корректировку_шорт(short_res["score"], sym)
                        short_res["score_final"] = ai_score
                        if ai_score >= MIN_SCORE:
                            det_sh = short_res.get("details", {})
                            if MA_CROSSOVER_ENABLED and not det_sh.get("ma_cross", True): continue
                            if not det_sh.get("vol_spike_ok", True): continue
                            log.info(f"🐻 Шорт-кандидат: {sym.split(':')[0]} скор={ai_score}")
                            выбрана = sym
                            фин_скор = ai_score
                            цена = short_res["price"]
                            sr_info = short_res.get("sr", {})
                            side = "short"
                            scores[sym] = short_res
                            break

            if выбрана is None:
                log.info(f"Нет кандидатов — ждём {SCAN_INTERVAL} сек")
                time.sleep(SCAN_INTERVAL)
                continue

            # Расчёт SL/TP
            atr_пт = 0.0
            try:
                raw_atr = safe_fetch_ohlcv(выбрана, TIMEFRAME_TA, limit=50)
                if len(raw_atr) >= 20:
                    df_atr = pd.DataFrame(raw_atr, columns=["ts","o","h","l","c","v"])
                    atr_пт = float(calc_atr(df_atr, 14).iloc[-1])
            except: pass

            if side == "long":
                sl_atr_dist = atr_пт * ATR_SL_MULT if atr_пт > 0 else цена * SL_PERCENT / 100
                sl_pct_dist = max(MIN_SL_PERCENT, min(MAX_SL_PERCENT, (sl_atr_dist / цена) * 100))
                sl_цена = цена * (1 - sl_pct_dist / 100)
                tp_atr_dist = atr_пт * ATR_TP_MULT if atr_пт > 0 else цена * TP_PERCENT / 100
                tp_pct_dist = max(TP_PERCENT, (tp_atr_dist / цена) * 100)
                tp_цена = цена * (1 + tp_pct_dist / 100)
                support = sr_info.get("support", sl_цена)
                if support < sl_цена and support > цена * 0.97:
                    sl_цена = support * 0.998
            else:
                sl_atr_dist = atr_пт * ATR_SL_MULT if atr_пт > 0 else цена * SL_PERCENT / 100
                sl_pct_dist = max(MIN_SL_PERCENT, min(MAX_SL_PERCENT, (sl_atr_dist / цена) * 100))
                sl_цена = цена * (1 + sl_pct_dist / 100)
                tp_atr_dist = atr_пт * ATR_TP_MULT if atr_пт > 0 else цена * TP_PERCENT / 100
                tp_pct_dist = max(TP_PERCENT, (tp_atr_dist / цена) * 100)
                tp_цена = цена * (1 - tp_pct_dist / 100)
                resistance = sr_info.get("resistance", sl_цена)
                if resistance > sl_цена and resistance < цена * 1.03:
                    sl_цена = resistance * 1.002

            sl_dist_pct = abs(цена - sl_цена) / цена * 100
            real_rr = abs(tp_цена - цена) / abs(цена - sl_цена) if abs(цена - sl_цена) > 0 else 0
            log.info(f"📐 ATR={atr_пт/цена*100:.2f}% SL={sl_dist_pct:.2f}% RR={real_rr:.1f}:1")

            if real_rr < 2.0:
                log.warning(f"⛔ RR={real_rr:.1f}:1 < 2:1 — пропуск {выбрана.split(':')[0]}")
                time.sleep(SCAN_INTERVAL)
                continue

            margin = рассчитать_размер_позиции(фин_скор, свободный, sl_dist_pct)

            if свободный < margin * 1.1:
                log.warning(f"⚠️ Баланс {свободный:.2f} < маржа {margin:.2f} — уменьшаем")
                margin = свободный * 0.8

            log.info(f"✅ ВХОД {side.upper()}: скор={фин_скор} | SL={sl_цена:.8f} | TP={tp_цена:.8f} | маржа={margin:.2f}U")

            # Подтверждение входа
            if ENTRY_CONFIRM_BARS > 0:
                if not подтвердить_вход(выбрана, фин_скор, side):
                    log.info(f"⛔ Вход в {выбрана} отменён по подтверждению")

                    fail_attempts[выбрана] = fail_attempts.get(выбрана, 0) + 1
                    текущий_счётчик = fail_attempts[выбрана]
                    log.warning(f"  ► {выбрана.split(':')[0]}: попытка {текущий_счётчик}/{SYMBOL_MAX_FAIL_ATTEMPTS}")
                    if текущий_счётчик >= SYMBOL_MAX_FAIL_ATTEMPTS:
                        заблокированные[выбрана] = time.time() + SYMBOL_BLOCK_AFTER_FAIL * 60
                        fail_attempts.pop(выбрана, None)
                        log.warning(f"  ⛔ {выбрана.split(':')[0]} заблокирован на {SYMBOL_BLOCK_AFTER_FAIL} мин")

                    time.sleep(30)
                    continue

            fail_attempts.pop(выбрана, None)

            баланс_до = полный_баланс_usdt()
            время_входа = time.time()
            вход_цена, кол_во = открыть_позицию(выбрана, margin, tp_цена, sl_цена, side)

            if вход_цена is None or кол_во is None:
                log.warning("Не удалось открыть позицию — пауза 30 сек")

                fail_attempts[выбрана] = fail_attempts.get(выбрана, 0) + 1
                текущий_счётчик = fail_attempts[выбрана]
                log.warning(f"  ► {выбрана.split(':')[0]}: попытка открытия {текущий_счётчик}/{SYMBOL_MAX_FAIL_ATTEMPTS}")
                if текущий_счётчик >= SYMBOL_MAX_FAIL_ATTEMPTS:
                    заблокированные[выбрана] = time.time() + SYMBOL_BLOCK_AFTER_FAIL * 60
                    fail_attempts.pop(выбрана, None)
                    log.warning(f"  ⛔ {выбрана.split(':')[0]} заблокирован на {SYMBOL_BLOCK_AFTER_FAIL} мин")

                time.sleep(30)
                continue

            fail_attempts.pop(выбрана, None)

            stats["сделок_всего"] += 1
            сохранить_состояние()

            результат = "sl"
            monitor_pnl = 0.0
            try:
                результат, monitor_pnl = мониторить_позицию(
                    выбрана, вход_цена, кол_во, время_входа, sl_цена, tp_цена, side
                )
            except Exception as monitor_err:
                log.error(f"💥 Краш мониторинга: {monitor_err}")
                закрыть_позицию_с_подтверждением(выбрана, кол_во, side)
                результат = "sl"

            time.sleep(3)
            баланс_после = полный_баланс_usdt()
            pnl_реальный = баланс_после - баланс_до
            длит_мин = (time.time() - время_входа) / 60

            if результат == "tp":
                stats["тейкпрофит"] += 1
                stats["прибыль_usdt"] += max(0, pnl_реальный)
                stats["sl_streak"] = 0
                log.info(f"✅ TP: прибыль ≈{pnl_реальный:+.4f} USDT")
                заблокированные[выбрана] = time.time() + SYMBOL_BLOCK_AFTER_TP * 60
            elif результат == "sl":
                stats["стоплосс"] += 1
                stats["убыток_usdt"] += abs(min(0, pnl_реальный))
                stats["sl_streak"] = stats.get("sl_streak", 0) + 1
                заблокированные[выбрана] = time.time() + SYMBOL_BLOCK_AFTER_SL * 60
                log.warning(f"❌ SL: убыток ≈{pnl_реальный:+.4f} USDT streak={stats['sl_streak']}/{SL_STREAK_LIMIT}")
            else:
                stats["таймаут"] += 1
                if pnl_реальный >= 0:
                    stats["прибыль_usdt"] += pnl_реальный
                else:
                    stats["убыток_usdt"] += abs(pnl_реальный)
                stats["sl_streak"] = 0
                заблокированные[выбрана] = time.time() + SYMBOL_BLOCK_AFTER_TP * 60
                log.warning(f"⏰ Таймаут: P&L ≈{pnl_реальный:+.4f} USDT")

            запись = {
                "id": stats["сделок_всего"],
                "время_входа": datetime.fromtimestamp(время_входа).strftime("%d.%m.%Y %H:%M:%S"),
                "время_выхода": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                "duration_min": round(длит_мин, 1),
                "symbol": выбрана,
                "side": side,
                "score": фин_скор,
                "entry_price": вход_цена,
                "sl_price": sl_цена,
                "tp_price": tp_цена,
                "sl_dist_pct": round(sl_dist_pct, 3),
                "margin_usdt": margin,
                "leverage": LEVERAGE,
                "результат": результат,
                "pnl_usdt": round(pnl_реальный, 4),
                "rr_ratio": round(real_rr, 2),
                "details": scores.get(выбрана, {}).get("details", {}),
            }
            сохранить_сделку(запись)
            обновить_статистику_индикаторов(запись)
            детальный_отчёт_сделки(запись)
            сохранить_состояние()

            log.info("Сделка завершена — пауза 60 сек")
            time.sleep(60)

        except Exception as e:
            log.error(f"Глобальная ошибка главного цикла: {e}", exc_info=True)
            time.sleep(60)


if __name__ == "__main__":
    main()
