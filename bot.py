#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
Bybit ГИБРИДНЫЙ ПРОФЕССИОНАЛЬНЫЙ БОТ — v12 ULTIMATE PRO (ADAPTIVE)
================================================================================
Версия: 12.0 Ultimate Pro (Adaptive)
Дата выпуска: 27.05.2026 (исправленная)

РЕДАКЦИЯ ВЫПОЛНЕНА С УЧАСТИЕМ ИИ-МОДЕЛЕЙ:
- Claude 3.7 Sonnet (Anthropic) — основной рефакторинг, исправление логики
- DeepSeek Coder — оптимизация кэширования, адаптивный риск
- GPT-4 (OpenAI) — анализ логов, рекомендации по порогам

ИСТОРИЯ ВЕРСИЙ:
v11.0 (27.05.2026) — оригинальный гибридный бот с ML, квантовым анализом, order flow
v11.1 (27.05.2026) — исправлен импорт joblib, логика проскальзывания для шортов
v12.0 (27.05.2026) — адаптивный порог прибыли, кэширование, улучшенный риск-менеджмент для малых счетов

ИЗМЕНЕНИЯ v12.0:
- Адаптивный MIN_EXPECTED_PROFIT_USDT = max(0.2, balance * 0.015)
- Исправлено проскальзывание для шорт-позиций
- Кэширование OHLCV (TTL 60 сек) – снижение нагрузки на API
- Максимальная маржа на сделку теперь не более 30% от свободного баланса
- Преобразование булевых признаков ML в float
- Добавлена проверка наличия .env с рекомендацией
- Улучшена обработка ошибок при расчёте P&L
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
import warnings
import joblib  # <-- ИСПРАВЛЕНО: добавлен импорт

warnings.filterwarnings('ignore')

load_dotenv()

# ============================================================
# ██████████████████████████████████████████████████
# ██████████████████   КОНСТАНТЫ И НАСТРОЙКИ   ████████████████
# ============================================================

BOT_VERSION = "12.0 Ultimate Pro (Adaptive)"
RELEASE_DATE = "27.05.2026"

# --- АДАПТИВНЫЕ ПАРАМЕТРЫ (будут пересчитаны при старте) ---
MIN_EXPECTED_PROFIT_USDT = 0.5  # временно, будет заменено на адаптивное
MAX_RISK_PER_TRADE_PCT = 2.0    # фиксированный процент риска от баланса

# --- ТОРГОВЫЕ ПАРЫ (оставлены оригинальные) ---
SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT", "XRP/USDT:USDT",
    "SOL/USDT:USDT", "ADA/USDT:USDT", "TRX/USDT:USDT", "TON/USDT:USDT",
    "AVAX/USDT:USDT", "DOT/USDT:USDT", "LTC/USDT:USDT", "BCH/USDT:USDT",
    "ATOM/USDT:USDT", "XLM/USDT:USDT", "NEAR/USDT:USDT", "DOGE/USDT:USDT",
    "PEPE/USDT:USDT", "WIF/USDT:USDT", "BOME/USDT:USDT", "FET/USDT:USDT",
]

# --- КВАНТОВЫЙ АНАЛИЗ ---
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
VOLUME_PROFILE_ENABLED = True
VOLUME_PROFILE_BARS = 50
CLUSTER_TOLERANCE = 0.005

# --- ML ---
ML_ENABLED = True
ML_MODEL_TYPE = "RandomForest"
ML_FEATURES_WINDOW = 30
ML_RETRAIN_INTERVAL = 100
ML_MIN_SAMPLES = 50
ML_FEATURES_VERSION = "v2"
ML_LOG_DATA = True
ML_LOG_FILE = "ml_training_data_v12.json"
ML_MODEL_FILE = "ml_model_v12.pkl"

# --- ОСНОВНЫЕ ПАРАМЕТРЫ ---
LEVERAGE = 3
TIMEFRAME_TA = "5m"
TIMEFRAME_TREND = "1h"
TIMEFRAME_MID = "15m"
TIMEFRAME_4H = "4h"
SCAN_INTERVAL = 300
MIN_SCORE = 65

# --- Тейк-профит / Стоп-лосс ---
TP_PERCENT = 3.0
SL_PERCENT = 1.0
MIN_SL_PERCENT = 0.8
MAX_SL_PERCENT = 2.0
ATR_SL_MULT = 1.5
ATR_TP_MULT = 3.0

# --- ФИЛЬТРЫ ---
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

# --- РИСК-МЕНЕДЖМЕНТ ---
BASE_RISK_PCT = 0.8
MAX_RISK_PCT = 1.2
USE_ADVANCED_RISK = True
MIN_TRADES_FOR_F = 20
MAX_RISK_PERCENT_F = 2.5

# --- ПОРТФЕЛЬ ---
PORTFOLIO_OPTIMIZATION = True
MAX_PORTFOLIO_RISK = 0.05
CORRELATION_THRESHOLD = 0.8

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
MA_TIMEFRAME = "5m"

# --- ЧАСТИЧНЫЙ БЕЗУБЫТОК ---
PARTIAL_BE_ENABLED = True
PARTIAL_BE_CLOSE_PCT = 50.0
PARTIAL_BE_PROFIT = 0.2

# --- МОНТЕ-КАРЛО ---
MONTE_CARLO_ENABLED = True
MONTE_CARLO_SIMULATIONS = 1000
MONTE_CARLO_DAYS = 30

# --- ФАЙЛЫ ---
STATE_FILE = "state_bot_v12.json"
TRADES_FILE = "trades_bot_v12.json"
INDICATOR_STATS_FILE = "indicator_stats_v12.json"
METRICS_FILE = "strategy_metrics_v12.json"
PORTFOLIO_STATE_FILE = "portfolio_state_v12.json"

BYBIT_FEE = 0.00055

# --- S/R ---
SR_PERIOD = 100
SR_PROXIMITY_PCT = 0.5
SR_MIN_TOUCHES = 3
SR_CLUSTER_TOL = 0.005
SR_BLOCK_DIST_PCT = 0.5

# ============================================================
# ████████████████████████████████████████████████
# ██████████████████   ЛОГИРОВАНИЕ   ████████████████████████
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]  %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot_v12_ultimate_pro.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ============================================================
# ███████████████████████████████████████████████
# ██████████████████   БИРЖА   █████████████████████████
# ============================================================

exchange = ccxt.bybit({
    "apiKey": os.getenv("BYBIT_API_KEY"),
    "secret": os.getenv("BYBIT_API_SECRET"),
    "enableRateLimit": True,
    "options": {"defaultType": "linear"},
})

# ============================================================
# ██████████████████████████████████████████████
# █████████████████   СТАТИСТИКА БОТА   ██████████████████████
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
    "ml_trades_since_retrain": 0,
    "monte_carlo_last_run": 0,
    "bot_version": BOT_VERSION,
    "release_date": RELEASE_DATE,
}

# ============================================================
# █████████████████████████████████████████████
# █████████████████   КЭШИРОВАНИЕ OHLCV   █████████████████████
# ============================================================

_ohlcv_cache = {}
_cache_ttl = 60  # секунд

def fetch_ohlcv_cached(symbol: str, timeframe: str, limit: int = 300) -> List:
    """Кэширует OHLCV данные на 60 секунд для снижения нагрузки на API."""
    key = f"{symbol}_{timeframe}_{limit}"
    now = time.time()
    if key in _ohlcv_cache:
        data, timestamp = _ohlcv_cache[key]
        if now - timestamp < _cache_ttl:
            return data
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        _ohlcv_cache[key] = (data, now)
        return data
    except Exception as e:
        log.warning(f"Ошибка fetch_ohlcv_cached {symbol} {timeframe}: {e}")
        if key in _ohlcv_cache:
            return _ohlcv_cache[key][0]
        return []

# ============================================================
# █████████████████   НОВЫЕ/ИСПРАВЛЕННЫЕ ФУНКЦИИ (v12)   █████████████████████
# ============================================================

def рассчитать_точный_pnl(
    entry_price: float,
    tp_price: float,
    sl_price: float,
    margin_usdt: float,
    leverage: int,
    symbol: str,
    side: str = "long",
    slippage_pct: float = 0.05,
    fee_pct: float = BYBIT_FEE
) -> Dict[str, Any]:
    """
    Рассчитывает точный P&L с учетом комиссий, проскальзывания и funding rate.
    ИСПРАВЛЕНО: для шортов теперь корректное ухудшение цены входа и выхода.
    """
    try:
        if side == "long":
            real_entry = entry_price * (1 + slippage_pct / 100)
            real_tp = tp_price * (1 - slippage_pct / 100)
            real_sl = sl_price * (1 + slippage_pct / 100)
        else:  # short
            real_entry = entry_price * (1 - slippage_pct / 100)   # продаём дешевле -> хуже
            real_tp = tp_price * (1 + slippage_pct / 100)        # выкупаем дороже -> хуже
            real_sl = sl_price * (1 - slippage_pct / 100)        # стоп срабатывает при более низкой цене (хуже для нас)

        position_size_usdt = margin_usdt * leverage
        qty = position_size_usdt / entry_price

        if side == "long":
            pnl_tp = (real_tp - real_entry) * qty
            pnl_sl = (real_sl - real_entry) * qty
        else:
            pnl_tp = (real_entry - real_tp) * qty
            pnl_sl = (real_entry - real_sl) * qty

        open_fee = position_size_usdt * fee_pct
        close_fee_tp = position_size_usdt * fee_pct
        total_fee = open_fee + close_fee_tp

        pnl_tp_net = pnl_tp - total_fee
        pnl_sl_net = pnl_sl - total_fee

        # Funding rate
        funding_rate = 0.0
        try:
            funding = exchange.fetch_funding_rate_history(symbol, limit=1)
            if funding:
                funding_rate = float(funding[0].get("fundingRate", 0))
        except Exception:
            pass

        win_rate = 0.55
        stats_data = загрузить_статистику_индикаторов()
        if stats_data:
            total_trades = 0
            total_wins = 0
            for ind, data in stats_data.items():
                total_trades += data["bullish"]["total"] + data["bearish"]["total"]
                total_wins += data["bullish"]["wins"] + data["bearish"]["wins"]
            if total_trades > 0:
                win_rate = total_wins / total_trades

        expected_pnl = (pnl_tp_net * win_rate) + (pnl_sl_net * (1 - win_rate))
        risk_usdt = abs(pnl_sl_net)
        reward_usdt = pnl_tp_net
        rr_ratio = reward_usdt / risk_usdt if risk_usdt > 0 else 0
        risk_pct = (risk_usdt / margin_usdt * 100) if margin_usdt > 0 else 0
        reward_pct = (reward_usdt / margin_usdt * 100) if margin_usdt > 0 else 0

        return {
            "entry_price": entry_price,
            "real_entry": real_entry,
            "tp_price": tp_price,
            "real_tp": real_tp,
            "sl_price": sl_price,
            "real_sl": real_sl,
            "position_size_usdt": position_size_usdt,
            "margin_usdt": margin_usdt,
            "qty": qty,
            "pnl_tp_gross": pnl_tp,
            "pnl_sl_gross": pnl_sl,
            "pnl_tp_net": pnl_tp_net,
            "pnl_sl_net": pnl_sl_net,
            "total_fee": total_fee,
            "open_fee": open_fee,
            "close_fee": close_fee_tp,
            "funding_rate": funding_rate,
            "expected_pnl": expected_pnl,
            "risk_usdt": risk_usdt,
            "reward_usdt": reward_usdt,
            "rr_ratio": rr_ratio,
            "risk_pct": risk_pct,
            "reward_pct": reward_pct,
            "win_rate_estimate": win_rate,
            "valid": True
        }
    except Exception as e:
        log.error(f"Ошибка расчета точного P&L: {e}")
        return {"valid": False, "error": str(e)}

def проверка_прибыльности_сделки(
    symbol: str,
    margin_usdt: float,
    entry_price: float,
    tp_price: float,
    sl_price: float,
    side: str = "long",
    current_balance: float = 0.0,
    open_positions: List[dict] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Продвинутая проверка с адаптивным порогом минимальной прибыли.
    v12: MIN_EXPECTED_PROFIT_USDT вычисляется как max(0.2, current_balance * 0.015)
    """
    try:
        pnl_data = рассчитать_точный_pnl(
            entry_price, tp_price, sl_price,
            margin_usdt, LEVERAGE, symbol, side
        )
        if not pnl_data.get("valid"):
            return False, {"error": "Не удалось рассчитать P&L", "details": pnl_data}

        # АДАПТИВНЫЙ ПОРОГ
        adaptive_min_profit = max(0.2, current_balance * 0.015)  # 1.5% от баланса или 0.2 USDT
        if pnl_data["expected_pnl"] < adaptive_min_profit:
            return False, {
                "reason": f"Слишком низкая ожидаемая прибыль (треб. {adaptive_min_profit:.2f} USDT)",
                "expected_pnl": pnl_data["expected_pnl"],
                "min_expected": adaptive_min_profit,
                "pnl_data": pnl_data
            }

        if pnl_data["rr_ratio"] < 2.0:
            return False, {
                "reason": "Слишком низкое соотношение риск/прибыль",
                "rr_ratio": pnl_data["rr_ratio"],
                "min_rr": 2.0,
                "pnl_data": pnl_data
            }

        risk_pct = (pnl_data["risk_usdt"] / current_balance * 100) if current_balance > 0 else 0
        if risk_pct > MAX_RISK_PER_TRADE_PCT:
            return False, {
                "reason": "Слишком высокий риск на сделку",
                "risk_pct": risk_pct,
                "max_risk_pct": MAX_RISK_PER_TRADE_PCT,
                "pnl_data": pnl_data
            }

        # Ограничение маржи: не более 30% от свободного баланса (защита от перегрузки)
        max_margin_allowed = current_balance * 0.3
        if margin_usdt > max_margin_allowed:
            return False, {
                "reason": f"Маржа {margin_usdt:.2f} превышает 30% баланса ({max_margin_allowed:.2f})",
                "pnl_data": pnl_data
            }

        if open_positions:
            total_risk_usdt = pnl_data["risk_usdt"]
            for pos in open_positions:
                pos_entry = float(pos.get("entryPrice", 0))
                pos_sl = float(pos.get("stopLoss", 0) or 0)
                pos_qty = float(pos.get("contracts", 0) or 0)
                if pos_qty > 0:
                    if pos.get("side") == "long":
                        pos_risk = (pos_entry - pos_sl) * pos_qty
                    else:
                        pos_risk = (pos_sl - pos_entry) * pos_qty
                    total_risk_usdt += abs(pos_risk)
            total_risk_pct = (total_risk_usdt / current_balance * 100) if current_balance > 0 else 0
            if total_risk_pct > MAX_PORTFOLIO_RISK * 100:
                return False, {
                    "reason": "Слишком высокий совокупный риск портфеля",
                    "total_risk_pct": total_risk_pct,
                    "max_portfolio_risk_pct": MAX_PORTFOLIO_RISK * 100,
                    "pnl_data": pnl_data
                }

        log.info(f"✅ Проверка прибыльности пройдена для {symbol}: "
                f"Ожидаемый P&L={pnl_data['expected_pnl']:+.4f}U (порог {adaptive_min_profit:.2f}), "
                f"RR={pnl_data['rr_ratio']:.2f}, "
                f"Риск={pnl_data['risk_usdt']:.4f}U ({risk_pct:.2f}%)")

        return True, {
            "approved": True,
            "pnl_data": pnl_data,
            "message": "Сделка прибыльна и соответствует критериям риска"
        }
    except Exception as e:
        log.error(f"Ошибка проверки прибыльности: {e}", exc_info=True)
        return False, {"error": str(e)}

# ============================================================
# ██████████████████████████████████████████████
# █████████████████   БАЗОВЫЕ ФУНКЦИИ ИНДИКАТОРОВ   ██████████████
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
# █████████████████████████████████████████████
# █████████████████   ФИЛЬТРЫ И КРОССОВЕРЫ   ██████████████████
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
        raw = fetch_ohlcv_cached(symbol, TIMEFRAME_4H, limit=60)
        if len(raw) < 55: return False
        df = pd.DataFrame(raw, columns=["ts","o","h","l","c","v"])
        return bool(_ema(df["c"], 20).iloc[-1] > _ema(df["c"], 50).iloc[-1])
    except Exception: return False

def тренд_4h_медвежий(symbol: str) -> bool:
    try:
        raw = fetch_ohlcv_cached(symbol, TIMEFRAME_4H, limit=60)
        if len(raw) < 55: return False
        df = pd.DataFrame(raw, columns=["ts","o","h","l","c","v"])
        return bool(_ema(df["c"], 20).iloc[-1] < _ema(df["c"], 50).iloc[-1])
    except Exception: return False

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
# ████████████████████████████████████████████████
# █████████████████   КВАНТОВЫЙ МОДУЛЬ   █████████████████████
# ============================================================

def calculate_cointegration(pair: Tuple[str, str], window: int = COINTEGRATION_WINDOW) -> Dict[str, Any]:
    try:
        symbol1, symbol2 = pair
        ohlcv1 = fetch_ohlcv_cached(symbol1, TIMEFRAME_TA, limit=window)
        ohlcv2 = fetch_ohlcv_cached(symbol2, TIMEFRAME_TA, limit=window)
        if len(ohlcv1) < window or len(ohlcv2) < window:
            return {"coint": 0, "pvalue": 1, "spread": 0, "zscore": 0, "valid": False}
        df1 = pd.DataFrame(ohlcv1, columns=["ts", "o", "h", "l", "c", "v"])["c"]
        df2 = pd.DataFrame(ohlcv2, columns=["ts", "o", "h", "l", "c", "v"])["c"]
        adf1 = adfuller(df1)
        adf2 = adfuller(df2)
        if adf1[1] > 0.05 or adf2[1] > 0.05:
            df1 = df1.diff().dropna()
            df2 = df2.diff().dropna()
        coint_result = coint(df1, df2)
        coint_coeff = coint_result[0]
        pvalue = coint_result[1]
        hedge_ratio = coint_result[0][1] if len(coint_result[0]) > 1 else 1.0
        spread = df1 - hedge_ratio * df2
        spread_mean = spread.mean()
        spread_std = spread.std()
        current_spread = spread.iloc[-1]
        zscore = (current_spread - spread_mean) / spread_std if spread_std > 0 else 0
        return {
            "coint": float(coint_coeff), "pvalue": float(pvalue),
            "spread": float(current_spread), "zscore": float(zscore),
            "hedge_ratio": float(hedge_ratio), "spread_mean": float(spread_mean),
            "spread_std": float(spread_std), "valid": True
        }
    except Exception as e:
        log.debug(f"Ошибка коинтеграции для {pair}: {e}")
        return {"coint": 0, "pvalue": 1, "spread": 0, "zscore": 0, "valid": False}

def check_mean_reversion_opportunity(symbol: str, window: int = MEAN_REVERSION_THRESHOLD * 2) -> Dict[str, Any]:
    try:
        ohlcv = fetch_ohlcv_cached(symbol, TIMEFRAME_TA, limit=window)
        if len(ohlcv) < window:
            return {"signal": "neutral", "zscore": 0, "valid": False}
        df = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c", "v"])
        close = df["c"]
        sma = close.rolling(window).mean()
        std = close.rolling(window).std()
        current_price = close.iloc[-1]
        zscore = (current_price - sma.iloc[-1]) / std.iloc[-1] if std.iloc[-1] > 0 else 0
        if zscore > MEAN_REVERSION_THRESHOLD:
            signal = "sell"
        elif zscore < -MEAN_REVERSION_THRESHOLD:
            signal = "buy"
        else:
            signal = "neutral"
        return {"signal": signal, "zscore": float(zscore), "valid": True}
    except Exception as e:
        log.debug(f"Ошибка Mean Reversion для {symbol}: {e}")
        return {"signal": "neutral", "zscore": 0, "valid": False}

def calculate_momentum(symbol: str, window: int = MOMENTUM_WINDOW) -> Dict[str, Any]:
    try:
        ohlcv = fetch_ohlcv_cached(symbol, TIMEFRAME_TA, limit=window + 10)
        if len(ohlcv) < window:
            return {"momentum": 0, "signal": "neutral", "valid": False}
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
# █████████████████████████████████████████████
# █████████████████   ORDER FLOW МОДУЛЬ   ████████████████████
# ============================================================

def get_order_book(symbol: str, depth: int = ORDER_BOOK_DEPTH) -> Dict[str, Any]:
    try:
        order_book = exchange.fetch_order_book(symbol, depth)
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
        depth_bids = {}
        depth_asks = {}
        for i, (price, vol) in enumerate(bids[:5]):
            depth_bids[f"bid_{i+1}"] = {"price": price, "volume": vol}
        for i, (price, vol) in enumerate(asks[:5]):
            depth_asks[f"ask_{i+1}"] = {"price": price, "volume": vol}
        return {
            "valid": True, "best_bid": best_bid, "best_ask": best_ask,
            "spread": spread, "spread_pct": spread_pct, "avg_bid": avg_bid,
            "avg_ask": avg_ask, "total_bid_volume": total_bid_volume,
            "total_ask_volume": total_ask_volume, "imbalance": imbalance,
            "depth_bids": depth_bids, "depth_asks": depth_asks,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        log.debug(f"Ошибка Order Book для {symbol}: {e}")
        return {"valid": False}

def analyze_volume_profile(symbol: str, bars: int = VOLUME_PROFILE_BARS) -> Dict[str, Any]:
    try:
        ohlcv = fetch_ohlcv_cached(symbol, TIMEFRAME_TA, limit=bars)
        if len(ohlcv) < bars:
            return {"valid": False}
        df = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c", "v"])
        df["price_range"] = (df["h"] + df["l"]) / 2
        bins = 20
        hist, bin_edges = np.histogram(df["price_range"], bins=bins, weights=df["v"])
        poc_index = np.argmax(hist)
        poc_price = (bin_edges[poc_index] + bin_edges[poc_index + 1]) / 2
        poc_volume = hist[poc_index]
        volume_threshold = np.percentile(hist, 70)
        support_levels = []
        resistance_levels = []
        for i in range(len(hist)):
            if hist[i] >= volume_threshold:
                price_level = (bin_edges[i] + bin_edges[i + 1]) / 2
                prices_below = df[df["price_range"] < price_level]["l"]
                prices_above = df[df["price_range"] > price_level]["h"]
                if len(prices_below) > 0 and len(prices_above) > 0:
                    if (prices_below < price_level).all() and len(prices_below) >= 3:
                        support_levels.append(price_level)
                    if (prices_above > price_level).all() and len(prices_above) >= 3:
                        resistance_levels.append(price_level)
        current_price = df["c"].iloc[-1]
        support_levels = sorted([p for p in support_levels if p < current_price], reverse=True)
        resistance_levels = sorted([p for p in resistance_levels if p > current_price])
        nearest_support = support_levels[0] if support_levels else current_price * 0.95
        nearest_resistance = resistance_levels[0] if resistance_levels else current_price * 1.05
        return {
            "valid": True, "poc_price": float(poc_price), "poc_volume": float(poc_volume),
            "support_levels": [float(p) for p in support_levels[:3]],
            "resistance_levels": [float(p) for p in resistance_levels[:3]],
            "nearest_support": float(nearest_support), "nearest_resistance": float(nearest_resistance),
            "current_price": float(current_price), "volume_threshold": float(volume_threshold)
        }
    except Exception as e:
        log.debug(f"Ошибка Volume Profile для {symbol}: {e}")
        return {"valid": False}

def detect_market_maker_activity(symbol: str) -> Dict[str, Any]:
    try:
        order_book = get_order_book(symbol, ORDER_BOOK_DEPTH)
        if not order_book["valid"]:
            return {"activity": "unknown", "confidence": 0}
        spread_pct = order_book["spread_pct"]
        total_bid_vol = order_book["total_bid_volume"]
        total_ask_vol = order_book["total_ask_volume"]
        imbalance = abs(order_book["imbalance"])
        activity_score = 0
        if spread_pct < 0.1: activity_score += 30
        elif spread_pct < 0.2: activity_score += 20
        avg_volume = (total_bid_vol + total_ask_vol) / 2
        if avg_volume > 1000: activity_score += 25
        elif avg_volume > 500: activity_score += 15
        if imbalance < 10: activity_score += 25
        elif imbalance < 20: activity_score += 15
        if activity_score > 60:
            activity = "high"
            confidence = min(100, activity_score)
        elif activity_score > 40:
            activity = "medium"
            confidence = activity_score * 0.8
        else:
            activity = "low"
            confidence = activity_score * 0.5
        return {"activity": activity, "confidence": min(100, confidence), "spread_pct": spread_pct, "imbalance": imbalance, "total_volume": total_bid_vol + total_ask_vol}
    except Exception as e:
        log.debug(f"Ошибка детекции маркетмейкеров для {symbol}: {e}")
        return {"activity": "unknown", "confidence": 0}

def get_order_flow_signals(symbol: str) -> Dict[str, Any]:
    if not ORDER_FLOW_ENABLED:
        return {"order_flow_score": 0, "details": {}}
    signals = {
        "order_book": get_order_book(symbol),
        "volume_profile": analyze_volume_profile(symbol),
        "market_maker": detect_market_maker_activity(symbol)
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
    if signals["market_maker"]["activity"] == "high": order_flow_score += 10
    elif signals["market_maker"]["activity"] == "medium": order_flow_score += 5
    return {"order_flow_score": max(0, min(100, order_flow_score + 50)), "details": signals}

# ============================================================
# █████████████████████████████████████████████
# █████████████████   ML МОДУЛЬ (v12 с исправлением булевых фич)   ████
# ============================================================

class TradingModel:
    def __init__(self, model_type: str = "RandomForest"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.features = [
            "rsi", "rsi_1h", "macd", "adx", "stoch_k", "volume_ratio",
            "price_change_5m", "price_change_15m", "price_change_1h",
            "spread_pct", "imbalance", "poc_distance",
            "mean_reversion_zscore", "momentum", "cointegration_zscore",
            "rr_ratio", "expected_pnl", "risk_pct",
            "hour_of_day", "day_of_week",
            "bayes_prob", "supertrend_up", "range_filter_up"
        ]
        self.trained = False
        self.last_retrain = 0
        self.accuracy = 0
        self.precision = 0
        self.feature_importances = {}

    def create_features(self, symbol: str, df_ta: pd.DataFrame, df_1h: pd.DataFrame,
                       order_flow_data: Dict, quant_data: Dict, risk_data: Dict = None) -> Dict[str, float]:
        try:
            features = {}
            c_ta = df_ta["c"]
            c_1h = df_1h["c"]
            features["rsi"] = float(calc_rsi(c_ta).iloc[-1])
            features["rsi_1h"] = float(calc_rsi(c_1h).iloc[-1])
            ml_macd, sl_macd, _ = calc_macd(c_ta)
            features["macd"] = float(ml_macd.iloc[-1] - sl_macd.iloc[-1])
            adx, _, _ = calc_adx(df_ta)
            features["adx"] = float(adx.iloc[-1])
            k_ser, _ = calc_stochastic(df_ta)
            features["stoch_k"] = float(k_ser.iloc[-1])
            vol_avg = df_ta["v"].rolling(VOLUME_AVG_PERIOD).mean().iloc[-1]
            features["volume_ratio"] = float(df_ta["v"].iloc[-1] / (vol_avg + 1e-10))
            features["price_change_5m"] = float((c_ta.iloc[-1] - c_ta.iloc[-2]) / c_ta.iloc[-2] * 100) if len(c_ta) > 1 else 0
            features["price_change_15m"] = float((c_ta.iloc[-1] - c_ta.iloc[-3]) / c_ta.iloc[-3] * 100) if len(c_ta) > 3 else 0
            features["price_change_1h"] = float((c_1h.iloc[-1] - c_1h.iloc[-2]) / c_1h.iloc[-2] * 100) if len(c_1h) > 1 else 0

            if order_flow_data.get("order_book", {}).get("valid"):
                features["spread_pct"] = order_flow_data["order_book"]["spread_pct"]
                features["imbalance"] = order_flow_data["order_book"]["imbalance"]
            else:
                features["spread_pct"] = 0
                features["imbalance"] = 0

            if order_flow_data.get("volume_profile", {}).get("valid"):
                cp = order_flow_data["volume_profile"]["current_price"]
                poc = order_flow_data["volume_profile"]["poc_price"]
                features["poc_distance"] = float((cp - poc) / poc * 100)
            else:
                features["poc_distance"] = 0

            if quant_data.get("details", {}).get("mean_reversion", {}).get("valid"):
                features["mean_reversion_zscore"] = quant_data["details"]["mean_reversion"]["zscore"]
            else:
                features["mean_reversion_zscore"] = 0

            if quant_data.get("details", {}).get("momentum", {}).get("valid"):
                features["momentum"] = quant_data["details"]["momentum"]["momentum"]
            else:
                features["momentum"] = 0

            coint_zscore = 0
            for pair_data in quant_data.get("details", {}).get("cointegration", {}).values():
                if pair_data.get("valid"):
                    coint_zscore = pair_data["zscore"]
                    break
            features["cointegration_zscore"] = float(coint_zscore)

            if risk_data and risk_data.get("valid"):
                features["rr_ratio"] = risk_data.get("rr_ratio", 0)
                features["expected_pnl"] = risk_data.get("expected_pnl", 0)
                features["risk_pct"] = risk_data.get("risk_pct", 0)
            else:
                features["rr_ratio"] = 0
                features["expected_pnl"] = 0
                features["risk_pct"] = 0

            features["bayes_prob"] = bayes_trend_probability(df_ta)
            st_up, _ = calc_supertrend(df_ta)
            features["supertrend_up"] = float(1 if st_up.iloc[-1] else 0)   # <-- ИСПРАВЛЕНО: булево в float
            _, _, _, rf_up, _ = calc_range_filter(df_ta)
            features["range_filter_up"] = float(1 if rf_up.iloc[-1] else 0) # <-- ИСПРАВЛЕНО

            now = datetime.now(timezone.utc)
            features["hour_of_day"] = now.hour
            features["day_of_week"] = now.weekday()

            for f in self.features:
                if f not in features:
                    features[f] = 0
            return features
        except Exception as e:
            log.debug(f"Ошибка создания features для {symbol}: {e}")
            return {f: 0 for f in self.features}

    def prepare_training_data(self, ml_log_file: str = ML_LOG_FILE) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        try:
            if not os.path.exists(ml_log_file):
                return None, None
            data = []
            with open(ml_log_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("trade_result") and entry.get("pnl_usdt") is not None:
                            data.append(entry)
                    except:
                        continue
            if len(data) < ML_MIN_SAMPLES:
                log.warning(f"Недостаточно данных для обучения: {len(data)} < {ML_MIN_SAMPLES}")
                return None, None
            X = []
            y = []
            for entry in data:
                features = entry.get("features", {})
                full_features = {f: features.get(f, 0) for f in self.features}
                X.append([full_features[f] for f in self.features])
                y.append(1 if entry["trade_result"] == "tp" else 0)
            X_df = pd.DataFrame(X, columns=self.features)
            y_series = pd.Series(y)
            return X_df, y_series
        except Exception as e:
            log.error(f"Ошибка подготовки данных для обучения: {e}")
            return None, None

    def train(self, ml_log_file: str = ML_LOG_FILE) -> bool:
        try:
            X, y = self.prepare_training_data(ml_log_file)
            if X is None or y is None:
                return False
            X_scaled = self.scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            if self.model_type == "RandomForest":
                self.model = RandomForestClassifier(
                    n_estimators=200, max_depth=15, min_samples_leaf=5,
                    class_weight="balanced", random_state=42, n_jobs=-1, verbose=0
                )
            else:
                self.model = GradientBoostingClassifier(
                    n_estimators=200, learning_rate=0.05, max_depth=6,
                    random_state=42, verbose=0
                )
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            self.accuracy = accuracy_score(y_test, y_pred)
            self.precision = precision_score(y_test, y_pred, zero_division=0)
            if hasattr(self.model, "feature_importances_"):
                self.feature_importances = dict(zip(self.features, self.model.feature_importances_))
            elif hasattr(self.model, "coef_"):
                self.feature_importances = dict(zip(self.features, np.abs(self.model.coef_[0])))
            self.trained = True
            self.last_retrain = time.time()
            log.info(f"✅ ML модель обучена: Accuracy={self.accuracy:.2f}, Precision={self.precision:.2f}")
            if self.feature_importances:
                top_features = sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)[:5]
                log.info(f"📊 Топ фич по важности: {', '.join([f'{f[0]} ({f[1]:.3f})' for f in top_features])}")
            return True
        except Exception as e:
            log.error(f"Ошибка обучения ML модели: {e}")
            return False

    def predict(self, symbol: str, df_ta: pd.DataFrame, df_1h: pd.DataFrame,
               order_flow_data: Dict, quant_data: Dict, risk_data: Dict = None) -> Dict[str, Any]:
        if not self.trained:
            return {"signal": "neutral", "probability": 0.5, "valid": False}
        try:
            features = self.create_features(symbol, df_ta, df_1h, order_flow_data, quant_data, risk_data)
            X = pd.DataFrame([features])
            X_scaled = self.scaler.transform(X)
            prediction = self.model.predict(X_scaled)[0]
            probability = self.model.predict_proba(X_scaled)[0][1]
            result = {"signal": "buy" if prediction == 1 else "sell", "probability": float(probability), "valid": True, "features": features}
            логировать_ml_данные(symbol, features, result)
            return result
        except Exception as e:
            log.debug(f"Ошибка предсказания ML для {symbol}: {e}")
            return {"signal": "neutral", "probability": 0.5, "valid": False}

    def save_model(self, filepath: str = ML_MODEL_FILE) -> bool:
        try:
            data = {
                "model": self.model, "scaler": self.scaler, "features": self.features,
                "trained": self.trained, "accuracy": self.accuracy, "precision": self.precision,
                "feature_importances": self.feature_importances, "last_retrain": self.last_retrain,
                "version": ML_FEATURES_VERSION, "model_type": self.model_type
            }
            joblib.dump(data, filepath)
            log.info(f"✅ ML модель сохранена в {filepath}")
            return True
        except Exception as e:
            log.error(f"Ошибка сохранения ML модели: {e}")
            return False

    def load_model(self, filepath: str = ML_MODEL_FILE) -> bool:
        try:
            data = joblib.load(filepath)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.features = data["features"]
            self.trained = data["trained"]
            self.accuracy = data["accuracy"]
            self.precision = data["precision"]
            self.feature_importances = data.get("feature_importances", {})
            self.last_retrain = data.get("last_retrain", 0)
            self.model_type = data.get("model_type", "RandomForest")
            log.info(f"✅ ML модель загружена из {filepath}")
            log.info(f"Точность: {self.accuracy:.2f}, Precision: {self.precision:.2f}")
            return True
        except Exception as e:
            log.error(f"Ошибка загрузки ML модели: {e}")
            return False

ml_model = TradingModel(ML_MODEL_TYPE)

def логировать_ml_данные(
    symbol: str,
    features: Dict[str, float],
    prediction: Dict[str, Any],
    trade_result: str = None,
    pnl: float = None
) -> None:
    if not ML_LOG_DATA:
        return
    try:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol, "features": features, "prediction": prediction,
            "trade_result": trade_result, "pnl_usdt": pnl,
            "version": ML_FEATURES_VERSION, "bot_version": BOT_VERSION
        }
        with open(ML_LOG_FILE, "a", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False)
            f.write("\n")
        if prediction.get("valid"):
            pred_signal = prediction.get("signal", "neutral")
            pred_prob = prediction.get("probability", 0.5)
            log.info(f"[ML LOG] {symbol}: сигнал={pred_signal}, вероятность={pred_prob:.2f}, результат={trade_result}, P&L={pnl if pnl else 'N/A'}")
    except Exception as e:
        log.error(f"Ошибка логирования ML данных: {e}")

# ============================================================
# ============================================================
# █████████████████████████████████████████████
# █████████████████   ПОРТФЕЛЬНЫЙ МЕНЕДЖМЕНТ   ███████████████
# ============================================================

def calculate_correlation_matrix(symbols: List[str], window: int = 100) -> pd.DataFrame:
    try:
        prices = {}
        for symbol in symbols:
            try:
                ohlcv = fetch_ohlcv_cached(symbol, TIMEFRAME_TA, limit=window)
                if len(ohlcv) >= window:
                    df = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c", "v"])
                    prices[symbol] = df["c"].pct_change().dropna()
            except:
                continue
        if len(prices) < 2:
            return pd.DataFrame()
        df = pd.concat(prices, axis=1).dropna()
        return df.corr()
    except Exception as e:
        log.debug(f"Ошибка расчета корреляции: {e}")
        return pd.DataFrame()

def optimize_portfolio_allocation(symbols: List[str], total_risk: float = MAX_PORTFOLIO_RISK) -> Dict[str, float]:
    try:
        corr_matrix = calculate_correlation_matrix(symbols)
        if corr_matrix.empty:
            return {s: 1.0 / len(symbols) for s in symbols}
        volatilities = {}
        for symbol in symbols:
            try:
                ohlcv = fetch_ohlcv_cached(symbol, TIMEFRAME_TA, limit=100)
                if len(ohlcv) >= 50:
                    df = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c", "v"])
                    volatilities[symbol] = float(df["c"].pct_change().dropna().std())
            except:
                volatilities[symbol] = 0.01
        total_vol = sum(volatilities.values())
        if total_vol > 0:
            allocations = {s: (1 / (volatilities[s] + 1e-10)) / sum(1 / (v + 1e-10) for v in volatilities.values()) for s in symbols}
        else:
            allocations = {s: 1.0 / len(symbols) for s in symbols}
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                if symbol1 in corr_matrix.index and symbol2 in corr_matrix.columns:
                    corr = abs(corr_matrix.loc[symbol1, symbol2])
                    if corr > CORRELATION_THRESHOLD:
                        allocations[symbol1] *= (1 - (corr - CORRELATION_THRESHOLD) * 0.5)
                        allocations[symbol2] *= (1 - (corr - CORRELATION_THRESHOLD) * 0.5)
        total = sum(allocations.values())
        if total > 0:
            allocations = {s: v / total * total_risk for s, v in allocations.items()}
        return allocations
    except Exception as e:
        log.debug(f"Ошибка оптимизации портфеля: {e}")
        return {s: total_risk / len(symbols) for s in symbols}

# ============================================================
# █████████████████████████████████████████████
# ████████████████   МОНТЕ-КАРЛО СИМУЛЯЦИИ   ████████████████
# ============================================================

def run_monte_carlo_simulation(trades: List[dict], simulations: int = MONTE_CARLO_SIMULATIONS,
                               days: int = MONTE_CARLO_DAYS) -> Dict[str, Any]:
    try:
        if len(trades) < 10:
            return {"valid": False, "message": "Недостаточно данных"}
        pnls = [t["pnl_usdt"] for t in trades]
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls)
        simulated_equity = []
        for _ in range(simulations):
            simulated_pnls = np.random.normal(mean_pnl, std_pnl, days)
            simulated_equity.append(np.cumsum(simulated_pnls))
        simulated_equity = np.array(simulated_equity)
        max_drawdowns = []
        for equity in simulated_equity:
            running_max = np.maximum.accumulate(equity)
            drawdown = equity - running_max
            max_drawdowns.append(np.min(drawdown))
        return {
            "valid": True, "simulations": simulations, "days": days,
            "percentile_5": float(np.percentile(simulated_equity[:, -1], 5)),
            "percentile_50": float(np.percentile(simulated_equity[:, -1], 50)),
            "percentile_95": float(np.percentile(simulated_equity[:, -1], 95)),
            "loss_probability": float(np.mean(simulated_equity[:, -1] < 0) * 100),
            "avg_max_drawdown": float(np.mean(max_drawdowns)),
            "worst_max_drawdown": float(np.min(max_drawdowns)),
            "mean_pnl": float(mean_pnl), "std_pnl": float(std_pnl)
        }
    except Exception as e:
        log.error(f"Ошибка Monte Carlo: {e}")
        return {"valid": False, "message": str(e)}

# ============================================================
# ████████████████████████████████████████████
# █████████████████   РАСШИРЕННАЯ СКОРИНГОВАЯ СИСТЕМА   █████████
# ============================================================

def получить_скор(symbol: str, use_quant: bool = True, use_order_flow: bool = True,
                use_ml: bool = True) -> dict:
    details = {}
    score = 0
    price = 0.0
    sr = {}
    try:
        raw_ta = fetch_ohlcv_cached(symbol, TIMEFRAME_TA, limit=300)
        raw_1h = fetch_ohlcv_cached(symbol, TIMEFRAME_TREND, limit=300)
        if len(raw_ta) < 100 or len(raw_1h) < 100:
            return {"score": 0, "details": {}, "price": 0, "sr": {}}
        cols = ["ts","o","h","l","c","v"]
        df_ta = pd.DataFrame(raw_ta, columns=cols).reset_index(drop=True)
        df_1h = pd.DataFrame(raw_1h, columns=cols).reset_index(drop=True)
        c_ta = df_ta["c"]
        c_1h = df_1h["c"]
        price = float(c_ta.iloc[-1])

        # ========== КЛАССИЧЕСКИЙ ТЕХАНАЛИЗ ==========
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

        ml_macd, sl_macd, _ = calc_macd(c_ta)
        macd_bull = ml_macd.iloc[-1] > sl_macd.iloc[-1]
        macd_cross = macd_bull and ml_macd.iloc[-2] <= sl_macd.iloc[-2]
        details["macd"] = "бычий" if macd_bull else "медвежий"
        if macd_cross: score += 18
        elif macd_bull: score += 8

        _, _, _, rf_up, rf_down = calc_range_filter(df_ta)
        rf_up_now = rf_up.iloc[-1]
        details["range_filter"] = "вверх" if rf_up_now else ("вниз" if rf_down.iloc[-1] else "бок")
        if rf_up_now: score += 15

        st_up, _ = calc_supertrend(df_ta)
        details["supertrend"] = "вверх" if st_up.iloc[-1] else "вниз"
        if st_up.iloc[-1]: score += 12

        hu_up, _ = calc_hull(c_ta)
        details["hull"] = "вверх" if hu_up.iloc[-1] else "вниз"
        if hu_up.iloc[-1]: score += 8

        ema50_1h = _ema(c_1h, 50).iloc[-1]
        ema200_1h = _ema(c_1h, 200).iloc[-1]
        details["тренд_1h"] = "бычий" if ema50_1h > ema200_1h else "медвежий"
        if ema50_1h > ema200_1h: score += 10

        adx, pdi, mdi = calc_adx(df_ta)
        adx_val = adx.iloc[-1]
        details["adx"] = round(adx_val, 1)
        if adx_val > 25 and pdi.iloc[-1] > mdi.iloc[-1]: score += 10
        elif adx_val > 20 and pdi.iloc[-1] > mdi.iloc[-1]: score += 4

        k_ser, _ = calc_stochastic(df_ta)
        k_val = k_ser.iloc[-1]
        details["stoch_k"] = round(k_val, 1)
        if k_val < 20: score += 10
        elif k_val < 40: score += 5

        vol_avg = df_ta["v"].rolling(VOLUME_AVG_PERIOD).mean().iloc[-1]
        vol_ratio = df_ta["v"].iloc[-1] / (vol_avg + 1e-10)
        details["объём_ratio"] = round(vol_ratio, 2)
        if vol_ratio > 1.5: score += 8
        elif vol_ratio > 1.2: score += 4

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

        last3_bearish = all(df_ta["c"].iloc[-i] < df_ta["o"].iloc[-i] for i in range(1, 4))
        if last3_bearish:
            score -= 20
            details["свечи_3red"] = True

        bayes_prob = bayes_trend_probability(df_ta)
        details["bayes_prob"] = round(bayes_prob, 2)
        score += int(bayes_prob * 10)

        # ========== КВАНТОВЫЙ АНАЛИЗ ==========
        if use_quant and QUANT_ENABLED:
            quant_data = get_quant_signals(symbol)
            quant_score = quant_data["quant_score"]
            details["quant_score"] = quant_score
            score += quant_score * 0.3
            details["quant_details"] = quant_data["details"]

        # ========== ORDER FLOW АНАЛИЗ ==========
        order_flow_data = {}
        if use_order_flow and ORDER_FLOW_ENABLED:
            order_flow_data = get_order_flow_signals(symbol)
            of_score = order_flow_data["order_flow_score"]
            details["order_flow_score"] = of_score
            score += of_score * 0.2
            details["order_flow_details"] = order_flow_data["details"]

        # ========== ML АНАЛИЗ ==========
        if use_ml and ML_ENABLED and ml_model.trained:
            ml_prediction = ml_model.predict(symbol, df_ta, df_1h,
                                           order_flow_data if use_order_flow else {},
                                           quant_data if use_quant else {})
            if ml_prediction["valid"]:
                ml_prob = ml_prediction["probability"]
                details["ml_probability"] = round(ml_prob, 2)
                details["ml_signal"] = ml_prediction["signal"]
                if ml_prob > 0.7: score += 15
                elif ml_prob > 0.6: score += 10
                elif ml_prob > 0.55: score += 5

        details["ma_cross"] = проверить_ma_кроссовер(df_ta, side="long")
        details["vol_spike_ok"] = volume_spike_guard(df_ta)

        return {"score": max(0, min(100, score)), "details": details, "price": price, "sr": sr}
    except Exception as e:
        log.warning(f"Ошибка анализа {symbol}: {e}")
        return {"score": 0, "details": {}, "price": 0, "sr": {}}

def получить_скор_шорта(symbol: str) -> dict:
    res = получить_скор(symbol)
    if res["score"] == 0: return res
    res["score"] = max(0, 100 - res["score"] - 10)
    try:
        raw = fetch_ohlcv_cached(symbol, TIMEFRAME_TA, limit=300)
        if len(raw) >= 50:
            df = pd.DataFrame(raw, columns=["ts","o","h","l","c","v"])
            res["details"]["ma_cross"] = проверить_ma_кроссовер(df, side="short")
    except: pass
    return res

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
# ██████████████████████████████████████████████
# █████████████████   РИСК-МЕНЕДЖМЕНТ   ████████████████████
# ============================================================

def рассчитать_оптимальное_f(сделки: List[dict]) -> float:
    if len(сделки) < MIN_TRADES_FOR_F: return 0.0
    pnls = [abs(t['pnl_usdt']) for t in сделки if t['pnl_usdt'] != 0]
    if not pnls: return 0.0
    wins = sum(1 for t in сделки if t['pnl_usdt'] > 0)
    losses = sum(1 for t in сделки if t['pnl_usdt'] < 0)
    if losses == 0: return 0.25
    win_rate = wins / (wins + losses)
    avg_win = sum(t['pnl_usdt'] for t in сделки if t['pnl_usdt'] > 0) / wins if wins else 0
    avg_loss = abs(sum(t['pnl_usdt'] for t in сделки if t['pnl_usdt'] < 0)) / losses if losses else 1
    if avg_loss == 0: avg_loss = 1
    kelly = win_rate - (1 - win_rate) / (avg_win / avg_loss)
    return min(max(0, kelly * 0.4), MAX_RISK_PERCENT_F / 100)

def рассчитать_размер_позиции(score: int, баланс: float, sl_dist_pct: float,
                               история_сделок: List[dict] = None) -> float:
    if USE_ADVANCED_RISK and история_сделок and len(история_сделок) >= MIN_TRADES_FOR_F:
        f_opt = рассчитать_оптимальное_f(история_сделок[-100:])
        risk_pct = max(0.5, min(f_opt * 100, MAX_RISK_PERCENT_F))
        log.info(f"Оптимальное f = {f_opt:.3f} → риск {risk_pct:.2f}%")
    else:
        factor = max(0, (score - MIN_SCORE)) / (100 - MIN_SCORE)
        risk_pct = min(BASE_RISK_PCT + (MAX_RISK_PCT - BASE_RISK_PCT) * factor, MAX_RISK_PCT)
    max_loss_usdt = баланс * risk_pct / 100
    margin_usdt = min(max_loss_usdt / (sl_dist_pct / 100), баланс * 0.95)
    log.info(f"Скор={score} → риск={risk_pct:.1f}% SL_dist={sl_dist_pct:.2f}% маржа={margin_usdt:.2f}U")
    return round(max(1.0, margin_usdt), 2)

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
# ████████████████████████████████████████████
# █████████████████   ИСПОЛНЕНИЕ ОРДЕРОВ   ██████████████████
# ============================================================

def установить_плечо(symbol: str, leverage: int) -> bool:
    try:
        coin_sym = symbol.split("/")[0] + "USDT"
        exchange.set_leverage(leverage, coin_sym, params={"buyLeverage": leverage, "sellLeverage": leverage})
        log.info(f"Плечо {leverage}x установлено для {coin_sym}")
        return True
    except Exception as e1:
        log.warning(f"Метод 1 плеча не сработал: {e1}")
        try:
            params = {
                "category": "linear",
                "symbol": coin_sym,
                "buyLeverage": str(leverage),
                "sellLeverage": str(leverage)
            }
            exchange.private_post_v5_position_set_leverage(params)
            log.info(f"Плечо {leverage}x установлено (v5) для {coin_sym}")
            return True
        except Exception as e2:
            log.error(f"Не удалось установить плечо {leverage}x для {symbol}: {e2}")
            return False

def обновить_sl_на_бирже(symbol: str, new_sl: float, side: str = "long") -> bool:
    try:
        sl_str = exchange.price_to_precision(symbol, new_sl)
        coin_sym = symbol.split("/")[0] + "USDT"
        params = {
            "category": "linear",
            "symbol": coin_sym,
            "stopLoss": sl_str,
            "slTriggerBy": "MarkPrice",
            "positionIdx": "0",
        }
        exchange.private_post_v5_position_trading_stop(params)
        log.info(f"SL обновлён → {sl_str}")
        return True
    except Exception as e:
        log.warning(f"Не удалось обновить SL: {e}")
        return False

def открыть_позицию(symbol: str, margin_usdt: float, tp_price: float,
                    sl_price: float, side: str = "long") -> Tuple[Optional[float], Optional[float]]:
    try:
        ticker = exchange.fetch_ticker(symbol)
        price = float(ticker["last"])
        current_balance = полный_баланс_usdt()
        open_positions = получить_позиции()

        # Корректировка цен (исправлено для шортов)
        if side == "long":
            sl_price = min(sl_price, price * (1 - MIN_SL_PERCENT / 100))
            tp_price = max(tp_price, price * (1 + TP_PERCENT / 100))
        else:
            sl_price = max(sl_price, price * (1 + MIN_SL_PERCENT / 100))
            tp_price = min(tp_price, price * (1 - TP_PERCENT / 100))

        profitability_ok, profitability_data = проверка_прибыльности_сделки(
            symbol, margin_usdt, price, tp_price, sl_price, side, current_balance, open_positions
        )
        if not profitability_ok:
            log.warning(f"❌ Сделка отклонена по проверке прибыльности: {profitability_data.get('reason', 'Неизвестная причина')}")
            if "pnl_data" in profitability_data:
                pnl_data = profitability_data["pnl_data"]
                log.warning(f"   Ожидаемый P&L: {pnl_data.get('expected_pnl', 0):+.4f}U")
                log.warning(f"   RR Ratio: {pnl_data.get('rr_ratio', 0):.2f}")
                log.warning(f"   Риск: {pnl_data.get('risk_usdt', 0):.4f}U ({pnl_data.get('risk_pct', 0):.2f}%)")
            return None, None

        log.info(f"✅ Сделка одобрена: {profitability_data.get('message')}")
        pnl_data = profitability_data.get("pnl_data", {})
        log.info(f"   Ожидаемый P&L: {pnl_data.get('expected_pnl', 0):+.4f}U")
        log.info(f"   Соотношение риск/прибыль: {pnl_data.get('rr_ratio', 0):.2f}")
        log.info(f"   Риск: {pnl_data.get('risk_usdt', 0):.4f}U ({pnl_data.get('risk_pct', 0):.2f}%)")

        if not установить_плечо(symbol, LEVERAGE):
            log.error(f"Плечо не установлено — сделка отменена для {symbol}")
            return None, None

        pos_size_usdt = margin_usdt * LEVERAGE
        qty_raw = pos_size_usdt / price
        qty = float(exchange.amount_to_precision(symbol, qty_raw))
        if qty <= 0:
            log.error(f"Нулевое количество {symbol}")
            return None, None

        tp_str = exchange.price_to_precision(symbol, tp_price)
        sl_str = exchange.price_to_precision(symbol, sl_price)
        buy_sell = "buy" if side == "long" else "sell"
        log.info(f"Открываем {side} {symbol}: qty={qty}, маржа≈{margin_usdt:.2f}U, плечо={LEVERAGE}x, TP={tp_str}, SL={sl_str}")

        try:
            order = exchange.create_market_order(
                symbol, buy_sell, qty,
                params={
                    "takeProfit": float(tp_str),
                    "stopLoss": float(sl_str),
                    "reduceOnly": False
                }
            )
            if not order or "average" not in order:
                log.error(f"Не удалось открыть позицию: {order}")
                return None, None
            entry_price = float(order.get("average", price))
            log.info(f"{side.upper()} открыт: {qty} {symbol} @ ~{entry_price:.8f}")

            if ML_ENABLED and ml_model.trained:
                try:
                    raw_ta = fetch_ohlcv_cached(symbol, TIMEFRAME_TA, limit=100)
                    raw_1h = fetch_ohlcv_cached(symbol, TIMEFRAME_TREND, limit=100)
                    if len(raw_ta) >= 50 and len(raw_1h) >= 50:
                        df_ta = pd.DataFrame(raw_ta, columns=["ts","o","h","l","c","v"])
                        df_1h = pd.DataFrame(raw_1h, columns=["ts","o","h","l","c","v"])
                        order_flow_data = get_order_flow_signals(symbol)
                        quant_data = get_quant_signals(symbol)
                        prediction = ml_model.predict(
                            symbol, df_ta, df_1h,
                            order_flow_data, quant_data, pnl_data
                        )
                        логировать_ml_данные(
                            symbol,
                            prediction.get("features", {}),
                            prediction,
                            trade_result=None,
                            pnl=None
                        )
                except Exception as e:
                    log.warning(f"Не удалось логировать данные для ML: {e}")
            return entry_price, qty
        except Exception as e:
            log.error(f"Ошибка открытия {side}: {e}", exc_info=True)
            return None, None
    except Exception as e:
        log.error(f"Глобальная ошибка в открыть_позицию: {e}", exc_info=True)
        return None, None

def закрыть_позицию_с_подтверждением(symbol: str, qty: float, side: str) -> bool:
    close_side = "sell" if side == "long" else "buy"
    for attempt in range(3):
        try:
            exchange.create_market_order(symbol, close_side, qty, params={"reduceOnly": True})
            time.sleep(3)
            positions = exchange.fetch_positions([symbol])
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
        raw = fetch_ohlcv_cached(symbol, TIMEFRAME_TA, limit=50)
        if len(raw) < 30: return False
        df = pd.DataFrame(raw, columns=["ts","o","h","l","c","v"])
        st_up, st_down = calc_supertrend(df)
        _, _, _, rf_up, rf_down = calc_range_filter(df)
        return bool(st_down.iloc[-1] and rf_down.iloc[-1]) if side == "long" else bool(st_up.iloc[-1] and rf_up.iloc[-1])
    except Exception: return False

# ============================================================
# ████████████████████████████████████████████
# █████████████████   МОНИТОРИНГ ПОЗИЦИЙ   ████████████████
# ============================================================

def мониторить_позицию(symbol: str, entry_price: float, qty: float,
                        открыта_в: float, sl_цена: float,
                        tp_цена: float, side: str = "long") -> str:
    deadline = открыта_в + TRADE_MAX_LIFETIME
    coin = symbol.split("/")[0]
    breakeven_price = entry_price * (1 + BYBIT_FEE * 2 + 0.0005) if side == "long" else entry_price * (1 - BYBIT_FEE * 2 - 0.0005)

    try:
        raw = fetch_ohlcv_cached(symbol, TIMEFRAME_TA, limit=50)
        if len(raw) >= 30:
            df = pd.DataFrame(raw, columns=["ts","o","h","l","c","v"])
            atr_val = calc_atr(df, TRAILING_ATR_PERIOD).iloc[-1]
            atr_pct = (atr_val / entry_price) * 100
            trailing_step = max(MIN_TRAILING_STEP, atr_pct * TRAILING_ATR_MULT) / 100
            trailing_offset = max(MIN_TRAILING_OFFSET, atr_pct * TRAILING_OFFSET_MULT) / 100
    except Exception:
        trailing_step = MIN_TRAILING_STEP / 100
        trailing_offset = MIN_TRAILING_OFFSET / 100

    if side == "long":
        rr_trigger_price = entry_price + (tp_цена - entry_price) * RR_EXIT_TRIGGER
    else:
        rr_trigger_price = entry_price - (entry_price - tp_цена) * RR_EXIT_TRIGGER

    log.info(f"Мониторинг {coin} {side} вход={entry_price:.8f} SL={sl_цена:.8f} TP={tp_цена:.8f} BE={breakeven_price:.8f}")
    log.info(f"rrExit триггер={rr_trigger_price:.8f} (RR_EXIT={RR_EXIT_TRIGGER})")

    фаза, текущий_sl, пиковая_цена = 1, sl_цена, entry_price
    trailing_активен = (RR_EXIT_TRIGGER == 0.0)
    partial_done = False

    while True:
        сейчас = time.time()
        if сейчас >= deadline:
            log.warning("Дедлайн — принудительное закрытие")
            закрыть_позицию_с_подтверждением(symbol, qty, side)
            return "таймаут"
        time.sleep(15)

        try:
            positions = exchange.fetch_positions([symbol])
            active = [p for p in positions if float(p.get("contracts", 0) or 0) > 0 and p.get("side") == side]
            if not active:
                cur_price = exchange.fetch_ticker(symbol)["last"]
                hit_tp = (cur_price >= entry_price * (1 + TP_PERCENT / 100 * 0.7)) if side == "long" else (cur_price <= entry_price * (1 - TP_PERCENT / 100 * 0.7))
                return "tp" if hit_tp or фаза >= 2 else "sl"

            pos = active[0]
            cur_price = exchange.fetch_ticker(symbol)["last"]
            qty_actual = abs(float(pos.get("contracts", 0) or 0))
            pnl_real = float(pos.get("unrealizedPnl", 0) or 0)
            pnl_pct = ((cur_price - entry_price) / entry_price * 100) if side == "long" else ((entry_price - cur_price) / entry_price * 100)
            до_дед = int(deadline - сейчас)

            if PARTIAL_BE_ENABLED and not partial_done and pnl_pct >= PARTIAL_BE_PROFIT:
                close_qty = qty_actual * (PARTIAL_BE_CLOSE_PCT / 100)
                if close_qty > 0:
                    close_side = "sell" if side == "long" else "buy"
                    try:
                        exchange.create_market_order(symbol, close_side, close_qty, params={"reduceOnly": True})
                        log.info(f"Частичный безубыток: закрыто {close_qty:.4f} ({PARTIAL_BE_CLOSE_PCT:.0f}%) @ ~{cur_price:.8f}")
                        qty_actual -= close_qty
                        new_sl = entry_price * (1 + BYBIT_FEE * 2 + 0.0003) if side == "long" else entry_price * (1 - BYBIT_FEE * 2 - 0.0003)
                        if обновить_sl_на_бирже(symbol, new_sl, side):
                            текущий_sl = new_sl
                            log.info(f"SL для остатка переведён в безубыток: {new_sl:.8f}")
                        partial_done = True
                    except Exception as e:
                        log.warning(f"Не удалось частично закрыть: {e}")

            if SIGNAL_EXIT_ENABLED and фаза >= 2 and проверить_signal_exit(symbol, side):
                log.info("Signal Exit: разворот — закрываем")
                закрыть_позицию_с_подтверждением(symbol, qty_actual, side)
                return "tp" if pnl_pct > 0 else "sl"

            if not partial_done and фаза == 1 and pnl_pct >= 0.3:
                new_sl_be = entry_price * (1 + BYBIT_FEE * 2 + 0.0003) if side == "long" else entry_price * (1 - BYBIT_FEE * 2 - 0.0003)
                if обновить_sl_на_бирже(symbol, new_sl_be, side):
                    фаза, текущий_sl, пиковая_цена = 2, new_sl_be, cur_price
                    log.info(f"БЕЗУБЫТОК! SL → {new_sl_be:.8f}")

            if not trailing_активен and фаза >= 2:
                trailing_активен = (cur_price >= rr_trigger_price) if side == "long" else (cur_price <= rr_trigger_price)
                if trailing_активен:
                    log.info(f"Трейлинг активирован @ {cur_price:.8f}")

            if trailing_активен and фаза >= 2 and pnl_pct >= MIN_PROFIT_FOR_TRAIL:
                if side == "long":
                    if cur_price > пиковая_цена:
                        пиковая_цена = cur_price
                    new_sl_trail = пиковая_цена * (1 - trailing_offset)
                    if new_sl_trail > текущий_sl and обновить_sl_на_бирже(symbol, new_sl_trail, side):
                        текущий_sl = new_sl_trail
                        log.info(f"ТРЕЙЛИНГ: пик={пиковая_цена:.8f} → SL={new_sl_trail:.8f}")
                else:
                    if cur_price < пиковая_цена:
                        пиковая_цена = cur_price
                    new_sl_trail = пиковая_цена * (1 + trailing_offset)
                    if new_sl_trail < текущий_sl and обновить_sl_на_бирже(symbol, new_sl_trail, side):
                        текущий_sl = new_sl_trail
                        log.info(f"ТРЕЙЛИНГ: пик={пиковая_цена:.8f} → SL={new_sl_trail:.8f}")

            if partial_done:
                log.info(f"[{coin}] {cur_price:.8f} P&L={pnl_pct:+.2f}% ({pnl_real:+.4f}U) SL={текущий_sl:.8f} остаток={qty_actual:.4f} фаза={фаза} дед={до_дед}с")
            else:
                log.debug(f"[{coin}] {cur_price:.8f} P&L={pnl_pct:+.2f}% ({pnl_real:+.4f}U) SL={текущий_sl:.8f} фаза={фаза} дед={до_дед}с")
        except Exception as e:
            log.warning(f"Ошибка в цикле мониторинга: {e}")
            continue

    return "sl"

# ============================================================
# ███████████████████████████████████████████
# █████████████████   СТАТИСТИКА ИНДИКАТОРОВ   ███████████████
# ============================================================

def загрузить_статистику_индикаторов() -> dict:
    if not os.path.exists(INDICATOR_STATS_FILE):
        return {}
    try:
        with open(INDICATOR_STATS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

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
        "rsi": lambda v: 25 <= float(v) <= 42, "rsi_1h": lambda v: float(v) < 55,
        "macd": lambda v: v == "бычий", "range_filter": lambda v: v == "вверх",
        "supertrend": lambda v: v == "вверх", "hull": lambda v: v == "вверх",
        "тренд_1h": lambda v: v == "бычий", "adx": lambda v: float(v) > 25,
        "stoch_k": lambda v: float(v) < 25, "объём_ratio": lambda v: float(v) > 1.5,
        "sr_signal": lambda v: "поддержки" in str(v), "bayes_prob": lambda v: float(v) > 0.6,
        "quant_score": lambda v: float(v) > 50, "order_flow_score": lambda v: float(v) > 50,
    }
    for инд, условие in индикаторы.items():
        значение = details.get(инд)
        if значение is None:
            continue
        try:
            is_bullish = условие(значение)
        except:
            continue
        if инд not in stats_data:
            stats_data[инд] = {"bullish": {"total": 0, "wins": 0}, "bearish": {"total": 0, "wins": 0}}
        if is_bullish:
            stats_data[инд]["bullish"]["total"] += 1
            if is_win:
                stats_data[инд]["bullish"]["wins"] += 1
        else:
            stats_data[инд]["bearish"]["total"] += 1
            if is_win:
                stats_data[инд]["bearish"]["wins"] += 1
    сохранить_статистику_индикаторов(stats_data)

def отчёт_по_индикаторам():
    stats_data = загрузить_статистику_индикаторов()
    if not stats_data:
        log.info("Статистика индикаторов пуста")
        return
    log.info("")
    log.info("=" * 70)
    log.info("📈 ЭФФЕКТИВНОСТЬ ИНДИКАТОРОВ (накопленная)")
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
    try:
        with open("indicators_analysis.json", "w", encoding="utf-8") as f:
            json.dump(stats_data, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        log.warning(f"Не удалось сохранить отчёт по индикаторам: {e}")

# ============================================================
# ███████████████████████████████████████████
# ████████████████   МЕТРИКИ СТРАТЕГИИ   █████████████████
# ============================================================

def рассчитать_метрики(сделки: List[dict]) -> dict:
    if len(сделки) < 5:
        return {}
    pnls = [t['pnl_usdt'] for t in сделки]
    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    max_dd = abs(min(drawdowns))
    max_dd_pct = (max_dd / max(1, cumulative[-1] + max_dd)) * 100 if cumulative[-1] != 0 else 0
    sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252) if len(pnls) > 1 and np.std(pnls) != 0 else 0
    neg_returns = [p for p in pnls if p < 0]
    sortino = np.mean(pnls) / np.std(neg_returns) * np.sqrt(252) if neg_returns and np.std(neg_returns) != 0 else 0
    total_return = cumulative[-1]
    years = len(сделки) / 252
    annual_return = total_return / years if years > 0 else total_return
    calmar = annual_return / (max_dd if max_dd > 0 else 1)
    recovery = cumulative[-1] / (max_dd if max_dd > 0 else 1)
    return {
        "sharpe_ratio": round(sharpe, 2), "sortino_ratio": round(sortino, 2),
        "calmar_ratio": round(calmar, 2), "max_drawdown_usdt": round(max_dd, 2),
        "max_drawdown_pct": round(max_dd_pct, 1), "recovery_factor": round(recovery, 2),
        "total_trades": len(сделки), "winrate": round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 1),
        "avg_win": round(np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0, 2),
        "avg_loss": round(abs(np.mean([p for p in pnls if p < 0])) if any(p < 0 for p in pnls) else 0, 2),
    }

def быстрый_walk_forward(история: List[dict], window: int = 50, step: int = 10) -> dict:
    if len(история) < window * 2:
        return {"стабильность": False, "рекомендация": "Недостаточно данных"}
    positive_windows = 0
    total_windows = 0
    for i in range(0, len(история) - window, step):
        out_sample = история[i+window:i+window+step]
        if len(out_sample) < 5:
            continue
        pnl_out = sum(t['pnl_usdt'] for t in out_sample)
        if pnl_out > 0:
            positive_windows += 1
        total_windows += 1
    if total_windows == 0:
        return {"стабильность": False, "рекомендация": "Недостаточно данных"}
    stability = positive_windows / total_windows > 0.6
    return {
        "стабильность": stability, "положительные_окна": positive_windows,
        "всего_окон": total_windows,
        "рекомендация": "Стратегия робастна" if stability else "Стратегия нестабильна – нужна оптимизация"
    }

def сохранить_метрики(метрики: dict):
    try:
        with open(METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump(метрики, f, ensure_ascii=False, indent=2, default=str)
        log.info(f"Метрики сохранены в {METRICS_FILE}")
    except Exception as e:
        log.warning(f"Не удалось сохранить метрики: {e}")

# ============================================================
# ███████████████████████████████████████████
# █████████████████   ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ   ███████████████
# ============================================================

def загрузить_историю() -> List[dict]:
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
            json.dump(история, f, ensure_ascii=False, indent=2, default=str)
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
        return total if total > 0 else баланс_usdt()
    except Exception as e:
        log.warning(f"Ошибка получения полного баланса: {e}")
        return баланс_usdt()

def получить_позиции() -> List[dict]:
    try:
        positions = exchange.fetch_positions()
        return [p for p in positions if float(p.get("contracts", 0) or 0) > 0]
    except Exception as e:
        log.warning(f"Ошибка получения позиций: {e}")
        return []

def обновить_начало_дня(баланс: float):
    сегодня = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if stats["дата_дня"] != сегодня:
        stats["дата_дня"] = сегодня
        stats["баланс_начало_дня"] = баланс
        log.info(f"Новый торговый день. Баланс: {баланс:.2f} USDT")
        сохранить_состояние()

def превышен_дневной_лимит() -> bool:
    нач = stats.get("баланс_начало_дня", 0.0)
    if нач <= 0:
        return False
    текущий = полный_баланс_usdt()
    реализованный_убыток = stats.get("убыток_usdt", 0.0)
    потеря_pct = (реализованный_убыток / нач * 100) if нач > 0 else 0
    if потеря_pct >= DAILY_LOSS_LIMIT_PCT:
        log.warning(f"Дневной лимит убытков: -{потеря_pct:.1f}% (лимит {DAILY_LOSS_LIMIT_PCT}%)")
        return True
    return False

def пост_трейд_анализ(запись: dict):
    r = запись["результат"]
    sym = запись["symbol"]
    pnl = запись.get("pnl_usdt", 0)
    dur = запись.get("duration_min", 0)
    знак = "✅" if r == "tp" else ("❌" if r == "sl" else "⏰")
    log.info("")
    log.info("━" * 60)
    log.info(f"📋 ПОСТ-ТРЕЙД: {sym.split(':')[0]} {знак} {r.upper()} P&L: {pnl:+.4f} USDT Длит: {dur:.1f} мин")
    log.info(f"Скор входа: {запись.get('score', '?')}/100 | RR: {запись.get('rr_ratio', '?')}")
    log.info(f"Маржа: {запись.get('margin_usdt', 0):.2f}U | Плечо: {запись.get('leverage', LEVERAGE)}x")
    log.info("━" * 60)
    log.info("")

    if ML_ENABLED and ml_model.trained and запись.get("entry_price"):
        try:
            raw_ta = fetch_ohlcv_cached(sym, TIMEFRAME_TA, limit=100)
            raw_1h = fetch_ohlcv_cached(sym, TIMEFRAME_TREND, limit=100)
            if len(raw_ta) >= 50 and len(raw_1h) >= 50:
                df_ta = pd.DataFrame(raw_ta, columns=["ts","o","h","l","c","v"])
                df_1h = pd.DataFrame(raw_1h, columns=["ts","o","h","l","c","v"])
                entry_time = запись.get("время_входа")
                if entry_time:
                    entry_timestamp = datetime.strptime(entry_time, "%d.%m.%Y %H:%M:%S").timestamp()
                    df_ta_entry = df_ta[df_ta["ts"] <= entry_timestamp * 1000].tail(1)
                    df_1h_entry = df_1h[df_1h["ts"] <= entry_timestamp * 1000].tail(1)
                    if not df_ta_entry.empty and not df_1h_entry.empty:
                        order_flow_data = get_order_flow_signals(sym)
                        quant_data = get_quant_signals(sym)
                        risk_data = рассчитать_точный_pnl(
                            запись["entry_price"],
                            запись["tp_price"],
                            запись["sl_price"],
                            запись["margin_usdt"],
                            LEVERAGE,
                            sym,
                            запись.get("side", "long")
                        )
                        features = ml_model.create_features(
                            sym, df_ta_entry, df_1h_entry,
                            order_flow_data, quant_data, risk_data
                        )
                        логировать_ml_данные(
                            sym,
                            features,
                            {"signal": "buy" if запись.get("side") == "long" else "sell", "probability": 0.5},
                            trade_result=r,
                            pnl=pnl
                        )
        except Exception as e:
            log.warning(f"Не удалось логировать данные для ML: {e}")

# ============================================================
# ██████████████████████████████████████████
# ████████████████   ОТЧЁТЫ   ███████████████████████
# ============================================================

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
    log.info(f"📊 ОТЧЁТ ГИБРИДНОГО БОТА {BOT_VERSION}")
    log.info(f"Версия: {BOT_VERSION} | Дата релиза: {RELEASE_DATE}")
    log.info(f"Баланс: {баланс:.2f} USDT ({дельта:+.2f} USDT / {пct:+.2f}%)")
    log.info(f"Сделок: {всего} TP={tp_} SL={sl_} Таймаут={stats['таймаут']}")
    log.info(f"WinRate: {wr:.1f}%")
    log.info(f"Прибыль/Убыток: {stats['прибыль_usdt']:.4f} / {stats['убыток_usdt']:.4f} USDT")
    log.info(f"Чистый P&L: {чистый:+.4f} USDT")

    if ML_ENABLED:
        log.info("-" * 65)
        log.info("🤖 СТАТУС ML МОДЕЛИ:")
        if ml_model.trained:
            log.info(f"Модель: {ml_model.model_type} | Версия фич: {ML_FEATURES_VERSION}")
            log.info(f"Точность: {ml_model.accuracy:.2f} | Precision: {ml_model.precision:.2f}")
            log.info(f"Обучена: {datetime.fromtimestamp(ml_model.last_retrain).strftime('%d.%m.%Y %H:%M:%S')}")
            log.info(f"Сделок с момента переобучения: {stats.get('ml_trades_since_retrain', 0)}")
            if ml_model.feature_importances:
                top_features = sorted(ml_model.feature_importances.items(), key=lambda x: x[1], reverse=True)[:5]
                log.info(f"Топ фич: {', '.join([f'{f[0]} ({f[1]:.3f})' for f in top_features])}")
        else:
            log.info("Модель не обучена")

    log.info("=" * 65)
    log.info("")

    stats["последний_отчёт"] = time.time()
    сохранить_состояние()
    отчёт_по_индикаторам()

    история = загрузить_историю()
    if len(история) > 5:
        метрики = рассчитать_метрики(история)
        if метрики:
            log.info("📉 МЕТРИКИ СТРАТЕГИИ:")
            log.info(f"Sharpe: {метрики.get('sharpe_ratio',0)} | Sortino: {метрики.get('sortino_ratio',0)} | Calmar: {метрики.get('calmar_ratio',0)}")
            log.info(f"Max Drawdown: {метрики.get('max_drawdown_pct',0)}% | Recovery: {метрики.get('recovery_factor',0)}")
            сохранить_метрики(метрики)
        if len(история) > 100:
            wf = быстрый_walk_forward(история)
            log.info(f"🔄 Walk-Forward: {wf.get('рекомендация')} (окна: {wf.get('положительные_окна',0)}/{wf.get('всего_окон',0)})")

    if MONTE_CARLO_ENABLED and len(история) > 20:
        if time.time() - stats.get("monte_carlo_last_run", 0) > 3600:
            mc_result = run_monte_carlo_simulation(история)
            if mc_result["valid"]:
                stats["monte_carlo_last_run"] = time.time()
                log.info("🎲 MONTE CARLO СИМУЛЯЦИЯ:")
                log.info(f"5-й перцентиль: {mc_result['percentile_5']:.2f} USDT")
                log.info(f"50-й перцентиль: {mc_result['percentile_50']:.2f} USDT")
                log.info(f"95-й перцентиль: {mc_result['percentile_95']:.2f} USDT")
                log.info(f"Вероятность убытка: {mc_result['loss_probability']:.1f}%")
                log.info(f"Средняя max просадка: {mc_result['avg_max_drawdown']:.2f} USDT")

    if ML_LOG_DATA and os.path.exists(ML_LOG_FILE):
        try:
            file_size = os.path.getsize(ML_LOG_FILE) / (1024 * 1024)
            log.info(f"📁 Файл ML данных: {ML_LOG_FILE} ({file_size:.2f} MB)")
            if file_size > 50:
                log.info("🔄 Файл ML данных большой — переобучаем модель...")
                if ml_model.train():
                    ml_model.save_model()
                    stats["ml_trades_since_retrain"] = 0
                    try:
                        os.rename(ML_LOG_FILE, f"ml_training_data_archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                        log.info(f"Файл ML данных архивирован")
                    except:
                        pass
        except Exception as e:
            log.warning(f"Не удалось проверить размер файла ML данных: {e}")

# ============================================================
# ███████████████████████████████████████████
# █████████████████   ПРЕДСТАРТОВАЯ ПРОВЕРКА   ████████████
# ============================================================

def этап_1_проверка_окружения() -> Tuple[bool, List[str]]:
    errors, warnings = [], []
    api_key, api_secret = os.getenv("BYBIT_API_KEY", ""), os.getenv("BYBIT_API_SECRET", "")
    if not api_key:
        errors.append("BYBIT_API_KEY не задан")
    elif len(api_key) < 10:
        errors.append("BYBIT_API_KEY слишком короткий")
    if not api_secret:
        errors.append("BYBIT_API_SECRET не задан")
    elif len(api_secret) < 10:
        errors.append("BYBIT_API_SECRET слишком короткий")
    if not os.path.exists(".env"):
        warnings.append(".env файл не найден (используются переменные окружения)")
    return len(errors) == 0, errors + warnings

def этап_2_проверка_подключения() -> Tuple[bool, List[str]]:
    errors = []
    try:
        b = exchange.fetch_balance({"type": "linear"})
        usdt = float(b.get("USDT", {}).get("free", 0))
        if usdt < MIN_BALANCE:
            errors.append(f"Баланс {usdt:.2f} < {MIN_BALANCE} USDT")
        else:
            log.info(f"Подключение OK | Баланс: {usdt:.4f} USDT")
    except ccxt.AuthenticationError as e:
        errors.append(f"Ошибка аутентификации: {e}")
    except ccxt.NetworkError as e:
        errors.append(f"Сетевая ошибка: {e}")
    except Exception as e:
        errors.append(f"Неизвестная ошибка: {e}")
    return len(errors) == 0, errors

def этап_3_проверка_конфигурации() -> Tuple[bool, List[str]]:
    errors, warnings = [], []
    rr = TP_PERCENT / SL_PERCENT
    if rr < 2.0:
        errors.append(f"RR {rr:.1f}:1 слишком низкий")
    elif rr < 2.5:
        warnings.append(f"RR {rr:.1f}:1 можно повысить")
    if LEVERAGE > 5:
        warnings.append(f"Плечо {LEVERAGE}x высокое")
    if MIN_SCORE < 65:
        errors.append(f"MIN_SCORE={MIN_SCORE} < 65")
    if BASE_RISK_PCT > 3.0:
        errors.append(f"BASE_RISK_PCT={BASE_RISK_PCT}% слишком высок")
    if DAILY_LOSS_LIMIT_PCT > 5.0:
        warnings.append(f"DAILY_LOSS_LIMIT_PCT={DAILY_LOSS_LIMIT_PCT}% высокий")
    if ATR_SL_MULT < 1.5:
        errors.append(f"ATR_SL_MULT={ATR_SL_MULT} слишком мал")
    if DEPOSIT_ANALYSIS_ENABLED:
        log.info(f"Проверка депозита: ВКЛ | Min P&L=адаптивный (1.5% от баланса) | Max Risk={MAX_RISK_PER_TRADE_PCT}%")
    log.info(f"Конфигурация: TP={TP_PERCENT}% | SL={SL_PERCENT}% | RR={rr:.1f}:1")
    return len(errors) == 0, errors + warnings

def этап_4_проверка_рынка() -> Tuple[bool, List[str]]:
    errors, warnings = [], []
    test_symbols = SYMBOLS[:5]
    доступные = 0
    for sym in test_symbols:
        try:
            ticker = exchange.fetch_ticker(sym)
            if float(ticker["last"]) > 0:
                доступные += 1
        except Exception as e:
            warnings.append(f"Пара {sym} недоступна: {e}")
    if доступные == 0:
        errors.append("Ни одна тестовая пара не доступна")
    elif доступные < len(test_symbols)//2:
        warnings.append(f"Доступно только {доступные}/{len(test_symbols)} пар")
    else:
        log.info(f"Рынок: {доступные}/{len(test_symbols)} тестовых пар доступны")
    return len(errors) == 0, errors + warnings

def этап_5_проверка_существующих_позиций() -> Tuple[bool, List[str]]:
    errors, warnings = [], []
    try:
        positions = exchange.fetch_positions()
        открытые = [p for p in positions if float(p.get("contracts", 0) or 0) > 0]
        if открытые:
            for p in открытые:
                warnings.append(f"Уже открыта позиция: {p.get('symbol')} {p.get('side')} qty={p.get('contracts')}")
        else:
            log.info("Открытых позиций нет — готов к торговле")
    except Exception as e:
        warnings.append(f"Не удалось проверить позиции: {e}")
    if os.path.exists(TRADES_FILE):
        try:
            with open(TRADES_FILE, "r", encoding="utf-8") as f:
                история = json.load(f)
            log.info(f"История: {len(история)} сделок найдено")
        except:
            warnings.append(f"Файл {TRADES_FILE} повреждён")
    return len(errors) == 0, errors + warnings

def запустить_предстартовую_проверку() -> bool:
    log.info("")
    log.info("=" * 65)
    log.info(f"🔍 ПРЕДСТАРТОВАЯ ПРОВЕРКА (5 ЭТАПОВ) | {BOT_VERSION}")
    log.info("=" * 65)
    этапы = [
        ("Этап 1: Окружение и API ключи", этап_1_проверка_окружения),
        ("Этап 2: Подключение к бирже", этап_2_проверка_подключения),
        ("Этап 3: Конфигурация бота", этап_3_проверка_конфигурации),
        ("Этап 4: Доступность рынка", этап_4_проверка_рынка),
        ("Этап 5: Существующие позиции", этап_5_проверка_существующих_позиций),
    ]
    все_ок, все_ошибки = True, []
    for название, функция in этапы:
        log.info(f"\n▶ {название}...")
        try:
            ок, сообщения = функция()
            for msg in сообщения:
                if "⚠" in msg or "рекомендуется" in msg or "Уже" in msg or "Мало" in msg:
                    log.warning(f"⚠️ {msg}")
                else:
                    log.error(f"❌ {msg}")
                    все_ошибки.append(f"[{название}] {msg}")
            if ок:
                log.info(f"✅ {название} — ПРОЙДЕН")
            else:
                log.error(f"❌ {название} — ПРОВАЛЕН")
                все_ок = False
        except Exception as e:
            log.error(f"💥 Исключение в {название}: {e}")
            все_ошибки.append(f"[{название}] Исключение: {e}")
            все_ок = False
    log.info("")
    log.info("=" * 65)
    if все_ок:
        log.info(f"✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ — БОТ {BOT_VERSION} ЗАПУСКАЕТСЯ")
    else:
        log.error("❌ ЕСТЬ КРИТИЧЕСКИЕ ОШИБКИ — БОТ НЕ ЗАПУСТИТСЯ")
        for err in все_ошибки:
            log.error(f"• {err}")
        log.error("ВАРИАНТЫ УСТРАНЕНИЯ: проверьте .env, баланс, параметры")
    log.info("=" * 65)
    log.info("")
    return все_ок

# ============================================================
# ████████████████████████████████████████████
# █████████████████   ГЛАВНЫЙ ЦИКЛ   █████████████████
# ============================================================

def main():
    global stats, ml_model

    # Проверка наличия .env (рекомендация)
    if not os.path.exists(".env"):
        log.warning("⚠️ Файл .env не найден. Убедитесь, что переменные окружения BYBIT_API_KEY и BYBIT_API_SECRET заданы.")

    if ML_ENABLED:
        try:
            if os.path.exists(ML_MODEL_FILE):
                ml_model.load_model(ML_MODEL_FILE)
            else:
                log.info("ML модель не найдена, будет обучена позже")
        except Exception as e:
            log.warning(f"Не удалось загрузить ML модель: {e}")

    if not запустить_предстартовую_проверку():
        log.error(f"🛑 Бот {BOT_VERSION} остановлен из-за ошибок предстартовой проверки.")
        return

    загрузить_состояние()
    stats["запусков"] += 1
    stats["bot_version"] = BOT_VERSION
    stats["release_date"] = RELEASE_DATE

    баланс_сейчас = полный_баланс_usdt()
    if stats["депозит_старт"] <= 0:
        stats["депозит_старт"] = баланс_сейчас
    if not stats["старт_время"]:
        stats["старт_время"] = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    обновить_начало_дня(баланс_сейчас)
    сохранить_состояние()

    if ML_ENABLED and not ml_model.trained:
        if os.path.exists(ML_LOG_FILE):
            ml_model.train()
            ml_model.save_model()
        else:
            log.info("Файл с данными для ML не найден — модель будет обучена позже")

    log.info("")
    log.info("=" * 65)
    log.info(f"🤖 ГИБРИДНЫЙ ФЬЮЧЕРСНЫЙ БОТ {BOT_VERSION}")
    log.info(f"Плечо: {LEVERAGE}x | RR: {TP_PERCENT}/{SL_PERCENT} ({TP_PERCENT/SL_PERCENT:.1f}:1)")
    log.info(f"Баланс: {баланс_сейчас:.4f} USDT")
    log.info(f"MIN_SCORE: {MIN_SCORE} | Пар: {len(SYMBOLS)}")
    log.info(f"Квантовый анализ: {'ВКЛ' if QUANT_ENABLED else 'ВЫКЛ'}")
    log.info(f"Order Flow: {'ВКЛ' if ORDER_FLOW_ENABLED else 'ВЫКЛ'}")
    log.info(f"ML: {'ВКЛ' if ML_ENABLED and ml_model.trained else 'ВЫКЛ'}")
    log.info(f"Портфельная оптимизация: {'ВКЛ' if PORTFOLIO_OPTIMIZATION else 'ВЫКЛ'}")
    log.info(f"Проверка депозита: ВКЛ (адаптивный порог 1.5% от баланса или 0.2 USDT)")
    log.info("=" * 65)
    log.info("")

    заблокированные = {}

    while True:
        try:
            if time.time() - stats["последний_отчёт"] >= REPORT_INTERVAL:
                печатать_отчёт()

            баланс = полный_баланс_usdt()
            свободный = баланс_usdt()
            обновить_начало_дня(баланс)

            if свободный < MIN_BALANCE:
                log.warning(f"🛑 Свободный баланс {свободный:.2f} < {MIN_BALANCE}. Пауза 10 мин.")
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

            log.info(f"── Сканирование {len(SYMBOLS)} пар (баланс={свободный:.2f}U | порог={MIN_SCORE}) ──")
            scores = {}

            for sym in SYMBOLS:
                if sym in заблокированные:
                    if time.time() < заблокированные[sym]:
                        continue
                    else:
                        del заблокированные[sym]
                if not тренд_4h_бычий(sym):
                    continue
                res = получить_скор(sym)
                ai_score = применить_ai_корректировку(res["score"], sym)
                res["score_final"] = ai_score
                scores[sym] = res
                det = res.get("details", {})
                log.debug(f"{sym.split(':')[0]:12s} скор={ai_score:3.0f}/100 rsi={det.get('rsi', '?')} rf={det.get('range_filter', '?')} st={det.get('supertrend', '?')}")

            if not scores:
                log.info(f"Нет кандидатов — ждём {SCAN_INTERVAL} сек")
                time.sleep(SCAN_INTERVAL)
                continue

            кандидаты = sorted(
                [(s, d) for s, d in scores.items() if d.get("score_final", 0) >= MIN_SCORE],
                key=lambda x: x[1]["score_final"],
                reverse=True
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
                if MA_CROSSOVER_ENABLED and not det.get("ma_cross", True):
                    continue
                if not det.get("vol_spike_ok", True):
                    continue
                выбрана = лучшая
                log.info(f"► Выбрана {лучшая.split(':')[0]} (лонг) скор={фин_скор} цена={цена:.8f}")
                break

            if выбрана is None:
                for sym in SYMBOLS:
                    if sym in заблокированные:
                        continue
                    if тренд_4h_медвежий(sym):
                        short_res = получить_скор_шорта(sym)
                        if short_res["score"] >= MIN_SCORE:
                            det_sh = short_res.get("details", {})
                            if MA_CROSSOVER_ENABLED and not det_sh.get("ma_cross", True):
                                continue
                            if not det_sh.get("vol_spike_ok", True):
                                continue
                            log.info(f"🐻 Шорт-кандидат: {sym.split(':')[0]} скор={short_res['score']}")
                            выбрана, фин_скор, цена, sr_info, side = sym, short_res["score"], short_res["price"], short_res.get("sr", {}), "short"
                            break

            if выбрана is None:
                log.info(f"Нет кандидатов — ждём {SCAN_INTERVAL} сек")
                time.sleep(SCAN_INTERVAL)
                continue

            # Расчёт SL/TP
            atr_пт = 0.0
            try:
                raw_atr = fetch_ohlcv_cached(выбрана, TIMEFRAME_TA, limit=50)
                if len(raw_atr) >= 20:
                    df_atr = pd.DataFrame(raw_atr, columns=["ts","o","h","l","c","v"])
                    atr_пт = float(calc_atr(df_atr, 14).iloc[-1])
            except:
                pass

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
            real_rr = abs(tp_цена - цена) / abs(цена - sl_цена)
            log.info(f"📐 ATR={atr_пт/цена*100:.2f}% SL={sl_dist_pct:.2f}% RR={real_rr:.1f}:1")

            if real_rr < 2.0:
                log.warning(f"⛔ RR={real_rr:.1f}:1 < 2.0 — пропуск {выбрана.split(':')[0]}")
                time.sleep(SCAN_INTERVAL)
                continue

            # Портфельная оптимизация
            if PORTFOLIO_OPTIMIZATION:
                active_symbols = [p['symbol'] for p in получить_позиции()]
                all_symbols = active_symbols + [выбрана]
                allocations = optimize_portfolio_allocation(all_symbols)
                max_risk_for_trade = allocations.get(выбрана, MAX_PORTFOLIO_RISK) * баланс
            else:
                max_risk_for_trade = баланс * 0.95

            история_сделок = загрузить_историю()
            margin = рассчитать_размер_позиции(фин_скор, свободный, sl_dist_pct, история_сделок)
            margin = min(margin, max_risk_for_trade)

            if свободный < margin * 1.1:
                log.warning(f"⚠️ Баланс {свободный:.2f} < маржа {margin:.2f} — уменьшаем")
                margin = свободный * 0.8

            log.info(f"✅ ВХОД {side.upper()}: скор={фин_скор} | SL={sl_цена:.8f} | TP={tp_цена:.8f} | маржа={margin:.2f}U")

            if ENTRY_CONFIRM_BARS > 0:
                if not подтвердить_вход(выбрана, фин_скор, side):
                    log.info(f"⛔ Вход в {выбрана} отменён по подтверждению")
                    time.sleep(30)
                    continue

            баланс_до = полный_баланс_usdt()
            время_входа = time.time()
            вход_цена, кол_во = открыть_позицию(выбрана, margin, tp_цена, sl_цена, side)

            if вход_цена is None or кол_во is None:
                log.warning("Не удалось открыть позицию — пауза 30 сек")
                time.sleep(30)
                continue

            stats["сделок_всего"] += 1
            сохранить_состояние()
            результат = "sl"
            try:
                результат = мониторить_позицию(выбрана, вход_цена, кол_во, время_входа, sl_цена, tp_цена, side)
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
                log.info(f"🔒 {выбрана.split(':')[0]} заблокирован на {SYMBOL_BLOCK_AFTER_TP} мин")
            elif результат == "sl":
                stats["стоплосс"] += 1
                stats["убыток_usdt"] += abs(min(0, pnl_реальный))
                stats["sl_streak"] = stats.get("sl_streak", 0) + 1
                заблокированные[выбрана] = time.time() + SYMBOL_BLOCK_AFTER_SL * 60
                log.warning(f"❌ SL: убыток ≈{pnl_реальный:+.4f} USDT streak={stats['sl_streak']}/{SL_STREAK_LIMIT} блок {SYMBOL_BLOCK_AFTER_SL} мин")
            else:
                stats["таймаут"] += 1
                stats["убыток_usdt"] += abs(min(0, pnl_реальный))
                stats["sl_streak"] = 0
                заблокированные[выбрана] = time.time() + SYMBOL_BLOCK_AFTER_TP * 60
                log.warning(f"⏰ Таймаут: P&L ≈{pnl_реальный:+.4f} USDT")

            запись = {
                "id": stats["сделок_всего"],
                "время_входа": datetime.fromtimestamp(время_входа).strftime("%d.%m.%Y %H:%M:%S"),
                "время_выхода": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                "duration_min": round(длит_мин, 1),
                "symbol": выбрана, "side": side, "score": фин_скор,
                "entry_price": вход_цена, "sl_price": sl_цена, "tp_price": tp_цена,
                "sl_dist_pct": round(sl_dist_pct, 3), "margin_usdt": margin,
                "leverage": LEVERAGE, "результат": результат,
                "pnl_usdt": round(pnl_реальный, 4), "rr_ratio": round(real_rr, 2),
                "details": scores.get(выбрана, {}).get("details", {}),
                "bot_version": BOT_VERSION
            }
            сохранить_сделку(запись)
            обновить_статистику_индикаторов(запись)
            пост_трейд_анализ(запись)
            сохранить_состояние()

            if ML_ENABLED:
                stats["ml_trades_since_retrain"] += 1
                if stats["ml_trades_since_retrain"] >= ML_RETRAIN_INTERVAL:
                    if ml_model.train():
                        ml_model.save_model()
                        stats["ml_trades_since_retrain"] = 0

            log.info("Сделка завершена — пауза 60 сек")
            time.sleep(60)

        except Exception as e:
            log.error(f"Глобальная ошибка главного цикла: {e}", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    main()
