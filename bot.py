#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Bybit ГИБРИДНЫЙ БОТ v11.0 — С СИМУЛЯТОРОМ ВХОДА И ЗАЩИТОЙ КАПИТАЛА
================================================================================
КЛЮЧЕВЫЕ ИЗМЕНЕНИЯ v11.0:
1. СИМУЛЯТОР ВХОДА: перед реальной сделкой — виртуальный прогон по всем
   индикаторам на нескольких свечах вперёд (Monte Carlo + исторический бэктест).
2. МОДУЛЬ ЗАЩИТЫ КАПИТАЛА: динамический расчёт SL/TP, режим восстановления
   при просадке, запрет новых входов при высоком риске.
3. УБРАНЫ нерабочие модули: cointegration, order_book depth, volume_profile,
   statsmodels, sklearn, scipy — только то что реально влияет на сигнал.
4. УЛУЧШЕН скоринг: Range Filter теперь полноценно считается, добавлен
   momentum-фильтр и фильтр волатильности.
5. PARTIAL CLOSE только при реальной прибыли > 0.3% (не 0.15%).
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
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

# ============================================================
#                   КОНСТАНТЫ
# ============================================================

SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT", "XRP/USDT:USDT",
    "SOL/USDT:USDT", "ADA/USDT:USDT", "TRX/USDT:USDT", "TON/USDT:USDT",
    "AVAX/USDT:USDT", "DOT/USDT:USDT", "LTC/USDT:USDT", "BCH/USDT:USDT",
    "XLM/USDT:USDT", "NEAR/USDT:USDT", "DOGE/USDT:USDT",
    "1000PEPE/USDT:USDT", "WIF/USDT:USDT", "RENDER/USDT:USDT",
    "TAO/USDT:USDT", "WLD/USDT:USDT", "ONDO/USDT:USDT", "UNI/USDT:USDT",
    "AAVE/USDT:USDT", "ARB/USDT:USDT", "OP/USDT:USDT", "LINK/USDT:USDT",
    "INJ/USDT:USDT", "SUI/USDT:USDT", "APT/USDT:USDT", "HBAR/USDT:USDT",
    "VET/USDT:USDT", "CATI/USDT:USDT",
]

# --- ОСНОВНЫЕ ---
LEVERAGE          = 3
TIMEFRAME_TA      = "5m"
TIMEFRAME_TREND   = "1h"
TIMEFRAME_4H      = "4h"
SCAN_INTERVAL     = 300

# --- ПОРОГИ ---
MIN_SCORE                 = 58
ENTRY_CONFIRM_MIN_SCORE   = 52
SIMULATOR_MIN_WIN_RATE    = 0.55   # симулятор: мин. исторический WR для входа
SIMULATOR_LOOKBACK        = 30     # свечей для бэктеста симулятора

# --- TP / SL ---
TP_PERCENT    = 2.0
SL_PERCENT    = 0.8
MIN_SL_PCT    = 0.6
MAX_SL_PCT    = 1.5
ATR_SL_MULT   = 1.2
ATR_TP_MULT   = 2.5

# --- ЧАСТИЧНЫЙ БЕЗУБЫТОК ---
PARTIAL_BE_ENABLED    = True
PARTIAL_BE_CLOSE_PCT  = 50.0
PARTIAL_BE_PROFIT_PCT = 0.30      # минимум 0.3% профита для частичного закрытия

# --- РИСК ---
BASE_RISK_PCT  = 0.7
MAX_RISK_PCT   = 1.0
# Режим восстановления (при просадке > RECOVERY_DRAWDOWN_PCT)
RECOVERY_RISK_PCT        = 1.5
RECOVERY_DRAWDOWN_PCT    = 5.0    # % от стартового депозита

# --- ТРЕЙЛИНГ ---
MIN_TRAILING_OFFSET   = 0.5
MIN_PROFIT_FOR_TRAIL  = 0.8
RR_EXIT_TRIGGER       = 0.55

# --- MA ---
MA1_LENGTH = 21
MA2_LENGTH = 50

# --- ФИЛЬТРЫ ---
VOLUME_SPIKE_MULT   = 3.0
VOLUME_AVG_PERIOD   = 20
BYBIT_LONG_BLOCK    = 0.63        # блок шортов если long_ratio > 63%

# --- БЛОКИРОВКИ ---
SYMBOL_BLOCK_AFTER_TP       = 90    # мин
SYMBOL_BLOCK_AFTER_SL       = 180   # мин
SYMBOL_MAX_FAIL_ATTEMPTS    = 3
SYMBOL_BLOCK_AFTER_FAIL     = 120   # мин

SL_STREAK_LIMIT      = 2
SL_STREAK_PAUSE      = 5400
SL_STREAK_EXTRA      = 300

MIN_BALANCE          = 5.0
MAX_DRAWDOWN_PCT     = 20.0
DAILY_LOSS_LIMIT_PCT = 3.5
DAILY_LOSS_PAUSE_SEC = 10800

TRADE_MAX_LIFETIME   = 3600
REPORT_INTERVAL      = 1800

# --- S/R ---
SR_PERIOD        = 100
SR_PROXIMITY_PCT = 0.5
SR_MIN_TOUCHES   = 3
SR_CLUSTER_TOL   = 0.005
SR_BLOCK_DIST    = 0.3

# --- API ---
API_CALL_DELAY       = 0.25
API_RATE_LIMIT_PAUSE = 5

BYBIT_FEE  = 0.00055
STATE_FILE  = "state_v11.json"
TRADES_FILE = "trades_v11.json"

# ============================================================
#                       ЛОГИРОВАНИЕ
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]  %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot_v11.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ============================================================
#                         БИРЖА
# ============================================================

exchange = ccxt.bybit({
    "apiKey":    os.getenv("BYBIT_API_KEY"),
    "secret":    os.getenv("BYBIT_API_SECRET"),
    "enableRateLimit": True,
    "options": {"defaultType": "linear"},
})

# ============================================================
#                    СТАТИСТИКА
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
    "sim_отфильтровано": 0,
}

# ============================================================
#              БЕЗОПАСНЫЕ API-ВЫЗОВЫ
# ============================================================

def safe_api_call(func, *args, retries=3, delay=1.0, **kwargs):
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except ccxt.RateLimitExceeded:
            log.warning(f"Rate limit — пауза {API_RATE_LIMIT_PAUSE}с")
            time.sleep(API_RATE_LIMIT_PAUSE)
        except ccxt.NetworkError as e:
            log.warning(f"Сеть: {e}, пауза {delay}с")
            time.sleep(delay); delay *= 2
        except Exception as e:
            log.error(f"API: {e} (попытка {attempt+1}/{retries})")
            if attempt < retries - 1:
                time.sleep(delay); delay *= 2
            else:
                raise
    return None

def safe_fetch_ohlcv(symbol: str, tf: str, limit: int = 300) -> List:
    try:
        r = safe_api_call(exchange.fetch_ohlcv, symbol, tf, limit=limit, retries=3)
        return r if r else []
    except Exception: return []

def safe_fetch_ticker(symbol: str) -> Optional[dict]:
    try:
        return safe_api_call(exchange.fetch_ticker, symbol, retries=3)
    except Exception: return None

def safe_fetch_positions(symbols=None) -> List[dict]:
    try:
        r = safe_api_call(exchange.fetch_positions, symbols, retries=3) if symbols \
            else safe_api_call(exchange.fetch_positions, retries=3)
        return r if r else []
    except Exception: return []

# ============================================================
#              ИНДИКАТОРЫ
# ============================================================

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1/n, adjust=False).mean()

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    gain = _rma(d.clip(lower=0), period)
    loss = _rma((-d).clip(lower=0), period)
    rs   = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(close: pd.Series):
    ml = _ema(close, 12) - _ema(close, 26)
    sl = _ema(ml, 9)
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
        else: trend.iloc[i] = pt
    return trend == 1, trend == -1

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
    return filt, up, down

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

def calc_stochastic(df: pd.DataFrame, k: int = 14, d: int = 3, smooth: int = 3):
    lo = df["l"].rolling(k).min()
    hi = df["h"].rolling(k).max()
    ks = (100 * (df["c"] - lo) / (hi - lo + 1e-10)).rolling(smooth).mean()
    return ks, ks.rolling(d).mean()

def calc_support_resistance(df: pd.DataFrame) -> dict:
    df_sr = df.tail(SR_PERIOD).reset_index(drop=True)
    highs, lows = df_sr["h"].values, df_sr["l"].values
    close = float(df["c"].iloc[-1])
    raw_res, raw_sup = [], []
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
           highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            raw_res.append(highs[i])
        if lows[i]  < lows[i-1]  and lows[i]  < lows[i-2]  and \
           lows[i]  < lows[i+1]  and lows[i]  < lows[i+2]:
            raw_sup.append(lows[i])

    def cluster(levels):
        if not levels: return []
        levels = sorted(levels); out = []; cur = [levels[0]]
        for lvl in levels[1:]:
            if (lvl - cur[0]) / (cur[0] + 1e-10) < SR_CLUSTER_TOL: cur.append(lvl)
            else: out.append((float(np.mean(cur)), len(cur))); cur = [lvl]
        out.append((float(np.mean(cur)), len(cur)))
        return out

    res_cl  = cluster(raw_res)
    sup_cl  = cluster(raw_sup)
    res_a   = sorted([(p,n) for p,n in res_cl if p > close], key=lambda x: x[0])
    sup_b   = sorted([(p,n) for p,n in sup_cl if p < close], key=lambda x: x[0], reverse=True)
    nr, rn  = res_a[0] if res_a else (close*1.05, 0)
    ns, sn  = sup_b[0] if sup_b else (close*0.95, 0)
    dr = (nr - close) / close * 100
    ds = (close - ns) / close * 100
    return {
        "support": ns, "resistance": nr,
        "dist_to_sup_pct": round(ds, 2), "dist_to_res_pct": round(dr, 2),
        "sup_cluster": sn, "res_cluster": rn,
        "near_support":    ds < SR_PROXIMITY_PCT and sn >= SR_MIN_TOUCHES,
        "near_resistance": dr < SR_PROXIMITY_PCT and rn >= SR_MIN_TOUCHES,
    }

# ============================================================
#           ВСПОМОГАТЕЛЬНЫЕ ФИЛЬТРЫ
# ============================================================

def volume_spike_guard(df: pd.DataFrame) -> bool:
    try:
        avg = df["v"].rolling(VOLUME_AVG_PERIOD).mean().iloc[-1]
        ratio = df["v"].iloc[-1] / (avg + 1e-10)
        if ratio > VOLUME_SPIKE_MULT:
            log.info(f"Volume spike {ratio:.1f}x — пропуск")
            return False
        return True
    except Exception: return True

def get_bybit_ratio(symbol: str) -> dict:
    result = {"signal": "neutral", "long_ratio": 0.5, "available": False}
    try:
        coin = symbol.split("/")[0]
        url  = f"https://api.bybit.com/v5/market/account-ratio?category=linear&symbol={coin}USDT&period=1h&limit=1"
        resp = requests.get(url, timeout=5).json()
        if resp.get("retCode") == 0:
            items = resp.get("result", {}).get("list", [])
            if items:
                lr = float(items[0].get("buyRatio", 0.5))
                result.update({"long_ratio": lr, "available": True,
                                "signal": "bullish" if lr > 0.6 else ("bearish" if lr < 0.4 else "neutral")})
    except Exception: pass
    return result

def trend_4h(symbol: str, bullish: bool = True) -> bool:
    try:
        raw = safe_fetch_ohlcv(symbol, TIMEFRAME_4H, limit=60)
        if len(raw) < 55: return False
        df = pd.DataFrame(raw, columns=["ts","o","h","l","c","v"])
        e20 = _ema(df["c"], 20).iloc[-1]
        e50 = _ema(df["c"], 50).iloc[-1]
        return (e20 > e50) if bullish else (e20 < e50)
    except Exception: return False

# ============================================================
#     ██████╗  СИМУЛЯТОР ВХОДА  ██████╗
# ============================================================

class TradeSimulator:
    """
    Виртуальный прогон стратегии на исторических данных перед реальным входом.
    Цель: убедиться что при текущих индикаторных условиях стратегия исторически
    прибыльна на данном символе.
    """

    def __init__(self, lookback: int = SIMULATOR_LOOKBACK):
        self.lookback = lookback

    def _build_signal(self, df: pd.DataFrame, idx: int, side: str) -> bool:
        """Проверяет наличие сигнала на свече idx (не последней)."""
        if idx < 50: return False
        sl  = df.iloc[:idx+1]
        c   = sl["c"]
        try:
            rsi_v  = calc_rsi(c).iloc[-1]
            ml, sg, _ = calc_macd(c)
            macd_bull = ml.iloc[-1] > sg.iloc[-1]
            _, rf_up, rf_dn = calc_range_filter(sl)
            st_up, st_dn   = calc_supertrend(sl)
            adx_v, pdi, mdi = calc_adx(sl)
            adx_val = adx_v.iloc[-1]

            if side == "long":
                return (
                    rsi_v < 55 and
                    macd_bull and
                    rf_up.iloc[-1] and
                    st_up.iloc[-1] and
                    adx_val > 18 and pdi.iloc[-1] > mdi.iloc[-1]
                )
            else:
                return (
                    rsi_v > 45 and
                    not macd_bull and
                    rf_dn.iloc[-1] and
                    st_dn.iloc[-1] and
                    adx_val > 18 and mdi.iloc[-1] > pdi.iloc[-1]
                )
        except Exception:
            return False

    def run(self, df: pd.DataFrame, side: str,
            tp_pct: float, sl_pct: float) -> dict:
        """
        Бэктест на последних `lookback` свечах.
        Возвращает: wins, losses, win_rate, avg_pnl, max_dd, passed.
        """
        wins = losses = 0
        pnls: List[float] = []
        trades_tried = 0

        # сдвигаемся по истории, ищем сигналы
        start = max(60, len(df) - self.lookback - 60)
        end   = len(df) - 5   # не трогаем последние 5 свечей (они "будущее")

        for i in range(start, end):
            if not self._build_signal(df, i, side):
                continue
            trades_tried += 1
            entry = float(df["c"].iloc[i])
            if side == "long":
                tp_price = entry * (1 + tp_pct / 100)
                sl_price = entry * (1 - sl_pct / 100)
            else:
                tp_price = entry * (1 - tp_pct / 100)
                sl_price = entry * (1 + sl_pct / 100)

            # смотрим следующие 20 свечей
            result = "timeout"
            for j in range(i+1, min(i+21, len(df))):
                hi = float(df["h"].iloc[j])
                lo = float(df["l"].iloc[j])
                if side == "long":
                    if lo <= sl_price: result = "sl"; break
                    if hi >= tp_price: result = "tp"; break
                else:
                    if hi >= sl_price: result = "sl"; break
                    if lo <= tp_price: result = "tp"; break

            if result == "tp":
                wins += 1
                pnls.append(tp_pct * LEVERAGE / 100)
            elif result == "sl":
                losses += 1
                pnls.append(-sl_pct * LEVERAGE / 100)
            # timeout — не считаем

        total = wins + losses
        if total == 0:
            return {"wins": 0, "losses": 0, "win_rate": 0.0,
                    "avg_pnl": 0.0, "max_dd": 0.0,
                    "passed": False, "trades_tried": trades_tried,
                    "reason": "нет_сигналов"}

        wr      = wins / total
        avg_pnl = float(np.mean(pnls)) if pnls else 0.0

        # максимальная просадка в серии
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        max_dd = float(abs(min(cumulative - running_max))) if len(pnls) > 1 else 0.0

        passed = (wr >= SIMULATOR_MIN_WIN_RATE and avg_pnl > 0 and total >= 3)

        return {
            "wins": wins, "losses": losses, "win_rate": round(wr, 2),
            "avg_pnl": round(avg_pnl, 4), "max_dd": round(max_dd, 4),
            "passed": passed, "trades_tried": trades_tried,
            "reason": "OK" if passed else
                      f"WR={wr:.0%}<{SIMULATOR_MIN_WIN_RATE:.0%} или avg_pnl={avg_pnl:.4f}<=0"
        }

    def log_result(self, symbol: str, side: str, result: dict):
        status = "✅ СИМУЛЯТОР ПРОЙДЕН" if result["passed"] else "⛔ СИМУЛЯТОР ОТКЛОНИЛ"
        log.info(
            f"[SIM] {status} | {symbol.split(':')[0]} {side.upper()} | "
            f"WR={result['win_rate']:.0%} ({result['wins']}W/{result['losses']}L) | "
            f"AvgPnL={result['avg_pnl']:.4f} | MaxDD={result['max_dd']:.4f} | "
            f"Сигналов найдено: {result['trades_tried']} | {result['reason']}"
        )

simulator = TradeSimulator()

# ============================================================
#   ██████╗  МОДУЛЬ ЗАЩИТЫ КАПИТАЛА  ██████╗
# ============================================================

class CapitalProtection:
    """
    Динамическое управление рисками:
    - Нормальный режим: стандартный риск
    - Режим ОСТОРОЖНОСТЬ (просадка > 3%): снижение риска
    - Режим ВОССТАНОВЛЕНИЕ (просадка > 5%): умеренное увеличение риска
      на ТОЛЬКО высококачественные сетапы (score >= 85)
    - Режим СТОП (просадка > MAX_DRAWDOWN_PCT): полная остановка
    """

    def __init__(self):
        self.mode = "normal"      # normal / caution / recovery / stop
        self.start_balance = 0.0
        self.peak_balance  = 0.0
        self.current_dd    = 0.0

    def update(self, balance: float):
        if self.start_balance <= 0:
            self.start_balance = balance
        self.peak_balance = max(self.peak_balance, balance)
        dd_from_peak  = (self.peak_balance - balance) / self.peak_balance * 100 if self.peak_balance > 0 else 0
        dd_from_start = (self.start_balance - balance) / self.start_balance * 100 if self.start_balance > 0 else 0
        self.current_dd = max(dd_from_peak, dd_from_start)

        if self.current_dd >= MAX_DRAWDOWN_PCT:
            self.mode = "stop"
        elif self.current_dd >= RECOVERY_DRAWDOWN_PCT:
            self.mode = "recovery"
        elif self.current_dd >= 3.0:
            self.mode = "caution"
        else:
            self.mode = "normal"

    def get_risk_pct(self, score: int) -> float:
        if self.mode == "stop":     return 0.0
        if self.mode == "caution":  return BASE_RISK_PCT * 0.6
        if self.mode == "recovery":
            if score >= 85: return RECOVERY_RISK_PCT
            return BASE_RISK_PCT * 0.5   # плохой сетап в режиме восстановления — минимум
        # normal
        factor = max(0, (score - MIN_SCORE)) / (100 - MIN_SCORE)
        return round(min(BASE_RISK_PCT + (MAX_RISK_PCT - BASE_RISK_PCT) * factor, MAX_RISK_PCT), 2)

    def can_trade(self, score: int) -> Tuple[bool, str]:
        if self.mode == "stop":
            return False, f"СТОП: просадка {self.current_dd:.1f}% >= {MAX_DRAWDOWN_PCT}%"
        if self.mode == "recovery" and score < 85:
            return False, f"ВОССТАНОВЛЕНИЕ: нужен скор >= 85, текущий {score}"
        return True, f"OK [{self.mode}] DD={self.current_dd:.1f}%"

    def adjust_sl_tp(self, entry: float, side: str, atr: float,
                     score: int) -> Tuple[float, float, float, float]:
        """
        Динамический расчёт SL/TP в зависимости от режима и ATR.
        Возвращает: sl_price, tp_price, sl_pct, tp_pct
        """
        atr_pct = (atr / entry) * 100 if entry > 0 else SL_PERCENT

        # Базовый SL по ATR
        sl_pct_raw = atr_pct * ATR_SL_MULT
        sl_pct     = float(np.clip(sl_pct_raw, MIN_SL_PCT, MAX_SL_PCT))

        # В режиме осторожности — уже SL
        if self.mode == "caution":
            sl_pct = max(MIN_SL_PCT, sl_pct * 0.8)

        # TP минимум RR 2.5:1
        tp_pct = max(TP_PERCENT, sl_pct * 2.5)
        # В режиме восстановления — более агрессивный TP (быстрее фиксируем)
        if self.mode == "recovery":
            tp_pct = sl_pct * 2.0

        if side == "long":
            sl_price = entry * (1 - sl_pct / 100)
            tp_price = entry * (1 + tp_pct / 100)
        else:
            sl_price = entry * (1 + sl_pct / 100)
            tp_price = entry * (1 - tp_pct / 100)

        return sl_price, tp_price, sl_pct, tp_pct

    def status_str(self) -> str:
        icons = {"normal": "🟢", "caution": "🟡", "recovery": "🔴", "stop": "⛔"}
        return f"{icons.get(self.mode,'?')} [{self.mode.upper()}] DD={self.current_dd:.1f}%"

capital = CapitalProtection()

# ============================================================
#               СКОРИНГОВАЯ СИСТЕМА
# ============================================================

def _score_common(df_ta: pd.DataFrame, df_1h: pd.DataFrame,
                  side: str) -> Tuple[int, dict]:
    """Общий расчёт скора для лонга и шорта."""
    score   = 0
    details = {}
    c_ta    = df_ta["c"]
    c_1h    = df_1h["c"]

    # --- RSI ---
    rsi_val = calc_rsi(c_ta).iloc[-1]
    details["rsi"] = round(rsi_val, 1)
    if side == "long":
        if   25 <= rsi_val <= 42: score += 22
        elif 42 <  rsi_val <= 52: score += 12
        elif rsi_val < 25:        score += 8
        elif rsi_val > 65:        score -= 20
    else:
        if   rsi_val >= 70: score += 25
        elif rsi_val >= 60: score += 15
        elif rsi_val <= 40: score -= 15

    # --- RSI 1h ---
    rsi_1h = calc_rsi(c_1h).iloc[-1]
    details["rsi_1h"] = round(rsi_1h, 1)
    if side == "long":
        if rsi_1h < 50:  score += 10
        elif rsi_1h < 58: score += 5
    else:
        if rsi_1h >= 60: score += 12
        elif rsi_1h >= 55: score += 6
        elif rsi_1h <= 45: score -= 10

    # --- MACD ---
    ml, sg, _ = calc_macd(c_ta)
    macd_bull  = ml.iloc[-1] > sg.iloc[-1]
    macd_cross = macd_bull and ml.iloc[-2] <= sg.iloc[-2]
    details["macd"] = "бычий" if macd_bull else "медвежий"
    if side == "long":
        if macd_cross: score += 20
        elif macd_bull: score += 8
        else: score -= 5
    else:
        death = (not macd_bull) and ml.iloc[-2] >= sg.iloc[-2]
        if death:        score += 22
        elif not macd_bull: score += 10
        else: score -= 5

    # --- Range Filter ---
    _, rf_up, rf_dn = calc_range_filter(df_ta)
    rf_up_now = bool(rf_up.iloc[-1])
    rf_dn_now = bool(rf_dn.iloc[-1])
    details["range_filter"] = "вверх" if rf_up_now else ("вниз" if rf_dn_now else "бок")
    if side == "long":
        if rf_up_now: score += 18
        elif rf_dn_now: score -= 12
    else:
        if rf_dn_now: score += 15
        elif rf_up_now: score -= 10

    # --- Supertrend ---
    st_up, st_dn = calc_supertrend(df_ta)
    details["supertrend"] = "вверх" if st_up.iloc[-1] else "вниз"
    if side == "long":
        if st_up.iloc[-1]: score += 14
        else: score -= 10
    else:
        if st_dn.iloc[-1]: score += 14
        else: score -= 10

    # --- EMA тренд 1h ---
    e50  = _ema(c_1h, 50).iloc[-1]
    e200 = _ema(c_1h, 200).iloc[-1]
    details["тренд_1h"] = "бычий" if e50 > e200 else "медвежий"
    if side == "long":
        if e50 > e200: score += 12
        else:          score -= 8
    else:
        if e50 < e200: score += 12
        else:          score -= 8

    # --- ADX ---
    adx, pdi, mdi = calc_adx(df_ta)
    adx_val = adx.iloc[-1]
    details["adx"] = round(adx_val, 1)
    if side == "long":
        if adx_val > 25 and pdi.iloc[-1] > mdi.iloc[-1]: score += 12
        elif adx_val > 20 and pdi.iloc[-1] > mdi.iloc[-1]: score += 5
        elif adx_val < 15: score -= 5   # слабый тренд — штраф
    else:
        if adx_val > 25 and mdi.iloc[-1] > pdi.iloc[-1]: score += 12
        elif adx_val > 20 and mdi.iloc[-1] > pdi.iloc[-1]: score += 5
        elif adx_val < 15: score -= 5

    # --- Stochastic ---
    k_ser, _ = calc_stochastic(df_ta)
    k_val    = k_ser.iloc[-1]
    details["stoch_k"] = round(k_val, 1)
    if side == "long":
        if k_val < 20:  score += 12
        elif k_val < 40: score += 6
        elif k_val > 80: score -= 10
    else:
        if k_val >= 80: score += 12
        elif k_val >= 65: score += 6
        elif k_val <= 20: score -= 10

    # --- Объём ---
    vol_avg = df_ta["v"].rolling(VOLUME_AVG_PERIOD).mean().iloc[-1]
    vol_r   = df_ta["v"].iloc[-1] / (vol_avg + 1e-10)
    details["объём_ratio"] = round(vol_r, 2)
    if vol_r > 1.5: score += 8
    elif vol_r > 1.2: score += 4
    elif vol_r < 0.5: score -= 5

    # --- S/R ---
    sr = calc_support_resistance(df_ta)
    details.update({"support": sr["support"], "resistance": sr["resistance"],
                    "dist_sup": sr["dist_to_sup_pct"], "dist_res": sr["dist_to_res_pct"]})
    if side == "long":
        if sr["near_support"]:
            score += 18
            details["sr_signal"] = f"у поддержки ✅ ({sr['sup_cluster']} каc)"
        elif sr["near_resistance"]:
            score -= 28
            details["sr_signal"] = f"у сопротивления ❌ ({sr['res_cluster']} каc)"
        else:
            details["sr_signal"] = f"нейтр (s={sr['dist_to_sup_pct']:.2f}% r={sr['dist_to_res_pct']:.2f}%)"
    else:
        if sr["near_resistance"]:
            score += 22
            details["sr_signal"] = f"у сопротивления ✅ ({sr['res_cluster']} каc)"
        elif sr["near_support"]:
            score -= 22
            details["sr_signal"] = f"у поддержки ❌ ({sr['sup_cluster']} каc)"
        else:
            details["sr_signal"] = f"нейтр"

    # --- 3 свечи подряд ---
    last3_bear = all(df_ta["c"].iloc[-i] < df_ta["o"].iloc[-i] for i in range(1, 4))
    last3_bull = all(df_ta["c"].iloc[-i] > df_ta["o"].iloc[-i] for i in range(1, 4))
    if side == "long" and last3_bear:
        score -= 20; details["свечи_3"] = "3 медв ❌"
    if side == "short" and last3_bear:
        score += 14; details["свечи_3"] = "3 медв ✅"
    if side == "long" and last3_bull:
        score += 6;  details["свечи_3"] = "3 бычьих ✅"

    # --- MA кроссовер ---
    ma1 = _ema(c_ta, MA1_LENGTH)
    ma2 = _ema(c_ta, MA2_LENGTH)
    ma_ok = (ma1.iloc[-1] > ma2.iloc[-1]) if side == "long" else (ma1.iloc[-1] < ma2.iloc[-1])
    details["ma_cross"] = ma_ok
    if not ma_ok: score -= 8

    # --- Momentum (5-свечной) ---
    m5 = (float(c_ta.iloc[-1]) - float(c_ta.iloc[-5])) / float(c_ta.iloc[-5]) * 100
    details["momentum_5"] = round(m5, 2)
    if side == "long" and m5 > 0.5:   score += 6
    if side == "short" and m5 < -0.5: score += 6

    details["vol_spike_ok"] = volume_spike_guard(df_ta)
    return max(0, min(100, score)), details

def get_score(symbol: str, side: str = "long") -> dict:
    try:
        raw_ta = safe_fetch_ohlcv(symbol, TIMEFRAME_TA, limit=300)
        time.sleep(API_CALL_DELAY)
        raw_1h = safe_fetch_ohlcv(symbol, TIMEFRAME_TREND, limit=300)
        if len(raw_ta) < 100 or len(raw_1h) < 100:
            return {"score": 0, "details": {}, "price": 0, "sr": {}, "df_ta": None}
        cols   = ["ts","o","h","l","c","v"]
        df_ta  = pd.DataFrame(raw_ta, columns=cols).reset_index(drop=True)
        df_1h  = pd.DataFrame(raw_1h, columns=cols).reset_index(drop=True)
        price  = float(df_ta["c"].iloc[-1])
        score, details = _score_common(df_ta, df_1h, side)
        sr     = calc_support_resistance(df_ta)
        return {"score": score, "details": details, "price": price, "sr": sr, "df_ta": df_ta}
    except Exception as e:
        log.warning(f"Ошибка скора {symbol}: {e}")
        return {"score": 0, "details": {}, "price": 0, "sr": {}, "df_ta": None}

def apply_ratio_correction(score: int, symbol: str, side: str) -> int:
    ai = get_bybit_ratio(symbol)
    if not ai["available"]: return score
    lr = ai["long_ratio"]
    log.info(f"Bybit ratio: long={lr:.1%} сигнал={ai['signal']}")
    if side == "long":
        if ai["signal"] == "bullish": return min(100, score + 5)
        if ai["signal"] == "bearish": return max(0, score - 15)
    else:
        if lr > 0.70: return max(0, score - 22)
        if lr > 0.65: return max(0, score - 12)
        if lr < 0.45: return min(100, score + 10)
    return score

# ============================================================
#               УПРАВЛЕНИЕ ПЛЕЧОМ
# ============================================================

def set_leverage(symbol: str, lev: int) -> bool:
    try:
        exchange.set_leverage(lev, symbol, params={"buyLeverage": lev, "sellLeverage": lev})
        return True
    except Exception as e1:
        if "110043" in str(e1) or "leverage not modified" in str(e1).lower():
            return True
    try:
        coin = symbol.replace("/","").replace(":USDT","")
        exchange.private_post_v5_position_set_leverage({
            "category": "linear", "symbol": coin,
            "buyLeverage": str(lev), "sellLeverage": str(lev),
        })
        return True
    except Exception as e2:
        if "110043" in str(e2) or "leverage not modified" in str(e2).lower():
            return True
    log.warning(f"Плечо не установлено явно для {symbol}")
    return True

def update_sl(symbol: str, new_sl: float) -> bool:
    try:
        sl_str  = exchange.price_to_precision(symbol, new_sl)
        coin    = symbol.replace("/","").replace(":USDT","")
        exchange.private_post_v5_position_trading_stop({
            "category": "linear", "symbol": coin,
            "stopLoss": sl_str, "slTriggerBy": "MarkPrice", "positionIdx": "0",
        })
        log.info(f"SL обновлён → {sl_str}")
        return True
    except Exception as e:
        log.warning(f"Не удалось обновить SL: {e}")
        return False

# ============================================================
#         ОТКРЫТИЕ / ЗАКРЫТИЕ ПОЗИЦИИ
# ============================================================

def open_position(symbol: str, margin_usdt: float,
                  tp_price: float, sl_price: float,
                  side: str = "long") -> Tuple[Optional[float], Optional[float]]:
    try:
        set_leverage(symbol, LEVERAGE)
        ticker = safe_fetch_ticker(symbol)
        if not ticker: return None, None
        price = float(ticker["last"])
        qty   = float(exchange.amount_to_precision(symbol, margin_usdt * LEVERAGE / price))
        if qty <= 0: return None, None

        if side == "long":
            sl_price = min(sl_price, price * (1 - MIN_SL_PCT/100))
            tp_price = max(tp_price, price * (1 + TP_PERCENT/100))
        else:
            sl_price = max(sl_price, price * (1 + MIN_SL_PCT/100))
            tp_price = min(tp_price, price * (1 - TP_PERCENT/100))

        tp_str = exchange.price_to_precision(symbol, tp_price)
        sl_str = exchange.price_to_precision(symbol, sl_price)
        bs     = "buy" if side == "long" else "sell"
        log.info(f"Открываем {side} {symbol}: qty={qty} маржа≈{margin_usdt:.2f}U "
                 f"плечо={LEVERAGE}x TP={tp_str} SL={sl_str}")
        order  = exchange.create_market_order(symbol, bs, qty, params={
            "takeProfit": float(tp_str), "stopLoss": float(sl_str)
        })

        entry_price = None
        for field in ("average", "price"):
            try:
                v = order.get(field)
                if v: entry_price = float(v); break
            except Exception: pass

        if not entry_price or entry_price <= 0:
            time.sleep(2)
            positions = safe_fetch_positions([symbol])
            for pos in positions:
                if float(pos.get("contracts", 0) or 0) > 0:
                    ep = pos.get("entryPrice") or pos.get("avgCost")
                    if ep: entry_price = float(ep); break

        if not entry_price or entry_price <= 0:
            entry_price = price

        log.info(f"{side.upper()} открыт: {qty} {symbol} @ ~{entry_price:.8f}")
        return entry_price, qty
    except Exception as e:
        log.error(f"Ошибка открытия {side} {symbol}: {e}")
        return None, None

def close_position(symbol: str, qty: float, side: str) -> bool:
    cs = "sell" if side == "long" else "buy"
    for attempt in range(3):
        try:
            exchange.create_market_order(symbol, cs, qty, params={"reduceOnly": True})
            time.sleep(3)
            positions = safe_fetch_positions([symbol])
            active = [p for p in positions
                      if float(p.get("contracts", 0) or 0) > 0 and p.get("side") == side]
            if not active:
                log.info(f"Позиция {symbol} закрыта")
                return True
            time.sleep(2)
        except Exception as e:
            log.warning(f"Закрытие {symbol} попытка {attempt+1}: {e}")
            time.sleep(2)
    log.error(f"Не удалось закрыть {symbol} после 3 попыток")
    return False

# ============================================================
#            МОНИТОРИНГ ПОЗИЦИИ
# ============================================================

def monitor_position(symbol: str, entry_price: float, qty: float,
                     opened_at: float, sl_price: float,
                     tp_price: float, side: str = "long") -> Tuple[str, float]:
    deadline = opened_at + TRADE_MAX_LIFETIME
    coin     = symbol.split("/")[0]

    # ATR для трейлинга
    trailing_offset = MIN_TRAILING_OFFSET / 100
    try:
        raw = safe_fetch_ohlcv(symbol, TIMEFRAME_TA, limit=50)
        if len(raw) >= 20:
            df  = pd.DataFrame(raw, columns=["ts","o","h","l","c","v"])
            atr = calc_atr(df, 14).iloc[-1]
            trailing_offset = max(MIN_TRAILING_OFFSET, (atr/entry_price)*100*1.5) / 100
    except Exception: pass

    if side == "long":
        rr_trigger = entry_price + (tp_price - entry_price) * RR_EXIT_TRIGGER
    else:
        rr_trigger = entry_price - (entry_price - tp_price) * RR_EXIT_TRIGGER

    log.info(f"rrTrigger={rr_trigger:.8f} trailing_offset={trailing_offset*100:.2f}%")

    phase          = 1
    cur_sl         = sl_price
    peak_price     = entry_price
    trailing_on    = False
    partial_done   = False
    accumulated    = 0.0

    log.info(f"Мониторинг {coin} {side} вход={entry_price:.8f} SL={sl_price:.8f} TP={tp_price:.8f}")

    while True:
        now = time.time()
        if now >= deadline:
            log.warning("Дедлайн — принудительное закрытие")
            close_position(symbol, qty, side)
            return "таймаут", accumulated
        time.sleep(15)

        try:
            positions = safe_fetch_positions([symbol])
            active    = [p for p in positions
                         if float(p.get("contracts", 0) or 0) > 0 and p.get("side") == side]
            if not active:
                ticker    = safe_fetch_ticker(symbol)
                cur_price = float(ticker["last"]) if ticker else entry_price
                hit_tp    = (cur_price >= entry_price * (1 + TP_PERCENT/100*0.7)) if side == "long" \
                            else (cur_price <= entry_price * (1 - TP_PERCENT/100*0.7))
                return ("tp" if (hit_tp or phase >= 2) else "sl"), accumulated

            pos        = active[0]
            ticker     = safe_fetch_ticker(symbol)
            cur_price  = float(ticker["last"]) if ticker else entry_price
            qty_actual = abs(float(pos.get("contracts", 0) or 0))
            pnl_real   = float(pos.get("unrealizedPnl", 0) or 0)
            pnl_pct    = ((cur_price - entry_price) / entry_price * 100) if side == "long" \
                         else ((entry_price - cur_price) / entry_price * 100)
            до_дед     = int(deadline - now)

            # Частичный безубыток
            if PARTIAL_BE_ENABLED and not partial_done and pnl_pct >= PARTIAL_BE_PROFIT_PCT:
                cqty   = qty_actual * (PARTIAL_BE_CLOSE_PCT / 100)
                cs     = "sell" if side == "long" else "buy"
                try:
                    exchange.create_market_order(symbol, cs, cqty, params={"reduceOnly": True})
                    partial_pnl = (cur_price - entry_price) * cqty if side == "long" \
                                  else (entry_price - cur_price) * cqty
                    accumulated += partial_pnl
                    qty_actual  -= cqty
                    new_be = entry_price * (1 + BYBIT_FEE*2 + 0.0003) if side == "long" \
                             else entry_price * (1 - BYBIT_FEE*2 - 0.0003)
                    if update_sl(symbol, new_be):
                        cur_sl = new_be
                    partial_done = True
                    log.info(f"Частичный BE: закрыто {cqty:.4f} ({PARTIAL_BE_CLOSE_PCT:.0f}%) "
                             f"@ {cur_price:.8f} PnL≈{partial_pnl:+.4f}U")
                except Exception as e:
                    log.warning(f"Ошибка частичного закрытия: {e}")

            # Полный безубыток
            if phase == 1 and pnl_pct >= 0.3:
                new_be = entry_price * (1 + BYBIT_FEE*2 + 0.0003) if side == "long" \
                         else entry_price * (1 - BYBIT_FEE*2 - 0.0003)
                if update_sl(symbol, new_be):
                    phase, cur_sl, peak_price = 2, new_be, cur_price
                    log.info(f"БЕЗУБЫТОК! SL → {new_be:.8f}")

            # Активация трейлинга
            if not trailing_on and phase >= 2:
                trailing_on = (cur_price >= rr_trigger) if side == "long" \
                              else (cur_price <= rr_trigger)
                if trailing_on:
                    log.info(f"Трейлинг активирован @ {cur_price:.8f}")

            # Трейлинг
            if trailing_on and phase >= 2 and pnl_pct >= MIN_PROFIT_FOR_TRAIL:
                if side == "long":
                    if cur_price > peak_price: peak_price = cur_price
                    new_trail = peak_price * (1 - trailing_offset)
                    if new_trail > cur_sl and update_sl(symbol, new_trail):
                        cur_sl = new_trail
                        log.info(f"ТРЕЙЛИНГ: пик={peak_price:.8f} SL={new_trail:.8f}")
                else:
                    if cur_price < peak_price: peak_price = cur_price
                    new_trail = peak_price * (1 + trailing_offset)
                    if new_trail < cur_sl and update_sl(symbol, new_trail):
                        cur_sl = new_trail
                        log.info(f"ТРЕЙЛИНГ: пик={peak_price:.8f} SL={new_trail:.8f}")

            log.info(f"[{coin}] {cur_price:.8f} P&L={pnl_pct:+.2f}% ({pnl_real:+.4f}U) "
                     f"SL={cur_sl:.8f} фаза={phase} дед={до_дед}с "
                     f"{'[частично]' if partial_done else ''} {capital.status_str()}")

        except Exception as e:
            log.warning(f"Ошибка мониторинга: {e}")

    return "sl", accumulated

# ============================================================
#              ПОДТВЕРЖДЕНИЕ ВХОДА
# ============================================================

def confirm_entry(symbol: str, initial_score: int, side: str) -> bool:
    tf_sec = {"1m": 60, "3m": 180, "5m": 300, "15m": 900}
    wait   = tf_sec.get(TIMEFRAME_TA, 300)
    log.info(f"Подтверждение: ждём {wait}с...")
    time.sleep(wait)
    res        = get_score(symbol, side)
    new_score  = res["score"]
    ai_score   = apply_ratio_correction(new_score, symbol, side)
    log.info(f"Перепроверка: {initial_score} → {ai_score} (мин={ENTRY_CONFIRM_MIN_SCORE})")
    if ai_score < ENTRY_CONFIRM_MIN_SCORE:
        log.info(f"Подтверждение не прошло: скор {ai_score}")
        return False
    if not res.get("details", {}).get("vol_spike_ok", True):
        log.info("Подтверждение не прошло: volume spike")
        return False
    log.info(f"Вход подтверждён. Скор {ai_score}/100")
    return True

# ============================================================
#            ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def load_history() -> List[dict]:
    if not os.path.exists(TRADES_FILE): return []
    try:
        with open(TRADES_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except Exception: return []

def save_trade(record: dict):
    history = load_history()
    history.append(record)
    try:
        with open(TRADES_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2, default=str)
        log.info(f"Сделка #{record['id']} сохранена")
    except Exception as e:
        log.warning(f"Не удалось сохранить сделку: {e}")

def save_state():
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
    except Exception: pass

def load_state():
    global stats
    if not os.path.exists(STATE_FILE): return
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        for k in stats:
            if k in saved: stats[k] = saved[k]
        log.info("Состояние восстановлено")
    except Exception: pass

def get_balance() -> float:
    try:
        b = exchange.fetch_balance({"type": "linear"})
        return float(b.get("USDT", {}).get("free", 0.0))
    except Exception: return 0.0

def get_full_balance() -> float:
    try:
        b = exchange.fetch_balance({"type": "linear"})
        for field in ("total", "equity"):
            v = float(b.get("USDT", {}).get(field, 0.0))
            if v > 0: return v
        return get_balance()
    except Exception: return get_balance()

def get_active_positions() -> List[dict]:
    return [p for p in safe_fetch_positions() if float(p.get("contracts", 0) or 0) > 0]

def update_day(balance: float):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if stats["дата_дня"] != today:
        stats["дата_дня"]            = today
        stats["баланс_начало_дня"]   = balance
        log.info(f"Новый день. Баланс: {balance:.2f} USDT")
        save_state()

def daily_limit_exceeded() -> bool:
    start = stats.get("баланс_начало_дня", 0.0)
    if start <= 0: return False
    cur = get_full_balance()
    loss_pct = (start - cur) / start * 100
    if loss_pct >= DAILY_LOSS_LIMIT_PCT:
        log.warning(f"Дневной лимит: -{loss_pct:.1f}%")
        return True
    return False

def calc_margin(score: int, balance: float, sl_pct: float) -> float:
    risk_pct  = capital.get_risk_pct(score)
    if risk_pct <= 0: return 0.0
    max_loss  = balance * risk_pct / 100
    margin    = min(max_loss / (sl_pct / 100), balance * 0.95)
    log.info(f"Скор={score} режим={capital.mode} риск={risk_pct:.1f}% "
             f"SL_dist={sl_pct:.2f}% маржа={margin:.2f}U")
    return round(max(1.0, margin), 2)

# ============================================================
#                       ОТЧЁТЫ
# ============================================================

def print_report():
    balance  = get_full_balance()
    start    = stats["депозит_старт"]
    delta    = balance - start
    net      = stats["прибыль_usdt"] - stats["убыток_usdt"]
    pct      = (delta / start * 100) if start > 0 else 0
    total    = stats["сделок_всего"]
    tp_      = stats["тейкпрофит"]
    wr       = (tp_ / total * 100) if total > 0 else 0

    log.info("")
    log.info("=" * 65)
    log.info("📊 ОТЧЁТ БОТА v11.0")
    log.info(f"  Баланс: {balance:.2f} USDT ({delta:+.2f} / {pct:+.2f}%)")
    log.info(f"  {capital.status_str()}")
    log.info(f"  Сделок: {total} | TP={tp_} | SL={stats['стоплосс']} | "
             f"Таймаут={stats['таймаут']}")
    log.info(f"  WinRate: {wr:.1f}%")
    log.info(f"  Прибыль: {stats['прибыль_usdt']:.4f} U  Убыток: {stats['убыток_usdt']:.4f} U")
    log.info(f"  Чистый P&L: {net:+.4f} USDT")
    log.info(f"  Симулятор отфильтровал: {stats['sim_отфильтровано']} входов")
    log.info("=" * 65)

    # Метрики
    history = load_history()
    if len(history) >= 5:
        pnls    = [t["pnl_usdt"] for t in history]
        wins    = sum(1 for p in pnls if p > 0)
        losses  = sum(1 for p in pnls if p < 0)
        cum     = np.cumsum(pnls)
        rmax    = np.maximum.accumulate(cum)
        max_dd  = abs(float(min(cum - rmax))) if len(pnls) > 1 else 0
        std     = np.std(pnls)
        sharpe  = np.mean(pnls) / std * np.sqrt(252) if std > 0 else 0
        pf_num  = sum(p for p in pnls if p > 0)
        pf_den  = abs(sum(p for p in pnls if p < 0))
        pf      = round(pf_num / pf_den, 2) if pf_den > 0 else 0
        log.info(f"  Sharpe={sharpe:.2f} MaxDD={max_dd:.2f}U PF={pf}")
    log.info("")
    stats["последний_отчёт"] = time.time()
    save_state()

def print_trade_report(record: dict):
    r     = record.get("результат", "?")
    sym   = record.get("symbol", "?").split(":")[0]
    pnl   = record.get("pnl_usdt", 0)
    side  = record.get("side", "?")
    score = record.get("score", "?")
    sim   = record.get("sim_wr", "?")
    d     = record.get("details", {})
    icon  = "✅ ТЕЙКПРОФИТ" if r == "tp" else ("❌ СТОПЛОСС" if r == "sl" else "⏰ ТАЙМАУТ")
    log.info("")
    log.info("━" * 60)
    log.info(f"📋 СДЕЛКА #{record.get('id','?')} | {sym} {side.upper()} | {icon}")
    log.info(f"   P&L: {pnl:+.4f} U | Скор: {score} | Sim WR: {sim}")
    log.info(f"   Вход: {record.get('entry_price',0):.8f}  "
             f"SL: {record.get('sl_price',0):.8f}  "
             f"TP: {record.get('tp_price',0):.8f}")
    log.info(f"   Длит: {record.get('duration_min',0):.1f} мин | "
             f"RR: {record.get('rr_ratio',0):.1f}:1")
    log.info(f"   RSI={d.get('rsi','?')} MACD={d.get('macd','?')} "
             f"ST={d.get('supertrend','?')} RF={d.get('range_filter','?')} "
             f"ADX={d.get('adx','?')}")
    log.info(f"   S/R: {d.get('sr_signal','?')} | Тренд1h: {d.get('тренд_1h','?')}")
    log.info("━" * 60)
    log.info("")

# ============================================================
#                   ПРЕДСТАРТОВАЯ ПРОВЕРКА
# ============================================================

def prestart_check() -> bool:
    log.info("=" * 60)
    log.info("🔍 ПРЕДСТАРТОВАЯ ПРОВЕРКА")
    ok = True
    # API
    api_key = os.getenv("BYBIT_API_KEY", "")
    api_sec = os.getenv("BYBIT_API_SECRET", "")
    if not api_key or len(api_key) < 10:
        log.error("❌ BYBIT_API_KEY не задан"); ok = False
    if not api_sec  or len(api_sec)  < 10:
        log.error("❌ BYBIT_API_SECRET не задан"); ok = False
    # Баланс
    try:
        b = exchange.fetch_balance({"type": "linear"})
        free = float(b.get("USDT", {}).get("free", 0))
        log.info(f"✅ Подключение OK. Свободно: {free:.4f} USDT")
        if free < MIN_BALANCE:
            log.warning(f"⚠️ Баланс {free:.2f} < {MIN_BALANCE} USDT")
    except Exception as e:
        log.error(f"❌ Ошибка подключения: {e}"); ok = False
    # Конфиг
    rr = TP_PERCENT / SL_PERCENT
    if rr < 2.0:
        log.error(f"❌ RR={rr:.1f} < 2:1"); ok = False
    log.info(f"✅ Конфиг: TP={TP_PERCENT}% SL={SL_PERCENT}% RR={rr:.1f}:1")
    log.info(f"   MIN_SCORE={MIN_SCORE} ENTRY_CONFIRM={ENTRY_CONFIRM_MIN_SCORE} "
             f"SIM_MIN_WR={SIMULATOR_MIN_WIN_RATE:.0%}")
    # Рынок
    ok_cnt = 0
    for sym in SYMBOLS[:5]:
        try:
            t = exchange.fetch_ticker(sym)
            if float(t["last"]) > 0: ok_cnt += 1
        except Exception: pass
    log.info(f"{'✅' if ok_cnt > 0 else '❌'} Рынок: {ok_cnt}/5 пар доступны")
    if ok_cnt == 0: ok = False
    # Позиции
    active = get_active_positions()
    if active:
        log.warning(f"⚠️ Открытые позиции: {[p['symbol'] for p in active]}")
    else:
        log.info("✅ Открытых позиций нет")
    log.info("✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ" if ok else "❌ ЕСТЬ КРИТИЧЕСКИЕ ОШИБКИ")
    log.info("=" * 60)
    return ok

# ============================================================
#                     ГЛАВНЫЙ ЦИКЛ
# ============================================================

def main():
    global stats

    if not prestart_check():
        log.error("🛑 Бот остановлен")
        return

    load_state()
    stats["запусков"] += 1
    bal_now = get_full_balance()
    if stats["депозит_старт"] <= 0:
        stats["депозит_старт"] = bal_now
    if not stats["старт_время"]:
        stats["старт_время"] = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    update_day(bal_now)
    capital.update(bal_now)
    capital.start_balance = stats["депозит_старт"]
    save_state()

    log.info("")
    log.info("=" * 65)
    log.info("🤖 ГИБРИДНЫЙ БОТ v11.0 С СИМУЛЯТОРОМ И ЗАЩИТОЙ КАПИТАЛА")
    log.info(f"  Плечо: {LEVERAGE}x | RR: {TP_PERCENT}/{SL_PERCENT} = {TP_PERCENT/SL_PERCENT:.1f}:1")
    log.info(f"  Баланс: {bal_now:.4f} USDT | {capital.status_str()}")
    log.info(f"  Симулятор: мин.WR={SIMULATOR_MIN_WIN_RATE:.0%} lookback={SIMULATOR_LOOKBACK}")
    log.info(f"  Защита: CAUTION>3% RECOVERY>5% STOP>{MAX_DRAWDOWN_PCT}%")
    log.info("=" * 65)
    log.info("")

    заблокированные: Dict[str, float] = {}
    fail_attempts:   Dict[str, int]   = {}

    while True:
        try:
            # Отчёт
            if time.time() - stats["последний_отчёт"] >= REPORT_INTERVAL:
                print_report()

            balance  = get_full_balance()
            free_bal = get_balance()
            update_day(balance)
            capital.update(balance)

            # Проверки защиты
            can_t, reason = capital.can_trade(0)
            if not can_t:
                log.warning(f"⛔ {reason} — пауза 10 мин")
                time.sleep(600); continue

            if free_bal < MIN_BALANCE:
                active = get_active_positions()
                if active:
                    log.info(f"⏳ Активные позиции: {[p['symbol'] for p in active]}")
                    time.sleep(60); continue
                log.warning(f"🛑 Баланс {free_bal:.2f} < {MIN_BALANCE} U — пауза 10 мин")
                time.sleep(600); continue

            if daily_limit_exceeded():
                log.warning(f"⛔ Дневной лимит. Пауза {DAILY_LOSS_PAUSE_SEC//60} мин.")
                time.sleep(DAILY_LOSS_PAUSE_SEC); continue

            if stats.get("sl_streak", 0) >= SL_STREAK_LIMIT:
                log.warning(f"🧊 {SL_STREAK_LIMIT} SL подряд — cooldown {SL_STREAK_PAUSE//60} мин.")
                stats["sl_streak"] = 0; save_state()
                time.sleep(SL_STREAK_PAUSE + SL_STREAK_EXTRA); continue

            active = get_active_positions()
            if active:
                log.info(f"⏳ Активные позиции: {[p['symbol'] for p in active]}")
                time.sleep(60); continue

            # Глобальный фильтр по рынку (шорты)
            ai_market = get_bybit_ratio("BTC/USDT:USDT")
            shorts_blocked = ai_market.get("long_ratio", 0) > BYBIT_LONG_BLOCK

            log.info(f"── Сканирование {len(SYMBOLS)} пар "
                     f"(баланс={free_bal:.2f}U порог={MIN_SCORE} "
                     f"{capital.status_str()}) ──")

            best_sym   = None
            best_score = 0
            best_side  = "long"
            best_data  = {}

            for sym in SYMBOLS:
                # Блокировки
                if sym in заблокированные:
                    if time.time() < заблокированные[sym]: continue
                    del заблокированные[sym]; fail_attempts.pop(sym, None)

                time.sleep(API_CALL_DELAY)

                # --- ЛОНГ ---
                if trend_4h(sym, bullish=True):
                    res = get_score(sym, "long")
                    ai_s = apply_ratio_correction(res["score"], sym, "long")
                    if ai_s >= MIN_SCORE:
                        det = res.get("details", {})
                        sr  = res.get("sr", {})
                        # Фильтры
                        if sr.get("near_resistance") and \
                           sr.get("dist_to_res_pct", 99) < SR_BLOCK_DIST:
                            log.info(f"⛔ {sym.split(':')[0]}: у сопротивления — пропуск")
                            continue
                        rsi_v = float(det.get("rsi", 50) or 50)
                        if rsi_v > 68 and not sr.get("near_support"):
                            continue
                        if not det.get("ma_cross", True): continue
                        if not det.get("vol_spike_ok", True): continue
                        if ai_s > best_score:
                            best_score = ai_s
                            best_sym   = sym
                            best_side  = "long"
                            best_data  = res

                # --- ШОРТ ---
                if not shorts_blocked and trend_4h(sym, bullish=False):
                    res_sh = get_score(sym, "short")
                    ai_sh  = apply_ratio_correction(res_sh["score"], sym, "short")
                    if ai_sh >= MIN_SCORE:
                        det_sh = res_sh.get("details", {})
                        if not det_sh.get("ma_cross", True): continue
                        if not det_sh.get("vol_spike_ok", True): continue
                        if ai_sh > best_score:
                            best_score = ai_sh
                            best_sym   = sym
                            best_side  = "short"
                            best_data  = res_sh

            if best_sym is None:
                log.info(f"Нет кандидатов — ждём {SCAN_INTERVAL} сек")
                time.sleep(SCAN_INTERVAL); continue

            log.info(f"► Лучший кандидат: {best_sym.split(':')[0]} "
                     f"({best_side.upper()}) скор={best_score}")

            # ============================================================
            #   СИМУЛЯТОР ВХОДА
            # ============================================================
            df_ta = best_data.get("df_ta")
            if df_ta is None or len(df_ta) < 80:
                log.warning("Недостаточно данных для симулятора — пропуск")
                time.sleep(SCAN_INTERVAL); continue

            price_now = best_data["price"]
            atr_now   = float(calc_atr(df_ta, 14).iloc[-1]) if len(df_ta) >= 14 else price_now*0.008

            # Получаем динамические SL/TP от модуля защиты
            sl_price, tp_price, sl_pct, tp_pct = capital.adjust_sl_tp(
                price_now, best_side, atr_now, best_score
            )

            sim_result = simulator.run(df_ta, best_side, tp_pct, sl_pct)
            simulator.log_result(best_sym, best_side, sim_result)

            if not sim_result["passed"]:
                stats["sim_отфильтровано"] += 1
                log.info(f"⛔ Симулятор отклонил вход в {best_sym.split(':')[0]} — "
                         f"ждём {SCAN_INTERVAL} сек")
                time.sleep(SCAN_INTERVAL); continue

            # RR проверка
            real_rr = abs(tp_price - price_now) / abs(price_now - sl_price) \
                      if abs(price_now - sl_price) > 0 else 0
            if real_rr < 2.0:
                log.warning(f"⛔ RR={real_rr:.1f} < 2:1 — пропуск")
                time.sleep(SCAN_INTERVAL); continue

            # Проверка разрешения торговли с учётом скора
            can_t, reason = capital.can_trade(best_score)
            if not can_t:
                log.warning(f"⛔ Защита капитала: {reason}")
                time.sleep(SCAN_INTERVAL); continue

            margin = calc_margin(best_score, free_bal, sl_pct)
            if free_bal < margin * 1.1:
                margin = free_bal * 0.8
                log.warning(f"⚠️ Баланс {free_bal:.2f} < маржа — уменьшаем до {margin:.2f}")

            sl_dist_pct = abs(price_now - sl_price) / price_now * 100

            log.info(f"📐 ATR={atr_now/price_now*100:.2f}% SL={sl_dist_pct:.2f}% "
                     f"TP={tp_pct:.2f}% RR={real_rr:.1f}:1 Sim.WR={sim_result['win_rate']:.0%}")
            log.info(f"✅ ВХОД {best_side.upper()}: скор={best_score} "
                     f"SL={sl_price:.8f} TP={tp_price:.8f} маржа={margin:.2f}U")

            # Подтверждение входа (пауза 1 свеча)
            if not confirm_entry(best_sym, best_score, best_side):
                log.info(f"⛔ Вход в {best_sym} отменён по подтверждению")
                fail_attempts[best_sym] = fail_attempts.get(best_sym, 0) + 1
                if fail_attempts[best_sym] >= SYMBOL_MAX_FAIL_ATTEMPTS:
                    заблокированные[best_sym] = time.time() + SYMBOL_BLOCK_AFTER_FAIL * 60
                    fail_attempts.pop(best_sym, None)
                    log.warning(f"⛔ {best_sym.split(':')[0]} заблокирован "
                                f"на {SYMBOL_BLOCK_AFTER_FAIL} мин")
                time.sleep(30); continue

            fail_attempts.pop(best_sym, None)

            # Открытие позиции
            bal_before   = get_full_balance()
            entered_at   = time.time()
            entry_price, qty = open_position(
                best_sym, margin, tp_price, sl_price, best_side
            )

            if entry_price is None or qty is None:
                log.warning("Не удалось открыть позицию")
                fail_attempts[best_sym] = fail_attempts.get(best_sym, 0) + 1
                if fail_attempts[best_sym] >= SYMBOL_MAX_FAIL_ATTEMPTS:
                    заблокированные[best_sym] = time.time() + SYMBOL_BLOCK_AFTER_FAIL * 60
                    fail_attempts.pop(best_sym, None)
                time.sleep(30); continue

            fail_attempts.pop(best_sym, None)
            stats["сделок_всего"] += 1
            save_state()

            # Мониторинг
            result_type = "sl"
            mon_pnl     = 0.0
            try:
                result_type, mon_pnl = monitor_position(
                    best_sym, entry_price, qty,
                    entered_at, sl_price, tp_price, best_side
                )
            except Exception as e:
                log.error(f"💥 Краш мониторинга: {e}")
                close_position(best_sym, qty, best_side)
                result_type = "sl"

            time.sleep(3)
            bal_after  = get_full_balance()
            pnl_real   = bal_after - bal_before
            dur_min    = (time.time() - entered_at) / 60

            if result_type == "tp":
                stats["тейкпрофит"] += 1
                stats["прибыль_usdt"] += max(0, pnl_real)
                stats["sl_streak"]    = 0
                log.info(f"✅ TP: прибыль ≈{pnl_real:+.4f} USDT")
                заблокированные[best_sym] = time.time() + SYMBOL_BLOCK_AFTER_TP * 60
            elif result_type == "sl":
                stats["стоплосс"] += 1
                stats["убыток_usdt"] += abs(min(0, pnl_real))
                stats["sl_streak"]   = stats.get("sl_streak", 0) + 1
                заблокированные[best_sym] = time.time() + SYMBOL_BLOCK_AFTER_SL * 60
                log.warning(f"❌ SL: убыток ≈{pnl_real:+.4f} USDT "
                            f"streak={stats['sl_streak']}/{SL_STREAK_LIMIT}")
            else:
                stats["таймаут"] += 1
                if pnl_real >= 0: stats["прибыль_usdt"] += pnl_real
                else: stats["убыток_usdt"] += abs(pnl_real)
                stats["sl_streak"] = 0
                заблокированные[best_sym] = time.time() + SYMBOL_BLOCK_AFTER_TP * 60
                log.warning(f"⏰ Таймаут: P&L ≈{pnl_real:+.4f} USDT")

            record = {
                "id":           stats["сделок_всего"],
                "время_входа":  datetime.fromtimestamp(entered_at).strftime("%d.%m.%Y %H:%M:%S"),
                "время_выхода": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                "duration_min": round(dur_min, 1),
                "symbol":       best_sym,
                "side":         best_side,
                "score":        best_score,
                "sim_wr":       sim_result["win_rate"],
                "sim_trades":   sim_result["wins"] + sim_result["losses"],
                "entry_price":  entry_price,
                "sl_price":     sl_price,
                "tp_price":     tp_price,
                "sl_dist_pct":  round(sl_dist_pct, 3),
                "margin_usdt":  margin,
                "leverage":     LEVERAGE,
                "результат":    result_type,
                "pnl_usdt":     round(pnl_real, 4),
                "rr_ratio":     round(real_rr, 2),
                "capital_mode": capital.mode,
                "details":      best_data.get("details", {}),
            }
            save_trade(record)
            print_trade_report(record)
            save_state()

            log.info("Сделка завершена — пауза 60 сек")
            time.sleep(60)

        except Exception as e:
            log.error(f"Глобальная ошибка: {e}", exc_info=True)
            time.sleep(60)


if __name__ == "__main__":
    main()
