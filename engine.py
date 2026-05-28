# engine.py — Индикаторы, скоринг, анализ, риск-менеджмент для мем-коинов
import os
import sys
import time
import json
import logging
import requests
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats as scipy_stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from statsmodels.tsa.stattools import coint, adfuller
import joblib
import warnings
import xgboost as xgb

warnings.filterwarnings('ignore')

from config import *

# ============================================================
# БИРЖА
# ============================================================
exchange = ccxt.bybit({
    "apiKey": API_KEY,
    "secret": API_SECRET,
    "enableRateLimit": True,
    "options": {"defaultType": "linear"},
})

# ============================================================
# ЛОГИРОВАНИЕ
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    handlers=[
        logging.FileHandler("engine_meme.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("engine")

# ============================================================
# ИСТОРИЧЕСКИЕ ДАННЫЕ ДЛЯ ML
# ============================================================
pending_ml_entries = {}  # trade_id → {features, symbol, entry_time}

# ============================================================
# ИНДИКАТОРЫ
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
    mt = ma_type.upper()
    if mt == "EMA":
        return _ema(s, length)
    elif mt == "SMA":
        return _sma(s, length)
    elif mt == "WMA":
        return _wma(s, length)
    elif mt == "HMA":
        return _hma(s, length)
    else:
        return _ema(s, length)

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
        if pt == 1 and c < lb.iloc[i]:
            trend.iloc[i] = -1
        elif pt == -1 and c > ub.iloc[i]:
            trend.iloc[i] = 1
        else:
            trend.iloc[i] = pt
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
        if c - r > pf:
            filt.iloc[i] = c - r
        elif c + r < pf:
            filt.iloc[i] = c + r
        else:
            filt.iloc[i] = pf
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

    def cluster(levels):
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

    res_cl = cluster(raw_res)
    sup_cl = cluster(raw_sup)
    res_above = sorted([(p, n) for p, n in res_cl if p > close], key=lambda x: x[0])
    sup_below = sorted([(p, n) for p, n in sup_cl if p < close], key=lambda x: x[0], reverse=True)

    nearest_res, res_n = res_above[0] if res_above else (close * 1.05, 0)
    nearest_sup, sup_n = sup_below[0] if sup_below else (close * 0.95, 0)
    dist_res = (nearest_res - close) / close * 100
    dist_sup = (close - nearest_sup) / close * 100

    return {
        "support": round(nearest_sup, 10),
        "resistance": round(nearest_res, 10),
        "dist_to_sup_pct": round(dist_sup, 2),
        "dist_to_res_pct": round(dist_res, 2),
        "sup_cluster": sup_n,
        "res_cluster": res_n,
        "near_support": dist_sup < SR_PROXIMITY_PCT and sup_n >= SR_MIN_TOUCHES,
        "near_resistance": dist_res < SR_PROXIMITY_PCT and res_n >= SR_MIN_TOUCHES,
    }

# ============================================================
# БАЙЕСОВСКИЙ ТРЕНД
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
    except Exception:
        return 0.5

# ============================================================
# ФИЛЬТРЫ
# ============================================================
def check_ma_crossover(df: pd.DataFrame, side: str = "long") -> bool:
    if not MA_CROSSOVER_ENABLED:
        return True
    try:
        min_len = max(MA1_LENGTH, MA2_LENGTH) * 2 + 5
        if len(df) < min_len:
            return True
        ma1 = calc_ma(df, MA1_TYPE, MA1_LENGTH)
        ma2 = calc_ma(df, MA2_TYPE, MA2_LENGTH)
        if side == "long":
            return bool(ma1.iloc[-1] > ma2.iloc[-1])
        else:
            return bool(ma1.iloc[-1] < ma2.iloc[-1])
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
    except Exception:
        return True

def is_trading_time_allowed() -> bool:
    if not SESSION_FILTER_ENABLED:
        return True
    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour
    if SESSION_BLOCK_START < SESSION_BLOCK_END:
        blocked = SESSION_BLOCK_START <= hour < SESSION_BLOCK_END
    else:
        blocked = hour >= SESSION_BLOCK_START or hour < SESSION_BLOCK_END
    if blocked:
        log.info(f"Session Filter: час {hour} UTC заблокирован")
    return not blocked

def trend_4h_bullish(symbol: str) -> bool:
    try:
        raw = exchange.fetch_ohlcv(symbol, TIMEFRAME_4H, limit=60)
        if len(raw) < 55:
            return False
        df = pd.DataFrame(raw, columns=["ts", "o", "h", "l", "c", "v"])
        return bool(_ema(df["c"], 20).iloc[-1] > _ema(df["c"], 50).iloc[-1])
    except Exception:
        return False

def trend_4h_bearish(symbol: str) -> bool:
    try:
        raw = exchange.fetch_ohlcv(symbol, TIMEFRAME_4H, limit=60)
        if len(raw) < 55:
            return False
        df = pd.DataFrame(raw, columns=["ts", "o", "h", "l", "c", "v"])
        return bool(_ema(df["c"], 20).iloc[-1] < _ema(df["c"], 50).iloc[-1])
    except Exception:
        return False

def is_false_breakout(df: pd.DataFrame, side: str) -> bool:
    """Проверяет, является ли пробой ложным."""
    if len(df) < 10:
        return False

    close = df["c"]
    high = df["h"]
    low = df["l"]

    if side == "long":
        last_high = high.iloc[-1]
        prev_highs = high.iloc[-5:-1]
        if last_high > max(prev_highs) and close.iloc[-1] < last_high * 0.995:
            return True
    else:
        last_low = low.iloc[-1]
        prev_lows = low.iloc[-5:-1]
        if last_low < min(prev_lows) and close.iloc[-1] > last_low * 1.005:
            return True

    return False

def check_correlation(symbol: str, open_positions: List[dict]) -> bool:
    """Проверяет корреляцию с уже открытыми позициями."""
    if not PORTFOLIO_OPTIMIZATION:
        return True

    try:
        raw = exchange.fetch_ohlcv(symbol, "1h", limit=100)
        if len(raw) < 50:
            return True
        df_new = pd.DataFrame(raw, columns=["ts", "o", "h", "l", "c", "v"])["c"]

        for pos in open_positions:
            pos_symbol = pos.get("symbol")
            if pos_symbol == symbol:
                continue

            raw_pos = exchange.fetch_ohlcv(pos_symbol, "1h", limit=100)
            if len(raw_pos) < 50:
                continue
            df_pos = pd.DataFrame(raw_pos, columns=["ts", "o", "h", "l", "c", "v"])["c"]

            correlation = df_new.corr(df_pos)
            if abs(correlation) > CORRELATION_THRESHOLD:
                log.warning(f"Высокая корреляция между {symbol} и {pos_symbol}: {correlation:.2f}")
                return False
    except Exception as e:
        log.warning(f"Ошибка проверки корреляции: {e}")

    return True

def check_liquidity(symbol: str) -> bool:
    """Проверяет ликвидность для мем-коинов (менее строгая проверка)."""
    try:
        order_book = exchange.fetch_order_book(symbol, 10)
        total_bid_vol = sum([b[1] for b in order_book["bids"]])
        total_ask_vol = sum([a[1] for a in order_book["asks"]])
        min_volume = 100  # Уменьшен порог для мем-коинов
        if total_bid_vol < min_volume or total_ask_vol < min_volume:
            log.warning(f"Низкая ликвидность для {symbol} (bid: {total_bid_vol}, ask: {total_ask_vol})")
            return False
    except Exception as e:
        log.warning(f"Ошибка проверки ликвидности: {e}")
        return False
    return True

# ============================================================
# ПРОВЕРКА НОВОСТЕЙ
# ============================================================
def is_high_impact_news(symbol: str) -> bool:
    """Проверяет, есть ли важные новости для символа."""
    try:
        coin = symbol.split("/")[0].lower()
        url = f"https://api.coingecko.com/api/v3/coins/{coin}/events"
        response = requests.get(url, timeout=5)
        data = response.json()

        now = datetime.now(timezone.utc)
        for event in data.get("data", []):
            event_time = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
            if abs((event_time - now).total_seconds()) < 3600:  # В течение часа
                if event.get("type") in ["conference", "meetup", "hard_fork", "airdrop"]:
                    log.warning(f"Важная новость для {symbol}: {event.get('title', 'Unknown')}")
                    return True
    except Exception as e:
        log.debug(f"Ошибка проверки новостей: {e}")
    return False

# ============================================================
# P&L И ПРОВЕРКА ПРИБЫЛЬНОСТИ
# ============================================================
def calc_exact_pnl(
    entry_price: float, tp_price: float, sl_price: float,
    margin_usdt: float, leverage: int, symbol: str,
    side: str = "long"
) -> Dict[str, Any]:
    """Рассчитывает точный P&L с учетом комиссий, проскальзывания и фондирования."""
    try:
        if side == "long":
            real_entry = entry_price * (1 + SLIPPAGE_PCT / 100)
            real_tp = tp_price * (1 - SLIPPAGE_PCT / 100)
            real_sl = sl_price * (1 + SLIPPAGE_PCT / 100)
        else:
            real_entry = entry_price * (1 - SLIPPAGE_PCT / 100)
            real_tp = tp_price * (1 + SLIPPAGE_PCT / 100)
            real_sl = sl_price * (1 - SLIPPAGE_PCT / 100)

        position_size_usdt = margin_usdt * leverage
        qty = position_size_usdt / entry_price

        if side == "long":
            pnl_tp = (real_tp - real_entry) * qty
            pnl_sl = (real_sl - real_entry) * qty
        else:
            pnl_tp = (real_entry - real_tp) * qty
            pnl_sl = (real_entry - real_sl) * qty

        open_fee = position_size_usdt * BYBIT_FEE
        close_fee = position_size_usdt * BYBIT_FEE
        total_fee = open_fee + close_fee

        pnl_tp_net = pnl_tp - total_fee
        pnl_sl_net = pnl_sl - total_fee

        funding_rate = 0.0
        if FUNDING_RATE_CHECK:
            try:
                funding = exchange.fetch_funding_rate_history(symbol, limit=1)
                if funding:
                    funding_rate = float(funding[0].get("fundingRate", 0))
            except Exception as e:
                log.debug(f"Funding rate недоступен: {e}")

        risk_usdt = abs(pnl_sl_net)
        reward_usdt = pnl_tp_net
        rr_ratio = reward_usdt / risk_usdt if risk_usdt > 0 else 0
        risk_pct = (risk_usdt / margin_usdt * 100) if margin_usdt > 0 else 0

        return {
            "real_entry": real_entry,
            "real_tp": real_tp,
            "real_sl": real_sl,
            "qty": qty,
            "pnl_tp_net": pnl_tp_net,
            "pnl_sl_net": pnl_sl_net,
            "total_fee": total_fee,
            "funding_rate": funding_rate,
            "risk_usdt": risk_usdt,
            "reward_usdt": reward_usdt,
            "rr_ratio": rr_ratio,
            "risk_pct": risk_pct,
            "valid": True
        }
    except Exception as e:
        log.error(f"Ошибка расчета P&L: {e}")
        return {"valid": False, "error": str(e)}

def check_trade_profitability(
    symbol: str, margin_usdt: float, entry_price: float,
    tp_price: float, sl_price: float, side: str = "long",
    current_balance: float = 0.0, open_positions: List[dict] = None
) -> Tuple[bool, Dict[str, Any]]:
    """Проверяет прибыльность сделки перед открытием."""
    if not DEPOSIT_ANALYSIS_ENABLED:
        return True, {"message": "Проверка отключена"}

    try:
        pnl_data = calc_exact_pnl(entry_price, tp_price, sl_price, margin_usdt, LEVERAGE, symbol, side)
        if not pnl_data.get("valid"):
            return False, {"error": "Не удалось рассчитать P&L"}

        if pnl_data["rr_ratio"] < MIN_RR_RATIO:
            return False, {"reason": "Низкое RR", "rr_ratio": pnl_data["rr_ratio"]}

        risk_pct = (pnl_data["risk_usdt"] / current_balance * 100) if current_balance > 0 else 0
        if risk_pct > MAX_RISK_PER_TRADE_PCT:
            return False, {"reason": "Высокий риск", "risk_pct": risk_pct}

        if open_positions:
            total_risk = pnl_data["risk_usdt"]
            for pos in open_positions:
                pos_sl = float(pos.get("stopLoss", 0) or 0)
                pos_qty = float(pos.get("contracts", 0) or 0)
                pos_entry = float(pos.get("entryPrice", 0))
                if pos_sl > 0 and pos_qty > 0:
                    total_risk += abs(pos_entry - pos_sl) * pos_qty

            total_risk_pct = (total_risk / current_balance * 100) if current_balance > 0 else 0
            if total_risk_pct > MAX_PORTFOLIO_RISK * 100:
                return False, {"reason": "Высокий совокупный риск", "total_risk_pct": total_risk_pct}

        return True, {"approved": True, "pnl_data": pnl_data}

    except Exception as e:
        log.error(f"Ошибка проверки прибыльности: {e}")
        return False, {"error": str(e)}

def check_slippage(symbol: str, entry_price: float, real_entry: float) -> bool:
    """Проверяет, что проскальзывание не превышает допустимого."""
    slippage_pct = abs(real_entry - entry_price) / entry_price * 100
    if slippage_pct > MAX_SLIPPAGE_PCT:
        log.warning(f"Высокое проскальзывание: {slippage_pct:.2f}% > {MAX_SLIPPAGE_PCT}%")
        return False
    return True

# ============================================================
# КВАНТОВЫЙ АНАЛИЗ (Отключён для мем-коинов)
# ============================================================
def get_quant_signals(symbol: str) -> Dict[str, Any]:
    """Получает квантовые сигналы (заглушка для мем-коинов)."""
    return {"quant_score": 0, "details": {}}

# ============================================================
# ORDER FLOW
# ============================================================
def get_order_flow_signals(symbol: str) -> Dict[str, Any]:
    """Получает Order Flow сигналы."""
    if not ORDER_FLOW_ENABLED:
        return {"order_flow_score": 0, "details": {}}

    order_flow_score = 50  # Нейтральный

    try:
        order_book = exchange.fetch_order_book(symbol, ORDER_BOOK_DEPTH)
        bids = order_book["bids"]
        asks = order_book["asks"]

        total_bid_vol = sum([b[1] for b in bids])
        total_ask_vol = sum([a[1] for a in asks])
        imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol + 1e-10) * 100

        if imbalance > 20:
            order_flow_score += 20
        elif imbalance < -20:
            order_flow_score -= 20

        best_bid = bids[0][0] if bids else 0
        best_ask = asks[0][0] if asks else 0
        spread_pct = ((best_ask - best_bid) / best_bid * 100) if best_bid > 0 else 0

        if spread_pct < 0.1:
            order_flow_score += 10

    except Exception as e:
        log.debug(f"Order Flow ошибка: {e}")

    return {"order_flow_score": max(0, min(100, order_flow_score)), "details": {"spread_pct": spread_pct, "imbalance": imbalance}}

# ============================================================
# ML МОДЕЛЬ (Отключена для мем-коинов)
# ============================================================
class TradingModel:
    """ML модель для предсказания (заглушка для мем-коинов)."""
    def __init__(self, model_type: str = "RandomForest"):
        self.model_type = model_type
        self.trained = False
        self.accuracy = 0
        self.precision = 0

    def create_features(self, *args, **kwargs) -> Dict[str, float]:
        return {}

    def train(self, *args, **kwargs) -> bool:
        return False

    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        return {"signal": "neutral", "probability": 0.5, "valid": False}

    def save_model(self, *args, **kwargs) -> bool:
        return False

    def load_model(self, *args, **kwargs) -> bool:
        return False

ml_model = TradingModel()

# ============================================================
# СКОРИНГ (Оптимизирован для мем-коинов)
# ============================================================
def get_meme_coin_score(symbol: str) -> dict:
    """Анализирует мем-коин с учётом его специфики (волатильность, объём, тренд)."""
    details = {}
    score = 0
    price = 0.0
    sr = {}

    try:
        # Получаем данные для анализа
        raw_ta = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA, limit=200)
        raw_1h = exchange.fetch_ohlcv(symbol, TIMEFRAME_TREND, limit=200)
        if len(raw_ta) < 50 or len(raw_1h) < 50:
            return {"score": 0, "details": {}, "price": 0, "sr": {}}

        cols = ["ts", "o", "h", "l", "c", "v"]
        df_ta = pd.DataFrame(raw_ta, columns=cols).reset_index(drop=True)
        df_1h = pd.DataFrame(raw_1h, columns=cols).reset_index(drop=True)
        c_ta = df_ta["c"]
        c_1h = df_1h["c"]
        price = float(c_ta.iloc[-1])

        # Проверяем волатильность (ATR)
        atr = calc_atr(df_ta, 14).iloc[-1]
        atr_pct = (atr / price) * 100 if price > 0 else 0
        details["atr_pct"] = round(atr_pct, 2)

        # Для мем-коинов высокая волатильность — это хорошо
        if atr_pct > 10.0:
            score += 20
        elif atr_pct > 5.0:
            score += 10

        # Проверяем тренд на 1H и 4H
        if trend_4h_bullish(symbol):
            score += 15
            details["trend_4h"] = "bullish"
        else:
            score -= 5
            details["trend_4h"] = "bearish"

        # RSI (для мем-коинов можно допускать перекупленность)
        rsi_val = calc_rsi(c_ta).iloc[-1]
        details["rsi"] = round(rsi_val, 1)
        if rsi_val < 30:
            score += 15
        elif rsi_val < 40:
            score += 10
        elif rsi_val > 70:
            score += 5  # Для мем-коинов перекупленность не всегда плохо

        # MACD (быстрый для мем-коинов)
        ml_macd, sl_macd, _ = calc_macd(c_ta, fast=8, slow=21, signal=9)
        macd_bull = ml_macd.iloc[-1] > sl_macd.iloc[-1]
        macd_cross = macd_bull and ml_macd.iloc[-2] <= sl_macd.iloc[-2]
        details["macd"] = "bullish" if macd_bull else "bearish"
        if macd_cross:
            score += 20
        elif macd_bull:
            score += 10

        # Объём (для мем-коинов важен рост объёма)
        vol_avg = df_ta["v"].rolling(10).mean().iloc[-1]
        vol_current = df_ta["v"].iloc[-1]
        vol_ratio = vol_current / (vol_avg + 1e-10)
        details["volume_ratio"] = round(vol_ratio, 2)
        if vol_ratio > 2.0:
            score += 15
        elif vol_ratio > 1.5:
            score += 10

        # Supertrend (быстрый для мем-коинов)
        st_up, _ = calc_supertrend(df_ta, period=10, mult=2.0)
        details["supertrend"] = "up" if st_up.iloc[-1] else "down"
        if st_up.iloc[-1]:
            score += 12

        # EMA кроссовер (быстрый для мем-коинов)
        ema9 = _ema(c_ta, 9).iloc[-1]
        ema21 = _ema(c_ta, 21).iloc[-1]
        if ema9 > ema21:
            score += 10
            details["ema_cross"] = "bullish"
        else:
            details["ema_cross"] = "bearish"

        # S/R уровни
        sr = calc_support_resistance(df_ta, period=50)
        details.update({
            "support": sr["support"],
            "resistance": sr["resistance"],
            "dist_sup": sr["dist_to_sup_pct"],
            "dist_res": sr["dist_to_res_pct"]
        })
        if sr["near_support"]:
            score += 10
            details["sr_signal"] = f"near_support ({sr['sup_cluster']} touches)"
        elif sr["near_resistance"]:
            score -= 5  # Для мем-коинов сопротивление не так критично
            details["sr_signal"] = f"near_resistance ({sr['res_cluster']} touches)"

        # Bayes вероятность
        bayes_prob = bayes_trend_probability(df_ta)
        details["bayes_prob"] = round(bayes_prob, 2)
        score += int(bayes_prob * 15)  # Увеличен вес для мем-коинов

        # Проверяем, есть ли сильный рост/падение за последние 10 свечей
        price_change_10 = (c_ta.iloc[-1] - c_ta.iloc[-10]) / c_ta.iloc[-10] * 100
        details["price_change_10"] = round(price_change_10, 2)
        if abs(price_change_10) > 5.0:
            score += 10

        return {
            "score": max(0, min(100, score)),
            "details": details,
            "price": price,
            "sr": sr,
            "df_ta": df_ta,
            "df_1h": df_1h
        }

    except Exception as e:
        log.warning(f"Ошибка анализа мем-коина {symbol}: {e}")
        return {"score": 0, "details": {}, "price": 0, "sr": {}}

def get_score(symbol: str, use_quant: bool = True, use_order_flow: bool = True) -> dict:
    """Получает скор для мем-коина."""
    return get_meme_coin_score(symbol)

def get_score_short(symbol: str) -> dict:
    """Получает скор для шорта мем-коина."""
    res = get_score(symbol)
    if res["score"] == 0:
        return res
    res["score"] = max(0, 100 - res["score"] - 10)
    try:
        raw = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA, limit=200)
        if len(raw) >= 50:
            df = pd.DataFrame(raw, columns=["ts", "o", "h", "l", "c", "v"])
            ema9 = _ema(df["c"], 9).iloc[-1]
            ema21 = _ema(df["c"], 21).iloc[-1]
            res["details"]["ema_cross"] = "bullish" if ema9 > ema21 else "bearish"
    except:
        pass
    return res

def apply_ai_correction(score: int, symbol: str) -> int:
    """Применяет AI корректировку (заглушка для мем-коинов)."""
    return score

# ============================================================
# РИСК-МЕНЕДЖМЕНТ
# ============================================================
def calc_optimal_f(trades: List[dict]) -> float:
    """Оптимальное f по Келли."""
    if len(trades) < MIN_TRADES_FOR_F:
        return 0.0
    wins = [t['pnl_usdt'] for t in trades if t['pnl_usdt'] > 0]
    losses = [abs(t['pnl_usdt']) for t in trades if t['pnl_usdt'] < 0]
    if not wins or not losses:
        return 0.0
    win_rate = len(wins) / (len(wins) + len(losses))
    avg_win = np.mean(wins)
    avg_loss = np.mean(losses)
    kelly = win_rate - (1 - win_rate) / (avg_win / avg_loss)
    return min(max(0, kelly * 0.4), MAX_RISK_PERCENT_F / 100)

def calc_position_size(score: int, balance: float, sl_dist_pct: float, trades_history: List[dict] = None) -> float:
    """Рассчитывает размер позиции с учётом волатильности для мем-коинов."""
    atr_pct = sl_dist_pct  # Используем sl_dist_pct как прокси для ATR

    # Корректируем риск в зависимости от волатильности
    volatility_factor = min(max(0.5, atr_pct / 2.0), 2.0)

    if USE_ADVANCED_RISK and trades_history and len(trades_history) >= MIN_TRADES_FOR_F:
        f_opt = calc_optimal_f(trades_history[-100:])
        risk_pct = max(0.5, min(f_opt * 100 * volatility_factor, MAX_RISK_PERCENT_F))
    else:
        factor = max(0, (score - MIN_SCORE)) / (100 - MIN_SCORE)
        risk_pct = min(BASE_RISK_PCT + (MAX_RISK_PCT - BASE_RISK_PCT) * factor * volatility_factor, MAX_RISK_PCT)

    max_loss_usdt = balance * risk_pct / 100
    margin_usdt = min(max_loss_usdt / (sl_dist_pct / 100), balance * 0.95)
    return round(max(1.0, margin_usdt), 2)

# ============================================================
# МОНТЕ-КАРЛО (Отключено для мем-коинов)
# ============================================================
def run_monte_carlo(trades: List[dict]) -> Dict[str, Any]:
    """Monte Carlo симуляция (заглушка для мем-коинов)."""
    return {"valid": False}

# ============================================================
# СТАТИСТИКА И МЕТРИКИ
# ============================================================
def load_indicator_stats() -> dict:
    if not os.path.exists(INDICATOR_STATS_FILE):
        return {}
    try:
        with open(INDICATOR_STATS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_indicator_stats(stats_data: dict):
    try:
        with open(INDICATOR_STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(stats_data, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        log.warning(f"Не удалось сохранить статистику индикаторов: {e}")

def update_indicator_stats(trade_record: dict):
    """Обновляет статистику индикаторов после сделки."""
    stats_data = load_indicator_stats()
    details = trade_record.get("details", {})
    result = trade_record.get("result", "")
    is_win = (result == "tp")

    indicators = {
        "atr_pct": lambda v: float(v) > 5.0,
        "trend_4h": lambda v: v == "bullish",
        "rsi": lambda v: 25 <= float(v) <= 40,
        "macd": lambda v: v == "bullish",
        "volume_ratio": lambda v: float(v) > 1.5,
        "supertrend": lambda v: v == "up",
        "ema_cross": lambda v: v == "bullish",
        "sr_signal": lambda v: "support" in str(v),
        "bayes_prob": lambda v: float(v) > 0.6,
        "price_change_10": lambda v: abs(float(v)) > 3.0,
    }

    for ind, condition in indicators.items():
        value = details.get(ind)
        if value is None:
            continue
        try:
            is_bullish = condition(value)
        except:
            continue

        if ind not in stats_data:
            stats_data[ind] = {"bullish": {"total": 0, "wins": 0}, "bearish": {"total": 0, "wins": 0}}

        if is_bullish:
            stats_data[ind]["bullish"]["total"] += 1
            if is_win:
                stats_data[ind]["bullish"]["wins"] += 1
        else:
            stats_data[ind]["bearish"]["total"] += 1
            if is_win:
                stats_data[ind]["bearish"]["wins"] += 1

    save_indicator_stats(stats_data)

def print_indicator_report():
    """Выводит отчёт по эффективности индикаторов."""
    stats_data = load_indicator_stats()
    if not stats_data:
        return

    log.info("=" * 70)
    log.info("📈 ЭФФЕКТИВНОСТЬ ИНДИКАТОРОВ (МЕМ-КОИНЫ)")
    log.info(f"{'Индикатор':<18} {'🟢Бычий WR%':>11}  {'n':>4}  {'🔴Медвежий WR%':>14}  {'n':>4}")
    log.info("─" * 70)

    for ind, data in stats_data.items():
        b_total = data["bullish"]["total"]
        b_wins = data["bullish"]["wins"]
        be_total = data["bearish"]["total"]
        be_wins = data["bearish"]["wins"]
        b_wr = (b_wins / b_total * 100) if b_total > 0 else 0
        be_wr = (be_wins / be_total * 100) if be_total > 0 else 0
        diff = b_wr - be_wr
        sign = "▲" if diff > 5 else ("▼" if diff < -5 else "≈")
        log.info(f"{ind:<18}  {b_wr:>9.1f}%  {b_total:>4}  {be_wr:>12.1f}%  {be_total:>4}  {sign}{diff:>+7.1f}%")

    log.info("=" * 70)

def calc_strategy_metrics(trades: List[dict]) -> dict:
    """Рассчитывает метрики стратегии."""
    if len(trades) < 5:
        return {}

    pnls = [t['pnl_usdt'] for t in trades]
    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    max_dd = abs(min(drawdowns))
    max_dd_pct = (max_dd / max(1, cumulative[-1] + max_dd)) * 100

    sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252) if len(pnls) > 1 and np.std(pnls) != 0 else 0
    neg_returns = [p for p in pnls if p < 0]
    sortino = np.mean(pnls) / np.std(neg_returns) * np.sqrt(252) if neg_returns and np.std(neg_returns) != 0 else 0

    win_trades = [p for p in pnls if p > 0]
    loss_trades = [p for p in pnls if p < 0]

    return {
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "max_drawdown_usdt": round(max_dd, 2),
        "max_drawdown_pct": round(max_dd_pct, 1),
        "total_trades": len(trades),
        "winrate": round(len(win_trades) / len(pnls) * 100, 1),
        "avg_win": round(np.mean(win_trades) if win_trades else 0, 2),
        "avg_loss": round(abs(np.mean(loss_trades)) if loss_trades else 0, 2),
        "profit_factor": round(sum(win_trades) / abs(sum(loss_trades)) if loss_trades and sum(loss_trades) != 0 else 0, 2),
    }

# ============================================================
# ЛОГИРОВАНИЕ ML ДАННЫХ (Заглушка для мем-коинов)
# ============================================================
def log_ml_data(*args, **kwargs):
    """Логирует данные для обучения ML (заглушка)."""
    pass

# ============================================================
# БЭКТЕСТЕР
# ============================================================
def backtest_simple(historical_data: pd.DataFrame, params: dict = None) -> dict:
    """
    Простой бэктестер на исторических данных.
    """
    if len(historical_data) < 100:
        return {"error": "Недостаточно данных для бэктеста"}

    results = []
    position = None
    entry_price = 0
    entry_idx = 0

    for i in range(50, len(historical_data) - 1):
        window = historical_data.iloc[i-50:i+1]
        df = pd.DataFrame(window, columns=["o", "h", "l", "c", "v"])

        if df is None or len(df) < 50:
            continue

        close = df["c"]

        # Простые сигналы для мем-коинов
        rsi = calc_rsi(close).iloc[-1]
        ema9 = _ema(close, 9).iloc[-1]
        ema21 = _ema(close, 21).iloc[-1]

        # Вход в лонг
        if position is None and rsi < 40 and ema9 > ema21:
            position = "long"
            entry_price = close.iloc[-1]
            entry_idx = i
        # Выход
        elif position == "long":
            pnl_pct = (close.iloc[-1] - entry_price) / entry_price * 100

            # TP 5% или SL 2%
            if pnl_pct >= 5.0:
                results.append({"result": "tp", "pnl_pct": pnl_pct, "bars": i - entry_idx})
                position = None
            elif pnl_pct <= -2.0:
                results.append({"result": "sl", "pnl_pct": pnl_pct, "bars": i - entry_idx})
                position = None
            # Таймаут 100 свечей
            elif i - entry_idx > 100:
                results.append({"result": "timeout", "pnl_pct": pnl_pct, "bars": i - entry_idx})
                position = None

    if not results:
        return {"error": "Нет сделок в бэктесте"}

    pnls = [r["pnl_pct"] for r in results]
    wins = [r for r in results if r["result"] == "tp"]
    losses = [r for r in results if r["result"] == "sl"]

    return {
        "total_trades": len(results),
        "wins": len(wins),
        "losses": len(losses),
        "winrate": round(len(wins) / len(results) * 100, 1),
        "avg_pnl": round(np.mean(pnls), 2),
        "total_pnl": round(sum(pnls), 2),
        "max_pnl": round(max(pnls), 2),
        "min_pnl": round(min(pnls), 2),
        "avg_bars": round(np.mean([r["bars"] for r in results]), 1),
    }

# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================
def load_trades_history() -> List[dict]:
    """Загружает историю сделок."""
    if not os.path.exists(TRADES_FILE):
        return []
    try:
        with open(TRADES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_trade(trade_record: dict):
    """Сохраняет сделку в историю."""
    history = load_trades_history()
    history.append(trade_record)
    try:
        with open(TRADES_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        log.warning(f"Не удалось сохранить сделку: {e}")

def load_state() -> dict:
    """Загружает состояние бота."""
    default_state = {
        "starts": 0,
        "trades_total": 0,
        "take_profit": 0,
        "stop_loss": 0,
        "timeout": 0,
        "profit_usdt": 0.0,
        "loss_usdt": 0.0,
        "deposit_start": 0.0,
        "balance_day_start": 0.0,
        "day_date": "",
        "start_time": "",
        "last_report": 0.0,
        "sl_streak": 0,
        "ml_trades_since_retrain": 0,
        "monte_carlo_last_run": 0,
        "blocked_symbols": {},
        "bot_version": BOT_VERSION,
    }

    if not os.path.exists(STATE_FILE):
        return default_state
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        # Обновляем только существующие ключи
        for key in default_state:
            if key in saved:
                default_state[key] = saved[key]
        return default_state
    except Exception:
        return default_state

def save_state(state: dict):
    """Сохраняет состояние бота."""
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        log.warning(f"Не удалось сохранить состояние: {e}")

def get_free_balance() -> float:
    """Получает свободный баланс USDT."""
    try:
        b = exchange.fetch_balance({"type": "linear"})
        return float(b.get("USDT", {}).get("free", 0.0))
    except Exception as e:
        log.warning(f"Ошибка получения баланса: {e}")
        return 0.0

def get_total_balance() -> float:
    """Получает полный баланс USDT."""
    try:
        b = exchange.fetch_balance({"type": "linear"})
        total = float(b.get("USDT", {}).get("total", 0.0))
        return total if total > 0 else get_free_balance()
    except Exception as e:
        log.warning(f"Ошибка получения баланса: {e}")
        return get_free_balance()

def get_positions() -> List[dict]:
    """Получает открытые позиции."""
    try:
        positions = exchange.fetch_positions()
        return [p for p in positions if float(p.get("contracts", 0) or 0) > 0]
    except Exception as e:
        log.warning(f"Ошибка получения позиций: {e}")
        return []

def update_day_start(state: dict, balance: float):
    """Обновляет начало дня."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state["day_date"] != today:
        state["day_date"] = today
        state["balance_day_start"] = balance
        log.info(f"Новый торговый день. Баланс: {balance:.2f} USDT")

def is_daily_loss_exceeded(state: dict) -> bool:
    """Проверяет превышение дневного лимита убытков."""
    start = state.get("balance_day_start", 0.0)
    if start <= 0:
        return False
    loss = state.get("loss_usdt", 0.0)
    if loss <= 0:
        return False
    loss_pct = (loss / start * 100) if start > 0 else 0
    if loss_pct >= DAILY_LOSS_LIMIT_PCT:
        log.warning(f"Дневной лимит убытков: -{loss_pct:.1f}% (лимит {DAILY_LOSS_LIMIT_PCT}%)")
        return True
    return False

# ============================================================
# ОРДЕРА
# ============================================================
def set_leverage(symbol: str, leverage: int) -> bool:
    """Устанавливает плечо для символа."""
    try:
        coin_sym = symbol.split("/")[0] + "USDT"
        exchange.set_leverage(leverage, coin_sym, params={
            "buyLeverage": leverage,
            "sellLeverage": leverage
        })
        log.info(f"Плечо {leverage}x установлено для {coin_sym}")
        return True
    except Exception as e1:
        log.debug(f"Метод 1 не сработал: {e1}")
        try:
            coin_sym = symbol.split("/")[0] + "USDT"
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
            log.error(f"Не удалось установить плечо: {e2}")
            return False

def update_sl_on_exchange(symbol: str, new_sl: float, side: str = "long") -> bool:
    """Обновляет SL на бирже."""
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

def open_position(symbol: str, margin_usdt: float, tp_price: float,
                 sl_price: float, side: str = "long") -> Tuple[Optional[float], Optional[float]]:
    """Открывает позицию с проверкой прибыльности и проскальзывания."""
    try:
        # Проверяем, есть ли уже позиция по этому символу
        positions = get_positions()
        for pos in positions:
            if pos.get("symbol") == symbol:
                log.warning(f"Позиция по {symbol} уже открыта!")
                return None, None

        # Получаем текущие данные
        ticker = exchange.fetch_ticker(symbol)
        price = float(ticker["last"])
        current_balance = get_total_balance()
        open_positions = get_positions()

        # Корректируем SL/TP с учётом минимальных отступов
        if side == "long":
            sl_price = min(sl_price, price * (1 - MIN_SL_PERCENT / 100))
            tp_price = max(tp_price, price * (1 + TP_PERCENT / 100))
        else:
            sl_price = max(sl_price, price * (1 + MIN_SL_PERCENT / 100))
            tp_price = min(tp_price, price * (1 - TP_PERCENT / 100))

        # Проверяем прибыльность
        profitability_ok, profitability_data = check_trade_profitability(
            symbol, margin_usdt, price, tp_price, sl_price, side, current_balance, open_positions
        )

        if not profitability_ok:
            log.warning(f"❌ Сделка отклонена: {profitability_data.get('reason', 'проверка не пройдена')}")
            return None, None

        pnl_data = profitability_data.get("pnl_data", {})
        log.info(f"✅ Сделка одобрена: RR={pnl_data.get('rr_ratio', 0):.2f}")

        # Устанавливаем плечо
        if not set_leverage(symbol, LEVERAGE):
            log.error(f"Плечо не установлено — сделка отменена")
            return None, None

        # Рассчитываем количество
        pos_size_usdt = margin_usdt * LEVERAGE
        qty_raw = pos_size_usdt / price
        qty = float(exchange.amount_to_precision(symbol, qty_raw))
        if qty <= 0:
            log.error(f"Нулевое количество {symbol}")
            return None, None

        # Проверяем минимальный объём
        try:
            min_qty = exchange.markets[symbol]["limits"]["amount"]["min"]
            if qty < min_qty:
                log.warning(f"Объём {qty} меньше минимального {min_qty} для {symbol}")
                return None, None
        except Exception as e:
            log.warning(f"Ошибка проверки минимального объёма: {e}")

        # Форматируем цены
        tp_str = exchange.price_to_precision(symbol, tp_price)
        sl_str = exchange.price_to_precision(symbol, sl_price)

        buy_sell = "buy" if side == "long" else "sell"
        log.info(f"Открываем {side} {symbol}: qty={qty}, маржа≈{margin_usdt:.2f}U, TP={tp_str}, SL={sl_str}")

        # Открываем позицию
        try:
            order = exchange.create_market_order(
                symbol, buy_sell, qty,
                params={
                    "takeProfit": float(tp_str),
                    "stopLoss": float(sl_str),
                    "reduceOnly": False
                }
            )

            if not order:
                log.error(f"Ордер не выполнен: {order}")
                return None, None

            entry_price = float(order.get("average", price)) if order.get("average") else price

            # Проверяем проскальзывание
            if not check_slippage(symbol, price, entry_price):
                log.warning("Высокое проскальзывание — сделка отменена")
                return None, None

            # Проверяем частичное исполнение
            filled = float(order.get("filled", qty))
            if filled < qty * 0.9:  # Менее 90% исполнено
                log.warning(f"Частичное исполнение: {filled}/{qty}")
                if filled <= 0:
                    log.error("Ордер не исполнен")
                    return None, None
                qty = filled

            log.info(f"{side.upper()} открыт: {qty} {symbol} @ ~{entry_price:.8f}")
            return entry_price, qty

        except Exception as e:
            log.error(f"Ошибка открытия: {e}", exc_info=True)
            return None, None

    except Exception as e:
        log.error(f"Глобальная ошибка open_position: {e}", exc_info=True)
        return None, None

def open_position_with_retries(symbol: str, margin_usdt: float, tp_price: float,
                              sl_price: float, side: str = "long", max_retries: int = 3) -> Tuple[Optional[float], Optional[float]]:
    """Открывает позицию с повторными попытками."""
    for attempt in range(max_retries):
        entry_price, qty = open_position(symbol, margin_usdt, tp_price, sl_price, side)
        if entry_price is not None and qty is not None:
            return entry_price, qty
        log.warning(f"Попытка {attempt + 1}/{max_retries} не удалась. Повтор...")
        time.sleep(5)  # Ждём перед повторной попыткой
    return None, None

def close_position_with_confirm(symbol: str, qty: float, side: str) -> bool:
    """Закрывает позицию с подтверждением."""
    close_side = "sell" if side == "long" else "buy"
    for attempt in range(3):
        try:
            exchange.create_market_order(symbol, close_side, qty, params={"reduceOnly": True})
            time.sleep(3)

            # Проверяем закрытие
            positions = exchange.fetch_positions([symbol])
            active = [p for p in positions if float(p.get("contracts", 0) or 0) > 0 and p.get("side") == side]

            if not active:
                log.info(f"Позиция {symbol} закрыта успешно")
                return True

            log.warning(f"Позиция {symbol} не закрылась, попытка {attempt + 1}/3")
            time.sleep(2)
        except Exception as e:
            log.warning(f"Попытка {attempt + 1} закрыть {symbol}: {e}")
            time.sleep(2)

    log.error(f"Не удалось закрыть {symbol} после 3 попыток")
    return False

def emergency_close_position(symbol: str, side: str) -> bool:
    """Аварийное закрытие позиции."""
    try:
        positions = exchange.fetch_positions([symbol])
        for pos in positions:
            if pos.get("side") == side:
                qty = abs(float(pos.get("contracts", 0)))
                close_side = "sell" if side == "long" else "buy"
                exchange.create_market_order(symbol, close_side, qty, params={"reduceOnly": True})
                log.info(f"Аварийное закрытие {symbol} {side}")
                return True
    except Exception as e:
        log.error(f"Ошибка аварийного закрытия: {e}")
        return False
    return False

def check_signal_exit(symbol: str, side: str) -> bool:
    """Проверяет сигнал для выхода по развороту."""
    if not SIGNAL_EXIT_ENABLED:
        return False
    try:
        raw = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA, limit=50)
        if len(raw) < 30:
            return False
        df = pd.DataFrame(raw, columns=["ts", "o", "h", "l", "c", "v"])
        st_up, st_down = calc_supertrend(df, period=10, mult=2.0)
        _, _, _, rf_up, rf_down = calc_range_filter(df, period=50, qty=2.5)

        if side == "long":
            return bool(st_down.iloc[-1] and rf_down.iloc[-1])
        else:
            return bool(st_up.iloc[-1] and rf_up.iloc[-1])
    except Exception:
        return False

# ============================================================
# МОНИТОРИНГ ПОЗИЦИИ
# ============================================================
def monitor_position(symbol: str, entry_price: float, qty: float,
                    opened_at: float, sl_price: float, tp_price: float,
                    side: str = "long") -> str:
    """
    Мониторит позицию с защитой от зависания и уведомлениями.
    Возвращает: 'tp', 'sl', 'timeout'
    """
    deadline = opened_at + TRADE_MAX_LIFETIME
    coin = symbol.split(":")[0]
    consecutive_errors = 0
    max_errors = 5  # Максимум последовательных ошибок

    # Рассчитываем цену безубытка
    if side == "long":
        breakeven_price = entry_price * (1 + BYBIT_FEE * 2 + 0.0005)
    else:
        breakeven_price = entry_price * (1 - BYBIT_FEE * 2 - 0.0005)

    # Параметры трейлинга
    try:
        raw = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA, limit=50)
        if len(raw) >= 30:
            df = pd.DataFrame(raw, columns=["ts", "o", "h", "l", "c", "v"])
            atr_val = calc_atr(df, TRAILING_ATR_PERIOD).iloc[-1]
            atr_pct = (atr_val / entry_price) * 100
            trailing_step = max(MIN_TRAILING_STEP, atr_pct * TRAILING_ATR_MULT) / 100
            trailing_offset = max(MIN_TRAILING_OFFSET, atr_pct * TRAILING_OFFSET_MULT) / 100
        else:
            trailing_step = MIN_TRAILING_STEP / 100
            trailing_offset = MIN_TRAILING_OFFSET / 100
    except Exception as e:
        log.warning(f"Не удалось получить ATR для {symbol}: {e}")
        trailing_step = MIN_TRAILING_STEP / 100
        trailing_offset = MIN_TRAILING_OFFSET / 100

    # RR Exit триггер
    if side == "long":
        rr_trigger_price = entry_price + (tp_price - entry_price) * RR_EXIT_TRIGGER
    else:
        rr_trigger_price = entry_price - (entry_price - tp_price) * RR_EXIT_TRIGGER

    log.info(f"Мониторинг {coin} {side}: вход={entry_price:.8f}, SL={sl_price:.8f}, TP={tp_price:.8f}")

    phase = 1
    current_sl = sl_price
    peak_price = entry_price
    trailing_active = False
    partial_done = False

    while True:
        now = time.time()

        # Проверка дедлайна
        if now >= deadline:
            log.warning("Дедлайн — принудительное закрытие")
            close_position_with_confirm(symbol, qty, side)
            return "timeout"

        try:
            # Проверяем соединение с биржей
            if not check_connection():
                log.error("Нет соединения с биржей — аварийное закрытие")
                emergency_close_position(symbol, side)
                return "sl"

            # Проверяем существование позиции
            positions = exchange.fetch_positions([symbol])
            active = [p for p in positions if float(p.get("contracts", 0) or 0) > 0 and p.get("side") == side]

            if not active:
                # Позиция закрыта — определяем результат
                cur_price = exchange.fetch_ticker(symbol)["last"]
                if side == "long":
                    hit_tp = cur_price >= tp_price * (1 - SLIPPAGE_PCT / 100)
                else:
                    hit_tp = cur_price <= tp_price * (1 + SLIPPAGE_PCT / 100)
                return "tp" if hit_tp else "sl"

            pos = active[0]
            cur_price = float(pos.get("markPrice") or exchange.fetch_ticker(symbol)["last"])
            qty_actual = abs(float(pos.get("contracts", 0) or 0))
            unrealized_pnl = float(pos.get("unrealizedPnl", 0) or 0)

            # P&L в процентах
            if side == "long":
                pnl_pct = ((cur_price - entry_price) / entry_price * 100)
            else:
                pnl_pct = ((entry_price - cur_price) / entry_price * 100)

            time_to_deadline = int(deadline - now)
            consecutive_errors = 0  # Сброс при успехе

            # Частичный безубыток
            if PARTIAL_BE_ENABLED and not partial_done and pnl_pct >= PARTIAL_BE_PROFIT:
                close_qty = qty_actual * (PARTIAL_BE_CLOSE_PCT / 100)
                if close_qty > 0:
                    close_side = "sell" if side == "long" else "buy"
                    try:
                        exchange.create_market_order(symbol, close_side, close_qty, params={"reduceOnly": True})
                        log.info(f"Частичный BE: закрыто {close_qty:.4f} ({PARTIAL_BE_CLOSE_PCT:.0f}%) @ ~{cur_price:.8f}")
                        qty_actual -= close_qty

                        # Переводим остаток в безубыток
                        if side == "long":
                            new_sl = entry_price * (1 + BYBIT_FEE * 2 + 0.0003)
                        else:
                            new_sl = entry_price * (1 - BYBIT_FEE * 2 - 0.0003)

                        if update_sl_on_exchange(symbol, new_sl, side):
                            current_sl = new_sl
                            partial_done = True
                            log.info(f"SL переведён в безубыток: {new_sl:.8f}")
                    except Exception as e:
                        log.warning(f"Ошибка частичного закрытия: {e}")

            # Signal Exit
            if SIGNAL_EXIT_ENABLED and phase >= 2 and check_signal_exit(symbol, side):
                log.info("Signal Exit: разворот — закрываем")
                close_position_with_confirm(symbol, qty_actual, side)
                return "tp" if pnl_pct > 0 else "sl"

            # Полный безубыток (если не было частичного)
            if not partial_done and phase == 1 and pnl_pct >= 0.3:
                if side == "long":
                    new_sl_be = entry_price * (1 + BYBIT_FEE * 2 + 0.0003)
                else:
                    new_sl_be = entry_price * (1 - BYBIT_FEE * 2 - 0.0003)

                if update_sl_on_exchange(symbol, new_sl_be, side):
                    phase = 2
                    current_sl = new_sl_be
                    peak_price = cur_price
                    log.info(f"БЕЗУБЫТОК! SL → {new_sl_be:.8f}")

            # Активация трейлинга
            if not trailing_active and phase >= 2:
                if side == "long":
                    trailing_active = cur_price >= rr_trigger_price
                else:
                    trailing_active = cur_price <= rr_trigger_price

                if trailing_active:
                    log.info(f"Трейлинг активирован @ {cur_price:.8f}")

            # Трейлинг-стоп
            if trailing_active and phase >= 2 and pnl_pct >= MIN_PROFIT_FOR_TRAIL:
                updated = False
                if side == "long":
                    if cur_price > peak_price:
                        peak_price = cur_price
                    new_sl_trail = peak_price * (1 - trailing_offset)
                    if new_sl_trail > current_sl:
                        updated = update_sl_on_exchange(symbol, new_sl_trail, side)
                else:
                    if cur_price < peak_price:
                        peak_price = cur_price
                    new_sl_trail = peak_price * (1 + trailing_offset)
                    if new_sl_trail < current_sl:
                        updated = update_sl_on_exchange(symbol, new_sl_trail, side)

                if updated:
                    current_sl = new_sl_trail
                    log.info(f"ТРЕЙЛИНГ: пик={peak_price:.8f} → SL={new_sl_trail:.8f}")

            # Логирование статуса
            if int(now) % 30 < 10:  # Каждые ~30 секунд
                log.info(f"[{coin}] {cur_price:.8f} P&L={pnl_pct:+.2f}% ({unrealized_pnl:+.4f}U) "
                         f"SL={current_sl:.8f} фаза={phase} до_дед={time_to_deadline}с")

        except Exception as e:
            consecutive_errors += 1
            log.warning(f"Ошибка мониторинга ({consecutive_errors}/{max_errors}): {e}")

            if consecutive_errors >= max_errors:
                log.critical("Слишком много ошибок подряд — аварийное закрытие")
                emergency_close_position(symbol, side)
                return "sl"

            time.sleep(10)
            continue

        time.sleep(15)  # Пауза между проверками

    return "sl"

# ============================================================
# УВЕДОМЛЕНИЯ В TELEGRAM
# ============================================================
def send_telegram_message(message: str) -> None:
    """Отправляет уведомление в Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        log.warning(f"Ошибка отправки уведомления в Telegram: {e}")

# ============================================================
# ПОДТВЕРЖДЕНИЕ ВХОДА
# ============================================================
def confirm_entry(symbol: str, original_score: int, side: str = "long") -> bool:
    """Подтверждает вход с задержкой (отключено для мем-коинов)."""
    if ENTRY_CONFIRM_BARS <= 0:
        return True
    return True  # Для мем-коинов всегда подтверждаем

# ============================================================
# ОТЧЁТЫ
# ============================================================
def print_report(state: dict):
    """Выводит полный отчёт."""
    balance = get_total_balance()
    start = state["deposit_start"]
    delta = balance - start
    net = state["profit_usdt"] - state["loss_usdt"]
    pct = (delta / start * 100) if start > 0 else 0
    total = state["trades_total"]
    tp = state["take_profit"]
    sl = state["stop_loss"]
    wr = (tp / total * 100) if total > 0 else 0.0

    log.info("")
    log.info("=" * 65)
    log.info(f"📊 ОТЧЁТ {BOT_VERSION} (МЕМ-КОИНЫ)")
    log.info(f"Баланс: {balance:.2f} USDT ({delta:+.2f} / {pct:+.2f}%)")
    log.info(f"Сделок: {total} TP={tp} SL={sl} Таймаут={state['timeout']}")
    log.info(f"WinRate: {wr:.1f}%")
    log.info(f"Прибыль/Убыток: {state['profit_usdt']:.4f} / {state['loss_usdt']:.4f} USDT")
    log.info(f"Чистый P&L: {net:+.4f} USDT")

    log.info("=" * 65)

    state["last_report"] = time.time()
    save_state(state)
    print_indicator_report()

    # Метрики стратегии
    history = load_trades_history()
    if len(history) > 5:
        metrics = calc_strategy_metrics(history)
        if metrics:
            log.info("📉 МЕТРИКИ:")
            log.info(f"Sharpe: {metrics.get('sharpe_ratio', 0)} | "
                     f"Sortino: {metrics.get('sortino_ratio', 0)} | "
                     f"PF: {metrics.get('profit_factor', 0)}")
            log.info(f"Max DD: {metrics.get('max_drawdown_pct', 0)}% | "
                     f"WinRate: {metrics.get('winrate', 0)}%")

            try:
                with open(METRICS_FILE, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

def post_trade_analysis(trade_record: dict, ml_model_instance=None):
    """Анализирует завершенную сделку."""
    result = trade_record.get("result", "")
    symbol = trade_record["symbol"]
    pnl = trade_record.get("pnl_usdt", 0)
    duration = trade_record.get("duration_min", 0)

    sign = "✅" if result == "tp" else ("❌" if result == "sl" else "⏰")
    side_text = "LONG" if trade_record.get("side") == "long" else "SHORT"

    log.info("")
    log.info("━" * 60)
    log.info(f"📋 ПОСТ-ТРЕЙД: {symbol.split(':')[0]} {side_text} {sign} {result.upper()}")
    log.info(f"P&L: {pnl:+.4f} USDT | Длительность: {duration:.1f} мин")
    log.info(f"Скор: {trade_record.get('score', '?')}/100 | RR: {trade_record.get('rr_ratio', '?')}")
    log.info(f"Маржа: {trade_record.get('margin_usdt', 0):.2f}U | Плечо: {trade_record.get('leverage', LEVERAGE)}x")
    log.info("━" * 60)
    log.info("")

    # Обновляем статистику индикаторов
    update_indicator_stats(trade_record)

# ============================================================
# РАСЧЁТ SL/TP
# ============================================================
def calc_sl_tp(symbol: str, price: float, side: str, sr_info: dict) -> Tuple[float, float, float, float]:
    """Рассчитывает SL, TP и расстояния с учётом ATR и S/R для мем-коинов."""
    atr_price = 0.0
    try:
        raw = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA, limit=50)
        if len(raw) >= 20:
            df = pd.DataFrame(raw, columns=["ts", "o", "h", "l", "c", "v"])
            atr_price = float(calc_atr(df, 14).iloc[-1])
    except:
        pass

    atr_pct = (atr_price / price) * 100 if atr_price > 0 else 0

    if side == "long":
        # Динамический SL на основе ATR
        sl_pct_dist = max(MIN_SL_PERCENT, min(MAX_SL_PERCENT, atr_pct * ATR_SL_MULT))
        sl_price = price * (1 - sl_pct_dist / 100)

        # Учитываем поддержку
        support = sr_info.get("support", sl_price)
        if support < sl_price and support > price * 0.97:
            sl_price = support * 0.998

        # TP на основе RR
        tp_pct_dist = sl_pct_dist * MIN_RR_RATIO
        tp_price = price * (1 + tp_pct_dist / 100)
    else:
        sl_pct_dist = max(MIN_SL_PERCENT, min(MAX_SL_PERCENT, atr_pct * ATR_SL_MULT))
        sl_price = price * (1 + sl_pct_dist / 100)

        # Учитываем сопротивление
        resistance = sr_info.get("resistance", sl_price)
        if resistance > sl_price and resistance < price * 1.03:
            sl_price = resistance * 1.002

        tp_pct_dist = sl_pct_dist * MIN_RR_RATIO
        tp_price = price * (1 - tp_pct_dist / 100)

    sl_dist_pct = abs(price - sl_price) / price * 100
    real_rr = abs(tp_price - price) / abs(price - sl_price)

    return sl_price, tp_price, sl_dist_pct, real_rr

# ============================================================
# ОБУЧЕНИЕ ML (Заглушка для мем-коинов)
# ============================================================
def maybe_retrain_ml():
    """Переобучает ML модель (заглушка для мем-коинов)."""
    pass
