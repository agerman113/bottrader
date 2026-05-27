# engine.py — Индикаторы, скоринг, анализ, ML, риск-менеджмент
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
        logging.FileHandler("bot_v12.log", encoding="utf-8"),
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

def get_bybit_ai(symbol: str) -> dict:
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
                if buy_r > 0.6:
                    result["signal"] = "bullish"
                elif buy_r < 0.4:
                    result["signal"] = "bearish"
    except Exception as e:
        log.debug(f"Bybit ratio недоступен: {e}")
    return result

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
    """Проверяет ликвидность пары."""
    try:
        order_book = exchange.fetch_order_book(symbol, 20)
        total_bid_vol = sum([b[1] for b in order_book["bids"]])
        total_ask_vol = sum([a[1] for a in order_book["asks"]])
        min_volume = 1000  # Минимальный объём в стакане
        if total_bid_vol < min_volume or total_ask_vol < min_volume:
            log.warning(f"Низкая ликвидность для {symbol}")
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
# КВАНТОВЫЙ АНАЛИЗ
# ============================================================
def get_quant_signals(symbol: str) -> Dict[str, Any]:
    """Получает квантовые сигналы."""
    if not QUANT_ENABLED:
        return {"quant_score": 0, "details": {}}

    signals = {"mean_reversion": {}, "momentum": {}, "cointegration": {}}
    quant_score = 0

    # Mean Reversion
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA, limit=50)
        if len(ohlcv) >= 40:
            df = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c", "v"])
            close = df["c"]
            sma = close.rolling(20).mean()
            std = close.rolling(20).std()
            zscore = float((close.iloc[-1] - sma.iloc[-1]) / std.iloc[-1]) if std.iloc[-1] > 0 else 0

            if abs(zscore) > MEAN_REVERSION_THRESHOLD:
                quant_score += 20
            elif abs(zscore) > MEAN_REVERSION_THRESHOLD * 0.7:
                quant_score += 10
            signals["mean_reversion"] = {"zscore": zscore, "valid": True}
    except Exception as e:
        log.debug(f"Mean Reversion ошибка: {e}")

    # Momentum
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA, limit=40)
        if len(ohlcv) >= 30:
            df = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c", "v"])
            close = df["c"]
            momentum = (close.iloc[-1] - close.iloc[-MOMENTUM_WINDOW]) / close.iloc[-MOMENTUM_WINDOW] * 100
            if abs(momentum) > 3:
                quant_score += 15
            signals["momentum"] = {"momentum": momentum, "valid": True}
    except Exception as e:
        log.debug(f"Momentum ошибка: {e}")

    return {"quant_score": min(100, quant_score), "details": signals}

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
# ML МОДЕЛЬ
# ============================================================
class TradingModel:
    """ML модель для предсказания."""

    def __init__(self, model_type: str = "RandomForest"):
        self.model_type = model_type
        self.models = {}  # Для ансамбля моделей
        self.ensemble = None
        self.scaler = StandardScaler()
        self.features = [
            "rsi", "rsi_1h", "macd", "adx", "stoch_k", "volume_ratio",
            "price_change_5m", "price_change_15m", "price_change_1h",
            "atr_pct", "spread_pct", "imbalance",
            "mean_reversion_zscore", "momentum",
            "rr_ratio", "risk_pct",
            "hour_of_day", "day_of_week",
            "bayes_prob", "supertrend_up", "range_filter_up",
            "dist_to_support", "dist_to_resistance"
        ]
        self.trained = False
        self.last_retrain = 0
        self.accuracy = 0
        self.precision = 0
        self.feature_importances = {}

    def create_features(self, symbol: str, df_ta: pd.DataFrame, df_1h: pd.DataFrame,
                       order_flow_data: Dict, quant_data: Dict, risk_data: Dict = None) -> Dict[str, float]:
        """Создает расширенный набор фич для модели."""
        features = {}
        try:
            c_ta = df_ta["c"]
            c_1h = df_1h["c"]

            # Технические индикаторы
            features["rsi"] = float(calc_rsi(c_ta).iloc[-1])
            features["rsi_1h"] = float(calc_rsi(c_1h).iloc[-1])
            ml_macd, sl_macd, _ = calc_macd(c_ta)
            features["macd"] = float(ml_macd.iloc[-1] - sl_macd.iloc[-1])
            adx, _, _ = calc_adx(df_ta)
            features["adx"] = float(adx.iloc[-1])
            k_ser, _ = calc_stochastic(df_ta)
            features["stoch_k"] = float(k_ser.iloc[-1])

            # Волатильность
            atr = calc_atr(df_ta, 14).iloc[-1]
            features["atr_pct"] = (atr / c_ta.iloc[-1]) * 100 if c_ta.iloc[-1] > 0 else 0

            # Объём
            vol_avg = df_ta["v"].rolling(VOLUME_AVG_PERIOD).mean().iloc[-1]
            features["volume_ratio"] = float(df_ta["v"].iloc[-1] / (vol_avg + 1e-10))

            # Изменения цены
            features["price_change_5m"] = float((c_ta.iloc[-1] - c_ta.iloc[-2]) / c_ta.iloc[-2] * 100)
            features["price_change_15m"] = float((c_ta.iloc[-1] - c_ta.iloc[-3]) / c_ta.iloc[-3] * 100)
            features["price_change_1h"] = float((c_1h.iloc[-1] - c_1h.iloc[-2]) / c_1h.iloc[-2] * 100)

            # Order Flow
            if order_flow_data.get("details"):
                features["spread_pct"] = order_flow_data["details"].get("spread_pct", 0)
                features["imbalance"] = order_flow_data["details"].get("imbalance", 0)

            # Квантовые сигналы
            mr = quant_data.get("details", {}).get("mean_reversion", {})
            features["mean_reversion_zscore"] = mr.get("zscore", 0) if mr.get("valid") else 0

            mom = quant_data.get("details", {}).get("momentum", {})
            features["momentum"] = mom.get("momentum", 0) if mom.get("valid") else 0

            # Риск
            if risk_data and risk_data.get("valid"):
                features["rr_ratio"] = risk_data.get("rr_ratio", 0)
                features["risk_pct"] = risk_data.get("risk_pct", 0)

            # Дополнительные фичи
            features["bayes_prob"] = bayes_trend_probability(df_ta)
            st_up, _ = calc_supertrend(df_ta)
            features["supertrend_up"] = float(st_up.iloc[-1])
            _, _, _, rf_up, _ = calc_range_filter(df_ta)
            features["range_filter_up"] = float(rf_up.iloc[-1])

            # Фичи на основе S/R
            sr = calc_support_resistance(df_ta)
            features["dist_to_support"] = sr["dist_to_sup_pct"]
            features["dist_to_resistance"] = sr["dist_to_res_pct"]

            # Временные фичи
            now = datetime.now(timezone.utc)
            features["hour_of_day"] = now.hour
            features["day_of_week"] = now.weekday()

        except Exception as e:
            log.debug(f"Ошибка создания фич: {e}")

        # Заполняем пропущенные фичи нулями
        for f in self.features:
            if f not in features:
                features[f] = 0
        return features

    def train(self, ml_log_file: str = ML_LOG_FILE) -> bool:
        """Обучает ансамбль моделей с кросс-валидацией."""
        try:
            if not os.path.exists(ml_log_file):
                return False

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
                return False

            X = []
            y = []
            for entry in data:
                features = entry.get("features", {})
                full_features = {f: features.get(f, 0) for f in self.features}
                X.append([full_features[f] for f in self.features])
                y.append(1 if entry["trade_result"] == "tp" else 0)

            X_df = pd.DataFrame(X, columns=self.features)
            y_series = pd.Series(y)
            X_scaled = self.scaler.fit_transform(X_df)

            # Разделяем данные на тренировочный и тестовый наборы
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_series, test_size=0.2, random_state=42
            )

            # Создаём несколько моделей
            models = {}
            if "RandomForest" in ML_MODEL_TYPE:
                models["rf"] = RandomForestClassifier(
                    n_estimators=200, max_depth=15, min_samples_leaf=5,
                    class_weight="balanced", random_state=42, n_jobs=-1
                )
            if "GradientBoosting" in ML_MODEL_TYPE:
                models["gb"] = GradientBoostingClassifier(
                    n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42
                )
            if "XGBoost" in ML_MODEL_TYPE:
                models["xgb"] = xgb.XGBClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42
                )

            # Обучаем каждую модель
            for name, model in models.items():
                model.fit(X_train, y_train)
                self.models[name] = model

            # Создаём ансамбль
            if len(models) > 1:
                self.ensemble = VotingClassifier(
                    estimators=[(name, model) for name, model in models.items()],
                    voting="soft"
                )
                self.ensemble.fit(X_train, y_train)
                y_pred = self.ensemble.predict(X_test)
            else:
                # Если одна модель, используем её
                model_name = list(models.keys())[0]
                self.ensemble = models[model_name]
                y_pred = self.ensemble.predict(X_test)

            # Оцениваем качество
            self.accuracy = accuracy_score(y_test, y_pred)
            self.precision = precision_score(y_test, y_pred, zero_division=0)

            if hasattr(self.ensemble, "feature_importances_"):
                self.feature_importances = dict(zip(self.features, self.ensemble.feature_importances_))

            self.trained = True
            self.last_retrain = time.time()
            log.info(f"✅ ML ансамбль обучен: Acc={self.accuracy:.2f}, Prec={self.precision:.2f}")
            return True
        except Exception as e:
            log.error(f"Ошибка обучения ML: {e}")
            return False

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Предсказывает сигнал."""
        if not self.trained or self.ensemble is None:
            return {"signal": "neutral", "probability": 0.5, "valid": False}
        try:
            X = pd.DataFrame([{f: features.get(f, 0) for f in self.features}])
            X_scaled = self.scaler.transform(X)
            prediction = self.ensemble.predict(X_scaled)[0]
            probability = self.ensemble.predict_proba(X_scaled)[0][1]
            return {"signal": "buy" if prediction == 1 else "sell", "probability": float(probability), "valid": True}
        except Exception as e:
            log.debug(f"Ошибка предсказания: {e}")
            return {"signal": "neutral", "probability": 0.5, "valid": False}

    def save_model(self, filepath: str = ML_MODEL_FILE) -> bool:
        try:
            joblib.dump({
                "models": self.models,
                "ensemble": self.ensemble,
                "scaler": self.scaler,
                "features": self.features,
                "trained": self.trained,
                "accuracy": self.accuracy,
                "precision": self.precision,
                "last_retrain": self.last_retrain
            }, filepath)
            return True
        except Exception as e:
            log.error(f"Ошибка сохранения: {e}")
            return False

    def load_model(self, filepath: str = ML_MODEL_FILE) -> bool:
        try:
            data = joblib.load(filepath)
            self.models = data.get("models", {})
            self.ensemble = data.get("ensemble")
            self.scaler = data.get("scaler")
            self.features = data.get("features", self.features)
            self.trained = data.get("trained", False)
            self.accuracy = data.get("accuracy", 0)
            self.precision = data.get("precision", 0)
            self.last_retrain = data.get("last_retrain", 0)
            return True
        except Exception as e:
            log.error(f"Ошибка загрузки: {e}")
            return False

# Инициализация ML модели
ml_model = TradingModel(ML_MODEL_TYPE)

# ============================================================
# СКОРИНГ
# ============================================================
def get_score(symbol: str, use_quant: bool = True, use_order_flow: bool = True) -> dict:
    """Получает скор для лонга с учётом рыночного контекста."""
    details = {}
    score = 0
    price = 0.0
    sr = {}

    try:
        raw_ta = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA, limit=300)
        raw_1h = exchange.fetch_ohlcv(symbol, TIMEFRAME_TREND, limit=300)
        if len(raw_ta) < 100 or len(raw_1h) < 100:
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
        if atr_pct > 5.0:  # Высокая волатильность
            score -= 10  # Уменьшаем скор при высокой волатильности
            details["high_volatility"] = True

        # Проверяем тренд на старших таймфреймах (4H)
        if not trend_4h_bullish(symbol):
            score -= 15  # Штраф за отсутствие тренда на 4H
            details["trend_4h"] = "bearish"

        # Проверяем объём
        vol_avg = df_ta["v"].rolling(20).mean().iloc[-1]
        vol_current = df_ta["v"].iloc[-1]
        if vol_current < vol_avg * 0.5:  # Низкий объём
            score -= 10
            details["low_volume"] = True

        # RSI
        rsi_val = calc_rsi(c_ta).iloc[-1]
        details["rsi"] = round(rsi_val, 1)
        if 25 <= rsi_val <= 40:
            score += 20
        elif 40 < rsi_val <= 50:
            score += 12
        elif rsi_val < 25:
            score += 10
        elif 50 < rsi_val <= 60:
            score += 5

        rsi_1h = calc_rsi(c_1h).iloc[-1]
        details["rsi_1h"] = round(rsi_1h, 1)
        if rsi_1h < 50:
            score += 10
        elif rsi_1h < 60:
            score += 5

        # MACD
        ml_macd, sl_macd, _ = calc_macd(c_ta)
        macd_bull = ml_macd.iloc[-1] > sl_macd.iloc[-1]
        macd_cross = macd_bull and ml_macd.iloc[-2] <= sl_macd.iloc[-2]
        details["macd"] = "bullish" if macd_bull else "bearish"
        if macd_cross:
            score += 18
        elif macd_bull:
            score += 8

        # Range Filter
        _, _, _, rf_up, rf_down = calc_range_filter(df_ta)
        details["range_filter"] = "up" if rf_up.iloc[-1] else ("down" if rf_down.iloc[-1] else "flat")
        if rf_up.iloc[-1]:
            score += 15

        # Supertrend
        st_up, _ = calc_supertrend(df_ta)
        details["supertrend"] = "up" if st_up.iloc[-1] else "down"
        if st_up.iloc[-1]:
            score += 12

        # Hull
        hu_up, _ = calc_hull(c_ta)
        details["hull"] = "up" if hu_up.iloc[-1] else "down"
        if hu_up.iloc[-1]:
            score += 8

        # EMA тренд 1h
        ema50_1h = _ema(c_1h, 50).iloc[-1]
        ema200_1h = _ema(c_1h, 200).iloc[-1]
        details["trend_1h"] = "bullish" if ema50_1h > ema200_1h else "bearish"
        if ema50_1h > ema200_1h:
            score += 10

        # ADX
        adx, pdi, mdi = calc_adx(df_ta)
        adx_val = adx.iloc[-1]
        details["adx"] = round(adx_val, 1)
        if adx_val > 25 and pdi.iloc[-1] > mdi.iloc[-1]:
            score += 10
        elif adx_val > 20 and pdi.iloc[-1] > mdi.iloc[-1]:
            score += 4

        # Stochastic
        k_ser, _ = calc_stochastic(df_ta)
        k_val = k_ser.iloc[-1]
        details["stoch_k"] = round(k_val, 1)
        if k_val < 20:
            score += 10
        elif k_val < 40:
            score += 5

        # Volume
        vol_ratio = vol_current / (vol_avg + 1e-10)
        details["volume_ratio"] = round(vol_ratio, 2)
        if vol_ratio > 1.5:
            score += 8
        elif vol_ratio > 1.2:
            score += 4

        # S/R
        sr = calc_support_resistance(df_ta)
        details.update({
            "support": sr["support"],
            "resistance": sr["resistance"],
            "dist_sup": sr["dist_to_sup_pct"],
            "dist_res": sr["dist_to_res_pct"]
        })
        if sr["near_support"]:
            score += 15
            details["sr_signal"] = f"near_support ({sr['sup_cluster']} touches)"
        elif sr["near_resistance"]:
            score -= 25
            details["sr_signal"] = f"near_resistance ({sr['res_cluster']} touches)"
        else:
            details["sr_signal"] = "neutral"

        # 3 красных свечи
        last3_bearish = all(df_ta["c"].iloc[-i] < df_ta["o"].iloc[-i] for i in range(1, 4))
        if last3_bearish:
            score -= 20
            details["3red_candles"] = True

        # Bayes
        bayes_prob = bayes_trend_probability(df_ta)
        details["bayes_prob"] = round(bayes_prob, 2)
        score += int(bayes_prob * 10)

        # Quant
        if use_quant and QUANT_ENABLED:
            quant_data = get_quant_signals(symbol)
            quant_score = quant_data["quant_score"]
            details["quant_score"] = quant_score
            score += int(quant_score * 0.3)

        # Order Flow
        if use_order_flow and ORDER_FLOW_ENABLED:
            order_flow_data = get_order_flow_signals(symbol)
            of_score = order_flow_data["order_flow_score"]
            details["order_flow_score"] = of_score
            score += int(of_score * 0.2)

        # ML
        details["ml_probability"] = 0
        details["ml_signal"] = "neutral"

        details["ma_cross"] = check_ma_crossover(df_ta, side="long")
        details["vol_spike_ok"] = volume_spike_guard(df_ta)

        return {
            "score": max(0, min(100, score)),
            "details": details,
            "price": price,
            "sr": sr,
            "df_ta": df_ta,
            "df_1h": df_1h
        }

    except Exception as e:
        log.warning(f"Ошибка анализа {symbol}: {e}")
        return {"score": 0, "details": {}, "price": 0, "sr": {}}

def get_score_short(symbol: str) -> dict:
    """Получает скор для шорта."""
    res = get_score(symbol)
    if res["score"] == 0:
        return res
    res["score"] = max(0, 100 - res["score"] - 10)
    try:
        raw = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA, limit=300)
        if len(raw) >= 50:
            df = pd.DataFrame(raw, columns=["ts", "o", "h", "l", "c", "v"])
            res["details"]["ma_cross"] = check_ma_crossover(df, side="short")
    except:
        pass
    return res

def apply_ai_correction(score: int, symbol: str) -> int:
    """Применяет AI корректировку."""
    ai = get_bybit_ai(symbol)
    if not ai["available"]:
        return score
    if ai["signal"] == "bullish":
        return min(100, score + 5)
    elif ai["signal"] == "bearish":
        return max(0, score - 15)
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
    """Рассчитывает размер позиции с учётом волатильности."""
    # Получаем ATR для оценки волатильности
    atr_pct = sl_dist_pct  # Можно заменить на реальный ATR

    # Корректируем риск в зависимости от волатильности
    volatility_factor = min(max(0.5, atr_pct / 2.0), 2.0)  # При высокой волатильности уменьшаем размер позиции

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
# МОНТЕ-КАРЛО
# ============================================================
def run_monte_carlo(trades: List[dict]) -> Dict[str, Any]:
    """Monte Carlo симуляция."""
    try:
        if len(trades) < 10:
            return {"valid": False}
        pnls = [t["pnl_usdt"] for t in trades]
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls)
        sim_results = np.random.normal(mean_pnl, std_pnl, (MONTE_CARLO_SIMULATIONS, MONTE_CARLO_DAYS)).cumsum(axis=1)

        return {
            "valid": True,
            "percentile_5": float(np.percentile(sim_results[:, -1], 5)),
            "percentile_50": float(np.percentile(sim_results[:, -1], 50)),
            "percentile_95": float(np.percentile(sim_results[:, -1], 95)),
            "loss_probability": float(np.mean(sim_results[:, -1] < 0) * 100),
        }
    except Exception as e:
        log.error(f"Monte Carlo ошибка: {e}")
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
        "rsi": lambda v: 25 <= float(v) <= 42,
        "rsi_1h": lambda v: float(v) < 55,
        "macd": lambda v: v == "bullish",
        "range_filter": lambda v: v == "up",
        "supertrend": lambda v: v == "up",
        "hull": lambda v: v == "up",
        "trend_1h": lambda v: v == "bullish",
        "adx": lambda v: float(v) > 25,
        "stoch_k": lambda v: float(v) < 25,
        "volume_ratio": lambda v: float(v) > 1.5,
        "sr_signal": lambda v: "support" in str(v),
        "bayes_prob": lambda v: float(v) > 0.6,
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
    log.info("📈 ЭФФЕКТИВНОСТЬ ИНДИКАТОРОВ")
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
# ЛОГИРОВАНИЕ ML ДАННЫХ
# ============================================================
def log_ml_data(symbol: str, features: Dict[str, float], prediction: Dict[str, Any],
               trade_result: str = None, pnl: float = None):
    """Логирует данные для обучения ML."""
    if not ML_LOG_DATA:
        return

    try:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "features": features,
            "prediction": prediction,
            "trade_result": trade_result,
            "pnl_usdt": pnl,
            "version": ML_FEATURES_VERSION,
        }
        with open(ML_LOG_FILE, "a", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False)
            f.write("\n")
    except Exception as e:
        log.warning(f"Ошибка логирования ML: {e}")

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

        # Простые сигналы
        rsi = calc_rsi(close).iloc[-1]
        ema20 = _ema(close, 20).iloc[-1]
        ema50 = _ema(close, 50).iloc[-1]

        # Вход в лонг
        if position is None and rsi < 40 and ema20 > ema50:
            position = "long"
            entry_price = close.iloc[-1]
            entry_idx = i
        # Выход
        elif position == "long":
            pnl_pct = (close.iloc[-1] - entry_price) / entry_price * 100

            # TP 3% или SL 1%
            if pnl_pct >= 3.0:
                results.append({"result": "tp", "pnl_pct": pnl_pct, "bars": i - entry_idx})
                position = None
            elif pnl_pct <= -1.0:
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
        st_up, st_down = calc_supertrend(df)
        _, _, _, rf_up, rf_down = calc_range_filter(df)

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
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                send_telegram_message(f"⏰ Таймаут для {symbol}: позиция закрыта по времени")
            return "timeout"

        try:
            # Проверяем соединение с биржей
            if not check_connection():
                log.error("Нет соединения с биржей — аварийное закрытие")
                emergency_close_position(symbol, side)
                if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                    send_telegram_message(f"🚨 Нет соединения с биржей! Позиция {symbol} закрыта аварийно")
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

                result = "tp" if hit_tp else "sl"
                if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                    pnl = (cur_price - entry_price) * qty * (1 if side == "long" else -1)
                    send_telegram_message(f"{'✅' if result == 'tp' else '❌'} {result.upper()} для {symbol}: P&L = {pnl:+.4f} USDT")
                return result

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
                if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                    send_telegram_message(f"🔄 Signal Exit для {symbol}: разворот")
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
            if int(now) % 60 < 20:  # Каждую ~минуту
                log.info(f"[{coin}] {cur_price:.8f} P&L={pnl_pct:+.2f}% ({unrealized_pnl:+.4f}U) "
                         f"SL={current_sl:.8f} фаза={phase} до_дед={time_to_deadline}с")

        except Exception as e:
            consecutive_errors += 1
            log.warning(f"Ошибка мониторинга ({consecutive_errors}/{max_errors}): {e}")

            if consecutive_errors >= max_errors:
                log.critical("Слишком много ошибок подряд — аварийное закрытие")
                emergency_close_position(symbol, side)
                if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                    send_telegram_message(f"🚨 Слишком много ошибок для {symbol}! Позиция закрыта аварийно")
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
    """Подтверждает вход с задержкой."""
    if ENTRY_CONFIRM_BARS <= 0:
        return True

    tf_seconds = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "1h": 3600}
    wait = tf_seconds.get(TIMEFRAME_TA, 300) * ENTRY_CONFIRM_BARS

    log.info(f"Подтверждение входа: ждём {wait}с ({ENTRY_CONFIRM_BARS} свечи)...")
    time.sleep(wait)

    new_result = get_score(symbol) if side == "long" else get_score_short(symbol)
    new_score = new_result["score"]

    log.info(f"Перепроверка: {original_score} → {new_score} (мин={ENTRY_CONFIRM_MIN_SCORE})")

    if new_score < ENTRY_CONFIRM_MIN_SCORE:
        log.info(f"Подтверждение не прошло: скор упал до {new_score}")
        return False

    if not new_result.get("details", {}).get("vol_spike_ok", True):
        log.info("Подтверждение не прошло: volume spike")
        return False

    # Проверяем, не изменился ли тренд
    if side == "long" and not trend_4h_bullish(symbol):
        log.info("Подтверждение не прошло: тренд на 4H изменился")
        return False
    elif side == "short" and not trend_4h_bearish(symbol):
        log.info("Подтверждение не прошло: тренд на 4H изменился")
        return False

    log.info(f"Вход подтверждён. Скор {new_score}/100")
    return True

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
    log.info(f"📊 ОТЧЁТ {BOT_VERSION}")
    log.info(f"Баланс: {balance:.2f} USDT ({delta:+.2f} / {pct:+.2f}%)")
    log.info(f"Сделок: {total} TP={tp} SL={sl} Таймаут={state['timeout']}")
    log.info(f"WinRate: {wr:.1f}%")
    log.info(f"Прибыль/Убыток: {state['profit_usdt']:.4f} / {state['loss_usdt']:.4f} USDT")
    log.info(f"Чистый P&L: {net:+.4f} USDT")

    # ML статус
    if ML_ENABLED:
        log.info("-" * 65)
        log.info("🤖 ML СТАТУС:")
        log.info(f"Обучена: {'Да' if ml_model.trained else 'Нет'}")
        if ml_model.trained:
            log.info(f"Accuracy: {ml_model.accuracy:.2f} | Precision: {ml_model.precision:.2f}")

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

    # Monte Carlo
    if MONTE_CARLO_ENABLED and len(history) > 20:
        if time.time() - state.get("monte_carlo_last_run", 0) > 3600:
            mc = run_monte_carlo(history)
            if mc["valid"]:
                state["monte_carlo_last_run"] = time.time()
                log.info("🎲 MONTE CARLO:")
                log.info(f"5-й перцентиль: {mc['percentile_5']:.2f} USDT")
                log.info(f"50-й перцентиль: {mc['percentile_50']:.2f} USDT")
                log.info(f"Вероятность убытка: {mc['loss_probability']:.1f}%")

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

    # Логируем для ML (если есть pending запись)
    trade_id = trade_record.get("id")
    if trade_id and trade_id in pending_ml_entries:
        entry = pending_ml_entries[trade_id]
        log_ml_data(
            symbol=entry["symbol"],
            features=entry["features"],
            prediction=entry["prediction"],
            trade_result=result,
            pnl=pnl
        )
        del pending_ml_entries[trade_id]

    # Обновляем статистику индикаторов
    update_indicator_stats(trade_record)

# ============================================================
# РАСЧЁТ SL/TP
# ============================================================
def calc_sl_tp(symbol: str, price: float, side: str, sr_info: dict) -> Tuple[float, float, float, float]:
    """Рассчитывает SL, TP и расстояния с учётом ATR и S/R."""
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
# ОБУЧЕНИЕ ML
# ============================================================
def maybe_retrain_ml():
    """Переобучает ML модель если нужно."""
    global stats

    if not ML_ENABLED or not ML_LOG_DATA:
        return

    stats["ml_trades_since_retrain"] = stats.get("ml_trades_since_retrain", 0) + 1

    if stats["ml_trades_since_retrain"] >= ML_RETRAIN_INTERVAL:
        log.info("🔄 Переобучение ML модели...")
        if ml_model.train():
            ml_model.save_model()
            stats["ml_trades_since_retrain"] = 0
            log.info("✅ ML переобучена")
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                send_telegram_message("🤖 ML модель переобучена")
