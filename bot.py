#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ГИБРИДНЫЙ БОТ v10.4_hybrid — СТАКАННЫЙ СКАЛЬПЕР + 7 СТРАТЕГИЙ
=============================================================
- Архитектура v10.4: подхват позиций, ATR SL/TP, трейлинг, частичный безубыток.
- Сигналы генерируются 7 стратегиями-лидерами (Keltner, Stochastic, CCI, MFI, Aroon, VWAP, Stoch+Trend).
- Каждый сигнал проверяется трендом 4h и стаканом.
- Лучший сигнал выбирается по объёму стены.
"""

import os, time, logging, ccxt
import pandas as pd, numpy as np
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

load_dotenv()

# ======================= КОНФИГУРАЦИЯ =======================
SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT", "XRP/USDT:USDT",
    "SOL/USDT:USDT", "ADA/USDT:USDT", "TRX/USDT:USDT",
    "AVAX/USDT:USDT", "DOT/USDT:USDT", "LTC/USDT:USDT", "BCH/USDT:USDT",
    "ATOM/USDT:USDT", "XLM/USDT:USDT", "NEAR/USDT:USDT", "DOGE/USDT:USDT",
    "1000PEPE/USDT:USDT", "WIF/USDT:USDT", "BOME/USDT:USDT",
    "RENDER/USDT:USDT", "TAO/USDT:USDT", "WLD/USDT:USDT", "ARKM/USDT:USDT",
    "IO/USDT:USDT", "ONDO/USDT:USDT", "VIRTUAL/USDT:USDT", "UNI/USDT:USDT",
    "AAVE/USDT:USDT", "ARB/USDT:USDT", "OP/USDT:USDT", "LINK/USDT:USDT",
    "GRT/USDT:USDT", "INJ/USDT:USDT", "SUI/USDT:USDT", "APT/USDT:USDT",
    "TIA/USDT:USDT", "JTO/USDT:USDT", "EIGEN/USDT:USDT", "HBAR/USDT:USDT",
    "VET/USDT:USDT", "NOT/USDT:USDT", "CATI/USDT:USDT",
]

TIMEFRAME_TA = "5m"
TIMEFRAME_4H = "4h"
SCAN_INTERVAL = 120

# --- Параметры стратегий (полностью из v10.4) ---
ORDER_BOOK_DEPTH = 20
WALL_THRESHOLD_VOL_RATIO = 3.0
MIN_WALL_VOLUME_USDT = 500
MAX_WALL_DISTANCE_PCT = 2.0
IMBALANCE_RATIO_LONG = 1.5
IMBALANCE_RATIO_SHORT = 1 / 1.5

ATR_SL_MULT = 1.5
ATR_TP_MULT = 3.0
MIN_SL_PERCENT = 0.8
MAX_SL_PERCENT = 2.0
TP_PERCENT = 3.0

PARTIAL_BE_ENABLED = True
PARTIAL_BE_CLOSE_PCT = 50.0
PARTIAL_BE_PROFIT = 0.2
TRAILING_ATR_PERIOD = 14
TRAILING_ATR_MULT = 2.0
TRAILING_OFFSET_MULT = 1.5
MIN_TRAILING_STEP = 0.4
MIN_TRAILING_OFFSET = 0.6
MIN_PROFIT_FOR_TRAIL = 1.0
RR_EXIT_TRIGGER = 0.6
SIGNAL_EXIT_ENABLED = True

DAILY_LOSS_LIMIT_PCT = 3.0
DAILY_LOSS_PAUSE_SEC = 10800
SYMBOL_BLOCK_AFTER_TP = 90 * 60
SYMBOL_BLOCK_AFTER_SL = 180 * 60
SYMBOL_MAX_FAIL_ATTEMPTS = 3
SYMBOL_BLOCK_AFTER_FAIL = 120 * 60
SL_STREAK_LIMIT = 3
SL_STREAK_PAUSE = 1800
MIN_BALANCE = 5.0
MAX_DRAWDOWN_PCT = 15.0
TRADE_MAX_LIFETIME = 7200
REPORT_INTERVAL = 1800
BYBIT_FEE = 0.00055
RISK_PCT = 0.8

LEVERAGE_MIN = 3
LEVERAGE_MAX = 5

STATE_FILE = "state_bot_v10.4_hybrid.json"
TRADES_FILE = "trades_bot_v10.4_hybrid.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler("bot_v10.4_hybrid.log", encoding="utf-8")],
)
log = logging.getLogger("HybridBot")

exchange = ccxt.bybit({
    "apiKey": os.getenv("BYBIT_API_KEY"),
    "secret": os.getenv("BYBIT_API_SECRET"),
    "enableRateLimit": True,
    "options": {"defaultType": "linear"},
})

stats = {
    "запусков": 0, "сделок_всего": 0, "тейкпрофит": 0, "стоплосс": 0,
    "таймаут": 0, "прибыль_usdt": 0.0, "убыток_usdt": 0.0,
    "депозит_старт": 0.0, "баланс_начало_дня": 0.0, "дата_дня": "",
    "старт_время": "", "последний_отчёт": 0.0, "sl_streak": 0,
}

# ======================= ИНДИКАТОРЫ =======================
def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def sma(s, p): return s.rolling(p).mean()
def rsi(s, p=14):
    d = s.diff(); g = d.clip(lower=0); l = (-d).clip(lower=0)
    ag = g.ewm(alpha=1/p, adjust=False).mean(); al = l.ewm(alpha=1/p, adjust=False).mean()
    return 100 - 100/(1+ag/al.replace(0, np.nan))
def atr(df, p=14):
    h, l, c = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h-l, (h-c).abs(), (l-c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, adjust=False).mean()
def bollinger(s, p=20, std=2):
    mb = sma(s, p); sd = s.rolling(p).std(); return mb+std*sd, mb, mb-std*sd
def macd(s, f=12, sl=26, sig=9):
    ml = ema(s, f)-ema(s, sl); sl_ = ema(ml, sig); return ml, sl_, ml-sl_
def stoch(df, kp=14, dp=3):
    lo = df["low"].rolling(kp).min(); hi = df["high"].rolling(kp).max()
    k = 100*(df["close"]-lo)/(hi-lo+1e-10); d = k.rolling(dp).mean()
    return k, d
def cci(df, p=20):
    tp = (df["high"]+df["low"]+df["close"])/3
    return (tp-sma(tp,p))/(0.015*tp.rolling(p).apply(lambda x: np.abs(x-x.mean()).mean()))
def mfi(df, p=14):
    tp = (df["high"]+df["low"]+df["close"])/3; mf = tp*df["volume"]
    pf = mf.where(tp>tp.shift(1),0); nf = mf.where(tp<tp.shift(1),0)
    return 100-100/(1+pf.rolling(p).sum()/nf.rolling(p).sum().replace(0,np.nan))
def aroon(df, p=25):
    h_max = df["high"].rolling(p).apply(lambda x: x.argmax())/p*100
    l_min = df["low"].rolling(p).apply(lambda x: x.argmin())/p*100
    return 100-h_max, 100-l_min
def keltner(df, p=20, mult=1.5):
    e = ema(df["close"], p); a = atr(df, p); return e+mult*a, e, e-mult*a
def vwap_bands(df, p=100):
    tp = (df["high"]+df["low"]+df["close"])/3
    cv = df["volume"].cumsum(); cvp = (tp*df["volume"]).cumsum()
    vwap = cvp/cv; std = tp.rolling(p).std()
    return vwap, std

# ======================= СТРАТЕГИИ =======================
class Strategy(ABC):
    def __init__(self, name): self.name = name
    @abstractmethod
    def signal(self, df: pd.DataFrame) -> int: pass

class KeltnerRev(Strategy):
    def signal(self, df):
        u, _, l = keltner(df); c = df["close"].iloc[-1]
        return 1 if c < l.iloc[-1] else (-1 if c > u.iloc[-1] else 0)

class StochasticStrat(Strategy):
    def signal(self, df):
        k, d = stoch(df); kc, dc = k.iloc[-1], d.iloc[-1]
        return 1 if (kc<20 and kc>dc) else (-1 if (kc>80 and kc<dc) else 0)

class CCIStrat(Strategy):
    def signal(self, df):
        c = cci(df).iloc[-1]; return 1 if c<-100 else (-1 if c>100 else 0)

class MFIStrat(Strategy):
    def signal(self, df):
        m = mfi(df).iloc[-1]; return 1 if m<20 else (-1 if m>80 else 0)

class AroonStrat(Strategy):
    def signal(self, df):
        u, d = aroon(df); return 1 if (u.iloc[-1]>70 and d.iloc[-1]<30) else (-1 if (d.iloc[-1]>70 and u.iloc[-1]<30) else 0)

class VWAPRev(Strategy):
    def signal(self, df):
        v, s = vwap_bands(df); c = df["close"].iloc[-1]
        return 1 if c < v.iloc[-1]-2*s.iloc[-1] else (-1 if c > v.iloc[-1]+2*s.iloc[-1] else 0)

class StochTrend(Strategy):
    def signal(self, df):
        k, d = stoch(df); e50 = ema(df["close"],50)
        return 1 if (k.iloc[-1]<20 and k.iloc[-1]>d.iloc[-1] and df["close"].iloc[-1]>e50.iloc[-1]) else (-1 if (k.iloc[-1]>80 and k.iloc[-1]<d.iloc[-1] and df["close"].iloc[-1]<e50.iloc[-1]) else 0)

STRATEGIES = [KeltnerRev("Keltner Rev"), StochasticStrat("Stochastic"), CCIStrat("CCI"),
              MFIStrat("MFI"), AroonStrat("Aroon"), VWAPRev("VWAP Rev"), StochTrend("Stoch+Trend")]

# ======================= ФИЛЬТРЫ =======================
def safe_api(func, *a, retries=3, delay=1.0, **kw):
    for attempt in range(retries):
        try:
            return func(*a, **kw)
        except (ccxt.RateLimitExceeded, ccxt.NetworkError) as e:
            log.warning(f"API retry {attempt+1}: {e}")
            time.sleep(delay); delay *= 2
        except Exception as e:
            if attempt == retries-1: raise
            time.sleep(delay); delay *= 2
    return None

def fetch_ohlcv(symbol, tf, limit=150):
    try: return safe_api(exchange.fetch_ohlcv, symbol, tf, limit=limit) or []
    except: return []

def fetch_ticker(symbol):
    try: return safe_api(exchange.fetch_ticker, symbol)
    except: return None

def fetch_positions(symbols=None):
    try:
        if symbols: return safe_api(exchange.fetch_positions, symbols) or []
        return safe_api(exchange.fetch_positions) or []
    except: return []

def get_balance(free=True):
    try:
        bal = exchange.fetch_balance({"type": "linear"})
        return float(bal.get("USDT", {}).get("free" if free else "total", 0))
    except: return 0.0

def trend_4h(symbol, direction="bull"):
    raw = fetch_ohlcv(symbol, TIMEFRAME_4H, limit=60)
    if len(raw) < 55: return False
    df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"]).set_index("timestamp")
    e20 = ema(df["close"],20).iloc[-1]; e50 = ema(df["close"],50).iloc[-1]
    return e20>e50 if direction=="bull" else e20<e50

def order_book_filter(symbol, side):
    ob = safe_api(exchange.fetch_order_book, symbol, ORDER_BOOK_DEPTH)
    if not ob or len(ob["bids"])<5 or len(ob["asks"])<5: return False, 0
    bids = ob["bids"]; asks = ob["asks"]
    spread = (asks[0][0]-bids[0][0])/bids[0][0]*100
    if spread > 1.0: return False, 0
    bid_vols = [v for _,v in bids[1:]]; ask_vols = [v for _,v in asks[1:]]
    med_bid = np.median(bid_vols) if bid_vols else 0
    med_ask = np.median(ask_vols) if ask_vols else 0
    total_bid = sum(v for _,v in bids[:ORDER_BOOK_DEPTH])
    total_ask = sum(v for _,v in asks[:ORDER_BOOK_DEPTH])
    imb = total_bid/(total_ask+1e-10)
    if side == 1:
        wall_vol = bids[0][1]
        for p,v in bids[:5]:
            if v >= WALL_THRESHOLD_VOL_RATIO*med_bid: wall_vol = v; break
        return (wall_vol*bids[0][0] >= MIN_WALL_VOLUME_USDT and imb > IMBALANCE_RATIO_LONG), wall_vol*bids[0][0]
    else:
        wall_vol = asks[0][1]
        for p,v in asks[:5]:
            if v >= WALL_THRESHOLD_VOL_RATIO*med_ask: wall_vol = v; break
        return (wall_vol*asks[0][0] >= MIN_WALL_VOLUME_USDT and imb < IMBALANCE_RATIO_SHORT), wall_vol*asks[0][0]

# ======================= ТОРГОВЫЕ ФУНКЦИИ (из v10.4 без изменений) =======================
def choose_leverage(atr_pct):
    if atr_pct > 1.5: return LEVERAGE_MIN
    elif atr_pct > 0.8: return LEVERAGE_MIN + 1
    else: return LEVERAGE_MAX

def set_leverage(symbol, lev):
    try:
        exchange.set_leverage(lev, symbol, params={"buyLeverage": lev, "sellLeverage": lev})
        return True
    except Exception as e1:
        if "leverage not modified" in str(e1).lower(): return True
    try:
        coin = symbol.replace("/", "").replace(":USDT", "")
        exchange.private_post_v5_position_set_leverage({
            "category": "linear", "symbol": coin,
            "buyLeverage": str(lev), "sellLeverage": str(lev),
        })
        return True
    except Exception as e2:
        if "leverage not modified" in str(e2).lower(): return True
    log.warning(f"Плечо не удалось установить для {symbol}")
    return True

def open_position(symbol, side, qty, tp_price, sl_price, leverage):
    set_leverage(symbol, leverage)
    ticker = fetch_ticker(symbol)
    if not ticker or ticker.get("last") is None:
        log.error(f"Не удалось получить цену для {symbol}")
        return None, None
    price = float(ticker["last"])
    s = "buy" if side == "long" else "sell"
    try:
        order = exchange.create_market_order(symbol, s, qty, params={
            "takeProfit": float(exchange.price_to_precision(symbol, tp_price)),
            "stopLoss": float(exchange.price_to_precision(symbol, sl_price)),
        })
        entry = float(order.get("average", price))
        log.info(f"{side.upper()} открыт: {qty} @ {entry:.6f} (плечо {leverage}x)")
        return entry, qty
    except Exception as e:
        log.error(f"Ошибка открытия {side} {symbol}: {e}")
        return None, None

def close_position(symbol, qty, side):
    s = "sell" if side == "long" else "buy"
    for _ in range(3):
        try:
            exchange.create_market_order(symbol, s, qty, params={"reduceOnly": True})
            return True
        except: time.sleep(2)
    return False

def update_sl(symbol, new_sl, side):
    try:
        coin = symbol.replace("/", "").replace(":USDT", "")
        exchange.private_post_v5_position_trading_stop({
            "category": "linear", "symbol": coin,
            "stopLoss": str(exchange.price_to_precision(symbol, new_sl)),
            "slTriggerBy": "MarkPrice", "positionIdx": "0",
        })
        log.info(f"SL обновлён → {new_sl:.6f}")
        return True
    except Exception as e:
        log.warning(f"Не удалось обновить SL: {e}")
        return False

def update_tp(symbol, new_tp, side):
    try:
        coin = symbol.replace("/", "").replace(":USDT", "")
        exchange.private_post_v5_position_trading_stop({
            "category": "linear", "symbol": coin,
            "takeProfit": str(exchange.price_to_precision(symbol, new_tp)),
            "tpTriggerBy": "MarkPrice", "positionIdx": "0",
        })
        log.info(f"TP обновлён → {new_tp:.6f}")
        return True
    except Exception as e:
        log.warning(f"Не удалось обновить TP: {e}")
        return False

def monitor_position(symbol, entry, qty, start_time, sl_price, tp_price, side, atr):
    deadline = start_time + TRADE_MAX_LIFETIME
    be_done = False; trailing_active = False; peak = entry; current_sl = sl_price
    trailing_offset = max(MIN_TRAILING_OFFSET/100, atr/entry * TRAILING_OFFSET_MULT)
    rr_trigger = entry + (tp_price - entry) * RR_EXIT_TRIGGER if side == "long" else entry - (entry - tp_price) * RR_EXIT_TRIGGER
    partial_done = False; accumulated_pnl = 0.0
    log.info(f"Мониторинг {symbol} {side} вход={entry:.6f} SL={sl_price:.6f} TP={tp_price:.6f}")
    while True:
        now = time.time()
        if now >= deadline:
            log.warning("Дедлайн — закрытие"); close_position(symbol, qty, side); return "timeout", accumulated_pnl
        time.sleep(15)
        pos_list = fetch_positions([symbol])
        active = [p for p in pos_list if float(p.get("contracts",0) or 0) > 0 and p.get("side") == side]
        if not active:
            ticker = fetch_ticker(symbol)
            cur = float(ticker["last"]) if ticker and ticker.get("last") else entry
            return ("tp", accumulated_pnl) if (cur >= tp_price if side=="long" else cur <= tp_price) or be_done else ("sl", accumulated_pnl)
        pos = active[0]; ticker = fetch_ticker(symbol)
        if not ticker or ticker.get("last") is None: continue
        cur = float(ticker["last"])
        qty_act = abs(float(pos.get("contracts",0) or 0))
        pnl_pct = (cur/entry - 1)*100 if side=="long" else (entry/cur - 1)*100
        if PARTIAL_BE_ENABLED and not partial_done and pnl_pct >= PARTIAL_BE_PROFIT:
            close_qty = qty_act * (PARTIAL_BE_CLOSE_PCT/100)
            if close_qty > 0:
                close_s = "sell" if side=="long" else "buy"
                try:
                    exchange.create_market_order(symbol, close_s, close_qty, params={"reduceOnly": True})
                    partial_pnl = (cur - entry) * close_qty if side=="long" else (entry - cur) * close_qty
                    accumulated_pnl += partial_pnl
                    log.info(f"Частичный безубыток: {close_qty:.4f} PnL≈{partial_pnl:+.4f}U")
                    qty_act -= close_qty
                    new_sl = entry * (1 + BYBIT_FEE*2 + 0.0003) if side=="long" else entry * (1 - BYBIT_FEE*2 - 0.0003)
                    if update_sl(symbol, new_sl, side): current_sl = new_sl
                    partial_done = True
                except Exception as e:
                    log.warning(f"Ошибка частичного закрытия: {e}")
        if SIGNAL_EXIT_ENABLED and be_done and pnl_pct > 0.5:
            if (side=="long" and not trend_4h(symbol,"bull")) or (side=="short" and not trend_4h(symbol,"bear")):
                log.info("Signal exit по 4h тренду"); close_position(symbol, qty_act, side); return "tp", accumulated_pnl
        if not partial_done and not be_done and pnl_pct >= 0.3:
            new_sl = entry * (1 + BYBIT_FEE*2 + 0.0003) if side=="long" else entry * (1 - BYBIT_FEE*2 - 0.0003)
            if update_sl(symbol, new_sl, side): current_sl = new_sl; be_done = True; log.info("Безубыток")
        if be_done:
            if not trailing_active and ((side=="long" and cur>=rr_trigger) or (side=="short" and cur<=rr_trigger)):
                trailing_active = True; peak = cur; log.info("Трейлинг активирован")
            if trailing_active and pnl_pct >= MIN_PROFIT_FOR_TRAIL:
                if side == "long":
                    if cur > peak: peak = cur
                    new_sl = peak * (1 - trailing_offset)
                    if new_sl > current_sl and update_sl(symbol, new_sl, side): current_sl = new_sl
                else:
                    if cur < peak: peak = cur
                    new_sl = peak * (1 + trailing_offset)
                    if new_sl < current_sl and update_sl(symbol, new_sl, side): current_sl = new_sl
        log.info(f"[{symbol}] {cur:.6f} P&L={pnl_pct:+.2f}% SL={current_sl:.6f}")
    return "sl", accumulated_pnl

def handle_existing_position(pos):
    symbol = pos["symbol"]; side = pos["side"]; entry = float(pos["entryPrice"] or pos["avgCost"] or 0)
    qty = abs(float(pos["contracts"]))
    if entry <= 0 or qty <= 0: return
    raw = fetch_ohlcv(symbol, TIMEFRAME_TA, 50)
    df_ta = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"]).set_index("timestamp")
    atr_val = atr(df_ta, 14).iloc[-1] if len(df_ta) > 14 else entry * 0.01
    price = float(fetch_ticker(symbol)["last"]) if fetch_ticker(symbol) else entry
    sl_dist = atr_val * ATR_SL_MULT; tp_dist = atr_val * ATR_TP_MULT
    if side == "long":
        sl = max(price * (1 - MAX_SL_PERCENT/100), min(price * (1 - MIN_SL_PERCENT/100), price - sl_dist))
        tp = max(price * (1 + TP_PERCENT/100), price + tp_dist)
    else:
        sl = min(price * (1 + MAX_SL_PERCENT/100), max(price * (1 + MIN_SL_PERCENT/100), price + sl_dist))
        tp = min(price * (1 - TP_PERCENT/100), price - tp_dist)
    update_sl(symbol, sl, side); update_tp(symbol, tp, side)
    log.info(f"Подхвачена позиция {symbol} {side} entry={entry:.6f} новые SL={sl:.6f} TP={tp:.6f}")
    result, pnl = monitor_position(symbol, entry, qty, time.time()-60, sl, tp, side, atr_val)
    cur_price = float(fetch_ticker(symbol).get("last",entry)) if fetch_ticker(symbol) else entry
    total_pnl = (cur_price - entry) * qty if side=="long" else (entry - cur_price) * qty
    stats["прибыль_usdt" if result=="tp" else "убыток_usdt"] += max(0,total_pnl) if result=="tp" else abs(min(0,total_pnl))
    stats["тейкпрофит" if result=="tp" else "стоплосс"] += 1
    stats["сделок_всего"] += 1
    stats["sl_streak"] = 0 if result=="tp" else stats.get("sl_streak",0)+1
    log.info(f"Завершена подхваченная позиция: {result} PnL={total_pnl:.4f} USDT")

# ======================= ГЛАВНЫЙ ЦИКЛ =======================
def main():
    global stats
    if not os.getenv("BYBIT_API_KEY"):
        log.error("Нет API ключей"); return
    stats["депозит_старт"] = get_balance(free=False)
    stats["старт_время"] = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    log.info(f"=== ГИБРИДНЫЙ БОТ v10.4_hybrid ===")
    log.info(f"Депозит: {stats['депозит_старт']:.2f} USDT")

    existing = [p for p in fetch_positions() if float(p.get("contracts",0))>0]
    if existing:
        log.info(f"Найдены открытые позиции: {[(p['symbol'], p['side']) for p in existing]}")
        for pos in existing: handle_existing_position(pos)
        log.info("Все существующие позиции обработаны")

    заблокированные: Dict[str, float] = {}
    fail_attempts: Dict[str, int] = {}

    while True:
        try:
            if time.time() - stats.get("последний_отчёт",0) >= REPORT_INTERVAL:
                bal = get_balance(free=False)
                log.info(f"📊 Отчёт: Баланс={bal:.2f} USDT | Сделок: {stats['сделок_всего']} | TP: {stats['тейкпрофит']} SL: {stats['стоплосс']}")
                stats["последний_отчёт"] = time.time()

            свободный = get_balance(free=True)
            if свободный < MIN_BALANCE:
                active = [p for p in fetch_positions() if float(p.get("contracts",0))>0]
                if not active:
                    log.warning("Мало средств, жду 300с"); time.sleep(300); continue

            if stats["депозит_старт"] > 0:
                loss = (stats["депозит_старт"] - get_balance(free=False)) / stats["депозит_старт"] * 100
                if loss >= DAILY_LOSS_LIMIT_PCT:
                    log.warning(f"Дневной лимит {loss:.2f}%, пауза {DAILY_LOSS_PAUSE_SEC//60} мин")
                    time.sleep(DAILY_LOSS_PAUSE_SEC); continue

            open_positions = [p for p in fetch_positions() if float(p.get("contracts",0))>0]
            if open_positions:
                log.info(f"Открытые позиции: {[(p['symbol'], p['side']) for p in open_positions]}")
                for pos in open_positions: handle_existing_position(pos)
                continue

            # СКАНИРОВАНИЕ
            log.info("── Сканирование ──")
            signals = []
            for sym in SYMBOLS:
                if sym in заблокированные and time.time() < заблокированные[sym]: continue
                raw = fetch_ohlcv(sym, TIMEFRAME_TA, 100)
                if len(raw) < 50: continue
                df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"]).set_index("timestamp")
                a = atr(df).iloc[-1] if len(df) > 14 else df["close"].iloc[-1]*0.01
                price = df["close"].iloc[-1]
                for st in STRATEGIES:
                    sig = st.signal(df)
                    if sig == 0: continue
                    side_str = "long" if sig == 1 else "short"
                    if not trend_4h(sym, "bull" if sig==1 else "bear"): continue
                    time.sleep(0.1)
                    ob_ok, wall_vol = order_book_filter(sym, sig)
                    if not ob_ok: continue
                    signals.append((wall_vol, sym, side_str, st.name, price, a))
                    log.info(f"Сигнал: {st.name} {sym} {side_str.upper()} wall={wall_vol:.0f} USDT")

            log.info(f"Найдено {len(signals)} сигналов")
            if not signals:
                time.sleep(SCAN_INTERVAL); continue

            signals.sort(reverse=True)
            _, sym, side, sname, price, a = signals[0]
            sl_dist = a * ATR_SL_MULT; tp_dist = a * ATR_TP_MULT
            sl = price - sl_dist if side=="long" else price + sl_dist
            tp = price + tp_dist if side=="long" else price - tp_dist
            sl = max(price*(1-MAX_SL_PERCENT/100), min(price*(1-MIN_SL_PERCENT/100), sl)) if side=="long" else min(price*(1+MAX_SL_PERCENT/100), max(price*(1+MIN_SL_PERCENT/100), sl))
            tp = max(price*(1+TP_PERCENT/100), tp) if side=="long" else min(price*(1-TP_PERCENT/100), tp)
            rr = abs(tp-price)/abs(sl-price) if abs(price-sl)>0 else 0
            if rr < 2.0: continue

            atr_pct = (a/price)*100
            leverage = choose_leverage(atr_pct)
            qty = (свободный * RISK_PCT / 100) / abs(sl - price)
            try: qty = float(exchange.amount_to_precision(sym, qty))
            except: pass
            if qty <= 0: continue

            log.info(f"✅ Вход {side.upper()} {sym} цена={price:.6f} SL={sl:.6f} TP={tp:.6f} плечо={leverage}x")
            entry, qty_open = open_position(sym, side, qty, tp, sl, leverage)
            if not entry:
                log.warning("Не удалось открыть позицию")
                fail_attempts[sym] = fail_attempts.get(sym,0)+1
                if fail_attempts[sym] >= SYMBOL_MAX_FAIL_ATTEMPTS:
                    заблокированные[sym] = time.time() + SYMBOL_BLOCK_AFTER_FAIL
                    fail_attempts.pop(sym, None)
                continue
            fail_attempts.pop(sym, None)

            stats["сделок_всего"] += 1
            start_t = time.time()
            result, partial_pnl = monitor_position(sym, entry, qty_open, start_t, sl, tp, side, a)
            cur_price = float(fetch_ticker(sym).get("last",entry)) if fetch_ticker(sym) else entry
            total_pnl = (cur_price-entry)*qty_open if side=="long" else (entry-cur_price)*qty_open
            total_pnl += partial_pnl
            stats["прибыль_usdt" if result=="tp" else "убыток_usdt"] += max(0,total_pnl) if result=="tp" else abs(min(0,total_pnl))
            stats["тейкпрофит" if result=="tp" else "стоплосс"] += 1
            stats["sl_streak"] = 0 if result=="tp" else stats.get("sl_streak",0)+1
            заблокированные[sym] = time.time() + (SYMBOL_BLOCK_AFTER_TP if result=="tp" else SYMBOL_BLOCK_AFTER_SL)
            log.info(f"Сделка закрыта: {result} PnL={total_pnl:.4f} USDT")
            if stats["sl_streak"] >= SL_STREAK_LIMIT:
                log.warning("Серия SL, пауза"); time.sleep(SL_STREAK_PAUSE); stats["sl_streak"] = 0
            time.sleep(30)

        except Exception as e:
            log.error(f"Ошибка в цикле: {e}", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    main()
