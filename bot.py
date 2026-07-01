#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AGGRESSIVE MULTI-STRATEGY SCALPER v10.7
=======================================
- Сигнал открывается, если минимум 2 стратегии дают одинаковое направление.
- ДОБАВЛЕН ФИЛЬТР ОБЪЁМА: текущий объём > 1.5 * средний объём за 20 свечей.
- Плечо 20x, фиксированная маржа 3 USDT.
- Улучшенный риск-менеджмент (ATR SL/TP 1.0/2.5).
- Отключён частичный безубыток.
- Фильтр минимальной стены 500 USDT.
"""

import os, time, logging, ccxt
import pandas as pd, numpy as np
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from collections import Counter

load_dotenv()

# ======================= КОНФИГУРАЦИЯ =======================
SYMBOLS = [
    "1000PEPE/USDT:USDT", "WIF/USDT:USDT", "BOME/USDT:USDT",
    "RENDER/USDT:USDT", "TAO/USDT:USDT", "WLD/USDT:USDT", "ARKM/USDT:USDT",
    "IO/USDT:USDT", "ONDO/USDT:USDT", "VIRTUAL/USDT:USDT",
    "GRT/USDT:USDT", "INJ/USDT:USDT", "SUI/USDT:USDT", "APT/USDT:USDT",
    "TIA/USDT:USDT", "JTO/USDT:USDT", "EIGEN/USDT:USDT", "HBAR/USDT:USDT",
    "VET/USDT:USDT", "NOT/USDT:USDT", "CATI/USDT:USDT",
    "1000FLOKI/USDT:USDT", "1000BONK/USDT:USDT", "PEOPLE/USDT:USDT",
    "MEME/USDT:USDT", "DOGE/USDT:USDT", "SHIB/USDT:USDT",
    "1000BABYDOGE/USDT:USDT", "1000LUNC/USDT:USDT",
    "1000RATS/USDT:USDT", "1000TURBO/USDT:USDT", "MYRO/USDT:USDT",
    "PONKE/USDT:USDT", "SLERF/USDT:USDT", "SAMO/USDT:USDT",
    "WEN/USDT:USDT", "MOODENG/USDT:USDT", "GOAT/USDT:USDT",
]

TIMEFRAME_TA = "5m"
SCAN_INTERVAL = 120

FIXED_MARGIN = 3.0
LEVERAGE = 20

MIN_CONSENSUS = 2            # можно повысить до 3 для ещё более строгого отбора

MIN_WALL_USDT = 500
ORDER_BOOK_DEPTH = 20
WALL_THRESHOLD_VOL_RATIO = 3.0
MAX_WALL_DISTANCE_PCT = 2.0
IMBALANCE_RATIO_LONG = 1.2
IMBALANCE_RATIO_SHORT = 1/1.2

ATR_SL_MULT = 1.0
ATR_TP_MULT = 2.5
MIN_SL_PERCENT = 0.5
MAX_SL_PERCENT = 1.5
TP_PERCENT = 3.0

PARTIAL_BE_ENABLED = False
TRAILING_ATR_MULT = 2.0
TRAILING_OFFSET_MULT = 1.5
MIN_PROFIT_FOR_TRAIL = 0.8
RR_EXIT_TRIGGER = 0.5

SYMBOL_BLOCK_AFTER_TP = 90 * 60
SYMBOL_BLOCK_AFTER_SL = 180 * 60
SYMBOL_MAX_FAIL_ATTEMPTS = 3
SYMBOL_BLOCK_AFTER_FAIL = 120 * 60
TRADE_MAX_LIFETIME = 7200
REPORT_INTERVAL = 1800
BYBIT_FEE = 0.00055

# === НОВЫЙ ФИЛЬТР ===
VOLUME_RATIO_THRESHOLD = 1.5   # текущий объём должен быть > среднего в 1.5 раза

STATE_FILE = "state_bot_v10.7_vol.json"
TRADES_FILE = "trades_bot_v10.7_vol.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler("bot_v10.7_vol.log", encoding="utf-8")],
)
log = logging.getLogger("VolumeScalper")

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

class VolumeSpike(Strategy):
    def signal(self, df):
        avg_vol = df["volume"].rolling(20).mean().iloc[-1]
        cur_vol = df["volume"].iloc[-1]
        if cur_vol > 2 * avg_vol:
            if df["close"].iloc[-1] > df["open"].iloc[-1]: return 1
            elif df["close"].iloc[-1] < df["open"].iloc[-1]: return -1
        return 0

STRATEGIES = [KeltnerRev("Keltner Rev"), StochasticStrat("Stochastic"), CCIStrat("CCI"),
              MFIStrat("MFI"), AroonStrat("Aroon"), VWAPRev("VWAP Rev"), StochTrend("Stoch+Trend"),
              VolumeSpike("Vol Spike")]

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

def order_book_filter(symbol, side):
    ob = safe_api(exchange.fetch_order_book, symbol, ORDER_BOOK_DEPTH)
    if not ob or len(ob["bids"])<5 or len(ob["asks"])<5: return False, 0
    bids = ob["bids"]; asks = ob["asks"]
    spread = (asks[0][0]-bids[0][0])/bids[0][0]*100
    if spread > 2.0: return False, 0
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
        return (wall_vol*bids[0][0] >= MIN_WALL_USDT and imb > IMBALANCE_RATIO_LONG), wall_vol*bids[0][0]
    else:
        wall_vol = asks[0][1]
        for p,v in asks[:5]:
            if v >= WALL_THRESHOLD_VOL_RATIO*med_ask: wall_vol = v; break
        return (wall_vol*asks[0][0] >= MIN_WALL_USDT and imb < IMBALANCE_RATIO_SHORT), wall_vol*asks[0][0]

# ======================= ТОРГОВЫЕ ФУНКЦИИ =======================
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
        entry = None
        if order.get("average") is not None:
            try: entry = float(order["average"])
            except: pass
        if not entry or entry <= 0:
            time.sleep(2)
            positions = fetch_positions([symbol])
            for pos in positions:
                if float(pos.get("contracts",0) or 0) > 0 and pos.get("side") == side:
                    ep = pos.get("entryPrice") or pos.get("avgCost")
                    if ep:
                        entry = float(ep)
                        break
        if not entry or entry <= 0:
            entry = price
            log.warning(f"Не удалось получить entry, использую рыночную цену {entry:.6f}")
        log.info(f"✅ {side.upper()} {symbol} qty={qty} @ {entry:.6f} SL={sl_price:.6f} TP={tp_price:.6f} плечо={leverage}x")
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

def monitor_position(symbol, entry, qty, start_time, sl_price, tp_price, side, atr):
    deadline = start_time + TRADE_MAX_LIFETIME
    be_done = False; trailing_active = False; peak = entry; current_sl = sl_price
    trailing_offset = max(0.6/100, atr/entry * TRAILING_OFFSET_MULT)
    rr_trigger = entry + (tp_price - entry) * RR_EXIT_TRIGGER if side == "long" else entry - (entry - tp_price) * RR_EXIT_TRIGGER
    log.info(f"Мониторинг {symbol} {side} вход={entry:.6f} SL={sl_price:.6f} TP={tp_price:.6f}")
    while True:
        now = time.time()
        if now >= deadline:
            log.warning("Дедлайн — закрытие"); close_position(symbol, qty, side); return "timeout"
        time.sleep(15)
        pos_list = fetch_positions([symbol])
        active = [p for p in pos_list if float(p.get("contracts",0) or 0) > 0 and p.get("side") == side]
        if not active:
            ticker = fetch_ticker(symbol)
            cur = float(ticker["last"]) if ticker and ticker.get("last") else entry
            return "tp" if (cur >= tp_price if side=="long" else cur <= tp_price) or be_done else "sl"
        pos = active[0]; ticker = fetch_ticker(symbol)
        if not ticker or ticker.get("last") is None: continue
        cur = float(ticker["last"])
        pnl_pct = (cur/entry - 1)*100 if side=="long" else (entry/cur - 1)*100
        qty_act = abs(float(pos.get("contracts",0) or 0))

        if not be_done and pnl_pct >= 0.3:
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

def handle_existing_position(pos):
    symbol = pos["symbol"]; side = pos["side"]; entry = float(pos["entryPrice"] or pos["avgCost"] or 0)
    qty = abs(float(pos["contracts"]))
    if entry <= 0 or qty <= 0: return
    raw = fetch_ohlcv(symbol, TIMEFRAME_TA, 50)
    df_ta = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"]).set_index("timestamp")
    atr_val = atr(df_ta, 14).iloc[-1] if len(df_ta) > 14 else entry * 0.01
    price = float(fetch_ticker(symbol)["last"]) if fetch_ticker(symbol) else entry
    sl = entry*(1-MAX_SL_PERCENT/100) if side=="long" else entry*(1+MAX_SL_PERCENT/100)
    tp = entry*(1+TP_PERCENT/100) if side=="long" else entry*(1-TP_PERCENT/100)
    update_sl(symbol, sl, side); update_sl(symbol, tp, side)
    log.info(f"Подхвачена позиция {symbol} {side} entry={entry:.6f}")
    result = monitor_position(symbol, entry, qty, time.time()-60, sl, tp, side, atr_val)
    cur_price = float(fetch_ticker(symbol).get("last",entry)) if fetch_ticker(symbol) else entry
    total_pnl = (cur_price - entry) * qty if side=="long" else (entry - cur_price) * qty
    stats["прибыль_usdt" if result=="tp" else "убыток_usdt"] += max(0,total_pnl) if result=="tp" else abs(min(0,total_pnl))
    stats["тейкпрофит" if result=="tp" else "стоплосс"] += 1
    stats["сделок_всего"] += 1
    log.info(f"Завершена подхваченная позиция: {result} PnL={total_pnl:.4f} USDT")

# ======================= ГЛАВНЫЙ ЦИКЛ =======================
def main():
    global stats
    if not os.getenv("BYBIT_API_KEY"):
        log.error("Нет API ключей"); return
    stats["депозит_старт"] = get_balance(free=False)
    stats["старт_время"] = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    log.info(f"=== VOLUME-FILTERED SCALPER v10.7 ===")
    log.info(f"Депозит: {stats['депозит_старт']:.2f} USDT | Маржа: {FIXED_MARGIN} USDT | Плечо: {LEVERAGE}x")

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
                time.sleep(0.2)
                if len(raw) < 50: continue
                df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"]).set_index("timestamp")
                a = atr(df).iloc[-1] if len(df) > 14 else df["close"].iloc[-1]*0.01
                price = df["close"].iloc[-1]

                # === НОВЫЙ ФИЛЬТР ОБЪЁМА ===
                avg_vol = df["volume"].tail(20).mean()
                cur_vol = df["volume"].iloc[-1]
                volume_ratio = cur_vol / avg_vol if avg_vol > 0 else 0
                if volume_ratio < VOLUME_RATIO_THRESHOLD:
                    continue  # пропускаем символ, если объём слабый

                # Собираем голоса стратегий
                votes = Counter()
                for st in STRATEGIES:
                    sig = st.signal(df)
                    if sig != 0:
                        votes[sig] += 1
                # Сигнал только если минимум MIN_CONSENSUS стратегий согласны
                for direction, count in votes.items():
                    if count >= MIN_CONSENSUS:
                        side_str = "long" if direction == 1 else "short"
                        # Проверяем стакан
                        time.sleep(0.1)
                        ob_ok, wall_vol = order_book_filter(sym, direction)
                        if not ob_ok: continue
                        if wall_vol < MIN_WALL_USDT: continue
                        signals.append((wall_vol, sym, side_str, f"Consensus({count})", price, a))
                        log.info(f"Сигнал: Consensus({count}) {sym} {side_str.upper()} wall={wall_vol:.0f} USDT vol_ratio={volume_ratio:.1f}")

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

            # Фиксированный размер позиции
            qty = (FIXED_MARGIN * LEVERAGE) / price
            try:
                min_qty = float(exchange.market(sym)["limits"]["amount"]["min"])
                if qty < min_qty: qty = min_qty
            except: pass
            try: qty = float(exchange.amount_to_precision(sym, qty))
            except: pass
            if qty <= 0: continue

            log.info(f"🎯 Вход {side.upper()} {sym} цена={price:.6f} SL={sl:.6f} TP={tp:.6f} плечо={LEVERAGE}x")
            entry, qty_open = open_position(sym, side, qty, tp, sl, LEVERAGE)
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
            result = monitor_position(sym, entry, qty_open, start_t, sl, tp, side, a)
            cur_price = float(fetch_ticker(sym).get("last",entry)) if fetch_ticker(sym) else entry
            total_pnl = (cur_price-entry)*qty_open if side=="long" else (entry-cur_price)*qty_open
            stats["прибыль_usdt" if result=="tp" else "убыток_usdt"] += max(0,total_pnl) if result=="tp" else abs(min(0,total_pnl))
            stats["тейкпрофит" if result=="tp" else "стоплосс"] += 1
            заблокированные[sym] = time.time() + (SYMBOL_BLOCK_AFTER_TP if result=="tp" else SYMBOL_BLOCK_AFTER_SL)
            log.info(f"Сделка закрыта: {result} PnL={total_pnl:.4f} USDT")
            time.sleep(30)

        except Exception as e:
            log.error(f"Ошибка в цикле: {e}", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    main()
