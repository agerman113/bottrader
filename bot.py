#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ГИБРИДНЫЙ СКОРИНГ-БОТ v10.9_fixed (rate-limit safe)
===================================
- 42 монеты, 7 стратегий-лидеров, динамические веса на основе винрейта.
- Многослойный фильтр: стакан (order book wall) + тренд 4h.
- Риск-менеджмент из v10.4 (ATR SL/TP, частичный безубыток, трейлинг).
- Параллельный бумажный трейдер для сбора статистики.
- Безопасные API-вызовы, логирование, подхват позиций.
- ДОБАВЛЕНЫ ЗАДЕРЖКИ для защиты от rate limit.
"""

import os, sys, time, logging, threading
import numpy as np, pandas as pd, ccxt
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from dotenv import load_dotenv

load_dotenv()

# ======================= КОНФИГУРАЦИЯ =======================
SYMBOLS = [
    "BTC/USDT:USDT","ETH/USDT:USDT","BNB/USDT:USDT","XRP/USDT:USDT",
    "SOL/USDT:USDT","ADA/USDT:USDT","TRX/USDT:USDT","AVAX/USDT:USDT",
    "DOT/USDT:USDT","LTC/USDT:USDT","BCH/USDT:USDT","ATOM/USDT:USDT",
    "XLM/USDT:USDT","NEAR/USDT:USDT","DOGE/USDT:USDT",
    "1000PEPE/USDT:USDT","WIF/USDT:USDT","BOME/USDT:USDT",
    "RENDER/USDT:USDT","TAO/USDT:USDT","WLD/USDT:USDT","ARKM/USDT:USDT",
    "IO/USDT:USDT","ONDO/USDT:USDT","VIRTUAL/USDT:USDT","UNI/USDT:USDT",
    "AAVE/USDT:USDT","ARB/USDT:USDT","OP/USDT:USDT","LINK/USDT:USDT",
    "GRT/USDT:USDT","INJ/USDT:USDT","SUI/USDT:USDT","APT/USDT:USDT",
    "TIA/USDT:USDT","JTO/USDT:USDT","EIGEN/USDT:USDT","HBAR/USDT:USDT",
    "VET/USDT:USDT","NOT/USDT:USDT","CATI/USDT:USDT",
]

REAL_TRADING = False
TIMEFRAME = "5m"
TREND_TF = "4h"
CANDLE_LIMIT = 100
SCAN_INTERVAL = 180

ATR_PERIOD = 14
ATR_SL_MULT = 1.5
ATR_TP_MULT = 3.0
MIN_SL_PCT = 0.8
MAX_SL_PCT = 2.0
MIN_TP_PCT = 3.0

RISK_PCT = 0.8
LEV_MIN, LEV_MAX = 3, 5
PARTIAL_BE_PROFIT = 0.2
PARTIAL_BE_PCT = 50.0
TRAIL_TRIGGER = 0.6
TRAIL_ATR_MULT = 1.5
MAX_TRADE_LIFE = 7200

DAILY_LOSS_LIMIT = 3.0
DAILY_LOSS_PAUSE = 10800
SL_STREAK_LIMIT = 3
SL_STREAK_PAUSE = 1800
COOLDOWN_TP = 5400
COOLDOWN_SL = 10800
COOLDOWN_FAIL = 7200

MIN_WALL_USDT = 5000
OB_LEVELS = 20
WALL_RATIO = 3.0
IMBALANCE_RATIO = 1.5
MAX_SPREAD_PCT = 1.0

WEIGHT_UPDATE_INTERVAL = 1800
MIN_TRADES_WEIGHT = 10
WEIGHT_LOOKBACK = 20

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler("bot_v10.9_fixed.log", encoding="utf-8")],
)
log = logging.getLogger("ScoringBot")

# ======================= БИРЖА =======================
exchange = ccxt.bybit({
    "apiKey": os.getenv("BYBIT_API_KEY"),
    "secret": os.getenv("BYBIT_API_SECRET"),
    "enableRateLimit": True,
    "options": {"defaultType": "linear"},
})

def safe_api(func, *a, **kw):
    for attempt in range(5):
        try:
            return func(*a, **kw)
        except (ccxt.RateLimitExceeded, ccxt.NetworkError) as e:
            log.warning(f"API retry {attempt+1}: {e}")
            time.sleep((attempt+1)*2)
        except Exception as e:
            log.error(f"API error: {e}")
            if attempt == 4: raise
            time.sleep(2)
    return None

# ======================= ИНДИКАТОРЫ =======================
def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def sma(s, p): return s.rolling(p).mean()
def rsi(s, p=14):
    d = s.diff(); g = d.clip(lower=0); l = (-d).clip(lower=0)
    ag = g.ewm(alpha=1/p, adjust=False).mean()
    al = l.ewm(alpha=1/p, adjust=False).mean()
    return 100 - 100/(1+ag/al.replace(0, np.nan))
def atr(df, p=14):
    h, l, c = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h-l, (h-c).abs(), (l-c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, adjust=False).mean()
def bollinger(s, p=20, std=2):
    mb = sma(s, p); sd = s.rolling(p).std()
    return mb+std*sd, mb, mb-std*sd
def macd(s, f=12, sl=26, sig=9):
    ml = ema(s, f)-ema(s, sl); sl_ = ema(ml, sig); return ml, sl_, ml-sl_
def stoch(df, kp=14, dp=3):
    lo = df["low"].rolling(kp).min(); hi = df["high"].rolling(kp).max()
    k = 100*(df["close"]-lo)/(hi-lo+1e-10); d = k.rolling(dp).mean()
    return k, d
def adx(df, p=14):
    a = atr(df, p); hd = df["high"].diff(); ld = -df["low"].diff()
    pdm = hd.where((hd>ld)&(hd>0),0); mdm = ld.where((ld>hd)&(ld>0),0)
    pdi = 100*pdm.ewm(alpha=1/p, adjust=False).mean()/a.replace(0,np.nan)
    mdi = 100*mdm.ewm(alpha=1/p, adjust=False).mean()/a.replace(0,np.nan)
    dx = 100*abs(pdi-mdi)/(pdi+mdi+1e-10); adx_ = dx.ewm(alpha=1/p, adjust=False).mean()
    return adx_, pdi, mdi
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
def donchian(df, p=20):
    return df["high"].rolling(p).max(), df["low"].rolling(p).min()
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
def trend_ok(symbol, side):
    raw = safe_api(exchange.fetch_ohlcv, symbol, TREND_TF, limit=60) or []
    if len(raw)<55: return False
    df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"]).set_index("timestamp")
    e20 = ema(df["close"],20).iloc[-1]; e50 = ema(df["close"],50).iloc[-1]
    return e20>e50 if side==1 else e20<e50

def order_book_filter(symbol, side):
    ob = safe_api(exchange.fetch_order_book, symbol, OB_LEVELS)
    if not ob or len(ob["bids"])<5 or len(ob["asks"])<5: return False, 0
    bids = ob["bids"]; asks = ob["asks"]
    spread = (asks[0][0]-bids[0][0])/bids[0][0]*100
    if spread > MAX_SPREAD_PCT: return False, 0
    bid_vols = [v for _,v in bids[1:]]; ask_vols = [v for _,v in asks[1:]]
    med_bid = np.median(bid_vols) if bid_vols else 0
    med_ask = np.median(ask_vols) if ask_vols else 0
    total_bid = sum(v for _,v in bids[:OB_LEVELS])
    total_ask = sum(v for _,v in asks[:OB_LEVELS])
    imb = total_bid/(total_ask+1e-10)
    if side == 1:
        wall_vol = bids[0][1]
        for p,v in bids[:5]:
            if v >= WALL_RATIO*med_bid: wall_vol = v; break
        return (wall_vol*bids[0][0] >= MIN_WALL_USDT and imb > IMBALANCE_RATIO), wall_vol*bids[0][0]
    else:
        wall_vol = asks[0][1]
        for p,v in asks[:5]:
            if v >= WALL_RATIO*med_ask: wall_vol = v; break
        return (wall_vol*asks[0][0] >= MIN_WALL_USDT and imb < 1/IMBALANCE_RATIO), wall_vol*asks[0][0]

# ======================= РИСК-МЕНЕДЖЕР =======================
class RiskManager:
    def __init__(self):
        self.balance = self.free = 0
        self.daily_loss = 0; self.start_balance = 0
        self.sl_streak = 0; self.cooldowns = {}
        self.daily_pause_until = 0; self.streak_pause_until = 0
    def update_balance(self):
        try:
            b = exchange.fetch_balance({"type":"linear"})
            self.balance = float(b["USDT"]["total"])
            self.free = float(b["USDT"]["free"])
        except: pass
    def can_trade(self, symbol):
        now = time.time()
        if now < self.daily_pause_until or now < self.streak_pause_until: return False
        if symbol in self.cooldowns and now < self.cooldowns[symbol]: return False
        if self.start_balance>0 and abs(self.daily_loss)/self.start_balance*100 >= DAILY_LOSS_LIMIT:
            self.daily_pause_until = now + DAILY_LOSS_PAUSE; return False
        if self.sl_streak >= SL_STREAK_LIMIT:
            self.streak_pause_until = now + SL_STREAK_PAUSE; self.sl_streak=0; return False
        return True
    def record(self, pnl):
        if pnl<0: self.daily_loss += abs(pnl); self.sl_streak+=1
        else: self.sl_streak=0
    def set_cooldown(self, sym, tp): self.cooldowns[sym]=time.time()+(COOLDOWN_TP if tp else COOLDOWN_SL)

# ======================= БУМАЖНЫЙ ТРЕЙДЕР =======================
class PaperTrader:
    def __init__(self):
        self.history = {s.name: [] for s in STRATEGIES}
        self.open = []
    def add(self, strat_name, sym, side, entry, sl, tp):
        self.open.append({"strat":strat_name,"sym":sym,"side":side,"entry":entry,"sl":sl,"tp":tp,"time":time.time()})
    def update(self):
        closed = []
        for p in self.open:
            ticker = safe_api(exchange.fetch_ticker, p["sym"])
            if not ticker: continue
            cur = ticker["last"]
            dead = p["time"]+MAX_TRADE_LIFE
            res = None
            if time.time()>=dead: res = "timeout"
            elif p["side"]==1:
                if cur>=p["tp"]: res="tp"
                elif cur<=p["sl"]: res="sl"
            else:
                if cur<=p["tp"]: res="tp"
                elif cur>=p["sl"]: res="sl"
            if res:
                pnl = (cur-p["entry"])/p["entry"]*100 if p["side"]==1 else (p["entry"]-cur)/p["entry"]*100
                self.history[p["strat"]].append(pnl>0)
                closed.append(p)
        for p in closed: self.open.remove(p)
    def weights(self):
        w = {}
        for s in STRATEGIES:
            hist = self.history[s.name][-WEIGHT_LOOKBACK:]
            if len(hist)>=MIN_TRADES_WEIGHT: w[s.name] = sum(hist)/len(hist)
            else: w[s.name] = 0
        return w
    def report(self):
        lines = []
        for s in STRATEGIES:
            hist = self.history[s.name]
            wr = sum(hist)/len(hist)*100 if hist else 0
            lines.append(f"{s.name}: trades={len(hist)} WR={wr:.1f}% weight={self.weights()[s.name]:.2f}")
        return "\n".join(lines)

# ======================= ИСПОЛНИТЕЛЬ =======================
class Executor:
    def open(self, sym, side, qty, tp, sl):
        set_lev(sym)
        s = "buy" if side==1 else "sell"
        tp_s = exchange.price_to_precision(sym, tp)
        sl_s = exchange.price_to_precision(sym, sl)
        try:
            o = exchange.create_order(sym, "market", s, qty, params={"takeProfit":tp_s,"stopLoss":sl_s})
            entry = float(o.get("average",0)) or float(exchange.fetch_ticker(sym)["last"])
            log.info(f"OPEN {s.upper()} {sym} qty={qty} @ {entry:.6f} SL={sl_s} TP={tp_s}")
            return entry
        except Exception as e:
            log.error(f"Open error {sym}: {e}")
            return None
    def close(self, sym, qty, side):
        s = "sell" if side==1 else "buy"
        try:
            exchange.create_order(sym, "market", s, qty, params={"reduceOnly":True})
            return True
        except: return False
    def move_sl(self, sym, new_sl, side):
        try:
            coin = sym.replace("/","").replace(":USDT","")
            exchange.private_post_v5_position_trading_stop({
                "category":"linear","symbol":coin,"stopLoss":str(exchange.price_to_precision(sym,new_sl)),
                "slTriggerBy":"MarkPrice","positionIdx":"0"
            })
            return True
        except: return False

def set_lev(sym):
    for lev in [LEV_MAX, LEV_MIN]:
        try:
            exchange.set_leverage(lev, sym)
            return lev
        except: pass
    return LEV_MIN

# ======================= МОНИТОР =======================
def monitor(sym, entry, qty, side, sl, tp, atr_val):
    start = time.time()
    be_done = False; trail_active = False; peak = entry; cur_sl = sl
    trail_off = max(0.6/100, atr_val/entry*TRAIL_ATR_MULT)
    rr_trig = entry+(tp-entry)*TRAIL_TRIGGER if side==1 else entry-(entry-tp)*TRAIL_TRIGGER
    partial_done = False
    while True:
        if time.time()-start > MAX_TRADE_LIFE:
            log.warning("Timeout close"); Executor().close(sym, qty, side); return "timeout"
        time.sleep(15)
        pos_list = safe_api(exchange.fetch_positions, [sym]) or []
        act = [p for p in pos_list if float(p.get("contracts",0) or 0)>0 and p.get("side")==("long" if side==1 else "short")]
        ticker = safe_api(exchange.fetch_ticker, sym)
        cur = float(ticker["last"]) if ticker else entry
        if not act: return "tp" if be_done or (cur>=tp if side==1 else cur<=tp) else "sl"
        pnl_pct = (cur/entry-1)*100 if side==1 else (entry/cur-1)*100
        qty_act = abs(float(act[0].get("contracts",0) or 0))
        if not partial_done and pnl_pct >= PARTIAL_BE_PROFIT:
            close_q = qty_act*(PARTIAL_BE_PCT/100)
            if close_q>0:
                Executor().close(sym, close_q, side)
                partial_done = True
                new_sl = entry*1.001 if side==1 else entry*0.999
                if Executor().move_sl(sym, new_sl, side): cur_sl = new_sl
        if not be_done and pnl_pct >= 0.3:
            new_sl = entry*1.001 if side==1 else entry*0.999
            if Executor().move_sl(sym, new_sl, side): cur_sl = new_sl; be_done = True
        if be_done:
            if not trail_active and ((side==1 and cur>=rr_trig) or (side==-1 and cur<=rr_trig)):
                trail_active = True; peak = cur; log.info("Trail activated")
            if trail_active and pnl_pct>=1.0:
                if side==1:
                    if cur>peak: peak=cur
                    new_sl = peak*(1-trail_off)
                    if new_sl>cur_sl and Executor().move_sl(sym, new_sl, side): cur_sl=new_sl
                else:
                    if cur<peak: peak=cur
                    new_sl = peak*(1+trail_off)
                    if new_sl<cur_sl and Executor().move_sl(sym, new_sl, side): cur_sl=new_sl
        log.info(f"[{sym}] {cur:.6f} P&L={pnl_pct:+.2f}% SL={cur_sl:.6f}")

# ======================= ГЛАВНЫЙ ЦИКЛ =======================
def main():
    risk = RiskManager(); risk.update_balance()
    risk.start_balance = risk.balance
    paper = PaperTrader()
    log.info("=== SCORING BOT v10.9_fixed ===")
    # Подхват позиций
    for pos in safe_api(exchange.fetch_positions) or []:
        if float(pos.get("contracts",0) or 0)>0:
            sym = pos["symbol"]; side = 1 if pos["side"]=="long" else -1
            entry = float(pos["entryPrice"]); qty = abs(float(pos["contracts"]))
            raw = safe_api(exchange.fetch_ohlcv, sym, TIMEFRAME, 50) or []
            time.sleep(0.2)
            if len(raw)<14: continue
            df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"]).set_index("timestamp")
            a = atr(df).iloc[-1]
            sl = entry*(1-MAX_SL_PCT/100) if side==1 else entry*(1+MAX_SL_PCT/100)
            tp = entry*(1+MIN_TP_PCT/100) if side==1 else entry*(1-MIN_TP_PCT/100)
            log.info(f"Подхват {sym} {side} entry={entry:.6f}")
            monitor(sym, entry, qty, side, sl, tp, a)

    while True:
        risk.update_balance()
        paper.update()
        if time.time() % WEIGHT_UPDATE_INTERVAL < SCAN_INTERVAL:
            weights = paper.weights()
            log.info(f"Weights: { {k:round(v,2) for k,v in weights.items()} }")
        if not risk.can_trade(""): time.sleep(60); continue
        signals = []
        for sym in SYMBOLS:
            raw = safe_api(exchange.fetch_ohlcv, sym, TIMEFRAME, CANDLE_LIMIT) or []
            time.sleep(0.3)  # ЗАДЕРЖКА МЕЖДУ ЗАПРОСАМИ OHLCV
            if len(raw)<50: continue
            df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"]).set_index("timestamp")
            a = atr(df).iloc[-1] if len(df)>14 else df["close"].iloc[-1]*0.01
            price = df["close"].iloc[-1]
            for st in STRATEGIES:
                sig = st.signal(df)
                if sig==0: continue
                if not trend_ok(sym, sig): continue
                # задержка перед запросом стакана
                time.sleep(0.2)
                ob_ok, wall_vol = order_book_filter(sym, sig)
                if not ob_ok: continue
                w = weights.get(st.name, 0)
                if w <= 0: continue
                signals.append((w*wall_vol, sym, sig, st.name, price, a, wall_vol))
                sl_dist = a*ATR_SL_MULT; tp_dist = a*ATR_TP_MULT
                sl = price-sl_dist if sig==1 else price+sl_dist
                tp = price+tp_dist if sig==1 else price-tp_dist
                sl = max(price*(1-MAX_SL_PCT/100), min(price*(1-MIN_SL_PCT/100), sl)) if sig==1 else min(price*(1+MAX_SL_PCT/100), max(price*(1+MIN_SL_PCT/100), sl))
                tp = max(price*(1+MIN_TP_PCT/100), tp) if sig==1 else min(price*(1-MIN_TP_PCT/100), tp)
                paper.add(st.name, sym, sig, price, sl, tp)
        if signals and risk.free >= 5 and REAL_TRADING:
            signals.sort(reverse=True)
            _, sym, sig, sname, price, a, wv = signals[0]
            qty = (risk.free*RISK_PCT/100)/abs(a*ATR_SL_MULT)
            entry = Executor().open(sym, sig, qty, price+a*ATR_TP_MULT if sig==1 else price-a*ATR_TP_MULT, price-a*ATR_SL_MULT if sig==1 else price+a*ATR_SL_MULT)
            if entry:
                res = monitor(sym, entry, qty, sig, entry-(a*ATR_SL_MULT if sig==1 else a*ATR_SL_MULT), entry+(a*ATR_TP_MULT if sig==1 else a*ATR_TP_MULT), a)
                pnl = (risk.balance - risk.start_balance)
                risk.record(pnl); risk.set_cooldown(sym, res=="tp")
                log.info(f"Trade closed: {res} PnL={pnl:.4f}")
        time.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    main()
