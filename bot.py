#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mini Speed Trader – оптимизированная версия для быстрого теста гипотез.
Только классический теханализ, экспресс‑режим, трейлинг.
"""

import os, sys, time, json, logging, requests, pandas as pd, numpy as np, math
from dotenv import load_dotenv
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any
from pybit.unified_trading import HTTP
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================
TESTNET_MODE = True
LEVERAGE = 3
TIMEFRAME_TA = "5m"
TIMEFRAME_TREND = "1h"
TIMEFRAME_4H = "4h"
SCAN_INTERVAL = 120          # чаще сканируем для ускорения

SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT", "XRP/USDT:USDT",
    "SOL/USDT:USDT", "ADA/USDT:USDT", "DOGE/USDT:USDT", "AVAX/USDT:USDT",
    "DOT/USDT:USDT", "LINK/USDT:USDT", "MATIC/USDT:USDT", "ATOM/USDT:USDT",
    "UNI/USDT:USDT", "LTC/USDT:USDT", "OP/USDT:USDT", "ARB/USDT:USDT",
    "APT/USDT:USDT", "FIL/USDT:USDT", "NEAR/USDT:USDT", "RUNE/USDT:USDT",
]

# --- Пороги ---
MIN_SCORE = 60               # классический скор для входа
MQS_MIN_SCORE = 25           # не используется, оставлен для совместимости

# --- Риск‑менеджмент ---
BASE_RISK_PCT = 0.8
MAX_RISK_PCT = 1.2
MAX_MARGIN_PCT = 25.0
MIN_BALANCE = 5.0
MAX_DRAWDOWN_PCT = 15.0
DAILY_LOSS_LIMIT_PCT = 3.0
DAILY_LOSS_PAUSE_SEC = 10800

# --- TP / SL ---
TP_PERCENT = 3.0
SL_PERCENT = 1.0
MIN_SL_PERCENT = 0.8
MAX_SL_PERCENT = 2.0
ATR_SL_MULT = 1.5
ATR_TP_MULT = 3.0

# --- Трейлинг / Безубыток ---
PARTIAL_BE_ENABLED = True
PARTIAL_BE_PROFIT = 0.5        # процент прибыли для перевода SL в безубыток
MIN_PROFIT_FOR_TRAIL = 1.0    # процент прибыли для активации трейлинга
TRAILING_OFFSET_PCT = 0.6     # отступ трейлинга от пика (%)

# --- Фильтры ---
VOLUME_SPIKE_MULT = 3.5
VOLUME_AVG_PERIOD = 20
ENTRY_CONFIRM_BARS = 0        # отключаем подтверждение для скорости
SIGNAL_EXIT_ENABLED = True
SYMBOL_BLOCK_AFTER_TP = 90
SYMBOL_BLOCK_AFTER_SL = 180
SL_STREAK_LIMIT = 2
SL_STREAK_PAUSE = 3600
TRADE_MAX_LIFETIME = 7200

# --- S/R ---
SR_PERIOD = 100
SR_PROXIMITY_PCT = 0.5
SR_MIN_TOUCHES = 3
SR_CLUSTER_TOL = 0.005

MARK_PRICE_DIFF_THRESHOLD = 0.5 if TESTNET_MODE else 0.1
BYBIT_FEE = 0.00055

STATE_FILE = "state_mini.json"
TRADES_FILE = "trades_mini.json"

# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler("mini_speed.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)

# ============================================================
# API‑ОБЁРТКА
# ============================================================
class BybitWrapper:
    def __init__(self, testnet, key, secret):
        self.session = HTTP(testnet=testnet, api_key=key, api_secret=secret)

    def fetch_balance(self):
        r = self.session.get_wallet_balance(accountType="UNIFIED")
        acc = r["result"]["list"][0]
        return float(acc.get("totalAvailableBalance") or 0)

    def fetch_ohlcv(self, symbol, timeframe, limit=200):
        sym = symbol.replace("/", "").replace(":USDT", "")
        tf_map = {"1m":"1","3m":"3","5m":"5","15m":"15","1h":"60","4h":"240"}
        interval = tf_map.get(timeframe, "5")
        r = self.session.get_kline(category="linear", symbol=sym, interval=interval, limit=limit)
        rows = list(reversed(r["result"]["list"]))
        return [[int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])] for x in rows]

    def fetch_ticker(self, symbol: str):
        sym = symbol.replace("/", "").replace(":USDT", "")
        r = self.session.get_tickers(category="linear", symbol=sym)
        tickers = r.get("result", {}).get("list", [])
        if not tickers:
            log.warning(f"Пустой список тикеров для {symbol}")
            return {"last": 0.0, "mark_price": 0.0}
        last = float(tickers[0].get("lastPrice", 0))
        mark_price = float(tickers[0].get("markPrice", last))
        return {"last": last, "mark_price": mark_price}

    def fetch_positions(self):
        r = self.session.get_positions(category="linear", settleCoin="USDT")
        positions = []
        for p in r["result"]["list"]:
            size = float(p.get("size",0))
            if size == 0: continue
            positions.append({
                "symbol": p["symbol"],
                "side": "long" if p["side"]=="Buy" else "short",
                "contracts": size,
                "unrealizedPnl": float(p.get("unrealisedPnl",0)),
                "entryPrice": float(p.get("avgPrice",0)),
            })
        return positions

    def set_leverage(self, symbol, leverage):
        sym = symbol.replace("/", "").replace(":USDT", "")
        try:
            self.session.set_leverage(category="linear", symbol=sym,
                                       buyLeverage=str(leverage), sellLeverage=str(leverage))
            return True
        except Exception as e:
            log.warning(f"Ошибка плеча: {e}")
            return False

    def create_market_order(self, symbol, side, qty, take_profit=None, stop_loss=None, reduce_only=False):
        sym = symbol.replace("/", "").replace(":USDT", "")
        order_side = "Buy" if side=="buy" else "Sell"
        params = {
            "category": "linear", "symbol": sym,
            "side": order_side, "orderType": "Market",
            "qty": str(qty), "timeInForce": "GTC",
        }
        if reduce_only: params["reduceOnly"] = True
        if take_profit: params["takeProfit"] = str(take_profit)
        if stop_loss: params["stopLoss"] = str(stop_loss)
        r = self.session.place_order(**params)
        order_id = r["result"]["orderId"]
        time.sleep(1)
        try:
            hist = self.session.get_order_history(category="linear", symbol=sym, orderId=order_id)
            avg = float(hist["result"]["list"][0].get("avgPrice",0) or 0)
        except:
            avg = 0
        return {"average": avg, "id": order_id}

    def price_to_precision(self, symbol, price):
        return str(round(price,2))

    def amount_to_precision(self, symbol, amount):
        sym = symbol.replace("/", "").replace(":USDT", "")
        info = INSTRUMENTS.get(sym, {"minOrderQty":0.001, "qtyStep":0.001})
        step = info["qtyStep"]
        min_qty = info["minOrderQty"]
        qty = math.floor(amount / step) * step
        qty = round(qty, 10)
        if qty < min_qty:
            return min_qty
        return qty

    def update_stop_loss(self, symbol, stop_price):
        sym = symbol.replace("/", "").replace(":USDT", "")
        try:
            self.session.set_trading_stop(
                category="linear", symbol=sym,
                stopLoss=str(stop_price),
                slTriggerBy="MarkPrice", positionIdx=0
            )
            return True
        except Exception as e:
            log.warning(f"Не удалось обновить SL: {e}")
            return False

# ------------------------------------------------------------
if TESTNET_MODE:
    exchange = BybitWrapper(True, os.getenv("BYBIT_TESTNET_API_KEY"), os.getenv("BYBIT_TESTNET_API_SECRET"))
    log.info("TESTNET")
else:
    exchange = BybitWrapper(False, os.getenv("BYBIT_API_KEY"), os.getenv("BYBIT_API_SECRET"))

# Загружаем инструменты
INSTRUMENTS = {}
try:
    r = exchange.session.get_instruments_info(category="linear")
    for item in r["result"]["list"]:
        lot = item.get("lotSizeFilter", {})
        INSTRUMENTS[item["symbol"]] = {
            "minOrderQty": float(lot.get("minOrderQty", 0.001)),
            "qtyStep": float(lot.get("qtyStep", 0.001)),
        }
    log.info(f"Загружено {len(INSTRUMENTS)} инструментов")
except Exception as e:
    log.warning(f"Ошибка загрузки инструментов: {e}")

# ============================================================
# ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ
# ============================================================
def _ema(s, span): return s.ewm(span=span, adjust=False).mean()
def _rma(s, span): return s.ewm(alpha=1/span, adjust=False).mean()

def calc_rsi(close, period=14):
    d = close.diff()
    gain = d.clip(lower=0)
    loss = (-d).clip(lower=0)
    avg_g = _rma(gain, period)
    avg_l = _rma(loss, period)
    rs = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(close, fast=12, slow=26, signal=9):
    ml = _ema(close, fast) - _ema(close, slow)
    sl = _ema(ml, signal)
    return ml, sl

def calc_atr(df, period=14):
    hi, lo, pc = df["h"], df["l"], df["c"].shift(1)
    tr = pd.concat([hi-lo, (hi-pc).abs(), (lo-pc).abs()], axis=1).max(axis=1)
    return _rma(tr, period)

def calc_supertrend(df, period=10, mult=3.0):
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

def calc_hull(close, period=55):
    half = max(1, period//2)
    sqrt_p = max(1, int(np.sqrt(period)))
    hma = _ema(2 * _ema(close, half) - _ema(close, period), sqrt_p)
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

def calc_stochastic(df, k=14, d=3, smooth=3):
    lo = df["l"].rolling(k).min()
    hi = df["h"].rolling(k).max()
    ks = (100 * (df["c"] - lo) / (hi - lo + 1e-10)).rolling(smooth).mean()
    return ks, ks.rolling(d).mean()

def calc_support_resistance(df, period=SR_PERIOD):
    df_sr = df.tail(period).reset_index(drop=True)
    highs = df_sr["h"].values
    lows = df_sr["l"].values
    close = float(df["c"].iloc[-1])
    raw_res, raw_sup = [], []
    for i in range(2, len(highs)-2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            raw_res.append(highs[i])
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            raw_sup.append(lows[i])

    def _cluster(levels):
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

    res_cl = _cluster(raw_res)
    sup_cl = _cluster(raw_sup)
    res_above = sorted([(p,n) for p,n in res_cl if p > close], key=lambda x: x[0])
    sup_below = sorted([(p,n) for p,n in sup_cl if p < close], key=lambda x: x[0], reverse=True)
    nearest_res, res_n = res_above[0] if res_above else (close*1.05, 0)
    nearest_sup, sup_n = sup_below[0] if sup_below else (close*0.95, 0)
    dist_res = (nearest_res - close) / close * 100
    dist_sup = (close - nearest_sup) / close * 100
    near_sup = dist_sup < SR_PROXIMITY_PCT and sup_n >= SR_MIN_TOUCHES
    near_res = dist_res < SR_PROXIMITY_PCT and res_n >= SR_MIN_TOUCHES
    return {
        "support": nearest_sup, "resistance": nearest_res,
        "dist_to_sup_pct": round(dist_sup,2), "dist_to_res_pct": round(dist_res,2),
        "near_support": near_sup, "near_resistance": near_res,
    }

# ============================================================
# СКОРИНГ (ТОЛЬКО КЛАССИКА)
# ============================================================
def получить_скор(symbol):
    try:
        raw_ta = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA, limit=100)
        raw_1h = exchange.fetch_ohlcv(symbol, TIMEFRAME_TREND, limit=100)
        if len(raw_ta) < 60 or len(raw_1h) < 60:
            return {"score": 0, "price": 0, "sr": {}}
        cols = ["ts","o","h","l","c","v"]
        df_ta = pd.DataFrame(raw_ta, columns=cols)
        df_1h = pd.DataFrame(raw_1h, columns=cols)
        c_ta, c_1h = df_ta["c"], df_1h["c"]
        price = float(c_ta.iloc[-1])
        score = 0

        # RSI 5m
        rsi = calc_rsi(c_ta).iloc[-1]
        if 25 <= rsi <= 40: score += 20
        elif 40 < rsi <= 50: score += 12
        elif rsi < 25: score += 10

        # RSI 1h
        rsi_1h = calc_rsi(c_1h).iloc[-1]
        if rsi_1h < 50: score += 10

        # MACD
        ml, sl = calc_macd(c_ta)
        if ml.iloc[-1] > sl.iloc[-1]:
            score += 10

        # Supertrend
        st_up, _ = calc_supertrend(df_ta)
        if st_up.iloc[-1]: score += 15

        # Hull
        hu_up, _ = calc_hull(c_ta)
        if hu_up.iloc[-1]: score += 8

        # EMA Trend 1h
        ema50_1h = _ema(c_1h, 50).iloc[-1]
        ema200_1h = _ema(c_1h, 200).iloc[-1] if len(c_1h) >= 200 else ema50_1h
        if ema50_1h > ema200_1h: score += 10

        # ADX
        adx, pdi, mdi = calc_adx(df_ta)
        if adx.iloc[-1] > 25 and pdi.iloc[-1] > mdi.iloc[-1]: score += 10

        # Stochastic
        k_ser, _ = calc_stochastic(df_ta)
        if k_ser.iloc[-1] < 20: score += 10

        # Volume
        vol_avg = df_ta["v"].rolling(VOLUME_AVG_PERIOD).mean().iloc[-1]
        vol_ratio = df_ta["v"].iloc[-1] / (vol_avg + 1e-10)
        if vol_ratio > 1.5: score += 8

        # S/R
        sr = calc_support_resistance(df_ta)
        if sr["near_support"]: score += 15
        elif sr["near_resistance"]: score -= 25

        # 3 red candles penalty
        last3_bear = all(df_ta["c"].iloc[-i] < df_ta["o"].iloc[-i] for i in range(1,4))
        if last3_bear: score -= 20

        # Bybit AI ratio (простая версия)
        try:
            coin = symbol.split("/")[0]
            url = f"https://api.bybit.com/v5/market/account-ratio?category=linear&symbol={coin}USDT&period=1h&limit=1"
            resp = requests.get(url, timeout=5)
            data = resp.json()
            if data.get("retCode")==0:
                items = data["result"]["list"]
                if items:
                    buy_r = float(items[0].get("buyRatio",0.5))
                    if buy_r > 0.6: score += 5
                    elif buy_r < 0.4: score -= 10
        except: pass

        # Mark‑Price safety
        ticker = exchange.fetch_ticker(symbol)
        if ticker["last"] == 0:
            return {"score": 0, "price": 0, "sr": {}}
        diff = abs(ticker["mark_price"] - ticker["last"]) / ticker["last"] * 100
        if diff >= MARK_PRICE_DIFF_THRESHOLD: score -= 30

        return {"score": max(0, min(100, score)), "price": price, "sr": sr}
    except Exception as e:
        log.debug(f"Ошибка анализа {symbol}: {e}")
        return {"score": 0, "price": 0, "sr": {}}

# ============================================================
# ИСПОЛНЕНИЕ ОРДЕРОВ
# ============================================================
def установить_плечо(symbol, leverage):
    try:
        exchange.set_leverage(symbol, leverage)
        return True
    except Exception:
        sym = symbol.replace("/", "").replace(":USDT", "")
        try:
            exchange.session.set_leverage(category="linear", symbol=sym,
                                          buyLeverage=str(leverage), sellLeverage=str(leverage))
            return True
        except: return False

def открыть_позицию(symbol, margin_usdt, tp_price, sl_price, side="long"):
    try:
        if not установить_плечо(symbol, LEVERAGE):
            log.error(f"Плечо не установлено – отмена {symbol}")
            return None, None
        ticker = exchange.fetch_ticker(symbol)
        price = ticker["last"]
        if price == 0:
            return None, None
        pos_size_usdt = margin_usdt * LEVERAGE
        qty_raw = pos_size_usdt / price
        qty = exchange.amount_to_precision(symbol, qty_raw)
        if qty <= 0: return None, None

        if side == "long":
            sl = min(sl_price, price - max(price*MIN_SL_PERCENT/100, price*0.001))
            tp = max(tp_price, price + price*TP_PERCENT/100)
        else:
            sl = max(sl_price, price + max(price*MIN_SL_PERCENT/100, price*0.001))
            tp = min(tp_price, price - price*TP_PERCENT/100)

        tp_str = exchange.price_to_precision(symbol, tp)
        sl_str = exchange.price_to_precision(symbol, sl)
        buy_sell = "buy" if side=="long" else "sell"
        log.info(f"Открываем {side} {symbol}: qty={qty}, маржа≈{margin_usdt:.2f}U, TP={tp_str}, SL={sl_str}")
        order = exchange.create_market_order(symbol, buy_sell, qty,
                                             take_profit=tp_str, stop_loss=sl_str)
        entry_price = float(order.get("average", price))
        log.info(f"{side.upper()} открыт: {qty} @ ~{entry_price:.8f}")
        return entry_price, qty
    except Exception as e:
        log.error(f"Ошибка открытия: {e}")
        return None, None

def закрыть_позицию(symbol, qty, side):
    close_side = "sell" if side=="long" else "buy"
    for attempt in range(3):
        try:
            exchange.create_market_order(symbol, close_side, qty, reduce_only=True)
            time.sleep(3)
            positions = exchange.fetch_positions()
            active = [p for p in positions if p["symbol"]==symbol.replace("/","").replace(":USDT","")]
            if not active:
                log.info(f"Позиция {symbol} закрыта")
                return True
            log.warning(f"Позиция {symbol} не закрылась, попытка {attempt+1}")
            time.sleep(2)
        except Exception as e:
            log.warning(f"Попытка {attempt+1} закрыть {symbol} не удалась: {e}")
            time.sleep(2)
    log.error(f"Не удалось закрыть {symbol} после 3 попыток")
    return False

def проверить_signal_exit(symbol, side):
    if not SIGNAL_EXIT_ENABLED: return False
    try:
        raw = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA, limit=50)
        if len(raw) < 30: return False
        df = pd.DataFrame(raw, columns=["ts","o","h","l","c","v"])
        st_up, st_down = calc_supertrend(df)
        if side=="long" and st_down.iloc[-1]: return True
        if side=="short" and st_up.iloc[-1]: return True
    except: pass
    return False

# ============================================================
# МОНИТОРИНГ ПОЗИЦИИ
# ============================================================
def мониторить_позицию(symbol, entry_price, qty, открыта_в, sl_цена, tp_цена, side="long"):
    deadline = открыта_в + TRADE_MAX_LIFETIME
    coin = symbol.split("/")[0]
    fee_buffer = 0.001
    breakeven_price = entry_price * (1 + BYBIT_FEE*2 + fee_buffer) if side=="long" else entry_price * (1 - BYBIT_FEE*2 - fee_buffer)

    текущий_sl = sl_цена
    пиковая_цена = entry_price
    be_done = False
    trailing_активен = False
    trailing_offset_pct = TRAILING_OFFSET_PCT / 100.0

    log.info(f"Мониторинг {coin} {side} вход={entry_price:.8f} SL={sl_цена:.8f} TP={tp_цена:.8f}")

    while True:
        now = time.time()
        if now >= deadline:
            log.warning("Дедлайн – закрываем")
            закрыть_позицию(symbol, qty, side)
            return "timeout"
        time.sleep(15)

        try:
            positions = exchange.fetch_positions()
            sym_clean = symbol.replace("/","").replace(":USDT","")
            active = [p for p in positions if p["symbol"]==sym_clean and p["side"]==side]
            if not active:
                cur_price = exchange.fetch_ticker(symbol)["last"]
                hit_tp = (cur_price >= entry_price * (1 + TP_PERCENT/100*0.7)) if side=="long" else (cur_price <= entry_price * (1 - TP_PERCENT/100*0.7))
                return "tp" if (hit_tp or be_done) else "sl"

            pos = active[0]
            cur_price = exchange.fetch_ticker(symbol)["last"]
            if cur_price == 0: continue
            pnl_pct = ((cur_price - entry_price) / entry_price * 100) if side=="long" else ((entry_price - cur_price) / entry_price * 100)
            qty_actual = abs(float(pos.get("contracts",0) or 0))

            # Безубыток
            if PARTIAL_BE_ENABLED and not be_done and pnl_pct >= PARTIAL_BE_PROFIT:
                mark = exchange.fetch_ticker(symbol).get("mark_price", cur_price)
                if (side=="long" and breakeven_price < mark * 0.9995) or (side=="short" and breakeven_price > mark * 1.0005):
                    if exchange.update_stop_loss(symbol, breakeven_price):
                        текущий_sl = breakeven_price
                        be_done = True
                        trailing_активен = True
                        log.info(f"🎯 SL → БЕЗУБЫТОК: {breakeven_price:.8f}")

            # Трейлинг
            if not trailing_активен and be_done:
                if pnl_pct >= MIN_PROFIT_FOR_TRAIL:
                    trailing_активен = True
                    log.info(f"🚀 ТРЕЙЛИНГ АКТИВИРОВАН @ {cur_price:.8f}")

            if trailing_активен and be_done:
                if side=="long":
                    if cur_price > пиковая_цена: пиковая_цена = cur_price
                    new_sl = пиковая_цена * (1 - trailing_offset_pct)
                    if new_sl > текущий_sl:
                        if exchange.update_stop_loss(symbol, new_sl):
                            текущий_sl = new_sl
                            log.info(f"📈 Трейлинг SL → {new_sl:.8f}")
                else:
                    if cur_price < пиковая_цена: пиковая_цена = cur_price
                    new_sl = пиковая_цена * (1 + trailing_offset_pct)
                    if new_sl < текущий_sl:
                        if exchange.update_stop_loss(symbol, new_sl):
                            текущий_sl = new_sl
                            log.info(f"📉 Трейлинг SL → {new_sl:.8f}")

            # Signal Exit
            if SIGNAL_EXIT_ENABLED and be_done and проверить_signal_exit(symbol, side):
                log.info("Signal Exit: разворот – закрываем")
                закрыть_позицию(symbol, qty_actual, side)
                return "tp" if pnl_pct > 0 else "sl"

            log.info(f"[{coin}] {cur_price:.4f} P&L={pnl_pct:+.2f}% SL={текущий_sl:.4f} BE={be_done} Trail={trailing_активен}")

        except Exception as e:
            log.warning(f"Ошибка мониторинга: {e}")

# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================
def баланс_usdt():
    try: return exchange.fetch_balance()
    except: return 0.0

def полный_баланс_usdt():
    try:
        b = exchange.session.get_wallet_balance(accountType="UNIFIED")
        return float(b["result"]["list"][0].get("totalWalletBalance") or 0)
    except: return баланс_usdt()

def получить_позиции():
    return exchange.fetch_positions()

def загрузить_историю():
    if not os.path.exists(TRADES_FILE): return []
    try:
        with open(TRADES_FILE, "r") as f: return json.load(f)
    except: return []

def сохранить_сделку(запись):
    история = загрузить_историю()
    история.append(запись)
    with open(TRADES_FILE, "w") as f: json.dump(история, f, indent=2, default=str)

def загрузить_состояние():
    global stats
    if not os.path.exists(STATE_FILE): return
    try:
        with open(STATE_FILE, "r") as f:
            saved = json.load(f)
        for k in stats:
            if k in saved: stats[k] = saved[k]
    except: pass

def сохранить_состояние():
    with open(STATE_FILE, "w") as f: json.dump(stats, f, indent=2, default=str)

# ============================================================
# ГЛОБАЛЬНАЯ СТАТИСТИКА
# ============================================================
stats = {
    "запусков": 0, "сделок_всего": 0, "тейкпрофит": 0, "стоплосс": 0, "таймаут": 0,
    "прибыль_usdt": 0.0, "убыток_usdt": 0.0, "депозит_старт": 0.0,
    "баланс_начало_дня": 0.0, "дата_дня": "", "старт_время": "",
    "последний_отчёт": 0.0, "sl_streak": 0,
}

# ============================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================
def main():
    global stats
    загрузить_состояние()
    stats["запусков"] += 1
    баланс = полный_баланс_usdt()
    if stats["депозит_старт"] <= 0: stats["депозит_старт"] = баланс
    if not stats["старт_время"]: stats["старт_время"] = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    log.info(f"Mini Speed Trader | Баланс: {баланс:.2f} USDT | Мин. скор: {MIN_SCORE}")

    заблокированные = {}
    while True:
        try:
            свободный = баланс_usdt()
            if свободный < MIN_BALANCE:
                log.warning(f"Баланс {свободный:.2f} < {MIN_BALANCE} – пауза 10 мин")
                time.sleep(600); continue

            активные = получить_позиции()
            if активные:
                log.info(f"Открытые позиции: {[p['symbol'] for p in активные]} – ждём")
                time.sleep(60); continue

            # Сканирование
            scores = {}
            for sym in SYMBOLS:
                if sym in заблокированные and time.time() < заблокированные[sym]: continue
                if sym in заблокированные: del заблокированные[sym]
                # Проверка Mark‑Price
                ticker = exchange.fetch_ticker(sym)
                if ticker["last"] == 0:
                    continue
                diff = abs(ticker["mark_price"] - ticker["last"]) / ticker["last"] * 100
                if diff >= MARK_PRICE_DIFF_THRESHOLD: continue
                # Требуем бычий 4h тренд
                raw_4h = exchange.fetch_ohlcv(sym, TIMEFRAME_4H, limit=60)
                if len(raw_4h) >= 55:
                    df_4h = pd.DataFrame(raw_4h, columns=["ts","o","h","l","c","v"])
                    if _ema(df_4h["c"], 20).iloc[-1] <= _ema(df_4h["c"], 50).iloc[-1]: continue
                else: continue

                res = получить_скор(sym)
                scores[sym] = res
                log.debug(f"{sym}: скор={res['score']}")

            # Выбор кандидата
            кандидаты = sorted([(s,d) for s,d in scores.items() if d["score"] >= MIN_SCORE],
                              key=lambda x: x[1]["score"], reverse=True)[:3]
            if not кандидаты:
                log.info("Нет кандидатов – ждём")
                time.sleep(SCAN_INTERVAL); continue

            выбрана, скор, цена, sr_info, side = None, 0, 0.0, {}, "long"
            for лучшая, данные in кандидаты:
                sr = данные["sr"]
                if sr.get("near_resistance") and sr.get("dist_to_res_pct", 99) < 0.3: continue
                выбрана, скор, цена, sr_info = лучшая, данные["score"], данные["price"], sr
                log.info(f"► Выбрана {лучшая.split(':')[0]} (лонг) скор={скор} цена={цена:.8f}")
                break

            if выбрана is None:
                # Попробуем шорт
                for sym in SYMBOLS:
                    raw_4h = exchange.fetch_ohlcv(sym, TIMEFRAME_4H, limit=60)
                    if len(raw_4h) < 55: continue
                    df_4h = pd.DataFrame(raw_4h, columns=["ts","o","h","l","c","v"])
                    if _ema(df_4h["c"], 20).iloc[-1] >= _ema(df_4h["c"], 50).iloc[-1]: continue
                    res = получить_скор(sym)
                    if res["score"] == 0: continue
                    inv_score = 100 - res["score"]   # простой инверсный скор для шорта
                    if inv_score >= MIN_SCORE:
                        выбрана, скор, цена, sr_info, side = sym, inv_score, res["price"], res["sr"], "short"
                        log.info(f"🐻 Шорт‑кандидат: {sym.split(':')[0]} скор={inv_score}")
                        break
                if выбрана is None:
                    log.info("Нет кандидатов – ждём")
                    time.sleep(SCAN_INTERVAL); continue

            # Расчёт TP/SL
            atr_pt = 0.0
            raw_atr = exchange.fetch_ohlcv(выбрана, TIMEFRAME_TA, limit=50)
            if len(raw_atr) >= 20:
                df_atr = pd.DataFrame(raw_atr, columns=["ts","o","h","l","c","v"])
                atr_pt = float(calc_atr(df_atr, 14).iloc[-1])

            sl_dist = max(MIN_SL_PERCENT, min(MAX_SL_PERCENT, (atr_pt*ATR_SL_MULT/цена)*100)) if atr_pt>0 else SL_PERCENT
            tp_dist = max(TP_PERCENT, sl_dist*2)

            if side == "long":
                sl_цена = цена * (1 - sl_dist/100)
                tp_цена = цена * (1 + tp_dist/100)
                support = sr_info.get("support", sl_цена)
                if support < sl_цена and support > цена*0.97: sl_цена = support*0.998
            else:
                sl_цена = цена * (1 + sl_dist/100)
                tp_цена = цена * (1 - tp_dist/100)
                resistance = sr_info.get("resistance", sl_цена)
                if resistance > sl_цена and resistance < цена*1.03: sl_цена = resistance*1.002

            sl_dist_pct = abs(цена - sl_цена) / цена * 100
            real_rr = abs(tp_цена - цена) / abs(цена - sl_цена)
            if real_rr < 1.999:
                log.warning(f"⛔ RR={real_rr:.1f}:1 < 2:1 – пропуск")
                time.sleep(SCAN_INTERVAL); continue

            margin = баланс_usdt() * BASE_RISK_PCT / 100
            margin = min(margin, свободный * 0.9)
            log.info(f"✅ ВХОД {side.upper()}: скор={скор} SL={sl_цена:.8f} TP={tp_цена:.8f} маржа={margin:.2f}U")

            время_входа = time.time()
            entry_price, qty = открыть_позицию(выбрана, margin, tp_цена, sl_цена, side)
            if entry_price is None:
                time.sleep(30); continue

            stats["сделок_всего"] += 1
            результат = мониторить_позицию(выбрана, entry_price, qty, время_входа, sl_цена, tp_цена, side)

            баланс_после = полный_баланс_usdt()
            pnl = баланс_после - баланс
            duration = (time.time() - время_входа) / 60

            if результат == "tp":
                stats["тейкпрофит"] += 1; stats["прибыль_usdt"] += max(0, pnl); stats["sl_streak"] = 0
                заблокированные[выбрана] = time.time() + SYMBOL_BLOCK_AFTER_TP*60
                log.info(f"✅ TP: прибыль ≈{pnl:+.4f} USDT")
            elif результат == "sl":
                stats["стоплосс"] += 1; stats["убыток_usdt"] += abs(min(0,pnl)); stats["sl_streak"] += 1
                заблокированные[выбрана] = time.time() + SYMBOL_BLOCK_AFTER_SL*60
                log.warning(f"❌ SL: убыток ≈{pnl:+.4f} USDT streak={stats['sl_streak']}")
            else:
                stats["таймаут"] += 1; stats["убыток_usdt"] += abs(min(0,pnl)); stats["sl_streak"] = 0
                заблокированные[выбрана] = time.time() + SYMBOL_BLOCK_AFTER_TP*60
                log.warning(f"⏰ Таймаут: P&L ≈{pnl:+.4f} USDT")

            запись = {
                "время": datetime.now().strftime("%d.%m.%Y %H:%M:%S"), "symbol": выбрана,
                "side": side, "score": скор, "entry": entry_price, "sl": sl_цена, "tp": tp_цена,
                "pnl": round(pnl,4), "duration_min": round(duration,1), "результат": результат
            }
            сохранить_сделку(запись)
            сохранить_состояние()
            log.info("Сделка завершена – пауза 60 сек")
            time.sleep(60)

        except Exception as e:
            log.error(f"Глобальная ошибка: {e}", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    main()
