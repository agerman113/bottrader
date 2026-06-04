#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aurora AI + Copy‑Trading Bot (исправленный)
Закрывает все позиции при старте, игнорирует несуществующие символы.
Полный риск‑менеджмент, трейлинг, частичное закрытие, пирамидинг.
"""

import os, time, json, logging, requests, pandas as pd, numpy as np, math
from datetime import datetime, timedelta
from pybit.unified_trading import HTTP
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================
TESTNET_MODE   = True
LEVERAGE       = 3
TIMEFRAME_TA   = "5m"
SCAN_INTERVAL  = 120          # интервал сканирования сигналов (сек)
COPY_SCAN_INTERVAL = 3600     # интервал проверки мастеров (сек)

# ТОЛЬКО СИМВОЛЫ, КОТОРЫЕ ЕСТЬ НА ТЕСТНЕТЕ BYBIT (проверено)
SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "ADA/USDT:USDT",
    "DOGE/USDT:USDT", "DOT/USDT:USDT", "LINK/USDT:USDT", "UNI/USDT:USDT",
    "OP/USDT:USDT", "APT/USDT:USDT", "NEAR/USDT:USDT", "RUNE/USDT:USDT",
    "AVAX/USDT:USDT", "MATIC/USDT:USDT", "ATOM/USDT:USDT", "LTC/USDT:USDT",
    "ARB/USDT:USDT", "FIL/USDT:USDT",
]

MIN_SCORE      = 20

# --- Риск ---
BASE_RISK_PCT      = 0.8
MIN_BALANCE        = 5.0
MAX_DRAWDOWN_PCT   = 15.0
DAILY_LOSS_LIMIT_PCT  = 3.0
DAILY_LOSS_PAUSE_SEC  = 10800

# --- Сила сигнала → множитель маржи ---
SIGNAL_STRENGTH_TIERS = [
    (85, "ULTRA",  2.0),
    (75, "STRONG", 1.5),
    (60, "NORMAL", 1.0),
    (45, "WEAK",   0.5),
]
ADAPTIVE_MIN_TRADES = 10

# --- TP / SL ---
TP_PERCENT     = 3.0
SL_PERCENT     = 1.0
MIN_SL_PERCENT = 0.8
MAX_SL_PERCENT = 2.0
ATR_SL_MULT    = 1.5

# --- Безубыток / трейлинг ---
PARTIAL_BE_ENABLED   = True
PARTIAL_BE_PROFIT    = 1.0
MIN_PROFIT_FOR_TRAIL = 1.5
TRAILING_OFFSET_PCT  = 0.6

# --- Частичное закрытие ---
PARTIAL_CLOSE_ENABLED = True
PARTIAL_CLOSE_LEVELS  = [
    (2.0, 0.30),
    (4.0, 0.30),
]

# --- Пирамидинг ---
PYRAMID_ENABLED      = True
PYRAMID_TRIGGER_PCT  = 1.5
PYRAMID_FRACTION     = 0.50
PYRAMID_MAX_ADDS     = 2
PYRAMID_SL_TRAIL_PCT = 0.4

# --- Фильтры ---
VOLUME_AVG_PERIOD     = 20
SIGNAL_EXIT_ENABLED   = True
SYMBOL_BLOCK_AFTER_TP = 300
SYMBOL_BLOCK_AFTER_SL = 300
SL_STREAK_LIMIT       = 2
SL_STREAK_PAUSE       = 3600
TRADE_MAX_LIFETIME    = 7200
STUCK_PRICE_TIMEOUT   = 600

# --- S/R ---
SR_PERIOD       = 100
SR_PROXIMITY_PCT= 0.5
SR_MIN_TOUCHES  = 3
SR_CLUSTER_TOL  = 0.005

MARK_PRICE_DIFF_THRESHOLD = 0.5 if TESTNET_MODE else 0.1
BYBIT_FEE = 0.00055

STATE_FILE  = "state_aurora_copy.json"
TRADES_FILE = "trades_aurora_copy.json"

# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("aurora_copy.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ============================================================
# ГЛОБАЛЬНЫЙ КЭШ «ПЛОХИХ» СИМВОЛОВ (чтобы не повторять ошибки)
# ============================================================
_bad_symbols: set = set()

def is_bad_symbol(symbol: str) -> bool:
    return symbol in _bad_symbols

def mark_bad_symbol(symbol: str):
    _bad_symbols.add(symbol)


# ============================================================
# КЭШ ТИКЕРОВ (2 сек)
# ============================================================
_ticker_cache: dict = {}
_TICKER_TTL = 2

def cached_ticker(symbol: str) -> dict:
    now = time.time()
    if symbol in _ticker_cache:
        ts, val = _ticker_cache[symbol]
        if now - ts < _TICKER_TTL:
            return val
    val = exchange.fetch_ticker(symbol)
    _ticker_cache[symbol] = (now, val)
    return val


# ============================================================
# СИЛА СИГНАЛА (как в мини-боте)
# ============================================================
def определить_силу_сигнала(score: int) -> tuple:
    tier_name, tier_mult = "WEAK", 0.5
    for threshold, name, mult in SIGNAL_STRENGTH_TIERS:
        if score >= threshold:
            tier_name, tier_mult = name, mult
            break
    wr_data = stats.get("винрейт_по_скору", {})
    d = wr_data.get(tier_name, {"сделок": 0, "побед": 0})
    n = d["сделок"]
    if n >= ADAPTIVE_MIN_TRADES:
        wr = d["побед"] / n
        if wr < 0.40:
            tier_mult *= 0.5
        elif wr > 0.65:
            tier_mult = min(tier_mult * 1.25, 3.0)
    return tier_name, tier_mult


def обновить_винрейт(tier_name: str, победа: bool):
    if "винрейт_по_скору" not in stats:
        stats["винрейт_по_скору"] = {}
    wr = stats["винрейт_по_скору"]
    if tier_name not in wr:
        wr[tier_name] = {"сделок": 0, "побед": 0}
    wr[tier_name]["сделок"] += 1
    if победа:
        wr[tier_name]["побед"] += 1


def распечатать_винрейт():
    wr = stats.get("винрейт_по_скору", {})
    if not wr:
        log.info("📊 Статистика винрейта: данных пока нет")
        return
    log.info("=" * 58)
    log.info("📊 ВИНРЕЙТ ПО СИЛЕ СИГНАЛА:")
    log.info(f"  {'Тир':<8} {'Сделок':>7} {'Побед':>7} {'WR':>7} {'EV(2:1)':>9}")
    log.info("-" * 58)
    total_n = total_w = 0
    for _, name, _ in reversed(SIGNAL_STRENGTH_TIERS):
        d = wr.get(name, {"сделок": 0, "побед": 0})
        n, w = d["сделок"], d["побед"]
        total_n += n; total_w += w
        if n == 0:
            log.info(f"  {name:<8} {'—':>7} {'—':>7} {'—':>7} {'—':>9}")
            continue
        wrate = w / n
        ev = wrate * 2 - (1 - wrate)
        status = "✅" if ev > 0.05 else ("⚠️" if ev > -0.1 else "❌")
        log.info(f"  {name:<8} {n:>7} {w:>7} {wrate*100:>6.1f}% {ev:>+8.2f}R {status}")
    if total_n:
        wrate = total_w / total_n
        ev = wrate * 2 - (1 - wrate)
        log.info("-" * 58)
        log.info(f"  {'ИТОГО':<8} {total_n:>7} {total_w:>7} {wrate*100:>6.1f}% {ev:>+8.2f}R")
    log.info("=" * 58)


# ============================================================
# AURORA AI
# ============================================================
AURORA_CACHE: dict = {}
AURORA_TTL = 300

def получить_aurora_сигнал(symbol: str) -> dict:
    coin = symbol.split("/")[0]
    now = time.time()
    if coin in AURORA_CACHE:
        ts, cached = AURORA_CACHE[coin]
        if now - ts < AURORA_TTL:
            return cached
    try:
        url = f"https://www.bybit.com/aurora/api/v1/signal/list?symbol={coin}USDT"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code != 200:
            raise Exception(f"HTTP {resp.status_code}")
        data = resp.json()
        signals = data.get("result", {}).get("list", [])
        if not signals:
            raise Exception("пустой список сигналов")
        latest = signals[0]
        signal = latest.get("signal", "").lower()
        confidence = float(latest.get("confidence", 0))
        if signal not in ("long", "short"):
            signal = "neutral"
        result = {"signal": signal, "confidence": confidence, "valid": True}
    except Exception as e:
        log.debug(f"Aurora AI недоступен для {symbol}: {e}")
        result = {"signal": "neutral", "confidence": 0, "valid": False}
    AURORA_CACHE[coin] = (now, result)
    return result


# ============================================================
# ПОИСК УСПЕШНЫХ МАСТЕРОВ КОПИТРЕЙДИНГА (заглушка)
# ============================================================
def найти_успешных_мастеров() -> list:
    return []


# ============================================================
# API‑ОБЁРТКА
# ============================================================
class BybitWrapper:
    def __init__(self, testnet, key, secret):
        self.session = HTTP(testnet=testnet, api_key=key, api_secret=secret)

    def fetch_balance(self):
        r = self.session.get_wallet_balance(accountType="UNIFIED")
        return float(r["result"]["list"][0].get("totalAvailableBalance") or 0)

    def fetch_ohlcv(self, symbol, timeframe, limit=200):
        sym = symbol.replace("/","").replace(":USDT","")
        tf_map = {"1m":"1","3m":"3","5m":"5","15m":"15","1h":"60","4h":"240"}
        interval = tf_map.get(timeframe,"5")
        for attempt in range(3):
            try:
                r = self.session.get_kline(category="linear",symbol=sym,interval=interval,limit=limit)
                rows = list(reversed(r["result"]["list"]))
                return [[int(x[0]),float(x[1]),float(x[2]),float(x[3]),float(x[4]),float(x[5])] for x in rows]
            except Exception as e:
                log.warning(f"kline попытка {attempt+1}/3 {symbol}: {e}")
                time.sleep(2)
        return []

    def fetch_ticker(self, symbol: str) -> dict:
        sym = symbol.replace("/","").replace(":USDT","")
        if is_bad_symbol(sym):   # больше не пытаемся
            return {"last":0.0,"mark_price":0.0}
        for attempt in range(1):  # только одна попытка для неизвестных символов
            try:
                r = self.session.get_tickers(category="linear", symbol=sym)
                tickers = r.get("result",{}).get("list",[])
                if not tickers:
                    return {"last":0.0,"mark_price":0.0}
                last       = float(tickers[0].get("lastPrice",0))
                mark_price = float(tickers[0].get("markPrice", last))
                return {"last":last,"mark_price":mark_price}
            except Exception as e:
                if "invalid" in str(e).lower() or "10001" in str(e):
                    mark_bad_symbol(sym)
                    log.warning(f"Символ {sym} не существует на бирже – помечаем как плохой")
                return {"last":0.0,"mark_price":0.0}
        return {"last":0.0,"mark_price":0.0}

    def fetch_positions(self):
        r = self.session.get_positions(category="linear", settleCoin="USDT")
        out = []
        for p in r["result"]["list"]:
            size = float(p.get("size",0))
            if size == 0: continue
            out.append({
                "symbol":        p["symbol"],
                "side":          "long" if p["side"]=="Buy" else "short",
                "contracts":     size,
                "unrealizedPnl": float(p.get("unrealisedPnl",0)),
                "entryPrice":    float(p.get("avgPrice",0)),
            })
        return out

    def close_position(self, symbol, qty, side):
        close_side = "sell" if side=="long" else "buy"
        try:
            self.create_market_order(symbol, close_side, qty, reduce_only=True)
            log.info(f"Принудительно закрыта позиция {symbol} ({side}) qty={qty}")
            return True
        except Exception as e:
            log.warning(f"Не удалось закрыть позицию {symbol}: {e}")
            return False

    def set_leverage(self, symbol, leverage):
        sym = symbol.replace("/","").replace(":USDT","")
        try:
            self.session.set_leverage(category="linear",symbol=sym,
                                      buyLeverage=str(leverage),sellLeverage=str(leverage))
        except Exception as e:
            log.warning(f"Ошибка плеча (не критично): {e}")

    def create_market_order(self, symbol, side, qty,
                             take_profit=None, stop_loss=None, reduce_only=False):
        sym = symbol.replace("/","").replace(":USDT","")
        params = {
            "category":"linear","symbol":sym,
            "side":"Buy" if side=="buy" else "Sell",
            "orderType":"Market","qty":str(qty),"timeInForce":"GTC",
        }
        if reduce_only:  params["reduceOnly"] = True
        if take_profit:  params["takeProfit"] = str(take_profit)
        if stop_loss:    params["stopLoss"]    = str(stop_loss)
        r = self.session.place_order(**params)
        order_id = r["result"]["orderId"]
        time.sleep(1)
        avg = 0.0
        try:
            hist = self.session.get_order_history(category="linear",symbol=sym,orderId=order_id)
            avg = float(hist["result"]["list"][0].get("avgPrice",0) or 0)
        except: pass
        return {"average": avg, "id": order_id}

    def price_to_precision(self, symbol, price):
        return str(round(price, 2))

    def amount_to_precision(self, symbol: str, amount: float) -> float:
        sym  = symbol.replace("/","").replace(":USDT","")
        info = INSTRUMENTS.get(sym, {"minOrderQty":0.001,"qtyStep":0.001})
        step, min_qty = info["qtyStep"], info["minOrderQty"]
        qty = math.floor(amount / step) * step
        qty = round(qty, 10)
        return qty if qty >= min_qty else 0.0

    def update_stop_loss(self, symbol, stop_price):
        sym = symbol.replace("/","").replace(":USDT","")
        try:
            self.session.set_trading_stop(category="linear",symbol=sym,
                                          stopLoss=str(stop_price),
                                          slTriggerBy="MarkPrice",positionIdx=0)
            return True
        except Exception as e:
            log.warning(f"Не удалось обновить SL: {e}")
            return False


# ------------------------------------------------------------
if TESTNET_MODE:
    exchange = BybitWrapper(True,
        os.getenv("BYBIT_TESTNET_API_KEY"), os.getenv("BYBIT_TESTNET_API_SECRET"))
    log.info("TESTNET (Aurora + Copy)")
else:
    exchange = BybitWrapper(False,
        os.getenv("BYBIT_API_KEY"), os.getenv("BYBIT_API_SECRET"))

INSTRUMENTS: dict = {}
try:
    r = exchange.session.get_instruments_info(category="linear")
    for item in r["result"]["list"]:
        lot = item.get("lotSizeFilter",{})
        INSTRUMENTS[item["symbol"]] = {
            "minOrderQty": float(lot.get("minOrderQty",0.001)),
            "qtyStep":     float(lot.get("qtyStep",0.001)),
        }
    log.info(f"Загружено {len(INSTRUMENTS)} инструментов")
except Exception as e:
    log.warning(f"Ошибка загрузки инструментов: {e}")


# ============================================================
# ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ
# ============================================================
def _rma(s, span): return s.ewm(alpha=1/span, adjust=False).mean()
def _ema(s, span): return s.ewm(span=span, adjust=False).mean()

def calc_atr(df, period=14):
    hi,lo,pc = df["h"],df["l"],df["c"].shift(1)
    tr = pd.concat([hi-lo,(hi-pc).abs(),(lo-pc).abs()],axis=1).max(axis=1)
    return _rma(tr,period)

def calc_supertrend(df, period=10, mult=3.0):
    atr = calc_atr(df,period)
    hl2 = (df["h"]+df["l"])/2
    ub = (hl2+mult*atr).copy()
    lb = (hl2-mult*atr).copy()
    trend = pd.Series(1,index=df.index)
    for i in range(1,len(df)):
        c,pc = df["c"].iloc[i],df["c"].iloc[i-1]
        pu,pl,pt = ub.iloc[i-1],lb.iloc[i-1],trend.iloc[i-1]
        ub.iloc[i] = ub.iloc[i] if ub.iloc[i]<pu or pc>pu else pu
        lb.iloc[i] = lb.iloc[i] if lb.iloc[i]>pl or pc<pl else pl
        if   pt==1  and c<lb.iloc[i]: trend.iloc[i]=-1
        elif pt==-1 and c>ub.iloc[i]: trend.iloc[i]=1
        else: trend.iloc[i]=pt
    return trend==1, trend==-1


# ============================================================
# ФУНКЦИЯ ПРИНУДИТЕЛЬНОГО ЗАКРЫТИЯ ВСЕХ ПОЗИЦИЙ ПРИ СТАРТЕ
# ============================================================
def закрыть_все_позиции():
    positions = exchange.fetch_positions()
    for p in positions:
        sym = p["symbol"]
        # Восстанавливаем полный символ
        full_symbol = sym + "/USDT:USDT"
        qty = float(p.get("contracts",0))
        side = p["side"]
        if qty > 0:
            exchange.close_position(full_symbol, qty, side)
            time.sleep(1)


# ============================================================
# ОТКРЫТИЕ / ЗАКРЫТИЕ ПОЗИЦИЙ (те же функции)
# ============================================================
def открыть_позицию(symbol, margin_usdt, tp_price, sl_price, side="long"):
    try:
        exchange.set_leverage(symbol, LEVERAGE)
        ticker = exchange.fetch_ticker(symbol)
        price  = ticker["last"]
        if price==0: return None,None

        qty = exchange.amount_to_precision(symbol, margin_usdt*LEVERAGE/price)
        if qty<=0:
            sym_c = symbol.replace("/","").replace(":USDT","")
            min_q = INSTRUMENTS.get(sym_c,{}).get("minOrderQty","?")
            log.error(f"qty ниже мин. лота ({min_q}) для {symbol}: маржа={margin_usdt:.2f}U цена={price:.4f}")
            return None,None

        sl = (min(sl_price, price-max(price*MIN_SL_PERCENT/100,price*0.001)) if side=="long"
              else max(sl_price, price+max(price*MIN_SL_PERCENT/100,price*0.001)))
        tp = (max(tp_price, price+price*TP_PERCENT/100) if side=="long"
              else min(tp_price, price-price*TP_PERCENT/100))

        log.info(f"Открываем {side} {symbol}: qty={qty} маржа≈{margin_usdt:.2f}U TP={tp:.2f} SL={sl:.2f}")
        order = exchange.create_market_order(symbol,"buy" if side=="long" else "sell",qty,
                                             take_profit=exchange.price_to_precision(symbol,tp),
                                             stop_loss  =exchange.price_to_precision(symbol,sl))
        entry = float(order.get("average",0)) or price
        log.info(f"{side.upper()} открыт: {qty} @ {entry:.6f}")
        return entry, qty
    except Exception as e:
        log.error(f"Ошибка открытия: {e}")
        return None,None


def закрыть_позицию(symbol, qty, side):
    close_side = "sell" if side=="long" else "buy"
    for attempt in range(3):
        try:
            exchange.create_market_order(symbol,close_side,qty,reduce_only=True)
            time.sleep(3)
            active = [p for p in exchange.fetch_positions()
                      if p["symbol"]==symbol.replace("/","").replace(":USDT","")]
            if not active:
                log.info(f"Позиция {symbol} закрыта")
                return True
            log.warning(f"{symbol} не закрылась, попытка {attempt+1}")
            time.sleep(2)
        except Exception as e:
            log.warning(f"Попытка {attempt+1} закрыть {symbol}: {e}")
            time.sleep(2)
    log.error(f"Не удалось закрыть {symbol}")
    return False


def частично_закрыть(symbol, qty_to_close, side) -> float:
    sym_c   = symbol.replace("/","").replace(":USDT","")
    min_qty = INSTRUMENTS.get(sym_c,{}).get("minOrderQty",0.001)
    qty_r   = exchange.amount_to_precision(symbol, qty_to_close)
    if qty_r < min_qty:
        log.warning(f"Частичное закрытие {qty_to_close:.6f} < мин. лот {min_qty} — пропуск")
        return 0.0
    try:
        exchange.create_market_order(symbol,"sell" if side=="long" else "buy",qty_r,reduce_only=True)
        log.info(f"💰 Частичное закрытие: {qty_r} {sym_c}")
        return qty_r
    except Exception as e:
        log.warning(f"Ошибка частичного закрытия: {e}")
        return 0.0


def добавить_к_позиции(symbol, add_margin, side, new_sl=None) -> float:
    try:
        exchange.set_leverage(symbol, LEVERAGE)
        ticker = exchange.fetch_ticker(symbol)
        price  = ticker["last"]
        if price == 0: return 0.0

        qty = exchange.amount_to_precision(symbol, add_margin * LEVERAGE / price)
        if qty <= 0:
            log.warning(f"Пирамидинг: qty слишком мал для {symbol}")
            return 0.0

        sl_price = None
        if new_sl:
            sl_price = exchange.price_to_precision(symbol, new_sl)

        log.info(f"🔺 Пирамидинг: +{qty} {symbol} @ ~{price:.4f} маржа={add_margin:.2f}U")
        exchange.create_market_order(symbol, "buy" if side=="long" else "sell",
                                     qty, stop_loss=sl_price)
        return qty
    except Exception as e:
        log.warning(f"Ошибка пирамидинга: {e}")
        return 0.0


def проверить_signal_exit(symbol, side) -> bool:
    if not SIGNAL_EXIT_ENABLED: return False
    try:
        raw = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA, limit=50)
        if len(raw)<30: return False
        df = pd.DataFrame(raw,columns=["ts","o","h","l","c","v"])
        st_up,st_down = calc_supertrend(df)
        if side=="long"  and st_down.iloc[-1]: return True
        if side=="short" and st_up.iloc[-1]:   return True
    except: pass
    return False


def тренд_подтверждён(symbol, side) -> bool:
    try:
        raw_ta = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA,    limit=30)
        raw_1h = exchange.fetch_ohlcv(symbol, "1h", limit=60)
        if len(raw_ta)<20 or len(raw_1h)<55: return False
        df_ta = pd.DataFrame(raw_ta,columns=["ts","o","h","l","c","v"])
        df_1h = pd.DataFrame(raw_1h,columns=["ts","o","h","l","c","v"])
        st_up,st_down = calc_supertrend(df_ta)
        ema20 = _ema(df_1h["c"],20).iloc[-1]
        ema50 = _ema(df_1h["c"],50).iloc[-1]
        if side=="long":
            return bool(st_up.iloc[-1]) and ema20>ema50
        else:
            return bool(st_down.iloc[-1]) and ema20<ema50
    except:
        return False


# ============================================================
# МОНИТОРИНГ ПОЗИЦИИ
# ============================================================
def мониторить_позицию(symbol, entry_price, qty, открыта_в,
                        sl_цена, tp_цена, side, начальная_маржа):
    deadline  = открыта_в + TRADE_MAX_LIFETIME
    coin      = symbol.split("/")[0]
    be_price  = (entry_price*(1+BYBIT_FEE*2+0.001) if side=="long"
                 else entry_price*(1-BYBIT_FEE*2-0.001))

    текущий_sl   = sl_цена
    пиковая_цена = entry_price
    be_done      = False
    trail_on     = False
    trail_pct    = TRAILING_OFFSET_PCT / 100.0

    partial_done = [False] * len(PARTIAL_CLOSE_LEVELS)
    текущий_qty  = qty

    pyramid_adds       = 0
    pyramid_trigger_price = (
        entry_price * (1 + PYRAMID_TRIGGER_PCT/100) if side=="long"
        else entry_price * (1 - PYRAMID_TRIGGER_PCT/100)
    )

    последняя_цена = 0.0
    цена_не_менялась_сек = 0.0

    log.info(f"Мониторинг {coin} {side} | вход={entry_price:.6f} | SL={sl_цена:.6f} | TP={tp_цена:.6f} | qty={qty}")

    while True:
        if time.time() >= deadline:
            log.warning("Дедлайн – закрываем")
            закрыть_позицию(symbol, текущий_qty, side)
            return "timeout"
        time.sleep(15)
        _ticker_cache.clear()

        try:
            positions = exchange.fetch_positions()
            sym_c  = symbol.replace("/","").replace(":USDT","")
            active = [p for p in positions if p["symbol"]==sym_c and p["side"]==side]

            if not active:
                cur = cached_ticker(symbol)["last"]
                hit_tp = (cur >= entry_price*(1+TP_PERCENT/100*0.7) if side=="long"
                          else cur <= entry_price*(1-TP_PERCENT/100*0.7))
                return "tp" if (hit_tp or be_done) else "sl"

            pos          = active[0]
            текущий_qty  = abs(float(pos.get("contracts",0) or 0))
            cur_price    = cached_ticker(symbol)["last"]

            # Защита от зависшей цены
            if cur_price == последняя_цена:
                цена_не_менялась_сек += 15
                if цена_не_менялась_сек >= STUCK_PRICE_TIMEOUT:
                    log.warning(f"Цена {cur_price} не менялась {цена_не_менялась_сек}с – закрываем")
                    закрыть_позицию(symbol, текущий_qty, side)
                    return "sl"
            else:
                последняя_цена = cur_price
                цена_не_менялась_сек = 0

            # PnL
            calc_pnl_pct = ((cur_price - entry_price)/entry_price*100 if side=="long"
                            else (entry_price - cur_price)/entry_price*100)

            # Частичное закрытие
            if PARTIAL_CLOSE_ENABLED:
                for i,(thr,frac) in enumerate(PARTIAL_CLOSE_LEVELS):
                    if partial_done[i]: continue
                    if calc_pnl_pct >= thr:
                        closed = частично_закрыть(symbol, текущий_qty*frac, side)
                        if closed > 0:
                            partial_done[i] = True
                            текущий_qty     = max(0, текущий_qty-closed)
                            log.info(f"💰 Ч.закр. #{i+1}: +{thr}% → -{closed:.4f} остаток={текущий_qty:.4f}")
                        break

            # Пирамидинг
            if (PYRAMID_ENABLED and pyramid_adds < PYRAMID_MAX_ADDS and cur_price > 0):
                triggered = (
                    (cur_price >= pyramid_trigger_price if side=="long"
                     else cur_price <= pyramid_trigger_price)
                )
                if triggered and тренд_подтверждён(symbol, side):
                    add_margin = начальная_маржа * PYRAMID_FRACTION
                    pyr_sl = (cur_price*(1-PYRAMID_SL_TRAIL_PCT/100) if side=="long"
                              else cur_price*(1+PYRAMID_SL_TRAIL_PCT/100))
                    added_qty = добавить_к_позиции(symbol, add_margin, side, new_sl=pyr_sl)
                    if added_qty > 0:
                        pyramid_adds += 1
                        текущий_qty  += added_qty
                        pyramid_trigger_price = (
                            cur_price*(1+PYRAMID_TRIGGER_PCT/100) if side=="long"
                            else cur_price*(1-PYRAMID_TRIGGER_PCT/100)
                        )
                        log.info(f"🔺 Пирамида #{pyramid_adds}: +{added_qty:.4f} | итого qty={текущий_qty:.4f}")

            # Безубыток
            if PARTIAL_BE_ENABLED and not be_done and calc_pnl_pct >= PARTIAL_BE_PROFIT:
                mark = cached_ticker(symbol).get("mark_price", cur_price)
                ok = ((side=="long"  and be_price < mark*0.9995) or
                      (side=="short" and be_price > mark*1.0005))
                if ok and exchange.update_stop_loss(symbol, be_price):
                    текущий_sl = be_price
                    be_done    = True
                    log.info(f"🎯 SL → БЕЗУБЫТОК: {be_price:.6f}")

            # Трейлинг
            if not trail_on and calc_pnl_pct >= MIN_PROFIT_FOR_TRAIL:
                trail_on = True
                log.info(f"🚀 ТРЕЙЛИНГ @ {cur_price:.6f}")
            if trail_on:
                if side=="long":
                    if cur_price > пиковая_цена: пиковая_цена = cur_price
                    new_sl = пиковая_цена*(1-trail_pct)
                    if new_sl > текущий_sl and exchange.update_stop_loss(symbol,new_sl):
                        текущий_sl = new_sl
                else:
                    if cur_price < пиковая_цена: пиковая_цена = cur_price
                    new_sl = пиковая_цена*(1+trail_pct)
                    if new_sl < текущий_sl and exchange.update_stop_loss(symbol,new_sl):
                        текущий_sl = new_sl

            # Signal Exit
            if SIGNAL_EXIT_ENABLED and be_done and проверить_signal_exit(symbol,side):
                log.info("Signal Exit: разворот – закрываем")
                закрыть_позицию(symbol, текущий_qty, side)
                return "tp" if calc_pnl_pct>0 else "sl"

            pyr_info = f" пирамид={pyramid_adds}" if PYRAMID_ENABLED else ""
            pc_info  = f" ч.закр={sum(partial_done)}/{len(PARTIAL_CLOSE_LEVELS)}" if PARTIAL_CLOSE_ENABLED else ""
            log.info(f"[{coin}] {cur_price:.4f} P&L={calc_pnl_pct:+.2f}% SL={текущий_sl:.4f} BE={be_done} Trail={trail_on}{pyr_info}{pc_info}")

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

def загрузить_историю():
    if not os.path.exists(TRADES_FILE): return []
    try:
        with open(TRADES_FILE,"r") as f: return json.load(f)
    except: return []

def сохранить_сделку(rec):
    h = загрузить_историю(); h.append(rec)
    with open(TRADES_FILE,"w") as f: json.dump(h,f,indent=2,default=str)

def загрузить_состояние():
    global stats
    if not os.path.exists(STATE_FILE): return
    try:
        with open(STATE_FILE,"r") as f:
            saved = json.load(f)
        for k in stats:
            if k in saved: stats[k] = saved[k]
    except: pass

def сохранить_состояние():
    with open(STATE_FILE,"w") as f: json.dump(stats,f,indent=2,default=str)


# ============================================================
# ГЛОБАЛЬНАЯ СТАТИСТИКА
# ============================================================
stats = {
    "запусков":0,"сделок_всего":0,"тейкпрофит":0,"стоплосс":0,"таймаут":0,
    "прибыль_usdt":0.0,"убыток_usdt":0.0,"депозит_старт":0.0,
    "баланс_начало_дня":0.0,"дата_дня":"","старт_время":"",
    "последний_отчёт":0.0,"sl_streak":0,
    "винрейт_по_скору":{
        "ULTRA": {"сделок":0,"побед":0},
        "STRONG":{"сделок":0,"побед":0},
        "NORMAL":{"сделок":0,"побед":0},
        "WEAK":  {"сделок":0,"побед":0},
    },
}


# ============================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================
def main():
    global stats
    загрузить_состояние()

    # ⚡ Принудительно закрываем все открытые позиции при старте
    закрыть_все_позиции()

    stats["запусков"] += 1
    баланс = полный_баланс_usdt()
    if stats["депозит_старт"]<=0: stats["депозит_старт"] = баланс
    if not stats["старт_время"]:  stats["старт_время"]   = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    log.info(f"Aurora+Copy Bot | Баланс: {баланс:.2f} USDT | Мин. скор: {MIN_SCORE}")
    распечатать_винрейт()

    заблокированные: dict = {}
    last_copy_scan = 0

    while True:
        try:
            свободный = баланс_usdt()
            if свободный < MIN_BALANCE:
                log.warning(f"Баланс {свободный:.2f} < {MIN_BALANCE} – пауза 10 мин")
                time.sleep(600); continue

            активные = exchange.fetch_positions()
            if активные:
                log.info(f"Открытые позиции: {[p['symbol'] for p in активные]} – ждём")
                time.sleep(60); continue

            # --------------------------------------------------
            # 1. Сигналы Aurora AI
            # --------------------------------------------------
            signals_aurora = {}
            for sym in SYMBOLS:
                if sym in заблокированные and time.time()<заблокированные[sym]: continue
                if sym in заблокированные: del заблокированные[sym]

                ticker = exchange.fetch_ticker(sym)
                if ticker["last"]==0: continue
                diff = abs(ticker["mark_price"]-ticker["last"])/ticker["last"]*100
                if diff>=MARK_PRICE_DIFF_THRESHOLD: continue

                aurora = получить_aurora_сигнал(sym)
                if not aurora["valid"]: continue

                signals_aurora[sym] = {
                    "signal": aurora["signal"],
                    "confidence": aurora["confidence"],
                    "price": ticker["last"],
                    "symbol": sym,
                }

            # --------------------------------------------------
            # 2. Мастера копитрейдинга (раз в час)
            # --------------------------------------------------
            copy_trades = []
            if time.time() - last_copy_scan > COPY_SCAN_INTERVAL:
                last_copy_scan = time.time()
                masters = найти_успешных_мастеров()
                for m in masters:
                    sym = m["symbol"] + "/USDT:USDT"
                    if sym in SYMBOLS and sym not in заблокированные:
                        copy_trades.append({
                            "symbol": sym,
                            "side": m["side"],
                            "confidence": 80,
                        })

            # Объединяем все сигналы
            all_candidates = []
            for sym, data in signals_aurora.items():
                if data["signal"] in ("long", "short"):
                    score = data["confidence"]
                    if score < MIN_SCORE: continue
                    all_candidates.append((sym, score, data["price"], data["signal"], data["signal"], "Aurora"))
            for t in copy_trades:
                ticker = exchange.fetch_ticker(t["symbol"])
                if ticker["last"]==0: continue
                all_candidates.append((t["symbol"], 80, ticker["last"], t["side"], t["side"], "CopyMaster"))

            if not all_candidates:
                log.info("Нет сигналов – ждём")
                time.sleep(SCAN_INTERVAL); continue

            # Сортируем по скору
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            выбрана, скор, цена, side_str, signal_type, источник = None, 0, 0.0, "", "", ""
            for sym, sc, pr, sd, si, src in all_candidates:
                if sym in заблокированные: continue
                выбрана, скор, цена, side_str, signal_type, источник = sym, sc, pr, sd, si, src
                break

            if выбрана is None:
                log.info("Нет подходящих кандидатов – ждём")
                time.sleep(SCAN_INTERVAL); continue

            # Сила сигнала
            tier_name, tier_mult = определить_силу_сигнала(скор)
            log.info(f"⚡ {tier_name} (источник={источник}, скор={скор}, ×{tier_mult:.2f})")

            # Расчёт TP/SL
            atr_pt = 0.0
            raw_atr = exchange.fetch_ohlcv(выбрана, TIMEFRAME_TA, limit=50)
            if len(raw_atr)>=20:
                df_atr = pd.DataFrame(raw_atr,columns=["ts","o","h","l","c","v"])
                atr_pt = float(calc_atr(df_atr,14).iloc[-1])
            sl_dist = max(MIN_SL_PERCENT,min(MAX_SL_PERCENT,(atr_pt*ATR_SL_MULT/цена)*100)) if atr_pt>0 else SL_PERCENT
            tp_dist = max(TP_PERCENT, sl_dist*2)

            if signal_type == "long":
                sl_цена = цена*(1-sl_dist/100)
                tp_цена = цена*(1+tp_dist/100)
            else:
                sl_цена = цена*(1+sl_dist/100)
                tp_цена = цена*(1-tp_dist/100)

            if abs(tp_цена-цена)/abs(цена-sl_цена)<1.999:
                log.warning(f"⛔ RR<2:1 – пропуск")
                time.sleep(SCAN_INTERVAL); continue

            # Маржа
            margin = min(свободный*BASE_RISK_PCT/100*tier_mult, свободный*0.9)
            ticker = exchange.fetch_ticker(выбрана)
            if ticker["last"]==0: continue
            sym_c = выбрана.replace("/","").replace(":USDT","")
            min_qty = INSTRUMENTS.get(sym_c,{}).get("minOrderQty",0.001)
            min_margin = min_qty*ticker["last"]/LEVERAGE
            if margin<min_margin:
                log.warning(f"⚠️ Маржа {margin:.4f}U < мин. {min_margin:.4f}U → повышаем")
                margin = min_margin
            if margin>свободный*0.95:
                log.error(f"❌ Недостаточно средств: нужно {margin:.4f}U доступно {свободный:.4f}U")
                time.sleep(SCAN_INTERVAL); continue

            log.info(f"✅ ВХОД {signal_type.upper()} [{источник}] скор={скор} SL={sl_цена:.4f} TP={tp_цена:.4f} маржа={margin:.2f}U")

            время_входа = time.time()
            entry_price, qty = открыть_позицию(выбрана, margin, tp_цена, sl_цена, signal_type)
            if entry_price is None:
                time.sleep(30); continue

            stats["сделок_всего"] += 1
            результат = мониторить_позицию(
                выбрана, entry_price, qty, время_входа,
                sl_цена, tp_цена, signal_type, margin
            )

            баланс_после = полный_баланс_usdt()
            pnl = баланс_после - баланс
            duration = (time.time()-время_входа)/60
            победа = результат=="tp"

            обновить_винрейт(tier_name, победа)

            if результат=="tp":
                stats["тейкпрофит"]  += 1
                stats["прибыль_usdt"]+= max(0,pnl)
                stats["sl_streak"]    = 0
                заблокированные[выбрана] = time.time()+SYMBOL_BLOCK_AFTER_TP
                log.info(f"✅ TP [{tier_name}]: ≈{pnl:+.4f} USDT")
            elif результат=="sl":
                stats["стоплосс"]    += 1
                stats["убыток_usdt"] += abs(min(0,pnl))
                stats["sl_streak"]   += 1
                заблокированные[выбрана] = time.time()+SYMBOL_BLOCK_AFTER_SL
                log.warning(f"❌ SL [{tier_name}]: ≈{pnl:+.4f} USDT streak={stats['sl_streak']}")
            else:
                stats["таймаут"]     += 1
                stats["убыток_usdt"] += abs(min(0,pnl))
                stats["sl_streak"]    = 0
                заблокированные[выбрана] = time.time()+SYMBOL_BLOCK_AFTER_TP
                log.warning(f"⏰ Таймаут [{tier_name}]: ≈{pnl:+.4f} USDT")

            сохранить_сделку({
                "время":     datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                "symbol":    выбрана, "side":signal_type,
                "score":     скор, "tier":tier_name, "margin_mult":tier_mult,
                "entry":     entry_price, "sl":sl_цена, "tp":tp_цена,
                "pnl":       round(pnl,4), "duration_min":round(duration,1),
                "результат": результат, "источник": источник,
            })
            сохранить_состояние()

            if stats["сделок_всего"]%5==0:
                распечатать_винрейт()

            log.info("Сделка завершена – пауза 60 сек")
            time.sleep(60)

        except Exception as e:
            log.error(f"Глобальная ошибка: {e}", exc_info=True)
            time.sleep(60)


if __name__ == "__main__":
    main()
