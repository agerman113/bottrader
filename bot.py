#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ГИБРИДНЫЙ БОТ v10.3 — СТАКАННЫЙ СКАЛЬПЕР НА ОСНОВЕ МОНСТРА
=============================================================
Взят каркас v10.2 (риск‑менеджмент, мониторинг, частичный безубыток,
трейлинг, отчёты), а сигналы заменены на анализ Order Book:
  • Поиск крупных лимитных стен (bid/ask wall)
  • Дисбаланс bid/ask объёмов
  • Фильтр тренда 4h, MA‑кроссовер, минимальная ATR
  • Выбор лучшей монеты по объёму стены в USDT

Работает с Bybit USDT‑M фьючерсами.
"""

import os, time, json, logging, ccxt
import pandas as pd, numpy as np
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

load_dotenv()

# ============================================================
#                      КОНФИГУРАЦИЯ
# ============================================================
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

LEVERAGE = 3
TIMEFRAME_TA = "5m"
TIMEFRAME_TREND = "1h"
TIMEFRAME_4H = "4h"
SCAN_INTERVAL = 120

MIN_SCORE = 1
ENTRY_CONFIRM_BARS = 0
MA_CROSSOVER_ENABLED = True
MA1_TYPE, MA2_TYPE = "EMA", "EMA"
MA1_LENGTH, MA2_LENGTH = 21, 50

# --- Стаканные сигналы ---
ORDER_BOOK_DEPTH = 20
WALL_THRESHOLD_VOL_RATIO = 3.0
MIN_WALL_VOLUME_USDT = 500
MAX_WALL_DISTANCE_PCT = 2.0
IMBALANCE_RATIO_LONG = 1.5
IMBALANCE_RATIO_SHORT = 1 / 1.5

# --- TP/SL на основе ATR ---
ATR_SL_MULT = 1.5
ATR_TP_MULT = 3.0
MIN_SL_PERCENT = 0.8
MAX_SL_PERCENT = 2.0
TP_PERCENT = 3.0

# --- Частичный безубыток, трейлинг, фильтры ---
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
VOLUME_SPIKE_MULT = 5.0
VOLUME_AVG_PERIOD = 20

DAILY_LOSS_LIMIT_PCT = 3.0
DAILY_LOSS_PAUSE_SEC = 10800
SYMBOL_BLOCK_AFTER_TP = 90 * 60
SYMBOL_BLOCK_AFTER_SL = 180 * 60
SYMBOL_MAX_FAIL_ATTEMPTS = 3
SYMBOL_BLOCK_AFTER_FAIL = 120 * 60
SL_STREAK_LIMIT = 3
SL_STREAK_PAUSE = 1800
SL_STREAK_EXTRA_PAUSE = 300
MIN_BALANCE = 5.0
MAX_DRAWDOWN_PCT = 15.0
TRADE_MAX_LIFETIME = 7200
REPORT_INTERVAL = 1800
BYBIT_FEE = 0.00055
RISK_PCT = 0.8

STATE_FILE = "state_bot_v10.3.json"
TRADES_FILE = "trades_bot_v10.3.json"

# ============================================================
#                      ЛОГИРОВАНИЕ
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler("bot_v10.3.log", encoding="utf-8")],
)
log = logging.getLogger("WallScalper")

# ============================================================
#                      БИРЖА
# ============================================================
exchange = ccxt.bybit({
    "apiKey": os.getenv("BYBIT_API_KEY"),
    "secret": os.getenv("BYBIT_API_SECRET"),
    "enableRateLimit": True,
    "options": {"defaultType": "linear"},
})

# ============================================================
#                      СТАТИСТИКА
# ============================================================
stats = {
    "запусков": 0, "сделок_всего": 0, "тейкпрофит": 0, "стоплосс": 0,
    "таймаут": 0, "прибыль_usdt": 0.0, "убыток_usdt": 0.0,
    "депозит_старт": 0.0, "баланс_начало_дня": 0.0, "дата_дня": "",
    "старт_время": "", "последний_отчёт": 0.0, "sl_streak": 0,
}

# ============================================================
#                 ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================
def safe_api(func, *a, retries=3, delay=1.0, **kw):
    for attempt in range(retries):
        try:
            return func(*a, **kw)
        except ccxt.RateLimitExceeded:
            log.warning("Rate limit, пауза 5с"); time.sleep(5)
        except ccxt.NetworkError as e:
            log.warning(f"Сеть: {e}"); time.sleep(delay); delay *= 2
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

def _ema(s, span): return s.ewm(span=span, adjust=False).mean()
def _rma(s, span): return s.ewm(alpha=1/span, adjust=False).mean()

def calc_atr(df, period=14):
    hi, lo, pc = df["h"], df["l"], df["c"].shift(1)
    tr = pd.concat([hi-lo, (hi-pc).abs(), (lo-pc).abs()], axis=1).max(axis=1)
    return _rma(tr, period)

def calc_ma(df, ma_type, length):
    s = df["c"]
    if ma_type.upper() == "EMA": return _ema(s, length)
    elif ma_type.upper() == "SMA": return s.rolling(length).mean()
    else: return _ema(s, length)

def ma_cross_ok(df, side):
    if not MA_CROSSOVER_ENABLED: return True
    try:
        ma1 = calc_ma(df, MA1_TYPE, MA1_LENGTH)
        ma2 = calc_ma(df, MA2_TYPE, MA2_LENGTH)
        return bool(ma1.iloc[-1] > ma2.iloc[-1]) if side == "long" else bool(ma1.iloc[-1] < ma2.iloc[-1])
    except: return True

def trend_4h(symbol, direction="bull"):
    try:
        raw = fetch_ohlcv(symbol, TIMEFRAME_4H, limit=60)
        if len(raw) < 55: return False
        df = pd.DataFrame(raw, columns=["ts","o","h","l","c","v"])
        ema20 = _ema(df["c"], 20).iloc[-1]
        ema50 = _ema(df["c"], 50).iloc[-1]
        return ema20 > ema50 if direction == "bull" else ema20 < ema50
    except: return False

def volume_spike_guard(df):
    try:
        avg = df["v"].rolling(VOLUME_AVG_PERIOD).mean().iloc[-1]
        return (df["v"].iloc[-1] / (avg+1e-10)) <= VOLUME_SPIKE_MULT
    except: return True

# ============================================================
#          ДЕТЕКТОР СТЕНЫ ЗАЯВОК (Order Book Wall)
# ============================================================
def detect_wall_signal(symbol: str) -> Optional[Dict]:
    """Возвращает сигнал на основе стены в стакане."""
    try:
        book = safe_api(exchange.fetch_order_book, symbol, ORDER_BOOK_DEPTH)
        if not book: return None
        bids = book["bids"]
        asks = book["asks"]
        if len(bids) < 5 or len(asks) < 5: return None

        # Медианные объёмы
        bid_volumes = [v for _, v in bids[1:]]
        ask_volumes = [v for _, v in asks[1:]]
        med_bid_vol = np.median(bid_volumes) if bid_volumes else 0
        med_ask_vol = np.median(ask_volumes) if ask_volumes else 0

        best_bid, best_bid_vol = bids[0]
        best_ask, best_ask_vol = asks[0]

        # Поиск bid wall
        bid_wall_vol = best_bid_vol
        bid_wall_price = best_bid
        for price, vol in bids[:5]:
            if vol >= WALL_THRESHOLD_VOL_RATIO * med_bid_vol and vol >= best_bid_vol * 0.8:
                dist_pct = (best_bid - price) / best_bid * 100 if best_bid > 0 else 0
                if dist_pct <= MAX_WALL_DISTANCE_PCT:
                    bid_wall_vol = vol
                    bid_wall_price = price
                    break

        # Поиск ask wall
        ask_wall_vol = best_ask_vol
        ask_wall_price = best_ask
        for price, vol in asks[:5]:
            if vol >= WALL_THRESHOLD_VOL_RATIO * med_ask_vol and vol >= best_ask_vol * 0.8:
                dist_pct = (price - best_ask) / best_ask * 100 if best_ask > 0 else 0
                if dist_pct <= MAX_WALL_DISTANCE_PCT:
                    ask_wall_vol = vol
                    ask_wall_price = price
                    break

        # Дисбаланс
        total_bid = sum(v for _, v in bids[:ORDER_BOOK_DEPTH])
        total_ask = sum(v for _, v in asks[:ORDER_BOOK_DEPTH])
        imbalance = total_bid / (total_ask + 1e-10)

        spread_pct = (best_ask - best_bid) / best_bid * 100 if best_bid > 0 else 0

        signal = None
        wall_usdt = 0
        if imbalance > IMBALANCE_RATIO_LONG and spread_pct < 1.0:
            signal = "long"
            wall_usdt = bid_wall_vol * bid_wall_price
        elif imbalance < IMBALANCE_RATIO_SHORT and spread_pct < 1.0:
            signal = "short"
            wall_usdt = ask_wall_vol * ask_wall_price
        else:
            if bid_wall_vol * best_bid > MIN_WALL_VOLUME_USDT * 5 and imbalance > 1.2:
                signal = "long"
                wall_usdt = bid_wall_vol * best_bid
            elif ask_wall_vol * best_ask > MIN_WALL_VOLUME_USDT * 5 and imbalance < 0.8:
                signal = "short"
                wall_usdt = ask_wall_vol * best_ask

        if signal and wall_usdt >= MIN_WALL_VOLUME_USDT:
            return {
                "signal": signal,
                "wall_usdt": wall_usdt,
                "price": best_ask if signal == "long" else best_bid,
                "spread_pct": spread_pct,
                "imbalance": imbalance,
            }
        return None
    except Exception as e:
        log.debug(f"Ошибка стакана {symbol}: {e}")
        return None

# ============================================================
#                  ОТКРЫТИЕ ПОЗИЦИИ (исправлено)
# ============================================================
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

def open_position(symbol, side, qty, tp_price, sl_price):
    set_leverage(symbol, LEVERAGE)
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
        log.info(f"{side.upper()} открыт: {qty} @ {entry:.6f}")
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
        return True
    except Exception as e:
        log.warning(f"Не удалось обновить SL: {e}")
        return False

# ============================================================
#           МОНИТОРИНГ ПОЗИЦИИ (взят из v10.2)
# ============================================================
def monitor_position(symbol, entry, qty, start_time, sl_price, tp_price, side, atr):
    deadline = start_time + TRADE_MAX_LIFETIME
    be_done = False
    trailing_active = False
    peak = entry
    current_sl = sl_price
    trailing_offset = max(MIN_TRAILING_OFFSET/100, atr/entry * TRAILING_OFFSET_MULT)
    trailing_step = max(MIN_TRAILING_STEP/100, atr/entry * TRAILING_ATR_MULT)
    rr_trigger = entry + (tp_price - entry) * RR_EXIT_TRIGGER if side == "long" else entry - (entry - tp_price) * RR_EXIT_TRIGGER
    partial_done = False

    log.info(f"Мониторинг {symbol} {side} вход={entry:.6f} SL={sl_price:.6f} TP={tp_price:.6f}")
    while True:
        now = time.time()
        if now >= deadline:
            log.warning("Дедлайн — закрытие")
            close_position(symbol, qty, side)
            return "timeout"
        time.sleep(15)
        pos_list = fetch_positions([symbol])
        active = [p for p in pos_list if float(p.get("contracts", 0) or 0) > 0 and p.get("side") == side]
        if not active:
            ticker = fetch_ticker(symbol)
            cur = float(ticker["last"]) if ticker and ticker.get("last") else entry
            return "tp" if (cur >= tp_price if side=="long" else cur <= tp_price) or be_done else "sl"
        pos = active[0]
        ticker = fetch_ticker(symbol)
        if not ticker or ticker.get("last") is None: continue
        cur = float(ticker["last"])
        qty_act = abs(float(pos.get("contracts", 0) or 0))
        pnl_pct = (cur/entry - 1)*100 if side=="long" else (entry/cur - 1)*100

        # Частичный безубыток
        if PARTIAL_BE_ENABLED and not partial_done and pnl_pct >= PARTIAL_BE_PROFIT:
            close_qty = qty_act * (PARTIAL_BE_CLOSE_PCT/100)
            if close_qty > 0:
                close_s = "sell" if side=="long" else "buy"
                try:
                    exchange.create_market_order(symbol, close_s, close_qty, params={"reduceOnly": True})
                    log.info(f"Частичный безубыток: {close_qty:.4f}")
                    qty_act -= close_qty
                    new_sl = entry * (1 + BYBIT_FEE*2 + 0.0003) if side=="long" else entry * (1 - BYBIT_FEE*2 - 0.0003)
                    if update_sl(symbol, new_sl, side):
                        current_sl = new_sl
                    partial_done = True
                except Exception as e:
                    log.warning(f"Ошибка частичного закрытия: {e}")

        # Signal exit по смене 4h тренда
        if SIGNAL_EXIT_ENABLED and be_done and pnl_pct > 0.5:
            if (side == "long" and not trend_4h(symbol, "bull")) or (side == "short" and not trend_4h(symbol, "bear")):
                log.info("Signal exit по 4h тренду")
                close_position(symbol, qty_act, side)
                return "tp"

        # Безубыток
        if not partial_done and not be_done and pnl_pct >= 0.3:
            new_sl = entry * (1 + BYBIT_FEE*2 + 0.0003) if side=="long" else entry * (1 - BYBIT_FEE*2 - 0.0003)
            if update_sl(symbol, new_sl, side):
                current_sl = new_sl
                be_done = True
                log.info("Безубыток")

        # Трейлинг
        if be_done:
            if not trailing_active:
                if (side=="long" and cur >= rr_trigger) or (side=="short" and cur <= rr_trigger):
                    trailing_active = True
                    peak = cur
                    log.info("Трейлинг активирован")
            if trailing_active and pnl_pct >= MIN_PROFIT_FOR_TRAIL:
                if side == "long":
                    if cur > peak: peak = cur
                    new_sl = peak * (1 - trailing_offset)
                    if new_sl > current_sl and update_sl(symbol, new_sl, side):
                        current_sl = new_sl
                else:
                    if cur < peak: peak = cur
                    new_sl = peak * (1 + trailing_offset)
                    if new_sl < current_sl and update_sl(symbol, new_sl, side):
                        current_sl = new_sl

        log.info(f"[{symbol}] {cur:.6f} P&L={pnl_pct:+.2f}% SL={current_sl:.6f}")
    return "sl"

# ============================================================
#                    ГЛАВНЫЙ ЦИКЛ
# ============================================================
def main():
    global stats
    if not os.getenv("BYBIT_API_KEY"):
        log.error("Нет API ключей")
        return

    stats["депозит_старт"] = get_balance(free=False)
    stats["старт_время"] = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    log.info(f"=== СТАКАННЫЙ БОТ v10.3 ===")
    заблокированные: Dict[str, float] = {}
    fail_attempts: Dict[str, int] = {}

    while True:
        try:
            if time.time() - stats.get("последний_отчёт", 0) >= REPORT_INTERVAL:
                bal = get_balance(free=False)
                log.info(f"Баланс: {bal:.2f} USDT | Сделок: {stats['сделок_всего']} | TP: {stats['тейкпрофит']} SL: {stats['стоплосс']}")
                stats["последний_отчёт"] = time.time()

            свободный = get_balance(free=True)
            if свободный < MIN_BALANCE:
                active = [p for p in fetch_positions() if float(p.get("contracts",0))>0]
                if not active:
                    log.warning("Мало средств")
                    time.sleep(300)
                    continue

            # Дневной лимит
            if stats["депозит_старт"] > 0:
                loss = (stats["депозит_старт"] - get_balance(free=False)) / stats["депозит_старт"] * 100
                if loss >= DAILY_LOSS_LIMIT_PCT:
                    log.warning("Дневной лимит")
                    time.sleep(DAILY_LOSS_PAUSE_SEC)
                    continue

            # Ждём закрытия открытых позиций
            if [p for p in fetch_positions() if float(p.get("contracts",0))>0]:
                time.sleep(30)
                continue

            # Сканирование стаканов
            signals = []
            for sym in SYMBOLS:
                if sym in заблокированные and time.time() < заблокированные[sym]:
                    continue
                sig = detect_wall_signal(sym)
                if sig:
                    if sig["signal"] == "long" and not trend_4h(sym, "bull"): continue
                    if sig["signal"] == "short" and not trend_4h(sym, "bear"): continue
                    signals.append((sym, sig))

            if not signals:
                log.info("Нет сигналов")
                time.sleep(SCAN_INTERVAL)
                continue

            signals.sort(key=lambda x: x[1]["wall_usdt"], reverse=True)
            sym, sig = signals[0]

            # Проверка MA кроссовера и объёма
            df_ta = pd.DataFrame(fetch_ohlcv(sym, TIMEFRAME_TA, limit=50), columns=["ts","o","h","l","c","v"])
            if len(df_ta) < 20: continue
            if not ma_cross_ok(df_ta, sig["signal"]):
                log.info(f"MA кроссовер не пройден для {sym}")
                continue
            if not volume_spike_guard(df_ta):
                continue

            atr_val = calc_atr(df_ta, 14).iloc[-1] if len(df_ta) > 14 else sig["price"] * 0.01
            price = sig["price"]
            sl_dist = atr_val * ATR_SL_MULT
            tp_dist = atr_val * ATR_TP_MULT
            if sig["signal"] == "long":
                sl = price - sl_dist
                tp = price + tp_dist
            else:
                sl = price + sl_dist
                tp = price - tp_dist

            # Ограничиваем SL/TP
            if sig["signal"] == "long":
                sl = max(price * (1 - MAX_SL_PERCENT/100), min(price * (1 - MIN_SL_PERCENT/100), sl))
                tp = max(price * (1 + TP_PERCENT/100), tp)
            else:
                sl = min(price * (1 + MAX_SL_PERCENT/100), max(price * (1 + MIN_SL_PERCENT/100), sl))
                tp = min(price * (1 - TP_PERCENT/100), tp)

            rr = abs(tp - price) / abs(sl - price) if abs(price - sl) > 0 else 0
            if rr < 2.0:
                log.info(f"RR={rr:.1f} < 2.0, пропуск")
                continue

            risk_usdt = свободный * RISK_PCT / 100
            qty = risk_usdt / abs(sl - price)
            try:
                qty = float(exchange.amount_to_precision(sym, qty))
            except:
                pass
            if qty <= 0: continue

            log.info(f"Вход {sig['signal'].upper()} {sym} стена={sig['wall_usdt']:.0f} USDT цена={price:.6f} SL={sl:.6f} TP={tp:.6f}")
            entry, qty_open = open_position(sym, sig["signal"], qty, tp, sl)
            if not entry:
                log.warning("Не удалось открыть позицию")
                fail_attempts[sym] = fail_attempts.get(sym, 0) + 1
                if fail_attempts[sym] >= SYMBOL_MAX_FAIL_ATTEMPTS:
                    заблокированные[sym] = time.time() + SYMBOL_BLOCK_AFTER_FAIL
                    fail_attempts.pop(sym, None)
                continue
            fail_attempts.pop(sym, None)

            stats["сделок_всего"] += 1
            start_t = time.time()
            result = monitor_position(sym, entry, qty_open, start_t, sl, tp, sig["signal"], atr_val)
            cur_price = float(fetch_ticker(sym).get("last", entry)) if fetch_ticker(sym) else entry
            pnl = (cur_price - entry) * qty_open if sig["signal"]=="long" else (entry - cur_price) * qty_open
            stats["прибыль_usdt" if result=="tp" else "убыток_usdt"] += max(0, pnl) if result=="tp" else abs(min(0, pnl))
            stats["тейкпрофит" if result=="tp" else "стоплосс"] += 1
            stats["sl_streak"] = 0 if result=="tp" else stats.get("sl_streak",0)+1
            if result == "tp":
                заблокированные[sym] = time.time() + SYMBOL_BLOCK_AFTER_TP
            else:
                заблокированные[sym] = time.time() + SYMBOL_BLOCK_AFTER_SL
                if stats["sl_streak"] >= SL_STREAK_LIMIT:
                    log.warning("Серия SL, пауза")
                    time.sleep(SL_STREAK_PAUSE)
                    stats["sl_streak"] = 0
            log.info(f"Сделка закрыта: {result} PnL={pnl:.4f} USDT")
            time.sleep(30)

        except Exception as e:
            log.error(f"Ошибка в цикле: {e}", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    main()
