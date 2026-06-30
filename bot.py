#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ГИБРИДНЫЙ БОТ v10.6 — СБОР ДАННЫХ ПО ВСЕМ СИГНАЛАМ (БУМАЖНЫЙ РЕЖИМ)
====================================================================
- Реальная торговля опциональна (REAL_TRADING_ENABLED = False).
- Бумажный трейдер анализирует ВСЕ сигналы, прошедшие фильтры.
- Все завершённые виртуальные сделки пишутся в paper_trades.csv.
- Каждый час выводится сводка.
"""

import os, time, json, logging, ccxt, csv
import pandas as pd, numpy as np
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

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

TIMEFRAME_TA = "5m"
TIMEFRAME_TREND = "1h"
TIMEFRAME_4H = "4h"
SCAN_INTERVAL = 120

REAL_TRADING_ENABLED = False          # Отключаем реальную торговлю для сбора данных
PAPER_TRADING_ENABLED = True
PAPER_REPORT_INTERVAL = 3600          # 1 час
CSV_FILE = "paper_trades.csv"

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

LEVERAGE_MIN = 3
LEVERAGE_MAX = 5

STATE_FILE = "state_bot_v10.6.json"
TRADES_FILE = "trades_bot_v10.6.json"

# ============================================================
#                      ЛОГИРОВАНИЕ
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler("bot_v10.6.log", encoding="utf-8")],
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

def choose_leverage(atr_pct):
    if atr_pct > 1.5: return LEVERAGE_MIN
    elif atr_pct > 0.8: return LEVERAGE_MIN + 1
    else: return LEVERAGE_MAX

# ============================================================
#          ДЕТЕКТОР СТЕНЫ ЗАЯВОК
# ============================================================
def detect_wall_signal(symbol: str) -> Optional[Dict]:
    try:
        book = safe_api(exchange.fetch_order_book, symbol, ORDER_BOOK_DEPTH)
        if not book: return None
        bids = book["bids"]
        asks = book["asks"]
        if len(bids) < 5 or len(asks) < 5: return None

        bid_volumes = [v for _, v in bids[1:]]
        ask_volumes = [v for _, v in asks[1:]]
        med_bid_vol = np.median(bid_volumes) if bid_volumes else 0
        med_ask_vol = np.median(ask_volumes) if ask_volumes else 0

        best_bid, best_bid_vol = bids[0]
        best_ask, best_ask_vol = asks[0]

        bid_wall_vol = best_bid_vol
        bid_wall_price = best_bid
        for price, vol in bids[:5]:
            if vol >= WALL_THRESHOLD_VOL_RATIO * med_bid_vol and vol >= best_bid_vol * 0.8:
                dist_pct = (best_bid - price) / best_bid * 100 if best_bid > 0 else 0
                if dist_pct <= MAX_WALL_DISTANCE_PCT:
                    bid_wall_vol = vol
                    bid_wall_price = price
                    break

        ask_wall_vol = best_ask_vol
        ask_wall_price = best_ask
        for price, vol in asks[:5]:
            if vol >= WALL_THRESHOLD_VOL_RATIO * med_ask_vol and vol >= best_ask_vol * 0.8:
                dist_pct = (price - best_ask) / best_ask * 100 if best_ask > 0 else 0
                if dist_pct <= MAX_WALL_DISTANCE_PCT:
                    ask_wall_vol = vol
                    ask_wall_price = price
                    break

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
#        БУМАЖНЫЙ ТРЕЙДЕР (принимает все сигналы)
# ============================================================
class PaperTrader:
    def __init__(self, csv_file):
        self.positions: List[Dict] = []
        self.closed_trades: List[Dict] = []
        self.last_report_time = time.time()
        self.total_signals = 0
        self.total_tp = 0
        self.total_sl = 0
        self.total_timeout = 0
        self.csv_file = csv_file
        # Инициализируем CSV с заголовками
        self._init_csv()

    def _init_csv(self):
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "symbol", "signal", "price", "wall_usdt", "imbalance", "spread_pct",
                    "atr_pct", "trend_4h", "ma_cross_ok", "volume_ratio", "rr",
                    "best_bid_vol", "best_ask_vol", "result", "pnl_pct", "duration_min"
                ])

    def add_signal(self, symbol, signal_info, atr, sl_price, tp_price, current_price, 
                   trend_4h_val, ma_ok, volume_ratio, rr):
        pos = {
            "symbol": symbol,
            "side": signal_info["signal"],
            "entry": current_price,
            "sl": sl_price,
            "tp": tp_price,
            "atr": atr,
            "time": time.time(),
            "signal_info": signal_info,
            "trend_4h": trend_4h_val,
            "ma_cross_ok": ma_ok,
            "volume_ratio": volume_ratio,
            "rr": rr,
            "best_bid_vol": 0,  # заполним позже при закрытии
            "best_ask_vol": 0,
        }
        self.positions.append(pos)
        self.total_signals += 1
        log.info(f"📝 Бумажный вход: {signal_info['signal'].upper()} {symbol} цена={current_price:.6f} wall={signal_info['wall_usdt']:.0f}")

    def update(self):
        now = time.time()
        closed = []
        for pos in self.positions:
            symbol = pos["symbol"]
            ticker = fetch_ticker(symbol)
            if not ticker: continue
            cur = float(ticker["last"])
            deadline = pos["time"] + TRADE_MAX_LIFETIME

            result = None
            if now >= deadline:
                result = "timeout"
            elif pos["side"] == "long":
                if cur >= pos["tp"]: result = "tp"
                elif cur <= pos["sl"]: result = "sl"
            else:
                if cur <= pos["tp"]: result = "tp"
                elif cur >= pos["sl"]: result = "sl"

            if result:
                pnl_pct = (cur/pos["entry"] - 1)*100 if pos["side"]=="long" else (pos["entry"]/cur - 1)*100
                trade = {
                    "timestamp": datetime.now().isoformat(),
                    "symbol": pos["symbol"],
                    "signal": pos["side"],
                    "price": pos["entry"],
                    "wall_usdt": pos["signal_info"]["wall_usdt"],
                    "imbalance": pos["signal_info"]["imbalance"],
                    "spread_pct": pos["signal_info"]["spread_pct"],
                    "atr_pct": (pos["atr"] / pos["entry"]) * 100,
                    "trend_4h": pos["trend_4h"],
                    "ma_cross_ok": pos["ma_cross_ok"],
                    "volume_ratio": pos["volume_ratio"],
                    "rr": pos["rr"],
                    "best_bid_vol": pos["best_bid_vol"],
                    "best_ask_vol": pos["best_ask_vol"],
                    "result": result,
                    "pnl_pct": pnl_pct,
                    "duration_min": (now - pos["time"]) / 60,
                    "close_time": now,
                }
                self.closed_trades.append(trade)
                # Записываем в CSV
                with open(self.csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        trade["timestamp"], trade["symbol"], trade["signal"], trade["price"],
                        trade["wall_usdt"], trade["imbalance"], trade["spread_pct"],
                        trade["atr_pct"], trade["trend_4h"], trade["ma_cross_ok"],
                        trade["volume_ratio"], trade["rr"], trade["best_bid_vol"], trade["best_ask_vol"],
                        trade["result"], trade["pnl_pct"], trade["duration_min"]
                    ])
                if result == "tp": self.total_tp += 1
                elif result == "sl": self.total_sl += 1
                else: self.total_timeout += 1
                closed.append(pos)

        for pos in closed:
            self.positions.remove(pos)

    def generate_hourly_report(self):
        now = time.time()
        recent_trades = [t for t in self.closed_trades if t.get("close_time", 0) >= now - 3600]
        total_hour = len(recent_trades)
        wins_hour = sum(1 for t in recent_trades if t["result"] == "tp") if total_hour > 0 else 0
        avg_pnl_hour = np.mean([t["pnl_pct"] for t in recent_trades]) if total_hour > 0 else 0.0

        total_all = len(self.closed_trades)
        winrate_all = self.total_tp / total_all * 100 if total_all > 0 else 0.0
        avg_pnl_all = np.mean([t["pnl_pct"] for t in self.closed_trades]) if total_all > 0 else 0.0

        log.info("=" * 60)
        log.info("📋 БУМАЖНАЯ СВОДКА")
        log.info(f"   Открытых позиций сейчас: {len(self.positions)}")
        log.info(f"   За последний час: сделок {total_hour}, TP {wins_hour}, средний P&L {avg_pnl_hour:+.2f}%")
        log.info(f"   За всё время: сделок {total_all}, TP {self.total_tp}, SL {self.total_sl}, Timeout {self.total_timeout}")
        log.info(f"   WinRate: {winrate_all:.1f}% | Средний P&L: {avg_pnl_all:+.2f}%")
        log.info(f"   Данные сохранены в {self.csv_file}")
        log.info("=" * 60)
        self.last_report_time = now

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
    log.info(f"=== СТАКАННЫЙ БОТ v10.6 (сбор данных) ===")
    log.info(f"Реальная торговля: {'ВКЛ' if REAL_TRADING_ENABLED else 'ВЫКЛ'}")
    log.info(f"Депозит: {stats['депозит_старт']:.2f} USDT")

    paper_trader = PaperTrader(CSV_FILE) if PAPER_TRADING_ENABLED else None

    # Подхват существующих реальных позиций, если реальная торговля включена
    if REAL_TRADING_ENABLED:
        existing = [p for p in fetch_positions() if float(p.get("contracts",0))>0]
        if existing:
            log.info(f"Найдены открытые позиции: {[(p['symbol'], p['side']) for p in existing]}")
            for pos in existing:
                handle_existing_position(pos)
            log.info("Все существующие позиции обработаны")

    заблокированные: Dict[str, float] = {}
    fail_attempts: Dict[str, int] = {}

    while True:
        try:
            now = time.time()
            # Обновление бумажного трейдера
            if paper_trader:
                paper_trader.update()
                if now - paper_trader.last_report_time >= PAPER_REPORT_INTERVAL:
                    paper_trader.generate_hourly_report()

            if REAL_TRADING_ENABLED:
                # ... (реальная торговля остаётся без изменений) ...
                pass  # в данной версии оставим только бумажный режим

            свободный = get_balance(free=True)

            # Сканирование
            log.info("── Сканирование стаканов ──")
            signals = []
            for sym in SYMBOLS:
                if sym in заблокированные and time.time() < заблокированные[sym]:
                    continue
                sig = detect_wall_signal(sym)
                if sig:
                    # Фильтр тренда 4h
                    if sig["signal"] == "long" and not trend_4h(sym, "bull"): continue
                    if sig["signal"] == "short" and not trend_4h(sym, "bear"): continue
                    # Быстрый расчёт MA и volume для информации (не блокируем)
                    df_ta = pd.DataFrame(fetch_ohlcv(sym, TIMEFRAME_TA, limit=20), columns=["ts","o","h","l","c","v"])
                    ma_ok = ma_cross_ok(df_ta, sig["signal"]) if len(df_ta) >= 5 else True
                    # Без volume_spike_guard – пускаем все, чтобы собрать статистику
                    signals.append((sym, sig, ma_ok))

            log.info(f"Найдено {len(signals)} сигналов после фильтра тренда")

            # Для каждого сигнала открываем бумажную позицию
            for sym, sig, ma_ok in signals:
                df_ta = pd.DataFrame(fetch_ohlcv(sym, TIMEFRAME_TA, limit=50), columns=["ts","o","h","l","c","v"])
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

                # Ограничения SL/TP
                if sig["signal"] == "long":
                    sl = max(price * (1 - MAX_SL_PERCENT/100), min(price * (1 - MIN_SL_PERCENT/100), sl))
                    tp = max(price * (1 + TP_PERCENT/100), tp)
                else:
                    sl = min(price * (1 + MAX_SL_PERCENT/100), max(price * (1 + MIN_SL_PERCENT/100), sl))
                    tp = min(price * (1 - TP_PERCENT/100), tp)

                rr = abs(tp - price) / abs(sl - price) if abs(price - sl) > 0 else 0
                current_price = float(fetch_ticker(sym)["last"]) if fetch_ticker(sym) else price
                trend_val = "bull" if trend_4h(sym, "bull") else ("bear" if trend_4h(sym, "bear") else "neutral")
                volume_ratio = df_ta["v"].iloc[-1] / (df_ta["v"].tail(20).mean() + 1e-10) if len(df_ta) >= 20 else 0

                paper_trader.add_signal(
                    symbol=sym,
                    signal_info=sig,
                    atr=atr_val,
                    sl_price=sl,
                    tp_price=tp,
                    current_price=current_price,
                    trend_4h_val=trend_val,
                    ma_ok=ma_ok,
                    volume_ratio=volume_ratio,
                    rr=rr
                )

            # Если реальная торговля включена, выбираем лучший сигнал (по желанию)
            if REAL_TRADING_ENABLED and signals:
                # ... (код реальной торговли) ...
                pass

            time.sleep(SCAN_INTERVAL)

        except Exception as e:
            log.error(f"Ошибка в цикле: {e}", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    main()
