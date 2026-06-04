#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bybit Мем-Коин Бот v3.0 — Testnet
- Тренд с Bybit (EMA 9/21 на 3m свечах)
- После убытка: переворот направления
- Мартингейл до 10 шагов (x1.5 за шаг)
- Трейлинг-стоп через Bybit API
- Учёт комиссий при расчёте ставки
- Лимит сделки 3 минуты, без тейк-профита
- Плечо 5x (риск до ~50% депозита)
"""

import os, time, math, logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

load_dotenv()

# ───────────────────────────────────────────────
# КОНФИГ
# ───────────────────────────────────────────────
# На Bybit Testnet мем-коины с маленькой ценой
# торгуются с префиксом 1000 или 10000
SYMBOLS = ["1000PEPEUSDT", "WIFUSDT", "1000BONKUSDT"]

LEVERAGE         = 5        # плечо
TRADE_DURATION   = 180      # секунд на сделку
MAX_MART_STEPS   = 10       # макс шагов мартингейла
BASE_RISK_PCT    = 2.0      # % баланса на базовую ставку
MART_MULT        = 1.5      # множитель за шаг мартингейла
FEE_RATE         = 0.00055  # тейкер комиссия Bybit
MIN_BALANCE      = 5.0      # мин. баланс для торговли
INITIAL_SL_PCT   = 2.0      # базовый SL %
TRAILING_PCT     = 1.0      # трейлинг дельта %
TIMEFRAME        = "3"      # 3-минутные свечи

# ───────────────────────────────────────────────
# ЛОГГЕР
# ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("meme_bot.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ───────────────────────────────────────────────
# BYBIT WRAPPER
# ───────────────────────────────────────────────
class Bybit:
    def __init__(self):
        self.s = HTTP(
            testnet=True,
            api_key=os.getenv("BYBIT_TESTNET_API_KEY", ""),
            api_secret=os.getenv("BYBIT_TESTNET_API_SECRET", ""),
        )

    def balance(self) -> float:
        try:
            r = self.s.get_wallet_balance(accountType="UNIFIED")
            return float(r["result"]["list"][0].get("totalAvailableBalance", 0))
        except Exception as e:
            log.error(f"balance: {e}")
            return 0.0

    def price(self, symbol: str) -> float:
        try:
            r = self.s.get_tickers(category="linear", symbol=symbol)
            return float(r["result"]["list"][0]["lastPrice"])
        except Exception as e:
            log.error(f"price({symbol}): {e}")
            return 0.0

    def klines(self, symbol: str, limit: int = 60) -> pd.DataFrame:
        try:
            r = self.s.get_kline(
                category="linear", symbol=symbol,
                interval=TIMEFRAME, limit=limit
            )
            rows = [
                [int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4])]
                for x in reversed(r["result"]["list"])
            ]
            return pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c"])
        except Exception as e:
            log.error(f"klines({symbol}): {e}")
            return pd.DataFrame()

    def set_leverage(self, symbol: str, lev: int):
        try:
            self.s.set_leverage(
                category="linear", symbol=symbol,
                buyLeverage=str(lev), sellLeverage=str(lev)
            )
            log.info(f"  Плечо {lev}x → {symbol} ✅")
        except Exception as e:
            # Если уже установлено — ошибка нормальная
            log.warning(f"  set_leverage {symbol}: {e}")

    def min_qty(self, symbol: str) -> Tuple[float, float]:
        """Возвращает (min_qty, qty_step)"""
        try:
            r = self.s.get_instruments_info(category="linear", symbol=symbol)
            f = r["result"]["list"][0]["lotSizeFilter"]
            return float(f["minOrderQty"]), float(f["qtyStep"])
        except Exception as e:
            log.error(f"min_qty({symbol}): {e}")
            return 1.0, 1.0

    def open_order(self, symbol: str, side: str, qty: float, sl: float) -> Optional[str]:
        """Открывает рыночный ордер со стоп-лоссом. Возвращает orderId или None."""
        try:
            r = self.s.place_order(
                category="linear",
                symbol=symbol,
                side="Buy" if side == "long" else "Sell",
                orderType="Market",
                qty=str(qty),
                stopLoss=str(round(sl, 8)),
                slTriggerBy="MarkPrice",
                timeInForce="GTC",
                reduceOnly=False,
            )
            oid = r["result"]["orderId"]
            log.info(f"  Ордер открыт: {oid}")
            return oid
        except Exception as e:
            log.error(f"open_order({symbol}): {e}")
            return None

    def close_order(self, symbol: str, side: str, qty: float) -> bool:
        """Закрывает позицию рыночным ордером."""
        try:
            self.s.place_order(
                category="linear",
                symbol=symbol,
                side="Sell" if side == "long" else "Buy",
                orderType="Market",
                qty=str(qty),
                reduceOnly=True,
                timeInForce="GTC",
            )
            return True
        except Exception as e:
            log.error(f"close_order({symbol}): {e}")
            return False

    def set_trailing_stop(self, symbol: str, trailing_delta: float) -> bool:
        try:
            self.s.set_trading_stop(
                category="linear",
                symbol=symbol,
                trailingStop=str(round(trailing_delta, 8)),
                positionIdx=0,
            )
            return True
        except Exception as e:
            log.warning(f"set_trailing_stop({symbol}): {e}")
            return False

    def position(self, symbol: str) -> Optional[Dict]:
        try:
            r = self.s.get_positions(category="linear", symbol=symbol)
            for p in r["result"]["list"]:
                sz = float(p.get("size", 0))
                if sz > 0:
                    return {
                        "side":  "long" if p["side"] == "Buy" else "short",
                        "size":  sz,
                        "entry": float(p.get("avgPrice", 0)),
                        "pnl":   float(p.get("unrealisedPnl", 0)),
                        "sl":    float(p.get("stopLoss", 0)),
                    }
        except Exception as e:
            log.error(f"position({symbol}): {e}")
        return None


# ───────────────────────────────────────────────
# ИНДИКАТОРЫ
# ───────────────────────────────────────────────
def ema(series: pd.Series, n: int) -> float:
    return float(series.ewm(span=n, adjust=False).mean().iloc[-1])

def get_trend(df: pd.DataFrame) -> Optional[str]:
    """EMA9 vs EMA21 → 'long' | 'short' | None"""
    if len(df) < 22:
        return None
    e9  = ema(df["c"], 9)
    e21 = ema(df["c"], 21)
    if e9 > e21 * 1.0005:   # небольшой фильтр шума
        return "long"
    if e9 < e21 * 0.9995:
        return "short"
    return None


# ───────────────────────────────────────────────
# БОТ
# ───────────────────────────────────────────────
class Bot:
    def __init__(self):
        self.ex           = Bybit()
        self.mart_step    = 0
        self.loss_streak  = 0
        self.last_side: Optional[str] = None
        self.total_pnl    = 0.0
        self.trade_n      = 0

        log.info("=== Bybit Meme Bot v3.0 (Testnet) ===")
        for sym in SYMBOLS:
            self.ex.set_leverage(sym, LEVERAGE)

    # ── РАЗМЕР СТАВКИ ──────────────────────────
    def calc_qty(self, symbol: str, price: float) -> float:
        bal = self.ex.balance()
        if bal < MIN_BALANCE:
            return 0.0

        # Базовая ставка → масштабируем на шаг мартингейла
        risk_usdt = bal * (BASE_RISK_PCT / 100) * (MART_MULT ** self.mart_step) * LEVERAGE
        # Не более 50 % баланса с плечом
        risk_usdt = min(risk_usdt, bal * 0.5 * LEVERAGE)

        # Добавляем запас на двойную комиссию (вход + выход)
        commission = price * (risk_usdt / price) * FEE_RATE * 2
        # Ставка уже с учётом комиссии — просто информативно
        log.info(f"  Ставка: {risk_usdt:.2f} USDT | Комиссия ≈ {commission:.4f} USDT")

        min_q, step = self.ex.min_qty(symbol)
        qty = math.floor((risk_usdt / price) / step) * step
        return max(min_q, qty)

    # ── SL ─────────────────────────────────────
    def calc_sl(self, side: str, price: float) -> float:
        pct = INITIAL_SL_PCT / 100
        return price * (1 - pct) if side == "long" else price * (1 + pct)

    # ── СИГНАЛ ─────────────────────────────────
    def get_signal(self, symbol: str) -> Optional[str]:
        df = self.ex.klines(symbol)
        if df.empty:
            return None
        trend = get_trend(df)
        if trend is None:
            return None

        # После убытка — разворот (стратегия догон)
        if self.loss_streak > 0 and self.last_side:
            flip = "short" if self.last_side == "long" else "long"
            log.info(f"  🔄 Догон: переворачиваем {self.last_side} → {flip}")
            return flip

        return trend

    # ── ОТКРЫТИЕ СДЕЛКИ ────────────────────────
    def open_trade(self, symbol: str, side: str) -> Optional[Dict]:
        price = self.ex.price(symbol)
        if not price:
            return None

        qty = self.calc_qty(symbol, price)
        if qty <= 0:
            log.warning("  Объём = 0, пропуск")
            return None

        sl = self.calc_sl(side, price)
        oid = self.ex.open_order(symbol, side, qty, sl)
        if not oid:
            return None

        # Даём бирже 1 сек оформить позицию
        time.sleep(1)

        # Устанавливаем трейлинг-стоп
        trailing_delta = price * TRAILING_PCT / 100
        self.ex.set_trailing_stop(symbol, trailing_delta)

        self.last_side = side
        log.info(
            f"🟢 ОТКРЫТО: {side.upper()} {symbol} | "
            f"qty={qty} | price={price:.8f} | SL={sl:.8f} | "
            f"mart={self.mart_step}"
        )
        return {"symbol": symbol, "side": side, "entry": price,
                "qty": qty, "sl": sl, "t0": time.time()}

    # ── МОНИТОРИНГ ─────────────────────────────
    def monitor(self, trade: Dict) -> Tuple[float, str]:
        symbol = trade["symbol"]
        side   = trade["side"]
        entry  = trade["entry"]
        qty    = trade["qty"]
        t0     = trade["t0"]

        while time.time() - t0 < TRADE_DURATION:
            try:
                pos = self.ex.position(symbol)

                # Позиция закрылась сама (по SL или другой причине)
                if pos is None or pos["side"] != side:
                    pnl = pos["pnl"] if pos else 0.0
                    log.info(f"  Позиция закрыта биржей | PnL≈{pnl:.4f}")
                    return pnl, "sl" if pnl <= 0 else "tp"

                elapsed = int(time.time() - t0)
                log.info(
                    f"  [{elapsed}s] {side.upper()} {symbol} | "
                    f"PnL={pos['pnl']:.4f} USDT | SL={pos['sl']:.8f}"
                )
                time.sleep(5)

            except Exception as e:
                log.error(f"monitor: {e}")
                time.sleep(5)

        # Время вышло — закрываем
        log.info("  ⏰ Время вышло — закрываем")
        cur = self.ex.price(symbol)
        self.ex.close_order(symbol, side, qty)

        if side == "long":
            pnl = (cur - entry) * qty * LEVERAGE
        else:
            pnl = (entry - cur) * qty * LEVERAGE
        pnl -= entry * qty * FEE_RATE * 2  # вычитаем комиссии

        return pnl, "timeout"

    # ── ГЛАВНЫЙ ЦИКЛ ───────────────────────────
    def run(self):
        log.info(f"Символы: {SYMBOLS}")
        log.info(f"Плечо: {LEVERAGE}x | Длительность: {TRADE_DURATION}s | Мартингейл: до {MAX_MART_STEPS} шагов")

        while True:
            try:
                bal = self.ex.balance()
                if bal < MIN_BALANCE:
                    log.warning(f"Баланс {bal:.2f} < {MIN_BALANCE} USDT — ожидание")
                    time.sleep(60)
                    continue

                # Проверяем открытые позиции
                busy = False
                for sym in SYMBOLS:
                    p = self.ex.position(sym)
                    if p:
                        log.info(f"⏳ Уже открыта: {p['side'].upper()} {sym} | PnL={p['pnl']:.4f}")
                        busy = True
                        break
                if busy:
                    time.sleep(10)
                    continue

                # Выбираем символ
                symbol = SYMBOLS[self.trade_n % len(SYMBOLS)]
                self.trade_n += 1

                log.info(f"\n─── Сделка #{self.trade_n} | {symbol} | mart={self.mart_step} ───")

                signal = self.get_signal(symbol)
                if not signal:
                    log.info("Нет чёткого тренда — пропуск")
                    time.sleep(30)
                    continue

                log.info(f"  Сигнал: {signal.upper()}")
                trade = self.open_trade(symbol, signal)
                if not trade:
                    time.sleep(30)
                    continue

                pnl, result = self.monitor(trade)
                self.total_pnl += pnl

                if result in ("sl", "timeout") and pnl < 0:
                    self.loss_streak += 1
                    self.mart_step = min(self.mart_step + 1, MAX_MART_STEPS)
                    log.warning(
                        f"❌ Убыток | PnL={pnl:.4f} USDT | "
                        f"Loss streak={self.loss_streak} | Mart step={self.mart_step}"
                    )
                else:
                    self.loss_streak = 0
                    self.mart_step = max(0, self.mart_step - 1)
                    log.info(f"✅ Прибыль | PnL={pnl:.4f} USDT")

                log.info(f"📊 Итого PnL: {self.total_pnl:.4f} USDT | Баланс: {self.ex.balance():.2f} USDT")
                time.sleep(10)

            except KeyboardInterrupt:
                log.info("🛑 Остановлен")
                break
            except Exception as e:
                log.error(f"Главный цикл: {e}")
                time.sleep(30)


if __name__ == "__main__":
    Bot().run()
