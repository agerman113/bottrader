#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bybit Мем-Коин Бот v3.1 — Testnet
- Тренд с Bybit (EMA 9/21 на 3m свечах)
- Мартингейл до 10 шагов (x1.5 за шаг) с переворотом после убытка
- При серии убытков ≥ 6 → переход в recovery-режим:
    * сделки только по тренду, TP в безубыток (0.15% от входа)
    * максимальное время удержания 15 минут
    * возврат к мартингейлу после выхода общего PnL в ноль/плюс
- Трейлинг-стоп (в обычном режиме), комиссии, плечо 5x
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
SYMBOLS = ["1000PEPEUSDT", "WIFUSDT", "1000BONKUSDT"]

LEVERAGE         = 5
TRADE_DURATION   = 180      # секунд на сделку (обычный режим)
MAX_MART_STEPS   = 10
BASE_RISK_PCT    = 2.0
MART_MULT        = 1.5
FEE_RATE         = 0.00055
MIN_BALANCE      = 5.0
INITIAL_SL_PCT   = 2.0
TRAILING_PCT     = 1.0
TIMEFRAME        = "3"

# Параметры recovery-режима
RECOVERY_TRIGGER    = 6       # количество убыточных сделок подряд для переключения
RECOVERY_MAX_DURATION = 900   # 15 минут
RECOVERY_TP_PCT     = 0.15    # тейк-профит в % от цены входа (0.15% покрывает комиссии)

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
# BYBIT WRAPPER (расширен)
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
            log.warning(f"  set_leverage {symbol}: {e}")

    def min_qty(self, symbol: str) -> Tuple[float, float]:
        try:
            r = self.s.get_instruments_info(category="linear", symbol=symbol)
            f = r["result"]["list"][0]["lotSizeFilter"]
            return float(f["minOrderQty"]), float(f["qtyStep"])
        except Exception as e:
            log.error(f"min_qty({symbol}): {e}")
            return 1.0, 1.0

    def open_order(self, symbol: str, side: str, qty: float,
                   sl: float, tp: Optional[float] = None) -> Optional[str]:
        """
        Открывает рыночный ордер со стоп-лоссом и опциональным тейк-профитом.
        Возвращает orderId или None.
        """
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": "Buy" if side == "long" else "Sell",
            "orderType": "Market",
            "qty": str(qty),
            "stopLoss": str(round(sl, 8)),
            "slTriggerBy": "MarkPrice",
            "timeInForce": "GTC",
            "reduceOnly": False,
        }
        if tp is not None:
            params["takeProfit"] = str(round(tp, 8))
            params["tpTriggerBy"] = "MarkPrice"

        try:
            r = self.s.place_order(**params)
            oid = r["result"]["orderId"]
            log.info(f"  Ордер открыт: {oid}")
            return oid
        except Exception as e:
            log.error(f"open_order({symbol}): {e}")
            return None

    def close_order(self, symbol: str, side: str, qty: float) -> bool:
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
                        "tp":    float(p.get("takeProfit", 0)),
                    }
        except Exception as e:
            log.error(f"position({symbol}): {e}")
        return None

    def cancel_all_orders(self, symbol: str):
        """Отмена всех активных ордеров (TP/SL)"""
        try:
            self.s.cancel_all_orders(category="linear", symbol=symbol)
        except Exception as e:
            log.warning(f"cancel_all_orders({symbol}): {e}")


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
    if e9 > e21 * 1.0005:
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
        self.recovery_mode = False   # флаг восстановительного режима

        log.info("=== Bybit Meme Bot v3.1 (Testnet) ===")
        for sym in SYMBOLS:
            self.ex.set_leverage(sym, LEVERAGE)

    # ── РАСЧЁТ ОБЪЁМА ──────────────────────────
    def calc_qty(self, symbol: str, price: float, use_martingale: bool = True) -> float:
        """Рассчитывает объём позиции с учётом мартингейла или без него"""
        bal = self.ex.balance()
        if bal < MIN_BALANCE:
            return 0.0

        if use_martingale:
            risk_usdt = bal * (BASE_RISK_PCT / 100) * (MART_MULT ** self.mart_step) * LEVERAGE
        else:
            risk_usdt = bal * (BASE_RISK_PCT / 100) * LEVERAGE   # базовый риск

        risk_usdt = min(risk_usdt, bal * 0.5 * LEVERAGE)

        min_q, step = self.ex.min_qty(symbol)
        qty = math.floor((risk_usdt / price) / step) * step
        return max(min_q, qty)

    # ── SL / TP ДЛЯ ОБЫЧНОГО РЕЖИМА ──────────────
    def calc_sl(self, side: str, price: float) -> float:
        pct = INITIAL_SL_PCT / 100
        return price * (1 - pct) if side == "long" else price * (1 + pct)

    # ── TP ДЛЯ RECOVERY (безубыток) ────────────
    def calc_recovery_tp(self, side: str, price: float) -> float:
        """TP с небольшим запасом, чтобы перекрыть комиссии и получить нулевой/слабоположительный результат"""
        pct = RECOVERY_TP_PCT / 100
        return price * (1 + pct) if side == "long" else price * (1 - pct)

    # ── СИГНАЛ ─────────────────────────────────
    def get_signal(self, symbol: str) -> Optional[str]:
        df = self.ex.klines(symbol)
        if df.empty:
            return None
        trend = get_trend(df)
        if trend is None:
            return None

        # В обычном режиме: переворот после убытка (догон)
        if not self.recovery_mode and self.loss_streak > 0 and self.last_side:
            flip = "short" if self.last_side == "long" else "long"
            log.info(f"  🔄 Догон: переворачиваем {self.last_side} → {flip}")
            return flip

        # В recovery или при отсутствии догона — просто тренд
        return trend

    # ── ОТКРЫТИЕ СДЕЛКИ (ОБЫЧНЫЙ РЕЖИМ) ────────
    def open_trade(self, symbol: str, side: str) -> Optional[Dict]:
        price = self.ex.price(symbol)
        if not price:
            return None

        qty = self.calc_qty(symbol, price, use_martingale=True)
        if qty <= 0:
            log.warning("  Объём = 0, пропуск")
            return None

        sl = self.calc_sl(side, price)
        oid = self.ex.open_order(symbol, side, qty, sl)
        if not oid:
            return None

        time.sleep(1)
        # Трейлинг-стоп (только в обычном режиме)
        trailing_delta = price * TRAILING_PCT / 100
        self.ex.set_trailing_stop(symbol, trailing_delta)

        self.last_side = side
        log.info(
            f"🟢 ОТКРЫТО (обычный): {side.upper()} {symbol} | "
            f"qty={qty} | price={price:.8f} | SL={sl:.8f} | "
            f"mart={self.mart_step}"
        )
        return {"symbol": symbol, "side": side, "entry": price,
                "qty": qty, "sl": sl, "t0": time.time()}

    # ── ОТКРЫТИЕ СДЕЛКИ В RECOVERY РЕЖИМЕ ───────
    def open_recovery_trade(self, symbol: str, side: str) -> Optional[Dict]:
        price = self.ex.price(symbol)
        if not price:
            return None

        qty = self.calc_qty(symbol, price, use_martingale=False)
        if qty <= 0:
            log.warning("  Recovery: объём = 0, пропуск")
            return None

        sl = self.calc_sl(side, price)
        tp = self.calc_recovery_tp(side, price)
        oid = self.ex.open_order(symbol, side, qty, sl, tp)
        if not oid:
            return None

        log.info(
            f"🔄 RECOVERY: открыта {side.upper()} {symbol} | "
            f"qty={qty} | price={price:.8f} | SL={sl:.8f} | TP={tp:.8f}"
        )
        return {"symbol": symbol, "side": side, "entry": price,
                "qty": qty, "sl": sl, "tp": tp, "t0": time.time()}

    # ── МОНИТОРИНГ ОБЫЧНОЙ СДЕЛКИ ──────────────
    def monitor(self, trade: Dict) -> Tuple[float, str]:
        symbol = trade["symbol"]
        side   = trade["side"]
        entry  = trade["entry"]
        qty    = trade["qty"]
        t0     = trade["t0"]

        while time.time() - t0 < TRADE_DURATION:
            try:
                pos = self.ex.position(symbol)
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

        # Таймаут — закрываем
        log.info("  ⏰ Время вышло — закрываем")
        cur = self.ex.price(symbol)
        self.ex.close_order(symbol, side, qty)

        if side == "long":
            pnl = (cur - entry) * qty * LEVERAGE
        else:
            pnl = (entry - cur) * qty * LEVERAGE
        pnl -= entry * qty * FEE_RATE * 2
        return pnl, "timeout"

    # ── МОНИТОРИНГ RECOVERY СДЕЛКИ (до 15 мин) ──
    def monitor_recovery(self, trade: Dict) -> Tuple[float, str]:
        symbol = trade["symbol"]
        side   = trade["side"]
        entry  = trade["entry"]
        qty    = trade["qty"]
        t0     = trade["t0"]
        tp_price = trade["tp"]

        log.info(f"  Recovery мониторинг: ждём TP={tp_price:.8f} или таймаут {RECOVERY_MAX_DURATION//60} мин")

        while time.time() - t0 < RECOVERY_MAX_DURATION:
            try:
                pos = self.ex.position(symbol)
                # Позиция закрыта (сработал TP или SL)
                if pos is None or pos["side"] != side:
                    pnl = pos["pnl"] if pos else 0.0
                    log.info(f"  Recovery: позиция закрыта | PnL={pnl:.4f}")
                    return pnl, "tp_closed" if pnl > 0 else "sl_closed"

                elapsed = int(time.time() - t0)
                cur_price = self.ex.price(symbol)
                log.info(
                    f"  [recovery {elapsed}s] {side.upper()} {symbol} | "
                    f"цена={cur_price:.8f} | TP={tp_price:.8f} | PnL={pos['pnl']:.4f}"
                )
                time.sleep(5)
            except Exception as e:
                log.error(f"monitor_recovery: {e}")
                time.sleep(5)

        # Таймаут — принудительное закрытие
        log.info("  ⏰ Recovery: время истекло, закрываем по рынку")
        cur = self.ex.price(symbol)
        self.ex.cancel_all_orders(symbol)          # убираем TP/SL ордера
        self.ex.close_order(symbol, side, qty)

        if side == "long":
            pnl = (cur - entry) * qty * LEVERAGE
        else:
            pnl = (entry - cur) * qty * LEVERAGE
        pnl -= entry * qty * FEE_RATE * 2
        return pnl, "timeout"

    # ── ВХОД В RECOVERY РЕЖИМ ──────────────────
    def enter_recovery_mode(self):
        log.warning(f"⚠️  Активирован RECOVERY РЕЖИМ (серия убытков = {self.loss_streak})")
        self.recovery_mode = True
        # Сбрасываем счётчики мартингейла, они не нужны в recovery
        self.mart_step = 0
        self.loss_streak = 0
        self.last_side = None

    # ── ВЫХОД ИЗ RECOVERY РЕЖИМА ───────────────
    def exit_recovery_mode(self):
        log.info("✅ Выход из RECOVERY режима — общий PnL стал неотрицательным, возвращаемся к мартингейлу")
        self.recovery_mode = False
        self.mart_step = 0
        self.loss_streak = 0
        self.last_side = None

    # ── ЦИКЛ RECOVERY (повторяем, пока PnL не ≥0) ─
    def run_recovery_cycle(self):
        """Выполняет сделки в recovery-режиме до восстановления общего PnL >= 0"""
        log.info("🔄 Запуск recovery-цикла, цель: вывести total_pnl в ноль или плюс")
        while self.recovery_mode:
            try:
                bal = self.ex.balance()
                if bal < MIN_BALANCE:
                    log.warning(f"Баланс {bal:.2f} < {MIN_BALANCE} USDT — ожидание")
                    time.sleep(60)
                    continue

                # Проверяем открытые позиции
                busy = False
                for sym in SYMBOLS:
                    if self.ex.position(sym):
                        log.info("⏳ Recovery: уже есть открытая позиция, ждём её закрытия")
                        busy = True
                        break
                if busy:
                    time.sleep(10)
                    continue

                # Выбираем символ по кругу
                symbol = SYMBOLS[self.trade_n % len(SYMBOLS)]
                self.trade_n += 1

                log.info(f"\n─── Recovery сделка #{self.trade_n} | {symbol} ───")
                signal = self.get_signal(symbol)   # в recovery переворот не работает, только тренд
                if not signal:
                    log.info("Recovery: нет чёткого тренда — пропуск")
                    time.sleep(30)
                    continue

                log.info(f"  Recovery сигнал: {signal.upper()}")
                trade = self.open_recovery_trade(symbol, signal)
                if not trade:
                    time.sleep(30)
                    continue

                pnl, _ = self.monitor_recovery(trade)
                self.total_pnl += pnl
                log.info(f"Recovery сделка завершена | PnL={pnl:.4f} | Общий PnL={self.total_pnl:.4f}")

                # Проверяем, вышли ли в плюс/ноль
                if self.total_pnl >= 0:
                    self.exit_recovery_mode()
                    break
                else:
                    log.info(f"Общий PnL всё ещё отрицательный ({self.total_pnl:.4f}), продолжаем recovery...")
                    time.sleep(10)

            except KeyboardInterrupt:
                log.info("🛑 Остановлен во время recovery")
                raise
            except Exception as e:
                log.error(f"Ошибка в recovery-цикле: {e}")
                time.sleep(30)

    # ── ГЛАВНЫЙ ЦИКЛ ───────────────────────────
    def run(self):
        log.info(f"Символы: {SYMBOLS}")
        log.info(f"Плечо: {LEVERAGE}x | Обычный таймаут: {TRADE_DURATION}s | Мартингейл: до {MAX_MART_STEPS} шагов")
        log.info(f"Recovery триггер: {RECOVERY_TRIGGER} убытков подряд | TP {RECOVERY_TP_PCT}% | макс. время {RECOVERY_MAX_DURATION//60} мин")

        while True:
            try:
                # Если мы в recovery-режиме — выполняем recovery-цикл
                if self.recovery_mode:
                    self.run_recovery_cycle()
                    # После выхода из recovery продолжаем обычную работу
                    continue

                # --- Обычный режим (мартингейл + трейлинг) ---
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

                # Обновляем счётчики мартингейла и убытков
                if result in ("sl", "timeout") and pnl < 0:
                    self.loss_streak += 1
                    self.mart_step = min(self.mart_step + 1, MAX_MART_STEPS)
                    log.warning(
                        f"❌ Убыток | PnL={pnl:.4f} USDT | "
                        f"Loss streak={self.loss_streak} | Mart step={self.mart_step}"
                    )
                    # Проверяем, не пора ли перейти в recovery
                    if self.loss_streak >= RECOVERY_TRIGGER:
                        self.enter_recovery_mode()
                        continue   # сразу уходим в recovery на следующей итерации
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
