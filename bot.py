#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bybit Мем-Коин Бот v2.0
Торгует 3 мем-коинами на фьючерсах (Testnet) с:
- Мартингейлом (макс. 10 шагов)
- Переворотом позиции после убытка
- Трейлинг-стопом на базе ATR
- Пирамидингом
- Стратегией догон
- Учетом комиссий
- Лимитом риска 50% от депозита
- Временным лимитом сделки (3 минуты)
"""

import os
import time
import logging
import math
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

load_dotenv()

# ============================================================
# 📌 КОНФИГУРАЦИЯ
# ============================================================
TESTNET_MODE       = True
SYMBOLS            = ["PEPEUSDT", "WIFUSDT", "BONKUSDT"]   # Bybit linear symbols
LEVERAGE           = 5          # Плечо (риск до 50 % от депозита)
TIMEFRAME          = "3"        # Таймфрейм в минутах (Bybit interval)
TRADE_DURATION     = 180        # Максимальная длительность сделки (сек)
MAX_MARTINGALE_STEPS = 10       # Максимум шагов мартингейла
BASE_RISK_PCT      = 1.0        # Базовый риск на сделку (% от баланса)
BYBIT_FEE          = 0.00055    # Комиссия Bybit (тейкер)
MIN_BALANCE_USDT   = 10.0       # Минимальный баланс для торговли

# Параметры стопов
INITIAL_SL_PCT     = 1.5        # Начальный стоп-лосс (%)
TRAILING_ATR_MULT  = 2.0        # Множитель ATR для трейлинга
MIN_TRAILING_PCT   = 0.3        # Минимальный шаг трейлинга (%)

# Параметры пирамидинга / мартингейла
PYRAMIDING_MULT    = 1.5        # Множитель размера позиции за шаг

# ============================================================
# 📋 ЛОГИРОВАНИЕ
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("meme_bot.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ============================================================
# 🤖 BYBIT API WRAPPER
# ============================================================
class BybitWrapper:
    def __init__(self):
        self.session = HTTP(
            testnet=TESTNET_MODE,
            api_key=os.getenv("BYBIT_TESTNET_API_KEY", ""),
            api_secret=os.getenv("BYBIT_TESTNET_API_SECRET", ""),
        )

    # ----------------------------------------------------------
    def fetch_ticker(self, symbol: str) -> Dict:
        try:
            r = self.session.get_tickers(category="linear", symbol=symbol)
            item = r["result"]["list"][0]
            return {
                "last": float(item["lastPrice"]),
                "mark_price": float(item.get("markPrice", item["lastPrice"])),
            }
        except Exception as e:
            log.error(f"fetch_ticker error: {e}")
            return {"last": 0.0, "mark_price": 0.0}

    # ----------------------------------------------------------
    def fetch_ohlcv(self, symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
        try:
            r = self.session.get_kline(
                category="linear", symbol=symbol, interval=interval, limit=limit
            )
            rows = [
                [int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])]
                for x in reversed(r["result"]["list"])
            ]
            df = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c", "v"])
            return df
        except Exception as e:
            log.error(f"fetch_ohlcv error: {e}")
            return pd.DataFrame()

    # ----------------------------------------------------------
    def fetch_balance(self) -> float:
        try:
            r = self.session.get_wallet_balance(accountType="UNIFIED")
            return float(r["result"]["list"][0].get("totalAvailableBalance", 0))
        except Exception as e:
            log.error(f"fetch_balance error: {e}")
            return 0.0

    # ----------------------------------------------------------
    def set_leverage(self, symbol: str, leverage: int) -> bool:
        try:
            self.session.set_leverage(
                category="linear",
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage),
            )
            return True
        except Exception as e:
            log.error(f"set_leverage error [{symbol}]: {e}")
            return False

    # ----------------------------------------------------------
    def get_instrument_info(self, symbol: str) -> Dict:
        """Возвращает параметры инструмента (шаг цены, мин. объём и т.д.)"""
        try:
            r = self.session.get_instruments_info(category="linear", symbol=symbol)
            info = r["result"]["list"][0]
            lot = info["lotSizeFilter"]
            price = info["priceFilter"]
            return {
                "qty_step": float(lot.get("qtyStep", 1)),
                "min_qty":  float(lot.get("minOrderQty", 1)),
                "price_tick": float(price.get("tickSize", 0.0001)),
            }
        except Exception as e:
            log.error(f"get_instrument_info error: {e}")
            return {"qty_step": 1.0, "min_qty": 1.0, "price_tick": 0.0001}

    # ----------------------------------------------------------
    def create_market_order(
        self,
        symbol: str,
        side: str,          # "long" | "short"
        qty: float,
        sl_price: Optional[float] = None,
        reduce_only: bool = False,
    ) -> Optional[Dict]:
        order_side = "Buy" if side == "long" else "Sell"
        params: Dict = {
            "category": "linear",
            "symbol": symbol,
            "side": order_side,
            "orderType": "Market",
            "qty": str(qty),
            "timeInForce": "GTC",
            "reduceOnly": reduce_only,
        }
        if sl_price and not reduce_only:
            params["stopLoss"] = str(round(sl_price, 8))
        try:
            r = self.session.place_order(**params)
            result = r["result"]
            avg = float(result.get("avgPrice", 0) or 0)
            return {"order_id": result["orderId"], "avg_price": avg}
        except Exception as e:
            log.error(f"create_market_order error: {e}")
            return None

    # ----------------------------------------------------------
    def set_trailing_stop(self, symbol: str, side: str, trailing_delta: float) -> bool:
        """Устанавливает трейлинг-стоп через Trading Stop API Bybit."""
        try:
            self.session.set_trading_stop(
                category="linear",
                symbol=symbol,
                trailingStop=str(round(trailing_delta, 8)),
                positionIdx=0,
            )
            return True
        except Exception as e:
            log.error(f"set_trailing_stop error: {e}")
            return False

    # ----------------------------------------------------------
    def close_position(self, symbol: str, side: str) -> bool:
        close_side = "Sell" if side == "long" else "Buy"
        try:
            positions = self.session.get_positions(category="linear", symbol=symbol)
            for p in positions["result"]["list"]:
                size = float(p.get("size", 0))
                pos_side = p.get("side", "")
                if size > 0 and pos_side == ("Buy" if side == "long" else "Sell"):
                    self.session.place_order(
                        category="linear",
                        symbol=symbol,
                        side=close_side,
                        orderType="Market",
                        qty=str(size),
                        reduceOnly=True,
                    )
                    return True
        except Exception as e:
            log.error(f"close_position error: {e}")
        return False

    # ----------------------------------------------------------
    def get_position(self, symbol: str) -> Optional[Dict]:
        try:
            r = self.session.get_positions(category="linear", symbol=symbol)
            for p in r["result"]["list"]:
                size = float(p.get("size", 0))
                if size > 0:
                    return {
                        "symbol": p["symbol"],
                        "side": "long" if p["side"] == "Buy" else "short",
                        "size": size,
                        "entry_price": float(p.get("avgPrice", 0)),
                        "pnl": float(p.get("unrealisedPnl", 0)),
                    }
        except Exception as e:
            log.error(f"get_position error: {e}")
        return None


# ============================================================
# 📊 ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ
# ============================================================
def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period + 1:
        return 0.0
    hi, lo, pc = df["h"], df["l"], df["c"].shift(1)
    tr = pd.concat([hi - lo, (hi - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    return float(atr)


def calc_ema(series: pd.Series, span: int) -> float:
    return float(series.ewm(span=span, adjust=False).mean().iloc[-1])


# ============================================================
# 🎯 ОСНОВНОЙ БОТ
# ============================================================
class MemeCoinBot:
    def __init__(self):
        self.exchange = BybitWrapper()
        self.martingale_step = 0
        self.trade_count = 0
        self.win_streak = 0
        self.loss_streak = 0
        self.total_pnl = 0.0
        self.last_signal: Optional[str] = None   # последний сигнал (для догона)

        # Кэш параметров инструментов
        self._instrument_cache: Dict[str, Dict] = {}

        # Устанавливаем плечо
        for symbol in SYMBOLS:
            ok = self.exchange.set_leverage(symbol, LEVERAGE)
            log.info(f"Плечо {LEVERAGE}x для {symbol}: {'✅' if ok else '⚠️ ошибка'}")

    # ----------------------------------------------------------
    def _instrument(self, symbol: str) -> Dict:
        if symbol not in self._instrument_cache:
            self._instrument_cache[symbol] = self.exchange.get_instrument_info(symbol)
        return self._instrument_cache[symbol]

    # ----------------------------------------------------------
    def get_balance(self) -> float:
        return self.exchange.fetch_balance()

    # ----------------------------------------------------------
    def calculate_position_size(self, symbol: str) -> float:
        """Размер позиции с учётом мартингейла и лимита риска 50 % от депозита."""
        balance = self.get_balance()
        if balance < MIN_BALANCE_USDT:
            return 0.0

        # Масштабируем ставку по шагу мартингейла
        risk_pct = BASE_RISK_PCT * (PYRAMIDING_MULT ** self.martingale_step)
        # Не более 50 % баланса с плечом
        max_risk_usdt = balance * 0.50
        bet_usdt = min(balance * risk_pct / 100 * LEVERAGE, max_risk_usdt)

        ticker = self.exchange.fetch_ticker(symbol)
        price = ticker["last"]
        if price == 0:
            return 0.0

        info = self._instrument(symbol)
        qty_step = info["qty_step"]
        min_qty  = info["min_qty"]

        qty = math.floor((bet_usdt / price) / qty_step) * qty_step
        return max(min_qty, qty)

    # ----------------------------------------------------------
    def get_trend_signal(self, symbol: str) -> Optional[str]:
        """EMA 9 vs EMA 21 на 3m свечах."""
        df = self.exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=50)
        if df.empty or len(df) < 22:
            return None
        ema9  = calc_ema(df["c"], 9)
        ema21 = calc_ema(df["c"], 21)
        if ema9 > ema21:
            return "long"
        if ema9 < ema21:
            return "short"
        return None

    # ----------------------------------------------------------
    def get_entry_signal(self, symbol: str) -> Optional[str]:
        """
        Сигнал входа:
        - При убытке: переворачиваем последний сигнал (стратегия догон).
        - Иначе: следуем тренду.
        """
        trend = self.get_trend_signal(symbol)
        if trend is None:
            return None

        if self.loss_streak > 0 and self.last_signal:
            # Разворот против предыдущего направления
            return "short" if self.last_signal == "long" else "long"

        return trend

    # ----------------------------------------------------------
    def calculate_sl(self, symbol: str, side: str, entry_price: float) -> float:
        df = self.exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=50)
        if df.empty or len(df) < 15:
            pct = INITIAL_SL_PCT / 100
            return entry_price * (1 - pct) if side == "long" else entry_price * (1 + pct)

        atr = calc_atr(df)
        atr_pct = (atr / entry_price) * 100
        sl_pct = max(INITIAL_SL_PCT, atr_pct * TRAILING_ATR_MULT)

        if side == "long":
            return entry_price * (1 - sl_pct / 100)
        else:
            return entry_price * (1 + sl_pct / 100)

    # ----------------------------------------------------------
    def update_trailing_stop(
        self, symbol: str, side: str, current_sl: float, current_price: float
    ) -> float:
        """Возвращает новый уровень SL (или тот же, если сдвиг не нужен)."""
        df = self.exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=50)
        if df.empty or len(df) < 15:
            return current_sl

        atr = calc_atr(df)
        atr_pct = (atr / current_price) * 100
        trailing_pct = max(MIN_TRAILING_PCT, atr_pct * 0.5)

        if side == "long":
            candidate = current_price * (1 - trailing_pct / 100)
            if candidate > current_sl:
                return candidate
        else:
            candidate = current_price * (1 + trailing_pct / 100)
            if candidate < current_sl:
                return candidate

        return current_sl

    # ----------------------------------------------------------
    def _commission_usdt(self, qty: float, price: float) -> float:
        """Суммарная комиссия за открытие + закрытие."""
        return price * qty * BYBIT_FEE * 2

    # ----------------------------------------------------------
    def open_trade(self, symbol: str, side: str) -> Optional[Dict]:
        balance = self.get_balance()
        if balance < MIN_BALANCE_USDT:
            log.warning(f"❌ Недостаточно средств: {balance:.2f} USDT")
            return None

        qty = self.calculate_position_size(symbol)
        if qty <= 0:
            log.warning(f"❌ Нулевой объём позиции для {symbol}")
            return None

        ticker = self.exchange.fetch_ticker(symbol)
        current_price = ticker["last"]
        if current_price == 0:
            return None

        # Проверяем, что минимальная прибыль перекрывает комиссию
        commission = self._commission_usdt(qty, current_price)
        min_profit_move_pct = (commission / (qty * current_price)) * 100 * 1.5  # буфер 1.5x
        log.info(
            f"💸 Комиссия: {commission:.4f} USDT | "
            f"Минимальный ход для покрытия: {min_profit_move_pct:.3f}%"
        )

        sl_price = self.calculate_sl(symbol, side, current_price)
        order = self.exchange.create_market_order(symbol, side, qty, sl_price=sl_price)
        if not order:
            return None

        entry_price = order["avg_price"] or current_price
        log.info(
            f"🔹 Открыта позиция: {side.upper()} {symbol} | "
            f"Qty={qty} | Entry={entry_price:.8f} | SL={sl_price:.8f} | "
            f"Мартингейл шаг={self.martingale_step}"
        )
        self.last_signal = side
        return {
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "sl_price": sl_price,
            "qty": qty,
            "open_time": time.time(),
            "order_id": order["order_id"],
        }

    # ----------------------------------------------------------
    def close_trade(
        self, symbol: str, side: str, entry_price: float, qty: float
    ) -> Tuple[float, str]:
        ticker = self.exchange.fetch_ticker(symbol)
        current_price = ticker["last"]

        success = self.exchange.close_position(symbol, side)
        if not success:
            return 0.0, "error"

        if side == "long":
            gross_pnl = (current_price - entry_price) * qty * LEVERAGE
        else:
            gross_pnl = (entry_price - current_price) * qty * LEVERAGE

        commission = self._commission_usdt(qty, entry_price)
        net_pnl = gross_pnl - commission

        result = "tp" if net_pnl >= 0 else "sl"
        return net_pnl, result

    # ----------------------------------------------------------
    def monitor_trade(self, trade: Dict) -> Tuple[float, str]:
        symbol     = trade["symbol"]
        side       = trade["side"]
        entry_price = trade["entry_price"]
        qty        = trade["qty"]
        start_time = trade["open_time"]
        current_sl = trade["sl_price"]

        # Устанавливаем трейлинг-стоп на бирже через Bybit API
        df = self.exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=50)
        if not df.empty and len(df) >= 15:
            atr = calc_atr(df)
            trailing_delta = max(entry_price * MIN_TRAILING_PCT / 100, atr * 0.5)
            self.exchange.set_trailing_stop(symbol, side, trailing_delta)

        while time.time() - start_time < TRADE_DURATION:
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker["last"]
                position = self.exchange.get_position(symbol)

                # Позиция уже закрыта биржей (по SL)
                if not position or position["side"] != side:
                    # Пытаемся узнать реальный PnL
                    pnl_est = position["pnl"] if position else 0.0
                    return pnl_est, "sl" if pnl_est < 0 else "tp"

                # Обновляем программный трейлинг-стоп
                new_sl = self.update_trailing_stop(symbol, side, current_sl, current_price)
                if new_sl != current_sl:
                    current_sl = new_sl
                    log.info(f"🔄 Трейлинг-стоп обновлён: {current_sl:.8f}")

                # Программная проверка SL (на случай проскальзывания)
                if side == "long" and current_price <= current_sl:
                    pnl, result = self.close_trade(symbol, side, entry_price, qty)
                    return pnl, result
                if side == "short" and current_price >= current_sl:
                    pnl, result = self.close_trade(symbol, side, entry_price, qty)
                    return pnl, result

                time.sleep(5)

            except Exception as e:
                log.error(f"Ошибка мониторинга: {e}")
                time.sleep(5)

        # Время вышло — принудительно закрываем
        log.info(f"⏰ Время сделки истекло ({TRADE_DURATION}с) — закрываем позицию")
        pnl, result = self.close_trade(symbol, side, entry_price, qty)
        return pnl, result

    # ----------------------------------------------------------
    def run(self):
        log.info("🚀 Запуск MemeCoin Bot (Testnet)")
        log.info(f"📌 Символы: {SYMBOLS}")
        log.info(f"💰 Плечо: {LEVERAGE}x | Риск: до 50% от депозита")
        log.info(f"⏱️  Время сделки: {TRADE_DURATION // 60} мин")
        log.info(f"🔄 Мартингейл: до {MAX_MARTINGALE_STEPS} шагов")

        while True:
            try:
                balance = self.get_balance()
                if balance < MIN_BALANCE_USDT:
                    log.warning(
                        f"❌ Баланс {balance:.2f} USDT < {MIN_BALANCE_USDT} USDT — ожидание"
                    )
                    time.sleep(60)
                    continue

                # Проверяем открытые позиции
                has_position = False
                for sym in SYMBOLS:
                    pos = self.exchange.get_position(sym)
                    if pos:
                        has_position = True
                        log.info(
                            f"⏳ Открыта позиция: {pos['side'].upper()} {sym} | "
                            f"PnL={pos['pnl']:.4f} USDT"
                        )
                        break

                if has_position:
                    time.sleep(10)
                    continue

                # Выбираем символ по очереди
                symbol = SYMBOLS[self.trade_count % len(SYMBOLS)]
                self.trade_count += 1

                signal = self.get_entry_signal(symbol)
                if not signal:
                    log.info(f"⏭️  Нет сигнала для {symbol} — пропуск")
                    time.sleep(30)
                    continue

                trade = self.open_trade(symbol, signal)
                if not trade:
                    time.sleep(30)
                    continue

                pnl, result = self.monitor_trade(trade)

                # Обрабатываем результат
                if result in ("sl", "commission"):
                    self.loss_streak += 1
                    self.win_streak = 0
                    self.martingale_step = min(
                        self.martingale_step + 1, MAX_MARTINGALE_STEPS
                    )
                    log.warning(
                        f"❌ {result.upper()}: {symbol} | {signal.upper()} | "
                        f"PnL={pnl:.4f} USDT | Убытков подряд: {self.loss_streak} | "
                        f"Мартингейл шаг: {self.martingale_step}"
                    )
                elif result in ("tp", "closed"):
                    self.win_streak += 1
                    self.loss_streak = 0
                    self.martingale_step = max(0, self.martingale_step - 1)
                    log.info(
                        f"✅ {result.upper()}: {symbol} | {signal.upper()} | "
                        f"PnL={pnl:.4f} USDT | Выигрышей подряд: {self.win_streak}"
                    )
                else:
                    log.info(f"⚠️  {result}: {symbol} | PnL={pnl:.4f} USDT")

                self.total_pnl += pnl
                log.info(f"📊 Общий PnL: {self.total_pnl:.4f} USDT | Баланс: {self.get_balance():.2f} USDT")

                time.sleep(15)

            except KeyboardInterrupt:
                log.info("🛑 Бот остановлен пользователем")
                break
            except Exception as e:
                log.error(f"💥 Ошибка в главном цикле: {e}")
                time.sleep(30)


# ============================================================
# 🏁 ЗАПУСК
# ============================================================
if __name__ == "__main__":
    bot = MemeCoinBot()
    bot.run()
