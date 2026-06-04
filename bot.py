#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bybit Мем-Коин Бот v4.0 — Testnet
────────────────────────────────────────────────────────
Торговые механизмы:
  • Тренд по EMA 9/21 (3m свечи с Bybit)
  • После убытка — разворот направления (стратегия догон)
  • Мартингейл до 10 шагов (x1.5 за шаг)
  • Трейлинг-стоп через Bybit API
  • Учёт комиссий в расчёте ставки
  • Лимит сделки 3 минуты, без тейк-профита
  • Плечо 5x (риск до 50% депозита)

Механизмы из задачи о разорении игрока:
  • Вычисление вероятности разорения P(ruin) перед входом
  • Блокировка торговли при P(ruin) > 80%
  • Смелая игра: при winrate < 50% — увеличиваем ставку
    (математически выгоднее дробных шагов при p < 0.5)
  • Целевой барьер +20% к стартовому балансу:
    достигли — сбрасываем мартингейл, фиксируем сессию
  • Стоп по просадке: баланс упал на 40% — торговля стоп
────────────────────────────────────────────────────────
"""

import os, time, math, logging
from collections import deque
from typing import Deque, Dict, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

load_dotenv()

# ══════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ══════════════════════════════════════════════════════
SYMBOLS        = ["1000PEPEUSDT", "WIFUSDT", "1000BONKUSDT"]
LEVERAGE       = 5        # плечо
TRADE_DURATION = 180      # сек на сделку
MAX_MART_STEPS = 10       # макс шагов мартингейла
BASE_RISK_PCT  = 2.0      # % баланса — базовая ставка
MART_MULT      = 1.5      # множитель мартингейла
FEE_RATE       = 0.00055  # тейкер-комиссия Bybit
MIN_BALANCE    = 5.0      # мин. баланс USDT
INITIAL_SL_PCT = 2.0      # базовый SL %
TRAILING_PCT   = 1.0      # трейлинг-дельта %
TIMEFRAME      = "3"      # 3-минутные свечи

# ── Параметры задачи о разорении ──────────────────────
RUIN_BLOCK_THRESHOLD = 0.80   # блокируем вход при P(ruin) > 80%
TARGET_PROFIT_PCT    = 20.0   # целевой барьер прибыли (% от старт. баланса)
MAX_DRAWDOWN_PCT     = 40.0   # стоп при просадке баланса (% от старт.)
WINRATE_WINDOW       = 20     # последние N сделок для оценки winrate

# ══════════════════════════════════════════════════════
# ЛОГГЕР
# ══════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════
# BYBIT WRAPPER
# ══════════════════════════════════════════════════════
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
                interval=TIMEFRAME, limit=limit,
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
                buyLeverage=str(lev), sellLeverage=str(lev),
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

    def open_order(self, symbol: str, side: str, qty: float, sl: float) -> Optional[str]:
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


# ══════════════════════════════════════════════════════
# ИНДИКАТОРЫ
# ══════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════
# МАТЕМАТИКА ЗАДАЧИ О РАЗОРЕНИИ ИГРОКА
# ══════════════════════════════════════════════════════
def ruin_probability(p: float, capital: float, target: float) -> float:
    """
    Вероятность разорения игрока с капиталом `capital`,
    играющего до достижения `target` (или нуля).
    Формула: P(ruin) = (r^capital - r^target) / (1 - r^target),
    где r = (1-p)/p, при p ≠ 0.5.
    При p = 0.5: P(ruin) = 1 - capital/target.
    """
    if target <= capital:
        return 0.0
    if abs(p - 0.5) < 1e-9:
        return 1.0 - capital / target
    r = (1.0 - p) / p
    try:
        num = r**capital - r**target
        den = 1.0 - r**target
        if den == 0:
            return 1.0
        return max(0.0, min(1.0, num / den))
    except (OverflowError, ZeroDivisionError):
        return 1.0 if p < 0.5 else 0.0

def bold_bet(p: float, balance: float, target: float) -> float:
    """
    Смелая игра: при p < 0.5 оптимально ставить как можно
    больше за раз (ближе к целевому барьеру за один шаг).
    Возвращает рекомендуемый размер ставки в USDT.
    При p >= 0.5 — осторожная игра (стандартный BASE_RISK_PCT).
    """
    if p >= 0.5:
        return balance * BASE_RISK_PCT / 100
    # Ставим минимум из: (target - balance) и 50% баланса
    bold = min(target - balance, balance * 0.5)
    return max(bold, balance * BASE_RISK_PCT / 100)


# ══════════════════════════════════════════════════════
# БОТ
# ══════════════════════════════════════════════════════
class Bot:
    def __init__(self):
        self.ex          = Bybit()
        self.mart_step   = 0
        self.loss_streak = 0
        self.last_side: Optional[str] = None
        self.total_pnl   = 0.0
        self.trade_n     = 0

        # История результатов для winrate (1=win, 0=loss)
        self.history: Deque[int] = deque(maxlen=WINRATE_WINDOW)

        # Стартовый баланс — для расчёта целевого барьера и просадки
        self.start_balance = self.ex.balance()
        self.target_balance = self.start_balance * (1 + TARGET_PROFIT_PCT / 100)
        self.floor_balance  = self.start_balance * (1 - MAX_DRAWDOWN_PCT / 100)

        log.info("═══════════════════════════════════════════════")
        log.info("  Bybit Meme Bot v4.0 (Testnet)")
        log.info("  + Задача о разорении игрока")
        log.info("═══════════════════════════════════════════════")
        log.info(f"  Стартовый баланс : {self.start_balance:.2f} USDT")
        log.info(f"  Цель сессии      : {self.target_balance:.2f} USDT (+{TARGET_PROFIT_PCT}%)")
        log.info(f"  Стоп просадки    : {self.floor_balance:.2f} USDT (-{MAX_DRAWDOWN_PCT}%)")
        log.info(f"  Символы          : {SYMBOLS}")
        log.info(f"  Плечо {LEVERAGE}x | Мартингейл до {MAX_MART_STEPS} шагов")

        for sym in SYMBOLS:
            self.ex.set_leverage(sym, LEVERAGE)

    # ── WINRATE ────────────────────────────────────────
    def winrate(self) -> float:
        """Доля побед в последних WINRATE_WINDOW сделках."""
        if not self.history:
            return 0.5  # нейтральная оценка при старте
        return sum(self.history) / len(self.history)

    # ── ФИЛЬТР РАЗОРЕНИЯ ───────────────────────────────
    def check_ruin_filter(self, balance: float) -> bool:
        """
        Возвращает True если торговать можно,
        False если вероятность разорения слишком высока.
        """
        p = self.winrate()
        # Расстояние до цели в условных единицах (шагах BASE_RISK_PCT)
        unit = balance * BASE_RISK_PCT / 100
        if unit <= 0:
            return False
        capital_units = balance / unit
        target_units  = self.target_balance / unit
        p_ruin = ruin_probability(p, capital_units, target_units)

        log.info(
            f"  📐 Разорение игрока: winrate={p:.1%} | "
            f"P(ruin)={p_ruin:.1%} | "
            f"капитал={capital_units:.1f}u / цель={target_units:.1f}u"
        )

        if p_ruin > RUIN_BLOCK_THRESHOLD:
            log.warning(
                f"  ⛔ P(ruin)={p_ruin:.1%} > {RUIN_BLOCK_THRESHOLD:.0%} — "
                f"вход заблокирован"
            )
            return False
        return True

    # ── РАЗМЕР СТАВКИ ──────────────────────────────────
    def calc_qty(self, symbol: str, price: float, balance: float) -> float:
        if balance < MIN_BALANCE:
            return 0.0

        p = self.winrate()

        # Смелая vs осторожная игра (из теории разорения)
        if len(self.history) >= 5:
            risk_usdt = bold_bet(p, balance, self.target_balance) * LEVERAGE
            game_mode = "СМЕЛАЯ" if p < 0.5 else "ОСТОРОЖНАЯ"
        else:
            # Мало данных — стандартный BASE_RISK_PCT
            risk_usdt = balance * (BASE_RISK_PCT / 100) * LEVERAGE
            game_mode = "СТАНДАРТ"

        # Мартингейл поверх базовой ставки
        risk_usdt *= (MART_MULT ** self.mart_step)

        # Жёсткий потолок — 50% баланса с плечом
        risk_usdt = min(risk_usdt, balance * 0.5 * LEVERAGE)

        commission = risk_usdt * FEE_RATE * 2
        log.info(
            f"  💰 Игра: {game_mode} | winrate={p:.1%} | "
            f"ставка={risk_usdt:.2f} USDT | комиссия≈{commission:.4f} USDT"
        )

        min_q, step = self.ex.min_qty(symbol)
        qty = math.floor((risk_usdt / price) / step) * step
        return max(min_q, qty)

    # ── SL ─────────────────────────────────────────────
    def calc_sl(self, side: str, price: float) -> float:
        pct = INITIAL_SL_PCT / 100
        return price * (1 - pct) if side == "long" else price * (1 + pct)

    # ── СИГНАЛ ─────────────────────────────────────────
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
            log.info(f"  🔄 Догон: {self.last_side} → {flip} (убытков подряд: {self.loss_streak})")
            return flip

        return trend

    # ── ОТКРЫТИЕ СДЕЛКИ ────────────────────────────────
    def open_trade(self, symbol: str, side: str, balance: float) -> Optional[Dict]:
        price = self.ex.price(symbol)
        if not price:
            return None

        qty = self.calc_qty(symbol, price, balance)
        if qty <= 0:
            log.warning("  Объём = 0, пропуск")
            return None

        sl = self.calc_sl(side, price)
        oid = self.ex.open_order(symbol, side, qty, sl)
        if not oid:
            return None

        time.sleep(1)

        trailing_delta = price * TRAILING_PCT / 100
        self.ex.set_trailing_stop(symbol, trailing_delta)

        self.last_side = side
        log.info(
            f"🟢 ОТКРЫТО: {side.upper()} {symbol} | "
            f"qty={qty} | price={price:.8f} | SL={sl:.8f} | mart={self.mart_step}"
        )
        return {"symbol": symbol, "side": side, "entry": price,
                "qty": qty, "sl": sl, "t0": time.time()}

    # ── МОНИТОРИНГ ─────────────────────────────────────
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
                    log.info(f"  Закрыта биржей | PnL≈{pnl:.4f} USDT")
                    return pnl, "sl" if pnl <= 0 else "tp"

                elapsed = int(time.time() - t0)
                log.info(
                    f"  [{elapsed}s/{TRADE_DURATION}s] {side.upper()} {symbol} | "
                    f"PnL={pos['pnl']:.4f} USDT | SL={pos['sl']:.8f}"
                )
                time.sleep(5)

            except Exception as e:
                log.error(f"monitor: {e}")
                time.sleep(5)

        log.info("  ⏰ Время вышло — закрываем принудительно")
        cur = self.ex.price(symbol)
        self.ex.close_order(symbol, side, qty)

        pnl = ((cur - entry) if side == "long" else (entry - cur)) * qty * LEVERAGE
        pnl -= entry * qty * FEE_RATE * 2
        return pnl, "timeout"

    # ── ПРОВЕРКА БАРЬЕРОВ ──────────────────────────────
    def check_barriers(self, balance: float) -> Optional[str]:
        """
        Возвращает:
          'target'   — достигнут целевой барьер (победа сессии)
          'drawdown' — достигнута максимальная просадка (стоп)
          None       — всё в норме
        """
        if balance >= self.target_balance:
            return "target"
        if balance <= self.floor_balance:
            return "drawdown"
        return None

    # ── ГЛАВНЫЙ ЦИКЛ ───────────────────────────────────
    def run(self):
        while True:
            try:
                bal = self.ex.balance()

                # ── Проверка барьеров ──────────────────
                barrier = self.check_barriers(bal)
                if barrier == "target":
                    log.info(
                        f"🏆 ЦЕЛЬ ДОСТИГНУТА! Баланс={bal:.2f} USDT "
                        f"(+{TARGET_PROFIT_PCT}% от старта) | "
                        f"Мартингейл сброшен. Пауза 60s."
                    )
                    self.mart_step = 0
                    self.loss_streak = 0
                    self.history.clear()
                    # Сдвигаем стартовый баланс и цель вперёд
                    self.start_balance  = bal
                    self.target_balance = bal * (1 + TARGET_PROFIT_PCT / 100)
                    self.floor_balance  = bal * (1 - MAX_DRAWDOWN_PCT  / 100)
                    log.info(
                        f"  Новая цель: {self.target_balance:.2f} USDT | "
                        f"Стоп: {self.floor_balance:.2f} USDT"
                    )
                    time.sleep(60)
                    continue

                if barrier == "drawdown":
                    log.error(
                        f"🛑 СТОП ПРОСАДКИ: баланс={bal:.2f} USDT "
                        f"(-{MAX_DRAWDOWN_PCT}% от старта {self.start_balance:.2f} USDT). "
                        f"Бот остановлен."
                    )
                    break

                if bal < MIN_BALANCE:
                    log.warning(f"Баланс {bal:.2f} < {MIN_BALANCE} USDT — ожидание")
                    time.sleep(60)
                    continue

                # ── Проверка открытых позиций ──────────
                busy = False
                for sym in SYMBOLS:
                    p = self.ex.position(sym)
                    if p:
                        log.info(f"⏳ Открыта: {p['side'].upper()} {sym} | PnL={p['pnl']:.4f}")
                        busy = True
                        break
                if busy:
                    time.sleep(10)
                    continue

                # ── Выбор символа ──────────────────────
                symbol = SYMBOLS[self.trade_n % len(SYMBOLS)]
                self.trade_n += 1

                log.info(
                    f"\n{'─'*50}\n"
                    f"  Сделка #{self.trade_n} | {symbol} | mart={self.mart_step} | "
                    f"баланс={bal:.2f} USDT"
                )

                # ── Фильтр разорения ───────────────────
                if not self.check_ruin_filter(bal):
                    log.info("  Ждём 30s...")
                    time.sleep(30)
                    continue

                # ── Сигнал ─────────────────────────────
                signal = self.get_signal(symbol)
                if not signal:
                    log.info("  Нет чёткого тренда — пропуск")
                    time.sleep(30)
                    continue

                log.info(f"  Сигнал: {signal.upper()}")
                trade = self.open_trade(symbol, signal, bal)
                if not trade:
                    time.sleep(30)
                    continue

                # ── Мониторинг ─────────────────────────
                pnl, result = self.monitor(trade)
                self.total_pnl += pnl

                is_win = pnl > 0
                self.history.append(1 if is_win else 0)

                if is_win:
                    self.loss_streak = 0
                    self.mart_step = max(0, self.mart_step - 1)
                    log.info(f"✅ ПРИБЫЛЬ | PnL={pnl:.4f} USDT | result={result}")
                else:
                    self.loss_streak += 1
                    self.mart_step = min(self.mart_step + 1, MAX_MART_STEPS)
                    log.warning(
                        f"❌ УБЫТОК | PnL={pnl:.4f} USDT | result={result} | "
                        f"loss_streak={self.loss_streak} | mart={self.mart_step}"
                    )

                new_bal = self.ex.balance()
                log.info(
                    f"📊 Итого PnL: {self.total_pnl:.4f} USDT | "
                    f"Баланс: {new_bal:.2f} USDT | "
                    f"Winrate: {self.winrate():.1%} ({sum(self.history)}/{len(self.history)})"
                )
                time.sleep(10)

            except KeyboardInterrupt:
                log.info("🛑 Остановлен пользователем")
                break
            except Exception as e:
                log.error(f"Главный цикл: {e}")
                time.sleep(30)


# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    Bot().run()
