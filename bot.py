#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ОБЪЕДИНЁННЫЙ АДАПТИВНЫЙ RSI-БОТ для Bybit фьючерсов.

Логика работы:
  1. (Опционально) Сканирует топ-N монет по объёму и RSI-активности/волатильности
     и выбирает лучшую монету для торговли.
  2. Запускает СИМУЛЯЦИЮ трёх режимов агрессивности RSI (Консервативный /
     Умеренный / Агрессивный) на реальных рыночных данных в течение 10 минут,
     виртуально открывая и закрывая позиции по TP/SL, и собирает статистику
     по каждому режиму (винрейт, суммарная прибыль и т.д.).
  3. Выбирает режим с максимальной суммарной виртуальной прибылью и тестирует
     его ещё 5 минут (тоже виртуально, на тех же условиях).
  4. Если тест прошёл успешно (суммарная прибыль > 0) — бот переходит в режим
     РЕАЛЬНОЙ торговли с параметрами выбранного режима.
  5. Каждые REOPTIMIZE_INTERVAL_SEC (по умолчанию 4 часа) бот приостанавливает
     реальную торговлю (открытые позиции НЕ закрываются) и повторяет цикл
     симуляция → тест, чтобы подстроиться под текущую волатильность рынка.
  6. Если включён авто-сканер монет (FIXED_SYMBOL == ""), он также пересканирует
     рынок каждые SCAN_INTERVAL секунд; при смене монеты цикл симуляции
     запускается заново.

ВНИМАНИЕ: это инструмент автоматической торговли реальными деньгами.
Перед использованием на реальном счёте обязательно проверь логику на
тестовом аккаунте / минимальным депозитом — автор не даёт гарантий
прибыльности и не несёт ответственности за торговые результаты.
"""

import os
import time
import logging
import threading
import ccxt
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ============================================================
#                         НАСТРОЙКИ
# ============================================================

# Если задан — всегда торгуем только эту пару, сканер монет не запускается.
# Например: "BTC/USDT:USDT". Оставь "" для автоматического выбора монеты.
FIXED_SYMBOL          = ""

TIMEFRAME             = "5m"
STOP_LOSS_PERCENT     = 0.25
TAKE_PROFIT_PERCENT   = 1.0
LEVERAGE              = 1
CHECK_INTERVAL        = 60     # пауза между проверками в реальной торговле (сек)
RSI_MONITOR_INTERVAL  = 10     # фоновый монитор RSI (сек)
SIM_TICK_INTERVAL      = 5      # частота "тиков" на этапах симуляции/теста (сек)

# --- Параметры сканера монет (используются только если FIXED_SYMBOL == "") ---
SCAN_TOP_N            = 30
SCAN_INTERVAL         = 14400      # пересканировать монеты каждые 4 часа
SCAN_BARS             = 60
MIN_VOLUME_USDT       = 5_000_000

# --- Режимы агрессивности (симулируются параллельно на этапе 1) ---
AGGRESSION_MODES = [
    {"name": "Консервативный", "period": 5, "oversold": 25, "overbought": 75},
    {"name": "Умеренный",      "period": 3, "oversold": 30, "overbought": 70},
    {"name": "Агрессивный",    "period": 2, "oversold": 35, "overbought": 65},
]
MODE_MAX_PERIOD = max(m["period"] for m in AGGRESSION_MODES)

SIMULATION_FIRST_STAGE_SEC = 600     # 10 минут — параллельная симуляция всех режимов
TEST_SECOND_STAGE_SEC      = 300     # 5 минут — тест выбранного режима
REOPTIMIZE_INTERVAL_SEC    = 14400   # 4 часа — периодическая переоптимизация в LIVE

MAX_SIM_BARS_PER_TRADE = 20   # тайм-аут виртуальной сделки (в ЗАКРЫТЫХ свечах)

# ============================================================
#                        ЛОГИРОВАНИЕ
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]  %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("unified_rsi_bot.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ============================================================
#                          БИРЖА
# ============================================================
exchange = ccxt.bybit({
    "apiKey":          os.getenv("BYBIT_API_KEY"),
    "secret":          os.getenv("BYBIT_API_SECRET"),
    "enableRateLimit": True,
    "timeout":         10_000,
    "options":         {"defaultType": "linear"},
})

# ============================================================
#                    ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def get_balance() -> float:
    try:
        balance = exchange.fetch_balance({"type": "linear"})
        return float(balance["USDT"]["free"])
    except Exception as e:
        log.error(f"Ошибка получения баланса: {e}")
        return 0.0


def get_ohlcv(symbol: str, timeframe: str, limit: int = 150) -> pd.DataFrame:
    try:
        raw = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        return df.iloc[:-1].reset_index(drop=True)   # убираем незакрытую свечу
    except Exception as e:
        log.error(f"Ошибка получения свечей {symbol}: {e}")
        return pd.DataFrame()


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def set_leverage_once(symbol: str, leverage: int) -> None:
    try:
        exchange.set_leverage(leverage, symbol)
        log.info(f"Плечо {leverage}x установлено для {symbol}")
    except Exception as e:
        log.warning(f"Не удалось установить плечо: {e}")


def get_open_position(symbol: str) -> dict | None:
    try:
        positions = exchange.fetch_positions([symbol])
        for pos in positions:
            if float(pos.get("contracts", 0)) > 0 and pos.get("side") in ("long", "short"):
                return pos
        return None
    except Exception as e:
        log.error(f"Ошибка получения позиции: {e}")
        return None


def open_position(symbol: str, side: str, amount: float,
                   stop_loss_price: float, take_profit_price: float) -> float | None:
    existing = get_open_position(symbol)
    if existing:
        log.warning("Позиция уже открыта, пропускаем ордер")
        return None
    try:
        order = exchange.create_order(
            symbol=symbol,
            type="market",
            side=side,
            amount=amount,
            params={"takeProfit": take_profit_price, "stopLoss": stop_loss_price},
        )
        log.info(f"Открыта позиция {side.upper()} {amount} {symbol}")
        entry = order.get("average") or order.get("price")
        if not entry:
            entry = exchange.fetch_ticker(symbol)["last"]
        return float(entry)
    except Exception as e:
        log.error(f"Ошибка открытия позиции: {e}")
        return None

# ============================================================
#                СКАНЕР МОНЕТ ПО ВОЛАТИЛЬНОСТИ
# ============================================================

def score_symbol(symbol: str, period: int = 14, oversold: int = 30, overbought: int = 70) -> dict | None:
    """
    Оценивает монету по чистым пересечениям уровней RSI и волатильности (ATR%).
    Итоговый скор = пересечения × (1 + ATR%/10).
    """
    try:
        df = get_ohlcv(symbol, TIMEFRAME, limit=SCAN_BARS + period + 5)
        if df.empty or len(df) < SCAN_BARS:
            return None

        close, high, low = df["close"], df["high"], df["low"]
        atr_pct = ((high - low) / close).mean() * 100

        rsi = calculate_rsi(close, period).dropna()
        crosses = 0
        for i in range(1, len(rsi) - 3):
            prev, curr = rsi.iloc[i - 1], rsi.iloc[i]
            if prev <= oversold < curr:
                if not any(rsi.iloc[i + j] <= oversold for j in range(1, 4) if i + j < len(rsi)):
                    crosses += 1
            if prev >= overbought > curr:
                if not any(rsi.iloc[i + j] >= overbought for j in range(1, 4) if i + j < len(rsi)):
                    crosses += 1

        score = crosses * (1 + atr_pct / 10)
        return {"symbol": symbol, "score": round(score, 2), "crosses": crosses, "atr_pct": round(atr_pct, 3)}
    except Exception:
        return None


def scan_best_symbol() -> str:
    """Берёт топ-N монет по объёму, оценивает каждую и возвращает символ с лучшим скором."""
    fallback = "BTC/USDT:USDT"
    log.info(f"🔍 Сканирование топ-{SCAN_TOP_N} монет по объёму...")

    try:
        tickers = exchange.fetch_tickers()
    except Exception as e:
        log.error(f"Ошибка получения тикеров: {e}")
        return fallback

    candidates = []
    for sym, t in tickers.items():
        if not sym.endswith(":USDT"):
            continue
        vol = (t.get("quoteVolume") or 0)
        if vol >= MIN_VOLUME_USDT:
            candidates.append((sym, vol))

    candidates.sort(key=lambda x: x[1], reverse=True)
    top = [sym for sym, _ in candidates[:SCAN_TOP_N]]
    log.info(f"  Отобрано {len(top)} монет для анализа RSI-активности")

    results = []
    for i, sym in enumerate(top, 1):
        result = score_symbol(sym)
        if result:
            results.append(result)
            log.info(
                f"  [{i:2d}/{len(top)}] {sym:<22} "
                f"скор={result['score']:6.2f}  "
                f"пересечений={result['crosses']}  "
                f"ATR={result['atr_pct']:.2f}%"
            )
        time.sleep(0.3)   # не спамим API

    if not results:
        log.warning("Сканер не нашёл подходящих монет, используем BTC")
        return fallback

    results.sort(key=lambda x: x["score"], reverse=True)
    best = results[0]

    log.info("─" * 55)
    log.info(f"🏆 Лучшая монета: {best['symbol']}  (скор={best['score']}  "
             f"пересечений={best['crosses']}  ATR={best['atr_pct']}%)")
    log.info("  Топ-5 монет по скору:")
    for r in results[:5]:
        log.info(f"    {r['symbol']:<22} скор={r['score']:6.2f}  пересечений={r['crosses']}")
    log.info("─" * 55)

    return best["symbol"]

# ============================================================
#          ВИРТУАЛЬНАЯ СДЕЛКА И ДВИЖОК СИМУЛЯЦИИ
# ============================================================

class SimulatedTrade:
    """Одна виртуальная позиция: следит за достижением TP/SL или тайм-аутом."""

    def __init__(self, side: str, entry_price: float, tp_price: float, sl_price: float,
                 max_bars: int = MAX_SIM_BARS_PER_TRADE):
        self.side        = side
        self.entry_price = entry_price
        self.tp_price    = tp_price
        self.sl_price    = sl_price
        self.max_bars    = max_bars
        self.closed      = False
        self._bars_passed = 0

    def update(self, candle_high: float, candle_low: float, candle_close: float):
        """Вызывается РОВНО ОДИН РАЗ за каждую новую закрытую свечу."""
        if self.closed:
            return None
        self._bars_passed += 1

        if self.side == "long":
            if candle_high >= self.tp_price:
                self.closed = True
                return ("tp", self.tp_price)
            if candle_low <= self.sl_price:
                self.closed = True
                return ("sl", self.sl_price)
        else:  # short
            if candle_low <= self.tp_price:
                self.closed = True
                return ("tp", self.tp_price)
            if candle_high >= self.sl_price:
                self.closed = True
                return ("sl", self.sl_price)

        if self.max_bars > 0 and self._bars_passed >= self.max_bars:
            self.closed = True
            return ("timeout", candle_close)

        return None


class SimulationEngine:
    """Виртуально торгует ОДНИМ режимом агрессивности на текущей монете."""

    def __init__(self, mode_config: dict):
        self.config = mode_config
        self.name   = mode_config["name"]
        self.trades: list[SimulatedTrade] = []
        self.closed_trades: list[tuple[str, float]] = []   # (exit_type, profit_pct)
        self._last_signal_bar_ts = None   # доп. защита от повторного сигнала на той же свече

    def check_signal(self, df: pd.DataFrame) -> str | None:
        close  = df["close"]
        period = self.config["period"]
        rsi    = calculate_rsi(close, period)
        if len(rsi) < 3:
            return None

        ts = df["timestamp"].iloc[-1]
        if ts == self._last_signal_bar_ts:
            return None  # эта свеча уже была обработана этим движком

        rsi_prev, rsi_curr = rsi.iloc[-2], rsi.iloc[-1]
        oversold, overbought = self.config["oversold"], self.config["overbought"]

        signal = None
        if rsi_prev <= oversold < rsi_curr:
            signal = "long"
        elif rsi_prev >= overbought > rsi_curr:
            signal = "short"

        if signal:
            self._last_signal_bar_ts = ts
        return signal

    def open_trade(self, side: str, entry_price: float) -> None:
        tp = entry_price * (1 + TAKE_PROFIT_PERCENT / 100) if side == "long" \
            else entry_price * (1 - TAKE_PROFIT_PERCENT / 100)
        sl = entry_price * (1 - STOP_LOSS_PERCENT / 100) if side == "long" \
            else entry_price * (1 + STOP_LOSS_PERCENT / 100)
        self.trades.append(SimulatedTrade(side, entry_price, tp, sl))

    def update_trades(self, candle: dict) -> None:
        """Вызывать РОВНО ОДИН РАЗ за каждую новую закрытую свечу!"""
        high, low, close = candle["high"], candle["low"], candle["close"]
        still_open = []
        for trade in self.trades:
            result = trade.update(high, low, close)
            if result is None:
                still_open.append(trade)
                continue
            exit_type, exit_price = result
            profit_pct = (exit_price - trade.entry_price) / trade.entry_price * 100
            if trade.side == "short":
                profit_pct = -profit_pct
            self.closed_trades.append((exit_type, profit_pct))
        self.trades = still_open

    def total_profit(self) -> float:
        return sum(p for _, p in self.closed_trades)

    def stats(self) -> str:
        if not self.closed_trades:
            return f"{self.name}: нет завершённых сделок (открытых сейчас: {len(self.trades)})"
        wins     = sum(1 for t, _ in self.closed_trades if t == "tp")
        losses   = sum(1 for t, _ in self.closed_trades if t == "sl")
        timeouts = sum(1 for t, _ in self.closed_trades if t == "timeout")
        total    = len(self.closed_trades)
        winrate  = wins / total * 100
        avg      = self.total_profit() / total
        return (f"{self.name} (период={self.config['period']}, "
                f"OS={self.config['oversold']}/OB={self.config['overbought']}): "
                f"сделок={total} TP={wins} SL={losses} Timeout={timeouts}  "
                f"winrate={winrate:.1f}%  сумм.прибыль={self.total_profit():.2f}%  "
                f"средняя={avg:.2f}%")

# ============================================================
#          ФОНОВЫЙ МОНИТОР RSI (каждые 10 сек)
# ============================================================

_state = {"symbol": "BTC/USDT:USDT", "period": 14, "oversold": 30, "overbought": 70, "active": True}


def rsi_monitor() -> None:
    def bar(v: float, w: int = 20) -> str:
        f = int(max(0.0, min(100.0, v)) / 100 * w)
        return "[" + "█" * f + "░" * (w - f) + "]"

    while _state["active"]:
        try:
            sym, period = _state["symbol"], _state["period"]
            raw = exchange.fetch_ohlcv(sym, TIMEFRAME, limit=period + 10)
            df  = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            rsi_val = calculate_rsi(df["close"], period).iloc[-1]
            price   = df["close"].iloc[-1]
            os_, ob_ = _state["oversold"], _state["overbought"]

            if rsi_val <= os_:
                zone = "🔵 ПЕРЕПРОДАННОСТЬ — жди ЛОНГ"
            elif rsi_val >= ob_:
                zone = "🔴 ПЕРЕКУПЛЕННОСТЬ — жди ШОРТ"
            elif rsi_val < 40:
                zone = "🟡 Зона слабости"
            elif rsi_val > 60:
                zone = "🟠 Зона силы"
            else:
                zone = "⚪ Нейтральная зона"

            log.info(
                f"📊 [{sym.split('/')[0]}] RSI={rsi_val:5.1f} {bar(rsi_val)}  "
                f"Цена={price:.6f}  {zone}"
            )
        except Exception as e:
            log.debug(f"Монитор RSI: {e}")
        time.sleep(RSI_MONITOR_INTERVAL)

# ============================================================
#              ОТКРЫТИЕ РЕАЛЬНОЙ ПОЗИЦИИ
# ============================================================

def try_open_real_position(symbol: str, signal: str, force_rescan: dict) -> None:
    """Считает размер позиции, проверяет минимумы лота, ставит TP/SL и открывает ордер."""
    free_balance = get_balance()
    if free_balance <= 0:
        log.warning("Баланс = 0, пропускаем")
        return

    current_price = float(exchange.fetch_ticker(symbol)["last"])
    raw_size = free_balance / current_price

    market   = exchange.market(symbol)
    min_amt  = float((market.get("limits") or {}).get("amount", {}).get("min") or 0)
    min_cost = float((market.get("limits") or {}).get("cost", {}).get("min") or 0)

    if min_amt > 0 and raw_size < min_amt:
        log.warning(
            f"Недостаточно для мин. лота: нужно {min_amt} {market['base']}, есть {raw_size:.6f}. "
            f"Запрашиваем пересканирование монеты..."
        )
        force_rescan["reset"] = True
        return

    if min_cost > 0 and raw_size * current_price < min_cost:
        log.warning(
            f"Сумма {raw_size * current_price:.2f} USDT < минимума {min_cost} USDT. "
            f"Запрашиваем пересканирование монеты..."
        )
        force_rescan["reset"] = True
        return

    position_size = float(exchange.amount_to_precision(symbol, raw_size))
    if position_size <= 0:
        log.warning("Размер позиции = 0 после округления, запрашиваем пересканирование...")
        force_rescan["reset"] = True
        return

    if signal == "long":
        sl_price = current_price * (1 - STOP_LOSS_PERCENT / 100)
        tp_price = current_price * (1 + TAKE_PROFIT_PERCENT / 100)
        side = "buy"
    else:
        sl_price = current_price * (1 + STOP_LOSS_PERCENT / 100)
        tp_price = current_price * (1 - TAKE_PROFIT_PERCENT / 100)
        side = "sell"

    sl_price = float(exchange.price_to_precision(symbol, sl_price))
    tp_price = float(exchange.price_to_precision(symbol, tp_price))

    log.info(
        f"Вход {signal.upper()}: цена={current_price:.6f}  "
        f"объём={position_size}  SL={sl_price}  TP={tp_price}"
    )

    entry = open_position(symbol, side, position_size, sl_price, tp_price)
    if entry:
        log.info(f"✅ Позиция открыта @ {entry:.6f}, ожидаем TP/SL")
    else:
        log.error("❌ Не удалось открыть позицию")

# ============================================================
#                        ОСНОВНАЯ ЛОГИКА
# ============================================================

STATE_SIMULATION = 1   # параллельная симуляция 3 режимов (10 мин)
STATE_TEST       = 2   # тест выбранного режима (5 мин)
STATE_LIVE       = 3   # реальная торговля


def main() -> None:
    log.info("=" * 70)
    log.info("🤖 ОБЪЕДИНЁННЫЙ АДАПТИВНЫЙ RSI-БОТ (Bybit фьючерсы)")
    log.info(f"  Таймфрейм: {TIMEFRAME} | Плечо: {LEVERAGE}x | "
             f"SL: {STOP_LOSS_PERCENT}% | TP: {TAKE_PROFIT_PERCENT}%")
    log.info(f"  Режимы агрессивности: {', '.join(m['name'] for m in AGGRESSION_MODES)}")
    log.info(f"  Симуляция: {SIMULATION_FIRST_STAGE_SEC // 60} мин  →  "
             f"Тест: {TEST_SECOND_STAGE_SEC // 60} мин  →  Реальная торговля")
    if FIXED_SYMBOL:
        log.info(f"  Монета зафиксирована: {FIXED_SYMBOL}")
    else:
        log.info(f"  Авто-сканер монет: топ-{SCAN_TOP_N}, пересканирование каждые "
                  f"{SCAN_INTERVAL // 3600}ч")
    log.info("=" * 70)

    threading.Thread(target=rsi_monitor, daemon=True).start()
    log.info(f"👁 Мониторинг RSI запущен (каждые {RSI_MONITOR_INTERVAL} сек)")

    current_symbol  = ""
    last_scan_time  = 0.0
    force_rescan    = {"reset": False}   # сигнал «пересканировать монету прямо сейчас»

    sim_engines: list[SimulationEngine] = []
    test_engine: SimulationEngine | None = None
    last_sim_candle_ts = None   # timestamp последней обработанной (на этапе сим./теста) свечи

    chosen_period     = 14
    chosen_oversold    = 30
    chosen_overbought = 70

    state       = STATE_SIMULATION
    state_start = time.time()

    while True:
        try:
            now = time.time()

            # ──────────────────────────────────────────────────────────
            # 0. ВЫБОР МОНЕТЫ (при старте, по таймеру и по явному запросу)
            # ──────────────────────────────────────────────────────────
            need_rescan = (
                not current_symbol
                or (not FIXED_SYMBOL and now - last_scan_time > SCAN_INTERVAL)
                or force_rescan["reset"]
            )
            if need_rescan:
                force_rescan["reset"] = False
                new_symbol = FIXED_SYMBOL if FIXED_SYMBOL else scan_best_symbol()
                if new_symbol != current_symbol:
                    log.info(f"🔄 Монета: {current_symbol or '—'} → {new_symbol}")
                    current_symbol = new_symbol
                    set_leverage_once(current_symbol, LEVERAGE)
                    _state["symbol"] = current_symbol
                    # При смене монеты обязательно пересимулируем все режимы заново
                    sim_engines         = [SimulationEngine(m) for m in AGGRESSION_MODES]
                    last_sim_candle_ts  = None
                    state       = STATE_SIMULATION
                    state_start = now
                last_scan_time = now

            # ──────────────────────────────────────────────────────────
            # Получаем последнюю ЗАКРЫТУЮ свечу (нужна на всех этапах)
            # ──────────────────────────────────────────────────────────
            needed_period = max(MODE_MAX_PERIOD, chosen_period)
            df = get_ohlcv(current_symbol, TIMEFRAME, limit=needed_period + 60)
            if df.empty or len(df) < needed_period + 2:
                log.warning("Недостаточно данных, ждём...")
                time.sleep(CHECK_INTERVAL)
                continue

            last_candle   = df.iloc[-1].to_dict()
            current_price = last_candle["close"]
            candle_ts     = last_candle["timestamp"]

            # ──────────────────────────────────────────────────────────
            # ЭТАП 1: СИМУЛЯЦИЯ 3 РЕЖИМОВ (10 МИНУТ)
            # ──────────────────────────────────────────────────────────
            if state == STATE_SIMULATION:
                # Обрабатываем сигналы/сделки ровно ОДИН РАЗ за новую закрытую свечу,
                # даже если тиков (проверок) за это время было несколько.
                if candle_ts != last_sim_candle_ts:
                    last_sim_candle_ts = candle_ts
                    for engine in sim_engines:
                        signal = engine.check_signal(df)
                        if signal:
                            engine.open_trade(signal, current_price)
                            log.info(f"[СИМ {engine.name}] Сигнал {signal.upper()} @ {current_price:.6f}")
                        engine.update_trades(last_candle)

                elapsed = now - state_start
                if elapsed >= SIMULATION_FIRST_STAGE_SEC:
                    log.info("=" * 60)
                    log.info(f"=== Симуляция завершена ({current_symbol}) ===")
                    for engine in sim_engines:
                        log.info("  " + engine.stats())

                    best = max(sim_engines, key=lambda e: e.total_profit())
                    if best.total_profit() <= 0:
                        log.warning(f"Все режимы убыточны/нулевые. Перезапускаем симуляцию "
                                    f"ещё на {SIMULATION_FIRST_STAGE_SEC // 60} мин.")
                        sim_engines         = [SimulationEngine(m) for m in AGGRESSION_MODES]
                        last_sim_candle_ts  = None
                        state_start = now
                    else:
                        log.info(f"✅ Выбран режим: {best.name} "
                                 f"(виртуальная прибыль {best.total_profit():.2f}%)")
                        test_engine         = SimulationEngine(best.config)
                        last_sim_candle_ts  = None
                        state       = STATE_TEST
                        state_start = now
                    log.info("=" * 60)
                else:
                    time.sleep(SIM_TICK_INTERVAL)
                    continue

            # ──────────────────────────────────────────────────────────
            # ЭТАП 2: ТЕСТ ВЫБРАННОГО РЕЖИМА (5 МИНУТ)
            # ──────────────────────────────────────────────────────────
            elif state == STATE_TEST:
                if candle_ts != last_sim_candle_ts:
                    last_sim_candle_ts = candle_ts
                    signal = test_engine.check_signal(df)
                    if signal:
                        test_engine.open_trade(signal, current_price)
                        log.info(f"[ТЕСТ {test_engine.name}] Сигнал {signal.upper()} @ {current_price:.6f}")
                    test_engine.update_trades(last_candle)

                elapsed = now - state_start
                if elapsed >= TEST_SECOND_STAGE_SEC:
                    log.info("=" * 60)
                    log.info(f"=== Тест завершён ({current_symbol}) ===")
                    log.info("  " + test_engine.stats())

                    if test_engine.total_profit() > 0:
                        chosen_period      = test_engine.config["period"]
                        chosen_oversold    = test_engine.config["oversold"]
                        chosen_overbought  = test_engine.config["overbought"]
                        _state["period"]     = chosen_period
                        _state["oversold"]   = chosen_oversold
                        _state["overbought"] = chosen_overbought
                        log.info(
                            f"✅ Тест успешен! Переходим в РЕАЛЬНУЮ торговлю с режимом "
                            f"'{test_engine.name}' (период={chosen_period}, "
                            f"OS={chosen_oversold}/OB={chosen_overbought})"
                        )
                        state       = STATE_LIVE
                        state_start = now
                    else:
                        log.warning("Тест показал убыток. Возвращаемся к симуляции всех режимов.")
                        sim_engines         = [SimulationEngine(m) for m in AGGRESSION_MODES]
                        last_sim_candle_ts  = None
                        state       = STATE_SIMULATION
                        state_start = now
                    log.info("=" * 60)
                else:
                    time.sleep(SIM_TICK_INTERVAL)
                    continue

            # ──────────────────────────────────────────────────────────
            # ЭТАП 3: РЕАЛЬНАЯ ТОРГОВЛЯ
            # ──────────────────────────────────────────────────────────
            elif state == STATE_LIVE:
                # Периодическая переоптимизация — каждые REOPTIMIZE_INTERVAL_SEC
                if now - state_start > REOPTIMIZE_INTERVAL_SEC:
                    log.info("🔄 Время переоптимизации режима агрессивности — "
                             "запускаем симуляцию заново (открытые позиции не закрываются).")
                    sim_engines         = [SimulationEngine(m) for m in AGGRESSION_MODES]
                    last_sim_candle_ts  = None
                    state       = STATE_SIMULATION
                    state_start = now
                    continue

                pos = get_open_position(current_symbol)
                if pos is not None:
                    log.debug(
                        f"Позиция: {pos['side'].upper()} {pos['contracts']} "
                        f"@ {pos.get('entryPrice', '?')}"
                    )
                    time.sleep(CHECK_INTERVAL)
                    continue

                last_candle_ts_sec = candle_ts / 1000
                candle_age = time.time() - last_candle_ts_sec
                if candle_age > 600:
                    log.warning(f"Данные устарели ({candle_age:.0f} сек), пропускаем")
                    time.sleep(CHECK_INTERVAL)
                    continue

                rsi = calculate_rsi(df["close"], chosen_period)
                rsi_prev, rsi_curr = rsi.iloc[-2], rsi.iloc[-1]
                log.info(
                    f"RSI (период={chosen_period}): "
                    f"предыд.={rsi_prev:.2f}  текущ.={rsi_curr:.2f}  "
                    f"свеча={candle_age:.0f}с назад  [{current_symbol}]"
                )

                signal = None
                if rsi_prev <= chosen_oversold < rsi_curr:
                    signal = "long"
                    log.info(f"📈 Сигнал ЛОНГ: RSI пересёк {chosen_oversold} снизу вверх")
                elif rsi_prev >= chosen_overbought > rsi_curr:
                    signal = "short"
                    log.info(f"📉 Сигнал ШОРТ: RSI пересёк {chosen_overbought} сверху вниз")

                if signal:
                    try_open_real_position(current_symbol, signal, force_rescan)

                time.sleep(CHECK_INTERVAL)

        except Exception as e:
            log.error(f"Ошибка в главном цикле: {e}", exc_info=True)
            time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
