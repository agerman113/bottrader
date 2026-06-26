#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
АДАПТИВНЫЙ RSI-БОТ для Bybit фьючерсов.
- Сканирует топ монеты по волатильности и выбирает лучшую для торговли
- Автоматически выбирает оптимальный период RSI
- Фоновый мониторинг RSI каждые 10 секунд
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

# Если задан — всегда торгуем только эту пару, сканер не запускается
FIXED_SYMBOL        = ""               # Например "BTC/USDT:USDT" или "" (авто)

TIMEFRAME           = "5m"            # Таймфрейм
RSI_OVERSOLD        = 20              # Порог перепроданности
RSI_OVERBOUGHT      = 80              # Порог перекупленности
STOP_LOSS_PERCENT   = 0.25            # SL в %
TAKE_PROFIT_PERCENT = 1.0             # TP в %
LEVERAGE            = 1               # Плечо (1x = без плеча)
CHECK_INTERVAL      = 60              # Пауза между проверками (сек)
RSI_MONITOR_INTERVAL = 10             # Мониторинг RSI каждые 10 сек

# Параметры адаптивного периода RSI
PERIODS_TO_TEST     = [7, 10, 14, 21, 30]
ANALYSIS_BARS       = 50              # Свечей для анализа (50 × 5m ≈ 4 ч)
REOPTIMIZE_INTERVAL = 1800            # Переоптимизация периода каждые 30 мин

# Параметры сканера монет
SCAN_TOP_N          = 30              # Сколько топ-монет по объёму анализировать
SCAN_INTERVAL       = 14400           # Пересканировать монеты каждые 4 часа
SCAN_BARS           = 60              # Свечей для оценки волатильности
MIN_VOLUME_USDT     = 5_000_000       # Минимальный суточный объём (фильтр ликвидности)

# ============================================================
#                        ЛОГИРОВАНИЕ
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]  %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("adaptive_rsi_bot.log", encoding="utf-8"),
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
        df  = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
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


def open_position(
    symbol: str,
    side: str,
    amount: float,
    stop_loss_price: float,
    take_profit_price: float,
) -> float | None:
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
#                    СКАНЕР МОНЕТ ПО ВОЛАТИЛЬНОСТИ
# ============================================================

def score_symbol(symbol: str, period: int = 14) -> dict | None:
    """
    Оценивает монету по трём критериям:
      1. rsi_crosses  — число чистых пересечений уровней RSI (главный критерий)
      2. volatility   — средний ATR% (амплитуда свечей относительно цены)
      3. volume_usdt  — суточный объём в USDT (фильтр ликвидности)

    Итоговый скор = rsi_crosses × (1 + volatility/10)
    Больше пересечений + выше волатильность = лучше для нашей стратегии.
    """
    try:
        df = get_ohlcv(symbol, TIMEFRAME, limit=SCAN_BARS + period + 5)
        if df.empty or len(df) < SCAN_BARS:
            return None

        close = df["close"]
        high  = df["high"]
        low   = df["low"]

        # Волатильность: средний (high-low)/close в %
        atr_pct = ((high - low) / close).mean() * 100

        # RSI-пересечения
        rsi = calculate_rsi(close, period).dropna()
        crosses = 0
        for i in range(1, len(rsi) - 3):
            prev, curr = rsi.iloc[i - 1], rsi.iloc[i]
            if prev <= RSI_OVERSOLD < curr:
                if not any(rsi.iloc[i+j] <= RSI_OVERSOLD for j in range(1, 4) if i+j < len(rsi)):
                    crosses += 1
            if prev >= RSI_OVERBOUGHT > curr:
                if not any(rsi.iloc[i+j] >= RSI_OVERBOUGHT for j in range(1, 4) if i+j < len(rsi)):
                    crosses += 1

        score = crosses * (1 + atr_pct / 10)

        return {
            "symbol":     symbol,
            "score":      round(score, 2),
            "crosses":    crosses,
            "atr_pct":    round(atr_pct, 3),
        }
    except Exception:
        return None


def scan_best_symbol() -> str:
    """
    Берёт топ-N монет по суточному объёму, оценивает каждую
    и возвращает символ с наибольшим скором.
    Возвращает BTC/USDT:USDT если ничего не нашли.
    """
    fallback = "BTC/USDT:USDT"
    log.info(f"🔍 Сканирование топ-{SCAN_TOP_N} монет по объёму...")

    try:
        tickers = exchange.fetch_tickers()
    except Exception as e:
        log.error(f"Ошибка получения тикеров: {e}")
        return fallback

    # Фильтруем: только USDT-линейные фьючерсы с достаточным объёмом
    candidates = []
    for sym, t in tickers.items():
        if not sym.endswith(":USDT"):
            continue
        vol = (t.get("quoteVolume") or 0)
        if vol >= MIN_VOLUME_USDT:
            candidates.append((sym, vol))

    # Сортируем по объёму, берём топ-N
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
    log.info(f"🏆 Лучшая монета: {best['symbol']}")
    log.info(f"   Скор={best['score']}  Пересечений={best['crosses']}  ATR={best['atr_pct']}%")
    log.info("─" * 55)

    # Топ-5 для наглядности
    log.info("  Топ-5 монет по скору:")
    for r in results[:5]:
        log.info(f"    {r['symbol']:<22} скор={r['score']:6.2f}  пересечений={r['crosses']}")

    return best["symbol"]

# ============================================================
#              АДАПТИВНЫЙ ВЫБОР ПЕРИОДА RSI
# ============================================================

def find_best_rsi_period(symbol: str, timeframe: str, periods: list[int],
                         lookback_bars: int, oversold: int = 20, overbought: int = 80) -> int:
    default = 14
    try:
        df = get_ohlcv(symbol, timeframe, limit=lookback_bars + max(periods) + 10)
        if df.empty or len(df) < lookback_bars:
            return default

        close = df["close"]
        best_period, best_score = default, -999.0

        for period in periods:
            rsi = calculate_rsi(close, period).dropna()
            if rsi.empty:
                continue
            clean = false_cnt = 0
            for i in range(1, len(rsi) - 3):
                prev, curr = rsi.iloc[i - 1], rsi.iloc[i]
                if prev <= oversold < curr:
                    rev = any(rsi.iloc[i+j] <= oversold for j in range(1, 4) if i+j < len(rsi))
                    clean += 0 if rev else 1
                    false_cnt += 1 if rev else 0
                if prev >= overbought > curr:
                    rev = any(rsi.iloc[i+j] >= overbought for j in range(1, 4) if i+j < len(rsi))
                    clean += 0 if rev else 1
                    false_cnt += 1 if rev else 0
            score = clean - false_cnt * 0.5
            log.debug(f"  Период {period:2d}: чистых={clean}  ложных={false_cnt}  скор={score:.1f}")
            if score > best_score:
                best_score, best_period = score, period

        if best_score <= 0:
            return default
        log.info(f"Выбран период RSI: {best_period} (скор={best_score:.1f})")
        return best_period
    except Exception as e:
        log.error(f"Ошибка анализа периодов: {e}")
        return default

# ============================================================
#              ФОНОВЫЙ МОНИТОРИНГ RSI (каждые 10 сек)
# ============================================================

_state = {
    "symbol": "BTC/USDT:USDT",
    "period": 14,
    "active": True,
}

def rsi_monitor() -> None:
    def bar(v: float, w: int = 20) -> str:
        f = int(v / 100 * w)
        return "[" + "█" * f + "░" * (w - f) + "]"

    while _state["active"]:
        try:
            sym    = _state["symbol"]
            period = _state["period"]
            raw    = exchange.fetch_ohlcv(sym, TIMEFRAME, limit=period + 10)
            df     = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
            rsi_val = calculate_rsi(df["close"], period).iloc[-1]
            price   = df["close"].iloc[-1]

            if rsi_val <= RSI_OVERSOLD:
                zone = "🔵 ПЕРЕПРОДАННОСТЬ — жди ЛОНГ"
            elif rsi_val >= RSI_OVERBOUGHT:
                zone = "🔴 ПЕРЕКУПЛЕННОСТЬ — жди ШОРТ"
            elif rsi_val < 40:
                zone = "🟡 Зона слабости"
            elif rsi_val > 60:
                zone = "🟠 Зона силы"
            else:
                zone = "⚪ Нейтральная зона"

            log.info(
                f"📊 [{sym.split('/')[0]}] RSI={rsi_val:5.1f} {bar(rsi_val)}  "
                f"Цена={price:.4f}  {zone}"
            )
        except Exception as e:
            log.debug(f"Монитор RSI: {e}")
        time.sleep(RSI_MONITOR_INTERVAL)

# ============================================================
#                        ОСНОВНАЯ ЛОГИКА
# ============================================================

def main() -> None:
    log.info("=" * 65)
    log.info("🤖 АДАПТИВНЫЙ RSI-БОТ (Bybit фьючерсы)")
    log.info(f"  Таймфрейм: {TIMEFRAME} | Плечо: {LEVERAGE}x")
    log.info(f"  SL: {STOP_LOSS_PERCENT}% | TP: {TAKE_PROFIT_PERCENT}%")
    log.info(f"  Сканер монет: каждые {SCAN_INTERVAL // 3600}ч | топ-{SCAN_TOP_N}")
    log.info("=" * 65)

    # Запускаем фоновый монитор RSI
    threading.Thread(target=rsi_monitor, daemon=True).start()
    log.info(f"👁 Мониторинг RSI запущен (каждые {RSI_MONITOR_INTERVAL} сек)")

    current_symbol = ""
    current_period = 14
    last_scan      = 0.0
    last_optimize  = 0.0

    while True:
        try:
            now = time.time()

            # ── 1. Сканер монет ──────────────────────────────────────────────
            if FIXED_SYMBOL:
                current_symbol = FIXED_SYMBOL
            elif now - last_scan > SCAN_INTERVAL:
                new_symbol = scan_best_symbol()
                if new_symbol != current_symbol:
                    log.info(f"🔄 Переключаемся на монету: {new_symbol}")
                    current_symbol       = new_symbol
                    _state["symbol"]     = new_symbol
                    set_leverage_once(current_symbol, LEVERAGE)
                    last_optimize = 0   # сбрасываем, чтобы сразу переоптимизировать период
                last_scan = time.time()

            if not current_symbol:
                time.sleep(10)
                continue

            # ── 2. Переоптимизация периода RSI ──────────────────────────────
            if now - last_optimize > REOPTIMIZE_INTERVAL:
                log.info("🔄 Анализ оптимального периода RSI...")
                current_period = find_best_rsi_period(
                    symbol=current_symbol, timeframe=TIMEFRAME,
                    periods=PERIODS_TO_TEST, lookback_bars=ANALYSIS_BARS,
                    oversold=RSI_OVERSOLD, overbought=RSI_OVERBOUGHT,
                )
                _state["period"] = current_period
                last_optimize    = time.time()
                log.info(f"✅ Период RSI: {current_period} | Монета: {current_symbol}")

            # ── 3. Проверяем открытую позицию ───────────────────────────────
            pos = get_open_position(current_symbol)
            if pos is not None:
                log.debug(
                    f"Позиция: {pos['side'].upper()} {pos['contracts']} "
                    f"@ {pos.get('entryPrice','?')}"
                )
                time.sleep(CHECK_INTERVAL)
                continue

            # ── 4. Получаем свечи и RSI ─────────────────────────────────────
            df = get_ohlcv(current_symbol, TIMEFRAME, limit=current_period + 60)
            if df.empty or len(df) < current_period + 2:
                log.warning("Недостаточно данных, ждём...")
                time.sleep(CHECK_INTERVAL)
                continue

            rsi      = calculate_rsi(df["close"], current_period)
            rsi_prev = rsi.iloc[-2]
            rsi_curr = rsi.iloc[-1]
            log.info(
                f"RSI (период={current_period}): "
                f"предыд.={rsi_prev:.2f}  текущ.={rsi_curr:.2f}  [{current_symbol}]"
            )

            # ── 5. Сигналы входа ─────────────────────────────────────────────
            signal = None
            if rsi_prev <= RSI_OVERSOLD and rsi_curr > RSI_OVERSOLD:
                signal = "long"
                log.info(f"📈 Сигнал ЛОНГ: RSI пересёк {RSI_OVERSOLD} снизу вверх")
            elif rsi_prev >= RSI_OVERBOUGHT and rsi_curr < RSI_OVERBOUGHT:
                signal = "short"
                log.info(f"📉 Сигнал ШОРТ: RSI пересёк {RSI_OVERBOUGHT} сверху вниз")

            if signal is None:
                time.sleep(CHECK_INTERVAL)
                continue

            # ── 6. Размер позиции ────────────────────────────────────────────
            free_balance = get_balance()
            if free_balance <= 0:
                log.warning("Баланс = 0, пропускаем")
                time.sleep(CHECK_INTERVAL)
                continue

            current_price = float(exchange.fetch_ticker(current_symbol)["last"])
            position_size = float(
                exchange.amount_to_precision(
                    current_symbol, free_balance / current_price
                )
            )
            if position_size <= 0:
                log.warning("Размер позиции слишком мал, пропускаем")
                time.sleep(CHECK_INTERVAL)
                continue

            # ── 7. TP / SL ───────────────────────────────────────────────────
            if signal == "long":
                sl_price = current_price * (1 - STOP_LOSS_PERCENT   / 100)
                tp_price = current_price * (1 + TAKE_PROFIT_PERCENT / 100)
                side     = "buy"
            else:
                sl_price = current_price * (1 + STOP_LOSS_PERCENT   / 100)
                tp_price = current_price * (1 - TAKE_PROFIT_PERCENT / 100)
                side     = "sell"

            sl_price = float(exchange.price_to_precision(current_symbol, sl_price))
            tp_price = float(exchange.price_to_precision(current_symbol, tp_price))

            log.info(
                f"Вход {signal.upper()}: цена={current_price:.4f}  "
                f"объём={position_size}  SL={sl_price}  TP={tp_price}"
            )

            # ── 8. Открываем позицию ─────────────────────────────────────────
            entry = open_position(current_symbol, side, position_size, sl_price, tp_price)
            if entry:
                log.info(f"✅ Позиция открыта @ {entry:.4f}, ожидаем TP/SL")
            else:
                log.error("❌ Не удалось открыть позицию")

            time.sleep(CHECK_INTERVAL)

        except Exception as e:
            log.error(f"Ошибка в главном цикле: {e}", exc_info=True)
            time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
