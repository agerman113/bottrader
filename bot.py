#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
АДАПТИВНЫЙ RSI-БОТ для Bybit фьючерсов.
Автоматически выбирает оптимальный период RSI на основе исторических данных.
"""

import os
import time
import logging
import ccxt
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ============================================================
#                         НАСТРОЙКИ
# ============================================================

SYMBOL              = "BTC/USDT:USDT"   # Торговая пара (USDT-фьючерсы)
TIMEFRAME           = "5m"              # Таймфрейм
RSI_OVERSOLD        = 20               # Порог перепроданности
RSI_OVERBOUGHT      = 80               # Порог перекупленности
STOP_LOSS_PERCENT   = 0.25             # SL в %
TAKE_PROFIT_PERCENT = 1.0              # TP в %
LEVERAGE            = 1                # Плечо (1x = без плеча)
CHECK_INTERVAL      = 60               # Пауза между проверками (сек)

# Параметры адаптивного периода
PERIODS_TO_TEST     = [7, 10, 14, 21, 30]
ANALYSIS_BARS       = 50               # Свечей для анализа (50 × 5m ≈ 4 ч)
REOPTIMIZE_INTERVAL = 1800             # Переоптимизация каждые 30 мин

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
    "apiKey":        os.getenv("BYBIT_API_KEY"),
    "secret":        os.getenv("BYBIT_API_SECRET"),
    "enableRateLimit": True,
    "timeout":       10_000,                          # FIX: таймаут 10 сек
    "options":       {"defaultType": "linear"},       # USDT-фьючерсы
})

# ============================================================
#                    ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def get_balance() -> float:
    """Свободный баланс USDT на фьючерсном счёте."""
    try:
        balance = exchange.fetch_balance({"type": "linear"})
        return float(balance["USDT"]["free"])
    except Exception as e:
        log.error(f"Ошибка получения баланса: {e}")
        return 0.0


def get_ohlcv(symbol: str, timeframe: str, limit: int = 150) -> pd.DataFrame:
    """OHLCV-данные в виде DataFrame."""
    try:
        raw = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        # Убираем незакрытую свечу (последнюю)
        return df.iloc[:-1].reset_index(drop=True)
    except Exception as e:
        log.error(f"Ошибка получения свечей: {e}")
        return pd.DataFrame()


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI классическим методом Wilder (EMA)."""
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def set_leverage_once(symbol: str, leverage: int) -> None:
    """Устанавливает плечо; ошибку логирует, не падает."""
    try:
        exchange.set_leverage(leverage, symbol)
        log.info(f"Плечо {leverage}x установлено для {symbol}")
    except Exception as e:
        log.warning(f"Не удалось установить плечо: {e}")


def get_open_position(symbol: str) -> dict | None:
    """
    Возвращает открытую позицию (long или short) или None.
    Источник правды — биржа, не локальный флаг.
    """
    try:
        positions = exchange.fetch_positions([symbol])
        for pos in positions:
            # contracts > 0 у обеих сторон; отличаем по side
            if float(pos.get("contracts", 0)) > 0 and pos.get("side") in ("long", "short"):
                return pos
        return None
    except Exception as e:
        log.error(f"Ошибка получения позиции: {e}")
        return None   # FIX: возвращаем None (консервативно), не открываем новую


def close_position(symbol: str, position: dict) -> bool:
    """Закрывает позицию рыночным ордером."""
    try:
        side   = "sell" if position["side"] == "long" else "buy"
        amount = abs(float(position["contracts"]))
        exchange.create_order(
            symbol=symbol,
            type="market",
            side=side,
            amount=amount,
            params={"reduceOnly": True},
        )
        log.info(f"Позиция закрыта (market {side})")
        return True
    except Exception as e:
        log.error(f"Ошибка закрытия позиции: {e}")
        return False


def open_position(
    symbol: str,
    side: str,
    amount: float,
    stop_loss_price: float,
    take_profit_price: float,
) -> float | None:
    """
    Открывает позицию рыночным ордером + TP/SL.
    Перед открытием повторно проверяет, нет ли уже позиции.
    Возвращает цену входа или None при ошибке.
    """
    # FIX: защита от дублирования — последняя проверка перед сделкой
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
            params={
                "takeProfit": take_profit_price,
                "stopLoss":   stop_loss_price,
            },
        )
        log.info(f"Открыта позиция {side.upper()} {amount} {symbol}")

        entry = order.get("average") or order.get("price")
        if not entry:
            ticker = exchange.fetch_ticker(symbol)
            entry = ticker["last"]
        return float(entry)
    except Exception as e:
        log.error(f"Ошибка открытия позиции: {e}")
        return None

# ============================================================
#              АДАПТИВНЫЙ ВЫБОР ПЕРИОДА RSI
# ============================================================

def find_best_rsi_period(
    symbol: str,
    timeframe: str,
    periods: list[int],
    lookback_bars: int,
    oversold: int = 20,
    overbought: int = 80,
) -> int:
    """
    Перебирает периоды RSI и возвращает тот, у которого
    наибольший скор «чистых пересечений» (без обратного касания
    через 3 свечи) за вычетом штрафа за ложные сигналы.
    """
    default = 14
    try:
        limit = lookback_bars + max(periods) + 10
        df = get_ohlcv(symbol, timeframe, limit=limit)
        if df.empty or len(df) < lookback_bars:
            log.warning("Недостаточно данных для анализа периодов")
            return default

        close = df["close"]
        best_period = default
        best_score  = -999.0

        for period in periods:
            rsi = calculate_rsi(close, period).dropna()
            if rsi.empty:
                continue

            clean_long = clean_short = false_long = false_short = 0

            for i in range(1, len(rsi) - 3):
                prev, curr = rsi.iloc[i - 1], rsi.iloc[i]

                if prev <= oversold < curr:           # лонг-сигнал
                    reverse = any(rsi.iloc[i + j] <= oversold for j in range(1, 4) if i + j < len(rsi))
                    if reverse:
                        false_long += 1
                    else:
                        clean_long += 1

                if prev >= overbought > curr:          # шорт-сигнал
                    reverse = any(rsi.iloc[i + j] >= overbought for j in range(1, 4) if i + j < len(rsi))
                    if reverse:
                        false_short += 1
                    else:
                        clean_short += 1

            score = (clean_long + clean_short) - (false_long + false_short) * 0.5
            log.debug(
                f"  Период {period:2d}: лонг={clean_long}, шорт={clean_short}, "
                f"ложных={false_long + false_short}, скор={score:.1f}"
            )

            if score > best_score:
                best_score  = score
                best_period = period

        if best_score <= 0:
            log.info("Нет качественных сигналов ни для одного периода → период 14")
            return default

        log.info(f"Выбран оптимальный период RSI: {best_period} (скор={best_score:.1f})")
        return best_period

    except Exception as e:
        log.error(f"Ошибка анализа периодов: {e}")
        return default

# ============================================================
#                        ОСНОВНАЯ ЛОГИКА
# ============================================================

def main() -> None:
    log.info("=" * 65)
    log.info("🤖 АДАПТИВНЫЙ RSI-БОТ (Bybit фьючерсы)")
    log.info(f"  Пара: {SYMBOL} | Таймфрейм: {TIMEFRAME} | Плечо: {LEVERAGE}x")
    log.info(f"  Периоды: {PERIODS_TO_TEST} | Переоптимизация: {REOPTIMIZE_INTERVAL // 60} мин")
    log.info(f"  SL: {STOP_LOSS_PERCENT}% | TP: {TAKE_PROFIT_PERCENT}%")
    log.info("=" * 65)

    # Устанавливаем плечо один раз при старте
    set_leverage_once(SYMBOL, LEVERAGE)

    current_period = 14
    last_optimize  = 0.0

    while True:
        try:
            # ── 1. Переоптимизация периода ─────────────────────────────────
            now = time.time()
            if now - last_optimize > REOPTIMIZE_INTERVAL:
                log.info("🔄 Анализ оптимального периода RSI...")
                current_period = find_best_rsi_period(
                    symbol=SYMBOL,
                    timeframe=TIMEFRAME,
                    periods=PERIODS_TO_TEST,
                    lookback_bars=ANALYSIS_BARS,
                    oversold=RSI_OVERSOLD,
                    overbought=RSI_OVERBOUGHT,
                )
                last_optimize = time.time()
                log.info(f"✅ Используемый период RSI: {current_period}")

            # ── 2. Проверяем открытую позицию (источник правды — биржа) ────
            # FIX: нет локального флага in_position, всегда спрашиваем биржу
            pos = get_open_position(SYMBOL)
            if pos is not None:
                log.debug(
                    f"Позиция открыта: {pos['side'].upper()} "
                    f"{pos['contracts']} @ {pos.get('entryPrice', '?')}"
                )
                time.sleep(CHECK_INTERVAL)
                continue

            # ── 3. Получаем свечи и RSI ─────────────────────────────────────
            df = get_ohlcv(SYMBOL, TIMEFRAME, limit=current_period + 60)
            if df.empty or len(df) < current_period + 2:
                log.warning("Недостаточно данных, ждём...")
                time.sleep(CHECK_INTERVAL)
                continue

            rsi = calculate_rsi(df["close"], current_period)
            if rsi.isna().all():
                log.warning("RSI не вычислен, ждём...")
                time.sleep(CHECK_INTERVAL)
                continue

            # FIX: сигнал по закрытым свечам [-2] и [-1], не через prev_rsi между итерациями
            rsi_prev = rsi.iloc[-2]
            rsi_curr = rsi.iloc[-1]
            log.info(
                f"RSI (период={current_period}): предыд.={rsi_prev:.2f}  текущ.={rsi_curr:.2f}"
            )

            # ── 4. Сигналы входа ─────────────────────────────────────────────
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

            # ── 5. Расчёт размера позиции ────────────────────────────────────
            free_balance = get_balance()
            if free_balance <= 0:
                log.warning("Баланс USDT = 0, пропускаем сделку")
                time.sleep(CHECK_INTERVAL)
                continue

            ticker        = exchange.fetch_ticker(SYMBOL)
            current_price = float(ticker["last"])

            # Весь баланс в позицию (депозит ~25 USDT, 1x плечо)
            raw_size      = free_balance / current_price
            position_size = float(exchange.amount_to_precision(SYMBOL, raw_size))

            if position_size <= 0:
                log.warning("Размер позиции слишком мал, пропускаем")
                time.sleep(CHECK_INTERVAL)
                continue

            # ── 6. TP / SL ───────────────────────────────────────────────────
            if signal == "long":
                sl_price = current_price * (1 - STOP_LOSS_PERCENT   / 100)
                tp_price = current_price * (1 + TAKE_PROFIT_PERCENT / 100)
                side     = "buy"
            else:
                sl_price = current_price * (1 + STOP_LOSS_PERCENT   / 100)
                tp_price = current_price * (1 - TAKE_PROFIT_PERCENT / 100)
                side     = "sell"

            sl_price = float(exchange.price_to_precision(SYMBOL, sl_price))
            tp_price = float(exchange.price_to_precision(SYMBOL, tp_price))

            log.info(
                f"Вход {signal.upper()}: цена={current_price:.2f}  "
                f"объём={position_size:.6f}  SL={sl_price:.2f}  TP={tp_price:.2f}  "
                f"(RSI период={current_period})"
            )

            # ── 7. Открываем позицию ─────────────────────────────────────────
            entry = open_position(SYMBOL, side, position_size, sl_price, tp_price)
            if entry:
                log.info(f"✅ Позиция открыта @ {entry:.2f}, ожидаем TP/SL")
            else:
                log.error("❌ Не удалось открыть позицию")

            time.sleep(CHECK_INTERVAL)

        except Exception as e:
            log.error(f"Ошибка в главном цикле: {e}", exc_info=True)
            time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
