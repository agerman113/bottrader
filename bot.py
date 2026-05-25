import os
import time
import json
import logging
import ccxt
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# ================== НАСТРОЙКИ ==================
SYMBOLS = [
    "DOGE/USDT:USDT", "SHIB/USDT:USDT", "PEPE/USDT:USDT",
    "FLOKI/USDT:USDT", "BONK/USDT:USDT", "WIF/USDT:USDT"
]

LEVERAGE = 2
INITIAL_MARGIN = 2.0
MARTINGALE_FACTOR = 1.5
MAX_MARGIN = 8.0
MAX_FLIPS = 3

TAKE_PROFIT_PCT = 4.0
TRAILING_ACTIVATE_PCT = 0.8
TRAILING_STEP_PCT = 0.3
STOP_LOSS_INITIAL_PCT = 1.0

MIN_ADX = 25
MAX_ADX_NEUTRAL = 20
MIN_ATR_PCT = 0.5

USE_LIMIT_ENTRY = False
TIMEFRAME_TREND = "1h"
TIMEFRAME_ADX = "1h"

SCAN_INTERVAL = 300
MIN_BALANCE = 5.0
REPORT_INTERVAL = 1800
MAX_DAILY_LOSS_PCT = 10.0
MAX_LOSING_SERIES_PER_DAY = 3

STATE_FILE = "flip_martingale_advanced.json"
LOG_FILE = "flip_bot_advanced.log"

TAKER_FEE = 0.00055
MAKER_FEE = 0.0002

# ================== ЛОГИРОВАНИЕ ==================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ================== БИРЖА ==================
exchange = ccxt.bybit({
    "apiKey":    os.getenv("BYBIT_API_KEY"),
    "secret":    os.getenv("BYBIT_API_SECRET"),
    "enableRateLimit": True,
    "options": {"defaultType": "linear"},
})

# ================== СТАТИСТИКА ==================
stats = {
    "запусков": 0,
    "сделок_всего": 0,
    "прибыльных": 0,
    "убыточных": 0,
    "прибыль_usdt": 0.0,
    "убыток_usdt": 0.0,
    "депозит_старт": 0.0,
    "старт_время": "",
    "последний_отчёт": 0.0,
    "ежедневный_убыток": 0.0,
    "последняя_дата_сброса": datetime.now().strftime("%Y-%m-%d"),
    "убыточные_серии_сегодня": 0,
    "последняя_убыточная_серия_время": 0.0,
}

def сохранить_состояние():
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning(f"Не сохранить состояние: {e}")

def загрузить_состояние():
    global stats
    if not os.path.exists(STATE_FILE):
        return False
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        for key in stats:
            if key in saved:
                stats[key] = saved[key]
        log.info(f"Состояние восстановлено из {STATE_FILE}")
        return True
    except Exception as e:
        log.warning(f"Не удалось загрузить состояние: {e}")
        return False

# ================== ИНВЕНТАРИЗАЦИЯ ==================
def отменить_все_ордера():
    try:
        orders = exchange.fetch_open_orders()
        for o in orders:
            exchange.cancel_order(o["id"], o["symbol"])
    except Exception as e:
        log.warning(f"Ошибка отмены ордеров: {e}")

def закрыть_все_позиции():
    try:
        positions = exchange.fetch_positions()
        for pos in positions:
            if float(pos.get("contracts", 0)) != 0:
                side = "sell" if pos["side"] == "long" else "buy"
                qty = abs(float(pos["contracts"]))
                exchange.create_market_order(pos["symbol"], side, qty, params={"reduceOnly": True})
                log.info(f"  Закрыта позиция {pos['symbol']} {pos['side']}")
    except Exception as e:
        log.warning(f"Ошибка закрытия позиций: {e}")

def полная_инвентаризация():
    log.info("🔄 Инвентаризация перед стартом...")
    отменить_все_ордера()
    закрыть_все_позиции()
    time.sleep(1)

# ================== БАЛАНС ==================
def баланс_usdt() -> float:
    try:
        b = exchange.fetch_balance({"type": "linear"})
        return float(b.get("USDT", {}).get("free", 0.0))
    except:
        return 0.0

def есть_открытая_позиция(symbol) -> bool:
    try:
        positions = exchange.fetch_positions([symbol])
        for pos in positions:
            if float(pos.get("contracts", 0)) != 0:
                return True
        return False
    except:
        return False

def установить_плечо(symbol: str, leverage: int):
    try:
        exchange.set_leverage(leverage, symbol)
    except Exception as e:
        log.warning(f"Не удалось установить плечо для {symbol}: {e}")

# ================== АНАЛИЗ РЫНКА (ADX, ATR, тренд) ==================
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rma(series, span):
    return series.ewm(alpha=1/span, adjust=False).mean()

def calc_adx(high, low, close, period=14):
    """
    Возвращает ADX (значение) и плюс/минус DI.
    Исправлена ошибка DeprecationWarning для np.maximum.
    """
    high = np.array(high)
    low = np.array(low)
    close = np.array(close)

    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Исправлено: np.maximum с двумя аргументами, затем результат с третьим
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)

    atr = rma(pd.Series(tr), period).values[-1] if len(tr) > 0 else 0
    plus_di = 100 * rma(pd.Series(plus_dm), period).values[-1] / atr if atr != 0 else 0
    minus_di = 100 * rma(pd.Series(minus_dm), period).values[-1] / atr if atr != 0 else 0
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) != 0 else 0
    adx = rma(pd.Series([dx]), period).values[-1] if len(pd.Series([dx])) > 0 else 0
    return adx, plus_di, minus_di

def анализ_рынка(symbol: str):
    """
    Возвращает словарь:
        trend_up     - bool, тренд вверх (EMA50>200 на 1h)
        adx          - значение ADX
        atr_pct      - ATR в процентах от цены
        can_trade    - можно ли торговать (ADX>MIN_ADX и ATR>MIN_ATR_PCT)
    """
    try:
        ohlcv_1h = exchange.fetch_ohlcv(symbol, TIMEFRAME_TREND, limit=200)
        if len(ohlcv_1h) < 200:
            return {"can_trade": False, "trend_up": None, "adx": 0, "atr_pct": 0}
        df = pd.DataFrame(ohlcv_1h, columns=["ts","o","h","l","c","v"])
        close = df["c"]
        # EMA тренд
        ema50 = ema(close, 50).iloc[-1]
        ema200 = ema(close, 200).iloc[-1]
        trend_up = ema50 > ema200
        # ADX
        adx_val, _, _ = calc_adx(df["h"].values, df["l"].values, df["c"].values, period=14)
        # ATR в процентах
        atr = rma((df["h"] - df["l"]), 14).iloc[-1]
        atr_pct = (atr / close.iloc[-1]) * 100 if close.iloc[-1] != 0 else 0
        can_trade = (adx_val > MIN_ADX) and (atr_pct > MIN_ATR_PCT)
        log.info(f"  📊 {symbol.split(':')[0]}: тренд={'вверх' if trend_up else 'вниз'}, ADX={adx_val:.1f}, ATR%={atr_pct:.2f}% -> {'✅ можно' if can_trade else '❌ нельзя'}")
        return {"can_trade": can_trade, "trend_up": trend_up, "adx": adx_val, "atr_pct": atr_pct}
    except Exception as e:
        log.warning(f"Ошибка анализа {symbol}: {e}")
        return {"can_trade": False, "trend_up": None, "adx": 0, "atr_pct": 0}

# ================== ПРОВЕРКА ДНЕВНЫХ ЛИМИТОВ ==================
def проверить_дневные_лимиты() -> bool:
    today = datetime.now().strftime("%Y-%m-%d")
    if stats["последняя_дата_сброса"] != today:
        stats["ежедневный_убыток"] = 0.0
        stats["убыточные_серии_сегодня"] = 0
        stats["последняя_дата_сброса"] = today
        сохранить_состояние()
    # Ежедневная просадка
    if stats["ежедневный_убыток"] > (stats["депозит_старт"] * MAX_DAILY_LOSS_PCT / 100):
        log.warning(f"  🛑 Дневной лимит убытка достигнут ({stats['ежедневный_убыток']:.2f} USDT). Пауза до завтра.")
        return False
    # Слишком много убыточных серий подряд
    if stats["убыточные_серии_сегодня"] >= MAX_LOSING_SERIES_PER_DAY:
        elapsed = time.time() - stats["последняя_убыточная_серия_время"]
        if elapsed < 3600:
            log.warning(f"  🛑 {MAX_LOSING_SERIES_PER_DAY} убыточные серии за сегодня. Пауза {3600 - int(elapsed)} сек.")
            return False
        else:
            stats["убыточные_серии_сегодня"] = 0
            сохранить_состояние()
    return True

# ================== ОТКРЫТИЕ ПОЗИЦИИ (исправлено) ==================
def открыть_позицию(symbol: str, side: str, margin_usdt: float):
    try:
        установить_плечо(symbol, LEVERAGE)
        ticker = exchange.fetch_ticker(symbol)
        price = ticker.get("last")
        if price is None or price <= 0:
            log.error(f"  Не удалось получить цену {symbol}")
            return None

        size_usdt = margin_usdt * LEVERAGE
        qty = size_usdt / price
        qty = float(exchange.amount_to_precision(symbol, qty))
        if qty <= 0:
            log.error(f"  Некорректное количество {qty} для {symbol}")
            return None

        tp_price = price * (1 + TAKE_PROFIT_PCT/100) if side == "buy" else price * (1 - TAKE_PROFIT_PCT/100)
        sl_price = price * (1 - STOP_LOSS_INITIAL_PCT/100) if side == "buy" else price * (1 + STOP_LOSS_INITIAL_PCT/100)

        if USE_LIMIT_ENTRY:
            limit_price = price * 0.999 if side == "buy" else price * 1.001
            order = exchange.create_limit_order(symbol, side, qty, limit_price,
                                                params={"takeProfit": tp_price, "stopLoss": sl_price})
            entry_price = limit_price
        else:
            order = exchange.create_market_order(symbol, side, qty,
                                                 params={"takeProfit": tp_price, "stopLoss": sl_price})
            entry_price = order.get("average", price)
            if entry_price is None:
                entry_price = price

        log.info(f"  📈 {side.upper()} {symbol} | маржа={margin_usdt:.2f}U | qty={qty} | вход={entry_price:.8f}")
        log.info(f"     TP={tp_price:.8f} (+{TAKE_PROFIT_PCT}%)  SL={sl_price:.8f}")
        return {
            "symbol": symbol,
            "side": side,
            "margin": margin_usdt,
            "entry_price": entry_price,
            "qty": qty,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "start_time": time.time(),
            "phase": 1,
            "peak_price": entry_price,
            "current_sl": sl_price,
        }
    except Exception as e:
        log.error(f"  ❌ Ошибка открытия позиции: {e}")
        return None

# ================== ОБНОВЛЕНИЕ SL ==================
def обновить_sl(symbol: str, new_sl: float):
    try:
        exchange.set_trading_stop(symbol, stopLoss=new_sl, params={"category": "linear"})
        log.info(f"  🔧 SL обновлён → {new_sl:.8f}")
    except Exception as e:
        log.warning(f"  ⚠️ Не удалось обновить SL: {e}")

# ================== МОНИТОРИНГ ПОЗИЦИИ ==================
def мониторить_позицию(pos_info: dict) -> str:
    symbol = pos_info["symbol"]
    side = pos_info["side"]
    entry = pos_info["entry_price"]
    qty = pos_info["qty"]
    start_time = pos_info["start_time"]
    tp_price = pos_info["tp_price"]
    current_sl = pos_info["current_sl"]
    phase = pos_info["phase"]
    peak_price = pos_info["peak_price"]

    deadline = start_time + 3600
    breakeven_price = entry * (1 + TRAILING_ACTIVATE_PCT/100) if side == "buy" else entry * (1 - TRAILING_ACTIVATE_PCT/100)

    while True:
        if time.time() > deadline:
            log.warning("  ⏰ Таймаут 1 час – закрываем принудительно")
            close_side = "sell" if side == "buy" else "buy"
            try:
                exchange.create_market_order(symbol, close_side, qty, params={"reduceOnly": True})
            except:
                pass
            return "timeout"

        time.sleep(10)
        try:
            positions = exchange.fetch_positions([symbol])
            active = [p for p in positions if float(p.get("contracts", 0)) != 0 and p.get("side") == side]
            if not active:
                cur = exchange.fetch_ticker(symbol)["last"]
                if (side == "buy" and cur >= tp_price * 0.99) or (side == "sell" and cur <= tp_price * 1.01):
                    return "tp"
                else:
                    return "sl"

            cur = exchange.fetch_ticker(symbol)["last"]
            pnl_pct = ((cur - entry) / entry * 100) if side == "buy" else ((entry - cur) / entry * 100)

            if phase == 1 and pnl_pct >= TRAILING_ACTIVATE_PCT:
                phase = 2
                new_sl = entry * (1 + (TRAILING_ACTIVATE_PCT - 0.1)/100) if side == "buy" else entry * (1 - (TRAILING_ACTIVATE_PCT - 0.1)/100)
                обновить_sl(symbol, new_sl)
                current_sl = new_sl
                log.info(f"  🔒 Безубыток активирован, SL={new_sl:.8f}")

            if phase >= 2:
                if (side == "buy" and cur > peak_price) or (side == "sell" and cur < peak_price):
                    peak_price = cur
                    if side == "buy":
                        new_sl = peak_price * (1 - TRAILING_STEP_PCT/100)
                    else:
                        new_sl = peak_price * (1 + TRAILING_STEP_PCT/100)
                    if (side == "buy" and new_sl > current_sl) or (side == "sell" and new_sl < current_sl):
                        обновить_sl(symbol, new_sl)
                        current_sl = new_sl
                        phase = 3
                        log.info(f"  📈 Трейлинг: peak={peak_price:.8f}, SL={new_sl:.8f}")

            log.info(f"  {symbol.split(':')[0]} {cur:.8f}  P&L={pnl_pct:+.2f}%  фаза={phase}")
            pos_info["peak_price"] = peak_price
            pos_info["current_sl"] = current_sl
            pos_info["phase"] = phase

        except Exception as e:
            log.warning(f"  Ошибка мониторинга: {e}")

# ================== ГЛАВНЫЙ ЦИКЛ ==================
def main():
    полная_инвентаризация()
    загрузить_состояние()
    баланс = баланс_usdt()
    if stats["депозит_старт"] == 0:
        stats["депозит_старт"] = баланс
        stats["старт_время"] = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    stats["запусков"] += 1
    stats["последний_отчёт"] = time.time()
    сохранить_состояние()

    log.info("=" * 60)
    log.info("  🔄 БОТ 'FLIP MARTINGALE' (улучшенная версия)")
    log.info(f"  Фильтры: ADX>{MIN_ADX}, ATR%>{MIN_ATR_PCT}% | TP={TAKE_PROFIT_PCT}% | Нач.маржа={INITIAL_MARGIN}U")
    log.info(f"  Дневной лимит убытка: {MAX_DAILY_LOSS_PCT}% | Серий убытков: {MAX_LOSING_SERIES_PER_DAY}")
    log.info(f"  Баланс: {баланс:.2f} USDT")
    log.info("=" * 60)

    while True:
        try:
            # 1. Дневные лимиты
            if not проверить_дневные_лимиты():
                time.sleep(600)
                continue

            # 2. Выбор монеты с сильным трендом
            best_pair = None
            best_trend_up = None
            for sym in SYMBOLS:
                анализ = анализ_рынка(sym)
                if not анализ["can_trade"]:
                    continue
                best_pair = sym
                best_trend_up = анализ["trend_up"]
                break
            if best_pair is None:
                log.info("  Нет пар, подходящих по ADX/ATR, ждём...")
                time.sleep(SCAN_INTERVAL)
                continue

            # 3. Определяем направление (с учётом тренда)
            direction = "buy" if best_trend_up else "sell"
            log.info(f"  🧭 Тренд {best_pair.split(':')[0]}: {'ЛОНГ' if direction=='buy' else 'ШОРТ'}")

            # 4. Серия переворотов
            margin = INITIAL_MARGIN
            flip_count = 0
            current_side = direction
            series_live = True

            while series_live and flip_count <= MAX_FLIPS:
                if баланс_usdt() < margin * 1.1:
                    log.warning(f"  Недостаточно средств, прерываем серию")
                    break

                log.info(f"  ━━━ Сделка {flip_count+1} | маржа={margin:.2f}U | сторона={current_side.upper()}")
                pos = открыть_позицию(best_pair, current_side, margin)
                if pos is None:
                    break

                stats["сделок_всего"] += 1
                сохранить_состояние()
                result = мониторить_позицию(pos)

                fee_rate = MAKER_FEE if USE_LIMIT_ENTRY else TAKER_FEE
                size = margin * LEVERAGE
                fee = size * fee_rate * 2
                pnl = 0.0
                if result == "tp":
                    pnl = size * TAKE_PROFIT_PCT / 100 - fee
                    stats["прибыльных"] += 1
                    stats["прибыль_usdt"] += max(0, pnl)
                    log.info(f"  ✅ TP! Прибыль ≈{pnl:.4f} USDT")
                    series_live = False
                else:
                    pnl = -(size * STOP_LOSS_INITIAL_PCT / 100 + fee)
                    stats["убыточных"] += 1
                    stats["убыток_usdt"] += abs(pnl)
                    stats["ежедневный_убыток"] += abs(pnl)
                    stats["убыточные_серии_сегодня"] += 1
                    stats["последняя_убыточная_серия_время"] = time.time()
                    log.warning(f"  ❌ SL! Убыток ≈{pnl:.4f} USDT")

                    # Переворот с проверкой тренда (не слепой)
                    if flip_count < MAX_FLIPS:
                        анализ = анализ_рынка(best_pair)
                        if not анализ["can_trade"]:
                            log.info("  Тренд ослаб, переворот отменяем, серия окончена")
                            series_live = False
                            break
                        new_direction = "buy" if анализ["trend_up"] else "sell"
                        if new_direction == current_side:
                            log.info("  Тренд не изменился – не переворачиваем, серия окончена")
                            series_live = False
                            break
                        # Переворот
                        flip_count += 1
                        margin = min(margin * MARTINGALE_FACTOR, MAX_MARGIN)
                        current_side = new_direction
                        log.info(f"  🔁 Переворот: новая сторона {current_side.upper()}, новая маржа {margin:.2f}U")
                        continue
                    else:
                        series_live = False

                # пауза между сделками в серии
                if series_live and result == "sl" and flip_count < MAX_FLIPS:
                    time.sleep(5)

            # Серия завершена
            log.info("  🎯 Серия завершена. Пауза 30 сек")
            time.sleep(30)
            полная_инвентаризация()

            # Отчёт каждые 30 минут
            if time.time() - stats["последний_отчёт"] >= REPORT_INTERVAL:
                bal_now = баланс_usdt()
                delta = bal_now - stats["депозит_старт"]
                pct = (delta / stats["депозит_старт"] * 100) if stats["депозит_старт"] > 0 else 0
                log.info("=" * 60)
                log.info(f"  📊 ОТЧЁТ | Депозит: {stats['депозит_старт']:.2f} → {bal_now:.2f} ({delta:+.2f}, {pct:+.1f}%)")
                log.info(f"  Сделок: {stats['сделок_всего']} | Прибыльных: {stats['прибыльных']} | Убыточных: {stats['убыточных']}")
                log.info(f"  P&L: {stats['прибыль_usdt'] - stats['убыток_usdt']:+.4f} USDT")
                log.info("=" * 60)
                stats["последний_отчёт"] = time.time()
                сохранить_состояние()

        except Exception as e:
            log.error(f"Глобальная ошибка: {e}", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    main()
