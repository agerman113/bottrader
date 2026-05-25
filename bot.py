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

# Время экспирации для каждого шага (секунды)
EXPIRE_TIMES = [60, 300, 900]       # 1 мин, 5 мин, 15 мин

# Фильтр тренда (опционально)
USE_TREND_FILTER = True
TIMEFRAME_TREND = "1h"

SCAN_INTERVAL = 300
MIN_BALANCE = 5.0
REPORT_INTERVAL = 1800

MAX_DAILY_LOSS_PCT = 10.0
MAX_LOSING_SERIES_PER_DAY = 3

STATE_FILE = "binary_flip_martingale.json"
LOG_FILE = "binary_bot.log"

TAKER_FEE = 0.00055

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

def установить_плечо(symbol: str, leverage: int):
    try:
        exchange.set_leverage(leverage, symbol)
    except Exception as e:
        log.warning(f"Не удалось установить плечо для {symbol}: {e}")

# ================== ОПРЕДЕЛЕНИЕ ТРЕНДА ==================
def определить_тренд(symbol: str) -> str:
    if not USE_TREND_FILTER:
        return "up"
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME_TREND, limit=200)
        if len(ohlcv) < 200:
            return "neutral"
        df = pd.DataFrame(ohlcv, columns=["ts","o","h","l","c","v"])
        close = df["c"]
        ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
        ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1]
        if ema50 > ema200 * 1.005:
            return "up"
        elif ema50 < ema200 * 0.995:
            return "down"
        else:
            return "neutral"
    except Exception as e:
        log.warning(f"Ошибка тренда {symbol}: {e}")
        return "neutral"

# ================== ОТКРЫТИЕ ПОЗИЦИИ ==================
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
            log.error(f"  Некорректное количество {qty}")
            return None

        # Открываем рыночный ордер
        order = exchange.create_market_order(symbol, side, qty)
        entry_price = order.get("average", price)
        if entry_price is None:
            entry_price = price

        # Ждём, чтобы позиция точно появилась в системе
        time.sleep(2)

        log.info(f"  📈 {side.upper()} {symbol} | маржа={margin_usdt:.2f}U | qty={qty} | вход={entry_price:.8f}")
        return {
            "symbol": symbol,
            "side": side,
            "margin": margin_usdt,
            "entry_price": entry_price,
            "qty": qty,
            "open_time": time.time(),
        }
    except Exception as e:
        log.error(f"  ❌ Ошибка открытия: {e}")
        return None

# ================== МОНИТОРИНГ ПОЗИЦИИ ПО ТАЙМЕРУ ==================
def мониторить_позицию(pos_info: dict, expire_sec: int) -> str:
    """
    Ждёт заданное количество секунд, затем закрывает позицию по рынку.
    Возвращает 'profit' если P&L положительный, иначе 'loss'.
    """
    symbol = pos_info["symbol"]
    side = pos_info["side"]
    entry = pos_info["entry_price"]
    qty = pos_info["qty"]
    open_time = pos_info["open_time"]

    deadline = open_time + expire_sec
    log.info(f"  ⏳ Позиция удерживается {expire_sec} сек (до {datetime.fromtimestamp(deadline).strftime('%H:%M:%S')})")

    # Ждём до таймаута, не закрывая досрочно
    while time.time() < deadline:
        time.sleep(5)

    # Время истекло – принудительно закрываем по рынку
    try:
        close_side = "sell" if side == "buy" else "buy"
        exchange.create_market_order(symbol, close_side, qty, params={"reduceOnly": True})

        # Дадим время на обновление баланса
        time.sleep(1)
        cur = exchange.fetch_ticker(symbol)["last"]
        pnl_pct = ((cur - entry) / entry * 100) if side == "buy" else ((entry - cur) / entry * 100)
        log.info(f"  ⏰ Экспирация! Закрыто. P&L={pnl_pct:+.2f}%")
        return "profit" if pnl_pct > 0 else "loss"
    except Exception as e:
        log.error(f"  Ошибка закрытия по таймеру: {e}")
        # Пытаемся ещё раз продать по рынку (если позиция всё ещё открыта)
        try:
            exchange.create_market_order(symbol, close_side, qty, params={"reduceOnly": True})
            cur = exchange.fetch_ticker(symbol)["last"]
            pnl_pct = ((cur - entry) / entry * 100) if side == "buy" else ((entry - cur) / entry * 100)
            return "profit" if pnl_pct > 0 else "loss"
        except:
            return "loss"

# ================== ДНЕВНЫЕ ЛИМИТЫ ==================
def проверить_дневные_лимиты() -> bool:
    today = datetime.now().strftime("%Y-%m-%d")
    if stats["последняя_дата_сброса"] != today:
        stats["ежедневный_убыток"] = 0.0
        stats["убыточные_серии_сегодня"] = 0
        stats["последняя_дата_сброса"] = today
        сохранить_состояние()
    if stats["ежедневный_убыток"] > (stats["депозит_старт"] * MAX_DAILY_LOSS_PCT / 100):
        log.warning(f"  🛑 Дневной лимит убытка достигнут. Пауза до завтра.")
        return False
    if stats["убыточные_серии_сегодня"] >= MAX_LOSING_SERIES_PER_DAY:
        elapsed = time.time() - stats["последняя_убыточная_серия_время"]
        if elapsed < 3600:
            log.warning(f"  🛑 {MAX_LOSING_SERIES_PER_DAY} убыточные серии. Пауза {3600 - int(elapsed)} сек.")
            return False
        else:
            stats["убыточные_серии_сегодня"] = 0
            сохранить_состояние()
    return True

# ================== ОТЧЁТ ==================
def печатать_отчёт():
    bal = баланс_usdt()
    start = stats["депозит_старт"]
    delta = bal - start
    pct = (delta / start * 100) if start > 0 else 0
    log.info("=" * 60)
    log.info(f"  📊 ОТЧЁТ | Депозит: {start:.2f} → {bal:.2f} ({delta:+.2f}, {pct:+.1f}%)")
    log.info(f"  Сделок: {stats['сделок_всего']} | Прибыльных: {stats['прибыльных']} | Убыточных: {stats['убыточных']}")
    log.info(f"  P&L: {stats['прибыль_usdt'] - stats['убыток_usdt']:+.4f} USDT")
    log.info("=" * 60)
    stats["последний_отчёт"] = time.time()
    сохранить_состояние()

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
    log.info("  🔄 БОТ 'BINARY FLIP MARTINGALE' (фиксированное время экспирации)")
    log.info(f"  Маржа: {INITIAL_MARGIN} -> {MAX_MARGIN} | Множитель {MARTINGALE_FACTOR}")
    log.info(f"  Таймфреймы экспирации: {EXPIRE_TIMES} сек ({EXPIRE_TIMES[0]//60}, {EXPIRE_TIMES[1]//60}, {EXPIRE_TIMES[2]//60} мин)")
    log.info(f"  Фильтр тренда: {'вкл' if USE_TREND_FILTER else 'выкл'}")
    log.info(f"  Дневной лимит убытка: {MAX_DAILY_LOSS_PCT}% | Макс убыточных серий: {MAX_LOSING_SERIES_PER_DAY}")
    log.info(f"  Баланс: {баланс:.2f} USDT")
    log.info("=" * 60)

    while True:
        try:
            if not проверить_дневные_лимиты():
                time.sleep(600)
                continue

            # Выбор монеты с трендом (если фильтр включён)
            best_pair = None
            best_trend = None
            for sym in SYMBOLS:
                trend = определить_тренд(sym)
                if trend == "up" or trend == "down":
                    best_pair = sym
                    best_trend = trend
                    break
            if best_pair is None:
                log.info("  Нет монеты с определённым трендом, ждём...")
                time.sleep(SCAN_INTERVAL)
                continue

            # Начальное направление (по тренду)
            direction = "buy" if best_trend == "up" else "sell"
            log.info(f"  🧭 {best_pair.split(':')[0]}: тренд {'вверх' if direction=='buy' else 'вниз'}, торгуем {direction.upper()}")

            # Серия переворотов
            margin = INITIAL_MARGIN
            flip = 0
            current_side = direction
            series_profit = False

            while flip <= MAX_FLIPS:
                if баланс_usdt() < margin * 1.1:
                    log.warning(f"  Недостаточно средств, серия прервана")
                    break

                expire_sec = EXPIRE_TIMES[min(flip, len(EXPIRE_TIMES)-1)]
                log.info(f"  ━━━ Сделка {flip+1} | маржа={margin:.2f}U | сторона={current_side.upper()} | экспирация={expire_sec//60} мин")

                pos = открыть_позицию(best_pair, current_side, margin)
                if pos is None:
                    break

                stats["сделок_всего"] += 1
                сохранить_состояние()

                result = мониторить_позицию(pos, expire_sec)

                # Расчёт оценочного P&L (приблизительный)
                size = margin * LEVERAGE
                fee = size * TAKER_FEE * 2
                if result == "profit":
                    # Прибыль считаем как 0.5% от объёма (минимальная прибыль, чтобы покрыть комиссии)
                    pnl_est = size * 0.005 - fee
                    stats["прибыльных"] += 1
                    stats["прибыль_usdt"] += max(0, pnl_est)
                    log.info(f"  ✅ Прибыльная сделка! Оценочно +{pnl_est:.4f} USDT")
                    series_profit = True
                    break
                else:
                    pnl_est = -(size * 0.005 + fee)
                    stats["убыточных"] += 1
                    stats["убыток_usdt"] += abs(pnl_est)
                    stats["ежедневный_убыток"] += abs(pnl_est)
                    log.warning(f"  ❌ Убыточная сделка! Оценочно {pnl_est:.4f} USDT")

                    # Переворот, если не последний шаг
                    if flip < MAX_FLIPS:
                        flip += 1
                        margin = min(margin * MARTINGALE_FACTOR, MAX_MARGIN)
                        current_side = "sell" if current_side == "buy" else "buy"
                        log.info(f"  🔁 Переворот! Новая сторона {current_side.upper()}, новая маржа {margin:.2f}U")
                        continue
                    else:
                        stats["убыточные_серии_сегодня"] += 1
                        stats["последняя_убыточная_серия_время"] = time.time()
                        log.warning(f"  🚫 Серия убыточна, лимит переворотов исчерпан")
                        break

            # Серия завершена
            if not series_profit:
                log.warning("  Серия убыточна")
            else:
                log.info("  Серия прибыльна – сброс мартингейла")

            log.info("  🎯 Серия завершена. Пауза 30 сек")
            time.sleep(30)
            полная_инвентаризация()

            # Отчёт по времени
            if time.time() - stats["последний_отчёт"] >= REPORT_INTERVAL:
                печатать_отчёт()

        except Exception as e:
            log.error(f"Глобальная ошибка: {e}", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    main()
