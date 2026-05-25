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

EXPIRE_TIMES = [60, 300, 900]          # 1, 5, 15 минут

USE_TREND_FILTER = True
TIMEFRAME_TREND = "1h"
SCAN_INTERVAL = 300
MIN_BALANCE = 5.0
REPORT_INTERVAL = 1800

MAX_DAILY_LOSS_PCT = 10.0
MAX_LOSING_SERIES_PER_DAY = 3

STATE_FILE = "binary_flip_martingale.json"
LOG_FILE = "binary_bot.log"

# Комиссии (тейкер для входов и выходов)
TAKER_FEE = 0.00055        # 0.055%
# Минимальная прибыль, чтобы перекрыть комиссии (в процентах от объёма)
MIN_PROFIT_PCT = 0.12      # 0.12% (комиссии ~0.11%, плюс запас)

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
    "реальная_прибыль_usdt": 0.0,
    "реальный_убыток_usdt": 0.0,
    "депозит_старт": 0.0,
    "старт_время": "",
    "последний_отчёт": 0.0,
    "ежедневный_убыток": 0.0,
    "последняя_дата_сброса": datetime.now().strftime("%Y-%m-%d"),
    "убыточные_серии_сегодня": 0,
    "последняя_убыточная_серия_время": 0.0,
}

def сохранить_состояние():
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

def загрузить_состояние():
    global stats
    if not os.path.exists(STATE_FILE):
        return False
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        saved = json.load(f)
        for key in stats:
            if key in saved:
                stats[key] = saved[key]
    log.info(f"Состояние восстановлено из {STATE_FILE}")
    return True

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
        log.warning(f"Не удалось установить плечо: {e}")

# ================== ТРЕНД ==================
def определить_тренд(symbol: str) -> str:
    if not USE_TREND_FILTER:
        return "up"
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME_TREND, limit=200)
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
    except:
        return "neutral"

# ================== ПРОВЕРКА ВЫГОДНОСТИ ПЕРЕД ВХОДОМ ==================
def оценить_сделку(symbol: str, side: str, expire_sec: int) -> (bool, str):
    """
    Возвращает (можно_ли_входить, причина_отказа).
    Оценивает волатильность (ATR) и ожидаемое изменение цены за время экспирации.
    Требует, чтобы ожидаемое движение было не менее MIN_PROFIT_PCT (с запасом на комиссии).
    """
    try:
        # Берём свечи 5m для расчёта волатильности
        ohlcv = exchange.fetch_ohlcv(symbol, "5m", limit=50)
        df = pd.DataFrame(ohlcv, columns=["ts","o","h","l","c","v"])
        atr = (df["h"] - df["l"]).rolling(14).mean().iloc[-1]
        price = df["c"].iloc[-1]
        atr_pct = (atr / price) * 100
        # Ожидаемое движение за expire_sec (грубо: ATR * (expire_sec / 300) , т.к. 5м = 300с)
        expected_move_pct = atr_pct * (expire_sec / 300)
        # Требуемый минимум: комиссии + запас
        required_pct = MIN_PROFIT_PCT
        if expected_move_pct < required_pct:
            return False, f"волатильность {expected_move_pct:.2f}% < {required_pct}%"
        return True, "ok"
    except Exception as e:
        return False, f"ошибка оценки: {e}"

# ================== ОТКРЫТИЕ ПОЗИЦИИ ==================
def открыть_позицию(symbol: str, side: str, margin_usdt: float):
    try:
        установить_плечо(symbol, LEVERAGE)
        ticker = exchange.fetch_ticker(symbol)
        price = ticker["last"]
        if price is None or price <= 0:
            return None
        size_usdt = margin_usdt * LEVERAGE
        qty = size_usdt / price
        qty = float(exchange.amount_to_precision(symbol, qty))
        if qty <= 0:
            return None
        order = exchange.create_market_order(symbol, side, qty)
        entry_price = order.get("average", price)
        time.sleep(2)   # даём время на регистрацию позиции
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
        log.error(f"Ошибка открытия: {e}")
        return None

# ================== МОНИТОРИНГ ПО ТАЙМЕРУ ==================
def мониторить_позицию(pos_info: dict, expire_sec: int) -> str:
    symbol = pos_info["symbol"]
    side = pos_info["side"]
    entry = pos_info["entry_price"]
    qty = pos_info["qty"]
    open_time = pos_info["open_time"]
    deadline = open_time + expire_sec
    log.info(f"  ⏳ Удерживается {expire_sec} сек до {datetime.fromtimestamp(deadline).strftime('%H:%M:%S')}")

    while time.time() < deadline:
        time.sleep(5)

    # Закрываем по рынку
    try:
        close_side = "sell" if side == "buy" else "buy"
        exchange.create_market_order(symbol, close_side, qty, params={"reduceOnly": True})
        time.sleep(1)
        cur = exchange.fetch_ticker(symbol)["last"]
        pnl_pct = ((cur - entry) / entry * 100) if side == "buy" else ((entry - cur) / entry * 100)
        log.info(f"  ⏰ Экспирация! P&L={pnl_pct:+.2f}%")
        return "profit" if pnl_pct > 0 else "loss"
    except Exception as e:
        log.error(f"Ошибка закрытия: {e}")
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
        log.warning("  🛑 Дневной лимит убытка достигнут. Пауза до завтра.")
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

# ================== РАСЧЁТ РЕАЛЬНОГО P&L ПО БАЛАНСУ ==================
def обновить_pnl_реальный(баланс_до: float, баланс_после: float, было_прибыльных: bool) -> float:
    изменение = баланс_после - баланс_до
    if изменение > 0:
        stats["реальная_прибыль_usdt"] += изменение
        if было_прибыльных:
            stats["прибыльных"] += 1
        log.info(f"  💰 Реальная прибыль: +{изменение:.4f} USDT")
    else:
        stats["реальный_убыток_usdt"] += abs(изменение)
        if not было_прибыльных:
            stats["убыточных"] += 1
        log.warning(f"  💸 Реальный убыток: {изменение:.4f} USDT")
        stats["ежедневный_убыток"] += abs(изменение)
    return изменение

# ================== ОТЧЁТ ==================
def печатать_отчёт():
    bal = баланс_usdt()
    start = stats["депозит_старт"]
    delta = bal - start
    pct = (delta / start * 100) if start > 0 else 0
    log.info("=" * 60)
    log.info(f"  📊 ОТЧЁТ | Депозит: {start:.2f} → {bal:.2f} ({delta:+.2f}, {pct:+.1f}%)")
    log.info(f"  Сделок: {stats['сделок_всего']} | По оценке бота: Прибыльных {stats['прибыльных']} | Убыточных {stats['убыточных']}")
    log.info(f"  Реальный P&L: {stats['реальная_прибыль_usdt'] - stats['реальный_убыток_usdt']:+.4f} USDT")
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
    log.info("  🔄 БОТ 'BINARY FLIP MARTINGALE' (с контролем комиссий)")
    log.info(f"  Маржа: {INITIAL_MARGIN} -> {MAX_MARGIN} | Множитель {MARTINGALE_FACTOR}")
    log.info(f"  Экспирация: {EXPIRE_TIMES} сек")
    log.info(f"  Минимальный профит для входа: {MIN_PROFIT_PCT}% (комиссии ~0.11%)")
    log.info(f"  Баланс: {баланс:.2f} USDT")
    log.info("=" * 60)

    while True:
        try:
            if not проверить_дневные_лимиты():
                time.sleep(600)
                continue

            # Выбор монеты с трендом
            best_pair = None
            best_trend = None
            for sym in SYMBOLS:
                trend = определить_тренд(sym)
                if trend in ("up", "down"):
                    best_pair = sym
                    best_trend = trend
                    break
            if best_pair is None:
                log.info("  Нет монеты с трендом, ждём...")
                time.sleep(SCAN_INTERVAL)
                continue

            direction = "buy" if best_trend == "up" else "sell"
            log.info(f"  🧭 {best_pair.split(':')[0]}: тренд {'вверх' if direction=='buy' else 'вниз'} -> торгуем {direction.upper()}")

            margin = INITIAL_MARGIN
            flip = 0
            current_side = direction
            series_profit = False
            balance_before_series = баланс_usdt()

            while flip <= MAX_FLIPS:
                if баланс_usdt() < margin * 1.1:
                    log.warning("  Недостаточно средств, прерываем серию")
                    break

                expire_sec = EXPIRE_TIMES[min(flip, len(EXPIRE_TIMES)-1)]
                # Проверка выгодности перед входом
                can_enter, reason = оценить_сделку(best_pair, current_side, expire_sec)
                if not can_enter:
                    log.warning(f"  🚫 Сделка невыгодна: {reason} → пропускаем эту возможность, ждём следующую")
                    time.sleep(SCAN_INTERVAL)
                    continue  # не выходим из цикла, а пробуем другую монету или ждём

                log.info(f"  ━━━ Сделка {flip+1} | маржа={margin:.2f}U | сторона={current_side.upper()} | экспирация={expire_sec//60} мин")
                pos = открыть_позицию(best_pair, current_side, margin)
                if pos is None:
                    break

                stats["сделок_всего"] += 1
                сохранить_состояние()

                result = мониторить_позицию(pos, expire_sec)

                # Реальный P&L после закрытия
                баланс_после = баланс_usdt()
                изменение = баланс_после - balance_before_series
                if изменение > 0:
                    stats["реальная_прибыль_usdt"] += изменение
                    stats["прибыльных"] += 1
                    log.info(f"  ✅ Прибыльная серия! +{изменение:.4f} USDT")
                    series_profit = True
                    break
                else:
                    stats["реальный_убыток_usdt"] += abs(изменение)
                    stats["убыточных"] += 1
                    stats["ежедневный_убыток"] += abs(изменение)
                    log.warning(f"  ❌ Убыточная сделка! -{abs(изменение):.4f} USDT")

                    if flip < MAX_FLIPS:
                        flip += 1
                        margin = min(margin * MARTINGALE_FACTOR, MAX_MARGIN)
                        current_side = "sell" if current_side == "buy" else "buy"
                        log.info(f"  🔁 Переворот! Новая сторона {current_side.upper()}, новая маржа {margin:.2f}U")
                        continue
                    else:
                        stats["убыточные_серии_сегодня"] += 1
                        stats["последняя_убыточная_серия_время"] = time.time()
                        log.warning("  🚫 Серия убыточна, лимит переворотов исчерпан")
                        break

            log.info("  🎯 Серия завершена. Пауза 30 сек")
            time.sleep(30)
            полная_инвентаризация()

            if time.time() - stats["последний_отчёт"] >= REPORT_INTERVAL:
                печатать_отчёт()

        except Exception as e:
            log.error(f"Глобальная ошибка: {e}", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    main()
