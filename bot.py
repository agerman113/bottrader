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

# Минимальный ожидаемый профит в процентах от объёма позиции (после комиссий)
# Чтобы сделка была экономически целесообразной
MIN_EXPECTED_PROFIT_PCT = 0.15   # 0.15% от объёма (с учётом комиссий)

STATE_FILE = "binary_flip_martingale.json"
LOG_FILE = "binary_bot.log"

TAKER_FEE = 0.00055
MAKER_FEE = 0.0002
FEE = TAKER_FEE  # используем тейкер

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
    "история_pnl": [],   # список последних 10 P&L
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

# ================== ОЦЕНКА ЭКОНОМИЧЕСКОЙ ЦЕЛЕСООБРАЗНОСТИ ==================
def сделка_выгодна(маржа_usdt: float) -> bool:
    """
    Проверяет, есть ли смысл входить в сделку с данной маржой.
    Учитывает комиссии и историческую прибыльность.
    """
    # 1. Комиссии составляют 2*FEE от объёма позиции (вход+выход)
    size = маржа_usdt * LEVERAGE
    fee = size * FEE * 2
    # Минимальный необходимый профит для безубыточности
    min_profit_needed = fee + 0.01  # +0.01 USDT на всякий случай
    # Профит от движения в 0.1% от объёма
    profit_from_01 = size * 0.001 - fee
    if profit_from_01 < min_profit_needed:
        log.warning(f"  ⚠️ При текущей марже {маржа_usdt}U комиссии {fee:.4f}U съедают прибыль даже при движении 0.1%. Сделка экономически невыгодна.")
        return False

    # 2. Историческая эффективность: если последние 5 сделок убыточны в среднем, не входить
    if len(stats["история_pnl"]) >= 5:
        avg_pnl = sum(stats["история_pnl"][-5:]) / 5
        if avg_pnl < 0:
            log.warning(f"  ⚠️ Средний P&L последних 5 сделок = {avg_pnl:.4f} USDT (убыточный). Пауза до улучшения.")
            return False
    return True

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

        # Запоминаем баланс до сделки
        баланс_до = баланс_usdt()

        order = exchange.create_market_order(symbol, side, qty)
        entry_price = order.get("average", price)
        if entry_price is None:
            entry_price = price

        time.sleep(2)  # ждём регистрации позиции

        log.info(f"  📈 {side.upper()} {symbol} | маржа={margin_usdt:.2f}U | qty={qty} | вход={entry_price:.8f}")
        return {
            "symbol": symbol,
            "side": side,
            "margin": margin_usdt,
            "entry_price": entry_price,
            "qty": qty,
            "open_time": time.time(),
            "balance_before": баланс_до,
        }
    except Exception as e:
        log.error(f"  ❌ Ошибка открытия: {e}")
        return None

# ================== МОНИТОРИНГ И ЗАКРЫТИЕ ==================
def мониторить_и_закрыть(pos_info: dict, expire_sec: int) -> float:
    """Закрывает позицию по таймеру, возвращает реальный P&L в USDT."""
    symbol = pos_info["symbol"]
    side = pos_info["side"]
    qty = pos_info["qty"]
    open_time = pos_info["open_time"]
    баланс_до = pos_info["balance_before"]

    deadline = open_time + expire_sec
    log.info(f"  ⏳ Позиция удерживается {expire_sec} сек (до {datetime.fromtimestamp(deadline).strftime('%H:%M:%S')})")

    while time.time() < deadline:
        time.sleep(5)

    # Закрываем по рынку
    try:
        close_side = "sell" if side == "buy" else "buy"
        exchange.create_market_order(symbol, close_side, qty, params={"reduceOnly": True})
        time.sleep(1)
        баланс_после = баланс_usdt()
        pnl = баланс_после - баланс_до
        cur = exchange.fetch_ticker(symbol)["last"]
        entry = pos_info["entry_price"]
        pnl_pct = ((cur - entry) / entry * 100) if side == "buy" else ((entry - cur) / entry * 100)
        log.info(f"  ⏰ Экспирация! Закрыто. P&L={pnl_pct:+.2f}% ({pnl:+.4f} USDT)")
        return pnl
    except Exception as e:
        log.error(f"  Ошибка закрытия: {e}")
        # Пытаемся ещё раз
        try:
            exchange.create_market_order(symbol, close_side, qty, params={"reduceOnly": True})
            баланс_после = баланс_usdt()
            pnl = баланс_после - баланс_до
            return pnl
        except:
            return 0.0

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
    winrate = (stats["прибыльных"] / stats["сделок_всего"] * 100) if stats["сделок_всего"] > 0 else 0
    log.info("=" * 60)
    log.info(f"  📊 ОТЧЁТ | Депозит: {start:.2f} → {bal:.2f} ({delta:+.2f}, {pct:+.1f}%)")
    log.info(f"  Сделок: {stats['сделок_всего']} | Прибыльных: {stats['прибыльных']} | Убыточных: {stats['убыточных']} | Winrate: {winrate:.1f}%")
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
    log.info("  🔄 БОТ 'BINARY FLIP MARTINGALE' (с контролем комиссий)")
    log.info(f"  Маржа: {INITIAL_MARGIN} -> {MAX_MARGIN} | Множитель {MARTINGALE_FACTOR}")
    log.info(f"  Таймфреймы экспирации: {EXPIRE_TIMES} сек")
    log.info(f"  Фильтр тренда: {'вкл' if USE_TREND_FILTER else 'выкл'}")
    log.info(f"  Дневной лимит убытка: {MAX_DAILY_LOSS_PCT}% | Макс убыточных серий: {MAX_LOSING_SERIES_PER_DAY}")
    log.info(f"  Комиссия: {FEE*100:.3f}% | Мин. выгодный профит: {MIN_EXPECTED_PROFIT_PCT}% от объёма")
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
                if trend == "up" or trend == "down":
                    best_pair = sym
                    best_trend = trend
                    break
            if best_pair is None:
                log.info("  Нет монеты с определённым трендом, ждём...")
                time.sleep(SCAN_INTERVAL)
                continue

            direction = "buy" if best_trend == "up" else "sell"
            log.info(f"  🧭 {best_pair.split(':')[0]}: тренд {'вверх' if direction=='buy' else 'вниз'}, торгуем {direction.upper()}")

            # Серия переворотов
            margin = INITIAL_MARGIN
            flip = 0
            current_side = direction
            series_profit = False
            total_series_pnl = 0.0

            while flip <= MAX_FLIPS:
                # Проверка экономической целесообразности
                if not сделка_выгодна(margin):
                    log.warning(f"  Сделка с маржой {margin:.2f}U невыгодна, серия прервана")
                    break

                if баланс_usdt() < margin * 1.1:
                    log.warning(f"  Недостаточно средств, серия прервана")
                    break

                expire_sec = EXPIRE_TIMES[min(flip, len(EXPIRE_TIMES)-1)]
                log.info(f"  ━━━ Сделка {flip+1} | маржа={margin:.2f}U | сторона={current_side.upper()} | экспирация={expire_sec//60} мин")

                pos = открыть_позицию(best_pair, current_side, margin)
                if pos is None:
                    break

                stats["сделок_всего"] += 1
                pnl = мониторить_и_закрыть(pos, expire_sec)

                # Обновляем историю P&L
                stats["история_pnl"].append(pnl)
                if len(stats["история_pnl"]) > 20:
                    stats["история_pnl"] = stats["история_pnl"][-20:]

                if pnl > 0:
                    stats["прибыльных"] += 1
                    stats["прибыль_usdt"] += pnl
                    log.info(f"  ✅ Прибыльная сделка! P&L = {pnl:+.4f} USDT")
                    total_series_pnl += pnl
                    series_profit = True
                    break
                else:
                    stats["убыточных"] += 1
                    stats["убыток_usdt"] += abs(pnl)
                    stats["ежедневный_убыток"] += abs(pnl)
                    total_series_pnl += pnl
                    log.warning(f"  ❌ Убыточная сделка! P&L = {pnl:+.4f} USDT")

                    # Переворот
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

            # После серии
            if series_profit:
                log.info(f"  ✅ Серия завершена с прибылью {total_series_pnl:+.4f} USDT")
            else:
                log.warning(f"  ❌ Серия убыточна, общий P&L = {total_series_pnl:+.4f} USDT")

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
