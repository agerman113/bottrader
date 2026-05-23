import os
import time
import json
import logging
import requests
import ccxt
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# ================== НАСТРОЙКИ ==================
SYMBOLS = [
    "PEPE/USDT", "DOGE/USDT", "SHIB/USDT", "FLOKI/USDT", "BONK/USDT",
    "WIF/USDT",  "MEME/USDT", "BOME/USDT", "DOGS/USDT",
    "SOL/USDT",  "AVAX/USDT", "LTC/USDT",  "LINK/USDT",
    "DOT/USDT",  "ADA/USDT",  "TRX/USDT",  "XRP/USDT",  "TON/USDT",
]

MARTINGALE_FACTOR  = 1.35
MAX_STEPS          = 2
TP_PERCENT         = 1.2   # ИСПРАВЛЕНО: было 0.8% — меньше комиссий (0.2% туда-обратно) почти не остаётся
SL_PERCENT         = 1.5   # ИСПРАВЛЕНО: было 1.0% — соотношение риск/прибыль хотя бы 1:0.8
TIMEFRAME_TA       = "5m"
TIMEFRAME_TREND    = "1h"
SCAN_INTERVAL      = 300
MIN_SCORE          = 55    # ИСПРАВЛЕНО: было 40 — бот торговал при любом рынке
TRADE_TIMEOUT      = 900   # ИСПРАВЛЕНО: было 600 — давать больше времени для TP
TRADE_MAX_LIFETIME = 86400
REPORT_INTERVAL    = 1800
STATE_FILE         = "state.json"

MIN_DEAL_AMOUNT = 6.0
MAX_DEAL_AMOUNT = 20.0

# Комиссия Bybit spot (maker/taker)
BYBIT_FEE = 0.001  # 0.1%

# ================== ЛОГИРОВАНИЕ ==================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ================== БИРЖА ==================
exchange = ccxt.bybit({
    "apiKey":    os.getenv("BYBIT_API_KEY"),
    "secret":    os.getenv("BYBIT_API_SECRET"),
    "enableRateLimit": True,
    "options": {"defaultType": "spot"},
})

# ================== СТАТИСТИКА ==================
stats = {
    "запусков":           0,
    "сделок_всего":       0,
    "тейкпрофит":         0,
    "стоплосс":           0,
    "таймаут":            0,
    "дедлайн_24ч":        0,
    "мартингейл_шагов":   0,
    "прибыль_usdt":       0.0,
    "убыток_usdt":        0.0,
    "депозит_старт":      0.0,
    "депозит_текущий":    0.0,
    "старт_время":        "",
    "последний_отчёт":    0.0,
}

def сохранить_состояние():
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning(f"Не удалось сохранить состояние: {e}")

def загрузить_состояние():
    global stats
    if not os.path.exists(STATE_FILE):
        return False
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        for key in ["сделок_всего", "тейкпрофит", "стоплосс", "таймаут",
                    "дедлайн_24ч", "мартингейл_шагов", "прибыль_usdt",
                    "убыток_usdt", "депозит_старт", "старт_время"]:
            if key in saved:
                stats[key] = saved[key]
        log.info(f"  Состояние восстановлено из {STATE_FILE}")
        return True
    except Exception as e:
        log.warning(f"Не удалось загрузить состояние: {e}")
        return False

# ================== ИНВЕНТАРИЗАЦИЯ ==================
def отменить_все_ордера():
    log.info("  🗑️  Отмена всех открытых ордеров...")
    try:
        ордера = exchange.fetch_open_orders()
        for ордер in ордера:
            try:
                exchange.cancel_order(ордер["id"], ордер["symbol"])
                log.info(f"    Отменён ордер {ордер['id']} на {ордер['symbol']}")
            except Exception as e:
                log.warning(f"    Не удалось отменить {ордер['id']}: {e}")
    except Exception as e:
        log.warning(f"  Ошибка при получении ордеров: {e}")

def продать_все_монеты():
    log.info("  💱 Продажа всех монет (кроме USDT)...")
    try:
        баланс = exchange.fetch_balance()
        свободные = баланс["free"]
        for монета, количество in свободные.items():
            if монета == "USDT" or количество == 0:
                continue
            пара = f"{монета}/USDT"
            try:
                ticker = exchange.fetch_ticker(пара)
                сумма_usdt = количество * ticker['last']
                if сумма_usdt < 1.0:
                    log.info(f"    Пропускаю {количество:.6f} {монета} (сумма {сумма_usdt:.4f} USDT < 1.0)")
                    continue
                # ИСПРАВЛЕНО: точность количества перед продажей
                кол_точное = exchange.amount_to_precision(пара, количество)
                log.info(f"    Продажа {кол_точное} {монета} по рынку")
                exchange.create_market_sell_order(пара, кол_точное)
                time.sleep(0.5)
            except Exception as e:
                log.warning(f"    Не удалось продать {монета}: {e}")
    except Exception as e:
        log.warning(f"  Ошибка при получении баланса: {e}")

def полная_инвентаризация():
    log.info("🔄 Выполняю полную инвентаризацию перед торговлей...")
    отменить_все_ордера()
    time.sleep(1)
    продать_все_монеты()
    log.info("✅ Инвентаризация завершена")
    time.sleep(2)

# ================== ИНДИКАТОРЫ ==================
def _ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def _rma(s, span):
    return s.ewm(alpha=1 / span, adjust=False).mean()

def calc_rsi(close, period=14):
    d = close.diff()
    gain = d.clip(lower=0)
    loss = (-d).clip(lower=0)
    avg_gain = _rma(gain, period)
    avg_loss = _rma(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(close, fast=12, slow=26, signal=9):
    ml = _ema(close, fast) - _ema(close, slow)
    sl = _ema(ml, signal)
    return ml, sl, ml - sl

def calc_atr(df, period=14):
    hi, lo, pc = df["h"], df["l"], df["c"].shift(1)
    tr = pd.concat([hi - lo, (hi - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    return _rma(tr, period)

def calc_range_filter(df, period=200, qty=3.0):
    close = df["c"]
    rng = qty * calc_atr(df, period)
    filt = close.copy()
    for i in range(1, len(close)):
        c, r, pf = close.iloc[i], rng.iloc[i], filt.iloc[i-1]
        if c - r > pf:
            filt.iloc[i] = c - r
        elif c + r < pf:
            filt.iloc[i] = c + r
        else:
            filt.iloc[i] = pf
    up = (filt > filt.shift(1)) & (close > filt)
    down = (filt < filt.shift(1)) & (close < filt)
    return filt, filt + rng, filt - rng, up, down

def calc_supertrend(df, period=10, mult=3.0):
    atr = calc_atr(df, period)
    hl2 = (df["h"] + df["l"]) / 2
    ub = (hl2 + mult * atr).copy()
    lb = (hl2 - mult * atr).copy()
    trend = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        c, pc = df["c"].iloc[i], df["c"].iloc[i-1]
        pu, pl, pt = ub.iloc[i-1], lb.iloc[i-1], trend.iloc[i-1]
        ub.iloc[i] = ub.iloc[i] if ub.iloc[i] < pu or pc > pu else pu
        lb.iloc[i] = lb.iloc[i] if lb.iloc[i] > pl or pc < pl else pl
        if pt == 1 and c < lb.iloc[i]:
            trend.iloc[i] = -1
        elif pt == -1 and c > ub.iloc[i]:
            trend.iloc[i] = 1
        else:
            trend.iloc[i] = pt
    return trend == 1, trend == -1

def calc_stochastic(df, k=14, d=3, smooth=3):
    lo = df["l"].rolling(k).min()
    hi = df["h"].rolling(k).max()
    ks = (100 * (df["c"] - lo) / (hi - lo + 1e-10)).rolling(smooth).mean()
    return ks, ks.rolling(d).mean()

def calc_qqe(close, rsi_period=14, sf=5):
    rsi = calc_rsi(close, rsi_period)
    rsi_s = _ema(rsi, sf)
    return rsi_s > 50, rsi_s < 50

def calc_hull(close, period=55):
    hma = _ema(2 * _ema(close, period//2) - _ema(close, period), int(np.sqrt(period)))
    return hma > hma.shift(2), hma < hma.shift(2)

def calc_adx(df, period=14):
    atr = calc_atr(df, period)
    pdm = (df["h"] - df["h"].shift(1)).clip(lower=0)
    mdm = (df["l"].shift(1) - df["l"]).clip(lower=0)
    pdm = pdm.where(pdm >= mdm, 0)
    mdm = mdm.where(mdm >= pdm, 0)
    pdi = 100 * _rma(pdm, period) / atr.replace(0, np.nan)
    mdi = 100 * _rma(mdm, period) / atr.replace(0, np.nan)
    adx = _rma(100 * (pdi - mdi).abs() / (pdi + mdi + 1e-10), period)
    return adx, pdi, mdi

# ДОБАВЛЕНО: фильтр объёма (VWAP упрощённый)
def calc_vwap_deviation(df, period=20):
    """Возвращает отклонение цены от VWAP — отрицательное значит цена ниже VWAP (потенциал роста)."""
    typical = (df["h"] + df["l"] + df["c"]) / 3
    vwap = (typical * df["v"]).rolling(period).sum() / df["v"].rolling(period).sum()
    deviation_pct = (df["c"] - vwap) / vwap * 100
    return deviation_pct

# ================== ТЕХНИЧЕСКИЙ СКОР ==================
def получить_скор(symbol: str) -> dict:
    details = {}
    score = 0
    price = 0
    try:
        raw5 = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA, limit=300)
        raw1h = exchange.fetch_ohlcv(symbol, TIMEFRAME_TREND, limit=300)
        raw15 = exchange.fetch_ohlcv(symbol, "15m", limit=100)  # ДОБАВЛЕНО: средний таймфрейм
        if len(raw5) < 100 or len(raw1h) < 100:
            return {"score": 0, "details": {}, "price": 0}

        cols = ["ts", "o", "h", "l", "c", "v"]
        df5  = pd.DataFrame(raw5,  columns=cols).reset_index(drop=True)
        df1h = pd.DataFrame(raw1h, columns=cols).reset_index(drop=True)
        df15 = pd.DataFrame(raw15, columns=cols).reset_index(drop=True)
        c5, c1h, c15 = df5["c"], df1h["c"], df15["c"]

        # --- RSI (5m) ---
        rsi_val = calc_rsi(c5).iloc[-1]
        details["rsi"] = round(rsi_val, 1)
        if 25 <= rsi_val <= 40:        # перепроданность — хороший вход
            score += 25
        elif 40 < rsi_val <= 50:
            score += 10
        elif rsi_val < 25:             # экстремальная перепроданность — риск продолжения
            score += 15
        elif rsi_val > 70:
            score -= 15                # перекупленность — не входить

        # --- RSI (1h) — дополнительный фильтр тренда ---
        rsi_1h = calc_rsi(c1h).iloc[-1]
        details["rsi_1h"] = round(rsi_1h, 1)
        if rsi_1h < 50:
            score += 5   # на часовом ещё не перекуплено
        elif rsi_1h > 65:
            score -= 10  # на часовом перекуплено — не входить

        # --- MACD (5m) ---
        ml, sl, _ = calc_macd(c5)
        macd_bull = ml.iloc[-1] > sl.iloc[-1]
        macd_cross = macd_bull and ml.iloc[-2] <= sl.iloc[-2]
        details["macd"] = "бычий" if macd_bull else "медвежий"
        score += 20 if macd_cross else (8 if macd_bull else -5)

        # --- Range Filter (5m) ---
        _, _, _, rf_up, rf_down = calc_range_filter(df5)
        details["range_filter"] = "вверх" if rf_up.iloc[-1] else "вниз"
        score += 15 if rf_up.iloc[-1] else (-10 if rf_down.iloc[-1] else 0)

        # --- Supertrend (5m) ---
        st_up, st_down = calc_supertrend(df5)
        details["supertrend"] = "вверх" if st_up.iloc[-1] else "вниз"
        score += 10 if st_up.iloc[-1] else -5

        # --- Supertrend (15m) — средний таймфрейм ---
        st_up_15, _ = calc_supertrend(df15)
        details["supertrend_15m"] = "вверх" if st_up_15.iloc[-1] else "вниз"
        score += 5 if st_up_15.iloc[-1] else -5

        # --- Hull MA (5m) ---
        hu_up, _ = calc_hull(c5)
        details["hull"] = "вверх" if hu_up.iloc[-1] else "вниз"
        score += 10 if hu_up.iloc[-1] else 0

        # --- EMA 50/200 тренд (1h) ---
        ema50  = _ema(c1h, 50).iloc[-1]
        ema200 = _ema(c1h, 200).iloc[-1]
        trend_bull = ema50 > ema200
        details["тренд_1h"] = "бычий" if trend_bull else "медвежий"
        # ИСПРАВЛЕНО: медвежий тренд на часовом — штраф, а не просто 0
        score += 10 if trend_bull else -10

        # --- EMA 20/50 тренд (15m) ---
        ema20_15 = _ema(c15, 20).iloc[-1]
        ema50_15 = _ema(c15, 50).iloc[-1]
        details["тренд_15m"] = "бычий" if ema20_15 > ema50_15 else "медвежий"
        score += 5 if ema20_15 > ema50_15 else -5

        # --- ADX (сила тренда) ---
        adx, pdi, mdi = calc_adx(df5)
        adx_val = adx.iloc[-1]
        details["adx"] = round(adx_val, 1)
        if adx_val > 25 and pdi.iloc[-1] > mdi.iloc[-1]:
            score += 10   # сильный восходящий тренд
        elif adx_val > 25 and mdi.iloc[-1] > pdi.iloc[-1]:
            score -= 10   # сильный нисходящий тренд
        elif adx_val < 15:
            score -= 5    # слабый боковой рынок — невыгодно

        # --- Stochastic ---
        k, d_val = calc_stochastic(df5)
        k_val = k.iloc[-1]
        details["stoch_k"] = round(k_val, 1)
        if k_val < 25:
            score += 10   # перепроданность
        elif k_val > 75:
            score -= 10   # перекупленность

        # --- Объём ---
        vol_avg = df5["v"].rolling(20).mean().iloc[-1]
        vol_ratio = df5["v"].iloc[-1] / (vol_avg + 1e-10)
        details["объём_ratio"] = round(vol_ratio, 2)
        if vol_ratio > 1.5:
            score += 10   # значительный рост объёма
        elif vol_ratio > 1.2:
            score += 5
        elif vol_ratio < 0.5:
            score -= 5    # объём падает — нет интереса

        # --- VWAP отклонение ---
        vwap_dev = calc_vwap_deviation(df5).iloc[-1]
        details["vwap_dev"] = round(vwap_dev, 2)
        if -3 <= vwap_dev <= -0.5:
            score += 10   # цена чуть ниже VWAP — хороший вход
        elif vwap_dev < -3:
            score += 5    # сильно ниже VWAP
        elif vwap_dev > 2:
            score -= 10   # цена сильно выше VWAP — поздно входить

        # --- QQE ---
        qqe_long, _ = calc_qqe(c5)
        details["qqe"] = "лонг" if qqe_long.iloc[-1] else "шорт"
        score += 5 if qqe_long.iloc[-1] else -5

        # ДОБАВЛЕНО: проверка, что последние 2 свечи не падающие подряд
        last_candles_bearish = (df5["c"].iloc[-1] < df5["o"].iloc[-1]) and \
                               (df5["c"].iloc[-2] < df5["o"].iloc[-2]) and \
                               (df5["c"].iloc[-3] < df5["o"].iloc[-3])
        if last_candles_bearish:
            score -= 15   # три красных свечи подряд — не входить

        score = max(0, min(100, score))
        price = df5["c"].iloc[-1]  # ИСПРАВЛЕНО: берём из уже загруженных данных, не делаем лишний запрос

    except Exception as e:
        log.warning(f"Ошибка анализа {symbol}: {e}")

    return {"score": score, "details": details, "price": price}

# ================== РАСЧЁТ РАЗМЕРА ВХОДА ==================
def рассчитать_размер_входа(баланс_usdt: float) -> float:
    # ИСПРАВЛЕНО: учитываем все шаги мартингейла, чтобы хватило на серию
    total_factor = 1 + MARTINGALE_FACTOR + MARTINGALE_FACTOR ** 2
    # Используем не более 60% баланса на всю серию мартингейла
    amount = min(баланс_usdt * 0.6 / total_factor, MAX_DEAL_AMOUNT)
    return round(max(MIN_DEAL_AMOUNT, amount), 2)

# ================== ТОРГОВЫЕ ФУНКЦИИ ==================
def баланс_монеты(symbol):
    coin = symbol.split("/")[0]
    try:
        bal = exchange.fetch_balance()
        return bal["free"].get(coin, 0.0)
    except:
        return 0.0

def баланс_usdt():
    try:
        bal = exchange.fetch_balance()
        return bal["free"].get("USDT", 0.0)
    except:
        return 0.0

def купить_рыночным(symbol, amount_usdt):
    """
    Покупает рыночным ордером. Возвращает (entry_price, qty_received).
    ИСПРАВЛЕНО: не используем встроенный TP/SL параметр (вызывает 170131),
    вместо этого ставим limit-ордер TP вручную после подтверждения покупки.
    """
    try:
        ticker = exchange.fetch_ticker(symbol)
        price  = ticker['last']
        qty_raw = amount_usdt / price

        # Приводим к точности биржи
        qty = float(exchange.amount_to_precision(symbol, qty_raw))
        if qty <= 0:
            log.error(f"  Количество {qty} <= 0, пропускаем")
            return None, None

        log.info(f"  Отправка рыночного ордера на покупку {qty} {symbol}...")
        order = exchange.create_market_buy_order(symbol, qty)

        # Ждём появления монет на балансе (до 10 сек)
        coin = symbol.split("/")[0]
        реальное_кол = 0.0
        for attempt in range(10):
            time.sleep(1)
            реальное_кол = баланс_монеты(symbol)
            if реальное_кол > qty * 0.9:
                log.info(f"  ✅ Монета появилась на балансе через {attempt+1} сек: {реальное_кол}")
                break
        else:
            log.warning(f"  ⚠️ Монета не появилась за 10 сек, используем расчётное кол-во")
            реальное_кол = qty * (1 - BYBIT_FEE)

        # Реальная цена входа из ордера (если доступна)
        entry_price = price
        try:
            if order.get("average") and float(order["average"]) > 0:
                entry_price = float(order["average"])
            elif order.get("price") and float(order.get("price", 0)) > 0:
                entry_price = float(order["price"])
        except:
            pass

        log.info(f"  📈 ПОКУПКА {реальное_кол:.6f} {coin} по ~{entry_price:.8f} ({amount_usdt:.2f} USDT)")
        return entry_price, реальное_кол

    except Exception as e:
        log.error(f"  ❌ Ошибка при покупке: {e}")
        return None, None

def выставить_tp_ордер(symbol, qty, tp_price):
    """
    ИСПРАВЛЕНО: ставим limit-ордер на продажу после покупки.
    qty — реальное количество монет (уже с учётом комиссии).
    Делаем 3 попытки с уменьшением qty на случай частичного расхода.
    """
    coin = symbol.split("/")[0]
    tp_price_str = exchange.price_to_precision(symbol, tp_price)

    for attempt in range(3):
        try:
            # Берём актуальный баланс монеты
            актуальное_кол = баланс_монеты(symbol)
            if актуальное_кол <= 0:
                log.warning(f"  TP: баланс {coin} = 0, TP не нужен")
                return None

            # Небольшая "подушка" — продаём чуть меньше, чтобы не словить 170131
            кол_для_tp = float(exchange.amount_to_precision(symbol, актуальное_кол * 0.999))
            if кол_для_tp <= 0:
                кол_для_tp = float(exchange.amount_to_precision(symbol, актуальное_кол))

            log.info(f"  Попытка {attempt+1}: выставляем TP limit {кол_для_tp} {coin} @ {tp_price_str}")
            order = exchange.create_limit_sell_order(symbol, кол_для_tp, float(tp_price_str))
            log.info(f"  🎯 TP ордер выставлен: id={order['id']}, цена={tp_price_str}")
            return order["id"]

        except Exception as e:
            log.warning(f"  Попытка {attempt+1} не удалось поставить TP: {e}")
            time.sleep(2)

    log.warning("  Не удалось выставить TP после 3 попыток — будем мониторить вручную")
    return None

def продать_по_рынку(symbol, qty, причина=""):
    """Продаёт по рынку с правильной точностью."""
    try:
        актуальное_кол = баланс_монеты(symbol)
        if актуальное_кол <= 0:
            log.info(f"  Нечего продавать по {symbol}, баланс = 0")
            return True
        # Продаём актуальный баланс, не старое qty (могло измениться)
        кол = float(exchange.amount_to_precision(symbol, актуальное_кол))
        exchange.create_market_sell_order(symbol, кол)
        log.info(f"  📉 ПРОДАЖА {кол} {symbol.split('/')[0]} по рынку{' (' + причина + ')' if причина else ''}")
        return True
    except Exception as e:
        log.warning(f"  Ошибка продажи: {e}")
        return False

def отменить_ордера_по_паре(symbol):
    try:
        ордера = exchange.fetch_open_orders(symbol)
        for ордер in ордера:
            try:
                exchange.cancel_order(ордер["id"], symbol)
                log.info(f"  🗑️  Отменён ордер {ордер['id']} на {symbol}")
            except Exception as e:
                log.warning(f"  Ошибка отмены ордера: {e}")
    except Exception as e:
        log.warning(f"  Ошибка отмены ордеров: {e}")

def мониторить_позицию(symbol, entry_price, qty, открыта_в: float, tp_order_id=None) -> str:
    """
    Следит за позицией. Приоритет: limit TP-ордер. Fallback: мониторинг цены.
    ИСПРАВЛЕНО: корректная логика определения TP (монеты исчезли с баланса).
    """
    deadline_обычный = открыта_в + TRADE_TIMEOUT
    deadline_24ч     = открыта_в + TRADE_MAX_LIFETIME
    coin = symbol.split("/")[0]

    while True:
        сейчас = time.time()

        if сейчас >= deadline_24ч:
            log.warning("  🔴 ДЕДЛАЙН 24 ЧАСА — принудительное закрытие позиции!")
            отменить_ордера_по_паре(symbol)
            продать_по_рынку(symbol, qty, "дедлайн 24ч")
            return "дедлайн_24ч"

        if сейчас >= deadline_обычный:
            log.info("  ⏰ Таймаут — закрываем позицию")
            отменить_ордера_по_паре(symbol)
            time.sleep(1)
            продать_по_рынку(symbol, qty, "таймаут")
            return "timeout"

        time.sleep(10)
        try:
            бал = баланс_монеты(symbol)

            # TP исполнен — монеты ушли
            if бал < qty * 0.05:
                log.info("  ✅ Тейк-профит исполнен!")
                return "tp"

            cur = exchange.fetch_ticker(symbol)["last"]
            pnl = (cur - entry_price) / entry_price * 100

            log.info(f"  [{symbol}] цена={cur:.8f}  P&L={pnl:+.2f}%  баланс={бал:.6f} {coin}  "
                     f"до TP={((entry_price*(1+TP_PERCENT/100))-cur)/cur*100:+.2f}%  "
                     f"до таймаута={int(deadline_обычный - сейчас)}с")

            # SL — цена упала ниже порога
            if pnl <= -SL_PERCENT:
                log.info(f"  ❌ Стоп-лосс! P&L={pnl:.2f}%")
                отменить_ордера_по_паре(symbol)
                time.sleep(0.5)
                продать_по_рынку(symbol, бал, "стоп-лосс")
                return "sl"

        except Exception as e:
            log.warning(f"  Ошибка мониторинга: {e}")

# ================== ТЕСТОВАЯ СДЕЛКА ==================
def тестовая_сделка():
    log.info("🧪 Тестовая сделка (покупка+продажа) для проверки API...")
    try:
        sym = "DOGE/USDT"
        ticker = exchange.fetch_ticker(sym)
        qty_raw = 1.5 / ticker['last']  # ~1.5 USDT (выше минимума биржи)
        qty = float(exchange.amount_to_precision(sym, qty_raw))
        log.info(f"  Покупка {qty} {sym} на ~1.5 USDT...")
        exchange.create_market_buy_order(sym, qty)
        time.sleep(3)
        # Продаём актуальный баланс
        актуальный = баланс_монеты(sym)
        if актуальный > 0:
            кол = float(exchange.amount_to_precision(sym, актуальный))
            exchange.create_market_sell_order(sym, кол)
        log.info("  ✅ Тестовая сделка успешна!")
    except Exception as e:
        log.error(f"  ❌ Тестовая сделка не удалась: {e}")
        log.error("  Бот продолжит работу, но возможны проблемы с API.")

# ================== ОТЧЁТЫ ==================
def печатать_отчёт():
    сейчас = баланс_usdt()
    старт  = stats["депозит_старт"]
    дельта = сейчас - старт
    знак   = "+" if дельта >= 0 else ""
    чистый = stats["прибыль_usdt"] - stats["убыток_usdt"]
    процент = (дельта / старт * 100) if старт > 0 else 0
    winrate = (stats["тейкпрофит"] / stats["сделок_всего"] * 100) if stats["сделок_всего"] > 0 else 0

    log.info("")
    log.info("=" * 60)
    log.info("  📊  ОТЧЁТ ЗА СЕССИЮ")
    log.info(f"  Время:                {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    log.info(f"  Работает с:           {stats['старт_время']}")
    log.info("  ──────────────────────────────────────────────────────────")
    log.info(f"  Депозит при старте:   {старт:.2f} USDT")
    log.info(f"  Баланс сейчас:        {сейчас:.2f} USDT  ({знак}{дельта:.2f})")
    log.info(f"  Изменение депозита:   {знак}{процент:.2f}%")
    log.info("  ──────────────────────────────────────────────────────────")
    log.info(f"  Сделок проведено:     {stats['сделок_всего']}")
    log.info(f"  ✅ Тейк-профит:       {stats['тейкпрофит']}  (winrate {winrate:.1f}%)")
    log.info(f"  ❌ Стоп-лосс:         {stats['стоплосс']}")
    log.info(f"  ⏰ Таймаут:           {stats['таймаут']}")
    log.info(f"  🔴 Дедлайн 24 часа:   {stats['дедлайн_24ч']}")
    log.info(f"  🔁 Шагов мартингейла: {stats['мартингейл_шагов']}")
    log.info("  ──────────────────────────────────────────────────────────")
    log.info(f"  💰 Прибыль:          +{stats['прибыль_usdt']:.4f} USDT")
    log.info(f"  💸 Убыток:           -{stats['убыток_usdt']:.4f} USDT")
    log.info(f"  📈 Чистый P&L:        {'+' if чистый >= 0 else ''}{чистый:.4f} USDT")
    log.info("=" * 60)
    log.info("")

    stats["последний_отчёт"] = time.time()
    сохранить_состояние()

# ================== ГЛАВНЫЙ ЦИКЛ ==================
def main():
    полная_инвентаризация()
    тестовая_сделка()

    восстановлен = загрузить_состояние()
    баланс_сейчас = баланс_usdt()
    stats["запусков"] += 1

    if not восстановлен or stats["депозит_старт"] == 0:
        stats["депозит_старт"] = баланс_сейчас
        stats["старт_время"]   = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

    stats["депозит_текущий"] = баланс_сейчас
    stats["последний_отчёт"] = time.time()

    log.info("")
    log.info("=" * 60)
    log.info("  🤖  БОТ ЗАПУЩЕН")
    log.info(f"  Запуск №:             {stats['запусков']}")
    log.info(f"  Дата/время:           {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    log.info(f"  Работает с:           {stats['старт_время']}")
    log.info(f"  Баланс:               {баланс_сейчас:.2f} USDT")
    log.info(f"  Стартовый депозит:    {stats['депозит_старт']:.2f} USDT")
    log.info(f"  Пар для торговли:     {len(SYMBOLS)}")
    log.info(f"  MIN_SCORE:            {MIN_SCORE}")
    log.info(f"  TP / SL:              {TP_PERCENT}% / {SL_PERCENT}%")
    log.info("=" * 60)
    log.info("")

    while True:
        try:
            if time.time() - stats["последний_отчёт"] >= REPORT_INTERVAL:
                печатать_отчёт()

            баланс = баланс_usdt()
            первый_вход = рассчитать_размер_входа(баланс)

            log.info(f"── Сканирование {len(SYMBOLS)} пар ──")
            scores = {}
            for sym in SYMBOLS:
                try:
                    res = получить_скор(sym)
                    scores[sym] = res
                    log.info(
                        f"  {sym:12s}  скор={res['score']:3d}/100  "
                        f"rsi={res['details'].get('rsi','?'):5}  "
                        f"rf={res['details'].get('range_filter','?'):5}  "
                        f"тренд={res['details'].get('тренд_1h','?')}"
                    )
                except Exception as e:
                    log.warning(f"  Ошибка скора {sym}: {e}")
                    scores[sym] = {"score": 0, "details": {}, "price": 0}

            if not scores:
                log.warning("  Нет данных по парам, ждём...")
                time.sleep(SCAN_INTERVAL)
                continue

            лучшая = max(scores, key=lambda s: scores[s]["score"])
            финальный_скор = scores[лучшая]["score"]
            цена = scores[лучшая]["price"]

            log.info(f"  ► Выбрана {лучшая}  скор={финальный_скор}  цена={цена:.8f}")

            if финальный_скор < MIN_SCORE:
                log.info(f"  Скор {финальный_скор} < порога {MIN_SCORE} — ждём {SCAN_INTERVAL} сек")
                time.sleep(SCAN_INTERVAL)
                continue

            # ИСПРАВЛЕНО: проверяем что хватает баланса хотя бы на первый вход
            if баланс < первый_вход:
                log.warning(f"  ⚠️ Баланс {баланс:.2f} USDT < первый вход {первый_вход:.2f} USDT — ждём")
                time.sleep(SCAN_INTERVAL)
                continue

            log.info(f"  ✅ Сигнал к покупке (скор {финальный_скор} >= {MIN_SCORE})")
            log.info(f"  💰 Первый вход: {первый_вход:.2f} USDT (баланс {баланс:.2f} USDT)")

            сумма = первый_вход
            шаг = 0
            успех = False

            while шаг <= MAX_STEPS:
                log.info(f"  ── Шаг мартингейла {шаг} / сумма {сумма:.2f} USDT ──")

                текущий_баланс = баланс_usdt()
                if текущий_баланс < сумма * 0.95:
                    log.warning(f"  Недостаточно USDT ({текущий_баланс:.2f}) для входа {сумма:.2f} — прерываем серию")
                    break

                время_входа = time.time()
                вход_цена, кол_во = купить_рыночным(лучшая, сумма)
                if вход_цена is None or кол_во is None or кол_во <= 0:
                    log.warning("  Покупка не удалась, прерываем серию")
                    break

                stats["сделок_всего"] += 1
                сохранить_состояние()

                # Рассчитываем TP и SL цены
                tp_цена = вход_цена * (1 + TP_PERCENT / 100)
                sl_цена = вход_цена * (1 - SL_PERCENT / 100)
                log.info(f"  🎯 Вход={вход_цена:.8f}  TP={tp_цена:.8f} (+{TP_PERCENT}%)  SL={sl_цена:.8f} (-{SL_PERCENT}%)")

                # Выставляем limit TP-ордер
                time.sleep(1)  # даём бирже время обработать покупку
                tp_order_id = выставить_tp_ордер(лучшая, кол_во, tp_цена)

                # Мониторим позицию
                результат = мониторить_позицию(лучшая, вход_цена, кол_во, время_входа, tp_order_id)

                if результат == "tp":
                    # Реальная прибыль с учётом комиссий
                    прибыль_брутто = сумма * TP_PERCENT / 100
                    комиссии = сумма * BYBIT_FEE * 2  # вход + выход
                    прибыль_нетто = прибыль_брутто - комиссии
                    stats["тейкпрофит"] += 1
                    stats["прибыль_usdt"] += max(0, прибыль_нетто)
                    log.info(f"  ✅ Прибыль зафиксирована ~{прибыль_нетто:.4f} USDT (после комиссий) — серия закончена")
                    успех = True
                    break

                elif результат == "sl":
                    убыток = сумма * SL_PERCENT / 100
                    stats["стоплосс"] += 1
                    stats["убыток_usdt"] += убыток
                    log.warning(f"  ❌ Убыток ~{убыток:.4f} USDT")
                    шаг += 1
                    if шаг <= MAX_STEPS:
                        сумма = round(сумма * MARTINGALE_FACTOR, 2)
                        stats["мартингейл_шагов"] += 1
                        log.info(f"  🔁 Мартингейл шаг {шаг} — следующая ставка {сумма:.2f} USDT")
                    else:
                        log.info("  🚫 Лимит шагов мартингейла исчерпан — серия окончена")
                        break

                elif результат == "timeout":
                    # ИСПРАВЛЕНО: таймаут = продали по рынку, убыток только комиссии
                    # Не увеличиваем ставку мартингейла — это не настоящий убыток
                    комиссии = сумма * BYBIT_FEE * 2
                    stats["таймаут"] += 1
                    stats["убыток_usdt"] += комиссии
                    log.warning(f"  ⏰ Таймаут, убыток на комиссиях ~{комиссии:.4f} USDT — серия окончена")
                    # При таймауте НЕ делаем следующий шаг мартингейла
                    break

                elif результат == "дедлайн_24ч":
                    stats["дедлайн_24ч"] += 1
                    stats["убыток_usdt"] += сумма * SL_PERCENT / 100
                    log.warning("  Серия прервана по дедлайну 24 часа")
                    break

            сохранить_состояние()
            log.info("  Серия завершена — пауза 30 сек")
            time.sleep(30)

        except Exception as e:
            log.error(f"Глобальная ошибка: {e}", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    main()
