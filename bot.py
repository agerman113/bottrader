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
TP_PERCENT         = 0.8
SL_PERCENT         = 1.0
TIMEFRAME_TA       = "5m"
TIMEFRAME_TREND    = "1h"
SCAN_INTERVAL      = 300
MIN_SCORE          = 40               # ← изменено с 50 на 40
TRADE_TIMEOUT      = 600
TRADE_MAX_LIFETIME = 86400
REPORT_INTERVAL    = 1800
STATE_FILE         = "state.json"

# Минимальная сумма сделки (USDT) – для входа и для тейк-профита
MIN_DEAL_AMOUNT = 6.0                 # увеличено с 5 до 6 для надёжности
MAX_DEAL_AMOUNT = 20.0

USE_AI = False

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
                if сумма_usdt < 0.5:
                    log.info(f"    Пропускаю {количество:.6f} {монета} (сумма {сумма_usdt:.4f} USDT < 0.5)")
                    continue
                log.info(f"    Продажа {количество:.6f} {монета} по рынку")
                exchange.create_market_sell_order(пара, количество)
                time.sleep(0.3)
            except Exception as e:
                log.warning(f"    Не удалось продать {монета}: {e}")
    except Exception as e:
        log.warning(f"  Ошибка при получении баланса: {e}")

def полная_инвентаризация():
    log.info("🔄 Выполняю полную инвентаризацию перед торговлей...")
    отменить_все_ордера()
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
    return 100 - (100 / (1 + avg_gain / avg_loss.replace(0, np.nan)))

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
    pdm[pdm < mdm] = 0
    mdm[mdm < pdm] = 0
    pdi = 100 * _rma(pdm, period) / atr.replace(0, np.nan)
    mdi = 100 * _rma(mdm, period) / atr.replace(0, np.nan)
    adx = _rma(100 * (pdi - mdi).abs() / (pdi + mdi + 1e-10), period)
    return adx, pdi, mdi

# ================== ТЕХНИЧЕСКИЙ СКОР ==================
def получить_скор(symbol: str) -> dict:
    details = {}
    score = 0
    price = 0
    try:
        raw5 = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA, limit=300)
        raw1h = exchange.fetch_ohlcv(symbol, TIMEFRAME_TREND, limit=300)
        if len(raw5) < 100 or len(raw1h) < 100:
            return {"score": 0, "details": {}, "price": 0}

        cols = ["ts", "o", "h", "l", "c", "v"]
        df5 = pd.DataFrame(raw5, columns=cols).reset_index(drop=True)
        df1h = pd.DataFrame(raw1h, columns=cols).reset_index(drop=True)
        c5, c1h = df5["c"], df1h["c"]

        # RSI
        rsi_val = calc_rsi(c5).iloc[-1]
        details["rsi"] = round(rsi_val, 1)
        if rsi_val < 30:
            score += 20
        elif rsi_val < 45:
            score += 10
        elif rsi_val > 70:
            score -= 5
        else:
            score += 5

        # MACD
        ml, sl, _ = calc_macd(c5)
        macd_bull = ml.iloc[-1] > sl.iloc[-1]
        macd_cross = macd_bull and ml.iloc[-2] <= sl.iloc[-2]
        details["macd"] = "бычий" if macd_bull else "медвежий"
        score += 15 if macd_cross else (8 if macd_bull else 0)

        # Range Filter
        _, _, _, rf_up, _ = calc_range_filter(df5)
        details["range_filter"] = "вверх" if rf_up.iloc[-1] else "вниз"
        score += 15 if rf_up.iloc[-1] else 0

        # Supertrend
        st_up, _ = calc_supertrend(df5)
        details["supertrend"] = "вверх" if st_up.iloc[-1] else "вниз"
        score += 10 if st_up.iloc[-1] else 0

        # Hull
        hu_up, _ = calc_hull(c5)
        details["hull"] = "вверх" if hu_up.iloc[-1] else "вниз"
        score += 10 if hu_up.iloc[-1] else 0

        # Тренд 1h
        ema50 = _ema(c1h, 50).iloc[-1]
        ema200 = _ema(c1h, 200).iloc[-1]
        trend_bull = ema50 > ema200
        details["тренд_1h"] = "бычий" if trend_bull else "медвежий"
        score += 10 if trend_bull else 0

        # ADX
        adx, pdi, mdi = calc_adx(df5)
        details["adx"] = round(adx.iloc[-1], 1)
        if adx.iloc[-1] > 20 and pdi.iloc[-1] > mdi.iloc[-1]:
            score += 5

        # Stochastic
        k, _ = calc_stochastic(df5)
        details["stoch_k"] = round(k.iloc[-1], 1)
        if k.iloc[-1] < 30:
            score += 5

        # Объём
        vol_surge = df5["v"].iloc[-1] > df5["v"].rolling(20).mean().iloc[-1] * 1.2
        details["объём_всплеск"] = vol_surge
        score += 5 if vol_surge else 0

        # QQE
        qqe_long, _ = calc_qqe(c5)
        details["qqe"] = "лонг" if qqe_long.iloc[-1] else "шорт"
        score += 5 if qqe_long.iloc[-1] else 0

        score = max(0, min(100, score))
        price = exchange.fetch_ticker(symbol)["last"]

    except Exception as e:
        log.warning(f"Ошибка анализа {symbol}: {e}")

    return {"score": score, "details": details, "price": price}

# ================== РАСЧЁТ РАЗМЕРА ВХОДА ==================
def рассчитать_размер_входа(баланс_usdt: float) -> float:
    total_factor = 1 + MARTINGALE_FACTOR + MARTINGALE_FACTOR ** 2
    amount = min(баланс_usdt * 0.15, баланс_usdt / total_factor * 0.9)
    return round(max(MIN_DEAL_AMOUNT, min(MAX_DEAL_AMOUNT, amount)), 2)

# ================== ИИ (ОТКЛЮЧЁН) ==================
def спросить_ии(symbol, price, score, details) -> tuple:
    if not USE_AI:
        return "buy", 1.0, "ИИ отключён"
    return "buy", 1.0, "ИИ отключён"

# ================== ТОРГОВЫЕ ФУНКЦИИ (С ИСПРАВЛЕНИЯМИ) ==================
def купить(symbol, amount_usdt):
    price = exchange.fetch_ticker(symbol)["last"]
    qty = amount_usdt / price
    order = exchange.create_market_buy_order(symbol, qty)
    log.info(f"  📈 ПОКУПКА {qty:.6f} {symbol.split('/')[0]} по ~{price:.8f} ({amount_usdt:.2f} USDT)")
    time.sleep(1.5)   # Ждём обновления баланса
    return price, qty

def поставить_тп(symbol, qty, entry_price):
    tp_price = entry_price * (1 + TP_PERCENT / 100)
    стоимость_ордера = tp_price * qty
    if стоимость_ордера < MIN_DEAL_AMOUNT:
        log.warning(f"  Сумма TP ({стоимость_ордера:.2f} USDT) ниже минимальной {MIN_DEAL_AMOUNT}. Продаю по рынку.")
        продать_по_рынку(symbol, qty, "TP малая сумма")
        return
    for попытка in range(3):
        try:
            exchange.create_limit_sell_order(symbol, qty, tp_price)
            log.info(f"  🎯 Тейк-профит установлен на {tp_price:.8f}")
            return
        except Exception as e:
            log.warning(f"  Попытка {попытка+1} не удалось поставить TP: {e}")
            time.sleep(1)
    # Если все попытки не удались – продаём по рынку
    log.warning("  Не удалось выставить TP после 3 попыток, продаю по рынку")
    продать_по_рынку(symbol, qty, "ошибка TP")

def продать_по_рынку(symbol, qty, причина=""):
    try:
        time.sleep(1)   # Даём время обновиться балансу
        exchange.create_market_sell_order(symbol, qty)
        log.info(f"  📉 ПРОДАЖА {qty:.6f} {symbol.split('/')[0]} по рынку{' (' + причина + ')' if причина else ''}")
    except Exception as e:
        log.warning(f"  Ошибка продажи: {e}")

def отменить_ордера_по_паре(symbol):
    try:
        ордера = exchange.fetch_open_orders(symbol)
        for ордер in ордера:
            exchange.cancel_order(ордер["id"], symbol)
            log.info(f"  🗑️  Отменён ордер {ордер['id']} на {symbol}")
    except Exception as e:
        log.warning(f"  Ошибка отмены ордеров: {e}")

def баланс_монеты(symbol):
    return exchange.fetch_balance()["free"].get(symbol.split("/")[0], 0.0)

def баланс_usdt():
    return exchange.fetch_balance()["free"].get("USDT", 0.0)

def мониторить_позицию(symbol, entry_price, qty, открыта_в: float) -> str:
    deadline_обычный = открыта_в + TRADE_TIMEOUT
    deadline_24ч = открыта_в + TRADE_MAX_LIFETIME

    while True:
        сейчас = time.time()
        if сейчас >= deadline_24ч:
            log.warning("  🔴 ДЕДЛАЙН 24 ЧАСА — принудительное закрытие позиции!")
            отменить_ордера_по_паре(symbol)
            остаток = баланс_монеты(symbol)
            if остаток > 0:
                продать_по_рынку(symbol, остаток, "дедлайн 24ч")
            cur = exchange.fetch_ticker(symbol)["last"]
            pnl = (cur - entry_price) / entry_price * 100
            log.warning(f"  Итог за 24ч: {'+' if pnl >= 0 else ''}{pnl:.2f}% от цены входа")
            return "дедлайн_24ч"

        if сейчас >= deadline_обычный:
            log.info("  ⏰ Таймаут 10 мин — закрываем позицию")
            отменить_ордера_по_паре(symbol)
            остаток = баланс_монеты(symbol)
            if остаток > 0:
                продать_по_рынку(symbol, остаток, "таймаут 10 мин")
            return "timeout"

        time.sleep(10)
        try:
            бал = баланс_монеты(symbol)
            if бал < qty * 0.05:
                log.info("  ✅ Тейк-профит исполнен!")
                return "tp"

            cur = exchange.fetch_ticker(symbol)["last"]
            pnl = (cur - entry_price) / entry_price * 100
            if pnl <= -SL_PERCENT:
                log.info(f"  ❌ Стоп-лосс! Убыток {pnl:.2f}%")
                отменить_ордера_по_паре(symbol)
                продать_по_рынку(symbol, бал, "стоп-лосс")
                return "sl"
        except Exception as e:
            log.warning(f"  Ошибка мониторинга: {e}")

# ================== ОТЧЁТЫ ==================
def печатать_отчёт():
    сейчас = баланс_usdt()
    старт = stats["депозит_старт"]
    дельта = сейчас - старт
    знак = "+" if дельта >= 0 else ""
    чистый = stats["прибыль_usdt"] - stats["убыток_usdt"]
    процент = (дельта / старт * 100) if старт > 0 else 0

    log.info("")
    log.info("=" * 58)
    log.info("  📊  ОТЧЁТ ЗА СЕССИЮ")
    log.info(f"  Время:                {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    log.info(f"  Работает с:           {stats['старт_время']}")
    log.info("  ──────────────────────────────────────────────────")
    log.info(f"  Депозит при старте:   {старт:.2f} USDT")
    log.info(f"  Баланс сейчас:        {сейчас:.2f} USDT  ({знак}{дельта:.2f})")
    log.info(f"  Изменение депозита:   {знак}{процент:.2f}%")
    log.info("  ──────────────────────────────────────────────────")
    log.info(f"  Сделок проведено:     {stats['сделок_всего']}")
    log.info(f"  ✅ Тейк-профит:       {stats['тейкпрофит']}")
    log.info(f"  ❌ Стоп-лосс:         {stats['стоплосс']}")
    log.info(f"  ⏰ Таймаут 10 мин:    {stats['таймаут']}")
    log.info(f"  🔴 Дедлайн 24 часа:   {stats['дедлайн_24ч']}")
    log.info(f"  🔁 Шагов мартингейла: {stats['мартингейл_шагов']}")
    log.info("  ──────────────────────────────────────────────────")
    log.info(f"  💰 Прибыль:          +{stats['прибыль_usdt']:.4f} USDT")
    log.info(f"  💸 Убыток:           -{stats['убыток_usdt']:.4f} USDT")
    log.info(f"  📈 Чистый P&L:        {'+' if чистый >= 0 else ''}{чистый:.4f} USDT")
    log.info("=" * 58)
    log.info("")

    stats["последний_отчёт"] = time.time()
    сохранить_состояние()

# ================== ГЛАВНЫЙ ЦИКЛ ==================
def main():
    полная_инвентаризация()

    восстановлен = загрузить_состояние()
    баланс_сейчас = баланс_usdt()
    stats["запусков"] += 1

    if not восстановлен or stats["депозит_старт"] == 0:
        stats["депозит_старт"] = баланс_сейчас
        stats["старт_время"] = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

    stats["депозит_текущий"] = баланс_сейчас
    stats["последний_отчёт"] = time.time()

    log.info("")
    log.info("=" * 58)
    log.info("  🤖  БОТ ЗАПУЩЕН")
    log.info(f"  Запуск №:             {stats['запусков']}")
    log.info(f"  Дата/время:           {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    log.info(f"  Работает с:           {stats['старт_время']}")
    log.info(f"  Баланс:               {баланс_сейчас:.2f} USDT")
    log.info(f"  Стартовый депозит:    {stats['депозит_старт']:.2f} USDT")
    log.info(f"  Пар для торговли:     {len(SYMBOLS)}")
    log.info(f"  MIN_SCORE:            {MIN_SCORE}")
    log.info(f"  ИИ модель:            выкл")
    log.info("=" * 58)
    log.info("")

    # Автокоррекция риска ОТКЛЮЧЕНА (чтобы MIN_SCORE не менялся)
    # скорректировать_риск(баланс_сейчас)

    while True:
        try:
            if time.time() - stats["последний_отчёт"] >= REPORT_INTERVAL:
                печатать_отчёт()

            баланс = баланс_usdt()
            первый_вход = рассчитать_размер_входа(баланс)

            log.info(f"── Сканирование {len(SYMBOLS)} пар ──")
            scores = {}
            for sym in SYMBOLS:
                res = получить_скор(sym)
                scores[sym] = res
                log.info(
                    f"  {sym:12s}  скор={res['score']:3d}/100  "
                    f"rsi={res['details'].get('rsi','?'):5}  "
                    f"rf={res['details'].get('range_filter','?'):5}  "
                    f"тренд={res['details'].get('тренд_1h','?')}"
                )

            прошедшие = {s: v for s, v in scores.items() if v["score"] >= MIN_SCORE}
            if not прошедшие:
                лучший_скор = max(scores.values(), key=lambda x: x["score"])["score"]
                log.info(f"  Ни одна пара не прошла порог {MIN_SCORE} (лучший скор: {лучший_скор}) — ждём {SCAN_INTERVAL} сек")
                time.sleep(SCAN_INTERVAL)
                continue

            лучшая = max(прошедшие, key=lambda s: прошедшие[s]["score"])
            скор = прошедшие[лучшая]["score"]
            детали = прошедшие[лучшая]["details"]
            цена = прошедшие[лучшая]["price"]

            log.info(f"  ► Выбрана {лучшая}  скор={скор}  цена={цена:.8f}  (прошло порог: {len(прошедшие)} из {len(SYMBOLS)} пар)")

            действие, уверенность, причина = спросить_ии(лучшая, цена, скор, детали)
            log.info(f"  🤖 ИИ: {действие}  уверенность={уверенность:.2f}  → {причина}")

            if действие != "buy":
                log.info(f"  ИИ отклонил сделку — ждём {SCAN_INTERVAL} сек")
                time.sleep(SCAN_INTERVAL)
                continue

            сумма = первый_вход
            шаг = 0
            log.info(f"  💰 Первый вход: {сумма:.2f} USDT (баланс {баланс:.2f} USDT)")

            while шаг <= MAX_STEPS:
                log.info(f"  ── Шаг мартингейла {шаг} / сумма {сумма:.2f} USDT ──")
                if баланс_usdt() < сумма:
                    log.warning(f"  Недостаточно USDT для входа {сумма:.2f} — прерываем серию")
                    break

                время_входа = time.time()
                вход_цена, кол_во = купить(лучшая, сумма)
                поставить_тп(лучшая, кол_во, вход_цена)
                stats["сделок_всего"] += 1
                сохранить_состояние()

                результат = мониторить_позицию(лучшая, вход_цена, кол_во, время_входа)

                if результат == "tp":
                    прибыль = сумма * TP_PERCENT / 100
                    stats["тейкпрофит"] += 1
                    stats["прибыль_usdt"] += прибыль
                    log.info(f"  ✅ Прибыль зафиксирована ~{прибыль:.4f} USDT — серия закончена")
                    break

                elif результат == "дедлайн_24ч":
                    stats["дедлайн_24ч"] += 1
                    stats["убыток_usdt"] += сумма * SL_PERCENT / 100
                    log.warning("  Серия прервана по дедлайну 24 часа")
                    шаг = MAX_STEPS + 1
                    break

                elif результат in ("sl", "timeout"):
                    убыток = сумма * SL_PERCENT / 100
                    stats["убыток_usdt"] += убыток
                    if результат == "sl":
                        stats["стоплосс"] += 1
                    else:
                        stats["таймаут"] += 1

                    шаг += 1
                    if шаг > MAX_STEPS:
                        log.info("  🚫 Лимит шагов мартингейла исчерпан — серия окончена")
                        break
                    сумма = round(сумма * MARTINGALE_FACTOR, 2)
                    stats["мартингейл_шагов"] += 1
                    log.info(f"  🔁 Мартингейл шаг {шаг} — следующая ставка {сумма:.2f} USDT")

            сохранить_состояние()
            log.info("  Серия завершена — пауза 30 сек")
            time.sleep(30)

        except Exception as e:
            log.error(f"Глобальная ошибка: {e}", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    main()
