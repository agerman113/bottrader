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

# ─────────────────────────────────────────────────────────────
#  ЛОГИРОВАНИЕ
# ─────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────
#  НАСТРОЙКИ
# ─────────────────────────────────────────────────────────────
SYMBOLS = [
    # Мемкоины
    "PEPE/USDT", "DOGE/USDT", "SHIB/USDT", "FLOKI/USDT", "BONK/USDT",
    "WIF/USDT",  "MEME/USDT", "BOME/USDT", "NEIRO/USDT", "DOGS/USDT",
    # Альткоины с высокой ликвидностью
    "SOL/USDT",  "AVAX/USDT", "MATIC/USDT", "LTC/USDT",  "LINK/USDT",
    "DOT/USDT",  "ADA/USDT",  "TRX/USDT",  "XRP/USDT",  "TON/USDT",
]

MARTINGALE_FACTOR  = 1.35
MAX_STEPS          = 2
TP_PERCENT         = 0.8
SL_PERCENT         = 1.0
TIMEFRAME_TA       = "5m"
TIMEFRAME_TREND    = "1h"
SCAN_INTERVAL      = 300        # секунд между сканированиями
MIN_SCORE          = 65         # минимальный ТА-скор
AI_MODEL           = "deepseek/deepseek-v4-flash:free"
AI_CONFIDENCE_MIN  = 0.60
TRADE_TIMEOUT      = 600        # обычный таймаут позиции (10 мин)
TRADE_MAX_LIFETIME = 86400      # жёсткий дедлайн позиции (24 часа)
REPORT_INTERVAL    = 1800       # отчёт каждые 30 минут
STATE_FILE         = "state.json"

# ─────────────────────────────────────────────────────────────
#  БИРЖА
# ─────────────────────────────────────────────────────────────
exchange = ccxt.bybit({
    "apiKey":    os.getenv("BYBIT_API_KEY"),
    "secret":    os.getenv("BYBIT_API_SECRET"),
    "enableRateLimit": True,
    "options": {"defaultType": "spot"},
})

# ─────────────────────────────────────────────────────────────
#  СТАТИСТИКА
# ─────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────
#  СОСТОЯНИЕ
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
#  ИНДИКАТОРЫ
# ─────────────────────────────────────────────────────────────

def _ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def _rma(s, span):
    return s.ewm(alpha=1 / span, adjust=False).mean()

def calc_rsi(close, period=14):
    d = close.diff()
    return 100 - (100 / (1 + _rma(d.clip(lower=0), period) /
                         _rma((-d).clip(lower=0), period).replace(0, np.nan)))

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
    rng   = qty * calc_atr(df, period)
    filt  = close.copy()
    for i in range(1, len(close)):
        c, r, pf = close.iloc[i], rng.iloc[i], filt.iloc[i - 1]
        filt.iloc[i] = c - r if c - r > pf else (c + r if c + r < pf else pf)
    up   = (filt > filt.shift(1)) & (close > filt)
    down = (filt < filt.shift(1)) & (close < filt)
    return filt, filt + rng, filt - rng, up, down

def calc_supertrend(df, period=10, mult=3.0):
    atr = calc_atr(df, period)
    hl2 = (df["h"] + df["l"]) / 2
    ub  = (hl2 + mult * atr).copy()
    lb  = (hl2 - mult * atr).copy()
    trend = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        c, pc = df["c"].iloc[i], df["c"].iloc[i - 1]
        pu, pl, pt = ub.iloc[i - 1], lb.iloc[i - 1], trend.iloc[i - 1]
        ub.iloc[i] = ub.iloc[i] if ub.iloc[i] < pu or pc > pu else pu
        lb.iloc[i] = lb.iloc[i] if lb.iloc[i] > pl or pc < pl else pl
        trend.iloc[i] = (-1 if pt == 1 and c < lb.iloc[i] else
                         (1 if pt == -1 and c > ub.iloc[i] else pt))
    return trend == 1, trend == -1

def calc_stochastic(df, k=14, d=3, smooth=3):
    lo  = df["l"].rolling(k).min()
    hi  = df["h"].rolling(k).max()
    ks  = (100 * (df["c"] - lo) / (hi - lo + 1e-10)).rolling(smooth).mean()
    return ks, ks.rolling(d).mean()

def calc_qqe(close, rsi_period=14, sf=5, qq_factor=4.236):
    rsi_s = _ema(calc_rsi(close, rsi_period), sf)
    _ema((rsi_s - rsi_s.shift(1)).abs(), rsi_period * 2)
    return rsi_s > 50, rsi_s < 50

def calc_hull(close, period=55):
    hma = _ema(2 * _ema(close, period // 2) - _ema(close, period), int(np.sqrt(period)))
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


# ─────────────────────────────────────────────────────────────
#  АВТОКАЛИБРОВКА СУММЫ ВХОДА
# ─────────────────────────────────────────────────────────────

def рассчитать_размер_входа(баланс_usdt: float) -> float:
    """15% депозита, с запасом на полный мартингейл. Мин 2, макс 20 USDT."""
    total_factor = 1 + MARTINGALE_FACTOR + MARTINGALE_FACTOR ** 2
    amount = min(баланс_usdt * 0.15, баланс_usdt / total_factor * 0.9)
    return round(max(2.0, min(20.0, amount)), 2)


# ─────────────────────────────────────────────────────────────
#  ТЕХНИЧЕСКИЙ СКОР  (0–100)
# ─────────────────────────────────────────────────────────────

def получить_скор(symbol: str) -> dict:
    details = {}
    score   = 0
    price   = 0
    try:
        raw5  = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA,    limit=300)
        raw1h = exchange.fetch_ohlcv(symbol, TIMEFRAME_TREND, limit=300)
        if len(raw5) < 100 or len(raw1h) < 100:
            return {"score": 0, "details": {}, "price": 0}

        cols = ["ts", "o", "h", "l", "c", "v"]
        df5  = pd.DataFrame(raw5,  columns=cols).reset_index(drop=True)
        df1h = pd.DataFrame(raw1h, columns=cols).reset_index(drop=True)
        c5, c1h = df5["c"], df1h["c"]

        # RSI — 20 очков
        rsi_val = calc_rsi(c5).iloc[-1]
        details["rsi"] = round(rsi_val, 1)
        score += (20 if rsi_val < 30 else 10 if rsi_val < 45 else -5 if rsi_val > 70 else 5)

        # MACD — 15 очков
        ml, sl, _ = calc_macd(c5)
        macd_bull  = ml.iloc[-1] > sl.iloc[-1]
        macd_cross = macd_bull and ml.iloc[-2] <= sl.iloc[-2]
        details["macd"] = "бычий" if macd_bull else "медвежий"
        score += 15 if macd_cross else (8 if macd_bull else 0)

        # Range Filter — 15 очков
        _, _, _, rf_up, _ = calc_range_filter(df5)
        details["range_filter"] = "вверх" if rf_up.iloc[-1] else "вниз"
        score += 15 if rf_up.iloc[-1] else 0

        # Supertrend — 10 очков
        st_up, _ = calc_supertrend(df5)
        details["supertrend"] = "вверх" if st_up.iloc[-1] else "вниз"
        score += 10 if st_up.iloc[-1] else 0

        # Hull Suite — 10 очков
        hu_up, _ = calc_hull(c5)
        details["hull"] = "вверх" if hu_up.iloc[-1] else "вниз"
        score += 10 if hu_up.iloc[-1] else 0

        # EMA тренд 1h — 10 очков
        trend_bull = _ema(c1h, 50).iloc[-1] > _ema(c1h, 200).iloc[-1]
        details["тренд_1h"] = "бычий" if trend_bull else "медвежий"
        score += 10 if trend_bull else 0

        # ADX — 5 очков
        adx, pdi, mdi = calc_adx(df5)
        details["adx"] = round(adx.iloc[-1], 1)
        score += 5 if adx.iloc[-1] > 20 and pdi.iloc[-1] > mdi.iloc[-1] else 0

        # Stochastic — 5 очков
        k, _ = calc_stochastic(df5)
        details["stoch_k"] = round(k.iloc[-1], 1)
        score += 5 if k.iloc[-1] < 30 else 0

        # Объём — 5 очков
        vol_surge = df5["v"].iloc[-1] > df5["v"].rolling(20).mean().iloc[-1] * 1.2
        details["объём_всплеск"] = vol_surge
        score += 5 if vol_surge else 0

        # QQE — 5 очков
        qqe_long, _ = calc_qqe(c5)
        details["qqe"] = "лонг" if qqe_long.iloc[-1] else "шорт"
        score += 5 if qqe_long.iloc[-1] else 0

        score = max(0, min(100, score))
        price = exchange.fetch_ticker(symbol)["last"]

    except Exception as e:
        log.warning(f"Ошибка анализа {symbol}: {e}")

    return {"score": score, "details": details, "price": price}


# ─────────────────────────────────────────────────────────────
#  ИИ-ФИЛЬТР
# ─────────────────────────────────────────────────────────────

def спросить_ии(symbol, price, score, details) -> tuple:
    prompt = f"""Ты помощник для крипто-скальпинга. Проанализируй данные и реши: купить (buy) или ждать (wait).

Пара: {symbol}
Цена: {price}
ТА-скор: {score}/100

Индикаторы:
- RSI: {details.get('rsi', '?')} (< 30 = перепроданность)
- MACD: {details.get('macd', '?')}
- Range Filter: {details.get('range_filter', '?')}
- Supertrend: {details.get('supertrend', '?')}
- Hull Suite: {details.get('hull', '?')}
- Тренд 1h EMA: {details.get('тренд_1h', '?')}
- ADX: {details.get('adx', '?')} (> 20 = тренд)
- Stochastic K: {details.get('stoch_k', '?')}
- Всплеск объёма: {details.get('объём_всплеск', False)}
- QQE: {details.get('qqe', '?')}

Стратегия: спот-скальпинг, TP={TP_PERCENT}%, SL={SL_PERCENT}%. Входить только при сильном стечении факторов.

Ответь ТОЛЬКО валидным JSON без markdown:
{{"action": "buy" или "wait", "confidence": 0.0-1.0, "reasoning": "одно предложение"}}"""

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       AI_MODEL,
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens":  150,
    }
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers, json=payload, timeout=12
        )
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            s, e = content.find("{"), content.rfind("}") + 1
            if s != -1:
                data = json.loads(content[s:e])
                return (data.get("action", "wait"),
                        float(data.get("confidence", 0)),
                        data.get("reasoning", ""))
        else:
            log.warning(f"ИИ вернул HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        log.warning(f"Ошибка ИИ: {e}")
    return "wait", 0.0, "ИИ недоступен"


# ─────────────────────────────────────────────────────────────
#  ТОРГОВЫЕ ФУНКЦИИ
# ─────────────────────────────────────────────────────────────

def купить(symbol, amount_usdt):
    price = exchange.fetch_ticker(symbol)["last"]
    qty   = amount_usdt / price
    exchange.create_market_buy_order(symbol, qty)
    log.info(f"  📈 ПОКУПКА {qty:.6f} {symbol.split('/')[0]} по ~{price:.8f} ({amount_usdt:.2f} USDT)")
    return price, qty

def поставить_тп(symbol, qty, entry_price):
    tp = entry_price * (1 + TP_PERCENT / 100)
    try:
        exchange.create_limit_sell_order(symbol, qty, tp)
        log.info(f"  🎯 Тейк-профит установлен на {tp:.8f}")
    except Exception as e:
        log.warning(f"  Не удалось поставить TP: {e}")

def продать_по_рынку(symbol, qty, причина=""):
    try:
        exchange.create_market_sell_order(symbol, qty)
        log.info(f"  📉 ПРОДАЖА {qty:.6f} {symbol.split('/')[0]} по рынку{' (' + причина + ')' if причина else ''}")
    except Exception as e:
        log.warning(f"  Ошибка продажи: {e}")

def отменить_все_ордера(symbol):
    """Отменяет все открытые лимитные ордера по паре (TP лимитки)."""
    try:
        открытые = exchange.fetch_open_orders(symbol)
        for ордер in открытые:
            exchange.cancel_order(ордер["id"], symbol)
            log.info(f"  🗑️  Отменён ордер {ордер['id']}")
    except Exception as e:
        log.warning(f"  Ошибка отмены ордеров: {e}")

def баланс_монеты(symbol):
    return exchange.fetch_balance()["free"].get(symbol.split("/")[0], 0.0)

def баланс_usdt():
    return exchange.fetch_balance()["free"].get("USDT", 0.0)


# ─────────────────────────────────────────────────────────────
#  МОНИТОРИНГ ПОЗИЦИИ  (обычный таймаут + жёсткий дедлайн 24ч)
# ─────────────────────────────────────────────────────────────

def мониторить_позицию(symbol, entry_price, qty, открыта_в: float) -> str:
    """
    Следит за позицией.
    Уровни закрытия по времени:
      — TRADE_TIMEOUT (10 мин)  → закрывает по рынку, серия продолжается
      — TRADE_MAX_LIFETIME (24ч) → жёсткий дедлайн, отменяет все ордера,
                                   продаёт остаток, серия заканчивается
    """
    deadline_обычный  = открыта_в + TRADE_TIMEOUT
    deadline_24ч      = открыта_в + TRADE_MAX_LIFETIME

    while True:
        сейчас = time.time()

        # ── Жёсткий дедлайн 24 часа ──────────────────────────
        if сейчас >= deadline_24ч:
            log.warning("  🔴 ДЕДЛАЙН 24 ЧАСА — принудительное закрытие позиции!")
            отменить_все_ордера(symbol)
            остаток = баланс_монеты(symbol)
            if остаток > 0:
                продать_по_рынку(symbol, остаток, "дедлайн 24ч")
            cur = exchange.fetch_ticker(symbol)["last"]
            pnl = (cur - entry_price) / entry_price * 100
            log.warning(f"  Итог за 24ч: {'+' if pnl >= 0 else ''}{pnl:.2f}% от цены входа")
            return "дедлайн_24ч"

        # ── Обычный 10-минутный таймаут ───────────────────────
        if сейчас >= deadline_обычный:
            log.info("  ⏰ Таймаут 10 мин — закрываем позицию")
            отменить_все_ордера(symbol)
            остаток = баланс_монеты(symbol)
            if остаток > 0:
                продать_по_рынку(symbol, остаток, "таймаут 10 мин")
            return "timeout"

        time.sleep(10)

        try:
            # ── Проверяем, не сработал ли TP лимиткой ─────────
            бал = баланс_монеты(symbol)
            if бал < qty * 0.05:
                log.info("  ✅ Тейк-профит исполнен!")
                return "tp"

            # ── Проверяем стоп-лосс ───────────────────────────
            cur = exchange.fetch_ticker(symbol)["last"]
            pnl = (cur - entry_price) / entry_price * 100

            прошло_мин = (time.time() - открыта_в) / 60
            прошло_ч   = прошло_мин / 60
            if прошло_ч >= 1:
                log.debug(f"  {symbol}  P&L: {pnl:+.2f}%  держим {прошло_ч:.1f}ч")
            else:
                log.debug(f"  {symbol}  P&L: {pnl:+.2f}%  держим {прошло_мин:.0f} мин")

            if pnl <= -SL_PERCENT:
                log.info(f"  ❌ Стоп-лосс! Убыток {pnl:.2f}%")
                отменить_все_ордера(symbol)
                продать_по_рынку(symbol, бал, "стоп-лосс")
                return "sl"

        except Exception as e:
            log.warning(f"  Ошибка мониторинга: {e}")


# ─────────────────────────────────────────────────────────────
#  30-МИНУТНЫЙ ОТЧЁТ
# ─────────────────────────────────────────────────────────────

def печатать_отчёт():
    сейчас  = баланс_usdt()
    старт   = stats["депозит_старт"]
    дельта  = сейчас - старт
    знак    = "+" if дельта >= 0 else ""
    чистый  = stats["прибыль_usdt"] - stats["убыток_usdt"]
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


# ─────────────────────────────────────────────────────────────
#  АВТОКОРРЕКТИРОВКА РИСКА
# ─────────────────────────────────────────────────────────────

def скорректировать_риск(баланс: float):
    global MIN_SCORE
    старт = stats["депозит_старт"]
    if старт == 0:
        return
    изм = (баланс - старт) / старт * 100
    if изм < -30:
        MIN_SCORE = 80
        log.warning(f"  ⚠️  Депозит упал на {abs(изм):.1f}% — режим осторожности, MIN_SCORE=80")
    elif изм < -15:
        MIN_SCORE = 72
        log.info(f"  ℹ️  Депозит ниже старта на {abs(изм):.1f}% — MIN_SCORE повышен до 72")
    else:
        MIN_SCORE = 65
        log.info(f"  ✅ Депозит в норме ({'+' if изм >= 0 else ''}{изм:.1f}%) — штатный режим, MIN_SCORE=65")


# ─────────────────────────────────────────────────────────────
#  ГЛАВНЫЙ ЦИКЛ
# ─────────────────────────────────────────────────────────────

def main():
    восстановлен = загрузить_состояние()
    баланс_сейчас = баланс_usdt()
    stats["запусков"] += 1

    if not восстановлен or stats["депозит_старт"] == 0:
        stats["депозит_старт"] = баланс_сейчас
        stats["старт_время"]   = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

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
    log.info(f"  Таймаут позиции:      10 мин  /  жёсткий дедлайн: 24 ч")
    log.info(f"  ИИ модель:            {AI_MODEL}")
    if восстановлен:
        log.info("  ♻️  Состояние восстановлено — продолжаем с прошлой сессии")
    log.info("=" * 58)
    log.info("")

    скорректировать_риск(баланс_сейчас)

    while True:
        try:
            # ── Периодический отчёт ───────────────────────────
            if time.time() - stats["последний_отчёт"] >= REPORT_INTERVAL:
                печатать_отчёт()

            # ── 1. Рассчитываем размер входа ──────────────────
            баланс = баланс_usdt()
            первый_вход = рассчитать_размер_входа(баланс)

            # ── 2. Скорим все пары ────────────────────────────
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

            # Фильтруем пары которые вообще прошли порог
            прошедшие = {s: v for s, v in scores.items() if v["score"] >= MIN_SCORE}
            if not прошедшие:
                лучший_скор = max(scores.values(), key=lambda x: x["score"])["score"]
                log.info(f"  Ни одна пара не прошла порог {MIN_SCORE} (лучший скор: {лучший_скор}) — ждём {SCAN_INTERVAL} сек")
                time.sleep(SCAN_INTERVAL)
                continue

            # Берём лучшую
            лучшая = max(прошедшие, key=lambda s: прошедшие[s]["score"])
            скор   = прошедшие[лучшая]["score"]
            детали = прошедшие[лучшая]["details"]
            цена   = прошедшие[лучшая]["price"]

            log.info(f"  ► Выбрана {лучшая}  скор={скор}  цена={цена:.8f}  (прошло порог: {len(прошедшие)} из {len(SYMBOLS)} пар)")

            # ── 3. ИИ фильтр ──────────────────────────────────
            действие, уверенность, причина = спросить_ии(лучшая, цена, скор, детали)
            log.info(f"  🤖 ИИ: {действие}  уверенность={уверенность:.2f}  → {причина}")

            if действие != "buy" or уверенность < AI_CONFIDENCE_MIN:
                log.info(f"  ИИ отклонил сделку — ждём {SCAN_INTERVAL} сек")
                time.sleep(SCAN_INTERVAL)
                continue

            # ── 4. Вход с мартингейлом ─────────────────────────
            шаг    = 0
            сумма  = первый_вход
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

                результат = мониторить_позицию(лучшая, вход_цена, кол_во, время_входа)

                if результат == "tp":
                    прибыль = сумма * TP_PERCENT / 100
                    stats["тейкпрофит"]   += 1
                    stats["прибыль_usdt"] += прибыль
                    log.info(f"  ✅ Прибыль зафиксирована ~{прибыль:.4f} USDT — серия закончена")
                    break

                elif результат == "дедлайн_24ч":
                    # Уже всё закрыто внутри мониторинга
                    stats["дедлайн_24ч"]  += 1
                    stats["убыток_usdt"]  += сумма * SL_PERCENT / 100
                    log.warning("  Серия прервана по дедлайну 24 часа")
                    шаг = MAX_STEPS + 1   # выходим из while
                    break

                elif результат in ("sl", "timeout"):
                    убыток = сумма * SL_PERCENT / 100
                    stats["убыток_usdt"] += убыток
                    if результат == "sl":
                        stats["стоплосс"] += 1
                    else:
                        stats["таймаут"]  += 1

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
