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

# Торговые параметры
LEVERAGE = 2                     # плечо
BASE_RISK_PERCENT = 2.0         # базовый процент от баланса на маржу (при скоре = 100)
MIN_MARGIN = 1.0                # минимальная маржа в USDT
MAX_MARGIN = 10.0               # максимальная маржа в USDT

TP_PERCENT = 1.2                # тейк-профит от цены входа
SL_PERCENT = 1.5                # стоп-лосс

TIMEFRAME_TA = "5m"
TIMEFRAME_TREND = "1h"
SCAN_INTERVAL = 300             # секунд между сканированиями
MIN_SCORE = 55                  # порог скора для входа
MIN_BALANCE = 5.0               # минимальный баланс USDT для работы

# AI Bybit и фильтр поддержки/сопротивления
USE_BYBIT_AI = True
USE_SUPPORT_RESISTANCE = True
SUPPORT_THRESHOLD = 0.5         # % близости к поддержке (добавляет баллы)
RESISTANCE_THRESHOLD = 0.5      # % близости к сопротивлению (блокирует вход)

# Комиссии (тейкер для рыночных ордеров)
TAKER_FEE = 0.0006

TRADE_TIMEOUT = 600             # 10 минут
TRADE_MAX_LIFETIME = 86400      # 24 часа
REPORT_INTERVAL = 1800          # 30 минут
STATE_FILE = "futures_state.json"

# ================== ЛОГИРОВАНИЕ ==================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("futures_bot.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ================== БИРЖА (ФЬЮЧЕРСЫ LINEAR) ==================
exchange = ccxt.bybit({
    "apiKey":    os.getenv("BYBIT_API_KEY"),
    "secret":    os.getenv("BYBIT_API_SECRET"),
    "enableRateLimit": True,
    "options": {
        "defaultType": "linear",   # USDT-M фьючерсы
    },
})

# ================== СТАТИСТИКА ==================
stats = {
    "запусков":           0,
    "сделок_всего":       0,
    "тейкпрофит":         0,
    "стоплосс":           0,
    "таймаут":            0,
    "дедлайн_24ч":        0,
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
                    "дедлайн_24ч", "прибыль_usdt", "убыток_usdt",
                    "депозит_старт", "старт_время"]:
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

def закрыть_все_позиции():
    log.info("  🔒 Закрытие всех открытых позиций...")
    try:
        позиции = exchange.fetch_positions()
        for pos in позиции:
            contracts = float(pos.get("contracts", 0))
            if contracts != 0:
                symbol = pos["symbol"]
                side = "sell" if pos["side"] == "long" else "buy"
                try:
                    exchange.create_market_order(symbol, side, contracts)
                    log.info(f"    Закрыта позиция {symbol} {pos['side']} {contracts}")
                except Exception as e:
                    log.warning(f"    Не удалось закрыть {symbol}: {e}")
    except Exception as e:
        log.warning(f"  Ошибка при получении позиций: {e}")

def полная_инвентаризация():
    log.info("🔄 Выполняю полную инвентаризацию перед торговлей...")
    отменить_все_ордера()
    закрыть_все_позиции()
    log.info("✅ Инвентаризация завершена")
    time.sleep(2)

# ================== ИНДИКАТОРЫ (сжатая версия) ==================
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
    return up

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
    return trend == 1

def calc_stochastic(df, k=14, d=3, smooth=3):
    lo = df["l"].rolling(k).min()
    hi = df["h"].rolling(k).max()
    ks = (100 * (df["c"] - lo) / (hi - lo + 1e-10)).rolling(smooth).mean()
    return ks

def calc_hull(close, period=55):
    hma = _ema(2 * _ema(close, period//2) - _ema(close, period), int(np.sqrt(period)))
    return hma > hma.shift(2)

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

def calc_vwap_deviation(df, period=20):
    typical = (df["h"] + df["l"] + df["c"]) / 3
    vwap = (typical * df["v"]).rolling(period).sum() / df["v"].rolling(period).sum()
    return (df["c"] - vwap) / vwap * 100

# ================== ТЕХНИЧЕСКИЙ СКОР ==================
def получить_скор(symbol: str) -> dict:
    details = {}
    score = 0
    try:
        raw5 = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA, limit=200)
        raw1h = exchange.fetch_ohlcv(symbol, TIMEFRAME_TREND, limit=200)
        raw15 = exchange.fetch_ohlcv(symbol, "15m", limit=100)
        if len(raw5) < 100:
            return {"score": 0, "details": {}, "price": 0}

        cols = ["ts", "o", "h", "l", "c", "v"]
        df5 = pd.DataFrame(raw5, columns=cols).reset_index(drop=True)
        df1h = pd.DataFrame(raw1h, columns=cols).reset_index(drop=True)
        df15 = pd.DataFrame(raw15, columns=cols).reset_index(drop=True)
        c5, c1h, c15 = df5["c"], df1h["c"], df15["c"]

        # RSI (5m)
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

        # RSI 1h
        rsi_1h = calc_rsi(c1h).iloc[-1]
        if rsi_1h < 55:
            score += 8
        elif rsi_1h < 65:
            score += 4

        # MACD
        ml, sl, _ = calc_macd(c5)
        macd_bull = ml.iloc[-1] > sl.iloc[-1]
        macd_cross = macd_bull and ml.iloc[-2] <= sl.iloc[-2]
        if macd_cross:
            score += 18
        elif macd_bull:
            score += 8

        # Range Filter
        rf_up = calc_range_filter(df5)
        if rf_up.iloc[-1]:
            score += 15

        # Supertrend (5m и 15m)
        st_up = calc_supertrend(df5)
        if st_up.iloc[-1]:
            score += 12
        st_up_15 = calc_supertrend(df15)
        if st_up_15.iloc[-1]:
            score += 8

        # Hull MA
        hu_up = calc_hull(c5)
        if hu_up.iloc[-1]:
            score += 8

        # EMA тренд 1h
        ema50 = _ema(c1h, 50).iloc[-1]
        ema200 = _ema(c1h, 200).iloc[-1]
        if ema50 > ema200:
            score += 10

        # EMA 20/50 15m
        ema20_15 = _ema(c15, 20).iloc[-1]
        ema50_15 = _ema(c15, 50).iloc[-1]
        if ema20_15 > ema50_15:
            score += 5

        # ADX
        adx, pdi, mdi = calc_adx(df5)
        adx_val = adx.iloc[-1]
        if adx_val > 25 and pdi.iloc[-1] > mdi.iloc[-1]:
            score += 8
        elif adx_val > 20:
            score += 3

        # Stochastic
        k_val = calc_stochastic(df5).iloc[-1]
        if k_val < 25:
            score += 8
        elif k_val < 50:
            score += 4

        # Объём
        vol_ratio = df5["v"].iloc[-1] / (df5["v"].rolling(20).mean().iloc[-1] + 1e-10)
        if vol_ratio > 1.5:
            score += 8
        elif vol_ratio > 1.2:
            score += 4

        # VWAP отклонение
        vwap_dev = calc_vwap_deviation(df5).iloc[-1]
        if -3 <= vwap_dev <= -0.3:
            score += 8
        elif vwap_dev < -3:
            score += 4
        elif vwap_dev <= 1:
            score += 2

        # Штраф за 3 красные свечи
        if all(df5["c"].iloc[-i] < df5["o"].iloc[-i] for i in range(1, 4)):
            score -= 15

        score = max(0, min(100, score))
        price = df5["c"].iloc[-1]

    except Exception as e:
        log.warning(f"Ошибка анализа {symbol}: {e}")

    return {"score": score, "details": details, "price": price}

# ================== BYBIT AI (с уровнями) ==================
def получить_ai_анализ_bybit(symbol):
    try:
        pair = symbol.replace("/", "")
        url = f"https://api.bybit.com/v5/market/ai-analysis?symbol={pair}"
        headers = {}
        if os.getenv("BYBIT_API_KEY"):
            headers["X-BAPI-API-KEY"] = os.getenv("BYBIT_API_KEY")
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("retCode") == 0:
                result = data["result"]
                trend = result.get("trend", "neutral").lower()
                bullish = result.get("bullishIndicators", 0)
                bearish = result.get("bearishIndicators", 0)
                support = result.get("support", [])
                resistance = result.get("resistance", [])
                return trend, bullish, bearish, support, resistance
    except Exception:
        pass
    return None

# ================== РАСЧЁТ МАРЖИ ОТ ВЕРОЯТНОСТИ ==================
def рассчитать_маржу(баланс_usdt: float, score: int) -> float:
    """Маржа = базовый процент * (score/100), с ограничениями"""
    risk = BASE_RISK_PERCENT * (score / 100.0)
    маржа = баланс_usdt * risk / 100.0
    return round(max(MIN_MARGIN, min(MAX_MARGIN, маржа)), 2)

# ================== ТОРГОВЫЕ ФУНКЦИИ ==================
def баланс_usdt():
    try:
        bal = exchange.fetch_balance()
        return bal["free"].get("USDT", 0.0)
    except:
        return 0.0

def установить_плечо(symbol, leverage):
    try:
        exchange.set_leverage(leverage, symbol)
        log.info(f"  Плечо {leverage}x для {symbol}")
    except Exception as e:
        log.warning(f"  Не удалось установить плечо {symbol}: {e}")

def открыть_лонг(symbol, маржа_usdt, score):
    """Открывает длинную позицию с TP/SL, встроенными в ордер"""
    try:
        ticker = exchange.fetch_ticker(symbol)
        цена = ticker['last']
        # Количество контрактов = (маржа * плечо) / цена
        количество = (маржа_usdt * LEVERAGE) / цена
        количество = float(exchange.amount_to_precision(symbol, количество))
        if количество <= 0:
            log.error(f"  Количество {количество} <= 0, пропускаем")
            return None, None

        tp_price = цена * (1 + TP_PERCENT / 100)
        sl_price = цена * (1 - SL_PERCENT / 100)

        # Создаём рыночный ордер с TP/SL для линейных фьючерсов
        order = exchange.create_order(
            symbol=symbol,
            type='market',
            side='buy',
            amount=количество,
            params={
                'takeProfit': tp_price,
                'stopLoss': sl_price,
            }
        )
        log.info(f"  📈 ЛОНГ {symbol} | маржа {маржа_usdt:.2f} USDT | кол-во {количество} | цена {цена:.8f}")
        log.info(f"  🎯 TP={tp_price:.8f} (+{TP_PERCENT}%)   SL={sl_price:.8f} (-{SL_PERCENT}%)")
        return цена, количество
    except Exception as e:
        log.error(f"  ❌ Ошибка открытия лонга: {e}")
        return None, None

def закрыть_позицию(symbol):
    try:
        позиции = exchange.fetch_positions([symbol])
        for pos in позиции:
            if float(pos.get("contracts", 0)) != 0:
                side = "sell" if pos["side"] == "long" else "buy"
                qty = float(pos["contracts"])
                exchange.create_market_order(symbol, side, qty)
                log.info(f"  🔒 Позиция {symbol} закрыта")
                return True
        return False
    except Exception as e:
        log.warning(f"  Ошибка закрытия {symbol}: {e}")
        return False

def есть_открытая_позиция(symbol):
    try:
        позиции = exchange.fetch_positions([symbol])
        for pos in позиции:
            if float(pos.get("contracts", 0)) != 0:
                return True
        return False
    except:
        return False

def мониторить_позицию(symbol, entry_price, открыта_в: float) -> str:
    """Следит, не зависла ли позиция (если TP/SL не сработали)"""
    deadline_обычный = открыта_в + TRADE_TIMEOUT
    deadline_24ч = открыта_в + TRADE_MAX_LIFETIME

    while True:
        сейчас = time.time()
        if сейчас >= deadline_24ч:
            log.warning("  🔴 ДЕДЛАЙН 24 ЧАСА — принудительное закрытие позиции!")
            закрыть_позицию(symbol)
            return "дедлайн_24ч"
        if сейчас >= deadline_обычный:
            log.info("  ⏰ Таймаут 10 мин — закрываем позицию")
            закрыть_позицию(symbol)
            return "timeout"
        time.sleep(10)
        if not есть_открытая_позиция(symbol):
            # Позиция закрыта – вероятно, сработал TP или SL
            return "closed"

# ================== ТЕСТОВАЯ СДЕЛКА ==================
def тестовая_сделка():
    log.info("🧪 Тестовая сделка (фьючерсы) для проверки API...")
    try:
        sym = "DOGE/USDT"
        установить_плечо(sym, LEVERAGE)
        ticker = exchange.fetch_ticker(sym)
        маржа = 1.0
        кол_во = (маржа * LEVERAGE) / ticker['last']
        кол_во = float(exchange.amount_to_precision(sym, кол_во))
        log.info(f"  Покупка {кол_во} {sym} на маржу {маржа} USDT...")
        # Открываем без TP/SL, чтобы не усложнять
        order = exchange.create_market_buy_order(sym, кол_во)
        time.sleep(2)
        exchange.create_market_sell_order(sym, кол_во)
        log.info("  ✅ Тестовая сделка успешна!")
    except Exception as e:
        log.error(f"  ❌ Тестовая сделка не удалась: {e}")

# ================== ОТЧЁТЫ ==================
def печатать_отчёт():
    сейчас = баланс_usdt()
    старт = stats["депозит_старт"]
    дельта = сейчас - старт
    знак = "+" if дельта >= 0 else ""
    чистый = stats["прибыль_usdt"] - stats["убыток_usdt"]
    процент = (дельта / старт * 100) if старт > 0 else 0
    winrate = (stats["тейкпрофит"] / stats["сделок_всего"] * 100) if stats["сделок_всего"] > 0 else 0

    log.info("")
    log.info("=" * 60)
    log.info("  📊  ОТЧЁТ (ФЬЮЧЕРСЫ)")
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
        stats["старт_время"] = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

    stats["депозит_текущий"] = баланс_сейчас
    stats["последний_отчёт"] = time.time()

    log.info("")
    log.info("=" * 60)
    log.info("  🤖  ФЬЮЧЕРСНЫЙ БОТ ЗАПУЩЕН (без мартингейла)")
    log.info(f"  Запуск №:             {stats['запусков']}")
    log.info(f"  Дата/время:           {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    log.info(f"  Работает с:           {stats['старт_время']}")
    log.info(f"  Баланс:               {баланс_сейчас:.2f} USDT")
    log.info(f"  Плечо:                {LEVERAGE}x")
    log.info(f"  Базовый риск:         {BASE_RISK_PERCENT}% (адаптивный от скора)")
    log.info(f"  MIN_SCORE:            {MIN_SCORE}")
    log.info(f"  TP / SL:              {TP_PERCENT}% / {SL_PERCENT}%")
    log.info(f"  AI Bybit:             {'вкл' if USE_BYBIT_AI else 'выкл'}")
    log.info(f"  Уровни sup/res:       {'вкл' if USE_SUPPORT_RESISTANCE else 'выкл'}")
    log.info("=" * 60)
    log.info("")

    # Устанавливаем плечо для всех символов
    for sym in SYMBOLS:
        установить_плечо(sym, LEVERAGE)

    while True:
        try:
            if time.time() - stats["последний_отчёт"] >= REPORT_INTERVAL:
                печатать_отчёт()

            баланс = баланс_usdt()
            if баланс < MIN_BALANCE:
                log.warning(f"  🛑 Баланс {баланс:.2f} < {MIN_BALANCE} USDT. Пауза 10 мин.")
                time.sleep(600)
                continue

            # Защита от просадки
            if stats["депозит_старт"] > 0:
                просадка = (stats["депозит_старт"] - баланс) / stats["депозит_старт"] * 100
                if просадка > 25:
                    log.warning(f"  ⛔ Просадка {просадка:.1f}% > 25%. Пауза 60 мин.")
                    time.sleep(3600)
                    continue

            log.info(f"── Сканирование {len(SYMBOLS)} пар (баланс={баланс:.2f}, порог={MIN_SCORE}) ──")
            scores = {}
            for sym in SYMBOLS:
                try:
                    res = получить_скор(sym)
                    scores[sym] = res
                    log.info(f"  {sym:12s}  скор={res['score']:3d}/100")
                except Exception as e:
                    log.warning(f"  Ошибка {sym}: {e}")
                    scores[sym] = {"score": 0, "details": {}, "price": 0}

            if not scores:
                time.sleep(SCAN_INTERVAL)
                continue

            лучшая = max(scores, key=lambda s: scores[s]["score"])
            сырой_скор = scores[лучшая]["score"]
            цена = scores[лучшая]["price"]
            финальный_скор = сырой_скор

            # AI Bybit с уровнями поддержки/сопротивления
            if USE_BYBIT_AI:
                ai = получить_ai_анализ_bybit(лучшая)
                if ai:
                    trend, bull_cnt, bear_cnt, support, resistance = ai
                    log.info(f"  🤖 Bybit AI: тренд={trend}, бычьих={bull_cnt}, медвежьих={bear_cnt}")
                    if support:
                        log.info(f"     Поддержка: {support[:3]}")
                    if resistance:
                        log.info(f"     Сопротивление: {resistance[:3]}")
                    if trend == "bullish" and bull_cnt >= 3:
                        финальный_скор += 15
                    elif trend == "bearish" and bear_cnt >= 3:
                        финальный_скор -= 15
                    # Фильтр по уровням
                    if USE_SUPPORT_RESISTANCE and support:
                        ближ_под = min(support, key=lambda x: abs(x - цена))
                        dist_to_support = abs(цена - ближ_под) / цена * 100
                        if dist_to_support <= SUPPORT_THRESHOLD:
                            log.info(f"     ✅ Цена близка к поддержке {ближ_под} (+{финальный_скор} → +10)")
                            финальный_скор += 10
                    if USE_SUPPORT_RESISTANCE and resistance:
                        ближ_рез = min(resistance, key=lambda x: abs(x - цена))
                        dist_to_res = abs(цена - ближ_рез) / цена * 100
                        if dist_to_res <= RESISTANCE_THRESHOLD:
                            log.info(f"     ⚠️ Цена у сопротивления {ближ_рез} — вход блокирован")
                            финальный_скор = 0
                else:
                    log.info(f"  🤖 Bybit AI: данные не получены")

            log.info(f"  ► Выбрана {лучшая}  сырой скор={сырой_скор} → финальный скор={финальный_скор}  цена={цена:.8f}")

            if финальный_скор < MIN_SCORE:
                log.info(f"  Финальный скор {финальный_скор} < {MIN_SCORE} — ждём {SCAN_INTERVAL} сек")
                time.sleep(SCAN_INTERVAL)
                continue

            if есть_открытая_позиция(лучшая):
                log.info("  Позиция уже открыта, ждём")
                time.sleep(SCAN_INTERVAL)
                continue

            # Рассчитываем маржу динамически от скора
            маржа = рассчитать_маржу(баланс, финальный_скор)
            log.info(f"  ✅ Сигнал к покупке (скор {финальный_скор})")
            log.info(f"  💰 Маржа: {маржа:.2f} USDT (баланс {баланс:.2f})")

            время_входа = time.time()
            цена_входа, кол_во = открыть_лонг(лучшая, маржа, финальный_скор)
            if цена_входа is None:
                log.warning("  Не удалось открыть позицию")
                time.sleep(30)
                continue

            stats["сделок_всего"] += 1
            сохранить_состояние()

            # Мониторим закрытие (TP/SL сработают автоматически)
            результат = мониторить_позицию(лучшая, цена_входа, время_входа)

            # Простая эвристика: если позиция закрылась сама, пытаемся понять, TP или SL.
            # Для этого смотрим изменение баланса.
            новый_баланс = баланс_usdt()
            pnl_netto = новый_баланс - баланс
            if pnl_netto > 0:
                stats["тейкпрофит"] += 1
                stats["прибыль_usdt"] += pnl_netto
                log.info(f"  ✅ Прибыль зафиксирована +{pnl_netto:.4f} USDT")
            else:
                stats["стоплосс"] += 1
                stats["убыток_usdt"] += -pnl_netto
                log.warning(f"  ❌ Убыток {pnl_netto:.4f} USDT")

            сохранить_состояние()
            log.info("  Серия завершена — пауза 30 сек")
            time.sleep(30)

        except Exception as e:
            log.error(f"Глобальная ошибка: {e}", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    main()
