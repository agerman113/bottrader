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
# ──────────────────────────────────────────────
# 1. Базовые параметры торговли
SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT",
    "XRP/USDT:USDT", "SOL/USDT:USDT", "ADA/USDT:USDT",
    "DOGE/USDT:USDT", "DOT/USDT:USDT", "LTC/USDT:USDT",
    "AVAX/USDT:USDT", "LINK/USDT:USDT", "MATIC/USDT:USDT",
    "UNI/USDT:USDT", "ATOM/USDT:USDT", "XLM/USDT:USDT",
    "ALGO/USDT:USDT", "VET/USDT:USDT", "FIL/USDT:USDT",
    "ICP/USDT:USDT", "NEAR/USDT:USDT", "APT/USDT:USDT",
    "ARB/USDT:USDT", "OP/USDT:USDT", "INJ/USDT:USDT",
    "SUI/USDT:USDT", "SEI/USDT:USDT", "TIA/USDT:USDT",
    "JUP/USDT:USDT", "JTO/USDT:USDT", "PYTH/USDT:USDT",
    # Мемкоины (высокий риск, но можно)
    "PEPE/USDT:USDT", "SHIB/USDT:USDT", "WIF/USDT:USDT",
    "BONK/USDT:USDT", "FLOKI/USDT:USDT", "MEME/USDT:USDT",
]

LEVERAGE           = 3          # плечо (2-3x)
ALLOW_SHORT        = True       # разрешить шорт (False = только лонг)
TREND_FILTER_REQUIRED = True    # вход только в направлении тренда (1h EMA)

# 2. Риск и размер позиции
BASE_RISK_PCT      = 1.5        # % баланса при скоре = MIN_SCORE
MAX_RISK_PCT       = 3.0        # % баланса при скоре = 100
MIN_SCORE          = 60         # порог входа (повышен)
# Формула: risk = BASE_RISK_PCT + (MAX_RISK_PCT - BASE_RISK_PCT) * (score - MIN_SCORE) / (100 - MIN_SCORE)

# 3. TP/SL и R:R (3:1)
TP_PERCENT         = 2.5        # тейк-профит
SL_PERCENT         = 0.8        # стоп-лосс (R:R = 3.125)

# 4. Таймауты и защита
TRADE_MAX_LIFETIME = 7200       # 2 часа максимум в позиции
CONSECUTIVE_SL_LIMIT = 3        # после 3 SL подряд – пауза 1 час
COOLDOWN_AFTER_SL_SEC = 3600    # 1 час

# 5. Трейлинг (после безубытка)
TRAILING_STEP_PCT   = 0.3
TRAILING_OFFSET_PCT = 0.4

# 6. Фильтры и анализ
TIMEFRAME_TA       = "5m"
TIMEFRAME_TREND    = "1h"
SCAN_INTERVAL      = 300        # 5 минут между сканами
MIN_BALANCE        = 5.0
REPORT_INTERVAL    = 1800
STATE_FILE         = "state_futures.json"
TRADES_HISTORY_FILE = "trades_history.json"

# 7. Уровни поддержки/сопротивления
SR_PERIOD          = 100
SR_PROXIMITY_PCT   = 0.3
SR_MIN_TOUCHES     = 3
SR_BLOCK_DIST_PCT  = 0.2        # блокируем вход если цена ближе 0.2% к сопротивлению (было 0.15)

# 8. Комиссия (тейкер)
BYBIT_FEE          = 0.00055

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

# ================== БИРЖА ==================
exchange = ccxt.bybit({
    "apiKey":    os.getenv("BYBIT_API_KEY"),
    "secret":    os.getenv("BYBIT_API_SECRET"),
    "enableRateLimit": True,
    "options": {"defaultType": "linear"},
})

# ================== СТАТИСТИКА ==================
stats = {
    "запусков":          0,
    "сделок_всего":      0,
    "тейкпрофит":        0,
    "стоплосс":          0,
    "таймаут":           0,
    "прибыль_usdt":      0.0,
    "убыток_usdt":       0.0,
    "депозит_старт":     0.0,
    "старт_время":       "",
    "последний_отчёт":   0.0,
    "подряд_sl":         0,         # счётчик убыточных сделок подряд
    "последний_sl_время": 0.0,
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
    log.info("  🗑️  Отмена всех открытых ордеров...")
    try:
        orders = exchange.fetch_open_orders()
        for o in orders:
            exchange.cancel_order(o["id"], o["symbol"])
    except Exception as e:
        log.warning(f"  Ошибка отмены ордеров: {e}")

def закрыть_все_позиции():
    log.info("  🔒 Закрытие всех открытых позиций...")
    try:
        positions = exchange.fetch_positions()
        for pos in positions:
            if float(pos.get("contracts", 0)) != 0:
                side = "sell" if pos["side"] == "long" else "buy"
                qty = abs(float(pos["contracts"]))
                exchange.create_market_order(pos["symbol"], side, qty, params={"reduceOnly": True})
                log.info(f"    Закрыта {pos['symbol']}")
    except Exception as e:
        log.warning(f"  Ошибка закрытия позиций: {e}")

def полная_инвентаризация():
    log.info("🔄 Инвентаризация перед торговлей...")
    отменить_все_ордера()
    закрыть_все_позиции()
    log.info("✅ Инвентаризация завершена")
    time.sleep(2)

# ================== БАЛАНС И ПОЗИЦИИ ==================
def баланс_usdt() -> float:
    try:
        b = exchange.fetch_balance({"type": "linear"})
        return float(b.get("USDT", {}).get("free", 0.0))
    except:
        return 0.0

def получить_позиции() -> list:
    try:
        return [p for p in exchange.fetch_positions() if float(p.get("contracts", 0)) != 0]
    except:
        return []

def есть_открытая_позиция(symbol) -> bool:
    try:
        for p in exchange.fetch_positions([symbol]):
            if float(p.get("contracts", 0)) != 0:
                return True
        return False
    except:
        return False

def установить_плечо(symbol: str, leverage: int):
    try:
        exchange.set_leverage(leverage, symbol)
    except Exception as e:
        log.warning(f"  Не удалось установить плечо {symbol}: {e}")

# ================== ИНДИКАТОРЫ ==================
def _ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def _rma(s, span):
    return s.ewm(alpha=1/span, adjust=False).mean()

def calc_rsi(close, period=14):
    d = close.diff()
    gain = d.clip(lower=0)
    loss = (-d).clip(lower=0)
    avg_g = _rma(gain, period)
    avg_l = _rma(loss, period)
    rs = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(close, fast=12, slow=26, signal=9):
    ml = _ema(close, fast) - _ema(close, slow)
    sl = _ema(ml, signal)
    return ml, sl, ml - sl

def calc_atr(df, period=14):
    hi, lo, pc = df["h"], df["l"], df["c"].shift(1)
    tr = pd.concat([hi - lo, (hi - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    return _rma(tr, period)

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

def calc_stochastic(df, k=14, d=3, smooth=3):
    lo = df["l"].rolling(k).min()
    hi = df["h"].rolling(k).max()
    ks = (100 * (df["c"] - lo) / (hi - lo + 1e-10)).rolling(smooth).mean()
    return ks, ks.rolling(d).mean()

def calc_vwap_deviation(df, period=20):
    typical = (df["h"] + df["l"] + df["c"]) / 3
    vwap = (typical * df["v"]).rolling(period).sum() / df["v"].rolling(period).sum()
    return (df["c"] - vwap) / vwap * 100

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

# ================== УРОВНИ S/R (с кластеризацией) ==================
def cluster_levels(levels, tolerance=0.005):
    if not levels:
        return []
    levels = sorted(levels)
    clusters = []
    cur = [levels[0]]
    for lvl in levels[1:]:
        if (lvl - cur[0]) / cur[0] < tolerance:
            cur.append(lvl)
        else:
            clusters.append((np.mean(cur), len(cur)))
            cur = [lvl]
    clusters.append((np.mean(cur), len(cur)))
    return clusters

def calc_support_resistance(df, period=SR_PERIOD):
    df_tail = df.tail(period).reset_index(drop=True)
    highs = df_tail["h"].values
    lows  = df_tail["l"].values
    close = df["c"].iloc[-1]

    resistances = []
    supports    = []
    for i in range(2, len(highs)-2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            resistances.append(highs[i])
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            supports.append(lows[i])

    res_cl = cluster_levels(resistances)
    sup_cl = cluster_levels(supports)

    res_above = [(p, n) for p, n in res_cl if p > close]
    sup_below = [(p, n) for p, n in sup_cl if p < close]

    nearest_res, res_touch = res_above[0] if res_above else (close*1.05, 0)
    nearest_sup, sup_touch = sup_below[0] if sup_below else (close*0.95, 0)

    dist_res = (nearest_res - close) / close * 100
    dist_sup = (close - nearest_sup) / close * 100

    near_sup = dist_sup < SR_PROXIMITY_PCT and sup_touch >= SR_MIN_TOUCHES
    near_res = dist_res < SR_PROXIMITY_PCT and res_touch >= SR_MIN_TOUCHES

    return {
        "support": nearest_sup, "resistance": nearest_res,
        "dist_sup": dist_sup, "dist_res": dist_res,
        "sup_touch": sup_touch, "res_touch": res_touch,
        "near_support": near_sup, "near_resistance": near_res,
    }

# ================== BYBIT AI (LONG/SHORT RATIO) ==================
def get_bybit_ai(symbol: str) -> dict:
    try:
        coin = symbol.split("/")[0]
        url = f"https://api.bybit.com/v5/market/account-ratio?category=linear&symbol={coin}USDT&period=1h&limit=1"
        resp = requests.get(url, timeout=5)
        data = resp.json()
        if data.get("retCode") == 0 and data.get("result", {}).get("list"):
            item = data["result"]["list"][0]
            buy_ratio = float(item.get("buyRatio", 0.5))
            sell_ratio = float(item.get("sellRatio", 0.5))
            signal = "bullish" if buy_ratio > 0.6 else ("bearish" if buy_ratio < 0.4 else "neutral")
            return {"available": True, "signal": signal, "long_ratio": buy_ratio}
    except:
        pass
    return {"available": False, "signal": "neutral", "long_ratio": 0.5}

# ================== ТЕХНИЧЕСКИЙ СКОР ==================
def get_score(symbol: str) -> dict:
    details = {}
    score = 0
    try:
        raw5  = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA, limit=300)
        raw1h = exchange.fetch_ohlcv(symbol, TIMEFRAME_TREND, limit=200)
        raw15 = exchange.fetch_ohlcv(symbol, "15m", limit=100)
        if len(raw5) < 100:
            return {"score": 0, "details": {}, "price": 0, "sr": {}}

        cols = ["ts", "o", "h", "l", "c", "v"]
        df5 = pd.DataFrame(raw5, columns=cols)
        df1h = pd.DataFrame(raw1h, columns=cols)
        df15 = pd.DataFrame(raw15, columns=cols)
        c5, c1h, c15 = df5["c"], df1h["c"], df15["c"]
        price = c5.iloc[-1]

        # RSI 5m
        rsi = calc_rsi(c5).iloc[-1]
        if 25 <= rsi <= 42: score += 20
        elif 42 < rsi <= 52: score += 10
        elif rsi < 25: score += 12
        elif 52 < rsi <= 65: score += 5
        details["rsi"] = round(rsi, 1)

        # RSI 1h
        rsi1h = calc_rsi(c1h).iloc[-1]
        if rsi1h < 55: score += 8
        elif rsi1h < 65: score += 4
        details["rsi_1h"] = round(rsi1h, 1)

        # MACD
        ml, sl, _ = calc_macd(c5)
        macd_bull = ml.iloc[-1] > sl.iloc[-1]
        macd_cross = macd_bull and ml.iloc[-2] <= sl.iloc[-2]
        details["macd"] = "бычий" if macd_bull else "медвежий"
        if macd_cross: score += 18
        elif macd_bull: score += 8

        # Range Filter
        rf_up = calc_range_filter(df5)
        if rf_up.iloc[-1]: score += 15
        details["range_filter"] = "вверх" if rf_up.iloc[-1] else "вниз"

        # Supertrend 5m
        st_up, _ = calc_supertrend(df5)
        if st_up.iloc[-1]: score += 12
        details["supertrend"] = "вверх" if st_up.iloc[-1] else "вниз"

        # Supertrend 15m
        st_up15, _ = calc_supertrend(df15)
        if st_up15.iloc[-1]: score += 8
        details["supertrend_15m"] = "вверх" if st_up15.iloc[-1] else "вниз"

        # Hull MA
        hu_up, _ = calc_hull(c5)
        if hu_up.iloc[-1]: score += 8
        details["hull"] = "вверх" if hu_up.iloc[-1] else "вниз"

        # Тренд 1h EMA
        ema50 = _ema(c1h, 50).iloc[-1]
        ema200 = _ema(c1h, 200).iloc[-1]
        trend_up = ema50 > ema200
        if trend_up: score += 10
        details["тренд_1h"] = "бычий" if trend_up else "медвежий"

        # Тренд 15m EMA20/50
        ema20_15 = _ema(c15, 20).iloc[-1]
        ema50_15 = _ema(c15, 50).iloc[-1]
        if ema20_15 > ema50_15: score += 5
        details["тренд_15m"] = "бычий" if ema20_15 > ema50_15 else "медвежий"

        # ADX
        adx, pdi, mdi = calc_adx(df5)
        if adx.iloc[-1] > 25 and pdi.iloc[-1] > mdi.iloc[-1]: score += 8
        elif adx.iloc[-1] > 20: score += 3
        details["adx"] = round(adx.iloc[-1], 1)

        # Stochastic
        k, _ = calc_stochastic(df5)
        k_val = k.iloc[-1]
        if k_val < 25: score += 8
        elif k_val < 50: score += 4
        details["stoch_k"] = round(k_val, 1)

        # Объём
        vol_ratio = df5["v"].iloc[-1] / (df5["v"].rolling(20).mean().iloc[-1] + 1e-10)
        if vol_ratio > 1.5: score += 8
        elif vol_ratio > 1.2: score += 4
        details["объём_ratio"] = round(vol_ratio, 2)

        # VWAP
        vwap_dev = calc_vwap_deviation(df5).iloc[-1]
        if -3 <= vwap_dev <= -0.3: score += 8
        elif vwap_dev < -3: score += 4
        elif vwap_dev <= 1: score += 2
        details["vwap_dev"] = round(vwap_dev, 2)

        # S/R
        sr = calc_support_resistance(df5)
        if sr["near_support"]:
            score += 12
            details["sr_signal"] = f"у поддержки ✅ ({sr['sup_touch']} касаний)"
        elif sr["near_resistance"]:
            score -= 20
            details["sr_signal"] = f"у сопротивления ❌ ({sr['res_touch']} касаний)"
        else:
            details["sr_signal"] = f"нейтрально (sup={sr['dist_sup']:.2f}% res={sr['dist_res']:.2f}%)"

        # Штраф 3 красные свечи
        if all(df5["c"].iloc[-i] < df5["o"].iloc[-i] for i in range(1,4)):
            score -= 15
            details["свечи_3red"] = True

        score = max(0, min(100, score))
        return {"score": score, "details": details, "price": price, "sr": sr}
    except Exception as e:
        log.warning(f"Ошибка анализа {symbol}: {e}")
        return {"score": 0, "details": {}, "price": 0, "sr": {}}

# ================== AI КОРРЕКЦИЯ ==================
def apply_ai_correction(score: int, symbol: str) -> int:
    ai = get_bybit_ai(symbol)
    if not ai["available"]:
        log.info(f"  🤖 Bybit AI: недоступен")
        return score
    signal = ai["signal"]
    log.info(f"  🤖 Bybit AI: сигнал={signal}, long_ratio={ai['long_ratio']:.1%}")
    if signal == "bullish":
        return min(100, score + 15)
    elif signal == "bearish":
        return max(0, score - 15)
    return score

# ================== РАЗМЕР ПОЗИЦИИ (АДАПТИВНЫЙ РИСК) ==================
def calc_margin(score: int, balance: float) -> float:
    if score <= MIN_SCORE:
        risk_pct = BASE_RISK_PCT
    else:
        risk_pct = BASE_RISK_PCT + (MAX_RISK_PCT - BASE_RISK_PCT) * (score - MIN_SCORE) / (100 - MIN_SCORE)
    max_loss = balance * risk_pct / 100
    margin = max_loss / (SL_PERCENT / 100)
    return round(max(1.0, margin), 2)

# ================== ОТКРЫТИЕ ПОЗИЦИИ (ЛОНГ ИЛИ ШОРТ) ==================
def open_position(symbol: str, side: str, margin_usdt: float, tp_price: float, sl_price: float):
    try:
        установить_плечо(symbol, LEVERAGE)
        ticker = exchange.fetch_ticker(symbol)
        price = ticker["last"]
        pos_size = margin_usdt * LEVERAGE
        qty = pos_size / price
        qty = float(exchange.amount_to_precision(symbol, qty))
        if qty <= 0:
            return None, None

        order = exchange.create_market_order(
            symbol, side, qty,
            params={"takeProfit": tp_price, "stopLoss": sl_price}
        )
        entry_price = order.get("average", price)
        log.info(f"  📈 {side.upper()} {symbol} маржа={margin_usdt:.2f}U, qty={qty}, цена={entry_price:.8f}")
        return entry_price, qty
    except Exception as e:
        log.error(f"  ❌ Ошибка открытия {side}: {e}")
        return None, None

# ================== МОНИТОРИНГ ПОЗИЦИИ (С ТРЕЙЛИНГОМ) ==================
def monitor_position(symbol: str, side: str, entry_price: float, qty: float,
                     sl_price: float, start_time: float) -> str:
    deadline = start_time + TRADE_MAX_LIFETIME
    breakeven = entry_price * (1 + BYBIT_FEE*2 + 0.0005) if side == "buy" else entry_price * (1 - (BYBIT_FEE*2 + 0.0005))
    trailing_step = TRAILING_STEP_PCT / 100
    trailing_offset = TRAILING_OFFSET_PCT / 100
    phase = 1
    current_sl = sl_price
    peak_price = entry_price
    next_trail = entry_price * (1 + trailing_step) if side == "buy" else entry_price * (1 - trailing_step)

    while True:
        if time.time() > deadline:
            log.warning("  ⏰ Дедлайн – закрываю")
            exchange.create_market_order(symbol, "sell" if side=="buy" else "buy", qty, params={"reduceOnly": True})
            return "timeout"

        time.sleep(10)
        try:
            pos = [p for p in exchange.fetch_positions([symbol]) if float(p.get("contracts",0)) != 0]
            if not pos:
                # позиция закрыта – определяем TP или SL по текущей цене
                cur = exchange.fetch_ticker(symbol)["last"]
                if (side == "buy" and cur >= entry_price * (1 + TP_PERCENT/100 * 0.7)) or \
                   (side == "sell" and cur <= entry_price * (1 - TP_PERCENT/100 * 0.7)):
                    return "tp"
                else:
                    return "sl"

            cur = exchange.fetch_ticker(symbol)["last"]
            pnl_pct = ((cur - entry_price) / entry_price * 100) if side=="buy" else ((entry_price - cur) / entry_price * 100)

            # Фаза 1 -> безубыток
            if phase == 1:
                if (side == "buy" and cur >= breakeven) or (side == "sell" and cur <= breakeven):
                    phase = 2
                    new_sl = entry_price * (1 + BYBIT_FEE*2 + 0.0003) if side=="buy" else entry_price * (1 - (BYBIT_FEE*2 + 0.0003))
                    if abs(new_sl - entry_price) > 0:
                        exchange.set_trading_stop(symbol, stopLoss=new_sl, params={"category":"linear"})
                        current_sl = new_sl
                        peak_price = cur
                        next_trail = cur * (1 + trailing_step) if side=="buy" else cur * (1 - trailing_step)
                        log.info(f"  🔒 Безубыток, SL={new_sl:.8f}")

            # Фаза трейлинга
            if phase >= 2:
                if (side == "buy" and cur > peak_price) or (side == "sell" and cur < peak_price):
                    peak_price = cur
                if (side == "buy" and cur >= next_trail) or (side == "sell" and cur <= next_trail):
                    new_sl = peak_price * (1 - trailing_offset) if side=="buy" else peak_price * (1 + trailing_offset)
                    if (side == "buy" and new_sl > current_sl) or (side == "sell" and new_sl < current_sl):
                        exchange.set_trading_stop(symbol, stopLoss=new_sl, params={"category":"linear"})
                        current_sl = new_sl
                        next_trail = cur * (1 + trailing_step) if side=="buy" else cur * (1 - trailing_step)
                        log.info(f"  📈 Трейлинг: SL={new_sl:.8f}")

            log.info(f"  {symbol.split(':')[0]} {cur:.8f}  P&L={pnl_pct:+.2f}%  phase={phase}")
        except Exception as e:
            log.warning(f"  Ошибка мониторинга: {e}")

# ================== ЗАЩИТА ОТ СЕРИИ УБЫТКОВ ==================
def check_cooldown():
    if stats["подряд_sl"] >= CONSECUTIVE_SL_LIMIT:
        elapsed = time.time() - stats["последний_sl_время"]
        if elapsed < COOLDOWN_AFTER_SL_SEC:
            log.warning(f"  🛑 После {CONSECUTIVE_SL_LIMIT} SL подряд – пауза {COOLDOWN_AFTER_SL_SEC//60} мин. Осталось {int((COOLDOWN_AFTER_SL_SEC - elapsed)/60)} мин.")
            time.sleep(COOLDOWN_AFTER_SL_SEC - elapsed)
            stats["подряд_sl"] = 0
            сохранить_состояние()

# ================== ОТЧЁТЫ ==================
def print_report():
    bal = баланс_usdt()
    start = stats["депозит_старт"]
    delta = bal - start
    pct = (delta / start * 100) if start>0 else 0
    wr = stats["тейкпрофит"] / stats["сделок_всего"]*100 if stats["сделок_всего"]>0 else 0
    log.info("="*60)
    log.info(f"  📊 ОТЧЁТ | Депозит: {start:.2f} → {bal:.2f} ({delta:+.2f}, {pct:+.1f}%)")
    log.info(f"  Сделок: {stats['сделок_всего']}  TP={stats['тейкпрофит']}  SL={stats['стоплосс']}  WR={wr:.1f}%")
    log.info(f"  P&L: {stats['прибыль_usdt']-stats['убыток_usdt']:+.4f} USDT")
    log.info("="*60)
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

    log.info("="*60)
    log.info("  🤖 ФЬЮЧЕРСНЫЙ БОТ (v3 – все исправления)")
    log.info(f"  Плечо: {LEVERAGE}x | MIN_SCORE={MIN_SCORE} | TP={TP_PERCENT}% SL={SL_PERCENT}%")
    log.info(f"  Риск: {BASE_RISK_PCT}–{MAX_RISK_PCT}% | Шорт {'разрешён' if ALLOW_SHORT else 'запрещён'}")
    log.info(f"  Баланс: {баланс:.2f} USDT")
    log.info("="*60)

    while True:
        try:
            if time.time() - stats["последний_отчёт"] > REPORT_INTERVAL:
                print_report()

            bal = баланс_usdt()
            if bal < MIN_BALANCE:
                log.warning(f"  Баланс {bal:.2f} < {MIN_BALANCE} – пауза 10 мин")
                time.sleep(600)
                continue

            # Защита от серии убытков
            check_cooldown()

            # Если есть открытая позиция – ждём
            if получить_позиции():
                log.info("  Есть открытая позиция – жду закрытия")
                time.sleep(30)
                continue

            # Сканируем
            log.info(f"── Сканирование {len(SYMBOLS)} пар (баланс={bal:.2f}) ──")
            candidates = []
            for sym in SYMBOLS:
                try:
                    data = get_score(sym)
                    final_score = apply_ai_correction(data["score"], sym)
                    if final_score < MIN_SCORE:
                        continue
                    # Фильтр тренда по направлению (если включен)
                    trend = data["details"].get("тренд_1h", "медвежий")
                    if TREND_FILTER_REQUIRED:
                        if not ALLOW_SHORT and trend != "бычий":
                            continue
                        if ALLOW_SHORT and trend != "бычий" and trend != "медвежий":
                            continue  # нейтральный тренд – пропускаем
                    candidates.append((sym, final_score, data))
                except Exception as e:
                    log.warning(f"  Ошибка {sym}: {e}")

            if not candidates:
                log.info(f"  Нет кандидатов с score>={MIN_SCORE} – ждём {SCAN_INTERVAL} сек")
                time.sleep(SCAN_INTERVAL)
                continue

            # Сортируем по скору
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_sym, best_score, best_data = candidates[0]
            price = best_data["price"]
            sr = best_data.get("sr", {})
            details = best_data["details"]

            # Определяем направление
            if not ALLOW_SHORT:
                side = "buy"
            else:
                # Используем AI сигнал + тренд
                ai = get_bybit_ai(best_sym)
                if ai["signal"] == "bullish" or details.get("тренд_1h") == "бычий":
                    side = "buy"
                elif ai["signal"] == "bearish" or details.get("тренд_1h") == "медвежий":
                    side = "sell"
                else:
                    side = "buy"   # по умолчанию лонг

            # Проверка сопротивления для лонга
            if side == "buy" and sr.get("near_resistance") and sr.get("dist_res", 99) < SR_BLOCK_DIST_PCT:
                log.info(f"  ⛔ {best_sym.split(':')[0]}: сопротивление {sr['dist_res']:.2f}% – блокируем лонг")
                time.sleep(SCAN_INTERVAL)
                continue

            # Рассчёт TP/SL с учётом S/R
            if side == "buy":
                sl_price = max(price * (1 - SL_PERCENT/100), sr.get("support", price*0.99))
                tp_price = min(price * (1 + TP_PERCENT/100), sr.get("resistance", price*1.02))
            else:
                sl_price = min(price * (1 + SL_PERCENT/100), sr.get("resistance", price*1.01))
                tp_price = max(price * (1 - TP_PERCENT/100), sr.get("support", price*0.98))

            # Размер позиции
            margin = calc_margin(best_score, bal)
            if margin > bal * 0.8:
                margin = bal * 0.8

            log.info(f"  🎯 {best_sym.split(':')[0]} | score={best_score} | side={side} | margin={margin:.2f}U")
            log.info(f"     TP={tp_price:.8f} ({'+' if side=='buy' else '-'}{TP_PERCENT}%)  SL={sl_price:.8f}")

            # Открываем позицию
            entry, qty = open_position(best_sym, side, margin, tp_price, sl_price)
            if entry is None:
                log.warning("  Не удалось открыть позицию")
                time.sleep(30)
                continue

            stats["сделок_всего"] += 1
            сохранить_состояние()

            start_time = time.time()
            result = monitor_position(best_sym, side, entry, qty, sl_price, start_time)

            # Расчёт P&L
            pos_size = margin * LEVERAGE
            fee = pos_size * BYBIT_FEE * 2
            if result == "tp":
                pnl = pos_size * TP_PERCENT / 100 - fee
                stats["тейкпрофит"] += 1
                stats["прибыль_usdt"] += max(0, pnl)
                stats["подряд_sl"] = 0
                log.info(f"  ✅ TP: +{pnl:.4f} USDT")
            elif result == "sl":
                pnl = -(pos_size * SL_PERCENT / 100 + fee)
                stats["стоплосс"] += 1
                stats["убыток_usdt"] -= pnl  # pnl отрицательный, вычитаем
                stats["подряд_sl"] += 1
                stats["последний_sl_время"] = time.time()
                log.warning(f"  ❌ SL: {pnl:.4f} USDT")
            else:
                pnl = -fee
                stats["таймаут"] += 1
                stats["убыток_usdt"] += fee
                log.warning(f"  ⏰ Таймаут: -{fee:.4f} USDT")

            # Сохраняем сделку в историю
            try:
                with open(TRADES_HISTORY_FILE, "a") as f:
                    json.dump({
                        "time": datetime.now().isoformat(),
                        "symbol": best_sym,
                        "side": side,
                        "score": best_score,
                        "margin": margin,
                        "entry": entry,
                        "tp": tp_price,
                        "sl": sl_price,
                        "result": result,
                        "pnl": round(pnl, 4),
                    }, f)
                    f.write("\n")
            except:
                pass

            сохранить_состояние()
            log.info("  Пауза 30 сек")
            time.sleep(30)

        except Exception as e:
            log.error(f"Глобальная ошибка: {e}", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    main()
