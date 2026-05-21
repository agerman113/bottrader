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
#  LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
#  SETTINGS
# ─────────────────────────────────────────────────────────────
SYMBOLS          = ["PEPE/USDT", "DOGE/USDT", "SHIB/USDT", "FLOKI/USDT", "BONK/USDT"]
INITIAL_AMOUNT   = 6.0          # USDT on first entry
MARTINGALE_FACTOR = 1.35
MAX_STEPS        = 2            # max martingale doubles (0 = off)
TP_PERCENT       = 0.8          # take-profit %
SL_PERCENT       = 1.0          # stop-loss %
TIMEFRAME_TA     = "5m"
TIMEFRAME_TREND  = "1h"
SCAN_INTERVAL    = 300          # seconds between scans
MIN_SCORE        = 65           # minimum TA score to consider entry (0-100)
AI_MODEL         = "deepseek/deepseek-v4-flash:free"
AI_CONFIDENCE_THRESHOLD = 0.60
TRADE_TIMEOUT    = 600          # max seconds to hold a position

# ─────────────────────────────────────────────────────────────
#  EXCHANGE
# ─────────────────────────────────────────────────────────────
exchange = ccxt.bybit({
    "apiKey":    os.getenv("BYBIT_API_KEY"),
    "secret":    os.getenv("BYBIT_API_SECRET"),
    "enableRateLimit": True,
    "options": {"defaultType": "spot"},
})


# ─────────────────────────────────────────────────────────────
#  INDICATOR HELPERS  (replicate DIY Custom Strategy Builder)
# ─────────────────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rma(series: pd.Series, span: int) -> pd.Series:
    """Wilder's smoothing (RMA) — used in RSI and ATR."""
    alpha = 1 / span
    return series.ewm(alpha=alpha, adjust=False).mean()


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = _rma(gain, period)
    avg_loss = _rma(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(close: pd.Series, fast=12, slow=26, signal=9):
    macd_line   = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hi, lo, prev_close = df["h"], df["l"], df["c"].shift(1)
    tr = pd.concat([hi - lo, (hi - prev_close).abs(), (lo - prev_close).abs()], axis=1).max(axis=1)
    return _rma(tr, period)


def calc_range_filter(df: pd.DataFrame, period: int = 200, qty: float = 3.0):
    """
    Simplified Python port of the Range Filter from the DIY indicator.
    Returns (filt, hi_band, lo_band, upward, downward) as Series.
    """
    close = df["c"]
    atr   = calc_atr(df, period)
    rng   = qty * atr

    filt = close.copy()
    for i in range(1, len(close)):
        c  = close.iloc[i]
        r  = rng.iloc[i]
        pf = filt.iloc[i - 1]
        if   c - r > pf:  filt.iloc[i] = c - r
        elif c + r < pf:  filt.iloc[i] = c + r
        else:             filt.iloc[i] = pf

    hi_band = filt + rng
    lo_band = filt - rng

    upward   = (filt > filt.shift(1)) & (close > filt)
    downward = (filt < filt.shift(1)) & (close < filt)
    return filt, hi_band, lo_band, upward, downward


def calc_supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0):
    atr  = calc_atr(df, period)
    hl2  = (df["h"] + df["l"]) / 2
    upper_basic = hl2 + mult * atr
    lower_basic = hl2 - mult * atr

    upper = upper_basic.copy()
    lower = lower_basic.copy()
    trend = pd.Series(1, index=df.index)

    for i in range(1, len(df)):
        c    = df["c"].iloc[i]
        pc   = df["c"].iloc[i - 1]
        pu   = upper.iloc[i - 1]
        pl   = lower.iloc[i - 1]
        pt   = trend.iloc[i - 1]

        upper.iloc[i] = upper_basic.iloc[i] if upper_basic.iloc[i] < pu or pc > pu else pu
        lower.iloc[i] = lower_basic.iloc[i] if lower_basic.iloc[i] > pl or pc < pl else pl

        if   pt ==  1 and c < lower.iloc[i]:  trend.iloc[i] = -1
        elif pt == -1 and c > upper.iloc[i]:  trend.iloc[i] =  1
        else:                                  trend.iloc[i] =  pt

    is_up   = trend ==  1
    is_down = trend == -1
    return is_up, is_down


def calc_stochastic(df: pd.DataFrame, k=14, d=3, smooth=3):
    lowest  = df["l"].rolling(k).min()
    highest = df["h"].rolling(k).max()
    k_line  = 100 * (df["c"] - lowest) / (highest - lowest + 1e-10)
    k_smooth = k_line.rolling(smooth).mean()
    d_line   = k_smooth.rolling(d).mean()
    return k_smooth, d_line


def calc_qqe(close: pd.Series, rsi_period=14, sf=5, qq_factor=4.236):
    """Simplified QQE Mod signal direction."""
    rsi        = calc_rsi(close, rsi_period)
    rsi_smooth = _ema(rsi, sf)
    tr_rsi     = (rsi_smooth - rsi_smooth.shift(1)).abs()
    atr_rsi    = _ema(tr_rsi, rsi_period * 2)
    threshold  = qq_factor * atr_rsi
    upper = rsi_smooth + threshold
    lower = rsi_smooth - threshold
    is_long  = rsi_smooth > 50
    is_short = rsi_smooth < 50
    return is_long, is_short


def calc_hull_suite(close: pd.Series, period: int = 55):
    half  = _ema(close, period // 2)
    full  = _ema(close, period)
    delta = 2 * half - full
    hma   = _ema(delta, int(np.sqrt(period)))
    is_up   = hma > hma.shift(2)
    is_down = hma < hma.shift(2)
    return is_up, is_down


def calc_adx(df: pd.DataFrame, period: int = 14):
    atr   = calc_atr(df, period)
    plus_dm  = (df["h"] - df["h"].shift(1)).clip(lower=0)
    minus_dm = (df["l"].shift(1) - df["l"]).clip(lower=0)
    # zero out when the other is larger
    mask = plus_dm < minus_dm;  plus_dm[mask]  = 0
    mask = minus_dm < plus_dm;  minus_dm[mask] = 0

    plus_di  = 100 * _rma(plus_dm,  period) / atr.replace(0, np.nan)
    minus_di = 100 * _rma(minus_dm, period) / atr.replace(0, np.nan)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx      = _rma(dx, period)
    return adx, plus_di, minus_di


# ─────────────────────────────────────────────────────────────
#  COMPOSITE TECHNICAL SCORE  (0-100)
# ─────────────────────────────────────────────────────────────

def get_technical_score(symbol: str) -> dict:
    """
    Runs a suite of indicators inspired by the DIY Custom Strategy Builder.
    Returns {'score': int, 'details': dict} where score ∈ [0, 100].
    """
    details = {}
    score   = 0

    try:
        # ── Fetch candles ──────────────────────────────────────
        raw_5m = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA, limit=300)
        raw_1h = exchange.fetch_ohlcv(symbol, TIMEFRAME_TREND, limit=300)
        if len(raw_5m) < 100 or len(raw_1h) < 100:
            return {"score": 0, "details": {}}

        cols = ["ts", "o", "h", "l", "c", "v"]
        df5  = pd.DataFrame(raw_5m, columns=cols).reset_index(drop=True)
        df1h = pd.DataFrame(raw_1h, columns=cols).reset_index(drop=True)

        close5  = df5["c"]
        close1h = df1h["c"]

        # ── 1. RSI (5m) — 20 pts ──────────────────────────────
        rsi = calc_rsi(close5)
        rsi_val = rsi.iloc[-1]
        details["rsi"] = round(rsi_val, 1)
        if   rsi_val < 30:  rsi_pts = 20
        elif rsi_val < 45:  rsi_pts = 10
        elif rsi_val > 70:  rsi_pts = -5
        else:               rsi_pts = 5
        score += rsi_pts

        # ── 2. MACD (5m) — 15 pts ─────────────────────────────
        macd, sig, hist = calc_macd(close5)
        macd_bull = macd.iloc[-1] > sig.iloc[-1]
        macd_cross = macd.iloc[-1] > sig.iloc[-1] and macd.iloc[-2] <= sig.iloc[-2]
        details["macd_bullish"] = macd_bull
        macd_pts = 15 if macd_cross else (8 if macd_bull else 0)
        score += macd_pts

        # ── 3. Range Filter (5m) — 15 pts ─────────────────────
        filt, hi_b, lo_b, rf_up, rf_dn = calc_range_filter(df5)
        rf_long = rf_up.iloc[-1]
        details["range_filter"] = "up" if rf_long else "down"
        score += 15 if rf_long else 0

        # ── 4. Supertrend (5m) — 10 pts ───────────────────────
        st_up, st_dn = calc_supertrend(df5)
        details["supertrend"] = "up" if st_up.iloc[-1] else "down"
        score += 10 if st_up.iloc[-1] else 0

        # ── 5. Hull Suite (5m) — 10 pts ───────────────────────
        hu_up, hu_dn = calc_hull_suite(close5)
        details["hull"] = "up" if hu_up.iloc[-1] else "down"
        score += 10 if hu_up.iloc[-1] else 0

        # ── 6. EMA trend (1h) — 10 pts ────────────────────────
        ema50_1h  = _ema(close1h, 50).iloc[-1]
        ema200_1h = _ema(close1h, 200).iloc[-1]
        trend_bull = ema50_1h > ema200_1h
        details["ema_trend_1h"] = "bullish" if trend_bull else "bearish"
        score += 10 if trend_bull else 0

        # ── 7. ADX (5m) — 5 pts ───────────────────────────────
        adx, plus_di, minus_di = calc_adx(df5)
        adx_val = adx.iloc[-1]
        adx_trending = adx_val > 20 and plus_di.iloc[-1] > minus_di.iloc[-1]
        details["adx"] = round(adx_val, 1)
        score += 5 if adx_trending else 0

        # ── 8. Stochastic (5m) — 5 pts ────────────────────────
        k, d = calc_stochastic(df5)
        stoch_bull = k.iloc[-1] < 30 or (k.iloc[-1] > d.iloc[-1] and k.iloc[-1] < 50)
        details["stoch_k"] = round(k.iloc[-1], 1)
        score += 5 if stoch_bull else 0

        # ── 9. Volume surge — 5 pts ───────────────────────────
        avg_vol  = df5["v"].rolling(20).mean().iloc[-1]
        last_vol = df5["v"].iloc[-1]
        vol_surge = last_vol > avg_vol * 1.2
        details["volume_surge"] = vol_surge
        score += 5 if vol_surge else 0

        # ── 10. QQE (5m) — 5 pts ──────────────────────────────
        qqe_long, _ = calc_qqe(close5)
        details["qqe"] = "long" if qqe_long.iloc[-1] else "short"
        score += 5 if qqe_long.iloc[-1] else 0

        # Clamp
        score = max(0, min(100, score))

    except Exception as e:
        log.warning(f"TA error for {symbol}: {e}")

    return {"score": score, "details": details, "price": exchange.fetch_ticker(symbol)["last"]}


# ─────────────────────────────────────────────────────────────
#  AI FILTER  (OpenRouter — deepseek-v4-flash:free)
# ─────────────────────────────────────────────────────────────

def ask_ai(symbol: str, price: float, score: int, details: dict) -> tuple[str, float, str]:
    """
    Returns (action, confidence, reasoning).
    action ∈ {"buy", "wait"}
    """
    prompt = f"""You are a crypto scalping assistant. Analyze this real-time data and decide to BUY or WAIT.

Symbol: {symbol}
Price: {price}
TA Score: {score}/100

Indicators:
- RSI: {details.get('rsi', 'N/A')} (< 30 oversold, > 70 overbought)
- MACD: {'bullish' if details.get('macd_bullish') else 'bearish'}
- Range Filter: {details.get('range_filter', 'N/A')}
- Supertrend: {details.get('supertrend', 'N/A')}
- Hull Suite: {details.get('hull', 'N/A')}
- 1h EMA trend: {details.get('ema_trend_1h', 'N/A')}
- ADX: {details.get('adx', 'N/A')} (> 20 = trending)
- Stochastic K: {details.get('stoch_k', 'N/A')}
- Volume surge: {details.get('volume_surge', False)}
- QQE: {details.get('qqe', 'N/A')}

Strategy: spot scalping, TP={TP_PERCENT}%, SL={SL_PERCENT}%. Enter only on strong confluence.

Respond ONLY with valid JSON (no markdown, no extra text):
{{"action": "buy" or "wait", "confidence": 0.0-1.0, "reasoning": "one sentence"}}"""

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
            # strip possible markdown code block
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            start = content.find("{")
            end   = content.rfind("}") + 1
            if start != -1:
                data = json.loads(content[start:end])
                return (
                    data.get("action", "wait"),
                    float(data.get("confidence", 0.0)),
                    data.get("reasoning", ""),
                )
        else:
            log.warning(f"AI HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        log.warning(f"AI error: {e}")
    return ("wait", 0.0, "AI unavailable")


# ─────────────────────────────────────────────────────────────
#  TRADING HELPERS
# ─────────────────────────────────────────────────────────────

def place_buy(symbol: str, amount_usdt: float) -> tuple[float, float]:
    ticker = exchange.fetch_ticker(symbol)
    price  = ticker["last"]
    qty    = amount_usdt / price
    order  = exchange.create_market_buy_order(symbol, qty)
    log.info(f"BUY  {qty:.6f} {symbol.split('/')[0]} @ ~{price:.8f}  ({amount_usdt:.2f} USDT)")
    return price, qty


def place_tp(symbol: str, qty: float, entry_price: float):
    tp_price = entry_price * (1 + TP_PERCENT / 100)
    try:
        exchange.create_limit_sell_order(symbol, qty, tp_price)
        log.info(f"TP set @ {tp_price:.8f}")
    except Exception as e:
        log.warning(f"Could not set TP limit order: {e}")


def market_sell(symbol: str, qty: float):
    exchange.create_market_sell_order(symbol, qty)
    log.info(f"SELL {qty:.6f} {symbol.split('/')[0]} @ market")


def free_balance(symbol: str) -> float:
    coin = symbol.split("/")[0]
    return exchange.fetch_balance()["free"].get(coin, 0.0)


def usdt_balance() -> float:
    return exchange.fetch_balance()["free"].get("USDT", 0.0)


# ─────────────────────────────────────────────────────────────
#  POSITION MONITOR  (TP / SL / timeout)
# ─────────────────────────────────────────────────────────────

def monitor_position(symbol: str, entry_price: float, qty: float) -> str:
    """
    Watches open position until TP, SL, or timeout.
    Returns 'tp', 'sl', or 'timeout'.
    """
    deadline = time.time() + TRADE_TIMEOUT
    while time.time() < deadline:
        time.sleep(10)
        try:
            # If coin is gone → TP limit was filled
            coin_bal = free_balance(symbol)
            if coin_bal < qty * 0.05:
                log.info("✅ TP filled!")
                return "tp"

            cur_price = exchange.fetch_ticker(symbol)["last"]
            pnl_pct   = (cur_price - entry_price) / entry_price * 100

            if pnl_pct <= -SL_PERCENT:
                log.info(f"❌ SL hit  {pnl_pct:.2f}%")
                market_sell(symbol, coin_bal)
                return "sl"

            log.debug(f"{symbol} P&L: {pnl_pct:+.2f}%")
        except Exception as e:
            log.warning(f"Monitor error: {e}")

    # Timeout → close
    log.info("⏰ Timeout — closing position")
    try:
        market_sell(symbol, free_balance(symbol))
    except Exception as e:
        log.warning(f"Timeout close error: {e}")
    return "timeout"


# ─────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("  BOT STARTED")
    log.info(f"  Symbols : {SYMBOLS}")
    log.info(f"  Deposit : {usdt_balance():.2f} USDT")
    log.info(f"  Model   : {AI_MODEL}")
    log.info("=" * 60)

    while True:
        try:
            # ── 1. Score all symbols ──────────────────────────
            log.info("── Scanning ──")
            scores = {}
            for sym in SYMBOLS:
                result = get_technical_score(sym)
                scores[sym] = result
                log.info(
                    f"  {sym:15s}  score={result['score']:3d}/100  "
                    f"rsi={result['details'].get('rsi','?')}  "
                    f"rf={result['details'].get('range_filter','?')}  "
                    f"st={result['details'].get('supertrend','?')}"
                )

            # Best candidate
            best = max(scores, key=lambda s: scores[s]["score"])
            best_score   = scores[best]["score"]
            best_details = scores[best]["details"]
            best_price   = scores[best]["price"]

            if best_score < MIN_SCORE:
                log.info(f"  Best score {best_score} < {MIN_SCORE} — waiting {SCAN_INTERVAL}s")
                time.sleep(SCAN_INTERVAL)
                continue

            log.info(f"  ► Selected {best}  score={best_score}")

            # ── 2. AI filter ──────────────────────────────────
            action, conf, reason = ask_ai(best, best_price, best_score, best_details)
            log.info(f"  🤖 AI: {action}  conf={conf:.2f}  → {reason}")

            if action != "buy" or conf < AI_CONFIDENCE_THRESHOLD:
                log.info(f"  AI rejected — waiting {SCAN_INTERVAL}s")
                time.sleep(SCAN_INTERVAL)
                continue

            # ── 3. Martingale entry loop ──────────────────────
            step   = 0
            amount = INITIAL_AMOUNT

            while step <= MAX_STEPS:
                log.info(f"  ── Step {step}  amount={amount:.2f} USDT ──")

                # Safety: check we have enough balance
                avail = usdt_balance()
                if avail < amount:
                    log.warning(f"  Not enough USDT ({avail:.2f}), skipping entry")
                    break

                entry_price, qty = place_buy(best, amount)
                place_tp(best, qty, entry_price)

                result = monitor_position(best, entry_price, qty)

                if result == "tp":
                    log.info(f"  ✅ Profit taken — series done")
                    break
                elif result in ("sl", "timeout"):
                    step += 1
                    if step > MAX_STEPS:
                        log.info(f"  🚫 Max martingale steps reached — series over")
                        break
                    amount = round(amount * MARTINGALE_FACTOR, 2)
                    log.info(f"  ↩ Martingale step {step}  next amount={amount:.2f} USDT")

            log.info("  Series finished — pause 30s")
            time.sleep(30)

        except Exception as e:
            log.error(f"Global error: {e}", exc_info=True)
            time.sleep(60)


if __name__ == "__main__":
    main()
