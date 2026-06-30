#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
МУЛЬТИ-СТРАТЕГИЧЕСКИЙ БЭКТЕСТЕР НА ИСТОРИЧЕСКИХ ДАННЫХ (v2, исправленный)
========================================================================
Загружает OHLCV для списка монет, прогоняет 31 стратегию,
выводит сводную таблицу результатов.
Исправлены ошибки int64 в Supertrend и Heikin-Ashi.
"""

import os, time, logging, numpy as np, pandas as pd, ccxt
from typing import Dict, List, Callable
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
#               КОНФИГУРАЦИЯ БЭКТЕСТА
# ============================================================
SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT", "XRP/USDT:USDT",
    "SOL/USDT:USDT", "ADA/USDT:USDT", "TRX/USDT:USDT", "AVAX/USDT:USDT",
    "DOT/USDT:USDT", "LTC/USDT:USDT", "BCH/USDT:USDT", "ATOM/USDT:USDT",
    "XLM/USDT:USDT", "NEAR/USDT:USDT", "DOGE/USDT:USDT",
    "1000PEPE/USDT:USDT", "WIF/USDT:USDT", "BOME/USDT:USDT",
    "RENDER/USDT:USDT", "TAO/USDT:USDT", "WLD/USDT:USDT", "ARKM/USDT:USDT",
    "IO/USDT:USDT", "ONDO/USDT:USDT", "VIRTUAL/USDT:USDT", "UNI/USDT:USDT",
    "AAVE/USDT:USDT", "ARB/USDT:USDT", "OP/USDT:USDT", "LINK/USDT:USDT",
    "GRT/USDT:USDT", "INJ/USDT:USDT", "SUI/USDT:USDT", "APT/USDT:USDT",
    "TIA/USDT:USDT", "JTO/USDT:USDT", "EIGEN/USDT:USDT", "HBAR/USDT:USDT",
    "VET/USDT:USDT", "NOT/USDT:USDT", "CATI/USDT:USDT",
]
TIMEFRAME = "5m"
LIMIT = 5000                     # больше свечей = точнее статистика
SL_ATR_MULT = 1.5
TP_ATR_MULT = 3.0
MAX_HOLD_BARS = 200
INITIAL_CAPITAL = 1000

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

exchange = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "linear"}})

def fetch_ohlcv(symbol: str, timeframe: str = "5m", limit: int = 5000) -> pd.DataFrame:
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    except Exception as e:
        log.error(f"Ошибка загрузки {symbol}: {e}")
        return pd.DataFrame()

# ---------------------- ИНДИКАТОРЫ (исправленные) ----------------------
def ema(series, span): return series.ewm(span=span, adjust=False).mean()
def sma(series, span): return series.rolling(span).mean()
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))
def macd(series, fast=12, slow=26, signal=9):
    ml = ema(series, fast) - ema(series, slow)
    sl = ema(ml, signal)
    return ml, sl, ml - sl
def atr(df, period=14):
    high, low, close = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([high - low, (high - close).abs(), (low - close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()
def bollinger_bands(series, period=20, std=2):
    mb = sma(series, period)
    std_dev = series.rolling(period).std()
    return mb + std*std_dev, mb, mb - std*std_dev
def supertrend(df, period=10, mult=3):
    atr_val = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    up = hl2 - mult * atr_val
    down = hl2 + mult * atr_val
    trend = pd.Series(1.0, index=df.index)          # <-- float, чтобы избежать int64
    for i in range(1, len(df)):
        if df["close"].iloc[i] > trend.iloc[i-1]:
            trend.iloc[i] = max(up.iloc[i], trend.iloc[i-1])
        else:
            trend.iloc[i] = min(down.iloc[i], trend.iloc[i-1])
    return trend
def stochastic(df, k_period=14, d_period=3):
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-10)
    d = k.rolling(d_period).mean()
    return k, d
def adx(df, period=14):
    atr_val = atr(df, period)
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff().clip(lower=0)
    minus_dm = -low.diff().clip(upper=0)
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_val.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_val.replace(0, np.nan))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
    adx_val = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx_val, plus_di, minus_di
def cci(df, period=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = sma(tp, period)
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma_tp) / (0.015 * mad)
def mfi(df, period=14):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    money_flow = tp * df["volume"]
    positive_flow = money_flow.where(tp > tp.shift(1), 0)
    negative_flow = money_flow.where(tp < tp.shift(1), 0)
    pos_sum = positive_flow.rolling(period).sum()
    neg_sum = negative_flow.rolling(period).sum()
    return 100 - (100 / (1 + pos_sum / neg_sum.replace(0, np.nan)))
def obv(df):
    return (df["volume"] * (df["close"].diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0))).cumsum()
def aroon(df, period=25):
    high_max = df["high"].rolling(period).apply(lambda x: x.argmax(), raw=True)
    low_min = df["low"].rolling(period).apply(lambda x: x.argmin(), raw=True)
    aroon_up = 100 * (period - high_max) / period
    aroon_down = 100 * (period - low_min) / period
    return aroon_up, aroon_down
def hull_ma(series, period=55):
    half = period // 2
    sqrt = int(np.sqrt(period))
    wma1 = 2 * ema(series, half) - ema(series, period)
    hma = ema(wma1, sqrt)
    return hma
def donchian(df, period=20):
    upper = df["high"].rolling(period).max()
    lower = df["low"].rolling(period).min()
    return upper, lower
def keltner(df, period=20, atr_mult=1.5):
    typical = (df["high"] + df["low"] + df["close"]) / 3
    mb = ema(typical, period)
    atr_val = atr(df, period)
    upper = mb + atr_mult * atr_val
    lower = mb - atr_mult * atr_val
    return upper, mb, lower
def parabolic_sar(df, acceleration=0.02, maximum=0.2):
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    sar = np.zeros(len(df))
    sar[0] = low[0]
    ep = high[0]
    af = acceleration
    trend = 1
    for i in range(1, len(df)):
        sar[i] = sar[i-1] + af * (ep - sar[i-1])
        if trend == 1:
            if low[i] < sar[i]:
                trend = -1
                sar[i] = ep
                ep = low[i]
                af = acceleration
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + acceleration, maximum)
                if low[i-1] < sar[i]: sar[i] = low[i-1]
                if low[i-2] < sar[i]: sar[i] = low[i-2]
        else:
            if high[i] > sar[i]:
                trend = 1
                sar[i] = ep
                ep = high[i]
                af = acceleration
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + acceleration, maximum)
                if high[i-1] > sar[i]: sar[i] = high[i-1]
                if high[i-2] > sar[i]: sar[i] = high[i-2]
    return pd.Series(sar, index=df.index)

# ---------------------- ФУНКЦИИ СИГНАЛОВ (31 стратегия) ----------------------
def ema_cross_9_21(df):
    e9 = ema(df["close"], 9)
    e21 = ema(df["close"], 21)
    signal = pd.Series(0, index=df.index)
    signal[(e9 > e21) & (e9.shift(1) <= e21.shift(1))] = 1
    signal[(e9 < e21) & (e9.shift(1) >= e21.shift(1))] = -1
    return signal
def ema_cross_21_50(df):
    e21 = ema(df["close"], 21)
    e50 = ema(df["close"], 50)
    signal = pd.Series(0, index=df.index)
    signal[(e21 > e50) & (e21.shift(1) <= e50.shift(1))] = 1
    signal[(e21 < e50) & (e21.shift(1) >= e50.shift(1))] = -1
    return signal
def sma_cross_50_200(df):
    s50 = sma(df["close"], 50)
    s200 = sma(df["close"], 200)
    signal = pd.Series(0, index=df.index)
    signal[(s50 > s200) & (s50.shift(1) <= s200.shift(1))] = 1
    signal[(s50 < s200) & (s50.shift(1) >= s200.shift(1))] = -1
    return signal
def macd_cross(df):
    ml, sl, _ = macd(df["close"])
    signal = pd.Series(0, index=df.index)
    signal[(ml > sl) & (ml.shift(1) <= sl.shift(1))] = 1
    signal[(ml < sl) & (ml.shift(1) >= sl.shift(1))] = -1
    return signal
def rsi_30_70(df):
    r = rsi(df["close"])
    signal = pd.Series(0, index=df.index)
    signal[r < 30] = 1
    signal[r > 70] = -1
    return signal
def rsi_ema200(df):
    r = rsi(df["close"])
    e200 = ema(df["close"], 200)
    signal = pd.Series(0, index=df.index)
    signal[(r < 30) & (df["close"] > e200)] = 1
    signal[(r > 70) & (df["close"] < e200)] = -1
    return signal
def bollinger_rev(df):
    upper, mid, lower = bollinger_bands(df["close"])
    signal = pd.Series(0, index=df.index)
    signal[df["close"] < lower] = 1
    signal[df["close"] > upper] = -1
    return signal
def bollinger_rsi(df):
    upper, mid, lower = bollinger_bands(df["close"])
    r = rsi(df["close"])
    signal = pd.Series(0, index=df.index)
    signal[(df["close"] < lower) & (r < 30)] = 1
    signal[(df["close"] > upper) & (r > 70)] = -1
    return signal
def supertrend_signal(df):
    st = supertrend(df)
    signal = pd.Series(0, index=df.index)
    signal[(st > st.shift(1)) & (df["close"] > st)] = 1
    signal[(st < st.shift(1)) & (df["close"] < st)] = -1
    return signal
def supertrend_rsi(df):
    st = supertrend(df)
    r = rsi(df["close"])
    signal = pd.Series(0, index=df.index)
    signal[(st > st.shift(1)) & (df["close"] > st) & (r < 50)] = 1
    signal[(st < st.shift(1)) & (df["close"] < st) & (r > 50)] = -1
    return signal
def donchian_breakout(df):
    upper, lower = donchian(df)
    signal = pd.Series(0, index=df.index)
    signal[df["close"] > upper.shift(1)] = 1
    signal[df["close"] < lower.shift(1)] = -1
    return signal
def keltner_rev(df):
    upper, mid, lower = keltner(df)
    signal = pd.Series(0, index=df.index)
    signal[df["close"] < lower] = 1
    signal[df["close"] > upper] = -1
    return signal
def stochastic_signal(df):
    k, d = stochastic(df)
    signal = pd.Series(0, index=df.index)
    signal[(k < 20) & (k > d)] = 1
    signal[(k > 80) & (k < d)] = -1
    return signal
def stochastic_trend(df):
    k, d = stochastic(df)
    e50 = ema(df["close"], 50)
    signal = pd.Series(0, index=df.index)
    signal[(k < 20) & (k > d) & (df["close"] > e50)] = 1
    signal[(k > 80) & (k < d) & (df["close"] < e50)] = -1
    return signal
def adx_signal(df):
    adx_val, plus_di, minus_di = adx(df)
    signal = pd.Series(0, index=df.index)
    signal[(adx_val > 25) & (plus_di > minus_di)] = 1
    signal[(adx_val > 25) & (minus_di > plus_di)] = -1
    return signal
def ichimoku(df):
    high9 = df["high"].rolling(9).max()
    low9 = df["low"].rolling(9).min()
    tenkan = (high9 + low9) / 2
    high26 = df["high"].rolling(26).max()
    low26 = df["low"].rolling(26).min()
    kijun = (high26 + low26) / 2
    signal = pd.Series(0, index=df.index)
    signal[(tenkan > kijun) & (df["close"] > kijun)] = 1
    signal[(tenkan < kijun) & (df["close"] < kijun)] = -1
    return signal
def parabolic_sar_signal(df):
    sar = parabolic_sar(df)
    signal = pd.Series(0, index=df.index)
    signal[df["close"] > sar] = 1
    signal[df["close"] < sar] = -1
    return signal
def cci_signal(df):
    c = cci(df)
    signal = pd.Series(0, index=df.index)
    signal[c > 100] = -1
    signal[c < -100] = 1
    return signal
def mfi_signal(df):
    m = mfi(df)
    signal = pd.Series(0, index=df.index)
    signal[m < 20] = 1
    signal[m > 80] = -1
    return signal
def obv_breakout(df):
    o = obv(df)
    signal = pd.Series(0, index=df.index)
    signal[o > o.rolling(20).mean()] = 1
    signal[o < o.rolling(20).mean()] = -1
    return signal
def aroon_signal(df):
    up, down = aroon(df)
    signal = pd.Series(0, index=df.index)
    signal[(up > 70) & (down < 30)] = 1
    signal[(down > 70) & (up < 30)] = -1
    return signal
def hull_ma_signal(df):
    hma = hull_ma(df["close"])
    signal = pd.Series(0, index=df.index)
    signal[(hma > hma.shift(1))] = 1
    signal[(hma < hma.shift(1))] = -1
    return signal
def volume_reversal(df):
    avg_vol = df["volume"].rolling(20).mean()
    body = abs(df["close"] - df["open"])
    signal = pd.Series(0, index=df.index)
    signal[(df["volume"] > 2 * avg_vol) & (df["close"] > df["open"]) & (body > body.shift(1))] = 1
    signal[(df["volume"] > 2 * avg_vol) & (df["close"] < df["open"]) & (body > body.shift(1))] = -1
    return signal
def engulfing(df):
    open, high, low, close = df["open"], df["high"], df["low"], df["close"]
    signal = pd.Series(0, index=df.index)
    bull_eng = (close > open) & (open.shift(1) > close.shift(1)) & (close > open.shift(1)) & (open < close.shift(1))
    bear_eng = (close < open) & (open.shift(1) < close.shift(1)) & (close < open.shift(1)) & (open > close.shift(1))
    signal[bull_eng] = 1
    signal[bear_eng] = -1
    return signal
def pin_bar(df):
    body = abs(df["close"] - df["open"])
    upper_shadow = df["high"] - df[["open", "close"]].max(axis=1)
    lower_shadow = df[["open", "close"]].min(axis=1) - df["low"]
    signal = pd.Series(0, index=df.index)
    signal[(lower_shadow > 2 * body) & (upper_shadow < 0.1 * body)] = 1
    signal[(upper_shadow > 2 * body) & (lower_shadow < 0.1 * body)] = -1
    return signal
def fractal_breakout(df, period=5):
    high_fractal = df["high"].rolling(2*period+1, center=True).apply(
        lambda x: x[period] if (x[period] == x.max()) and (np.sum(x == x.max()) == 1) else np.nan, raw=True)
    low_fractal = df["low"].rolling(2*period+1, center=True).apply(
        lambda x: x[period] if (x[period] == x.min()) and (np.sum(x == x.min()) == 1) else np.nan, raw=True)
    signal = pd.Series(0, index=df.index)
    signal[df["high"] > high_fractal.shift(1)] = 1
    signal[df["low"] < low_fractal.shift(1)] = -1
    return signal
def heikin_ashi(df):
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_open = pd.Series(0.0, index=df.index)          # <-- float
    ha_open.iloc[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
    ha_color = ha_close > ha_open
    signal = pd.Series(0, index=df.index)
    signal[(ha_color) & (~ha_color.shift(1))] = 1
    signal[(~ha_color) & (ha_color.shift(1))] = -1
    return signal
def range_filter(df, period=100, qty=2.5):
    close = df["close"]
    atr_val = atr(df, period)
    rng = qty * atr_val
    filt = close.copy()
    for i in range(1, len(close)):
        c, r, pf = close.iloc[i], rng.iloc[i], filt.iloc[i-1]
        if c - r > pf: filt.iloc[i] = c - r
        elif c + r < pf: filt.iloc[i] = c + r
        else: filt.iloc[i] = pf
    signal = pd.Series(0, index=df.index)
    signal[(close > filt) & (close.shift(1) <= filt.shift(1))] = 1
    signal[(close < filt) & (close.shift(1) >= filt.shift(1))] = -1
    return signal
def vwap_reversal(df):
    typical = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].cumsum()
    cum_vp = (typical * df["volume"]).cumsum()
    vwap = cum_vp / cum_vol
    std = (typical - vwap).rolling(100).std()
    upper = vwap + 2 * std
    lower = vwap - 2 * std
    signal = pd.Series(0, index=df.index)
    signal[df["close"] < lower] = 1
    signal[df["close"] > upper] = -1
    return signal
def zscore_reversal(df, period=20):
    sma20 = sma(df["close"], period)
    std20 = df["close"].rolling(period).std()
    z = (df["close"] - sma20) / std20
    signal = pd.Series(0, index=df.index)
    signal[z < -2] = 1
    signal[z > 2] = -1
    return signal
def momentum(df, period=10):
    mom = df["close"].pct_change(period)
    signal = pd.Series(0, index=df.index)
    signal[mom > 0.02] = 1
    signal[mom < -0.02] = -1
    return signal

class Strategy:
    def __init__(self, name: str, func: Callable):
        self.name = name
        self.func = func
    def get_signals(self, df): return self.func(df)

strategies = [
    Strategy("EMA Cross 9/21", ema_cross_9_21),
    Strategy("EMA Cross 21/50", ema_cross_21_50),
    Strategy("SMA Cross 50/200", sma_cross_50_200),
    Strategy("MACD Cross", macd_cross),
    Strategy("RSI 30/70", rsi_30_70),
    Strategy("RSI + EMA200", rsi_ema200),
    Strategy("Bollinger Reversal", bollinger_rev),
    Strategy("Bollinger + RSI", bollinger_rsi),
    Strategy("Supertrend", supertrend_signal),
    Strategy("Supertrend + RSI", supertrend_rsi),
    Strategy("Donchian Breakout", donchian_breakout),
    Strategy("Keltner Reversal", keltner_rev),
    Strategy("Stochastic", stochastic_signal),
    Strategy("Stochastic + Trend", stochastic_trend),
    Strategy("ADX Trend", adx_signal),
    Strategy("Ichimoku", ichimoku),
    Strategy("Parabolic SAR", parabolic_sar_signal),
    Strategy("CCI", cci_signal),
    Strategy("MFI", mfi_signal),
    Strategy("OBV Breakout", obv_breakout),
    Strategy("Aroon", aroon_signal),
    Strategy("Hull MA", hull_ma_signal),
    Strategy("Volume Reversal", volume_reversal),
    Strategy("Engulfing", engulfing),
    Strategy("Pin Bar", pin_bar),
    Strategy("Fractal Breakout", fractal_breakout),
    Strategy("Heikin-Ashi", heikin_ashi),
    Strategy("Range Filter", range_filter),
    Strategy("VWAP Reversal", vwap_reversal),
    Strategy("Z-Score Reversal", zscore_reversal),
    Strategy("Momentum", momentum),
]

# ---------------------- БЭКТЕСТ ----------------------
def backtest_strategy(df, signals, sl_mult, tp_mult, max_bars):
    capital = INITIAL_CAPITAL
    in_pos = False; side = 0; entry = 0; bars = 0
    trades = []
    for i in range(1, len(df)):
        if in_pos:
            bars += 1
            cur = df["close"].iloc[i]
            if side == 1:
                if cur <= entry - sl_mult * atr(df).iloc[i] or cur >= entry + tp_mult * atr(df).iloc[i] or bars >= max_bars:
                    pnl = (cur - entry) / entry * 100
                    trades.append(pnl); in_pos = False
            else:
                if cur >= entry + sl_mult * atr(df).iloc[i] or cur <= entry - tp_mult * atr(df).iloc[i] or bars >= max_bars:
                    pnl = (entry - cur) / entry * 100
                    trades.append(pnl); in_pos = False
        else:
            sig = signals.iloc[i]
            if sig != 0 and not pd.isna(sig):
                in_pos = True; side = sig; entry = df["close"].iloc[i]; bars = 0
    if not trades: return {"trades": 0, "winrate": 0, "avg_pnl": 0, "total_pnl": 0, "maxdd": 0}
    wins = sum(1 for p in trades if p > 0)
    wr = wins / len(trades) * 100
    avg = np.mean(trades)
    eq = [INITIAL_CAPITAL]
    for p in trades: eq.append(eq[-1] * (1 + p/100))
    total_pnl = (eq[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    eqs = pd.Series(eq)
    maxdd = (eqs.cummax() - eqs).max() / eqs.cummax().max() * 100
    return {"trades": len(trades), "winrate": wr, "avg_pnl": avg, "total_pnl": total_pnl, "maxdd": maxdd}

# ---------------------- ГЛАВНОЕ ----------------------
def main():
    log.info("Загрузка данных...")
    data = {}
    for sym in SYMBOLS:
        df = fetch_ohlcv(sym, TIMEFRAME, LIMIT)
        if not df.empty:
            data[sym] = df
            log.info(f"{sym}: {len(df)} свечей")
        else:
            log.warning(f"{sym} нет данных")
    if not data:
        log.error("Нет данных")
        return

    log.info(f"Тестирование {len(strategies)} стратегий на {len(data)} монетах...")
    results = []
    for st in strategies:
        total_trades = 0; pnls = []
        for sym, df in data.items():
            try:
                sig = st.get_signals(df)
                res = backtest_strategy(df, sig, SL_ATR_MULT, TP_ATR_MULT, MAX_HOLD_BARS)
                if res["trades"] > 0:
                    total_trades += res["trades"]
                    pnls.extend([res["avg_pnl"]] * res["trades"])
            except Exception as e:
                log.error(f"Ошибка {st.name} на {sym}: {e}")
        if total_trades == 0:
            results.append((st.name, 0, 0.0, 0.0, 0.0, 0.0))
            continue
        avg_pnl = np.mean(pnls)
        # Моделируем рост капитала по всем сделкам подряд
        eq = [INITIAL_CAPITAL]
        for p in pnls: eq.append(eq[-1] * (1 + p/100))
        total_pnl = (eq[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        eqs = pd.Series(eq)
        maxdd = (eqs.cummax() - eqs).max() / eqs.cummax().max() * 100
        wr = (sum(1 for p in pnls if p > 0) / len(pnls)) * 100
        results.append((st.name, total_trades, wr, avg_pnl, total_pnl, maxdd))

    # Сортировка по общему P&L
    results.sort(key=lambda x: x[4], reverse=True)
    print("\n" + "="*90)
    print(f"{'Стратегия':<25} {'Сделок':>6} {'WinRate%':>9} {'Avg P&L%':>10} {'Total P&L%':>11} {'MaxDD%':>8}")
    print("-"*90)
    for r in results:
        print(f"{r[0]:<25} {r[1]:>6} {r[2]:>8.1f} {r[3]:>9.2f} {r[4]:>10.2f} {r[5]:>7.2f}")
    print("="*90)

if __name__ == "__main__":
    main()
