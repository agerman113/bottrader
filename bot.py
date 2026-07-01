#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
СИНТЕТИЧЕСКИЙ СТРЕСС-ТЕСТ СТРАТЕГИЙ-ЛИДЕРОВ
=============================================
- 42 монеты, 5000 свечей 5m (~17 дней истории).
- Реалистичная комиссия 0.04% + случайное проскальзывание 0.01%-0.05%.
- Три периода: train (60%), validation (20%), test (20%).
- Метрики: WinRate, Total P&L, MaxDD, Recovery Factor, Sharpe Ratio.
"""

import os, time, logging, numpy as np, pandas as pd, ccxt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
#               КОНФИГУРАЦИЯ
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
LIMIT = 5000                     # 5000 свечей 5m ≈ 17 дней
SL_ATR_MULT = 1.5
TP_ATR_MULT = 3.0
MAX_HOLD_BARS = 200
INITIAL_CAPITAL = 1000
COMMISSION = 0.0004              # 0.04% от объёма
SLIPPAGE_MIN = 0.0001           # 0.01%
SLIPPAGE_MAX = 0.0005           # 0.05%

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

exchange = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "linear"}})

def fetch_ohlcv(symbol, timeframe, limit):
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    except Exception as e:
        log.error(f"Ошибка загрузки {symbol}: {e}")
        return pd.DataFrame()

# ---------------------- ИНДИКАТОРЫ ----------------------
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
    trend = pd.Series(1.0, index=df.index)
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

# ---------------------- ФУНКЦИИ СИГНАЛОВ ----------------------
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

def aroon_signal(df):
    up, down = aroon(df)
    signal = pd.Series(0, index=df.index)
    signal[(up > 70) & (down < 30)] = 1
    signal[(down > 70) & (up < 30)] = -1
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

def stochastic_trend(df):
    k, d = stochastic(df)
    e50 = ema(df["close"], 50)
    signal = pd.Series(0, index=df.index)
    signal[(k < 20) & (k > d) & (df["close"] > e50)] = 1
    signal[(k > 80) & (k < d) & (df["close"] < e50)] = -1
    return signal

STRATEGIES = {
    "Keltner Reversal": keltner_rev,
    "Stochastic": stochastic_signal,
    "CCI": cci_signal,
    "MFI": mfi_signal,
    "Aroon": aroon_signal,
    "VWAP Reversal": vwap_reversal,
    "Stochastic + Trend": stochastic_trend,
}

# ---------------------- БЭКТЕСТ С РЕАЛИСТИЧНЫМИ УСЛОВИЯМИ ----------------------
def backtest_realistic(df, signals, sl_mult, tp_mult, max_bars, start_date, end_date):
    """Тест на заданном временном отрезке с комиссией и проскальзыванием."""
    mask = (df.index >= start_date) & (df.index <= end_date)
    data = df[mask].copy()
    if data.empty:
        return []
    capital = INITIAL_CAPITAL
    in_pos = False; side = 0; entry = 0; bars = 0
    trades = []
    for i in range(1, len(data)):
        if in_pos:
            bars += 1
            cur = data["close"].iloc[i]
            atr_val = atr(data).iloc[i]
            slippage = np.random.uniform(SLIPPAGE_MIN, SLIPPAGE_MAX)
            if side == 1:
                if cur <= entry - sl_mult * atr_val or cur >= entry + tp_mult * atr_val or bars >= max_bars:
                    exit_price = cur * (1 - slippage)  # проскальзывание против нас при закрытии long
                    pnl = (exit_price - entry) / entry * 100 - COMMISSION * 100
                    trades.append(pnl); in_pos = False
            else:
                if cur >= entry + sl_mult * atr_val or cur <= entry - tp_mult * atr_val or bars >= max_bars:
                    exit_price = cur * (1 + slippage)  # проскальзывание при закрытии short
                    pnl = (entry - exit_price) / entry * 100 - COMMISSION * 100
                    trades.append(pnl); in_pos = False
        else:
            sig = signals.loc[data.index].iloc[i]
            if sig != 0 and not pd.isna(sig):
                in_pos = True; side = sig; entry = data["close"].iloc[i]; bars = 0
    return trades

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

    # Определяем общий временной диапазон
    min_date = min(df.index.min() for df in data.values())
    max_date = max(df.index.max() for df in data.values())
    total_duration = max_date - min_date
    train_end = min_date + 0.6 * total_duration
    val_end = min_date + 0.8 * total_duration

    log.info(f"Train: {min_date} -> {train_end}")
    log.info(f"Validation: {train_end} -> {val_end}")
    log.info(f"Test: {val_end} -> {max_date}")

    results = []
    for name, func in STRATEGIES.items():
        log.info(f"--- {name} ---")
        train_trades = []; val_trades = []; test_trades = []
        for sym, df in data.items():
            try:
                signals = func(df)
                train_trades.extend(backtest_realistic(df, signals, SL_ATR_MULT, TP_ATR_MULT, MAX_HOLD_BARS, min_date, train_end))
                val_trades.extend(backtest_realistic(df, signals, SL_ATR_MULT, TP_ATR_MULT, MAX_HOLD_BARS, train_end, val_end))
                test_trades.extend(backtest_realistic(df, signals, SL_ATR_MULT, TP_ATR_MULT, MAX_HOLD_BARS, val_end, max_date))
            except Exception as e:
                log.error(f"   Ошибка {sym}: {e}")

        def metrics(trades):
            if not trades: return (0,0,0,0,0,0)
            wins = sum(1 for p in trades if p > 0)
            wr = wins / len(trades) * 100
            avg = np.mean(trades)
            eq = [INITIAL_CAPITAL]
            for p in trades: eq.append(eq[-1] * (1 + p/100))
            total_pnl = (eq[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
            eqs = pd.Series(eq)
            maxdd = (eqs.cummax() - eqs).max() / eqs.cummax().max() * 100
            # Sharpe Ratio (условный, без процентной ставки)
            sharpe = np.mean(trades) / np.std(trades) * np.sqrt(len(trades)) if np.std(trades) > 0 else 0
            recovery = eq[-1] / max(eq) if max(eq) > 0 else 0
            return (len(trades), wr, avg, total_pnl, maxdd, sharpe, recovery)

        tr_m = metrics(train_trades)
        v_m = metrics(val_trades)
        te_m = metrics(test_trades)
        results.append((name, tr_m, v_m, te_m))

    # Вывод итоговой таблицы
    print("\n" + "="*140)
    print(f"{'Стратегия':<20} {'Train сделок':>11} {'Train WR%':>9} {'Train P&L%':>11} {'Train DD%':>9} {'Valid сделок':>11} {'Valid WR%':>9} {'Valid P&L%':>11} {'Test сделок':>11} {'Test WR%':>9} {'Test P&L%':>11} {'Test DD%':>9}")
    print("-"*140)
    for name, tr, v, te in sorted(results, key=lambda x: x[3][3] if x[3][0] > 0 else 0, reverse=True):
        print(f"{name:<20} {tr[0]:>11} {tr[1]:>8.1f} {tr[3]:>10.2f} {tr[4]:>8.2f} "
              f"{v[0]:>11} {v[1]:>8.1f} {v[3]:>10.2f} "
              f"{te[0]:>11} {te[1]:>8.1f} {te[3]:>10.2f} {te[4]:>8.2f}")
    print("="*140)

    # Сохраняем результаты в CSV для дальнейшего анализа
    csv_data = []
    for name, tr, v, te in results:
        csv_data.append({
            "strategy": name,
            "train_trades": tr[0], "train_wr": tr[1], "train_pnl": tr[3], "train_dd": tr[4],
            "val_trades": v[0], "val_wr": v[1], "val_pnl": v[3],
            "test_trades": te[0], "test_wr": te[1], "test_pnl": te[3], "test_dd": te[4],
        })
    pd.DataFrame(csv_data).to_csv("synthetic_test_results.csv", index=False)
    log.info("Результаты сохранены в synthetic_test_results.csv")

if __name__ == "__main__":
    main()
