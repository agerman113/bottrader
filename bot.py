#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ГЛУБОКИЙ АНАЛИЗ ЛИДЕРОВ БЭКТЕСТА
=================================
Тестируем отобранные стратегии на расширенной истории (2000 свечей 5m),
с разбивкой по периодам и расчётом дополнительных метрик.
"""

import os, time, logging, numpy as np, pandas as pd, ccxt
from typing import Dict, List, Callable
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
#               КОНФИГУРАЦИЯ
# ============================================================
SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT", "XRP/USDT:USDT",
    "SOL/USDT:USDT", "ADA/USDT:USDT", "TRX/USDT:USDT", "AVAX/USDT:USDT",
    "DOT/USDT:USDT", "LTC/USDT:USDT",
]
TIMEFRAME = "5m"           # можно заменить на "15m" или "1h"
LIMIT = 2000               # 2000 свечей 5m ≈ 7 дней
SL_ATR_MULT = 1.5
TP_ATR_MULT = 3.0
MAX_HOLD_BARS = 200
INITIAL_CAPITAL = 1000

# Стратегии-лидеры (название и функция сигнала)
STRATEGIES = {
    "Keltner Reversal": keltner_rev,
    "Stochastic": stochastic_signal,
    "CCI": cci_signal,
    "MFI": mfi_signal,
    "Aroon": aroon_signal,
    "VWAP Reversal": vwap_reversal,
    "Stochastic + Trend": stochastic_trend,
}

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

# ---------------------- БЭКТЕСТ С РАЗБИВКОЙ ----------------------
def backtest_split(df, signals, sl_mult, tp_mult, max_bars, split_date):
    """Делит данные на train и test по дате split_date, возвращает метрики для каждого."""
    train_df = df[df.index < split_date]
    test_df = df[df.index >= split_date]
    def run(data):
        capital = INITIAL_CAPITAL
        in_pos = False; side = 0; entry = 0; bars = 0
        trades = []
        for i in range(1, len(data)):
            if in_pos:
                bars += 1
                cur = data["close"].iloc[i]
                atr_val = atr(data).iloc[i]
                if side == 1:
                    if cur <= entry - sl_mult * atr_val or cur >= entry + tp_mult * atr_val or bars >= max_bars:
                        pnl = (cur - entry) / entry * 100
                        trades.append(pnl); in_pos = False
                else:
                    if cur >= entry + sl_mult * atr_val or cur <= entry - tp_mult * atr_val or bars >= max_bars:
                        pnl = (entry - cur) / entry * 100
                        trades.append(pnl); in_pos = False
            else:
                sig = signals.loc[data.index].iloc[i]
                if sig != 0 and not pd.isna(sig):
                    in_pos = True; side = sig; entry = data["close"].iloc[i]; bars = 0
        return trades
    train_trades = run(train_df)
    test_trades = run(test_df)
    def metrics(trades):
        if not trades: return {"trades":0,"winrate":0,"avg_pnl":0,"total_pnl":0,"maxdd":0}
        wins = sum(1 for p in trades if p > 0)
        wr = wins / len(trades) * 100
        avg = np.mean(trades)
        eq = [INITIAL_CAPITAL]
        for p in trades: eq.append(eq[-1] * (1 + p/100))
        total_pnl = (eq[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        eqs = pd.Series(eq)
        maxdd = (eqs.cummax() - eqs).max() / eqs.cummax().max() * 100
        return {"trades":len(trades),"winrate":wr,"avg_pnl":avg,"total_pnl":total_pnl,"maxdd":maxdd}
    return metrics(train_trades), metrics(test_trades)

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

    # Общий split_date: 70% времени в train, 30% в test
    all_dates = pd.concat([df.index for df in data.values()])
    split_date = all_dates.quantile(0.7)

    log.info(f"Дата разделения train/test: {split_date}")
    log.info(f"Тестируем {len(STRATEGIES)} стратегий на {len(data)} монетах...")

    # Таблица результатов
    results = []
    for name, func in STRATEGIES.items():
        log.info(f"--- {name} ---")
        all_train_trades = []
        all_test_trades = []
        for sym, df in data.items():
            try:
                signals = func(df)
                train_m, test_m = backtest_split(df, signals, SL_ATR_MULT, TP_ATR_MULT, MAX_HOLD_BARS, split_date)
                if train_m["trades"] > 0:
                    all_train_trades.extend([train_m["avg_pnl"]] * train_m["trades"])  # упрощённо для сводки
                if test_m["trades"] > 0:
                    all_test_trades.extend([test_m["avg_pnl"]] * test_m["trades"])
            except Exception as e:
                log.error(f"   Ошибка {sym}: {e}")
        if len(all_train_trades) + len(all_test_trades) == 0:
            results.append((name, 0,0,0,0,0,0,0,0,0))
            continue
        # Общие метрики train
        train_pnls = all_train_trades
        test_pnls = all_test_trades
        def agg_metrics(pnls):
            if not pnls: return (0,0,0,0)
            avg = np.mean(pnls)
            eq = [INITIAL_CAPITAL]
            for p in pnls: eq.append(eq[-1] * (1 + p/100))
            total = (eq[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
            eqs = pd.Series(eq)
            dd = (eqs.cummax() - eqs).max() / eqs.cummax().max() * 100
            wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
            return (len(pnls), wr, avg, total, dd)
        tr_t, tr_wr, tr_avg, tr_pnl, tr_dd = agg_metrics(train_pnls)
        te_t, te_wr, te_avg, te_pnl, te_dd = agg_metrics(test_pnls)
        results.append((name, tr_t, tr_wr, tr_pnl, tr_dd, te_t, te_wr, te_pnl, te_dd))

    # Вывод
    print("\n" + "="*120)
    print(f"{'Стратегия':<20} {'Train сделок':>11} {'Train WR%':>9} {'Train P&L%':>11} {'Train DD%':>9} {'Test сделок':>11} {'Test WR%':>9} {'Test P&L%':>11} {'Test DD%':>9}")
    print("-"*120)
    for r in sorted(results, key=lambda x: x[6], reverse=True):  # сортировка по Test P&L
        print(f"{r[0]:<20} {r[1]:>11} {r[2]:>8.1f} {r[3]:>10.2f} {r[4]:>8.2f} {r[5]:>11} {r[6]:>8.1f} {r[7]:>10.2f} {r[8]:>8.2f}")
    print("="*120)

if __name__ == "__main__":
    main()
