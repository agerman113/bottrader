#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ГИБРИДНЫЙ БОТ: SuperTrend + RSI + НЕЙРОСЕТЬ + БАЛЛЫ (архитектура v10.2)
- Сканирует топ-монеты по волатильности (или белый список)
- Сигналы на основе RSI-пересечений, фильтруются SuperTrend и нейросетью
- Первая сделка (10 мин) – максимальный риск, далее риск плавно снижается
- Полноценный мониторинг позиции с трейлинг-стопом и частичным безубытком
- Автоматическая симуляция при нехватке реальных средств
"""

import os, time, logging, json, requests, numpy as np, ccxt, pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import MinMaxScaler

load_dotenv()

# ============================================================
#                 КОНФИГУРАЦИЯ
# ============================================================
FIXED_SYMBOL        = ""               # если пусто – авто‑сканер
WHITELIST_SYMBOLS   = [                # белый список (используется при FIXED_SYMBOL="")
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
    "XRP/USDT:USDT", "DOGE/USDT:USDT", "ADA/USDT:USDT",
    "BNB/USDT:USDT", "LINK/USDT:USDT", "MATIC/USDT:USDT", "DOT/USDT:USDT",
]
TIMEFRAME           = "1m"             # торговый таймфрейм
SCAN_TIMEFRAME      = "5m"             # таймфрейм для сканера волатильности
LEVERAGE            = 1                # кредитное плечо
STOP_LOSS_PERCENT   = 0.25             # базовый SL (подстраивается под ATR)
TAKE_PROFIT_PERCENT = 1.0              # базовый TP (подстраивается под ATR)

# Система баллов и стилей
SCORE_WIN = 2; SCORE_LOSS = -2; SCORE_TIMEOUT = 0; SCORE_PARTIAL = 0.5
MIN_TRADES_FOR_ADAPT = 5
MODES = [
    {"name": "Агрессивный",  "period": 3,  "oversold": 40, "overbought": 60},
    {"name": "Умеренный",    "period": 5,  "oversold": 30, "overbought": 70},
    {"name": "Консервативный","period": 7, "oversold": 20, "overbought": 80},
]

# Нейросеть
NN_HIDDEN=32; NN_LR=0.01; NN_EPOCHS=30; NN_BATCH=16
NN_RETRAIN_INTERVAL = 1800           # секунд
NN_LOOKBACK = 500
NN_CONFIDENCE_MIN = 0.6              # адаптивный порог

# Сканер монет
SCAN_TOP_N = 30
SCAN_INTERVAL = 14400                # 4 часа
SCAN_BARS = 60
MIN_VOLUME_USDT = 5_000_000
MIN_ATR_PCT = 0.5                    # минимальная волатильность

# SuperTrend
SUPERTREND_PERIOD = 10
SUPERTREND_MULT = 3.0

# Управление риском
INITIAL_RISK = 1.0                  # первая сделка 100%
RISK_REDUCTION_STEP = 0.2           # –20% после каждой сделки
MIN_RISK = 0.1                      # минимальный риск 10%

# Трейлинг и частичный безубыток
TRAILING_ATR_MULT = 2.0
TRAILING_OFFSET_MULT = 1.5
MIN_TRAILING_STEP = 0.4             # %
MIN_TRAILING_OFFSET = 0.6           # %
MIN_PROFIT_FOR_TRAIL = 1.0          # %
PARTIAL_BE_ENABLED = True
PARTIAL_BE_PROFIT = 0.2             # % прибыли для частичного закрытия

# Прочее
TRADE_MAX_LIFETIME = 7200           # секунд
REPORT_INTERVAL = 1800
STATE_FILE = "hybrid_state.json"
TRADES_FILE = "hybrid_trades.json"

# ============================================================
#                        ЛОГИРОВАНИЕ
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]  %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler("hybrid_bot.log", encoding="utf-8")],
)
log = logging.getLogger("HybridBot")

# ============================================================
#                    НЕЙРОСЕТЕВОЙ ПРЕДИКТОР
# ============================================================
class NeuralPredictor:
    def __init__(self, input_size=10, hidden=32, lr=0.01):
        self.W1 = np.random.randn(input_size, hidden) * np.sqrt(2./input_size)
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, 1) * np.sqrt(2./hidden)
        self.b2 = np.zeros(1)
        self.m_W1, self.v_W1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.m_b1, self.v_b1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.m_W2, self.v_W2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.m_b2, self.v_b2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        self.lr = lr; self.beta1=0.9; self.beta2=0.999; self.eps=1e-8; self.t=0

    def sigmoid(self, x): return 1/(1+np.exp(-x))
    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    def backward(self, X, y, output):
        m = X.shape[0]
        dZ2 = output - y
        dW2 = (self.a1.T @ dZ2)/m
        db2 = np.sum(dZ2, axis=0, keepdims=True).flatten()/m
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (self.z1 > 0)
        dW1 = (X.T @ dZ1)/m
        db1 = np.sum(dZ1, axis=0)/m
        self.t += 1
        for param, m_, v_, grad in zip(
            [self.W1, self.b1, self.W2, self.b2],
            [self.m_W1, self.m_b1, self.m_W2, self.m_b2],
            [self.v_W1, self.v_b1, self.v_W2, self.v_b2],
            [dW1, db1, dW2, db2]
        ):
            m_ = self.beta1*m_ + (1-self.beta1)*grad
            v_ = self.beta2*v_ + (1-self.beta2)*(grad**2)
            m_hat = m_/(1-self.beta1**self.t)
            v_hat = v_/(1-self.beta2**self.t)
            param -= self.lr * m_hat / (np.sqrt(v_hat)+self.eps)
    def fit(self, X, y, epochs=30, batch=16):
        for epoch in range(epochs):
            idx = np.random.permutation(len(X))
            for i in range(0, len(X), batch):
                bi = idx[i:i+batch]
                self.forward(X[bi])
                self.backward(X[bi], y[bi].reshape(-1,1), self.a2)
    def predict_proba(self, X): return self.forward(X)

# ============================================================
#             БЕЗОПАСНЫЕ ОБЁРТКИ API
# ============================================================
exchange = ccxt.bybit({
    "apiKey": os.getenv("BYBIT_API_KEY"),
    "secret": os.getenv("BYBIT_API_SECRET"),
    "enableRateLimit": True, "timeout": 10_000,
    "options": {"defaultType": "linear"},
})

def safe_api_call(func, *args, retries=3, delay=1.0, **kwargs):
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except ccxt.RateLimitExceeded:
            log.warning("Rate limit, пауза 5с")
            time.sleep(5)
        except ccxt.NetworkError as e:
            log.warning(f"Сеть: {e}")
            time.sleep(delay); delay *= 2
        except Exception as e:
            if attempt == retries-1: raise
            time.sleep(delay); delay *= 2
    return None

def safe_fetch_ohlcv(symbol, tf, limit=150):
    try: return safe_api_call(exchange.fetch_ohlcv, symbol, tf, limit=limit) or []
    except: return []

def safe_fetch_ticker(symbol):
    try: return safe_api_call(exchange.fetch_ticker, symbol)
    except: return None

def safe_fetch_positions(symbols=None):
    try: return safe_api_call(exchange.fetch_positions, symbols) or []
    except: return []

# ============================================================
#             ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================
def calc_rsi(close, period):
    d = close.diff()
    gain = d.clip(lower=0)
    loss = (-d).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float('nan'))
    return 100 - (100/(1+rs))

def calc_atr(df, period=14):
    hi, lo, pc = df['high'], df['low'], df['close'].shift(1)
    tr = pd.concat([hi-lo, (hi-pc).abs(), (lo-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def calc_supertrend(df, period=10, mult=3.0):
    atr = calc_atr(df, period)
    hl2 = (df['high'] + df['low'])/2
    ub = hl2 + mult*atr
    lb = hl2 - mult*atr
    trend = pd.Series(1, index=df.index)
    st_line = pd.Series(0., index=df.index)
    for i in range(1, len(df)):
        c = df['close'].iloc[i]
        if c > st_line.iloc[i-1]:
            st_line.iloc[i] = max(lb.iloc[i], st_line.iloc[i-1])
            trend.iloc[i] = 1
        else:
            st_line.iloc[i] = min(ub.iloc[i], st_line.iloc[i-1])
            trend.iloc[i] = -1
        if st_line.iloc[i-1] == 0:
            st_line.iloc[i] = lb.iloc[i] if c > lb.iloc[i] else ub.iloc[i]
            trend.iloc[i] = 1 if c > lb.iloc[i] else -1
    return trend == 1, st_line

# ============================================================
#                КЛАСС ГИБРИДНОГО БОТА
# ============================================================
class HybridBot:
    def __init__(self):
        self.symbol = None
        self.timeframe = TIMEFRAME
        self.total_trades = 0
        self.wins = 0; self.losses = 0; self.timeouts = 0
        self.score = 0
        self.active_mode_idx = 0
        self.mode_stats = [{'trades':0,'score':0,'avg':0} for _ in MODES]

        self.nn = None
        self.scaler = MinMaxScaler()
        self.nn_trained = False
        self.last_nn_train = 0
        self.current_atr = 0.0
        self.supertrend_dir = "neutral"

        self.live_trading = True   # будет переопределено, если не хватает средств
        self.risk_fraction = INITIAL_RISK
        self.first_trade_done = False
        self.start_time = time.time()

        self.last_scan_time = 0
        self._last_status_time = 0
        self._last_stat_time = 0
        self.active_trade = None

    # ---------- Сканер монет ----------
    def scan_best_symbol(self):
        if WHITELIST_SYMBOLS:
            syms = WHITELIST_SYMBOLS
            log.info(f"Сканирование белого списка ({len(syms)} монет)")
        else:
            # старый сканер по объёму
            try:
                tickers = exchange.fetch_tickers()
            except: return "BTC/USDT:USDT"
            candidates = []
            for sym, t in tickers.items():
                if not sym.endswith(":USDT"): continue
                vol = t.get('quoteVolume', 0)
                if vol >= MIN_VOLUME_USDT: candidates.append((sym, vol))
            candidates.sort(key=lambda x: x[1], reverse=True)
            syms = [s for s,_ in candidates[:SCAN_TOP_N]]
        best_sym = "BTC/USDT:USDT"
        best_atr = 0
        for sym in syms:
            df = safe_fetch_ohlcv(sym, SCAN_TIMEFRAME, limit=SCAN_BARS+5)
            if len(df) < SCAN_BARS: continue
            df = pd.DataFrame(df, columns=['ts','o','h','l','c','v'])
            atr = ((df['h'] - df['l'])/df['c']).mean()
            if atr > best_atr and atr >= MIN_ATR_PCT:
                best_atr = atr
                best_sym = sym
        log.info(f"Лучшая монета по ATR: {best_sym} ({best_atr*100:.2f}%)")
        return best_sym

    # ---------- Обучение нейросети ----------
    def train_nn(self, df):
        if len(df) < 200: return
        df = df.copy()
        close = df['close']
        df['rsi'] = calc_rsi(close, 14)
        df['ema_short'] = close.ewm(9).mean()
        df['ema_long'] = close.ewm(21).mean()
        macd = close.ewm(12).mean() - close.ewm(26).mean()
        df['macd'] = macd
        df['macd_signal'] = macd.ewm(9).mean()
        df['macd_hist'] = macd - df['macd_signal']
        for lag in [1,2,3]: df[f'close_lag_{lag}'] = close.shift(lag)
        feat_cols = ['rsi','ema_short','ema_long','macd','macd_signal','macd_hist',
                     'close_lag_1','close_lag_2','close_lag_3']
        df = df.dropna()
        X = df[feat_cols].values
        y = (df['close'].shift(-1) > df['close']).astype(int).values[:-1]
        X, y = X[:len(y)], y
        if len(X) < 100: return
        self.scaler.fit(X)
        X_s = self.scaler.transform(X)
        if self.nn is None or self.nn.W1.shape[0] != X.shape[1]:
            self.nn = NeuralPredictor(input_size=X.shape[1], hidden=NN_HIDDEN, lr=NN_LR)
        self.nn.fit(X_s, y, epochs=NN_EPOCHS, batch=NN_BATCH)
        self.nn_trained = True
        self.last_nn_train = time.time()

    def nn_predict(self, df):
        if not self.nn_trained: return 0.5
        close = df['close']
        df = df.copy()
        df['rsi'] = calc_rsi(close, 14)
        df['ema_short'] = close.ewm(9).mean()
        df['ema_long'] = close.ewm(21).mean()
        macd = close.ewm(12).mean() - close.ewm(26).mean()
        df['macd'] = macd
        df['macd_signal'] = macd.ewm(9).mean()
        df['macd_hist'] = macd - df['macd_signal']
        for lag in [1,2,3]: df[f'close_lag_{lag}'] = close.shift(lag)
        feat_cols = ['rsi','ema_short','ema_long','macd','macd_signal','macd_hist',
                     'close_lag_1','close_lag_2','close_lag_3']
        last = df[feat_cols].iloc[-1:].values
        try: last_s = self.scaler.transform(last)
        except: return 0.5
        return self.nn.predict_proba(last_s)[0,0]

    # ---------- Сигнал ----------
    def get_signal(self, df, mode, forced=False):
        if len(df) < mode['period']+2: return None, None
        close = df['close']
        rsi = calc_rsi(close, mode['period'])
        rsi_val = rsi.iloc[-1]
        signal = None
        if rsi.iloc[-2] <= mode['oversold'] and rsi.iloc[-1] > mode['oversold']:
            signal = 'long'
        elif rsi.iloc[-2] >= mode['overbought'] and rsi.iloc[-1] < mode['overbought']:
            signal = 'short'
        if signal and not forced:
            # трендовый фильтр
            if self.first_trade_done:
                if signal == 'long' and self.supertrend_dir == 'bearish': signal = None
                elif signal == 'short' and self.supertrend_dir == 'bullish': signal = None
            # нейросетевой фильтр (адаптивный порог)
            if signal:
                proba = self.nn_predict(df)
                if signal == 'long' and proba < self.adaptive_confidence: signal = None
                elif signal == 'short' and proba > 1-self.adaptive_confidence: signal = None
        return signal, rsi_val

    # ---------- Открытие позиции ----------
    def open_position(self, signal, price, risk_override=None):
        fraction = risk_override if risk_override is not None else self.risk_fraction
        if self.live_trading:
            free = exchange.fetch_balance()['USDT']['free']
            avail = free * fraction
            if avail <= 0: return False
            raw_qty = avail / price
            market = exchange.market(self.symbol)
            min_amt = float(market.get('limits', {}).get('amount', {}).get('min', 0) or 0)
            min_cost = float(market.get('limits', {}).get('cost', {}).get('min', 0) or 0)
            if min_amt and raw_qty < min_amt:
                if free >= min_amt * price: raw_qty = min_amt
                else: return False
            if min_cost and raw_qty * price < min_cost:
                if free >= min_cost: raw_qty = min_cost / price
                else: return False
            qty = float(exchange.amount_to_precision(self.symbol, raw_qty))
            if qty <= 0: return False
            # динамические TP/SL
            atr = self.current_atr if self.current_atr > 0 else 0.003
            tp_move = max(TAKE_PROFIT_PERCENT/100, atr*2)
            sl_move = max(STOP_LOSS_PERCENT/100, atr*0.5)
            if signal == 'long':
                side='buy'; tp=price*(1+tp_move); sl=price*(1-sl_move)
            else:
                side='sell'; tp=price*(1-tp_move); sl=price*(1+sl_move)
            tp = float(exchange.price_to_precision(self.symbol, tp))
            sl = float(exchange.price_to_precision(self.symbol, sl))
            try:
                exchange.create_order(self.symbol, 'market', side, qty,
                                      params={'takeProfit':tp, 'stopLoss':sl})
                log.info(f"✅ {signal.upper()} {qty} @ {price:.4f} TP={tp:.4f} SL={sl:.4f}")
                self.active_trade = {'side':signal, 'entry':price, 'tp':tp, 'sl':sl, 'qty':qty,
                                     'time':time.time(), 'peak':price, 'phase':1, 'partial_done':False,
                                     'trailing_active':False}
                return True
            except Exception as e:
                log.error(f"Ошибка открытия: {e}")
                return False
        else:
            # симуляция
            log.info(f"📈 СИМУЛЯЦИЯ {signal.upper()} @ {price:.4f}")
            self.active_trade = {'side':signal, 'entry':price, 'tp':price, 'sl':price,
                                 'qty':0, 'time':time.time(), 'virtual':True, 'bars_held':0}
            return True

    # ---------- Мониторинг позиции ----------
    def monitor_position(self):
        if not self.active_trade: return
        trade = self.active_trade
        if trade.get('virtual'):
            # виртуальная сделка
            df = safe_fetch_ohlcv(self.symbol, self.timeframe, limit=10)
            if len(df) < 2: return
            df = pd.DataFrame(df, columns=['ts','o','h','l','c','v'])
            high, low, close = df['h'].iloc[-1], df['l'].iloc[-1], df['c'].iloc[-1]
            trade['bars_held'] += 1
            result = None
            if trade['side'] == 'long':
                if high >= trade['tp']: result = 'tp'
                elif low <= trade['sl']: result = 'sl'
            else:
                if low <= trade['tp']: result = 'tp'
                elif high >= trade['sl']: result = 'sl'
            if result is None and trade['bars_held'] >= 10: result = 'timeout'
            if result:
                self.add_trade_result(result, trade['entry'], close, trade['side'])
                self.active_trade = None
            return

        # реальная позиция
        deadline = trade['time'] + TRADE_MAX_LIFETIME
        while True:
            if time.time() >= deadline:
                log.warning("Дедлайн — закрываем")
                self.close_position(trade['qty'], trade['side'])
                self.add_trade_result('timeout', trade['entry'], exchange.fetch_ticker(self.symbol)['last'], trade['side'])
                self.active_trade = None
                return
            time.sleep(15)
            pos = safe_fetch_positions([self.symbol])
            active = [p for p in pos if float(p.get('contracts',0))>0 and p.get('side')==trade['side']]
            if not active:
                # позиция уже закрыта биржей
                cur = exchange.fetch_ticker(self.symbol)['last']
                self.add_trade_result('tp' if cur>trade['entry'] else 'sl', trade['entry'], cur, trade['side'])
                self.active_trade = None
                return
            cur = exchange.fetch_ticker(self.symbol)['last']
            pnl_pct = (cur/trade['entry']-1)*100 if trade['side']=='long' else (trade['entry']/cur-1)*100

            # Частичный безубыток
            if PARTIAL_BE_ENABLED and not trade['partial_done'] and pnl_pct >= PARTIAL_BE_PROFIT:
                close_qty = trade['qty'] * 0.5
                if close_qty > 0:
                    try: exchange.create_order(self.symbol, 'market', 'sell' if trade['side']=='long' else 'buy',
                                               close_qty, params={'reduceOnly':True})
                    except: pass
                    trade['partial_done'] = True
                    # двигаем SL в безубыток
                    new_sl = trade['entry'] * (1 + 0.0005) if trade['side']=='long' else trade['entry'] * (1 - 0.0005)
                    try: exchange.create_order(self.symbol, 'market', 'sell' if trade['side']=='long' else 'buy',
                                               0, params={'stopLoss':exchange.price_to_precision(self.symbol, new_sl)})
                    except: pass
                    trade['sl'] = new_sl
                    trade['phase'] = 2
                    log.info(f"Частичный безубыток, новый SL={new_sl:.4f}")

            # Активация трейлинга при достижении MIN_PROFIT_FOR_TRAIL%
            if not trade.get('trailing_active') and pnl_pct >= MIN_PROFIT_FOR_TRAIL:
                trade['trailing_active'] = True
                log.info(f"Трейлинг активирован @ {cur:.4f}")

            # Трейлинг
            if trade.get('trailing_active') and trade['phase']>=2:
                if trade['side']=='long':
                    if cur > trade.get('peak', cur): trade['peak'] = cur
                    new_sl = trade['peak'] * (1 - MIN_TRAILING_OFFSET/100)
                    if new_sl > trade['sl']:
                        try: exchange.create_order(self.symbol, 'market', 'sell', 0,
                                                   params={'stopLoss':exchange.price_to_precision(self.symbol, new_sl)})
                        except: pass
                        trade['sl'] = new_sl
                        log.info(f"Трейлинг: SL → {new_sl:.4f}")
                else:
                    if cur < trade.get('peak', cur): trade['peak'] = cur
                    new_sl = trade['peak'] * (1 + MIN_TRAILING_OFFSET/100)
                    if new_sl < trade['sl']:
                        try: exchange.create_order(self.symbol, 'market', 'buy', 0,
                                                   params={'stopLoss':exchange.price_to_precision(self.symbol, new_sl)})
                        except: pass
                        trade['sl'] = new_sl
                        log.info(f"Трейлинг: SL → {new_sl:.4f}")

            log.info(f"Мониторинг: цена={cur:.4f} P&L={pnl_pct:+.2f}% SL={trade['sl']:.4f}")

    def close_position(self, qty, side):
        close_side = 'sell' if side=='long' else 'buy'
        try: exchange.create_order(self.symbol, 'market', close_side, qty, params={'reduceOnly':True})
        except Exception as e: log.error(f"Закрытие не удалось: {e}")

    # ---------- Учёт результатов ----------
    def add_trade_result(self, exit_type, entry, exit_price, side):
        profit = (exit_price-entry)/entry*100 if side=='long' else (entry-exit_price)/entry*100
        if exit_type=='tp': sc=SCORE_WIN; self.wins+=1
        elif exit_type=='sl': sc=SCORE_LOSS; self.losses+=1
        else:
            sc = SCORE_PARTIAL if profit >= TAKE_PROFIT_PERCENT/2 else SCORE_TIMEOUT
            self.timeouts+=1
        self.score += sc
        self.total_trades += 1
        idx = self.active_mode_idx
        self.mode_stats[idx]['trades'] += 1
        self.mode_stats[idx]['score'] += sc
        log.info(f"🏁 {exit_type.upper()} {side} {profit:+.2f}% балл {sc:+d} (всего {self.score})")
        if not self.first_trade_done:
            self.first_trade_done = True
            self.risk_fraction = max(MIN_RISK, self.risk_fraction - RISK_REDUCTION_STEP)
            log.info(f"Риск снижен до {self.risk_fraction*100:.0f}%")
        if self.total_trades >= MIN_TRADES_FOR_ADAPT:
            for i, s in enumerate(self.mode_stats):
                if s['trades']>0: s['avg']=s['score']/s['trades']
            best = max(range(len(self.mode_stats)), key=lambda i: self.mode_stats[i]['avg'])
            if best != self.active_mode_idx:
                log.info(f"Стиль изменён: {MODES[self.active_mode_idx]['name']} → {MODES[best]['name']}")
                self.active_mode_idx = best

    # ---------- Главный цикл ----------
    def run(self):
        log.info("Запуск гибридного бота")
        # Проверка баланса
        try:
            balance = exchange.fetch_balance()['USDT']['free']
            if balance < 5:
                log.warning(f"Баланс {balance:.2f}USDT — переход в симуляцию")
                self.live_trading = False
        except: self.live_trading = False

        if not FIXED_SYMBOL:
            self.symbol = self.scan_best_symbol()
        else:
            self.symbol = FIXED_SYMBOL

        log.info(f"Пара: {self.symbol} | Режим: {'LIVE' if self.live_trading else 'SIM'} | Риск: {self.risk_fraction*100:.0f}%")
        if self.live_trading:
            try: exchange.set_leverage(LEVERAGE, self.symbol)
            except: pass

        # первичное обучение
        df = safe_fetch_ohlcv(self.symbol, self.timeframe, limit=NN_LOOKBACK)
        if len(df) > 100:
            df = pd.DataFrame(df, columns=['ts','o','h','l','c','v'])
            self.train_nn(df)
            self.current_atr = ((df['h']-df['l'])/df['c']).tail(20).mean()
            _, st_line = calc_supertrend(df, SUPERTREND_PERIOD, SUPERTREND_MULT)
            self.supertrend_dir = 'bullish' if (df['c'].iloc[-1] > st_line.iloc[-1]) else 'bearish'

        self.adaptive_confidence = NN_CONFIDENCE_MIN
        self.last_signal_time = time.time()
        FORCED_TIMEOUT = 9.5*60

        while True:
            now = time.time()
            if not self.first_trade_done and (now - self.start_time) > 600:
                log.info("10 мин истекли, первая сделка не выполнена — переходим в обычный режим")
                self.first_trade_done = True   # фейковое завершение, чтобы включить фильтры

            df = safe_fetch_ohlcv(self.symbol, self.timeframe, limit=200)
            if len(df) < 50:
                time.sleep(10); continue
            df = pd.DataFrame(df, columns=['ts','o','h','l','c','v'])
            self.current_atr = ((df['h']-df['l'])/df['c']).tail(20).mean()
            _, st_line = calc_supertrend(df, SUPERTREND_PERIOD, SUPERTREND_MULT)
            self.supertrend_dir = 'bullish' if (df['c'].iloc[-1] > st_line.iloc[-1]) else 'bearish'

            # мониторинг позиции
            if self.active_trade:
                self.monitor_position()
                continue

            # адаптивный порог нейросети
            if self.first_trade_done:
                if now - self.last_signal_time > 900:
                    self.adaptive_confidence = max(0.3, self.adaptive_confidence - 0.1)
                    log.info(f"Порог NN снижен до {self.adaptive_confidence:.2f}")
                elif self.adaptive_confidence < NN_CONFIDENCE_MIN:
                    self.adaptive_confidence = min(NN_CONFIDENCE_MIN, self.adaptive_confidence + 0.05)

            mode = MODES[self.active_mode_idx]
            signal, rsi_val = self.get_signal(df, mode, forced=not self.first_trade_done)
            price = df['c'].iloc[-1]

            if not self.first_trade_done:
                if signal:
                    log.info(f"Первая сделка: {signal.upper()}")
                    self.open_position(signal, price, risk_override=1.0)
                elif now - self.start_time > FORCED_TIMEOUT:
                    force = 'long' if self.supertrend_dir != 'bearish' else 'short'
                    log.info(f"Принудительный вход {force.upper()}")
                    self.open_position(force, price, risk_override=1.0)
            else:
                if signal:
                    log.info(f"Сигнал {signal.upper()}")
                    self.open_position(signal, price)

            if now - self._last_status_time > 30:
                log.info(f"👁 {self.symbol} цена={price:.4f} RSI={rsi_val:.1f} ST={self.supertrend_dir} риск={self.risk_fraction*100:.0f}%")
                self._last_status_time = now

            if now - self.last_nn_train > NN_RETRAIN_INTERVAL:
                self.train_nn(df)
                self.last_nn_train = now

            time.sleep(5)

if __name__ == "__main__":
    bot = HybridBot()
    bot.run()
