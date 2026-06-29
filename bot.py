#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ГИБРИДНЫЙ БОТ (v2.0) – ПРИБЫЛЬНАЯ ВЕРСИЯ
- SuperTrend + RSI + нейросеть + адаптивный риск
- Улучшенные сигналы: подтверждение свечных паттернов, ATR-фильтр, старший тренд
- Корректный трейлинг‑стоп (через edit_order)
- Асинхронный мониторинг позиции без блокировки
- Защита от серий убытков (временная остановка торгов)
"""

import os, time, logging, json, numpy as np, ccxt, pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import MinMaxScaler

load_dotenv()

# ============================================================
#                 КОНФИГУРАЦИЯ (обновлена)
# ============================================================
FIXED_SYMBOL        = ""               # авто‑сканер
WHITELIST_SYMBOLS   = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
    "XRP/USDT:USDT", "DOGE/USDT:USDT", "ADA/USDT:USDT",
    "BNB/USDT:USDT", "LINK/USDT:USDT", "MATIC/USDT:USDT", "DOT/USDT:USDT",
]
TIMEFRAME           = "5m"             # основной ТФ для сигналов
FAST_TIMEFRAME      = "1m"             # для уточнения входа
LEVERAGE            = 1
STOP_LOSS_ATR_MULT  = 2.0             # SL = ATR*2
TAKE_PROFIT_ATR_MULT= 3.5             # TP = ATR*3.5
MIN_ATR_FOR_TRADE   = 0.003           # минимальная ATR (0.3%)

SCORE_WIN = 2; SCORE_LOSS = -2; SCORE_TIMEOUT = 0; SCORE_PARTIAL = 0.5
MIN_TRADES_FOR_ADAPT = 5
MODES = [
    {"name": "Агрессивный",  "period": 5,  "oversold": 35, "overbought": 65},
    {"name": "Умеренный",    "period": 7,  "oversold": 30, "overbought": 70},
    {"name": "Консервативный","period": 9, "oversold": 25, "overbought": 75},
]

NN_HIDDEN=32; NN_LR=0.01; NN_EPOCHS=30; NN_BATCH=16
NN_RETRAIN_INTERVAL = 1800
NN_LOOKBACK = 500
NN_CONFIDENCE_MIN = 0.55            # снижена для большей чувствительности
NN_CONFIDENCE_CONTRA = 0.65         # порог для противотрендовых сигналов

SCAN_TOP_N = 30
SCAN_INTERVAL = 14400
SCAN_BARS = 60
MIN_VOLUME_USDT = 5_000_000
MIN_ATR_PCT = 0.5

SUPERTREND_PERIOD = 10
SUPERTREND_MULT = 3.0

INITIAL_RISK = 1.0
RISK_REDUCTION_STEP = 0.15
MIN_RISK = 0.1

TRAILING_ATR_MULT = 2.5
TRAILING_STEP_ATR_MULT = 1.0
MIN_PROFIT_FOR_TRAIL = 1.0        # % прибыли для активации трейлинга
PARTIAL_BE_ENABLED = True
PARTIAL_BE_PROFIT = 0.4           # % для частичного безубытка

TRADE_MAX_LIFETIME = 7200
REPORT_INTERVAL = 1800
STATE_FILE = "hybrid_state.json"
TRADES_FILE = "hybrid_trades.json"

# Дополнительные защиты
MAX_CONSECUTIVE_LOSSES = 4         # остановка торгов после 4 убытков подряд
COOLDOWN_MINUTES = 60              # пауза после серии убытков

# ============================================================
#                        ЛОГИРОВАНИЕ
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]  %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler("hybrid_bot_v2.log", encoding="utf-8")],
)
log = logging.getLogger("HybridBotV2")

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
            m_[:] = self.beta1*m_ + (1-self.beta1)*grad
            v_[:] = self.beta2*v_ + (1-self.beta2)*(grad**2)
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
#             ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ
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
    hi, lo, pc = df['h'], df['l'], df['c'].shift(1)
    tr = pd.concat([hi-lo, (hi-pc).abs(), (lo-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def calc_supertrend(df, period=10, mult=3.0):
    atr = calc_atr(df, period)
    hl2 = (df['h'] + df['l'])/2
    ub = hl2 + mult*atr
    lb = hl2 - mult*atr
    trend = pd.Series(1, index=df.index)
    st_line = pd.Series(0., index=df.index)
    for i in range(1, len(df)):
        c = df['c'].iloc[i]
        if st_line.iloc[i-1] == 0:
            st_line.iloc[i] = lb.iloc[i] if c > lb.iloc[i] else ub.iloc[i]
            trend.iloc[i] = 1 if c > lb.iloc[i] else -1
        elif trend.iloc[i-1] == 1:
            st_line.iloc[i] = max(lb.iloc[i], st_line.iloc[i-1])
            if c < st_line.iloc[i]:
                trend.iloc[i] = -1
                st_line.iloc[i] = ub.iloc[i]
            else:
                trend.iloc[i] = 1
        else:
            st_line.iloc[i] = min(ub.iloc[i], st_line.iloc[i-1])
            if c > st_line.iloc[i]:
                trend.iloc[i] = 1
                st_line.iloc[i] = lb.iloc[i]
            else:
                trend.iloc[i] = -1
    return trend == 1, st_line

def is_bullish_engulfing(df):
    """Поглощение на понижение (бычий разворот)"""
    if len(df) < 2: return False
    prev, curr = df.iloc[-2], df.iloc[-1]
    return (prev['c'] < prev['o'] and curr['c'] > curr['o'] and
            curr['c'] > prev['o'] and curr['o'] < prev['c'])

def is_bearish_engulfing(df):
    if len(df) < 2: return False
    prev, curr = df.iloc[-2], df.iloc[-1]
    return (prev['c'] > prev['o'] and curr['c'] < curr['o'] and
            curr['o'] > prev['c'] and curr['c'] < prev['o'])

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

        self.live_trading = True
        self.risk_fraction = INITIAL_RISK
        self.first_trade_done = False
        self.start_time = time.time()

        self.last_scan_time = 0
        self._last_status_time = 0
        self.active_trade = None

        # Защита от серии убытков
        self.consecutive_losses = 0
        self.cooldown_until = 0

        # Для асинхронного мониторинга храним ордера SL/TP
        self.tp_order_id = None
        self.sl_order_id = None

    # ---------- Сканер монет ----------
    def scan_best_symbol(self):
        syms = WHITELIST_SYMBOLS if WHITELIST_SYMBOLS else ["BTC/USDT:USDT"]
        log.info(f"Сканирование {len(syms)} монет на {SCAN_TIMEFRAME}")
        best_sym = "BTC/USDT:USDT"
        best_atr = 0
        for sym in syms:
            raw = safe_fetch_ohlcv(sym, "5m", limit=SCAN_BARS+5)
            if len(raw) < SCAN_BARS: continue
            df = pd.DataFrame(raw, columns=['ts','o','h','l','c','v'])
            atr = ((df['h'] - df['l'])/df['c']).mean()
            if atr > best_atr and atr >= MIN_ATR_PCT/100:
                best_atr = atr
                best_sym = sym
        log.info(f"Выбрана: {best_sym} (средняя ATR={best_atr*100:.2f}%)")
        return best_sym

    # ---------- Обучение нейросети ----------
    def train_nn(self, df):
        if len(df) < 200: return
        df = df.copy()
        close = df['c']
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
        y = (df['c'].shift(-1) > df['c']).astype(int).values[:-1]
        X, y = X[:len(y)], y
        if len(X) < 100: return
        self.scaler.fit(X)
        X_s = self.scaler.transform(X)
        if self.nn is None or self.nn.W1.shape[0] != X.shape[1]:
            self.nn = NeuralPredictor(input_size=X.shape[1], hidden=NN_HIDDEN, lr=NN_LR)
        self.nn.fit(X_s, y, epochs=NN_EPOCHS, batch=NN_BATCH)
        self.nn_trained = True
        self.last_nn_train = time.time()
        log.info(f"Нейросеть обучена на {len(X)} примерах")

    def nn_predict(self, df):
        if not self.nn_trained: return 0.5
        df = df.copy()
        close = df['c']
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

    # ---------- Сигнал с улучшенными фильтрами ----------
    def get_signal(self, df, mode):
        """Возвращает (signal, rsi_val) с учётом свечных паттернов, ATR и Supertrend"""
        if len(df) < mode['period']+2: return None, None
        close = df['c']
        rsi = calc_rsi(close, mode['period'])
        rsi_val = rsi.iloc[-1]
        atr = self.current_atr
        if atr < MIN_ATR_FOR_TRADE:
            return None, rsi_val

        # Свечной паттерн
        engulfing_bull = is_bullish_engulfing(df)
        engulfing_bear = is_bearish_engulfing(df)

        signal = None
        # Long: RSI выходит из перепроданности + бычье поглощение
        if (rsi.iloc[-2] <= mode['oversold'] and rsi.iloc[-1] > mode['oversold']) or \
           (rsi_val <= mode['oversold'] + 5 and engulfing_bull):
            signal = 'long'
        # Short: RSI выходит из перекупленности + медвежье поглощение
        elif (rsi.iloc[-2] >= mode['overbought'] and rsi.iloc[-1] < mode['overbought']) or \
             (rsi_val >= mode['overbought'] - 5 and engulfing_bear):
            signal = 'short'

        if signal and self.first_trade_done:
            # Фильтр Supertrend (только по тренду)
            if signal == 'long' and self.supertrend_dir == 'bearish':
                signal = None
            elif signal == 'short' and self.supertrend_dir == 'bullish':
                signal = None

        if signal:
            proba = self.nn_predict(df)
            # Для противотрендовых сигналов используем более высокий порог
            if (signal == 'long' and self.supertrend_dir == 'bearish') or \
               (signal == 'short' and self.supertrend_dir == 'bullish'):
                conf_threshold = NN_CONFIDENCE_CONTRA
                log.info(f"Сигнал {signal.upper()} против тренда (ST={self.supertrend_dir}) — требуется повышенная уверенность NN")
            else:
                conf_threshold = self.adaptive_confidence

            if signal == 'long' and proba < conf_threshold:
                log.info(f"Сигнал LONG отклонён NN (proba={proba:.2f} < порог={conf_threshold:.2f})")
                return None, rsi_val
            elif signal == 'short' and proba > (1 - conf_threshold):
                log.info(f"Сигнал SHORT отклонён NN (proba={proba:.2f} > порог={1-conf_threshold:.2f})")
                return None, rsi_val
            self.last_signal_time = time.time()
        return signal, rsi_val

    # ---------- Открытие позиции ----------
    def open_position(self, signal, price, risk_override=None):
        if self.cooldown_until > time.time():
            log.info("Пропуск входа – действует кулдаун после серии убытков")
            return False
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
            atr = self.current_atr
            sl_dist = STOP_LOSS_ATR_MULT * atr
            tp_dist = TAKE_PROFIT_ATR_MULT * atr
            if signal == 'long':
                side='buy'; tp=price*(1+tp_dist); sl=price*(1-sl_dist)
            else:
                side='sell'; tp=price*(1-tp_dist); sl=price*(1+sl_dist)
            tp = float(exchange.price_to_precision(self.symbol, tp))
            sl = float(exchange.price_to_precision(self.symbol, sl))
            try:
                order = exchange.create_order(
                    self.symbol, 'market', side, qty,
                    params={'takeProfit': tp, 'stopLoss': sl}
                )
                log.info(f"✅ {signal.upper()} {qty} @ {price:.4f} TP={tp:.4f} SL={sl:.4f}")
                self.active_trade = {
                    'side': signal, 'entry': price, 'tp': tp, 'sl': sl, 'qty': qty,
                    'time': time.time(), 'peak': price, 'phase': 1, 'partial_done': False,
                    'trailing_active': False, 'sl_order_id': order.get('stopLossOrderId'),
                    'tp_order_id': order.get('takeProfitOrderId')
                }
                return True
            except Exception as e:
                log.error(f"Ошибка открытия: {e}")
                return False
        else:
            log.info(f"📈 СИМУЛЯЦИЯ {signal.upper()} @ {price:.4f}")
            self.active_trade = {'side':signal, 'entry':price, 'tp':price, 'sl':price,
                                 'qty':0, 'time':time.time(), 'virtual':True, 'bars_held':0}
            return True

    # ---------- Обновление стоп-лосса (реальное) ----------
    def update_stop_loss(self, new_sl):
        if not self.live_trading or self.active_trade is None: return
        trade = self.active_trade
        try:
            # Отменяем старый SL
            if trade.get('sl_order_id'):
                exchange.cancel_order(trade['sl_order_id'], self.symbol)
            # Создаём новый SL ордер (условный, reduce-only)
            side_close = 'sell' if trade['side'] == 'long' else 'buy'
            sl_order = exchange.create_order(
                self.symbol, 'stop_market', side_close, trade['qty'],
                params={'stopPrice': new_sl, 'reduceOnly': True}
            )
            trade['sl_order_id'] = sl_order['id']
            trade['sl'] = new_sl
            log.info(f"Стоп-лосс обновлён → {new_sl:.4f}")
        except Exception as e:
            log.error(f"Не удалось обновить SL: {e}")

    # ---------- Частичное закрытие ----------
    def partial_close(self, fraction=0.5):
        if not self.live_trading or self.active_trade is None: return
        trade = self.active_trade
        close_qty = trade['qty'] * fraction
        close_side = 'sell' if trade['side'] == 'long' else 'buy'
        try:
            exchange.create_order(self.symbol, 'market', close_side, close_qty,
                                  params={'reduceOnly': True})
            trade['qty'] -= close_qty
            trade['partial_done'] = True
            log.info(f"Частичное закрытие {fraction*100:.0f}% позиции")
        except Exception as e:
            log.error(f"Ошибка частичного закрытия: {e}")

    # ---------- Асинхронный мониторинг (вызывается часто) ----------
    def monitor_position(self):
        if not self.active_trade: return
        trade = self.active_trade
        # Симуляция
        if trade.get('virtual'):
            raw = safe_fetch_ohlcv(self.symbol, FAST_TIMEFRAME, limit=10)
            if len(raw) < 2: return
            df = pd.DataFrame(raw, columns=['ts','o','h','l','c','v'])
            high, low, close = df['h'].iloc[-1], df['l'].iloc[-1], df['c'].iloc[-1]
            trade['bars_held'] += 1
            result = None
            if trade['side'] == 'long':
                if high >= trade['tp']: result = 'tp'
                elif low <= trade['sl']: result = 'sl'
            else:
                if low <= trade['tp']: result = 'tp'
                elif high >= trade['sl']: result = 'sl'
            if result is None and trade['bars_held'] >= 50: result = 'timeout'
            if result:
                self.add_trade_result(result, trade['entry'], close, trade['side'])
                self.active_trade = None
            return

        # Реальная позиция: проверяем, существует ли ещё
        pos = safe_fetch_positions([self.symbol])
        active = [p for p in pos if float(p.get('contracts',0))>0 and p.get('side')==trade['side']]
        if not active:
            cur = exchange.fetch_ticker(self.symbol)['last']
            # Определяем результат: если цена за TP или SL – считаем соответствующим
            if trade['side'] == 'long':
                result = 'tp' if cur >= trade['tp'] else ('sl' if cur <= trade['sl'] else 'timeout')
            else:
                result = 'tp' if cur <= trade['tp'] else ('sl' if cur >= trade['sl'] else 'timeout')
            self.add_trade_result(result, trade['entry'], cur, trade['side'])
            self.active_trade = None
            return

        # Позиция ещё открыта – проверяем условия трейлинга и частичного безубытка
        ticker = safe_fetch_ticker(self.symbol)
        if ticker is None: return
        cur = ticker['last']
        pnl_pct = (cur/trade['entry']-1)*100 if trade['side']=='long' else (trade['entry']/cur-1)*100

        # Частичный безубыток
        if PARTIAL_BE_ENABLED and not trade['partial_done'] and pnl_pct >= PARTIAL_BE_PROFIT:
            self.partial_close(0.5)
            # После частичного закрытия подтягиваем SL в безубыток
            new_sl = trade['entry']  # безубыток
            self.update_stop_loss(new_sl)
            trade['phase'] = 2

        # Активация трейлинга
        if not trade.get('trailing_active') and pnl_pct >= MIN_PROFIT_FOR_TRAIL:
            trade['trailing_active'] = True
            trade['peak'] = cur
            log.info("Трейлинг активирован")

        # Трейлинг (только если фаза >=2 и трейлинг активен)
        if trade.get('trailing_active') and trade['phase'] >= 2:
            atr = self.current_atr
            trail_offset = TRAILING_STEP_ATR_MULT * atr
            if trade['side'] == 'long':
                if cur > trade['peak']:
                    trade['peak'] = cur
                new_sl = trade['peak'] * (1 - trail_offset)
                if new_sl > trade['sl'] + trail_offset * 0.5:
                    self.update_stop_loss(new_sl)
            else:
                if cur < trade['peak']:
                    trade['peak'] = cur
                new_sl = trade['peak'] * (1 + trail_offset)
                if new_sl < trade['sl'] - trail_offset * 0.5:
                    self.update_stop_loss(new_sl)

        # Проверка максимального времени жизни
        if time.time() - trade['time'] > TRADE_MAX_LIFETIME:
            log.warning("Дедлайн — закрываем позицию")
            close_side = 'sell' if trade['side']=='long' else 'buy'
            try:
                exchange.create_order(self.symbol, 'market', close_side, trade['qty'],
                                      params={'reduceOnly': True})
            except: pass
            self.add_trade_result('timeout', trade['entry'], cur, trade['side'])
            self.active_trade = None
            return

        log.info(f"Мониторинг: цена={cur:.4f} P&L={pnl_pct:+.2f}% SL={trade['sl']:.4f}")

    # ---------- Учёт результатов ----------
    def add_trade_result(self, exit_type, entry, exit_price, side):
        profit = (exit_price-entry)/entry*100 if side=='long' else (entry-exit_price)/entry*100
        if exit_type=='tp':
            sc=SCORE_WIN; self.wins+=1
            self.consecutive_losses = 0
        elif exit_type=='sl':
            sc=SCORE_LOSS; self.losses+=1
            self.consecutive_losses += 1
            if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                self.cooldown_until = time.time() + COOLDOWN_MINUTES * 60
                log.warning(f"Серия из {MAX_CONSECUTIVE_LOSSES} убытков – пауза на {COOLDOWN_MINUTES} мин")
        else:
            sc = SCORE_PARTIAL if profit >= TAKE_PROFIT_ATR_MULT/2 else SCORE_TIMEOUT
            self.timeouts+=1
            self.consecutive_losses = 0
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
        log.info("🚀 Запуск Гибридного Бота v2.0")
        try:
            balance = exchange.fetch_balance()['USDT']['free']
            if balance < 5:
                log.warning(f"Баланс {balance:.2f}USDT – переход в симуляцию")
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

        # Первичное обучение NN и расчёт индикаторов
        raw = safe_fetch_ohlcv(self.symbol, self.timeframe, limit=NN_LOOKBACK)
        if len(raw) > 100:
            df = pd.DataFrame(raw, columns=['ts','o','h','l','c','v'])
            self.train_nn(df)
            self.current_atr = calc_atr(df, 14).iloc[-1]
            _, st_line = calc_supertrend(df, SUPERTREND_PERIOD, SUPERTREND_MULT)
            self.supertrend_dir = 'bullish' if (df['c'].iloc[-1] > st_line.iloc[-1]) else 'bearish'

        self.adaptive_confidence = NN_CONFIDENCE_MIN
        self.last_signal_time = time.time()
        FORCED_TIMEOUT = 9.5 * 60

        while True:
            now = time.time()
            # Кулдаун
            if self.cooldown_until > now:
                log.info(f"Ожидание окончания кулдауна до {datetime.fromtimestamp(self.cooldown_until).strftime('%H:%M:%S')}")
                time.sleep(30)
                continue

            # Принудительный вход для первой сделки (если не было)
            if not self.first_trade_done and (now - self.start_time) > FORCED_TIMEOUT:
                self.first_trade_done = True

            # Обновление данных
            raw = safe_fetch_ohlcv(self.symbol, self.timeframe, limit=200)
            if len(raw) < 50:
                time.sleep(10); continue
            df = pd.DataFrame(raw, columns=['ts','o','h','l','c','v'])
            self.current_atr = calc_atr(df, 14).iloc[-1]
            _, st_line = calc_supertrend(df, SUPERTREND_PERIOD, SUPERTREND_MULT)
            self.supertrend_dir = 'bullish' if (df['c'].iloc[-1] > st_line.iloc[-1]) else 'bearish'

            # Мониторинг активной позиции (не блокирует)
            if self.active_trade:
                self.monitor_position()
                time.sleep(5)
                continue

            # Адаптивный порог NN
            if self.first_trade_done:
                if now - self.last_signal_time > 900:
                    self.adaptive_confidence = max(0.3, self.adaptive_confidence - 0.05)
                    log.info(f"Порог NN снижен до {self.adaptive_confidence:.2f}")
                elif self.adaptive_confidence < NN_CONFIDENCE_MIN:
                    self.adaptive_confidence = min(NN_CONFIDENCE_MIN, self.adaptive_confidence + 0.02)

            mode = MODES[self.active_mode_idx]
            signal, rsi_val = self.get_signal(df, mode)
            price = df['c'].iloc[-1]

            if signal:
                log.info(f"Сигнал {signal.upper()}")
                self.open_position(signal, price, risk_override=1.0 if not self.first_trade_done else None)
            else:
                # Если первая сделка не совершена и время вышло – принудительный вход
                if not self.first_trade_done and (now - self.start_time) > FORCED_TIMEOUT:
                    force = 'long' if self.supertrend_dir != 'bearish' else 'short'
                    log.info(f"Принудительный вход {force.upper()}")
                    self.open_position(force, price, risk_override=1.0)

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
