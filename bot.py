#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ГИБРИДНЫЙ БОТ: SuperTrend + RSI + НЕЙРОСЕТЬ + БАЛЛЫ (ИСПРАВЛЕННЫЙ v2)
- Сканирует белый список монет по волатильности
- Сигналы RSI фильтруются SuperTrend и нейросетью
- Первая сделка (10 мин) – максимальный риск, потом риск снижается
- Мониторинг позиции с трейлинг-стопом и частичным безубытком
- Авто-симуляция при недостатке средств

ИСПРАВЛЕНИЯ В ЭТОЙ ВЕРСИИ (см. CHANGELOG.md):
  1. self.last_signal_time теперь обновляется при каждом реальном сигнале
     и при открытии позиции -> адаптивный порог NN больше не "застывает"
     на 0.30 и не спамит лог каждые 5 секунд.
  2. В симуляции (виртуальные сделки) TP/SL теперь считаются от ATR/процентов,
     как и в реальной торговле, а не равны цене входа -> сделки больше не
     закрываются мгновенно с фиктивным "TP" при нулевом PnL.
  3. Добавлено логирование смены порога только при фактическом ИЗМЕНЕНИИ
     значения (а не на каждой итерации цикла).
  4. Мелкие правки устойчивости (защита от деления на ноль / пустых данных).
"""

import os, time, logging, json, numpy as np, ccxt, pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import MinMaxScaler

load_dotenv()

# ============================================================
#                 КОНФИГУРАЦИЯ
# ============================================================
FIXED_SYMBOL        = ""               # если пусто – авто‑сканер
# Из белого списка убраны BTC и ETH: при текущем балансе их минимальный
# объём ордера на бирже (например 0.001 BTC ≈ $60) превышает доступные
# средства, и сделки по ним просто не открываются ("Открытие отклонено").
# Оставлены более дешёвые монеты, где минимальный лот стоит заметно меньше.
WHITELIST_SYMBOLS   = [
    "SOL/USDT:USDT",
    "XRP/USDT:USDT", "DOGE/USDT:USDT", "ADA/USDT:USDT",
    "LINK/USDT:USDT", "MATIC/USDT:USDT", "DOT/USDT:USDT",
]
TIMEFRAME           = "1m"
SCAN_TIMEFRAME      = "5m"
LEVERAGE            = 1
STOP_LOSS_PERCENT   = 0.25
TAKE_PROFIT_PERCENT = 1.0

SCORE_WIN = 2; SCORE_LOSS = -2; SCORE_TIMEOUT = 0; SCORE_PARTIAL = 0.5
MIN_TRADES_FOR_ADAPT = 5
MODES = [
    {"name": "Агрессивный",  "period": 3,  "oversold": 40, "overbought": 60},
    {"name": "Умеренный",    "period": 5,  "oversold": 30, "overbought": 70},
    {"name": "Консервативный","period": 7, "oversold": 20, "overbought": 80},
]

NN_HIDDEN=32; NN_LR=0.01; NN_EPOCHS=30; NN_BATCH=16
NN_RETRAIN_INTERVAL = 1800
NN_LOOKBACK = 500
NN_CONFIDENCE_MIN = 0.45

SCAN_TOP_N = 30
SCAN_INTERVAL = 14400
SCAN_BARS = 60
MIN_VOLUME_USDT = 5_000_000
MIN_ATR_PCT = 0.5

SUPERTREND_PERIOD = 10
SUPERTREND_MULT = 3.0

INITIAL_RISK = 1.0
RISK_REDUCTION_STEP = 0.2
MIN_RISK = 0.1

TRAILING_ATR_MULT = 2.0
TRAILING_OFFSET_MULT = 1.5
MIN_TRAILING_STEP = 0.4
MIN_TRAILING_OFFSET = 0.6
MIN_PROFIT_FOR_TRAIL = 1.0
PARTIAL_BE_ENABLED = True
PARTIAL_BE_PROFIT = 0.2

# Сколько секунд без нового сигнала считаем "застоем" перед смягчением порога NN
SIGNAL_STALE_SECONDS = 900

TRADE_MAX_LIFETIME = 7200
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
#             ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (с короткими именами)
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

        self.live_trading = True
        self.risk_fraction = INITIAL_RISK
        self.first_trade_done = False
        self.start_time = time.time()

        self.last_scan_time = 0
        self._last_status_time = 0
        self._last_stat_time = 0
        self.active_trade = None

        # FIX #1: инициализируем здесь же, чтобы не зависеть от порядка вызовов в run()
        self.adaptive_confidence = NN_CONFIDENCE_MIN
        self.last_signal_time = time.time()

    # ---------- Сканер монет ----------
    def scan_best_symbol(self):
        syms = WHITELIST_SYMBOLS if WHITELIST_SYMBOLS else ["BTC/USDT:USDT"]
        log.info(f"Сканирование белого списка ({len(syms)} монет)")

        # Проверяем доступный баланс, чтобы не выбрать монету, минимальный
        # лот которой всё равно не пройдёт при открытии позиции (как было
        # с BTC при балансе 19 USDT — сигнал срабатывал, но ордер
        # отклонялся биржей из-за min_amt/min_cost).
        try:
            free_balance = exchange.fetch_balance()['USDT']['free']
        except Exception as e:
            log.warning(f"Не удалось получить баланс для проверки сканера: {e}")
            free_balance = None

        affordable = []   # [(sym, atr), ...] - монеты, которые прошли проверку баланса
        skipped = []
        for sym in syms:
            raw = safe_fetch_ohlcv(sym, SCAN_TIMEFRAME, limit=SCAN_BARS+5)
            if len(raw) < SCAN_BARS: continue
            df = pd.DataFrame(raw, columns=['ts','o','h','l','c','v'])
            last_price = df['c'].iloc[-1]

            if free_balance is not None:
                try:
                    market = exchange.market(sym)
                    min_amt = float(market.get('limits', {}).get('amount', {}).get('min', 0) or 0)
                    min_cost = float(market.get('limits', {}).get('cost', {}).get('min', 0) or 0)
                    required = max(min_amt * last_price, min_cost)
                    if required > free_balance:
                        skipped.append(f"{sym} (нужно ~{required:.2f}, есть {free_balance:.2f})")
                        continue
                except Exception:
                    pass  # если не удалось получить лимиты рынка - не блокируем монету из-за этого

            atr = ((df['h'] - df['l'])/df['c']).mean()
            affordable.append((sym, atr))

        if skipped:
            log.info(f"Пропущены монеты (не хватает баланса на мин. лот): {', '.join(skipped)}")

        if not affordable:
            # ни одна монета не прошла даже проверку баланса - дальше
            # сканировать нечего, явно сообщаем об этом и просим расширить
            # список монет или пополнить счёт
            log.error("Ни одна монета из белого списка не проходит по балансу — "
                      "проверьте WHITELIST_SYMBOLS или пополните счёт")
            return syms[0]  # последний fallback, чтобы бот не упал с исключением

        # среди доступных по балансу выбираем монету с максимальным ATR,
        # удовлетворяющим MIN_ATR_PCT; если ни одна не дотягивает до
        # порога волатильности - берём просто самую волатильную из доступных
        above_threshold = [(s, a) for s, a in affordable if a >= MIN_ATR_PCT]
        if above_threshold:
            best_sym, best_atr = max(above_threshold, key=lambda x: x[1])
        else:
            best_sym, best_atr = max(affordable, key=lambda x: x[1])
            log.warning(f"Ни одна доступная монета не достигла MIN_ATR_PCT={MIN_ATR_PCT}%, "
                        f"берём самую волатильную из доступных: {best_sym}")

        log.info(f"Лучшая монета по ATR: {best_sym} ({best_atr*100:.2f}%)")
        return best_sym

    # ---------- Обучение нейросети ----------
    def train_nn(self, df):
        if len(df) < 200: return
        df = df.copy()  # не мутируем переданный df на месте
        close = df['c'].copy()
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
        df = df.copy()  # не мутируем переданный df на месте
        close = df['c'].copy()
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
        if np.isnan(last).any(): return 0.5
        try: last_s = self.scaler.transform(last)
        except: return 0.5
        return self.nn.predict_proba(last_s)[0,0]

    # ---------- Сигнал ----------
    def get_signal(self, df, mode, forced=False):
        if len(df) < mode['period']+2: return None, None
        close = df['c']
        rsi = calc_rsi(close, mode['period'])
        rsi_val = rsi.iloc[-1]
        signal = None
        if rsi.iloc[-2] <= mode['oversold'] and rsi.iloc[-1] > mode['oversold']:
            signal = 'long'
        elif rsi.iloc[-2] >= mode['overbought'] and rsi.iloc[-1] < mode['overbought']:
            signal = 'short'
        if signal and not forced:
            # Фильтр SuperTrend ослаблен: раньше контр-трендовый сигнал
            # полностью блокировался. Теперь он не отбрасывается, а просто
            # требует более высокой уверенности NN (более строгий локальный
            # порог), чтобы пройти. Это увеличивает число сигналов, но не
            # убирает тренд как фактор риска совсем.
            against_trend = False
            if self.first_trade_done:
                if signal == 'long' and self.supertrend_dir == 'bearish':
                    against_trend = True
                elif signal == 'short' and self.supertrend_dir == 'bullish':
                    against_trend = True
                if against_trend:
                    log.info(f"Сигнал {signal.upper()} против тренда "
                             f"(ST={self.supertrend_dir}) — требуется повышенная уверенность NN")
            if signal:
                proba = self.nn_predict(df)
                # Контр-трендовый сигнал требует доп. запас уверенности (+0.1)
                local_threshold = self.adaptive_confidence + (0.1 if against_trend else 0.0)
                if signal == 'long' and proba < local_threshold:
                    log.info(f"Сигнал LONG отклонён NN (proba={proba:.2f} < порог={local_threshold:.2f})")
                    signal = None
                elif signal == 'short' and proba > 1-local_threshold:
                    log.info(f"Сигнал SHORT отклонён NN (proba={proba:.2f} > порог={1-local_threshold:.2f})")
                    signal = None
        return signal, rsi_val

    # ---------- Расчёт уровней TP/SL (общий для live и для симуляции) ----------
    def calc_tp_sl(self, signal, price):
        """FIX #2: вынесено в отдельный метод, чтобы симуляция считала
        TP/SL так же, как реальная торговля, а не приравнивала их к цене входа."""
        atr = self.current_atr if self.current_atr > 0 else 0.003
        tp_move = max(TAKE_PROFIT_PERCENT/100, atr*2)
        sl_move = max(STOP_LOSS_PERCENT/100, atr*0.5)
        if signal == 'long':
            tp = price*(1+tp_move); sl = price*(1-sl_move)
        else:
            tp = price*(1-tp_move); sl = price*(1+sl_move)
        return tp, sl

    # ---------- Открытие позиции ----------
    def open_position(self, signal, price, risk_override=None):
        fraction = risk_override if risk_override is not None else self.risk_fraction

        # FIX #1: фиксируем момент сигнала здесь
        self.last_signal_time = time.time()

        free = exchange.fetch_balance()['USDT']['free']
        avail = free * fraction
        if avail <= 0:
            log.warning(f"Открытие отклонено: avail={avail:.4f} (free={free:.4f}, fraction={fraction:.2f})")
            return False
        raw_qty = avail / price
        market = exchange.market(self.symbol)
        min_amt = float(market.get('limits', {}).get('amount', {}).get('min', 0) or 0)
        min_cost = float(market.get('limits', {}).get('cost', {}).get('min', 0) or 0)
        if min_amt and raw_qty < min_amt:
            if free >= min_amt * price:
                raw_qty = min_amt
            else:
                log.warning(f"Открытие отклонено: нужно min_amt={min_amt}, "
                            f"но free={free:.4f} недостаточно для {min_amt}@{price:.4f}")
                return False
        if min_cost and raw_qty * price < min_cost:
            if free >= min_cost:
                raw_qty = min_cost / price
            else:
                log.warning(f"Открытие отклонено: нужно min_cost={min_cost}, "
                            f"но free={free:.4f} недостаточно")
                return False
        qty = float(exchange.amount_to_precision(self.symbol, raw_qty))
        if qty <= 0:
            log.warning(f"Открытие отклонено: рассчитанный qty={qty} <= 0 "
                        f"(raw_qty={raw_qty}, avail={avail:.4f})")
            return False
        tp, sl = self.calc_tp_sl(signal, price)
        side = 'buy' if signal == 'long' else 'sell'
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

    # ---------- Мониторинг позиции ----------
    def monitor_position(self):
        if not self.active_trade: return
        trade = self.active_trade
        # Реальная позиция (симуляция убрана — счёт пополнен, торгуем live)
        deadline = trade['time'] + TRADE_MAX_LIFETIME
        while True:
            if time.time() >= deadline:
                log.warning("Дедлайн — закрываем")
                self.close_position(trade['qty'], trade['side'])
                cur = exchange.fetch_ticker(self.symbol)['last']
                self.add_trade_result('timeout', trade['entry'], cur, trade['side'])
                self.active_trade = None
                return
            time.sleep(15)
            pos = safe_fetch_positions([self.symbol])
            active = [p for p in pos if float(p.get('contracts',0))>0 and p.get('side')==trade['side']]
            if not active:
                cur = exchange.fetch_ticker(self.symbol)['last']
                # определяем результат
                if trade['side']=='long': res = 'tp' if cur>trade['entry'] else 'sl'
                else: res = 'tp' if cur<trade['entry'] else 'sl'
                self.add_trade_result(res, trade['entry'], cur, trade['side'])
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
                    new_sl = trade['entry'] * (1 + 0.0005) if trade['side']=='long' else trade['entry'] * (1 - 0.0005)
                    try: exchange.create_order(self.symbol, 'market', 'sell' if trade['side']=='long' else 'buy',
                                               0, params={'stopLoss':exchange.price_to_precision(self.symbol, new_sl)})
                    except: pass
                    trade['sl'] = new_sl
                    trade['phase'] = 2
                    log.info(f"Частичный безубыток, SL→{new_sl:.4f}")

            # Активация трейлинга
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
                        log.info(f"Трейлинг SL→{new_sl:.4f}")
                else:
                    if cur < trade.get('peak', cur): trade['peak'] = cur
                    new_sl = trade['peak'] * (1 + MIN_TRAILING_OFFSET/100)
                    if new_sl < trade['sl']:
                        try: exchange.create_order(self.symbol, 'market', 'buy', 0,
                                                   params={'stopLoss':exchange.price_to_precision(self.symbol, new_sl)})
                        except: pass
                        trade['sl'] = new_sl
                        log.info(f"Трейлинг SL→{new_sl:.4f}")
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
        # Проверка баланса/автопереход в симуляцию убраны — счёт пополнен,
        # бот всегда торгует live (self.live_trading = True задаётся в __init__).
        try:
            balance = exchange.fetch_balance()['USDT']['free']
            log.info(f"Баланс {balance:.2f} USDT")
        except Exception as e:
            log.warning(f"Не удалось получить баланс: {e}")

        if not FIXED_SYMBOL:
            self.symbol = self.scan_best_symbol()
        else:
            self.symbol = FIXED_SYMBOL

        log.info(f"Пара: {self.symbol} | Режим: LIVE | Риск: {self.risk_fraction*100:.0f}%")
        try: exchange.set_leverage(LEVERAGE, self.symbol)
        except: pass

        # первичное обучение
        raw = safe_fetch_ohlcv(self.symbol, self.timeframe, limit=NN_LOOKBACK)
        if len(raw) > 100:
            df = pd.DataFrame(raw, columns=['ts','o','h','l','c','v'])
            self.train_nn(df)
            self.current_atr = ((df['h']-df['l'])/df['c']).tail(20).mean()
            _, st_line = calc_supertrend(df, SUPERTREND_PERIOD, SUPERTREND_MULT)
            self.supertrend_dir = 'bullish' if (df['c'].iloc[-1] > st_line.iloc[-1]) else 'bearish'

        # last_signal_time/adaptive_confidence уже инициализированы в __init__ (FIX #1),
        # но обновим время старта здесь же, чтобы отсчёт "застоя" начинался от запуска цикла.
        self.last_signal_time = time.time()
        FORCED_TIMEOUT = 9.5*60

        while True:
            now = time.time()
            if not self.first_trade_done and (now - self.start_time) > 600:
                log.info("10 мин истекли, первая сделка не выполнена — включаем фильтры")
                self.first_trade_done = True

            raw = safe_fetch_ohlcv(self.symbol, self.timeframe, limit=200)
            if len(raw) < 50:
                time.sleep(10); continue
            df = pd.DataFrame(raw, columns=['ts','o','h','l','c','v'])
            self.current_atr = ((df['h']-df['l'])/df['c']).tail(20).mean()
            _, st_line = calc_supertrend(df, SUPERTREND_PERIOD, SUPERTREND_MULT)
            self.supertrend_dir = 'bullish' if (df['c'].iloc[-1] > st_line.iloc[-1]) else 'bearish'

            if self.active_trade:
                self.monitor_position()
                continue

            # FIX #1: адаптивный порог — логируем только когда значение
            # действительно меняется, иначе лог не спамится при "застывшем" пороге.
            if self.first_trade_done:
                prev_conf = self.adaptive_confidence
                if now - self.last_signal_time > SIGNAL_STALE_SECONDS:
                    self.adaptive_confidence = max(0.3, self.adaptive_confidence - 0.1)
                elif self.adaptive_confidence < NN_CONFIDENCE_MIN:
                    self.adaptive_confidence = min(NN_CONFIDENCE_MIN, self.adaptive_confidence + 0.05)
                if abs(self.adaptive_confidence - prev_conf) > 1e-9:
                    log.info(f"Порог NN изменён: {prev_conf:.2f} → {self.adaptive_confidence:.2f}")

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
                    # FIX #1: last_signal_time также обновляется внутри open_position(),
                    # эта строка избыточна, но оставлена явной для читаемости намерения.
                    self.last_signal_time = now

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
