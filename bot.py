#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
АВТОНОМНЫЙ БОТ: СКАНЕР МОНЕТ + RSI + НЕЙРОСЕТЬ + БАЛЛЫ
- Реальная торговля (LIVE_TRADING = True)
- Сканирует топ-30 монет по объёму, отбирает с ATR ≥ 0.5%
- Нейросеть фильтрует сигналы
- Динамические TP/SL под волатильность
- Система баллов и адаптация стиля
- Торгует на весь свободный баланс
"""

import os
import time
import logging
import numpy as np
import ccxt
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler

load_dotenv()

# ============================================================
#                 КОНФИГУРАЦИЯ
# ============================================================
FIXED_SYMBOL        = ""               # Оставьте пустым для авто-сканирования
TIMEFRAME           = "1m"             # Таймфрейм торговли
SCAN_TIMEFRAME      = "5m"             # ТФ для сканера (оценка волатильности)
LEVERAGE            = 1                # Плечо
STOP_LOSS_PERCENT   = 0.25             # Базовый SL (будет увеличен при низкой волатильности)
TAKE_PROFIT_PERCENT = 1.0              # Базовый TP (будет увеличен при низкой волатильности)
LIVE_TRADING        = True             # Реальная торговля (True) или симуляция (False)

# Система баллов
SCORE_WIN           = 2
SCORE_LOSS          = -2
SCORE_TIMEOUT       = 0
SCORE_PARTIAL_MOVE  = 0.5              # если цена прошла половину TP при тайм-ауте
MIN_TRADES_FOR_ADAPT = 5

# Режимы агрессивности RSI (период, пороги)
MODES = [
    {"name": "Агрессивный",  "period": 3,  "oversold": 40, "overbought": 60},
    {"name": "Умеренный",    "period": 5,  "oversold": 30, "overbought": 70},
    {"name": "Консервативный","period": 7, "oversold": 20, "overbought": 80},
]

# Нейросеть
NN_HIDDEN           = 32
NN_LEARNING_RATE    = 0.01
NN_EPOCHS           = 30
NN_BATCH_SIZE       = 16
NN_RETRAIN_INTERVAL = 1800            # переобучение каждые 30 мин
NN_LOOKBACK         = 500
NN_CONFIDENCE_MIN   = 0.6

# Сканер монет
SCAN_TOP_N          = 30
SCAN_INTERVAL       = 14400           # каждые 4 часа
SCAN_BARS           = 60              # свечей для оценки
MIN_VOLUME_USDT     = 5_000_000       # мин. суточный объём
MIN_ATR_PCT         = 0.5             # минимальная волатильность (high-low)/close в %
SCAN_RSI_OVERSOLD   = 20
SCAN_RSI_OVERBOUGHT = 80

# ============================================================
#                        ЛОГИРОВАНИЕ
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]  %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("neuro_rsi_live.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("NeuroRSILive")

# ============================================================
#                    НЕЙРОСЕТЕВОЙ ПРЕДИКТОР
# ============================================================
class NeuralPredictor:
    def __init__(self, input_size=10, hidden_size=32, output_size=1, lr=0.01):
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros(output_size)

        self.m_W1, self.v_W1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.m_b1, self.v_b1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.m_W2, self.v_W2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.m_b2, self.v_b2 = np.zeros_like(self.b2), np.zeros_like(self.b2)

        self.lr = lr
        self.beta1, self.beta2 = 0.9, 0.999
        self.eps = 1e-8
        self.t = 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output):
        m = X.shape[0]
        dZ2 = output - y
        dW2 = (self.a1.T @ dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True).flatten() / m
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (self.z1 > 0)
        dW1 = (X.T @ dZ1) / m
        db1 = np.sum(dZ1, axis=0) / m

        self.t += 1
        for param, m_param, v_param, grad in zip(
            [self.W1, self.b1, self.W2, self.b2],
            [self.m_W1, self.m_b1, self.m_W2, self.m_b2],
            [self.v_W1, self.v_b1, self.v_W2, self.v_b2],
            [dW1, db1, dW2, db2]
        ):
            m_param = self.beta1 * m_param + (1 - self.beta1) * grad
            v_param = self.beta2 * v_param + (1 - self.beta2) * (grad ** 2)
            m_hat = m_param / (1 - self.beta1 ** self.t)
            v_hat = v_param / (1 - self.beta2 ** self.t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def fit(self, X, y, epochs=30, batch_size=16):
        for epoch in range(epochs):
            idx = np.arange(len(X))
            np.random.shuffle(idx)
            for i in range(0, len(X), batch_size):
                batch_idx = idx[i:i+batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx].reshape(-1, 1)
                self.forward(X_batch)
                self.backward(X_batch, y_batch, self.a2)
            if epoch % 10 == 0:
                out = self.forward(X)
                loss = -np.mean(y.reshape(-1,1) * np.log(out + 1e-10) +
                                (1 - y.reshape(-1,1)) * np.log(1 - out + 1e-10))
                log.debug(f"NN Epoch {epoch}, loss {loss:.4f}")

    def predict_proba(self, X):
        return self.forward(X)

# ============================================================
#             ОСНОВНОЙ КЛАСС БОТА
# ============================================================
class LiveBot:
    def __init__(self):
        self.symbol = None
        self.timeframe = TIMEFRAME
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.timeouts = 0
        self.score = 0
        self.active_mode_idx = 0
        self.mode_stats = [{'trades': 0, 'score': 0, 'avg': 0} for _ in MODES]

        self.nn = None
        self.scaler = MinMaxScaler()
        self.nn_trained = False
        self.last_nn_train_time = 0
        self.current_atr = 0.0

        self.exchange = ccxt.bybit({
            "apiKey": os.getenv("BYBIT_API_KEY"),
            "secret": os.getenv("BYBIT_API_SECRET"),
            "enableRateLimit": True,
            "timeout": 10_000,
            "options": {"defaultType": "linear"},
        })

        self.last_scan_time = 0
        self._last_status_time = 0
        self._last_stat_time = 0
        self.active_trade = None      # для отслеживания открытой позиции

    # ---------- получение свечей ----------
    def fetch_ohlcv(self, symbol, timeframe, limit=150):
        try:
            raw = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
            return df.iloc[:-1].reset_index(drop=True)
        except Exception as e:
            log.error(f"Ошибка получения свечей {symbol}: {e}")
            return pd.DataFrame()

    @staticmethod
    def calculate_rsi(close, period):
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, float('nan'))
        return 100 - (100 / (1 + rs))

    # ---------- обновление ATR ----------
    def update_atr(self, df, window=20):
        if len(df) < window:
            return
        atr_pct = ((df['high'] - df['low']) / df['close']).tail(window).mean()
        self.current_atr = atr_pct

    # ---------- сканер монет ----------
    def score_symbol(self, symbol, period=14):
        try:
            df = self.fetch_ohlcv(symbol, SCAN_TIMEFRAME, limit=SCAN_BARS + period + 5)
            if df.empty or len(df) < SCAN_BARS:
                return None
            close = df["close"]
            high, low = df["high"], df["low"]
            atr_pct = ((high - low) / close).mean()
            rsi = self.calculate_rsi(close, period).dropna()
            crosses = 0
            for i in range(1, len(rsi) - 3):
                prev, curr = rsi.iloc[i-1], rsi.iloc[i]
                if prev <= SCAN_RSI_OVERSOLD < curr:
                    if not any(rsi.iloc[i+j] <= SCAN_RSI_OVERSOLD for j in range(1,4) if i+j < len(rsi)):
                        crosses += 1
                if prev >= SCAN_RSI_OVERBOUGHT > curr:
                    if not any(rsi.iloc[i+j] >= SCAN_RSI_OVERBOUGHT for j in range(1,4) if i+j < len(rsi)):
                        crosses += 1
            score = crosses * (1 + atr_pct / 10)
            return {"symbol": symbol, "score": round(score,2), "crosses": crosses, "atr_pct": round(atr_pct,3)}
        except Exception:
            return None

    def scan_best_symbol(self):
        fallback = "BTC/USDT:USDT"
        log.info(f"🔍 Сканирование топ-{SCAN_TOP_N} монет по объёму...")
        try:
            tickers = self.exchange.fetch_tickers()
        except Exception as e:
            log.error(f"Ошибка получения тикеров: {e}")
            return fallback

        candidates = []
        for sym, t in tickers.items():
            if not sym.endswith(":USDT"):
                continue
            vol = (t.get("quoteVolume") or 0)
            if vol >= MIN_VOLUME_USDT:
                candidates.append((sym, vol))

        candidates.sort(key=lambda x: x[1], reverse=True)
        top = [sym for sym, _ in candidates[:SCAN_TOP_N]]
        log.info(f"  Отобрано {len(top)} монет для анализа")

        results = []
        for i, sym in enumerate(top, 1):
            res = self.score_symbol(sym)
            if res and res['atr_pct'] >= MIN_ATR_PCT:
                results.append(res)
                log.info(f"  [{i:2d}/{len(top)}] {sym:<22} скор={res['score']:6.2f} пересечений={res['crosses']} ATR={res['atr_pct']:.2f}%")
            time.sleep(0.2)

        if not results:
            log.warning("Нет монет с достаточной волатильностью. Выбираем монету с максимальным ATR.")
            best_atr = -1
            best_sym = fallback
            for sym in top:
                df = self.fetch_ohlcv(sym, SCAN_TIMEFRAME, limit=SCAN_BARS + 5)
                if df.empty:
                    continue
                atr = ((df['high'] - df['low']) / df['close']).mean()
                if atr > best_atr:
                    best_atr = atr
                    best_sym = sym
            return best_sym

        results.sort(key=lambda x: x['score'], reverse=True)
        best = results[0]
        log.info("─" * 55)
        log.info(f"🏆 Лучшая монета: {best['symbol']}")
        log.info(f"   Скор={best['score']}  Пересечений={best['crosses']}  ATR={best['atr_pct']}%")
        log.info("─" * 55)
        for r in results[:5]:
            log.info(f"    {r['symbol']:<22} скор={r['score']:6.2f}  пересечений={r['crosses']}")
        return best['symbol']

    # ---------- индикаторы для нейросети ----------
    @staticmethod
    def add_features(df):
        df = df.copy()
        close = df['close']
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, float('nan'))
        df['rsi'] = 100 - (100 / (1 + rs))
        df['ema_short'] = close.ewm(span=9).mean()
        df['ema_long'] = close.ewm(span=21).mean()
        macd_line = close.ewm(span=12).mean() - close.ewm(span=26).mean()
        signal_line = macd_line.ewm(span=9).mean()
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_hist'] = macd_line - signal_line
        for lag in [1, 2, 3]:
            df[f'close_lag_{lag}'] = close.shift(lag)
        return df.dropna()

    # ---------- обучение нейросети ----------
    def train_nn(self, df):
        if len(df) < 200:
            return False
        feats = self.add_features(df)
        feature_cols = ['rsi', 'ema_short', 'ema_long', 'macd', 'macd_signal', 'macd_hist',
                        'close_lag_1', 'close_lag_2', 'close_lag_3']
        X = feats[feature_cols].values
        y = (df.loc[feats.index, 'close'].shift(-1) > df.loc[feats.index, 'close']).astype(int).values
        valid = ~np.isnan(y)
        X, y = X[valid], y[valid]
        if len(X) < 100:
            return False
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        input_size = X.shape[1]
        if self.nn is None or self.nn.W1.shape[0] != input_size:
            self.nn = NeuralPredictor(input_size=input_size, hidden_size=NN_HIDDEN, lr=NN_LEARNING_RATE)
        self.nn.fit(X_scaled, y, epochs=NN_EPOCHS, batch_size=NN_BATCH_SIZE)
        self.nn_trained = True
        self.last_nn_train_time = time.time()
        log.info(f"🧠 Нейросеть обучена на {len(X)} примерах")
        return True

    # ---------- предсказание нейросети ----------
    def nn_predict(self, df):
        if not self.nn_trained or self.nn is None:
            return 0.5
        feats = self.add_features(df)
        feature_cols = ['rsi', 'ema_short', 'ema_long', 'macd', 'macd_signal', 'macd_hist',
                        'close_lag_1', 'close_lag_2', 'close_lag_3']
        if len(feats) == 0:
            return 0.5
        last = feats[feature_cols].iloc[-1].values.reshape(1, -1)
        try:
            last_scaled = self.scaler.transform(last)
        except:
            return 0.5
        return self.nn.predict_proba(last_scaled)[0][0]

    # ---------- сигнал RSI с фильтром нейросети ----------
    def get_signal(self, df, mode):
        if len(df) < mode["period"] + 2:
            return None, None
        close = df["close"]
        rsi_series = self.calculate_rsi(close, mode["period"])
        rsi_val = rsi_series.iloc[-1]
        signal = None
        if rsi_series.iloc[-2] <= mode["oversold"] and rsi_series.iloc[-1] > mode["oversold"]:
            signal = "long"
        elif rsi_series.iloc[-2] >= mode["overbought"] and rsi_series.iloc[-1] < mode["overbought"]:
            signal = "short"

        proba = self.nn_predict(df) if signal else None
        if signal:
            if signal == "long" and proba is not None and proba < NN_CONFIDENCE_MIN:
                log.debug(f"Нейросеть против long (p={proba:.2f})")
                signal = None
            elif signal == "short" and proba is not None and proba > (1 - NN_CONFIDENCE_MIN):
                log.debug(f"Нейросеть против short (p={proba:.2f})")
                signal = None
        return signal, rsi_val

    # ---------- открытие реальной позиции ----------
    def open_real_position(self, signal, current_price):
        """Открывает рыночный ордер на весь доступный баланс с TP/SL."""
        try:
            free_balance = self.exchange.fetch_balance()["USDT"]["free"]
        except:
            log.error("Не удалось получить баланс")
            return False

        if free_balance <= 0:
            log.warning("Баланс нулевой")
            return False

        # Расчёт объёма
        raw_size = free_balance / current_price
        market = self.exchange.market(self.symbol)
        min_amt = float((market.get("limits") or {}).get("amount", {}).get("min") or 0)
        min_cost = float((market.get("limits") or {}).get("cost", {}).get("min") or 0)

        if min_amt > 0 and raw_size < min_amt:
            log.warning(f"Недостаточно для мин. лота: нужно {min_amt}, есть {raw_size:.6f}")
            return False
        if min_cost > 0 and raw_size * current_price < min_cost:
            log.warning(f"Сумма {raw_size * current_price:.2f} USDT < мин. стоимости {min_cost} USDT")
            return False

        amount = float(self.exchange.amount_to_precision(self.symbol, raw_size))
        if amount <= 0:
            log.warning("Объём после округления = 0")
            return False

        # Динамические TP/SL на основе ATR
        atr = self.current_atr if self.current_atr > 0 else 0.003
        tp_move = max(TAKE_PROFIT_PERCENT / 100, atr * 2)
        sl_move = max(STOP_LOSS_PERCENT / 100, atr * 0.5)

        if signal == "long":
            side = "buy"
            tp_price = current_price * (1 + tp_move)
            sl_price = current_price * (1 - sl_move)
        else:
            side = "sell"
            tp_price = current_price * (1 - tp_move)
            sl_price = current_price * (1 + sl_move)

        tp_price = float(self.exchange.price_to_precision(self.symbol, tp_price))
        sl_price = float(self.exchange.price_to_precision(self.symbol, sl_price))

        try:
            order = self.exchange.create_order(
                symbol=self.symbol,
                type="market",
                side=side,
                amount=amount,
                params={"takeProfit": tp_price, "stopLoss": sl_price}
            )
            log.info(f"✅ Открыта позиция {signal.upper()} {amount} @ ~{current_price:.4f} TP={tp_price:.4f} SL={sl_price:.4f}")
            self.active_trade = {"side": signal, "entry_price": current_price}
            return True
        except Exception as e:
            log.error(f"Ошибка открытия позиции: {e}")
            return False

    # ---------- проверка статуса позиции ----------
    def check_position(self):
        """Проверяет, есть ли открытая позиция по текущему символу."""
        try:
            positions = self.exchange.fetch_positions([self.symbol])
            for pos in positions:
                if float(pos.get("contracts", 0)) > 0 and pos.get("side") in ("long", "short"):
                    return pos
        except Exception as e:
            log.error(f"Ошибка получения позиции: {e}")
        return None

    # ---------- адаптация стиля по баллам ----------
    def adapt_mode(self):
        if self.total_trades < MIN_TRADES_FOR_ADAPT:
            return
        for i, stats in enumerate(self.mode_stats):
            if stats['trades'] > 0:
                stats['avg'] = stats['score'] / stats['trades']
        best_idx = max(range(len(self.mode_stats)), key=lambda i: self.mode_stats[i]['avg'])
        if best_idx != self.active_mode_idx:
            old = MODES[self.active_mode_idx]['name']
            new = MODES[best_idx]['name']
            log.info(f"🔄 Смена стиля: {old} → {new} (средний балл: {self.mode_stats[best_idx]['avg']:.2f})")
            self.active_mode_idx = best_idx

    # ---------- начисление баллов за закрытую позицию ----------
    def add_trade_result(self, exit_type, entry_price, exit_price, side):
        if side == "long":
            profit_pct = (exit_price - entry_price) / entry_price * 100
        else:
            profit_pct = (entry_price - exit_price) / entry_price * 100

        if exit_type == "tp":
            score_delta = SCORE_WIN
            self.wins += 1
        elif exit_type == "sl":
            score_delta = SCORE_LOSS
            self.losses += 1
        else:  # timeout / manual
            # проверяем частичное движение
            tp_move = max(TAKE_PROFIT_PERCENT / 100, self.current_atr * 2)
            half_tp = tp_move * 0.5
            if profit_pct >= half_tp * 100:
                score_delta = SCORE_PARTIAL_MOVE
            else:
                score_delta = SCORE_TIMEOUT
            self.timeouts += 1

        self.score += score_delta
        self.total_trades += 1
        idx = self.active_mode_idx
        self.mode_stats[idx]['trades'] += 1
        self.mode_stats[idx]['score'] += score_delta
        log.info(f"🏁 Сделка закрыта ({exit_type}) {side.upper()} вход {entry_price:.4f} выход {exit_price:.4f} | прибыль {profit_pct:.2f}% | баллы: {score_delta:+d} (всего: {self.score})")
        self.active_trade = None
        self.adapt_mode()

    # ---------- вывод статистики ----------
    def print_stats(self):
        win_rate = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0
        mode_name = MODES[self.active_mode_idx]['name']
        log.info(f"📊 СТАТИСТИКА | Режим: {mode_name} | Баллы: {self.score} | "
                 f"Сделок: {self.total_trades} (W:{self.wins} L:{self.losses} T:{self.timeouts}) | "
                 f"Винрейт: {win_rate:.1f}%")
        for i, m in enumerate(MODES):
            s = self.mode_stats[i]
            if s['trades'] > 0:
                log.info(f"   {m['name']}: сделок {s['trades']}, средний балл {s['avg']:.2f}")

    # ---------- главный цикл ----------
    def run(self):
        log.info("🚀 Бот запущен в РЕАЛЬНОМ режиме")
        # Определяем начальную монету
        if FIXED_SYMBOL:
            self.symbol = FIXED_SYMBOL
        else:
            self.symbol = self.scan_best_symbol()
            self.last_scan_time = time.time()

        log.info(f"Торговая пара: {self.symbol}")
        try:
            self.exchange.set_leverage(LEVERAGE, self.symbol)
        except:
            pass

        # Первичное обучение нейросети
        df_hist = self.fetch_ohlcv(self.symbol, self.timeframe, limit=NN_LOOKBACK)
        if not df_hist.empty:
            self.train_nn(df_hist)
            self.update_atr(df_hist)
        else:
            log.warning("Нет данных для обучения нейросети")

        self._last_status_time = time.time()
        self._last_stat_time = time.time()

        while True:
            try:
                now = time.time()

                # Сканирование новой монеты (если не фиксирована)
                if not FIXED_SYMBOL and now - self.last_scan_time > SCAN_INTERVAL:
                    new_sym = self.scan_best_symbol()
                    if new_sym != self.symbol:
                        # Закрываем текущую позицию, если есть
                        pos = self.check_position()
                        if pos:
                            # Закрываем принудительно? Лучше не закрывать, пусть доходит TP/SL
                            log.warning(f"Обнаружена открытая позиция на {self.symbol}, не закрываем, но смену монеты откладываем")
                        else:
                            log.info(f"🔄 Переключение на {new_sym}")
                            self.symbol = new_sym
                            try:
                                self.exchange.set_leverage(LEVERAGE, self.symbol)
                            except:
                                pass
                            df_hist = self.fetch_ohlcv(self.symbol, self.timeframe, limit=NN_LOOKBACK)
                            if not df_hist.empty:
                                self.train_nn(df_hist)
                                self.update_atr(df_hist)
                    self.last_scan_time = now

                # Периодическая статистика
                if now - self._last_stat_time > 120:
                    self.print_stats()
                    self._last_stat_time = now

                # Обновление ATR по последним свечам
                df = self.fetch_ohlcv(self.symbol, self.timeframe, limit=200)
                if not df.empty:
                    self.update_atr(df)

                # Проверяем открытую позицию
                pos = self.check_position()
                if pos is not None:
                    # Позиция есть — ждём закрытия
                    if now - self._last_status_time > 30:
                        entry = float(pos.get("entryPrice", 0))
                        side = pos.get("side", "")
                        pnl = float(pos.get("unrealizedPnl", 0))
                        log.info(f"⚡ Открыта позиция {side.upper()} вх. {entry:.4f} PnL: {pnl:.2f} USDT")
                        self._last_status_time = now
                    time.sleep(5)
                    continue

                # Если позиции нет, ищем сигнал
                if self.active_trade:
                    # Только что закрылась позиция — нужно определить результат по истории ордеров
                    # Пока упростим: считаем, что если позиции нет, а active_trade не None, то она закрылась.
                    # Определяем exit_price по последней цене закрытия (не идеально, но для баллов сойдёт)
                    exit_price = df['close'].iloc[-1]
                    # Не знаем точно, TP или SL. Предположим, что смотрим на PnL: если профит положительный > 0.5% -> tp, иначе sl
                    entry_price = self.active_trade['entry_price']
                    side = self.active_trade['side']
                    profit_pct = (exit_price - entry_price) / entry_price * 100 if side == "long" else (entry_price - exit_price) / entry_price * 100
                    if profit_pct > 0.5 * TAKE_PROFIT_PERCENT:
                        exit_type = "tp"
                    elif profit_pct < -0.5 * STOP_LOSS_PERCENT:
                        exit_type = "sl"
                    else:
                        exit_type = "timeout"
                    self.add_trade_result(exit_type, entry_price, exit_price, side)
                    self.active_trade = None

                # Мониторинг и поиск сигнала
                if df.empty or len(df) < 20:
                    time.sleep(5)
                    continue

                mode = MODES[self.active_mode_idx]
                signal, rsi_val = self.get_signal(df, mode)
                current_price = df['close'].iloc[-1]

                if now - self._last_status_time > 30:
                    proba = self.nn_predict(df)
                    log.info(f"👁 Мониторинг [{self.symbol}] Цена: {current_price:.4f} | RSI({mode['period']}): {rsi_val:.1f} | "
                             f"NN proba: {proba:.2f} | Режим: {mode['name']} | Баллы: {self.score} | ATR: {self.current_atr*100:.2f}%")
                    self._last_status_time = now

                if signal:
                    log.info(f"📈 Сигнал {signal.upper()} на {self.symbol}")
                    if LIVE_TRADING:
                        success = self.open_real_position(signal, current_price)
                        if not success:
                            log.warning("Не удалось открыть позицию")
                    else:
                        # Симуляция (не должна вызываться при LIVE_TRADING=True, но на всякий случай)
                        log.info("Симуляция: сигнал получен")

                # Переобучение нейросети
                if now - self.last_nn_train_time > NN_RETRAIN_INTERVAL:
                    df_train = self.fetch_ohlcv(self.symbol, self.timeframe, limit=NN_LOOKBACK)
                    if not df_train.empty:
                        self.train_nn(df_train)
                    self.last_nn_train_time = now

                time.sleep(5)

            except KeyboardInterrupt:
                log.info("Остановка бота")
                self.print_stats()
                break
            except Exception as e:
                log.error(f"Ошибка: {e}", exc_info=True)
                time.sleep(5)

if __name__ == "__main__":
    bot = LiveBot()
    bot.run()
