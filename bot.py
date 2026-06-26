#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
АВТОНОМНЫЙ ТОРГОВЫЙ БОТ: RSI + НЕЙРОСЕТЬ + БАЛЛЫ
- Частая торговля (1m/5m)
- Адаптивный выбор агрессивности по баллам
- Самообучение нейросети каждые 30 минут
- Реальная торговля на Bybit (или симуляция)
"""

import os
import time
import logging
import threading
import datetime
import numpy as np
import ccxt
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler

load_dotenv()

# ============================================================
#                 КОНФИГУРАЦИЯ
# ============================================================
SYMBOL              = "BTC/USDT:USDT"   # торгуемая пара
TIMEFRAME           = "1m"              # 1 минута для частой торговли
INITIAL_CAPITAL     = 100               # стартовый депозит USDT
LEVERAGE            = 1                 # без плеча
STOP_LOSS_PERCENT   = 0.25
TAKE_PROFIT_PERCENT = 1.0
MONITOR_INTERVAL    = 10                # секунд между проверками (для симуляции)
LIVE_TRADING        = False             # False = симуляция, True = реальные ордера
SCORE_WIN           = 2                 # баллы за тейк-профит
SCORE_LOSS          = -2                # баллы за стоп-лосс
SCORE_TIMEOUT       = 0                 # баллы за тайм-аут
MIN_TRADES_FOR_ADAPT = 5               # после скольки сделок начинаем адаптацию

# Режимы агрессивности RSI (период, пороги)
MODES = [
    {"name": "Агрессивный",  "period": 3,  "oversold": 40, "overbought": 60},
    {"name": "Умеренный",    "period": 5,  "oversold": 30, "overbought": 70},
    {"name": "Консервативный","period": 7, "oversold": 20, "overbought": 80},
]

# Параметры нейросети
NN_HIDDEN           = 32
NN_LEARNING_RATE    = 0.01
NN_EPOCHS           = 30
NN_BATCH_SIZE       = 16
NN_RETRAIN_INTERVAL = 1800              # переобучение каждые 30 минут
NN_LOOKBACK         = 500               # свечей для обучения
NN_CONFIDENCE_MIN   = 0.6               # порог уверенности для фильтра

# ============================================================
#                        ЛОГИРОВАНИЕ
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]  %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("neuro_rsi_bot.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("NeuroRSIBot")

# ============================================================
#                    НЕЙРОСЕТЕВОЙ ПРЕДИКТОР
# ============================================================
class NeuralPredictor:
    def __init__(self, input_size=10, hidden_size=32, output_size=1, lr=0.01):
        # Xavier init
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros(output_size)

        # Adam state
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
        self.a1 = np.maximum(0, self.z1)  # ReLU
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
class AdaptiveScoreBot:
    def __init__(self):
        self.symbol = SYMBOL
        self.timeframe = TIMEFRAME
        self.balance = INITIAL_CAPITAL
        self.position = 0          # 1 = long, -1 = short, 0 = none
        self.entry_price = 0.0
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.score = 0             # накопленные баллы
        self.active_mode_idx = 0   # индекс текущего режима
        self.mode_stats = [{'trades': 0, 'score': 0, 'avg': 0} for _ in MODES]

        # Нейросеть
        self.nn = None
        self.scaler = MinMaxScaler()
        self.nn_trained = False
        self.last_nn_train_time = 0

        # Подключение к бирже (только для получения данных и реальных ордеров)
        self.exchange = ccxt.bybit({
            "apiKey": os.getenv("BYBIT_API_KEY"),
            "secret": os.getenv("BYBIT_API_SECRET"),
            "enableRateLimit": True,
            "timeout": 10_000,
            "options": {"defaultType": "linear"},
        })

        if LIVE_TRADING:
            try:
                self.exchange.set_leverage(LEVERAGE, self.symbol)
            except Exception as e:
                log.warning(f"Не удалось установить плечо: {e}")

    # ---------- получение свечей ----------
    def fetch_ohlcv(self, limit=150):
        try:
            raw = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
            df = df.iloc[:-1].reset_index(drop=True)  # убираем незакрытую
            return df
        except Exception as e:
            log.error(f"Ошибка получения свечей: {e}")
            return pd.DataFrame()

    # ---------- индикаторы для нейросети ----------
    @staticmethod
    def add_features(df):
        df = df.copy()
        close = df['close']
        # RSI 14
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, float('nan'))
        df['rsi'] = 100 - (100 / (1 + rs))

        # EMA
        df['ema_short'] = close.ewm(span=9).mean()
        df['ema_long'] = close.ewm(span=21).mean()

        # MACD
        macd_line = close.ewm(span=12).mean() - close.ewm(span=26).mean()
        signal_line = macd_line.ewm(span=9).mean()
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_hist'] = macd_line - signal_line

        # Лаги цены
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
        # удаляем последний NaN
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
        log.info(f"Нейросеть обучена на {len(X)} примерах")
        return True

    # ---------- предсказание нейросети ----------
    def nn_predict(self, df):
        if not self.nn_trained or self.nn is None:
            return 0.5  # нейтрально
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
        proba = self.nn.predict_proba(last_scaled)[0][0]
        return proba

    # ---------- генерация сигнала по RSI + фильтр нейросети ----------
    def get_signal(self, df, mode):
        if len(df) < mode["period"] + 2:
            return None
        close = df["close"]
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/mode["period"], min_periods=mode["period"], adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/mode["period"], min_periods=mode["period"], adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, float('nan'))
        rsi = 100 - (100 / (1 + rs))

        rsi_prev = rsi.iloc[-2]
        rsi_curr = rsi.iloc[-1]
        signal = None
        if rsi_prev <= mode["oversold"] and rsi_curr > mode["oversold"]:
            signal = "long"
        elif rsi_prev >= mode["overbought"] and rsi_curr < mode["overbought"]:
            signal = "short"

        if signal is None:
            return None

        # Нейросетевой фильтр: проверяем, что предсказание совпадает
        proba = self.nn_predict(df)
        if signal == "long" and proba < NN_CONFIDENCE_MIN:
            log.debug(f"Нейросеть против long (p={proba:.2f})")
            return None
        if signal == "short" and proba > (1 - NN_CONFIDENCE_MIN):
            log.debug(f"Нейросеть против short (p={proba:.2f})")
            return None

        return signal

    # ---------- исполнение сделки (реальной или виртуальной) ----------
    def execute_trade(self, signal, price, mode_name):
        """Возвращает результат: 'tp', 'sl', 'timeout' (для симуляции)"""
        if LIVE_TRADING:
            # Здесь логика открытия реального ордера, как в оригинальном боте
            # (пока заглушка)
            log.info(f"[LIVE] Сигнал {signal} @ {price:.4f}")
            return "tp"  # заглушка
        else:
            # Виртуальная сделка: открываем и ждём закрытия по TP/SL на следующих свечах
            # (будет обработано в цикле симуляции)
            self.virtual_trade = {
                "signal": signal,
                "entry_price": price,
                "mode": mode_name,
                "tp": price * (1 + TAKE_PROFIT_PERCENT/100) if signal == "long" else price * (1 - TAKE_PROFIT_PERCENT/100),
                "sl": price * (1 - STOP_LOSS_PERCENT/100) if signal == "long" else price * (1 + STOP_LOSS_PERCENT/100),
                "bars_held": 0
            }
            return None

    # ---------- адаптация стиля по баллам ----------
    def adapt_mode(self):
        if self.total_trades < MIN_TRADES_FOR_ADAPT:
            return
        # обновляем средний балл для каждого режима
        for i, stats in enumerate(self.mode_stats):
            if stats['trades'] > 0:
                stats['avg'] = stats['score'] / stats['trades']
        # выбираем режим с наивысшим средним баллом
        best_idx = max(range(len(self.mode_stats)), key=lambda i: self.mode_stats[i]['avg'])
        if best_idx != self.active_mode_idx:
            old = MODES[self.active_mode_idx]['name']
            new = MODES[best_idx]['name']
            log.info(f"🔄 Смена стиля: {old} → {new} (баллы: {self.mode_stats[best_idx]['avg']:.2f})")
            self.active_mode_idx = best_idx

    # ---------- главный цикл симуляции ----------
    def run_simulation(self):
        log.info("Запуск симуляции с самообучением и баллами")
        self.last_nn_train_time = time.time() - NN_RETRAIN_INTERVAL  # сразу обучим

        # загружаем историю для первого обучения
        df_hist = self.fetch_ohlcv(limit=1000)
        if not df_hist.empty:
            self.train_nn(df_hist)
        else:
            log.warning("Не удалось загрузить историю для обучения")

        # инициализация виртуальной сделки
        self.virtual_trade = None

        while True:
            try:
                # получаем свежие свечи
                df = self.fetch_ohlcv(limit=200)
                if df.empty or len(df) < 20:
                    time.sleep(MONITOR_INTERVAL)
                    continue

                current_price = df['close'].iloc[-1]

                # проверка виртуальной сделки (закрытие по TP/SL/тайм-аут)
                if self.virtual_trade is not None:
                    vt = self.virtual_trade
                    high = df['high'].iloc[-1]
                    low = df['low'].iloc[-1]
                    vt['bars_held'] += 1
                    result = None

                    if vt['signal'] == 'long':
                        if high >= vt['tp']:
                            result = 'tp'
                        elif low <= vt['sl']:
                            result = 'sl'
                    else:
                        if low <= vt['tp']:
                            result = 'tp'
                        elif high >= vt['sl']:
                            result = 'sl'

                    if result is None and vt['bars_held'] >= 10:
                        result = 'timeout'

                    if result is not None:
                        # завершаем сделку, начисляем баллы
                        if result == 'tp':
                            score_delta = SCORE_WIN
                            self.wins += 1
                        elif result == 'sl':
                            score_delta = SCORE_LOSS
                            self.losses += 1
                        else:
                            score_delta = SCORE_TIMEOUT

                        self.score += score_delta
                        self.total_trades += 1
                        # обновляем статистику текущего режима
                        idx = self.active_mode_idx
                        self.mode_stats[idx]['trades'] += 1
                        self.mode_stats[idx]['score'] += score_delta

                        log.info(
                            f"[{vt['mode']}] Сделка закрыта ({result}): "
                            f"вход {vt['entry_price']:.4f} -> выход {current_price:.4f} | "
                            f"баллы: {score_delta:+d} (всего: {self.score})"
                        )
                        self.virtual_trade = None

                        # адаптируем стиль после каждой сделки
                        self.adapt_mode()

                # если нет активной сделки — ищем сигнал
                if self.virtual_trade is None:
                    mode = MODES[self.active_mode_idx]
                    signal = self.get_signal(df, mode)
                    if signal:
                        log.info(f"[{mode['name']}] Сигнал {signal.upper()} @ {current_price:.4f}")
                        self.execute_trade(signal, current_price, mode['name'])

                # периодическое обучение нейросети
                if time.time() - self.last_nn_train_time > NN_RETRAIN_INTERVAL:
                    df_train = self.fetch_ohlcv(limit=NN_LOOKBACK)
                    if not df_train.empty:
                        self.train_nn(df_train)
                        self.last_nn_train_time = time.time()

                time.sleep(MONITOR_INTERVAL)

            except KeyboardInterrupt:
                log.info("Остановка бота")
                break
            except Exception as e:
                log.error(f"Ошибка: {e}", exc_info=True)
                time.sleep(MONITOR_INTERVAL)

if __name__ == "__main__":
    bot = AdaptiveScoreBot()
    bot.run_simulation()
