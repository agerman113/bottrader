#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
АВТОНОМНЫЙ БОТ: СКАНЕР МОНЕТ + RSI + НЕЙРОСЕТЬ + БАЛЛЫ
- Сканирует топ-30 монет по объёму, выбирает лучшую для RSI-торговли
- Нейросеть фильтрует сигналы, обучается под каждую монету
- Система баллов и адаптация стиля (агрессивный/умеренный/консервативный)
- Постоянная обратная связь
"""

import os
import time
import logging
import threading
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
SCAN_TIMEFRAME      = "5m"             # Таймфрейм для сканера монет (оценка волатильности)
STOP_LOSS_PERCENT   = 0.25
TAKE_PROFIT_PERCENT = 1.0
LEVERAGE            = 1
LIVE_TRADING        = False            # Пока только симуляция

# Система баллов
SCORE_WIN           = 2
SCORE_LOSS          = -2
SCORE_TIMEOUT       = 0
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
SCAN_BARS           = 60              # свечей для оценки волатильности
MIN_VOLUME_USDT     = 5_000_000       # мин. суточный объём
SCAN_RSI_OVERSOLD   = 20              # пороги для подсчёта пересечений в сканере
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
        logging.FileHandler("neuro_rsi_scanner_bot.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("NeuroRSIScanner")

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
class AdaptiveScoreBot:
    def __init__(self):
        self.symbol = None
        self.timeframe = TIMEFRAME
        self.balance = 100  # виртуальный баланс
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

        self.exchange = ccxt.bybit({
            "apiKey": os.getenv("BYBIT_API_KEY"),
            "secret": os.getenv("BYBIT_API_SECRET"),
            "enableRateLimit": True,
            "timeout": 10_000,
            "options": {"defaultType": "linear"},
        })

        self.virtual_trade = None
        self.last_scan_time = 0
        self._last_status_time = 0
        self._last_stat_time = 0

        # Параметры сканера
        self.scan_top_n = SCAN_TOP_N
        self.scan_interval = SCAN_INTERVAL
        self.scan_bars = SCAN_BARS
        self.min_volume_usdt = MIN_VOLUME_USDT
        self.scan_tf = SCAN_TIMEFRAME

    # ---------- Вспомогательные функции для RSI и свечей ----------
    def fetch_ohlcv(self, symbol, timeframe, limit=150):
        try:
            raw = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
            return df.iloc[:-1].reset_index(drop=True)  # убираем незакрытую свечу
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

    # ---------- СКАНЕР МОНЕТ (из первого бота) ----------
    def score_symbol(self, symbol, period=14):
        """Оценивает монету по количеству RSI-пересечений и волатильности."""
        try:
            df = self.fetch_ohlcv(symbol, self.scan_tf, limit=self.scan_bars + period + 5)
            if df.empty or len(df) < self.scan_bars:
                return None
            close = df["close"]
            high = df["high"]
            low = df["low"]
            atr_pct = ((high - low) / close).mean() * 100
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
        log.info(f"🔍 Сканирование топ-{self.scan_top_n} монет по объёму...")
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
            if vol >= self.min_volume_usdt:
                candidates.append((sym, vol))

        candidates.sort(key=lambda x: x[1], reverse=True)
        top = [sym for sym, _ in candidates[:self.scan_top_n]]
        log.info(f"  Отобрано {len(top)} монет для анализа RSI-активности")

        results = []
        for i, sym in enumerate(top, 1):
            result = self.score_symbol(sym)
            if result:
                results.append(result)
                log.info(f"  [{i:2d}/{len(top)}] {sym:<22} скор={result['score']:6.2f}  пересечений={result['crosses']}  ATR={result['atr_pct']:.2f}%")
            time.sleep(0.2)

        if not results:
            log.warning("Сканер не нашёл подходящих монет, используем BTC")
            return fallback

        results.sort(key=lambda x: x["score"], reverse=True)
        best = results[0]
        log.info("─" * 55)
        log.info(f"🏆 Лучшая монета: {best['symbol']}")
        log.info(f"   Скор={best['score']}  Пересечений={best['crosses']}  ATR={best['atr_pct']}%")
        log.info("─" * 55)
        log.info("  Топ-5 монет по скору:")
        for r in results[:5]:
            log.info(f"    {r['symbol']:<22} скор={r['score']:6.2f}  пересечений={r['crosses']}")
        return best["symbol"]

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
    def train_nn(self, df, symbol):
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
        log.info(f"🧠 Нейросеть обучена для {symbol} на {len(X)} примерах")
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

    # ---------- открытие виртуальной сделки ----------
    def execute_trade(self, signal, price, mode_name):
        tp_price = price * (1 + TAKE_PROFIT_PERCENT/100) if signal == "long" else price * (1 - TAKE_PROFIT_PERCENT/100)
        sl_price = price * (1 - STOP_LOSS_PERCENT/100) if signal == "long" else price * (1 + STOP_LOSS_PERCENT/100)
        self.virtual_trade = {
            "signal": signal,
            "entry_price": price,
            "mode": mode_name,
            "tp": tp_price,
            "sl": sl_price,
            "bars_held": 0,
            "start_time": time.time()
        }
        log.info(f"📈 ОТКРЫТА СДЕЛКА [{mode_name}] {signal.upper()} по {price:.4f} | TP={tp_price:.4f} SL={sl_price:.4f}")

    # ---------- закрытие виртуальной сделки и начисление баллов ----------
    def close_trade(self, result, current_price):
        vt = self.virtual_trade
        if vt is None:
            return
        if result == 'tp':
            score_delta = SCORE_WIN
            self.wins += 1
        elif result == 'sl':
            score_delta = SCORE_LOSS
            self.losses += 1
        else:
            score_delta = SCORE_TIMEOUT
            self.timeouts += 1
        self.score += score_delta
        self.total_trades += 1
        idx = self.active_mode_idx
        self.mode_stats[idx]['trades'] += 1
        self.mode_stats[idx]['score'] += score_delta
        log.info(f"🏁 ЗАКРЫТА [{vt['mode']}] {vt['signal'].upper()} ({result}) вход {vt['entry_price']:.4f} → выход {current_price:.4f} | баллы: {score_delta:+d} (всего: {self.score})")
        self.virtual_trade = None
        self.adapt_mode()

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

    # ---------- вывод информации об открытой сделке ----------
    def print_trade_status(self, current_price):
        if self.virtual_trade is None:
            return
        vt = self.virtual_trade
        bars = vt['bars_held']
        if vt['signal'] == 'long':
            dist_tp = (vt['tp'] - current_price) / current_price * 100
            dist_sl = (current_price - vt['sl']) / current_price * 100
        else:
            dist_tp = (current_price - vt['tp']) / current_price * 100
            dist_sl = (vt['sl'] - current_price) / current_price * 100
        elapsed = time.time() - vt['start_time']
        log.info(f"⚡ Сделка [{vt['mode']}] {vt['signal'].upper()} | Цена: {current_price:.4f} | Баров: {bars} | "
                 f"До TP: {dist_tp:.3f}% | До SL: {dist_sl:.3f}% | Длится: {elapsed:.0f}с")

    # ---------- статистика ----------
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

    # ---------- ГЛАВНЫЙ ЦИКЛ ----------
    def run(self):
        log.info("🚀 Бот запущен")

        # Определяем начальную монету
        if FIXED_SYMBOL:
            self.symbol = FIXED_SYMBOL
            log.info(f"Торгуем фиксированную пару: {self.symbol}")
        else:
            log.info("Запускаю первичный скан монет...")
            self.symbol = self.scan_best_symbol()
            self.last_scan_time = time.time()

        # Устанавливаем плечо (если нужно)
        try:
            self.exchange.set_leverage(LEVERAGE, self.symbol)
        except:
            pass

        # Обучаем нейросеть
        df_hist = self.fetch_ohlcv(self.symbol, self.timeframe, limit=NN_LOOKBACK)
        if not df_hist.empty:
            self.train_nn(df_hist, self.symbol)
        else:
            log.warning("Не удалось загрузить историю для обучения нейросети")

        self._last_status_time = time.time()
        self._last_stat_time = time.time()

        while True:
            try:
                now = time.time()

                # Сканирование новой монеты (если не фиксированная)
                if not FIXED_SYMBOL and now - self.last_scan_time > SCAN_INTERVAL:
                    new_symbol = self.scan_best_symbol()
                    if new_symbol != self.symbol:
                        log.info(f"🔄 Переключение на монету: {new_symbol}")
                        self.symbol = new_symbol
                        try:
                            self.exchange.set_leverage(LEVERAGE, self.symbol)
                        except:
                            pass
                        # Переобучаем нейросеть под новую монету
                        df_hist = self.fetch_ohlcv(self.symbol, self.timeframe, limit=NN_LOOKBACK)
                        if not df_hist.empty:
                            self.train_nn(df_hist, self.symbol)
                        self.virtual_trade = None  # закрываем старую виртуальную сделку
                    self.last_scan_time = now

                # Периодическая статистика
                if now - self._last_stat_time > 120:
                    self.print_stats()
                    self._last_stat_time = now

                # Получаем свежие свечи
                df = self.fetch_ohlcv(self.symbol, self.timeframe, limit=200)
                if df.empty or len(df) < 20:
                    time.sleep(5)
                    continue

                current_price = df['close'].iloc[-1]
                mode = MODES[self.active_mode_idx]

                # Проверка виртуальной сделки
                if self.virtual_trade:
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
                    if result:
                        self.close_trade(result, current_price)
                    elif now - self._last_status_time > 30:
                        self.print_trade_status(current_price)
                        self._last_status_time = now
                else:
                    # Мониторинг
                    if now - self._last_status_time > 30:
                        signal, rsi_val = self.get_signal(df, mode)
                        proba = self.nn_predict(df)
                        log.info(f"👁 Мониторинг [{self.symbol}] Цена: {current_price:.4f} | RSI({mode['period']}): {rsi_val:.1f} | "
                                 f"NN proba: {proba:.2f} | Режим: {mode['name']} | Баллы: {self.score}")
                        self._last_status_time = now
                    # Поиск сигнала
                    signal, rsi_val = self.get_signal(df, mode)
                    if signal:
                        self.execute_trade(signal, current_price, mode['name'])

                # Переобучение нейросети
                if now - self.last_nn_train_time > NN_RETRAIN_INTERVAL:
                    log.info("🔄 Переобучение нейросети...")
                    df_train = self.fetch_ohlcv(self.symbol, self.timeframe, limit=NN_LOOKBACK)
                    if not df_train.empty:
                        self.train_nn(df_train, self.symbol)
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
    bot = AdaptiveScoreBot()
    bot.run()
