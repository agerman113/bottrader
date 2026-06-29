#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ГИБРИДНЫЙ БОТ v2 (трендовая стратегия + улучшенный риск-менеджмент)
====================================================================
Ключевые отличия от v1:
  • Вход ТОЛЬКО по тренду (15m SuperTrend + EMA200 как фильтр старшего ТФ).
  • SL/TP рассчитываются от ATR, а не фиксированные проценты:
        SL  = 1.5 * ATR,  TP = 3.0 * ATR  (соотношение R:R = 1:2).
  • Размер позиции — по фиксированному риску на сделку (1% от баланса),
    а не дамп 80% в один ордер.
  • Вход по откату RSI внутри тренда (RSI 14), а не на шумных кроссах.
  • Нет «принудительных» сделок — терпеливо ждём сигнал.
  • Мониторинг НЕ блокирует основной цикл (неблокирующий poll).
  • Грамотный трейлинг: безубыток при +1 ATR, трейл при +2 ATR.
  • Улучшенная нейросеть: цель — предсказать движение вперёд,
    расширенный набор фичей, больше эпох.
"""

import os, time, logging, json
import numpy as np
import ccxt
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timezone
from sklearn.preprocessing import MinMaxScaler

load_dotenv()

# ============================================================
#                      КОНФИГУРАЦИЯ
# ============================================================
FIXED_SYMBOL = ""                       # "" -> авто-сканер
WHITELIST_SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
    "XRP/USDT:USDT", "DOGE/USDT:USDT", "ADA/USDT:USDT",
    "BNB/USDT:USDT", "LINK/USDT:USDT", "DOT/USDT:USDT",
    "AVAX/USDT:USDT",
]

# --- Таймфреймы ---
ENTRY_TIMEFRAME = "1m"                 # рабочие бары
TREND_TIMEFRAME = "15m"                # старший ТФ для направления тренда

# --- Риск-менеджмент (ATR-based) ---
LEVERAGE = 1
SL_ATR_MULT = 1.5                      # стоп = 1.5 * ATR(entry TF)
TP_ATR_MULT = 3.0                      # тейк = 3.0 * ATR
RISK_PER_TRADE = 0.01                  # 1% капитала на сделку
ATR_PERIOD = 14
MIN_ATR_PCT = 0.25                     # если волатильность ниже — пропускаем (нет движения)

# --- RSI: вход по откату внутри тренда ---
RSI_PERIOD = 14
RSI_LONG_PULLBACK_MAX = 50             # в аптренде: вход в лонг, когда RSI откатился <=50
RSI_LONG_EXIT_MIN = 52                 # и развернулся вверх (закрытие > 52)
RSI_SHORT_PULLBACK_MIN = 50
RSI_SHORT_EXIT_MAX = 48

# --- Тренд (старший ТФ) ---
TREND_SUPERTREND_PERIOD = 10
TREND_SUPERTREND_MULT = 3.0
TREND_USE_EMA = True
TREND_EMA = 200

# --- Подтверждение входа ---
REQUIRE_PULLBACK_CONFIRM = True        # ждать разворотного бара (close выше prev close для long)

# --- Трейлинг / безубыток ---
BE_TRIGGER_ATR = 1.0                   # +1 ATR -> безубыток
TRAIL_TRIGGER_ATR = 2.0                # +2 ATR -> трейлинг
TRAIL_ATR_MULT = 1.0                   # отступ трейла в ATR
TRADE_MAX_LIFETIME = 3600              # максимум в сделке, сек

# --- Нейросеть ---
NN_LOOKBACK = 600
NN_EPOCHS = 50
NN_BATCH = 32
NN_LR = 0.005
NN_HIDDEN = 48
NN_FORWARD_BARS = 3                    # предсказываем движение через 3 бара
NN_TARGET_PCT = 0.0015                 # «успех» = +0.15% вперёд
NN_CONFIDENCE_MIN = 0.55               # подтверждение направления (long: proba long-класса)
NN_RETRAIN_INTERVAL = 1800

# --- Сканер ---
SCAN_TIMEFRAME = "5m"
SCAN_BARS = 60
MIN_VOLUME_USDT = 5_000_000
SCAN_INTERVAL = 14400
MAX_SPREAD_PCT = 0.12

# --- Режимы (адаптация) ---
MIN_TRADES_FOR_ADAPT = 6
MODES = [
    {"name": "Сбалансированный", "sl_mult": 1.5, "tp_mult": 3.0, "rsi_pb": 50},
    {"name": "Консервативный",   "sl_mult": 2.0, "tp_mult": 4.0, "rsi_pb": 45},
    {"name": "Агрессивный",      "sl_mult": 1.3, "tp_mult": 2.5, "rsi_pb": 55},
]

# --- Оценка сделок ---
SCORE_WIN = 2
SCORE_LOSS = -2
SCORE_BE = 0.5
SCORE_TIMEOUT = -0.5

# --- Файлы состояния ---
STATE_FILE = "hybrid_state_v2.json"
REPORT_INTERVAL = 900

# ============================================================
#                       ЛОГИРОВАНИЕ
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]  %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler("hybrid_bot_v2.log", encoding="utf-8")],
)
log = logging.getLogger("HybridBotV2")

# ============================================================
#                  НЕЙРОСЕТЬ (Adam, 2 скрытых)
# ============================================================
class NeuralPredictor:
    def __init__(self, input_size=10, hidden=48, lr=0.005):
        rng = np.random.default_rng(42)
        self.W1 = rng.standard_normal((input_size, hidden)) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden)
        self.W2 = rng.standard_normal((hidden, 1)) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(1)
        self._adam_init()
        self.lr = lr
        self.beta1, self.beta2, self.eps, self.t = 0.9, 0.999, 1e-8, 0

    def _adam_init(self):
        self.mW1, self.vW1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.mb1, self.vb1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.mW2, self.vW2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.mb2, self.vb2 = np.zeros_like(self.b2), np.zeros_like(self.b2)

    @staticmethod
    def _relu(z): return np.maximum(0, z)
    @staticmethod
    def _sig(x): return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self._relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self._sig(self.z2)
        return self.a2

    def backward(self, X, y):
        m = X.shape[0]
        dZ2 = self.a2 - y
        dW2 = (self.a1.T @ dZ2) / m
        db2 = dZ2.mean(axis=0)
        dZ1 = (dZ2 @ self.W2.T) * (self.z1 > 0)
        dW1 = (X.T @ dZ1) / m
        db1 = dZ1.mean(axis=0)
        self.t += 1
        for p, mp, vp, g in [
            (self.W1, self.mW1, self.vW1, dW1),
            (self.b1, self.mb1, self.vb1, db1),
            (self.W2, self.mW2, self.vW2, dW2),
            (self.b2, self.mb2, self.vb2, db2),
        ]:
            mp *= self.beta1; mp += (1 - self.beta1) * g
            vp *= self.beta2; vp += (1 - self.beta2) * (g * g)
            mhat = mp / (1 - self.beta1 ** self.t)
            vhat = vp / (1 - self.beta2 ** self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

    def fit(self, X, y, epochs=50, batch=32):
        for _ in range(epochs):
            idx = np.random.permutation(len(X))
            for i in range(0, len(X), batch):
                bi = idx[i:i + batch]
                self.forward(X[bi])
                self.backward(X[bi], y[bi].reshape(-1, 1))

    def predict(self, X):
        return self.forward(X)[0, 0]

# ============================================================
#                  БЕЗОПАСНЫЕ ОБЁРТКИ API
# ============================================================
exchange = ccxt.bybit({
    "apiKey": os.getenv("BYBIT_API_KEY"),
    "secret": os.getenv("BYBIT_API_SECRET"),
    "enableRateLimit": True,
    "timeout": 10_000,
    "options": {"defaultType": "linear"},
})

def safe_api(func, *a, retries=3, delay=1.0, **kw):
    for attempt in range(retries):
        try:
            return func(*a, **kw)
        except ccxt.RateLimitExceeded:
            log.warning("Rate limit, пауза 5с"); time.sleep(5)
        except ccxt.NetworkError as e:
            log.warning(f"Сеть: {e}"); time.sleep(delay); delay *= 2
        except Exception as e:
            if attempt == retries - 1: raise
            log.warning(f"API ошибка: {e}"); time.sleep(delay); delay *= 2
    return None

def fetch_ohlcv(symbol, tf, limit=150):
    try: return safe_api(exchange.fetch_ohlcv, symbol, tf, limit=limit) or []
    except Exception: return []

def fetch_positions(symbols=None):
    try: return safe_api(exchange.fetch_positions, symbols) or []
    except Exception: return []

def fetch_price(symbol):
    try:
        t = safe_api(exchange.fetch_ticker, symbol)
        return float(t["last"]) if t else None
    except Exception: return None

# ============================================================
#                   ИНДИКАТОРЫ
# ============================================================
def calc_rsi(close, period):
    d = close.diff()
    gain = d.clip(lower=0)
    loss = (-d).clip(lower=0)
    ag = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    al = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_atr(df, period=14):
    hi, lo, pc = df["h"], df["l"], df["c"].shift(1)
    tr = pd.concat([hi - lo, (hi - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()

def calc_supertrend(df, period=10, mult=3.0):
    atr = calc_atr(df, period)
    hl2 = (df["h"] + df["l"]) / 2
    ub = hl2 + mult * atr
    lb = hl2 - mult * atr
    trend = pd.Series(1, index=df.index)
    st_line = pd.Series(0.0, index=df.index)
    for i in range(1, len(df)):
        c = df["c"].iloc[i]
        if st_line.iloc[i - 1] == 0:
            st_line.iloc[i] = lb.iloc[i] if c > lb.iloc[i] else ub.iloc[i]
            trend.iloc[i] = 1 if c > lb.iloc[i] else -1
        elif c > st_line.iloc[i - 1]:
            st_line.iloc[i] = max(lb.iloc[i], st_line.iloc[i - 1])
            trend.iloc[i] = 1
        else:
            st_line.iloc[i] = min(ub.iloc[i], st_line.iloc[i - 1])
            trend.iloc[i] = -1
    return (trend == 1), st_line, atr

def build_features(df):
    """Возвращает DataFrame с признаками для NN (и сам df с доп. колонками)."""
    c = df["c"]
    df = df.copy()
    df["ret1"] = c.pct_change(1)
    df["ret3"] = c.pct_change(3)
    df["ret5"] = c.pct_change(5)
    df["rsi"] = calc_rsi(c, RSI_PERIOD)
    df["ema9"] = c.ewm(9).mean()
    df["ema21"] = c.ewm(21).mean()
    df["ema_slope"] = df["ema9"] - df["ema9"].shift(3)
    macd = c.ewm(12).mean() - c.ewm(26).mean()
    df["macd_hist"] = macd - macd.ewm(9).mean()
    df["vol_ratio"] = df["v"] / (df["v"].rolling(20).mean().replace(0, np.nan))
    df["atr_pct"] = calc_atr(df, ATR_PERIOD) / c
    df["dist_ema"] = (c - df["ema21"]) / c
    feat = ["ret1", "ret3", "ret5", "rsi", "ema_slope", "macd_hist",
            "vol_ratio", "atr_pct", "dist_ema"]
    return df, feat

# ============================================================
#                       БОТ
# ============================================================
class HybridBotV2:
    def __init__(self):
        self.symbol = None
        self.total_trades = 0
        self.wins = self.losses = self.timeouts = 0
        self.score = 0
        self.active_mode_idx = 0
        self.mode_stats = [{"trades": 0, "score": 0} for _ in MODES]

        self.nn = None
        self.scaler = MinMaxScaler()
        self.nn_trained = False
        self.nn_confidence = NN_CONFIDENCE_MIN
        self.last_nn_train = 0
        self.feat_cols = None

        self.live_trading = True
        self.start_time = time.time()
        self.last_scan_time = 0
        self._last_status = 0
        self.active_trade = None
        self.last_signal_time = 0

    # ---------- Сохранение/загрузка состояния ----------
    def save_state(self):
        try:
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "total_trades": self.total_trades, "wins": self.wins,
                    "losses": self.losses, "timeouts": self.timeouts,
                    "score": self.score, "mode_idx": self.active_mode_idx,
                    "mode_stats": self.mode_stats,
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.warning(f"Не удалось сохранить состояние: {e}")

    def load_state(self):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
            self.total_trades = d.get("total_trades", 0)
            self.wins = d.get("wins", 0)
            self.losses = d.get("losses", 0)
            self.timeouts = d.get("timeouts", 0)
            self.score = d.get("score", 0)
            self.active_mode_idx = d.get("mode_idx", 0)
            self.mode_stats = d.get("mode_stats", self.mode_stats)
            log.info(f"Состояние загружено: сделок={self.total_trades} счёт={self.score}")
        except Exception:
            pass

    # ---------- Сканер монет по ATR и объёму ----------
    def scan_best_symbol(self):
        syms = WHITELIST_SYMBOLS or ["BTC/USDT:USDT"]
        log.info(f"Сканирование {len(syms)} монет (ATR + объём)")
        best, best_score = "BTC/USDT:USDT", 0
        for sym in syms:
            raw = fetch_ohlcv(sym, SCAN_TIMEFRAME, limit=SCAN_BARS + 5)
            if len(raw) < SCAN_BARS: continue
            df = pd.DataFrame(raw, columns=["ts", "o", "h", "l", "c", "v"])
            atr_pct = (calc_atr(df, 14) / df["c"]).tail(20).mean()
            vol_usdt = (df["c"] * df["v"]).tail(20).mean()
            if vol_usdt < MIN_VOLUME_USDT: continue
            score = atr_pct * (1 + np.log10(vol_usdt / MIN_VOLUME_USDT + 1))
            if atr_pct >= MIN_ATR_PCT / 100 and score > best_score:
                best_score, best = score, sym
        log.info(f"Лучшая монета: {best} (score={best_score:.3f})")
        return best

    # ---------- Обучение NN ----------
    def train_nn(self, df_raw):
        if len(df_raw) < 200: return
        df, feat = build_features(df_raw)
        self.feat_cols = feat
        future = df["c"].shift(-NN_FORWARD_BARS)
        fwd_ret = (future - df["c"]) / df["c"]
        target = (fwd_ret > NN_TARGET_PCT).astype(int)
        target.name = "y"
        work = df[feat + ["c"]].join(target).dropna()
        X = work[feat].values
        y = work["y"].astype(int).values
        if len(X) < 100:
            log.info(f"Недостаточно данных для NN ({len(X)})"); return
        # балансировка классов
        pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
        if len(pos) < 10 or len(neg) < 10:
            log.info("NN: классы несбалансированы, пропуск"); return
        n = min(len(pos), len(neg))
        idx = np.concatenate([np.random.choice(pos, n), np.random.choice(neg, n)])
        X, y = X[idx], y[idx]
        self.scaler.fit(X)
        Xs = self.scaler.transform(X)
        if self.nn is None or self.nn.W1.shape[0] != X.shape[1]:
            self.nn = NeuralPredictor(X.shape[1], NN_HIDDEN, NN_LR)
        self.nn.fit(Xs, y, NN_EPOCHS, NN_BATCH)
        self.nn_trained = True
        self.last_nn_train = time.time()
        # точность
        acc = (self.nn.forward(Xs).flatten().round() == y).mean()
        log.info(f"NN обучена: {len(X)} примеров, точность {acc*100:.1f}%")

    def nn_predict(self, df_raw):
        if not self.nn_trained or self.feat_cols is None: return 0.5
        try:
            df, feat = build_features(df_raw)
            last = df[feat].iloc[-1:].values
            last_s = self.scaler.transform(last)
            return float(self.nn.predict(last_s))
        except Exception:
            return 0.5

    # ---------- Тренд старшего ТФ ----------
    def get_trend(self, symbol):
        raw = fetch_ohlcv(symbol, TREND_TIMEFRAME, limit=TREND_EMA + 50)
        if len(raw) < TREND_EMA + 20:
            return "neutral", None
        df = pd.DataFrame(raw, columns=["ts", "o", "h", "l", "c", "v"])
        bull_st, st_line, _ = calc_supertrend(df, TREND_SUPERTREND_PERIOD, TREND_SUPERTREND_MULT)
        st_dir = "bullish" if bull_st.iloc[-1] else "bearish"
        if TREND_USE_EMA:
            ema = df["c"].ewm(TREND_EMA).mean().iloc[-1]
            price = df["c"].iloc[-1]
            ema_dir = "bullish" if price > ema else "bearish"
            if st_dir == ema_dir:
                return st_dir, st_line.iloc[-1]
            return "neutral", st_line.iloc[-1]
        return st_dir, st_line.iloc[-1]

    # ---------- Сигнал ----------
    def get_signal(self, df, mode):
        if len(df) < max(RSI_PERIOD, 30): return None
        rsi = calc_rsi(df["c"], RSI_PERIOD)
        r1, r2 = rsi.iloc[-2], rsi.iloc[-1]
        c = df["c"]; prev_close = c.iloc[-2]
        trend, _ = self.get_trend(self.symbol)
        if trend == "neutral": return None

        signal = None
        if trend == "bullish":
            # откат: RSI был <= порога и развернулся вверх
            if r1 <= mode["rsi_pb"] and r2 >= RSI_LONG_EXIT_MIN:
                signal = "long"
                if REQUIRE_PULLBACK_CONFIRM and c.iloc[-1] <= prev_close:
                    signal = None
        else:  # bearish
            if r1 >= (100 - mode["rsi_pb"]) and r2 <= RSI_SHORT_EXIT_MAX:
                signal = "short"
                if REQUIRE_PULLBACK_CONFIRM and c.iloc[-1] >= prev_close:
                    signal = None

        if signal is None: return None

        # подтверждение нейросетью: long -> ждём proba роста >= порога
        proba = self.nn_predict(df)
        if signal == "long" and proba < self.nn_confidence:
            log.info(f"NN отклонил LONG (proba={proba:.2f} < {self.nn_confidence:.2f})")
            return None
        if signal == "short" and (1 - proba) < self.nn_confidence:
            log.info(f"NN отклонил SHORT (proba падения={1-proba:.2f} < {self.nn_confidence:.2f})")
            return None
        return signal

    # ---------- Размер позиции по риску ----------
    def calc_qty(self, entry, sl_price, side):
        try:
            bal = exchange.fetch_balance()
            equity = float(bal.get("USDT", {}).get("free", 0))
        except Exception:
            return None
        if equity <= 0: return None
        risk_amt = equity * RISK_PER_TRADE
        sl_dist = abs(entry - sl_price)
        if sl_dist <= 0: return None
        raw_qty = risk_amt / sl_dist
        market = exchange.market(self.symbol)
        min_amt = float(market.get("limits", {}).get("amount", {}).get("min", 0) or 0)
        min_cost = float(market.get("limits", {}).get("cost", {}).get("min", 0) or 0)
        if min_amt and raw_qty < min_amt: raw_qty = min_amt
        if min_cost and raw_qty * entry < min_cost: raw_qty = min_cost / entry
        try:
            qty = float(exchange.amount_to_precision(self.symbol, raw_qty))
        except Exception:
            qty = round(raw_qty, 4)
        if qty <= 0: return None
        return qty

    # ---------- Открытие позиции ----------
    def open_position(self, signal, price):
        atr = calc_atr(self._last_df, ATR_PERIOD).iloc[-1] if self._last_df is not None else price * 0.005
        mode = MODES[self.active_mode_idx]
        sl_dist = atr * mode["sl_mult"]
        tp_dist = atr * mode["tp_mult"]
        if signal == "long":
            sl, tp = price - sl_dist, price + tp_dist
        else:
            sl, tp = price + sl_dist, price - tp_dist

        if self.live_trading:
            qty = self.calc_qty(price, sl, signal)
            if not qty:
                log.warning("Не удалось рассчитать размер позиции"); return False
            tp_p = float(exchange.price_to_precision(self.symbol, tp))
            sl_p = float(exchange.price_to_precision(self.symbol, sl))
            side = "buy" if signal == "long" else "sell"
            try:
                exchange.create_order(self.symbol, "market", side, qty,
                                      params={"takeProfit": tp_p, "stopLoss": sl_p, "tpTrigger": "LastPrice"})
                log.info(f"✅ {signal.upper()} {qty} @ {price:.4f} TP={tp_p:.4f} SL={sl_p:.4f} "
                         f"(ATR={atr:.4f}, R:R=1:{mode['tp_mult']/mode['sl_mult']:.1f})")
                self.active_trade = {
                    "side": signal, "entry": price, "tp": tp_p, "sl": sl_p, "qty": qty,
                    "atr": atr, "time": time.time(), "peak": price, "be_done": False,
                    "trail_active": False, "virtual": False,
                }
                return True
            except Exception as e:
                log.error(f"Ошибка открытия: {e}"); return False
        else:
            log.info(f"📈 СИМ {signal.upper()} @ {price:.4f} SL={sl:.4f} TP={tp:.4f}")
            self.active_trade = {
                "side": signal, "entry": price, "tp": tp, "sl": sl, "qty": 0,
                "atr": atr, "time": time.time(), "peak": price, "be_done": False,
                "trail_active": False, "virtual": True, "bars": 0,
            }
            return True

    # ---------- Мониторинг (неблокирующий, один шаг за вызов) ----------
    def monitor_position(self):
        if not self.active_trade: return
        t = self.active_trade
        price = fetch_price(self.symbol)
        if price is None: return
        entry = t["entry"]

        # --- симуляция ---
        if t.get("virtual"):
            t["bars"] += 1
            raw = fetch_ohlcv(self.symbol, ENTRY_TIMEFRAME, limit=3)
            if len(raw) >= 2:
                hi, lo = raw[-1][2], raw[-1][3]
                if t["side"] == "long":
                    if hi >= t["tp"]: self._close_virtual("tp", price); return
                    if lo <= t["sl"]: self._close_virtual("sl", price); return
                else:
                    if lo <= t["tp"]: self._close_virtual("tp", price); return
                    if hi >= t["sl"]: self._close_virtual("sl", price); return
            if t["bars"] >= 60: self._close_virtual("timeout", price)
            return

        # --- реальная позиция ---
        # время жизни
        if time.time() - t["time"] > TRADE_MAX_LIFETIME:
            log.warning("Дедлайн — закрываем по рынку")
            self._market_close(t["qty"], t["side"])
            self._record_result("timeout", entry, price, t["side"]); return

        pos = fetch_positions([self.symbol])
        active = [p for p in pos if float(p.get("contracts", 0) or 0) > 0 and p.get("side") == t["side"]]
        if not active:
            # позиция закрылась биржей (SL/TP). Определяем результат по цене.
            if t["side"] == "long":
                res = "tp" if price >= t["tp"] else ("sl" if price <= t["sl"] else "be")
            else:
                res = "tp" if price <= t["tp"] else ("sl" if price >= t["sl"] else "be")
            self._record_result(res, entry, price, t["side"]); return

        # P&L %
        pnl = (price / entry - 1) * 100 if t["side"] == "long" else (entry / price - 1) * 100
        atr = t["atr"]

        # безубыток при +1 ATR
        if not t["be_done"]:
            be_trig = BE_TRIGGER_ATR * atr
            profit_dist = (price - entry) if t["side"] == "long" else (entry - price)
            if profit_dist >= be_trig:
                new_sl = entry * (1 + 0.0005) if t["side"] == "long" else entry * (1 - 0.0005)
                new_sl = float(exchange.price_to_precision(self.symbol, new_sl))
                self._move_sl(new_sl, t["side"])
                t["sl"] = new_sl; t["be_done"] = True
                log.info(f"🔒 Безубыток: SL→{new_sl:.4f} (P&L {pnl:+.2f}%)")

        # трейлинг при +2 ATR
        if not t["trail_active"]:
            trig = TRAIL_TRIGGER_ATR * atr
            profit_dist = (price - entry) if t["side"] == "long" else (entry - price)
            if profit_dist >= trig:
                t["trail_active"] = True
                log.info(f"📐 Трейлинг активирован (P&L {pnl:+.2f}%)")

        if t["trail_active"]:
            if t["side"] == "long":
                if price > t["peak"]: t["peak"] = price
                new_sl = t["peak"] - TRAIL_ATR_MULT * atr
            else:
                if price < t["peak"]: t["peak"] = price
                new_sl = t["peak"] + TRAIL_ATR_MULT * atr
            new_sl = float(exchange.price_to_precision(self.symbol, new_sl))
            if (t["side"] == "long" and new_sl > t["sl"]) or (t["side"] == "short" and new_sl < t["sl"]):
                self._move_sl(new_sl, t["side"])
                t["sl"] = new_sl

        log.info(f"👁 {self.symbol} {t['side']} цена={price:.4f} P&L={pnl:+.2f}% SL={t['sl']:.4f}")

    def _move_sl(self, new_sl, side):
        try:
            exchange.create_order(self.symbol, "market",
                                  "sell" if side == "long" else "buy", 0,
                                  params={"stopLoss": new_sl, "reduceOnly": True})
            log.info(f"   SL перемещён → {new_sl:.4f}")
        except Exception as e:
            log.warning(f"   не удалось переставить SL: {e}")

    def _market_close(self, qty, side):
        try:
            exchange.create_order(self.symbol, "market",
                                  "sell" if side == "long" else "buy", qty,
                                  params={"reduceOnly": True})
        except Exception as e:
            log.error(f"Закрытие не удалось: {e}")

    def _close_virtual(self, exit_type, price):
        t = self.active_trade
        self._record_result(exit_type, t["entry"], price, t["side"])

    # ---------- Учёт результата ----------
    def _record_result(self, exit_type, entry, exit_price, side):
        profit = ((exit_price - entry) / entry * 100) if side == "long" \
            else ((entry - exit_price) / entry * 100)
        if exit_type == "tp":
            sc = SCORE_WIN; self.wins += 1
        elif exit_type == "sl":
            sc = SCORE_LOSS; self.losses += 1
        elif exit_type == "be":
            sc = SCORE_BE
        else:  # timeout
            sc = SCORE_BE if profit >= 0 else SCORE_TIMEOUT
            self.timeouts += 1
        self.score += sc
        self.total_trades += 1
        self.mode_stats[self.active_mode_idx]["trades"] += 1
        self.mode_stats[self.active_mode_idx]["score"] += sc
        wr = (self.wins / max(1, self.wins + self.losses)) * 100
        log.info(f"🏁 {exit_type.upper()} {side} {profit:+.2f}% | балл {sc:+.1f} "
                 f"| всего {self.score} | winrate {wr:.0f}%")
        self.active_trade = None
        self.last_signal_time = time.time()
        self.save_state()

        # адаптация режима
        if self.total_trades >= MIN_TRADES_FOR_ADAPT:
            best = max(range(len(MODES)), key=lambda i: self.mode_stats[i]["score"])
            if best != self.active_mode_idx and self.mode_stats[best]["trades"] >= 3:
                log.info(f"⚙ Стиль: {MODES[self.active_mode_idx]['name']} → {MODES[best]['name']}")
                self.active_mode_idx = best

    # ---------- Главный цикл ----------
    def run(self):
        self.load_state()
        log.info("=== Запуск гибридного бота v2 ===")
        try:
            bal = exchange.fetch_balance()["USDT"]["free"]
            if bal < 5:
                log.warning(f"Баланс {bal:.2f} USDT — режим симуляции")
                self.live_trading = False
            else:
                log.info(f"Баланс: {bal:.2f} USDT")
        except Exception:
            self.live_trading = False

        self.symbol = FIXED_SYMBOL or self.scan_best_symbol()
        log.info(f"Пара: {self.symbol} | Режим: {'LIVE' if self.live_trading else 'SIM'}")
        if self.live_trading:
            try: exchange.set_leverage(LEVERAGE, self.symbol)
            except Exception: pass

        self._last_df = None
        # первичное обучение
        raw = fetch_ohlcv(self.symbol, ENTRY_TIMEFRAME, limit=NN_LOOKBACK)
        if len(raw) > 200:
            self._last_df = pd.DataFrame(raw, columns=["ts", "o", "h", "l", "c", "v"])
            self.train_nn(self._last_df)

        self.last_signal_time = time.time()

        while True:
            now = time.time()
            # периодический рескан
            if not FIXED_SYMBOL and now - self.last_scan_time > SCAN_INTERVAL and not self.active_trade:
                self.symbol = self.scan_best_symbol()
                self.last_scan_time = now

            raw = fetch_ohlcv(self.symbol, ENTRY_TIMEFRAME, limit=200)
            if len(raw) < 60:
                time.sleep(10); continue
            df = pd.DataFrame(raw, columns=["ts", "o", "h", "l", "c", "v"])
            self._last_df = df
            price = df["c"].iloc[-1]

            # мониторинг активной сделки
            if self.active_trade:
                self.monitor_position()
                # мягкая адаптация порога NN по времени простоя
                if self.live_trading and now - self.last_signal_time > 1800 \
                        and self.nn_confidence > 0.45:
                    self.nn_confidence = max(0.45, self.nn_confidence - 0.025)
                    log.info(f"Порог NN снижен до {self.nn_confidence:.2f}")
                time.sleep(15); continue

            # нет позиции -> ищем сигнал
            mode = MODES[self.active_mode_idx]
            signal = self.get_signal(df, mode)
            if signal:
                self.open_position(signal, price)
            else:
                # плавно повышаем порог обратно, если давно не было сделок
                if now - self.last_signal_time > 1800 and self.nn_confidence < NN_CONFIDENCE_MIN:
                    self.nn_confidence = min(NN_CONFIDENCE_MIN, self.nn_confidence + 0.02)

            if now - self._last_status > 60:
                rsi = calc_rsi(df["c"], RSI_PERIOD).iloc[-1]
                trend, _ = self.get_trend(self.symbol)
                log.info(f"👁 {self.symbol} {price:.4f} RSI={rsi:.1f} тренд(15m)={trend} "
                         f"стиль={mode['name']} счёт={self.score}")
                self._last_status = now

            # переобучение NN
            if now - self.last_nn_train > NN_RETRAIN_INTERVAL:
                self.train_nn(df)
                self.last_nn_train = now

            time.sleep(10)


if __name__ == "__main__":
    bot = HybridBotV2()
    try:
        bot.run()
    except KeyboardInterrupt:
        log.info("Остановлен пользователем")
    except Exception as e:
        log.exception(f"Критическая ошибка: {e}")
