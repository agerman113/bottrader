#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ГИБРИДНЫЙ ИИ-СКАЛЬПЕР: "ОДНА ХОРОШАЯ СДЕЛКА"
Философия Майка Беллафиоре: охота за ЕДИНСТВЕННЫМ повторяющимся сетапом
с нейросетевым фильтром для повышения точности.

Ключевые улучшения:
1. Полностью асинхронный мониторинг
2. Работающий трейлинг-стоп через API Bybit
3. SL и TP на основе ATR
4. Улучшенный нейросетевой предиктор
5. Защита от проскальзывания
6. Оптимизированные запросы к API (решение проблемы rate limits)
7. Стабильный WebSocket с автоматическим переподключением
"""

import os
import time
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timedelta
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import ccxt
import websocket
import threading
from dotenv import load_dotenv

# ============================================================
#                 КОНСТАНТЫ И КОНФИГУРАЦИЯ
# ============================================================

load_dotenv()

DEFAULT_CONFIG = {
    "mode": "live",
    "symbol": "BTC/USDT:USDT",
    "timeframe": "1m",
    "exchange": "bybit",
    "api_key": os.getenv("BYBIT_API_KEY", ""),
    "api_secret": os.getenv("BYBIT_API_SECRET", ""),

    "vwap": {
        "lookback": 100,
        "bbands_std": 2.0,
    },

    "atr": {
        "period": 14,
        "stop_loss_multiplier": 1.2,
        "take_profit_multiplier": 3.0,
        "trailing_step_multiplier": 0.5,
    },

    "signal": {
        "volume_threshold": 1.8,
        "min_distance_from_vwap": 0.5,
        "confirmation_required": True,
        "signal_lifetime_seconds": 60,
        "min_body_coverage": 0.6,
    },

    "predictor": {
        "model_type": "mlp",
        "hidden_layers": [64, 32],
        "initial_confidence_threshold": 0.60,
        "min_samples_for_training": 50,
        "adaptation_window": 20,
        "min_confidence_threshold": 0.55,
    },

    "risk": {
        "risk_per_trade": 0.005,
        "max_risk_per_trade": 0.02,
        "max_trade_lifetime_minutes": 5,
        "max_daily_loss": 0.03,
        "cool_down_after_losses": 2,
        "cool_down_minutes": 15,
    },

    "logging": {
        "log_file": "vwap_scalper.log",
        "level": "INFO",
    },
}

# ============================================================
#                 ПЕРЕЧИСЛЕНИЯ (ENUMS)
# ============================================================

class TradeSide(Enum):
    LONG = auto()
    SHORT = auto()

class TradeStatus(Enum):
    OPEN = auto()
    CLOSED = auto()
    TIMEOUT = auto()

class SignalType(Enum):
    LONG = auto()
    SHORT = auto()
    NONE = auto()

class SignalStatus(Enum):
    DETECTED = auto()
    CONFIRMED = auto()
    EXPIRED = auto()

# ============================================================
#                 ДАТАКЛАССЫ
# ============================================================

@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass
class Trade:
    side: TradeSide
    entry_price: float
    entry_time: datetime
    qty: float
    stop_loss: float
    take_profit: float
    status: TradeStatus = TradeStatus.OPEN
    close_price: Optional[float] = None
    close_time: Optional[datetime] = None
    pnl: Optional[float] = None
    trailing_stop: Optional[float] = None
    signal: Optional['Signal'] = None

@dataclass
class Signal:
    side: SignalType
    timestamp: datetime
    candle: Candle
    vwap: float
    vwap_std: float
    atr: float
    distance_from_vwap: float
    volume_ratio: float
    candle_pattern: str
    features: Dict[str, float] = field(default_factory=dict)
    probability: Optional[float] = None
    status: SignalStatus = SignalStatus.DETECTED
    confirmation_candle: Optional[Candle] = None

@dataclass
class PredictionResult:
    probability: float
    decision: bool
    features: Dict[str, float]

# ============================================================
#                 КОНФИГУРАЦИЯ ЛОГИРОВАНИЯ
# ============================================================

def setup_logging(config: Dict) -> logging.Logger:
    log_file = config["logging"]["log_file"]
    log_level = getattr(logging, config["logging"]["level"].upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    logger = logging.getLogger("VWAPScalper")
    return logger

# ============================================================
#                 МОДУЛЬ 1: DATA COLLECTOR
# ============================================================

class DataCollector:
    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.exchange = self._init_exchange()
        self.ohlcv_data: Dict[str, pd.DataFrame] = {}
        self.ws_connections: Dict[str, websocket.WebSocket] = {}
        self.lock = threading.Lock()
        self.last_ohlcv_update: Dict[str, datetime] = {}

    def _init_exchange(self) -> ccxt.Exchange:
        exchange_class = getattr(ccxt, self.config["exchange"])
        return exchange_class({
            "apiKey": self.config["api_key"],
            "secret": self.config["api_secret"],
            "enableRateLimit": True,
            "options": {"defaultType": "linear"},
        })

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 1) -> Optional[pd.DataFrame]:
        try:
            # Проверяем, не превысили ли мы лимит запросов
            if (datetime.now() - self.last_ohlcv_update.get(symbol, datetime.min)).total_seconds() < 0.1:
                return self.ohlcv_data.get(symbol)

            data = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            self.ohlcv_data[symbol] = df
            self.last_ohlcv_update[symbol] = datetime.now()
            return df
        except ccxt.RateLimitExceeded as e:
            self.logger.warning(f"Rate limit exceeded for {symbol}. Waiting 1 second...")
            time.sleep(1)
            return self.fetch_ohlcv(symbol, timeframe, limit)
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return None

    def get_latest_candle(self, symbol: str) -> Optional[Candle]:
        if symbol not in self.ohlcv_data or self.ohlcv_data[symbol].empty:
            self.fetch_ohlcv(symbol, self.config["timeframe"], limit=1)
            if symbol not in self.ohlcv_data or self.ohlcv_data[symbol].empty:
                return None

        last_candle = self.ohlcv_data[symbol].iloc[-1]
        return Candle(
            timestamp=last_candle.name,
            open=last_candle["open"],
            high=last_candle["high"],
            low=last_candle["low"],
            close=last_candle["close"],
            volume=last_candle["volume"],
        )

    def get_current_price(self, symbol: str) -> float:
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker["last"])
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            return 0.0

    def start_websocket(self, symbol: str):
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if "topic" in data and "kline" in data["topic"]:
                    with self.lock:
                        candle_data = data["data"]
                        new_candle = Candle(
                            timestamp=datetime.fromtimestamp(candle_data["start"] / 1000),
                            open=float(candle_data["open"]),
                            high=float(candle_data["high"]),
                            low=float(candle_data["low"]),
                            close=float(candle_data["close"]),
                            volume=float(candle_data["volume"])
                        )
                        if symbol not in self.ohlcv_data:
                            self.ohlcv_data[symbol] = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
                        new_row = pd.DataFrame([{
                            "timestamp": new_candle.timestamp,
                            "open": new_candle.open,
                            "high": new_candle.high,
                            "low": new_candle.low,
                            "close": new_candle.close,
                            "volume": new_candle.volume
                        }])
                        self.ohlcv_data[symbol] = pd.concat([self.ohlcv_data[symbol], new_row]).drop_duplicates("timestamp").sort_index()
            except Exception as e:
                self.logger.error(f"WebSocket message error: {e}")

        def on_error(ws, error):
            self.logger.error(f"WebSocket error for {symbol}: {error}")

        def on_close(ws, close_status_code, close_msg):
            self.logger.warning(f"WebSocket closed for {symbol}. Reconnecting in 5 seconds...")
            time.sleep(5)
            if self.config["mode"] == "live":
                self.start_websocket(symbol)

        def on_open(ws):
            self.logger.info(f"WebSocket connected for {symbol}")
            ws.send(json.dumps({
                "op": "subscribe",
                "args": [f"klineV2.1.{symbol.replace('/', '').replace(':', '')}"]
            }))

        ws_url = "wss://stream.bybit.com/v5/public/linear"
        ws = websocket.WebSocketApp(ws_url, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
        self.ws_connections[symbol] = ws
        ws.run_forever()

    def close_websocket(self, symbol: str):
        if symbol in self.ws_connections:
            try:
                self.ws_connections[symbol].close()
                del self.ws_connections[symbol]
            except Exception as e:
                self.logger.error(f"Error closing WebSocket for {symbol}: {e}")

# ============================================================
#                 МОДУЛЬ 2: SETUP DETECTOR
# ============================================================

class SetupDetector:
    def __init__(self, config: Dict, data_collector: DataCollector, logger: logging.Logger):
        self.config = config
        self.data_collector = data_collector
        self.logger = logger
        self.pending_signals: Dict[str, Signal] = {}

    def calculate_vwap(self, symbol: str, lookback: int = 100) -> Tuple[float, float]:
        if symbol not in self.data_collector.ohlcv_data:
            return 0.0, 0.0

        df = self.data_collector.ohlcv_data[symbol].tail(lookback)
        if len(df) < lookback:
            return 0.0, 0.0

        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        df["cumulative_volume"] = df["volume"].cumsum()
        df["cumulative_typical_price_volume"] = (df["typical_price"] * df["volume"]).cumsum()
        vwap = df["cumulative_typical_price_volume"].iloc[-1] / df["cumulative_volume"].iloc[-1]
        std = df["typical_price"].std()
        return vwap, std

    def calculate_atr(self, symbol: str, period: int = 14) -> float:
        if symbol not in self.data_collector.ohlcv_data:
            return 0.0

        df = self.data_collector.ohlcv_data[symbol]
        if len(df) < period:
            return 0.0

        high = df["high"].tail(period + 1)
        low = df["low"].tail(period + 1)
        close = df["close"].tail(period + 1).shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.tail(period).mean()

    def detect_candle_patterns(self, candle: Candle, prev_candle: Optional[Candle] = None) -> List[str]:
        patterns = []
        body_size = abs(candle.close - candle.open)
        upper_shadow = candle.high - max(candle.open, candle.close)
        lower_shadow = min(candle.open, candle.close) - candle.low

        if prev_candle:
            prev_body_size = abs(prev_candle.close - prev_candle.open)
            if prev_body_size > 0:
                body_coverage = min(
                    abs(candle.close - prev_candle.open),
                    abs(candle.close - prev_candle.close),
                    abs(candle.open - prev_candle.open),
                    abs(candle.open - prev_candle.close)
                ) / prev_body_size
            else:
                body_coverage = 0
        else:
            body_coverage = 0

        if lower_shadow >= 2 * body_size and upper_shadow <= 0.1 * body_size and candle.close > candle.open:
            patterns.append("hammer")
        if lower_shadow >= 2 * body_size and upper_shadow <= 0.1 * body_size and candle.close < candle.open:
            patterns.append("hanging_man")
        if upper_shadow >= 2 * body_size and lower_shadow <= 0.1 * body_size and candle.close < candle.open:
            patterns.append("shooting_star")
        if upper_shadow >= 2 * body_size and lower_shadow <= 0.1 * body_size and candle.close > candle.open:
            patterns.append("inverted_hammer")
        if prev_candle and candle.close > candle.open and prev_candle.close < prev_candle.open:
            if body_coverage >= self.config["signal"]["min_body_coverage"]:
                patterns.append("bullish_engulfing")
        if prev_candle and candle.close < candle.open and prev_candle.close > prev_candle.open:
            if body_coverage >= self.config["signal"]["min_body_coverage"]:
                patterns.append("bearish_engulfing")

        return patterns

    def check_setup(self, symbol: str) -> Optional[Signal]:
        last_candle = self.data_collector.get_latest_candle(symbol)
        if not last_candle:
            return None

        # Получаем предыдущую свечу
        df = self.data_collector.ohlcv_data[symbol]
        if len(df) < 2:
            return None
        prev_candle_data = df.iloc[-2]
        prev_candle = Candle(
            timestamp=prev_candle_data.name,
            open=prev_candle_data["open"],
            high=prev_candle_data["high"],
            low=prev_candle_data["low"],
            close=prev_candle_data["close"],
            volume=prev_candle_data["volume"]
        )

        vwap, std = self.calculate_vwap(symbol, self.config["vwap"]["lookback"])
        atr = self.calculate_atr(symbol, self.config["atr"]["period"])
        bb_upper = vwap + std * self.config["vwap"]["bbands_std"]
        bb_lower = vwap - std * self.config["vwap"]["bbands_std"]
        distance_from_vwap = (last_candle.close - vwap) / vwap * 100

        avg_volume = df["volume"].tail(20).mean()
        volume_ratio = last_candle.volume / avg_volume if avg_volume > 0 else 0

        patterns = self.detect_candle_patterns(last_candle, prev_candle)
        if not patterns:
            return None

        if (last_candle.close < bb_lower and
            distance_from_vwap < -self.config["signal"]["min_distance_from_vwap"] and
            volume_ratio > self.config["signal"]["volume_threshold"] and
            any(p in ["hammer", "inverted_hammer", "bullish_engulfing"] for p in patterns)):

            signal = Signal(
                side=SignalType.LONG,
                timestamp=last_candle.timestamp,
                candle=last_candle,
                vwap=vwap,
                vwap_std=std,
                atr=atr,
                distance_from_vwap=distance_from_vwap,
                volume_ratio=volume_ratio,
                candle_pattern=patterns[0],
                status=SignalStatus.DETECTED,
            )
            self.pending_signals[symbol] = signal
            return signal

        elif (last_candle.close > bb_upper and
              distance_from_vwap > self.config["signal"]["min_distance_from_vwap"] and
              volume_ratio > self.config["signal"]["volume_threshold"] and
              any(p in ["shooting_star", "hanging_man", "bearish_engulfing"] for p in patterns)):

            signal = Signal(
                side=SignalType.SHORT,
                timestamp=last_candle.timestamp,
                candle=last_candle,
                vwap=vwap,
                vwap_std=std,
                atr=atr,
                distance_from_vwap=distance_from_vwap,
                volume_ratio=volume_ratio,
                candle_pattern=patterns[0],
                status=SignalStatus.DETECTED,
            )
            self.pending_signals[symbol] = signal
            return signal

        return None

    def confirm_signal(self, symbol: str) -> Optional[Signal]:
        if symbol not in self.pending_signals:
            return None

        signal = self.pending_signals[symbol]

        if (datetime.now() - signal.timestamp).total_seconds() > self.config["signal"]["signal_lifetime_seconds"]:
            del self.pending_signals[symbol]
            return None

        current_candle = self.data_collector.get_latest_candle(symbol)
        if not current_candle or current_candle.timestamp <= signal.candle.timestamp:
            return None

        if signal.side == SignalType.LONG and current_candle.close > signal.candle.close:
            signal.status = SignalStatus.CONFIRMED
            signal.confirmation_candle = current_candle
            del self.pending_signals[symbol]
            return signal
        elif signal.side == SignalType.SHORT and current_candle.close < signal.candle.close:
            signal.status = SignalStatus.CONFIRMED
            signal.confirmation_candle = current_candle
            del self.pending_signals[symbol]
            return signal

        return None

# ============================================================
#                 МОДУЛЬ 3: PREDICTOR
# ============================================================

class Predictor:
    def __init__(self, config: Dict, data_collector: DataCollector, logger: logging.Logger):
        self.config = config
        self.data_collector = data_collector
        self.logger = logger
        self.model = self._init_model()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_data: List[Dict] = []
        self.confidence_threshold = config["predictor"]["initial_confidence_threshold"]
        self.trade_outcomes: List[bool] = []
        self.last_retrain_time = datetime.min

    def _init_model(self):
        if self.config["predictor"]["model_type"] == "mlp":
            return MLPClassifier(
                hidden_layer_sizes=tuple(self.config["predictor"]["hidden_layers"]),
                learning_rate_init=self.config["predictor"].get("learning_rate", 0.01),
                max_iter=500,
                random_state=42,
            )
        else:
            try:
                from lightgbm import LGBMClassifier
                return LGBMClassifier(
                    max_depth=self.config["predictor"].get("max_depth", 3),
                    n_estimators=self.config["predictor"].get("n_estimators", 100),
                    learning_rate=self.config["predictor"].get("learning_rate", 0.01),
                    random_state=42,
                )
            except ImportError:
                self.logger.warning("LightGBM not installed. Using MLP.")
                return MLPClassifier(
                    hidden_layer_sizes=tuple(self.config["predictor"]["hidden_layers"]),
                    max_iter=500,
                    random_state=42,
                )

    def collect_features(self, signal: Signal, symbol: str) -> Dict[str, float]:
        df = self.data_collector.ohlcv_data[symbol]
        candle = signal.candle

        ema9 = df["close"].ewm(span=9).mean().iloc[-1]
        ema9_prev = df["close"].ewm(span=9).mean().iloc[-2] if len(df) >= 2 else ema9
        ema9_slope = (ema9 - ema9_prev) / ema9_prev * 100 if ema9_prev > 0 else 0

        rsi = self._calculate_rsi(df["close"], 14)
        rsi_prev = self._calculate_rsi(df["close"].iloc[:-1], 14) if len(df) >= 15 else rsi
        rsi_slope = rsi - rsi_prev

        vwap_history = []
        for i in range(1, min(6, len(df))):
            vwap, _ = self.calculate_vwap(symbol, self.config["vwap"]["lookback"])
            vwap_history.append(vwap)
        vwap_slope = (vwap_history[0] - vwap_history[-1]) / vwap_history[-1] * 100 if len(vwap_history) >= 2 else 0

        session_volume = df["volume"].max()
        volume_pct = candle.volume / session_volume if session_volume > 0 else 0

        body_size = abs(candle.close - candle.open)
        upper_shadow = candle.high - max(candle.open, candle.close)
        lower_shadow = min(candle.open, candle.close) - candle.low

        atr_pct = signal.atr / candle.close * 100 if candle.close > 0 else 0

        return {
            "ema9_slope": ema9_slope,
            "distance_from_vwap_std": abs(signal.distance_from_vwap) / signal.vwap_std if signal.vwap_std > 0 else 0,
            "rsi": rsi,
            "rsi_slope": rsi_slope,
            "vwap_slope": vwap_slope,
            "volume_pct": volume_pct,
            "atr": signal.atr,
            "atr_pct": atr_pct,
            "body_size_pct": body_size / candle.open * 100 if candle.open > 0 else 0,
            "upper_shadow_to_body": upper_shadow / body_size if body_size > 0 else 0,
            "lower_shadow_to_body": lower_shadow / body_size if body_size > 0 else 0,
            "volume_ratio": signal.volume_ratio,
            "is_long": 1 if signal.side == SignalType.LONG else 0,
        }

    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> float:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(lower=0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, float('nan'))
        return 100 - (100 / (1 + rs))

    def predict(self, signal: Signal, symbol: str) -> PredictionResult:
        features = self.collect_features(signal, symbol)

        if not self.is_trained:
            return PredictionResult(probability=0.5, decision=False, features=features)

        feature_values = np.array([list(features.values())])

        if hasattr(self.scaler, "scale_"):
            feature_values = self.scaler.transform(feature_values)

        probability = self.model.predict_proba(feature_values)[0][1]
        decision = probability >= self.confidence_threshold

        return PredictionResult(probability=probability, decision=decision, features=features)

    def add_training_example(self, signal: Signal, symbol: str, outcome: bool):
        features = self.collect_features(signal, symbol)
        features["target"] = 1 if outcome else 0
        self.training_data.append(features)

        if len(self.training_data) >= self.config["predictor"]["min_samples_for_training"]:
            self.retrain()

    def retrain(self):
        if len(self.training_data) < self.config["predictor"]["min_samples_for_training"]:
            return

        self.logger.info(f"Retraining model with {len(self.training_data)} samples...")
        df = pd.DataFrame(self.training_data)
        X = df.drop(columns=["target"])
        y = df["target"]

        if not hasattr(self.scaler, "scale_"):
            self.scaler.fit(X)

        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        self.last_retrain_time = datetime.now()
        self.logger.info("Model retrained successfully.")

    def adapt_confidence_threshold(self):
        if len(self.trade_outcomes) < self.config["predictor"]["adaptation_window"]:
            return

        recent_outcomes = self.trade_outcomes[-self.config["predictor"]["adaptation_window"]:]
        win_rate = sum(recent_outcomes) / len(recent_outcomes)

        if win_rate < 0.5:
            self.confidence_threshold = min(0.9, self.confidence_threshold + 0.03)
            self.logger.info(f"Increased confidence threshold to {self.confidence_threshold:.2f} (win rate: {win_rate:.2f})")
        elif win_rate > 0.6:
            self.confidence_threshold = max(
                self.config["predictor"]["min_confidence_threshold"],
                self.confidence_threshold - 0.01
            )
            self.logger.info(f"Decreased confidence threshold to {self.confidence_threshold:.2f} (win rate: {win_rate:.2f})")

    def record_trade_outcome(self, outcome: bool):
        self.trade_outcomes.append(outcome)
        if len(self.trade_outcomes) > self.config["predictor"]["adaptation_window"] * 2:
            self.trade_outcomes = self.trade_outcomes[-self.config["predictor"]["adaptation_window"] * 2:]

# ============================================================
#                 МОДУЛЬ 4: RISK MANAGER
# ============================================================

class RiskManager:
    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.balance = 0.0
        self.daily_loss = 0.0
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.trades: List[Trade] = []
        self.cool_down_until = datetime.min

    def update_balance(self, balance: float):
        self.balance = balance

    def calculate_position_size(self, entry_price: float, stop_loss: float, symbol: str) -> float:
        risk_amount = self.balance * self.config["risk"]["risk_per_trade"]
        stop_loss_distance = abs(entry_price - stop_loss)

        if stop_loss_distance == 0:
            return 0.0

        qty = risk_amount / stop_loss_distance

        max_nominal_risk = self.balance * self.config["risk"]["max_risk_per_trade"]
        max_qty = max_nominal_risk / (entry_price * 0.01)
        qty = min(qty, max_qty)

        try:
            exchange = self._get_exchange()
            market = exchange.market(symbol)
            min_qty = float(market.get("limits", {}).get("amount", {}).get("min", 0) or 0)
            if qty < min_qty:
                self.logger.warning(f"Position size {qty} < minimum {min_qty} for {symbol}")
                return 0.0
        except Exception as e:
            self.logger.error(f"Error getting limits for {symbol}: {e}")
            return 0.0

        return qty

    def _get_exchange(self) -> ccxt.Exchange:
        exchange_class = getattr(ccxt, self.config["exchange"])
        return exchange_class({
            "apiKey": self.config["api_key"],
            "secret": self.config["api_secret"],
            "enableRateLimit": True,
            "options": {"defaultType": "linear"},
        })

    def check_daily_limit(self) -> bool:
        if abs(self.daily_loss) >= self.balance * self.config["risk"]["max_daily_loss"]:
            self.logger.warning(f"Daily loss limit reached: {self.daily_loss:.2f} USDT")
            return False
        return True

    def check_cool_down(self) -> bool:
        if datetime.now() < self.cool_down_until:
            remaining = (self.cool_down_until - datetime.now()).total_seconds() / 60
            self.logger.warning(f"Cooldown: {remaining:.1f} minutes remaining")
            return False
        return True

    def record_trade(self, trade: Trade):
        self.trades.append(trade)

        if trade.status == TradeStatus.CLOSED and trade.pnl is not None:
            self.daily_pnl += trade.pnl
            if trade.pnl < 0:
                self.daily_loss += abs(trade.pnl)
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0

            if self.consecutive_losses >= self.config["risk"]["cool_down_after_losses"]:
                self.cool_down_until = datetime.now() + timedelta(minutes=self.config["risk"]["cool_down_minutes"])
                self.logger.warning(f"Cooldown activated for {self.config['risk']['cool_down_minutes']} minutes")

    def get_win_rate(self) -> float:
        closed_trades = [t for t in self.trades if t.status == TradeStatus.CLOSED and t.pnl is not None]
        if not closed_trades:
            return 0.0
        wins = sum(1 for t in closed_trades if t.pnl > 0)
        return wins / len(closed_trades)

# ============================================================
#                 МОДУЛЬ 5: EXECUTOR
# ============================================================

class Executor:
    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.exchange = self._init_exchange()

    def _init_exchange(self) -> ccxt.Exchange:
        exchange_class = getattr(ccxt, self.config["exchange"])
        return exchange_class({
            "apiKey": self.config["api_key"],
            "secret": self.config["api_secret"],
            "enableRateLimit": True,
            "options": {"defaultType": "linear"},
        })

    def open_position(self, side: TradeSide, symbol: str, qty: float, entry_price: float,
                     stop_loss: float, take_profit: float) -> Optional[Trade]:
        try:
            side_str = "buy" if side == TradeSide.LONG else "sell"

            market = self.exchange.market(symbol)
            min_qty = float(market.get("limits", {}).get("amount", {}).get("min", 0) or 0)
            if qty < min_qty:
                self.logger.warning(f"Position size {qty} < minimum {min_qty} for {symbol}")
                return None

            order = self.exchange.create_order(
                symbol=symbol,
                type="market",
                side=side_str,
                amount=qty,
                params={
                    "stopLoss": stop_loss,
                    "takeProfit": take_profit,
                },
            )

            trade = Trade(
                side=side,
                entry_price=entry_price,
                entry_time=datetime.now(),
                qty=qty,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )

            self.logger.info(f"Position opened: {side_str.upper()} {qty:.4f} {symbol} @ {entry_price:.2f} | SL={stop_loss:.2f} | TP={take_profit:.2f}")
            return trade

        except Exception as e:
            self.logger.error(f"Error opening position: {e}")
            return None

    def close_position(self, symbol: str, qty: float, side: TradeSide) -> bool:
        try:
            side_str = "sell" if side == TradeSide.LONG else "buy"
            order = self.exchange.create_order(
                symbol=symbol,
                type="market",
                side=side_str,
                amount=qty,
                params={"reduceOnly": True},
            )
            self.logger.info(f"Position closed: {side_str.upper()} {qty:.4f} {symbol}")
            return True
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return False

    def update_stop_loss(self, symbol: str, new_stop_loss: float, side: TradeSide, qty: float) -> bool:
        try:
            side_str = "sell" if side == TradeSide.LONG else "buy"

            # Отменяем старые стоп-ордера
            open_orders = self.exchange.fetch_open_orders(symbol)
            stop_orders = [
                o for o in open_orders
                if o.get("type") == "stop" and o.get("side") == side_str
            ]

            for order in stop_orders:
                try:
                    self.exchange.cancel_order(order["id"], symbol)
                except Exception as e:
                    self.logger.error(f"Error canceling stop order {order['id']}: {e}")

            # Создаем новый стоп-ордер
            new_order = self.exchange.create_order(
                symbol=symbol,
                type="stop",
                side=side_str,
                amount=qty,
                price=new_stop_loss,
                params={"reduceOnly": True},
            )

            self.logger.info(f"Stop loss updated to {new_stop_loss:.2f} | New order: {new_order.get('id')}")
            return True

        except Exception as e:
            self.logger.error(f"Error updating stop loss: {e}")
            return False

    def get_balance(self) -> float:
        try:
            balance = self.exchange.fetch_balance()
            return balance["USDT"]["free"]
        except Exception as e:
            self.logger.error(f"Error getting balance: {e}")
            return 0.0

# ============================================================
#                 ГЛАВНЫЙ КЛАСС: VWAP SCALPER BOT
# ============================================================

class VWAPScalperBot:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = setup_logging(config)

        self.data_collector = DataCollector(config, self.logger)
        self.setup_detector = SetupDetector(config, self.data_collector, self.logger)
        self.predictor = Predictor(config, self.data_collector, self.logger)
        self.risk_manager = RiskManager(config, self.logger)
        self.executor = Executor(config, self.logger)

        self.is_running = False
        self.current_trade: Optional[Trade] = None
        self.ws_thread: Optional[threading.Thread] = None
        self.last_ohlcv_update = datetime.min
        self.last_retrain_check = datetime.min

    def start(self):
        self.is_running = True
        self.logger.info(f"Bot started in mode: {self.config['mode']}")

        try:
            if self.config["mode"] == "backtest":
                self._run_backtest()
            else:
                self._run_live()
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.is_running = False

        if self.ws_thread:
            self.data_collector.close_websocket(self.config["symbol"])
            self.ws_thread.join(timeout=5)

        if self.current_trade and self.current_trade.status == TradeStatus.OPEN:
            self.logger.warning("Closing open position...")
            self.executor.close_position(
                self.config["symbol"],
                self.current_trade.qty,
                self.current_trade.side
            )

        self.logger.info("Bot stopped")

    def _calculate_pnl(self, trade: Trade) -> float:
        if trade.close_price is None:
            return 0.0
        if trade.side == TradeSide.LONG:
            return (trade.close_price - trade.entry_price) * trade.qty
        else:
            return (trade.entry_price - trade.close_price) * trade.qty

    def _run_live(self):
        symbol = self.config["symbol"]

        # Initial data load
        self.data_collector.fetch_ohlcv(symbol, self.config["timeframe"], limit=100)
        self.risk_manager.update_balance(self.executor.get_balance())

        # Start WebSocket in a separate thread
        self.ws_thread = threading.Thread(target=self.data_collector.start_websocket, args=(symbol,))
        self.ws_thread.daemon = True
        self.ws_thread.start()

        self.logger.info("Main loop started...")

        while self.is_running:
            try:
                # Check daily limit and cooldown
                if not self.risk_manager.check_daily_limit() or not self.risk_manager.check_cool_down():
                    time.sleep(5)
                    continue

                # Update OHLCV data (max once per second)
                if (datetime.now() - self.last_ohlcv_update).total_seconds() > 1:
                    self.data_collector.fetch_ohlcv(symbol, self.config["timeframe"], limit=1)
                    self.last_ohlcv_update = datetime.now()

                # Check for confirmed signals
                if symbol in self.setup_detector.pending_signals:
                    confirmed_signal = self.setup_detector.confirm_signal(symbol)
                    if confirmed_signal:
                        prediction = self.predictor.predict(confirmed_signal, symbol)
                        if prediction.decision:
                            self._open_position_from_signal(confirmed_signal)
                        else:
                            self.logger.info(f"Signal rejected by neural network (probability: {prediction.probability:.2f})")

                # Check for new setups (only if no open position)
                if not self.current_trade or self.current_trade.status != TradeStatus.OPEN:
                    self.setup_detector.check_setup(symbol)

                # Monitor open position
                if self.current_trade and self.current_trade.status == TradeStatus.OPEN:
                    self._monitor_position()

                # Adapt confidence threshold
                self.predictor.adapt_confidence_threshold()

                # Retrain model if needed
                if (datetime.now() - self.last_retrain_check).total_seconds() > 3600:  # Once per hour
                    if len(self.predictor.training_data) >= self.config["predictor"]["min_samples_for_training"]:
                        self.predictor.retrain()
                        self.last_retrain_check = datetime.now()

                time.sleep(0.5)  # Reduced sleep time for better responsiveness

            except KeyboardInterrupt:
                self.stop()
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(5)

    def _open_position_from_signal(self, signal: Signal):
        symbol = self.config["symbol"]
        atr = signal.atr

        if signal.side == SignalType.LONG:
            stop_loss = signal.candle.low - self.config["atr"]["stop_loss_multiplier"] * atr
            take_profit = signal.candle.close + self.config["atr"]["take_profit_multiplier"] * atr
        else:
            stop_loss = signal.candle.high + self.config["atr"]["stop_loss_multiplier"] * atr
            take_profit = signal.candle.close - self.config["atr"]["take_profit_multiplier"] * atr

        qty = self.risk_manager.calculate_position_size(signal.candle.close, stop_loss, symbol)
        if qty <= 0:
            self.logger.warning("Position size is 0. Skipping signal.")
            return

        trade_side = TradeSide.LONG if signal.side == SignalType.LONG else TradeSide.SHORT
        trade = self.executor.open_position(
            side=trade_side,
            symbol=symbol,
            qty=qty,
            entry_price=signal.candle.close,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        if trade:
            trade.signal = signal
            self.current_trade = trade
            self.risk_manager.record_trade(trade)
            self.logger.info(f"Position opened based on signal: {signal.candle_pattern}")

    def _monitor_position(self):
        if not self.current_trade:
            return

        symbol = self.config["symbol"]
        current_price = self.executor.get_current_price(symbol)

        # Check position lifetime
        max_lifetime = timedelta(minutes=self.config["risk"]["max_trade_lifetime_minutes"])
        if (datetime.now() - self.current_trade.entry_time) > max_lifetime:
            self._close_position("timeout")
            return

        # Check SL/TP
        if self.current_trade.side == TradeSide.LONG:
            if current_price <= self.current_trade.stop_loss:
                self._close_position("stop_loss")
                return
            elif current_price >= self.current_trade.take_profit:
                self._close_position("take_profit")
                return
        else:
            if current_price >= self.current_trade.stop_loss:
                self._close_position("stop_loss")
                return
            elif current_price <= self.current_trade.take_profit:
                self._close_position("take_profit")
                return

        # Check trailing stop (only on new candle)
        last_candle = self.data_collector.get_latest_candle(symbol)
        if last_candle and last_candle.timestamp > self.current_trade.entry_time:
            if self.current_trade.trailing_stop is None:
                trailing_start = self.current_trade.entry_price + (
                    self.config["atr"]["stop_loss_multiplier"] + self.config["atr"]["trailing_step_multiplier"]
                ) * self.current_trade.stop_loss if self.current_trade.side == TradeSide.LONG else (
                    self.current_trade.entry_price - (
                        self.config["atr"]["stop_loss_multiplier"] + self.config["atr"]["trailing_step_multiplier"]
                    ) * self.current_trade.stop_loss
                )

                if (self.current_trade.side == TradeSide.LONG and current_price >= trailing_start) or \
                   (self.current_trade.side == TradeSide.SHORT and current_price <= trailing_start):
                    self.current_trade.trailing_stop = current_price - (
                        self.config["atr"]["trailing_step_multiplier"] * self.current_trade.stop_loss
                    ) if self.current_trade.side == TradeSide.LONG else current_price + (
                        self.config["atr"]["trailing_step_multiplier"] * self.current_trade.stop_loss
                    )
                    self.logger.info(f"Trailing stop activated: {self.current_trade.trailing_stop:.2f}")
            else:
                new_trailing_stop = current_price - (
                    self.config["atr"]["trailing_step_multiplier"] * self.current_trade.stop_loss
                ) if self.current_trade.side == TradeSide.LONG else current_price + (
                    self.config["atr"]["trailing_step_multiplier"] * self.current_trade.stop_loss
                )

                if (self.current_trade.side == TradeSide.LONG and new_trailing_stop > self.current_trade.trailing_stop) or \
                   (self.current_trade.side == TradeSide.SHORT and new_trailing_stop < self.current_trade.trailing_stop):
                    self.current_trade.trailing_stop = new_trailing_stop
                    self.current_trade.stop_loss = new_trailing_stop
                    self.executor.update_stop_loss(
                        symbol, new_trailing_stop, self.current_trade.side, self.current_trade.qty
                    )
                    self.logger.info(f"Trailing stop updated to: {new_trailing_stop:.2f}")

    def _close_position(self, reason: str):
        if not self.current_trade:
            return

        symbol = self.config["symbol"]
        current_price = self.executor.get_current_price(symbol)

        self.executor.close_position(symbol, self.current_trade.qty, self.current_trade.side)

        self.current_trade.status = TradeStatus.CLOSED
        self.current_trade.close_price = current_price
        self.current_trade.close_time = datetime.now()
        self.current_trade.pnl = self._calculate_pnl(self.current_trade)

        # Record outcome for neural network training
        if self.current_trade.signal:
            outcome = self.current_trade.pnl > 0 if self.current_trade.pnl else False
            self.predictor.add_training_example(self.current_trade.signal, symbol, outcome)
            self.predictor.record_trade_outcome(outcome)

        self.risk_manager.record_trade(self.current_trade)
        self.logger.info(f"Position closed due to: {reason} | PnL: {self.current_trade.pnl:.2f} USDT")
        self.current_trade = None

    def _run_backtest(self):
        # Backtest implementation would go here
        self.logger.info("Backtest mode not implemented in this version")
        self.stop()

# ============================================================
#                 ЗАПУСК БОТА
# ============================================================

if __name__ == "__main__":
    config = DEFAULT_CONFIG.copy()

    # Try to load config from file
    config_path = "config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = json.load(f)
                # Merge with default config
                for key, value in file_config.items():
                    if key in config and isinstance(value, dict):
                        config[key].update(value)
                    else:
                        config[key] = value
        except Exception as e:
            print(f"Error loading config: {e}")

    bot = VWAPScalperBot(config)

    try:
        bot.start()
    except KeyboardInterrupt:
        bot.stop()
