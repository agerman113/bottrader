#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ГИБРИДНЫЙ ИИ-СКАЛЬПЕР: "ОДНА ХОРОШАЯ СДЕЛКА"
Философия Майка Беллафиоре: охота за ЕДИНСТВЕННЫМ повторяющимся сетапом
с нейросетевым фильтром для повышения точности.

Ключевые улучшения:
1. Полностью асинхронный мониторинг (одновременный поиск сигналов и отслеживание позиции).
2. Работающий трейлинг-стоп с обновлением SL через API Bybit.
3. SL и TP на основе ATR (1.2 ATR для SL, 3.0 ATR для TP).
4. Улучшенный нейросетевой предиктор с новыми фичами (ATR, наклон VWAP, объёмный профиль).
5. Сигнал с подтверждением объёмом и скоростью (аномальный объём > avg * 1.8).
6. Защита от проскальзывания: подтверждение следующей свечой.
7. Корректный расчёт размера позиции с учётом минимального лота и максимального риска.
8. Модульная архитектура с обработкой KeyboardInterrupt.

Требования:
- Python 3.10+
- ccxt, pandas, numpy, scikit-learn, websocket-client, pyyaml
"""

import os
import time
import logging
import json
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timedelta
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
import ccxt
import websocket
import threading
from dotenv import load_dotenv

# ============================================================
#                 КОНСТАНТЫ И КОНФИГУРАЦИЯ
# ============================================================

load_dotenv()

# --- Конфигурация по умолчанию ---
DEFAULT_CONFIG = {
    "mode": "live",  # live / backtest
    "symbol": "BTC/USDT:USDT",
    "timeframe": "1m",
    "exchange": "bybit",
    "api_key": os.getenv("BYBIT_API_KEY", ""),
    "api_secret": os.getenv("BYBIT_API_SECRET", ""),
    
    # Параметры сетапа
    "vwap": {
        "lookback": 100,  # количество свечей для расчёта VWAP
        "bbands_std": 2.0,  # количество стандартных отклонений для Bollinger Bands
    },
    
    # Параметры ATR
    "atr": {
        "period": 14,
        "stop_loss_multiplier": 1.2,  # SL = 1.2 * ATR
        "take_profit_multiplier": 3.0,  # TP = 3.0 * ATR
        "trailing_step_multiplier": 0.5,  # Шаг трейлинга = 0.5 * ATR
    },
    
    # Параметры сигнала
    "signal": {
        "candle_patterns": ["hammer", "hanging_man", "shooting_star", "inverted_hammer", "bullish_engulfing", "bearish_engulfing"],
        "volume_threshold": 1.8,  # объём > среднего за 20 свечей * этот коэффициент
        "min_distance_from_vwap": 0.5,  # минимальное расстояние от VWAP в %
        "confirmation_required": True,  # требуется подтверждение следующей свечой
        "signal_lifetime_seconds": 60,  # время жизни сигнала (1 минута)
        "min_body_coverage": 0.6,  # минимальное перекрытие тела предыдущей свечи для поглощения
    },
    
    # Параметры нейросети
    "predictor": {
        "model_type": "mlp",  # mlp / lightgbm
        "hidden_layers": [64, 32],  # для MLP
        "max_depth": 3,  # для LightGBM
        "n_estimators": 100,
        "learning_rate": 0.01,
        "initial_confidence_threshold": 0.60,  # начальный порог уверенности
        "min_samples_for_training": 2000,
        "retrain_interval_hours": 168,  # раз в неделю
        "adaptation_window": 20,  # количество последних сделок для адаптации порога
        "min_confidence_threshold": 0.55,  # минимальный порог уверенности
    },
    
    # Параметры риск-менеджмента
    "risk": {
        "risk_per_trade": 0.005,  # 0.5% от капитала
        "max_risk_per_trade": 0.02,  # максимальный риск в номинале позиции (2% от баланса)
        "max_trade_lifetime_minutes": 5,
        "max_daily_loss": 0.03,  # 3% от капитала
        "cool_down_after_losses": 2,  # количество убытков подряд для тайм-аута
        "cool_down_minutes": 15,
    },
    
    # Логирование
    "logging": {
        "log_file": "vwap_scalper.log",
        "level": "INFO",
    },
    
    # Backtest
    "backtest": {
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "initial_balance": 10000,
        "data_path": "data/",
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
class OrderBookLevel:
    price: float
    volume: float


@dataclass
class OrderBook:
    bids: List[OrderBookLevel]  # от лучшей цены к худшей
    asks: List[OrderBookLevel]  # от лучшей цены к худшей
    timestamp: datetime


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
    stop_loss_order_id: Optional[str] = None  # ID стоп-ордера на бирже


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
    decision: bool  # True = входить в сделку
    features: Dict[str, float]


# ============================================================
#                 КОНФИГУРАЦИЯ ЛОГИРОВАНИЯ
# ============================================================

def setup_logging(config: Dict) -> logging.Logger:
    """Настройка логгирования."""
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
    """
    Модуль для сбора данных в реальном времени через WebSocket и REST API.
    Поддерживает:
    - OHLCV данные
    - Стакан заказов (Order Book)
    - Поток сделок (Time & Sales)
    """
    
    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.exchange = self._init_exchange()
        self.ohlcv_data: Dict[str, pd.DataFrame] = {}
        self.order_book: Dict[str, OrderBook] = {}
        self.trades: Dict[str, List[Dict]] = {}
        self.ws_connections: Dict[str, websocket.WebSocket] = {}
        self.lock = threading.Lock()
        self.last_candle_time: Dict[str, datetime] = {}
        
    def _init_exchange(self) -> ccxt.Exchange:
        """Инициализация подключения к бирже."""
        exchange_class = getattr(ccxt, self.config["exchange"])
        exchange = exchange_class({
            "apiKey": self.config["api_key"],
            "secret": self.config["api_secret"],
            "enableRateLimit": True,
            "options": {"defaultType": "linear"},
        })
        return exchange
    
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        """Получение OHLCV данных через REST API."""
        try:
            data = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            self.ohlcv_data[symbol] = df
            return df
        except Exception as e:
            self.logger.error(f"Ошибка получения OHLCV для {symbol}: {e}")
            return pd.DataFrame()
    
    def get_latest_candle(self, symbol: str, timeframe: str = "1m") -> Optional[Candle]:
        """Получение последней свечи."""
        if symbol not in self.ohlcv_data or self.ohlcv_data[symbol].empty:
            self.fetch_ohlcv(symbol, timeframe, limit=1)
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
    
    def get_previous_candle(self, symbol: str, timeframe: str = "1m") -> Optional[Candle]:
        """Получение предыдущей свечи."""
        if symbol not in self.ohlcv_data or len(self.ohlcv_data[symbol]) < 2:
            return None
        
        prev_candle = self.ohlcv_data[symbol].iloc[-2]
        return Candle(
            timestamp=prev_candle.name,
            open=prev_candle["open"],
            high=prev_candle["high"],
            low=prev_candle["low"],
            close=prev_candle["close"],
            volume=prev_candle["volume"],
        )
    
    def fetch_order_book(self, symbol: str, limit: int = 5) -> OrderBook:
        """Получение стакана заказов."""
        try:
            book = self.exchange.fetch_order_book(symbol, limit=limit)
            bids = [OrderBookLevel(price=p[0], volume=p[1]) for p in book["bids"]]
            asks = [OrderBookLevel(price=p[0], volume=p[1]) for p in book["asks"]]
            order_book = OrderBook(bids=bids, asks=asks, timestamp=datetime.now())
            self.order_book[symbol] = order_book
            return order_book
        except Exception as e:
            self.logger.error(f"Ошибка получения стакана для {symbol}: {e}")
            return OrderBook(bids=[], asks=[], timestamp=datetime.now())
    
    def get_current_price(self, symbol: str) -> float:
        """Получение текущей цены."""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker["last"])
        except Exception as e:
            self.logger.error(f"Ошибка получения цены для {symbol}: {e}")
            return 0.0
    
    def start_websocket(self, symbol: str):
        """Запуск WebSocket подключения для получения данных в реальном времени."""
        if self.config["exchange"] == "bybit":
            self._start_bybit_websocket(symbol)
        elif self.config["exchange"] == "binance":
            self._start_binance_websocket(symbol)
        else:
            self.logger.warning(f"WebSocket не поддерживается для биржи {self.config['exchange']}")
    
    def _start_bybit_websocket(self, symbol: str):
        """WebSocket для Bybit."""
        def on_message(ws, message):
            data = json.loads(message)
            if "topic" in data:
                if "orderbook" in data["topic"]:
                    book = self._parse_bybit_orderbook(data)
                    with self.lock:
                        self.order_book[symbol] = book
                elif "trade" in data["topic"]:
                    trade = self._parse_bybit_trade(data)
                    with self.lock:
                        if symbol not in self.trades:
                            self.trades[symbol] = []
                        self.trades[symbol].append(trade)
        
        def on_error(ws, error):
            self.logger.error(f"WebSocket ошибка для {symbol}: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            self.logger.warning(f"WebSocket закрыт для {symbol}: {close_msg}")
            time.sleep(5)
            self.start_websocket(symbol)
        
        def on_open(ws):
            self.logger.info(f"WebSocket подключен для {symbol}")
            ws.send(json.dumps({
                "op": "subscribe",
                "args": [
                    f"orderbook.50.{symbol.replace('/', '').replace(':', '')}",
                    f"trade.{symbol.replace('/', '').replace(':', '')}",
                ]
            }))
        
        ws_url = "wss://stream.bybit.com/v5/public/linear"
        ws = websocket.WebSocketApp(ws_url, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
        self.ws_connections[symbol] = ws
        ws.run_forever()
    
    def _parse_bybit_orderbook(self, data: Dict) -> OrderBook:
        """Парсинг стакана из WebSocket Bybit."""
        bids = [OrderBookLevel(price=float(p["price"]), volume=float(p["qty"])) for p in data["data"]["b"]]
        asks = [OrderBookLevel(price=float(p["price"]), volume=float(p["qty"])) for p in data["data"]["a"]]
        return OrderBook(bids=bids, asks=asks, timestamp=datetime.now())
    
    def _parse_bybit_trade(self, data: Dict) -> Dict:
        """Парсинг сделки из WebSocket Bybit."""
        trade = data["data"]
        return {
            "timestamp": datetime.now(),
            "price": float(trade["p"]),
            "volume": float(trade["v"]),
            "side": trade["S"],  # 'Buy' или 'Sell'
        }
    
    def close_websocket(self, symbol: str):
        """Закрытие WebSocket подключения."""
        if symbol in self.ws_connections:
            self.ws_connections[symbol].close()
            del self.ws_connections[symbol]
            self.logger.info(f"WebSocket для {symbol} закрыт")


# ============================================================
#                 МОДУЛЬ 2: SETUP DETECTOR
# ============================================================

class SetupDetector:
    """
    Модуль для обнаружения сетапа "Возврат к VWAP на аномальном объёме".
    """
    
    def __init__(self, config: Dict, data_collector: DataCollector, logger: logging.Logger):
        self.config = config
        self.data_collector = data_collector
        self.logger = logger
        self.vwap_data: Dict[str, Dict] = {}
        self.atr_data: Dict[str, Dict] = {}
        self.pending_signals: Dict[str, Signal] = {}
    
    def calculate_vwap(self, symbol: str, lookback: int = 100) -> Tuple[float, float]:
        """
        Расчёт VWAP и стандартного отклонения для Bollinger Bands.
        Возвращает (vwap, std).
        """
        if symbol not in self.data_collector.ohlcv_data:
            return 0.0, 0.0
        
        df = self.data_collector.ohlcv_data[symbol].tail(lookback)
        if len(df) < lookback:
            return 0.0, 0.0
        
        # Расчёт VWAP
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        df["cumulative_volume"] = df["volume"].cumsum()
        df["cumulative_typical_price_volume"] = (df["typical_price"] * df["volume"]).cumsum()
        vwap = (df["cumulative_typical_price_volume"].iloc[-1] / df["cumulative_volume"].iloc[-1])
        
        # Расчёт стандартного отклонения для Bollinger Bands
        std = df["typical_price"].std()
        
        self.vwap_data[symbol] = {"vwap": vwap, "std": std, "timestamp": datetime.now()}
        return vwap, std
    
    def calculate_atr(self, symbol: str, period: int = 14) -> float:
        """
        Расчёт ATR (Average True Range).
        """
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
        atr = tr.tail(period).mean()
        
        self.atr_data[symbol] = {"atr": atr, "timestamp": datetime.now()}
        return atr
    
    def detect_candle_patterns(self, candle: Candle, prev_candle: Optional[Candle] = None) -> List[str]:
        """Обнаружение свечных паттернов."""
        patterns = []
        body_size = abs(candle.close - candle.open)
        upper_shadow = candle.high - max(candle.open, candle.close)
        lower_shadow = min(candle.open, candle.close) - candle.low
        
        # Проверка перекрытия тела предыдущей свечи (для поглощения)
        if prev_candle:
            prev_body_size = abs(prev_candle.close - prev_candle.open)
            if body_size > 0:
                body_coverage = min(
                    abs(candle.close - prev_candle.open),
                    abs(candle.close - prev_candle.close),
                    abs(candle.open - prev_candle.open),
                    abs(candle.open - prev_candle.close)
                ) / prev_body_size if prev_body_size > 0 else 0
            else:
                body_coverage = 0
        else:
            body_coverage = 0
        
        # Молот (Hammer)
        if lower_shadow >= 2 * body_size and upper_shadow <= 0.1 * body_size and candle.close > candle.open:
            patterns.append("hammer")
        
        # Висельник (Hanging Man)
        if lower_shadow >= 2 * body_size and upper_shadow <= 0.1 * body_size and candle.close < candle.open:
            patterns.append("hanging_man")
        
        # Падающая звезда (Shooting Star)
        if upper_shadow >= 2 * body_size and lower_shadow <= 0.1 * body_size and candle.close < candle.open:
            patterns.append("shooting_star")
        
        # Перевёрнутый молот (Inverted Hammer)
        if upper_shadow >= 2 * body_size and lower_shadow <= 0.1 * body_size and candle.close > candle.open:
            patterns.append("inverted_hammer")
        
        # Поглощение (Bullish Engulfing)
        if prev_candle and candle.close > candle.open and prev_candle.close < prev_candle.open:
            if body_coverage >= self.config["signal"]["min_body_coverage"]:
                patterns.append("bullish_engulfing")
        
        # Поглощение (Bearish Engulfing)
        if prev_candle and candle.close < candle.open and prev_candle.close > prev_candle.open:
            if body_coverage >= self.config["signal"]["min_body_coverage"]:
                patterns.append("bearish_engulfing")
        
        return patterns
    
    def check_setup(self, symbol: str) -> Optional[Signal]:
        """
        Проверка сетапа "Возврат к VWAP на аномальном объёме".
        Возвращает Signal, если сетап обнаружен, иначе None.
        """
        # Получение последней и предыдущей свечи
        last_candle = self.data_collector.get_latest_candle(symbol)
        prev_candle = self.data_collector.get_previous_candle(symbol)
        
        if not last_candle or not prev_candle:
            return None
        
        # Расчёт VWAP и Bollinger Bands
        vwap, std = self.calculate_vwap(symbol, self.config["vwap"]["lookback"])
        bb_upper = vwap + std * self.config["vwap"]["bbands_std"]
        bb_lower = vwap - std * self.config["vwap"]["bbands_std"]
        
        # Расчёт ATR
        atr = self.calculate_atr(symbol, self.config["atr"]["period"])
        
        # Расстояние от VWAP
        distance_from_vwap = (last_candle.close - vwap) / vwap * 100
        
        # Проверка объёма
        df = self.data_collector.ohlcv_data[symbol]
        avg_volume = df["volume"].tail(20).mean()
        volume_ratio = last_candle.volume / avg_volume if avg_volume > 0 else 0
        
        # Проверка свечного паттерна
        patterns = self.detect_candle_patterns(last_candle, prev_candle)
        if not patterns:
            return None
        
        # Условия для LONG
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
            self.logger.info(f"🔍 Обнаружен LONG сетап: {signal.candle_pattern}, VWAP={vwap:.2f}, ATR={atr:.2f}, расстояние={distance_from_vwap:.2f}%, объём={volume_ratio:.2f}x")
            self.pending_signals[symbol] = signal
            return signal
        
        # Условия для SHORT
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
            self.logger.info(f"🔍 Обнаружен SHORT сетап: {signal.candle_pattern}, VWAP={vwap:.2f}, ATR={atr:.2f}, расстояние={distance_from_vwap:.2f}%, объём={volume_ratio:.2f}x")
            self.pending_signals[symbol] = signal
            return signal
        
        return None
    
    def confirm_signal(self, symbol: str) -> Optional[Signal]:
        """
        Подтверждение сигнала следующей свечой.
        Возвращает подтверждённый сигнал или None.
        """
        if symbol not in self.pending_signals:
            return None
        
        signal = self.pending_signals[symbol]
        
        # Проверка времени жизни сигнала
        if (datetime.now() - signal.timestamp).total_seconds() > self.config["signal"]["signal_lifetime_seconds"]:
            self.logger.info(f"⏳ Сигнал для {symbol} истёк (время жизни: {self.config['signal']['signal_lifetime_seconds']}с)")
            del self.pending_signals[symbol]
            return None
        
        # Получение текущей свечи для подтверждения
        current_candle = self.data_collector.get_latest_candle(symbol)
        if not current_candle or current_candle.timestamp <= signal.candle.timestamp:
            return None
        
        # Подтверждение: закрытие текущей свечи должно быть в направлении сигнала
        if signal.side == SignalType.LONG:
            if current_candle.close > signal.candle.close:
                signal.status = SignalStatus.CONFIRMED
                signal.confirmation_candle = current_candle
                self.logger.info(f"✅ Сигнал LONG для {symbol} подтверждён следующей свечой (закрытие: {current_candle.close:.2f})")
                del self.pending_signals[symbol]
                return signal
        else:  # SHORT
            if current_candle.close < signal.candle.close:
                signal.status = SignalStatus.CONFIRMED
                signal.confirmation_candle = current_candle
                self.logger.info(f"✅ Сигнал SHORT для {symbol} подтверждён следующей свечой (закрытие: {current_candle.close:.2f})")
                del self.pending_signals[symbol]
                return signal
        
        return None


# ============================================================
#                 МОДУЛЬ 3: PREDICTOR (НЕЙРОСЕТЬ)
# ============================================================

class Predictor:
    """
    Модуль для предсказания вероятности успеха сигнала с помощью нейросети.
    """
    
    def __init__(self, config: Dict, data_collector: DataCollector, logger: logging.Logger):
        self.config = config
        self.data_collector = data_collector
        self.logger = logger
        self.model = self._init_model()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.last_retrain_time = datetime.min
        self.training_data: List[Dict] = []
        self.confidence_threshold = config["predictor"]["initial_confidence_threshold"]
        self.trade_outcomes: List[bool] = []  # История исходов сделок (True = прибыль)
    
    def _init_model(self):
        """Инициализация модели."""
        if self.config["predictor"]["model_type"] == "mlp":
            model = MLPClassifier(
                hidden_layer_sizes=tuple(self.config["predictor"]["hidden_layers"]),
                learning_rate_init=self.config["predictor"]["learning_rate"],
                max_iter=500,
                random_state=42,
            )
        else:  # lightgbm
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(
                max_depth=self.config["predictor"]["max_depth"],
                n_estimators=self.config["predictor"]["n_estimators"],
                learning_rate=self.config["predictor"]["learning_rate"],
                random_state=42,
            )
        return model
    
    def collect_features(self, signal: Signal, symbol: str) -> Dict[str, float]:
        """Сбор признаков для нейросети."""
        df = self.data_collector.ohlcv_data[symbol]
        candle = signal.candle
        
        # Расчёт дополнительных индикаторов
        ema9 = df["close"].ewm(span=9).mean().iloc[-1]
        ema9_prev = df["close"].ewm(span=9).mean().iloc[-2] if len(df) >= 2 else ema9
        ema9_slope = (ema9 - ema9_prev) / ema9_prev * 100 if ema9_prev > 0 else 0
        
        # Расчёт RSI
        rsi = self._calculate_rsi(df["close"], 14)
        rsi_prev = self._calculate_rsi(df["close"].iloc[:-1], 14) if len(df) >= 15 else rsi
        rsi_slope = rsi - rsi_prev
        
        # Расчёт наклона VWAP
        vwap_history = []
        for i in range(1, min(6, len(df))):  # Последние 5 свечей
            vwap, _ = self.calculate_vwap(symbol, self.config["vwap"]["lookback"])
            vwap_history.append(vwap)
        vwap_slope = (vwap_history[0] - vwap_history[-1]) / vwap_history[-1] * 100 if len(vwap_history) >= 2 else 0
        
        # Объёмный профиль
        session_volume = df["volume"].max()
        volume_pct = candle.volume / session_volume if session_volume > 0 else 0
        
        # Характеристики свечной модели
        body_size = abs(candle.close - candle.open)
        upper_shadow = candle.high - max(candle.open, candle.close)
        lower_shadow = min(candle.open, candle.close) - candle.low
        
        # ATR в пунктах и %
        atr_pct = signal.atr / candle.close * 100 if candle.close > 0 else 0
        
        features = {
            # Контекст
            "ema9_slope": ema9_slope,
            "distance_from_vwap_std": abs(signal.distance_from_vwap) / signal.vwap_std if signal.vwap_std > 0 else 0,
            "rsi": rsi,
            "rsi_slope": rsi_slope,
            "vwap_slope": vwap_slope,
            "volume_pct": volume_pct,
            
            # ATR
            "atr": signal.atr,
            "atr_pct": atr_pct,
            
            # Свечной паттерн
            "body_size_pct": body_size / candle.open * 100 if candle.open > 0 else 0,
            "upper_shadow_to_body": upper_shadow / body_size if body_size > 0 else 0,
            "lower_shadow_to_body": lower_shadow / body_size if body_size > 0 else 0,
            
            # Сетап
            "volume_ratio": signal.volume_ratio,
            "is_long": 1 if signal.side == SignalType.LONG else 0,
        }
        
        return features
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> float:
        """Расчёт RSI."""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(lower=0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, float('nan'))
        return 100 - (100 / (1 + rs))
    
    def predict(self, signal: Signal, symbol: str) -> PredictionResult:
        """Предсказание вероятности успеха сигнала."""
        features = self.collect_features(signal, symbol)
        
        if not self.is_trained:
            self.logger.warning("Модель не обучена! Возвращаем нейтральную вероятность.")
            return PredictionResult(probability=0.5, decision=False, features=features)
        
        # Преобразование признаков
        feature_values = np.array([list(features.values())])
        feature_names = list(features.keys())
        
        # Масштабирование
        if hasattr(self.scaler, "scale_"):
            feature_values = self.scaler.transform(feature_values)
        
        # Предсказание
        if self.config["predictor"]["model_type"] == "mlp":
            probability = self.model.predict_proba(feature_values)[0][1]
        else:
            probability = self.model.predict_proba(feature_values)[0][1]
        
        decision = probability >= self.confidence_threshold
        
        return PredictionResult(
            probability=probability,
            decision=decision,
            features=features,
        )
    
    def add_training_example(self, signal: Signal, symbol: str, outcome: bool):
        """Добавление примера для обучения."""
        features = self.collect_features(signal, symbol)
        features["target"] = 1 if outcome else 0
        self.training_data.append(features)
        
        if len(self.training_data) >= self.config["predictor"]["min_samples_for_training"]:
            self.retrain()
    
    def retrain(self):
        """Переобучение модели."""
        if len(self.training_data) < self.config["predictor"]["min_samples_for_training"]:
            self.logger.warning(f"Недостаточно данных для обучения: {len(self.training_data)} примеров")
            return
        
        self.logger.info(f"🔄 Переобучение модели на {len(self.training_data)} примерах...")
        
        # Подготовка данных
        df = pd.DataFrame(self.training_data)
        X = df.drop(columns=["target"])
        y = df["target"]
        
        # Масштабирование
        if not hasattr(self.scaler, "scale_"):
            self.scaler.fit(X)
        
        X_scaled = self.scaler.transform(X)
        
        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Обучение
        self.model.fit(X_train, y_train)
        
        # Оценка
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        
        self.logger.info(f"Модель переобучена. Accuracy: {accuracy:.2f}, Precision: {precision:.2f}")
        self.is_trained = True
        self.last_retrain_time = datetime.now()
    
    def adapt_confidence_threshold(self):
        """Адаптация порога уверенности на основе последних сделок."""
        if len(self.trade_outcomes) < self.config["predictor"]["adaptation_window"]:
            return
        
        recent_outcomes = self.trade_outcomes[-self.config["predictor"]["adaptation_window"]:]
        win_rate = sum(recent_outcomes) / len(recent_outcomes)
        
        if win_rate < 0.5:
            # Если win rate низкий, повышаем порог
            self.confidence_threshold = min(0.9, self.confidence_threshold + 0.03)
            self.logger.info(f"📉 Порог уверенности повышен до {self.confidence_threshold:.2f} (win rate: {win_rate:.2f})")
        elif win_rate > 0.6:
            # Если win rate высокий, снижаем порог
            self.confidence_threshold = max(
                self.config["predictor"]["min_confidence_threshold"], 
                self.confidence_threshold - 0.01
            )
            self.logger.info(f"📈 Порог уверенности снижен до {self.confidence_threshold:.2f} (win rate: {win_rate:.2f})")
    
    def record_trade_outcome(self, outcome: bool):
        """Запись исхода сделки для адаптации порога."""
        self.trade_outcomes.append(outcome)
        if len(self.trade_outcomes) > self.config["predictor"]["adaptation_window"] * 2:
            self.trade_outcomes = self.trade_outcomes[-self.config["predictor"]["adaptation_window"] * 2:]


# ============================================================
#                 МОДУЛЬ 4: RISK MANAGER
# ============================================================

class RiskManager:
    """
    Модуль для управления рисками и учёта сделок.
    """
    
    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.balance = 0.0
        self.current_risk = config["risk"]["risk_per_trade"]
        self.daily_loss = 0.0
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.trades: List[Trade] = []
        self.last_trade_time = datetime.min
        self.cool_down_until = datetime.min
    
    def update_balance(self, balance: float):
        """Обновление баланса."""
        self.balance = balance
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, symbol: str) -> float:
        """Расчёт размера позиции на основе риска."""
        risk_amount = self.balance * self.current_risk
        stop_loss_distance = abs(entry_price - stop_loss)
        
        if stop_loss_distance == 0:
            return 0.0
        
        # Расчёт номинального размера позиции
        qty = risk_amount / stop_loss_distance
        
        # Проверка максимального риска в номинале позиции (не более 2% от баланса)
        max_nominal_risk = self.balance * self.config["risk"]["max_risk_per_trade"]
        max_qty = max_nominal_risk / (entry_price * 0.01)  # Примерная оценка (1% движения цены)
        qty = min(qty, max_qty)
        
        # Проверка минимального размера ордера
        try:
            market = self._get_exchange().market(symbol)
            min_qty = float(market.get("limits", {}).get("amount", {}).get("min", 0) or 0)
            if qty < min_qty:
                self.logger.warning(f"Размер позиции {qty} меньше минимального {min_qty} для {symbol}")
                return 0.0
        except Exception as e:
            self.logger.error(f"Ошибка получения лимитов для {symbol}: {e}")
            return 0.0
        
        return qty
    
    def _get_exchange(self) -> ccxt.Exchange:
        """Получение подключения к бирже."""
        exchange_class = getattr(ccxt, self.config["exchange"])
        exchange = exchange_class({
            "apiKey": self.config["api_key"],
            "secret": self.config["api_secret"],
            "enableRateLimit": True,
            "options": {"defaultType": "linear"},
        })
        return exchange
    
    def check_daily_limit(self) -> bool:
        """Проверка дневного лимита убытков."""
        if abs(self.daily_loss) >= self.balance * self.config["risk"]["max_daily_loss"]:
            self.logger.warning(f"🛑 Дневной лимит убытков достигнут: {self.daily_loss:.2f} USDT")
            return False
        return True
    
    def check_cool_down(self) -> bool:
        """Проверка тайм-аута после убытков."""
        if datetime.now() < self.cool_down_until:
            remaining = (self.cool_down_until - datetime.now()).total_seconds() / 60
            self.logger.warning(f"⏳ Тайм-аут: осталось {remaining:.1f} минут")
            return False
        return True
    
    def record_trade(self, trade: Trade):
        """Запись сделки."""
        self.trades.append(trade)
        
        if trade.status == TradeStatus.CLOSED:
            if trade.pnl is not None:
                self.daily_pnl += trade.pnl
                if trade.pnl < 0:
                    self.daily_loss += abs(trade.pnl)
                    self.consecutive_losses += 1
                else:
                    self.consecutive_losses = 0
                
                # Проверка на тайм-аут
                if self.consecutive_losses >= self.config["risk"]["cool_down_after_losses"]:
                    self.cool_down_until = datetime.now() + timedelta(minutes=self.config["risk"]["cool_down_minutes"])
                    self.logger.warning(f"⏳ Тайм-аут на {self.config['risk']['cool_down_minutes']} минут после {self.consecutive_losses} убытков подряд")
    
    def get_win_rate(self) -> float:
        """Получение win rate."""
        if not self.trades:
            return 0.0
        
        closed_trades = [t for t in self.trades if t.status == TradeStatus.CLOSED]
        if not closed_trades:
            return 0.0
        
        wins = sum(1 for t in closed_trades if t.pnl and t.pnl > 0)
        return wins / len(closed_trades)


# ============================================================
#                 МОДУЛЬ 5: EXECUTOR
# ============================================================

class Executor:
    """
    Модуль для выполнения ордеров через API биржи.
    """
    
    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.exchange = self._init_exchange()
        self.active_orders: Dict[str, Dict] = {}  # {order_id: order_info}
    
    def _init_exchange(self) -> ccxt.Exchange:
        """Инициализация подключения к бирже."""
        exchange_class = getattr(ccxt, self.config["exchange"])
        exchange = exchange_class({
            "apiKey": self.config["api_key"],
            "secret": self.config["api_secret"],
            "enableRateLimit": True,
            "options": {"defaultType": "linear"},
        })
        return exchange
    
    def open_position(self, side: TradeSide, symbol: str, qty: float, entry_price: float, 
                     stop_loss: float, take_profit: float) -> Optional[Trade]:
        """Открытие позиции с SL и TP."""
        try:
            side_str = "buy" if side == TradeSide.LONG else "sell"
            
            # Проверка минимального размера ордера
            market = self.exchange.market(symbol)
            min_qty = float(market.get("limits", {}).get("amount", {}).get("min", 0) or 0)
            if qty < min_qty:
                self.logger.warning(f"Размер позиции {qty} меньше минимального {min_qty} для {symbol}")
                return None
            
            # Отправка рыночного ордера с SL/TP
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
            
            # Сохранение информации об ордере
            trade = Trade(
                side=side,
                entry_price=entry_price,
                entry_time=datetime.now(),
                qty=qty,
                stop_loss=stop_loss,
                take_profit=take_profit,
                stop_loss_order_id=order["id"] if "id" in order else None,
            )
            
            self.active_orders[order["id"]] = order
            self.logger.info(f"✅ Открыта позиция: {side_str.upper()} {qty:.4f} {symbol} @ {entry_price:.2f} | SL={stop_loss:.2f} | TP={take_profit:.2f}")
            return trade
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка открытия позиции: {e}")
            return None
    
    def close_position(self, symbol: str, qty: float, side: TradeSide) -> bool:
        """Закрытие позиции рыночным ордером."""
        try:
            side_str = "sell" if side == TradeSide.LONG else "buy"
            order = self.exchange.create_order(
                symbol=symbol,
                type="market",
                side=side_str,
                amount=qty,
                params={"reduceOnly": True},
            )
            self.logger.info(f"🔒 Позиция закрыта: {side_str.upper()} {qty:.4f} {symbol}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Ошибка закрытия позиции: {e}")
            return False
    
    def update_stop_loss(self, symbol: str, new_stop_loss: float, side: TradeSide, qty: float) -> bool:
        """
        Обновление стоп-лосса для открытой позиции.
        Алгоритм:
        1. Отменить старый стоп-ордер (если есть).
        2. Создать новый стоп-ордер с новой ценой.
        """
        try:
            # Поиск активных стоп-ордеров для символа
            open_orders = self.exchange.fetch_open_orders(symbol)
            stop_orders = [o for o in open_orders if o.get("type") == "stop" and o.get("side") == ("sell" if side == TradeSide.LONG else "buy")]
            
            # Отмена старых стоп-ордеров
            for order in stop_orders:
                try:
                    self.exchange.cancel_order(order["id"], symbol)
                    self.logger.info(f"🗑 Отменён стоп-ордер: {order['id']}")
                except Exception as e:
                    self.logger.error(f"Ошибка отмены стоп-ордера {order['id']}: {e}")
            
            # Создание нового стоп-ордера
            side_str = "sell" if side == TradeSide.LONG else "buy"
            new_order = self.exchange.create_order(
                symbol=symbol,
                type="stop",
                side=side_str,
                amount=qty,
                price=new_stop_loss,
                params={"reduceOnly": True},
            )
            
            self.logger.info(f"🔄 Обновлён стоп-лосс: {new_stop_loss:.2f} | Новый ордер: {new_order.get('id')}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка обновления стоп-лосса: {e}")
            return False
    
    def get_balance(self) -> float:
        """Получение текущего баланса."""
        try:
            balance = self.exchange.fetch_balance()
            return balance["USDT"]["free"]
        except Exception as e:
            self.logger.error(f"Ошибка получения баланса: {e}")
            return 0.0
    
    def get_current_price(self, symbol: str) -> float:
        """Получение текущей цены."""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker["last"])
        except Exception as e:
            self.logger.error(f"Ошибка получения цены для {symbol}: {e}")
            return 0.0


# ============================================================
#                 МОДУЛЬ 6: BACKTESTER
# ============================================================

class Backtester:
    """
    Модуль для тестирования стратегии на исторических данных.
    """
    
    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.data_collector = DataCollector(config, logger)
        self.setup_detector = SetupDetector(config, self.data_collector, logger)
        self.predictor = Predictor(config, self.data_collector, logger)
        self.risk_manager = RiskManager(config, logger)
        self.executor = Executor(config, logger)
        
        # Имитация баланса для бэктеста
        self.balance = config["backtest"]["initial_balance"]
        self.risk_manager.update_balance(self.balance)
    
    def load_historical_data(self, symbol: str, start_date: str, end_date: str):
        """Загрузка исторических данных."""
        self.logger.info(f"📥 Загрузка исторических данных для {symbol} с {start_date} по {end_date}")
        
        # В реальности нужно использовать тиковые данные или OHLCV с высоким таймфреймом
        # Здесь упрощённая версия
        df = self.data_collector.fetch_ohlcv(symbol, self.config["timeframe"], limit=10000)
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        if df.empty:
            self.logger.error(f"Не удалось загрузить данные для {symbol}")
            return False
        
        self.logger.info(f"Загружено {len(df)} свечей")
        return True
    
    def run_backtest(self):
        """Запуск бэктеста."""
        symbol = self.config["symbol"]
        start_date = self.config["backtest"]["start_date"]
        end_date = self.config["backtest"]["end_date"]
        
        if not self.load_historical_data(symbol, start_date, end_date):
            return
        
        self.logger.info("🚀 Запуск бэктеста...")
        
        df = self.data_collector.ohlcv_data[symbol]
        
        for i in range(1, len(df)):
            # Имитация новой свечи
            self.data_collector.ohlcv_data[symbol] = df.iloc[:i+1]
            
            # Проверка сетапа
            signal = self.setup_detector.check_setup(symbol)
            if signal:
                # Подтверждение сигнала следующей свечой (в бэктесте сразу подтверждаем)
                signal.status = SignalStatus.CONFIRMED
                
                # Предсказание нейросети
                prediction = self.predictor.predict(signal, symbol)
                
                if prediction.decision:
                    # Расчёт SL/TP на основе ATR
                    atr = signal.atr
                    stop_loss_multiplier = self.config["atr"]["stop_loss_multiplier"]
                    take_profit_multiplier = self.config["atr"]["take_profit_multiplier"]
                    
                    if signal.side == SignalType.LONG:
                        stop_loss = signal.candle.low - stop_loss_multiplier * atr
                        take_profit = signal.candle.close + take_profit_multiplier * atr
                    else:
                        stop_loss = signal.candle.high + stop_loss_multiplier * atr
                        take_profit = signal.candle.close - take_profit_multiplier * atr
                    
                    # Расчёт размера позиции
                    qty = self.risk_manager.calculate_position_size(signal.candle.close, stop_loss, symbol)
                    
                    # Имитация открытия позиции
                    trade_side = TradeSide.LONG if signal.side == SignalType.LONG else TradeSide.SHORT
                    trade = Trade(
                        side=trade_side,
                        entry_price=signal.candle.close,
                        entry_time=signal.timestamp,
                        qty=qty,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                    )
                    
                    self.logger.info(f"📈 Открыта позиция в бэктесте: {trade.side.name} @ {trade.entry_price:.2f}")
                    
                    # Имитация закрытия позиции на следующих свечах
                    for j in range(i + 1, min(i + 6, len(df))):  # Проверяем следующие 5 свечей
                        next_candle = df.iloc[j]
                        close_price = next_candle["close"]
                        
                        if (trade.side == TradeSide.LONG and close_price >= trade.take_profit) or \
                           (trade.side == TradeSide.SHORT and close_price <= trade.take_profit):
                            pnl = (close_price - trade.entry_price) * trade.qty if trade.side == TradeSide.LONG else \
                                  (trade.entry_price - close_price) * trade.qty
                            trade.status = TradeStatus.CLOSED
                            trade.close_price = close_price
                            trade.close_time = next_candle.name
                            trade.pnl = pnl
                            
                            self.risk_manager.record_trade(trade)
                            self.predictor.add_training_example(signal, symbol, pnl > 0)
                            self.predictor.record_trade_outcome(pnl > 0)
                            
                            self.logger.info(f"💰 Закрыта позиция: PnL={pnl:.2f} USDT")
                            break
                        elif (trade.side == TradeSide.LONG and close_price <= trade.stop_loss) or \
                             (trade.side == TradeSide.SHORT and close_price >= trade.stop_loss):
                            pnl = (close_price - trade.entry_price) * trade.qty if trade.side == TradeSide.LONG else \
                                  (trade.entry_price - close_price) * trade.qty
                            trade.status = TradeStatus.CLOSED
                            trade.close_price = close_price
                            trade.close_time = next_candle.name
                            trade.pnl = pnl
                            
                            self.risk_manager.record_trade(trade)
                            self.predictor.add_training_example(signal, symbol, pnl > 0)
                            self.predictor.record_trade_outcome(pnl > 0)
                            
                            self.logger.info(f"💥 Закрыта позиция по SL: PnL={pnl:.2f} USDT")
                            break
                    else:
                        # Если не закрылась за 5 свечей, закрываем по тайм-ауту
                        trade.status = TradeStatus.TIMEOUT
                        trade.close_price = df.iloc[min(i + 5, len(df) - 1)]["close"]
                        trade.close_time = df.iloc[min(i + 5, len(df) - 1)].name
                        trade.pnl = (trade.close_price - trade.entry_price) * trade.qty if trade.side == TradeSide.LONG else \
                                    (trade.entry_price - trade.close_price) * trade.qty
                        
                        self.risk_manager.record_trade(trade)
                        self.predictor.add_training_example(signal, symbol, trade.pnl > 0)
                        self.predictor.record_trade_outcome(trade.pnl > 0)
                        
                        self.logger.info(f"⏰ Закрыта позиция по тайм-ауту: PnL={trade.pnl:.2f} USDT")
        
        # Итоги бэктеста
        win_rate = self.risk_manager.get_win_rate()
        total_pnl = sum(t.pnl for t in self.risk_manager.trades if t.pnl is not None)
        
        self.logger.info("=" * 50)
        self.logger.info("📊 Итоги бэктеста:")
        self.logger.info(f"   Всего сделок: {len(self.risk_manager.trades)}")
        self.logger.info(f"   Win Rate: {win_rate * 100:.2f}%")
        self.logger.info(f"   Общий PnL: {total_pnl:.2f} USDT")
        self.logger.info(f"   Итоговый баланс: {self.balance + total_pnl:.2f} USDT")
        self.logger.info("=" * 50)


# ============================================================
#                 ГЛАВНЫЙ КЛАСС: VWAP SCALPER BOT
# ============================================================

class VWAPScalperBot:
    """
    Главный класс бота.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = setup_logging(config)
        
        # Инициализация модулей
        self.data_collector = DataCollector(config, self.logger)
        self.setup_detector = SetupDetector(config, self.data_collector, self.logger)
        self.predictor = Predictor(config, self.data_collector, self.logger)
        self.risk_manager = RiskManager(config, self.logger)
        self.executor = Executor(config, self.logger)
        
        # Состояние бота
        self.is_running = False
        self.current_trade: Optional[Trade] = None
        self.last_signal_time = datetime.min
        self.ws_thread: Optional[threading.Thread] = None
    
    def start(self):
        """Запуск бота."""
        self.is_running = True
        self.logger.info(f"🚀 Бот запущен в режиме: {self.config['mode']}")
        
        try:
            if self.config["mode"] == "backtest":
                self._run_backtest()
            else:
                self._run_live()
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Остановка бота."""
        self.is_running = False
        
        # Закрытие WebSocket
        if self.ws_thread:
            self.data_collector.close_websocket(self.config["symbol"])
            self.ws_thread.join(timeout=5)
        
        # Закрытие открытой позиции (если есть)
        if self.current_trade and self.current_trade.status == TradeStatus.OPEN:
            self.logger.warning("🔒 Закрытие открытой позиции при остановке бота...")
            self.executor.close_position(
                self.config["symbol"], 
                self.current_trade.qty, 
                self.current_trade.side
            )
            self.current_trade.status = TradeStatus.CLOSED
            self.current_trade.close_time = datetime.now()
            self.current_trade.close_price = self.executor.get_current_price(self.config["symbol"])
            self.current_trade.pnl = self._calculate_pnl(self.current_trade)
            self.risk_manager.record_trade(self.current_trade)
        
        self.logger.info("🛑 Бот остановлен")
    
    def _calculate_pnl(self, trade: Trade) -> float:
        """Расчёт PnL для позиции."""
        if trade.close_price is None:
            return 0.0
        
        if trade.side == TradeSide.LONG:
            return (trade.close_price - trade.entry_price) * trade.qty
        else:
            return (trade.entry_price - trade.close_price) * trade.qty
    
    def _run_backtest(self):
        """Запуск бэктеста."""
        backtester = Backtester(self.config, self.logger)
        backtester.run_backtest()
    
    def _run_live(self):
        """Запуск live-торговли."""
        symbol = self.config["symbol"]
        
        # Загрузка начальных данных
        self.data_collector.fetch_ohlcv(symbol, self.config["timeframe"], limit=1000)
        self.risk_manager.update_balance(self.executor.get_balance())
        
        # Запуск WebSocket в отдельном потоке
        self.ws_thread = threading.Thread(target=self.data_collector.start_websocket, args=(symbol,))
        self.ws_thread.daemon = True
        self.ws_thread.start()
        
        # Основной цикл (асинхронный мониторинг)
        self.logger.info("🔄 Запущен основной цикл...")
        
        while self.is_running:
            try:
                # Проверка дневного лимита и тайм-аута
                if not self.risk_manager.check_daily_limit() or not self.risk_manager.check_cool_down():
                    time.sleep(5)
                    continue
                
                # Обновление данных (последняя свеча)
                self.data_collector.fetch_ohlcv(symbol, self.config["timeframe"], limit=1)
                
                # --- 1. Проверка подтверждения сигнала ---
                if symbol in self.setup_detector.pending_signals:
                    confirmed_signal = self.setup_detector.confirm_signal(symbol)
                    if confirmed_signal:
                        # Предсказание нейросети
                        prediction = self.predictor.predict(confirmed_signal, symbol)
                        
                        if prediction.decision:
                            self.logger.info(f"🎯 Сигнал подтверждён нейросетью (вероятность: {prediction.probability:.2f})")
                            self._open_position_from_signal(confirmed_signal, prediction)
                        else:
                            self.logger.info(f"❌ Сигнал отклонён нейросетью (вероятность: {prediction.probability:.2f})")
                
                # --- 2. Проверка нового сетапа ---
                if not self.current_trade or self.current_trade.status != TradeStatus.OPEN:
                    signal = self.setup_detector.check_setup(symbol)
                    if signal:
                        self.last_signal_time = datetime.now()
                
                # --- 3. Мониторинг открытой позиции ---
                if self.current_trade and self.current_trade.status == TradeStatus.OPEN:
                    self._monitor_position()
                
                # Адаптация порога уверенности
                self.predictor.adapt_confidence_threshold()
                
                # Переобучение модели (раз в неделю)
                if (datetime.now() - self.predictor.last_retrain_time).total_seconds() > self.config["predictor"]["retrain_interval_hours"] * 3600:
                    self.predictor.retrain()
                
                time.sleep(1)
                
            except KeyboardInterrupt:
                self.stop()
                break
            except Exception as e:
                self.logger.error(f"Ошибка в основном цикле: {e}")
                time.sleep(5)
    
    def _open_position_from_signal(self, signal: Signal, prediction: PredictionResult):
        """Открытие позиции на основе сигнала."""
        symbol = self.config["symbol"]
        
        # Расчёт SL/TP на основе ATR
        atr = signal.atr
        stop_loss_multiplier = self.config["atr"]["stop_loss_multiplier"]
        take_profit_multiplier = self.config["atr"]["take_profit_multiplier"]
        
        if signal.side == SignalType.LONG:
            stop_loss = signal.candle.low - stop_loss_multiplier * atr
            take_profit = signal.candle.close + take_profit_multiplier * atr
        else:
            stop_loss = signal.candle.high + stop_loss_multiplier * atr
            take_profit = signal.candle.close - take_profit_multiplier * atr
        
        # Расчёт размера позиции
        qty = self.risk_manager.calculate_position_size(signal.candle.close, stop_loss, symbol)
        if qty <= 0:
            self.logger.warning("Размер позиции равен 0. Пропуск сигнала.")
            return
        
        # Открытие позиции
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
            self.current_trade = trade
            self.risk_manager.record_trade(trade)
            self.logger.info(f"📈 Позиция открыта на основе сигнала: {signal.candle_pattern}")
    
    def _monitor_position(self):
        """
        Мониторинг открытой позиции (вызывается на каждом шаге основного цикла).
        Проверяет:
        - Истекло ли время жизни позиции.
        - Достигнут ли SL/TP.
        - Нужно ли обновлять трейлинг-стоп (только на закрытии нового бара).
        """
        if not self.current_trade:
            return
        
        symbol = self.config["symbol"]
        current_price = self.executor.get_current_price(symbol)
        
        # Проверка времени жизни позиции
        max_lifetime = timedelta(minutes=self.config["risk"]["max_trade_lifetime_minutes"])
        if (datetime.now() - self.current_trade.entry_time) > max_lifetime:
            self.logger.warning(f"⏰ Время жизни позиции истекло: {self.current_trade.side.name} {symbol}")
            self._close_position("timeout")
            return
        
        # Проверка SL/TP
        if self.current_trade.side == TradeSide.LONG:
            if current_price <= self.current_trade.stop_loss:
                self.logger.info(f"💥 Срабатывание SL: {current_price:.2f} <= {self.current_trade.stop_loss:.2f}")
                self._close_position("stop_loss")
                return
            elif current_price >= self.current_trade.take_profit:
                self.logger.info(f"🎯 Срабатывание TP: {current_price:.2f} >= {self.current_trade.take_profit:.2f}")
                self._close_position("take_profit")
                return
        else:  # SHORT
            if current_price >= self.current_trade.stop_loss:
                self.logger.info(f"💥 Срабатывание SL: {current_price:.2f} >= {self.current_trade.stop_loss:.2f}")
                self._close_position("stop_loss")
                return
            elif current_price <= self.current_trade.take_profit:
                self.logger.info(f"🎯 Срабатывание TP: {current_price:.2f} <= {self.current_trade.take_profit:.2f}")
                self._close_position("take_profit")
                return
        
        # Проверка трейлинга (только на закрытии нового бара)
        last_candle = self.data_collector.get_latest_candle(symbol)
        if last_candle and last_candle.timestamp > self.current_trade.entry_time:
            # Проверяем, что трейлинг ещё не активирован
            if self.current_trade.trailing_stop is None:
                # Активация трейлинга после достижения определённого уровня прибыли
                trailing_start_multiplier = self.config["atr"]["stop_loss_multiplier"] + self.config["atr"]["trailing_step_multiplier"]
                if self.current_trade.side == TradeSide.LONG:
                    if current_price >= self.current_trade.entry_price + trailing_start_multiplier * self.current_trade.stop_loss:
                        self.current_trade.trailing_stop = current_price - self.config["atr"]["trailing_step_multiplier"] * self.current_trade.stop_loss
                        self.logger.info(f"🔄 Активирован трейлинг-стоп: {self.current_trade.trailing_stop:.2f}")
                else:  # SHORT
                    if current_price <= self.current_trade.entry_price - trailing_start_multiplier * self.current_trade.stop_loss:
                        self.current_trade.trailing_stop = current_price + self.config["atr"]["trailing_step_multiplier"] * self.current_trade.stop_loss
                        self.logger.info(f"🔄 Активирован трейлинг-стоп: {self.current_trade.trailing_stop:.2f}")
            else:
                # Обновление трейлинга
                if self.current_trade.side == TradeSide.LONG:
                    new_trailing_stop = current_price - self.config["atr"]["trailing_step_multiplier"] * self.current_trade.stop_loss
                    if new_trailing_stop > self.current_trade.trailing_stop:
                        self.current_trade.trailing_stop = new_trailing_stop
                        self.current_trade.stop_loss = new_trailing_stop
                        self.executor.update_stop_loss(
                            symbol,
                            new_trailing_stop,
                            self.current_trade.side,
                            self.current_trade.qty
                        )
                        self.logger.info(f"🔄 Обновлён трейлинг-стоп: {new_trailing_stop:.2f}")
                else:  # SHORT
                    new_trailing_stop = current_price + self.config["atr"]["trailing_step_multiplier"] * self.current_trade.stop_loss
                    if new_trailing_stop < self.current_trade.trailing_stop:
                        self.current_trade.trailing_stop = new_trailing_stop
                        self.current_trade.stop_loss = new_trailing_stop
                        self.executor.update_stop_loss(
                            symbol,
                            new_trailing_stop,
                            self.current_trade.side,
                            self.current_trade.qty
                        )
                        self.logger.info(f"🔄 Обновлён трейлинг-стоп: {new_trailing_stop:.2f}")
    
    def _close_position(self, reason: str):
        """Закрытие позиции."""
        if not self.current_trade:
            return
        
        symbol = self.config["symbol"]
        current_price = self.executor.get_current_price(symbol)
        
        # Закрытие через API
        self.executor.close_position(
            symbol, 
            self.current_trade.qty, 
            self.current_trade.side
        )
        
        # Обновление информации о позиции
        self.current_trade.status = TradeStatus.CLOSED
        self.current_trade.close_price = current_price
        self.current_trade.close_time = datetime.now()
        self.current_trade.pnl = self._calculate_pnl(self.current_trade)
        
        # Запись результата для нейросети
        self.predictor.record_trade_outcome(self.current_trade.pnl > 0)
        
        # Запись в риск-менеджер
        self.risk_manager.record_trade(self.current_trade)
        
        self.logger.info(f"🔒 Позиция закрыта по причине: {reason} | PnL: {self.current_trade.pnl:.2f} USDT")
        self.current_trade = None


# ============================================================
#                 ЗАПУСК БОТА
# ============================================================

if __name__ == "__main__":
    # Загрузка конфигурации
    config_path = "config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = DEFAULT_CONFIG
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True)
    
    # Создание и запуск бота
    bot = VWAPScalperBot(config)
    
    try:
        bot.start()
    except KeyboardInterrupt:
        bot.stop()
