#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
МУЛЬТИ-КОИН СКАЛЬПЕР НА ОСНОВЕ АНАЛИЗА СТАКАНА (Order Book Strategy)
- Анализирует стаканы в реальном времени через WebSocket
- Ищет стены заявок, дисбаланс спроса/предложения, поглощение стен
- Торгует на Bybit с строгим риск-менеджментом
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
import ccxt
import websocket
import threading
from dotenv import load_dotenv

# ============================================================
#                 КОНФИГУРАЦИЯ
# ============================================================

load_dotenv()

# Список коинов для анализа
COINS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "PEPE/USDT:USDT",
    "WIF/USDT:USDT",
    "BONK/USDT:USDT",
    "DOGE/USDT:USDT",
    "SHIB/USDT:USDT"
]

DEFAULT_CONFIG = {
    "mode": "live",
    "symbols": COINS,
    "timeframe": "1m",
    "exchange": "bybit",
    "api_key": os.getenv("BYBIT_API_KEY", ""),
    "api_secret": os.getenv("BYBIT_API_SECRET", ""),

    # Параметры стратегии стакана
    "orderbook_strategy": {
        "enabled": True,
        "wall_threshold": 3.0,          # Минимальное соотношение объёма стены к медиане
        "max_wall_distance_percent": 2.0,  # Максимальное расстояние стены от лучшей цены (%)
        "imbalance_ratio": 2.0,        # Минимальное соотношение дисбаланса
        "depth": 10,                    # Глубина анализа для дисбаланса
        "sl_offset_ticks": 2,          # Отступ SL от стены (в тиках)
        "tp_ticks": 8,                 # Фиксированный TP в тиках
        "use_atr_for_tp": False,       # Использовать ATR для TP
        "tp_atr_multiplier": 2.0,      # Множитель ATR для TP
        "max_trade_lifetime_minutes": 1,  # Максимальное время удержания позиции
        "min_trade_interval_seconds": 10, # Минимальный интервал между сделками на одном символе
        "absorption_enabled": True,    # Включить детекцию поглощения стен
        "absorption_pct": 70.0,        # Процент поглощения для пробоя
        "absorption_time_seconds": 5, # Время для поглощения (секунды)
    },

    # Параметры ATR (если используется для TP)
    "atr": {
        "period": 14,
    },

    # Параметры риск-менеджмента
    "risk": {
        "risk_per_trade": 0.005,      # 0.5% от капитала
        "max_risk_per_trade": 0.02,   # Максимум 2% от капитала в позиции
        "max_daily_loss": 0.03,        # Дневной лимит убытков (3%)
        "cool_down_after_losses": 2,   # Тайм-аут после 2 убытков подряд
        "cool_down_minutes": 10,       # Длительность тайм-аута (минут)
    },

    # Логирование
    "logging": {
        "log_file": "orderbook_scalper.log",
        "level": "INFO",
    },

    # Настройки стакана (для логирования)
    "orderbook": {
        "depth": 5,                   # Глубина стакана для логирования
        "min_volume_threshold": 0.1,  # Минимальный объём для логирования заявки
    }
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

class SignalReason(Enum):
    WALL = auto()          # Стена заявок
    IMBALANCE = auto()    # Дисбаланс спроса/предложения
    ABSORPTION = auto()   # Поглощение стены

# ============================================================
#                 ДАТАКЛАССЫ
# ============================================================

@dataclass
class OrderBookLevel:
    price: float
    volume: float

@dataclass
class OrderBook:
    bids: List[OrderBookLevel]  # От лучшей цены к худшей
    asks: List[OrderBookLevel]  # От лучшей цены к худшей
    timestamp: datetime

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
    symbol: str = ""
    signal_reason: Optional[SignalReason] = None
    wall_price: Optional[float] = None  # Цена стены (для SL)
    spread: Optional[float] = None      # Спред при входе
    wall_volume: Optional[float] = None # Объём стены

@dataclass
class Signal:
    side: SignalType
    symbol: str
    entry_price: float
    stop_loss: float
    take_profit: float
    reason: SignalReason
    wall_price: float
    spread: float
    wall_volume: float
    timestamp: datetime = field(default_factory=datetime.now)

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

    logger = logging.getLogger("OrderBookScalper")
    return logger

# ============================================================
#                 DATA COLLECTOR (МНОГОВАЛЮТНЫЙ)
# ============================================================

class DataCollector:
    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.exchange = self._init_exchange()
        self.ohlcv_data: Dict[str, pd.DataFrame] = {}
        self.order_books: Dict[str, OrderBook] = {}
        self.ws_connections: Dict[str, websocket.WebSocket] = {}
        self.lock = threading.Lock()
        self.last_ohlcv_update: Dict[str, datetime] = {}
        self.last_orderbook_update: Dict[str, datetime] = {}
        self.wall_history: Dict[str, Dict[float, float]] = {}  # {symbol: {price: volume}}

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
            if (datetime.now() - self.last_ohlcv_update.get(symbol, datetime.min)).total_seconds() < 0.1:
                return self.ohlcv_data.get(symbol)

            data = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            self.ohlcv_data[symbol] = df
            self.last_ohlcv_update[symbol] = datetime.now()
            return df
        except ccxt.RateLimitExceeded:
            self.logger.warning(f"Rate limit for {symbol}. Waiting 1 second...")
            time.sleep(1)
            return self.fetch_ohlcv(symbol, timeframe, limit)
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return None

    def fetch_order_book(self, symbol: str, limit: int = 50) -> Optional[OrderBook]:
        try:
            if (datetime.now() - self.last_orderbook_update.get(symbol, datetime.min)).total_seconds() < 0.5:
                return self.order_books.get(symbol)

            book = self.exchange.fetch_order_book(symbol, limit=limit)
            bids = [OrderBookLevel(price=float(p[0]), volume=float(p[1])) for p in book["bids"]]
            asks = [OrderBookLevel(price=float(p[0]), volume=float(p[1])) for p in book["asks"]]
            order_book = OrderBook(bids=bids, asks=asks, timestamp=datetime.now())
            self.order_books[symbol] = order_book
            self.last_orderbook_update[symbol] = datetime.now()

            # Сохраняем историю стен для детекции поглощения
            if symbol not in self.wall_history:
                self.wall_history[symbol] = {}
            self.wall_history[symbol][order_book.timestamp] = {
                'bids': {level.price: level.volume for level in bids},
                'asks': {level.price: level.volume for level in asks}
            }

            return order_book
        except Exception as e:
            self.logger.error(f"Error fetching order book for {symbol}: {e}")
            return None

    def get_current_price(self, symbol: str) -> float:
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker["last"])
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            return 0.0

    def get_tick_size(self, symbol: str) -> float:
        """Получение размера тика для символа"""
        try:
            market = self.exchange.market(symbol)
            return float(market.get("precision", {}).get("price", 0.0001))
        except:
            return 0.0001  # Дефолтное значение

    def start_websockets(self):
        """Запускаем WebSocket для всех символов"""
        for symbol in self.config["symbols"]:
            self._start_websocket_for_symbol(symbol)

    def _start_websocket_for_symbol(self, symbol: str):
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if "topic" in data:
                    if "orderbook" in data["topic"]:
                        with self.lock:
                            book_data = data["data"]
                            bids = [OrderBookLevel(price=float(p["price"]), volume=float(p["qty"])) for p in book_data["b"]]
                            asks = [OrderBookLevel(price=float(p["price"]), volume=float(p["qty"])) for p in book_data["a"]]
                            order_book = OrderBook(bids=bids, asks=asks, timestamp=datetime.now())
                            self.order_books[symbol] = order_book

                            # Сохраняем историю для поглощения
                            if symbol not in self.wall_history:
                                self.wall_history[symbol] = {}
                            self.wall_history[symbol][order_book.timestamp] = {
                                'bids': {level.price: level.volume for level in bids},
                                'asks': {level.price: level.volume for level in asks}
                            }

                            # Логируем лучшие лимитные заявки
                            self._log_limit_orders(symbol, order_book)

            except Exception as e:
                self.logger.error(f"WebSocket message error for {symbol}: {e}")

        def on_error(ws, error):
            self.logger.error(f"WebSocket error for {symbol}: {error}")

        def on_close(ws, close_status_code, close_msg):
            self.logger.warning(f"WebSocket closed for {symbol}. Reconnecting in 3 seconds...")
            time.sleep(3)
            if self.config["mode"] == "live":
                self._start_websocket_for_symbol(symbol)

        def on_open(ws):
            self.logger.info(f"WebSocket connected for {symbol}")
            ws.send(json.dumps({
                "op": "subscribe",
                "args": [
                    f"orderbook.50.{symbol.replace('/', '').replace(':', '')}"
                ]
            }))

        ws_url = "wss://stream.bybit.com/v5/public/linear"
        ws = websocket.WebSocketApp(ws_url, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
        self.ws_connections[symbol] = ws
        ws.run_forever()

    def _log_limit_orders(self, symbol: str, order_book: OrderBook):
        """Логируем лучшие лимитные заявки"""
        if not order_book.bids or not order_book.asks:
            return

        mid_price = (order_book.bids[0].price + order_book.asks[0].price) / 2
        depth = self.config["orderbook"]["depth"]
        min_volume = self.config["orderbook"]["min_volume_threshold"]

        # Логируем лучшие bid заявки
        for i, bid in enumerate(order_book.bids[:depth]):
            distance = (mid_price - bid.price) / mid_price * 100
            if bid.volume >= min_volume:
                self.logger.info(
                    f"💎 BID {symbol}: Price={bid.price:.8f} | Volume={bid.volume:.2f} | "
                    f"Distance={distance:.2f}%"
                )

        # Логируем лучшие ask заявки
        for i, ask in enumerate(order_book.asks[:depth]):
            distance = (ask.price - mid_price) / mid_price * 100
            if ask.volume >= min_volume:
                self.logger.info(
                    f"💎 ASK {symbol}: Price={ask.price:.8f} | Volume={ask.volume:.2f} | "
                    f"Distance={distance:.2f}%"
                )

    def close_all_websockets(self):
        for symbol, ws in self.ws_connections.items():
            try:
                ws.close()
            except:
                pass
        self.ws_connections.clear()

    def get_wall_history(self, symbol: str, price: float, side: str) -> List[Dict]:
        """Получаем историю объёмов для конкретной цены (для детекции поглощения)"""
        if symbol not in self.wall_history:
            return []

        history = []
        for timestamp, data in self.wall_history[symbol].items():
            if side == "bid" and price in data['bids']:
                history.append({"timestamp": timestamp, "volume": data['bids'][price]})
            elif side == "ask" and price in data['asks']:
                history.append({"timestamp": timestamp, "volume": data['asks'][price]})

        return history

# ============================================================
#                 ORDER BOOK STRATEGY (НОВАЯ ЛОГИКА)
# ============================================================

class OrderBookStrategy:
    def __init__(self, config: Dict, data_collector: DataCollector, logger: logging.Logger):
        self.config = config
        self.data_collector = data_collector
        self.logger = logger
        self.last_trade_time: Dict[str, datetime] = {}  # Время последней сделки по символу
        self.wall_detection_time: Dict[str, Dict] = {}  # Время обнаружения стен {symbol: {price: timestamp}}

    def analyze(self, symbol: str, order_book: OrderBook) -> List[Signal]:
        """Анализируем стакан и возвращаем список сигналов"""
        signals = []

        if not order_book or not order_book.bids or not order_book.asks:
            return signals

        # Проверяем минимальный интервал между сделками
        if symbol in self.last_trade_time:
            min_interval = self.config["orderbook_strategy"]["min_trade_interval_seconds"]
            if (datetime.now() - self.last_trade_time[symbol]).total_seconds() < min_interval:
                return signals

        # 1. Ищем стены заявок
        wall_signals = self._detect_walls(symbol, order_book)
        signals.extend(wall_signals)

        # 2. Ищем дисбаланс
        imbalance_signal = self._detect_imbalance(symbol, order_book)
        if imbalance_signal:
            signals.append(imbalance_signal)

        # 3. Ищем поглощение стен (если включено)
        if self.config["orderbook_strategy"]["absorption_enabled"]:
            absorption_signals = self._detect_absorption(symbol, order_book)
            signals.extend(absorption_signals)

        return signals

    def _detect_walls(self, symbol: str, order_book: OrderBook) -> List[Signal]:
        """Обнаружение стен заявок"""
        signals = []
        config = self.config["orderbook_strategy"]
        tick_size = self.data_collector.get_tick_size(symbol)

        # Анализируем bid-стены
        best_bid = order_book.bids[0].price if order_book.bids else 0
        bid_volumes = [level.volume for level in order_book.bids]
        if bid_volumes:
            median_bid_volume = np.median(bid_volumes)
            for level in order_book.bids:
                if level.volume >= config["wall_threshold"] * median_bid_volume:
                    distance = (best_bid - level.price) / best_bid * 100
                    if abs(distance) <= config["max_wall_distance_percent"]:
                        # Проверяем, что это новая стена (не детектировали её раньше)
                        if symbol not in self.wall_detection_time:
                            self.wall_detection_time[symbol] = {}
                        if level.price not in self.wall_detection_time[symbol]:
                            self.wall_detection_time[symbol][level.price] = datetime.now()
                            spread = (order_book.asks[0].price - order_book.bids[0].price) / order_book.bids[0].price * 100
                            entry_price = order_book.asks[0].price  # Для LONG входим по best ask
                            stop_loss = level.price - config["sl_offset_ticks"] * tick_size
                            take_profit = self._calculate_tp(symbol, entry_price, TradeSide.LONG)

                            signal = Signal(
                                side=SignalType.LONG,
                                symbol=symbol,
                                entry_price=entry_price,
                                stop_loss=stop_loss,
                                take_profit=take_profit,
                                reason=SignalReason.WALL,
                                wall_price=level.price,
                                spread=spread,
                                wall_volume=level.volume
                            )
                            signals.append(signal)
                            self.logger.info(
                                f"🪨 BID WALL {symbol}: Price={level.price:.8f} | Volume={level.volume:.2f} | "
                                f"Distance={distance:.2f}% | Entry={entry_price:.8f} | SL={stop_loss:.8f} | TP={take_profit:.8f}"
                            )

        # Анализируем ask-стены
        best_ask = order_book.asks[0].price if order_book.asks else 0
        ask_volumes = [level.volume for level in order_book.asks]
        if ask_volumes:
            median_ask_volume = np.median(ask_volumes)
            for level in order_book.asks:
                if level.volume >= config["wall_threshold"] * median_ask_volume:
                    distance = (level.price - best_ask) / best_ask * 100
                    if abs(distance) <= config["max_wall_distance_percent"]:
                        if symbol not in self.wall_detection_time:
                            self.wall_detection_time[symbol] = {}
                        if level.price not in self.wall_detection_time[symbol]:
                            self.wall_detection_time[symbol][level.price] = datetime.now()
                            spread = (order_book.asks[0].price - order_book.bids[0].price) / order_book.bids[0].price * 100
                            entry_price = order_book.bids[0].price  # Для SHORT входим по best bid
                            stop_loss = level.price + config["sl_offset_ticks"] * tick_size
                            take_profit = self._calculate_tp(symbol, entry_price, TradeSide.SHORT)

                            signal = Signal(
                                side=SignalType.SHORT,
                                symbol=symbol,
                                entry_price=entry_price,
                                stop_loss=stop_loss,
                                take_profit=take_profit,
                                reason=SignalReason.WALL,
                                wall_price=level.price,
                                spread=spread,
                                wall_volume=level.volume
                            )
                            signals.append(signal)
                            self.logger.info(
                                f"🪨 ASK WALL {symbol}: Price={level.price:.8f} | Volume={level.volume:.2f} | "
                                f"Distance={distance:.2f}% | Entry={entry_price:.8f} | SL={stop_loss:.8f} | TP={take_profit:.8f}"
                            )

        return signals

    def _detect_imbalance(self, symbol: str, order_book: OrderBook) -> Optional[Signal]:
        """Обнаружение дисбаланса спроса/предложения"""
        config = self.config["orderbook_strategy"]
        depth = config["depth"]

        if len(order_book.bids) < depth or len(order_book.asks) < depth:
            return None

        # Суммарный объём на первых depth уровнях
        total_bid_volume = sum(level.volume for level in order_book.bids[:depth])
        total_ask_volume = sum(level.volume for level in order_book.asks[:depth])

        if total_bid_volume == 0 or total_ask_volume == 0:
            return None

        imbalance_ratio = total_bid_volume / total_ask_volume

        spread = (order_book.asks[0].price - order_book.bids[0].price) / order_book.bids[0].price * 100

        if imbalance_ratio >= config["imbalance_ratio"]:
            # Бычий сигнал (больше покупок)
            entry_price = order_book.asks[0].price
            stop_loss = order_book.bids[0].price - config["sl_offset_ticks"] * self.data_collector.get_tick_size(symbol)
            take_profit = self._calculate_tp(symbol, entry_price, TradeSide.LONG)

            signal = Signal(
                side=SignalType.LONG,
                symbol=symbol,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=SignalReason.IMBALANCE,
                wall_price=0,
                spread=spread,
                wall_volume=total_bid_volume
            )
            self.logger.info(
                f"⚖️ IMBALANCE {symbol}: Bid/Ask={imbalance_ratio:.2f} | "
                f"Entry={entry_price:.8f} | SL={stop_loss:.8f} | TP={take_profit:.8f}"
            )
            return signal

        elif imbalance_ratio <= 1 / config["imbalance_ratio"]:
            # Медвежий сигнал (больше продаж)
            entry_price = order_book.bids[0].price
            stop_loss = order_book.asks[0].price + config["sl_offset_ticks"] * self.data_collector.get_tick_size(symbol)
            take_profit = self._calculate_tp(symbol, entry_price, TradeSide.SHORT)

            signal = Signal(
                side=SignalType.SHORT,
                symbol=symbol,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=SignalReason.IMBALANCE,
                wall_price=0,
                spread=spread,
                wall_volume=total_ask_volume
            )
            self.logger.info(
                f"⚖️ IMBALANCE {symbol}: Bid/Ask={imbalance_ratio:.2f} | "
                f"Entry={entry_price:.8f} | SL={stop_loss:.8f} | TP={take_profit:.8f}"
            )
            return signal

        return None

    def _detect_absorption(self, symbol: str, order_book: OrderBook) -> List[Signal]:
        """Обнаружение поглощения стен"""
        config = self.config["orderbook_strategy"]
        signals = []
        tick_size = self.data_collector.get_tick_size(symbol)
        current_price = self.data_collector.get_current_price(symbol)

        if symbol not in self.wall_detection_time:
            return signals

        # Проверяем bid-стены на поглощение
        for wall_price, detection_time in list(self.wall_detection_time[symbol].items()):
            if (datetime.now() - detection_time).total_seconds() > config["absorption_time_seconds"]:
                del self.wall_detection_time[symbol][wall_price]
                continue

            # Получаем историю объёмов для этой стены
            history = self.data_collector.get_wall_history(symbol, wall_price, "bid")
            if len(history) < 2:
                continue

            # Проверяем уменьшение объёма
            first_volume = history[0]["volume"]
            last_volume = history[-1]["volume"]
            volume_decrease_pct = (first_volume - last_volume) / first_volume * 100

            if volume_decrease_pct >= config["absorption_pct"]:
                # Проверяем, что цена сдвинулась в сторону стены (пробой)
                if current_price > wall_price:
                    # Пробой вверх → LONG
                    entry_price = order_book.asks[0].price
                    stop_loss = wall_price - config["sl_offset_ticks"] * tick_size
                    take_profit = self._calculate_tp(symbol, entry_price, TradeSide.LONG)

                    signal = Signal(
                        side=SignalType.LONG,
                        symbol=symbol,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        reason=SignalReason.ABSORPTION,
                        wall_price=wall_price,
                        spread=(order_book.asks[0].price - order_book.bids[0].price) / order_book.bids[0].price * 100,
                        wall_volume=last_volume
                    )
                    signals.append(signal)
                    del self.wall_detection_time[symbol][wall_price]  # Удаляем, чтобы не детектировать повторно
                    self.logger.info(
                        f"💥 ABSORPTION {symbol}: Wall at {wall_price:.8f} absorbed ({volume_decrease_pct:.1f}%) | "
                        f"Entry={entry_price:.8f} | SL={stop_loss:.8f} | TP={take_profit:.8f}"
                    )

        # Проверяем ask-стены на поглощение
        for wall_price, detection_time in list(self.wall_detection_time[symbol].items()):
            if (datetime.now() - detection_time).total_seconds() > config["absorption_time_seconds"]:
                del self.wall_detection_time[symbol][wall_price]
                continue

            history = self.data_collector.get_wall_history(symbol, wall_price, "ask")
            if len(history) < 2:
                continue

            first_volume = history[0]["volume"]
            last_volume = history[-1]["volume"]
            volume_decrease_pct = (first_volume - last_volume) / first_volume * 100

            if volume_decrease_pct >= config["absorption_pct"]:
                if current_price < wall_price:
                    # Пробой вниз → SHORT
                    entry_price = order_book.bids[0].price
                    stop_loss = wall_price + config["sl_offset_ticks"] * tick_size
                    take_profit = self._calculate_tp(symbol, entry_price, TradeSide.SHORT)

                    signal = Signal(
                        side=SignalType.SHORT,
                        symbol=symbol,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        reason=SignalReason.ABSORPTION,
                        wall_price=wall_price,
                        spread=(order_book.asks[0].price - order_book.bids[0].price) / order_book.bids[0].price * 100,
                        wall_volume=last_volume
                    )
                    signals.append(signal)
                    del self.wall_detection_time[symbol][wall_price]
                    self.logger.info(
                        f"💥 ABSORPTION {symbol}: Wall at {wall_price:.8f} absorbed ({volume_decrease_pct:.1f}%) | "
                        f"Entry={entry_price:.8f} | SL={stop_loss:.8f} | TP={take_profit:.8f}"
                    )

        return signals

    def _calculate_tp(self, symbol: str, entry_price: float, side: TradeSide) -> float:
        """Расчёт тейк-профита"""
        config = self.config["orderbook_strategy"]
        tick_size = self.data_collector.get_tick_size(symbol)

        if config["use_atr_for_tp"]:
            atr = self._calculate_atr(symbol)
            if side == TradeSide.LONG:
                return entry_price + config["tp_atr_multiplier"] * atr
            else:
                return entry_price - config["tp_atr_multiplier"] * atr
        else:
            if side == TradeSide.LONG:
                return entry_price + config["tp_ticks"] * tick_size
            else:
                return entry_price - config["tp_ticks"] * tick_size

    def _calculate_atr(self, symbol: str, period: int = 14) -> float:
        """Расчёт ATR для символа"""
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

    def record_trade(self, symbol: str):
        """Записываем время последней сделки по символу"""
        self.last_trade_time[symbol] = datetime.now()

# ============================================================
#                 RISK MANAGER
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
        self.active_positions: Dict[str, Trade] = {}  # symbol -> Trade

    def update_balance(self, balance: float):
        self.balance = balance

    def calculate_position_size(self, entry_price: float, stop_loss: float, symbol: str) -> float:
        risk_amount = self.balance * self.config["risk"]["risk_per_trade"]
        stop_loss_distance = abs(entry_price - stop_loss)

        if stop_loss_distance == 0:
            return 0.0

        qty = risk_amount / stop_loss_distance

        # Ограничиваем максимальный размер позиции
        max_nominal_risk = self.balance * self.config["risk"]["max_risk_per_trade"]
        max_qty = max_nominal_risk / (entry_price * 0.01)  # Примерная оценка
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
            self.logger.warning(f"🛑 Daily loss limit reached: {self.daily_loss:.2f} USDT")
            return False
        return True

    def check_cool_down(self) -> bool:
        if datetime.now() < self.cool_down_until:
            remaining = (self.cool_down_until - datetime.now()).total_seconds() / 60
            self.logger.warning(f"⏳ Cooldown: {remaining:.1f} minutes remaining")
            return False
        return True

    def can_open_position(self, symbol: str) -> bool:
        """Проверяем, можно ли открывать позицию для этого символа"""
        if symbol in self.active_positions:
            return False
        return self.check_daily_limit() and self.check_cool_down()

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
                self.logger.warning(f"⏳ Cooldown activated for {self.config['risk']['cool_down_minutes']} minutes")

    def get_win_rate(self) -> float:
        closed_trades = [t for t in self.trades if t.status == TradeStatus.CLOSED and t.pnl is not None]
        if not closed_trades:
            return 0.0
        wins = sum(1 for t in closed_trades if t.pnl > 0)
        return wins / len(closed_trades)

# ============================================================
#                 EXECUTOR
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
                     stop_loss: float, take_profit: float, signal_reason: SignalReason,
                     wall_price: float, spread: float, wall_volume: float) -> Optional[Trade]:
        try:
            side_str = "buy" if side == TradeSide.LONG else "sell"

            market = self.exchange.market(symbol)
            min_qty = float(market.get("limits", {}).get("amount", {}).get("min", 0) or 0)
            if qty < min_qty:
                self.logger.warning(f"⚠️ Position size {qty} < minimum {min_qty} for {symbol}")
                return None

            # Открываем позицию с SL/TP
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
                symbol=symbol,
                signal_reason=signal_reason,
                wall_price=wall_price,
                spread=spread,
                wall_volume=wall_volume,
            )

            self.logger.info(
                f"🚀 OPENED {side_str.upper()} {qty:.4f} {symbol} @ {entry_price:.8f} | "
                f"SL={stop_loss:.8f} | TP={take_profit:.8f} | "
                f"Reason: {signal_reason.name} | Wall: {wall_price:.8f} ({wall_volume:.2f})"
            )
            return trade

        except Exception as e:
            self.logger.error(f"❌ Error opening position for {symbol}: {e}")
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
            self.logger.info(f"🔒 CLOSED {side_str.upper()} {qty:.4f} {symbol}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error closing position for {symbol}: {e}")
            return False

    def get_balance(self) -> float:
        try:
            balance = self.exchange.fetch_balance()
            return balance["USDT"]["free"]
        except Exception as e:
            self.logger.error(f"Error getting balance: {e}")
            return 0.0

    def get_current_price(self, symbol: str) -> float:
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker["last"])
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            return 0.0

# ============================================================
#                 ГЛАВНЫЙ КЛАСС
# ============================================================

class OrderBookScalperBot:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = setup_logging(config)

        self.data_collector = DataCollector(config, self.logger)
        self.orderbook_strategy = OrderBookStrategy(config, self.data_collector, self.logger)
        self.risk_manager = RiskManager(config, self.logger)
        self.executor = Executor(config, self.logger)

        self.is_running = False
        self.ws_thread: Optional[threading.Thread] = None
        self.last_orderbook_check = datetime.min

    def start(self):
        self.is_running = True
        self.logger.info("🚀 ORDER BOOK SCALPER STARTED")
        self.logger.info(f"📊 Monitoring symbols: {', '.join(self.config['symbols'])}")
        self.logger.info(f"🎯 Strategy: Order Book Analysis (Walls, Imbalance, Absorption)")

        try:
            self._run_live()
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.is_running = False
        self.logger.info("🛑 Stopping bot...")

        if self.ws_thread:
            self.data_collector.close_all_websockets()
            self.ws_thread.join(timeout=5)

        # Закрываем все открытые позиции
        for symbol, trade in list(self.risk_manager.active_positions.items()):
            if trade.status == TradeStatus.OPEN:
                self.logger.warning(f"🔒 Closing open position for {symbol}...")
                self.executor.close_position(symbol, trade.qty, trade.side)
                trade.status = TradeStatus.CLOSED
                trade.close_time = datetime.now()
                trade.close_price = self.executor.get_current_price(symbol)
                trade.pnl = self._calculate_pnl(trade)
                self.risk_manager.record_trade(trade)

        self.logger.info("✅ Bot stopped")

    def _calculate_pnl(self, trade: Trade) -> float:
        if trade.close_price is None:
            return 0.0
        if trade.side == TradeSide.LONG:
            return (trade.close_price - trade.entry_price) * trade.qty
        else:
            return (trade.entry_price - trade.close_price) * trade.qty

    def _run_live(self):
        # Получаем начальный баланс
        self.risk_manager.update_balance(self.executor.get_balance())
        self.logger.info(f"💰 Current balance: {self.risk_manager.balance:.2f} USDT")

        # Запускаем WebSocket для всех символов
        self.ws_thread = threading.Thread(target=self.data_collector.start_websockets)
        self.ws_thread.daemon = True
        self.ws_thread.start()

        # Даём время на подключение WebSocket
        time.sleep(3)

        self.logger.info("🔄 Main loop started...")

        while self.is_running:
            try:
                # Проверяем дневной лимит и кулдаун
                if not self.risk_manager.check_daily_limit() or not self.risk_manager.check_cool_down():
                    time.sleep(5)
                    continue

                # Анализируем стаканы для всех символов
                for symbol in self.config["symbols"]:
                    if symbol in self.data_collector.order_books:
                        order_book = self.data_collector.order_books[symbol]
                        signals = self.orderbook_strategy.analyze(symbol, order_book)

                        for signal in signals:
                            self._process_signal(signal)

                # Мониторинг открытых позиций
                self._monitor_positions()

                time.sleep(0.1)  # Минимальная задержка для обработки

            except KeyboardInterrupt:
                self.stop()
                break
            except Exception as e:
                self.logger.error(f"❌ Error in main loop: {e}")
                time.sleep(5)

    def _process_signal(self, signal: Signal):
        """Обрабатываем сигнал от OrderBookStrategy"""
        symbol = signal.symbol

        # Проверяем, можно ли открывать позицию
        if not self.risk_manager.can_open_position(symbol):
            self.logger.warning(f"⚠️ Cannot open position for {symbol} (daily limit or cooldown)")
            return

        # Рассчитываем размер позиции
        qty = self.risk_manager.calculate_position_size(
            signal.entry_price,
            signal.stop_loss,
            symbol
        )
        if qty <= 0:
            self.logger.warning(f"⚠️ Position size is 0 for {symbol}. Skipping.")
            return

        # Открываем позицию
        trade_side = TradeSide.LONG if signal.side == SignalType.LONG else TradeSide.SHORT
        trade = self.executor.open_position(
            side=trade_side,
            symbol=symbol,
            qty=qty,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            signal_reason=signal.reason,
            wall_price=signal.wall_price,
            spread=signal.spread,
            wall_volume=signal.wall_volume
        )

        if trade:
            self.risk_manager.active_positions[symbol] = trade
            self.risk_manager.record_trade(trade)
            self.orderbook_strategy.record_trade(symbol)  # Записываем время последней сделки

    def _monitor_positions(self):
        """Мониторинг всех открытых позиций"""
        for symbol, trade in list(self.risk_manager.active_positions.items()):
            if trade.status != TradeStatus.OPEN:
                continue

            current_price = self.executor.get_current_price(symbol)
            if current_price == 0:
                continue

            # Проверяем время жизни позиции
            max_lifetime = timedelta(minutes=self.config["orderbook_strategy"]["max_trade_lifetime_minutes"])
            if (datetime.now() - trade.entry_time) > max_lifetime:
                self._close_position(trade, "timeout")
                continue

            # Проверяем SL/TP
            if trade.side == TradeSide.LONG:
                if current_price <= trade.stop_loss:
                    self._close_position(trade, "stop_loss")
                    continue
                elif current_price >= trade.take_profit:
                    self._close_position(trade, "take_profit")
                    continue
            else:  # SHORT
                if current_price >= trade.stop_loss:
                    self._close_position(trade, "stop_loss")
                    continue
                elif current_price <= trade.take_profit:
                    self._close_position(trade, "take_profit")
                    continue

    def _close_position(self, trade: Trade, reason: str):
        """Закрываем позицию"""
        symbol = trade.symbol
        current_price = self.executor.get_current_price(symbol)

        success = self.executor.close_position(symbol, trade.qty, trade.side)
        if not success:
            self.logger.error(f"❌ Failed to close position for {symbol}")
            return

        trade.status = TradeStatus.CLOSED
        trade.close_price = current_price
        trade.close_time = datetime.now()
        trade.pnl = self._calculate_pnl(trade)

        self.risk_manager.record_trade(trade)
        del self.risk_manager.active_positions[symbol]

        self.logger.info(
            f"💰 CLOSED {trade.side.name} {symbol} | "
            f"Entry={trade.entry_price:.8f} | Exit={current_price:.8f} | "
            f"PnL={trade.pnl:.2f} USDT | Reason: {reason} | "
            f"Signal: {trade.signal_reason.name}"
        )

# ============================================================
#                 ЗАПУСК
# ============================================================

if __name__ == "__main__":
    # Проверяем наличие API ключей
    if not os.getenv("BYBIT_API_KEY") or not os.getenv("BYBIT_API_SECRET"):
        print("❌ ERROR: BYBIT_API_KEY and BYBIT_API_SECRET must be set in environment variables")
        exit(1)

    config = DEFAULT_CONFIG.copy()

    # Пробуем загрузить конфиг из файла
    config_path = "orderbook_scalper_config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = json.load(f)
                # Объединяем конфиги
                for key, value in file_config.items():
                    if key in config and isinstance(value, dict):
                        config[key].update(value)
                    else:
                        config[key] = value
        except Exception as e:
            print(f"⚠️ Warning: Error loading config file: {e}. Using defaults.")

    # Создаём и запускаем бота
    bot = OrderBookScalperBot(config)

    try:
        bot.start()
    except KeyboardInterrupt:
        bot.stop()
