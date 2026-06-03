#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
Bybit ГИБРИДНЫЙ ПРОФЕССИОНАЛЬНЫЙ БОТ — v22 "SEMANTIC QUANTUM MATRIX CORE"
================================================================================
Версия: 22.0 Ultimate
Дата выпуска: 03.06.2026

ОПИСАНИЕ:
Полноценный квантовый фьючерсный бот для биржи Bybit, объединяющий:
- Лучшие практики из книг Чан, Винс, Пардо, Лопес де Прадо
- Теханализ (15+ индикаторов)
- Квантовый анализ (Mean Reversion, Momentum, Коинтеграция)
- Микроструктуру рынка (Order Flow, Volume Profile, Order Book)
- Машинное обучение (Random Forest, Gradient Boosting)
- Портфельный подход и риск‑менеджмент
- НОВОВВЕДЕНИЯ v22:
  1️⃣ Матрично‑Квантовый Скоринг (MQS) с цветными смайликами  
  2️⃣ Семантическое ядро (7 методов анализа)  
  3️⃣ Теория игр (Nash Equilibrium)  
  4️⃣ Цепи Маркова (Markov Chains)  
  5️⃣ Теория хаоса (Hurst Exponent)  
  6️⃣ Динамическая монетка (Coin Flip → Weighted Signal)  
  7️⃣ Байесовская вероятность
"""

# ============================================================
# 📦 Библиотеки и импорт
# ============================================================
import os
import sys
import time
import json
import logging
import warnings
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import stats as scipy_stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from pybit.unified_trading import HTTP, WebSocket
import math

warnings.filterwarnings('ignore')

# ============================================================
# ⚙️ Константы и настройки
# ============================================================
# --- Режим работы ---
TESTNET_MODE = True  # True – Testnet, False – Mainnet

# --- Трейдинг‑пары ---
SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT", "XRP/USDT:USDT",
    "SOL/USDT:USDT", "ADA/USDT:USDT", "TRX/USDT:USDT", "TON/USDT:USDT",
    "AVAX/USDT:USDT", "DOT/USDT:USDT", "LTC/USDT:USDT", "BCH/USDT:USDT",
    "ATOM/USDT:USDT", "XLM/USDT:USDT", "NEAR/USDT:USDT", "DOGE/USDT:USDT",
    "PEPE/USDT:USDT", "WIF/USDT:USDT", "BOME/USDT:USDT", "FET/USDT:USDT",
    "RENDER/USDT:USDT", "TAO/USDT:USDT", "WLD/USDT:USDT", "ARKM/USDT:USDT",
    "IO/USDT:USDT", "ONDO/USDT:USDT", "VIRTUAL/USDT:USDT", "UNI/USDT:USDT",
    "AAVE/USDT:USDT", "ARB/USDT:USDT", "OP/USDT:USDT", "LINK/USDT:USDT",
    "GRT/USDT:USDT", "INJ/USDT:USDT", "SUI/USDT:USDT", "APT/USDT:USDT",
    "TIA/USDT:USDT", "JTO/USDT:USDT", "EIGEN/USDT:USDT", "HBAR/USDT:USDT",
    "VET/USDT:USDT", "NOT/USDT:USDT", "CATI/USDT:USDT",
]

# --- Параметры квант‑анализа ---
QUANT_ENABLED = True
COINTEGRATION_PAIRS = [
    ("BTC/USDT:USDT", "ETH/USDT:USDT"),
    ("BTC/USDT:USDT", "BNB/USDT:USDT"),
    ("ETH/USDT:USDT", "SOL/USDT:USDT"),
]
COINTEGRATION_WINDOW = 100
MEAN_REVERSION_THRESHOLD = 2.0
MOMENTUM_WINDOW = 20

# --- Параметры Order Flow ---
ORDER_FLOW_ENABLED = True
ORDER_BOOK_DEPTH = 20
VOLUME_PROFILE_ENABLED = True
VOLUME_PROFILE_BARS = 50
CLUSTER_TOLERANCE = 0.005

# --- Параметры машинного обучения ---
ML_ENABLED = True
ML_MODEL_TYPE = "RandomForest"
ML_FEATURES_WINDOW = 30
ML_RETRAIN_INTERVAL = 100
ML_MIN_SAMPLES = 50

# --- Основные параметры ---
LEVERAGE = 3
TIMEFRAME_TA = "5m"
TIMEFRAME_TREND = "1h"
TIMEFRAME_MID = "15m"
TIMEFRAME_4H = "4h"
SCAN_INTERVAL = 300
MIN_SCORE = 65      # классический порог
MQS_MIN_SCORE = 25  # порог входа MQS

# --- MQS (Матрично‑Квантовый Скоринг) ---
MQS_WEIGHTS = {
    "classic_ta": 0.50,   # было 0.25
    "decision_matrix": 0.15,
    "nash_equilibrium": 0.10,
    "markov_chain": 0.05,
    "hurst_exponent": 0.05,
    "coin_flip": 0.05,
    "bayesian": 0.10,
}
MQS_EMOJIS = {
    (90, 100): "🟢",  # сильный сигнал
    (70, 89):  "🟡",  # хороший сигнал
    (50, 69):  "🟠",  # нейтральный
    (30, 49):  "🔴",  # слабый сигнал
    (0,  29):  "⚫",  # плохой сигнал
}
MQS_DESCRIPTIONS = {
    (90, 100): "СИЛЬНЫЙ СИГНАЛ (ВХОДИМ!)",
    (70, 89):  "ХОРОШИЙ СИГНАЛ (МОЖНО ВХОДИТЬ)",
    (50, 69):  "НЕЙТРАЛЬНЫЙ (ЖДЁМ ЛУЧШЕГО)",
    (30, 49):  "СЛАБЫЙ СИГНАЛ (РИСКОВАННО)",
    (0,  29):  "ПЛОХОЙ СИГНАЛ (ПРОПУСКАЕМ)",
}

# --- Тейк‑профит / Стоп‑лосс ---
TP_PERCENT = 3.0
SL_PERCENT = 1.0
MIN_SL_PERCENT = 0.8
MAX_SL_PERCENT = 2.0
ATR_SL_MULT = 1.5
ATR_TP_MULT = 3.0

# --- Трейлинг‑стоп ---
PARTIAL_BE_ENABLED = True
PARTIAL_BE_PROFIT = 0.2
TRAILING_ATR_PERIOD = 14
TRAILING_ATR_MULT = 2.0
TRAILING_OFFSET_MULT = 1.5
MIN_TRAILING_STEP = 0.4
MIN_TRAILING_OFFSET = 0.6
MIN_PROFIT_FOR_TRAIL = 1.0
RR_EXIT_TRIGGER = 0.6

# --- Риск‑менеджмент ---
BASE_RISK_PCT = 0.8
MAX_RISK_PCT = 1.2
USE_ADVANCED_RISK = True
MIN_TRADES_FOR_F = 20
MAX_RISK_PERCENT_F = 2.5

# --- Портфельная оптимизация ---
PORTFOLIO_OPTIMIZATION = True
MAX_PORTFOLIO_RISK = 0.05
CORRELATION_THRESHOLD = 0.8

# --- Фильтры ---
SESSION_FILTER_ENABLED = False
SESSION_BLOCK_START = 0
SESSION_BLOCK_END = 4
DAILY_LOSS_LIMIT_PCT = 3.0
DAILY_LOSS_PAUSE_SEC = 10800
VOLUME_SPIKE_MULT = 3.5
VOLUME_AVG_PERIOD = 20
SIGNAL_EXIT_ENABLED = True
ENTRY_CONFIRM_BARS = 1
ENTRY_CONFIRM_MIN_SCORE = 60
SYMBOL_BLOCK_AFTER_TP = 90
SYMBOL_BLOCK_AFTER_SL = 180
SL_STREAK_LIMIT = 2
SL_STREAK_PAUSE = 3600
SL_STREAK_EXTRA_PAUSE = 300
MIN_BALANCE = 5.0
MAX_DRAWDOWN_PCT = 15.0
TRADE_MAX_LIFETIME = 7200
REPORT_INTERVAL = 1800

# --- Monte‑Carlo симуляции ---
MONTE_CARLO_ENABLED = True
MONTE_CARLO_SIMULATIONS = 1000
MONTE_CARLO_DAYS = 30

# --- Экспресс‑режим ---
EXPRESS_MODE = True
EXPRESS_MIN_SCORE = 50
EXPRESS_MQS_MIN = 52
EXPRESS_MAX_MARGIN_PCT = 0.5

# --- Разрешение конфликтов индикаторов ---
CONFLICT_RESOLUTION_ENABLED = True

# --- Файлы ---
STATE_FILE = "state_bot_v22.json"
TRADES_FILE = "trades_bot_v22.json"
INDICATOR_STATS_FILE = "indicator_stats_v22.json"
METRICS_FILE = "strategy_metrics_v22.json"
ML_MODEL_FILE = "ml_model_v22.pkl"
PORTFOLIO_STATE_FILE = "portfolio_state_v22.json"

BYBIT_FEE = 0.00055

# --- S/R уровни ---
SR_PERIOD = 100
SR_PROXIMITY_PCT = 0.5
SR_MIN_TOUCHES = 3
SR_CLUSTER_TOL = 0.005
SR_BLOCK_DIST_PCT = 0.3

# --- Параметры для тик‑данных ---
TICK_DATA_ENABLED = False
TICK_DATA_WINDOW = 1000

# --- Новые параметры Bybit API ---
FUNDING_RATE_ENABLED = True
FUNDING_RATE_THRESHOLD = 0.1
OPEN_INTEREST_ENABLED = True
OI_GROWTH_THRESHOLD = 5.0
LIQUIDATIONS_ENABLED = True
LIQUIDATION_THRESHOLD_USD = 1_000_000
INSURANCE_FUND_ENABLED = True
MARK_PRICE_CHECK_ENABLED = True
MARK_PRICE_DIFF_THRESHOLD = 0.5 if TESTNET_MODE else 0.1
TAKER_VOLUME_ENABLED = True
TAKER_BUY_SELL_RATIO_THRESHOLD = 1.5

# --- MA‑кроссовер ---
MA_CROSSOVER_ENABLED = True
MA1_TYPE = "EMA"
MA2_TYPE = "EMA"
MA1_LENGTH = 21
MA2_LENGTH = 50
MA_TIMEFRAME = "5m"

# ------------------------------------------------------------
# 📜 Логирование
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]  %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot_v22_ultimate.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ============================================================
# 🧩 Обёртка API Bybit
# ============================================================
class BybitWrapper:
    """Обёртка для Unified Trading API."""

    def __init__(self, testnet: bool, api_key: str, api_secret: str):
        self.session = HTTP(
            testnet=testnet,
            api_key=api_key,
            api_secret=api_secret,
        )

    # ------------------------------------------------------------------
    # Основные запросы
    # ------------------------------------------------------------------
    def fetch_balance(self, params=None):
        r = self.session.get_wallet_balance(accountType="UNIFIED")
        account = r["result"]["list"][0]
        total = float(account.get("totalWalletBalance") or 0)
        free = float(account.get("totalAvailableBalance") or 0)
        return {"USDT": {"free": free, "total": total}}

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200):
        sym = symbol.replace("/", "").replace(":USDT", "")
        tf_map = {"1m": "1", "3m": "3", "5m": "5", "15m": "15",
                  "1h": "60", "4h": "240"}
        interval = tf_map.get(timeframe, "5")
        r = self.session.get_kline(category="linear", symbol=sym,
                                  interval=interval, limit=limit)
        rows = list(reversed(r["result"]["list"]))
        return [[int(x[0]), float(x[1]), float(x[2]), float(x[3]),
                 float(x[4]), float(x[5])] for x in rows]

    def fetch_ticker(self, symbol: str):
        sym = symbol.replace("/", "").replace(":USDT", "")
        r = self.session.get_tickers(category="linear", symbol=sym)
        last = float(r["result"]["list"][0]["lastPrice"])
        mark_price = float(r["result"]["list"][0].get("markPrice", last))
        return {"last": last, "mark_price": mark_price}

    def fetch_order_book(self, symbol: str, limit: int = 20):
        sym = symbol.replace("/", "").replace(":USDT", "")
        r = self.session.get_orderbook(category="linear", symbol=sym,
                                      limit=limit)
        bids = [[float(x[0]), float(x[1])] for x in r["result"]["b"]]
        asks = [[float(x[0]), float(x[1])] for x in r["result"]["a"]]
        return {"bids": bids, "asks": asks}

    def fetch_positions(self, symbols=None):
        r = self.session.get_positions(category="linear", settleCoin="USDT")
        positions = []
        for p in r["result"]["list"]:
            size = float(p.get("size", 0))
            if size == 0:
                continue
            positions.append({
                "symbol": p["symbol"],
                "side": "long" if p["side"] == "Buy" else "short",
                "contracts": size,
                "unrealizedPnl": float(p.get("unrealisedPnl", 0)),
                "entryPrice": float(p.get("avgPrice", 0)),
            })
        return positions

    def set_leverage(self, leverage: int, symbol: str, params=None):
        sym = symbol.replace("/", "").replace(":USDT", "")
        try:
            self.session.set_leverage(
                category="linear", symbol=sym,
                buyLeverage=str(leverage), sellLeverage=str(leverage)
            )
            return True
        except Exception as e:
            log.warning(f"Ошибка установки плеча (метод 1): {e}")
            return False

    def create_market_order(self, symbol: str, side: str,
                           qty: float, params=None):
        params = params or {}
        sym = symbol.replace("/", "").replace(":USDT", "")
        order_side = "Buy" if side == "buy" else "Sell"
        kwargs = dict(
            category="linear", symbol=sym,
            side=order_side, orderType="Market",
            qty=str(qty), timeInForce="GTC",
        )
        if params.get("reduceOnly"):
            kwargs["reduceOnly"] = True
        if params.get("takeProfit"):
            kwargs["takeProfit"] = str(params["takeProfit"])
        if params.get("stopLoss"):
            kwargs["stopLoss"] = str(params["stopLoss"])
        r = self.session.place_order(**kwargs)
        order_id = r["result"]["orderId"]
        time.sleep(1)
        try:
            hist = self.session.get_order_history(
                category="linear", symbol=sym, orderId=order_id
            )
            avg_price = float(hist["result"]["list"][0].get("avgPrice", 0) or 0)
        except Exception:
            avg_price = 0
        return {"average": avg_price, "id": order_id}

    def price_to_precision(self, symbol: str, price: float):
        """Округление цены до 2‑х знаков."""
        return str(round(price, 2))

    # --------------------------------------------------------------
    # Методы, связанные с точностью количества и управлением позицией
    # --------------------------------------------------------------
    def amount_to_precision(self, symbol: str, amount: float) -> float:
        """Возвращает количество, округлённое в соответствии с step‑size."""
        sym = symbol.replace("/", "").replace(":USDT", "")
        info = INSTRUMENTS.get(sym,
                {"minOrderQty": 0.001, "qtyStep": 0.001})
        step = info["qtyStep"]
        min_qty = info["minOrderQty"]
        qty = math.floor(amount / step) * step
        qty = round(qty, 10)
        if qty < min_qty:
            log.warning(
                f"qty={qty} < minOrderQty={min_qty} для {sym} — возвращаем минимум")
            return min_qty
        return qty

    def private_post_v5_position_trading_stop(self, params: dict):
        """Устанавливает stop‑loss через v5‑endpoint."""
        self.session.set_trading_stop(
            category="linear",
            symbol=params["symbol"],
            stopLoss=params.get("stopLoss", ""),
            slTriggerBy=params.get("slTriggerBy", "MarkPrice"),
            positionIdx=0,
        )

    def private_post_v5_position_set_leverage(self, params: dict):
        """Устанавливает плечо через v5‑endpoint."""
        try:
            self.session.set_leverage(
                category="linear",
                symbol=params["symbol"],
                buyLeverage=params.get("buyLeverage", "3"),
                sellLeverage=params.get("sellLeverage", "3"),
            )
        except Exception as e:
            log.warning(f"Ошибка установки плеча (v5): {e}")

    # ------------------------------------------------------------------
    # Дополнительные запросы (Open‑Interest, Funding, Liquidations, Taker‑Volume)
    # ------------------------------------------------------------------
    def fetch_open_interest(self, symbol: str) -> float:
        sym = symbol.replace("/", "").replace(":USDT", "")
        try:
            r = self.session.get_open_interest(category="linear", symbol=sym)
            return float(r["result"]["list"][0].get("openInterest", 0))
        except Exception as e:
            log.debug(f"Ошибка получения Open Interest для {symbol}: {e}")
            return 0.0

    def fetch_funding_rate(self, symbol: str) -> Dict[str, Any]:
        sym = symbol.replace("/", "").replace(":USDT", "")
        try:
            r = self.session.get_funding_rate(category="linear", symbol=sym)
            data = r["result"]["list"][0]
            return {
                "fundingRate": float(data.get("fundingRate", 0)),
                "nextFundingTime": data.get("nextFundingTime", ""),
                "valid": True,
            }
        except Exception as e:
            log.debug(f"Ошибка получения Funding Rate для {symbol}: {e}")
            return {"fundingRate": 0.0, "nextFundingTime": "", "valid": False}

    def fetch_liquidations(self, symbol: str,
                           limit: int = 50) -> List[Dict[str, Any]]:
        sym = symbol.replace("/", "").replace(":USDT", "")
        try:
            r = self.session.get_liquidation(
                category="linear", symbol=sym, limit=limit)
            return r["result"]["list"]
        except Exception as e:
            log.debug(f"Ошибка получения ликвидаций для {symbol}: {e}")
            return []

    def fetch_taker_volume(self, symbol: str) -> Dict[str, Any]:
        sym = symbol.replace("/", "").replace(":USDT", "")
        try:
            r = self.session.get_taker_volume(category="linear", symbol=sym)
            data = r["result"]["list"][0]
            return {
                "buyVol": float(data.get("buyVol", 0)),
                "sellVol": float(data.get("sellVol", 0)),
                "valid": True,
            }
        except Exception as e:
            log.debug(f"Ошибка получения Taker Volume для {symbol}: {e}")
            return {"buyVol": 0.0, "sellVol": 0.0, "valid": False}


# ============================================================
# 🚀 Инициализация API‑клиента
# ============================================================
if TESTNET_MODE:
    exchange = BybitWrapper(
        testnet=True,
        api_key=os.getenv("BYBIT_TESTNET_API_KEY"),
        api_secret=os.getenv("BYBIT_TESTNET_API_SECRET"),
    )
    log.info("✅ Режим TESTNET активирован (pybit)")
else:
    exchange = BybitWrapper(
        testnet=False,
        api_key=os.getenv("BYBIT_API_KEY"),
        api_secret=os.getenv("BYBIT_API_SECRET"),
    )
    log.info("✅ Режим MAINNET активирован (pybit)")

# ------------------------------------------------------------
# 📂 Загрузка информации об инструментах
# ------------------------------------------------------------
def загрузить_инструменты() -> dict:
    """Получает step‑size и минимальные ордеры для всех символов."""
    instruments = {}
    try:
        r = exchange.session.get_instruments_info(category="linear")
        for item in r["result"]["list"]:
            sym = item["symbol"]
            lot = item.get("lotSizeFilter", {})
            instruments[sym] = {
                "minOrderQty": float(lot.get("minOrderQty", 0.001)),
                "qtyStep":     float(lot.get("qtyStep", 0.001)),
            }
        log.info(f"✅ Загружено {len(instruments)} инструментов (minOrderQty/qtyStep)")
    except Exception as e:
        log.warning(f"Не удалось загрузить инструменты: {e}")
    return instruments

INSTRUMENTS = загрузить_инструменты()

# ============================================================
# 🧠 Функции MQS и вспомогательные расчёты
# ============================================================
def get_open_interest_data(symbol: str,
                           window: int = 24) -> Dict[str, Any]:
    """Open‑Interest + цена‑изменения."""
    try:
        oi_now = exchange.fetch_open_interest(symbol)
        if oi_now == 0:
            return {"oi_value": 0, "oi_change_pct": 0,
                    "signal": "neutral", "valid": False}
        ticker = exchange.fetch_ticker(symbol)
        price_now = ticker["last"]
        price_past = exchange.fetch_ohlcv(symbol, "1h",
                                         limit=window)[-window][4]
        price_change_pct = (price_now - price_past) / price_past * 100
        oi_change_pct = 5.0  # упрощённо
        if oi_change_pct > OI_GROWTH_THRESHOLD and price_change_pct > 0:
            signal = "bullish"
        elif oi_change_pct > OI_GROWTH_THRESHOLD and price_change_pct < 0:
            signal = "bearish"
        else:
            signal = "neutral"
        return {"oi_value": oi_now, "oi_change_pct": oi_change_pct,
                "signal": signal, "valid": True}
    except Exception as e:
        log.debug(f"Ошибка Open Interest для {symbol}: {e}")
        return {"oi_value": 0, "oi_change_pct": 0,
                "signal": "neutral", "valid": False}


def get_funding_rate_data(symbol: str) -> Dict[str, Any]:
    """Funding Rate + экстремум."""
    try:
        data = exchange.fetch_funding_rate(symbol)
        if not data["valid"]:
            return {"funding_rate": 0, "signal": "neutral",
                    "extreme": False, "valid": False}
        funding_rate = data["fundingRate"] * 100
        extreme = abs(funding_rate) > FUNDING_RATE_THRESHOLD
        if funding_rate > FUNDING_RATE_THRESHOLD:
            signal = "bearish"
        elif funding_rate < -FUNDING_RATE_THRESHOLD:
            signal = "bullish"
        else:
            signal = "neutral"
        return {"funding_rate": funding_rate, "signal": signal,
                "extreme": extreme, "valid": True}
    except Exception as e:
        log.debug(f"Ошибка Funding Rate для {symbol}: {e}")
        return {"funding_rate": 0, "signal": "neutral",
                "extreme": False, "valid": False}


def get_liquidations_data(symbol: str) -> Dict[str, Any]:
    """Суммарные ликвидации в USD + сигнал."""
    try:
        liquidations = exchange.fetch_liquidations(symbol, limit=50)
        if not liquidations:
            return {"total_liquidations_usd": 0,
                    "long_liquidations_usd": 0,
                    "short_liquidations_usd": 0,
                    "signal": "neutral", "valid": False}
        total_long = total_short = 0.0
        for liq in liquidations:
            side = liq.get("side", "")
            size = float(liq.get("size", 0))
            price = float(liq.get("price", 0))
            volume_usd = size * price
            if side == "Buy":
                total_short += volume_usd
            elif side == "Sell":
                total_long += volume_usd
        total = total_long + total_short
        if total > LIQUIDATION_THRESHOLD_USD:
            if total_long > total_short * 1.5:
                signal = "bearish"
            elif total_short > total_long * 1.5:
                signal = "bullish"
            else:
                signal = "neutral"
        else:
            signal = "neutral"
        return {"total_liquidations_usd": total,
                "long_liquidations_usd": total_long,
                "short_liquidations_usd": total_short,
                "signal": signal, "valid": True}
    except Exception as e:
        log.debug(f"Ошибка Liquidations для {symbol}: {e}")
        return {"total_liquidations_usd": 0,
                "long_liquidations_usd": 0,
                "short_liquidations_usd": 0,
                "signal": "neutral", "valid": False}


def get_mark_price_diff(symbol: str) -> Dict[str, Any]:
    """Проверка отклонения Mark‑Price от Last‑Price."""
    try:
        ticker = exchange.fetch_ticker(symbol)
        last_price = ticker["last"]
        mark_price = ticker.get("mark_price", last_price)
        if last_price == 0:
            return {"last_price": 0, "mark_price": 0,
                    "diff_pct": 0, "safe": False, "valid": False}
        diff_pct = abs(mark_price - last_price) / last_price * 100
        safe = diff_pct < MARK_PRICE_DIFF_THRESHOLD
        return {"last_price": last_price, "mark_price": mark_price,
                "diff_pct": diff_pct, "safe": safe, "valid": True}
    except Exception as e:
        log.debug(f"Ошибка Mark Price Diff для {symbol}: {e}")
        return {"last_price": 0, "mark_price": 0,
                "diff_pct": 0, "safe": False, "valid": False}


def get_taker_volume_data(symbol: str) -> Dict[str, Any]:
    """Taker Volume + отношение покупок к продажам."""
    try:
        data = exchange.fetch_taker_volume(symbol)
        if not data["valid"]:
            return {"buy_vol": 0, "sell_vol": 0,
                    "ratio": 1, "signal": "neutral", "valid": False}
        buy_vol = data["buyVol"]
        sell_vol = data["sellVol"]
        ratio = buy_vol / (sell_vol + 1e-10)
        if ratio > TAKER_BUY_SELL_RATIO_THRESHOLD:
            signal = "bullish"
        elif ratio < 1 / TAKER_BUY_SELL_RATIO_THRESHOLD:
            signal = "bearish"
        else:
            signal = "neutral"
        return {"buy_vol": buy_vol, "sell_vol": sell_vol,
                "ratio": ratio, "signal": signal, "valid": True}
    except Exception as e:
        log.debug(f"Ошибка Taker Volume для {symbol}: {e}")
        return {"buy_vol": 0, "sell_vol": 0,
                "ratio": 1, "signal": "neutral", "valid": False}


def get_bybit_advanced_signals(symbol: str) -> Dict[str, Any]:
    """Агрегация всех advanced‑сигналов."""
    signals = {
        "open_interest": get_open_interest_data(symbol),
        "funding_rate": get_funding_rate_data(symbol),
        "liquidations": get_liquidations_data(symbol),
        "mark_price_diff": get_mark_price_diff(symbol),
        "taker_volume": get_taker_volume_data(symbol),
    }
    advanced_score = 0
    if signals["open_interest"]["valid"]:
        if signals["open_interest"]["signal"] == "bullish":
            advanced_score += 10
        elif signals["open_interest"]["signal"] == "bearish":
            advanced_score -= 10
    if signals["funding_rate"]["valid"]:
        if signals["funding_rate"]["extreme"]:
            if signals["funding_rate"]["signal"] == "bullish":
                advanced_score += 15
            else:
                advanced_score -= 15
        else:
            if signals["funding_rate"]["signal"] == "bullish":
                advanced_score += 5
            elif signals["funding_rate"]["signal"] == "bearish":
                advanced_score -= 5
    if signals["liquidations"]["valid"]:
        if signals["liquidations"]["signal"] == "bullish":
            advanced_score += 10
        elif signals["liquidations"]["signal"] == "bearish":
            advanced_score -= 10
    if signals["mark_price_diff"]["valid"] and not signals["mark_price_diff"]["safe"]:
        advanced_score -= 20
    if signals["taker_volume"]["valid"]:
        if signals["taker_volume"]["signal"] == "bullish":
            advanced_score += 10
        elif signals["taker_volume"]["signal"] == "bearish":
            advanced_score -= 10
    return {"advanced_score": max(-50, min(50, advanced_score)),
            "details": signals}


# ------------------------------------------------------------
# 📈 Технические индикаторы (поддержка pandas)
# ------------------------------------------------------------
def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rma(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(alpha=1 / span, adjust=False).mean()


def _sma(s: pd.Series, span: int) -> pd.Series:
    return s.rolling(span).mean()


def _wma(s: pd.Series, span: int) -> pd.Series:
    weights = np.arange(1, span + 1, dtype=float)
    return s.rolling(span).apply(lambda x: np.dot(x, weights) / weights.sum(),
                               raw=True)


def _hma(s: pd.Series, span: int) -> pd.Series:
    half = max(1, span // 2)
    sqrt_p = max(1, int(np.sqrt(span)))
    return _wma(2 * _wma(s, half) - _wma(s, span), sqrt_p)


def calc_ma(df: pd.DataFrame, ma_type: str, length: int) -> pd.Series:
    s = df["c"]
    ma_type = ma_type.upper()
    if ma_type == "EMA":
        return _ema(s, length)
    elif ma_type == "SMA":
        return _sma(s, length)
    elif ma_type == "WMA":
        return _wma(s, length)
    elif ma_type == "HMA":
        return _hma(s, length)
    else:
        return _ema(s, length)


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0)
    loss = (-d).clip(lower=0)
    avg_g = _rma(gain, period)
    avg_l = _rma(loss, period)
    rs = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(close: pd.Series,
              fast: int = 12, slow: int = 26, signal: int = 9):
    ml = _ema(close, fast) - _ema(close, slow)
    sl = _ema(ml, signal)
    return ml, sl, ml - sl


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hi, lo, pc = df["h"], df["l"], df["c"].shift(1)
    tr = pd.concat([hi - lo,
                    (hi - pc).abs(),
                    (lo - pc).abs()], axis=1).max(axis=1)
    return _rma(tr, period)


def calc_supertrend(df: pd.DataFrame,
                    period: int = 10, mult: float = 3.0):
    atr = calc_atr(df, period)
    hl2 = (df["h"] + df["l"]) / 2
    ub = (hl2 + mult * atr).copy()
    lb = (hl2 - mult * atr).copy()
    trend = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        c, pc = df["c"].iloc[i], df["c"].iloc[i - 1]
        pu, pl, pt = ub.iloc[i - 1], lb.iloc[i - 1], trend.iloc[i - 1]
        ub.iloc[i] = ub.iloc[i] if ub.iloc[i] < pu or pc > pu else pu
        lb.iloc[i] = lb.iloc[i] if lb.iloc[i] > pl or pc < pl else pl
        if pt == 1 and c < lb.iloc[i]:
            trend.iloc[i] = -1
        elif pt == -1 and c > ub.iloc[i]:
            trend.iloc[i] = 1
        else:
            trend.iloc[i] = pt
    return trend == 1, trend == -1


def calc_stochastic(df: pd.DataFrame,
                    k: int = 14, d: int = 3, smooth: int = 3):
    lo = df["l"].rolling(k).min()
    hi = df["h"].rolling(k).max()
    ks = (100 * (df["c"] - lo) / (hi - lo + 1e-10)).rolling(smooth).mean()
    return ks, ks.rolling(d).mean()


def calc_hull(close: pd.Series, period: int = 55):
    hma = _ema(2 * _ema(close, period // 2) - _ema(close, period),
                int(np.sqrt(period)))
    return hma > hma.shift(2), hma < hma.shift(2)


def calc_adx(df: pd.DataFrame, period: int = 14):
    atr = calc_atr(df, period)
    pdm = (df["h"] - df["h"].shift(1)).clip(lower=0)
    mdm = (df["l"].shift(1) - df["l"]).clip(lower=0)
    pdm = pdm.where(pdm >= mdm, 0)
    mdm = mdm.where(mdm >= pdm, 0)
    pdi = 100 * _rma(pdm, period) / atr.replace(0, np.nan)
    mdi = 100 * _rma(mdm, period) / atr.replace(0, np.nan)
    adx = _rma(100 * (pdi - mdi).abs() / (pdi + mdi + 1e-10), period)
    return adx, pdi, mdi


def calc_range_filter(df: pd.DataFrame, period: int = 100,
                     qty: float = 2.5):
    close = df["c"]
    rng = qty * calc_atr(df, period)
    filt = close.copy()
    for i in range(1, len(close)):
        c, r, pf = close.iloc[i], rng.iloc[i], filt.iloc[i - 1]
        if c - r > pf:
            filt.iloc[i] = c - r
        elif c + r < pf:
            filt.iloc[i] = c + r
        else:
            filt.iloc[i] = pf
    up = (filt > filt.shift(1)) & (close > filt)
    down = (filt < filt.shift(1)) & (close < filt)
    return filt, filt + rng, filt - rng, up, down


def calc_support_resistance(df: pd.DataFrame,
                            period: int = SR_PERIOD) -> dict:
    """Определение уровней поддержки/сопротивления."""
    df_sr = df.tail(period).reset_index(drop=True)
    highs = df_sr["h"].values
    lows = df_sr["l"].values
    close = float(df["c"].iloc[-1])
    raw_res, raw_sup = [], []

    for i in range(2, len(highs) - 2):
        if (highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and
                highs[i] > highs[i + 1] and highs[i] > highs[i + 2]):
            raw_res.append(highs[i])
        if (lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and
                lows[i] < lows[i + 1] and lows[i] < lows[i + 2]):
            raw_sup.append(lows[i])

    def _cluster(levels):
        if not levels:
            return []
        levels = sorted(levels)
        out = []
        cur = [levels[0]]
        for lvl in levels[1:]:
            if (lvl - cur[0]) / (cur[0] + 1e-10) < SR_CLUSTER_TOL:
                cur.append(lvl)
            else:
                out.append((float(np.mean(cur)), len(cur)))
                cur = [lvl]
        out.append((float(np.mean(cur)), len(cur)))
        return out

    res_cl = _cluster(raw_res)
    sup_cl = _cluster(raw_sup)
    res_above = sorted([(p, n) for p, n in res_cl if p > close],
                       key=lambda x: x[0])
    sup_below = sorted([(p, n) for p, n in sup_cl if p < close],
                       key=lambda x: x[0], reverse=True)

    nearest_res, res_n = res_above[0] if res_above else (close * 1.05, 0)
    nearest_sup, sup_n = sup_below[0] if sup_below else (close * 0.95, 0)
    dist_res = (nearest_res - close) / close * 100
    dist_sup = (close - nearest_sup) / close * 100
    near_sup = dist_sup < SR_PROXIMITY_PCT and sup_n >= SR_MIN_TOUCHES
    near_res = dist_res < SR_PROXIMITY_PCT and res_n >= SR_MIN_TOUCHES

    return {
        "support": round(nearest_sup, 10),
        "resistance": round(nearest_res, 10),
        "dist_to_sup_pct": round(dist_sup, 2),
        "dist_to_res_pct": round(dist_res, 2),
        "sup_cluster": sup_n,
        "res_cluster": res_n,
        "near_support": near_sup,
        "near_resistance": near_res,
    }

# ------------------------------------------------------------
# 📊 Оценочные функции (Decision Matrix, Nash, Markov, Hurst, Weighted)
# ------------------------------------------------------------
def decision_matrix_score(details: dict, side: str) -> float:
    """Взвешенный скор по набору индикаторов."""
    weights = {
        "rsi": 0.15,
        "macd": 0.12,
        "supertrend": 0.10,
        "funding_rate": 0.15,
        "liquidations": 0.10,
        "open_interest": 0.08,
        "taker_volume": 0.05,
    }

    signals = {
        "rsi": 1 if details.get("rsi", 50) < 40 else (
            -1 if details.get("rsi", 50) > 60 else 0),
        "macd": 1 if details.get("macd") == "бычий" else (
            -1 if details.get("macd") == "медвежий" else 0),
        "supertrend": 1 if details.get("supertrend") == "вверх" else (
            -1 if details.get("supertrend") == "вниз" else 0),
        "funding_rate": -1 if details.get("funding_rate", {}).get("signal") == "bearish"
                        else (1 if details.get("funding_rate", {}).get("signal") == "bullish"
                              else 0),
        "liquidations": 1 if details.get("liquidations", {}).get("signal") == "bullish"
                        else (-1 if details.get("liquidations", {}).get("signal") == "bearish"
                              else 0),
        "open_interest": 1 if details.get("open_interest", {}).get("signal") == "bullish"
                         else (-1 if details.get("open_interest", {}).get("signal") == "bearish"
                               else 0),
        "taker_volume": 1 if details.get("taker_volume", {}).get("signal") == "bullish"
                        else (-1 if details.get("taker_volume", {}).get("signal") == "bearish"
                              else 0),
    }

    score = 0.0
    for indicator, weight in weights.items():
        signal = signals.get(indicator, 0)
        if side == "short":
            signal *= -1
        score += signal * weight
    return score


def nash_equilibrium_score(symbol: str, side: str) -> float:
    """Теория игр — Nash Equilibrium."""
    ai_data = получить_bybit_ai(symbol)
    long_ratio = ai_data.get("long_ratio", 0.5)
    if side == "long":
        if long_ratio > 0.7:
            return -1.0
        elif long_ratio < 0.3:
            return 1.0
        else:
            return 0.0
    else:
        if long_ratio > 0.7:
            return 1.0
        elif long_ratio < 0.3:
            return -1.0
        else:
            return 0.0


class MarkovChain:
    """Простая цепочка Маркова."""

    def __init__(self):
        self.states = ["bullish", "bearish", "neutral"]
        self.transition_matrix = np.array([
            [0.7, 0.2, 0.1],   # из bullish
            [0.1, 0.7, 0.2],   # из bearish
            [0.3, 0.3, 0.4],   # из neutral
        ])
        self.current_state = "neutral"

    def update_state(self, new_state: str):
        if new_state in self.states:
            self.current_state = new_state

    def predict_next_state(self) -> str:
        """Детерминированный вывод — берём состояние с максимальной вероятностью."""
        idx = self.states.index(self.current_state)
        probabilities = self.transition_matrix[idx]
        next_state_idx = int(np.argmax(probabilities))
        return self.states[next_state_idx]

    def get_score(self, side: str) -> float:
        next_state = self.predict_next_state()
        if side == "long":
            return 1.0 if next_state == "bullish" else (
                -1.0 if next_state == "bearish" else 0.0)
        else:
            return 1.0 if next_state == "bearish" else (
                -1.0 if next_state == "bullish" else 0.0)


markov_model = MarkovChain()


def update_markov_state(details: dict):
    """Обновление состояния модели Маркова."""
    rsi = details.get("rsi", 50)
    macd = details.get("macd", "нейтральный")
    if rsi < 30 and macd == "бычий":
        markov_model.update_state("bullish")
    elif rsi > 70 and macd == "медвежий":
        markov_model.update_state("bearish")
    else:
        markov_model.update_state("neutral")


def markov_chain_score(details: dict, side: str) -> float:
    """Счёт по цепочке Маркова."""
    update_markov_state(details)
    return markov_model.get_score(side)


def calculate_hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
    """Оценка индекса Хёрста."""
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(series[lag:].values,
                               series[:-lag].values)) for lag in lags]
    tau = np.array(tau)
    n = np.array(lags)
    hurst = np.polyfit(np.log(n), np.log(tau), 1)[0]
    return hurst


def hurst_signal(symbol: str, timeframe: str = "1h", limit: int = 100) -> str:
    """Сигнал Хёрста (trend / mean_reverting / neutral)."""
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    if len(ohlcv) < limit:
        return "neutral"
    df = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c", "v"])
    close = df["c"]
    hurst = calculate_hurst_exponent(close)
    if hurst > 0.6:
        return "trend"
    elif hurst < 0.4:
        return "mean_reverting"
    else:
        return "neutral"


def hurst_exponent_score(details: dict, side: str, symbol: str) -> float:
    """Счёт по Хёрсту."""
    hurst = hurst_signal(symbol)
    if hurst == "trend":
        if side == "long" and тренд_4h_бычий(symbol):
            return 1.0
        elif side == "short" and тренд_4h_медвежий(symbol):
            return 1.0
        else:
            return -0.5
    elif hurst == "mean_reverting":
        if side == "long" and details.get("rsi", 50) < 30:
            return 1.0
        elif side == "short" and details.get("rsi", 50) > 70:
            return 1.0
        else:
            return -0.5
    else:
        return 0.0


def weighted_signal_score(symbol: str, details: dict, side: str) -> float:
    """
    Детерминированная взвешенная оценка сигналов.
    Заменяет `dynamic_coin_flip` — убирает случайность.
    Возвращает значение от -1 до +1.
    """
    score = 0.0

    # RSI (вес 0.30)
    rsi = details.get("rsi", 50)
    if rsi < 25:
        score += 0.30
    elif rsi < 35:
        score += 0.20
    elif rsi < 45:
        score += 0.10
    elif rsi > 75:
        score -= 0.30
    elif rsi > 65:
        score -= 0.15

    # MACD (вес 0.25)
    macd_signal = details.get("macd", "нейтральный")
    if macd_signal == "бычий":
        score += 0.25
    elif macd_signal == "медвежий":
        score -= 0.25

    # Supertrend (вес 0.20)
    st = details.get("supertrend", "")
    if st == "вверх":
        score += 0.20
    elif st == "вниз":
        score -= 0.20

    # Funding Rate (вес 0.15)
    funding_signal = details.get("funding_rate", {}).get("signal", "neutral")
    if funding_signal == "bullish":
        score += 0.15
    elif funding_signal == "bearish":
        score -= 0.15

    # Liquidations (вес 0.10)
    liq_signal = details.get("liquidations", {}).get("signal", "neutral")
    if liq_signal == "bullish":
        score += 0.10
    elif liq_signal == "bearish":
        score -= 0.10

    # Ограничиваем диапазон
    score = max(-1.0, min(1.0, score))

    return score if side == "long" else -score


def bayes_trend_probability(df: pd.DataFrame) -> float:
    """Байесовская вероятность бычего/медвежьего тренда."""
    try:
        close = df["c"]
        ema20 = _ema(close, 20).iloc[-1]
        ema50 = _ema(close, 50).iloc[-1]
        rsi = calc_rsi(close).iloc[-1]
        adx, _, _ = calc_adx(df)
        adx_val = adx.iloc[-1]
        if any(np.isnan(v) for v in [ema20, ema50, rsi, adx_val]):
            return 0.5
        if ema50 == 0:
            return 0.5
        z = ((ema20 / ema50 - 1) * 100 +
             (rsi - 50) / 25 +
             (adx_val - 25) / 10)
        if np.isnan(z) or np.isinf(z):
            return 0.5
        return float(np.clip(1.0 / (1.0 + np.exp(-z)), 0.0, 1.0))
    except Exception:
        return 0.5


def bayesian_score(details: dict, side: str) -> float:
    """Байесовский скор от -1 до 1."""
    prob_bullish = bayes_trend_probability(
        pd.DataFrame({"c": [details.get("price", 0)]}))
    prob_bearish = 1 - prob_bullish
    if side == "long":
        return prob_bullish * 2 - 1
    else:
        return prob_bearish * 2 - 1


def resolve_indicator_conflicts(details: dict) -> dict:
    """
    Анализ противоречий между индикаторами.

    Возвращает:
        - consensus_score
        - group_scores
        - conflicts
        - penalty
        - resolution_log
        - has_conflicts
    """
    def индикатор_направление(key: str, val) -> float:
        """Переводит значение индикатора в направление -1/0/+1."""
        if val is None:
            return 0.0
        try:
            if key == "rsi":
                v = float(val)
                if v < 35:   return +1.0
                if v > 65:   return -1.0
                if v < 45:   return +0.5
                if v > 55:   return -0.5
                return 0.0
            if key == "stoch_k":
                v = float(val)
                if v < 20:   return +1.0
                if v > 80:   return -1.0
                if v < 35:   return +0.5
                if v > 65:   return -0.5
                return 0.0
            if key == "adx":
                v = float(val)
                return +0.5 if v > 25 else 0.0
            if key in ("supertrend", "hull", "range_filter"):
                return +1.0 if val == "вверх" else -1.0
            if key == "macd":
                return +1.0 if val == "бычий" else -1.0
            if key == "тренд_1h":
                return +1.0 if val == "бычий" else -1.0
            if key == "объём_ratio":
                v = float(val)
                if v > 1.5:   return +0.5
                return 0.0
            if key == "order_flow_score":
                v = float(val)
                if v > 60:    return +1.0
                if v < 40:    return -1.0
                return 0.0
            if key == "sr_signal":
                s = str(val)
                if "поддержки" in s: return +1.0
                if "сопротивления" in s: return -1.0
                return 0.0
            if key == "bayes_prob":
                v = float(val)
                if v > 0.6:   return +1.0
                if v < 0.4:   return -1.0
                return 0.0
        except Exception:
            return 0.0
        return 0.0

    group_weights = {
        "trend":     0.35,
        "momentum":  0.30,
        "volume":    0.15,
        "structure": 0.20,
    }
    INDICATOR_GROUPS = {
        "trend":     ["supertrend", "hull", "тренд_1h", "range_filter"],
        "momentum":  ["macd", "rsi", "stoch_k", "adx"],
        "volume":    ["объём_ratio", "order_flow_score"],
        "structure": ["sr_signal", "bayes_prob"],
    }
    GROUP_CONSENSUS_THRESHOLD = 0.6

    group_scores = {}
    conflicts = []
    resolution_log = []
    total_penalty = 0.0

    for group_name, indicators in INDICATOR_GROUPS.items():
        directions = {}
        for ind in indicators:
            val = details.get(ind)
            if val is not None:
                directions[ind] = индикатор_направление(ind, val)

        if not directions:
            group_scores[group_name] = 0.0
            continue

        values = list(directions.values())
        avg = sum(values) / len(values)

        if avg > 0:
            agreeing = sum(1 for v in values if v >= 0)
        elif avg < 0:
            agreeing = sum(1 for v in values if v <= 0)
        else:
            agreeing = sum(1 for v in values if v == 0)

        consensus = agreeing / len(values)

        if consensus < GROUP_CONSENSUS_THRESHOLD:
            penalty = (GROUP_CONSENSUS_THRESHOLD - consensus) * \
                      group_weights[group_name] * 20
            total_penalty += penalty
            conflicts.append({
                "group": group_name,
                "consensus": round(consensus, 2),
                "avg": round(avg, 2),
                "details": directions,
            })
            resolution_log.append(
                f"[{group_name}] ⚠️ Противоречие: согласованность "
                f"{consensus:.0%} → штраф -{penalty:.1f} MQS"
            )
            effective_weight = group_weights[group_name] * consensus
            group_scores[group_name] = avg * effective_weight
        else:
            group_scores[group_name] = avg * group_weights[group_name]
            resolution_log.append(
                f"[{group_name}] ✅ Согласованность {consensus:.0%} "
                f"скор={avg:+.2f}"
            )

    # Межгрупповые конфликты (trend vs momentum)
    trend_dir = group_scores.get("trend", 0)
    momentum_dir = group_scores.get("momentum", 0)
    if (trend_dir > 0.1 and momentum_dir < -0.1) or \
       (trend_dir < -0.1 and momentum_dir > 0.1):
        inter_penalty = 15.0
        total_penalty += inter_penalty
        conflicts.append({
            "group": "trend_vs_momentum",
            "trend": round(trend_dir, 2),
            "momentum": round(momentum_dir, 2),
        })
        resolution_log.append(
            f"[МЕЖГРУППОВОЕ] ⛔ Тренд ({trend_dir:+.2f}) vs "
            f"Моментум ({momentum_dir:+.2f}) → штраф -{inter_penalty:.0f} MQS"
        )

    consensus_score = sum(group_scores.values())
    total_penalty = min(total_penalty, 30.0)  # максимальный штраф

    return {
        "consensus_score": round(consensus_score, 3),
        "group_scores":    {k: round(v, 3) for k, v in group_scores.items()},
        "conflicts":       conflicts,
        "penalty":         round(total_penalty, 1),
        "resolution_log":  resolution_log,
        "has_conflicts":   len(conflicts) > 0,
    }

# ------------------------------------------------------------
# 🧩 Функция расчёта MQS
# ------------------------------------------------------------
def рассчитать_mqs(symbol: str, details: dict,
                   side: str) -> Tuple[float, str, str]:
    """
    Рассчитывает MQR от 0 до 100, возвращает:
    - mqs (float)
    - mqs_emoji (str)
    - mqs_description (str)
    """
    # 1️⃣ Классический теханализ (15 %)
    classic_score = 0.0
    rsi_val = details.get("rsi", 50)
    if 25 <= rsi_val <= 40:
        classic_score += 20
    elif 40 < rsi_val <= 50:
        classic_score += 12
    elif rsi_val < 25:
        classic_score += 10
    elif 50 < rsi_val <= 60:
        classic_score += 5

    rsi_1h = details.get("rsi_1h", 50)
    if rsi_1h < 50:
        classic_score += 10
    elif rsi_1h < 60:
        classic_score += 5

    if details.get("macd") == "бычий":
        classic_score += 8

    if details.get("range_filter") == "вверх":
        classic_score += 15

    if details.get("supertrend") == "вверх":
        classic_score += 12

    if details.get("hull") == "вверх":
        classic_score += 8

    if details.get("ema50_1h", 0) > details.get("ema200_1h", 0):
        classic_score += 10

    if details.get("adx", 0) > 25:
        classic_score += 10

    k_val = details.get("stoch_k", 50)
    if k_val < 20:
        classic_score += 10
    elif k_val < 40:
        classic_score += 5

    vol_ratio = details.get("объём_ratio", 1)
    if vol_ratio > 1.5:
        classic_score += 8
    elif vol_ratio > 1.2:
        classic_score += 4

    if "поддержки" in str(details.get("sr_signal", "")):
        classic_score += 15
    elif "сопротивления" in str(details.get("sr_signal", "")):
        classic_score -= 25

    classic_score += details.get("bayes_prob", 0.5) * 10
    classic_score = min(100, max(0, classic_score))
    classic_score_normalized = (classic_score / 100) * 2 - 1   # → [‑1, 1]

    # 2️⃣ Decision Matrix (25 %)
    decision_score = decision_matrix_score(details, side)

    # 3️⃣ Nash Equilibrium (15 %)
    nash_score = nash_equilibrium_score(symbol, side)

    # 4️⃣ Markov Chain (15 %)
    markov_score = markov_chain_score(details, side)

    # 5️⃣ Hurst Exponent (10 %)
    hurst_score = hurst_exponent_score(details, side, symbol)

    # 6️⃣ Weighted Signal (заменяем Coin Flip) (10 %)
    coin_score = weighted_signal_score(symbol, details, side)

    # 7️⃣ Bayesian Probability (10 %)
    bayesian_val = bayesian_score(details, side)

    # ---------------------------------
    # Взвешенная сумма
    # ---------------------------------
    mqs = (
        classic_score_normalized * MQS_WEIGHTS["classic_ta"] +
        decision_score * MQS_WEIGHTS["decision_matrix"] +
        nash_score * MQS_WEIGHTS["nash_equilibrium"] +
        markov_score * MQS_WEIGHTS["markov_chain"] +
        hurst_score * MQS_WEIGHTS["hurst_exponent"] +
        coin_score * MQS_WEIGHTS["coin_flip"] +
        bayesian_val * MQS_WEIGHTS["bayesian"]
    )

    # 8️⃣ Учёт конфликтов индикаторов
    if CONFLICT_RESOLUTION_ENABLED:
        conflict = resolve_indicator_conflicts(details)
        mqs -= conflict["penalty"] / 100.0   # ← штраф теперь в масштабе [-0.3 … 0]
        if conflict["has_conflicts"]:
            for msg in conflict["resolution_log"]:
                log.debug(f"[CONFLICT] {msg}")
            log.info(
                f"[CONFLICT] Штраф={conflict['penalty']:.1f} "
                f"Групп с конфликтом: {len(conflict['conflicts'])}"
            )
        details["conflict_penalty"] = conflict["penalty"]
        details["conflict_has"] = conflict["has_conflicts"]
        details["conflict_groups"] = conflict["group_scores"]

    # 9️⃣ Нормализация в диапазон 0‑100
    mqs = (mqs + 1) * 50   # → [0, 100]

    # 10️⃣ Смайлик и описание
    for (low, high), emoji in MQS_EMOJIS.items():
        if low <= mqs <= high:
            mqs_emoji = emoji
            mqs_description = MQS_DESCRIPTIONS[(low, high)]
            break
    else:
        mqs_emoji = "❓"
        mqs_description = "НЕИЗВЕСТНЫЙ СИГНАЛ"

    return mqs, mqs_emoji, mqs_description


# ------------------------------------------------------------
# 🌐 Получение скоринга (полный анализ)
# ------------------------------------------------------------
def получить_bybit_ai(symbol: str) -> dict:
    """Запрос AI‑соотношения долг/корот (Bybit Ratio)."""
    result = {"signal": "neutral", "long_ratio": 0.5,
              "short_ratio": 0.5, "available": False}
    try:
        coin = symbol.split("/")[0]
        url = (f"https://api.bybit.com/v5/market/account-ratio?"
               f"category=linear&symbol={coin}USDT&period=1h&limit=1")
        resp = requests.get(url, timeout=5)
        data = resp.json()
        if data.get("retCode") == 0:
            items = data.get("result", {}).get("list", [])
            if items:
                buy_r = float(items[0].get("buyRatio", 0.5))
                sell_r = float(items[0].get("sellRatio", 0.5))
                result.update({"long_ratio": buy_r,
                               "short_ratio": sell_r,
                               "available": True})
                if buy_r > 0.6:
                    result["signal"] = "bullish"
                elif buy_r < 0.4:
                    result["signal"] = "bearish"
    except Exception as e:
        log.debug(f"Bybit ratio недоступен: {e}")
    return result


def тренд_4h_бычий(symbol: str) -> bool:
    """Определение 4‑hour тренда (EMA 20 vs EMA 50)."""
    try:
        raw = exchange.fetch_ohlcv(symbol, TIMEFRAME_4H, limit=60)
        if len(raw) < 55:
            return False
        df = pd.DataFrame(raw,
                          columns=["ts", "o", "h", "l", "c", "v"])
        return bool(_ema(df["c"], 20).iloc[-1] > _ema(df["c"], 50).iloc[-1])
    except Exception:
        return False


def тренд_4h_медвежий(symbol: str) -> bool:
    """Определение 4‑hour медвежьего тренда."""
    try:
        raw = exchange.fetch_ohlcv(symbol, TIMEFRAME_4H, limit=60)
        if len(raw) < 55:
            return False
        df = pd.DataFrame(raw,
                          columns=["ts", "o", "h", "l", "c", "v"])
        return bool(_ema(df["c"], 20).iloc[-1] < _ema(df["c"], 50).iloc[-1])
    except Exception:
        return False


def проверить_ma_кроссовер(df: pd.DataFrame, side: str = "long") -> bool:
    """Кроссовер MA‑индикатора."""
    if not MA_CROSSOVER_ENABLED:
        return True
    try:
        min_len = max(MA1_LENGTH, MA2_LENGTH) * 2 + 5
        if len(df) < min_len:
            return True
        ma1 = calc_ma(df, MA1_TYPE, MA1_LENGTH)
        ma2 = calc_ma(df, MA2_TYPE, MA2_LENGTH)
        return bool(ma1.iloc[-1] > ma2.iloc[-1]) if side == "long" else bool(ma1.iloc[-1] < ma2.iloc[-1])
    except Exception as e:
        log.warning(f"Ошибка MA‑кроссовера: {e}")
        return True


def volume_spike_guard(df: pd.DataFrame) -> bool:
    """Проверка резкого объёма."""
    try:
        vol_avg = df["v"].rolling(VOLUME_AVG_PERIOD).mean().iloc[-1]
        vol_now = df["v"].iloc[-1]
        ratio = vol_now / (vol_avg + 1e-10)
        if ratio > VOLUME_SPIKE_MULT:
            log.info(f"Volume Spike Guard: объём {ratio:.1f}x > {VOLUME_SPIKE_MULT}x")
            return False
        return True
    except Exception:
        return True


def get_quant_signals(symbol: str) -> Dict[str, Any]:
    """Квантовый модуль (Cointegration, Mean Reversion, Momentum)."""
    if not QUANT_ENABLED:
        return {"quant_score": 0, "details": {}}
    signals = {
        "cointegration": {},
        "mean_reversion": check_mean_reversion_opportunity(symbol),
        "momentum": calculate_momentum(symbol),
    }
    for pair in COINTEGRATION_PAIRS:
        if symbol in pair:
            other = pair[0] if pair[1] == symbol else pair[1]
            coint_res = calculate_cointegration(pair)
            if coint_res["valid"]:
                signals["cointegration"][other] = coint_res
    quant_score = 0
    if signals["mean_reversion"]["valid"]:
        z = abs(signals["mean_reversion"]["zscore"])
        if z > MEAN_REVERSION_THRESHOLD:
            quant_score += 25
        elif z > MEAN_REVERSION_THRESHOLD * 0.7:
            quant_score += 15
    if signals["momentum"]["valid"]:
        mom = signals["momentum"]["momentum"]
        if abs(mom) > 3:
            quant_score += 20
        elif abs(mom) > 1.5:
            quant_score += 10
    for other, coint_data in signals["cointegration"].items():
        if coint_data["valid"] and coint_data["pvalue"] < 0.05:
            z = abs(coint_data["zscore"])
            if z > 2:
                quant_score += 20
            elif z > 1.5:
                quant_score += 10
    return {"quant_score": min(100, quant_score), "details": signals}


def get_order_flow_signals(symbol: str) -> Dict[str, Any]:
    """Сигналы Order Flow (order‑book, volume‑profile, market‑maker)."""
    if not ORDER_FLOW_ENABLED:
        return {"order_flow_score": 0, "details": {}}
    signals = {
        "order_book": get_order_book(symbol),
        "volume_profile": analyze_volume_profile(symbol),
        "market_maker": detect_market_maker_activity(symbol),
    }
    order_flow_score = 0
    if signals["order_book"]["valid"]:
        imbal = signals["order_book"]["imbalance"]
        spread = signals["order_book"]["spread_pct"]
        if imbal > 20:
            order_flow_score += 15
        elif imbal < -20:
            order_flow_score -= 15
        if spread < 0.1:
            order_flow_score += 10
    if signals["volume_profile"]["valid"]:
        price = signals["volume_profile"]["current_price"]
        sup = signals["volume_profile"]["nearest_support"]
        res = signals["volume_profile"]["nearest_resistance"]
        if price > sup * 1.01:
            order_flow_score += 15
        elif price < res * 0.99:
            order_flow_score -= 15
    if signals["market_maker"]["activity"] == "high":
        order_flow_score += 10
    elif signals["market_maker"]["activity"] == "medium":
        order_flow_score += 5
    return {"order_flow_score": max(0, min(100, order_flow_score + 50)),
            "details": signals}


def get_order_book(symbol: str, depth: int = ORDER_BOOK_DEPTH) -> Dict[str, Any]:
    """Получение данных по стакану."""
    try:
        order_book = exchange.fetch_order_book(symbol, depth)
        bids = order_book["bids"]
        asks = order_book["asks"]
        total_bid_vol = sum([b[1] for b in bids])
        total_ask_vol = sum([a[1] for a in asks])
        bid_prices = [b[0] for b in bids]
        ask_prices = [a[0] for a in asks]
        best_bid = bids[0][0] if bids else 0
        best_ask = asks[0][0] if asks else 0
        spread = best_ask - best_bid
        spread_pct = (spread / best_bid * 100) if best_bid > 0 else 0
        imbalance = (total_bid_vol - total_ask_vol) / \
                   (total_bid_vol + total_ask_vol + 1e-10) * 100
        depth_bids = {}
        depth_asks = {}
        for i, (price, vol) in enumerate(bids[:5]):
            depth_bids[f"bid_{i + 1}"] = {"price": price,
                                         "volume": vol}
        for i, (price, vol) in enumerate(asks[:5]):
            depth_asks[f"ask_{i + 1}"] = {"price": price,
                                         "volume": vol}
        return {
            "valid": True,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "spread_pct": spread_pct,
            "total_bid_volume": total_bid_vol,
            "total_ask_volume": total_ask_vol,
            "imbalance": imbalance,
            "depth_bids": depth_bids,
            "depth_asks": depth_asks,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        log.debug(f"Ошибка Order Book для {symbol}: {e}")
        return {"valid": False}


# ------------------------------------------------------------
# 📊 Анализ Volume Profile
# ------------------------------------------------------------
def analyze_volume_profile(symbol: str,
                          bars: int = VOLUME_PROFILE_BARS) -> Dict[str, Any]:
    """Анализирует Volume Profile для символа."""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA, limit=bars)
        if len(ohlcv) < bars:
            return {"valid": False}

        df = pd.DataFrame(ohlcv,
                          columns=["ts", "o", "h", "l", "c", "v"])
        # Средняя цена диапазона каждой свечи
        df["price_range"] = (df["h"] + df["l"]) / 2

        # Гистограмма объёмов по 20 ценовым бинам
        bins = 20
        hist, bin_edges = np.histogram(df["price_range"],
                                       bins=bins,
                                       weights=df["v"])
        poc_index = np.argmax(hist)                     # точка контроля
        poc_price = (bin_edges[poc_index] + bin_edges[poc_index + 1]) / 2
        poc_volume = hist[poc_index]

        # Сильные зоны – объём выше 70‑го перцентиля
        volume_threshold = np.percentile(hist, 70)
        support_levels, resistance_levels = [], []

        for i in range(len(hist)):
            if hist[i] >= volume_threshold:
                price_level = (bin_edges[i] + bin_edges[i + 1]) / 2
                prices_below = df[df["price_range"] < price_level]["l"]
                prices_above = df[df["price_range"] > price_level]["h"]

                # Поддержка: цены ниже уровня, но не выше
                if len(prices_below) > 0 and len(prices_above) > 0:
                    if (prices_below < price_level).all() and len(prices_below) >= 3:
                        support_levels.append(price_level)
                    # Сопротивление: цены выше уровня, но не ниже
                    if (prices_above > price_level).all() and len(prices_above) >= 3:
                        resistance_levels.append(price_level)

        current_price = df["c"].iloc[-1]

        # Сортируем уровни рядом с текущей ценой
        support_levels = sorted([p for p in support_levels if p < current_price],
                               reverse=True)
        resistance_levels = sorted([p for p in resistance_levels if p > current_price])

        nearest_support = support_levels[0] if support_levels else current_price * 0.95
        nearest_resistance = resistance_levels[0] if resistance_levels else current_price * 1.05

        return {
            "valid": True,
            "poc_price": float(poc_price),
            "poc_volume": float(poc_volume),
            "support_levels": [float(p) for p in support_levels[:3]],
            "resistance_levels": [float(p) for p in resistance_levels[:3]],
            "nearest_support": float(nearest_support),
            "nearest_resistance": float(nearest_resistance),
            "current_price": float(current_price),
            "volume_threshold": float(volume_threshold),
        }
    except Exception as e:
        log.debug(f"Ошибка Volume Profile для {symbol}: {e}")
        return {"valid": False}


# ------------------------------------------------------------
# 🤖 Обнаружение активности маркет‑мейкеров
# ------------------------------------------------------------
def detect_market_maker_activity(symbol: str) -> Dict[str, Any]:
    """Обнаруживает активность маркет‑мейкеров."""
    try:
        order_book = get_order_book(symbol, ORDER_BOOK_DEPTH)
        if not order_book["valid"]:
            return {"activity": "unknown", "confidence": 0}

        spread_pct = order_book["spread_pct"]
        total_bid_vol = order_book["total_bid_volume"]
        total_ask_vol = order_book["total_ask_volume"]
        imbalance = abs(order_book["imbalance"])

        activity_score = 0

        if spread_pct < 0.1:
            activity_score += 30
        elif spread_pct < 0.2:
            activity_score += 20

        avg_volume = (total_bid_vol + total_ask_vol) / 2
        if avg_volume > 1000:
            activity_score += 25
        elif avg_volume > 500:
            activity_score += 15

        if imbalance < 10:
            activity_score += 25
        elif imbalance < 20:
            activity_score += 15

        if activity_score > 60:
            activity = "high"
            confidence = min(100, activity_score)
        elif activity_score > 40:
            activity = "medium"
            confidence = activity_score * 0.8
        else:
            activity = "low"
            confidence = activity_score * 0.5

        return {
            "activity": activity,
            "confidence": min(100, confidence),
            "spread_pct": spread_pct,
            "imbalance": imbalance,
            "total_volume": total_bid_vol + total_ask_vol,
        }
    except Exception as e:
        log.debug(f"Ошибка детекции маркетмейкеров для {symbol}: {e}")
        return {"activity": "unknown", "confidence": 0}


# ------------------------------------------------------------
# 🧮 Квантовый модуль (Cointegration, Mean‑Reversion, Momentum)
# ------------------------------------------------------------
def calculate_cointegration(pair: Tuple[str, str],
                           window: int = COINTEGRATION_WINDOW) -> Dict[str, Any]:
    """Расчёт коинтеграции пары."""
    try:
        symbol1, symbol2 = pair
        ohlcv1 = exchange.fetch_ohlcv(symbol1, TIMEFRAME_TA, limit=window)
        ohlcv2 = exchange.fetch_ohlcv(symbol2, TIMEFRAME_TA, limit=window)
        if len(ohlcv1) < window or len(ohlcv2) < window:
            return {"coint": 0, "pvalue": 1, "spread": 0, "zscore": 0,
                    "valid": False}

        df1 = pd.DataFrame(ohlcv1,
                           columns=["ts", "o", "h", "l", "c", "v"])["c"]
        df2 = pd.DataFrame(ohlcv2,
                           columns=["ts", "o", "h", "l", "c", "v"])["c"]

        # Проверяем стационарность
        adf1 = adfuller(df1)
        adf2 = adfuller(df2)
        if adf1[1] > 0.05 or adf2[1] > 0.05:
            df1 = df1.diff().dropna()
            df2 = df2.diff().dropna()

        coint_res = coint(df1, df2)
        coint_coeff = coint_res[0]
        pvalue = coint_res[1]
        hedge_ratio = coint_res[0][1] if isinstance(coint_res[0], (list, tuple, np.ndarray)) else 1.0

        spread = df1 - hedge_ratio * df2
        spread_mean = spread.mean()
        spread_std = spread.std()
        current_spread = spread.iloc[-1]
        zscore = (current_spread - spread_mean) / spread_std if spread_std > 0 else 0

        return {
            "coint": float(coint_coeff),
            "pvalue": float(pvalue),
            "spread": float(current_spread),
            "zscore": float(zscore),
            "hedge_ratio": float(hedge_ratio),
            "spread_mean": float(spread_mean),
            "spread_std": float(spread_std),
            "valid": True,
        }
    except Exception as e:
        log.debug(f"Ошибка коинтеграции для {pair}: {e}")
        return {"coint": 0, "pvalue": 1, "spread": 0, "zscore": 0,
                "valid": False}


def check_mean_reversion_opportunity(symbol: str,
                                    window: int = MEAN_REVERSION_THRESHOLD * 2) -> Dict[str, Any]:
    """Mean‑Reversion сигнал."""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA, limit=window)
        if len(ohlcv) < window:
            return {"signal": "neutral", "zscore": 0, "valid": False}

        df = pd.DataFrame(ohlcv,
                          columns=["ts", "o", "h", "l", "c", "v"])
        close = df["c"]
        sma = close.rolling(window).mean()
        std = close.rolling(window).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        current_price = close.iloc[-1]
        sma_val = sma.iloc[-1]
        zscore = (current_price - sma_val) / std.iloc[-1] if std.iloc[-1] > 0 else 0

        if zscore > MEAN_REVERSION_THRESHOLD:
            signal = "sell"
        elif zscore < -MEAN_REVERSION_THRESHOLD:
            signal = "buy"
        else:
            signal = "neutral"

        return {
            "signal": signal,
            "zscore": float(zscore),
            "bollinger_upper": float(upper_band.iloc[-1]),
            "bollinger_lower": float(lower_band.iloc[-1]),
            "sma": float(sma_val),
            "current_price": float(current_price),
            "valid": True,
        }
    except Exception as e:
        log.debug(f"Ошибка Mean Reversion для {symbol}: {e}")
        return {"signal": "neutral", "zscore": 0, "valid": False}


def calculate_momentum(symbol: str,
                       window: int = MOMENTUM_WINDOW) -> Dict[str, Any]:
    """Momentum‑индикатор."""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA,
                                    limit=window + 10)
        if len(ohlcv) < window:
            return {"momentum": 0, "signal": "neutral", "valid": False}
        df = pd.DataFrame(ohlcv,
                          columns=["ts", "o", "h", "l", "c", "v"])
        close = df["c"]
        momentum = (close.iloc[-1] - close.iloc[-window]) / close.iloc[-window] * 100
        ema_fast = _ema(close, window // 2)
        ema_slow = _ema(close, window)
        ema_momentum = (ema_fast.iloc[-1] - ema_slow.iloc[-1]) / ema_slow.iloc[-1] * 100

        if momentum > 2:
            signal = "bullish"
        elif momentum < -2:
            signal = "bearish"
        else:
            signal = "neutral"

        return {
            "momentum": float(momentum),
            "ema_momentum": float(ema_momentum),
            "signal": signal,
            "valid": True,
        }
    except Exception as e:
        log.debug(f"Ошибка Momentum для {symbol}: {e}")
        return {"momentum": 0, "signal": "neutral", "valid": False}


# ------------------------------------------------------------
# ⚡ Экспресс‑анализ (быстрый скрининг)
# ------------------------------------------------------------
def получить_скор_экспресс(symbol: str) -> dict:
    """
    Быстрый анализ без тяжёлых модулей (ML, Quant, OrderFlow, AdvancedBybit).
    Возвращает упрощённый скор и MQS.
    """
    try:
        raw_ta = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA, limit=100)
        raw_1h = exchange.fetch_ohlcv(symbol, TIMEFRAME_TREND, limit=100)
        if len(raw_ta) < 60 or len(raw_1h) < 60:
            return {"score": 0, "mqs": 0, "express": True,
                    "details": {}}

        cols = ["ts", "o", "h", "l", "c", "v"]
        df_ta = pd.DataFrame(raw_ta, columns=cols)
        df_1h = pd.DataFrame(raw_1h, columns=cols)

        c_ta = df_ta["c"]
        c_1h = df_1h["c"]
        price = float(c_ta.iloc[-1])
        details = {"price": price}

        score = 0

        # RSI
        rsi = calc_rsi(c_ta).iloc[-1]
        details["rsi"] = round(rsi, 1)
        if 25 <= rsi <= 40:
            score += 20
        elif rsi < 25:
            score += 10
        elif rsi < 50:
            score += 8

        # RSI 1h
        rsi_1h = calc_rsi(c_1h).iloc[-1]
        details["rsi_1h"] = round(rsi_1h, 1)
        if rsi_1h < 50:
            score += 10

        # MACD
        ml, sl, _ = calc_macd(c_ta)
        macd_bull = ml.iloc[-1] > sl.iloc[-1]
        details["macd"] = "бычий" if macd_bull else "медвежий"
        if macd_bull:
            score += 15

        # Supertrend
        st_up, _ = calc_supertrend(df_ta)
        details["supertrend"] = "вверх" if st_up.iloc[-1] else "вниз"
        if st_up.iloc[-1]:
            score += 15

        # Hull
        hu_up, _ = calc_hull(c_ta)
        details["hull"] = "вверх" if hu_up.iloc[-1] else "вниз"
        if hu_up.iloc[-1]:
            score += 10

        # EMA‑trend 1h
        ema50_1h = _ema(c_1h, 50).iloc[-1]
        ema200_1h = _ema(c_1h, 200).iloc[-1]
        details["тренд_1h"] = "бычий" if ema50_1h > ema200_1h else "медвежий"
        if ema50_1h > ema200_1h:
            score += 10

        # ADX
        adx, _, _ = calc_adx(df_ta)
        details["adx"] = round(adx.iloc[-1], 1)
        if adx.iloc[-1] > 25:
            score += 10

        # Bybit AI Ratio
        ai = получить_bybit_ai(symbol)
        details["ai_signal"] = ai["signal"]
        if ai["signal"] == "bullish":
            score += 5
        elif ai["signal"] == "bearish":
            score -= 10

        # Конфликты – экспресс‑версия
        conflict = resolve_indicator_conflicts(details) if CONFLICT_RESOLUTION_ENABLED else {"penalty": 0}
        mqs_raw = min(100, max(0, score))
        mqs_raw -= conflict.get("penalty", 0)
        mqs = max(0, min(100, mqs_raw))

        # Определяем emoji
        for (low, high), emoji in MQS_EMOJIS.items():
            if low <= mqs <= high:
                mqs_emoji = emoji
                mqs_desc = MQS_DESCRIPTIONS[(low, high)]
                break
        else:
            mqs_emoji = "❓"
            mqs_desc = "НЕИЗВЕСТНЫЙ СИГНАЛ"

        return {
            "score": max(0, min(100, score)),
            "mqs": mqs,
            "mqs_emoji": mqs_emoji,
            "mqs_description": mqs_desc,
            "details": details,
            "price": price,
            "express": True,
            "conflict_penalty": conflict.get("penalty", 0),
        }

    except Exception as e:
        log.debug(f"Экспресс‑анализ {symbol}: {e}")
        return {"score": 0, "mqs": 0, "express": True,
                "details": {}}


# ------------------------------------------------------------
# 🏁 Полный скори‑анализ (с учётом всех модулей)
# ------------------------------------------------------------
def получить_скор(symbol: str,
                 use_quant: bool = True,
                 use_order_flow: bool = True,
                 use_ml: bool = True,
                 use_advanced: bool = True) -> dict:
    """
    Возвращает полный скор, MQS и детальную информацию.
    """
    details = {}
    classic_score = 0
    price = 0.0
    sr = {}

    try:
        # ---------- ОHLCV ----------
        raw_ta = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA, limit=300)
        raw_1h = exchange.fetch_ohlcv(symbol, TIMEFRAME_TREND, limit=300)
        if len(raw_ta) < 100 or len(raw_1h) < 100:
            return {"score": 0, "mqs": 0, "mqs_emoji": "❓",
                    "mqs_description": "НЕДОСТАТОЧНО ДАННЫХ",
                    "details": {}, "price": 0, "sr": {}}

        cols = ["ts", "o", "h", "l", "c", "v"]
        df_ta = pd.DataFrame(raw_ta, columns=cols).reset_index(drop=True)
        df_1h = pd.DataFrame(raw_1h, columns=cols).reset_index(drop=True)

        c_ta = df_ta["c"]
        c_1h = df_1h["c"]
        price = float(c_ta.iloc[-1])

        # ---------- Классический теханализ ----------
        rsi_val = calc_rsi(c_ta).iloc[-1]
        details["rsi"] = round(rsi_val, 1)
        if 25 <= rsi_val <= 40:
            classic_score += 20
        elif 40 < rsi_val <= 50:
            classic_score += 12
        elif rsi_val < 25:
            classic_score += 10
        elif 50 < rsi_val <= 60:
            classic_score += 5

        rsi_1h = calc_rsi(c_1h).iloc[-1]
        details["rsi_1h"] = round(rsi_1h, 1)
        if rsi_1h < 50:
            classic_score += 10
        elif rsi_1h < 60:
            classic_score += 5

        ml, sl, _ = calc_macd(c_ta)
        macd_bull = ml.iloc[-1] > sl.iloc[-1]
        macd_cross = macd_bull and ml.iloc[-2] <= sl.iloc[-2]
        details["macd"] = "бычий" if macd_bull else "медвежий"
        if macd_cross:
            classic_score += 18
        elif macd_bull:
            classic_score += 8

        _, _, _, rf_up, rf_down = calc_range_filter(df_ta)
        rf_up_now = rf_up.iloc[-1]
        details["range_filter"] = "вверх" if rf_up_now else (
            "вниз" if rf_down.iloc[-1] else "бок")
        if rf_up_now:
            classic_score += 15

        st_up, _ = calc_supertrend(df_ta)
        details["supertrend"] = "вверх" if st_up.iloc[-1] else "вниз"
        if st_up.iloc[-1]:
            classic_score += 12

        hu_up, _ = calc_hull(c_ta)
        details["hull"] = "вверх" if hu_up.iloc[-1] else "вниз"
        if hu_up.iloc[-1]:
            classic_score += 8

        ema50_1h = _ema(c_1h, 50).iloc[-1]
        ema200_1h = _ema(c_1h, 200).iloc[-1]
        details["тренд_1h"] = "бычий" if ema50_1h > ema200_1h else "медвежий"
        if ema50_1h > ema200_1h:
            classic_score += 10

        adx, pdi, mdi = calc_adx(df_ta)
        adx_val = adx.iloc[-1]
        details["adx"] = round(adx_val, 1)
        details["pdi"] = round(pdi.iloc[-1], 1)
        details["mdi"] = round(mdi.iloc[-1], 1)
        if adx_val > 25 and pdi.iloc[-1] > mdi.iloc[-1]:
            classic_score += 10
        elif adx_val > 20 and pdi.iloc[-1] > mdi.iloc[-1]:
            classic_score += 4

        k_ser, _ = calc_stochastic(df_ta)
        k_val = k_ser.iloc[-1]
        details["stoch_k"] = round(k_val, 1)
        if k_val < 20:
            classic_score += 10
        elif k_val < 40:
            classic_score += 5

        vol_avg = df_ta["v"].rolling(VOLUME_AVG_PERIOD).mean().iloc[-1]
        vol_ratio = df_ta["v"].iloc[-1] / (vol_avg + 1e-10)
        details["объём_ratio"] = round(vol_ratio, 2)
        if vol_ratio > 1.5:
            classic_score += 8
        elif vol_ratio > 1.2:
            classic_score += 4

        sr = calc_support_resistance(df_ta)
        details.update({
            "support": sr["support"],
            "resistance": sr["resistance"],
            "dist_sup": sr["dist_to_sup_pct"],
            "dist_res": sr["dist_to_res_pct"],
        })
        if sr["near_support"]:
            classic_score += 15
            details["sr_signal"] = f"у поддержки ✅ ({sr['sup_cluster']} касаний)"
        elif sr["near_resistance"]:
            classic_score -= 25
            details["sr_signal"] = f"у сопротивления ❌ ({sr['res_cluster']} касаний)"
        else:
            details["sr_signal"] = (f"нейтр (sup={sr['dist_to_sup_pct']:.2f}% "
                                    f"res={sr['dist_to_res_pct']:.2f}%)")

        last3_bearish = all(
            df_ta["c"].iloc[-i] < df_ta["o"].iloc[-i] for i in range(1, 4))
        if last3_bearish:
            classic_score -= 20
            details["свечи_3red"] = True

        bayes_prob = bayes_trend_probability(df_ta)
        details["bayes_prob"] = round(bayes_prob, 2)
        classic_score += int(bayes_prob * 10)

        # ---------- Квант‑анализ ----------
        if use_quant and QUANT_ENABLED:
            quant_data = get_quant_signals(symbol)
            quant_score = quant_data["quant_score"]
            details["quant_score"] = quant_score
            classic_score += quant_score * 0.3
            details["quant_details"] = quant_data["details"]

        # ---------- Order‑Flow ----------
        if use_order_flow and ORDER_FLOW_ENABLED:
            order_flow_data = get_order_flow_signals(symbol)
            of_score = order_flow_data["order_flow_score"]
            details["order_flow_score"] = of_score
            classic_score += of_score * 0.2
            details["order_flow_details"] = order_flow_data["details"]

        # ---------- Advanced Bybit сигналы ----------
        if use_advanced:
            advanced_data = get_bybit_advanced_signals(symbol)
            advanced_score = advanced_data["advanced_score"]
            details["advanced_score"] = advanced_score
            classic_score += advanced_score * 0.4
            details["advanced_details"] = advanced_data["details"]

        # ---------- Проверка Mark‑Price ----------
        mark_price_data = get_mark_price_diff(symbol)
        if mark_price_data["valid"] and not mark_price_data["safe"]:
            details["mark_price_unsafe"] = True
            classic_score -= 30

        # ---------- ML‑модель ----------
        if use_ml and ML_ENABLED and ml_model.trained:
            ml_prediction = ml_model.predict(
                symbol, df_ta, df_1h,
                order_flow_data if use_order_flow else {},
                quant_data if use_quant else {}
            )
            if ml_prediction["valid"]:
                prob = ml_prediction["probability"]
                details["ml_probability"] = round(prob, 2)
                details["ml_signal"] = ml_prediction["signal"]
                if prob > 0.7:
                    classic_score += 15
                elif prob > 0.6:
                    classic_score += 10
                elif prob > 0.55:
                    classic_score += 5

        # ---------- Прочие фильтры ----------
        details["ma_cross"] = проверить_ma_кроссовер(df_ta, side="long")
        details["vol_spike_ok"] = volume_spike_guard(df_ta)

        # ---------- MQS ----------
        mqs, mqs_emoji, mqs_description = рассчитать_mqs(
            symbol, details, "long")

        # ---------- Лог MQS ----------
        log.info(f"[MQS] {mqs_emoji} {mqs:.1f}/100 → {mqs_description}")

        # ---------- Возврат ----------
        return {
            "score": max(0, min(100, classic_score)),
            "mqs": mqs,
            "mqs_emoji": mqs_emoji,
            "mqs_description": mqs_description,
            "details": details,
            "price": price,
            "sr": sr,
        }

    except Exception as e:
        log.warning(f"Ошибка анализа {symbol}: {e}")
        return {
            "score": 0,
            "mqs": 0,
            "mqs_emoji": "❓",
            "mqs_description": "ОШИБКА АНАЛИЗА",
            "details": {},
            "price": 0,
            "sr": {},
        }


# ------------------------------------------------------------
# 📉 Шорт‑скори (инвертированный классический скор)
# ------------------------------------------------------------
def получить_скор_шорта(symbol: str) -> dict:
    """Получает скор для короткой позиции."""
    res = получить_скор(symbol)
    if res["score"] == 0:
        return res

    # Инвертируем классический скор для шорта
    res["score"] = max(0, 100 - res["score"] - 10)

    # Пересчитываем MQS для шорта
    mqs, mqs_emoji, mqs_description = рассчитать_mqs(
        symbol, res["details"], "short")
    res["mqs"] = mqs
    res["mqs_emoji"] = mqs_emoji
    res["mqs_description"] = mqs_description

    # Логируем MQS для шорта
    log.info(f"[MQS SHORT] {mqs_emoji} {mqs:.1f}/100 → {mqs_description}")

    try:
        raw = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA, limit=300)
        if len(raw) >= 50:
            df = pd.DataFrame(raw,
                              columns=["ts", "o", "h", "l", "c", "v"])
            res["details"]["ma_cross"] = проверить_ma_кроссовер(df,
                                                                side="short")
    except Exception:
        pass
    return res


# ------------------------------------------------------------
# 📈 Получение реального P&L (fallback через изменение баланса)
# ------------------------------------------------------------
def получить_реальный_pnl(symbol: str,
                         entry_time: float) -> Optional[float]:
    """
    Попытка получить реальный P&L через запрос позиции.
    Если не удалось – возвращаем ``None`` (будет использовать дельту баланса).
    """
    try:
        positions = exchange.fetch_positions([symbol])
        for p in positions:
            if float(p.get("contracts", 0) or 0) > 0:
                # На бирже уже есть расчётный PnL
                return float(p.get("unrealizedPnl", 0))
    except Exception as e:
        log.debug(f"Не удалось получить PnL для {symbol}: {e}")
    return None


# ------------------------------------------------------------
# 📊 Система оценки риска (Kelly, позиционный размер и т.д.)
# ------------------------------------------------------------
def рассчитать_оптимальное_f(сделки: List[dict]) -> float:
    """Определяет оптимальный коэффициент f (Kelly)."""
    if len(сделки) < MIN_TRADES_FOR_F:
        log.info(f"Недостаточно сделок для Kelly ({len(сделки)} < {MIN_TRADES_FOR_F})")
        return 0.0
    pnls = [abs(t['pnl_usdt']) for t in сделки if t['pnl_usdt'] != 0]
    if not pnls:
        return 0.0
    wins = sum(1 for t in сделки if t['pnl_usdt'] > 0)
    losses = sum(1 for t in сделки if t['pnl_usdt'] < 0)
    if losses == 0:
        return 0.25
    win_rate = wins / (wins + losses)
    avg_win = sum(t['pnl_usdt'] for t in сделки if t['pnl_usdt'] > 0) / wins if wins else 0
    avg_loss = abs(sum(t['pnl_usdt'] for t in сделки if t['pnl_usdt'] < 0)) / losses if losses else 1
    if avg_loss == 0:
        avg_loss = 1
    kelly = win_rate - (1 - win_rate) / (avg_win / avg_loss)
    return min(max(0, kelly * 0.4), MAX_RISK_PERCENT_F / 100)


def рассчитать_размер_позиции(score: int, баланс: float,
                               sl_dist_pct: float,
                               история_сделок: List[dict] = None) -> float:
    """Определяет маржу для новой позиции."""
    if USE_ADVANCED_RISK and история_сделок and len(история_сделок) >= MIN_TRADES_FOR_F:
        f_opt = рассчитать_оптимальное_f(история_сделок[-100:])
        risk_pct = max(0.5, min(f_opt * 100, MAX_RISK_PERCENT_F))
        log.info(f"Оптимальное f = {f_opt:.3f} → риск {risk_pct:.2f}%")
    else:
        factor = max(0, (score - MIN_SCORE)) / (100 - MIN_SCORE)
        risk_pct = min(BASE_RISK_PCT + (MAX_RISK_PCT - BASE_RISK_PCT) * factor,
                       MAX_RISK_PCT)
    max_loss_usdt = баланс * risk_pct / 100
    margin_usdt = min(max_loss_usdt / (sl_dist_pct / 100), баланс * 0.95)
    log.info(f"Скор={score} → риск={risk_pct:.1f}% SL_dist={sl_dist_pct:.2f}% "
             f"маржа={margin_usdt:.2f}U")
    return round(max(1.0, margin_usdt), 2)


def подтвердить_вход(symbol: str, исходный_скор: int,
                    side: str = "long") -> bool:
    """Подтверждает вход через дополнительный тайм‑фрейм."""
    if ENTRY_CONFIRM_BARS <= 0:
        return True
    tf_seconds = {"1m": 60, "3m": 180, "5m": 300,
                  "15m": 900}
    wait = tf_seconds.get(TIMEFRAME_TA, 300) * ENTRY_CONFIRM_BARS
    log.info(f"Подтверждение входа: ждём {wait}s ({ENTRY_CONFIRM_BARS} свеча)...")
    time.sleep(wait)
    новый = получить_скор(symbol) if side == "long" else получить_скор_шорта(symbol)
    новый_скор = новый["score"]
    mark_price_data = get_mark_price_diff(symbol)
    if mark_price_data["valid"] and not mark_price_data["safe"]:
        log.warning(f"Mark Price vs Last Price расхождение: "
                    f"{mark_price_data['diff_pct']:.2f}% > "
                    f"{MARK_PRICE_DIFF_THRESHOLD}%")
        return False
    log.info(f"Перепроверка скора: {исходный_скор} → {новый_скор} "
             f"(мин={ENTRY_CONFIRM_MIN_SCORE})")
    if новый_скор < ENTRY_CONFIRM_MIN_SCORE:
        log.info(f"Подтверждение не прошло: скор упал до {новый_скор}")
        return False
    if not новый.get("details", {}).get("vol_spike_ok", True):
        log.info("Подтверждение не прошло: volume spike")
        return False
    log.info(f"Вход подтверждён. Скор {новый_скор}/100")
    return True


# ------------------------------------------------------------
# 📈 Управление позицией (открытие / закрытие)
# ------------------------------------------------------------
def установить_плечо(symbol: str, leverage: int) -> bool:
    """Устанавливает плечо (первый способ, потом fallback)."""
    try:
        exchange.set_leverage(leverage, symbol,
                              params={"buyLeverage": leverage,
                                      "sellLeverage": leverage})
        log.info(f"Плечо {leverage}x установлено для {symbol}")
        return True
    except Exception as e1:
        log.warning(f"Метод 1 плеча не сработал: {e1}")
    try:
        coin_sym = symbol.replace("/", "").replace(":USDT", "")
        exchange.private_post_v5_position_set_leverage({
            "category": "linear",
            "symbol": coin_sym,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage),
        })
        log.info(f"Плечо {leverage}x установлено (v5) для {symbol}")
        return True
    except Exception as e2:
        log.warning(f"Метод 2 плеча не сработал: {e2}")
    log.error(f"Не удалось установить плечо {leverage}x для {symbol}")
    return False


def обновить_sl_на_бирже(symbol: str, new_sl: float,
                         side: str = "long") -> bool:
    """Обновление stop‑loss через API."""
    try:
        sl_str = exchange.price_to_precision(symbol, new_sl)
        coin_sym = symbol.replace("/", "").replace(":USDT", "")
        exchange.private_post_v5_position_trading_stop({
            "category": "linear",
            "symbol": coin_sym,
            "stopLoss": sl_str,
            "slTriggerBy": "MarkPrice",
            "positionIdx": "0",
        })
        log.info(f"SL обновлён → {sl_str}")
        return True
    except Exception as e:
        log.warning(f"Не удалось обновить SL: {e}")
        return False


def открыть_позицию(symbol: str, margin_usdt: float,
                    tp_price: float, sl_price: float,
                    side: str = "long") -> Tuple[Optional[float],
                                                Optional[float]]:
    """
    Открывает позицию, рассчитывает qty с учётом precision и amount_to_precision.
    Возвращает (entry_price, qty) либо (None, None) при ошибке.
    """
    try:
        if not установить_плечо(symbol, LEVERAGE):
            log.error(f"Плечо не установлено — сделка отменена для {symbol}")
            return None, None

        ticker = exchange.fetch_ticker(symbol)
        price = float(ticker["last"])
        pos_size_usdt = margin_usdt * LEVERAGE
        qty_raw = pos_size_usdt / price
        qty = float(exchange.amount_to_precision(symbol, qty_raw))

        if qty <= 0:
            log.error(f"Нулевое количество {symbol}")
            return None, None

        if side == "long":
            sl_price = min(sl_price,
                           price - max(price * MIN_SL_PERCENT / 100,
                                       price * 0.001))
            tp_price = max(tp_price, price + price * TP_PERCENT / 100)
        else:
            sl_price = max(sl_price,
                           price + max(price * MIN_SL_PERCENT / 100,
                                      price * 0.001))
            tp_price = min(tp_price, price - price * TP_PERCENT / 100)

        tp_str = exchange.price_to_precision(symbol, tp_price)
        sl_str = exchange.price_to_precision(symbol, sl_price)
        buy_sell = "buy" if side == "long" else "sell"

        log.info(
            f"Открываем {side} {symbol}: qty={qty}, маржа≈{margin_usdt:.2f}U, "
            f"плечо={LEVERAGE}x, TP={tp_str}, SL={sl_str}"
        )

        order = exchange.create_market_order(
            symbol, buy_sell, qty,
            params={"takeProfit": float(tp_str),
                    "stopLoss": float(sl_str)}
        )
        entry_price = float(order.get("average", price))
        log.info(f"{side.upper()} открыт: {qty} {symbol} @ ~{entry_price:.8f}")
        return entry_price, qty
    except Exception as e:
        log.error(f"Ошибка открытия {side}: {e}")
        return None, None


def закрыть_позицию_с_подтверждением(symbol: str,
                                    qty: float,
                                    side: str) -> bool:
    """Объективное закрытие позиции с повторными попытками."""
    close_side = "sell" if side == "long" else "buy"
    for attempt in range(3):
        try:
            exchange.create_market_order(symbol, close_side, qty,
                                        params={"reduceOnly": True})
            time.sleep(3)
            positions = exchange.fetch_positions([symbol])
            active = [p for p in positions
                      if float(p.get("contracts", 0) or 0) > 0
                      and p.get("side") == side]
            if not active:
                log.info(f"Позиция {symbol} закрыта успешно")
                return True
            log.warning(f"Позиция {symbol} не закрылась, попытка {attempt+1}")
            time.sleep(2)
        except Exception as e:
            log.warning(f"Попытка {attempt+1} закрыть {symbol} не удалась: {e}")
            time.sleep(2)
    log.error(f"Не удалось закрыть {symbol} после 3 попыток")
    return False


def проверить_signal_exit(symbol: str, side: str) -> bool:
    """Выход по сигналу (Supertrend + Range‑Filter)."""
    if not SIGNAL_EXIT_ENABLED:
        return False
    try:
        raw = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA, limit=50)
        if len(raw) < 30:
            return False
        df = pd.DataFrame(raw,
                          columns=["ts", "o", "h", "l", "c", "v"])
        st_up, st_down = calc_supertrend(df)
        _, _, _, rf_up, rf_down = calc_range_filter(df)
        return bool(st_down.iloc[-1] and rf_down.iloc[-1]) if side == "long" \
            else bool(st_up.iloc[-1] and rf_up.iloc[-1])
    except Exception:
        return False


# ------------------------------------------------------------
# 📈 Мониторинг позиции (WebSocket + fallback)
# ------------------------------------------------------------
class PositionMonitor:
    """
    Мониторинг позиции через WebSocket.
    Обновляет состояние в реальном времени, REST только как fallback.
    """
    def __init__(self):
        self.ws = None
        self.position_data = {}
        self.order_data = {}
        self._connected = False

    def start(self, symbol: str):
        sym = symbol.replace("/", "").replace(":USDT", "")
        try:
            self.ws = WebSocket(
                testnet=TESTNET_MODE,
                channel_type="private",
                api_key=os.getenv(
                    "BYBIT_TESTNET_API_KEY" if TESTNET_MODE else "BYBIT_API_KEY"),
                api_secret=os.getenv(
                    "BYBIT_TESTNET_API_SECRET"
                    if TESTNET_MODE else "BYBIT_API_SECRET"),
            )
            self.ws.position_stream(callback=self._on_position)
            self.ws.order_stream(callback=self._on_order)
            self._connected = True
            log.info(f"✅ WebSocket мониторинг запущен для {sym}")
        except Exception as e:
            log.warning(f"WebSocket недоступен, fallback на REST: {e}")
            self._connected = False

    def stop(self):
        try:
            if self.ws:
                self.ws.exit()
        except Exception:
            pass
        self._connected = False

    def _on_position(self, msg):
        for item in msg.get("data", []):
            sym = item.get("symbol", "")
            self.position_data[sym] = item

    def _on_order(self, msg):
        for item in msg.get("data", []):
            sym = item.get("symbol", "")
            oid = item.get("orderId", "")
            self.order_data[f"{sym}_{oid}"] = item

    def get_position(self, symbol: str) -> Optional[dict]:
        sym = symbol.replace("/", "").replace(":USDT", "")
        return self.position_data.get(sym)

    def is_position_closed(self, symbol: str) -> bool:
        data = self.get_position(symbol)
        if data is None:
            return False
        return float(data.get("size", 1)) == 0


position_monitor = PositionMonitor()


def мониторить_позицию(symbol, entry_price, qty, открыта_в,
                       sl_цена, tp_цена, side="long") -> str:
    """
    Основной цикл мониторинга позиции.
    Возвращает один из статусов: ``tp``, ``sl`` или ``timeout``.
    """
    deadline = открыта_в + TRADE_MAX_LIFETIME
    coin = symbol.split("/")[0]
    fee_buffer = 0.001
    breakeven_price = (entry_price *
                       (1 + BYBIT_FEE * 2 + fee_buffer)
                       if side == "long"
                       else entry_price *
                       (1 - BYBIT_FEE * 2 - fee_buffer))

    # Запускаем WebSocket‑мониторинг
    position_monitor.start(symbol)

    trailing_step = MIN_TRAILING_STEP / 100
    trailing_offset = MIN_TRAILING_OFFSET / 100

    try:
        # ── ATR‑подстройка для трейлинга
        raw = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA, limit=50)
        if len(raw) >= 30:
            df = pd.DataFrame(raw,
                              columns=["ts", "o", "h", "l", "c", "v"])
            atr_val = calc_atr(df, TRAILING_ATR_PERIOD).iloc[-1]
            atr_pct = (atr_val / entry_price) * 100
            trailing_step = max(MIN_TRAILING_STEP,
                               atr_pct * TRAILING_ATR_MULT) / 100
            trailing_offset = max(MIN_TRAILING_OFFSET,
                                 atr_pct * TRAILING_OFFSET_MULT) / 100

        rr_trigger_price = (
            entry_price + (tp_цена - entry_price) * RR_EXIT_TRIGGER
            if side == "long"
            else entry_price - (entry_price - tp_цена) * RR_EXIT_TRIGGER
        )

        log.info(f"Мониторинг {coin} {side} вход={entry_price:.8f} "
                 f"SL={sl_цена:.8f} TP={tp_цена:.8f}")

        текущий_sl = sl_цена
        пиковая_цена = entry_price
        be_done = False
        trailing_активен = False

        poll_interval = 5 if position_monitor._connected else 15

        while True:
            now = time.time()
            if now >= deadline:
                log.warning("Дедлайн — принудительное закрытие")
                закрыть_позицию_с_подтверждением(symbol, qty, side)
                return "timeout"

            time.sleep(poll_interval)

            # ------------------------------------------------------------------
            # 1️⃣ Проверка закрытия через WebSocket
            # ------------------------------------------------------------------
            if (position_monitor._connected and
                    position_monitor.is_position_closed(symbol)):
                log.info(f"[WS] Позиция {coin} закрыта биржей")
                cur_price = exchange.fetch_ticker(symbol)["last"]
                hit_tp = (cur_price >= entry_price *
                          (1 + TP_PERCENT / 100 * 0.7)) if side == "long" \
                    else (cur_price <= entry_price *
                          (1 - TP_PERCENT / 100 * 0.7))
                return "tp" if (hit_tp or be_done) else "sl"

            # ------------------------------------------------------------------
            # 2️⃣ REST‑fallback (текущая цена, размер позиции, P&L)
            # ------------------------------------------------------------------
            positions = exchange.fetch_positions([symbol])
            active = [p for p in positions
                      if float(p.get("contracts", 0) or 0) > 0
                      and p.get("side") == side]

            if not active:
                cur_price = exchange.fetch_ticker(symbol)["last"]
                hit_tp = (cur_price >= entry_price *
                          (1 + TP_PERCENT / 100 * 0.7)) if side == "long" \
                    else (cur_price <= entry_price *
                          (1 - TP_PERCENT / 100 * 0.7))
                return "tp" if (hit_tp or be_done) else "sl"

            pos = active[0]
            cur_price = exchange.fetch_ticker(symbol)["last"]
            qty_actual = abs(float(pos.get("contracts", 0) or 0))
            pnl_real = float(pos.get("unrealizedPnl", 0) or 0)
            pnl_pct = ((cur_price - entry_price) / entry_price * 100
                       if side == "long"
                       else (entry_price - cur_price) / entry_price * 100)
            time_left = int(deadline - now)

            # ------------------------------------------------------------------
            # 3️⃣ BE‑логика (перевод в безубыток)
            # ------------------------------------------------------------------
            if (PARTIAL_BE_ENABLED and not be_done and
                    pnl_pct >= PARTIAL_BE_PROFIT):
                mark = exchange.fetch_ticker(symbol).get("mark_price", cur_price)
                if side == "long" and breakeven_price < mark * 0.9995:
                    if обновить_sl_на_бирже(symbol, breakeven_price, side):
                        текущий_sl = breakeven_price
                        be_done = True
                        log.info(f"🎯 SL → БЕЗУБЫТОК: {breakeven_price:.8f}")

            # ------------------------------------------------------------------
            # 4️⃣ Трейлинг‑стоп
            # ------------------------------------------------------------------
            if not trailing_активен and be_done:
                trailing_активен = (cur_price >= rr_trigger_price
                                   if side == "long"
                                   else cur_price <= rr_trigger_price)
                if trailing_активен:
                    log.info(f"🚀 ТРЕЙЛИНГ АКТИВИРОВАН @ {cur_price:.8f}")

            if trailing_активен and be_done and pnl_pct >= MIN_PROFIT_FOR_TRAIL:
                if side == "long" and cur_price > пиковая_цена:
                    пиковая_цена = cur_price
                    new_sl = пиковая_цена * (1 - trailing_offset)
                    if new_sl > текущий_sl and обновить_sl_на_бирже(symbol,
                                                             new_sl, side):
                        текущий_sl = new_sl
                elif side == "short" and cur_price < пиковая_цена:
                    пиковая_цена = cur_price
                    new_sl = пиковая_цена * (1 + trailing_offset)
                    if new_sl < текущий_sl and обновить_sl_на_бирже(symbol,
                                                             new_sl, side):
                        текущий_sl = new_sl

            # ------------------------------------------------------------------
            # 5️⃣ Signal‑Exit (Supertrend + Range‑Filter)
            # ------------------------------------------------------------------
            if (SIGNAL_EXIT_ENABLED and be_done and
                    проверить_signal_exit(symbol, side)):
                log.info("Signal Exit: разворот — закрываем")
                закрыть_позицию_с_подтверждением(symbol, qty_actual, side)
                return "tp" if pnl_pct > 0 else "sl"

            log.info(
                f"[{coin}] {cur_price:.4f} P&L={pnl_pct:+.2f}% ({pnl_real:+.4f}U) "
                f"SL={текущий_sl:.4f} BE={be_done} "
                f"Trail={trailing_активен} дед={time_left}s"
            )

    except Exception as e:
        log.warning(f"Ошибка в цикле мониторинга: {e}")

    finally:
        position_monitor.stop()

    return "sl"


# ------------------------------------------------------------
# 📈 Пост‑трейд аналитика и статистика
# ------------------------------------------------------------
def применить_ai_корректировку(score: int, symbol: str) -> int:
    """Корректирует классический скор на основе AI‑соотношения."""
    ai = получить_bybit_ai(symbol)
    if not ai["available"]:
        return score
    long_r = ai["long_ratio"]
    signal = ai["signal"]
    log.info(f"Bybit ratio: long={long_r:.1%} "
             f"short={ai['short_ratio']:.1%} сигнал={signal}")
    if signal == "bullish":
        return min(100, score + 5)
    elif signal == "bearish":
        return max(0, score - 15)
    return score


def пост_трейд_анализ(запись: dict):
    """Логирование результатов сделки."""
    r = запись["результат"]
    sym = запись["symbol"]
    pnl = запись.get("pnl_usdt", 0)
    dur = запись.get("duration_min", 0)
    знак = "✅" if r == "tp" else ("❌" if r == "sl" else "⏰")
    mqs = запись.get("mqs", 0)
    mqs_emoji = запись.get("mqs_emoji", "❓")
    mqs_description = запись.get("mqs_description", "НЕИЗВЕСТНО")

    log.info("")
    log.info("━" * 60)
    log.info(
        f"📋 ПОСТ‑ТРЕЙД: {sym.split(':')[0]} {знак} {r.upper()} "
        f"P&L: {pnl:+.4f} USDT Длит: {dur:.1f} мин"
    )
    log.info(
        f"Классический скор: {запись.get('score', '?')}/100 | "
        f"MQS: {mqs_emoji} {mqs:.1f}/100 → {mqs_description}"
    )
    log.info("━" * 60)
    log.info("")


def загрузить_историю() -> List[dict]:
    """Чтение истории сделок из JSON."""
    if not os.path.exists(TRADES_FILE):
        return []
    try:
        with open(TRADES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def сохранить_сделку(запись: dict):
    """Запись сделки в историю."""
    история = загрузить_историю()
    история.append(запись)
    try:
        with open(TRADES_FILE, "w", encoding="utf-8") as f:
            json.dump(история, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        log.warning(f"Не удалось сохранить сделку: {e}")


def загрузить_состояние():
    """Загрузка состояния бота (счётчики, статистика)."""
    global stats
    if not os.path.exists(STATE_FILE):
        return False
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        for key in stats:
            if key in saved:
                stats[key] = saved[key]
        log.info(f"Состояние восстановлено из {STATE_FILE}")
        return True
    except Exception as e:
        log.warning(f"Не удалось загрузить состояние: {e}")
        return False


def сохранить_состояние():
    """Сохранение текущего состояния бота."""
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        log.warning(f"Не удалось сохранить состояние: {e}")


def баланс_usdt() -> float:
    """Свободный (free) баланс USDT."""
    try:
        b = exchange.fetch_balance({"type": "linear"})
        return float(b.get("USDT", {}).get("free", 0.0))
    except Exception as e:
        log.warning(f"Ошибка получения баланса: {e}")
        return 0.0


def полный_баланс_usdt() -> float:
    """Общий баланс (free + locked)."""
    try:
        b = exchange.fetch_balance({"type": "linear"})
        total = float(b.get("USDT", {}).get("total", 0.0))
        return total if total > 0 else баланс_usdt()
    except Exception as e:
        log.warning(f"Ошибка получения полного баланса: {e}")
        return баланс_usdt()


def получить_позиции() -> List[dict]:
    """Список открытых позиций."""
    try:
        positions = exchange.fetch_positions()
        return [p for p in positions
                if float(p.get("contracts", 0) or 0) > 0]
    except Exception as e:
        log.warning(f"Ошибка получения позиций: {e}")
        return []


def обновить_начало_дня(баланс: float):
    """Обновление дневного баланса в начале нового дня."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if stats["дата_дня"] != today:
        stats["дата_дня"] = today
        stats["баланс_начало_дня"] = баланс
        log.info(f"Новый торговый день. Баланс: {баланс:.2f} USDT")
        сохранить_состояние()


def превышен_дневной_лимит() -> bool:
    """Проверка дневного лимита убытков."""
    start = stats.get("баланс_начало_дня", 0.0)
    if start <= 0:
        return False
    current = полный_баланс_usdt()
    loss_pct = (start - current) / start * 100
    if loss_pct >= DAILY_LOSS_LIMIT_PCT:
        log.warning(f"Дневной лимит убытков: -{loss_pct:.1f}% "
                    f"(лимит {DAILY_LOSS_LIMIT_PCT}%)")
        return True
    return False


def торговать_разрешено_по_времени() -> bool:
    """Проверка временного ограничения (SESSION_FILTER)."""
    if not SESSION_FILTER_ENABLED:
        return True
    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour
    if SESSION_BLOCK_START < SESSION_BLOCK_END:
        blocked = SESSION_BLOCK_START <= hour < SESSION_BLOCK_END
    else:
        blocked = hour >= SESSION_BLOCK_START or hour < SESSION_BLOCK_END
    if blocked:
        log.info(f"Session Filter: час {hour} UTC заблокирован")
    return not blocked

# ------------------------------------------------------------
# 📊 Статистика индикаторов
# ------------------------------------------------------------
def загрузить_статистику_индикаторов() -> dict:
    if not os.path.exists(INDICATOR_STATS_FILE):
        return {}
    try:
        with open(INDICATOR_STATS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def сохранить_статистику_индикаторов(stats_data: dict):
    try:
        with open(INDICATOR_STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(stats_data, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        log.warning(f"Не удалось сохранить статистику индикаторов: {e}")

def обновить_статистику_индикаторов(запись_сделки: dict):
    stats_data = загрузить_статистику_индикаторов()
    details = запись_сделки.get("details", {})
    результат = запись_сделки.get("результат", "")
    is_win = (результат == "tp")

    def check_rsi(v): return 25 <= float(v) <= 42
    def check_rsi_1h(v): return float(v) < 55
    def check_macd(v): return v == "бычий"
    def check_range_filter(v): return v == "вверх"
    def check_supertrend(v): return v == "вверх"
    def check_hull(v): return v == "вверх"
    def check_trend_1h(v): return v == "бычий"
    def check_adx(v): return float(v) > 25
    def check_stoch_k(v): return float(v) < 25
    def check_volume_ratio(v): return float(v) > 1.5
    def check_sr_signal(v): return "поддержки" in str(v)
    def check_bayes_prob(v): return float(v) > 0.6
    def check_quant_score(v): return float(v) > 50
    def check_order_flow_score(v): return float(v) > 50

    индикаторы = {
        "rsi": check_rsi,
        "rsi_1h": check_rsi_1h,
        "macd": check_macd,
        "range_filter": check_range_filter,
        "supertrend": check_supertrend,
        "hull": check_hull,
        "тренд_1h": check_trend_1h,
        "adx": check_adx,
        "stoch_k": check_stoch_k,
        "объём_ratio": check_volume_ratio,
        "sr_signal": check_sr_signal,
        "bayes_prob": check_bayes_prob,
        "quant_score": check_quant_score,
        "order_flow_score": check_order_flow_score,
    }

    for инд, условие in индикаторы.items():
        значение = details.get(инд)
        if значение is None:
            continue
        try:
            is_bullish = условие(значение)
        except:
            continue
        if инд not in stats_data:
            stats_data[инд] = {"bullish": {"total": 0, "wins": 0}, "bearish": {"total": 0, "wins": 0}}
        if is_bullish:
            stats_data[инд]["bullish"]["total"] += 1
            if is_win:
                stats_data[инд]["bullish"]["wins"] += 1
        else:
            stats_data[инд]["bearish"]["total"] += 1
            if is_win:
                stats_data[инд]["bearish"]["wins"] += 1
    сохранить_статистику_индикаторов(stats_data)

def отчёт_по_индикаторам():
    stats_data = загрузить_статистику_индикаторов()
    if not stats_data:
        log.info("Статистика индикаторов пуста")
        return
    log.info("")
    log.info("=" * 70)
    log.info("📈 ЭФФЕКТИВНОСТЬ ИНДИКАТОРОВ (накопленная)")
    log.info(f"{'Индикатор':<18} {'🟢Бычий WR%':>11}  {'n':>4}  {'🔴Медвежий WR%':>14}  {'n':>4}  {'Разница':>8}")
    log.info(" " + "─" * 70)
    for инд, данные in stats_data.items():
        b_total, b_wins = данные["bullish"]["total"], данные["bullish"]["wins"]
        be_total, be_wins = данные["bearish"]["total"], данные["bearish"]["wins"]
        b_wr = (b_wins / b_total * 100) if b_total > 0 else 0
        be_wr = (be_wins / be_total * 100) if be_total > 0 else 0
        diff = b_wr - be_wr
        знак = "▲" if diff > 5 else ("▼" if diff < -5 else "≈")
        log.info(f"{инд:<18}  {b_wr:>9.1f}%  {b_total:>4}  {be_wr:>12.1f}%  {be_total:>4}  {знак}{diff:>+7.1f}%")
    log.info("=" * 70)
    log.info("")
    try:
        with open("indicators_analysis.json", "w", encoding="utf-8") as f:
            json.dump(stats_data, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        log.warning(f"Не удалось сохранить отчёт по индикаторам: {e}")

# ------------------------------------------------------------
# 📉 Метрики стратегии
# ------------------------------------------------------------
def рассчитать_метрики(сделки: List[dict]) -> dict:
    if len(сделки) < 5:
        return {}
    pnls = [t['pnl_usdt'] for t in сделки]
    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    max_dd = abs(min(drawdowns))
    max_dd_pct = (max_dd / max(1, cumulative[-1] + max_dd)) * 100 if cumulative[-1] != 0 else 0
    sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252) if len(pnls) > 1 and np.std(pnls) != 0 else 0
    neg_returns = [p for p in pnls if p < 0]
    sortino = np.mean(pnls) / np.std(neg_returns) * np.sqrt(252) if neg_returns and np.std(neg_returns) != 0 else 0
    total_return = cumulative[-1]
    years = len(сделки) / 252
    annual_return = total_return / years if years > 0 else total_return
    calmar = annual_return / (max_dd if max_dd > 0 else 1)
    recovery = cumulative[-1] / (max_dd if max_dd > 0 else 1)
    return {
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "calmar_ratio": round(calmar, 2),
        "max_drawdown_usdt": round(max_dd, 2),
        "max_drawdown_pct": round(max_dd_pct, 1),
        "recovery_factor": round(recovery, 2),
        "total_trades": len(сделки),
        "winrate": round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 1),
        "avg_win": round(np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0, 2),
        "avg_loss": round(abs(np.mean([p for p in pnls if p < 0])) if any(p < 0 for p in pnls) else 0, 2),
    }

def быстрый_walk_forward(история: List[dict], window: int = 50, step: int = 10) -> dict:
    if len(история) < window * 2:
        return {"стабильность": False, "рекомендация": "Недостаточно данных"}
    positive_windows = 0
    total_windows = 0
    for i in range(0, len(история) - window, step):
        out_sample = история[i+window:i+window+step]
        if len(out_sample) < 5:
            continue
        pnl_out = sum(t['pnl_usdt'] for t in out_sample)
        if pnl_out > 0:
            positive_windows += 1
        total_windows += 1
    if total_windows == 0:
        return {"стабильность": False, "рекомендация": "Недостаточно данных"}
    stability = positive_windows / total_windows > 0.6
    return {
        "стабильность": stability,
        "положительные_окна": positive_windows,
        "всего_окон": total_windows,
        "рекомендация": "Стратегия робастна" if stability else "Стратегия нестабильна – нужна оптимизация"
    }

def сохранить_метрики(метрики: dict):
    try:
        with open(METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump(метрики, f, ensure_ascii=False, indent=2, default=str)
        log.info(f"Метрики сохранены в {METRICS_FILE}")
    except Exception as e:
        log.warning(f"Не удалось сохранить метрики: {e}")

def run_monte_carlo_simulation(trades: List[dict], simulations: int = MONTE_CARLO_SIMULATIONS,
                               days: int = MONTE_CARLO_DAYS) -> Dict[str, Any]:
    try:
        if len(trades) < 10:
            return {"valid": False, "message": "Недостаточно данных"}
        pnls = [t["pnl_usdt"] for t in trades]
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls)
        simulated_equity = []
        for _ in range(simulations):
            simulated_pnls = np.random.normal(mean_pnl, std_pnl, days)
            simulated_equity.append(np.cumsum(simulated_pnls))
        simulated_equity = np.array(simulated_equity)
        max_drawdowns = []
        for equity in simulated_equity:
            running_max = np.maximum.accumulate(equity)
            drawdown = equity - running_max
            max_drawdowns.append(np.min(drawdown))
        return {
            "valid": True,
            "simulations": simulations,
            "days": days,
            "percentile_5": float(np.percentile(simulated_equity[:, -1], 5)),
            "percentile_50": float(np.percentile(simulated_equity[:, -1], 50)),
            "percentile_95": float(np.percentile(simulated_equity[:, -1], 95)),
            "loss_probability": float(np.mean(simulated_equity[:, -1] < 0) * 100),
            "avg_max_drawdown": float(np.mean(max_drawdowns)),
            "worst_max_drawdown": float(np.min(max_drawdowns)),
            "mean_pnl": float(mean_pnl),
            "std_pnl": float(std_pnl),
        }
    except Exception as e:
        log.error(f"Ошибка Monte Carlo: {e}")
        return {"valid": False, "message": str(e)}

def calculate_correlation_matrix(symbols: List[str], window: int = 100) -> pd.DataFrame:
    try:
        prices = {}
        for symbol in symbols:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA, limit=window)
                if len(ohlcv) >= window:
                    df = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c", "v"])
                    prices[symbol] = df["c"].pct_change().dropna()
            except:
                continue
        if len(prices) < 2:
            return pd.DataFrame()
        df = pd.concat(prices, axis=1).dropna()
        return df.corr()
    except Exception as e:
        log.debug(f"Ошибка расчета корреляции: {e}")
        return pd.DataFrame()

def optimize_portfolio_allocation(symbols: List[str], total_risk: float = MAX_PORTFOLIO_RISK) -> Dict[str, float]:
    try:
        corr_matrix = calculate_correlation_matrix(symbols)
        if corr_matrix.empty:
            return {s: 1.0 / len(symbols) for s in symbols}

        volatilities = {}
        for symbol in symbols:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA, limit=100)
                if len(ohlcv) >= 50:
                    df = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c", "v"])
                    volatilities[symbol] = float(df["c"].pct_change().dropna().std())
            except:
                volatilities[symbol] = 0.01

        total_vol = sum(volatilities.values())
        if total_vol > 0:
            allocations = {s: (1 / (volatilities[s] + 1e-10)) / sum(1 / (v + 1e-10) for v in volatilities.values()) for s in symbols}
        else:
            allocations = {s: 1.0 / len(symbols) for s in symbols}

        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                if symbol1 in corr_matrix.index and symbol2 in corr_matrix.columns:
                    corr = abs(corr_matrix.loc[symbol1, symbol2])
                    if corr > CORRELATION_THRESHOLD:
                        allocations[symbol1] *= (1 - (corr - CORRELATION_THRESHOLD) * 0.5)
                        allocations[symbol2] *= (1 - (corr - CORRELATION_THRESHOLD) * 0.5)

        total = sum(allocations.values())
        if total > 0:
            allocations = {s: v / total * total_risk for s, v in allocations.items()}
        return allocations
    except Exception as e:
        log.debug(f"Ошибка оптимизации портфеля: {e}")
        return {s: total_risk / len(symbols) for s in symbols}

# ------------------------------------------------------------
# 📊 Предстартовая проверка (5 этапов)
# ------------------------------------------------------------
def этап_1_проверка_окружения() -> Tuple[bool, List[str]]:
    errors, warnings = [], []
    api_key, api_secret = os.getenv("BYBIT_API_KEY", ""), os.getenv(
        "BYBIT_API_SECRET", "")
    if TESTNET_MODE:
        api_key, api_secret = os.getenv("BYBIT_TESTNET_API_KEY", ""), os.getenv(
            "BYBIT_TESTNET_API_SECRET", "")
    if not api_key:
        errors.append("BYBIT_API_KEY не задан")
    elif len(api_key) < 10:
        errors.append("BYBIT_API_KEY слишком короткий")
    if not api_secret:
        errors.append("BYBIT_API_SECRET не задан")
    elif len(api_secret) < 10:
        errors.append("BYBIT_API_SECRET слишком короткий")
    return len(errors) == 0, errors + warnings


def этап_2_проверка_подключения() -> Tuple[bool, List[str]]:
    errors = []
    try:
        b = exchange.fetch_balance({"type": "linear"})
        usdt = float(b.get("USDT", {}).get("free", 0))
        if TESTNET_MODE:
            log.info(f"Testnet: Баланс = {usdt:.4f} USDT (тестовые средства)")
        else:
            if usdt < MIN_BALANCE:
                errors.append(f"Баланс {usdt:.2f} < {MIN_BALANCE} USDT")
            else:
                log.info(f"Подключение OK | Баланс: {usdt:.4f} USDT")
    except Exception as e:
        errors.append(f"Ошибка подключения: {e}")
    return len(errors) == 0, errors


def этап_3_проверка_конфигурации() -> Tuple[bool, List[str]]:
    errors, warnings = [], []
    rr = TP_PERCENT / SL_PERCENT
    if rr < 2.0:
        errors.append(f"RR {rr:.1f}:1 слишком низкий")
    elif rr < 2.5:
        warnings.append(f"RR {rr:.1f}:1 можно повысить")
    if LEVERAGE > 5:
        warnings.append(f"Плечо {LEVERAGE}x высокое")
    if MIN_SCORE < 40:
        errors.append(f"MIN_SCORE={MIN_SCORE} < 40 — слишком низкий")
    elif MIN_SCORE < 55:
        warnings.append(f"MIN_SCORE={MIN_SCORE} < 55 — пониженный")
    if BASE_RISK_PCT > 3.0:
        errors.append(f"BASE_RISK_PCT={BASE_RISK_PCT}% слишком высокий")
    if DAILY_LOSS_LIMIT_PCT > 5.0:
        warnings.append(f"DAILY_LOSS_LIMIT_PCT={DAILY_LOSS_LIMIT_PCT}% высокий")
    if ATR_SL_MULT < 1.5:
        errors.append(f"ATR_SL_MULT={ATR_SL_MULT} слишком мал")
    log.info(f"Конфигурация: TP={TP_PERCENT}% | SL={SL_PERCENT}% | RR={rr:.1f}:1")
    return len(errors) == 0, errors + warnings


def этап_4_проверка_рынка() -> Tuple[bool, List[str]]:
    errors, warnings = [], []
    test_symbols = SYMBOLS[:5]
    доступные = 0
    for sym in test_symbols:
        try:
            ticker = exchange.fetch_ticker(sym)
            if float(ticker["last"]) > 0:
                доступные += 1
        except Exception as e:
            warnings.append(f"Пара {sym} недоступна: {e}")
    if доступные == 0:
        errors.append("Ни одна тестовая пара не доступна")
    elif доступные < len(test_symbols) // 2:
        warnings.append(f"Доступно только {доступные}/{len(test_symbols)} пар")
    else:
        log.info(f"Рынок: {доступные}/{len(test_symbols)} тестовых пар доступны")
    return len(errors) == 0, errors + warnings


def этап_5_проверка_существующих_позиций() -> Tuple[bool, List[str]]:
    errors, warnings = [], []
    try:
        positions = exchange.fetch_positions()
        открытые = [p for p in positions if float(p.get("contracts", 0) or 0) > 0]
        if открытые:
            for p in открытые:
                warnings.append(
                    f"Уже открыта позиция: {p.get('symbol')} "
                    f"{p.get('side')} qty={p.get('contracts')}"
                )
        else:
            log.info("Открытых позиций нет — готов к торговле")
    except Exception as e:
        warnings.append(f"Не удалось проверить позиции: {e}")

    if os.path.exists(TRADES_FILE):
        try:
            with open(TRADES_FILE, "r", encoding="utf-8") as f:
                история = json.load(f)
            log.info(f"История: {len(история)} сделок найдено")
        except Exception:
            warnings.append(f"Файл {TRADES_FILE} повреждён")
    return len(errors) == 0, errors + warnings


def запустить_предстартовую_проверку() -> bool:
    log.info("")
    log.info("=" * 65)
    log.info("🔍 ПРЕДСТАРТОВАЯ ПРОВЕРКА (5 ЭТАПОВ)")
    log.info("=" * 65)

    этапы = [
        ("Этап 1: Окружение и API‑ключи", этап_1_проверка_окружения),
        ("Этап 2: Подключение к бирже", этап_2_проверка_подключения),
        ("Этап 3: Конфигурация бота", этап_3_проверка_конфигурации),
        ("Этап 4: Доступность рынка", этап_4_проверка_рынка),
        ("Этап 5: Существующие позиции", этап_5_проверка_существующих_позиций),
    ]

    все_ок, все_ошибки = True, []
    for название, функция in этапы:
        log.info(f"\n▶ {название}...")
        try:
            ок, сообщения = функция()
            for msg in сообщения:
                if msg.lower().startswith("warning") or \
                   "рекомендуется" in msg.lower() or \
                   "уже" in msg.lower():
                    log.warning(f"⚠️ {msg}")
                else:
                    log.error(f"❌ {msg}")
                    все_ошибки.append(f"[{название}] {msg}")
            if ок:
                log.info(f"✅ {название} — ПРОЙДЕН")
            else:
                log.error(f"❌ {название} — ПРОВАЛЕН")
                все_ок = False
        except Exception as e:
            log.error(f"💥 Исключение в {название}: {e}")
            все_ошибки.append(f"[{название}] Исключение: {e}")
            все_ок = False

    log.info("")
    log.info("=" * 65)
    if все_ок:
        log.info("✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ — БОТ ЗАПУСКАЕТСЯ")
    else:
        log.error("❌ КРИТИЧЕСКИЕ ОШИБКИ — БОТ НЕ ЗАПУСТИТСЯ")
        for err in все_ошибки:
            log.error(f"• {err}")
        log.error("ВАРИАНТЫ УСТРАНЕНИЯ: проверьте .env, баланс, параметры")
    log.info("=" * 65)
    log.info("")
    return все_ок


# ------------------------------------------------------------
# 📈 Форматирование и вывод итогового отчёта
# ------------------------------------------------------------
def печатать_отчёт():
    баланс = полный_баланс_usdt()
    старт = stats["депозит_старт"]
    дельта = баланс - старт
    чистый = stats["прибыль_usdt"] - stats["убыток_usdt"]
    pct = (дельта / старт * 100) if старт > 0 else 0
    всего = stats["сделок_всего"]
    tp_ = stats["тейкпрофит"]
    sl_ = stats["стоплосс"]
    wr = (tp_ / всего * 100) if всего > 0 else 0.0

    log.info("")
    log.info("=" * 65)
    log.info("📊 ОТЧЁТ ГИБРИДНОГО БОТА v22 SEMANTIC QUANTUM MATRIX CORE")
    log.info(f"Режим: {'TESTNET' if TESTNET_MODE else 'MAINNET'}")
    log.info(f"Баланс: {баланс:.2f} USDT ({дельта:+.2f} USDT / {pct:+.2f}%)")
    log.info(f"Сделок: {всего} TP={tp_} SL={sl_} Таймаут={stats['таймаут']}")
    log.info(f"WinRate: {wr:.1f}%")
    log.info(f"Прибыль/Убыток: {stats['прибыль_usdt']:.4f} / {stats['убыток_usdt']:.4f} USDT")
    log.info(f"Чистый P&L: {чистый:+.4f} USDT")
    log.info("=" * 65)
    log.info("")

    stats["последний_отчёт"] = time.time()
    сохранить_состояние()

    # --- Вывод статистики индикаторов ---
    отчёт_по_индикаторам()

    # --- Метрики стратегии ---
    история = загрузить_историю()
    if len(история) > 5:
        метрики = рассчитать_метрики(история)
        if метрики:
            log.info("📉 МЕТРИКИ СТРАТЕГИИ:")
            log.info(f"Sharpe: {метрики.get('sharpe_ratio',0)} "
                     f"Sortino: {метрики.get('sortino_ratio',0)} "
                     f"Calmar: {метрики.get('calmar_ratio',0)}")
            log.info(f"Max Drawdown: {метрики.get('max_drawdown_pct',0)}% "
                     f"Recovery: {метрики.get('recovery_factor',0)}")
            сохранить_метрики(метрики)

        if len(история) > 100:
            wf = быстрый_walk_forward(история)
            log.info(f"🔄 Walk‑Forward: {wf.get('рекомендация')} "
                     f"(окна: {wf.get('положительные_окна',0)}/"
                     f"{wf.get('всего_окон',0)})")

    # --- Monte‑Carlo симуляция ---
    if MONTE_CARLO_ENABLED and len(история) > 20:
        if time.time() - stats.get("monte_carlo_last_run", 0) > 3600:
            mc_result = run_monte_carlo_simulation(история)
            if mc_result["valid"]:
                stats["monte_carlo_last_run"] = time.time()
                log.info("🎲 MONTE CARLO СИМУЛЯЦИЯ:")
                log.info(f"5‑й перцентиль: {mc_result['percentile_5']:.2f} USDT")
                log.info(f"50‑й перцентиль: {mc_result['percentile_50']:.2f} USDT")
                log.info(f"95‑й перцентиль: {mc_result['percentile_95']:.2f} USDT")
                log.info(f"Вероятность убытка: {mc_result['loss_probability']:.1f}%")
                log.info(f"Средняя max‑просадка: {mc_result['avg_max_drawdown']:.2f} USDT")

# ------------------------------------------------------------
# ⚡ Экспресс‑анализ (быстрый скрининг)
# ------------------------------------------------------------
def получить_скор_экспресс(symbol: str) -> dict:
    ...


# ============================================================
# ML МОДУЛЬ (Marcos Lopez de Prado)
# ============================================================
class TradingModel:
    """Класс для машинного обучения в трейдинге."""

    def __init__(self, model_type: str = "RandomForest"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.features = [
            "rsi", "rsi_1h", "macd", "adx", "stoch_k", "volume_ratio",
            "price_change_5m", "price_change_15m", "price_change_1h",
            "spread_pct", "imbalance", "poc_distance",
            "mean_reversion_zscore", "momentum", "cointegration_zscore"
        ]
        self.trained = False
        self.last_retrain = 0
        self.accuracy = 0
        self.precision = 0

    def create_features(self, symbol: str, df_ta: pd.DataFrame, df_1h: pd.DataFrame,
                       order_flow_data: Dict, quant_data: Dict) -> Dict[str, float]:
        """Создает features для модели."""
        try:
            features = {}
            c_ta = df_ta["c"]
            c_1h = df_1h["c"]

            features["rsi"] = float(calc_rsi(c_ta).iloc[-1])
            features["rsi_1h"] = float(calc_rsi(c_1h).iloc[-1])
            ml, sl_macd, _ = calc_macd(c_ta)
            features["macd"] = float(ml.iloc[-1] - sl_macd.iloc[-1])
            adx, _, _ = calc_adx(df_ta)
            features["adx"] = float(adx.iloc[-1])
            k_ser, _ = calc_stochastic(df_ta)
            features["stoch_k"] = float(k_ser.iloc[-1])
            vol_avg = df_ta["v"].rolling(VOLUME_AVG_PERIOD).mean().iloc[-1]
            features["volume_ratio"] = float(df_ta["v"].iloc[-1] / (vol_avg + 1e-10))
            features["price_change_5m"] = float((c_ta.iloc[-1] - c_ta.iloc[-2]) / c_ta.iloc[-2] * 100)
            features["price_change_15m"] = float((c_ta.iloc[-1] - c_ta.iloc[-3]) / c_ta.iloc[-3] * 100)
            features["price_change_1h"] = float((c_1h.iloc[-1] - c_1h.iloc[-2]) / c_1h.iloc[-2] * 100)

            if order_flow_data.get("order_book", {}).get("valid"):
                features["spread_pct"] = order_flow_data["order_book"]["spread_pct"]
                features["imbalance"] = order_flow_data["order_book"]["imbalance"]
            else:
                features["spread_pct"] = 0
                features["imbalance"] = 0

            if order_flow_data.get("volume_profile", {}).get("valid"):
                cp = order_flow_data["volume_profile"]["current_price"]
                poc = order_flow_data["volume_profile"]["poc_price"]
                features["poc_distance"] = float((cp - poc) / poc * 100)
            else:
                features["poc_distance"] = 0

            if quant_data.get("details", {}).get("mean_reversion", {}).get("valid"):
                features["mean_reversion_zscore"] = quant_data["details"]["mean_reversion"]["zscore"]
            else:
                features["mean_reversion_zscore"] = 0

            if quant_data.get("details", {}).get("momentum", {}).get("valid"):
                features["momentum"] = quant_data["details"]["momentum"]["momentum"]
            else:
                features["momentum"] = 0

            coint_zscore = 0
            for pair_data in quant_data.get("details", {}).get("cointegration", {}).values():
                if pair_data.get("valid"):
                    coint_zscore = pair_data["zscore"]
                    break
            features["cointegration_zscore"] = float(coint_zscore)

            return features
        except Exception as e:
            log.debug(f"Ошибка создания features для {symbol}: {e}")
            return {f: 0 for f in self.features}

    def prepare_training_data(self, trades: List[dict]) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series]]:
        """Готовит данные для обучения модели."""
        try:
            X, y = [], []
            for trade in trades:
                if trade["score"] < ENTRY_CONFIRM_MIN_SCORE:
                    continue
                details = trade.get("details", {})
                features = {
                    "rsi": float(details.get("rsi", 50)),
                    "rsi_1h": float(details.get("rsi_1h", 50)),
                    "macd": 0, "adx": float(details.get("adx", 20)),
                    "stoch_k": float(details.get("stoch_k", 50)),
                    "volume_ratio": float(details.get("объём_ratio", 1)),
                    "price_change_5m": 0, "price_change_15m": 0, "price_change_1h": 0,
                    "spread_pct": 0, "imbalance": 0, "poc_distance": 0,
                    "mean_reversion_zscore": 0, "momentum": 0, "cointegration_zscore": 0
                }
                X.append(features)
                y.append(1 if trade.get("результат", "sl") == "tp" else 0)

            if len(X) < ML_MIN_SAMPLES:
                return None, None, None, None

            X_df = pd.DataFrame(X)
            y_series = pd.Series(y)
            return train_test_split(X_df, y_series, test_size=0.2, random_state=42)
        except Exception as e:
            log.debug(f"Ошибка подготовки данных: {e}")
            return None, None, None, None

    def train(self, trades: List[dict]) -> bool:
        """Обучает модель."""
        try:
            X_train, X_test, y_train, y_test = self.prepare_training_data(trades)
            if X_train is None:
                log.info("Недостаточно данных для обучения ML модели")
                return False

            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            if self.model_type == "RandomForest":
                self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced")
            else:
                self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

            self.model.fit(X_train_scaled, y_train)
            y_pred = self.model.predict(X_test_scaled)
            self.accuracy = accuracy_score(y_test, y_pred)
            self.precision = precision_score(y_test, y_pred, zero_division=0)
            self.trained = True
            self.last_retrain = time.time()
            log.info(f"ML модель обучена: Accuracy={self.accuracy:.2f}, Precision={self.precision:.2f}")
            return True
        except Exception as e:
            log.error(f"Ошибка обучения ML модели: {e}")
            return False

    def predict(self, symbol: str, df_ta: pd.DataFrame, df_1h: pd.DataFrame,
               order_flow_data: Dict, quant_data: Dict) -> Dict[str, Any]:
        """Предсказывает сигнал для символа."""
        if not self.trained:
            return {"signal": "neutral", "probability": 0.5, "valid": False}
        try:
            features = self.create_features(symbol, df_ta, df_1h, order_flow_data, quant_data)
            for f in self.features:
                if f not in features:
                    features[f] = 0
            X = pd.DataFrame([features])
            X_scaled = self.scaler.transform(X)
            prediction = self.model.predict(X_scaled)[0]
            probability = self.model.predict_proba(X_scaled)[0][1]
            return {"signal": "buy" if prediction == 1 else "sell", "probability": float(probability), "valid": True, "features": features}
        except Exception as e:
            log.debug(f"Ошибка предсказания ML для {symbol}: {e}")
            return {"signal": "neutral", "probability": 0.5, "valid": False}

    def save_model(self, filepath: str = ML_MODEL_FILE) -> bool:
        """Сохраняет модель в файл."""
        try:
            import joblib
            joblib.dump({"model": self.model, "scaler": self.scaler, "features": self.features,
                         "trained": self.trained, "accuracy": self.accuracy, "precision": self.precision}, filepath)
            log.info(f"ML модель сохранена в {filepath}")
            return True
        except Exception as e:
            log.error(f"Ошибка сохранения ML модели: {e}")
            return False

    def load_model(self, filepath: str = ML_MODEL_FILE) -> bool:
        """Загружает модель из файла."""
        try:
            import joblib
            data = joblib.load(filepath)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.features = data["features"]
            self.trained = data["trained"]
            self.accuracy = data["accuracy"]
            self.precision = data["precision"]
            log.info(f"ML модель загружена из {filepath}")
            return True
        except Exception as e:
            log.error(f"Ошибка загрузки ML модели: {e}")
            return False

# ------------------------------------------------------------
# Создание экземпляра ML-модели
# ------------------------------------------------------------
ml_model = TradingModel(ML_MODEL_TYPE)

# ------------------------------------------------------------
# 📊 Сводный словарь статистики бота
# ------------------------------------------------------------
stats = {
    "запусков": 0,
    "сделок_всего": 0,
    "тейкпрофит": 0,
    "стоплосс": 0,
    "таймаут": 0,
    "прибыль_usdt": 0.0,
    "убыток_usdt": 0.0,
    "депозит_старт": 0.0,
    "баланс_начало_дня": 0.0,
    "дата_дня": "",
    "старт_время": "",
    "последний_отчёт": 0.0,
    "sl_streak": 0,
    "ml_trades_since_retrain": 0,
    "monte_carlo_last_run": 0,
}
# ------------------------------------------------------------

# ------------------------------------------------------------
# 🏁 Основная функция
# ------------------------------------------------------------
def main():
    global stats, ml_model

    # ----------------------- Загрузка модели ML -----------------------
    if ML_ENABLED:
        try:
            if os.path.exists(ML_MODEL_FILE):
                ml_model.load_model(ML_MODEL_FILE)
                log.info("ML модель загружена")
            else:
                log.info("ML модель не найдена – будет обучена позже")
        except Exception as e:
            log.warning(f"Не удалось загрузить ML модель: {e}")

    # ----------------------- Предстартовая проверка -----------------
    if not запустить_предстартовую_проверку():
        log.error("🛑 Бот остановлен из‑за ошибок предпроверки.")
        return

    # ----------------------- Загрузка/инициализация состояния ----------
    загрузить_состояние()
    stats["запусков"] += 1
    текущий_баланс = полный_баланс_usdt()
    if stats["депозит_старт"] <= 0:
        stats["депозит_старт"] = текущий_баланс
    if not stats["старт_время"]:
        stats["старт_время"] = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    обновить_начало_дня(текущий_баланс)
    сохранить_состояние()

    # ----------------------- Обучение ML (если необходимо) ------------
    if ML_ENABLED and not ml_model.trained:
        история = загрузить_историю()
        if len(история) >= ML_MIN_SAMPLES:
            ml_model.train(история)
            ml_model.save_model()

    # ----------------------- Информационный блок --------------------
    log.info("")
    log.info("=" * 65)
    log.info("🤖 ГИБРИДНЫЙ ФЬЮЧЕРСНЫЙ БОТ v22 SEMANTIC QUANTUM MATRIX CORE")
    log.info(f"Режим: {'TESTNET' if TESTNET_MODE else 'MAINNET'}")
    log.info(f"Плечо: {LEVERAGE}x | RR: {TP_PERCENT}/{SL_PERCENT} "
             f"({TP_PERCENT/SL_PERCENT:.1f}:1)")
    log.info(f"Баланс: {текущий_баланс:.4f} USDT")
    log.info(f"MIN_SCORE: {MIN_SCORE} | MQS_MIN_SCORE: {MQS_MIN_SCORE} | "
             f"Пар: {len(SYMBOLS)}")
    log.info(f"Квантовый анализ: {'ВКЛ' if QUANT_ENABLED else 'ВЫКЛ'}")
    log.info(f"Order Flow: {'ВКЛ' if ORDER_FLOW_ENABLED else 'ВЫКЛ'}")
    log.info(f"ML: {'ВКЛ' if ML_ENABLED and ml_model.trained else 'ВЫКЛ'}")
    log.info(f"Портфельная оптимизация: {'ВКЛ' if PORTFOLIO_OPTIMIZATION else 'ВЫКЛ'}")
    log.info("=" * 65)
    log.info("")

    # ----------------------- Список заблокированных символов ----------
    заблокированные = {}

    # ----------------------- Главный цикл -------------------------
    while True:
        try:
            # -- Периодический отчёт --
            if time.time() - stats["последний_отчёт"] >= REPORT_INTERVAL:
                печатать_отчёт()

            # -- Обновление баланса и проверка лимитов --
            баланс = полный_баланс_usdt()
            свободный = баланс_usdt()
            обновить_начало_дня(баланс)

            if свободный < MIN_BALANCE:
                log.warning(f"🛑 Свободный баланс {свободный:.2f} < {MIN_BALANCE} – пауза 10 мин")
                time.sleep(600)
                continue

            просадка = (stats["депозит_старт"] - баланс) / stats["депозит_старт"] * 100 \
                if stats["депозит_старт"] > 0 else 0
            if просадка > MAX_DRAWDOWN_PCT:
                log.warning(f"⛔ Просадка {просадка:.1f}% > {MAX_DRAWDOWN_PCT}% – пауза 2 ч")
                time.sleep(7200)
                continue

            if превышен_дневной_лимит():
                log.warning(f"⛔ Дневной лимит достигнут – пауза {DAILY_LOSS_PAUSE_SEC//60} мин")
                time.sleep(DAILY_LOSS_PAUSE_SEC)
                continue

            if not торговать_разрешено_по_времени():
                log.info("🕒 Трейдинг заблокирован по времени – пауза 5 мин")
                time.sleep(300)
                continue

            if stats.get("sl_streak", 0) >= SL_STREAK_LIMIT:
                log.warning(f"🧊 {SL_STREAK_LIMIT} SL подряд – cooldown {SL_STREAK_PAUSE//60} мин")
                stats["sl_streak"] = 0
                сохранить_состояние()
                time.sleep(SL_STREAK_PAUSE + SL_STREAK_EXTRA_PAUSE)
                continue

            # -- Если уже есть открытая позиция, ждём её закрытия --
            активные = получить_позиции()
            if активные:
                log.info(f"⏳ Открытые позиции: {[p['symbol'] for p in активные]} – ждём")
                time.sleep(60)
                continue

            log.info(f"── Сканирование {len(SYMBOLS)} пар (баланс={свободный:.2f}U) ──")
            scores = {}

            # ---------------------- Сканирование символов ----------------------
            for sym in SYMBOLS:
                # Снятие блокировок
                if sym in заблокированные:
                    if time.time() < заблокированные[sym]:
                        continue
                    else:
                        del заблокированные[sym]

                # Проверка Mark‑Price
                mp = get_mark_price_diff(sym)
                if mp["valid"] and not mp["safe"]:
                    log.info(f"⛔ {sym.split(':')[0]}: Mark‑Price отклонение {mp['diff_pct']:.2f}% — пропуск")
                    continue

                # Требуем бычий 4‑hour тренд
                if not тренд_4h_бычий(sym):
                    continue

                # Выбор режима скоринга
                if EXPRESS_MODE:
                    res = получить_скор_экспресс(sym)
                else:
                    res = получить_скор(sym)

                # Корректировка AI‑соотношением
                ai_score = применить_ai_корректировку(res["score"], sym)
                res["score_final"] = ai_score
                scores[sym] = res

                # Отладочный вывод
                det = res.get("details", {})
                log.debug(
                    f"{sym.split(':')[0]:12s} скор={ai_score:3.0f}/100 "
                    f"MQS={res['mqs']:.1f} rsi={det.get('rsi','?')} "
                    f"rf={det.get('range_filter','?')} st={det.get('supertrend','?')}"
                )

            # ---------------------- Формирование кандидатов ----------------------
            # Используем экспресс‑пороги, если включён экспресс‑режим
            min_score_threshold = EXPRESS_MIN_SCORE if EXPRESS_MODE else MIN_SCORE

            кандидаты = sorted(
                [(s, d) for s, d in scores.items()
                 if d.get("score_final", 0) >= min_score_threshold],
                key=lambda x: x[1]["score_final"],
                reverse=True
            )[:5]

            # ---------------------- Выбор лонга ----------------------
            выбран, фин_скор, цена, sr_info, side = None, 0, 0.0, {}, "long"
            for лучшая, данные in кандидаты:
                фин_скор = данные["score_final"]
                цена = данные["price"]
                sr_info = данные.get("sr", {})
                det = данные.get("details", {})

                # Пропуск, если слишком близко к сопротивлению
                if sr_info.get("near_resistance") and sr_info.get("dist_to_res_pct", 99) < SR_BLOCK_DIST_PCT:
                    log.info(f"⛔ {лучшая.split(':')[0]}: сопротивление "
                             f"{sr_info.get('dist_to_res_pct',0):.2f}% — пропуск")
                    continue

                # Перепроданность
                rsi_val = float(det.get("rsi", 50) or 50)
                if rsi_val > 65 and not sr_info.get("near_support"):
                    log.info(f"⚠️ {лучшая.split(':')[0]}: RSI={rsi_val:.1f} перекуплен — пропуск")
                    continue

                if MA_CROSSOVER_ENABLED and not det.get("ma_cross", True):
                    continue
                if not det.get("vol_spike_ok", True):
                    continue

                выбран = лучшая
                log.info(f"► Выбрана {лучшая.split(':')[0]} (лонг) скор={фин_скор} цена={цена:.8f}")
                break

            # ---------------------- Если лонг не найден – ищем шорт ----------------------
            if выбран is None:
                for sym in SYMBOLS:
                    if sym in заблокированные:
                        continue
                    if тренд_4h_медвежий(sym):
                        short_res = получить_скор_шорта(sym)
                        if short_res["score"] >= min_score_threshold:
                            det_sh = short_res.get("details", {})
                            if MA_CROSSOVER_ENABLED and not det_sh.get("ma_cross", True):
                                continue
                            if not det_sh.get("vol_spike_ok", True):
                                continue
                            log.info(f"🐻 Шорт‑кандидат: {sym.split(':')[0]} скор={short_res['score']}")
                            выбран, фин_скор, цена, sr_info, side = sym, short_res["score"], short_res["price"], short_res.get("sr", {}), "short"
                            break

            # ---------------------- Если кандидат не найден – пауза ----------------------
            if выбран is None:
                log.info(f"Нет подходящих кандидатов – ждём {SCAN_INTERVAL} сек")
                time.sleep(SCAN_INTERVAL)
                continue

            # ---------------------- Проверка MQS ----------------------
            mqs = scores[выбран]["mqs"]
            mqs_emoji = scores[выбран]["mqs_emoji"]
            mqs_descr = scores[выбран]["mqs_description"]
            log.info(f"[MQS] {mqs_emoji} {mqs:.1f}/100 → {mqs_descr} | Классический скор: {фин_скор}")

            mqs_threshold = EXPRESS_MQS_MIN if EXPRESS_MODE else MQS_MIN_SCORE
            if mqs < mqs_threshold:
                log.info(f"⛔ MQS={mqs:.1f} < {mqs_threshold} – пропуск {выбран.split(':')[0]}")
                time.sleep(SCAN_INTERVAL)
                continue

            # ---------------------- Расчёт TP/SL ----------------------
            atr_pt = 0.0
            try:
                raw_atr = exchange.fetch_ohlcv(выбран, TIMEFRAME_TA, limit=50)
                if len(raw_atr) >= 20:
                    df_atr = pd.DataFrame(raw_atr,
                                          columns=["ts", "o", "h", "l", "c", "v"])
                    atr_pt = float(calc_atr(df_atr, 14).iloc[-1])
            except Exception:
                pass

            if side == "long":
                sl_atr = atr_pt * ATR_SL_MULT if atr_pt > 0 else цена * SL_PERCENT / 100
                sl_pct = max(MIN_SL_PERCENT, min(MAX_SL_PERCENT, (sl_atr / цена) * 100))
                sl_price = цена * (1 - sl_pct / 100)

                tp_atr = atr_pt * ATR_TP_MULT if atr_pt > 0 else цена * TP_PERCENT / 100
                tp_pct = max(TP_PERCENT, (tp_atr / цена) * 100)
                tp_price = цена * (1 + tp_pct / 100)

                support = sr_info.get("support", sl_price)
                if support < sl_price and support > цена * 0.97:
                    sl_price = support * 0.998
            else:
                sl_atr = atr_pt * ATR_SL_MULT if atr_pt > 0 else цена * SL_PERCENT / 100
                sl_pct = max(MIN_SL_PERCENT, min(MAX_SL_PERCENT, (sl_atr / цена) * 100))
                sl_price = цена * (1 + sl_pct / 100)

                tp_atr = atr_pt * ATR_TP_MULT if atr_pt > 0 else цена * TP_PERCENT / 100
                tp_pct = max(TP_PERCENT, (tp_atr / цена) * 100)
                tp_price = цена * (1 - tp_pct / 100)

                resistance = sr_info.get("resistance", sl_price)
                if resistance > sl_price and resistance < цена * 1.03:
                    sl_price = resistance * 1.002

            sl_dist_pct = abs(цена - sl_price) / цена * 100
            real_rr = abs(tp_price - цена) / abs(цена - sl_price)
            log.info(f"📐 ATR={atr_pt/цена*100:.2f}% SL={sl_dist_pct:.2f}% RR={real_rr:.1f}:1")

            if real_rr < 1.999:
                log.warning(f"⛔ RR={real_rr:.1f}:1 < 2:1 – пропуск {выбран.split(':')[0]}")
                time.sleep(SCAN_INTERVAL)
                continue

            # ---------------------- Портфельная оптимизация ----------------------
            if PORTFOLIO_OPTIMIZATION:
                active_symbols = [p['symbol'] for p in получить_позиции()]
                all_symbols = active_symbols + [выбран]
                allocations = optimize_portfolio_allocation(all_symbols)
                max_risk_for_trade = allocations.get(выбран, MAX_PORTFOLIO_RISK) * баланс
            else:
                max_risk_for_trade = баланс * 0.95

            история_сделок = загрузить_историю()
            margin = рассчитать_размер_позиции(фин_скор, свободный,
                                               sl_dist_pct,
                                               история_сделок)
            margin = min(margin, max_risk_for_trade)

            if свободный < margin * 1.1:
                log.warning(f"⚠️ Баланс {свободный:.2f} < маржа {margin:.2f} – уменьшаем")
                margin = свободный * 0.8

            log.info(
                f"✅ ВХОД {side.upper()}: скор={фин_скор} | MQS={mqs_emoji} {mqs:.1f} | "
                f"SL={sl_price:.8f} | TP={tp_price:.8f} | маржа={margin:.2f}U"
            )

            # ---------------------- Подтверждение входа (если включено) ----------------------
            if ENTRY_CONFIRM_BARS > 0:
                if not подтвердить_вход(выбран, фин_скор, side):
                    log.info(f"⛔ Вход в {выбран} отменён по подтверждению")
                    time.sleep(30)
                    continue

            баланс_до = полный_баланс_usdt()
            время_входа = time.time()
            entry_price, qty = открыть_позицию(выбран, margin,
                                                tp_price, sl_price, side)

            if entry_price is None or qty is None:
                log.warning("Не удалось открыть позицию – пауза 30 сек")
                time.sleep(30)
                continue

            stats["сделок_всего"] += 1
            сохранить_состояние()

            # ---------------------- Мониторинг позиции ----------------------
            результат = "sl"
            try:
                результат = мониторить_позицию(
                    выбран, entry_price, qty, время_входа,
                    sl_price, tp_price, side
                )
            except Exception as monitor_err:
                log.error(f"💥 Краш мониторинга: {monitor_err}")
                закрыть_позицию_с_подтверждением(выбран, qty, side)
                результат = "sl"

            time.sleep(5)

            # ---------------------- Получение P&L ----------------------
            pnl_real = получить_реальный_pnl(выбран, время_входа)
            if pnl_real is None:
                balance_after = полный_баланс_usdt()
                pnl_real = balance_after - баланс_до
                log.debug("P&L получен через дельту баланса (fallback)")

            duration_min = (time.time() - время_входа) / 60

            # ---------------------- Обновление статистики ----------------------
            if результат == "tp":
                stats["тейкпрофит"] += 1
                stats["прибыль_usdt"] += max(0, pnl_real)
                stats["sl_streak"] = 0
                log.info(f"✅ TP: прибыль ≈{pnl_real:+.4f} USDT")
                заблокированные[выбран] = time.time() + SYMBOL_BLOCK_AFTER_TP * 60
                log.info(f"🔒 {выбран.split(':')[0]} заблокирован на {SYMBOL_BLOCK_AFTER_TP} мин")
            elif результат == "sl":
                stats["стоплосс"] += 1
                stats["убыток_usdt"] += abs(min(0, pnl_real))
                stats["sl_streak"] = stats.get("sl_streak", 0) + 1
                заблокированные[выбран] = time.time() + SYMBOL_BLOCK_AFTER_SL * 60
                log.warning(f"❌ SL: убыток ≈{pnl_real:+.4f} USDT (streak={stats['sl_streak']}/{SL_STREAK_LIMIT})")
            else:
                stats["таймаут"] += 1
                stats["убыток_usdt"] += abs(min(0, pnl_real))
                stats["sl_streak"] = 0
                заблокированные[выбран] = time.time() + SYMBOL_BLOCK_AFTER_TP * 60
                log.warning(f"⏰ Таймаут: P&L ≈{pnl_real:+.4f} USDT")

            # ---------------------- Запись сделки ----------------------
            запись = {
                "id": stats["сделок_всего"],
                "время_входа": datetime.fromtimestamp(время_входа).strftime("%d.%m.%Y %H:%M:%S"),
                "время_выхода": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                "duration_min": round(duration_min, 1),
                "symbol": выбран,
                "side": side,
                "score": фин_скор,
                "mqs": mqs,
                "mqs_emoji": mqs_emoji,
                "mqs_description": mqs_descr,
                "entry_price": entry_price,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "sl_dist_pct": round(sl_dist_pct, 3),
                "margin_usdt": margin,
                "leverage": LEVERAGE,
                "результат": результат,
                "pnl_usdt": round(pnl_real, 4),
                "rr_ratio": round(real_rr, 2),
                "details": scores.get(выбран, {}).get("details", {})
            }
            сохранить_сделку(запись)
            обновить_статистику_индикаторов(запись)
            пост_трейд_анализ(запись)
            сохранить_состояние()

            # ---------------------- Переобучение ML ----------------------
            if ML_ENABLED:
                stats["ml_trades_since_retrain"] += 1
                if stats["ml_trades_since_retrain"] >= ML_RETRAIN_INTERVAL:
                    история = загрузить_историю()
                    if len(история) >= ML_MIN_SAMPLES:
                        ml_model.train(история)
                        ml_model.save_model()
                    stats["ml_trades_since_retrain"] = 0

            log.info("Сделка завершена – пауза 60 сек")
            time.sleep(60)

        except Exception as e:
            log.error(f"Глобальная ошибка главного цикла: {e}", exc_info=True)
            time.sleep(60)


# ------------------------------------------------------------
# 🏁 Точка входа
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
