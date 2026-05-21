import os
import time
import json
import requests
import ccxt
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ========== НАСТРОЙКИ ==========
SYMBOLS = ["DOGE/USDT", "FLOKI/USDT", "PEPE/USDT"]  # убрал SHIB и BONK (слишком дешёвые)
INITIAL_AMOUNT = 6          # увеличено до 12 USDT (чтобы гарантированно пройти минималку)
MARTINGALE_FACTOR = 1.35
MAX_STEPS = 0                # 0 = без мартингейла
TP_PERCENT = 1.0             # тейк-профит 1%
SL_PERCENT = 1.2             # стоп-лосс 1.2%
TIMEFRAME_TA = "5m"
TIMEFRAME_TREND = "1h"
SCAN_INTERVAL = 300
MIN_SCORE = 30
USE_AI = False
AI_CONFIDENCE_THRESHOLD = 0.6

# Подключение к Bybit (реальный аккаунт)
exchange = ccxt.bybit({
    'apiKey': os.getenv('BYBIT_API_KEY'),
    'secret': os.getenv('BYBIT_API_SECRET'),
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
})

# ========== ТЕХНИЧЕСКИЙ АНАЛИЗ (без изменений) ==========
def get_technical_score(symbol):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME_TA, limit=200)
        if len(ohlcv) < 50:
            return 0
        df = pd.DataFrame(ohlcv, columns=['ts','o','h','l','c','v'])
        close = df['c']
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        if current_rsi < 30:
            rsi_score = 40
        elif current_rsi < 40:
            rsi_score = 20
        elif current_rsi > 70:
            rsi_score = 0
        else:
            rsi_score = 10

        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_score = 30 if macd.iloc[-1] > signal.iloc[-1] else 0

        ohlcv_1h = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=200)
        df_h = pd.DataFrame(ohlcv_1h, columns=['ts','o','h','l','c','v'])
        ema50_h = df_h['c'].ewm(span=50).mean().iloc[-1]
        ema200_h = df_h['c'].ewm(span=200).mean().iloc[-1]
        trend_score = 20 if ema50_h > ema200_h else 0

        avg_volume = df['v'].rolling(20).mean().iloc[-1]
        last_volume = df['v'].iloc[-1]
        volume_score = 10 if last_volume > avg_volume * 1.2 else 0

        min_price = df['l'].rolling(20).min().iloc[-1]
        current_price = close.iloc[-1]
        distance_to_min = (current_price - min_price) / min_price * 100
        if distance_to_min < 0.5:
            proximity_score = 10
        elif distance_to_min < 1.5:
            proximity_score = 5
        else:
            proximity_score = 0

        total = rsi_score + macd_score + trend_score + volume_score + proximity_score
        return min(100, total)
    except Exception as e:
        print(f"Ошибка ТА для {symbol}: {e}")
        return 0

# ========== ТОРГОВЫЕ ФУНКЦИИ С ПРОВЕРКОЙ БАЛАНСА ==========
def get_free_usdt_balance():
    """Возвращает свободный баланс USDT в едином торговом аккаунте"""
    bal = exchange.fetch_balance()
    return bal['free'].get('USDT', 0.0)

def place_buy_order(symbol, amount_usdt):
    """Покупает по рынку, если хватает баланса"""
    free_usdt = get_free_usdt_balance()
    if free_usdt < amount_usdt:
        print(f"⚠️ Недостаточно USDT: нужно {amount_usdt}, доступно {free_usdt}")
        return None, None
    ticker = exchange.fetch_ticker(symbol)
    price = ticker['last']
    qty = amount_usdt / price
    order = exchange.create_market_buy_order(symbol, qty)
    print(f"✅ Покупка {qty:.8f} {symbol} по ~{price:.8f} (сумма {amount_usdt} USDT)")
    return price, qty

def set_take_profit(symbol, price, qty, tp_percent):
    tp_price = price * (1 + tp_percent/100)
    exchange.create_limit_sell_order(symbol, qty, tp_price)
    print(f"🎯 Установлен TP {tp_price:.8f} (+{tp_percent}%)")

def close_position(symbol, qty):
    exchange.create_market_sell_order(symbol, qty)
    print("🔒 Позиция закрыта по рынку")

def get_coin_balance(symbol):
    coin = symbol.split('/')[0]
    bal = exchange.fetch_balance()
    return bal['free'].get(coin, 0)

# ========== ОСНОВНОЙ ЦИКЛ ==========
def main():
    print("🚀 Бот запущен (реальный счёт).")
    print(f"Доступно USDT: {get_free_usdt_balance():.2f}")
    print(f"Настройки: сумма ордера = {INITIAL_AMOUNT} USDT, TP={TP_PERCENT}%, SL={SL_PERCENT}%")
    while True:
        try:
            # 1. Выбираем лучшую пару
            scores = {}
            for sym in SYMBOLS:
                score = get_technical_score(sym)
                scores[sym] = score
                print(f"{sym}: score = {score}")
            best_symbol = max(scores, key=scores.get)
            best_score = scores[best_symbol]
            if best_score < MIN_SCORE:
                print(f"Лучший score {best_score} ниже порога {MIN_SCORE}, ждём {SCAN_INTERVAL} сек")
                time.sleep(SCAN_INTERVAL)
                continue

            print(f"⭐ Выбрана {best_symbol} со score {best_score}")

            # 2. Проверка свободного баланса USDT
            free_usdt = get_free_usdt_balance()
            if free_usdt < INITIAL_AMOUNT:
                print(f"⚠️ Недостаточно USDT для сделки (нужно {INITIAL_AMOUNT}, доступно {free_usdt}). Ждём 60 сек...")
                time.sleep(60)
                continue

            # 3. Вход в сделку (без мартингейла, т.к. MAX_STEPS=0)
            entry_price, qty = place_buy_order(best_symbol, INITIAL_AMOUNT)
            if entry_price is None:
                print("Не удалось купить, пропускаем цикл")
                time.sleep(30)
                continue

            set_take_profit(best_symbol, entry_price, qty, TP_PERCENT)

            # 4. Мониторим сделку (ждём TP или SL)
            wait_time = 0
            trade_closed = False
            while wait_time < 600:  # 10 минут
                time.sleep(10)
                coin_balance = get_coin_balance(best_symbol)
                if coin_balance < 1e-8:
                    print("✅ Тейк-профит сработал! Прибыль зафиксирована.")
                    trade_closed = True
                    break
                cur_price = exchange.fetch_ticker(best_symbol)['last']
                loss_pct = (cur_price - entry_price) / entry_price * 100
                if loss_pct <= -SL_PERCENT:
                    print(f"❌ Стоп-лосс! Убыток {loss_pct:.2f}%")
                    close_position(best_symbol, qty)
                    trade_closed = True
                    break
                wait_time += 10

            if not trade_closed:
                print("⏰ Таймаут, закрываем позицию принудительно")
                close_position(best_symbol, qty)

            print("Серия завершена. Пауза 30 секунд...")
            time.sleep(30)

        except Exception as e:
            print(f"⚠️ Глобальная ошибка: {e}")
            # Если ошибка связана с балансом, не спамим
            if "Insufficient balance" in str(e):
                print("Ошибка баланса. Ждём 120 секунд...")
                time.sleep(120)
            else:
                time.sleep(60)

if __name__ == "__main__":
    main()
