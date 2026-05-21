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
SYMBOLS = ["DOGE/USDT", "FLOKI/USDT", "PEPE/USDT"]  # мемкоины (SHIB и BONK убраны)
INITIAL_AMOUNT = 6          # сумма первого ордера (USDT) – для DOGE должно пройти
MARTINGALE_FACTOR = 1.35    # не используется при MAX_STEPS=0
MAX_STEPS = 0               # 0 = без мартингейла (только одна попытка)
TP_PERCENT = 1.0            # тейк-профит +1% от цены входа
SL_PERCENT = 1.2            # стоп-лосс -1.2%
TIMEFRAME_TA = "5m"         # для технического анализа
TIMEFRAME_TREND = "1h"
SCAN_INTERVAL = 300         # сканировать пары каждые 5 минут
MIN_SCORE = 30              # минимальный score (30)
USE_AI = False              # ИИ отключён (можно включить после получения ключа)
AI_CONFIDENCE_THRESHOLD = 0.6

# Файл для сохранения состояния незавершённой сделки
STATE_FILE = "trade_state.json"

# Подключение к Bybit (реальный аккаунт – без sandbox)
exchange = ccxt.bybit({
    'apiKey': os.getenv('BYBIT_API_KEY'),
    'secret': os.getenv('BYBIT_API_SECRET'),
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
})

# ========== ФУНКЦИИ ДЛЯ РАБОТЫ С СОСТОЯНИЕМ ==========
def save_trade_state(symbol, entry_price, qty, tp_price, sl_price, start_time):
    state = {
        "symbol": symbol,
        "entry_price": entry_price,
        "qty": qty,
        "tp_price": tp_price,
        "sl_price": sl_price,
        "start_time": start_time
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

def load_trade_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return None

def clear_trade_state():
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)

# ========== ТЕХНИЧЕСКИЙ АНАЛИЗ ==========
def get_technical_score(symbol):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME_TA, limit=200)
        if len(ohlcv) < 50:
            return 0
        df = pd.DataFrame(ohlcv, columns=['ts','o','h','l','c','v'])
        close = df['c']
        # RSI (14)
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

        # MACD (12,26,9)
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_score = 30 if macd.iloc[-1] > signal.iloc[-1] else 0

        # Тренд на 1h (EMA50 > EMA200)
        ohlcv_1h = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=200)
        df_h = pd.DataFrame(ohlcv_1h, columns=['ts','o','h','l','c','v'])
        ema50_h = df_h['c'].ewm(span=50).mean().iloc[-1]
        ema200_h = df_h['c'].ewm(span=200).mean().iloc[-1]
        trend_score = 20 if ema50_h > ema200_h else 0

        # Объём выше среднего за 20 свечей
        avg_volume = df['v'].rolling(20).mean().iloc[-1]
        last_volume = df['v'].iloc[-1]
        volume_score = 10 if last_volume > avg_volume * 1.2 else 0

        # Цена близка к локальному минимуму (за 20 свечей)
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

# ========== ИНТЕГРАЦИЯ ИИ (DeepSeek через OpenRouter) ==========
def ask_ai_for_signal(symbol, price, rsi, macd_bullish, trend_bullish, volume_surge):
    prompt = f"""
You are a professional crypto scalper. Based on the technical data, decide to BUY or WAIT.

Symbol: {symbol}
Current price: {price}
RSI (14): {rsi:.1f} (oversold <30, overbought >70)
MACD: {'bullish (line > signal)' if macd_bullish else 'bearish'}
1h trend (EMA50>EMA200): {trend_bullish}
Volume surge (>20% above avg): {volume_surge}

We use a scalping strategy: enter on pullback in uptrend, exit with +1% TP or -1.2% SL within 2 minutes.

Answer in JSON: {{"action": "buy" or "wait", "confidence": 0.0-1.0, "reasoning": "short"}}
"""
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek/deepseek-v4-flash:free",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 120
    }
    try:
        resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=10)
        if resp.status_code == 200:
            content = resp.json()['choices'][0]['message']['content']
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1:
                data = json.loads(content[start:end])
                return data.get('action', 'wait'), float(data.get('confidence', 0.5)), data.get('reasoning', '')
        print(f"AI HTTP {resp.status_code}: {resp.text[:100]}")
    except Exception as e:
        print(f"AI exception: {e}")
    return ('wait', 0.0, 'AI error')

# ========== ТОРГОВЫЕ ФУНКЦИИ ==========
def get_free_usdt_balance():
    bal = exchange.fetch_balance()
    return bal['free'].get('USDT', 0.0)

def get_coin_balance(symbol):
    coin = symbol.split('/')[0]
    bal = exchange.fetch_balance()
    return bal['free'].get(coin, 0)

def place_buy_order(symbol, amount_usdt):
    free_usdt = get_free_usdt_balance()
    if free_usdt < amount_usdt + 0.2:   # запас на комиссию
        print(f"⚠️ Недостаточно USDT: нужно {amount_usdt + 0.2:.2f}, доступно {free_usdt:.2f}")
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

# ========== ОСНОВНОЙ ЦИКЛ ==========
def main():
    print("🚀 Бот запущен (реальный счёт). Проверяю остатки...")

    # 1. При старте – закрываем все монеты (кроме USDT) и чистим файл состояния
    try:
        balance = exchange.fetch_balance()
        for coin, free in balance['free'].items():
            if coin != 'USDT' and free > 0:
                sym = f"{coin}/USDT"
                print(f"🔒 Найден остаток {free} {coin}, продаю по рынку")
                exchange.create_market_sell_order(sym, free)
                time.sleep(1)
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
    except Exception as e:
        print(f"Ошибка очистки баланса: {e}")

    # 2. Восстанавливаем незавершённую сделку (если есть) – продаём
    saved = load_trade_state()
    if saved:
        print("🔁 Обнаружена незавершённая сделка. Закрываю по рынку.")
        try:
            close_position(saved["symbol"], saved["qty"])
        except Exception as e:
            print(f"Ошибка закрытия сохранённой позиции: {e}")
        clear_trade_state()

    print(f"💼 Доступно USDT: {get_free_usdt_balance():.2f}")
    print(f"⚙️ Настройки: сумма ордера = {INITIAL_AMOUNT} USDT, TP={TP_PERCENT}%, SL={SL_PERCENT}%, таймаут=120с")

    while True:
        try:
            # 3. Выбираем лучшую пару по score
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

            # 4. Дополнительная проверка баланса USDT
            free_usdt = get_free_usdt_balance()
            if free_usdt < INITIAL_AMOUNT + 0.2:
                print(f"⚠️ Недостаточно USDT для сделки (нужно {INITIAL_AMOUNT+0.2:.2f}, доступно {free_usdt:.2f}). Ждём 60 сек...")
                time.sleep(60)
                continue

            # 5. ИИ-фильтр (опционально)
            if USE_AI:
                ohlcv = exchange.fetch_ohlcv(best_symbol, timeframe=TIMEFRAME_TA, limit=100)
                df = pd.DataFrame(ohlcv, columns=['ts','o','h','l','c','v'])
                close = df['c']
                # RSI
                delta = close.diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                avg_gain = gain.rolling(14).mean()
                avg_loss = loss.rolling(14).mean()
                rsi_val = 100 - (100 / (1 + (avg_gain.iloc[-1] / avg_loss.iloc[-1]))) if avg_loss.iloc[-1] != 0 else 50
                # MACD
                exp12 = close.ewm(span=12).mean()
                exp26 = close.ewm(span=26).mean()
                macd_line = exp12 - exp26
                signal_line = macd_line.ewm(span=9).mean()
                macd_bull = macd_line.iloc[-1] > signal_line.iloc[-1]
                # Тренд
                ohlcv_1h = exchange.fetch_ohlcv(best_symbol, timeframe='1h', limit=200)
                df_h = pd.DataFrame(ohlcv_1h, columns=['ts','o','h','l','c','v'])
                ema50_h = df_h['c'].ewm(span=50).mean().iloc[-1]
                ema200_h = df_h['c'].ewm(span=200).mean().iloc[-1]
                trend_bull = ema50_h > ema200_h
                # Объём
                avg_vol = df['v'].rolling(20).mean().iloc[-1]
                vol_surge = df['v'].iloc[-1] > avg_vol * 1.2

                action, conf, reason = ask_ai_for_signal(best_symbol, close.iloc[-1], rsi_val, macd_bull, trend_bull, vol_surge)
                print(f"🤖 AI: {action} (conf={conf:.2f}) - {reason}")
                if action != 'buy' or conf < AI_CONFIDENCE_THRESHOLD:
                    print("AI отклонил сделку, ждём")
                    time.sleep(SCAN_INTERVAL)
                    continue

            # 6. Покупка
            entry_price, qty = place_buy_order(best_symbol, INITIAL_AMOUNT)
            if entry_price is None:
                print("Не удалось купить, пропускаем цикл")
                time.sleep(30)
                continue

            # 7. Установка тейк-профита и сохранение состояния
            tp_price = entry_price * (1 + TP_PERCENT/100)
            sl_price = entry_price * (1 - SL_PERCENT/100)
            set_take_profit(best_symbol, entry_price, qty, TP_PERCENT)
            save_trade_state(best_symbol, entry_price, qty, tp_price, sl_price, time.time())

            # 8. Ожидание закрытия сделки (максимум 120 секунд)
            wait_time = 0
            trade_closed = False
            while wait_time < 120:
                time.sleep(10)
                coin_bal = get_coin_balance(best_symbol)
                if coin_bal < 1e-8:
                    print("✅ Тейк-профит сработал! Прибыль зафиксирована.")
                    clear_trade_state()
                    trade_closed = True
                    break
                cur_price = exchange.fetch_ticker(best_symbol)['last']
                loss_pct = (cur_price - entry_price) / entry_price * 100
                print(f"⌛ Цена {cur_price:.8f}, от входа {loss_pct:+.2f}%, до таймаута {120-wait_time}с")
                if loss_pct <= -SL_PERCENT:
                    print(f"❌ Стоп-лосс! Убыток {loss_pct:.2f}%")
                    close_position(best_symbol, qty)
                    clear_trade_state()
                    trade_closed = True
                    break
                wait_time += 10
            else:
                print("⏰ Таймаут 2 минуты, закрываю принудительно")
                close_position(best_symbol, qty)
                clear_trade_state()
                trade_closed = True

            print("✅ Серия завершена. Пауза 30 секунд...")
            time.sleep(30)

        except Exception as e:
            print(f"⚠️ Глобальная ошибка: {e}")
            if "Insufficient balance" in str(e):
                print("Ошибка баланса. Ждём 120 секунд...")
                time.sleep(120)
            else:
                time.sleep(60)

if __name__ == "__main__":
    main()
