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
SYMBOLS = ["PEPE/USDT", "DOGE/USDT", "SHIB/USDT", "FLOKI/USDT", "BONK/USDT"]  # мемкоины
INITIAL_AMOUNT = 1.0          # для первого ордера (USDT) — при 2 шагах: 6 + 8.1 = 14.1 USDT
MARTINGALE_FACTOR = 1.35
MAX_STEPS = 0                 # 0 = без мартингейла, 1 или 2 для лёгкого
TP_PERCENT = 1.0              # тейк-профит от цены входа (для каждой сделки, если мартингейл выкл)
SL_PERCENT = 1.2              # стоп-лосс
TIMEFRAME_TA = "5m"           # для анализа (RSE, MACD) — 5 минут для частых сигналов
TIMEFRAME_TREND = "1h"        # для глобального тренда (EMA50/200)
SCAN_INTERVAL = 300           # сканировать пары каждые 5 минут
MIN_SCORE = 30                # минимальный score ТА для входа (0-100)
USE_AI = False                 # использовать ИИ как доп. фильтр
AI_CONFIDENCE_THRESHOLD = 0.6

# Подключение к Bybit
exchange = ccxt.bybit({
    'apiKey': os.getenv('BYBIT_API_KEY'),
    'secret': os.getenv('BYBIT_API_SECRET'),
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
})

# ========== ТЕХНИЧЕСКИЙ АНАЛИЗ ==========
def get_technical_score(symbol):
    """Возвращает score от 0 до 100, чем выше, тем лучше для покупки."""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME_TA, limit=200)
        if len(ohlcv) < 50:
            return 0
        df = pd.DataFrame(ohlcv, columns=['ts','o','h','l','c','v'])
        close = df['c']
        # 1. RSI (14) — перепроданность даёт очки
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
            rsi_score = 0    # перекуплен
        else:
            rsi_score = 10

        # 2. MACD (12,26,9) — бычье пересечение
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_score = 30 if macd.iloc[-1] > signal.iloc[-1] else 0

        # 3. EMA50 > EMA200 (тренд на 1h)
        ohlcv_1h = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=200)
        df_h = pd.DataFrame(ohlcv_1h, columns=['ts','o','h','l','c','v'])
        ema50_h = df_h['c'].ewm(span=50).mean().iloc[-1]
        ema200_h = df_h['c'].ewm(span=200).mean().iloc[-1]
        trend_score = 20 if ema50_h > ema200_h else 0

        # 4. Объём выше среднего за 20 свечей
        avg_volume = df['v'].rolling(20).mean().iloc[-1]
        last_volume = df['v'].iloc[-1]
        volume_score = 10 if last_volume > avg_volume * 1.2 else 0

        # 5. Цена близка к локальному минимуму (за последние 20 свечей)
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
        # Максимум: 40+30+20+10+10 = 110, но нормируем до 100
        return min(100, total)
    except Exception as e:
        print(f"Ошибка ТА для {symbol}: {e}")
        return 0

# ========== ИНТЕГРАЦИЯ OPENROUTER (ИИ) ==========
def ask_ai_for_signal(symbol, price, rsi, macd_bullish, trend_bullish, volume_surge):
    prompt = f"""
You are a crypto trading assistant. Based on technical analysis, decide to BUY or WAIT for {symbol}.
Current price: {price}
RSI: {rsi} (below 30 is oversold)
MACD bullish: {macd_bullish}
Trend (1h EMA50>EMA200): {trend_bullish}
Volume surge: {volume_surge}

We use a scalping strategy with tight stop-loss. Only answer in JSON: {{"action": "buy" or "wait", "confidence": 0.0-1.0, "reasoning": "short"}}
"""
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 120
    }
    try:
        resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=8)
        if resp.status_code == 200:
            content = resp.json()['choices'][0]['message']['content']
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1:
                data = json.loads(content[start:end])
                return data.get('action', 'wait'), float(data.get('confidence', 0.5)), data.get('reasoning', '')
    except Exception as e:
        print(f"AI error: {e}")
    return ('wait', 0.0, 'AI error')

# ========== ТОРГОВЫЕ ФУНКЦИИ ==========
def place_buy_order(symbol, amount_usdt):
    ticker = exchange.fetch_ticker(symbol)
    price = ticker['last']
    qty = amount_usdt / price
    order = exchange.create_market_buy_order(symbol, qty)
    print(f"Покупка {qty} {symbol} по ~{price}")
    return price, qty

def set_take_profit(symbol, price, qty, tp_percent):
    tp_price = price * (1 + tp_percent/100)
    exchange.create_limit_sell_order(symbol, qty, tp_price)
    print(f"Установлен TP {tp_price}")

def close_position(symbol, qty):
    exchange.create_market_sell_order(symbol, qty)
    print("Позиция закрыта по рынку")

def get_balance(symbol):
    coin = symbol.split('/')[0]
    bal = exchange.fetch_balance()
    return bal['free'].get(coin, 0)

# ========== ОСНОВНОЙ ЦИКЛ ==========
def main():
    print("Бот запущен. Депозит: 38 USDT. Режим: скальпинг + легкий мартингейл")
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

            print(f"Выбрана {best_symbol} со score {best_score}")

            # 2. Получаем свежие индикаторы для ИИ
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

            # 3. ИИ-фильтр
            if USE_AI:
                action, conf, reason = ask_ai_for_signal(best_symbol, close.iloc[-1], rsi_val, macd_bull, trend_bull, vol_surge)
                print(f"🤖 AI: {action} (conf={conf:.2f}) - {reason}")
                if action != 'buy' or conf < AI_CONFIDENCE_THRESHOLD:
                    print("AI отклонил сделку, ждём")
                    time.sleep(SCAN_INTERVAL)
                    continue

            # 4. Вход в сделку (мартингейл до MAX_STEPS)
            step = 0
            current_amount = INITIAL_AMOUNT
            entry_price = 0
            qty = 0
            active = True

            while active and step <= MAX_STEPS:
                # Покупаем
                entry_price, qty = place_buy_order(best_symbol, current_amount)
                set_take_profit(best_symbol, entry_price, qty, TP_PERCENT)

                # Ждём либо TP, либо SL, либо таймаут
                wait_time = 0
                while wait_time < 600:  # 10 минут максимум
                    time.sleep(10)
                    balance_coin = get_balance(best_symbol)
                    if balance_coin < 1e-8:
                        print("✅ TP сработал! Прибыль зафиксирована.")
                        active = False
                        break
                    cur_price = exchange.fetch_ticker(best_symbol)['last']
                    loss_pct = (cur_price - entry_price) / entry_price * 100
                    if loss_pct <= -SL_PERCENT:
                        print(f"❌ Стоп-лосс! Убыток {loss_pct:.2f}%")
                        close_position(best_symbol, qty)
                        step += 1
                        if step <= MAX_STEPS:
                            current_amount = current_amount * MARTINGALE_FACTOR
                            print(f"Мартингейл шаг {step}, следующая ставка {current_amount:.2f} USDT")
                        else:
                            print("Достигнут лимит шагов, серия закончена.")
                            active = False
                        break
                    wait_time += 10
                else:
                    # Таймаут
                    print("Таймаут, закрываем принудительно")
                    close_position(best_symbol, qty)
                    active = False

                if not active:
                    break

            # 5. Пауза перед новым сканированием
            print("Серия завершена. Пауза 30 секунд...")
            time.sleep(30)

        except Exception as e:
            print(f"Глобальная ошибка: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
