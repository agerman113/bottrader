import os
import sys
import time
import logging
from pybit.unified_trading import HTTP

# --- Настройка логирования для отслеживания работы бота на Railway ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def main():
    # --- 1. Получение API-ключей из переменных окружения Railway ---
    BYBIT_TESTNET_API_KEY = os.getenv("BYBIT_TESTNET_API_KEY")
    BYBIT_TESTNET_API_SECRET = os.getenv("BYBIT_TESTNET_API_SECRET")

    if not BYBIT_TESTNET_API_KEY or not BYBIT_TESTNET_API_SECRET:
        logger.error("❌ Ошибка: API-ключи не найдены в переменных окружения.")
        sys.exit(1)

    # --- 2. Подключение к Bybit Testnet через единый API ---
    try:
        # testnet=True - ключевой параметр для работы с тестовой сетью
        session = HTTP(
            testnet=True,
            api_key=BYBIT_TESTNET_API_KEY,
            api_secret=BYBIT_TESTNET_API_SECRET,
        )
        logger.info("✅ Успешное подключение к Bybit Testnet API")
    except Exception as e:
        logger.error(f"❌ Ошибка подключения к API: {e}")
        sys.exit(1)

    # --- 3. Проверка баланса перед сделкой ---
    try:
        wallet_balance = session.get_wallet_balance(accountType="UNIFIED", coin="USDC")
        usdc_balance = float(wallet_balance['result']['list'][0]['coin'][0]['walletBalance'])
        logger.info(f"💰 Баланс USDC на торговом счете: {usdc_balance}")
        if usdc_balance < 50:  # Минимальная сумма для сделки
            logger.warning(f"⚠️ Баланс USDC ({usdc_balance}) меньше 50. Сделка может быть отклонена биржей.")
    except Exception as e:
        logger.warning(f"⚠️ Не удалось проверить баланс: {e}")

    # --- 4. Размещение одного рыночного ордера ---
    try:
        order = session.place_order(
            category="spot",       # Торгуем на спотовом рынке
            symbol="BTCUSDC",     # Пара для торговли
            side="Buy",           # Покупка. Можно поменять на "Sell" для продажи
            orderType="Market",   # Рыночный ордер
            qty="0.0001",         # Количество BTC к покупке
            timeInForce="GTC"     # Ордер активен до отмены
        )
        logger.info(f"✅ Рыночный ордер успешно размещен! Детали: {order}")
    except Exception as e:
        logger.error(f"❌ Ошибка при размещении ордера: {e}")

    # Небольшая пауза, чтобы убедиться, что все логи успели отправиться
    time.sleep(2)

if __name__ == "__main__":
    main()
