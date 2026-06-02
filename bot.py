import os
import sys
import time
import logging
from pybit.unified_trading import HTTP

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    BYBIT_TESTNET_API_KEY = os.getenv("BYBIT_TESTNET_API_KEY")
    BYBIT_TESTNET_API_SECRET = os.getenv("BYBIT_TESTNET_API_SECRET")

    if not BYBIT_TESTNET_API_KEY or not BYBIT_TESTNET_API_SECRET:
        logger.error("❌ API keys missing")
        sys.exit(1)

    session = HTTP(testnet=True, api_key=BYBIT_TESTNET_API_KEY, api_secret=BYBIT_TESTNET_API_SECRET)
    logger.info("✅ Connected to Bybit Testnet")

    # 1. Получаем текущую цену BTCUSDC
    try:
        ticker = session.get_tickers(category="spot", symbol="BTCUSDC")
        price = float(ticker['result']['list'][0]['lastPrice'])
        logger.info(f"📈 Current BTCUSDC price: {price} USDC")
    except Exception as e:
        logger.error(f"❌ Failed to get price: {e}")
        sys.exit(1)

    # 2. Рассчитываем количество для покупки на 15 USDC (чуть выше минимальных 10)
    target_usdc = 15.0
    qty = round(target_usdc / price, 6)  # округляем до 6 знаков
    logger.info(f"🔢 Buying {qty} BTC for ~{target_usdc} USDC")

    # 3. Проверяем баланс USDC (опционально)
    try:
        wallet = session.get_wallet_balance(accountType="UNIFIED", coin="USDC")
        balance = float(wallet['result']['list'][0]['coin'][0]['walletBalance'])
        logger.info(f"💰 USDC balance: {balance}")
        if balance < target_usdc:
            logger.warning(f"⚠️ Low balance: {balance} USDC (need ~{target_usdc})")
    except Exception as e:
        logger.warning(f"⚠️ Could not check balance: {e}")

    # 4. Размещаем рыночный ордер
    try:
        order = session.place_order(
            category="spot",
            symbol="BTCUSDC",
            side="Buy",
            orderType="Market",
            qty=str(qty),
            timeInForce="GTC"
        )
        logger.info(f"✅ Market order placed! Response: {order}")
    except Exception as e:
        logger.error(f"❌ Order failed: {e}")

if __name__ == "__main__":
    main()
