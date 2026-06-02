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

    # 1. Получаем баланс USDT
    try:
        wallet = session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        balance = float(wallet['result']['list'][0]['coin'][0]['walletBalance'])
        logger.info(f"💰 USDT balance: {balance}")
        
        if balance < 15:
            logger.error(f"❌ Insufficient USDT balance ({balance}). Please request test USDT from faucet: https://testnet.bybit.com/faucet")
            sys.exit(1)
    except Exception as e:
        logger.warning(f"⚠️ Could not check USDT balance: {e}")

    # 2. Получаем текущую цену BTCUSDT
    try:
        ticker = session.get_tickers(category="spot", symbol="BTCUSDT")
        price = float(ticker['result']['list'][0]['lastPrice'])
        logger.info(f"📈 Current BTCUSDT price: {price} USDT")
    except Exception as e:
        logger.error(f"❌ Failed to get price: {e}")
        sys.exit(1)

    # 3. Рассчитываем количество для покупки на 15 USDT
    target_usdt = 15.0
    qty = round(target_usdt / price, 6)  # BTC количество
    logger.info(f"🔢 Buying {qty} BTC for ~{target_usdt} USDT")

    # 4. Размещаем рыночный ордер
    try:
        order = session.place_order(
            category="spot",
            symbol="BTCUSDT",
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
