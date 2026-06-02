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

    # 1. Получаем минимальную сумму ордера (minOrderAmt) для BTCUSDT
    try:
        instrument = session.get_instruments_info(category="spot", symbol="BTCUSDT")
        lot_size_filter = instrument['result']['list'][0]['lotSizeFilter']
        min_order_amt = float(lot_size_filter['minOrderAmt'])
        logger.info(f"📏 Minimal order amount for BTCUSDT: {min_order_amt} USDT")
    except Exception as e:
        logger.error(f"❌ Failed to get instrument info: {e}")
        sys.exit(1)

    # 2. Рассчитываем сумму сделки: минимум + 20%
    target_usdt = round(min_order_amt * 1.2, 2)
    logger.info(f"🎯 Order amount: {target_usdt} USDT")

    # 3. Проверяем баланс
    try:
        wallet = session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        balance = float(wallet['result']['list'][0]['coin'][0]['walletBalance'])
        logger.info(f"💰 USDT balance: {balance}")
        if balance < target_usdt:
            logger.error(f"❌ Insufficient balance ({balance} < {target_usdt})")
            sys.exit(1)
    except Exception as e:
        logger.warning(f"⚠️ Could not check balance: {e}")

    # 4. Получаем текущую цену
    try:
        ticker = session.get_tickers(category="spot", symbol="BTCUSDT")
        price = float(ticker['result']['list'][0]['lastPrice'])
        logger.info(f"📈 Current BTCUSDT price: {price} USDT")
    except Exception as e:
        logger.error(f"❌ Failed to get price: {e}")
        sys.exit(1)

    # 5. Рассчитываем количество (округление до 6 знаков)
    qty = round(target_usdt / price, 6)
    logger.info(f"🔢 Buying {qty} BTC (~{target_usdt} USDT)")

    # 6. Размещаем ордер
    try:
        order = session.place_order(
            category="spot",
            symbol="BTCUSDT",
            side="Buy",
            orderType="Market",
            qty=str(qty),
            timeInForce="GTC"
        )
        logger.info(f"✅ Market order placed! {order}")
    except Exception as e:
        logger.error(f"❌ Order failed: {e}")

if __name__ == "__main__":
    main()
