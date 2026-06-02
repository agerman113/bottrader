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
        logger.error("❌ API keys missing in environment variables")
        sys.exit(1)

    try:
        session = HTTP(testnet=True, api_key=BYBIT_TESTNET_API_KEY, api_secret=BYBIT_TESTNET_API_SECRET)
        logger.info("✅ Connected to Bybit Testnet")
    except Exception as e:
        logger.error(f"❌ Connection error: {e}")
        sys.exit(1)

    # Place a single market order
    try:
        order = session.place_order(
            category="spot",
            symbol="BTCUSDC",
            side="Buy",
            orderType="Market",
            qty="0.0001",
            timeInForce="GTC"
        )
        logger.info(f"✅ Order placed: {order}")
    except Exception as e:
        logger.error(f"❌ Order failed: {e}")

    time.sleep(2)

if __name__ == "__main__":
    main()
