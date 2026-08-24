import asyncio
import logging

from app.config.settings import settings
from app.database.models import (
    initialize_database,
    initialize_candle_database,
)
from app.market.collector import MarketCollector
from app.market.binance import BinanceProvider


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


async def main() -> None:
    print("=" * 60)
    print("ATLAS AI Trading Agent")
    print(f"Environment: {settings.environment}")
    print("=" * 60)

    initialize_database()
    initialize_candle_database()

    provider = BinanceProvider()

    collector = MarketCollector(
        provider=provider,
        symbols=settings.symbols,
        interval_seconds=settings.market_poll_interval_seconds,
    )

    await collector.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nATLAS stopped.")