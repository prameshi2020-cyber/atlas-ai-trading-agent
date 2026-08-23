import asyncio
import logging

from app.config.settings import settings
from app.database.models import initialize_database
from app.market.collector import MarketCollector
from app.market.coingecko import CoinGeckoProvider


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


async def main() -> None:
    print("=" * 60)
    print(f"{settings.app_name} AI Trading Agent")
    print(f"Environment: {settings.environment}")
    print("=" * 60)

    initialize_database()

    provider = CoinGeckoProvider()

    collector = MarketCollector(
        provider=provider,
        symbols=[
            "BTCUSDT",
            "ETHUSDT",
            "SOLUSDT",
        ],
        interval_seconds=60,
    )

    await collector.run()


if __name__ == "__main__":
    asyncio.run(main())