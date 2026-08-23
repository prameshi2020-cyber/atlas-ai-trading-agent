import asyncio
import logging

from app.database.models import save_ticker
from app.market.base import MarketDataProvider


logger = logging.getLogger(__name__)


class MarketCollector:

    def __init__(
        self,
        provider: MarketDataProvider,
        symbols: list[str],
        interval_seconds: int = 60,
    ) -> None:
        self.provider = provider
        self.symbols = symbols
        self.interval_seconds = interval_seconds

    async def collect_once(self) -> None:
        for symbol in self.symbols:
            try:
                ticker = await self.provider.get_ticker(symbol)
                save_ticker(ticker)

                logger.info(
                    "Collected %s: $%.2f",
                    ticker.symbol,
                    ticker.price,
                )

            except Exception:
                logger.exception(
                    "Failed to collect %s",
                    symbol,
                )

    async def run(self) -> None:
        while True:
            await self.collect_once()
            await asyncio.sleep(self.interval_seconds)