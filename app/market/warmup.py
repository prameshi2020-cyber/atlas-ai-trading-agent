from __future__ import annotations

import logging

from app.database.models import save_candles
from app.market.base import MarketDataProvider

logger = logging.getLogger(__name__)


class MarketWarmup:
    """
    Loads recent closed market candles into the local database.

    Warmup is a startup concern only. It does not perform trading,
    strategy evaluation, risk management, or execution.
    """

    def __init__(
        self,
        *,
        provider: MarketDataProvider,
        symbols: list[str],
        limit: int = 100,
    ) -> None:
        if provider is None:
            raise ValueError("provider is required")
        if not symbols:
            raise ValueError("symbols cannot be empty")
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")

        self.provider = provider
        self.symbols = symbols
        self.limit = limit

    async def warmup(self) -> dict[str, int]:
        """
        Fetch recent closed candles for every configured symbol.

        Returns:
            Mapping of symbol -> number of newly inserted candles.

        A failure for one symbol does not prevent other symbols
        from being warmed up.
        """
        results: dict[str, int] = {}

        for symbol in self.symbols:
            try:
                candles = await self.provider.get_historical_candles(
                    symbol=symbol,
                    timeframe="1m",
                    limit=self.limit,
                )

                inserted = save_candles(candles)
                results[symbol] = inserted

                logger.info(
                    "Warmup %s: fetched=%d inserted=%d",
                    symbol,
                    len(candles),
                    inserted,
                )

            except Exception:
                logger.exception(
                    "Warmup failed for %s",
                    symbol,
                )
                results[symbol] = 0

        return results
