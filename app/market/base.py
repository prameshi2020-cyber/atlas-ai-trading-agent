from abc import ABC, abstractmethod

from app.market.models import Ticker
from app.market.models import Candle


class MarketDataProvider(ABC):

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        """Return the latest ticker for a symbol."""
        raise NotImplementedError

    async def get_tickers(self, symbols: list[str]) -> list[Ticker]:
        """Return a list of tickers for the provided symbols.

        Default implementation calls `get_ticker` sequentially and
        returns only successful results. Providers may override with a
        batch implementation for efficiency.
        """
        results: list[Ticker] = []
        for s in symbols:
            try:
                t = await self.get_ticker(s)
                results.append(t)
            except Exception:
                # Don't let one failure stop the batch; caller will log
                continue

        return results

    async def get_candle(self, symbol: str, timeframe: str = "1m") -> Candle:
        """Return the latest candle for a symbol and timeframe.

        Providers should override with a proper implementation. Default
        raises NotImplementedError so callers can detect support.
        """
        raise NotImplementedError

    async def get_candles(self, symbols: list[str], timeframe: str = "1m") -> list[Candle]:
        """Return list of latest candles for provided symbols.

        Default implementation sequentially calls `get_candle` and
        returns successful results.
        """
        results: list[Candle] = []
        for s in symbols:
            try:
                c = await self.get_candle(s, timeframe)
                results.append(c)
            except Exception:
                continue

        return results