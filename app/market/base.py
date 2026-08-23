from abc import ABC, abstractmethod

from app.market.models import Ticker


class MarketDataProvider(ABC):

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        """Return the latest ticker for a symbol."""
        raise NotImplementedError