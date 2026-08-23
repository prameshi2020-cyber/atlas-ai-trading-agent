from datetime import datetime, timezone

import httpx

from app.market.base import MarketDataProvider
from app.market.models import Ticker


class BinanceProvider(MarketDataProvider):

    BASE_URL = "https://api.binance.com"

    async def get_ticker(self, symbol: str) -> Ticker:
        symbol = symbol.upper()

        async with httpx.AsyncClient(
            base_url=self.BASE_URL,
            timeout=10.0,
        ) as client:

            response = await client.get(
                "/api/v3/ticker/24hr",
                params={"symbol": symbol},
            )

            response.raise_for_status()
            data = response.json()

        return Ticker(
            symbol=data["symbol"],
            price=float(data["lastPrice"]),
            price_change_24h=float(data["priceChangePercent"]),
            high_24h=float(data["highPrice"]),
            low_24h=float(data["lowPrice"]),
            volume_24h=float(data["volume"]),
            timestamp=datetime.now(timezone.utc),
        )