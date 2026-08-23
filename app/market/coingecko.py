from datetime import datetime, timezone

import httpx

from app.market.base import MarketDataProvider
from app.market.models import Ticker


class CoinGeckoProvider(MarketDataProvider):

    BASE_URL = "https://api.coingecko.com/api/v3"

    async def get_ticker(self, symbol: str) -> Ticker:
        symbol = symbol.upper()

        symbol_map = {
            "BTCUSDT": "bitcoin",
            "ETHUSDT": "ethereum",
            "SOLUSDT": "solana",
        }

        if symbol not in symbol_map:
            raise ValueError(
                f"Unsupported symbol for CoinGecko: {symbol}"
            )

        coin_id = symbol_map[symbol]

        params = {
            "ids": coin_id,
            "vs_currencies": "usd",
            "include_24hr_change": "true",
            "include_24hr_high": "true",
            "include_24hr_low": "true",
            "include_24hr_vol": "true",
        }

        async with httpx.AsyncClient(
            base_url=self.BASE_URL,
            timeout=10.0,
        ) as client:

            response = await client.get(
                "/simple/price",
                params=params,
            )

            response.raise_for_status()
            data = response.json()

        coin = data[coin_id]

        return Ticker(
            symbol=symbol,
            price=float(coin["usd"]),
            price_change_24h=float(
                coin.get("usd_24h_change", 0.0)
            ),
            high_24h=float(
                coin.get("usd_24h_high", coin["usd"])
            ),
            low_24h=float(
                coin.get("usd_24h_low", coin["usd"])
            ),
            volume_24h=float(
                coin.get("usd_24h_vol", 0.0)
            ),
            timestamp=datetime.now(timezone.utc),
        )