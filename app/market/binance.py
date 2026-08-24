from datetime import datetime, timezone

import httpx

from app.market.base import MarketDataProvider
from app.market.models import Ticker, Candle


class BinanceProvider(MarketDataProvider):

    BASE_URL = "https://api.binance.us"

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

    async def get_candle(
        self,
        symbol: str,
        timeframe: str = "1m",
    ) -> Candle:
        symbol = symbol.upper()

        async with httpx.AsyncClient(
            base_url=self.BASE_URL,
            timeout=10.0,
        ) as client:
            response = await client.get(
                "/api/v3/klines",
                params={
                    "symbol": symbol,
                    "interval": timeframe,
                    "limit": 1,
                },
            )

            response.raise_for_status()
            data = response.json()

        if not data:
            raise ValueError(
                f"No candle data returned for {symbol} {timeframe}"
            )

        k = data[0]

        open_time_ms = int(k[0])

        return Candle(
            symbol=symbol,
            timeframe=timeframe,
            open=float(k[1]),
            high=float(k[2]),
            low=float(k[3]),
            close=float(k[4]),
            volume=float(k[5]),
            timestamp=datetime.fromtimestamp(
                open_time_ms / 1000.0,
                tz=timezone.utc,
            ),
        )

    async def get_candles(
        self,
        symbols: list[str],
        timeframe: str = "1m",
    ) -> list[Candle]:
        results: list[Candle] = []

        for symbol in symbols:
            try:
                candle = await self.get_candle(symbol, timeframe)
                results.append(candle)
            except Exception:
                continue

        return results

    async def get_historical_candles(
        self,
        symbol: str,
        timeframe: str = "1m",
        limit: int = 500,
    ) -> list[Candle]:
        """
        Fetch historical OHLCV candles from Binance US.

        Binance allows a maximum of 1000 candles per request.
        """
        symbol = symbol.upper()

        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")

        async with httpx.AsyncClient(
            base_url=self.BASE_URL,
            timeout=15.0,
        ) as client:
            response = await client.get(
                "/api/v3/klines",
                params={
                    "symbol": symbol,
                    "interval": timeframe,
                    "limit": limit,
                },
            )

            response.raise_for_status()
            data = response.json()

        candles: list[Candle] = []

        for k in data:
            open_time_ms = int(k[0])

            candles.append(
                Candle(
                    symbol=symbol,
                    timeframe=timeframe,
                    open=float(k[1]),
                    high=float(k[2]),
                    low=float(k[3]),
                    close=float(k[4]),
                    volume=float(k[5]),
                    timestamp=datetime.fromtimestamp(
                        open_time_ms / 1000.0,
                        tz=timezone.utc,
                    ),
                )
            )

        return candles