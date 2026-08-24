import asyncio

from app.market.binance import BinanceProvider
from app.database.models import save_candles


async def main():
    provider = BinanceProvider()

    symbols = [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
    ]

    for symbol in symbols:
        print(f"Fetching {symbol}...")

        candles = await provider.get_historical_candles(
            symbol=symbol,
            timeframe="1m",
            limit=500,
        )

        inserted = save_candles(candles)

        print(
            f"{symbol}: "
            f"received={len(candles)}, "
            f"inserted={inserted}"
        )


if __name__ == "__main__":
    asyncio.run(main())