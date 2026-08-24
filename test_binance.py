import asyncio

from app.market.binance import BinanceProvider


async def main():
    provider = BinanceProvider()

    ticker = await provider.get_ticker("BTCUSDT")
    print("TICKER:")
    print(ticker)

    candle = await provider.get_candle("BTCUSDT", "1m")
    print("\nCANDLE:")
    print(candle)


if __name__ == "__main__":
    asyncio.run(main())