import asyncio
import logging

from app.database.models import save_ticker, save_candle
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
        # -------------------------
        # Tickers
        # -------------------------
        try:
            tickers = await self.provider.get_tickers(self.symbols)

            for ticker in tickers:
                try:
                    save_ticker(ticker)

                    logger.info(
                        "Collected %s ticker: $%.2f",
                        ticker.symbol,
                        ticker.price,
                    )

                except Exception:
                    logger.exception(
                        "Failed to save ticker %s",
                        ticker.symbol,
                    )

        except Exception:
            logger.exception("Failed to collect tickers")

        # -------------------------
        # Candles
        # -------------------------
        try:
            candles = await self.provider.get_candles(
                self.symbols,
                timeframe="1m",
            )

            for candle in candles:
                try:
                    save_candle(candle)

                    logger.info(
                        "Collected %s 1m candle: "
                        "O=%.2f H=%.2f L=%.2f C=%.2f V=%.2f",
                        candle.symbol,
                        candle.open,
                        candle.high,
                        candle.low,
                        candle.close,
                        candle.volume,
                    )

                except Exception:
                    logger.exception(
                        "Failed to save candle %s",
                        candle.symbol,
                    )

        except Exception:
            logger.exception("Failed to collect candles")

    async def run(self) -> None:
        try:
            while True:
                await self.collect_once()
                await asyncio.sleep(self.interval_seconds)

        except asyncio.CancelledError:
            logger.info("MarketCollector stopped by cancellation.")
            return