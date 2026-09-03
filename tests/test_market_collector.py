import asyncio
from datetime import datetime, timezone

from app.market.collector import MarketCollector
from app.market.models import Candle, Ticker


TIMESTAMP = datetime(
    2026,
    9,
    2,
    20,
    0,
    0,
    tzinfo=timezone.utc,
)


class FakeProvider:
    def __init__(self) -> None:
        self.ticker_symbols = None
        self.candle_symbols = None
        self.candle_timeframe = None

        self.tickers = [
            Ticker(
                symbol="BTCUSDT",
                price=100_000.0,
                price_change_24h=1.5,
                high_24h=101_000.0,
                low_24h=98_000.0,
                volume_24h=1_000_000.0,
                timestamp=TIMESTAMP,
            )
        ]

        self.candles = [
            Candle(
                symbol="BTCUSDT",
                timeframe="1m",
                open=99_900.0,
                high=100_100.0,
                low=99_800.0,
                close=100_000.0,
                volume=500.0,
                timestamp=TIMESTAMP,
            )
        ]

    async def get_tickers(self, symbols: list[str]) -> list[Ticker]:
        self.ticker_symbols = symbols
        return self.tickers

    async def get_candles(
        self,
        symbols: list[str],
        timeframe: str = "1m",
    ) -> list[Candle]:
        self.candle_symbols = symbols
        self.candle_timeframe = timeframe
        return self.candles


def test_collector_requests_tickers_and_candles():
    provider = FakeProvider()

    collector = MarketCollector(
        provider=provider,
        symbols=["BTCUSDT", "ETHUSDT"],
    )

    asyncio.run(collector.collect_once())

    assert provider.ticker_symbols == ["BTCUSDT", "ETHUSDT"]
    assert provider.candle_symbols == ["BTCUSDT", "ETHUSDT"]
    assert provider.candle_timeframe == "1m"


def test_collector_saves_tickers(monkeypatch):
    provider = FakeProvider()
    saved_tickers = []

    monkeypatch.setattr(
        "app.market.collector.save_ticker",
        saved_tickers.append,
    )

    collector = MarketCollector(
        provider=provider,
        symbols=["BTCUSDT"],
    )

    asyncio.run(collector.collect_once())

    assert len(saved_tickers) == 1
    assert saved_tickers[0] is provider.tickers[0]


def test_collector_saves_candles(monkeypatch):
    provider = FakeProvider()
    saved_candles = []

    monkeypatch.setattr(
        "app.market.collector.save_candle",
        saved_candles.append,
    )

    collector = MarketCollector(
        provider=provider,
        symbols=["BTCUSDT"],
    )

    asyncio.run(collector.collect_once())

    assert len(saved_candles) == 1
    assert saved_candles[0] is provider.candles[0]


def test_collector_uses_configured_symbols():
    provider = FakeProvider()

    collector = MarketCollector(
        provider=provider,
        symbols=["BTCUSDT"],
    )

    assert collector.symbols == ["BTCUSDT"]


def test_collector_uses_configured_interval():
    provider = FakeProvider()

    collector = MarketCollector(
        provider=provider,
        symbols=["BTCUSDT"],
        interval_seconds=30,
    )

    assert collector.interval_seconds == 30