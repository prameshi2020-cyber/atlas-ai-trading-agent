import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from app.market.models import Candle
from app.market.warmup import MarketWarmup


def make_candle(
    symbol: str,
    minute: int,
) -> Candle:
    return Candle(
        symbol=symbol,
        timeframe="1m",
        open=100.0 + minute,
        high=102.0 + minute,
        low=99.0 + minute,
        close=101.0 + minute,
        volume=1000.0 + minute,
        timestamp=datetime(
            2099,
            1,
            1,
            0,
            minute,
            tzinfo=timezone.utc,
        ),
    )


def test_warmup_fetches_and_persists_historical_candles():
    provider = AsyncMock()

    candles = [
        make_candle("BTCUSDT", 0),
        make_candle("BTCUSDT", 1),
        make_candle("BTCUSDT", 2),
    ]

    provider.get_historical_candles.return_value = candles

    warmup = MarketWarmup(
        provider=provider,
        symbols=["BTCUSDT"],
        limit=100,
    )

    with patch(
        "app.market.warmup.save_candles",
        return_value=3,
    ) as save_mock:
        result = asyncio.run(warmup.warmup())

    assert result == {"BTCUSDT": 3}

    provider.get_historical_candles.assert_awaited_once_with(
        symbol="BTCUSDT",
        timeframe="1m",
        limit=100,
    )

    save_mock.assert_called_once_with(candles)


def test_warmup_isolates_symbol_failures():
    provider = AsyncMock()

    provider.get_historical_candles.side_effect = [
        RuntimeError("BTC unavailable"),
        [
            make_candle("ETHUSDT", 0),
            make_candle("ETHUSDT", 1),
        ],
    ]

    warmup = MarketWarmup(
        provider=provider,
        symbols=["BTCUSDT", "ETHUSDT"],
        limit=50,
    )

    with patch(
        "app.market.warmup.save_candles",
        return_value=2,
    ):
        result = asyncio.run(warmup.warmup())

    assert result == {
        "BTCUSDT": 0,
        "ETHUSDT": 2,
    }


def test_warmup_validates_configuration():
    provider = AsyncMock()

    with pytest.raises(ValueError, match="symbols cannot be empty"):
        MarketWarmup(
            provider=provider,
            symbols=[],
        )

    with pytest.raises(ValueError, match="limit must be between 1 and 1000"):
        MarketWarmup(
            provider=provider,
            symbols=["BTCUSDT"],
            limit=0,
        )

    with pytest.raises(ValueError, match="limit must be between 1 and 1000"):
        MarketWarmup(
            provider=provider,
            symbols=["BTCUSDT"],
            limit=1001,
        )
