from datetime import datetime, timedelta, timezone

from app.database.connection import get_connection
from app.database.models import (
    initialize_candle_database,
    load_candles,
    save_candles,
)
from app.market.models import Candle


TEST_SYMBOL = "TESTBTC"
TEST_ETH_SYMBOL = "TESTETH"


def test_load_candles_returns_saved_candles():
    initialize_candle_database()

    timestamp = datetime(
        2099,
        1,
        1,
        0,
        0,
        0,
        tzinfo=timezone.utc,
    )

    candles = [
        Candle(
            symbol=TEST_SYMBOL,
            timeframe="1m",
            open=100.0,
            high=102.0,
            low=99.0,
            close=101.0,
            volume=1000.0,
            timestamp=timestamp,
        ),
        Candle(
            symbol=TEST_SYMBOL,
            timeframe="1m",
            open=101.0,
            high=103.0,
            low=100.0,
            close=102.0,
            volume=1100.0,
            timestamp=timestamp + timedelta(minutes=1),
        ),
    ]

    save_candles(candles)

    loaded = load_candles(
        symbol=TEST_SYMBOL,
        timeframe="1m",
    )

    assert len(loaded) == 2
    assert loaded == candles


def test_load_candles_filters_by_symbol():
    initialize_candle_database()

    timestamp = datetime(
        2099,
        1,
        2,
        21,
        0,
        0,
        tzinfo=timezone.utc,
    )

    candles = [
        Candle(
            symbol=TEST_SYMBOL,
            timeframe="1m",
            open=100.0,
            high=102.0,
            low=99.0,
            close=101.0,
            volume=1000.0,
            timestamp=timestamp,
        ),
        Candle(
            symbol=TEST_ETH_SYMBOL,
            timeframe="1m",
            open=200.0,
            high=202.0,
            low=199.0,
            close=201.0,
            volume=500.0,
            timestamp=timestamp,
        ),
    ]

    save_candles(candles)

    loaded = load_candles(
        symbol=TEST_SYMBOL,
        timeframe="1m",
    )

    assert all(candle.symbol == TEST_SYMBOL for candle in loaded)


def test_load_candles_filters_by_timeframe():
    initialize_candle_database()

    timestamp = datetime(
        2099,
        1,
        2,
        22,
        0,
        0,
        tzinfo=timezone.utc,
    )

    candles = [
        Candle(
            symbol=TEST_SYMBOL,
            timeframe="1m",
            open=100.0,
            high=102.0,
            low=99.0,
            close=101.0,
            volume=1000.0,
            timestamp=timestamp,
        ),
        Candle(
            symbol=TEST_SYMBOL,
            timeframe="5m",
            open=100.0,
            high=105.0,
            low=98.0,
            close=104.0,
            volume=5000.0,
            timestamp=timestamp,
        ),
    ]

    save_candles(candles)

    loaded = load_candles(
        symbol=TEST_SYMBOL,
        timeframe="1m",
    )

    assert all(candle.timeframe == "1m" for candle in loaded)


def test_load_candles_returns_chronological_order():
    initialize_candle_database()

    base_time = datetime(
        2099,
        1,
        2,
        23,
        0,
        0,
        tzinfo=timezone.utc,
    )

    candles = [
        Candle(
            symbol=TEST_SYMBOL,
            timeframe="1m",
            open=102.0,
            high=104.0,
            low=101.0,
            close=103.0,
            volume=1000.0,
            timestamp=base_time + timedelta(minutes=2),
        ),
        Candle(
            symbol=TEST_SYMBOL,
            timeframe="1m",
            open=100.0,
            high=102.0,
            low=99.0,
            close=101.0,
            volume=1000.0,
            timestamp=base_time,
        ),
        Candle(
            symbol=TEST_SYMBOL,
            timeframe="1m",
            open=101.0,
            high=103.0,
            low=100.0,
            close=102.0,
            volume=1000.0,
            timestamp=base_time + timedelta(minutes=1),
        ),
    ]

    save_candles(candles)

    loaded = load_candles(
        symbol=TEST_SYMBOL,
        timeframe="1m",
    )

    timestamps = [candle.timestamp for candle in loaded]

    assert timestamps == sorted(timestamps)


def test_load_candles_returns_empty_for_unknown_symbol():
    initialize_candle_database()

    loaded = load_candles(
        symbol="UNKNOWN",
        timeframe="1m",
    )

    assert loaded == []
