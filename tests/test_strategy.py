from datetime import datetime, timedelta, timezone

import pytest

from app.market.models import Candle
from app.strategy.engine import StrategyEngine


START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_candles(
    closes: list[float],
) -> list[Candle]:

    candles = []

    for index, close in enumerate(closes):
        candles.append(
            Candle(
                symbol="BTCUSDT",
                timeframe="1m",
                open=close,
                high=close + 2.0,
                low=close - 2.0,
                close=close,
                volume=1000.0,
                timestamp=START + timedelta(minutes=index),
            )
        )

    return candles


def test_strategy_returns_long_when_fast_sma_is_above_slow_sma():
    engine = StrategyEngine(
        fast_period=3,
        slow_period=5,
        atr_period=3,
    )

    candles = make_candles(
        [100.0, 100.0, 100.0, 110.0, 120.0]
    )

    signal = engine.generate_signal(
        symbol="BTCUSDT",
        candles=candles,
    )

    assert signal.side == "LONG"
    assert signal.price == 120.0
    assert signal.reason == "fast SMA above slow SMA"


def test_strategy_returns_short_when_fast_sma_is_below_slow_sma():
    engine = StrategyEngine(
        fast_period=3,
        slow_period=5,
        atr_period=3,
    )

    candles = make_candles(
        [120.0, 120.0, 120.0, 110.0, 100.0]
    )

    signal = engine.generate_signal(
        symbol="BTCUSDT",
        candles=candles,
    )

    assert signal.side == "SHORT"
    assert signal.price == 100.0
    assert signal.reason == "fast SMA below slow SMA"


def test_strategy_returns_flat_during_warmup():
    engine = StrategyEngine(
        fast_period=3,
        slow_period=5,
        atr_period=3,
    )

    candles = make_candles(
        [100.0, 105.0, 110.0]
    )

    signal = engine.generate_signal(
        symbol="BTCUSDT",
        candles=candles,
    )

    assert signal.side == "FLAT"
    assert signal.reason == "strategy warmup"


def test_strategy_returns_flat_when_atr_is_below_threshold():
    engine = StrategyEngine(
        fast_period=3,
        slow_period=5,
        atr_period=3,
        minimum_atr=10.0,
    )

    candles = make_candles(
        [100.0, 101.0, 102.0, 103.0, 104.0]
    )

    signal = engine.generate_signal(
        symbol="BTCUSDT",
        candles=candles,
    )

    assert signal.side == "FLAT"
    assert signal.reason == "ATR below minimum threshold"


def test_strategy_returns_flat_when_smas_are_equal():
    engine = StrategyEngine(
        fast_period=3,
        slow_period=5,
        atr_period=3,
    )

    candles = make_candles(
        [100.0, 100.0, 100.0, 100.0, 100.0]
    )

    signal = engine.generate_signal(
        symbol="BTCUSDT",
        candles=candles,
    )

    assert signal.side == "FLAT"
    assert signal.reason == "fast SMA equals slow SMA"


def test_strategy_rejects_invalid_period_configuration():
    with pytest.raises(
        ValueError,
        match="fast_period must be smaller than slow_period",
    ):
        StrategyEngine(
            fast_period=5,
            slow_period=3,
        )


def test_strategy_rejects_empty_candles():
    engine = StrategyEngine()

    with pytest.raises(
        ValueError,
        match="candles cannot be empty",
    ):
        engine.generate_signal(
            symbol="BTCUSDT",
            candles=[],
        )


def test_strategy_rejects_invalid_close():
    engine = StrategyEngine()

    candles = make_candles([100.0, 101.0, 0.0])

    with pytest.raises(
        ValueError,
        match="latest candle close must be greater than zero",
    ):
        engine.generate_signal(
            symbol="BTCUSDT",
            candles=candles,
        )

def test_strategy_signal_exposes_atr():
    engine = StrategyEngine(
        fast_period=3,
        slow_period=5,
        atr_period=3,
    )

    candles = make_candles(
        [100.0, 100.0, 100.0, 110.0, 120.0]
    )

    signal = engine.generate_signal(
        symbol="BTCUSDT",
        candles=candles,
    )

    assert signal.atr == pytest.approx(28.0 / 3.0)

