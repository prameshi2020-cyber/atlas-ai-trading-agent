import pytest

from app.strategy.indicators import (
    calculate_atr,
    calculate_sma,
)


def test_sma_calculates_correctly():
    result = calculate_sma(
        [10.0, 20.0, 30.0],
        period=3,
    )

    assert result == 20.0


def test_sma_uses_latest_period():
    result = calculate_sma(
        [10.0, 20.0, 30.0, 40.0],
        period=2,
    )

    assert result == 35.0


def test_sma_returns_none_during_warmup():
    result = calculate_sma(
        [10.0, 20.0],
        period=3,
    )

    assert result is None


def test_sma_rejects_invalid_period():
    with pytest.raises(
        ValueError,
        match="period must be greater than zero",
    ):
        calculate_sma(
            [10.0, 20.0],
            period=0,
        )


def test_atr_calculates_correctly():
    highs = [11.0, 13.0, 14.0]
    lows = [9.0, 10.0, 12.0]
    closes = [10.0, 12.0, 13.0]

    result = calculate_atr(
        highs,
        lows,
        closes,
        period=2,
    )

    assert result == 2.5


def test_atr_returns_none_during_warmup():
    result = calculate_atr(
        [11.0, 12.0],
        [9.0, 10.0],
        [10.0, 11.0],
        period=2,
    )

    assert result is None


def test_atr_rejects_mismatched_lengths():
    with pytest.raises(
        ValueError,
        match="equal length",
    ):
        calculate_atr(
            [11.0, 12.0],
            [9.0],
            [10.0, 11.0],
            period=2,
        )


def test_atr_rejects_invalid_high_low():
    with pytest.raises(
        ValueError,
        match="high must be greater than or equal to low",
    ):
        calculate_atr(
            [11.0, 9.0, 12.0],
            [9.0, 10.0, 11.0],
            [10.0, 10.0, 11.0],
            period=2,
        )