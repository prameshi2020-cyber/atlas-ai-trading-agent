from __future__ import annotations

import math


def calculate_sma(
    values: list[float],
    period: int,
) -> float | None:
    if period <= 0:
        raise ValueError("period must be greater than zero")

    if len(values) < period:
        return None

    window = values[-period:]

    if not all(math.isfinite(value) for value in window):
        raise ValueError("all values must be finite")

    return sum(window) / period


def calculate_atr(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int,
) -> float | None:
    if period <= 0:
        raise ValueError("period must be greater than zero")

    if not (
        len(highs) == len(lows) == len(closes)
    ):
        raise ValueError("highs, lows, and closes must have equal length")

    if len(closes) < period + 1:
        return None

    true_ranges: list[float] = []

    for index in range(1, len(closes)):
        high = highs[index]
        low = lows[index]
        previous_close = closes[index - 1]

        if not all(
            math.isfinite(value)
            for value in (high, low, previous_close)
        ):
            raise ValueError("all candle values must be finite")

        if high < low:
            raise ValueError("high must be greater than or equal to low")

        true_range = max(
            high - low,
            abs(high - previous_close),
            abs(low - previous_close),
        )

        true_ranges.append(true_range)

    return sum(true_ranges[-period:]) / period