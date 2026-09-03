from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from app.market.models import Candle


SignalSide = Literal["LONG", "SHORT", "FLAT"]


@dataclass(slots=True)
class TradingSignal:
    symbol: str
    side: SignalSide
    price: float
    timestamp: datetime
    reason: str
    atr: float | None = None


def calculate_sma(values: list[float], period: int) -> float | None:
    if period <= 0:
        raise ValueError("period must be greater than zero")

    if len(values) < period:
        return None

    return sum(values[-period:]) / period


def calculate_atr(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int,
) -> float | None:
    if period <= 0:
        raise ValueError("period must be greater than zero")

    if len(highs) != len(lows) or len(highs) != len(closes):
        raise ValueError("highs, lows, and closes must have the same length")

    if len(highs) < period + 1:
        return None

    true_ranges: list[float] = []
    for index in range(1, len(closes)):
        current_high = highs[index]
        current_low = lows[index]
        previous_close = closes[index - 1]
        true_ranges.append(
            max(
                current_high - current_low,
                abs(current_high - previous_close),
                abs(current_low - previous_close),
            )
        )

    if len(true_ranges) < period:
        return None

    return sum(true_ranges[-period:]) / period


def generate_signal(
    self,
    *,
    symbol: str,
    candles: list[Candle],
) -> TradingSignal:

    if not candles:
        raise ValueError("candles cannot be empty")

    latest = candles[-1]

    if not math.isfinite(latest.close) or latest.close <= 0:
        raise ValueError(
            "latest candle close must be greater than zero"
        )

    closes = [candle.close for candle in candles]
    highs = [candle.high for candle in candles]
    lows = [candle.low for candle in candles]

    fast_sma = calculate_sma(
        closes,
        self.fast_period,
    )

    slow_sma = calculate_sma(
        closes,
        self.slow_period,
    )

    atr = calculate_atr(
        highs,
        lows,
        closes,
        self.atr_period,
    )

    if fast_sma is None or slow_sma is None or atr is None:
        return TradingSignal(
            symbol=symbol,
            side="FLAT",
            price=latest.close,
            timestamp=latest.timestamp,
            reason="strategy warmup",
            atr=atr,
        )

    if atr < self.minimum_atr:
        return TradingSignal(
            symbol=symbol,
            side="FLAT",
            price=latest.close,
            timestamp=latest.timestamp,
            reason="ATR below minimum threshold",
            atr=atr,
        )

    if fast_sma > slow_sma:
        return TradingSignal(
            symbol=symbol,
            side="LONG",
            price=latest.close,
            timestamp=latest.timestamp,
            reason="fast SMA above slow SMA",
            atr=atr,
        )

    if fast_sma < slow_sma:
        return TradingSignal(
            symbol=symbol,
            side="SHORT",
            price=latest.close,
            timestamp=latest.timestamp,
            reason="fast SMA below slow SMA",
            atr=atr,
        )

    return TradingSignal(
        symbol=symbol,
        side="FLAT",
        price=latest.close,
        timestamp=latest.timestamp,
        reason="fast SMA equals slow SMA",
        atr=atr,
    )