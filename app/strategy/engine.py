from __future__ import annotations

import math

from app.market.models import Candle
from app.strategy.indicators import calculate_atr, calculate_sma
from app.strategy.models import TradingSignal


class StrategyEngine:
    """
    Deterministic SMA + ATR strategy.

    The strategy only generates signals.
    It does not execute trades or manage risk.
    """

    def __init__(
        self,
        *,
        fast_period: int = 3,
        slow_period: int = 5,
        atr_period: int = 3,
        minimum_atr: float = 0.0,
    ) -> None:

        if fast_period <= 0:
            raise ValueError("fast_period must be greater than zero")

        if slow_period <= 0:
            raise ValueError("slow_period must be greater than zero")

        if fast_period >= slow_period:
            raise ValueError(
                "fast_period must be smaller than slow_period"
            )

        if atr_period <= 0:
            raise ValueError("atr_period must be greater than zero")

        if not math.isfinite(minimum_atr) or minimum_atr < 0:
            raise ValueError(
                "minimum_atr must be non-negative"
            )

        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self.minimum_atr = minimum_atr

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