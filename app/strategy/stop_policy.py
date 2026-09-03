from __future__ import annotations

import math
from typing import Literal


StopSide = Literal["LONG", "SHORT"]


class StopPolicy:
    """
    ATR-based stop-loss policy.

    LONG:
        stop = entry_price - ATR * multiplier

    SHORT:
        stop = entry_price + ATR * multiplier
    """

    def __init__(
        self,
        *,
        atr_multiplier: float = 1.5,
    ) -> None:
        if (
            not math.isfinite(atr_multiplier)
            or atr_multiplier <= 0
        ):
            raise ValueError(
                "atr_multiplier must be greater than zero"
            )

        self.atr_multiplier = atr_multiplier

    def calculate(
        self,
        *,
        side: StopSide,
        entry_price: float,
        atr: float,
    ) -> float:

        if side not in {"LONG", "SHORT"}:
            raise ValueError(
                "side must be LONG or SHORT"
            )

        if (
            not math.isfinite(entry_price)
            or entry_price <= 0
        ):
            raise ValueError(
                "entry_price must be greater than zero"
            )

        if not math.isfinite(atr) or atr <= 0:
            raise ValueError(
                "atr must be greater than zero"
            )

        distance = atr * self.atr_multiplier

        if side == "LONG":
            stop_price = entry_price - distance
        else:
            stop_price = entry_price + distance

        if stop_price <= 0:
            raise ValueError(
                "calculated stop_price must be greater than zero"
            )

        return stop_price

    def is_triggered(
        self,
    *,
    side: StopSide,
    stop_price: float,
    high: float,
    low: float,
) -> bool:
        if side not in {"LONG", "SHORT"}:
            raise ValueError("side must be LONG or SHORT")

        if not math.isfinite(stop_price) or stop_price <= 0:
            raise ValueError(
                "stop_price must be greater than zero"
        )

        if not math.isfinite(high) or high <= 0:
            raise ValueError(
                "high must be greater than zero"
        )

        if not math.isfinite(low) or low <= 0:
            raise ValueError(
                "low must be greater than zero"
        )

        if high < low:
            raise ValueError(
                "high must be greater than or equal to low"
        )

        if side == "LONG":
            return low <= stop_price

        return high >= stop_price