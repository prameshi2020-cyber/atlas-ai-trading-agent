from __future__ import annotations

import math


def calculate_position_size(
    *,
    equity: float,
    entry_price: float,
    stop_price: float,
    risk_per_trade: float,
    maximum_position_pct: float = 1.0,
    minimum_position_size: float = 0.0,
) -> tuple[float, float, float]:
    """
    Calculate risk-based position sizing.

    Returns:
        quantity
        risk_amount
        position_notional
    """

    values = (
        equity,
        entry_price,
        stop_price,
        risk_per_trade,
        maximum_position_pct,
        minimum_position_size,
    )

    if not all(math.isfinite(value) for value in values):
        raise ValueError("all sizing inputs must be finite")

    if equity <= 0:
        raise ValueError("equity must be greater than zero")

    if entry_price <= 0:
        raise ValueError("entry_price must be greater than zero")

    if stop_price <= 0:
        raise ValueError("stop_price must be greater than zero")

    if risk_per_trade <= 0:
        raise ValueError("risk_per_trade must be greater than zero")

    if maximum_position_pct <= 0:
        raise ValueError("maximum_position_pct must be greater than zero")

    if minimum_position_size < 0:
        raise ValueError("minimum_position_size cannot be negative")

    stop_distance = abs(entry_price - stop_price)

    if stop_distance <= 0:
        raise ValueError("stop distance must be greater than zero")

    risk_amount = equity * risk_per_trade

    raw_quantity = risk_amount / stop_distance

    max_notional = equity * maximum_position_pct
    max_quantity = max_notional / entry_price

    quantity = min(raw_quantity, max_quantity)

    if quantity < minimum_position_size:
        return 0.0, risk_amount, 0.0

    position_notional = quantity * entry_price

    return quantity, risk_amount, position_notional