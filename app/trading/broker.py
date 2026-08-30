from __future__ import annotations

import math

from app.trading.models import (
    ExecutionResult,
    OrderRequest,
)


class PaperBroker:
    """
    Simulated broker for paper trading.

    This broker NEVER sends real orders to an exchange.
    Orders are executed internally at the requested price.
    """

    def __init__(
        self,
        *,
        initial_cash: float = 10_000.0,
        fee_pct: float = 0.10,
    ) -> None:

        if not math.isfinite(initial_cash) or initial_cash <= 0:
            raise ValueError(
                "initial_cash must be greater than zero"
            )

        if not math.isfinite(fee_pct) or fee_pct < 0:
            raise ValueError(
                "fee_pct must be non-negative"
            )

        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.fee_pct = fee_pct

    def execute(
        self,
        order: OrderRequest,
    ) -> ExecutionResult:

        if not math.isfinite(order.quantity):
            return ExecutionResult(
                success=False,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                execution_price=order.price,
                fee=0.0,
                timestamp=order.timestamp,
                reason="quantity must be finite",
            )

        if order.quantity <= 0:
            return ExecutionResult(
                success=False,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                execution_price=order.price,
                fee=0.0,
                timestamp=order.timestamp,
                reason="quantity must be greater than zero",
            )

        if not math.isfinite(order.price) or order.price <= 0:
            return ExecutionResult(
                success=False,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                execution_price=order.price,
                fee=0.0,
                timestamp=order.timestamp,
                reason="price must be greater than zero",
            )

        notional = order.quantity * order.price

        fee = notional * self.fee_pct / 100

        total_cost = notional + fee

        if order.side == "BUY":

            if total_cost > self.cash:
                return ExecutionResult(
                    success=False,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    execution_price=order.price,
                    fee=0.0,
                    timestamp=order.timestamp,
                    reason="insufficient paper cash",
                )

            self.cash -= total_cost

        else:

            self.cash += notional - fee

        return ExecutionResult(
            success=True,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            execution_price=order.price,
            fee=fee,
            timestamp=order.timestamp,
            reason="paper order executed",
        )