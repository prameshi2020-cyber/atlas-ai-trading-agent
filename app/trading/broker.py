from __future__ import annotations

import math

from app.database.models import (
    initialize_paper_broker_database,
    load_paper_broker_state,
    save_paper_broker_state,
)
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
        fee_rate: float | None = None,
    ) -> None:

        if not math.isfinite(initial_cash) or initial_cash <= 0:
            raise ValueError(
                "initial_cash must be greater than zero"
            )

        if fee_rate is not None:
            if not math.isfinite(fee_rate) or fee_rate < 0:
                raise ValueError(
                    "fee_rate must be non-negative"
                )

            fee_pct = fee_rate * 100

        if not math.isfinite(fee_pct) or fee_pct < 0:
            raise ValueError(
                "fee_pct must be non-negative"
            )

        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.fee_pct = fee_pct
        self.positions: dict[str, float] = {}
        initialize_paper_broker_database()

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
            self.positions[order.symbol] = (
                self.positions.get(order.symbol, 0.0)
                + order.quantity
            )

        else:

            self.cash += notional - fee

            self.positions[order.symbol] = (
                self.positions.get(order.symbol, 0.0)
                - order.quantity
            )

        self.save_state()
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

    def execute_market_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        timestamp,
    ) -> ExecutionResult:

        order = OrderRequest(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
        )

        return self.execute(order)

    def reverse_execution(
        self,
        result: ExecutionResult,
    ) -> None:
        if not result.success:
            raise ValueError(
                "cannot reverse failed execution"
            )

        if result.quantity <= 0:
            raise ValueError(
                "execution quantity must be greater than zero"
            )

        if result.execution_price <= 0:
            raise ValueError(
                "execution price must be greater than zero"
            )

        notional = (
            result.quantity * result.execution_price
        )

        if result.side == "BUY":
            self.cash += notional + result.fee

            current_quantity = self.positions.get(
                result.symbol,
                0.0,
            )

            new_quantity = (
                current_quantity - result.quantity
            )

        else:
            self.cash -= notional - result.fee

            current_quantity = self.positions.get(
                result.symbol,
                0.0,
            )

            new_quantity = (
                current_quantity + result.quantity
            )

        if math.isclose(
            new_quantity,
            0.0,
            abs_tol=1e-12,
        ):
            self.positions.pop(
                result.symbol,
                None,
            )
        else:
            self.positions[result.symbol] = new_quantity

        self.save_state()

    def equity(
        self,
        *,
        symbol: str,
        market_price: float,
    ) -> float:

        if not math.isfinite(market_price) or market_price <= 0:
            raise ValueError(
                "market_price must be greater than zero"
            )

        position_quantity = self.positions.get(symbol, 0.0)
        position_value = position_quantity * market_price

        return self.cash + position_value

    def save_state(self) -> None:
        persisted_positions = {
            symbol: quantity
            for symbol, quantity in self.positions.items()
            if not math.isclose(
                quantity,
                0.0,
                abs_tol=1e-12,
            )
        }

        save_paper_broker_state(
            cash=self.cash,
            positions=persisted_positions,
        )

    def load_state(self) -> bool:
        state = load_paper_broker_state()

        if state is None:
            return False

        cash, positions = state

        if not math.isfinite(cash) or cash <= 0:
            raise ValueError(
                "persisted broker cash must be greater than zero"
            )

        for symbol, quantity in positions.items():
            if not math.isfinite(quantity):
                raise ValueError(
                    f"persisted broker quantity must be finite: {symbol}"
                )

            if math.isclose(quantity, 0.0, abs_tol=1e-12):
                raise ValueError(
                    f"persisted broker quantity cannot be zero: {symbol}"
                )

        self.cash = cash
        self.positions = positions

        return True

    def restore_state(
        self,
        *,
        cash: float,
        positions: dict[str, float],
    ) -> None:
        if not math.isfinite(cash) or cash <= 0:
            raise ValueError(
                "restored broker cash must be greater than zero"
            )

        for symbol, quantity in positions.items():
            if not math.isfinite(quantity):
                raise ValueError(
                    f"restored broker quantity must be finite: {symbol}"
                )

            if math.isclose(
                quantity,
                0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"restored broker quantity cannot be zero: {symbol}"
                )

        self.cash = cash
        self.positions = positions.copy()

        self.save_state()