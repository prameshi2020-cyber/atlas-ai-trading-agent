from __future__ import annotations

import math

from app.database.models import (
    delete_position,
    load_positions,
    save_position,
)
from app.trading.models import Position, ExecutionResult

class PositionManager:
    def __init__(self) -> None:
        self._positions: dict[str, Position] = {}

    def load_positions(self) -> None:
        positions = load_positions()

        for position in positions:
            if position.quantity <= 0:
                raise ValueError(
                    f"invalid persisted position quantity for "
                    f"{position.symbol}"
                )

            if position.entry_price <= 0:
                raise ValueError(
                    f"invalid persisted position entry_price for "
                    f"{position.symbol}"
                )

            if position.entry_fee < 0:
                raise ValueError(
                    f"invalid persisted position entry_fee for "
                    f"{position.symbol}"
               )

            if position.side not in ("LONG", "SHORT"):
                raise ValueError(
                    f"invalid persisted position side for "
                    f"{position.symbol}"
                )

        self._positions = {
            position.symbol: position
            for position in positions
    }

    def get_position(self, symbol: str) -> Position | None:
        return self._positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        return symbol in self._positions

    def open_position(
        self,
        *,
        result: ExecutionResult,
        stop_price: float | None = None,
    ) -> Position:
        if not result.success:
            raise ValueError("cannot open position from failed execution")

        if result.quantity <= 0:
            raise ValueError("position quantity must be greater than zero")

        if result.execution_price <= 0:
            raise ValueError("execution price must be greater than zero")

        if result.symbol in self._positions:
            raise ValueError(
                f"position already exists for {result.symbol}"
            )

        if result.side == "BUY":
            side = "LONG"
        else:
            side = "SHORT"

        position = Position(
            symbol=result.symbol,
            side=side,
            quantity=result.quantity,
            entry_price=result.execution_price,
            entry_time=result.timestamp,
            entry_fee=result.fee,
            stop_price=stop_price,
        )

        self._positions[result.symbol] = position

        try:
            save_position(position)
        except Exception:
            del self._positions[result.symbol]
            raise

        return position

    def rollback_open_position(
        self,
        result: ExecutionResult,
    ) -> None:
        if not result.success:
            raise ValueError(
                "cannot rollback failed execution"
            )

        position = self._positions.get(result.symbol)

        if position is None:
            raise ValueError(
                f"no open position to rollback for {result.symbol}"
            )

        if not math.isclose(
            position.quantity,
            result.quantity,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "position quantity does not match execution"
            )

        if not math.isclose(
            position.entry_price,
            result.execution_price,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "position entry price does not match execution"
            )

        if not math.isclose(
            position.entry_fee,
            result.fee,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "position entry fee does not match execution"
            )

        del self._positions[result.symbol]

    def close_position(
        self,
        *,
        result: ExecutionResult,
    ) -> tuple[Position, float]:
        if not result.success:
            raise ValueError("cannot close position from failed execution")

        position = self._positions.get(result.symbol)

        if position is None:
            raise ValueError(
                f"no open position for {result.symbol}"
            )

        if result.quantity <= 0:
            raise ValueError("close quantity must be greater than zero")

        if result.quantity > position.quantity:
            raise ValueError(
                "close quantity cannot exceed position quantity"
            )

        if result.execution_price <= 0:
            raise ValueError("execution price must be greater than zero")

        if position.side == "LONG":
            expected_side = "SELL"

            pnl = (
                result.execution_price - position.entry_price
            ) * result.quantity

        else:
            expected_side = "BUY"

            pnl = (
                position.entry_price - result.execution_price
            ) * result.quantity

        if result.side != expected_side:
            raise ValueError(
                f"{position.side} position must be closed with "
                f"{expected_side}"
            )

        entry_fee_allocated = (
            position.entry_fee
            * result.quantity
            / position.quantity
     )

        pnl -= entry_fee_allocated
        pnl -= result.fee

        remaining_quantity = (
            position.quantity - result.quantity
        )

        remaining_entry_fee = (
            position.entry_fee - entry_fee_allocated
        )

        if math.isclose(remaining_quantity, 0.0, abs_tol=1e-12):
            delete_position(result.symbol)
            del self._positions[result.symbol]

        else:
            updated_position = Position(
                symbol=position.symbol,
                side=position.side,
                quantity=remaining_quantity,
                entry_price=position.entry_price,
                entry_time=position.entry_time,
                entry_fee=remaining_entry_fee,
                stop_price=position.stop_price,
            )

            save_position(updated_position)

            self._positions[result.symbol] = updated_position

            position = updated_position

        return position, pnl

    def restore_position(self, position: Position) -> None:
        if position.quantity <= 0:
            raise ValueError(
                "position quantity must be greater than zero"
            )

        if position.entry_price <= 0:
            raise ValueError(
                "position entry price must be greater than zero"
            )

        if position.entry_fee < 0:
            raise ValueError(
                "position entry fee cannot be negative"
            )

        if position.side not in ("LONG", "SHORT"):
            raise ValueError(
                "position side must be LONG or SHORT"
            )

        save_position(position)

        self._positions[position.symbol] = position

    def all_positions(self) -> list[Position]:
        return list(self._positions.values())

    def position_count(self) -> int:
        return len(self._positions)