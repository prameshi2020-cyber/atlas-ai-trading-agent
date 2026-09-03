from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime

from app.risk.manager import RiskManager
from app.trading.broker import PaperBroker
from app.trading.models import ExecutionResult
from app.trading.position_manager import PositionManager
from app.trading.journal import (
    initialize_trade_journal,
    save_trade,
)

@dataclass(slots=True)
class TradeExecution:
    result: ExecutionResult
    pnl: float | None = None


class ExecutionEngine:
    def __init__(
        self,
        *,
        broker: PaperBroker,
        risk_manager: RiskManager,
        position_manager: PositionManager,
    ) -> None:
        self.broker = broker
        self.risk_manager = risk_manager
        self.position_manager = position_manager

        initialize_trade_journal()

    def open_position(
        self,
        *,
        symbol: str,
        side: str,
        entry_price: float,
        stop_price: float,
        timestamp: datetime,
    ) -> TradeExecution:
        if self.position_manager.has_position(symbol):
            result = ExecutionResult(
                success=False,
                symbol=symbol,
                side="BUY" if side == "LONG" else "SELL",
                quantity=0.0,
                execution_price=entry_price,
                fee=0.0,
                timestamp=timestamp,
                reason="position already exists",
            )

            return TradeExecution(result=result)

        equity = self.broker.equity(
            symbol=symbol,
            market_price=entry_price,
       )

        decision = self.risk_manager.evaluate_trade(
            equity=equity,
            entry_price=entry_price,
            stop_price=stop_price,
            timestamp=timestamp,
        )

        if not decision.allowed:
            result = ExecutionResult(
                success=False,
                symbol=symbol,
                side="BUY" if side == "LONG" else "SELL",
                quantity=0.0,
                execution_price=entry_price,
                fee=0.0,
                timestamp=timestamp,
                reason=decision.reason,
            )

            return TradeExecution(result=result)

        order_side = "BUY" if side == "LONG" else "SELL"

        result = self.broker.execute_market_order(
            symbol=symbol,
            side=order_side,
            quantity=decision.quantity,
            price=entry_price,
            timestamp=timestamp,
        )

        if not result.success:
            return TradeExecution(result=result)

        try:
            self.position_manager.open_position(
                result=result,
                stop_price=stop_price,
            )
        except Exception as exc:
            self.broker.reverse_execution(result)

            return TradeExecution(
                result=ExecutionResult(
                    success=False,
                    symbol=symbol,
                    side=order_side,
                    quantity=decision.quantity,
                    execution_price=entry_price,
                    fee=result.fee,
                    timestamp=timestamp,
                    reason=str(exc),
                )
            )

        try:
            registered = self.risk_manager.register_position_open()

            if not registered:
                raise RuntimeError(
                    "risk manager rejected position registration"
                )

        except Exception as exc:
            self.position_manager.rollback_open_position(
                result=result,
            )

            self.broker.reverse_execution(result)

            return TradeExecution(
                result=ExecutionResult(
                    success=False,
                    symbol=symbol,
                    side=order_side,
                    quantity=decision.quantity,
                    execution_price=entry_price,
                    fee=result.fee,
                    timestamp=timestamp,
                    reason=str(exc),
                )
            )

        return TradeExecution(result=result)

    def close_position(
        self,
        *,
        symbol: str,
        exit_price: float,
        timestamp: datetime,
        quantity: float | None = None,
    ) -> TradeExecution:
        position = self.position_manager.get_position(symbol)

        if position is None:
            result = ExecutionResult(
                success=False,
                symbol=symbol,
                side="SELL",
                quantity=0.0,
                execution_price=exit_price,
                fee=0.0,
                timestamp=timestamp,
                reason="no open position",
            )
            return TradeExecution(result=result)

        close_quantity = (
            position.quantity
            if quantity is None
            else quantity
        )

        if close_quantity <= 0:
            result = ExecutionResult(
                success=False,
                symbol=symbol,
                side="SELL" if position.side == "LONG" else "BUY",
                quantity=0.0,
                execution_price=exit_price,
                fee=0.0,
                timestamp=timestamp,
                reason="close quantity must be greater than zero",
            )
            return TradeExecution(result=result)

        if close_quantity > position.quantity:
            result = ExecutionResult(
                success=False,
                symbol=symbol,
                side="SELL" if position.side == "LONG" else "BUY",
                quantity=0.0,
                execution_price=exit_price,
                fee=0.0,
                timestamp=timestamp,
                reason="close quantity cannot exceed position",
            )
            return TradeExecution(result=result)

        original_position = deepcopy(position)
        original_risk_state = deepcopy(
            self.risk_manager.state
        )

        original_cash = self.broker.cash
        original_positions = self.broker.positions.copy()

        order_side = (
            "SELL"
            if position.side == "LONG"
            else "BUY"
        )

        result = self.broker.execute_market_order(
            symbol=symbol,
            side=order_side,
            quantity=close_quantity,
            price=exit_price,
            timestamp=timestamp,
        )

        if not result.success:
            return TradeExecution(result=result)

        try:
            closed_position, pnl = self.position_manager.close_position(
                result=result,
            )

            if close_quantity == position.quantity:
                self.risk_manager.register_position_close()

            self.risk_manager.record_realized_pnl(
                pnl=pnl,
                timestamp=timestamp,
            )

            save_trade(
                symbol=original_position.symbol,
                side=original_position.side,
                quantity=result.quantity,
                entry_price=original_position.entry_price,
                exit_price=result.execution_price,
                entry_time=original_position.entry_time,
                exit_time=result.timestamp,
                pnl=pnl,
                fee=(
                    original_position.entry_fee
                    * result.quantity
                    / original_position.quantity
                    + result.fee
                ),
                status="CLOSED",
            )
        except Exception:
            try:
                self.position_manager.restore_position(
                    original_position,
                )
            finally:
                self.risk_manager.restore_state(
                    original_risk_state,
                )
                self.broker.restore_state(
                    cash=original_cash,
                    positions=original_positions,
                )

            raise

        return TradeExecution(
            result=result,
            pnl=pnl,
        )