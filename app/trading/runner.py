from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.trading.execution import ExecutionEngine, TradeExecution
from app.trading.journal import save_trade


@dataclass(slots=True)
class RunnerResult:
    action: str
    execution: TradeExecution | None
    reason: str


class PaperTradingRunner:
    """
    Coordinates trading decisions with the ExecutionEngine
    and persists completed trades in the trade journal.
    """

    def __init__(
        self,
        *,
        execution_engine: ExecutionEngine,
    ) -> None:
        self.execution_engine = execution_engine

    def open(
        self,
        *,
        symbol: str,
        side: str,
        entry_price: float,
        stop_price: float,
        timestamp: datetime,
    ) -> RunnerResult:

        if self.execution_engine.position_manager.has_position(symbol):
            return RunnerResult(
                action="OPEN",
                execution=None,
                reason="position already exists",
            )

        execution = self.execution_engine.open_position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            stop_price=stop_price,
            timestamp=timestamp,
        )

        return RunnerResult(
            action="OPEN",
            execution=execution,
            reason=execution.result.reason,
        )

    def close(
        self,
        *,
        symbol: str,
        exit_price: float,
        timestamp: datetime,
        exit_reason: str = "manual",
    ) -> RunnerResult:

        position = self.execution_engine.position_manager.get_position(
            symbol
        )

        if position is None:
            return RunnerResult(
                action="CLOSE",
                execution=None,
                reason="no open position",
            )

        execution = self.execution_engine.close_position(
            symbol=symbol,
            exit_price=exit_price,
            timestamp=timestamp,
        )

        if not execution.result.success:
            return RunnerResult(
                action="CLOSE",
                execution=execution,
                reason=execution.result.reason,
            )

        if execution.pnl is None:
            raise RuntimeError(
                "successful close execution must have PnL"
            )

        save_trade(
            symbol=position.symbol,
            side=position.side,
            quantity=execution.result.quantity,
            entry_price=position.entry_price,
            exit_price=execution.result.execution_price,
            entry_time=position.entry_time,
            exit_time=execution.result.timestamp,
            pnl=execution.pnl,
            fee=execution.result.fee,
            status="CLOSED",
            exit_reason=exit_reason,
        )

        return RunnerResult(
            action="CLOSE",
            execution=execution,
            reason=execution.result.reason,
        )