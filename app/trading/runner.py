from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.strategy.stop_policy import StopPolicy
from app.market.models import Candle
from app.trading.execution import ExecutionEngine, TradeExecution


@dataclass(slots=True)
class RunnerResult:
    action: str
    execution: TradeExecution | None
    reason: str


class PaperTradingRunner:
    """
    Coordinates strategy decisions with the ExecutionEngine
    and persists completed trades in the trade journal.
    """

    def __init__(
        self,
        *,
        execution_engine: ExecutionEngine,
        stop_policy: StopPolicy,
    ) -> None:
        self.execution_engine = execution_engine
        self.stop_policy = stop_policy

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
            exit_reason=exit_reason,
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

        return RunnerResult(
            action="CLOSE",
            execution=execution,
            reason=exit_reason,
        )

    def process_candles(
        self,
        *,
        symbol: str,
        candles: list[dict] | list[Candle],
        strategy,
        timestamp: datetime,
    ) -> RunnerResult:
        """
        Process the latest closed candle through the strategy.

        Strategy output is expected to be:
            LONG
            SHORT
            FLAT
        """

        if not candles:
            return RunnerResult(
                action="REJECT",
                execution=None,
                reason="candles cannot be empty",
            )

        candle_objects = [
            Candle(
                symbol=symbol,
                timeframe=candle.get("timeframe", "unknown"),
                open=candle["open"],
                high=candle["high"],
                low=candle["low"],
                close=candle["close"],
                volume=candle.get("volume", 0.0),
                timestamp=candle.get("timestamp", timestamp),
            )
            if isinstance(candle, dict)
            else candle
            for candle in candles
        ]

        latest_candle = candle_objects[-1]

        try:
            latest_high = float(latest_candle.high)
            latest_low = float(latest_candle.low)
        except (AttributeError, TypeError, ValueError):
            return RunnerResult(
                action="REJECT",
                execution=None,
                reason="invalid latest candle range",
            )

        existing_position = (
            self.execution_engine.position_manager.get_position(symbol)
        )

        if existing_position is not None:
            if existing_position.stop_price is None:
                return RunnerResult(
                    action="REJECT",
                    execution=None,
                    reason="existing position is missing stop price",
                )

            if self.stop_policy.is_triggered(
                side=existing_position.side,
                stop_price=existing_position.stop_price,
                high=latest_high,
                low=latest_low,
            ):
                if existing_position.side == "LONG":
                    if latest_candle.open <= existing_position.stop_price:
                        stop_exit_price = float(latest_candle.open)
                    else:
                        stop_exit_price = existing_position.stop_price
                else:
                    if latest_candle.open >= existing_position.stop_price:
                        stop_exit_price = float(latest_candle.open)
                    else:
                        stop_exit_price = existing_position.stop_price

                close_result = self.close(
                    symbol=symbol,
                    exit_price=stop_exit_price,
                    timestamp=timestamp,
                    exit_reason="stop loss",
                )

                return close_result

        signal = strategy.generate_signal(
            symbol=symbol,
            candles=candle_objects,
        )

        if signal.side == "FLAT":
            return RunnerResult(
                action="HOLD",
                execution=None,
                reason="strategy returned FLAT",
            )

        if signal.side not in {"LONG", "SHORT"}:
            return RunnerResult(
                action="REJECT",
                execution=None,
                reason=f"invalid strategy signal: {signal}",
            )

        if signal.atr is None:
            return RunnerResult(
                action="REJECT",
                execution=None,
                reason="strategy signal is missing ATR",
            )

        stop_price = self.stop_policy.calculate(
            side=signal.side,
            entry_price=signal.price,
            atr=signal.atr,
        )

        latest_candle = candle_objects[-1]

        try:
            entry_price = float(latest_candle.close)
        except (AttributeError, TypeError, ValueError):
            return RunnerResult(
                action="REJECT",
                execution=None,
                reason="invalid latest candle close",
            )

        if self.execution_engine.position_manager.has_position(symbol):
            position = self.execution_engine.position_manager.get_position(
                symbol
            )

            if position is not None and position.side == signal.side:
                return RunnerResult(
                    action="HOLD",
                    execution=None,
                    reason="existing position matches signal",
                )

            close_result = self.close(
                symbol=symbol,
                exit_price=entry_price,
                timestamp=timestamp,
                exit_reason="strategy reversal",
            )

            if close_result.execution is None:
                return close_result

            if not close_result.execution.result.success:
                return close_result

        return self.open(
            symbol=symbol,
            side=signal.side,
            entry_price=entry_price,
            stop_price=stop_price,
            timestamp=timestamp,
        )