from __future__ import annotations

from datetime import datetime

from app.trading.runner import PaperTradingRunner, RunnerResult


class TradingCoordinator:
    """
    Coordinates market candles with the paper trading runner.

    The coordinator is intentionally thin:
    strategy, risk management, stop logic, and execution
    remain owned by their respective components.
    """

    def __init__(
        self,
        *,
        runner: PaperTradingRunner,
    ) -> None:
        self.runner = runner

    def process_candles(
        self,
        *,
        symbol: str,
        candles: list[dict],
        strategy,
        timestamp: datetime,
    ) -> RunnerResult:
        return self.runner.process_candles(
            symbol=symbol,
            candles=candles,
            strategy=strategy,
            timestamp=timestamp,
        )