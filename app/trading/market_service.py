from __future__ import annotations

from datetime import datetime

from app.database.models import load_candles
from app.trading.coordinator import TradingCoordinator
from app.trading.runner import RunnerResult


class MarketTradingService:
    """
    Connects persisted market candles to the trading coordinator.

    Responsibilities:
    - Load candle history for a symbol.
    - Pass the history to TradingCoordinator.
    - Return the coordinator's result.

    It does not own strategy, risk, execution, or market collection.
    """

    def __init__(
        self,
        *,
        candle_repository=None,
        coordinator: TradingCoordinator,
    ) -> None:
        self.candle_repository = (
            candle_repository
            if candle_repository is not None
            else _DatabaseCandleRepository()
        )
        self.coordinator = coordinator

    def process_symbol(
        self,
        *,
        symbol: str,
        strategy,
        timestamp: datetime,
    ) -> RunnerResult:
        candles = self.candle_repository.load_candles(
            symbol=symbol,
            timeframe="1m",
        )

        if not candles:
            return RunnerResult(
                action="REJECT",
                execution=None,
                reason="no candles available",
            )

        return self.coordinator.process_candles(
            symbol=symbol,
            candles=candles,
            strategy=strategy,
            timestamp=timestamp,
        )


class _DatabaseCandleRepository:
    def load_candles(
        self,
        *,
        symbol: str,
        timeframe: str,
    ):
        return load_candles(
            symbol=symbol,
            timeframe=timeframe,
        )