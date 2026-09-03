from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from app.trading.runner import RunnerResult


logger = logging.getLogger(__name__)


class TradingOrchestrator:
    """
    Coordinates market collection and trading decisions.

    Responsibilities:
    - Collect market data.
    - Process each configured symbol through MarketTradingService.
    - Isolate symbol-level trading failures.
    - Maintain the polling loop.

    It does not own:
    - market data collection logic
    - strategy logic
    - risk management
    - execution
    - persistence
    """

    def __init__(
        self,
        *,
        collector,
        trading_service,
        strategies: dict[str, object],
        interval_seconds: int = 60,
    ) -> None:
        if collector is None:
            raise ValueError("collector is required")

        if trading_service is None:
            raise ValueError("trading_service is required")

        if not strategies:
            raise ValueError(
                "strategy configuration is required"
            )

        if interval_seconds <= 0:
            raise ValueError(
                "interval_seconds must be greater than zero"
            )

        self.collector = collector
        self.trading_service = trading_service
        self.strategies = strategies
        self.interval_seconds = interval_seconds

    async def run_once(
        self,
        *,
        timestamp: datetime | None = None,
    ) -> dict[str, RunnerResult]:
        """
        Execute exactly one market/trading cycle.
        """

        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        await self.collector.collect_once()

        results: dict[str, RunnerResult] = {}

        for symbol, strategy in self.strategies.items():
            try:
                results[symbol] = (
                    self.trading_service.process_symbol(
                        symbol=symbol,
                        strategy=strategy,
                        timestamp=timestamp,
                    )
                )

            except asyncio.CancelledError:
                raise

            except Exception:
                logger.exception(
                    "Trading cycle failed for %s",
                    symbol,
                )

                results[symbol] = RunnerResult(
                    action="REJECT",
                    execution=None,
                    reason="trading service error",
                )

        return results

    async def run(self) -> None:
        """
        Run the trading cycle continuously.
        """

        try:
            while True:
                await self.run_once()

                await asyncio.sleep(
                    self.interval_seconds
                )

        except asyncio.CancelledError:
            logger.info(
                "TradingOrchestrator stopped by cancellation."
            )
            return