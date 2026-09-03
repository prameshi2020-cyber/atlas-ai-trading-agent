import asyncio
from datetime import datetime, timezone

import pytest

from app.trading.runner import RunnerResult


TIMESTAMP = datetime(
    2026,
    9,
    2,
    20,
    0,
    0,
    tzinfo=timezone.utc,
)


class FakeCollector:
    def __init__(self):
        self.calls = 0
        self.events = []

    async def collect_once(self):
        self.calls += 1
        self.events.append("collect")


class FakeTradingService:
    def __init__(self, results=None, errors=None, events=None):
        self.calls = []
        self.results = results or {}
        self.errors = errors or {}
        self.events = events if events is not None else []

    def process_symbol(
        self,
        *,
        symbol,
        strategy,
        timestamp,
    ):
        self.calls.append(
            {
                "symbol": symbol,
                "strategy": strategy,
                "timestamp": timestamp,
            }
        )

        self.events.append(f"trade:{symbol}")

        if symbol in self.errors:
            raise self.errors[symbol]

        return self.results.get(
            symbol,
            RunnerResult(
                action="HOLD",
                execution=None,
                reason="test",
            ),
        )


def test_orchestrator_collects_before_processing_symbols():
    from app.trading.orchestrator import TradingOrchestrator

    events = []

    collector = FakeCollector()
    collector.events = events

    service = FakeTradingService(events=events)

    strategies = {
        "BTCUSDT": object(),
        "ETHUSDT": object(),
    }

    orchestrator = TradingOrchestrator(
        collector=collector,
        trading_service=service,
        strategies=strategies,
        interval_seconds=60,
    )

    results = asyncio.run(
        orchestrator.run_once(timestamp=TIMESTAMP)
    )

    assert events == [
        "collect",
        "trade:BTCUSDT",
        "trade:ETHUSDT",
    ]

    assert collector.calls == 1
    assert len(service.calls) == 2

    assert results["BTCUSDT"].action == "HOLD"
    assert results["ETHUSDT"].action == "HOLD"


def test_orchestrator_passes_correct_strategy_and_timestamp():
    from app.trading.orchestrator import TradingOrchestrator

    collector = FakeCollector()
    service = FakeTradingService()

    btc_strategy = object()
    eth_strategy = object()

    orchestrator = TradingOrchestrator(
        collector=collector,
        trading_service=service,
        strategies={
            "BTCUSDT": btc_strategy,
            "ETHUSDT": eth_strategy,
        },
        interval_seconds=60,
    )

    asyncio.run(
        orchestrator.run_once(timestamp=TIMESTAMP)
    )

    assert service.calls == [
        {
            "symbol": "BTCUSDT",
            "strategy": btc_strategy,
            "timestamp": TIMESTAMP,
        },
        {
            "symbol": "ETHUSDT",
            "strategy": eth_strategy,
            "timestamp": TIMESTAMP,
        },
    ]


def test_orchestrator_isolates_symbol_failure():
    from app.trading.orchestrator import TradingOrchestrator

    collector = FakeCollector()

    service = FakeTradingService(
        errors={
            "BTCUSDT": RuntimeError("BTC failed"),
        }
    )

    orchestrator = TradingOrchestrator(
        collector=collector,
        trading_service=service,
        strategies={
            "BTCUSDT": object(),
            "ETHUSDT": object(),
        },
        interval_seconds=60,
    )

    results = asyncio.run(
        orchestrator.run_once(timestamp=TIMESTAMP)
    )

    assert "BTCUSDT" in results
    assert "ETHUSDT" in results

    assert results["BTCUSDT"].action == "REJECT"
    assert results["BTCUSDT"].reason == "trading service error"

    assert results["ETHUSDT"].action == "HOLD"


def test_orchestrator_rejects_invalid_interval():
    from app.trading.orchestrator import TradingOrchestrator

    with pytest.raises(ValueError, match="interval_seconds"):
        TradingOrchestrator(
            collector=FakeCollector(),
            trading_service=FakeTradingService(),
            strategies={"BTCUSDT": object()},
            interval_seconds=0,
        )


def test_orchestrator_requires_strategy_for_each_symbol():
    from app.trading.orchestrator import TradingOrchestrator

    with pytest.raises(ValueError, match="strategy"):
        TradingOrchestrator(
            collector=FakeCollector(),
            trading_service=FakeTradingService(),
            strategies={},
            interval_seconds=60,
        )


def test_orchestrator_run_can_be_cancelled():
    from app.trading.orchestrator import TradingOrchestrator

    class CancellingCollector:
        async def collect_once(self):
            raise asyncio.CancelledError

    orchestrator = TradingOrchestrator(
        collector=CancellingCollector(),
        trading_service=FakeTradingService(),
        strategies={"BTCUSDT": object()},
        interval_seconds=60,
    )

    async def scenario():
        await orchestrator.run()

    asyncio.run(scenario())


def test_orchestrator_rejects_invalid_constructor_dependencies():
    from app.trading.orchestrator import TradingOrchestrator

    with pytest.raises(ValueError, match="collector"):
        TradingOrchestrator(
            collector=None,
            trading_service=FakeTradingService(),
            strategies={"BTCUSDT": object()},
            interval_seconds=60,
        )

    with pytest.raises(ValueError, match="trading_service"):
        TradingOrchestrator(
            collector=FakeCollector(),
            trading_service=None,
            strategies={"BTCUSDT": object()},
            interval_seconds=60,
        )