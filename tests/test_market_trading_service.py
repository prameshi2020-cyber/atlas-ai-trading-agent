from datetime import datetime, timedelta, timezone

import pytest

from app.market.models import Candle
from app.trading.coordinator import TradingCoordinator
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


class FakeCandleRepository:
    def __init__(self, candles):
        self.candles = candles
        self.calls = []

    def load_candles(
        self,
        *,
        symbol: str,
        timeframe: str,
    ) -> list[Candle]:
        self.calls.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
            }
        )

        return [
            candle
            for candle in self.candles
            if candle.symbol == symbol
            and candle.timeframe == timeframe
        ]


class FakeCoordinator:
    def __init__(self):
        self.calls = []

    def process_candles(
        self,
        *,
        symbol: str,
        candles: list[dict],
        strategy,
        timestamp: datetime,
    ) -> RunnerResult:
        self.calls.append(
            {
                "symbol": symbol,
                "candles": candles,
                "strategy": strategy,
                "timestamp": timestamp,
            }
        )

        return RunnerResult(
            action="HOLD",
            execution=None,
            reason="test",
        )


def make_candles() -> list[Candle]:
    return [
        Candle(
            symbol="BTCUSDT",
            timeframe="1m",
            open=100.0,
            high=102.0,
            low=99.0,
            close=101.0,
            volume=1000.0,
            timestamp=TIMESTAMP,
        ),
        Candle(
            symbol="BTCUSDT",
            timeframe="1m",
            open=101.0,
            high=103.0,
            low=100.0,
            close=102.0,
            volume=1100.0,
            timestamp=TIMESTAMP + timedelta(minutes=1),
        ),
    ]


def test_service_loads_candles_for_requested_symbol_and_timeframe():
    from app.trading.market_service import MarketTradingService

    repository = FakeCandleRepository(make_candles())
    coordinator = FakeCoordinator()

    service = MarketTradingService(
        candle_repository=repository,
        coordinator=coordinator,
    )

    service.process_symbol(
        symbol="BTCUSDT",
        strategy=object(),
        timestamp=TIMESTAMP,
    )

    assert repository.calls == [
        {
            "symbol": "BTCUSDT",
            "timeframe": "1m",
        }
    ]


def test_service_passes_loaded_candles_to_coordinator():
    from app.trading.market_service import MarketTradingService

    candles = make_candles()
    repository = FakeCandleRepository(candles)
    coordinator = FakeCoordinator()
    strategy = object()

    service = MarketTradingService(
        candle_repository=repository,
        coordinator=coordinator,
    )

    service.process_symbol(
        symbol="BTCUSDT",
        strategy=strategy,
        timestamp=TIMESTAMP,
    )

    assert len(coordinator.calls) == 1
    assert coordinator.calls[0]["symbol"] == "BTCUSDT"
    assert coordinator.calls[0]["candles"] == candles
    assert coordinator.calls[0]["strategy"] is strategy
    assert coordinator.calls[0]["timestamp"] == TIMESTAMP


def test_service_returns_coordinator_result():
    from app.trading.market_service import MarketTradingService

    repository = FakeCandleRepository(make_candles())
    coordinator = FakeCoordinator()

    service = MarketTradingService(
        candle_repository=repository,
        coordinator=coordinator,
    )

    result = service.process_symbol(
        symbol="BTCUSDT",
        strategy=object(),
        timestamp=TIMESTAMP,
    )

    assert result.action == "HOLD"
    assert result.execution is None
    assert result.reason == "test"


def test_service_rejects_empty_candle_history():
    from app.trading.market_service import MarketTradingService

    repository = FakeCandleRepository([])
    coordinator = FakeCoordinator()

    service = MarketTradingService(
        candle_repository=repository,
        coordinator=coordinator,
    )

    result = service.process_symbol(
        symbol="BTCUSDT",
        strategy=object(),
        timestamp=TIMESTAMP,
    )

    assert result.action == "REJECT"
    assert result.execution is None
    assert result.reason == "no candles available"

    assert coordinator.calls == []


def test_service_uses_default_one_minute_timeframe():
    from app.trading.market_service import MarketTradingService

    repository = FakeCandleRepository(make_candles())
    coordinator = FakeCoordinator()

    service = MarketTradingService(
        candle_repository=repository,
        coordinator=coordinator,
    )

    service.process_symbol(
        symbol="BTCUSDT",
        strategy=object(),
        timestamp=TIMESTAMP,
    )

    assert repository.calls[0]["timeframe"] == "1m"

def test_service_with_real_coordinator_accepts_candle_objects():
    from dataclasses import dataclass

    from app.trading.market_service import MarketTradingService
    from app.trading.runner import PaperTradingRunner

    @dataclass
    class TestSignal:
        symbol: str
        side: str
        price: float
        timestamp: datetime
        atr: float | None = None

    class LongStrategy:
        def generate_signal(self, *, symbol, candles):
            assert all(isinstance(candle, Candle) for candle in candles)

            return TestSignal(
                symbol=symbol,
                side="LONG",
                price=candles[-1].close,
                timestamp=candles[-1].timestamp,
                atr=2.0,
            )

    from tests.test_runner import make_runner

    runner: PaperTradingRunner = make_runner()
    coordinator = TradingCoordinator(runner=runner)

    repository = FakeCandleRepository(
        [
            Candle(
                symbol="BTCUSDT",
                timeframe="1m",
                open=100.0,
                high=102.0,
                low=99.0,
                close=101.0,
                volume=1000.0,
                timestamp=TIMESTAMP,
            ),
            Candle(
                symbol="BTCUSDT",
                timeframe="1m",
                open=101.0,
                high=104.0,
                low=100.0,
                close=103.0,
                volume=1100.0,
                timestamp=TIMESTAMP + timedelta(minutes=1),
            ),
        ]
    )

    service = MarketTradingService(
        candle_repository=repository,
        coordinator=coordinator,
    )

    result = service.process_symbol(
        symbol="BTCUSDT",
        strategy=LongStrategy(),
        timestamp=TIMESTAMP + timedelta(minutes=1),
    )

    assert result.action == "OPEN", result.reason
    assert result.execution is not None
    assert result.execution.result.success

    position = runner.execution_engine.position_manager.get_position(
        "BTCUSDT"
    )

    assert position is not None
    assert position.side == "LONG"