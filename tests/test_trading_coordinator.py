from datetime import datetime, timezone

import pytest

from app.trading.coordinator import TradingCoordinator
from app.trading.runner import RunnerResult
from app.strategy.engine import StrategyEngine
from app.trading.broker import PaperBroker
from app.risk.manager import RiskManager
from app.risk.models import RiskConfig
from app.trading.execution import ExecutionEngine
from app.trading.position_manager import PositionManager
from app.trading.runner import PaperTradingRunner
from app.strategy.stop_policy import StopPolicy

TIMESTAMP = datetime(
    2026,
    9,
    2,
    20,
    0,
    0,
    tzinfo=timezone.utc,
)


def test_coordinator_delegates_candles_to_runner():
    candles = [
        {
            "open": 100.0,
            "high": 102.0,
            "low": 98.0,
            "close": 101.0,
            "volume": 10.0,
            "timestamp": TIMESTAMP,
        }
    ]

    strategy = object()

    expected_result = RunnerResult(
        action="HOLD",
        execution=None,
        reason="strategy returned FLAT",
    )

    class FakeRunner:
        def __init__(self):
            self.calls = []

        def process_candles(
            self,
            *,
            symbol,
            candles,
            strategy,
            timestamp,
        ):
            self.calls.append(
                {
                    "symbol": symbol,
                    "candles": candles,
                    "strategy": strategy,
                    "timestamp": timestamp,
                }
            )

            return expected_result

    runner = FakeRunner()

    coordinator = TradingCoordinator(
        runner=runner,
    )

    result = coordinator.process_candles(
        symbol="BTCUSDT",
        candles=candles,
        strategy=strategy,
        timestamp=TIMESTAMP,
    )

    assert result is expected_result

    assert runner.calls == [
        {
            "symbol": "BTCUSDT",
            "candles": candles,
            "strategy": strategy,
            "timestamp": TIMESTAMP,
        }
    ]

def test_coordinator_delegates_empty_candles_to_runner():
    expected_result = RunnerResult(
        action="REJECT",
        execution=None,
        reason="candles cannot be empty",
    )

    class FakeRunner:
        def __init__(self):
            self.calls = []

        def process_candles(
            self,
            *,
            symbol,
            candles,
            strategy,
            timestamp,
        ):
            self.calls.append(
                {
                    "symbol": symbol,
                    "candles": candles,
                    "strategy": strategy,
                    "timestamp": timestamp,
                }
            )

            return expected_result

    runner = FakeRunner()
    strategy = object()

    coordinator = TradingCoordinator(
        runner=runner,
    )

    result = coordinator.process_candles(
        symbol="BTCUSDT",
        candles=[],
        strategy=strategy,
        timestamp=TIMESTAMP,
    )

    assert result is expected_result

    assert runner.calls == [
        {
            "symbol": "BTCUSDT",
            "candles": [],
            "strategy": strategy,
            "timestamp": TIMESTAMP,
        }
    ]

def test_coordinator_propagates_runner_exception():
    class FakeRunner:
        def process_candles(
            self,
            *,
            symbol,
            candles,
            strategy,
            timestamp,
        ):
            raise RuntimeError("runner failure")

    coordinator = TradingCoordinator(
        runner=FakeRunner(),
    )

    try:
        coordinator.process_candles(
            symbol="BTCUSDT",
            candles=[],
            strategy=object(),
            timestamp=TIMESTAMP,
        )
    except RuntimeError as exc:
        assert str(exc) == "runner failure"
    else:
        raise AssertionError(
            "TradingCoordinator must propagate runner exceptions"
        )

def test_coordinator_passes_strategy_unchanged_to_runner():
    strategy = object()
    received = {}

    class FakeRunner:
        def process_candles(
            self,
            *,
            symbol,
            candles,
            strategy,
            timestamp,
        ):
            received["strategy"] = strategy

            return RunnerResult(
                action="HOLD",
                execution=None,
                reason="strategy returned FLAT",
            )

    coordinator = TradingCoordinator(
        runner=FakeRunner(),
    )

    coordinator.process_candles(
        symbol="BTCUSDT",
        candles=[],
        strategy=strategy,
        timestamp=TIMESTAMP,
    )

    assert received["strategy"] is strategy

def test_coordinator_passes_symbol_unchanged_to_runner():
    received = {}

    class FakeRunner:
        def process_candles(
            self,
            *,
            symbol,
            candles,
            strategy,
            timestamp,
        ):
            received["symbol"] = symbol

            return RunnerResult(
                action="HOLD",
                execution=None,
                reason="strategy returned FLAT",
            )

    coordinator = TradingCoordinator(
        runner=FakeRunner(),
    )

    coordinator.process_candles(
        symbol="ETHUSDT",
        candles=[],
        strategy=object(),
        timestamp=TIMESTAMP,
    )

    assert received["symbol"] == "ETHUSDT"


def test_coordinator_passes_timestamp_unchanged_to_runner():
    received = {}

    class FakeRunner:
        def process_candles(
            self,
            *,
            symbol,
            candles,
            strategy,
            timestamp,
        ):
            received["timestamp"] = timestamp

            return RunnerResult(
                action="HOLD",
                execution=None,
                reason="strategy returned FLAT",
            )

    coordinator = TradingCoordinator(
        runner=FakeRunner(),
    )

    coordinator.process_candles(
        symbol="BTCUSDT",
        candles=[],
        strategy=object(),
        timestamp=TIMESTAMP,
    )

    assert received["timestamp"] is TIMESTAMP


def test_coordinator_passes_candles_without_modifying_them():
    candles = [
        {
            "open": 100.0,
            "high": 102.0,
            "low": 98.0,
            "close": 101.0,
            "volume": 25.0,
            "timestamp": TIMESTAMP,
        },
        {
            "open": 101.0,
            "high": 104.0,
            "low": 100.0,
            "close": 103.0,
            "volume": 30.0,
            "timestamp": TIMESTAMP,
        },
    ]

    received = {}

    class FakeRunner:
        def process_candles(
            self,
            *,
            symbol,
            candles,
            strategy,
            timestamp,
        ):
            received["candles"] = candles

            return RunnerResult(
                action="HOLD",
                execution=None,
                reason="strategy returned FLAT",
            )

    coordinator = TradingCoordinator(
        runner=FakeRunner(),
    )

    coordinator.process_candles(
        symbol="BTCUSDT",
        candles=candles,
        strategy=object(),
        timestamp=TIMESTAMP,
    )

    assert received["candles"] is candles
    assert received["candles"] == candles


def test_coordinator_returns_runner_result_without_modifying_it():
    expected_result = RunnerResult(
        action="CLOSE",
        execution=None,
        reason="stop loss",
    )

    class FakeRunner:
        def process_candles(
            self,
            *,
            symbol,
            candles,
            strategy,
            timestamp,
        ):
            return expected_result

    coordinator = TradingCoordinator(
        runner=FakeRunner(),
    )

    result = coordinator.process_candles(
        symbol="BTCUSDT",
        candles=[],
        strategy=object(),
        timestamp=TIMESTAMP,
    )

    assert result is expected_result
    assert result.action == "CLOSE"
    assert result.reason == "stop loss"

def make_real_coordinator() -> TradingCoordinator:
    broker = PaperBroker(
        initial_cash=10_000.0,
        fee_rate=0.001,
    )

    risk_manager = RiskManager(
        initial_equity=10_000.0,
        config=RiskConfig(
            risk_per_trade=0.01,
            maximum_position_pct=1.0,
            minimum_position_size=0.0,
            max_concurrent_positions=3,
            daily_loss_limit_pct=0.05,
            max_drawdown_pct=0.10,
        ),
    )

    execution_engine = ExecutionEngine(
        broker=broker,
        risk_manager=risk_manager,
        position_manager=PositionManager(),
    )

    runner = PaperTradingRunner(
        execution_engine=execution_engine,
        stop_policy=StopPolicy(
            atr_multiplier=1.5,
        ),
    )

    return TradingCoordinator(
        runner=runner,
    )

def test_coordinator_with_real_strategy_can_open_long():
    coordinator = make_real_coordinator()
    strategy = StrategyEngine()

    candles = [
        {
            "open": price,
            "high": price + 2,
            "low": price - 2,
            "close": price,
            "volume": 1000.0,
        }
        for price in [100, 100, 100, 100, 101, 102]
    ]

    result = coordinator.process_candles(
        symbol="BTCUSDT",
        candles=candles,
        strategy=strategy,
        timestamp=TIMESTAMP,
    )

    assert result.action == "OPEN"
    assert result.execution is not None
    assert result.execution.result.success

    position = coordinator.runner.execution_engine.position_manager.get_position(
        "BTCUSDT"
    )

    assert position is not None
    assert position.side == "LONG"
    assert position.entry_price == 102.0


def test_coordinator_with_real_strategy_holds_when_signal_is_flat():
    coordinator = make_real_coordinator()
    strategy = StrategyEngine()

    candles = [
        {
            "open": price,
            "high": price + 2,
            "low": price - 2,
            "close": price,
            "volume": 1000.0,
        }
        for price in [100, 100, 100, 100, 100, 100]
    ]

    result = coordinator.process_candles(
        symbol="BTCUSDT",
        candles=candles,
        strategy=strategy,
        timestamp=TIMESTAMP,
    )

    assert result.action == "HOLD"
    assert result.execution is None

    position = coordinator.runner.execution_engine.position_manager.get_position(
        "BTCUSDT"
    )

    assert position is None


def test_coordinator_with_real_strategy_calculates_atr_stop():
    coordinator = make_real_coordinator()

    strategy = StrategyEngine(
        fast_period=3,
        slow_period=5,
        atr_period=3,
    )

    candles = [
        {
            "open": price,
            "high": price + 2,
            "low": price - 2,
            "close": price,
            "volume": 1000.0,
        }
        for price in [100, 100, 100, 110, 120]
    ]

    result = coordinator.process_candles(
        symbol="BTCUSDT",
        candles=candles,
        strategy=strategy,
        timestamp=TIMESTAMP,
    )

    assert result.action == "OPEN"
    assert result.execution is not None
    assert result.execution.result.success

    position = coordinator.runner.execution_engine.position_manager.get_position(
        "BTCUSDT"
    )

    assert position is not None

    assert position.side == "LONG"
    assert position.entry_price == 120.0
    assert position.stop_price == pytest.approx(106.0)


def test_coordinator_stop_loss_closes_real_position():
    coordinator = make_real_coordinator()

    strategy = StrategyEngine()

    open_candles = [
        {
            "open": price,
            "high": price + 2,
            "low": price - 2,
            "close": price,
            "volume": 1000.0,
        }
        for price in [100, 100, 100, 100, 101, 102]
    ]

    opened = coordinator.process_candles(
        symbol="BTCUSDT",
        candles=open_candles,
        strategy=strategy,
        timestamp=TIMESTAMP,
    )

    assert opened.action == "OPEN"
    assert opened.execution is not None
    assert opened.execution.result.success

    position = coordinator.runner.execution_engine.position_manager.get_position(
        "BTCUSDT"
    )

    assert position is not None
    assert position.stop_price is not None

    stop_price = position.stop_price

    stop_candles = [
        {
            "open": 102.0,
            "high": 103.0,
            "low": stop_price - 1.0,
            "close": stop_price,
            "volume": 1000.0,
        }
    ]

    closed = coordinator.process_candles(
        symbol="BTCUSDT",
        candles=stop_candles,
        strategy=strategy,
        timestamp=TIMESTAMP,
    )

    assert closed.action == "CLOSE"
    assert closed.execution is not None
    assert closed.execution.result.success
    assert closed.reason == "stop loss"

    assert (
        coordinator.runner.execution_engine.position_manager.get_position(
            "BTCUSDT"
        )
        is None
    )