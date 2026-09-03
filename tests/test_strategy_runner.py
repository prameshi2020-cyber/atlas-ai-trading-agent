from datetime import datetime, timezone
import pytest

from app.strategy.stop_policy import StopPolicy
from app.risk.manager import RiskManager
from app.risk.models import RiskConfig
from app.trading.broker import PaperBroker
from app.trading.execution import ExecutionEngine
from app.trading.position_manager import PositionManager
from app.trading.runner import PaperTradingRunner
from app.strategy.engine import StrategyEngine


TIMESTAMP = datetime.now(timezone.utc)


def make_runner() -> PaperTradingRunner:
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

    return PaperTradingRunner(
        execution_engine=execution_engine,
        stop_policy=StopPolicy(
            atr_multiplier=1.5,
        ),
    )



def make_candles(prices: list[float]) -> list[dict]:
    return [
        {
            "open": price,
            "high": price + 2,
            "low": price - 2,
            "close": price,
            "volume": 1000.0,
        }
        for price in prices
    ]


def test_strategy_runner_can_open_long():
    runner = make_runner()
    strategy = StrategyEngine()

    candles = make_candles(
        [100, 100, 100, 100, 101, 102]
    )

    result = runner.process_candles(
        symbol="BTCUSDT",
        candles=candles,
        strategy=strategy,
        timestamp=TIMESTAMP,
    )

    assert result.action == "OPEN"
    assert result.execution is not None
    assert result.execution.result.success

    position = runner.execution_engine.position_manager.get_position(
        "BTCUSDT"
    )

    assert position is not None
    assert position.side == "LONG"


def test_strategy_runner_does_nothing_when_flat():
    runner = make_runner()
    strategy = StrategyEngine()

    candles = make_candles(
        [100, 100, 100, 100, 100, 100]
    )

    result = runner.process_candles(
        symbol="BTCUSDT",
        candles=candles,
        strategy=strategy,
        timestamp=TIMESTAMP,
    )

    assert result.action == "HOLD"
    assert result.execution is None


def test_strategy_runner_rejects_empty_candles():
    runner = make_runner()
    strategy = StrategyEngine()

    result = runner.process_candles(
        symbol="BTCUSDT",
        candles=[],
        strategy=strategy,
        timestamp=TIMESTAMP,
    )

    assert result.action == "REJECT"
    assert result.execution is None

def test_strategy_runner_calculates_stop_from_strategy_atr():
    runner = make_runner()
    strategy = StrategyEngine(
        fast_period=3,
        slow_period=5,
        atr_period=3,
    )

    candles = make_candles(
        [100, 100, 100, 110, 120]
    )

    result = runner.process_candles(
        symbol="BTCUSDT",
        candles=candles,
        strategy=strategy,
        timestamp=TIMESTAMP,
    )

    assert result.action == "OPEN"
    assert result.execution is not None
    assert result.execution.result.success

    position = runner.execution_engine.position_manager.get_position(
        "BTCUSDT"
    )

    assert position is not None
    assert position.entry_price == 120.0

def test_strategy_runner_passes_atr_stop_price_to_execution():
    runner = make_runner()
    strategy = StrategyEngine(
        fast_period=3,
        slow_period=5,
        atr_period=3,
    )

    candles = make_candles(
        [100, 100, 100, 110, 120]
    )

    captured = {}

    original_open = runner.execution_engine.open_position

    def spy_open_position(
        *,
        symbol,
        side,
        entry_price,
        stop_price,
        timestamp,
    ):
        captured["stop_price"] = stop_price

        return original_open(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            stop_price=stop_price,
            timestamp=timestamp,
        )

    runner.execution_engine.open_position = spy_open_position

    result = runner.process_candles(
        symbol="BTCUSDT",
        candles=candles,
        strategy=strategy,
        timestamp=TIMESTAMP,
    )

    assert result.action == "OPEN"
    assert result.execution is not None
    assert result.execution.result.success

    assert captured["stop_price"] == pytest.approx(106.0)