from datetime import datetime, timezone

from app.trading.journal import initialize_trade_journal
from app.risk.manager import RiskManager
from app.risk.models import RiskConfig
from app.strategy.stop_policy import StopPolicy
from app.trading.broker import PaperBroker
from app.trading.execution import ExecutionEngine
from app.trading.position_manager import PositionManager
from app.trading.runner import PaperTradingRunner


TIMESTAMP = datetime.now(timezone.utc)


def make_runner(cash: float = 10_000.0):
    broker = PaperBroker(
        initial_cash=cash,
        fee_rate=0.001,
    )

    risk_manager = RiskManager(
        initial_equity=cash,
        config=RiskConfig(
            risk_per_trade=0.01,
            maximum_position_pct=1.0,
            minimum_position_size=0.0,
            max_concurrent_positions=3,
            daily_loss_limit_pct=0.05,
            max_drawdown_pct=0.10,
        ),
    )

    position_manager = PositionManager()

    engine = ExecutionEngine(
        broker=broker,
        risk_manager=risk_manager,
        position_manager=position_manager,
    )

    return PaperTradingRunner(
        execution_engine=engine,
        stop_policy=StopPolicy(
            atr_multiplier=1.5,
        ),
    )


def test_long_profit_flow():
    runner = make_runner()

    opened = runner.open(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    assert opened.execution is not None
    assert opened.execution.result.success

    closed = runner.close(
        symbol="BTCUSDT",
        exit_price=110.0,
        timestamp=TIMESTAMP,
        exit_reason="take_profit",
    )

    assert closed.execution is not None
    assert closed.execution.result.success
    assert closed.execution.pnl is not None
    assert closed.execution.pnl > 0


def test_long_loss_flow():
    runner = make_runner()

    runner.open(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    closed = runner.close(
        symbol="BTCUSDT",
        exit_price=90.0,
        timestamp=TIMESTAMP,
        exit_reason="stop_loss",
    )

    assert closed.execution is not None
    assert closed.execution.result.success
    assert closed.execution.pnl is not None
    assert closed.execution.pnl < 0


def test_short_profit_flow():
    runner = make_runner()

    runner.open(
        symbol="BTCUSDT",
        side="SHORT",
        entry_price=100.0,
        stop_price=105.0,
        timestamp=TIMESTAMP,
    )

    closed = runner.close(
        symbol="BTCUSDT",
        exit_price=90.0,
        timestamp=TIMESTAMP,
        exit_reason="take_profit",
    )

    assert closed.execution is not None
    assert closed.execution.result.success
    assert closed.execution.pnl is not None
    assert closed.execution.pnl > 0


def test_short_loss_flow():
    runner = make_runner()

    runner.open(
        symbol="BTCUSDT",
        side="SHORT",
        entry_price=100.0,
        stop_price=105.0,
        timestamp=TIMESTAMP,
    )

    closed = runner.close(
        symbol="BTCUSDT",
        exit_price=110.0,
        timestamp=TIMESTAMP,
        exit_reason="stop_loss",
    )

    assert closed.execution is not None
    assert closed.execution.result.success
    assert closed.execution.pnl is not None
    assert closed.execution.pnl < 0


def test_closed_position_is_removed():
    runner = make_runner()

    runner.open(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    assert runner.execution_engine.position_manager.has_position(
        "BTCUSDT"
    )

    runner.close(
        symbol="BTCUSDT",
        exit_price=110.0,
        timestamp=TIMESTAMP,
    )

    assert not runner.execution_engine.position_manager.has_position(
        "BTCUSDT"
    )


def test_realized_daily_pnl_is_updated_after_close():
    runner = make_runner()

    runner.open(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    closed = runner.close(
        symbol="BTCUSDT",
        exit_price=110.0,
        timestamp=TIMESTAMP,
    )

    assert closed.execution is not None
    assert closed.execution.pnl is not None

    assert (
        runner.execution_engine.risk_manager.state.realized_daily_pnl
        == closed.execution.pnl
    )


def test_risk_rejection_does_not_create_position():
    runner = make_runner()

    risk_manager = runner.execution_engine.risk_manager
    risk_manager.state.halted = True
    risk_manager.state.halt_reason = "test halt"

    result = runner.open(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    assert result.execution is not None
    assert not result.execution.result.success

    assert not runner.execution_engine.position_manager.has_position(
        "BTCUSDT"
    )


def test_duplicate_position_is_rejected():
    runner = make_runner()

    first = runner.open(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    assert first.execution is not None
    assert first.execution.result.success

    second = runner.open(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    assert second.execution is None
    assert second.reason == "position already exists"


def test_close_without_position_is_rejected():
    runner = make_runner()

    result = runner.close(
        symbol="BTCUSDT",
        exit_price=100.0,
        timestamp=TIMESTAMP,
    )

    assert result.execution is None
    assert result.reason == "no open position"