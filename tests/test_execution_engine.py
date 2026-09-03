from datetime import datetime, timezone

from app.database.models import initialize_position_database
from app.risk.manager import RiskManager
from app.risk.models import RiskConfig
from app.trading.broker import PaperBroker
from app.trading.execution import ExecutionEngine
from app.trading.position_manager import PositionManager


TIMESTAMP = datetime.now(timezone.utc)


def make_engine(
    *,
    cash: float = 10_000.0,
) -> ExecutionEngine:
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

    return ExecutionEngine(
        broker=broker,
        risk_manager=risk_manager,
        position_manager=position_manager,
    )


def test_open_long_position():
    engine = make_engine()

    execution = engine.open_position(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    assert execution.result.success
    assert execution.result.side == "BUY"
    assert execution.result.quantity > 0

    position = engine.position_manager.get_position("BTCUSDT")

    assert position is not None
    assert position.side == "LONG"


def test_open_short_position():
    engine = make_engine()

    execution = engine.open_position(
        symbol="BTCUSDT",
        side="SHORT",
        entry_price=100.0,
        stop_price=105.0,
        timestamp=TIMESTAMP,
    )

    assert execution.result.success
    assert execution.result.side == "SELL"

    position = engine.position_manager.get_position("BTCUSDT")

    assert position is not None
    assert position.side == "SHORT"


def test_risk_rejection_does_not_create_position():
    engine = make_engine()

    engine.risk_manager.state.halted = True
    engine.risk_manager.state.halt_reason = "test halt"

    execution = engine.open_position(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    assert not execution.result.success
    assert execution.result.reason == "test halt"
    assert not engine.position_manager.has_position("BTCUSDT")


def test_close_long_position_calculates_pnl():
    engine = make_engine()

    engine.open_position(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    execution = engine.close_position(
        symbol="BTCUSDT",
        exit_price=110.0,
        timestamp=TIMESTAMP,
    )

    assert execution.result.success
    assert execution.result.side == "SELL"
    assert execution.pnl is not None
    assert execution.pnl > 0
    assert not engine.position_manager.has_position("BTCUSDT")


def test_close_short_position_calculates_pnl():
    engine = make_engine()

    engine.open_position(
        symbol="BTCUSDT",
        side="SHORT",
        entry_price=100.0,
        stop_price=105.0,
        timestamp=TIMESTAMP,
    )

    execution = engine.close_position(
        symbol="BTCUSDT",
        exit_price=90.0,
        timestamp=TIMESTAMP,
    )

    assert execution.result.success
    assert execution.result.side == "BUY"
    assert execution.pnl is not None
    assert execution.pnl > 0
    assert not engine.position_manager.has_position("BTCUSDT")


def test_close_without_position_is_rejected():
    engine = make_engine()

    execution = engine.close_position(
        symbol="BTCUSDT",
        exit_price=100.0,
        timestamp=TIMESTAMP,
    )

    assert not execution.result.success
    assert execution.result.reason == "no open position"

def test_open_position_is_rejected_when_symbol_already_has_position():
    engine = make_engine()

    first = engine.open_position(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    assert first.result.success is True

    cash_after_first_trade = engine.broker.cash

    second = engine.open_position(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=105.0,
        stop_price=100.0,
        timestamp=TIMESTAMP,
    )

    assert second.result.success is False
    assert second.result.reason == "position already exists"

    assert engine.broker.cash == cash_after_first_trade

    position = engine.position_manager.get_position("BTCUSDT")

    assert position is not None
    assert position.entry_price == 100.0

def test_failed_position_registration_rolls_back_broker_state():
    engine = make_engine()

    initial_cash = engine.broker.cash

    def failing_open_position(*, result, stop_price):
        raise RuntimeError("position registration failed")

    engine.position_manager.open_position = failing_open_position

    execution = engine.open_position(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    assert execution.result.success is False
    assert execution.result.reason == "position registration failed"

    assert engine.broker.cash == initial_cash
    assert not engine.position_manager.has_position("BTCUSDT")

def test_failed_risk_registration_rolls_back_position_and_broker_state():
    engine = make_engine()

    initial_cash = engine.broker.cash

    def failing_register_position_open():
        raise RuntimeError("risk registration failed")

    engine.risk_manager.register_position_open = (
        failing_register_position_open
    )

    execution = engine.open_position(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    assert execution.result.success is False
    assert execution.result.reason == "risk registration failed"

    assert engine.broker.cash == initial_cash
    assert not engine.position_manager.has_position("BTCUSDT")

def test_open_position_persists_stop_price():
    initialize_position_database()

    engine = make_engine()

    execution = engine.open_position(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    assert execution.result.success

    position = engine.position_manager.get_position("BTCUSDT")

    assert position is not None
    assert position.stop_price == 95.0

def test_false_risk_registration_rolls_back_position_and_broker_state():
    engine = make_engine()

    initial_cash = engine.broker.cash
    initial_concurrent_positions = (
        engine.risk_manager.state.concurrent_positions
    )

    def rejected_register_position_open():
        return False

    engine.risk_manager.register_position_open = (
        rejected_register_position_open
    )

    execution = engine.open_position(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    assert execution.result.success is False
    assert (
        execution.result.reason
        == "risk manager rejected position registration"
    )

    assert engine.broker.cash == initial_cash
    assert not engine.position_manager.has_position("BTCUSDT")

    assert (
        engine.risk_manager.state.concurrent_positions
        == initial_concurrent_positions
    )

def test_failed_position_close_rolls_back_broker_state():
    engine = make_engine()

    opening = engine.open_position(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    assert opening.result.success

    initial_cash = engine.broker.cash
    initial_position = engine.position_manager.get_position("BTCUSDT")

    assert initial_position is not None

    def failing_close_position(*, result):
        raise RuntimeError("position close failed")

    engine.position_manager.close_position = failing_close_position

    try:
        engine.close_position(
            symbol="BTCUSDT",
            exit_price=110.0,
            timestamp=TIMESTAMP,
        )
    except RuntimeError as exc:
        assert str(exc) == "position close failed"
    else:
        raise AssertionError(
            "expected position close failure"
        )

    assert engine.broker.cash == initial_cash
    assert engine.position_manager.has_position("BTCUSDT")

    current_position = engine.position_manager.get_position(
        "BTCUSDT"
    )

    assert current_position == initial_position

def test_failed_realized_pnl_rolls_back_broker_and_position_state():
    engine = make_engine()

    opening = engine.open_position(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    assert opening.result.success

    initial_cash = engine.broker.cash
    initial_position = engine.position_manager.get_position("BTCUSDT")

    assert initial_position is not None

    initial_concurrent_positions = (
        engine.risk_manager.state.concurrent_positions
    )
    initial_daily_pnl = (
        engine.risk_manager.state.realized_daily_pnl
    )

    def failing_record_realized_pnl(*, pnl, timestamp):
        raise RuntimeError("realized pnl recording failed")

    engine.risk_manager.record_realized_pnl = (
        failing_record_realized_pnl
    )

    try:
        engine.close_position(
            symbol="BTCUSDT",
            exit_price=110.0,
            timestamp=TIMESTAMP,
        )
    except RuntimeError as exc:
        assert str(exc) == "realized pnl recording failed"
    else:
        raise AssertionError(
            "expected realized pnl recording failure"
        )

    assert engine.broker.cash == initial_cash
    assert engine.position_manager.has_position("BTCUSDT")

    current_position = engine.position_manager.get_position(
        "BTCUSDT"
    )

    assert current_position == initial_position

    assert (
        engine.risk_manager.state.concurrent_positions
        == initial_concurrent_positions
    )

    assert (
        engine.risk_manager.state.realized_daily_pnl
        == initial_daily_pnl
    )