from datetime import datetime, timezone

from app.database.connection import get_connection
from app.trading.journal import initialize_trade_journal
from app.risk.manager import RiskManager
from app.risk.models import RiskConfig
from app.trading.broker import PaperBroker
from app.trading.execution import ExecutionEngine
from app.trading.position_manager import PositionManager
from app.trading.runner import PaperTradingRunner


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

    position_manager = PositionManager()

    engine = ExecutionEngine(
        broker=broker,
        risk_manager=risk_manager,
        position_manager=position_manager,
    )

    initialize_trade_journal()
    return PaperTradingRunner(
        execution_engine=engine,
    )


def test_runner_opens_long_position():
    runner = make_runner()

    result = runner.open(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    assert result.action == "OPEN"
    assert result.execution is not None
    assert result.execution.result.success


def test_runner_opens_short_position():
    runner = make_runner()

    result = runner.open(
        symbol="BTCUSDT",
        side="SHORT",
        entry_price=100.0,
        stop_price=105.0,
        timestamp=TIMESTAMP,
    )

    assert result.action == "OPEN"
    assert result.execution is not None
    assert result.execution.result.success


def test_runner_rejects_duplicate_position():
    runner = make_runner()

    runner.open(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    result = runner.open(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=101.0,
        stop_price=96.0,
        timestamp=TIMESTAMP,
    )

    assert result.action == "OPEN"
    assert result.execution is None
    assert result.reason == "position already exists"


def test_runner_closes_position():
    runner = make_runner()

    runner.open(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    result = runner.close(
        symbol="BTCUSDT",
        exit_price=110.0,
        timestamp=TIMESTAMP,
    )

    assert result.action == "CLOSE"
    assert result.execution is not None
    assert result.execution.result.success
    assert result.execution.pnl is not None
    assert result.execution.pnl > 0


def test_runner_rejects_close_without_position():
    runner = make_runner()

    result = runner.close(
        symbol="BTCUSDT",
        exit_price=100.0,
        timestamp=TIMESTAMP,
    )

    assert result.action == "CLOSE"
    assert result.execution is None
    assert result.reason == "no open position"

def setup_function():
    initialize_trade_journal()

    connection = get_connection()

    try:
        connection.execute("DELETE FROM trade_journal")
        connection.commit()
    finally:
        connection.close()

def test_runner_persists_closed_trade():
    runner = make_runner()

    runner.open(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    result = runner.close(
        symbol="BTCUSDT",
        exit_price=110.0,
        timestamp=TIMESTAMP,
        exit_reason="take_profit",
    )

    assert result.execution is not None
    assert result.execution.result.success

    connection = get_connection()

    try:
        row = connection.execute(
            """
            SELECT
                symbol,
                side,
                quantity,
                entry_price,
                exit_price,
                pnl,
                fee,
                status,
                exit_reason
            FROM trade_journal
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()

        assert row is not None
        assert row["symbol"] == "BTCUSDT"
        assert row["side"] == "LONG"
        assert row["quantity"] > 0
        assert row["entry_price"] == 100.0
        assert row["exit_price"] == 110.0
        assert row["pnl"] > 0
        assert row["fee"] >= 0
        assert row["status"] == "CLOSED"
        assert row["exit_reason"] == "take_profit"

    finally:
        connection.close()