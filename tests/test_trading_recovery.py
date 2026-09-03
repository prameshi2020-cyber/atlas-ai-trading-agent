from copy import deepcopy

from datetime import datetime, timezone

import pytest


from app.trading.journal import initialize_trade_journal
from app.risk.manager import RiskManager
from app.risk.models import RiskConfig
from app.trading.bootstrap import create_paper_trading_engine
from app.trading import execution as execution_module

TIMESTAMP = datetime.now(timezone.utc)


def make_engine():
    return create_paper_trading_engine(
        initial_cash=10_000.0,
        fee_rate=0.001,
        risk_config=RiskConfig(
            risk_per_trade=0.01,
            daily_loss_limit_pct=0.05,
            max_drawdown_pct=0.10,
            max_concurrent_positions=3,
            maximum_position_pct=1.0,
            minimum_position_size=0.0,
        ),
    )


def test_position_and_risk_count_survive_engine_restart():
    engine = make_engine()

    opening = engine.open_position(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    assert opening.result.success

    original_position = (
        engine.position_manager.get_position("BTCUSDT")
    )

    assert original_position is not None
    assert engine.risk_manager.state.concurrent_positions == 1

    restarted_engine = make_engine()

    restored_position = (
        restarted_engine.position_manager.get_position("BTCUSDT")
    )

    assert restored_position == original_position

    assert (
        restarted_engine.risk_manager.state.concurrent_positions
        == 1
    )


def test_broker_state_matches_restored_position_after_restart():
    engine = make_engine()

    opening = engine.open_position(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    assert opening.result.success

    original_position = (
        engine.position_manager.get_position("BTCUSDT")
    )

    assert original_position is not None

    restarted_engine = make_engine()

    restored_position = (
        restarted_engine.position_manager.get_position("BTCUSDT")
    )

    assert restored_position is not None

    assert (
        restarted_engine.broker.positions.get("BTCUSDT")
        == restored_position.quantity
    )

    assert restarted_engine.broker.cash == pytest.approx(
        engine.broker.cash
    )


def test_broker_equity_is_consistent_after_restart():
    engine = make_engine()

    opening = engine.open_position(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    assert opening.result.success

    market_price = 110.0

    expected_equity = engine.broker.equity(
        symbol="BTCUSDT",
        market_price=market_price,
    )

    restarted_engine = make_engine()

    restored_equity = restarted_engine.broker.equity(
        symbol="BTCUSDT",
        market_price=market_price,
    )

    assert restored_equity == expected_equity


def test_failed_close_restores_broker_database_state(monkeypatch):
    engine = make_engine()

    opening = engine.open_position(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )
    assert opening.result.success

    original_cash = engine.broker.cash
    original_positions = engine.broker.positions.copy()

    def fail_close(*args, **kwargs):
        raise RuntimeError("simulated close failure")

    monkeypatch.setattr(
        engine.position_manager,
        "close_position",
        fail_close,
    )

    with pytest.raises(RuntimeError, match="simulated close failure"):
        engine.close_position(
            symbol="BTCUSDT",
            exit_price=110.0,
            timestamp=TIMESTAMP,
        )

    restarted_engine = make_engine()

    assert restarted_engine.broker.cash == pytest.approx(
        original_cash
    )
    assert restarted_engine.broker.positions == original_positions


def test_failed_close_restores_position_database_state(monkeypatch):
    engine = make_engine()

    opening = engine.open_position(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )
    assert opening.result.success

    original_position = engine.position_manager.get_position(
        "BTCUSDT"
    )
    assert original_position is not None

    def fail_close(*args, **kwargs):
        raise RuntimeError("simulated close failure")

    monkeypatch.setattr(
        engine.position_manager,
        "close_position",
        fail_close,
    )

    with pytest.raises(RuntimeError, match="simulated close failure"):
        engine.close_position(
            symbol="BTCUSDT",
            exit_price=110.0,
            timestamp=TIMESTAMP,
        )

    restarted_engine = make_engine()

    restored_position = (
        restarted_engine.position_manager.get_position("BTCUSDT")
    )

    assert restored_position == original_position


def test_restart_preserves_broker_position_consistency():
    engine = make_engine()

    opening = engine.open_position(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )
    assert opening.result.success

    restarted_engine = make_engine()

    restored_position = (
        restarted_engine.position_manager.get_position("BTCUSDT")
    )

    assert restored_position is not None

    broker_quantity = (
        restarted_engine.broker.positions.get("BTCUSDT")
    )

    assert broker_quantity == pytest.approx(
        restored_position.quantity
    )

def test_risk_state_survives_restart():
    engine = make_engine()

    engine.risk_manager.update_equity(
        equity=10_500.0,
        timestamp=TIMESTAMP,
    )

    engine.risk_manager.record_realized_pnl(
        pnl=-250.0,
        timestamp=TIMESTAMP,
    )

    restarted_engine = make_engine()

    state = restarted_engine.risk_manager.state

    assert state.equity_peak == pytest.approx(10_500.0)
    assert state.day_start_equity == pytest.approx(10_000.0)
    assert state.realized_daily_pnl == pytest.approx(-250.0)
    assert state.current_day == TIMESTAMP.date().isoformat()
    assert state.halted is False
    assert state.halt_reason is None


def test_realized_daily_pnl_survives_restart():
    engine = make_engine()

    engine.risk_manager.record_realized_pnl(
        pnl=-100.0,
        timestamp=TIMESTAMP,
    )
    engine.risk_manager.record_realized_pnl(
        pnl=-150.0,
        timestamp=TIMESTAMP,
    )

    assert (
        engine.risk_manager.state.realized_daily_pnl
        == pytest.approx(-250.0)
    )

    restarted_engine = make_engine()

    assert (
        restarted_engine.risk_manager.state.realized_daily_pnl
        == pytest.approx(-250.0)
    )


def test_drawdown_halt_survives_restart():
    engine = make_engine()

    engine.risk_manager.update_equity(
        equity=9_000.0,
        timestamp=TIMESTAMP,
    )

    assert engine.risk_manager.state.halted is True
    assert (
        engine.risk_manager.state.halt_reason
        == "drawdown limit exceeded"
    )

    restarted_engine = make_engine()

    assert restarted_engine.risk_manager.state.halted is True
    assert (
        restarted_engine.risk_manager.state.halt_reason
        == "drawdown limit exceeded"
    )


def test_daily_loss_halt_survives_restart():
    engine = make_engine()

    engine.risk_manager.record_realized_pnl(
        pnl=-500.0,
        timestamp=TIMESTAMP,
    )

    assert engine.risk_manager.state.halted is True
    assert (
        engine.risk_manager.state.halt_reason
        == "daily loss limit exceeded"
    )

    restarted_engine = make_engine()

    assert restarted_engine.risk_manager.state.halted is True
    assert (
        restarted_engine.risk_manager.state.halt_reason
        == "daily loss limit exceeded"
    )


def test_risk_state_and_open_position_survive_restart_together():
    engine = make_engine()

    opening = engine.open_position(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    assert opening.result.success

    engine.risk_manager.update_equity(
        equity=10_500.0,
        timestamp=TIMESTAMP,
    )

    engine.risk_manager.record_realized_pnl(
        pnl=-250.0,
        timestamp=TIMESTAMP,
    )

    restarted_engine = make_engine()

    restored_position = (
        restarted_engine.position_manager.get_position("BTCUSDT")
    )
    assert restored_position is not None

    assert (
        restarted_engine.broker.positions.get("BTCUSDT")
        == pytest.approx(restored_position.quantity)
    )

    state = restarted_engine.risk_manager.state

    assert state.concurrent_positions == 1
    assert state.equity_peak == pytest.approx(10_500.0)
    assert state.day_start_equity == pytest.approx(10_000.0)
    assert state.realized_daily_pnl == pytest.approx(-250.0)
    assert state.halted is False

def test_closed_trade_survives_restart_in_journal():
    engine = make_engine()

    opening = engine.open_position(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    assert opening.result.success

    closing = engine.close_position(
        symbol="BTCUSDT",
        exit_price=110.0,
        timestamp=TIMESTAMP,
    )

    assert closing.result.success
    assert closing.pnl == pytest.approx(195.8)

    restarted_engine = make_engine()

    assert restarted_engine.position_manager.get_position(
        "BTCUSDT"
    ) is None

    assert (
        restarted_engine.broker.positions.get(
            "BTCUSDT",
            0.0,
        )
        == pytest.approx(0.0)
    )

def test_open_trade_and_close_leave_no_stale_position_after_restart():
    engine = make_engine()

    opening = engine.open_position(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    assert opening.result.success

    assert engine.position_manager.has_position("BTCUSDT")
    assert engine.broker.positions["BTCUSDT"] == pytest.approx(20.0)

    closing = engine.close_position(
        symbol="BTCUSDT",
        exit_price=110.0,
        timestamp=TIMESTAMP,
    )

    assert closing.result.success

    restarted_engine = make_engine()

    assert not restarted_engine.position_manager.has_position(
        "BTCUSDT"
    )
    assert (
        restarted_engine.broker.positions.get(
            "BTCUSDT",
            0.0,
        )
        == pytest.approx(0.0)
    )
    assert restarted_engine.risk_manager.state.concurrent_positions == 0

def test_closed_trade_realized_pnl_survives_restart():
    engine = make_engine()

    opening = engine.open_position(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    assert opening.result.success

    closing = engine.close_position(
        symbol="BTCUSDT",
        exit_price=110.0,
        timestamp=TIMESTAMP,
    )

    assert closing.result.success
    assert closing.pnl == pytest.approx(195.8)

    restarted_engine = make_engine()

    assert (
        restarted_engine.risk_manager.state.realized_daily_pnl
        == pytest.approx(195.8)
    )

def test_closed_trade_is_recorded_in_trade_journal():
    engine = make_engine()

    opening = engine.open_position(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    assert opening.result.success

    closing = engine.close_position(
        symbol="BTCUSDT",
        exit_price=110.0,
        timestamp=TIMESTAMP,
    )

    assert closing.result.success
    assert closing.pnl == pytest.approx(195.8)

    journal = TradeJournal()
    entries = journal.load_all()

    assert len(entries) == 1

    entry = entries[0]

    assert entry.symbol == "BTCUSDT"
    assert entry.side == "LONG"
    assert entry.entry_price == pytest.approx(100.0)
    assert entry.exit_price == pytest.approx(110.0)
    assert entry.quantity == pytest.approx(20.0)
    assert entry.pnl == pytest.approx(195.8)


def test_trade_journal_survives_restart():
    engine = make_engine()

    opening = engine.open_position(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    assert opening.result.success

    closing = engine.close_position(
        symbol="BTCUSDT",
        exit_price=110.0,
        timestamp=TIMESTAMP,
    )

    assert closing.result.success
    assert closing.pnl == pytest.approx(195.8)

    restarted_engine = make_engine()

    journal = TradeJournal()
    entries = journal.load_all()

    assert len(entries) == 1
    assert entries[0].symbol == "BTCUSDT"
    assert entries[0].pnl == pytest.approx(195.8)

    assert (
        restarted_engine.risk_manager.state.realized_daily_pnl
        == pytest.approx(195.8)
    )


def test_journal_matches_broker_after_closed_trade():
    initialize_trade_journal()

    engine = make_engine()

    opening = engine.open_position(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    assert opening.result.success

    closing = engine.close_position(
        symbol="BTCUSDT",
        exit_price=110.0,
        timestamp=TIMESTAMP,
    )

    assert closing.result.success
    assert closing.pnl == pytest.approx(195.8)

    from app.database.connection import get_connection

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
                status
            FROM trade_journal
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
    finally:
        connection.close()

    assert row is not None

    assert row["symbol"] == "BTCUSDT"
    assert row["side"] == "LONG"
    assert row["quantity"] == pytest.approx(20.0)
    assert row["entry_price"] == pytest.approx(100.0)
    assert row["exit_price"] == pytest.approx(110.0)
    assert row["pnl"] == pytest.approx(closing.pnl)
    assert row["status"] == "CLOSED"

    assert (
        engine.broker.positions.get(
            "BTCUSDT",
            0.0,
        )
        == pytest.approx(0.0)
    )

    assert (
        engine.position_manager.has_position("BTCUSDT")
        is False
    )
def test_closed_trade_is_recorded_in_trade_journal():
    initialize_trade_journal()

    engine = make_engine()

    opening = engine.open_position(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    assert opening.result.success

    closing = engine.close_position(
        symbol="BTCUSDT",
        exit_price=110.0,
        timestamp=TIMESTAMP,
    )

    assert closing.result.success
    assert closing.pnl == pytest.approx(195.8)

    # Journal ID is not currently returned by close_position(),
    # so inspect the SQLite table directly for the closed trade.
    from app.database.connection import get_connection

    connection = get_connection()

    try:
        row = connection.execute(
            """
            SELECT
                id,
                symbol,
                side,
                quantity,
                entry_price,
                exit_price,
                entry_time,
                exit_time,
                pnl,
                fee,
                status,
                exit_reason
            FROM trade_journal
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
    finally:
        connection.close()

    assert row is not None
    assert row["symbol"] == "BTCUSDT"
    assert row["side"] == "LONG"
    assert row["quantity"] == pytest.approx(20.0)
    assert row["entry_price"] == pytest.approx(100.0)
    assert row["exit_price"] == pytest.approx(110.0)
    assert row["pnl"] == pytest.approx(195.8)
    assert row["status"] == "CLOSED"


def test_trade_journal_survives_restart():
    initialize_trade_journal()

    engine = make_engine()

    opening = engine.open_position(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    assert opening.result.success

    closing = engine.close_position(
        symbol="BTCUSDT",
        exit_price=110.0,
        timestamp=TIMESTAMP,
    )

    assert closing.result.success

    restarted_engine = make_engine()

    assert (
        restarted_engine.position_manager.get_position(
            "BTCUSDT"
        )
        is None
    )

    from app.database.connection import get_connection

    connection = get_connection()

    try:
        row = connection.execute(
            """
            SELECT
                id,
                symbol,
                side,
                quantity,
                pnl,
                status
            FROM trade_journal
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
    finally:
        connection.close()

    assert row is not None
    assert row["symbol"] == "BTCUSDT"
    assert row["side"] == "LONG"
    assert row["quantity"] == pytest.approx(20.0)
    assert row["pnl"] == pytest.approx(195.8)
    assert row["status"] == "CLOSED"


def test_trade_journal_matches_risk_realized_pnl():
    initialize_trade_journal()

    engine = make_engine()

    opening = engine.open_position(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    assert opening.result.success

    closing = engine.close_position(
        symbol="BTCUSDT",
        exit_price=110.0,
        timestamp=TIMESTAMP,
    )

    assert closing.result.success
    assert closing.pnl == pytest.approx(195.8)

    assert (
        engine.risk_manager.state.realized_daily_pnl
        == pytest.approx(195.8)
    )

    from app.database.connection import get_connection

    connection = get_connection()

    try:
        row = connection.execute(
            """
            SELECT pnl
            FROM trade_journal
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
    finally:
        connection.close()

    assert row is not None
    assert row["pnl"] == pytest.approx(
        engine.risk_manager.state.realized_daily_pnl
    )

def test_failed_journal_write_rolls_back_entire_close(monkeypatch):
    engine = make_engine()

    opening = engine.open_position(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    assert opening.result.success

    original_position = (
        engine.position_manager.get_position("BTCUSDT")
    )
    assert original_position is not None

    original_cash = engine.broker.cash
    original_positions = engine.broker.positions.copy()
    original_risk_state = deepcopy(
        engine.risk_manager.state
    )

    def fail_journal_write(*args, **kwargs):
        raise RuntimeError("simulated journal failure")

    monkeypatch.setattr(
        execution_module,
        "save_trade",
        fail_journal_write,
    )

    with pytest.raises(
        RuntimeError,
        match="simulated journal failure",
    ):
        engine.close_position(
            symbol="BTCUSDT",
            exit_price=110.0,
            timestamp=TIMESTAMP,
        )

    # In-memory state must be restored.
    restored_position = (
        engine.position_manager.get_position("BTCUSDT")
    )

    assert restored_position == original_position
    assert engine.broker.cash == pytest.approx(
        original_cash
    )
    assert engine.broker.positions == original_positions
    assert engine.risk_manager.state == original_risk_state

    # Restart must see the same original state.
    restarted_engine = make_engine()

    restarted_position = (
        restarted_engine.position_manager.get_position(
            "BTCUSDT"
        )
    )

    assert restarted_position == original_position
    assert restarted_engine.broker.cash == pytest.approx(
        original_cash
    )
    assert restarted_engine.broker.positions == original_positions
    assert restarted_engine.risk_manager.state == original_risk_state

    # No closed trade should have been journaled.
    from app.database.connection import get_connection

    connection = get_connection()

    try:
        row = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM trade_journal
            """
        ).fetchone()
    finally:
        connection.close()

    assert row["count"] == 0

def test_partial_close_preserves_remaining_position_and_journals_closed_quantity():
    initialize_trade_journal()

    engine = make_engine()

    opening = engine.open_position(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    assert opening.result.success

    original_position = (
        engine.position_manager.get_position("BTCUSDT")
    )
    assert original_position is not None
    assert original_position.quantity == pytest.approx(20.0)

    # Close half of the position.
    result = engine.broker.execute_market_order(
        symbol="BTCUSDT",
        side="SELL",
        quantity=10.0,
        price=110.0,
        timestamp=TIMESTAMP,
    )

    assert result.success

    closed_position, pnl = (
        engine.position_manager.close_position(
            result=result,
        )
    )

    assert closed_position.quantity == pytest.approx(10.0)

    # Remaining broker position.
    assert engine.broker.positions["BTCUSDT"] == pytest.approx(10.0)

    # PnL for only the closed half:
    # gross = 100
    # entry fee allocated = 1
    # exit fee = 1.1
    # pnl = 97.9
    assert pnl == pytest.approx(97.9)

    remaining = (
        engine.position_manager.get_position("BTCUSDT")
    )

    assert remaining is not None
    assert remaining.quantity == pytest.approx(10.0)
    assert remaining.entry_price == pytest.approx(100.0)
    assert remaining.entry_fee == pytest.approx(1.0)

def test_execution_engine_partial_close_keeps_remaining_position_consistent():
    initialize_trade_journal()

    engine = make_engine()

    opening = engine.open_position(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    assert opening.result.success

    # The current ExecutionEngine.close_position() closes the
    # entire persisted position, so this test intentionally exposes
    # whether partial-close support exists at the orchestration layer.
    result = engine.broker.execute_market_order(
        symbol="BTCUSDT",
        side="SELL",
        quantity=10.0,
        price=110.0,
        timestamp=TIMESTAMP,
    )

    assert result.success

    closed_position, pnl = engine.position_manager.close_position(
        result=result,
    )

    assert closed_position.quantity == pytest.approx(10.0)
    assert pnl == pytest.approx(97.9)

    assert engine.broker.positions["BTCUSDT"] == pytest.approx(10.0)

    remaining = engine.position_manager.get_position("BTCUSDT")

    assert remaining is not None
    assert remaining.quantity == pytest.approx(10.0)

    assert engine.risk_manager.state.concurrent_positions == 1

def test_execution_engine_partial_close_updates_all_state():
    initialize_trade_journal()

    engine = make_engine()

    opening = engine.open_position(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    assert opening.result.success

    closing = engine.close_position(
        symbol="BTCUSDT",
        quantity=10.0,
        exit_price=110.0,
        timestamp=TIMESTAMP,
    )

    assert closing.result.success
    assert closing.pnl == pytest.approx(97.9)

    remaining = engine.position_manager.get_position(
        "BTCUSDT"
    )

    assert remaining is not None
    assert remaining.quantity == pytest.approx(10.0)
    assert remaining.entry_fee == pytest.approx(1.0)

    assert engine.broker.positions["BTCUSDT"] == pytest.approx(
        10.0
    )

    assert engine.risk_manager.state.concurrent_positions == 1
    assert engine.risk_manager.state.realized_daily_pnl == pytest.approx(
        97.9
    )

    from app.database.connection import get_connection

    connection = get_connection()

    try:
        row = connection.execute(
            """
            SELECT
                quantity,
                entry_price,
                exit_price,
                pnl,
                fee,
                status
            FROM trade_journal
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
    finally:
        connection.close()

    assert row is not None
    assert row["quantity"] == pytest.approx(10.0)
    assert row["entry_price"] == pytest.approx(100.0)
    assert row["exit_price"] == pytest.approx(110.0)
    assert row["pnl"] == pytest.approx(97.9)
    assert row["fee"] == pytest.approx(2.1)
    assert row["status"] == "CLOSED"