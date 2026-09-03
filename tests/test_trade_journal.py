from datetime import datetime, timezone

from app.database.connection import get_connection
from app.trading.journal import (
    get_trade,
    initialize_trade_journal,
    save_trade,
)


TIMESTAMP_ENTRY = datetime(
    2026,
    1,
    1,
    10,
    0,
    tzinfo=timezone.utc,
)

TIMESTAMP_EXIT = datetime(
    2026,
    1,
    1,
    11,
    0,
    tzinfo=timezone.utc,
)


def setup_function():
    """
    Ensure the trade journal table exists before every test.
    """

    initialize_trade_journal()

    connection = get_connection()

    try:
        connection.execute("DELETE FROM trade_journal")
        connection.commit()
    finally:
        connection.close()


def test_trade_journal_table_is_created():
    connection = get_connection()

    try:
        row = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            AND name = 'trade_journal'
            """
        ).fetchone()

        assert row is not None
        assert row["name"] == "trade_journal"

    finally:
        connection.close()


def test_save_trade_returns_trade_id():
    trade_id = save_trade(
        symbol="BTCUSDT",
        side="LONG",
        quantity=10.0,
        entry_price=100.0,
        exit_price=110.0,
        entry_time=TIMESTAMP_ENTRY,
        exit_time=TIMESTAMP_EXIT,
        pnl=99.0,
        fee=1.0,
    )

    assert trade_id > 0


def test_saved_trade_can_be_retrieved():
    trade_id = save_trade(
        symbol="BTCUSDT",
        side="LONG",
        quantity=10.0,
        entry_price=100.0,
        exit_price=110.0,
        entry_time=TIMESTAMP_ENTRY,
        exit_time=TIMESTAMP_EXIT,
        pnl=99.0,
        fee=1.0,
        exit_reason="take_profit",
    )

    trade = get_trade(trade_id)

    assert trade is not None
    assert trade["id"] == trade_id
    assert trade["symbol"] == "BTCUSDT"
    assert trade["side"] == "LONG"
    assert trade["quantity"] == 10.0
    assert trade["entry_price"] == 100.0
    assert trade["exit_price"] == 110.0
    assert trade["pnl"] == 99.0
    assert trade["fee"] == 1.0
    assert trade["status"] == "CLOSED"
    assert trade["exit_reason"] == "take_profit"


def test_trade_timestamps_are_stored_as_isoformat():
    trade_id = save_trade(
        symbol="ETHUSDT",
        side="SHORT",
        quantity=2.0,
        entry_price=2000.0,
        exit_price=1900.0,
        entry_time=TIMESTAMP_ENTRY,
        exit_time=TIMESTAMP_EXIT,
        pnl=199.0,
        fee=1.0,
    )

    trade = get_trade(trade_id)

    assert trade is not None
    assert trade["entry_time"] == TIMESTAMP_ENTRY.isoformat()
    assert trade["exit_time"] == TIMESTAMP_EXIT.isoformat()


def test_missing_trade_returns_none():
    trade = get_trade(999999)

    assert trade is None