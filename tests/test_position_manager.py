from datetime import datetime, timezone

import pytest

from app.database.models import (
    initialize_position_database,
    load_positions,
)
from app.trading.models import ExecutionResult
from app.trading.position_manager import PositionManager


TIMESTAMP = datetime.now(timezone.utc)

@pytest.fixture
def test_database(tmp_path, monkeypatch):
    database_path = tmp_path / "test_atlas.db"

    monkeypatch.setattr(
        "app.database.connection.DATABASE_PATH",
        database_path,
    )

    monkeypatch.setattr(
        "app.database.connection.DATA_DIR",
        tmp_path,
    )

    initialize_position_database()

    yield database_path


def execution(
    *,
    symbol="BTCUSDT",
    side="BUY",
    quantity=1.0,
    price=100.0,
    fee=0.0,
    success=True,
):
    return ExecutionResult(
        success=success,
        symbol=symbol,
        side=side,
        quantity=quantity,
        execution_price=price,
        fee=fee,
        timestamp=TIMESTAMP,
        reason="executed",
    )


def test_buy_execution_opens_long_position():
    manager = PositionManager()

    position = manager.open_position(
        result=execution(
            side="BUY",
            quantity=2.0,
            price=100.0,
        )
    )

    assert position.symbol == "BTCUSDT"
    assert position.side == "LONG"
    assert position.quantity == 2.0
    assert position.entry_price == 100.0
    assert manager.has_position("BTCUSDT")

def test_open_position_stores_stop_price():
    manager = PositionManager()

    position = manager.open_position(
        result=execution(
            side="BUY",
            quantity=2.0,
            price=120.0,
        ),
        stop_price=106.0,
    )

    assert position.stop_price == pytest.approx(106.0)

def test_sell_execution_opens_short_position():
    manager = PositionManager()

    position = manager.open_position(
        result=execution(
            side="SELL",
            quantity=2.0,
            price=100.0,
        )
    )

    assert position.side == "SHORT"


def test_long_position_closes_with_sell_and_calculates_pnl():
    manager = PositionManager()

    manager.open_position(
        result=execution(
            side="BUY",
            quantity=2.0,
            price=100.0,
        )
    )

    position, pnl = manager.close_position(
        result=execution(
            side="SELL",
            quantity=2.0,
            price=110.0,
        )
    )

    assert pnl == 20.0
    assert position.side == "LONG"
    assert not manager.has_position("BTCUSDT")


def test_short_position_closes_with_buy_and_calculates_pnl():
    manager = PositionManager()

    manager.open_position(
        result=execution(
            side="SELL",
            quantity=2.0,
            price=100.0,
        )
    )

    position, pnl = manager.close_position(
        result=execution(
            side="BUY",
            quantity=2.0,
            price=90.0,
        )
    )

    assert pnl == 20.0
    assert position.side == "SHORT"
    assert not manager.has_position("BTCUSDT")


def test_close_quantity_cannot_exceed_position():
    manager = PositionManager()

    manager.open_position(
        result=execution(
            quantity=1.0,
            price=100.0,
        )
    )

    with pytest.raises(ValueError):
        manager.close_position(
            result=execution(
                side="SELL",
                quantity=2.0,
                price=110.0,
            )
        )


def test_duplicate_position_is_rejected():
    manager = PositionManager()

    manager.open_position(
        result=execution()
    )

    with pytest.raises(ValueError):
        manager.open_position(
            result=execution()
        )


def test_partial_close_reduces_position_quantity():
    manager = PositionManager()

    manager.open_position(
        result=execution(
            quantity=2.0,
            price=100.0,
        )
    )

    manager.close_position(
        result=execution(
            side="SELL",
            quantity=0.5,
            price=110.0,
        )
    )

    position = manager.get_position("BTCUSDT")

    assert position is not None
    assert position.quantity == 1.5


def test_close_fee_is_included_in_pnl():
    manager = PositionManager()

    manager.open_position(
        result=execution(
            quantity=1.0,
            price=100.0,
        )
    )

    _, pnl = manager.close_position(
        result=execution(
            side="SELL",
            quantity=1.0,
            price=110.0,
            fee=1.5,
        )
    )

    assert pnl == 8.5

def test_entry_and_exit_fees_are_both_included_in_pnl():
    manager = PositionManager()

    manager.open_position(
        result=execution(
            side="BUY",
            quantity=1.0,
            price=100.0,
            fee=0.10,
        )
    )

    _, pnl = manager.close_position(
        result=execution(
            side="SELL",
            quantity=1.0,
            price=110.0,
            fee=0.11,
        )
    )

    assert pnl == pytest.approx(9.79)

def test_partial_close_allocates_entry_fee_proportionally():
    manager = PositionManager()

    manager.open_position(
        result=execution(
            side="BUY",
            quantity=2.0,
            price=100.0,
            fee=0.20,
        )
    )

    _, pnl = manager.close_position(
        result=execution(
            side="SELL",
            quantity=1.0,
            price=110.0,
            fee=0.11,
        )
    )

    assert pnl == pytest.approx(9.79)

    position = manager.get_position("BTCUSDT")

    assert position is not None
    assert position.quantity == pytest.approx(1.0)
    assert position.entry_fee == pytest.approx(0.10)

def test_rollback_open_position_removes_registered_position():
    manager = PositionManager()

    result = execution(
        side="BUY",
        quantity=2.0,
        price=100.0,
        fee=0.20,
    )

    manager.open_position(result=result)

    assert manager.has_position("BTCUSDT")

    manager.rollback_open_position(result)

    assert not manager.has_position("BTCUSDT")

def test_open_position_persists_position(test_database):
    manager = PositionManager()

    result = execution(
        symbol="BTCUSDT",
        side="BUY",
        quantity=2.0,
        price=100.0,
        fee=0.20,
    )

    manager.open_position(
        result=result,
        stop_price=106.0,
    )

    from app.database.models import load_positions

    positions = load_positions()

    assert len(positions) == 1

    saved = positions[0]

    assert saved.symbol == "BTCUSDT"
    assert saved.side == "LONG"
    assert saved.quantity == pytest.approx(2.0)
    assert saved.entry_price == pytest.approx(100.0)
    assert saved.entry_fee == pytest.approx(0.20)
    assert saved.stop_price == pytest.approx(106.0)

def test_open_position_rolls_back_memory_when_persistence_fails(
    monkeypatch,
):
    manager = PositionManager()

    def failing_save_position(position):
        raise RuntimeError("database write failed")

    monkeypatch.setattr(
        "app.trading.position_manager.save_position",
        failing_save_position,
    )

    result = execution(
        symbol="BTCUSDT",
        side="BUY",
        quantity=2.0,
        price=100.0,
        fee=0.20,
    )

    with pytest.raises(
        RuntimeError,
        match="database write failed",
    ):
        manager.open_position(result=result)

    assert not manager.has_position("BTCUSDT")
    assert manager.get_position("BTCUSDT") is None
    assert manager.position_count() == 0

def test_full_close_removes_position_from_database(test_database):
    manager = PositionManager()

    manager.open_position(
        result=execution(
            symbol="BTCUSDT",
            side="BUY",
            quantity=2.0,
            price=100.0,
            fee=0.20,
        )
    )

    assert len(load_positions()) == 1

    manager.close_position(
        result=execution(
            symbol="BTCUSDT",
            side="SELL",
            quantity=2.0,
            price=110.0,
            fee=0.22,
        )
    )

    assert not manager.has_position("BTCUSDT")
    assert load_positions() == []

def test_partial_close_updates_position_in_database(test_database):
    manager = PositionManager()

    manager.open_position(
        result=execution(
            symbol="BTCUSDT",
            side="BUY",
            quantity=2.0,
            price=100.0,
            fee=0.20,
        )
    )

    manager.close_position(
        result=execution(
            symbol="BTCUSDT",
            side="SELL",
            quantity=0.5,
            price=110.0,
            fee=0.05,
        )
    )

    positions = load_positions()

    assert len(positions) == 1

    saved = positions[0]

    assert saved.symbol == "BTCUSDT"
    assert saved.side == "LONG"
    assert saved.quantity == pytest.approx(1.5)
    assert saved.entry_price == pytest.approx(100.0)
    assert saved.entry_fee == pytest.approx(0.15)

def test_partial_close_rolls_back_memory_when_persistence_fails(
    test_database,
    monkeypatch,
):
    manager = PositionManager()

    manager.open_position(
        result=execution(
            symbol="BTCUSDT",
            side="BUY",
            quantity=2.0,
            price=100.0,
            fee=0.20,
        )
    )

    def failing_save_position(position):
        raise RuntimeError("database update failed")

    monkeypatch.setattr(
        "app.trading.position_manager.save_position",
        failing_save_position,
    )

    with pytest.raises(
        RuntimeError,
        match="database update failed",
    ):
        manager.close_position(
            result=execution(
                symbol="BTCUSDT",
                side="SELL",
                quantity=0.5,
                price=110.0,
                fee=0.05,
            )
        )

    position = manager.get_position("BTCUSDT")

    assert position is not None
    assert position.quantity == pytest.approx(2.0)
    assert position.entry_fee == pytest.approx(0.20)

    saved_positions = load_positions()

    assert len(saved_positions) == 1
    assert saved_positions[0].quantity == pytest.approx(2.0)
    assert saved_positions[0].entry_fee == pytest.approx(0.20)

def test_full_close_keeps_memory_position_when_persistence_fails(
    test_database,
    monkeypatch,
):
    manager = PositionManager()

    manager.open_position(
        result=execution(
            symbol="BTCUSDT",
            side="BUY",
            quantity=2.0,
            price=100.0,
            fee=0.20,
        )
    )

    def failing_delete_position(symbol):
        raise RuntimeError("database delete failed")

    monkeypatch.setattr(
        "app.trading.position_manager.delete_position",
        failing_delete_position,
    )

    with pytest.raises(
        RuntimeError,
        match="database delete failed",
    ):
        manager.close_position(
            result=execution(
                symbol="BTCUSDT",
                side="SELL",
                quantity=2.0,
                price=110.0,
                fee=0.22,
            )
        )

    position = manager.get_position("BTCUSDT")

    assert position is not None
    assert position.quantity == pytest.approx(2.0)
    assert position.entry_fee == pytest.approx(0.20)

    saved_positions = load_positions()

    assert len(saved_positions) == 1
    assert saved_positions[0].quantity == pytest.approx(2.0)
    assert saved_positions[0].entry_fee == pytest.approx(0.20)

def test_position_manager_can_restore_positions_from_database(
    test_database,
):
    manager = PositionManager()

    original_position = manager.open_position(
        result=execution(
            symbol="BTCUSDT",
            side="BUY",
            quantity=2.0,
            price=100.0,
            fee=0.20,
        )
    )

    assert original_position.symbol == "BTCUSDT"

    restored_manager = PositionManager()

    restored_manager.load_positions()

    restored_position = restored_manager.get_position(
        "BTCUSDT"
    )

    assert restored_position is not None
    assert restored_position.symbol == "BTCUSDT"
    assert restored_position.side == "LONG"
    assert restored_position.quantity == pytest.approx(2.0)
    assert restored_position.entry_price == pytest.approx(100.0)
    assert restored_position.entry_fee == pytest.approx(0.20)
    assert restored_position.entry_time == TIMESTAMP

def test_position_manager_restores_multiple_positions(
    test_database,
):
    manager = PositionManager()

    manager.open_position(
        result=execution(
            symbol="BTCUSDT",
            side="BUY",
            quantity=2.0,
            price=100.0,
            fee=0.20,
        )
    )

    manager.open_position(
        result=execution(
            symbol="ETHUSDT",
            side="SELL",
            quantity=1.5,
            price=200.0,
            fee=0.15,
        )
    )

    restored_manager = PositionManager()

    restored_manager.load_positions()

    btc = restored_manager.get_position("BTCUSDT")
    eth = restored_manager.get_position("ETHUSDT")

    assert btc is not None
    assert btc.symbol == "BTCUSDT"
    assert btc.side == "LONG"
    assert btc.quantity == pytest.approx(2.0)
    assert btc.entry_price == pytest.approx(100.0)
    assert btc.entry_fee == pytest.approx(0.20)

    assert eth is not None
    assert eth.symbol == "ETHUSDT"
    assert eth.side == "SHORT"
    assert eth.quantity == pytest.approx(1.5)
    assert eth.entry_price == pytest.approx(200.0)
    assert eth.entry_fee == pytest.approx(0.15)

    assert restored_manager.position_count() == 2

def test_load_positions_replaces_existing_memory_state(
    test_database,
):
    manager = PositionManager()

    manager.open_position(
        result=execution(
            symbol="BTCUSDT",
            side="BUY",
            quantity=2.0,
            price=100.0,
            fee=0.20,
        )
    )

    manager.open_position(
        result=execution(
            symbol="ETHUSDT",
            side="SELL",
            quantity=1.5,
            price=200.0,
            fee=0.15,
        )
    )

    assert manager.position_count() == 2

    # Simulate a persisted state that contains only BTC.
    from app.database.models import delete_position

    delete_position("ETHUSDT")

    manager.load_positions()

    assert manager.position_count() == 1

    btc = manager.get_position("BTCUSDT")
    eth = manager.get_position("ETHUSDT")

    assert btc is not None
    assert btc.symbol == "BTCUSDT"

    assert eth is None

def test_load_positions_rejects_invalid_persisted_quantity(
    test_database,
):
    from app.database.connection import get_connection

    connection = get_connection()

    try:
        connection.execute(
            """
            INSERT INTO positions (
                symbol,
                side,
                quantity,
                entry_price,
                entry_time,
                entry_fee
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "BTCUSDT",
                "LONG",
                -1.0,
                100.0,
                TIMESTAMP.isoformat(),
                0.20,
            ),
        )

        connection.commit()

    finally:
        connection.close()

    manager = PositionManager()

    with pytest.raises(
        ValueError,
        match="quantity",
    ):
        manager.load_positions()

    assert manager.position_count() == 0

def test_load_positions_rejects_invalid_persisted_entry_price(
    test_database,
):
    from app.database.connection import get_connection

    connection = get_connection()

    try:
        connection.execute(
            """
            INSERT INTO positions (
                symbol,
                side,
                quantity,
                entry_price,
                entry_time,
                entry_fee
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "BTCUSDT",
                "LONG",
                1.0,
                0.0,
                TIMESTAMP.isoformat(),
                0.20,
            ),
        )

        connection.commit()

    finally:
        connection.close()

    manager = PositionManager()

    with pytest.raises(
        ValueError,
        match="entry_price",
    ):
        manager.load_positions()

    assert manager.position_count() == 0

def test_load_positions_rejects_invalid_persisted_entry_fee(
    test_database,
):
    from app.database.connection import get_connection

    connection = get_connection()

    try:
        connection.execute(
            """
            INSERT INTO positions (
                symbol,
                side,
                quantity,
                entry_price,
                entry_time,
                entry_fee
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "BTCUSDT",
                "LONG",
                1.0,
                100.0,
                TIMESTAMP.isoformat(),
                -0.20,
            ),
        )

        connection.commit()

    finally:
        connection.close()

    manager = PositionManager()

    with pytest.raises(
        ValueError,
        match="entry_fee",
    ):
        manager.load_positions()

    assert manager.position_count() == 0

def test_load_positions_rejects_invalid_persisted_side(
    test_database,
):
    from app.database.connection import get_connection

    connection = get_connection()

    try:
        connection.execute(
            """
            INSERT INTO positions (
                symbol,
                side,
                quantity,
                entry_price,
                entry_time,
                entry_fee
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "BTCUSDT",
                "INVALID",
                1.0,
                100.0,
                TIMESTAMP.isoformat(),
                0.20,
            ),
        )

        connection.commit()

    finally:
        connection.close()

    manager = PositionManager()

    with pytest.raises(
        ValueError,
        match="side",
    ):
        manager.load_positions()

    assert manager.position_count() == 0

def test_load_positions_is_atomic_when_one_persisted_position_is_invalid(
    test_database,
):
    from app.database.connection import get_connection

    connection = get_connection()

    try:
        connection.executemany(
            """
            INSERT INTO positions (
                symbol,
                side,
                quantity,
                entry_price,
                entry_time,
                entry_fee
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "BTCUSDT",
                    "LONG",
                    1.0,
                    100.0,
                    TIMESTAMP.isoformat(),
                    0.20,
                ),
                (
                    "ETHUSDT",
                    "SHORT",
                    -1.0,
                    200.0,
                    TIMESTAMP.isoformat(),
                    0.15,
                ),
                (
                    "SOLUSDT",
                    "LONG",
                    3.0,
                    50.0,
                    TIMESTAMP.isoformat(),
                    0.10,
                ),
            ],
        )

        connection.commit()

    finally:
        connection.close()

    manager = PositionManager()

    with pytest.raises(
        ValueError,
        match="quantity",
    ):
        manager.load_positions()

    assert manager.position_count() == 0

    assert manager.get_position("BTCUSDT") is None
    assert manager.get_position("ETHUSDT") is None
    assert manager.get_position("SOLUSDT") is None

def test_position_database_migrates_stop_price_column(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "legacy_atlas.db"

    monkeypatch.setattr(
        "app.database.connection.DATABASE_PATH",
        database_path,
    )

    monkeypatch.setattr(
        "app.database.connection.DATA_DIR",
        tmp_path,
    )

    from app.database.connection import get_connection
    from app.database.models import initialize_position_database

    connection = get_connection()

    try:
        connection.execute(
            """
            CREATE TABLE positions (
                symbol TEXT PRIMARY KEY,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                entry_price REAL NOT NULL,
                entry_time TEXT NOT NULL,
                entry_fee REAL NOT NULL
            )
            """
        )
        connection.commit()
    finally:
        connection.close()

    initialize_position_database()

    connection = get_connection()

    try:
        columns = connection.execute(
            "PRAGMA table_info(positions)"
        ).fetchall()
    finally:
        connection.close()

    column_names = {
        column["name"]
        for column in columns
    }

    assert "stop_price" in column_names

def test_position_recovery_preserves_stop_price(test_database):
    manager = PositionManager()

    result = execution(
        symbol="BTCUSDT",
        side="BUY",
        quantity=2.0,
        price=100.0,
        fee=0.20,
    )

    manager.open_position(
        result=result,
        stop_price=106.0,
    )

    recovered_manager = PositionManager()
    recovered_manager.load_positions()

    recovered = recovered_manager.get_position("BTCUSDT")

    assert recovered is not None
    assert recovered.symbol == "BTCUSDT"
    assert recovered.side == "LONG"
    assert recovered.quantity == pytest.approx(2.0)
    assert recovered.entry_price == pytest.approx(100.0)
    assert recovered.entry_fee == pytest.approx(0.20)
    assert recovered.stop_price == pytest.approx(106.0)

def test_partial_close_preserves_stop_price(test_database):
    manager = PositionManager()

    result = execution(
        symbol="BTCUSDT",
        side="BUY",
        quantity=2.0,
        price=100.0,
        fee=0.20,
    )

    manager.open_position(
        result=result,
        stop_price=95.0,
    )

    close_result = execution(
        symbol="BTCUSDT",
        side="SELL",
        quantity=1.0,
        price=110.0,
        fee=0.11,
    )

    manager.close_position(
        result=close_result,
    )

    position = manager.get_position("BTCUSDT")

    assert position is not None
    assert position.quantity == pytest.approx(1.0)
    assert position.stop_price == pytest.approx(95.0)