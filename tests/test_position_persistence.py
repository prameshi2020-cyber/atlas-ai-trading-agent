from datetime import datetime, timezone

import pytest

from app.database.models import (
    delete_position,
    initialize_position_database,
    load_positions,
    save_position,
)
from app.trading.models import Position


TIMESTAMP = datetime(
    2026,
    8,
    23,
    12,
    0,
    tzinfo=timezone.utc,
)


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


def test_position_can_be_saved_and_loaded(test_database):
    position = Position(
        symbol="BTCUSDT",
        side="LONG",
        quantity=0.25,
        entry_price=100_000.0,
        entry_time=TIMESTAMP,
        entry_fee=2.5,
    )

    save_position(position)

    positions = load_positions()

    assert len(positions) == 1

    loaded = positions[0]

    assert loaded.symbol == "BTCUSDT"
    assert loaded.side == "LONG"
    assert loaded.quantity == pytest.approx(0.25)
    assert loaded.entry_price == pytest.approx(100_000.0)
    assert loaded.entry_time == TIMESTAMP
    assert loaded.entry_fee == pytest.approx(2.5)


def test_position_can_be_deleted(test_database):
    position = Position(
        symbol="ETHUSDT",
        side="LONG",
        quantity=1.5,
        entry_price=2_500.0,
        entry_time=TIMESTAMP,
        entry_fee=1.25,
    )

    save_position(position)

    assert len(load_positions()) == 1

    delete_position("ETHUSDT")

    assert load_positions() == []

def test_multiple_positions_can_be_saved_and_loaded(test_database):
    positions = [
        Position(
            symbol="BTCUSDT",
            side="LONG",
            quantity=0.25,
            entry_price=100_000.0,
            entry_time=TIMESTAMP,
            entry_fee=2.5,
        ),
        Position(
            symbol="ETHUSDT",
            side="SHORT",
            quantity=1.5,
            entry_price=2_500.0,
            entry_time=TIMESTAMP,
            entry_fee=1.25,
        ),
    ]

    for position in positions:
        save_position(position)

    loaded_positions = load_positions()

    assert len(loaded_positions) == 2

    loaded_by_symbol = {
        position.symbol: position
        for position in loaded_positions
    }

    assert loaded_by_symbol["BTCUSDT"].side == "LONG"
    assert loaded_by_symbol["BTCUSDT"].quantity == pytest.approx(0.25)

    assert loaded_by_symbol["ETHUSDT"].side == "SHORT"
    assert loaded_by_symbol["ETHUSDT"].quantity == pytest.approx(1.5)

def test_position_database_initialization_is_idempotent(
    test_database,
):
    initialize_position_database()
    initialize_position_database()

    position = Position(
        symbol="BTCUSDT",
        side="LONG",
        quantity=1.0,
        entry_price=100.0,
        entry_time=TIMESTAMP,
        entry_fee=0.10,
    )

    save_position(position)

    loaded = load_positions()

    assert len(loaded) == 1
    assert loaded[0].symbol == "BTCUSDT"