from datetime import datetime, timezone

import pytest

from app.database.models import (
    initialize_candle_database,
    initialize_database,
    initialize_position_database,
)
from app.risk.models import RiskConfig
from app.database.models import initialize_position_database, save_position
from app.trading.models import Position
from app.trading.bootstrap import create_paper_trading_engine


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


def test_bootstrap_restores_persisted_positions(test_database):
    save_position(
        Position(
            symbol="BTCUSDT",
            side="LONG",
            quantity=2.0,
            entry_price=100.0,
            entry_time=TIMESTAMP,
            entry_fee=0.20,
        )
    )

    engine = create_paper_trading_engine(
        initial_cash=10_000.0,
    )

    position = engine.position_manager.get_position("BTCUSDT")

    assert position is not None
    assert position.symbol == "BTCUSDT"
    assert position.side == "LONG"
    assert position.quantity == pytest.approx(2.0)
    assert position.entry_price == pytest.approx(100.0)
    assert position.entry_fee == pytest.approx(0.20)

def test_bootstrap_restores_risk_manager_position_count(
    test_database,
):
    save_position(
        Position(
            symbol="BTCUSDT",
            side="LONG",
            quantity=2.0,
            entry_price=100.0,
            entry_time=TIMESTAMP,
            entry_fee=0.20,
        )
    )

    save_position(
        Position(
            symbol="ETHUSDT",
            side="SHORT",
            quantity=1.5,
            entry_price=200.0,
            entry_time=TIMESTAMP,
            entry_fee=0.15,
        )
    )

    engine = create_paper_trading_engine(
        initial_cash=10_000.0,
        risk_config=RiskConfig(
            max_concurrent_positions=3,
        ),
    )

    assert engine.position_manager.position_count() == 2
    assert engine.risk_manager.state.concurrent_positions == 2

def test_bootstrap_rejects_too_many_persisted_positions(
    test_database,
):
    save_position(
        Position(
            symbol="BTCUSDT",
            side="LONG",
            quantity=2.0,
            entry_price=100.0,
            entry_time=TIMESTAMP,
            entry_fee=0.20,
        )
    )

    save_position(
        Position(
            symbol="ETHUSDT",
            side="SHORT",
            quantity=1.5,
            entry_price=200.0,
            entry_time=TIMESTAMP,
            entry_fee=0.15,
        )
    )

    save_position(
        Position(
            symbol="SOLUSDT",
            side="LONG",
            quantity=3.0,
            entry_price=50.0,
            entry_time=TIMESTAMP,
            entry_fee=0.15,
        )
    )

    with pytest.raises(
        ValueError,
        match="restored open positions exceed",
    ):
        create_paper_trading_engine(
            initial_cash=10_000.0,
            risk_config=RiskConfig(
                max_concurrent_positions=2,
            ),
        )

def test_bootstrap_creates_complete_paper_trading_stack(
    test_database,
):
    engine = create_paper_trading_engine(
        initial_cash=10_000.0,
        risk_config=RiskConfig(
            max_concurrent_positions=3,
        ),
    )

    assert engine.broker is not None
    assert engine.risk_manager is not None
    assert engine.position_manager is not None

def test_bootstrap_initializes_position_database(
    tmp_path,
    monkeypatch,
):
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

    engine = create_paper_trading_engine(
        initial_cash=10_000.0,
    )

    assert engine.position_manager.position_count() == 0