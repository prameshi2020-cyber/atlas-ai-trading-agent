from datetime import datetime, timezone

import pytest

from app.risk.manager import RiskManager
from app.risk.models import RiskConfig
from app.trading.bootstrap import create_paper_trading_engine


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