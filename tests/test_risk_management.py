from datetime import datetime, timezone, timedelta

import pytest

from app.risk.manager import RiskManager
from app.risk.models import RiskConfig
from app.risk.sizing import calculate_position_size


def test_one_percent_risk_sizing():
    quantity, risk_amount, notional = calculate_position_size(
        equity=10_000.0,
        entry_price=100.0,
        stop_price=99.0,
        risk_per_trade=0.01,
    )

    assert risk_amount == pytest.approx(100.0)
    assert quantity == pytest.approx(100.0)
    assert notional == pytest.approx(10_000.0)


def test_position_notional_cap_is_enforced():
    quantity, risk_amount, notional = calculate_position_size(
        equity=10_000.0,
        entry_price=100.0,
        stop_price=99.9,
        risk_per_trade=0.01,
        maximum_position_pct=0.5,
    )

    assert risk_amount == pytest.approx(100.0)
    assert notional <= 5_000.0 + 1e-9


def test_invalid_stop_distance_is_rejected():
    with pytest.raises(ValueError, match="stop distance"):
        calculate_position_size(
            equity=10_000.0,
            entry_price=100.0,
            stop_price=100.0,
            risk_per_trade=0.01,
        )


def test_invalid_equity_is_rejected():
    with pytest.raises(ValueError, match="equity"):
        calculate_position_size(
            equity=0.0,
            entry_price=100.0,
            stop_price=99.0,
            risk_per_trade=0.01,
        )


def test_risk_decision_approves_valid_trade():
    manager = RiskManager(initial_equity=10_000.0)

    decision = manager.evaluate_trade(
        equity=10_000.0,
        entry_price=100.0,
        stop_price=99.0,
        timestamp=datetime.now(timezone.utc),
    )

    assert decision.allowed is True
    assert decision.quantity > 0
    assert decision.risk_amount == pytest.approx(100.0)
    assert decision.reason == "approved"


def test_maximum_concurrent_positions_is_enforced():
    manager = RiskManager(
        initial_equity=10_000.0,
        config=RiskConfig(
            max_concurrent_positions=1,
        ),
    )

    assert manager.register_position_open() is True

    decision = manager.evaluate_trade(
        equity=10_000.0,
        entry_price=100.0,
        stop_price=99.0,
        timestamp=datetime.now(timezone.utc),
    )

    assert decision.allowed is False
    assert decision.quantity == 0.0
    assert decision.reason == "maximum concurrent positions reached"


def test_drawdown_circuit_breaker_activates():
    manager = RiskManager(
        initial_equity=10_000.0,
        config=RiskConfig(
            max_drawdown_pct=0.10,
        ),
    )

    timestamp = datetime.now(timezone.utc)

    manager.update_equity(
        equity=9_000.0,
        timestamp=timestamp,
    )

    assert manager.state.halted is True
    assert manager.state.halt_reason == "drawdown limit exceeded"


def test_drawdown_halt_remains_active():
    manager = RiskManager(
        initial_equity=10_000.0,
        config=RiskConfig(
            max_drawdown_pct=0.10,
        ),
    )

    timestamp = datetime.now(timezone.utc)

    manager.update_equity(
        equity=9_000.0,
        timestamp=timestamp,
    )

    manager.update_equity(
        equity=9_500.0,
        timestamp=timestamp,
    )

    assert manager.state.halted is True


def test_daily_loss_limit_activates():
    manager = RiskManager(
        initial_equity=10_000.0,
        config=RiskConfig(
            daily_loss_limit_pct=0.03,
        ),
    )

    timestamp = datetime.now(timezone.utc)

    manager.record_realized_pnl(
        pnl=-300.0,
        timestamp=timestamp,
    )

    assert manager.state.halted is True
    assert manager.state.halt_reason == "daily loss limit exceeded"


def test_new_day_resets_realized_daily_pnl():
    manager = RiskManager(
        initial_equity=10_000.0,
        config=RiskConfig(
            daily_loss_limit_pct=0.03,
        ),
    )

    day_one = datetime(
        2026,
        8,
        25,
        12,
        0,
        tzinfo=timezone.utc,
    )

    day_two = day_one + timedelta(days=1)

    manager.record_realized_pnl(
        pnl=-100.0,
        timestamp=day_one,
    )

    assert manager.state.realized_daily_pnl == pytest.approx(-100.0)

    manager.record_realized_pnl(
        pnl=50.0,
        timestamp=day_two,
    )

    assert manager.state.realized_daily_pnl == pytest.approx(50.0)


def test_halted_trade_returns_zero_quantity():
    manager = RiskManager(
        initial_equity=10_000.0,
        config=RiskConfig(
            max_drawdown_pct=0.10,
        ),
    )

    timestamp = datetime.now(timezone.utc)

    manager.update_equity(
        equity=9_000.0,
        timestamp=timestamp,
    )

    decision = manager.evaluate_trade(
        equity=9_000.0,
        entry_price=100.0,
        stop_price=99.0,
        timestamp=timestamp,
    )

    assert decision.allowed is False
    assert decision.quantity == 0.0


def test_new_day_resets_day_start_equity():
    manager = RiskManager(
        initial_equity=10_000.0,
        config=RiskConfig(
            daily_loss_limit_pct=0.03,
        ),
    )

    day_one = datetime(
        2026,
        8,
        25,
        12,
        0,
        tzinfo=timezone.utc,
    )

    day_two = day_one + timedelta(days=1)

    manager.update_equity(
        equity=9_500.0,
        timestamp=day_one,
    )

    manager.update_equity(
        equity=9_400.0,
        timestamp=day_two,
    )

    assert manager.state.day_start_equity == pytest.approx(9_400.0)
    assert manager.state.realized_daily_pnl == pytest.approx(0.0)


def test_invalid_maximum_position_pct_is_rejected():
    manager = RiskManager(
        initial_equity=10_000.0,
        config=RiskConfig(
            maximum_position_pct=0.0,
        ),
    )

    with pytest.raises(
        ValueError,
        match="maximum_position_pct",
    ):
        manager.evaluate_trade(
            equity=10_000.0,
            entry_price=100.0,
            stop_price=99.0,
            timestamp=datetime.now(timezone.utc),
        )

def test_restore_open_positions_updates_concurrent_position_count():
    manager = RiskManager(
        initial_equity=10_000.0,
        config=RiskConfig(
            max_concurrent_positions=3,
        ),
    )

    manager.restore_open_positions(2)

    assert manager.state.concurrent_positions == 2

def test_restore_open_positions_rejects_negative_count():
    manager = RiskManager(
        initial_equity=10_000.0,
        config=RiskConfig(
            max_concurrent_positions=3,
        ),
    )

    with pytest.raises(
        ValueError,
        match="count must be greater than or equal to zero",
    ):
        manager.restore_open_positions(-1)


def test_restore_open_positions_rejects_count_above_maximum():
    manager = RiskManager(
        initial_equity=10_000.0,
        config=RiskConfig(
            max_concurrent_positions=3,
        ),
    )

    with pytest.raises(
        ValueError,
        match="restored open positions exceed",
    ):
        manager.restore_open_positions(4)