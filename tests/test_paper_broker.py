from datetime import datetime, timezone

import pytest

from app.trading.broker import PaperBroker
from app.trading.models import OrderRequest


def make_order(
    *,
    side: str = "BUY",
    quantity: float = 1.0,
    price: float = 100.0,
) -> OrderRequest:

    return OrderRequest(
        symbol="BTCUSDT",
        side=side,
        quantity=quantity,
        price=price,
        timestamp=datetime(
            2026,
            8,
            30,
            tzinfo=timezone.utc,
        ),
    )


def test_buy_order_executes_and_deducts_cash():

    broker = PaperBroker(
        initial_cash=1_000.0,
        fee_pct=0.10,
    )

    result = broker.execute(
        make_order(
            side="BUY",
            quantity=2.0,
            price=100.0,
        )
    )

    assert result.success is True
    assert result.execution_price == pytest.approx(100.0)
    assert result.fee == pytest.approx(0.20)
    assert broker.cash == pytest.approx(799.80)


def test_sell_order_adds_cash_after_fee():

    broker = PaperBroker(
        initial_cash=1_000.0,
        fee_pct=0.10,
    )

    result = broker.execute(
        make_order(
            side="SELL",
            quantity=2.0,
            price=100.0,
        )
    )

    assert result.success is True
    assert result.fee == pytest.approx(0.20)
    assert broker.cash == pytest.approx(1_199.80)


def test_buy_rejected_when_cash_is_insufficient():

    broker = PaperBroker(
        initial_cash=100.0,
        fee_pct=0.10,
    )

    result = broker.execute(
        make_order(
            side="BUY",
            quantity=2.0,
            price=100.0,
        )
    )

    assert result.success is False
    assert result.reason == "insufficient paper cash"
    assert broker.cash == pytest.approx(100.0)


def test_invalid_price_is_rejected():

    broker = PaperBroker()

    result = broker.execute(
        make_order(
            price=0.0,
        )
    )

    assert result.success is False
    assert result.reason == "price must be greater than zero"


def test_equity_includes_open_long_position_value():
    broker = PaperBroker(
        initial_cash=1_000.0,
        fee_pct=0.10,
    )

    result = broker.execute(
        make_order(
            side="BUY",
            quantity=2.0,
            price=100.0,
        )
    )

    assert result.success is True

    equity = broker.equity(
        symbol="BTCUSDT",
        market_price=100.0,
    )

    assert equity == pytest.approx(999.80)

def test_equity_reflects_unrealized_long_pnl():
    broker = PaperBroker(
        initial_cash=1_000.0,
        fee_pct=0.10,
    )

    result = broker.execute(
        make_order(
            side="BUY",
            quantity=2.0,
            price=100.0,
        )
    )

    assert result.success is True

    equity = broker.equity(
        symbol="BTCUSDT",
        market_price=110.0,
    )

    assert equity == pytest.approx(1_019.80)

def test_reverse_execution_restores_buy_state():
    broker = PaperBroker(
        initial_cash=1_000.0,
        fee_pct=0.10,
    )

    result = broker.execute_market_order(
        symbol="BTCUSDT",
        side="BUY",
        quantity=2.0,
        price=100.0,
        timestamp=datetime.now(timezone.utc),
    )

    assert result.success is True
    assert broker.cash == pytest.approx(799.80)
    assert broker.positions["BTCUSDT"] == pytest.approx(2.0)

    broker.reverse_execution(result)

    assert broker.cash == pytest.approx(1_000.0)
    assert broker.positions.get("BTCUSDT", 0.0) == pytest.approx(0.0)

def test_paper_broker_state_survives_restart():
    broker = PaperBroker(
        initial_cash=1_000.0,
        fee_pct=0.10,
    )

    result = broker.execute(
        make_order(
            side="BUY",
            quantity=2.0,
            price=100.0,
        )
    )

    assert result.success is True

    restarted_broker = PaperBroker(
        initial_cash=1_000.0,
        fee_pct=0.10,
    )

    loaded = restarted_broker.load_state()

    assert loaded is True
    assert restarted_broker.cash == pytest.approx(799.80)
    assert restarted_broker.positions["BTCUSDT"] == pytest.approx(2.0)


def test_paper_broker_without_persisted_state_uses_initial_state():
    broker = PaperBroker(
        initial_cash=1_000.0,
        fee_pct=0.10,
    )

    assert broker.cash == pytest.approx(1_000.0)
    assert broker.positions == {}

def test_paper_broker_state_survives_restart():
    broker = PaperBroker(
        initial_cash=1_000.0,
        fee_pct=0.10,
    )

    result = broker.execute(
        make_order(
            side="BUY",
            quantity=2.0,
            price=100.0,
        )
    )

    assert result.success is True

    restarted_broker = PaperBroker(
        initial_cash=1_000.0,
        fee_pct=0.10,
    )

    loaded = restarted_broker.load_state()

    assert loaded is True
    assert restarted_broker.cash == pytest.approx(799.80)
    assert restarted_broker.positions["BTCUSDT"] == pytest.approx(2.0)