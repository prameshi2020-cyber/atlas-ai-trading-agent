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


def test_invalid_quantity_is_rejected():

    broker = PaperBroker()

    result = broker.execute(
        make_order(
            quantity=0.0,
        )
    )

    assert result.success is False
    assert result.reason == "quantity must be greater than zero"


def test_invalid_price_is_rejected():

    broker = PaperBroker()

    result = broker.execute(
        make_order(
            price=0.0,
        )
    )

    assert result.success is False
    assert result.reason == "price must be greater than zero"