from datetime import datetime, timezone
from uuid import uuid4

import pytest

from app.trading import intent_manager
from app.database.models import load_pending_intents as db_load_pending_intents


def setup_module():
    # ensure DB table exists
    intent_manager.initialize_intent_db()


def make_intent_payload(symbol: str = "BTCUSDT"):
    now = datetime.now(timezone.utc)
    return dict(
        symbol=symbol,
        side="LONG",
        decision_timestamp=now,
        source_candle_timestamp=now,
        atr=1.23,
        reason="test",
    )


def test_create_and_load_pending_intent():
    payload = make_intent_payload()

    intent = intent_manager.create_intent(**payload)

    loaded = intent_manager.load_pending_intents()

    assert any(i.intent_id == intent.intent_id for i in loaded)

    # cleanup
    intent_manager.delete(intent.intent_id)


def test_update_status_and_executed_fields():
    payload = make_intent_payload()

    intent = intent_manager.create_intent(**payload)

    # PENDING -> EXECUTING
    intent_manager.update_status(intent_id=intent.intent_id, status="EXECUTING")

    rows = db_load_pending_intents()
    row = next(r for r in rows if r["intent_id"] == intent.intent_id)
    assert row["status"] == "EXECUTING"

    # EXECUTING -> EXECUTED with executed_at and trade_id
    executed_at = datetime.now(timezone.utc)
    intent_manager.update_status(intent_id=intent.intent_id, status="EXECUTED", executed_at=executed_at, trade_id="trade123")

    rows = db_load_pending_intents()
    row = next(r for r in rows if r["intent_id"] == intent.intent_id)
    assert row["status"] == "EXECUTED"
    assert row["executed_at"] is not None
    assert row["trade_id"] == "trade123"

    # cleanup
    intent_manager.delete(intent.intent_id)


def test_delete_intent():
    payload = make_intent_payload()
    intent = intent_manager.create_intent(**payload)

    intent_manager.delete(intent.intent_id)

    rows = db_load_pending_intents()
    assert not any(r["intent_id"] == intent.intent_id for r in rows)


def test_duplicate_intent_id_rejected():
    payload = make_intent_payload()

    intent_id = uuid4().hex

    i1 = intent_manager.create_intent(intent_id=intent_id, **payload)

    with pytest.raises(Exception):
        # second insert with same id should fail
        intent_manager.create_intent(intent_id=intent_id, **payload)

    # cleanup
    intent_manager.delete(i1.intent_id)


def test_persistence_across_connections():
    payload = make_intent_payload(symbol="PERSIST")

    intent = intent_manager.create_intent(**payload)

    # load directly from DB layer to simulate fresh connection
    rows = db_load_pending_intents()
    assert any(r["intent_id"] == intent.intent_id for r in rows)

    # cleanup
    intent_manager.delete(intent.intent_id)


def test_nullable_executed_and_trade_id():
    payload = make_intent_payload()

    intent = intent_manager.create_intent(**payload)

    rows = db_load_pending_intents()
    row = next(r for r in rows if r["intent_id"] == intent.intent_id)

    assert row["executed_at"] is None
    assert row["trade_id"] is None

    # cleanup
    intent_manager.delete(intent.intent_id)


def test_no_execution_price_field_present():
    payload = make_intent_payload()
    intent = intent_manager.create_intent(**payload)

    rows = db_load_pending_intents()
    row = next(r for r in rows if r["intent_id"] == intent.intent_id)

    assert "execution_price" not in row

    # cleanup
    intent_manager.delete(intent.intent_id)


def test_invalid_status_rejected_on_update():
    payload = make_intent_payload()
    intent = intent_manager.create_intent(**payload)

    with pytest.raises(ValueError):
        intent_manager.update_status(intent_id=intent.intent_id, status="BOGUS")

    intent_manager.delete(intent.intent_id)


def test_multiple_pending_intents_same_symbol_allowed():
    payload = make_intent_payload(symbol="DUPSYM")

    i1 = intent_manager.create_intent(**payload)
    i2 = intent_manager.create_intent(**payload)

    rows = db_load_pending_intents()
    matches = [r for r in rows if r["symbol"] == "DUPSYM"]

    assert len(matches) >= 2

    intent_manager.delete(i1.intent_id)
    intent_manager.delete(i2.intent_id)
