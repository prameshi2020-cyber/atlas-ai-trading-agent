from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List

from uuid import uuid4

from app.database.models import (
    initialize_pending_intents_database,
    save_intent,
    load_pending_intents as db_load_pending_intents,
    update_intent_status as db_update_intent_status,
    delete_intent as db_delete_intent,
)


AllowedStatus = frozenset(
    {"PENDING", "EXECUTING", "EXECUTED", "CANCELLED", "FAILED"}
)


@dataclass(slots=True)
class ExecutionIntent:
    intent_id: str
    symbol: str
    side: str
    decision_timestamp: datetime
    source_candle_timestamp: datetime
    status: str
    atr: float | None
    created_at: datetime
    executed_at: datetime | None = None
    trade_id: str | None = None
    reason: str | None = None


def initialize_intent_db() -> None:
    initialize_pending_intents_database()


def create_intent(
    *,
    symbol: str,
    side: str,
    decision_timestamp: datetime,
    source_candle_timestamp: datetime,
    atr: float | None,
    reason: str | None = None,
    intent_id: str | None = None,
) -> ExecutionIntent:
    # Ensure DB table exists in the current (possibly monkeypatched) environment.
    initialize_pending_intents_database()
    if intent_id is None:
        intent_id = uuid4().hex

    status = "PENDING"

    if status not in AllowedStatus:
        raise ValueError("invalid status")

    if side not in {"LONG", "SHORT"}:
        raise ValueError("side must be LONG or SHORT")

    created_at = datetime.now()

    # persist
    save_intent(
        intent_id=intent_id,
        symbol=symbol,
        side=side,
        decision_timestamp=decision_timestamp.isoformat(),
        source_candle_timestamp=source_candle_timestamp.isoformat(),
        status=status,
        atr=atr,
        created_at=created_at.isoformat(),
        executed_at=None,
        trade_id=None,
        reason=reason,
    )

    return ExecutionIntent(
        intent_id=intent_id,
        symbol=symbol,
        side=side,
        decision_timestamp=decision_timestamp,
        source_candle_timestamp=source_candle_timestamp,
        status=status,
        atr=atr,
        created_at=created_at,
        executed_at=None,
        trade_id=None,
        reason=reason,
    )


def load_pending_intents(status: str | None = None) -> List[ExecutionIntent]:
    rows = db_load_pending_intents(status=status)

    intents: List[ExecutionIntent] = []

    for row in rows:
        intents.append(
            ExecutionIntent(
                intent_id=row["intent_id"],
                symbol=row["symbol"],
                side=row["side"],
                decision_timestamp=datetime.fromisoformat(row["decision_timestamp"]),
                source_candle_timestamp=datetime.fromisoformat(row["source_candle_timestamp"]),
                status=row["status"],
                atr=row["atr"],
                created_at=datetime.fromisoformat(row["created_at"]),
                executed_at=(None if row["executed_at"] is None else datetime.fromisoformat(row["executed_at"])),
                trade_id=row["trade_id"],
                reason=row.get("reason"),
            )
        )

    return intents


def update_status(*, intent_id: str, status: str, executed_at: datetime | None = None, trade_id: str | None = None) -> None:
    if status not in AllowedStatus:
        raise ValueError("invalid status")

    executed_at_iso = executed_at.isoformat() if executed_at is not None else None

    db_update_intent_status(
        intent_id=intent_id,
        status=status,
        executed_at=executed_at_iso,
        trade_id=trade_id,
    )


def delete(intent_id: str) -> None:
    db_delete_intent(intent_id)
