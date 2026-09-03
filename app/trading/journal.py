from __future__ import annotations

from datetime import datetime

from app.database.connection import get_connection


def initialize_trade_journal() -> None:
    """
    Create the trade journal table if it does not already exist.
    """

    connection = get_connection()

    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT NOT NULL,
                pnl REAL NOT NULL,
                fee REAL NOT NULL,
                status TEXT NOT NULL,
                exit_reason TEXT
            )
            """
        )

        columns = connection.execute(
            "PRAGMA table_info(trade_journal)"
        ).fetchall()

        column_names = {
            column["name"]
            for column in columns
        }

        if "trade_id" not in column_names:
            connection.execute(
                """
                ALTER TABLE trade_journal
                ADD COLUMN trade_id TEXT
                """
            )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_trade_journal_trade_id
            ON trade_journal(trade_id)
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_trade_journal_symbol_exit_time
            ON trade_journal(symbol, exit_time)
            """
        )

        connection.commit()

    finally:
        connection.close()


def save_trade(
    *,
    symbol: str,
    side: str,
    quantity: float,
    entry_price: float,
    exit_price: float,
    entry_time: datetime,
    exit_time: datetime,
    pnl: float,
    fee: float,
    trade_id: str | None = None,
    status: str = "CLOSED",
    exit_reason: str | None = None,
) -> int:
    """
    Save a completed trade to the trade journal.

    Returns:
        Database ID of the inserted trade.
    """

    connection = get_connection()

    try:
        cursor = connection.execute(
            """
            INSERT INTO trade_journal (
                trade_id,
                symbol,
                side,
                quantity,
                entry_price,
                exit_price,
                entry_time,
                exit_time,
                pnl,
                fee,
                status,
                exit_reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade_id,
                symbol,
                side,
                quantity,
                entry_price,
                exit_price,
                entry_time.isoformat(),
                exit_time.isoformat(),
                pnl,
                fee,
                status,
                exit_reason,
            )
        )

        connection.commit()

        return int(cursor.lastrowid)

    finally:
        connection.close()


def get_trade(
    trade_id: int,
) -> dict | None:
    """
    Retrieve one trade by database ID.
    """

    connection = get_connection()

    try:
        row = connection.execute(
            """
            SELECT
                id,
                trade_id,
                symbol,
                side,
                quantity,
                entry_price,
                exit_price,
                entry_time,
                exit_time,
                pnl,
                fee,
                status,
                exit_reason
            FROM trade_journal
            WHERE id = ?
            """,
            (trade_id,),
        ).fetchone()

        if row is None:
            return None

        return dict(row)

    finally:
        connection.close()


def load_all_trades() -> list[dict]:
    """
    Return all persisted journal rows in insertion order.
    Uses the existing SQLite journal table rather than introducing a
    separate database access layer.
    """
    connection = get_connection()

    try:
        rows = connection.execute(
            """
            SELECT
                id,
                trade_id,
                symbol,
                side,
                quantity,
                entry_price,
                exit_price,
                entry_time,
                exit_time,
                pnl,
                fee,
                status,
                exit_reason
            FROM trade_journal
            ORDER BY id ASC
            """
        ).fetchall()

        return [dict(row) for row in rows]

    finally:
        connection.close()


def get_trades_by_trade_id(
    trade_id: str,
) -> list[dict]:
    connection = get_connection()

    try:
        rows = connection.execute(
            """
            SELECT
                id,
                trade_id,
                symbol,
                side,
                quantity,
                entry_price,
                exit_price,
                entry_time,
                exit_time,
                pnl,
                fee,
                status,
                exit_reason
            FROM trade_journal
            WHERE trade_id = ?
            ORDER BY id ASC
            """,
            (trade_id,),
        ).fetchall()

        return [dict(row) for row in rows]

    finally:
        connection.close()