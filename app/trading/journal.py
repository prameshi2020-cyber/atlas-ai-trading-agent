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
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
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
            ),
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