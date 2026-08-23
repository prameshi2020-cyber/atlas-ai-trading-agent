from app.database.connection import get_connection
from app.market.models import Ticker


def initialize_database() -> None:
    connection = get_connection()

    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS market_ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                price_change_24h REAL,
                high_24h REAL,
                low_24h REAL,
                volume_24h REAL,
                timestamp TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_market_ticks_symbol_timestamp
            ON market_ticks(symbol, timestamp)
            """
        )

        connection.commit()

    finally:
        connection.close()


def save_ticker(ticker: Ticker) -> None:
    connection = get_connection()

    try:
        connection.execute(
            """
            INSERT INTO market_ticks (
                symbol,
                price,
                price_change_24h,
                high_24h,
                low_24h,
                volume_24h,
                timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker.symbol,
                ticker.price,
                ticker.price_change_24h,
                ticker.high_24h,
                ticker.low_24h,
                ticker.volume_24h,
                ticker.timestamp.isoformat(),
            ),
        )

        connection.commit()

    finally:
        connection.close()