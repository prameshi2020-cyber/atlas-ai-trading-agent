from app.database.connection import get_connection
from app.market.models import Ticker, Candle


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


def initialize_candle_database() -> None:
    connection = get_connection()

    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS market_candles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                timestamp TEXT NOT NULL,
                UNIQUE(symbol, timeframe, timestamp)
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_market_candles_symbol_timeframe_timestamp
            ON market_candles(symbol, timeframe, timestamp)
            """
        )

        connection.commit()

    finally:
        connection.close()


def save_candle(candle: Candle) -> None:
    connection = get_connection()

    try:
        connection.execute(
            """
            INSERT OR IGNORE INTO market_candles (
                symbol,
                timeframe,
                open,
                high,
                low,
                close,
                volume,
                timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                candle.symbol,
                candle.timeframe,
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
                candle.timestamp.isoformat(),
            ),
        )

        connection.commit()

    finally:
        connection.close()


def save_candles(candles: list[Candle]) -> int:
    """
    Save multiple candles.

    Returns the number of newly inserted candles.
    Existing candles are ignored because of the UNIQUE constraint.
    """
    if not candles:
        return 0

    connection = get_connection()

    try:
        inserted = 0

        for candle in candles:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO market_candles (
                    symbol,
                    timeframe,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candle.symbol,
                    candle.timeframe,
                    candle.open,
                    candle.high,
                    candle.low,
                    candle.close,
                    candle.volume,
                    candle.timestamp.isoformat(),
                ),
            )

            inserted += cursor.rowcount

        connection.commit()
        return inserted

    finally:
        connection.close()