from datetime import datetime

from app.database.connection import get_connection
from app.market.models import Ticker, Candle
from app.trading.models import Position


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

def initialize_position_database() -> None:
    connection = get_connection()

    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                entry_price REAL NOT NULL,
                entry_time TEXT NOT NULL,
                entry_fee REAL NOT NULL,
                stop_price REAL
            )
            """
        )

        columns = connection.execute(
            "PRAGMA table_info(positions)"
        ).fetchall()

        column_names = {column["name"] for column in columns}

        if "stop_price" not in column_names:
            connection.execute(
                """
                ALTER TABLE positions
                ADD COLUMN stop_price REAL
                """
                )

        connection.commit()

    finally:
        connection.close()


def save_position(position: Position) -> None:
    connection = get_connection()

    try:
        connection.execute(
            """
            INSERT INTO positions (
                symbol,
                side,
                quantity,
                entry_price,
                entry_time,
                entry_fee,
                stop_price
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                side = excluded.side,
                quantity = excluded.quantity,
                entry_price = excluded.entry_price,
                entry_time = excluded.entry_time,
                entry_fee = excluded.entry_fee,
                stop_price = excluded.stop_price
            """,
            (
                position.symbol,
                position.side,
                position.quantity,
                position.entry_price,
                position.entry_time.isoformat(),
                position.entry_fee,
                position.stop_price,
            ),
        )

        connection.commit()

    finally:
        connection.close()

def delete_position(symbol: str) -> None:
    connection = get_connection()

    try:
        connection.execute(
            """
            DELETE FROM positions
            WHERE symbol = ?
            """,
            (symbol,),
        )

        connection.commit()

    finally:
        connection.close()

def load_positions() -> list[Position]:
    connection = get_connection()

    try:
        rows = connection.execute(
            """
            SELECT
                symbol,
                side,
                quantity,
                entry_price,
                entry_time,
                entry_fee,
                stop_price
            FROM positions
            ORDER BY entry_time
            """
        ).fetchall()

        return [
            Position(
                symbol=row["symbol"],
                side=row["side"],
                quantity=row["quantity"],
                entry_price=row["entry_price"],
                entry_time=datetime.fromisoformat(
                    row["entry_time"]
                ),
                entry_fee=row["entry_fee"],
                stop_price=row["stop_price"],
            )
            for row in rows
        ]

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

def load_candles(
    symbol: str,
    timeframe: str = "1m",
) -> list[Candle]:
    connection = get_connection()

    try:
        rows = connection.execute(
            """
            SELECT
                symbol,
                timeframe,
                open,
                high,
                low,
                close,
                volume,
                timestamp
            FROM market_candles
            WHERE symbol = ?
              AND timeframe = ?
            ORDER BY timestamp ASC
            """,
            (
                symbol,
                timeframe,
            ),
        ).fetchall()

        return [
            Candle(
                symbol=row["symbol"],
                timeframe=row["timeframe"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                timestamp=datetime.fromisoformat(
                    row["timestamp"]
                ),
            )
            for row in rows
        ]

    finally:
        connection.close()

def initialize_paper_broker_database() -> None:
    connection = get_connection()

    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_account (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                cash REAL NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_positions (
                symbol TEXT PRIMARY KEY,
                quantity REAL NOT NULL
            )
            """
        )

        connection.commit()

    finally:
        connection.close()


def save_paper_broker_state(
    *,
    cash: float,
    positions: dict[str, float],
) -> None:
    connection = get_connection()

    try:
        connection.execute(
            """
            INSERT INTO paper_account (
                id,
                cash
            )
            VALUES (1, ?)
            ON CONFLICT(id) DO UPDATE SET
                cash = excluded.cash
            """,
            (cash,),
        )

        connection.execute(
            "DELETE FROM paper_positions"
        )

        for symbol, quantity in positions.items():
            connection.execute(
                """
                INSERT INTO paper_positions (
                    symbol,
                    quantity
                )
                VALUES (?, ?)
                """,
                (symbol, quantity),
            )

        connection.commit()

    finally:
        connection.close()


def load_paper_broker_state() -> tuple[float, dict[str, float]] | None:
    connection = get_connection()

    try:
        account = connection.execute(
            """
            SELECT cash
            FROM paper_account
            WHERE id = 1
            """
        ).fetchone()

        if account is None:
            return None

        rows = connection.execute(
            """
            SELECT symbol, quantity
            FROM paper_positions
            ORDER BY symbol
            """
        ).fetchall()

        positions = {
            row["symbol"]: row["quantity"]
            for row in rows
        }

        return account["cash"], positions

    finally:
        connection.close()

def initialize_risk_state_database() -> None:
    connection = get_connection()

    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS risk_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                equity_peak REAL NOT NULL,
                day_start_equity REAL NOT NULL,
                realized_daily_pnl REAL NOT NULL,
                concurrent_positions INTEGER NOT NULL,
                current_day TEXT,
                halted INTEGER NOT NULL,
                halt_reason TEXT
            )
            """
        )

        connection.commit()

    finally:
        connection.close()


def save_risk_state(
    *,
    equity_peak: float,
    day_start_equity: float,
    realized_daily_pnl: float,
    concurrent_positions: int,
    current_day: str | None,
    halted: bool,
    halt_reason: str | None,
) -> None:
    connection = get_connection()

    try:
        connection.execute(
            """
            INSERT INTO risk_state (
                id,
                equity_peak,
                day_start_equity,
                realized_daily_pnl,
                concurrent_positions,
                current_day,
                halted,
                halt_reason
            )
            VALUES (1, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                equity_peak = excluded.equity_peak,
                day_start_equity = excluded.day_start_equity,
                realized_daily_pnl = excluded.realized_daily_pnl,
                concurrent_positions = excluded.concurrent_positions,
                current_day = excluded.current_day,
                halted = excluded.halted,
                halt_reason = excluded.halt_reason
            """,
            (
                equity_peak,
                day_start_equity,
                realized_daily_pnl,
                concurrent_positions,
                current_day,
                int(halted),
                halt_reason,
            ),
        )

        connection.commit()

    finally:
        connection.close()


def load_risk_state() -> dict | None:
    connection = get_connection()

    try:
        row = connection.execute(
            """
            SELECT
                equity_peak,
                day_start_equity,
                realized_daily_pnl,
                concurrent_positions,
                current_day,
                halted,
                halt_reason
            FROM risk_state
            WHERE id = 1
            """
        ).fetchone()

        if row is None:
            return None

        return {
            "equity_peak": row["equity_peak"],
            "day_start_equity": row["day_start_equity"],
            "realized_daily_pnl": row["realized_daily_pnl"],
            "concurrent_positions": row["concurrent_positions"],
            "current_day": row["current_day"],
            "halted": bool(row["halted"]),
            "halt_reason": row["halt_reason"],
        }

    finally:
        connection.close()