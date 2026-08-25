from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "atlas.db"
OUTPUT_DIR = ROOT / "data" / "backtest"

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
TIMEFRAME = "1m"


def validate_dataframe(
    df: pd.DataFrame,
    symbol: str,
) -> None:
    required = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{symbol}: missing columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError(
            f"{symbol}: dataset is empty"
        )

    timestamps = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    if timestamps.isna().any():
        raise ValueError(
            f"{symbol}: invalid timestamp detected"
        )

    if not timestamps.is_monotonic_increasing:
        raise ValueError(
            f"{symbol}: timestamps are not chronological"
        )

    if timestamps.duplicated().any():
        duplicates = int(timestamps.duplicated().sum())

        raise ValueError(
            f"{symbol}: {duplicates} duplicate timestamp(s)"
        )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        if df[column].isna().any():
            raise ValueError(
                f"{symbol}: NaN detected in {column}"
            )

    if (df["volume"] < 0).any():
        raise ValueError(
            f"{symbol}: negative volume detected"
        )

    if (df["high"] < df[["open", "close"]].max(axis=1)).any():
        raise ValueError(
            f"{symbol}: invalid high price detected"
        )

    if (df["low"] > df[["open", "close"]].min(axis=1)).any():
        raise ValueError(
            f"{symbol}: invalid low price detected"
        )

    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(
            f"{symbol}: non-positive OHLC price detected"
        )

    expected_delta = pd.Timedelta(minutes=1)

    deltas = timestamps.diff().dropna()

    gaps = deltas[deltas != expected_delta]

    if not gaps.empty:
        print(
            f"WARNING: {symbol}: "
            f"{len(gaps)} timestamp gap(s) detected"
        )

        for index in gaps.index[:10]:
            previous = timestamps.iloc[index - 1]
            current = timestamps.iloc[index]

            print(
                f"  gap: {previous.isoformat()} "
                f"-> {current.isoformat()} "
                f"({current - previous})"
            )

        if len(gaps) > 10:
            print(
                f"  ... and {len(gaps) - 10} more"
            )


def export_symbol(
    connection: sqlite3.Connection,
    symbol: str,
) -> None:

    query = """
        SELECT
            timestamp,
            open,
            high,
            low,
            close,
            volume
        FROM market_candles
        WHERE symbol = ?
          AND timeframe = ?
        ORDER BY timestamp ASC
    """

    df = pd.read_sql_query(
        query,
        connection,
        params=(symbol, TIMEFRAME),
    )

    if df.empty:
        raise ValueError(
            f"{symbol}: no {TIMEFRAME} candles found in database"
        )

    validate_dataframe(
        df,
        symbol,
    )

    output_path = (
        OUTPUT_DIR
        / f"{symbol}_{TIMEFRAME}.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    timestamps = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    print()
    print("=" * 70)
    print(f"{symbol} DATASET")
    print("=" * 70)
    print(f"Rows:       {len(df)}")
    print(f"Start:      {timestamps.iloc[0].isoformat()}")
    print(f"End:        {timestamps.iloc[-1].isoformat()}")
    print(f"Output:     {output_path}")
    print("Validation: PASSED")


def main() -> None:

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("ATLAS Frozen Backtest Dataset Exporter")
    print("=" * 70)

    print(f"Database:   {DB_PATH}")
    print(f"Output:     {OUTPUT_DIR}")
    print(f"Timeframe:  {TIMEFRAME}")

    with sqlite3.connect(DB_PATH) as connection:

        for symbol in SYMBOLS:
            export_symbol(
                connection,
                symbol,
            )

    print()
    print("=" * 70)
    print("EXPORT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()