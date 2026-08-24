import sqlite3

import pandas as pd

from app.analysis.indicators import calculate_indicators


DB_PATH = "data/atlas.db"


def main() -> None:
    connection = sqlite3.connect(DB_PATH)

    try:
        df = pd.read_sql_query(
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
            connection,
            params=("BTCUSDT", "1m"),
        )
    finally:
        connection.close()

    print("=" * 70)
    print("ATLAS Indicator Engine Test")
    print("=" * 70)

    print(f"Raw candles: {len(df)}")

    result = calculate_indicators(df)

    print(f"Feature rows: {len(result)}")

    columns = [
        "timestamp",
        "close",
        "ema_9",
        "ema_21",
        "sma_20",
        "sma_50",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_histogram",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "bb_position",
        "atr_14",
        "volume_ratio",
        "momentum_10",
        "momentum_20",
        "volatility_20",
        "ema_spread",
        "price_vs_ema21",
        "price_vs_sma50",
    ]

    print()
    print("Latest feature row:")
    print("-" * 70)

    print(
        result[columns]
        .tail(1)
        .to_string(index=False)
    )

    print()
    print("NaN counts:")
    print("-" * 70)

    indicator_columns = [
        c for c in columns
        if c not in {"timestamp"}
    ]

    print(
        result[indicator_columns]
        .tail(1)
        .isna()
        .sum()
    )

    print()
    print("SUCCESS")


if __name__ == "__main__":
    main()