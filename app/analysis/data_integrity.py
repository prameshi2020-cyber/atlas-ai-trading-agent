from __future__ import annotations

from collections import Counter
from datetime import timedelta

from app.database.models import load_candles


SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def analyze_symbol(symbol: str) -> None:
    candles = load_candles(
        symbol=symbol,
        timeframe="1m",
    )

    print()
    print("=" * 72)
    print(symbol)
    print("=" * 72)

    if not candles:
        print("STATUS        : NO DATA")
        return

    timestamps = [c.timestamp for c in candles]
    prices = [c.close for c in candles]

    print(f"Candles       : {len(candles)}")
    print(f"First time    : {timestamps[0]}")
    print(f"Latest time   : {timestamps[-1]}")
    print(f"Min close     : {min(prices):.8f}")
    print(f"Max close     : {max(prices):.8f}")
    print(f"Latest close  : {prices[-1]:.8f}")

    # ------------------------------------------------------------
    # Duplicate timestamps
    # ------------------------------------------------------------
    counts = Counter(timestamps)
    duplicates = {
        timestamp: count
        for timestamp, count in counts.items()
        if count > 1
    }

    print(f"Duplicate TS  : {len(duplicates)}")

    # ------------------------------------------------------------
    # Timestamp gaps
    # ------------------------------------------------------------
    gaps = []

    for previous, current in zip(timestamps, timestamps[1:]):
        delta = current - previous

        if delta != timedelta(minutes=1):
            gaps.append((previous, current, delta))

    print(f"Timestamp gaps: {len(gaps)}")

    if gaps:
        print("  First gaps:")
        for previous, current, delta in gaps[:5]:
            print(
                f"    {previous} -> {current} "
                f"({delta})"
            )

    # ------------------------------------------------------------
    # OHLC consistency
    # ------------------------------------------------------------
    invalid_ohlc = []

    for candle in candles:
        if candle.high < max(candle.open, candle.close):
            invalid_ohlc.append(candle)

        elif candle.low > min(candle.open, candle.close):
            invalid_ohlc.append(candle)

        elif candle.high < candle.low:
            invalid_ohlc.append(candle)

    print(f"Invalid OHLC  : {len(invalid_ohlc)}")

    # ------------------------------------------------------------
    # Zero-volume candles
    # ------------------------------------------------------------
    zero_volume = [
        candle
        for candle in candles
        if candle.volume == 0
    ]

    print(f"Zero volume   : {len(zero_volume)}")

    # ------------------------------------------------------------
    # Suspicious price jumps
    # ------------------------------------------------------------
    suspicious_jumps = []

    for previous, current in zip(candles, candles[1:]):
        if previous.close <= 0:
            continue

        change_pct = (
            abs(current.close - previous.close)
            / previous.close
            * 100
        )

        if change_pct >= 5:
            suspicious_jumps.append(
                (
                    previous.timestamp,
                    previous.close,
                    current.timestamp,
                    current.close,
                    change_pct,
                )
            )

    print(f">=5% jumps    : {len(suspicious_jumps)}")

    if suspicious_jumps:
        print("  First jumps:")
        for item in suspicious_jumps[:5]:
            (
                previous_time,
                previous_close,
                current_time,
                current_close,
                change_pct,
            ) = item

            print(
                f"    {previous_time} "
                f"{previous_close:.8f} -> "
                f"{current_time} "
                f"{current_close:.8f} "
                f"({change_pct:.2f}%)"
            )

    # ------------------------------------------------------------
    # Last 20 candles
    # ------------------------------------------------------------
    print()
    print("Last 20 candles:")
    print(
        "  Timestamp                       "
        "Open          High          Low           Close         Volume"
    )

    for candle in candles[-20:]:
        print(
            f"  {candle.timestamp} "
            f"{candle.open:12.8f} "
            f"{candle.high:12.8f} "
            f"{candle.low:12.8f} "
            f"{candle.close:12.8f} "
            f"{candle.volume:12.4f}"
        )


def main() -> None:
    print("=" * 72)
    print("ATLAS — MARKET DATA INTEGRITY DIAGNOSTIC")
    print("READ-ONLY | NO DELETE | NO UPDATE | NO TRADES")
    print("=" * 72)

    for symbol in SYMBOLS:
        analyze_symbol(symbol)

    print()
    print("=" * 72)
    print("Integrity diagnostic completed.")
    print("=" * 72)


if __name__ == "__main__":
    main()
