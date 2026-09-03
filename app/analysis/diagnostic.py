from __future__ import annotations

from app.database.models import load_candles
from app.strategy.engine import StrategyEngine


SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def main() -> None:
    print("=" * 72)
    print("ATLAS — REAL MARKET STRATEGY DIAGNOSTIC")
    print("READ-ONLY | NO ORDERS | NO POSITION CHANGES")
    print("=" * 72)

    engine = StrategyEngine(
        fast_period=3,
        slow_period=5,
        atr_period=3,
        minimum_atr=0.0,
    )

    for symbol in SYMBOLS:
        print()
        print("-" * 72)
        print(symbol)
        print("-" * 72)

        candles = load_candles(
            symbol=symbol,
            timeframe="1m",
        )

        print(f"Candles       : {len(candles)}")

        if not candles:
            print("Status        : NO DATA")
            continue

        latest = candles[-1]

        signal = engine.generate_signal(
            symbol=symbol,
            candles=candles,
        )

        print(f"Latest time   : {latest.timestamp}")
        print(f"Latest close  : {latest.close:.8f}")
        print(f"Signal        : {signal.side}")
        print(f"ATR(3)        : {signal.atr:.8f}" if signal.atr is not None else "ATR(3)        : None")
        print(f"Reason        : {signal.reason}")

    print()
    print("=" * 72)
    print("Diagnostic completed.")
    print("=" * 72)


if __name__ == "__main__":
    main()
