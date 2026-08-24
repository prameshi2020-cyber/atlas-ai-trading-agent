import asyncio

import pandas as pd

from app.market.binance import BinanceProvider
from app.analysis.indicators import calculate_indicators
from app.analysis.strategy import StrategyEngine


async def main() -> None:
    print("=" * 70)
    print("ATLAS Strategy Engine Test")
    print("=" * 70)

    provider = BinanceProvider()

    symbol = "BTCUSDT"

    print(f"\nFetching {symbol} historical candles...")

    candles = await provider.get_historical_candles(
        symbol=symbol,
        timeframe="1m",
        limit=500,
    )

    print(f"Raw candles: {len(candles)}")

    # ---------------------------------------------------------
    # Convert candles to DataFrame
    # ---------------------------------------------------------

    df = pd.DataFrame(
        [
            {
                "timestamp": candle.timestamp,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
            }
            for candle in candles
        ]
    )

    # ---------------------------------------------------------
    # Calculate indicators
    # ---------------------------------------------------------

    features = calculate_indicators(df)

    print(f"Feature rows: {len(features)}")

    # ---------------------------------------------------------
    # Generate strategy signal
    # ---------------------------------------------------------

    strategy = StrategyEngine()

    signal = strategy.generate_signal(
        symbol=symbol,
        features=features,
    )

    # ---------------------------------------------------------
    # Display result
    # ---------------------------------------------------------

    print("\nSIGNAL")
    print("-" * 70)

    print(f"Symbol:     {signal.symbol}")
    print(f"Action:     {signal.action}")
    print(f"Score:      {signal.score}")
    print(f"Confidence: {signal.confidence}")

    print("\nReasons:")

    for reason in signal.reasons:
        print(f"  - {reason}")

    print("\nLatest market data")
    print("-" * 70)

    latest = features.iloc[-1]

    print(f"Close:          {latest['close']:.2f}")
    print(f"EMA9:           {latest['ema_9']:.2f}")
    print(f"EMA21:          {latest['ema_21']:.2f}")
    print(f"RSI14:          {latest['rsi_14']:.2f}")
    print(f"MACD histogram: {latest['macd_histogram']:.4f}")
    print(f"Momentum 10:    {latest['momentum_10']:.4f}%")
    print(f"Momentum 20:    {latest['momentum_20']:.4f}%")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(main())