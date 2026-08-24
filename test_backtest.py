import asyncio

import pandas as pd

from app.market.binance import BinanceProvider
from app.analysis.indicators import calculate_indicators
from app.analysis.strategy import StrategyEngine
from app.analysis.backtest import BacktestEngine


async def main() -> None:

    print("=" * 70)
    print("ATLAS Backtest Engine V2")
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

    features = calculate_indicators(df)

    print(f"Feature rows: {len(features)}")

    strategy = StrategyEngine()

    backtest = BacktestEngine(
        strategy=strategy,
        initial_capital=10_000.0,

        # 0.10% fee per side
        fee_pct=0.10,

        # 0.05% execution slippage
        slippage_pct=0.05,

        # Risk management
        stop_loss_atr=1.5,
        take_profit_atr=3.0,
    )

    result = backtest.run(
        symbol=symbol,
        features=features,
    )

    print("\nRESULT")
    print("-" * 70)

    print(
        f"Symbol:             {result.symbol}"
    )

    print(
        f"Initial capital:    "
        f"${result.initial_capital:,.2f}"
    )

    print(
        f"Final capital:      "
        f"${result.final_capital:,.2f}"
    )

    print(
        f"Total return:       "
        f"{result.total_return_pct:.2f}%"
    )

    print(
        f"Total trades:       "
        f"{result.total_trades}"
    )

    print(
        f"Winning trades:     "
        f"{result.winning_trades}"
    )

    print(
        f"Losing trades:      "
        f"{result.losing_trades}"
    )

    print(
        f"Win rate:           "
        f"{result.win_rate_pct:.2f}%"
    )

    print(
        f"Max drawdown:       "
        f"{result.max_drawdown_pct:.2f}%"
    )

    print(
        f"Profit factor:      "
        f"{result.profit_factor:.3f}"
    )

    print(
        f"Expectancy:         "
        f"{result.expectancy_pct:.4f}%"
    )

    print(
        f"Average trade:      "
        f"{result.average_trade_pct:.4f}%"
    )

    print(
        f"Average win:        "
        f"{result.average_win_pct:.4f}%"
    )

    print(
        f"Average loss:       "
        f"{result.average_loss_pct:.4f}%"
    )

    print(
        f"Largest win:        "
        f"{result.largest_win_pct:.4f}%"
    )

    print(
        f"Largest loss:       "
        f"{result.largest_loss_pct:.4f}%"
    )

    print("\nPOSITION STATISTICS")
    print("-" * 70)

    long_win_rate = (
        result.long_wins
        / result.long_trades
        * 100
        if result.long_trades
        else 0.0
    )

    short_win_rate = (
        result.short_wins
        / result.short_trades
        * 100
        if result.short_trades
        else 0.0
    )

    print(
        f"LONG trades:        "
        f"{result.long_trades}"
    )

    print(
        f"LONG win rate:      "
        f"{long_win_rate:.2f}%"
    )

    print(
        f"SHORT trades:       "
        f"{result.short_trades}"
    )

    print(
        f"SHORT win rate:     "
        f"{short_win_rate:.2f}%"
    )

    print("\nTRADES")
    print("-" * 70)

    for trade in result.trades[-15:]:

        print(
            f"{trade.side:5} | "
            f"Entry={trade.entry_price:.2f} | "
            f"Exit={trade.exit_price:.2f} | "
            f"PnL={trade.pnl_pct:.3f}% | "
            f"${trade.pnl_amount:.2f} | "
            f"{trade.exit_reason}"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(main())