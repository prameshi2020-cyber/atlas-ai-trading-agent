from pathlib import Path

import pandas as pd

from app.analysis.indicators import calculate_indicators
from app.analysis.strategy import StrategyEngine
from app.analysis.backtest import BacktestEngine


SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

INITIAL_CAPITAL = 10_000.0
FEE_PCT = 0.10
SLIPPAGE_PCT = 0.05
STOP_LOSS_ATR = 1.5
TAKE_PROFIT_ATR = 3.0


def load_dataset(symbol: str) -> pd.DataFrame:
    path = (
        Path(__file__).parent
        / "data"
        / "backtest"
        / f"{symbol}_1m.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Frozen dataset not found: {path}"
        )

    df = pd.read_csv(path)

    required_columns = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{symbol}: missing columns: {sorted(missing)}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    df = (
        df.sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    return df


def run_backtest(symbol: str) -> None:

    print("\n" + "=" * 70)
    print(f"{symbol} FROZEN BACKTEST")
    print("=" * 70)

    df = load_dataset(symbol)

    print(f"Dataset rows:      {len(df)}")
    print(f"Start:              {df['timestamp'].iloc[0]}")
    print(f"End:                {df['timestamp'].iloc[-1]}")

    features = calculate_indicators(df)

    print(f"Feature rows:       {len(features)}")

    strategy = StrategyEngine()

    backtest = BacktestEngine(
        strategy=strategy,
        initial_capital=INITIAL_CAPITAL,
        fee_pct=FEE_PCT,
        slippage_pct=SLIPPAGE_PCT,
        stop_loss_atr=STOP_LOSS_ATR,
        take_profit_atr=TAKE_PROFIT_ATR,
    )

    result = backtest.run(
        symbol=symbol,
        features=features,
    )

    print("\nRESULT")
    print("-" * 70)

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
        f"Largest win:       "
        f"{result.largest_win_pct:.4f}%"
    )

    print(
        f"Largest loss:      "
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

    return result


def main() -> None:

    print("=" * 70)
    print("ATLAS FROZEN BACKTEST BENCHMARK")
    print("=" * 70)

    print("\nConfiguration")
    print("-" * 70)
    print(f"Initial capital:    ${INITIAL_CAPITAL:,.2f}")
    print(f"Fee per side:       {FEE_PCT:.2f}%")
    print(f"Slippage per side:  {SLIPPAGE_PCT:.2f}%")
    print(f"Stop loss:          {STOP_LOSS_ATR:.1f} ATR")
    print(f"Take profit:        {TAKE_PROFIT_ATR:.1f} ATR")

    results = []

    for symbol in SYMBOLS:
        result = run_backtest(symbol)
        results.append(result)

    print("\n\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(
        f"{'SYMBOL':<10}"
        f"{'RETURN':>12}"
        f"{'TRADES':>10}"
        f"{'WIN RATE':>12}"
        f"{'MAX DD':>12}"
        f"{'PF':>10}"
    )

    print("-" * 70)

    for result in results:
        print(
            f"{result.symbol:<10}"
            f"{result.total_return_pct:>11.2f}%"
            f"{result.total_trades:>10}"
            f"{result.win_rate_pct:>11.2f}%"
            f"{result.max_drawdown_pct:>11.2f}%"
            f"{result.profit_factor:>10.3f}"
        )

    print("=" * 70)
    print("FROZEN BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()