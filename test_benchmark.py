from pathlib import Path

import pandas as pd


SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

INITIAL_CAPITAL = 10_000.0

# Round-trip transaction cost:
# 0.10% entry fee + 0.10% exit fee
# 0.05% entry slippage + 0.05% exit slippage
ROUND_TRIP_COST_PCT = 0.30


def load_dataset(symbol: str) -> pd.DataFrame:

    path = (
        Path(__file__).parent
        / "data"
        / "backtest"
        / f"{symbol}_1m.csv"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    return (
        df.sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )


def calculate_buy_and_hold(
    df: pd.DataFrame,
) -> dict:

    entry_price = float(df.iloc[0]["open"])
    exit_price = float(df.iloc[-1]["close"])

    raw_return_pct = (
        (exit_price - entry_price)
        / entry_price
        * 100
    )

    # Apply one buy-side fee + one sell-side fee
    # and equivalent slippage.
    net_return_pct = raw_return_pct - ROUND_TRIP_COST_PCT

    final_capital = (
        INITIAL_CAPITAL
        * (1 + net_return_pct / 100)
    )

    return {
        "entry_price": entry_price,
        "exit_price": exit_price,
        "raw_return_pct": raw_return_pct,
        "net_return_pct": net_return_pct,
        "final_capital": final_capital,
    }


def calculate_market_stats(
    df: pd.DataFrame,
) -> dict:

    closes = df["close"]

    returns = closes.pct_change().dropna()

    volatility_pct = (
        returns.std()
        * 100
    )

    max_price = closes.cummax()

    drawdown = (
        (closes - max_price)
        / max_price
        * 100
    )

    max_drawdown_pct = drawdown.min()

    return {
        "volatility_pct": volatility_pct,
        "max_drawdown_pct": max_drawdown_pct,
    }


def main() -> None:

    print("=" * 80)
    print("ATLAS MARKET BENCHMARK")
    print("=" * 80)

    print("\nThis benchmark uses the exact same frozen datasets")
    print("used by the ATLAS strategy backtest.")

    print("\n")

    for symbol in SYMBOLS:

        df = load_dataset(symbol)

        buy_hold = calculate_buy_and_hold(df)
        market = calculate_market_stats(df)

        print("=" * 80)
        print(symbol)
        print("=" * 80)

        print(
            f"Start:                  "
            f"{df['timestamp'].iloc[0]}"
        )

        print(
            f"End:                    "
            f"{df['timestamp'].iloc[-1]}"
        )

        print(
            f"Starting price:         "
            f"${buy_hold['entry_price']:,.4f}"
        )

        print(
            f"Ending price:           "
            f"${buy_hold['exit_price']:,.4f}"
        )

        print("\nBUY & HOLD")
        print("-" * 80)

        print(
            f"Raw return:             "
            f"{buy_hold['raw_return_pct']:.4f}%"
        )

        print(
            f"Round-trip cost:        "
            f"{ROUND_TRIP_COST_PCT:.2f}%"
        )

        print(
            f"Net return:             "
            f"{buy_hold['net_return_pct']:.4f}%"
        )

        print(
            f"Final capital:          "
            f"${buy_hold['final_capital']:,.2f}"
        )

        print("\nMARKET STATISTICS")
        print("-" * 80)

        print(
            f"1-minute volatility:    "
            f"{market['volatility_pct']:.4f}%"
        )

        print(
            f"Market max drawdown:    "
            f"{market['max_drawdown_pct']:.4f}%"
        )

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()