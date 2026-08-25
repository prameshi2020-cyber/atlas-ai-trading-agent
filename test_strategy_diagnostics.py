from pathlib import Path

import pandas as pd

from app.analysis.indicators import calculate_indicators
from app.analysis.strategy import StrategyEngine


SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def load_dataset(symbol: str) -> pd.DataFrame:
    path = (
        Path(__file__).parent
        / "data"
        / "backtest"
        / f"{symbol}_1m.csv"
    )

    df = pd.read_csv(path)

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

    return (
        df.sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )


def main() -> None:

    print("=" * 80)
    print("ATLAS STRATEGY DIAGNOSTICS")
    print("=" * 80)

    strategy = StrategyEngine()

    for symbol in SYMBOLS:

        print("\n" + "=" * 80)
        print(symbol)
        print("=" * 80)

        df = load_dataset(symbol)
        features = calculate_indicators(df)

        # Remove indicator warm-up rows.
        features = features.dropna().reset_index(drop=True)

        signal_counts = {
            "BUY": 0,
            "SELL": 0,
            "HOLD": 0,
        }

        scores = []
        confidences = []

        reasons = {}

        for i in range(len(features)):

            history = features.iloc[: i + 1]

            signal = strategy.generate_signal(
                symbol=symbol,
                features=history,
            )

            action = signal.action

            signal_counts[action] += 1

            scores.append(signal.score)
            confidences.append(signal.confidence)

            for reason in signal.reasons:
                reasons[reason] = reasons.get(reason, 0) + 1

        total = len(features)

        print("\nSIGNAL DISTRIBUTION")
        print("-" * 80)

        for action in ["BUY", "SELL", "HOLD"]:

            count = signal_counts[action]

            percentage = (
                count / total * 100
                if total
                else 0
            )

            print(
                f"{action:<8}"
                f"{count:>8}"
                f"{percentage:>12.2f}%"
            )

        print("\nSCORE STATISTICS")
        print("-" * 80)

        print(
            f"Minimum score:       {min(scores):.4f}"
        )

        print(
            f"Maximum score:       {max(scores):.4f}"
        )

        print(
            f"Average score:       "
            f"{sum(scores) / len(scores):.4f}"
        )

        print(
            f"Average confidence:  "
            f"{sum(confidences) / len(confidences):.4f}"
        )

        print("\nREASON FREQUENCY")
        print("-" * 80)

        for reason, count in sorted(
            reasons.items(),
            key=lambda item: item[1],
            reverse=True,
        ):

            percentage = count / total * 100

            print(
                f"{count:>5} "
                f"({percentage:>6.2f}%)  "
                f"{reason}"
            )

        print("\nPRICE MOVEMENT")
        print("-" * 80)

        start_price = float(features.iloc[0]["close"])
        end_price = float(features.iloc[-1]["close"])

        market_return = (
            (end_price - start_price)
            / start_price
            * 100
        )

        print(
            f"Feature-period return: "
            f"{market_return:.4f}%"
        )

        print(
            f"Starting close:        "
            f"{start_price:.4f}"
        )

        print(
            f"Ending close:          "
            f"{end_price:.4f}"
        )

    print("\n" + "=" * 80)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()