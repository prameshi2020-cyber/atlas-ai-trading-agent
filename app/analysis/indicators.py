from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators from OHLCV candle data.

    Required columns:
        open, high, low, close, volume

    The input DataFrame must be sorted chronologically.
    """

    required_columns = {
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    result = df.copy()

    result = result.sort_values("timestamp").reset_index(drop=True)

    close = result["close"]
    high = result["high"]
    low = result["low"]
    volume = result["volume"]

    # ---------------------------------------------------------
    # Moving averages
    # ---------------------------------------------------------

    result["sma_20"] = close.rolling(window=20).mean()
    result["sma_50"] = close.rolling(window=50).mean()

    result["ema_9"] = close.ewm(
        span=9,
        adjust=False,
    ).mean()

    result["ema_21"] = close.ewm(
        span=21,
        adjust=False,
    ).mean()

    # ---------------------------------------------------------
    # RSI 14
    # ---------------------------------------------------------

    delta = close.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    avg_loss = losses.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    result["rsi_14"] = 100 - (
        100 / (1 + rs)
    )

    # ---------------------------------------------------------
    # MACD
    # ---------------------------------------------------------

    ema_12 = close.ewm(
        span=12,
        adjust=False,
    ).mean()

    ema_26 = close.ewm(
        span=26,
        adjust=False,
    ).mean()

    result["macd"] = ema_12 - ema_26

    result["macd_signal"] = result["macd"].ewm(
        span=9,
        adjust=False,
    ).mean()

    result["macd_histogram"] = (
        result["macd"]
        - result["macd_signal"]
    )

    # ---------------------------------------------------------
    # Bollinger Bands
    # ---------------------------------------------------------

    bb_middle = close.rolling(
        window=20
    ).mean()

    bb_std = close.rolling(
        window=20
    ).std()

    result["bb_middle"] = bb_middle

    result["bb_upper"] = (
        bb_middle + (2 * bb_std)
    )

    result["bb_lower"] = (
        bb_middle - (2 * bb_std)
    )

    # Position of price inside Bollinger Bands.
    band_width = (
        result["bb_upper"]
        - result["bb_lower"]
    )

    result["bb_position"] = (
        (close - result["bb_lower"])
        / band_width.replace(0, np.nan)
    )

    # ---------------------------------------------------------
    # ATR 14
    # ---------------------------------------------------------

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    result["tr"] = true_range

    result["atr_14"] = true_range.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    # ---------------------------------------------------------
    # Volume
    # ---------------------------------------------------------

    result["volume_sma_20"] = volume.rolling(
        window=20
    ).mean()

    result["volume_ratio"] = (
        volume
        / result["volume_sma_20"].replace(0, np.nan)
    )

    # ---------------------------------------------------------
    # Momentum
    # ---------------------------------------------------------

    result["momentum_10"] = (
        close.pct_change(periods=10) * 100
    )

    result["momentum_20"] = (
        close.pct_change(periods=20) * 100
    )

    # ---------------------------------------------------------
    # Volatility
    # ---------------------------------------------------------

    returns = close.pct_change()

    result["volatility_20"] = (
        returns.rolling(window=20).std() * 100
    )

    # ---------------------------------------------------------
    # Trend features
    # ---------------------------------------------------------

    result["ema_spread"] = (
        result["ema_9"]
        - result["ema_21"]
    )

    result["price_vs_ema21"] = (
        (close - result["ema_21"])
        / result["ema_21"]
        * 100
    )

    result["price_vs_sma50"] = (
        (close - result["sma_50"])
        / result["sma_50"]
        * 100
    )

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    result.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    return result


# ---------------------------------------------------------------------
# Warm-up handling
# ---------------------------------------------------------------------
#
# The rolling/EWM windows above need a minimum number of prior candles
# before they produce a real value (e.g. sma_50 needs 50 candles, so
# the first 49 rows are NaN by construction -- this is correct and
# expected, not a bug in calculate_indicators()).
#
# The bug is downstream: naive comparisons like
#   row["ema_9"] > row["ema_21"]
# silently evaluate to False when either side is NaN, which biases
# every warm-up row toward a fabricated bearish signal instead of an
# honest "not enough data yet". REQUIRED_FEATURE_COLUMNS and
# drop_warmup_rows() let callers (the backtest engine, in particular)
# strip those rows out *before* they ever reach StrategyEngine, so no
# trading decision is ever generated from incomplete indicator data.

# Every column StrategyEngine.generate_signal() reads from the latest
# row, plus atr_14 which BacktestEngine needs for stop-loss /
# take-profit sizing. sma_50 has the longest warm-up window (50
# candles) and is therefore the binding constraint in practice.
REQUIRED_FEATURE_COLUMNS: tuple[str, ...] = (
    "ema_9",
    "ema_21",
    "sma_50",
    "rsi_14",
    "macd_histogram",
    "atr_14",
    "price_vs_ema21",
    "price_vs_sma50",
    "momentum_10",
    "momentum_20",
)


def drop_warmup_rows(
    df: pd.DataFrame,
    required_columns: tuple[str, ...] = REQUIRED_FEATURE_COLUMNS,
) -> pd.DataFrame:
    """
    Remove rows that do not yet have every required indicator value
    populated.

    This is a deterministic, order-preserving filter: it keeps every
    row where all `required_columns` are non-NaN, and drops the rest.
    In the normal case (indicators computed chronologically with
    rolling/EWM windows) the dropped rows are exactly the leading
    warm-up period, but the filter itself makes no assumption about
    row order -- it simply will never pass through a row with missing
    required indicator data.

    Raises ValueError if `df` is missing any of `required_columns`,
    since that indicates a caller/indicator mismatch rather than a
    normal warm-up gap.
    """
    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    valid_mask = df[list(required_columns)].notna().all(axis=1)

    return df.loc[valid_mask].reset_index(drop=True)