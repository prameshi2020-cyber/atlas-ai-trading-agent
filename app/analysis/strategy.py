from dataclasses import dataclass
from typing import Literal

import pandas as pd


Action = Literal["BUY", "SELL", "HOLD"]


@dataclass(slots=True)
class Signal:
    symbol: str
    action: Action
    score: float
    confidence: float
    reasons: list[str]


class StrategyEngine:
    """
    Deterministic rule-based strategy.

    The engine consumes the feature dataframe produced by the
    indicator engine and generates BUY / SELL / HOLD signals.
    """

    def __init__(
        self,
        buy_threshold: float = 0.60,
        sell_threshold: float = -0.60,
    ) -> None:
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def generate_signal(
        self,
        symbol: str,
        features: pd.DataFrame,
    ) -> Signal:
        if features.empty:
            raise ValueError("Feature dataframe is empty")

        row = features.iloc[-1]

        score = 0.0
        reasons: list[str] = []

        # ---------------------------------------------------------
        # 1. EMA trend
        # ---------------------------------------------------------
        if row["ema_9"] > row["ema_21"]:
            score += 0.20
            reasons.append("EMA9 above EMA21")
        else:
            score -= 0.20
            reasons.append("EMA9 below EMA21")

        # ---------------------------------------------------------
        # 2. MACD momentum
        # ---------------------------------------------------------
        if row["macd_histogram"] > 0:
            score += 0.20
            reasons.append("MACD histogram positive")
        else:
            score -= 0.20
            reasons.append("MACD histogram negative")

        # ---------------------------------------------------------
        # 3. RSI
        # ---------------------------------------------------------
        rsi = float(row["rsi_14"])

        if 50 <= rsi <= 70:
            score += 0.15
            reasons.append(f"RSI bullish ({rsi:.1f})")
        elif 30 <= rsi < 50:
            score -= 0.05
            reasons.append(f"RSI weak ({rsi:.1f})")
        elif rsi < 30:
            score += 0.10
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > 70:
            score -= 0.10
            reasons.append(f"RSI overbought ({rsi:.1f})")

        # ---------------------------------------------------------
        # 4. Price vs EMA21
        # ---------------------------------------------------------
        if row["price_vs_ema21"] > 0:
            score += 0.15
            reasons.append("Price above EMA21")
        else:
            score -= 0.15
            reasons.append("Price below EMA21")

        # ---------------------------------------------------------
        # 5. Price vs SMA50
        # ---------------------------------------------------------
        if row["price_vs_sma50"] > 0:
            score += 0.10
            reasons.append("Price above SMA50")
        else:
            score -= 0.10
            reasons.append("Price below SMA50")

        # ---------------------------------------------------------
        # 6. Momentum
        # ---------------------------------------------------------
        if row["momentum_10"] > 0:
            score += 0.10
            reasons.append("10-period momentum positive")
        else:
            score -= 0.10
            reasons.append("10-period momentum negative")

        if row["momentum_20"] > 0:
            score += 0.10
            reasons.append("20-period momentum positive")
        else:
            score -= 0.10
            reasons.append("20-period momentum negative")

        # ---------------------------------------------------------
        # Normalize score
        # ---------------------------------------------------------
        score = max(-1.0, min(1.0, score))

        # ---------------------------------------------------------
        # Signal
        # ---------------------------------------------------------
        if score >= self.buy_threshold:
            action: Action = "BUY"
        elif score <= self.sell_threshold:
            action = "SELL"
        else:
            action = "HOLD"

        confidence = abs(score)

        return Signal(
            symbol=symbol,
            action=action,
            score=round(score, 4),
            confidence=round(confidence, 4),
            reasons=reasons,
        )