from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import timedelta
import math


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _non_negative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be an integer >= 0")
    return value


def _name_set(value: object, label: str) -> frozenset[str] | None:
    if value is None:
        return None
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise TypeError(f"{label} must be an iterable of strings or None")
    names = frozenset(str(item).strip() for item in value)
    if not names or any(not name for name in names):
        raise ValueError(f"{label} must be a non-empty set of non-empty strings, or None")
    return names


@dataclass(frozen=True, slots=True)
class DeploymentPolicy:
    """Deterministic thresholds applied to research output. Nothing here recomputes a metric."""

    allowed_symbols: frozenset[str] | None = None
    allowed_timeframes: frozenset[str] | None = None
    minimum_walk_forward_windows: int = 3
    minimum_oos_pass_rate_pct: float = 60.0
    maximum_drawdown_pct: float = 20.0  # positive magnitude, per the backtest drawdown contract
    minimum_total_trades: int = 10
    maximum_dataset_age: timedelta | None = timedelta(hours=24)
    require_no_batch_failures: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_symbols", _name_set(self.allowed_symbols, "allowed_symbols"))
        object.__setattr__(self, "allowed_timeframes", _name_set(self.allowed_timeframes, "allowed_timeframes"))

        object.__setattr__(
            self,
            "minimum_walk_forward_windows",
            _non_negative_int(self.minimum_walk_forward_windows, "minimum_walk_forward_windows"),
        )
        object.__setattr__(self, "minimum_total_trades", _non_negative_int(self.minimum_total_trades, "minimum_total_trades"))

        pass_rate = _finite_number(self.minimum_oos_pass_rate_pct, "minimum_oos_pass_rate_pct")
        if not 0.0 <= pass_rate <= 100.0:
            raise ValueError("minimum_oos_pass_rate_pct must be within [0, 100]")
        object.__setattr__(self, "minimum_oos_pass_rate_pct", pass_rate)

        drawdown = _finite_number(self.maximum_drawdown_pct, "maximum_drawdown_pct")
        if drawdown < 0.0:
            raise ValueError("maximum_drawdown_pct is a positive magnitude and cannot be negative")
        object.__setattr__(self, "maximum_drawdown_pct", drawdown)

        if self.maximum_dataset_age is not None:
            if not isinstance(self.maximum_dataset_age, timedelta) or self.maximum_dataset_age <= timedelta(0):
                raise ValueError("maximum_dataset_age must be a positive timedelta or None")

        if not isinstance(self.require_no_batch_failures, bool):
            raise TypeError("require_no_batch_failures must be a bool")