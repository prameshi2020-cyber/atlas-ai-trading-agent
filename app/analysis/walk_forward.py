from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import pandas as pd
import inspect

from app.analysis.backtest import BacktestEngine, BacktestResult
from app.analysis.evaluation import StrategyEvaluation, StrategyEvaluationPolicy, StrategyEvaluator
from app.analysis.strategy import StrategyEngine

RobustnessDecision = Literal["PASS", "FAIL", "REVIEW"]

# Constructor arguments BacktestEngine accepts for a window backtest. Derived from the
# signature so WalkForwardEngine never re-declares (or drifts from) BacktestEngine's rules.
_BACKTEST_KWARG_NAMES: frozenset[str] = frozenset(
    name for name in inspect.signature(BacktestEngine.__init__).parameters if name not in {"self", "strategy"}
)

@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    """
    Sequential walk-forward configuration for strategy robustness checks.

    This layer evaluates a strategy on separately sliced historical windows
    and keeps the in-sample and out-of-sample slices strictly separate.
    It does not perform parameter optimization; it only measures the
    stability of a given strategy across time.
    """

    train_bars: int
    test_bars: int
    step_bars: int
    minimum_train_trades: int = 1
    minimum_test_trades: int = 1
    evaluation_policy: StrategyEvaluationPolicy | None = None
    minimum_oos_pass_rate_pct: float = 0.0
    minimum_oos_return_pct: float = 0.0
    maximum_oos_drawdown_pct: float = 100.0
    minimum_oos_profit_factor: float = 0.0
    strategy_factory: Callable[[], object] | None = None

    def __post_init__(self) -> None:
        if self.train_bars <= 0:
            raise ValueError("train_bars must be greater than zero")
        if self.test_bars <= 0:
            raise ValueError("test_bars must be greater than zero")
        if self.step_bars <= 0:
            raise ValueError("step_bars must be greater than zero")
        if self.minimum_train_trades < 0:
            raise ValueError("minimum_train_trades cannot be negative")
        if self.minimum_test_trades < 0:
            raise ValueError("minimum_test_trades cannot be negative")
        if self.minimum_oos_pass_rate_pct < 0:
            raise ValueError("minimum_oos_pass_rate_pct cannot be negative")
        if self.minimum_oos_return_pct < 0:
            raise ValueError("minimum_oos_return_pct cannot be negative")
        if self.maximum_oos_drawdown_pct < 0:
            raise ValueError("maximum_oos_drawdown_pct cannot be negative")
        if self.minimum_oos_profit_factor < 0:
            raise ValueError("minimum_oos_profit_factor cannot be negative")


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    in_sample_result: BacktestResult | None = None
    out_of_sample_result: BacktestResult | None = None
    in_sample_evaluation: StrategyEvaluation | None = None
    out_of_sample_evaluation: StrategyEvaluation | None = None


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    windows: list[WalkForwardWindow]
    total_windows: int
    passed_windows: int
    failed_windows: int
    oos_pass_rate_pct: float
    aggregate_oos_return_pct: float
    aggregate_oos_pnl: float
    robustness_decision: RobustnessDecision
    reasons: list[str] = field(default_factory=list)


class WalkForwardEngine:
    """
    Evaluate a strategy across sequential in-sample / out-of-sample windows.

    Window semantics: [start, end) half-open intervals. The strategy is backtested
    on the IS slice and then on a strictly later OOS slice with no shared rows.
    This prevents look-ahead and keeps the evaluation temporally causal.

    Important: this is not parameter tuning. The same strategy instance is not
    optimized across windows; it is evaluated as-is on each chronological slice.
    """

    def __init__(
        self,
        *,
        strategy_factory: Callable[[], object] | None = None,
        evaluator: StrategyEvaluator | None = None,
        backtest_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.strategy_factory = strategy_factory or (lambda: StrategyEngine())
        self.evaluator = evaluator or StrategyEvaluator()
        self.backtest_kwargs = self._normalize_backtest_kwargs(backtest_kwargs)

    @staticmethod
    def _normalize_backtest_kwargs(kwargs: Mapping[str, Any] | None) -> Mapping[str, Any]:
        """Keys are checked against BacktestEngine's signature; values are validated by BacktestEngine itself."""
        if kwargs is None:
            return MappingProxyType({})
        if not isinstance(kwargs, Mapping):
            raise TypeError("backtest_kwargs must be a mapping of BacktestEngine constructor arguments")
        unknown = sorted(str(key) for key in kwargs if key not in _BACKTEST_KWARG_NAMES)
        if unknown:
            raise TypeError(f"backtest_kwargs contains arguments BacktestEngine does not accept: {unknown}")
        return MappingProxyType(dict(kwargs))

    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if data is None or data.empty:
            raise ValueError("Feature dataframe cannot be empty")
        if "timestamp" not in data.columns:
            raise ValueError("Feature dataframe is missing the required 'timestamp' column")

        timestamps = pd.to_datetime(data["timestamp"], errors="coerce")
        if timestamps.isna().any():
            raise ValueError("Feature dataframe contains invalid timestamps")

        if not timestamps.is_monotonic_increasing:
            raise ValueError("Feature dataframe timestamps must be chronological and non-decreasing")

        return data.copy().sort_values("timestamp").reset_index(drop=True)

    def _generate_windows(
        self,
        data: pd.DataFrame,
        config: WalkForwardConfig,
    ) -> list[WalkForwardWindow]:
        if len(data) < config.train_bars + config.test_bars:
            raise ValueError(
                "insufficient data for a valid walk-forward configuration: "
                "data length is smaller than train_bars + test_bars"
            )

        windows: list[WalkForwardWindow] = []
        start = 0
        max_start = len(data) - (config.train_bars + config.test_bars)

        while start <= max_start:
            train_start = start
            train_end = start + config.train_bars
            test_start = train_end
            test_end = test_start + config.test_bars

            if train_end <= train_start or test_end <= test_start:
                raise ValueError("invalid window bounds: train/test windows must be positive half-open intervals")

            windows.append(
                WalkForwardWindow(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )

            start += config.step_bars

        if not windows:
            raise ValueError("insufficient data to build any valid walk-forward windows")

        return windows

    def _backtest_slice(
        self,
        symbol: str,
        features: pd.DataFrame,
        *,
        strategy: object,
    ) -> BacktestResult:
        if features is None or features.empty:
            raise ValueError("cannot backtest on an empty slice")

        engine = BacktestEngine(strategy=strategy, **self.backtest_kwargs)
        try:
            return engine.run(symbol=symbol, features=features)
        except ValueError as exc:
            message = str(exc).lower()
            if "warm" in message or "at least 3" in message:
                raise ValueError(
                    "walk-forward evaluation failed because the slice is too short after warm-up filtering"
                ) from exc
            raise

    def _evaluate_window(
        self,
        data: pd.DataFrame,
        window: WalkForwardWindow,
        config: WalkForwardConfig,
    ) -> WalkForwardWindow:
        train_slice = data.iloc[window.train_start:window.train_end].copy()
        test_slice = data.iloc[window.test_start:window.test_end].copy()

        if len(train_slice) < config.train_bars:
            raise ValueError("train slice is shorter than the configured window")
        if len(test_slice) < config.test_bars:
            raise ValueError("test slice is shorter than the configured window")

        strategy = self.strategy_factory()
        in_sample_result = self._backtest_slice("WALK_FORWARD_IS", train_slice, strategy=strategy)

        if in_sample_result.total_trades < config.minimum_train_trades:
            raise ValueError(
                f"in-sample window [{window.train_start}:{window.train_end}) had "
                f"{in_sample_result.total_trades} trades, below minimum_train_trades={config.minimum_train_trades}"
            )

        if len(train_slice) < 3:
            raise ValueError("in-sample window is too short for backtest execution")

        strategy_oos = self.strategy_factory()
        out_of_sample_result = self._backtest_slice("WALK_FORWARD_OOS", test_slice, strategy=strategy_oos)

        evaluator_policy = config.evaluation_policy or StrategyEvaluationPolicy()
        in_sample_eval = self.evaluator.evaluate(in_sample_result, policy=evaluator_policy)
        out_of_sample_eval = self.evaluator.evaluate(out_of_sample_result, policy=evaluator_policy)

        if out_of_sample_result.total_trades < config.minimum_test_trades:
            out_of_sample_eval = StrategyEvaluation(
                decision="FAIL",
                passed=False,
                failed_criteria=["minimum_test_trades"],
                reasons=[f"Out-of-sample trade count {out_of_sample_result.total_trades} is below minimum_test_trades={config.minimum_test_trades}."],
                metrics=out_of_sample_eval.metrics,
                metric_definitions=out_of_sample_eval.metric_definitions,
            )

        return WalkForwardWindow(
            train_start=window.train_start,
            train_end=window.train_end,
            test_start=window.test_start,
            test_end=window.test_end,
            in_sample_result=in_sample_result,
            out_of_sample_result=out_of_sample_result,
            in_sample_evaluation=in_sample_eval,
            out_of_sample_evaluation=out_of_sample_eval,
        )

    def run(
        self,
        data: pd.DataFrame,
        config: WalkForwardConfig,
    ) -> WalkForwardResult:
        checked = self._validate_data(data)

        windows = [
            self._evaluate_window(checked, window, config)
            for window in self._generate_windows(checked, config)
        ]

        if not windows:
            raise ValueError("no walk-forward windows were generated")

        total_windows = len(windows)
        passed_windows = sum(
            1 for window in windows if window.out_of_sample_evaluation is not None and window.out_of_sample_evaluation.passed
        )
        failed_windows = total_windows - passed_windows
        oos_pass_rate_pct = (passed_windows / total_windows) * 100.0 if total_windows else 0.0

        aggregate_oos_pnl = 0.0
        total_oos_capital = 0.0
        for window in windows:
            result = window.out_of_sample_result
            if result is None:
                continue
            aggregate_oos_pnl += sum(trade.pnl_amount for trade in result.trades)
            total_oos_capital += result.initial_capital

        aggregate_oos_return_pct = (
            (aggregate_oos_pnl / total_oos_capital) * 100.0 if total_oos_capital > 0 else 0.0
        )

        reasons: list[str] = []
        if oos_pass_rate_pct < config.minimum_oos_pass_rate_pct:
            reasons.append(
                f"Out-of-sample pass rate {oos_pass_rate_pct:.2f}% is below the minimum {config.minimum_oos_pass_rate_pct:.2f}%."
            )

        max_oos_drawdown = max(
            (window.out_of_sample_result.max_drawdown_pct for window in windows if window.out_of_sample_result is not None),
            default=0.0,
        )
        if max_oos_drawdown > config.maximum_oos_drawdown_pct:
            reasons.append(
                f"Maximum OOS drawdown {max_oos_drawdown:.2f}% exceeds the configured maximum {config.maximum_oos_drawdown_pct:.2f}%."
            )

        if aggregate_oos_return_pct < config.minimum_oos_return_pct:
            reasons.append(
                f"Aggregate OOS return {aggregate_oos_return_pct:.2f}% is below the configured minimum {config.minimum_oos_return_pct:.2f}%."
            )

        if windows and any(
            window.out_of_sample_result is not None and window.out_of_sample_result.profit_factor < config.minimum_oos_profit_factor
            for window in windows
        ):
            reasons.append(
                f"At least one OOS window has profit factor below the configured minimum {config.minimum_oos_profit_factor:.2f}."
            )

        if not reasons:
            robustness_decision: RobustnessDecision = "PASS"
        elif failed_windows > 0 and oos_pass_rate_pct < config.minimum_oos_pass_rate_pct:
            robustness_decision = "FAIL"
        elif failed_windows > 0:
            robustness_decision = "REVIEW"
        else:
            robustness_decision = "PASS"

        return WalkForwardResult(
            windows=windows,
            total_windows=total_windows,
            passed_windows=passed_windows,
            failed_windows=failed_windows,
            oos_pass_rate_pct=oos_pass_rate_pct,
            aggregate_oos_return_pct=aggregate_oos_return_pct,
            aggregate_oos_pnl=aggregate_oos_pnl,
            robustness_decision=robustness_decision,
            reasons=reasons,
        )
