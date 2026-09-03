from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal

from app.analysis.backtest import BacktestResult


EvaluationDecision = Literal["PASS", "FAIL", "REVIEW"]


@dataclass(frozen=True, slots=True)
class StrategyEvaluationPolicy:
    """
    Quality gate for a completed backtest.

    All thresholds are intentionally configurable so the policy remains the
    acceptance layer and does not hardcode arbitrary values in the evaluator.
    """

    minimum_total_return_pct: float = 0.0
    minimum_profit_factor: float = 0.0
    minimum_win_rate_pct: float = 0.0
    maximum_drawdown_pct: float = 100.0
    minimum_trade_count: int = 1
    minimum_sharpe_ratio: float = 0.0
    minimum_sortino_ratio: float = 0.0


@dataclass(frozen=True, slots=True)
class StrategyEvaluation:
    decision: EvaluationDecision
    passed: bool
    failed_criteria: list[str]
    reasons: list[str]
    metrics: dict[str, float]
    metric_definitions: dict[str, str] = field(default_factory=dict)


class StrategyEvaluator:
    """
    Evaluate a BacktestResult against configurable quality thresholds.

    The metric definitions are intentionally documented and conservative:

    - Sharpe and Sortino use per-trade fractional returns as the return series.
    - No annualization is assumed because the current result object does not
      contain a defined time basis or sampling frequency.
    - Risk-free rate is treated as 0.0.
    - A zero-volatility series resolves to 0.0 rather than NaN or inf.
    """

    @staticmethod
    def _mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _population_stddev(values: list[float]) -> float:
        if len(values) <= 1:
            return 0.0
        mean = StrategyEvaluator._mean(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return math.sqrt(variance)

    @staticmethod
    def _downside_deviation(values: list[float]) -> float:
        negatives = [value for value in values if value < 0.0]
        if not negatives:
            return 0.0
        mean = StrategyEvaluator._mean(negatives)
        variance = sum((value - mean) ** 2 for value in negatives) / len(negatives)
        return math.sqrt(variance)

    @staticmethod
    def _sharpe_ratio(returns: list[float]) -> float:
        if not returns:
            return 0.0
        mean_return = StrategyEvaluator._mean(returns)
        std_dev = StrategyEvaluator._population_stddev(returns)
        if std_dev == 0.0:
            return float("inf") if mean_return > 0.0 else 0.0
        return mean_return / std_dev

    @staticmethod
    def _sortino_ratio(returns: list[float]) -> float:
        if not returns:
            return 0.0
        mean_return = StrategyEvaluator._mean(returns)
        downside = StrategyEvaluator._downside_deviation(returns)
        if downside == 0.0:
            return float("inf") if mean_return > 0.0 else 0.0
        return mean_return / downside

    @staticmethod
    def _average_holding_time_seconds(result: BacktestResult) -> float:
        if not result.trades:
            return 0.0

        durations = []
        for trade in result.trades:
            if trade.entry_time is None or trade.exit_time is None:
                continue
            delta = trade.exit_time - trade.entry_time
            durations.append(delta.total_seconds())

        if not durations:
            return 0.0

        return sum(durations) / len(durations)

    def evaluate(
        self,
        result: BacktestResult,
        *,
        policy: StrategyEvaluationPolicy | None = None,
    ) -> StrategyEvaluation:
        policy = policy or StrategyEvaluationPolicy()

        if result.total_trades == 0:
            metrics = {
                "total_return_pct": float(result.total_return_pct),
                "net_pnl": 0.0,
                "total_trades": 0,
                "win_rate_pct": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
                "max_drawdown_pct": float(result.max_drawdown_pct),
                "average_trade_pct": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "average_holding_time_seconds": 0.0,
            }
            return StrategyEvaluation(
                decision="FAIL",
                passed=False,
                failed_criteria=["no_trades"],
                reasons=["No trades were executed."],
                metrics=metrics,
                metric_definitions={
                    "sharpe_ratio": "Non-annualized Sharpe ratio using per-trade fractional returns (pct/100) as the return series; risk-free rate is 0.0 and no annualization factor is applied.",
                    "sortino_ratio": "Non-annualized Sortino ratio using per-trade fractional returns (pct/100) as the return series; downside deviation is computed on negative returns only; risk-free rate is 0.0 and no annualization factor is applied.",
                },
            )

        trade_returns = [trade.pnl_pct / 100.0 for trade in result.trades]
        sharpe = self._sharpe_ratio(trade_returns)
        sortino = self._sortino_ratio(trade_returns)

        metrics = {
            "total_return_pct": float(result.total_return_pct),
            "net_pnl": float(sum(trade.pnl_amount for trade in result.trades)),
            "total_trades": int(result.total_trades),
            "win_rate_pct": float(result.win_rate_pct),
            "profit_factor": float(result.profit_factor),
            "expectancy": float(result.expectancy_pct),
            "max_drawdown_pct": float(result.max_drawdown_pct),
            "average_trade_pct": float(result.average_trade_pct),
            "sharpe_ratio": float(sharpe) if math.isfinite(sharpe) else 0.0,
            "sortino_ratio": float(sortino) if math.isfinite(sortino) else 0.0,
            "average_holding_time_seconds": self._average_holding_time_seconds(result),
        }

        failed_criteria: list[str] = []
        reasons: list[str] = []

        if result.total_return_pct < policy.minimum_total_return_pct:
            failed_criteria.append("total_return_pct")
            reasons.append(
                f"Total return {result.total_return_pct:.2f}% is below the minimum {policy.minimum_total_return_pct:.2f}%."
            )

        if result.profit_factor < policy.minimum_profit_factor:
            failed_criteria.append("profit_factor")
            reasons.append(
                f"Profit factor {result.profit_factor:.2f} is below the minimum {policy.minimum_profit_factor:.2f}."
            )

        if result.win_rate_pct < policy.minimum_win_rate_pct:
            failed_criteria.append("win_rate_pct")
            reasons.append(
                f"Win rate {result.win_rate_pct:.2f}% is below the minimum {policy.minimum_win_rate_pct:.2f}%."
            )

        if result.max_drawdown_pct > policy.maximum_drawdown_pct:
            failed_criteria.append("max_drawdown_pct")
            reasons.append(
                f"Max drawdown {result.max_drawdown_pct:.2f}% exceeds the maximum {policy.maximum_drawdown_pct:.2f}%."
            )

        if result.total_trades < policy.minimum_trade_count:
            failed_criteria.append("minimum_trade_count")
            reasons.append(
                f"Trade count {result.total_trades} is below the minimum {policy.minimum_trade_count}."
            )

        if metrics["sharpe_ratio"] < policy.minimum_sharpe_ratio:
            failed_criteria.append("sharpe_ratio")
            reasons.append(
                f"Sharpe ratio {metrics['sharpe_ratio']:.4f} is below the minimum {policy.minimum_sharpe_ratio:.4f}."
            )

        if metrics["sortino_ratio"] < policy.minimum_sortino_ratio:
            failed_criteria.append("sortino_ratio")
            reasons.append(
                f"Sortino ratio {metrics['sortino_ratio']:.4f} is below the minimum {policy.minimum_sortino_ratio:.4f}."
            )

        if failed_criteria:
            decision: EvaluationDecision = "FAIL"
            passed = False
        else:
            decision = "PASS"
            passed = True

        return StrategyEvaluation(
            decision=decision,
            passed=passed,
            failed_criteria=failed_criteria,
            reasons=reasons,
            metrics=metrics,
            metric_definitions={
                "sharpe_ratio": "Non-annualized Sharpe ratio using per-trade fractional returns (pct/100) as the return series; risk-free rate is 0.0 and no annualization factor is applied.",
                "sortino_ratio": "Non-annualized Sortino ratio using per-trade fractional returns (pct/100) as the return series; downside deviation is computed on negative returns only; risk-free rate is 0.0 and no annualization factor is applied.",
            },
        )
