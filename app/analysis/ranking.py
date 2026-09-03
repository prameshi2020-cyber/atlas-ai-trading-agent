from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from app.analysis.backtest import BacktestResult
from app.analysis.evaluation import StrategyEvaluation
from app.analysis.walk_forward import WalkForwardResult

RankingDecision = Literal["PASS", "FAIL", "REVIEW"]


@dataclass(frozen=True, slots=True)
class Candidate:
    """A strategy candidate evaluated by upstream layers."""

    candidate_id: str
    name: str
    strategy: object

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id cannot be empty")
        if not self.name:
            raise ValueError("name cannot be empty")


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """Persisted evaluation payload for a single strategy candidate."""

    candidate: Candidate
    backtest_result: BacktestResult | None = None
    evaluation: StrategyEvaluation | None = None
    walk_forward_result: WalkForwardResult | None = None


@dataclass(frozen=True, slots=True)
class RankingPolicy:
    """Configurable deterministic ranking of already-evaluated strategy candidates."""

    minimum_sample_trades: int = 1
    minimum_total_return_pct: float = 0.0
    maximum_drawdown_pct: float = 100.0
    minimum_win_rate_pct: float = 0.0
    minimum_profit_factor: float = 0.0
    minimum_sharpe_ratio: float = 0.0
    minimum_sortino_ratio: float = 0.0

    robustness_weight: float = 0.30
    evaluation_weight: float = 0.15
    return_weight: float = 0.20
    drawdown_weight: float = 0.20
    sample_weight: float = 0.05
    profit_weight: float = 0.05
    win_rate_weight: float = 0.03
    risk_adjusted_weight: float = 0.02

    def __post_init__(self) -> None:
        if self.minimum_sample_trades < 0:
            raise ValueError("minimum_sample_trades cannot be negative")
        if self.minimum_total_return_pct < 0:
            raise ValueError("minimum_total_return_pct cannot be negative")
        if self.maximum_drawdown_pct < 0:
            raise ValueError("maximum_drawdown_pct cannot be negative")
        if self.minimum_win_rate_pct < 0:
            raise ValueError("minimum_win_rate_pct cannot be negative")
        if self.minimum_profit_factor < 0:
            raise ValueError("minimum_profit_factor cannot be negative")
        if self.minimum_sharpe_ratio < 0:
            raise ValueError("minimum_sharpe_ratio cannot be negative")
        if self.minimum_sortino_ratio < 0:
            raise ValueError("minimum_sortino_ratio cannot be negative")


@dataclass(frozen=True, slots=True)
class CandidateScore:
    candidate_id: str
    candidate_name: str
    score: float
    accepted: bool
    robustness_decision: RankingDecision
    evaluation_decision: RankingDecision
    reasons: list[str] = field(default_factory=list)
    metrics_used: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RankingResult:
    ranked_candidates: list[CandidateScore]
    best_candidate: CandidateScore | None = None

    def __post_init__(self) -> None:
        if self.ranked_candidates:
            object.__setattr__(self, "best_candidate", self.ranked_candidates[0])
        else:
            object.__setattr__(self, "best_candidate", None)


class RankingEngine:
    """Deterministic ranker for already-evaluated strategy candidates."""

    @staticmethod
    def _decision_value(decision: str | None) -> float:
        if decision is None:
            return 0.0
        mapping = {"PASS": 1.0, "REVIEW": 0.5, "FAIL": 0.0}
        return mapping.get(str(decision).upper(), 0.0)

    @staticmethod
    def _safe_numeric(mapping: dict[str, float], key: str, default: float = 0.0) -> float:
        value = mapping.get(key, default)
        return float(value) if value is not None else float(default)

    def _metrics_from_candidate(
        self,
        evaluation: CandidateEvaluation,
    ) -> dict[str, float]:
        result = evaluation.backtest_result
        strategy_eval = evaluation.evaluation

        if result is not None:
            total_return_pct = float(result.total_return_pct)
            max_drawdown_pct = float(result.max_drawdown_pct)
            trade_count = int(result.total_trades)
            win_rate_pct = float(result.win_rate_pct)
            profit_factor = float(result.profit_factor)
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
            if strategy_eval is not None and strategy_eval.metrics:
                sharpe_ratio = self._safe_numeric(strategy_eval.metrics, "sharpe_ratio", 0.0)
                sortino_ratio = self._safe_numeric(strategy_eval.metrics, "sortino_ratio", 0.0)
        else:
            total_return_pct = 0.0
            max_drawdown_pct = 0.0
            trade_count = 0
            win_rate_pct = 0.0
            profit_factor = 0.0
            sharpe_ratio = 0.0
            sortino_ratio = 0.0

        if strategy_eval is not None and strategy_eval.metrics:
            total_return_pct = self._safe_numeric(strategy_eval.metrics, "total_return_pct", total_return_pct)
            max_drawdown_pct = self._safe_numeric(strategy_eval.metrics, "max_drawdown_pct", max_drawdown_pct)
            trade_count = int(self._safe_numeric(strategy_eval.metrics, "total_trades", trade_count))
            win_rate_pct = self._safe_numeric(strategy_eval.metrics, "win_rate_pct", win_rate_pct)
            profit_factor = self._safe_numeric(strategy_eval.metrics, "profit_factor", profit_factor)
            sharpe_ratio = self._safe_numeric(strategy_eval.metrics, "sharpe_ratio", sharpe_ratio)
            sortino_ratio = self._safe_numeric(strategy_eval.metrics, "sortino_ratio", sortino_ratio)

        return {
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "total_trades": trade_count,
            "win_rate_pct": win_rate_pct,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
        }

    def score_candidate(
        self,
        evaluation: CandidateEvaluation,
        *,
        policy: RankingPolicy | None = None,
    ) -> CandidateScore:
        policy = policy or RankingPolicy()

        candidate = evaluation.candidate
        metrics = self._metrics_from_candidate(evaluation)

        walk_forward = evaluation.walk_forward_result
        robustness_decision: RankingDecision = "FAIL"
        if walk_forward is not None:
            robustness_decision = str(walk_forward.robustness_decision).upper()  # type: ignore[arg-type]
        elif evaluation.evaluation is not None:
            robustness_decision = str(evaluation.evaluation.decision).upper()

        evaluation_decision: RankingDecision = "FAIL"
        if evaluation.evaluation is not None:
            evaluation_decision = str(evaluation.evaluation.decision).upper()

        total_return_pct = metrics["total_return_pct"]
        max_drawdown_pct = metrics["max_drawdown_pct"]
        trade_count = int(metrics["total_trades"])
        win_rate_pct = metrics["win_rate_pct"]
        profit_factor = metrics["profit_factor"]
        sharpe_ratio = metrics["sharpe_ratio"]
        sortino_ratio = metrics["sortino_ratio"]

        reasons: list[str] = []
        if walk_forward is not None and walk_forward.reasons:
            reasons.extend(walk_forward.reasons)
        if evaluation.evaluation is not None and evaluation.evaluation.reasons:
            reasons.extend(evaluation.evaluation.reasons)

        if trade_count < policy.minimum_sample_trades:
            reasons.append(
                f"Trade count {trade_count} is below the minimum_sample_trades threshold of {policy.minimum_sample_trades}."
            )
        if total_return_pct < policy.minimum_total_return_pct:
            reasons.append(
                f"Total return {total_return_pct:.2f}% is below the minimum_total_return_pct threshold of {policy.minimum_total_return_pct:.2f}%."
            )
        if max_drawdown_pct > policy.maximum_drawdown_pct:
            reasons.append(
                f"Max drawdown {max_drawdown_pct:.2f}% exceeds the maximum_drawdown_pct threshold of {policy.maximum_drawdown_pct:.2f}%."
            )
        if win_rate_pct < policy.minimum_win_rate_pct:
            reasons.append(
                f"Win rate {win_rate_pct:.2f}% is below the minimum_win_rate_pct threshold of {policy.minimum_win_rate_pct:.2f}%."
            )
        if profit_factor < policy.minimum_profit_factor:
            reasons.append(
                f"Profit factor {profit_factor:.2f} is below the minimum_profit_factor threshold of {policy.minimum_profit_factor:.2f}."
            )
        if sharpe_ratio < policy.minimum_sharpe_ratio:
            reasons.append(
                f"Sharpe ratio {sharpe_ratio:.4f} is below the minimum_sharpe_ratio threshold of {policy.minimum_sharpe_ratio:.4f}."
            )
        if sortino_ratio < policy.minimum_sortino_ratio:
            reasons.append(
                f"Sortino ratio {sortino_ratio:.4f} is below the minimum_sortino_ratio threshold of {policy.minimum_sortino_ratio:.4f}."
            )

        accepted = (
            robustness_decision == "PASS"
            and evaluation_decision == "PASS"
            and trade_count >= policy.minimum_sample_trades
            and total_return_pct >= policy.minimum_total_return_pct
            and max_drawdown_pct <= policy.maximum_drawdown_pct
            and win_rate_pct >= policy.minimum_win_rate_pct
            and profit_factor >= policy.minimum_profit_factor
            and sharpe_ratio >= policy.minimum_sharpe_ratio
            and sortino_ratio >= policy.minimum_sortino_ratio
        )

        robustness_value = self._decision_value(robustness_decision)
        evaluation_value = self._decision_value(evaluation_decision)
        return_value = max(0.0, min(1.0, total_return_pct / 100.0))
        drawdown_value = max(0.0, 1.0 - (max_drawdown_pct / 100.0))
        sample_value = 1.0 if trade_count == 0 else min(1.0, trade_count / max(policy.minimum_sample_trades, 1))
        profit_value = max(0.0, min(1.0, (profit_factor - 1.0) / 2.0 if profit_factor > 0 else 0.0))
        win_rate_value = max(0.0, min(1.0, win_rate_pct / 100.0))
        risk_adj_value = max(0.0, min(1.0, (sharpe_ratio + sortino_ratio) / 4.0))

        weighted_score = (
            policy.robustness_weight * robustness_value
            + policy.evaluation_weight * evaluation_value
            + policy.return_weight * return_value
            + policy.drawdown_weight * drawdown_value
            + policy.sample_weight * sample_value
            + policy.profit_weight * profit_value
            + policy.win_rate_weight * win_rate_value
            + policy.risk_adjusted_weight * risk_adj_value
        )
        score = float(weighted_score * 100.0)

        score_components = {
            "robustness": robustness_value,
            "evaluation": evaluation_value,
            "return_ratio": return_value,
            "drawdown_ratio": drawdown_value,
            "sample_ratio": sample_value,
            "profit_ratio": profit_value,
            "win_rate_ratio": win_rate_value,
            "risk_adjusted_ratio": risk_adj_value,
            "score_components": {
                "robustness": robustness_value,
                "evaluation": evaluation_value,
                "return_ratio": return_value,
                "drawdown_ratio": drawdown_value,
                "sample_ratio": sample_value,
                "profit_ratio": profit_value,
                "win_rate_ratio": win_rate_value,
                "risk_adjusted_ratio": risk_adj_value,
            },
        }

        return CandidateScore(
            candidate_id=candidate.candidate_id,
            candidate_name=candidate.name,
            score=score,
            accepted=accepted,
            robustness_decision=robustness_decision,
            evaluation_decision=evaluation_decision,
            reasons=reasons,
            metrics_used={
                "total_return_pct": total_return_pct,
                "max_drawdown_pct": max_drawdown_pct,
                "total_trades": trade_count,
                "win_rate_pct": win_rate_pct,
                "profit_factor": profit_factor,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "score_components": score_components,
            },
        )

    def rank_candidates(
        self,
        candidates: list[CandidateEvaluation],
        *,
        policy: RankingPolicy | None = None,
    ) -> RankingResult:
        if not candidates:
            raise ValueError("candidate set is empty")

        policy = policy or RankingPolicy()
        seen: set[str] = set()
        for item in candidates:
            candidate_id = item.candidate.candidate_id
            if candidate_id in seen:
                raise ValueError(f"duplicate candidate_id encountered: {candidate_id}")
            seen.add(candidate_id)

        ranked = [self.score_candidate(item, policy=policy) for item in candidates]
        ranked.sort(key=lambda item: (-item.score, item.candidate_id))
        return RankingResult(ranked_candidates=ranked)
