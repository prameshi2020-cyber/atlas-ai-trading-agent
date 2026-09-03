from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.analysis.backtest import BacktestResult, BacktestTrade
from app.analysis.evaluation import StrategyEvaluation
from app.analysis.walk_forward import WalkForwardResult
from app.analysis.ranking import (
    Candidate,
    CandidateEvaluation,
    RankingEngine,
    RankingPolicy,
)


@dataclass
class DummyStrategy:
    label: str = "strategy"
    mutable: dict | None = None

    def __post_init__(self) -> None:
        if self.mutable is None:
            self.mutable = {"seen": 0}


def _trade(*, pnl_pct: float, side: str = "LONG") -> BacktestTrade:
    return BacktestTrade(
        symbol="BTCUSDT",
        side=side,
        entry_time=None,
        exit_time=None,
        entry_price=100.0,
        exit_price=100.0 + pnl_pct,
        pnl_pct=pnl_pct,
        pnl_amount=pnl_pct,
        exit_reason="TEST",
    )


def _result(
    *,
    total_return_pct: float,
    max_drawdown_pct: float,
    total_trades: int,
    win_rate_pct: float = 50.0,
    profit_factor: float = 1.5,
    sharpe_ratio: float = 1.0,
    sortino_ratio: float = 1.0,
) -> BacktestResult:
    trades = [_trade(pnl_pct=5.0)] * total_trades if total_trades else []
    if total_trades > 0 and total_return_pct < 0:
        trades = [_trade(pnl_pct=total_return_pct / total_trades)] * total_trades

    return BacktestResult(
        symbol="BTCUSDT",
        initial_capital=10_000.0,
        final_capital=10_000.0 * (1 + (total_return_pct / 100.0)),
        total_return_pct=total_return_pct,
        total_trades=total_trades,
        winning_trades=max(1, int(total_trades * win_rate_pct / 100.0)) if total_trades else 0,
        losing_trades=max(0, total_trades - (max(1, int(total_trades * win_rate_pct / 100.0)) if total_trades else 0)),
        win_rate_pct=win_rate_pct,
        max_drawdown_pct=max_drawdown_pct,
        profit_factor=profit_factor,
        expectancy_pct=total_return_pct / max(total_trades, 1),
        average_trade_pct=total_return_pct / max(total_trades, 1),
        average_win_pct=5.0,
        average_loss_pct=-5.0,
        largest_win_pct=10.0,
        largest_loss_pct=-10.0,
        long_trades=total_trades,
        long_wins=max(1, int(total_trades * win_rate_pct / 100.0)) if total_trades else 0,
        short_trades=0,
        short_wins=0,
        trades=trades,
    )


def _evaluation(
    *,
    decision: str = "PASS",
    max_drawdown_pct: float = 10.0,
    total_return_pct: float = 5.0,
    total_trades: int = 10,
) -> StrategyEvaluation:
    return StrategyEvaluation(
        decision=decision,
        passed=decision == "PASS",
        failed_criteria=[] if decision == "PASS" else ["drawdown"],
        reasons=["ok"] if decision == "PASS" else ["bad"],
        metrics={
            "total_return_pct": total_return_pct,
            "net_pnl": total_return_pct,
            "total_trades": total_trades,
            "win_rate_pct": 50.0,
            "profit_factor": 1.5,
            "expectancy": total_return_pct / max(total_trades, 1),
            "max_drawdown_pct": max_drawdown_pct,
            "average_trade_pct": total_return_pct / max(total_trades, 1),
            "sharpe_ratio": 1.0,
            "sortino_ratio": 1.0,
            "average_holding_time_seconds": 300.0,
        },
        metric_definitions={
            "sharpe_ratio": "test",
            "sortino_ratio": "test",
        },
    )


def _wf(decision: str = "PASS") -> WalkForwardResult:
    return WalkForwardResult(
        windows=[],
        total_windows=1,
        passed_windows=1 if decision == "PASS" else 0,
        failed_windows=0 if decision == "PASS" else 1,
        oos_pass_rate_pct=100.0 if decision == "PASS" else 0.0,
        aggregate_oos_return_pct=5.0 if decision == "PASS" else -10.0,
        aggregate_oos_pnl=500.0 if decision == "PASS" else -1000.0,
        robustness_decision=decision,
        reasons=[decision],
    )


def test_one_candidate_ranks_first_when_it_is_consistently_better():
    policy = RankingPolicy()
    winner = Candidate(
        candidate_id="winner",
        name="Winner",
        strategy=DummyStrategy(label="winner"),
    )
    loser = Candidate(
        candidate_id="loser",
        name="Loser",
        strategy=DummyStrategy(label="loser"),
    )

    winner_eval = CandidateEvaluation(
        candidate=winner,
        backtest_result=_result(total_return_pct=20.0, max_drawdown_pct=12.0, total_trades=30),
        evaluation=_evaluation(decision="PASS", total_return_pct=20.0, max_drawdown_pct=12.0, total_trades=30),
        walk_forward_result=_wf("PASS"),
    )
    loser_eval = CandidateEvaluation(
        candidate=loser,
        backtest_result=_result(total_return_pct=10.0, max_drawdown_pct=12.0, total_trades=30),
        evaluation=_evaluation(decision="PASS", total_return_pct=10.0, max_drawdown_pct=12.0, total_trades=30),
        walk_forward_result=_wf("PASS"),
    )

    result = RankingEngine().rank_candidates([winner_eval, loser_eval], policy=policy)

    assert result.best_candidate is not None
    assert result.best_candidate.candidate_id == "winner"
    assert result.ranked_candidates[0].candidate_id == "winner"


def test_raw_return_alone_cannot_override_severe_drawdown():
    policy = RankingPolicy()
    large_gain = Candidate(
        candidate_id="gain",
        name="Gain",
        strategy=DummyStrategy(label="gain"),
    )
    stable = Candidate(
        candidate_id="stable",
        name="Stable",
        strategy=DummyStrategy(label="stable"),
    )

    large_gain_eval = CandidateEvaluation(
        candidate=large_gain,
        backtest_result=_result(total_return_pct=35.0, max_drawdown_pct=50.0, total_trades=20),
        evaluation=_evaluation(decision="PASS", total_return_pct=35.0, max_drawdown_pct=50.0, total_trades=20),
        walk_forward_result=_wf("PASS"),
    )
    stable_eval = CandidateEvaluation(
        candidate=stable,
        backtest_result=_result(total_return_pct=12.0, max_drawdown_pct=8.0, total_trades=18),
        evaluation=_evaluation(decision="PASS", total_return_pct=12.0, max_drawdown_pct=8.0, total_trades=18),
        walk_forward_result=_wf("PASS"),
    )

    result = RankingEngine().rank_candidates([large_gain_eval, stable_eval], policy=policy)

    assert result.best_candidate.candidate_id == "stable"


def test_robustness_fail_cannot_become_accept():
    policy = RankingPolicy()
    failed = Candidate(candidate_id="failed", name="Failed", strategy=DummyStrategy(label="failed"))
    candidate_eval = CandidateEvaluation(
        candidate=failed,
        backtest_result=_result(total_return_pct=50.0, max_drawdown_pct=5.0, total_trades=25),
        evaluation=_evaluation(decision="FAIL", total_return_pct=50.0, max_drawdown_pct=5.0, total_trades=25),
        walk_forward_result=_wf("FAIL"),
    )

    score = RankingEngine().score_candidate(candidate_eval, policy=policy)

    assert score.robustness_decision == "FAIL"
    assert score.accepted is False


def test_robustness_review_remains_review():
    policy = RankingPolicy()
    reviewed = Candidate(candidate_id="review", name="Review", strategy=DummyStrategy(label="review"))
    candidate_eval = CandidateEvaluation(
        candidate=reviewed,
        backtest_result=_result(total_return_pct=18.0, max_drawdown_pct=15.0, total_trades=20),
        evaluation=_evaluation(decision="REVIEW", total_return_pct=18.0, max_drawdown_pct=15.0, total_trades=20),
        walk_forward_result=_wf("REVIEW"),
    )

    score = RankingEngine().score_candidate(candidate_eval, policy=policy)

    assert score.robustness_decision == "REVIEW"
    assert score.accepted is False


def test_trade_sample_size_affects_confidence_deterministically():
    policy = RankingPolicy(minimum_sample_trades=10)
    few = Candidate(candidate_id="few", name="Few", strategy=DummyStrategy(label="few"))
    many = Candidate(candidate_id="many", name="Many", strategy=DummyStrategy(label="many"))

    few_eval = CandidateEvaluation(
        candidate=few,
        backtest_result=_result(total_return_pct=15.0, max_drawdown_pct=10.0, total_trades=4),
        evaluation=_evaluation(decision="PASS", total_return_pct=15.0, max_drawdown_pct=10.0, total_trades=4),
        walk_forward_result=_wf("PASS"),
    )
    many_eval = CandidateEvaluation(
        candidate=many,
        backtest_result=_result(total_return_pct=15.0, max_drawdown_pct=10.0, total_trades=24),
        evaluation=_evaluation(decision="PASS", total_return_pct=15.0, max_drawdown_pct=10.0, total_trades=24),
        walk_forward_result=_wf("PASS"),
    )

    few_score = RankingEngine().score_candidate(few_eval, policy=policy)
    many_score = RankingEngine().score_candidate(many_eval, policy=policy)

    assert many_score.score > few_score.score


def test_tie_handling_is_deterministic():
    policy = RankingPolicy()
    b = Candidate(candidate_id="b", name="B", strategy=DummyStrategy(label="b"))
    a = Candidate(candidate_id="a", name="A", strategy=DummyStrategy(label="a"))

    b_eval = CandidateEvaluation(
        candidate=b,
        backtest_result=_result(total_return_pct=10.0, max_drawdown_pct=12.0, total_trades=10),
        evaluation=_evaluation(decision="PASS", total_return_pct=10.0, max_drawdown_pct=12.0, total_trades=10),
        walk_forward_result=_wf("PASS"),
    )
    a_eval = CandidateEvaluation(
        candidate=a,
        backtest_result=_result(total_return_pct=10.0, max_drawdown_pct=12.0, total_trades=10),
        evaluation=_evaluation(decision="PASS", total_return_pct=10.0, max_drawdown_pct=12.0, total_trades=10),
        walk_forward_result=_wf("PASS"),
    )

    result = RankingEngine().rank_candidates([b_eval, a_eval], policy=policy)

    assert [item.candidate_id for item in result.ranked_candidates] == ["a", "b"]


def test_identical_candidates_produce_identical_rankings():
    policy = RankingPolicy()
    cand_a = Candidate(candidate_id="x", name="X", strategy=DummyStrategy(label="x"))
    cand_b = Candidate(candidate_id="y", name="Y", strategy=DummyStrategy(label="y"))

    first = CandidateEvaluation(
        candidate=cand_a,
        backtest_result=_result(total_return_pct=9.0, max_drawdown_pct=8.0, total_trades=12),
        evaluation=_evaluation(decision="PASS", total_return_pct=9.0, max_drawdown_pct=8.0, total_trades=12),
        walk_forward_result=_wf("PASS"),
    )
    second = CandidateEvaluation(
        candidate=cand_b,
        backtest_result=_result(total_return_pct=9.0, max_drawdown_pct=8.0, total_trades=12),
        evaluation=_evaluation(decision="PASS", total_return_pct=9.0, max_drawdown_pct=8.0, total_trades=12),
        walk_forward_result=_wf("PASS"),
    )

    result = RankingEngine().rank_candidates([first, second], policy=policy)

    assert result.ranked_candidates[0].candidate_id == "x"
    assert result.ranked_candidates[1].candidate_id == "y"


def test_ranking_does_not_mutate_candidate_strategies():
    policy = RankingPolicy()
    strategy = DummyStrategy(label="mutated")
    candidate = Candidate(candidate_id="c1", name="C1", strategy=strategy)
    candidate_eval = CandidateEvaluation(
        candidate=candidate,
        backtest_result=_result(total_return_pct=11.0, max_drawdown_pct=9.0, total_trades=10),
        evaluation=_evaluation(decision="PASS", total_return_pct=11.0, max_drawdown_pct=9.0, total_trades=10),
        walk_forward_result=_wf("PASS"),
    )

    before = strategy.mutable.copy()
    RankingEngine().score_candidate(candidate_eval, policy=policy)

    assert strategy.mutable == before


def test_empty_candidate_set_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        RankingEngine().rank_candidates([], policy=RankingPolicy())


def test_duplicate_candidate_id_is_rejected():
    policy = RankingPolicy()
    candidate_a = Candidate(candidate_id="dup", name="One", strategy=DummyStrategy(label="one"))
    candidate_b = Candidate(candidate_id="dup", name="Two", strategy=DummyStrategy(label="two"))

    eval_a = CandidateEvaluation(
        candidate=candidate_a,
        backtest_result=_result(total_return_pct=8.0, max_drawdown_pct=10.0, total_trades=9),
        evaluation=_evaluation(decision="PASS", total_return_pct=8.0, max_drawdown_pct=10.0, total_trades=9),
        walk_forward_result=_wf("PASS"),
    )
    eval_b = CandidateEvaluation(
        candidate=candidate_b,
        backtest_result=_result(total_return_pct=9.0, max_drawdown_pct=10.0, total_trades=9),
        evaluation=_evaluation(decision="PASS", total_return_pct=9.0, max_drawdown_pct=10.0, total_trades=9),
        walk_forward_result=_wf("PASS"),
    )

    with pytest.raises(ValueError, match="duplicate"):
        RankingEngine().rank_candidates([eval_a, eval_b], policy=policy)


def test_score_components_are_inspectable():
    policy = RankingPolicy()
    candidate = Candidate(candidate_id="inspect", name="Inspect", strategy=DummyStrategy(label="inspect"))
    candidate_eval = CandidateEvaluation(
        candidate=candidate,
        backtest_result=_result(total_return_pct=15.0, max_drawdown_pct=14.0, total_trades=12),
        evaluation=_evaluation(decision="PASS", total_return_pct=15.0, max_drawdown_pct=14.0, total_trades=12),
        walk_forward_result=_wf("PASS"),
    )

    score = RankingEngine().score_candidate(candidate_eval, policy=policy)

    assert "score_components" in score.metrics_used
    assert score.metrics_used["score_components"]


def test_ranking_is_deterministic_across_runs():
    policy = RankingPolicy()
    created = [
        CandidateEvaluation(
            candidate=Candidate(candidate_id=f"c{i}", name=f"C{i}", strategy=DummyStrategy(label=f"c{i}")),
            backtest_result=_result(total_return_pct=10.0 + i, max_drawdown_pct=8.0 + i, total_trades=8 + i),
            evaluation=_evaluation(decision="PASS", total_return_pct=10.0 + i, max_drawdown_pct=8.0 + i, total_trades=8 + i),
            walk_forward_result=_wf("PASS"),
        )
        for i in range(3)
    ]

    first = RankingEngine().rank_candidates(created, policy=policy)
    second = RankingEngine().rank_candidates(list(created), policy=policy)

    assert [item.candidate_id for item in first.ranked_candidates] == [item.candidate_id for item in second.ranked_candidates]
    assert first.best_candidate.candidate_id == second.best_candidate.candidate_id


def test_ranking_uses_existing_metrics_instead_of_recomputing():
    candidate = Candidate(candidate_id="existing", name="Existing", strategy=DummyStrategy(label="existing"))
    evaluation = _evaluation(decision="PASS", total_return_pct=25.0, max_drawdown_pct=8.0, total_trades=15)
    backtest = _result(total_return_pct=25.0, max_drawdown_pct=8.0, total_trades=15)

    candidate_eval = CandidateEvaluation(
        candidate=candidate,
        backtest_result=backtest,
        evaluation=evaluation,
        walk_forward_result=_wf("PASS"),
    )

    score = RankingEngine().score_candidate(candidate_eval, policy=RankingPolicy())

    assert score.metrics_used["total_return_pct"] == pytest.approx(25.0)
    assert score.metrics_used["max_drawdown_pct"] == pytest.approx(8.0)
