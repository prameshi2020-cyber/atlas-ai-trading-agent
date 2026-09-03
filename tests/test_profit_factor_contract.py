from __future__ import annotations

import math

import pandas as pd
import pytest

import app.agents.candidate_selector as candidate_selector
import app.agents.claude_selection_provider as claude_selection_provider
from app.agents.candidate_selector import UNBOUNDED_ABOVE_METRICS, CandidateSelector, SelectionRequest
from app.agents.claude_selection_provider import UNBOUNDED_METRIC_MARKER, build_selection_payload, render_user_prompt
from app.analysis.backtest import BacktestEngine, BacktestResult, BacktestTrade
from app.analysis.evaluation import StrategyEvaluationPolicy, StrategyEvaluator
from app.analysis.indicators import REQUIRED_FEATURE_COLUMNS
from app.analysis.ranking import Candidate, CandidateEvaluation, RankingEngine, RankingPolicy
from app.analysis.strategy import Signal
from app.analysis.walk_forward import WalkForwardResult

TS = pd.date_range("2026-01-01", periods=120, freq="1min", tz="UTC")


def _row(ts, price: float) -> dict:
    row = {"timestamp": ts, "open": price, "high": price + 0.8, "low": price - 0.8, "close": price + 0.2, "volume": 10.0, "atr_14": 1.0}
    for col in REQUIRED_FEATURE_COLUMNS:
        row.setdefault(col, 50.0 if col == "rsi_14" else 0.0)
    return row


def _uptrend(n: int = 120) -> pd.DataFrame:
    return pd.DataFrame([_row(TS[i], 100.0 + i * 0.15) for i in range(n)])


class ScriptedStrategy:
    def __init__(self, actions: list[str]) -> None:
        self._actions = actions
        self._calls = 0

    def generate_signal(self, symbol: str, features: pd.DataFrame) -> Signal:
        self._calls += 1
        action = self._actions[min(self._calls - 1, len(self._actions) - 1)]
        strength = 0.0 if action == "HOLD" else 1.0
        return Signal(symbol=symbol, action=action, score=strength, confidence=strength, reasons=[action])


def _result(pnl_pcts: list[float]) -> BacktestResult:
    trades = [
        BacktestTrade(symbol="BTCUSDT", side="LONG", entry_time=None, exit_time=None, entry_price=100.0,
                      exit_price=100.0 + p, pnl_pct=p, pnl_amount=p * 100.0, exit_reason="TEST")
        for p in pnl_pcts
    ]
    wins = [p for p in pnl_pcts if p > 0]
    losses = [p for p in pnl_pcts if p <= 0]
    gross_loss = abs(sum(losses))
    total = len(pnl_pcts)
    return BacktestResult(
        symbol="BTCUSDT",
        initial_capital=10_000.0,
        final_capital=10_000.0 + sum(t.pnl_amount for t in trades),
        total_return_pct=sum(pnl_pcts),
        total_trades=total,
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate_pct=(len(wins) / total * 100.0) if total else 0.0,
        max_drawdown_pct=0.0,
        profit_factor=(sum(wins) / gross_loss) if gross_loss > 0 else float("inf"),
        expectancy_pct=(sum(pnl_pcts) / total) if total else 0.0,
        average_trade_pct=(sum(pnl_pcts) / total) if total else 0.0,
        average_win_pct=(sum(wins) / len(wins)) if wins else 0.0,
        average_loss_pct=(sum(losses) / len(losses)) if losses else 0.0,
        largest_win_pct=max(wins) if wins else 0.0,
        largest_loss_pct=min(losses) if losses else 0.0,
        long_trades=total,
        long_wins=len(wins),
        short_trades=0,
        short_wins=0,
        trades=trades,
    )


def _wf_pass() -> WalkForwardResult:
    return WalkForwardResult(windows=[], total_windows=1, passed_windows=1, failed_windows=0, oos_pass_rate_pct=100.0,
                             aggregate_oos_return_pct=5.0, aggregate_oos_pnl=500.0, robustness_decision="PASS", reasons=["PASS"])


def _candidate_evaluation(candidate_id: str, result: BacktestResult) -> CandidateEvaluation:
    return CandidateEvaluation(
        candidate=Candidate(candidate_id=candidate_id, name=candidate_id.upper(), strategy=object()),
        backtest_result=result,
        evaluation=StrategyEvaluator().evaluate(result),
        walk_forward_result=_wf_pass(),
    )


# --- producers -------------------------------------------------------------


def test_backtest_with_no_losing_trades_reports_infinite_profit_factor():
    result = BacktestEngine(strategy=ScriptedStrategy(["BUY", "HOLD"])).run(symbol="BTCUSDT", features=_uptrend())

    assert result.total_trades >= 1
    assert result.losing_trades == 0
    assert result.winning_trades == result.total_trades
    assert isinstance(result.profit_factor, float)
    assert result.profit_factor == math.inf


def test_backtest_with_losses_reports_finite_profit_factor():
    result = BacktestEngine(strategy=ScriptedStrategy(["SELL", "BUY", "HOLD"])).run(symbol="BTCUSDT", features=_uptrend())

    assert result.losing_trades >= 1
    assert math.isfinite(result.profit_factor)
    assert result.profit_factor >= 0.0


def test_evaluator_preserves_infinite_profit_factor_and_passes_thresholds():
    evaluation = StrategyEvaluator().evaluate(_result([2.0, 3.0]), policy=StrategyEvaluationPolicy(minimum_profit_factor=3.0))

    assert evaluation.metrics["profit_factor"] == math.inf
    assert "profit_factor" not in evaluation.failed_criteria
    assert evaluation.decision == "PASS"


def test_evaluator_zero_trade_profit_factor_semantics_preserved():
    evaluation = StrategyEvaluator().evaluate(_result([]))

    assert evaluation.metrics["profit_factor"] == 0.0
    assert evaluation.decision == "FAIL"
    assert evaluation.failed_criteria == ["no_trades"]


# --- ranking ---------------------------------------------------------------


def test_ranking_scores_zero_loss_candidate_deterministically():
    engine = RankingEngine()
    evaluation = _candidate_evaluation("perfect", _result([2.0, 3.0]))

    first = engine.score_candidate(evaluation, policy=RankingPolicy(minimum_profit_factor=5.0))
    second = engine.score_candidate(evaluation, policy=RankingPolicy(minimum_profit_factor=5.0))

    assert first.accepted is True
    assert math.isfinite(first.score)
    assert first.metrics_used["profit_factor"] == math.inf
    assert first.metrics_used["score_components"]["profit_ratio"] == 1.0
    assert first == second


def test_ranking_profit_ratio_saturates_identically_for_inf_and_finite_pf():
    engine = RankingEngine()
    unbounded = engine.score_candidate(_candidate_evaluation("inf", _result([2.0, 3.0])))
    saturated = engine.score_candidate(_candidate_evaluation("three", _result([3.0, -1.0])))

    assert saturated.metrics_used["profit_factor"] == pytest.approx(3.0)
    assert unbounded.metrics_used["score_components"]["profit_ratio"] == saturated.metrics_used["score_components"]["profit_ratio"] == 1.0


# --- selector ----------------------------------------------------------------


def test_selector_accepts_ranked_zero_loss_candidate_end_to_end():
    ranking = RankingEngine().rank_candidates(
        [_candidate_evaluation("perfect", _result([2.0, 3.0])), _candidate_evaluation("ordinary", _result([3.0, -1.0]))]
    )

    decision = CandidateSelector(provider=None).select(SelectionRequest(ranking_result=ranking))

    assert decision.eligible_candidate_count == 2
    assert decision.selected_candidate_id == ranking.best_candidate.candidate_id
    assert {c.candidate_id for c in decision.eligible_candidates} == {"perfect", "ordinary"}


# --- claude payload ----------------------------------------------------------


def test_claude_payload_marks_unbounded_profit_factor_explicitly():
    ranking = RankingEngine().rank_candidates([_candidate_evaluation("perfect", _result([2.0, 3.0]))])
    payload = build_selection_payload(ranking.ranked_candidates)
    entry = payload["candidates"][0]

    assert entry["profit_factor"] == UNBOUNDED_METRIC_MARKER
    assert math.isfinite(entry["total_return_pct"])
    rendered = render_user_prompt(payload)
    assert UNBOUNDED_METRIC_MARKER in rendered
    assert "Infinity" not in rendered and "inf" not in rendered.replace(UNBOUNDED_METRIC_MARKER, "")


def test_claude_payload_still_omits_invalid_non_finite_values():
    score = RankingEngine().score_candidate(_candidate_evaluation("x", _result([2.0, -1.0])))
    score.metrics_used["profit_factor"] = float("-inf")
    score.metrics_used["sharpe_ratio"] = float("nan")
    score.metrics_used["win_rate_pct"] = float("inf")

    entry = build_selection_payload([score])["candidates"][0]

    assert "profit_factor" not in entry
    assert "sharpe_ratio" not in entry
    assert "win_rate_pct" not in entry


def test_unbounded_metric_registry_is_single_source_of_truth():
    assert UNBOUNDED_ABOVE_METRICS == frozenset({"profit_factor"})
    assert claude_selection_provider.UNBOUNDED_ABOVE_METRICS is candidate_selector.UNBOUNDED_ABOVE_METRICS