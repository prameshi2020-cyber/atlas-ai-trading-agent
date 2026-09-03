from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import math
from pathlib import Path
import re

import pandas as pd
import pytest

import app.analysis.research as research
from app.agents.candidate_selector import ProviderSelection, SelectionRequest
from app.analysis.evaluation import StrategyEvaluationPolicy, StrategyEvaluator
from app.analysis.indicators import REQUIRED_FEATURE_COLUMNS
from app.analysis.ranking import RankingEngine, RankingPolicy, RankingResult
from app.analysis.research import (
    BacktestConfig,
    DatasetDescriptor,
    ResearchBatchResult,
    ResearchCandidate,
    ResearchFailure,
    ResearchPipeline,
    ResearchRequest,
    canonical_json,
    fingerprint_frame,
)
from app.analysis.strategy import Signal
from app.analysis.walk_forward import WalkForwardConfig

HEX = re.compile(r"^[0-9a-f]{64}$")
TS = pd.date_range("2026-01-01", periods=120, freq="1min", tz="UTC")

GOOD_A = "SELL,BUY,HOLD"          # one losing short, one winning long -> finite metrics
GOOD_B = "SELL,SELL,BUY,HOLD"     # larger losing short -> different, lower score
HOLD = "HOLD"                     # never trades
WINS_ONLY = "BUY,HOLD,HOLD"       # no losing trade -> infinite profit factor upstream

WF = WalkForwardConfig(train_bars=30, test_bars=10, step_bars=10)
WF_LENIENT = WalkForwardConfig(train_bars=30, test_bars=10, step_bars=10, minimum_train_trades=0, minimum_test_trades=0)
WINDOWS = 9  # (120 - 40) / 10 + 1


# --- fixtures / helpers ----------------------------------------------------


def make_row(ts, o, h, l, c, atr, v=10.0) -> dict:
    row = {"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": v, "atr_14": atr}
    for col in REQUIRED_FEATURE_COLUMNS:
        row.setdefault(col, 50.0 if col == "rsi_14" else 0.0)
    return row


def build_frame(n: int = 120, base: float = 100.0, step: float = 0.15) -> pd.DataFrame:
    rows = []
    for i in range(n):
        price = base + i * step
        rows.append(make_row(TS[i], price, price + 0.8, price - 0.8, price + 0.2, atr=1.0))
    return pd.DataFrame(rows)


@pytest.fixture
def frame() -> pd.DataFrame:
    return build_frame()


@pytest.fixture
def dataset(frame: pd.DataFrame) -> DatasetDescriptor:
    return DatasetDescriptor.from_frame(frame, source="test", symbol="BTCUSDT", timeframe="1m")


class ScriptedStrategy:
    def __init__(self, actions: list[str]) -> None:
        self._actions = actions
        self.calls = 0

    def generate_signal(self, symbol: str, features: pd.DataFrame) -> Signal:
        self.calls += 1
        action = self._actions[min(self.calls - 1, len(self._actions) - 1)]
        strength = 1.0 if action != "HOLD" else 0.0
        return Signal(symbol=symbol, action=action, score=strength, confidence=strength, reasons=[action])


class ExplodingStrategy:
    def generate_signal(self, symbol: str, features: pd.DataFrame) -> Signal:
        raise RuntimeError("strategy exploded")


def make_factories(registry: list | None = None) -> dict:
    def scripted(params):
        strategy = ScriptedStrategy(str(params["actions"]).split(","))
        if registry is not None:
            registry.append(strategy)
        return strategy

    def exploding(params):
        return ExplodingStrategy()

    def broken(params):
        raise ValueError("factory rejected parameters")

    return {"scripted": scripted, "exploding": exploding, "broken": broken}


def cand(candidate_id: str, actions: str = GOOD_A, *, strategy_id: str = "scripted", **kwargs) -> ResearchCandidate:
    return ResearchCandidate(candidate_id=candidate_id, strategy_id=strategy_id, parameters={"actions": actions}, **kwargs)


def request(dataset: DatasetDescriptor, *candidates: ResearchCandidate, wf: WalkForwardConfig = WF, **kwargs) -> ResearchRequest:
    return ResearchRequest(dataset=dataset, candidates=list(candidates), walk_forward_config=wf, **kwargs)


def pipeline(provider=None, registry: list | None = None, **kwargs) -> ResearchPipeline:
    return ResearchPipeline(strategy_factories=make_factories(registry), selection_provider=provider, **kwargs)


class RecordingProvider:
    def __init__(self, choose: str | None = None) -> None:
        self.requests: list[SelectionRequest] = []
        self.choose = choose

    def select(self, req: SelectionRequest) -> ProviderSelection:
        self.requests.append(req)
        if self.choose == "NONE":
            return ProviderSelection(candidate_id=None, reason="declined")
        if self.choose is not None:
            return ProviderSelection(candidate_id=self.choose, reason="forced")
        ranked = req.ranking_result.ranked_candidates
        if not ranked:
            return ProviderSelection(candidate_id=None, reason="empty")
        return ProviderSelection(candidate_id=ranked[0].candidate_id, reason="first")


class ZeroTradeRaisingEvaluator(StrategyEvaluator):
    def evaluate(self, result, *, policy=None):
        if result.total_trades == 0:
            raise RuntimeError("evaluator refused a zero-trade result")
        return super().evaluate(result, policy=policy)


class BrokenRankingEngine:
    def rank_candidates(self, candidates, *, policy=None):
        raise RuntimeError("ranking exploded")


# --- batch shapes ------------------------------------------------------------


def test_empty_candidate_batch(dataset, frame):
    provider = RecordingProvider()
    result = pipeline(provider).run(request(dataset), frame)

    assert isinstance(result, ResearchBatchResult)
    assert result.total_candidates == 0
    assert result.results == () and result.failures == ()
    assert result.ranking_result is None and result.selection_decision is None
    assert provider.requests == []


def test_single_valid_candidate_runs_end_to_end(dataset, frame):
    result = pipeline().run(request(dataset, cand("a")), frame)

    assert result.successful_candidates == 1 and result.failed_candidates == 0
    item = result.result_for("a")
    assert item is not None
    assert item.backtest_result.total_trades >= 2
    assert item.evaluation.decision == "PASS"
    assert item.walk_forward_result.robustness_decision == "PASS"
    assert item.walk_forward_result.total_windows == WINDOWS
    assert result.ranked_candidate_ids == ("a",)
    assert result.score_for("a").accepted is True
    assert result.selected_candidate_id == "a"
    assert result.selection_decision.selection_source == "provider"


def test_multiple_valid_candidates(dataset, frame):
    result = pipeline().run(request(dataset, cand("a", GOOD_A), cand("b", GOOD_B)), frame)

    assert result.successful_candidates == 2
    assert set(result.ranked_candidate_ids) == {"a", "b"}
    assert result.selected_candidate_id == result.ranking_result.best_candidate.candidate_id
    assert result.score_for("a").score != result.score_for("b").score


def test_all_candidates_failing_yields_no_ranking_and_no_selection(dataset, frame):
    provider = RecordingProvider()
    result = pipeline(provider).run(request(dataset, cand("x", strategy_id="broken"), cand("y", strategy_id="broken")), frame)

    assert result.successful_candidates == 0 and result.failed_candidates == 2
    assert result.ranking_result is None and result.selection_decision is None
    assert provider.requests == []


# --- determinism ---------------------------------------------------------------


def test_results_follow_request_order_and_ranking_is_order_independent(dataset, frame):
    forward = pipeline().run(request(dataset, cand("a", GOOD_A), cand("b", GOOD_B), cand("z", strategy_id="broken")), frame)
    reverse = pipeline().run(request(dataset, cand("z", strategy_id="broken"), cand("b", GOOD_B), cand("a", GOOD_A)), frame)

    assert [item.candidate_id for item in forward.results] == ["a", "b"]
    assert [item.candidate_id for item in reverse.results] == ["b", "a"]
    assert forward.ranked_candidate_ids == reverse.ranked_candidate_ids
    assert forward.selected_candidate_id == reverse.selected_candidate_id


def test_repeated_execution_is_deterministic(dataset, frame):
    req = request(dataset, cand("a", GOOD_A), cand("b", GOOD_B))
    first = pipeline().run(req, frame)
    second = pipeline().run(req, frame.copy())

    assert first.to_dict() == second.to_dict()
    assert canonical_json(first.to_dict()) == canonical_json(second.to_dict())
    assert [item.evaluation_fingerprint for item in first.results] == [item.evaluation_fingerprint for item in second.results]


def test_result_serialization_roundtrips_json(dataset, frame):
    result = pipeline().run(request(dataset, cand("a"), cand("bad", strategy_id="broken")), frame)
    payload = json.loads(canonical_json(result.to_dict()))

    assert set(payload) == {"request", "summary", "results", "failures", "ranking", "selection"}
    assert payload["summary"]["total_candidates"] == 2
    assert payload["summary"]["successful_candidates"] == 1
    assert payload["summary"]["failed_candidates"] == 1
    assert payload["summary"]["selected_candidate_id"] == "a"
    assert payload["failures"][0]["candidate_id"] == "bad"


# --- candidate validation --------------------------------------------------------


def test_invalid_candidate_construction_is_rejected():
    with pytest.raises(ValueError, match="candidate_id"):
        ResearchCandidate(candidate_id="  ", strategy_id="scripted")
    with pytest.raises(ValueError, match="strategy_id"):
        ResearchCandidate(candidate_id="a", strategy_id="")
    with pytest.raises(ValueError, match="finite"):
        ResearchCandidate(candidate_id="a", strategy_id="s", parameters={"x": float("nan")})
    with pytest.raises(ValueError, match="JSON scalar"):
        ResearchCandidate(candidate_id="a", strategy_id="s", parameters={"x": [1, 2]})
    with pytest.raises(ValueError, match="duplicate parameter"):
        ResearchCandidate(candidate_id="a", strategy_id="s", parameters=[("x", 1), ("x", 2)])


def test_invalid_backtest_configuration_is_rejected():
    with pytest.raises(ValueError, match="initial_capital"):
        BacktestConfig(initial_capital=0)
    with pytest.raises(ValueError, match="fee_pct"):
        BacktestConfig(fee_pct=-0.1)
    with pytest.raises(ValueError, match="stop_loss_atr"):
        BacktestConfig(stop_loss_atr=0)
    with pytest.raises(ValueError, match="finite"):
        BacktestConfig(slippage_pct=float("inf"))


def test_duplicate_candidate_ids_are_rejected_explicitly(dataset):
    with pytest.raises(ValueError, match="duplicate candidate_id"):
        request(dataset, cand("dup"), cand("dup"))


def test_request_rejects_preset_strategy_factory_and_insufficient_rows(dataset):
    with pytest.raises(ValueError, match="strategy_factory"):
        request(dataset, cand("a"), wf=WalkForwardConfig(train_bars=30, test_bars=10, step_bars=10, strategy_factory=lambda: None))
    with pytest.raises(ValueError, match="train_bars \\+ test_bars"):
        request(dataset, cand("a"), wf=WalkForwardConfig(train_bars=100, test_bars=30, step_bars=10))


def test_invalid_dates_and_timeframes_are_rejected(frame):
    fp = "0" * 64
    with pytest.raises(ValueError, match="end cannot be earlier"):
        DatasetDescriptor(source="t", symbol="X", timeframe="1m", start="2026-01-02", end="2026-01-01", row_count=10, fingerprint=fp)
    with pytest.raises(ValueError, match="timeframe"):
        DatasetDescriptor(source="t", symbol="X", timeframe="1x", start="2026-01-01", end="2026-01-02", row_count=10, fingerprint=fp)
    reversed_frame = frame.sort_values("timestamp", ascending=False).reset_index(drop=True)
    with pytest.raises(ValueError, match="chronological"):
        DatasetDescriptor.from_frame(reversed_frame, source="t", symbol="X", timeframe="1m")
    broken = frame.copy()
    broken.loc[5, "timestamp"] = None
    with pytest.raises(ValueError, match="invalid timestamps"):
        DatasetDescriptor.from_frame(broken, source="t", symbol="X", timeframe="1m")


def test_unknown_strategy_is_a_validation_failure_not_an_exception(dataset, frame):
    result = pipeline().run(request(dataset, cand("ghost", strategy_id="nope")), frame)

    failure = result.failures_for("ghost")[0]
    assert failure.stage == "validation"
    assert "unknown strategy_id" in failure.message


def test_non_default_backtest_config_is_rejected_until_walk_forward_parity_exists(dataset, frame):
    result = pipeline().run(request(dataset, cand("a"), cand("cheap", backtest=BacktestConfig(fee_pct=0.0))), frame)

    assert result.result_for("a") is not None
    failure = result.failures_for("cheap")[0]
    assert failure.stage == "validation"
    assert "walk-forward" in failure.message


def test_dataset_mismatch_is_rejected_before_any_evaluation(dataset, frame):
    tampered = frame.copy()
    tampered.loc[10, "close"] += 1.0
    with pytest.raises(ValueError, match="does not match"):
        pipeline().run(request(dataset, cand("a")), tampered)


# --- failure isolation ---------------------------------------------------------


def test_malformed_candidate_does_not_destroy_the_batch(dataset, frame):
    result = pipeline().run(request(dataset, cand("a", GOOD_A), cand("b", strategy_id="broken"), cand("c", GOOD_B)), frame)

    assert [item.candidate_id for item in result.results] == ["a", "c"]
    failure = result.failures_for("b")[0]
    assert failure.stage == "validation" and failure.error_type == "ValueError"
    assert "factory rejected parameters" in failure.message
    assert set(result.ranked_candidate_ids) == {"a", "c"}


def test_backtest_failure_is_isolated(dataset, frame):
    result = pipeline().run(request(dataset, cand("boom", strategy_id="exploding"), cand("a")), frame)

    failure = result.failures_for("boom")[0]
    assert failure.stage == "backtest" and failure.error_type == "RuntimeError"
    assert result.ranked_candidate_ids == ("a",)


def test_evaluation_failure_is_isolated(dataset, frame):
    result = pipeline(evaluator=ZeroTradeRaisingEvaluator()).run(request(dataset, cand("hold", HOLD), cand("a")), frame)

    failure = result.failures_for("hold")[0]
    assert failure.stage == "evaluation" and "refused" in failure.message
    assert result.ranked_candidate_ids == ("a",)


def test_walk_forward_failure_is_isolated(dataset, frame):
    result = pipeline().run(request(dataset, cand("hold", HOLD), cand("a")), frame)

    failure = result.failures_for("hold")[0]
    assert failure.stage == "walk_forward"
    assert "minimum_train_trades" in failure.message
    assert result.ranked_candidate_ids == ("a",)


def test_ranking_failure_is_recorded_at_batch_level(dataset, frame):
    provider = RecordingProvider()
    result = pipeline(provider, ranking_engine=BrokenRankingEngine()).run(request(dataset, cand("a")), frame)

    assert result.successful_candidates == 1
    assert result.ranking_result is None and result.selection_decision is None
    (failure,) = result.batch_failures
    assert failure.stage == "ranking" and failure.candidate_id is None
    assert provider.requests == []


def test_failed_candidates_never_enter_ranking_or_selection(dataset, frame):
    provider = RecordingProvider()
    result = pipeline(provider).run(request(dataset, cand("a"), cand("x", strategy_id="broken"), cand("y", strategy_id="exploding")), frame)

    failed = {item.candidate_id for item in result.candidate_failures}
    assert failed == {"x", "y"}
    assert failed.isdisjoint(result.ranked_candidate_ids)
    offered = {score.candidate_id for score in provider.requests[0].ranking_result.ranked_candidates}
    assert failed.isdisjoint(offered)


def test_no_metrics_are_fabricated_for_failed_candidates(dataset, frame):
    result = pipeline().run(request(dataset, cand("x", strategy_id="broken")), frame)

    assert result.result_for("x") is None
    assert result.score_for("x") is None
    assert result.to_dict()["results"] == []


# --- state isolation -------------------------------------------------------------


def test_candidate_results_are_independent_of_batch_composition(dataset, frame):
    alone = pipeline().run(request(dataset, cand("a")), frame)
    together = pipeline().run(request(dataset, cand("a"), cand("b", GOOD_B), cand("bad", strategy_id="broken")), frame)

    assert alone.result_for("a").to_dict() == together.result_for("a").to_dict()
    assert alone.result_for("a").evaluation_fingerprint == together.result_for("a").evaluation_fingerprint
    assert alone.score_for("a").score == pytest.approx(together.score_for("a").score)


def test_fresh_strategy_instance_per_backtest_and_per_window(dataset, frame):
    registry: list = []
    pipeline(registry=registry).run(request(dataset, cand("a")), frame)

    assert len(registry) == 1 + 2 * WINDOWS
    assert len({id(item) for item in registry}) == len(registry)
    assert all(item.calls > 0 for item in registry)


def test_strategy_instances_are_never_shared_between_candidates(dataset, frame):
    registry: list = []
    pipeline(registry=registry).run(request(dataset, cand("a", GOOD_A), cand("b", GOOD_B)), frame)

    assert len(registry) == 2 * (1 + 2 * WINDOWS)
    assert len({id(item) for item in registry}) == len(registry)


def test_inputs_are_not_mutated(dataset, frame):
    snapshot = frame.copy(deep=True)
    a, b = cand("a", GOOD_A), cand("b", GOOD_B)
    before = (a.to_dict(), b.to_dict())
    req = request(dataset, a, b)
    req_fp = req.fingerprint

    pipeline().run(req, frame)

    assert frame.equals(snapshot)
    assert (a.to_dict(), b.to_dict()) == before
    assert req.fingerprint == req_fp


def test_module_has_no_mutable_global_state():
    for name in dir(research):
        if name.startswith("__"):
            continue
        assert not isinstance(getattr(research, name), (list, dict, set)), name


# --- ranking and AI boundary ---------------------------------------------------


def test_ranking_semantics_match_ranking_engine_exactly(dataset, frame):
    req = request(dataset, cand("a", GOOD_A), cand("b", GOOD_B))
    result = pipeline().run(req, frame)

    independent = RankingEngine().rank_candidates([item.candidate_evaluation for item in result.results], policy=req.ranking_policy)
    assert result.ranked_candidate_ids == tuple(score.candidate_id for score in independent.ranked_candidates)
    assert [score.score for score in result.ranked_candidates] == [score.score for score in independent.ranked_candidates]


def test_candidate_selector_receives_a_ranking_result(dataset, frame):
    provider = RecordingProvider()
    pipeline(provider).run(request(dataset, cand("a")), frame)

    (req,) = provider.requests
    assert isinstance(req, SelectionRequest)
    assert isinstance(req.ranking_result, RankingResult)


def test_ai_provider_sees_only_eligible_candidates(dataset, frame):
    provider = RecordingProvider()
    result = pipeline(provider).run(request(dataset, cand("a", GOOD_A), cand("hold", HOLD), cand("b", GOOD_B), wf=WF_LENIENT), frame)

    assert "hold" in result.ranked_candidate_ids
    assert result.score_for("hold").accepted is False
    offered = [score.candidate_id for score in provider.requests[0].ranking_result.ranked_candidates]
    assert "hold" not in offered
    assert all(result.score_for(cid).accepted for cid in offered)
    assert all(score.robustness_decision == "PASS" and score.evaluation_decision == "PASS" for score in provider.requests[0].ranking_result.ranked_candidates)


def test_ai_cannot_bypass_deterministic_acceptance(dataset, frame):
    result = pipeline(RecordingProvider(choose="hold")).run(request(dataset, cand("a"), cand("hold", HOLD), wf=WF_LENIENT), frame)

    assert result.ranking_result is not None
    assert result.selection_decision is None
    (failure,) = result.batch_failures
    assert failure.stage == "selection" and failure.error_type == "ValueError"
    assert "ineligible" in failure.message


def test_ai_unknown_candidate_is_rejected(dataset, frame):
    result = pipeline(RecordingProvider(choose="ghost")).run(request(dataset, cand("a")), frame)

    assert result.selection_decision is None
    assert result.batch_failures[0].stage == "selection"


def test_ai_null_selection_survives_pipeline(dataset, frame):
    result = pipeline(RecordingProvider(choose="NONE")).run(request(dataset, cand("a")), frame)

    assert result.selection_decision is not None
    assert result.selected_candidate_id is None
    assert result.selection_decision.selection_source == "none"
    assert result.selection_decision.eligible_candidate_count == 1


def test_no_candidate_duplication_in_results_or_ranking(dataset, frame):
    result = pipeline().run(request(dataset, cand("a", GOOD_A), cand("b", GOOD_B)), frame)

    ids = [item.candidate_id for item in result.results]
    assert len(ids) == len(set(ids))
    assert len(result.ranked_candidate_ids) == len(set(result.ranked_candidate_ids))


def test_candidate_with_infinite_profit_factor_is_selectable(dataset, frame):
    result = pipeline().run(request(dataset, cand("wins", WINS_ONLY)), frame)

    item = result.result_for("wins")
    assert item.backtest_result.losing_trades == 0
    assert item.backtest_result.profit_factor == math.inf
    assert item.evaluation.metrics["profit_factor"] == math.inf
    assert result.score_for("wins").accepted is True
    assert result.score_for("wins").metrics_used["profit_factor"] == math.inf
    assert result.batch_failures == ()
    assert result.selected_candidate_id == "wins"
    assert result.to_dict()["results"][0]["metrics"]["profit_factor"] == "inf"


# --- provenance / fingerprints ---------------------------------------------------


def test_candidate_fingerprint_depends_on_content_not_label():
    base = cand("a", GOOD_A)
    assert base.fingerprint == cand("renamed", GOOD_A, name="Other").fingerprint
    assert base.fingerprint != cand("a", GOOD_B).fingerprint
    assert base.fingerprint != cand("a", GOOD_A, backtest=BacktestConfig(fee_pct=0.2)).fingerprint
    assert HEX.match(base.fingerprint)


def test_dataset_fingerprint_is_stable_and_content_sensitive(frame):
    same = fingerprint_frame(frame.copy())
    reordered = fingerprint_frame(frame[list(reversed(frame.columns))])
    tampered = frame.copy()
    tampered.loc[3, "close"] += 0.01

    assert fingerprint_frame(frame) == same == reordered
    assert fingerprint_frame(tampered) != same
    assert HEX.match(same)


def test_configuration_fingerprint_reflects_policies(dataset):
    base = request(dataset, cand("a"))
    stricter = request(dataset, cand("a"), ranking_policy=RankingPolicy(minimum_sample_trades=5))
    tighter = request(dataset, cand("a"), evaluation_policy=StrategyEvaluationPolicy(minimum_trade_count=3))

    assert base.configuration_fingerprint != stricter.configuration_fingerprint
    assert base.configuration_fingerprint != tighter.configuration_fingerprint
    assert base.configuration_fingerprint == request(dataset, cand("a")).configuration_fingerprint
    assert base.walk_forward_config.evaluation_policy == base.evaluation_policy


def test_provenance_survives_the_full_pipeline(dataset, frame):
    req = request(dataset, cand("a", GOOD_A), cand("b", GOOD_B))
    result = pipeline().run(req, frame)
    payload = result.to_dict()

    assert result.request is req
    assert payload["request"]["fingerprint"] == req.fingerprint
    assert payload["request"]["dataset"]["fingerprint"] == dataset.fingerprint
    assert payload["request"]["configuration_fingerprint"] == req.configuration_fingerprint
    for item in result.results:
        assert HEX.match(item.evaluation_fingerprint)
    assert result.result_for("a").evaluation_fingerprint != result.result_for("b").evaluation_fingerprint


def test_configuration_snapshot_is_immutable(dataset):
    req = request(dataset, cand("a"))
    with pytest.raises(FrozenInstanceError):
        req.ranking_policy = RankingPolicy()
    with pytest.raises(FrozenInstanceError):
        req.candidates[0].candidate_id = "z"
    with pytest.raises(FrozenInstanceError):
        req.dataset.symbol = "ETHUSDT"
    with pytest.raises(FrozenInstanceError):
        BacktestConfig().fee_pct = 1.0


def test_canonical_json_rejects_non_finite_values():
    with pytest.raises(ValueError, match="serializable"):
        canonical_json({"x": float("nan")})


def test_batch_result_rejects_foreign_candidates(dataset):
    req = request(dataset, cand("a"))
    with pytest.raises(ValueError, match="unknown candidate"):
        ResearchBatchResult(
            request=req,
            results=(),
            failures=(ResearchFailure(candidate_id="stranger", stage="backtest", error_type="E", message="m"),),
            ranking_result=None,
            selection_decision=None,
        )


# --- import boundary -------------------------------------------------------------

_FORBIDDEN = ("app.trading", "app.risk", "app.database", "app.market", "journal", "broker",
              "execution", "position_manager", "runner", "analytics", "anthropic", "openai")


def test_research_module_has_no_execution_risk_broker_or_journal_imports():
    source = Path(research.__file__).read_text(encoding="utf-8")
    imports = [line.strip() for line in source.splitlines() if line.strip().startswith(("import ", "from "))]
    assert imports
    for line in imports:
        for token in _FORBIDDEN:
            assert token not in line.lower(), line