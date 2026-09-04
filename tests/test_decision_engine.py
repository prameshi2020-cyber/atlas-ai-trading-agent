"""Deterministic deployment gate between research output and strategy activation.

Contract under test
- input is a real ResearchBatchResult; nothing is recomputed, only compared
- APPROVE iff every policy check passes
- quality failures (drawdown, trades, walk-forward, gates)      -> REJECT
- operational failures (no candidate, batch failure, stale data,
  symbol/timeframe not allowed)                                 -> HOLD
- REJECT outranks HOLD when both kinds of failure are present
- time is injected; the engine never reads the clock
- output is immutable, JSON-serialisable and carries research provenance
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
import json

import pandas as pd
import pytest

from app.analysis.research import DatasetDescriptor, ResearchBatchResult, ResearchFailure
from app.decision.engine import DecisionEngine
from app.decision.models import DeploymentDecision, PolicyCheck
from app.decision.policy import DeploymentPolicy
from tests.test_research_pipeline import GOOD_A, HOLD, WINS_ONLY, build_frame, cand, pipeline, request


# --- fixtures / helpers ------------------------------------------------------


@pytest.fixture
def frame() -> pd.DataFrame:
    return build_frame()


@pytest.fixture
def dataset(frame: pd.DataFrame) -> DatasetDescriptor:
    return DatasetDescriptor.from_frame(frame, source="test", symbol="BTCUSDT", timeframe="1m")


@pytest.fixture
def selected_batch(dataset, frame) -> ResearchBatchResult:
    batch = pipeline().run(request(dataset, cand("wins", WINS_ONLY)), frame)
    assert batch.selected_candidate_id == "wins"  # precondition, established upstream
    return batch


@pytest.fixture
def losing_batch(dataset, frame) -> ResearchBatchResult:
    batch = pipeline().run(request(dataset, cand("a", GOOD_A)), frame)
    assert batch.selected_candidate_id == "a"
    assert batch.result_for("a").backtest_result.max_drawdown_pct > 0.0
    return batch


@pytest.fixture
def empty_batch(dataset, frame) -> ResearchBatchResult:
    batch = pipeline().run(request(dataset, cand("idle", HOLD)), frame)
    assert batch.selected_candidate_id is None
    return batch


def _utc(ts: str) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def fresh_now(batch: ResearchBatchResult, **delta) -> datetime:
    return (_utc(batch.request.dataset.end) + pd.Timedelta(**(delta or {"minutes": 5}))).to_pydatetime()


def permissive() -> DeploymentPolicy:
    """Every configurable check passes; only the hard PASS-gate checks remain active."""
    return DeploymentPolicy(
        allowed_symbols=None,
        allowed_timeframes=None,
        minimum_walk_forward_windows=0,
        minimum_oos_pass_rate_pct=0.0,
        maximum_drawdown_pct=100.0,
        minimum_total_trades=0,
        maximum_dataset_age=None,
        require_no_batch_failures=False,
    )


def observed(batch: ResearchBatchResult):
    item = batch.result_for(batch.selected_candidate_id)
    return item.backtest_result, item.walk_forward_result


def failed_names(decision: DeploymentDecision) -> list[str]:
    return [c.name for c in decision.checks if not c.passed]


def with_failure(batch: ResearchBatchResult) -> ResearchBatchResult:
    failure = ResearchFailure.from_exception(None, "selection", RuntimeError("boom"))
    return ResearchBatchResult(
        request=batch.request,
        results=batch.results,
        failures=(*batch.failures, failure),
        ranking_result=batch.ranking_result,
        selection_decision=batch.selection_decision,
    )

def with_oos_pass_rate(batch: ResearchBatchResult, rate: float) -> ResearchBatchResult:
    """Copy of the batch whose selected candidate reports a different OOS pass rate.

    Every end-to-end fixture in this suite passes 100% of walk-forward windows on the
    monotonic test frame, so the REJECT side of the pass-rate boundary is unreachable
    through ResearchPipeline alone. The engine only *reads* walk_forward_result, so
    substituting the value exercises exactly the comparison under test; all batch
    invariants and fingerprints are preserved.
    """
    item = batch.result_for(batch.selected_candidate_id)
    wf = replace(item.walk_forward_result, oos_pass_rate_pct=rate)
    evaluation = replace(item.candidate_evaluation, walk_forward_result=wf)
    patched = replace(item, candidate_evaluation=evaluation)
    results = tuple(patched if r.candidate_id == item.candidate_id else r for r in batch.results)
    return replace(batch, results=results)


# --- approve path --------------------------------------------------------------


def test_all_checks_pass_yields_approve_with_provenance(selected_batch):
    now = fresh_now(selected_batch)
    decision = DecisionEngine(permissive()).decide(selected_batch, now=now)

    assert decision.verdict == "APPROVE"
    assert decision.approved is True
    assert decision.candidate_id == "wins"
    assert decision.strategy_id == "scripted"
    assert dict(decision.parameters) == {"actions": WINS_ONLY}
    assert decision.symbol == "BTCUSDT"
    assert decision.timeframe == "1m"
    assert decision.research_fingerprint == selected_batch.request.fingerprint
    assert decision.configuration_fingerprint == selected_batch.request.configuration_fingerprint
    assert decision.dataset_fingerprint == selected_batch.request.dataset.fingerprint
    assert decision.decided_at == now
    assert decision.checks and all(c.passed for c in decision.checks)
    assert all(isinstance(c, PolicyCheck) for c in decision.checks)


def test_decision_is_deterministic_and_immutable(selected_batch):
    engine = DecisionEngine(permissive())
    now = fresh_now(selected_batch)

    first = engine.decide(selected_batch, now=now)
    second = engine.decide(selected_batch, now=now)

    assert first == second
    with pytest.raises(FrozenInstanceError):
        first.verdict = "REJECT"  # type: ignore[misc]


def test_to_dict_is_json_serialisable_and_complete(selected_batch):
    decision = DecisionEngine(permissive()).decide(selected_batch, now=fresh_now(selected_batch))
    payload = decision.to_dict()

    json.dumps(payload, allow_nan=False)
    assert payload["verdict"] == "APPROVE"
    assert payload["candidate_id"] == "wins"
    assert payload["decided_at"] == decision.decided_at.isoformat()
    assert [c["name"] for c in payload["checks"]] == [c.name for c in decision.checks]


def test_check_order_is_stable(selected_batch):
    engine = DecisionEngine(permissive())
    a = engine.decide(selected_batch, now=fresh_now(selected_batch))
    b = engine.decide(selected_batch, now=fresh_now(selected_batch, minutes=6))
    assert [c.name for c in a.checks] == [c.name for c in b.checks]


# --- HOLD: operational ---------------------------------------------------------


def test_no_selected_candidate_holds(empty_batch):
    decision = DecisionEngine(permissive()).decide(empty_batch, now=fresh_now(empty_batch))

    assert decision.verdict == "HOLD"
    assert decision.candidate_id is None
    assert decision.strategy_id is None
    assert decision.parameters == ()
    assert "selected_candidate" in failed_names(decision)
    assert decision.symbol == "BTCUSDT" and decision.timeframe == "1m"


def test_batch_failure_holds_when_required(selected_batch):
    tainted = with_failure(selected_batch)
    policy = replace(permissive(), require_no_batch_failures=True)

    decision = DecisionEngine(policy).decide(tainted, now=fresh_now(tainted))

    assert decision.verdict == "HOLD"
    assert failed_names(decision) == ["batch_failures"]


def test_batch_failure_ignored_when_not_required(selected_batch):
    tainted = with_failure(selected_batch)
    decision = DecisionEngine(permissive()).decide(tainted, now=fresh_now(tainted))
    assert decision.verdict == "APPROVE"


def test_dataset_freshness_boundary(selected_batch):
    policy = replace(permissive(), maximum_dataset_age=timedelta(hours=1))
    engine = DecisionEngine(policy)

    fresh = engine.decide(selected_batch, now=fresh_now(selected_batch, hours=1))
    stale = engine.decide(selected_batch, now=fresh_now(selected_batch, hours=1, seconds=1))

    assert fresh.verdict == "APPROVE"
    assert stale.verdict == "HOLD"
    assert failed_names(stale) == ["dataset_fresh"]


@pytest.mark.parametrize(
    "field, value, check",
    [
        ("allowed_symbols", frozenset({"ETHUSDT"}), "symbol_allowed"),
        ("allowed_timeframes", frozenset({"1h"}), "timeframe_allowed"),
    ],
)
def test_symbol_or_timeframe_not_allowed_holds(selected_batch, field, value, check):
    policy = replace(permissive(), **{field: value})
    decision = DecisionEngine(policy).decide(selected_batch, now=fresh_now(selected_batch))

    assert decision.verdict == "HOLD"
    assert failed_names(decision) == [check]


def test_allowed_symbol_match_is_exact_and_case_sensitive(selected_batch):
    ok = replace(permissive(), allowed_symbols=frozenset({"BTCUSDT", "ETHUSDT"}))
    assert DecisionEngine(ok).decide(selected_batch, now=fresh_now(selected_batch)).verdict == "APPROVE"


# --- REJECT: quality -----------------------------------------------------------


def test_drawdown_boundary_uses_positive_magnitude(losing_batch):
    bt, _ = observed(losing_batch)
    at = replace(permissive(), maximum_drawdown_pct=bt.max_drawdown_pct)
    below = replace(permissive(), maximum_drawdown_pct=bt.max_drawdown_pct - 1e-9)

    assert DecisionEngine(at).decide(losing_batch, now=fresh_now(losing_batch)).verdict == "APPROVE"
    rejected = DecisionEngine(below).decide(losing_batch, now=fresh_now(losing_batch))
    assert rejected.verdict == "REJECT"
    assert failed_names(rejected) == ["max_drawdown"]
    (check,) = [c for c in rejected.checks if c.name == "max_drawdown"]
    assert check.observed == bt.max_drawdown_pct > 0.0


def test_total_trades_boundary(selected_batch):
    bt, _ = observed(selected_batch)
    at = replace(permissive(), minimum_total_trades=bt.total_trades)
    above = replace(permissive(), minimum_total_trades=bt.total_trades + 1)

    assert DecisionEngine(at).decide(selected_batch, now=fresh_now(selected_batch)).verdict == "APPROVE"
    rejected = DecisionEngine(above).decide(selected_batch, now=fresh_now(selected_batch))
    assert rejected.verdict == "REJECT"
    assert failed_names(rejected) == ["total_trades"]


def test_walk_forward_windows_boundary(selected_batch):
    _, wf = observed(selected_batch)
    at = replace(permissive(), minimum_walk_forward_windows=wf.total_windows)
    above = replace(permissive(), minimum_walk_forward_windows=wf.total_windows + 1)

    assert DecisionEngine(at).decide(selected_batch, now=fresh_now(selected_batch)).verdict == "APPROVE"
    rejected = DecisionEngine(above).decide(selected_batch, now=fresh_now(selected_batch))
    assert rejected.verdict == "REJECT"
    assert failed_names(rejected) == ["walk_forward_windows"]


def test_oos_pass_rate_boundary(selected_batch):
    partial = with_oos_pass_rate(selected_batch, 66.0)
    _, wf = observed(partial)
    assert wf.oos_pass_rate_pct == 66.0

    at = replace(permissive(), minimum_oos_pass_rate_pct=66.0)
    above = replace(permissive(), minimum_oos_pass_rate_pct=66.5)

    assert DecisionEngine(at).decide(partial, now=fresh_now(partial)).verdict == "APPROVE"
    rejected = DecisionEngine(above).decide(partial, now=fresh_now(partial))
    assert rejected.verdict == "REJECT"
    assert failed_names(rejected) == ["oos_pass_rate"]
    (check,) = [c for c in rejected.checks if c.name == "oos_pass_rate"]
    assert check.observed == 66.0 and check.threshold == 66.5


def test_full_oos_pass_rate_cannot_be_rejected_by_any_valid_policy(selected_batch):
    _, wf = observed(selected_batch)
    assert wf.oos_pass_rate_pct == 100.0  # established by the WINS_ONLY fixture

    strictest = replace(permissive(), minimum_oos_pass_rate_pct=100.0)
    assert DecisionEngine(strictest).decide(selected_batch, now=fresh_now(selected_batch)).verdict == "APPROVE"
    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        replace(permissive(), minimum_oos_pass_rate_pct=100.0000001)


def test_reject_outranks_hold(selected_batch):
    bt, _ = observed(selected_batch)
    policy = replace(
        permissive(),
        minimum_total_trades=bt.total_trades + 1,  # quality  -> REJECT
        allowed_symbols=frozenset({"ETHUSDT"}),  # operational -> HOLD
    )
    decision = DecisionEngine(policy).decide(selected_batch, now=fresh_now(selected_batch))

    assert decision.verdict == "REJECT"
    assert set(failed_names(decision)) == {"symbol_allowed", "total_trades"}
    assert len(decision.reasons) == 2


# --- input contracts -----------------------------------------------------------


def test_now_must_be_timezone_aware(selected_batch):
    with pytest.raises(ValueError, match="timezone"):
        DecisionEngine(permissive()).decide(selected_batch, now=datetime(2026, 1, 1, 3, 0))


def test_engine_rejects_non_batch_input():
    with pytest.raises(TypeError):
        DecisionEngine(permissive()).decide(object(), now=datetime.now(timezone.utc))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum_walk_forward_windows": -1},
        {"minimum_oos_pass_rate_pct": -0.1},
        {"maximum_drawdown_pct": -5.0},
        {"minimum_total_trades": -1},
        {"maximum_dataset_age": timedelta(0)},
        {"allowed_symbols": frozenset()},
        {"maximum_drawdown_pct": float("nan")},
    ],
)
def test_policy_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        DeploymentPolicy(**kwargs)


def test_default_policy_is_conservative():
    policy = DeploymentPolicy()
    assert policy.minimum_walk_forward_windows >= 3
    assert policy.minimum_total_trades >= 10
    assert 0.0 < policy.maximum_drawdown_pct <= 30.0
    assert policy.require_no_batch_failures is True
    assert policy.maximum_dataset_age is not None