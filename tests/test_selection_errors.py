"""Structured selection errors: callers match on type and fields, not message text."""
from __future__ import annotations

import math

import pytest

from dataclasses import replace

from app.agents.candidate_selector import CandidateSelector, ProviderSelection, SelectionRequest
from app.agents.errors import InvalidMetricError, MalformedCandidateError, ProviderViolationError, SelectionError
from app.analysis.ranking import RankingResult
from tests.test_candidate_selector import CandidateFactory


class StubProvider:
    def __init__(self, selection: ProviderSelection) -> None:
        self._selection = selection

    def select(self, request: SelectionRequest) -> ProviderSelection:
        return self._selection


def ok(candidate_id: str = "a", score: float = 80.0):
    return CandidateFactory.build(candidate_id=candidate_id, name=candidate_id.upper(), score=score, accepted=True)


def request(*candidates) -> SelectionRequest:
    return SelectionRequest(ranking_result=RankingResult(ranked_candidates=list(candidates)))


def select(*candidates, provider=None):
    return CandidateSelector(provider=provider).select(request(*candidates))


# --- hierarchy ---------------------------------------------------------------


@pytest.mark.parametrize("cls", [MalformedCandidateError, InvalidMetricError, ProviderViolationError])
def test_every_selection_error_is_a_value_error(cls):
    assert issubclass(cls, SelectionError) and issubclass(cls, ValueError)


# --- metric domain --------------------------------------------------------------


@pytest.mark.parametrize("value", [float("nan"), float("-inf")])
def test_invalid_profit_factor_carries_metric_value_and_allowed_domain(value):
    bad = ok()
    bad.metrics_used["profit_factor"] = value

    with pytest.raises(InvalidMetricError) as info:
        select(bad)

    err = info.value
    assert (err.candidate_id, err.metric, err.allowed) == ("a", "profit_factor", "finite or +inf")
    assert (math.isnan(err.value) if math.isnan(value) else err.value == value)


@pytest.mark.parametrize("metric", ["total_return_pct", "max_drawdown_pct", "win_rate_pct", "sharpe_ratio"])
def test_positive_infinity_on_bounded_metric_reports_finite_domain(metric):
    bad = ok()
    bad.metrics_used[metric] = float("inf")

    with pytest.raises(InvalidMetricError) as info:
        select(bad)

    assert (info.value.candidate_id, info.value.metric, info.value.allowed) == ("a", metric, "finite")
    assert info.value.value == math.inf


def test_nested_score_component_uses_dotted_metric_path():
    bad = ok()
    bad.metrics_used["score_components"] = {"profit_ratio": float("nan")}

    with pytest.raises(InvalidMetricError) as info:
        select(bad)

    assert info.value.metric == "score_components.profit_ratio"
    assert info.value.candidate_id == "a"


def test_non_dict_score_components_reports_dict_domain():
    bad = ok()
    bad.metrics_used["score_components"] = 1.0

    with pytest.raises(InvalidMetricError) as info:
        select(bad)

    assert (info.value.metric, info.value.allowed) == ("score_components", "dict")


# --- candidate structure ----------------------------------------------------------


def test_non_finite_score_is_reported_as_malformed_score_field():
    from app.agents.candidate_selector import _validate_candidate_score

    with pytest.raises(MalformedCandidateError) as info:
        _validate_candidate_score(replace(ok(), score=float("inf")))

    assert (info.value.candidate_id, info.value.field) == ("a", "score")


def test_non_finite_score_is_silently_ineligible_without_message_matching():
    # Contract (see test_candidate_selector): a non-finite score makes the candidate
    # ineligible; it does not abort selection. Exclusion must key off the structured
    # field, not the exception text.
    decision = select(replace(ok("bad"), score=float("nan")), ok("good", 70.0))

    assert decision.selected_candidate_id == "good"
    assert decision.eligible_candidate_count == 1
    assert [c.candidate_id for c in decision.eligible_candidates] == ["good"]


def test_blank_candidate_id_is_malformed_candidate():
    bad = ok()
    bad.candidate_id = "   "

    with pytest.raises(MalformedCandidateError) as info:
        select(bad)

    assert info.value.field == "candidate_id"


# --- provider violations -------------------------------------------------------------


def test_ineligible_provider_choice_exposes_offered_set():
    provider = StubProvider(ProviderSelection(candidate_id="ghost", reason="forced"))

    with pytest.raises(ProviderViolationError) as info:
        select(ok("a"), ok("b", 70.0), provider=provider)

    err = info.value
    assert err.violation == "ineligible_candidate"
    assert err.candidate_id == "ghost"
    assert err.eligible_ids == ("a", "b")


def test_blank_provider_candidate_id_is_a_provider_violation():
    with pytest.raises(ProviderViolationError) as info:
        ProviderSelection(candidate_id="   ", reason="x")

    assert info.value.violation == "empty_candidate_id"


def test_forbidden_provider_metadata_lists_offending_keys():
    with pytest.raises(ProviderViolationError) as info:
        ProviderSelection(candidate_id="a", reason="x", metadata={"action": "BUY", "note": "fine"})

    assert info.value.violation == "forbidden_metadata"
    assert info.value.forbidden_keys == ("action",)


# --- backward compatibility ----------------------------------------------------------


def test_legacy_message_matching_still_works():
    bad = ok()
    bad.metrics_used["sharpe_ratio"] = float("nan")

    with pytest.raises(ValueError, match=r"metric 'sharpe_ratio'.*must be finite"):
        select(bad)

def test_non_finite_score_is_reported_as_malformed_score_field():
    from app.agents.candidate_selector import _validate_candidate_score

    with pytest.raises(MalformedCandidateError) as info:
        _validate_candidate_score(replace(ok(), score=float("inf")))

    assert (info.value.candidate_id, info.value.field) == ("a", "score")


def test_non_finite_score_is_silently_ineligible_without_message_matching():
    # Contract (see test_candidate_selector): a non-finite score makes the candidate
    # ineligible; it does not abort selection. Exclusion must key off the structured
    # field, not the exception text.
    decision = select(replace(ok("bad"), score=float("nan")), ok("good", 70.0))

    assert decision.selected_candidate_id == "good"
    assert decision.eligible_candidate_count == 1
    assert [c.candidate_id for c in decision.eligible_candidates] == ["good"]


def test_blank_candidate_id_is_malformed_candidate():
    bad = replace(ok(), candidate_id="   ")

    with pytest.raises(MalformedCandidateError) as info:
        select(bad)

    assert info.value.field == "candidate_id"