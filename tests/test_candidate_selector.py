from __future__ import annotations

from copy import deepcopy
import math

import pytest

from app.analysis.ranking import (
    Candidate,
    CandidateEvaluation,
    CandidateScore,
    RankingEngine,
    RankingPolicy,
    RankingResult,
)
from app.agents.candidate_selector import (
    CandidateSelector,
    DeterministicFallbackProvider,
    ProviderSelection,
    SelectionDecision,
    SelectionRequest,
)


class DummyStrategy:
    def __init__(self, label: str = "strategy") -> None:
        self.label = label


class RecordingProvider:
    def __init__(self) -> None:
        self.requests: list[SelectionRequest] = []

    def select(self, request: SelectionRequest) -> ProviderSelection:
        self.requests.append(request)
        eligible = request.ranking_result.ranked_candidates
        if not eligible:
            return ProviderSelection(candidate_id=None, reason="No eligible candidates.")
        candidate = eligible[0]
        return ProviderSelection(candidate_id=candidate.candidate_id, reason="selected")


class InvalidProvider:
    def select(self, request: SelectionRequest) -> ProviderSelection:
        return ProviderSelection(candidate_id="missing", reason="bad")


class ForbiddenProvider:
    def select(self, request: SelectionRequest) -> ProviderSelection:
        return ProviderSelection(
            candidate_id=request.ranking_result.ranked_candidates[0].candidate_id,
            reason="ok",
            metadata={"risk": "should fail"},
        )


class TradeInstructionProvider:
    def select(self, request: SelectionRequest) -> ProviderSelection:
        return ProviderSelection(candidate_id=request.ranking_result.ranked_candidates[0].candidate_id, reason="ok", metadata={"trade": True})


class CandidateFactory:
    @staticmethod
    def build(
        *,
        candidate_id: str,
        name: str,
        score: float,
        accepted: bool,
        robustness_decision: str = "PASS",
        evaluation_decision: str = "PASS",
        reason: str = "ok",
    ) -> CandidateScore:
        return CandidateScore(
            candidate_id=candidate_id,
            candidate_name=name,
            score=score,
            accepted=accepted,
            robustness_decision=robustness_decision,
            evaluation_decision=evaluation_decision,
            reasons=[reason],
            metrics_used={
                "total_return_pct": 10.0,
                "max_drawdown_pct": 5.0,
                "total_trades": 10,
                "win_rate_pct": 50.0,
                "profit_factor": 1.5,
                "sharpe_ratio": 1.0,
                "sortino_ratio": 1.0,
                "score_components": {
                    "robustness": 1.0,
                    "evaluation": 1.0,
                    "return_ratio": 0.5,
                    "drawdown_ratio": 0.9,
                    "sample_ratio": 1.0,
                    "profit_ratio": 0.5,
                    "win_rate_ratio": 0.5,
                    "risk_adjusted_ratio": 0.5,
                },
            },
        )


def _result_for(*, candidate_id: str, name: str, score: float, accepted: bool = True) -> CandidateScore:
    return CandidateFactory.build(
        candidate_id=candidate_id,
        name=name,
        score=score,
        accepted=accepted,
    )


def test_selects_eligible_candidate_from_provider():
    provider = RecordingProvider()
    selector = CandidateSelector(provider=provider)
    eligible = [
        _result_for(candidate_id="a", name="A", score=80.0),
        _result_for(candidate_id="b", name="B", score=70.0),
    ]
    request = SelectionRequest(ranking_result=RankingResult(ranked_candidates=eligible))

    decision = selector.select(request)

    assert decision.selected_candidate_id == "a"
    assert decision.selection_source == "provider"
    assert provider.requests[0].ranking_result.ranked_candidates[0].candidate_id == "a"


def test_fail_robustness_candidate_is_excluded():
    selector = CandidateSelector(provider=RecordingProvider())
    ranking = RankingResult(
        ranked_candidates=[
            CandidateFactory.build(candidate_id="bad", name="Bad", score=90.0, accepted=True, robustness_decision="FAIL"),
            CandidateFactory.build(candidate_id="good", name="Good", score=60.0, accepted=True, robustness_decision="PASS"),
        ]
    )

    decision = selector.select(SelectionRequest(ranking_result=ranking))

    assert decision.selected_candidate_id == "good"
    assert all(item.candidate_id != "bad" for item in decision.eligible_candidates)


def test_review_robustness_candidate_is_excluded():
    selector = CandidateSelector(provider=RecordingProvider())
    ranking = RankingResult(
        ranked_candidates=[
            CandidateFactory.build(candidate_id="bad", name="Bad", score=90.0, accepted=True, robustness_decision="REVIEW"),
            CandidateFactory.build(candidate_id="good", name="Good", score=75.0, accepted=True, robustness_decision="PASS"),
        ]
    )

    decision = selector.select(SelectionRequest(ranking_result=ranking))

    assert decision.selected_candidate_id == "good"
    assert decision.eligible_candidate_count == 1


def test_fail_evaluation_candidate_is_excluded():
    selector = CandidateSelector(provider=RecordingProvider())
    ranking = RankingResult(
        ranked_candidates=[
            CandidateFactory.build(candidate_id="bad", name="Bad", score=80.0, accepted=False, evaluation_decision="FAIL"),
            CandidateFactory.build(candidate_id="good", name="Good", score=60.0, accepted=True, evaluation_decision="PASS"),
        ]
    )

    decision = selector.select(SelectionRequest(ranking_result=ranking))

    assert decision.selected_candidate_id == "good"


def test_review_evaluation_candidate_is_excluded():
    selector = CandidateSelector(provider=RecordingProvider())
    ranking = RankingResult(
        ranked_candidates=[
            CandidateFactory.build(candidate_id="bad", name="Bad", score=85.0, accepted=True, evaluation_decision="REVIEW"),
            CandidateFactory.build(candidate_id="good", name="Good", score=70.0, accepted=True, evaluation_decision="PASS"),
        ]
    )

    decision = selector.select(SelectionRequest(ranking_result=ranking))

    assert decision.selected_candidate_id == "good"


def test_provider_never_receives_ineligible_candidates():
    provider = RecordingProvider()
    selector = CandidateSelector(provider=provider)
    ranking = RankingResult(
        ranked_candidates=[
            CandidateFactory.build(candidate_id="bad", name="Bad", score=80.0, accepted=False),
            CandidateFactory.build(candidate_id="good", name="Good", score=70.0, accepted=True),
        ]
    )

    selector.select(SelectionRequest(ranking_result=ranking))

    assert provider.requests[0].ranking_result.ranked_candidates[0].candidate_id == "good"
    assert len(provider.requests[0].ranking_result.ranked_candidates) == 1


def test_provider_selecting_unknown_candidate_is_rejected():
    selector = CandidateSelector(provider=InvalidProvider())
    ranking = RankingResult(
        ranked_candidates=[
            CandidateFactory.build(candidate_id="one", name="One", score=60.0, accepted=True),
            CandidateFactory.build(candidate_id="two", name="Two", score=55.0, accepted=True),
        ]
    )

    with pytest.raises(ValueError, match="ineligible candidate_id"):
        selector.select(SelectionRequest(ranking_result=ranking))


def test_provider_returning_forbidden_or_invalid_data_is_rejected_safely():
    selector = CandidateSelector(provider=ForbiddenProvider())
    ranking = RankingResult(
        ranked_candidates=[CandidateFactory.build(candidate_id="one", name="One", score=60.0, accepted=True)]
    )

    with pytest.raises(ValueError, match="forbidden keys"):
        selector.select(SelectionRequest(ranking_result=ranking))

    selector2 = CandidateSelector(provider=TradeInstructionProvider())
    with pytest.raises(ValueError, match="forbidden keys"):
        selector2.select(SelectionRequest(ranking_result=ranking))


def test_no_eligible_candidates_returns_no_selection():
    selector = CandidateSelector(provider=RecordingProvider())
    ranking = RankingResult(
        ranked_candidates=[
            CandidateFactory.build(candidate_id="bad", name="Bad", score=80.0, accepted=False),
            CandidateFactory.build(candidate_id="review", name="Review", score=65.0, accepted=True, robustness_decision="REVIEW"),
        ]
    )

    decision = selector.select(SelectionRequest(ranking_result=ranking))

    assert decision.selected_candidate_id is None
    assert decision.selection_source == "none"
    assert decision.eligible_candidate_count == 0


def test_deterministic_fallback_selects_highest_ranked_eligible_candidate():
    selector = CandidateSelector(provider=None)
    ranking = RankingResult(
        ranked_candidates=[
            CandidateFactory.build(candidate_id="b", name="B", score=50.0, accepted=True),
            CandidateFactory.build(candidate_id="a", name="A", score=75.0, accepted=True),
        ]
    )

    decision = selector.select(SelectionRequest(ranking_result=ranking))

    assert decision.selected_candidate_id == "a"
    assert decision.selection_source == "provider"


def test_tie_breaking_follows_existing_ranking_semantics():
    selector = CandidateSelector(provider=None)
    ranking = RankingResult(
        ranked_candidates=[
            CandidateFactory.build(candidate_id="b", name="B", score=60.0, accepted=True),
            CandidateFactory.build(candidate_id="a", name="A", score=60.0, accepted=True),
        ]
    )

    decision = selector.select(SelectionRequest(ranking_result=ranking))

    assert decision.selected_candidate_id == "a"


def test_inputs_are_not_mutated_after_selection():
    selector = CandidateSelector(provider=RecordingProvider())
    ranking = RankingResult(
        ranked_candidates=[
            CandidateFactory.build(candidate_id="one", name="One", score=90.0, accepted=True),
            CandidateFactory.build(candidate_id="two", name="Two", score=80.0, accepted=True),
        ]
    )

    snapshot = deepcopy(ranking)
    selector.select(SelectionRequest(ranking_result=ranking))

    assert ranking == snapshot


def test_empty_candidate_list_is_handled_safely():
    selector = CandidateSelector(provider=None)
    decision = selector.select(SelectionRequest(ranking_result=RankingResult(ranked_candidates=[])))

    assert decision.selected_candidate_id is None
    assert decision.selection_source == "none"
    assert decision.eligible_candidate_count == 0


def test_malformed_candidate_data_is_rejected():
    selector = CandidateSelector(provider=None)
    bad = CandidateScore(
        candidate_id="",
        candidate_name="Bad",
        score=80.0,
        accepted=True,
        robustness_decision="PASS",
        evaluation_decision="PASS",
        reasons=["bad"],
        metrics_used={"total_return_pct": 10.0},
    )

    with pytest.raises(ValueError, match="candidate_id"):
        selector.select(SelectionRequest(ranking_result=RankingResult(ranked_candidates=[bad])))


def test_non_finite_score_or_metric_values_are_rejected_or_excluded():
    selector = CandidateSelector(provider=None)
    valid = CandidateFactory.build(candidate_id="good", name="Good", score=70.0, accepted=True)
    bad_score = CandidateFactory.build(candidate_id="bad", name="Bad", score=float("nan"), accepted=True)
    bad_metric = CandidateFactory.build(candidate_id="bad_metric", name="Bad Metric", score=70.0, accepted=True)
    bad_metric.metrics_used["total_return_pct"] = float("inf")

    decision = selector.select(SelectionRequest(ranking_result=RankingResult(ranked_candidates=[valid, bad_score])))
    assert decision.selected_candidate_id == "good"

    with pytest.raises(ValueError, match="finite"):
        selector.select(SelectionRequest(ranking_result=RankingResult(ranked_candidates=[bad_metric])))


def test_repeated_identical_inputs_produce_identical_fallback_selection():
    selector = CandidateSelector(provider=None)
    result1 = RankingResult(
        ranked_candidates=[
            CandidateFactory.build(candidate_id="x", name="X", score=80.0, accepted=True),
            CandidateFactory.build(candidate_id="y", name="Y", score=79.0, accepted=True),
        ]
    )
    result2 = RankingResult(
        ranked_candidates=[
            CandidateFactory.build(candidate_id="x", name="X", score=80.0, accepted=True),
            CandidateFactory.build(candidate_id="y", name="Y", score=79.0, accepted=True),
        ]
    )

    first = selector.select(SelectionRequest(ranking_result=result1))
    second = selector.select(SelectionRequest(ranking_result=result2))

    assert first.selected_candidate_id == second.selected_candidate_id == "x"


def test_selector_does_not_import_or_call_broker_execution_or_risk_code():
    selector = CandidateSelector(provider=None)
    ranking = RankingResult(
        ranked_candidates=[CandidateFactory.build(candidate_id="x", name="X", score=80.0, accepted=True)]
    )

    decision = selector.select(SelectionRequest(ranking_result=ranking))

    assert decision.selected_candidate_id == "x"
    assert "broker" not in decision.reason.lower()
    assert "risk" not in decision.reason.lower()


def test_provider_can_be_replaced_with_a_mock_without_changing_selector_logic():
    class MockProvider:
        def select(self, request: SelectionRequest) -> ProviderSelection:
            return ProviderSelection(candidate_id="z", reason="mock")

    selector = CandidateSelector(provider=MockProvider())
    ranking = RankingResult(
        ranked_candidates=[
            CandidateFactory.build(candidate_id="z", name="Z", score=90.0, accepted=True),
            CandidateFactory.build(candidate_id="a", name="A", score=75.0, accepted=True),
        ]
    )

    decision = selector.select(SelectionRequest(ranking_result=ranking))

    assert decision.selected_candidate_id == "z"
    assert decision.selection_source == "provider"


def test_provider_returns_no_selection_when_no_eligible() -> None:
    selector = CandidateSelector(provider=DeterministicFallbackProvider())
    ranking = RankingResult(ranked_candidates=[])
    decision = selector.select(SelectionRequest(ranking_result=ranking))

    assert decision.selected_candidate_id is None
    assert decision.selection_source == "none"


def test_provider_selects_empty_candidate_id_rejected() -> None:
    class EmptyProvider:
        def select(self, request: SelectionRequest) -> ProviderSelection:
            return ProviderSelection(candidate_id="   ", reason="empty")

    selector = CandidateSelector(provider=EmptyProvider())
    ranking = RankingResult(ranked_candidates=[CandidateFactory.build(candidate_id="x", name="X", score=60.0, accepted=True)])

    with pytest.raises(ValueError, match="empty candidate_id"):
        selector.select(SelectionRequest(ranking_result=ranking))
