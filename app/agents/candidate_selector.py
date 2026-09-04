from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Literal, Protocol

from app.analysis.ranking import CandidateScore, RankingResult
from app.agents.errors import InvalidMetricError, MalformedCandidateError, ProviderViolationError

SelectionSource = Literal["provider", "fallback", "none"]

_FORBIDDEN_METADATA_KEYS = {
    "risk",
    "risk_config",
    "capital",
    "position_size",
    "position_sizing",
    "leverage",
    "stop_loss",
    "take_profit",
    "execution",
    "broker",
    "order",
    "trade",
    "trade_instruction",
    "action",
}

# Metrics whose domain legitimately includes +inf.
#
# profit_factor = gross_profit / gross_loss. A strategy with positive gross profit and
# no losing trades has an unbounded profit factor. BacktestEngine, StrategyEvaluator and
# AnalyticsEngine all represent that state as float("inf") and RankingEngine clamps it
# deterministically when scoring, so +inf is a valid value for this metric and must not
# be treated as malformed data. NaN and -inf remain invalid for every metric, and +inf
# remains invalid for every metric not declared here.
UNBOUNDED_ABOVE_METRICS: frozenset[str] = frozenset({"profit_factor"})

@dataclass(frozen=True, slots=True)
class ProviderSelection:
    """Provider output is intentionally limited to a candidate id or an explicit no-selection."""

    candidate_id: str | None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.candidate_id is not None:
            candidate_id = str(self.candidate_id).strip()
            if not candidate_id:
                raise ProviderViolationError("empty candidate_id", violation="empty_candidate_id")
            object.__setattr__(self, "candidate_id", candidate_id)

        if self.reason is not None:
            reason = str(self.reason).strip()
            object.__setattr__(self, "reason", reason)

        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

        forbidden = set(self.metadata.keys()) & _FORBIDDEN_METADATA_KEYS
        if forbidden:
            raise ProviderViolationError(
                f"provider metadata contains forbidden keys: {sorted(forbidden)}",
                violation="forbidden_metadata",
                candidate_id=self.candidate_id,
                forbidden_keys=tuple(sorted(forbidden)),
            )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"candidate_id": self.candidate_id, "reason": self.reason}
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


class SelectionProvider(Protocol):
    def select(self, request: "SelectionRequest") -> ProviderSelection:
        ...


@dataclass(frozen=True, slots=True)
class SelectionRequest:
    """Selection input built from a deterministic ranking result and a provider."""

    ranking_result: RankingResult
    provider: SelectionProvider | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.ranking_result, RankingResult):
            raise TypeError("ranking_result must be a RankingResult")
        if self.provider is not None and not callable(getattr(self.provider, "select", None)):
            raise TypeError("provider must implement select(self, request)")
        if self.provider_metadata is None:
            object.__setattr__(self, "provider_metadata", {})


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    """Safe, deterministic selection result returned from the selector."""

    selected_candidate_id: str | None
    selected_candidate_name: str | None
    selection_source: SelectionSource
    reason: str
    eligible_candidate_count: int
    eligible_candidates: tuple[CandidateScore, ...] = field(default_factory=tuple)
    provider_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.selected_candidate_id is not None:
            selected_candidate_id = str(self.selected_candidate_id).strip()
            if not selected_candidate_id:
                raise ValueError("selected_candidate_id cannot be empty")
            object.__setattr__(self, "selected_candidate_id", selected_candidate_id)

        if self.selected_candidate_name is not None:
            selected_candidate_name = str(self.selected_candidate_name).strip()
            if not selected_candidate_name:
                raise ValueError("selected_candidate_name cannot be empty")
            object.__setattr__(self, "selected_candidate_name", selected_candidate_name)

        if self.provider_metadata is None:
            object.__setattr__(self, "provider_metadata", {})


class DeterministicFallbackProvider:
    """Select the highest-ranked eligible candidate using the existing ranking order."""

    def select(self, request: SelectionRequest) -> ProviderSelection:
        eligible = _eligible_candidates_from_request(request)
        if not eligible:
            return ProviderSelection(candidate_id=None, reason="No eligible candidates available.")

        winner = eligible[0]
        return ProviderSelection(
            candidate_id=winner.candidate_id,
            reason="Selected highest-ranked eligible candidate using deterministic ranking order.",
            metadata={
                "candidate_name": winner.candidate_name,
                "score": winner.score,
                "ranking_position": 1,
            },
        )


class CandidateSelector:
    """Domain boundary that filters ranking results before AI/provider selection."""

    def __init__(self, provider: SelectionProvider | None = None) -> None:
        self.provider = provider or DeterministicFallbackProvider()

    def _eligible_candidates(self, ranking_result: RankingResult) -> tuple[CandidateScore, ...]:
        return _eligible_candidates_from_result(ranking_result)

    def select(self, request: SelectionRequest) -> SelectionDecision:
        if request is None:
            raise ValueError("selection request cannot be None")

        if not isinstance(request, SelectionRequest):
            raise TypeError("request must be a SelectionRequest")

        eligible = self._eligible_candidates(request.ranking_result)

        provider = request.provider or self.provider
        filtered_request = SelectionRequest(
            ranking_result=RankingResult(ranked_candidates=list(eligible)),
            provider=None,
            provider_metadata=dict(request.provider_metadata),
        )

        if not eligible:
            return SelectionDecision(
                selected_candidate_id=None,
                selected_candidate_name=None,
                selection_source="none",
                reason="No eligible candidates remain after deterministic gates.",
                eligible_candidate_count=0,
                eligible_candidates=(),
                provider_metadata=dict(request.provider_metadata),
            )

        provider_result = provider.select(filtered_request)
        if provider_result.candidate_id is not None:
            provider_candidate_id = str(provider_result.candidate_id).strip()
            if not provider_candidate_id:
                raise ProviderViolationError("provider returned an empty candidate_id", violation="empty_candidate_id")
            eligible_ids = {candidate.candidate_id for candidate in eligible}
            if provider_candidate_id not in eligible_ids:
                raise ProviderViolationError(
                    f"provider returned ineligible candidate_id {provider_candidate_id!r}; "
                    f"eligible set is {sorted(eligible_ids)}",
                    violation="ineligible_candidate",
                    candidate_id=provider_candidate_id,
                    eligible_ids=tuple(sorted(eligible_ids)),
                )
            selected_score = next(score for score in eligible if score.candidate_id == provider_candidate_id)
            return SelectionDecision(
                selected_candidate_id=selected_score.candidate_id,
                selected_candidate_name=selected_score.candidate_name,
                selection_source="provider",
                reason=provider_result.reason or "Provider selected an eligible candidate.",
                eligible_candidate_count=len(eligible),
                eligible_candidates=eligible,
                provider_metadata=dict(provider_result.metadata),
            )

        return SelectionDecision(
            selected_candidate_id=None,
            selected_candidate_name=None,
            selection_source="none",
            reason=provider_result.reason or "Provider explicitly returned no selection.",
            eligible_candidate_count=len(eligible),
            eligible_candidates=eligible,
            provider_metadata=dict(provider_result.metadata),
        )


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return False

def _is_valid_metric_value(key: str, value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    if math.isfinite(number):
        return True
    return key in UNBOUNDED_ABOVE_METRICS and number == math.inf

def _validate_candidate_score(candidate: CandidateScore) -> None:
    if not isinstance(candidate, CandidateScore):
        raise TypeError("candidate score entries must be CandidateScore instances")
    if not isinstance(candidate.candidate_id, str) or not candidate.candidate_id.strip():
        raise MalformedCandidateError("candidate_id must be a non-empty string", candidate_id=None, field="candidate_id")
    if not isinstance(candidate.candidate_name, str) or not candidate.candidate_name.strip():
        raise MalformedCandidateError(
            "candidate_name must be a non-empty string", candidate_id=candidate.candidate_id, field="candidate_name"
        )
    if not _is_finite_number(candidate.score):
        raise MalformedCandidateError(
            f"candidate score for {candidate.candidate_id!r} must be finite",
            candidate_id=candidate.candidate_id,
            field="score",
        )

    if candidate.accepted is not True:
        return

    if candidate.robustness_decision != "PASS":
        return
    if candidate.evaluation_decision != "PASS":
        return

    for key, value in candidate.metrics_used.items():
        if key == "score_components":
            nested = value
            if not isinstance(nested, dict):
                raise InvalidMetricError(
                    f"score_components for {candidate.candidate_id!r} must be a dict",
                    candidate_id=candidate.candidate_id,
                    metric="score_components",
                    value=nested,
                    allowed="dict",
                )
            for nested_key, nested_value in nested.items():
                if not _is_finite_number(nested_value) and not isinstance(nested_value, dict):
                    raise InvalidMetricError(
                        f"score component {nested_key!r} for {candidate.candidate_id!r} is not finite",
                        candidate_id=candidate.candidate_id,
                        metric=f"score_components.{nested_key}",
                        value=nested_value,
                        allowed="finite",
                    )
            continue
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                if not _is_finite_number(nested_value):
                    raise InvalidMetricError(
                        f"metric {key}.{nested_key} for {candidate.candidate_id!r} must be finite",
                        candidate_id=candidate.candidate_id,
                        metric=f"{key}.{nested_key}",
                        value=nested_value,
                        allowed="finite",
                    )
        elif not _is_valid_metric_value(key, value):
            allowed = "finite or +inf" if key in UNBOUNDED_ABOVE_METRICS else "finite"
            raise InvalidMetricError(
                f"metric {key!r} for {candidate.candidate_id!r} must be {allowed}",
                candidate_id=candidate.candidate_id,
                metric=key,
                value=value,
                allowed=allowed,
            )


def _eligible_candidates_from_result(ranking_result: RankingResult) -> tuple[CandidateScore, ...]:
    ranked = list(ranking_result.ranked_candidates)
    eligible: list[CandidateScore] = []

    for candidate in ranked:
        try:
            _validate_candidate_score(candidate)
        except MalformedCandidateError as exc:
            if exc.field == "score":
                continue
            raise

        if candidate.accepted is not True:
            continue
        if candidate.robustness_decision != "PASS":
            continue
        if candidate.evaluation_decision != "PASS":
            continue
        eligible.append(candidate)

    eligible.sort(key=lambda item: (-float(item.score), item.candidate_id))
    return tuple(eligible)


def _eligible_candidates_from_request(request: SelectionRequest) -> tuple[CandidateScore, ...]:
    return _eligible_candidates_from_result(request.ranking_result)
