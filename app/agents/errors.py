"""Structured exceptions raised by the deterministic selection layer.

Every class subclasses ValueError, so existing ``pytest.raises(ValueError, match=...)``
call sites keep working. New code should match on *type and fields*, never on message
text. Messages are preserved verbatim from the original raise sites.
"""
from __future__ import annotations

from typing import Any


class SelectionError(ValueError):
    """Base class for deterministic CandidateSelector rejections."""


class MalformedCandidateError(SelectionError):
    """A CandidateScore violates a structural rule (empty id/name, non-finite score, ...)."""

    def __init__(self, message: str, *, candidate_id: str | None, field: str) -> None:
        super().__init__(message)
        self.candidate_id = candidate_id
        self.field = field


class InvalidMetricError(SelectionError):
    """A metric value lies outside its declared domain (NaN, -inf, or +inf where not allowed)."""

    def __init__(self, message: str, *, candidate_id: str, metric: str, value: Any, allowed: str) -> None:
        super().__init__(message)
        self.candidate_id = candidate_id
        self.metric = metric  # dotted for nested values, e.g. "score_components.profit_ratio"
        self.value = value
        self.allowed = allowed  # "finite" | "finite or +inf" | "dict"


class ProviderViolationError(SelectionError):
    """The advisory provider returned something the deterministic gate refuses to accept."""

    def __init__(
        self,
        message: str,
        *,
        violation: str,  # "empty_candidate_id" | "ineligible_candidate" | "forbidden_metadata"
        candidate_id: str | None = None,
        eligible_ids: tuple[str, ...] = (),
        forbidden_keys: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.violation = violation
        self.candidate_id = candidate_id
        self.eligible_ids = eligible_ids
        self.forbidden_keys = forbidden_keys