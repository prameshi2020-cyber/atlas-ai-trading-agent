from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from typing import Any, Literal

DeploymentVerdict = Literal["APPROVE", "HOLD", "REJECT"]
FailureVerdict = Literal["HOLD", "REJECT"]

_VERDICTS: frozenset[str] = frozenset({"APPROVE", "HOLD", "REJECT"})
_FAILURE_VERDICTS: frozenset[str] = frozenset({"HOLD", "REJECT"})


def _serializable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, dict):
        return {str(k): _serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_serializable(v) for v in value]
    return str(value)


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class PolicyCheck:
    """One deterministic rule evaluated against already-computed research output."""

    name: str
    passed: bool
    failure_verdict: FailureVerdict
    observed: Any = None
    threshold: Any = None
    detail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_text(self.name, "check name"))
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be a bool")
        if self.failure_verdict not in _FAILURE_VERDICTS:
            raise ValueError(f"failure_verdict must be one of {sorted(_FAILURE_VERDICTS)}")
        if not isinstance(self.detail, str):
            raise TypeError("detail must be a string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "failure_verdict": self.failure_verdict,
            "observed": _serializable(self.observed),
            "threshold": _serializable(self.threshold),
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class DeploymentDecision:
    """Immutable, auditable verdict on whether a researched candidate may be activated.

    This is a *strategy-level* decision. It never describes an order, a trade or
    a position; per-candle execution intents remain the responsibility of the
    trading runner behind the risk layer.
    """

    verdict: DeploymentVerdict
    candidate_id: str | None
    strategy_id: str | None
    parameters: tuple[tuple[str, Any], ...]
    symbol: str
    timeframe: str
    reasons: tuple[str, ...]
    checks: tuple[PolicyCheck, ...]
    research_fingerprint: str
    configuration_fingerprint: str
    dataset_fingerprint: str
    decided_at: datetime

    def __post_init__(self) -> None:
        if self.verdict not in _VERDICTS:
            raise ValueError(f"verdict must be one of {sorted(_VERDICTS)}")

        checks = tuple(self.checks)
        if not all(isinstance(c, PolicyCheck) for c in checks):
            raise TypeError("checks must be PolicyCheck instances")
        object.__setattr__(self, "checks", checks)

        object.__setattr__(self, "reasons", tuple(str(r) for r in self.reasons))
        object.__setattr__(self, "parameters", tuple((str(k), v) for k, v in self.parameters))

        if self.candidate_id is not None:
            object.__setattr__(self, "candidate_id", _require_text(self.candidate_id, "candidate_id"))
        if self.strategy_id is not None:
            object.__setattr__(self, "strategy_id", _require_text(self.strategy_id, "strategy_id"))

        object.__setattr__(self, "symbol", _require_text(self.symbol, "symbol"))
        object.__setattr__(self, "timeframe", _require_text(self.timeframe, "timeframe"))
        for label in ("research_fingerprint", "configuration_fingerprint", "dataset_fingerprint"):
            object.__setattr__(self, label, _require_text(getattr(self, label), label))

        if not isinstance(self.decided_at, datetime) or self.decided_at.utcoffset() is None:
            raise ValueError("decided_at must be a timezone-aware datetime")

        all_passed = all(c.passed for c in checks)
        if self.verdict == "APPROVE" and (not all_passed or self.candidate_id is None):
            raise ValueError("APPROVE requires a candidate and every check to pass")
        if self.verdict != "APPROVE" and all_passed:
            raise ValueError(f"{self.verdict} requires at least one failed check")

    @property
    def approved(self) -> bool:
        return self.verdict == "APPROVE"

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "candidate_id": self.candidate_id,
            "strategy_id": self.strategy_id,
            "parameters": _serializable(dict(self.parameters)),
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "reasons": list(self.reasons),
            "checks": [c.to_dict() for c in self.checks],
            "research_fingerprint": self.research_fingerprint,
            "configuration_fingerprint": self.configuration_fingerprint,
            "dataset_fingerprint": self.dataset_fingerprint,
            "decided_at": self.decided_at.isoformat(),
        }