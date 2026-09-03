from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
from typing import Any

from app.agents.candidate_selector import ProviderSelection, SelectionRequest
from app.agents.llm_client import LLMClient
from app.analysis.ranking import CandidateScore
from app.config.settings import settings

PROVIDER_NAME = "claude"

# Only these already-computed numeric metrics are ever forwarded to the model.
CANDIDATE_METRIC_KEYS: tuple[str, ...] = (
    "total_return_pct",
    "max_drawdown_pct",
    "total_trades",
    "win_rate_pct",
    "profit_factor",
    "sharpe_ratio",
    "sortino_ratio",
)

MAX_REASON_LENGTH = 500

SYSTEM_PROMPT = (
    "You are a strategy-candidate selection assistant for ATLAS.\n"
    "You will receive a JSON document listing pre-approved strategy candidates. "
    "Every candidate has already passed deterministic robustness and evaluation gates. "
    "You cannot add, remove, or modify candidates, scores, or metrics.\n"
    "Your only task is to choose at most one candidate_id from the provided candidate_ids, "
    "or to choose none.\n"
    "Respond with exactly one JSON object and nothing else, in this shape:\n"
    '{"candidate_id": "<one of the provided candidate_ids>" or null, "reason": "<one or two sentences>"}\n'
    "Rules:\n"
    "- Output only the JSON object. No markdown, no prose outside the object, no extra keys.\n"
    "- candidate_id must be copied exactly from candidate_ids, or be null.\n"
    "- Selecting null is acceptable when no candidate is clearly preferable.\n"
    "- Your answer is advisory. It is validated by deterministic code and has no authority "
    "over anything other than this single choice."
)


class ClaudeSelectionProviderError(ValueError):
    """The provider could not produce a trustworthy selection."""


class ClaudeTransportError(ClaudeSelectionProviderError):
    """The injected LLM client failed before a response was obtained."""


class ClaudeResponseError(ClaudeSelectionProviderError):
    """The model output was empty, malformed, or structurally invalid."""


@dataclass(frozen=True, slots=True)
class ParsedSelectionResponse:
    """Validated shape of the model output. Nothing else from the output is retained."""

    candidate_id: str | None
    reason: str


def _finite_number(value: object) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return None


def build_selection_payload(candidates: Sequence[CandidateScore]) -> dict[str, Any]:
    """Build the minimal, read-only candidate context sent to the model."""
    entries: list[dict[str, Any]] = []
    for position, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, CandidateScore):
            raise TypeError("candidates must be CandidateScore instances")

        candidate_id = str(candidate.candidate_id).strip()
        if not candidate_id:
            raise ValueError("candidate_id cannot be empty")

        score = _finite_number(candidate.score)
        if score is None:
            raise ValueError(f"score for candidate {candidate_id!r} must be finite")

        entry: dict[str, Any] = {
            "candidate_id": candidate_id,
            "ranking_position": position,
            "score": float(score),
            "robustness_decision": str(candidate.robustness_decision),
            "evaluation_decision": str(candidate.evaluation_decision),
        }

        metrics = candidate.metrics_used if isinstance(candidate.metrics_used, dict) else {}
        for key in CANDIDATE_METRIC_KEYS:
            value = _finite_number(metrics.get(key))
            if value is not None:
                entry[key] = value

        entries.append(entry)

    return {
        "task": (
            "Choose at most one candidate_id from candidate_ids, or null if none should be chosen. "
            'Reply with a single JSON object: {"candidate_id": <string or null>, "reason": <string>}.'
        ),
        "candidate_ids": [entry["candidate_id"] for entry in entries],
        "candidates": entries,
    }


def render_user_prompt(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) < 2 or not lines[-1].strip().startswith("```"):
        return text
    return "\n".join(lines[1:-1]).strip()


def parse_selection_response(raw: object) -> ParsedSelectionResponse:
    """Parse untrusted model output into the strict selection schema."""
    if not isinstance(raw, str):
        raise ClaudeResponseError(f"model output must be text, got {type(raw).__name__}")

    text = _strip_code_fence(raw.strip())
    if not text:
        raise ClaudeResponseError("model output was empty")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ClaudeResponseError(f"model output is not valid JSON: {exc.msg}") from exc

    if not isinstance(data, dict):
        raise ClaudeResponseError("model output must be a JSON object")

    if "candidate_id" not in data:
        raise ClaudeResponseError("model output is missing 'candidate_id'")
    candidate_id = data["candidate_id"]
    if candidate_id is not None:
        if not isinstance(candidate_id, str):
            raise ClaudeResponseError(
                f"model output 'candidate_id' must be a string or null, got {type(candidate_id).__name__}"
            )
        candidate_id = candidate_id.strip()
        if not candidate_id:
            raise ClaudeResponseError("model output 'candidate_id' is empty")

    if "reason" not in data:
        raise ClaudeResponseError("model output is missing 'reason'")
    reason = data["reason"]
    if not isinstance(reason, str):
        raise ClaudeResponseError(f"model output 'reason' must be a string, got {type(reason).__name__}")
    reason = reason.strip()[:MAX_REASON_LENGTH]

    return ParsedSelectionResponse(candidate_id=candidate_id, reason=reason)


class ClaudeSelectionProvider:
    """
    SelectionProvider adapter that asks a Claude model to pick one eligible candidate.

    The provider receives only the candidates CandidateSelector already deemed
    eligible, forwards a minimal read-only summary through an injected LLMClient,
    and converts the untrusted model output into a ProviderSelection. It never
    ranks, never chooses on its own, and never touches trading state.
    """

    def __init__(self, client: LLMClient, *, model: str | None = None) -> None:
        if client is None or not callable(getattr(client, "complete", None)):
            raise TypeError("client must implement complete(*, model, system_prompt, user_prompt)")

        resolved_model = settings.claude_model if model is None else model
        resolved_model = str(resolved_model).strip()
        if not resolved_model:
            raise ValueError("model identifier cannot be empty")

        self._client = client
        self._model = resolved_model

    @property
    def model(self) -> str:
        return self._model

    def _metadata(self) -> dict[str, Any]:
        return {"provider": PROVIDER_NAME, "model": self._model}

    @staticmethod
    def _ensure_known_candidate(candidate_id: str, known_ids: frozenset[str]) -> None:
        if candidate_id not in known_ids:
            raise ClaudeResponseError(
                f"model selected unknown candidate_id {candidate_id!r}; offered set is {sorted(known_ids)}"
            )

    def select(self, request: SelectionRequest) -> ProviderSelection:
        if not isinstance(request, SelectionRequest):
            raise TypeError("request must be a SelectionRequest")

        candidates = tuple(request.ranking_result.ranked_candidates)
        if not candidates:
            return ProviderSelection(
                candidate_id=None,
                reason="No eligible candidates were provided.",
                metadata=self._metadata(),
            )

        payload = build_selection_payload(candidates)
        user_prompt = render_user_prompt(payload)

        try:
            raw = self._client.complete(
                model=self._model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except Exception as exc:
            raise ClaudeTransportError(f"LLM client failed: {exc}") from exc

        parsed = parse_selection_response(raw)

        if parsed.candidate_id is not None:
            known_ids = frozenset(payload["candidate_ids"])
            self._ensure_known_candidate(parsed.candidate_id, known_ids)

        return ProviderSelection(
            candidate_id=parsed.candidate_id,
            reason=parsed.reason,
            metadata=self._metadata(),
        )