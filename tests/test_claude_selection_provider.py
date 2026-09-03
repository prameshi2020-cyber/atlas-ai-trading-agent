from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

import app.agents.claude_selection_provider as claude_selection_provider
import app.agents.llm_client as llm_client
from app.agents.candidate_selector import CandidateSelector, ProviderSelection, SelectionRequest
from app.agents.claude_selection_provider import (
    CANDIDATE_METRIC_KEYS,
    SYSTEM_PROMPT,
    ClaudeResponseError,
    ClaudeSelectionProvider,
    ClaudeSelectionProviderError,
    ClaudeTransportError,
    build_selection_payload,
    parse_selection_response,
)
from app.analysis.ranking import CandidateScore, RankingResult
from app.config.settings import DEFAULT_CLAUDE_MODEL, settings


class MockLLMClient:
    """Test-only LLMClient with scripted responses. Never touches the network."""

    def __init__(self, *responses: object) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, str]] = []

    def complete(self, *, model: str, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({"model": model, "system_prompt": system_prompt, "user_prompt": user_prompt})
        if not self._responses:
            raise AssertionError("MockLLMClient has no scripted responses left")
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response  # type: ignore[return-value]


def _score(
    candidate_id: str,
    *,
    score: float = 70.0,
    accepted: bool = True,
    robustness: str = "PASS",
    evaluation: str = "PASS",
    extra_metrics: dict | None = None,
) -> CandidateScore:
    metrics: dict[str, object] = {
        "total_return_pct": 12.0,
        "max_drawdown_pct": 6.0,
        "total_trades": 20,
        "win_rate_pct": 55.0,
        "profit_factor": 1.6,
        "sharpe_ratio": 0.8,
        "sortino_ratio": 1.1,
        "score_components": {"robustness": 1.0, "evaluation": 1.0},
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    return CandidateScore(
        candidate_id=candidate_id,
        candidate_name=candidate_id.upper(),
        score=score,
        accepted=accepted,
        robustness_decision=robustness,  # type: ignore[arg-type]
        evaluation_decision=evaluation,  # type: ignore[arg-type]
        reasons=["ok"],
        metrics_used=metrics,
    )


def _ranking(*scores: CandidateScore) -> RankingResult:
    return RankingResult(ranked_candidates=list(scores))


def _response(candidate_id: str | None, reason: str = "Best balance of return and drawdown.") -> str:
    return json.dumps({"candidate_id": candidate_id, "reason": reason})


def _user_payload(client: MockLLMClient, call: int = 0) -> dict:
    return json.loads(client.calls[call]["user_prompt"])


# --- happy paths -----------------------------------------------------------


def test_valid_candidate_selection_flows_through_selector():
    client = MockLLMClient(_response("b"))
    selector = CandidateSelector(provider=ClaudeSelectionProvider(client))
    ranking = _ranking(_score("a", score=80.0), _score("b", score=75.0))

    decision = selector.select(SelectionRequest(ranking_result=ranking))

    assert decision.selected_candidate_id == "b"
    assert decision.selected_candidate_name == "B"
    assert decision.selection_source == "provider"
    assert decision.eligible_candidate_count == 2


def test_null_selection_survives_selector_contract():
    client = MockLLMClient(_response(None, "No candidate selected."))
    selector = CandidateSelector(provider=ClaudeSelectionProvider(client))

    decision = selector.select(SelectionRequest(ranking_result=_ranking(_score("a"))))

    assert decision.selected_candidate_id is None
    assert decision.selected_candidate_name is None
    assert decision.selection_source == "none"
    assert decision.reason == "No candidate selected."
    assert decision.eligible_candidate_count == 1


def test_reason_is_parsed_and_surfaced():
    client = MockLLMClient(_response("a", "  Lowest drawdown among peers.  "))
    provider = ClaudeSelectionProvider(client)

    selection = provider.select(SelectionRequest(ranking_result=_ranking(_score("a"))))

    assert isinstance(selection, ProviderSelection)
    assert selection.candidate_id == "a"
    assert selection.reason == "Lowest drawdown among peers."


def test_code_fenced_json_is_tolerated():
    client = MockLLMClient("```json\n" + _response("a") + "\n```")
    provider = ClaudeSelectionProvider(client)

    selection = provider.select(SelectionRequest(ranking_result=_ranking(_score("a"))))

    assert selection.candidate_id == "a"


def test_extra_output_fields_are_ignored_and_never_become_metadata():
    raw = json.dumps({"candidate_id": "a", "reason": "ok", "action": "buy", "leverage": 10, "stop_loss": 1.0})
    client = MockLLMClient(raw)
    selector = CandidateSelector(provider=ClaudeSelectionProvider(client, model="m-test"))

    decision = selector.select(SelectionRequest(ranking_result=_ranking(_score("a"))))

    assert decision.selected_candidate_id == "a"
    assert decision.provider_metadata == {"provider": "claude", "model": "m-test"}


def test_empty_candidate_list_returns_no_selection_without_calling_model():
    client = MockLLMClient()
    provider = ClaudeSelectionProvider(client)

    selection = provider.select(SelectionRequest(ranking_result=_ranking()))

    assert selection.candidate_id is None
    assert client.calls == []


# --- fail-safe parsing -----------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "not json at all",
        "{'candidate_id': 'a'}",
        '{"candidate_id": "a", "reason": "x"',
        "[]",
        '["a"]',
        '"a"',
        "null",
        '{"reason": "missing id"}',
        '{"candidate_id": 1, "reason": "int id"}',
        '{"candidate_id": ["a"], "reason": "list id"}',
        '{"candidate_id": true, "reason": "bool id"}',
        '{"candidate_id": "", "reason": "empty id"}',
        '{"candidate_id": "   ", "reason": "blank id"}',
        '{"candidate_id": "a"}',
        '{"candidate_id": "a", "reason": 42}',
        '{"candidate_id": "a", "reason": null}',
        '{"candidate_id": "a", "reason": ["x"]}',
    ],
)
def test_malformed_output_never_becomes_a_selection(raw: str):
    client = MockLLMClient(raw)
    selector = CandidateSelector(provider=ClaudeSelectionProvider(client))
    ranking = _ranking(_score("a"), _score("b", score=60.0))

    with pytest.raises(ClaudeResponseError):
        selector.select(SelectionRequest(ranking_result=ranking))


def test_non_text_model_output_is_rejected():
    client = MockLLMClient({"candidate_id": "a", "reason": "dict not str"})
    provider = ClaudeSelectionProvider(client)

    with pytest.raises(ClaudeResponseError, match="must be text"):
        provider.select(SelectionRequest(ranking_result=_ranking(_score("a"))))


def test_missing_candidate_id_is_rejected_with_specific_error():
    with pytest.raises(ClaudeResponseError, match="missing 'candidate_id'"):
        parse_selection_response('{"reason": "x"}')


def test_wrong_candidate_id_type_is_rejected_with_specific_error():
    with pytest.raises(ClaudeResponseError, match="'candidate_id' must be a string or null"):
        parse_selection_response('{"candidate_id": 7, "reason": "x"}')


def test_wrong_reason_type_is_rejected_with_specific_error():
    with pytest.raises(ClaudeResponseError, match="'reason' must be a string"):
        parse_selection_response('{"candidate_id": "a", "reason": 7}')


def test_unknown_candidate_id_is_rejected_by_provider():
    client = MockLLMClient(_response("ghost"))
    provider = ClaudeSelectionProvider(client)

    with pytest.raises(ClaudeResponseError, match="unknown candidate_id 'ghost'"):
        provider.select(SelectionRequest(ranking_result=_ranking(_score("a"), _score("b"))))


def test_client_exception_is_wrapped_as_transport_error():
    client = MockLLMClient(llm_client.LLMClientError("connection reset"))
    provider = ClaudeSelectionProvider(client)

    with pytest.raises(ClaudeTransportError, match="connection reset") as excinfo:
        provider.select(SelectionRequest(ranking_result=_ranking(_score("a"))))

    assert isinstance(excinfo.value.__cause__, llm_client.LLMClientError)


def test_provider_errors_are_value_errors_for_uniform_handling():
    assert issubclass(ClaudeSelectionProviderError, ValueError)
    assert issubclass(ClaudeResponseError, ClaudeSelectionProviderError)
    assert issubclass(ClaudeTransportError, ClaudeSelectionProviderError)


def test_invalid_client_is_rejected():
    with pytest.raises(TypeError):
        ClaudeSelectionProvider(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ClaudeSelectionProvider(None)  # type: ignore[arg-type]


# --- prompt context --------------------------------------------------------


def test_provider_receives_expected_candidate_context():
    client = MockLLMClient(_response("a"))
    provider = ClaudeSelectionProvider(client)
    ranking = _ranking(_score("a", score=81.5), _score("b", score=64.0))

    provider.select(SelectionRequest(ranking_result=ranking))

    assert len(client.calls) == 1
    assert client.calls[0]["system_prompt"] == SYSTEM_PROMPT
    payload = _user_payload(client)
    assert payload["candidate_ids"] == ["a", "b"]
    assert [entry["ranking_position"] for entry in payload["candidates"]] == [1, 2]
    first = payload["candidates"][0]
    assert first["candidate_id"] == "a"
    assert first["score"] == pytest.approx(81.5)
    assert first["robustness_decision"] == "PASS"
    assert first["evaluation_decision"] == "PASS"
    for key in CANDIDATE_METRIC_KEYS:
        assert key in first


def test_prompt_contains_only_whitelisted_candidate_fields():
    allowed = {"candidate_id", "ranking_position", "score", "robustness_decision", "evaluation_decision", *CANDIDATE_METRIC_KEYS}
    client = MockLLMClient(_response("a"))
    provider = ClaudeSelectionProvider(client)

    provider.select(SelectionRequest(ranking_result=_ranking(_score("a"))))

    for entry in _user_payload(client)["candidates"]:
        assert set(entry) <= allowed
    assert set(_user_payload(client)) == {"task", "candidate_ids", "candidates"}


def test_provider_does_not_receive_secrets(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env-secret-value")
    leaky = _score(
        "a",
        extra_metrics={"api_key": "sk-metric-secret", "credentials": {"token": "abc"}, "score_components": {"x": 1.0}},
    )
    client = MockLLMClient(_response("a"))
    provider = ClaudeSelectionProvider(client)

    provider.select(SelectionRequest(ranking_result=_ranking(leaky)))

    sent = client.calls[0]["system_prompt"] + client.calls[0]["user_prompt"]
    for token in ("sk-env-secret-value", "sk-metric-secret", "api_key", "credentials", "token", "abc", "score_components"):
        assert token not in sent


def _all_keys(node: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            keys.add(str(key))
            keys |= _all_keys(value)
    elif isinstance(node, list):
        for value in node:
            keys |= _all_keys(value)
    return keys


def test_provider_does_not_receive_trading_execution_state():
    leaked_keys = {"open_positions": 3, "account_balance": 10_000.0, "leverage": 5, "stop_loss": 0.02}
    leaky = _score("a", extra_metrics=leaked_keys)
    client = MockLLMClient(_response("a"))
    provider = ClaudeSelectionProvider(client)

    provider.select(SelectionRequest(ranking_result=_ranking(leaky)))

    payload = _user_payload(client)
    assert _all_keys(payload).isdisjoint(leaked_keys)
    assert "10000.0" not in client.calls[0]["user_prompt"]

    sent = (client.calls[0]["system_prompt"] + client.calls[0]["user_prompt"]).lower()
    for token in (
        "open_position",
        "positions",
        "balance",
        "broker",
        "leverage",
        "stop_loss",
        "take_profit",
        "order",
        "capital",
        "journal",
    ):
        assert token not in sent
    assert "A" not in payload["candidate_ids"]  # candidate_name is never forwarded

def test_candidate_name_is_not_forwarded():
    client = MockLLMClient(_response("alpha"))
    provider = ClaudeSelectionProvider(client)

    provider.select(SelectionRequest(ranking_result=_ranking(_score("alpha"))))

    assert "ALPHA" not in client.calls[0]["user_prompt"]


def test_non_finite_metrics_are_omitted_from_prompt_not_sent_as_nan():
    candidate = _score("a", extra_metrics={"sharpe_ratio": float("nan"), "sortino_ratio": float("inf")})
    payload = build_selection_payload([candidate])

    assert "sharpe_ratio" not in payload["candidates"][0]
    assert "sortino_ratio" not in payload["candidates"][0]
    assert "total_return_pct" in payload["candidates"][0]


# --- model identifier ------------------------------------------------------


def test_model_identifier_is_configured_once_and_defaults_from_settings():
    assert DEFAULT_CLAUDE_MODEL == "claude-fable-5-1"
    assert isinstance(settings.claude_model, str) and settings.claude_model.strip()

    client = MockLLMClient(_response("a"))
    provider = ClaudeSelectionProvider(client)
    provider.select(SelectionRequest(ranking_result=_ranking(_score("a"))))

    assert provider.model == settings.claude_model
    assert client.calls[0]["model"] == settings.claude_model


def test_model_identifier_is_overridable_per_provider():
    client = MockLLMClient(_response("a"))
    provider = ClaudeSelectionProvider(client, model="claude-custom-model")

    selection = provider.select(SelectionRequest(ranking_result=_ranking(_score("a"))))

    assert client.calls[0]["model"] == "claude-custom-model"
    assert selection.metadata["model"] == "claude-custom-model"


def test_empty_model_identifier_is_rejected():
    with pytest.raises(ValueError, match="model identifier"):
        ClaudeSelectionProvider(MockLLMClient(), model="   ")


# --- determinism, authority, immutability ----------------------------------


def test_repeated_calls_with_same_injected_response_are_consistent():
    client = MockLLMClient(_response("b", "same"), _response("b", "same"))
    selector = CandidateSelector(provider=ClaudeSelectionProvider(client))

    first = selector.select(SelectionRequest(ranking_result=_ranking(_score("a"), _score("b"))))
    second = selector.select(SelectionRequest(ranking_result=_ranking(_score("a"), _score("b"))))

    assert first.selected_candidate_id == second.selected_candidate_id == "b"
    assert first.reason == second.reason == "same"
    assert client.calls[0]["user_prompt"] == client.calls[1]["user_prompt"]


def test_ineligible_candidates_never_reach_the_model():
    client = MockLLMClient(_response("good"))
    selector = CandidateSelector(provider=ClaudeSelectionProvider(client))
    ranking = _ranking(
        _score("review", score=95.0, robustness="REVIEW"),
        _score("failed", score=90.0, accepted=False, evaluation="FAIL"),
        _score("rejected", score=85.0, accepted=False),
        _score("good", score=60.0),
    )

    decision = selector.select(SelectionRequest(ranking_result=ranking))

    assert decision.selected_candidate_id == "good"
    assert _user_payload(client)["candidate_ids"] == ["good"]


def test_selector_remains_final_authority_even_if_provider_check_is_bypassed(monkeypatch: pytest.MonkeyPatch):
    client = MockLLMClient(_response("review"))
    provider = ClaudeSelectionProvider(client)
    monkeypatch.setattr(provider, "_ensure_known_candidate", lambda candidate_id, known_ids: None)
    selector = CandidateSelector(provider=provider)
    ranking = _ranking(_score("review", score=95.0, robustness="REVIEW"), _score("good", score=60.0))

    with pytest.raises(ValueError, match="ineligible candidate_id 'review'"):
        selector.select(SelectionRequest(ranking_result=ranking))


def test_inputs_are_not_mutated():
    client = MockLLMClient(_response("a"))
    selector = CandidateSelector(provider=ClaudeSelectionProvider(client))
    ranking = _ranking(_score("a"), _score("b"))
    snapshot = deepcopy(ranking)
    metrics_ref = ranking.ranked_candidates[0].metrics_used

    selector.select(SelectionRequest(ranking_result=ranking))

    assert ranking == snapshot
    assert ranking.ranked_candidates[0].metrics_used is metrics_ref


# --- import boundary -------------------------------------------------------

_FORBIDDEN_IMPORT_TOKENS = (
    "app.trading",
    "app.risk",
    "app.database",
    "app.market",
    "broker",
    "execution",
    "position_manager",
    "runner",
    "anthropic",
    "openai",
    "requests",
    "httpx",
)


def _import_lines(module) -> list[str]:
    source = Path(module.__file__).read_text(encoding="utf-8")
    return [line.strip() for line in source.splitlines() if line.strip().startswith(("import ", "from "))]


@pytest.mark.parametrize("module", [llm_client, claude_selection_provider])
def test_new_modules_do_not_import_trading_risk_execution_or_vendor_sdks(module):
    for line in _import_lines(module):
        lowered = line.lower()
        for token in _FORBIDDEN_IMPORT_TOKENS:
            assert token not in lowered, f"{module.__name__} has forbidden import: {line}"


def test_llm_client_module_knows_nothing_about_atlas_domain():
    for line in _import_lines(llm_client):
        assert not line.startswith("from app"), line
        assert "import app" not in line, line


def test_no_vendor_sdk_is_loaded_by_unit_tests():
    assert "anthropic" not in sys.modules