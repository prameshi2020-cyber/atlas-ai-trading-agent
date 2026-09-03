from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
import hashlib
import json
import math
import numbers
import re
from typing import Any, Literal

import pandas as pd

from app.agents.candidate_selector import CandidateSelector, SelectionDecision, SelectionProvider, SelectionRequest
from app.analysis.backtest import BacktestEngine, BacktestResult
from app.analysis.evaluation import StrategyEvaluation, StrategyEvaluationPolicy, StrategyEvaluator
from app.analysis.indicators import REQUIRED_FEATURE_COLUMNS
from app.analysis.ranking import Candidate, CandidateEvaluation, CandidateScore, RankingEngine, RankingPolicy, RankingResult
from app.analysis.strategy import StrategyEngine
from app.analysis.walk_forward import WalkForwardConfig, WalkForwardEngine, WalkForwardResult

ResearchStage = Literal["validation", "backtest", "evaluation", "walk_forward", "ranking", "selection"]
RESEARCH_STAGES: tuple[str, ...] = ("validation", "backtest", "evaluation", "walk_forward", "ranking", "selection")

StrategyFactory = Callable[[Mapping[str, Any]], object]

_TIMEFRAME_PATTERN = re.compile(r"^[1-9][0-9]*[mhdw]$")
_HEX_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_JSON_SCALAR_TYPES = (str, int, float, bool, type(None))
_MINIMUM_ROWS = 3


# ---------------------------------------------------------------------------
# Deterministic serialization / fingerprints
# ---------------------------------------------------------------------------


def canonical_json(payload: Any) -> str:
    """Stable JSON: sorted keys, compact separators, no NaN/inf, no Python hash() involved."""
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"payload is not canonically serializable: {exc}") from exc


def fingerprint(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return False


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _config_payload(obj: Any) -> dict[str, Any]:
    """Serializable view of a configuration dataclass. Callables (e.g. factories) are excluded."""
    if not is_dataclass(obj) or isinstance(obj, type):
        raise TypeError("configuration payload requires a dataclass instance")
    payload: dict[str, Any] = {}
    for item in fields(obj):
        value = getattr(obj, item.name)
        if callable(value):
            continue
        if is_dataclass(value) and not isinstance(value, type):
            value = _config_payload(value)
        payload[item.name] = value
    return payload


def _serializable(value: Any) -> Any:
    """
    Recursively convert engine outputs into canonical-JSON-safe data for fingerprinting.

    Non-finite floats (e.g. an infinite profit factor) become their repr strings so
    provenance is preserved instead of failing or being silently dropped.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        return number if math.isfinite(number) else repr(number)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _serializable(getattr(value, item.name))
            for item in fields(value)
            if not callable(getattr(value, item.name))
        }
    if isinstance(value, Mapping):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    raise TypeError(f"value of type {type(value).__name__} is not serializable for fingerprinting")


def _normalize_parameters(raw: Any) -> tuple[tuple[str, Any], ...]:
    if isinstance(raw, Mapping):
        items = list(raw.items())
    elif isinstance(raw, (tuple, list)):
        items = []
        for entry in raw:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise TypeError("parameters must be a mapping or a sequence of (key, value) pairs")
            items.append((entry[0], entry[1]))
    else:
        raise TypeError("parameters must be a mapping or a sequence of (key, value) pairs")

    normalized: dict[str, Any] = {}
    for key, value in items:
        key_text = _require_text(key, "parameter key")
        if key_text in normalized:
            raise ValueError(f"duplicate parameter key {key_text!r}")
        if not isinstance(value, _JSON_SCALAR_TYPES):
            raise ValueError(f"parameter {key_text!r} must be a JSON scalar (str, int, float, bool, None)")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"parameter {key_text!r} must be finite")
        normalized[key_text] = value

    return tuple(sorted(normalized.items()))


# ---------------------------------------------------------------------------
# Market data identity
# ---------------------------------------------------------------------------


def validate_feature_frame(features: Any) -> pd.DataFrame:
    """Structural validation of a feature dataframe. Read-only; returns the same object."""
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    if features.empty:
        raise ValueError("Feature dataframe cannot be empty")
    if "timestamp" not in features.columns:
        raise ValueError("Feature dataframe is missing the required 'timestamp' column")

    missing = set(REQUIRED_FEATURE_COLUMNS) - set(features.columns)
    if missing:
        raise ValueError(f"Feature dataframe is missing required columns: {sorted(missing)}")

    if len(set(str(column) for column in features.columns)) != len(features.columns):
        raise ValueError("Feature dataframe contains duplicate column names")

    timestamps = pd.to_datetime(features["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Feature dataframe contains invalid timestamps")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Feature dataframe timestamps must be chronological and non-decreasing")

    if len(features) < _MINIMUM_ROWS:
        raise ValueError(f"Feature dataframe must contain at least {_MINIMUM_ROWS} rows")

    return features


def fingerprint_frame(features: pd.DataFrame) -> str:
    """SHA-256 over a canonical, column-order-independent serialization of the frame."""
    validate_feature_frame(features)
    ordered = features.copy()
    ordered.columns = [str(column) for column in ordered.columns]
    ordered = ordered.reset_index(drop=True)
    columns = sorted(ordered.columns)
    body = ordered[columns].to_json(orient="values", date_format="iso", double_precision=15)

    digest = hashlib.sha256()
    digest.update(canonical_json(columns).encode("utf-8"))
    digest.update(b"\n")
    digest.update(body.encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class DatasetDescriptor:
    """Minimal deterministic identity of the market dataset a research batch was run on."""

    source: str
    symbol: str
    timeframe: str
    start: str
    end: str
    row_count: int
    fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _require_text(self.source, "source"))
        symbol = _require_text(self.symbol, "symbol")
        if any(character.isspace() for character in symbol):
            raise ValueError("symbol cannot contain whitespace")
        object.__setattr__(self, "symbol", symbol)

        timeframe = _require_text(self.timeframe, "timeframe")
        if not _TIMEFRAME_PATTERN.match(timeframe):
            raise ValueError(f"invalid timeframe {timeframe!r}; expected e.g. '1m', '15m', '1h', '4h', '1d'")
        object.__setattr__(self, "timeframe", timeframe)

        object.__setattr__(self, "start", _require_text(self.start, "start"))
        object.__setattr__(self, "end", _require_text(self.end, "end"))
        if pd.Timestamp(self.end) < pd.Timestamp(self.start):
            raise ValueError("dataset end cannot be earlier than dataset start")

        if isinstance(self.row_count, bool) or not isinstance(self.row_count, int) or self.row_count < _MINIMUM_ROWS:
            raise ValueError(f"row_count must be an integer >= {_MINIMUM_ROWS}")

        if not isinstance(self.fingerprint, str) or not _HEX_DIGEST_PATTERN.match(self.fingerprint):
            raise ValueError("fingerprint must be a 64-character lowercase hex SHA-256 digest")

    @classmethod
    def from_frame(cls, features: pd.DataFrame, *, source: str, symbol: str, timeframe: str) -> "DatasetDescriptor":
        validate_feature_frame(features)
        timestamps = pd.to_datetime(features["timestamp"])
        return cls(
            source=source,
            symbol=symbol,
            timeframe=timeframe,
            start=timestamps.iloc[0].isoformat(),
            end=timestamps.iloc[-1].isoformat(),
            row_count=int(len(features)),
            fingerprint=fingerprint_frame(features),
        )

    def to_dict(self) -> dict[str, Any]:
        return _config_payload(self)


# ---------------------------------------------------------------------------
# Candidate model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Full-sample backtest configuration; mirrors BacktestEngine's constructor arguments."""

    initial_capital: float = 10_000.0
    fee_pct: float = 0.10
    slippage_pct: float = 0.05
    stop_loss_atr: float | None = 1.5
    take_profit_atr: float | None = 3.0

    def __post_init__(self) -> None:
        for name in ("initial_capital", "fee_pct", "slippage_pct"):
            if not _is_finite_number(getattr(self, name)):
                raise ValueError(f"{name} must be a finite number")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.fee_pct < 0:
            raise ValueError("fee_pct cannot be negative")
        if self.slippage_pct < 0:
            raise ValueError("slippage_pct cannot be negative")
        for name in ("stop_loss_atr", "take_profit_atr"):
            value = getattr(self, name)
            if value is None:
                continue
            if not _is_finite_number(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number or None")

    def engine_kwargs(self) -> dict[str, Any]:
        return {
            "initial_capital": float(self.initial_capital),
            "fee_pct": float(self.fee_pct),
            "slippage_pct": float(self.slippage_pct),
            "stop_loss_atr": None if self.stop_loss_atr is None else float(self.stop_loss_atr),
            "take_profit_atr": None if self.take_profit_atr is None else float(self.take_profit_atr),
        }


# WalkForwardEngine constructs BacktestEngine with its defaults for every window and offers
# no hook to change that. Until such a hook exists, research candidates must use the same
# defaults so the headline backtest and the robustness check share execution assumptions.
WALK_FORWARD_BACKTEST_CONFIG = BacktestConfig()


@dataclass(frozen=True, slots=True)
class ResearchCandidate:
    """
    Pure-data description of one strategy candidate.

    No runtime strategy object is stored; strategies are built fresh from
    (strategy_id, parameters) by the pipeline for every backtest and every
    walk-forward window, so no state can leak between evaluations.
    """

    candidate_id: str
    strategy_id: str
    parameters: Mapping[str, Any] | tuple[tuple[str, Any], ...] = ()
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _require_text(self.candidate_id, "candidate_id"))
        object.__setattr__(self, "strategy_id", _require_text(self.strategy_id, "strategy_id"))
        object.__setattr__(self, "parameters", _normalize_parameters(self.parameters))
        if not isinstance(self.backtest, BacktestConfig):
            raise TypeError("backtest must be a BacktestConfig")
        name = self.candidate_id if self.name is None else _require_text(self.name, "name")
        object.__setattr__(self, "name", name)

    @property
    def parameter_map(self) -> dict[str, Any]:
        return dict(self.parameters)

    @property
    def fingerprint(self) -> str:
        """Content fingerprint: what is evaluated, independent of the label it carries."""
        return fingerprint(
            {
                "strategy_id": self.strategy_id,
                "parameters": self.parameter_map,
                "backtest": _config_payload(self.backtest),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "name": self.name,
            "strategy_id": self.strategy_id,
            "parameters": self.parameter_map,
            "backtest": _config_payload(self.backtest),
            "fingerprint": self.fingerprint,
        }


# ---------------------------------------------------------------------------
# Request (immutable configuration snapshot)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResearchRequest:
    """
    Everything needed to reproduce a research batch, except the market data
    itself, which is supplied to ResearchPipeline.run and verified against
    `dataset.fingerprint`.
    """

    dataset: DatasetDescriptor
    candidates: Sequence[ResearchCandidate]
    walk_forward_config: WalkForwardConfig
    evaluation_policy: StrategyEvaluationPolicy = field(default_factory=StrategyEvaluationPolicy)
    ranking_policy: RankingPolicy = field(default_factory=RankingPolicy)

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, DatasetDescriptor):
            raise TypeError("dataset must be a DatasetDescriptor")
        if not isinstance(self.evaluation_policy, StrategyEvaluationPolicy):
            raise TypeError("evaluation_policy must be a StrategyEvaluationPolicy")
        if not isinstance(self.ranking_policy, RankingPolicy):
            raise TypeError("ranking_policy must be a RankingPolicy")
        if not isinstance(self.walk_forward_config, WalkForwardConfig):
            raise TypeError("walk_forward_config must be a WalkForwardConfig")

        if isinstance(self.candidates, (str, bytes)) or not isinstance(self.candidates, Sequence):
            raise TypeError("candidates must be a sequence of ResearchCandidate")
        candidates = tuple(self.candidates)
        seen: set[str] = set()
        for candidate in candidates:
            if not isinstance(candidate, ResearchCandidate):
                raise TypeError("candidates must be ResearchCandidate instances")
            if candidate.candidate_id in seen:
                raise ValueError(f"duplicate candidate_id encountered: {candidate.candidate_id}")
            seen.add(candidate.candidate_id)
        object.__setattr__(self, "candidates", candidates)

        config = self.walk_forward_config
        if config.strategy_factory is not None:
            raise ValueError(
                "walk_forward_config.strategy_factory must be None; the research pipeline "
                "builds strategy instances per candidate and per window"
            )
        if config.evaluation_policy is None:
            config = replace(config, evaluation_policy=self.evaluation_policy)
            object.__setattr__(self, "walk_forward_config", config)

        required_rows = config.train_bars + config.test_bars
        if self.dataset.row_count < required_rows:
            raise ValueError(
                f"dataset has {self.dataset.row_count} rows but walk-forward requires at least "
                f"train_bars + test_bars = {required_rows}"
            )

        # Fails loudly on non-finite or non-serializable configuration values.
        self.configuration_fingerprint

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(candidate.candidate_id for candidate in self.candidates)

    @property
    def configuration_payload(self) -> dict[str, Any]:
        return {
            "evaluation_policy": _config_payload(self.evaluation_policy),
            "walk_forward": _config_payload(self.walk_forward_config),
            "ranking_policy": _config_payload(self.ranking_policy),
        }

    @property
    def configuration_fingerprint(self) -> str:
        return fingerprint(self.configuration_payload)

    @property
    def fingerprint(self) -> str:
        return fingerprint(
            {
                "dataset": self.dataset.to_dict(),
                "configuration": self.configuration_payload,
                "candidates": [
                    {"candidate_id": candidate.candidate_id, "name": candidate.name, "fingerprint": candidate.fingerprint}
                    for candidate in self.candidates
                ],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "dataset": self.dataset.to_dict(),
            "configuration": self.configuration_payload,
            "configuration_fingerprint": self.configuration_fingerprint,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResearchFailure:
    """An observable, non-fabricated failure. candidate_id is None for batch-level stages."""

    candidate_id: str | None
    stage: ResearchStage
    error_type: str
    message: str

    def __post_init__(self) -> None:
        if self.candidate_id is not None:
            object.__setattr__(self, "candidate_id", _require_text(self.candidate_id, "candidate_id"))
        if self.stage not in RESEARCH_STAGES:
            raise ValueError(f"unknown research stage {self.stage!r}")
        object.__setattr__(self, "error_type", _require_text(self.error_type, "error_type"))
        if not isinstance(self.message, str):
            raise TypeError("message must be a string")

    @classmethod
    def from_exception(cls, candidate_id: str | None, stage: ResearchStage, exc: BaseException) -> "ResearchFailure":
        return cls(
            candidate_id=candidate_id,
            stage=stage,
            error_type=type(exc).__name__,
            message=str(exc) or type(exc).__name__,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "stage": self.stage,
            "error_type": self.error_type,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class ResearchCandidateResult:
    """A fully evaluated candidate, expressed through the existing CandidateEvaluation model."""

    candidate: ResearchCandidate
    evaluation_fingerprint: str
    candidate_evaluation: CandidateEvaluation

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, ResearchCandidate):
            raise TypeError("candidate must be a ResearchCandidate")
        if not isinstance(self.evaluation_fingerprint, str) or not _HEX_DIGEST_PATTERN.match(self.evaluation_fingerprint):
            raise ValueError("evaluation_fingerprint must be a 64-character lowercase hex SHA-256 digest")
        if not isinstance(self.candidate_evaluation, CandidateEvaluation):
            raise TypeError("candidate_evaluation must be a CandidateEvaluation")
        evaluation = self.candidate_evaluation
        if evaluation.candidate.candidate_id != self.candidate.candidate_id:
            raise ValueError("candidate_evaluation does not belong to this candidate")
        if evaluation.backtest_result is None or evaluation.evaluation is None or evaluation.walk_forward_result is None:
            raise ValueError("a successful research result requires backtest, evaluation and walk-forward results")

    @property
    def candidate_id(self) -> str:
        return self.candidate.candidate_id

    @property
    def backtest_result(self) -> BacktestResult:
        return self.candidate_evaluation.backtest_result  # type: ignore[return-value]

    @property
    def evaluation(self) -> StrategyEvaluation:
        return self.candidate_evaluation.evaluation  # type: ignore[return-value]

    @property
    def walk_forward_result(self) -> WalkForwardResult:
        return self.candidate_evaluation.walk_forward_result  # type: ignore[return-value]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_fingerprint": self.candidate.fingerprint,
            "evaluation_fingerprint": self.evaluation_fingerprint,
            "evaluation_decision": self.evaluation.decision,
            "robustness_decision": self.walk_forward_result.robustness_decision,
            "metrics": _serializable(dict(self.evaluation.metrics)),
            "walk_forward": {
                "total_windows": self.walk_forward_result.total_windows,
                "passed_windows": self.walk_forward_result.passed_windows,
                "failed_windows": self.walk_forward_result.failed_windows,
                "oos_pass_rate_pct": self.walk_forward_result.oos_pass_rate_pct,
                "aggregate_oos_return_pct": self.walk_forward_result.aggregate_oos_return_pct,
                "reasons": list(self.walk_forward_result.reasons),
            },
        }


@dataclass(frozen=True, slots=True)
class ResearchBatchResult:
    """Auditable outcome of one research batch."""

    request: ResearchRequest
    results: Sequence[ResearchCandidateResult]
    failures: Sequence[ResearchFailure]
    ranking_result: RankingResult | None
    selection_decision: SelectionDecision | None

    def __post_init__(self) -> None:
        if not isinstance(self.request, ResearchRequest):
            raise TypeError("request must be a ResearchRequest")
        results = tuple(self.results)
        failures = tuple(self.failures)
        known_ids = set(self.request.candidate_ids)
        for item in results:
            if not isinstance(item, ResearchCandidateResult):
                raise TypeError("results must be ResearchCandidateResult instances")
            if item.candidate_id not in known_ids:
                raise ValueError(f"result for unknown candidate {item.candidate_id!r}")
        for item in failures:
            if not isinstance(item, ResearchFailure):
                raise TypeError("failures must be ResearchFailure instances")
            if item.candidate_id is not None and item.candidate_id not in known_ids:
                raise ValueError(f"failure for unknown candidate {item.candidate_id!r}")
        if self.ranking_result is not None and not isinstance(self.ranking_result, RankingResult):
            raise TypeError("ranking_result must be a RankingResult or None")
        if self.selection_decision is not None and not isinstance(self.selection_decision, SelectionDecision):
            raise TypeError("selection_decision must be a SelectionDecision or None")
        object.__setattr__(self, "results", results)
        object.__setattr__(self, "failures", failures)

    @property
    def total_candidates(self) -> int:
        return len(self.request.candidates)

    @property
    def successful_candidates(self) -> int:
        return len(self.results)

    @property
    def candidate_failures(self) -> tuple[ResearchFailure, ...]:
        return tuple(item for item in self.failures if item.candidate_id is not None)

    @property
    def batch_failures(self) -> tuple[ResearchFailure, ...]:
        return tuple(item for item in self.failures if item.candidate_id is None)

    @property
    def failed_candidates(self) -> int:
        return len({item.candidate_id for item in self.candidate_failures})

    @property
    def ranked_candidates(self) -> tuple[CandidateScore, ...]:
        if self.ranking_result is None:
            return ()
        return tuple(self.ranking_result.ranked_candidates)

    @property
    def ranked_candidate_ids(self) -> tuple[str, ...]:
        return tuple(score.candidate_id for score in self.ranked_candidates)

    @property
    def selected_candidate_id(self) -> str | None:
        if self.selection_decision is None:
            return None
        return self.selection_decision.selected_candidate_id

    @property
    def selected_candidate(self) -> CandidateScore | None:
        selected = self.selected_candidate_id
        return None if selected is None else self.score_for(selected)

    def result_for(self, candidate_id: str) -> ResearchCandidateResult | None:
        return next((item for item in self.results if item.candidate_id == candidate_id), None)

    def score_for(self, candidate_id: str) -> CandidateScore | None:
        return next((score for score in self.ranked_candidates if score.candidate_id == candidate_id), None)

    def failures_for(self, candidate_id: str) -> tuple[ResearchFailure, ...]:
        return tuple(item for item in self.failures if item.candidate_id == candidate_id)

    def to_dict(self) -> dict[str, Any]:
        ranking = None
        if self.ranking_result is not None:
            ranking = [
                {
                    "candidate_id": score.candidate_id,
                    "candidate_name": score.candidate_name,
                    "score": score.score,
                    "accepted": score.accepted,
                    "robustness_decision": score.robustness_decision,
                    "evaluation_decision": score.evaluation_decision,
                    "reasons": list(score.reasons),
                    "metrics_used": _serializable(dict(score.metrics_used)),
                }
                for score in self.ranking_result.ranked_candidates
            ]

        selection = None
        if self.selection_decision is not None:
            decision = self.selection_decision
            selection = {
                "selected_candidate_id": decision.selected_candidate_id,
                "selected_candidate_name": decision.selected_candidate_name,
                "selection_source": decision.selection_source,
                "reason": decision.reason,
                "eligible_candidate_count": decision.eligible_candidate_count,
                "eligible_candidate_ids": [score.candidate_id for score in decision.eligible_candidates],
                "provider_metadata": _serializable(dict(decision.provider_metadata)),
            }

        return {
            "request": self.request.to_dict(),
            "summary": {
                "total_candidates": self.total_candidates,
                "successful_candidates": self.successful_candidates,
                "failed_candidates": self.failed_candidates,
                "ranked_candidate_ids": list(self.ranked_candidate_ids),
                "selected_candidate_id": self.selected_candidate_id,
            },
            "results": [item.to_dict() for item in self.results],
            "failures": [item.to_dict() for item in self.failures],
            "ranking": ranking,
            "selection": selection,
        }


# ---------------------------------------------------------------------------
# Strategy factories
# ---------------------------------------------------------------------------


def _build_rule_based_strategy(parameters: Mapping[str, Any]) -> StrategyEngine:
    return StrategyEngine(**dict(parameters))


def default_strategy_factories() -> dict[str, StrategyFactory]:
    """Factories for the strategies shipped with ATLAS' analysis layer."""
    return {"rule_based": _build_rule_based_strategy}


def _strategy_builder(factory: StrategyFactory, parameters: Mapping[str, Any], candidate_id: str) -> Callable[[], object]:
    """Zero-argument builder returning a fresh, validated strategy instance on every call."""

    def build() -> object:
        strategy = factory(dict(parameters))
        if strategy is None or not callable(getattr(strategy, "generate_signal", None)):
            raise TypeError(
                f"strategy factory for candidate {candidate_id!r} returned an object without a callable generate_signal"
            )
        return strategy

    return build


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class ResearchPipeline:
    """
    Orchestrates existing research components over a finite candidate batch.

    Per candidate, in request order: validation -> full-sample backtest ->
    evaluation -> walk-forward. Then, over successful candidates only:
    RankingEngine -> CandidateSelector (the only path to any SelectionProvider).

    The pipeline owns sequencing, validation, provenance and aggregation. It
    contains no backtest, evaluation, walk-forward, ranking, risk or execution
    logic, and it never produces orders.
    """

    def __init__(
        self,
        *,
        strategy_factories: Mapping[str, StrategyFactory],
        selection_provider: SelectionProvider | None = None,
        evaluator: StrategyEvaluator | None = None,
        ranking_engine: RankingEngine | None = None,
    ) -> None:
        if not isinstance(strategy_factories, Mapping):
            raise TypeError("strategy_factories must be a mapping of strategy_id -> factory")
        factories: dict[str, StrategyFactory] = {}
        for strategy_id, factory in strategy_factories.items():
            key = _require_text(strategy_id, "strategy_id")
            if not callable(factory):
                raise TypeError(f"strategy factory for {key!r} must be callable")
            factories[key] = factory

        self._strategy_factories = factories
        self._evaluator = evaluator or StrategyEvaluator()
        self._ranking_engine = ranking_engine or RankingEngine()
        self._selector = CandidateSelector(provider=selection_provider)

    @property
    def strategy_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._strategy_factories))

    def run(self, request: ResearchRequest, features: pd.DataFrame) -> ResearchBatchResult:
        if not isinstance(request, ResearchRequest):
            raise TypeError("request must be a ResearchRequest")

        observed = DatasetDescriptor.from_frame(
            features,
            source=request.dataset.source,
            symbol=request.dataset.symbol,
            timeframe=request.dataset.timeframe,
        )
        if observed != request.dataset:
            raise ValueError(
                "market data does not match the request's dataset descriptor "
                f"(expected fingerprint {request.dataset.fingerprint}, observed {observed.fingerprint})"
            )

        frame = features.copy().reset_index(drop=True)

        results: list[ResearchCandidateResult] = []
        failures: list[ResearchFailure] = []
        for candidate in request.candidates:
            outcome = self._research_candidate(candidate, request, frame)
            if isinstance(outcome, ResearchCandidateResult):
                results.append(outcome)
            else:
                failures.append(outcome)

        ranking_result: RankingResult | None = None
        if results:
            try:
                ranking_result = self._ranking_engine.rank_candidates(
                    [item.candidate_evaluation for item in results],
                    policy=request.ranking_policy,
                )
            except Exception as exc:
                failures.append(ResearchFailure.from_exception(None, "ranking", exc))

        selection_decision: SelectionDecision | None = None
        if ranking_result is not None:
            try:
                selection_decision = self._selector.select(SelectionRequest(ranking_result=ranking_result))
            except Exception as exc:
                failures.append(ResearchFailure.from_exception(None, "selection", exc))

        return ResearchBatchResult(
            request=request,
            results=tuple(results),
            failures=tuple(failures),
            ranking_result=ranking_result,
            selection_decision=selection_decision,
        )

    # -- per-candidate stages ---------------------------------------------

    def _prepare_strategy(
        self,
        candidate: ResearchCandidate,
    ) -> tuple[Callable[[], object], object] | ResearchFailure:
        candidate_id = candidate.candidate_id

        factory = self._strategy_factories.get(candidate.strategy_id)
        if factory is None:
            return ResearchFailure(
                candidate_id=candidate_id,
                stage="validation",
                error_type="ValueError",
                message=f"unknown strategy_id {candidate.strategy_id!r}; known strategies: {list(self.strategy_ids)}",
            )

        if candidate.backtest != WALK_FORWARD_BACKTEST_CONFIG:
            return ResearchFailure(
                candidate_id=candidate_id,
                stage="validation",
                error_type="ValueError",
                message=(
                    "walk-forward windows are evaluated with BacktestEngine defaults; a non-default "
                    "backtest configuration is rejected until walk-forward backtest parity exists"
                ),
            )

        build = _strategy_builder(factory, candidate.parameters, candidate_id)
        try:
            strategy = build()
        except Exception as exc:
            return ResearchFailure(
                candidate_id=candidate_id,
                stage="validation",
                error_type=type(exc).__name__,
                message=f"strategy factory failed for candidate {candidate_id!r}: {exc}",
            )
        return build, strategy

    @staticmethod
    def _evaluation_fingerprint(
        candidate: ResearchCandidate,
        request: ResearchRequest,
        backtest_result: BacktestResult,
        evaluation: StrategyEvaluation,
        walk_forward_result: WalkForwardResult,
    ) -> str:
        return fingerprint(
            {
                "candidate": candidate.to_dict(),
                "dataset_fingerprint": request.dataset.fingerprint,
                "configuration_fingerprint": request.configuration_fingerprint,
                "backtest": _serializable(backtest_result),
                "evaluation": _serializable(evaluation),
                "walk_forward": _serializable(walk_forward_result),
            }
        )

    def _research_candidate(
        self,
        candidate: ResearchCandidate,
        request: ResearchRequest,
        frame: pd.DataFrame,
    ) -> ResearchCandidateResult | ResearchFailure:
        candidate_id = candidate.candidate_id

        prepared = self._prepare_strategy(candidate)
        if isinstance(prepared, ResearchFailure):
            return prepared
        build, strategy = prepared

        try:
            engine = BacktestEngine(strategy=strategy, **candidate.backtest.engine_kwargs())
            backtest_result = engine.run(symbol=request.dataset.symbol, features=frame)
        except Exception as exc:
            return ResearchFailure.from_exception(candidate_id, "backtest", exc)

        try:
            evaluation = self._evaluator.evaluate(backtest_result, policy=request.evaluation_policy)
        except Exception as exc:
            return ResearchFailure.from_exception(candidate_id, "evaluation", exc)

        try:
            walk_forward_engine = WalkForwardEngine(strategy_factory=build, evaluator=self._evaluator)
            walk_forward_result = walk_forward_engine.run(frame, request.walk_forward_config)
        except Exception as exc:
            return ResearchFailure.from_exception(candidate_id, "walk_forward", exc)

        candidate_evaluation = CandidateEvaluation(
            candidate=Candidate(candidate_id=candidate_id, name=candidate.name, strategy=candidate),
            backtest_result=backtest_result,
            evaluation=evaluation,
            walk_forward_result=walk_forward_result,
        )
        return ResearchCandidateResult(
            candidate=candidate,
            evaluation_fingerprint=self._evaluation_fingerprint(
                candidate, request, backtest_result, evaluation, walk_forward_result
            ),
            candidate_evaluation=candidate_evaluation,
        )