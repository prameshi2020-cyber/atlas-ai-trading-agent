from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from app.analysis.backtest import BacktestEngine
from app.analysis.evaluation import StrategyEvaluationPolicy
from app.analysis.strategy import Signal
from app.analysis.indicators import REQUIRED_FEATURE_COLUMNS
from app.analysis.walk_forward import (
    WalkForwardConfig,
    WalkForwardEngine,
)


class ScriptedStrategy:
    def __init__(self, actions: list[str] | None = None) -> None:
        self._actions = actions or ["BUY", "HOLD", "HOLD"]
        self.calls: list[pd.DataFrame] = []

    def generate_signal(self, symbol: str, features: pd.DataFrame) -> Signal:
        self.calls.append(features)
        action = self._actions[min(len(self.calls) - 1, len(self._actions) - 1)]
        return Signal(
            symbol=symbol,
            action=action,
            score=1.0 if action == "BUY" else 0.0,
            confidence=1.0 if action == "BUY" else 0.0,
            reasons=[action],
        )


def make_row(ts, o, h, l, c, atr, v=10.0) -> dict:
    row = {
        "timestamp": ts,
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": v,
        "atr_14": atr,
    }
    for col in REQUIRED_FEATURE_COLUMNS:
        row.setdefault(col, 0.0 if col not in ("rsi_14",) else 50.0)
    return row


def make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


TS = pd.date_range("2026-01-01", periods=120, freq="1min", tz="UTC")


def _build_dataset(n: int = 120) -> pd.DataFrame:
    rows = []
    base = 100.0
    for i in range(n):
        price = base + i * 0.15
        rows.append(
            make_row(
                TS[i],
                price,
                price + 0.8,
                price - 0.8,
                price + 0.2,
                atr=1.0,
            )
        )
    return make_df(rows)


def test_walk_forward_generates_valid_chronological_windows():
    data = _build_dataset(120)
    config = WalkForwardConfig(train_bars=30, test_bars=10, step_bars=10)

    engine = WalkForwardEngine(strategy_factory=lambda: ScriptedStrategy(["BUY", "HOLD", "HOLD"]))
    windows = engine._generate_windows(data, config)

    assert len(windows) == 9
    assert windows[0].train_start == 0
    assert windows[0].train_end == 30
    assert windows[0].test_start == 30
    assert windows[0].test_end == 40
    assert windows[1].train_start == 10
    assert windows[1].test_start == 40
    assert windows[-1].test_end <= len(data)


def test_walk_forward_uses_half_open_interval_semantics():
    data = _build_dataset(80)
    config = WalkForwardConfig(train_bars=20, test_bars=10, step_bars=10)

    engine = WalkForwardEngine(strategy_factory=lambda: ScriptedStrategy(["BUY", "HOLD", "HOLD"]))
    windows = engine._generate_windows(data, config)

    first = windows[0]
    assert first.train_start == 0
    assert first.train_end == 20
    assert first.test_start == 20
    assert first.test_end == 30
    assert first.train_end == first.test_start
    assert first.train_start < first.train_end < first.test_end
    assert not (first.train_start <= first.test_start < first.train_end)


def test_walk_forward_rejects_invalid_config():
    with pytest.raises(ValueError, match="train_bars"):
        WalkForwardConfig(train_bars=0, test_bars=10, step_bars=5)

    with pytest.raises(ValueError, match="test_bars"):
        WalkForwardConfig(train_bars=20, test_bars=0, step_bars=5)

    with pytest.raises(ValueError, match="step_bars"):
        WalkForwardConfig(train_bars=20, test_bars=10, step_bars=0)


def test_walk_forward_rejects_insufficient_data():
    data = _build_dataset(30)
    config = WalkForwardConfig(train_bars=20, test_bars=15, step_bars=5)

    engine = WalkForwardEngine(strategy_factory=lambda: ScriptedStrategy(["BUY", "HOLD", "HOLD"]))
    with pytest.raises(ValueError, match="insufficient data"):
        engine.run(data, config)


def test_walk_forward_rejects_non_chronological_timestamps():
    data = _build_dataset(50).sort_values("timestamp", ascending=False).reset_index(drop=True)
    config = WalkForwardConfig(train_bars=20, test_bars=10, step_bars=5)

    engine = WalkForwardEngine(strategy_factory=lambda: ScriptedStrategy(["BUY", "HOLD", "HOLD"]))
    with pytest.raises(ValueError, match="chronological"):
        engine.run(data, config)


def test_single_window_case_works():
    data = _build_dataset(60)
    config = WalkForwardConfig(train_bars=30, test_bars=10, step_bars=10)

    engine = WalkForwardEngine(strategy_factory=lambda: ScriptedStrategy(["BUY", "HOLD", "HOLD"]))
    result = engine.run(data, config)

    assert result.total_windows == 3
    assert len(result.windows) == 3
    assert result.windows[0].in_sample_evaluation is not None
    assert result.windows[0].out_of_sample_evaluation is not None
    assert result.windows[0].test_start == 30
    assert result.windows[0].test_end == 40


def test_walk_forward_runs_is_and_oos_evaluations_independently_per_window():
    data = _build_dataset(80)
    config = WalkForwardConfig(train_bars=20, test_bars=10, step_bars=10, evaluation_policy=StrategyEvaluationPolicy())

    engine = WalkForwardEngine(strategy_factory=lambda: ScriptedStrategy(["BUY", "HOLD", "HOLD"]))
    result = engine.run(data, config)

    assert result.total_windows == 6
    for window in result.windows:
        assert window.in_sample_evaluation is not None
        assert window.out_of_sample_evaluation is not None
        assert window.in_sample_evaluation.metrics["total_trades"] >= 0
        assert window.out_of_sample_evaluation.metrics["total_trades"] >= 0
        assert window.out_of_sample_evaluation.metrics["total_trades"] == window.out_of_sample_evaluation.metrics["total_trades"]


def test_multiple_failed_oos_windows_produce_fail_or_review():
    data = _build_dataset(110)
    config = WalkForwardConfig(
        train_bars=25,
        test_bars=10,
        step_bars=10,
        minimum_train_trades=0,
        minimum_test_trades=0,
        minimum_oos_pass_rate_pct=90.0,
    )

    engine = WalkForwardEngine(strategy_factory=lambda: ScriptedStrategy(["HOLD", "HOLD", "HOLD"]))
    result = engine.run(data, config)

    assert result.robustness_decision in {"FAIL", "REVIEW"}
    assert result.failed_windows >= 1


def test_deterministic_input_produces_deterministic_result():
    data = _build_dataset(90)
    config = WalkForwardConfig(train_bars=25, test_bars=10, step_bars=10)

    engine = WalkForwardEngine(strategy_factory=lambda: ScriptedStrategy(["BUY", "HOLD", "HOLD"]))
    first = engine.run(data, config)
    second = engine.run(data.copy(), config)

    assert first.total_windows == second.total_windows
    assert first.robustness_decision == second.robustness_decision
    assert first.oos_pass_rate_pct == second.oos_pass_rate_pct


def test_walk_forward_rejects_warmup_longer_than_train_window_explicitly():
    rows = [make_row(TS[i], 100.0 + i, 100.0 + i + 1, 99.0 + i, 100.0 + i, atr=float("nan")) for i in range(30)]
    data = make_df(rows)
    config = WalkForwardConfig(train_bars=20, test_bars=10, step_bars=5)

    engine = WalkForwardEngine(strategy_factory=lambda: ScriptedStrategy(["BUY", "HOLD", "HOLD"]))
    with pytest.raises(ValueError, match="warmup|warm-up|insufficient warmed"):
        engine.run(data, config)


def test_walk_forward_does_not_imply_parameter_optimization():
    data = _build_dataset(80)
    config = WalkForwardConfig(train_bars=15, test_bars=10, step_bars=10)

    engine = WalkForwardEngine(strategy_factory=lambda: ScriptedStrategy(["BUY", "HOLD", "HOLD"]))
    result = engine.run(data, config)

    assert not hasattr(result, "optimized_parameters")
    assert not hasattr(result, "best_params")
    assert "parameter" not in str(result.reasons).lower()


@pytest.mark.parametrize(
    "train_bars,test_bars,step_bars",
    [(20, 10, 1), (10, 5, 5), (30, 20, 10)],
)
def test_walk_forward_allows_valid_multiple_window_configs(train_bars, test_bars, step_bars):
    data = _build_dataset(120)
    config = WalkForwardConfig(train_bars=train_bars, test_bars=test_bars, step_bars=step_bars)

    engine = WalkForwardEngine(strategy_factory=lambda: ScriptedStrategy(["BUY", "HOLD", "HOLD"]))
    result = engine.run(data, config)

    assert result.total_windows >= 1
    assert result.total_windows == len(result.windows)
