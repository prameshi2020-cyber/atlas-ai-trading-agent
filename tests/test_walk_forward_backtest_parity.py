"""Walk-forward / backtest execution parity.

Contract: every walk-forward window is backtested with exactly the same
BacktestEngine configuration (capital, fee, slippage, ATR exits) as the
headline backtest. Otherwise the robustness verdict is not comparable with
the full-sample result. Values are never re-validated here; BacktestEngine
remains the single owner of its argument rules.
"""
from __future__ import annotations

import pytest

from app.analysis import walk_forward as wf_module
from app.analysis.backtest import BacktestEngine, BacktestResult
from app.analysis.walk_forward import WalkForwardEngine
from tests.test_research_pipeline import GOOD_A, WF, ScriptedStrategy, build_frame

CUSTOM = {
    "initial_capital": 2_500.0,
    "fee_pct": 0.0,
    "slippage_pct": 0.0,
    "stop_loss_atr": None,
    "take_profit_atr": None,
}


def factory() -> ScriptedStrategy:
    return ScriptedStrategy(GOOD_A.split(","))


def engine(kwargs: dict | None) -> WalkForwardEngine:
    return WalkForwardEngine(strategy_factory=factory, backtest_kwargs=kwargs)


def signature(result: BacktestResult) -> tuple:
    return (
        result.initial_capital,
        result.final_capital,
        result.total_return_pct,
        result.total_trades,
        result.winning_trades,
        result.losing_trades,
        result.max_drawdown_pct,
        result.profit_factor,
        tuple((t.side, t.entry_price, t.exit_price, t.pnl_pct, t.exit_reason) for t in result.trades),
    )


@pytest.fixture
def frame():
    return build_frame()


# --- forwarding -------------------------------------------------------------


def test_every_window_engine_receives_exactly_the_configured_kwargs(monkeypatch, frame):
    real = wf_module.BacktestEngine
    calls: list[dict] = []

    class Spy(real):
        def __init__(self, *args, **kwargs):
            calls.append({k: v for k, v in kwargs.items() if k != "strategy"})
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(wf_module, "BacktestEngine", Spy)
    result = engine(CUSTOM).run(frame, WF)

    assert result.total_windows > 0
    assert len(calls) == 2 * result.total_windows  # one IS + one OOS engine per window
    assert all(call == CUSTOM for call in calls)


@pytest.mark.parametrize(
    "kwargs, field",
    [
        ({"initial_capital": 2_500.0}, "final_capital"),
        ({"fee_pct": 0.0}, "total_return_pct"),
        ({"slippage_pct": 0.0}, "total_return_pct"),
    ],
)
def test_changing_execution_assumptions_changes_oos_results(frame, kwargs, field):
    base = engine(None).run(frame, WF)
    changed = engine(kwargs).run(frame, WF)

    assert any(
        getattr(b.out_of_sample_result, field) != getattr(c.out_of_sample_result, field)
        for b, c in zip(base.windows, changed.windows, strict=True)
    )


# --- hard parity -----------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [None, CUSTOM, {"stop_loss_atr": None, "take_profit_atr": None}, {"initial_capital": 250.0}],
    ids=["defaults", "custom", "no-atr-exits", "small-capital"],
)
def test_oos_windows_are_identical_to_a_direct_backtest_with_the_same_config(frame, kwargs):
    result = engine(kwargs).run(frame, WF)
    assert result.total_windows > 0

    for window in result.windows:
        test_slice = frame.iloc[window.test_start:window.test_end].copy()
        direct = BacktestEngine(strategy=factory(), **(kwargs or {})).run(symbol="WALK_FORWARD_OOS", features=test_slice)
        assert signature(direct) == signature(window.out_of_sample_result)


def test_default_engine_matches_explicit_empty_kwargs(frame):
    a = engine(None).run(frame, WF)
    b = engine({}).run(frame, WF)
    assert [signature(w.out_of_sample_result) for w in a.windows] == [signature(w.out_of_sample_result) for w in b.windows]
    assert dict(engine(None).backtest_kwargs) == {}


# --- input contracts (keys checked here, values owned by BacktestEngine) -------------


def test_unknown_backtest_kwarg_is_rejected_at_construction():
    with pytest.raises(TypeError, match="leverage"):
        WalkForwardEngine(strategy_factory=factory, backtest_kwargs={"leverage": 3})


def test_strategy_cannot_be_injected_through_backtest_kwargs():
    with pytest.raises(TypeError, match="strategy"):
        WalkForwardEngine(strategy_factory=factory, backtest_kwargs={"strategy": object()})


def test_non_mapping_backtest_kwargs_rejected():
    with pytest.raises(TypeError):
        WalkForwardEngine(strategy_factory=factory, backtest_kwargs=[("fee_pct", 0.0)])  # type: ignore[arg-type]


def test_invalid_value_surfaces_backtest_engine_error_unchanged(frame):
    with pytest.raises(ValueError, match="fee_pct"):
        engine({"fee_pct": -0.1}).run(frame, WF)


def test_backtest_kwargs_are_copied_and_read_only():
    kwargs = {"fee_pct": 0.0}
    wf = engine(kwargs)
    kwargs["fee_pct"] = 5.0

    assert wf.backtest_kwargs["fee_pct"] == 0.0
    with pytest.raises(TypeError):
        wf.backtest_kwargs["fee_pct"] = 1.0  # type: ignore[index]