from __future__ import annotations

import pandas as pd
import pytest

from app.analysis.backtest import BacktestEngine
from app.analysis.ranking import RankingEngine, Candidate, CandidateEvaluation
from app.analysis.evaluation import StrategyEvaluator, StrategyEvaluationPolicy
from app.analysis.backtest import BacktestTrade


class ScriptedStrategy:
    def __init__(self, actions: list[str]) -> None:
        self._actions = actions
        self.calls = 0

    def generate_signal(self, symbol: str, features: pd.DataFrame):
        self.calls += 1
        action = self._actions[min(self.calls - 1, len(self._actions) - 1)]
        # simple Signal-like object
        class S:
            def __init__(self, action):
                self.action = action

        return S(action)


from app.analysis.indicators import REQUIRED_FEATURE_COLUMNS


def make_row(ts, o, h, l, c, atr, v=10.0) -> dict:
    row = {"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": v, "atr_14": atr}
    for col in REQUIRED_FEATURE_COLUMNS:
        row.setdefault(col, 50.0 if col == "rsi_14" else 0.0)
    return row


TS = pd.date_range("2026-01-01", periods=4, freq="1min", tz="UTC")


def test_peak_trough_drawdown_positive_magnitude():
    # initial capital 1000, create a single losing LONG trade that reduces
    # equity to 900 -> 10% drawdown expected (positive)
    rows = [
        make_row(TS[0], 100, 100, 100, 100, atr=1.0),
        # entry at open=100, this bar's close will be later used
        make_row(TS[1], 100, 100, 90, 90, atr=1.0),
        make_row(TS[2], 90, 90, 90, 90, atr=1.0),
        make_row(TS[3], 90, 90, 90, 90, atr=1.0),
    ]
    features = pd.DataFrame(rows)

    strategy = ScriptedStrategy(["BUY", "HOLD", "HOLD"])
    engine = BacktestEngine(strategy=strategy, initial_capital=1000.0, fee_pct=0.0, slippage_pct=0.0, stop_loss_atr=None, take_profit_atr=None)

    result = engine.run(symbol="TEST", features=features)

    # Contract: positive magnitude
    assert result.max_drawdown_pct == pytest.approx(10.0)


def test_no_drawdown_is_zero():
    rows = [
        make_row(TS[0], 100, 100, 100, 100, atr=1.0),
        make_row(TS[1], 101, 101, 101, 101, atr=1.0),
        make_row(TS[2], 102, 102, 102, 102, atr=1.0),
        make_row(TS[3], 103, 103, 103, 103, atr=1.0),
    ]
    features = pd.DataFrame(rows)

    strategy = ScriptedStrategy(["HOLD", "HOLD", "HOLD"])
    engine = BacktestEngine(strategy=strategy, initial_capital=1000.0, fee_pct=0.0, slippage_pct=0.0, stop_loss_atr=None, take_profit_atr=None)

    result = engine.run(symbol="TEST", features=features)
    assert result.max_drawdown_pct == pytest.approx(0.0)


def test_ranking_drawdown_component_within_unit_interval():
    # construct a BacktestResult-like object with 10% drawdown and ensure
    # the ranking drawdown component is within [0,1]
    class DummyCandidate:
        def __init__(self, cid: str):
            self.candidate_id = cid
            self.name = cid
            self.strategy = object()

    # Use a minimal BacktestResult-like object via a small fabricated result
    trades = []
    # We'll directly craft a BacktestResult via BacktestEngine above and reuse
    rows = [
        make_row(TS[0], 100, 100, 100, 100, atr=1.0),
        make_row(TS[1], 100, 100, 90, 90, atr=1.0),
        make_row(TS[2], 90, 90, 90, 90, atr=1.0),
    ]
    features = pd.DataFrame(rows)
    strategy = ScriptedStrategy(["BUY", "HOLD"])
    engine = BacktestEngine(strategy=strategy, initial_capital=1000.0, fee_pct=0.0, slippage_pct=0.0, stop_loss_atr=None, take_profit_atr=None)
    result = engine.run(symbol="TEST", features=features)

    cand = DummyCandidate("c1")
    ce = CandidateEvaluation(candidate=cand, backtest_result=result, evaluation=None, walk_forward_result=None)
    score = RankingEngine().score_candidate(ce)

    drawdown_ratio = score.metrics_used["score_components"]["drawdown_ratio"]
    assert 0.0 <= drawdown_ratio <= 1.0


def test_evaluator_threshold_and_equality_behavior():
    # evaluator should fail when actual drawdown > threshold and pass when equal
    from app.analysis.backtest import BacktestResult
    from datetime import datetime, timezone

    UTC = timezone.utc
    t = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)

    # Create minimal trade list
    trade = BacktestTrade(symbol="X", side="LONG", entry_time=t, exit_time=t, entry_price=100.0, exit_price=90.0, pnl_pct=-10.0, pnl_amount=-10.0, exit_reason="TEST")
    r = BacktestResult(symbol="X", initial_capital=1000.0, final_capital=900.0, total_return_pct=-10.0, total_trades=1, winning_trades=0, losing_trades=1, win_rate_pct=0.0, max_drawdown_pct=12.0, profit_factor=0.0, expectancy_pct=0.0, average_trade_pct=-10.0, average_win_pct=0.0, average_loss_pct=-10.0, largest_win_pct=0.0, largest_loss_pct=-10.0, long_trades=1, long_wins=0, short_trades=0, short_wins=0, trades=[trade])

    evalr = StrategyEvaluator()
    policy_fail = StrategyEvaluationPolicy(maximum_drawdown_pct=10.0, minimum_total_return_pct=-1000.0, minimum_trade_count=1, minimum_profit_factor=0.0)
    evaluation = evalr.evaluate(r, policy=policy_fail)
    assert evaluation.decision == "FAIL"

    policy_equal = StrategyEvaluationPolicy(maximum_drawdown_pct=12.0, minimum_total_return_pct=-1000.0, minimum_trade_count=1, minimum_profit_factor=0.0)
    evaluation2 = evalr.evaluate(r, policy=policy_equal)
    assert evaluation2.decision == "PASS"
