from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.analysis.backtest import BacktestResult, BacktestTrade
from app.analysis.evaluation import (
    StrategyEvaluationPolicy,
    StrategyEvaluator,
)


UTC = timezone.utc


def _trade(*, pnl_pct: float, entry_time: datetime, exit_time: datetime, side: str = "LONG") -> BacktestTrade:
    return BacktestTrade(
        symbol="BTCUSDT",
        side=side,
        entry_time=entry_time,
        exit_time=exit_time,
        entry_price=100.0,
        exit_price=100.0 + pnl_pct,
        pnl_pct=pnl_pct,
        pnl_amount=pnl_pct,
        exit_reason="TEST",
    )


def _result(trades: list[BacktestTrade], *, max_drawdown_pct: float = 10.0) -> BacktestResult:
    total_trades = len(trades)
    total_return_pct = sum(trade.pnl_pct for trade in trades)
    win_count = sum(1 for trade in trades if trade.pnl_pct > 0)
    loss_count = total_trades - win_count
    average_trade_pct = total_return_pct / total_trades if total_trades else 0.0
    average_win_pct = sum(trade.pnl_pct for trade in trades if trade.pnl_pct > 0) / win_count if win_count else 0.0
    average_loss_pct = sum(trade.pnl_pct for trade in trades if trade.pnl_pct <= 0) / loss_count if loss_count else 0.0
    profit_factor = (
        sum(trade.pnl_pct for trade in trades if trade.pnl_pct > 0)
        / abs(sum(trade.pnl_pct for trade in trades if trade.pnl_pct <= 0))
        if sum(trade.pnl_pct for trade in trades if trade.pnl_pct <= 0) < 0
        else float("inf")
    )
    expectancy_pct = (
        (win_count / total_trades * average_win_pct)
        + (loss_count / total_trades * average_loss_pct)
        if total_trades
        else 0.0
    )

    return BacktestResult(
        symbol="BTCUSDT",
        initial_capital=10_000.0,
        final_capital=10_000.0 * (1 + (total_return_pct / 100.0)),
        total_return_pct=total_return_pct,
        total_trades=total_trades,
        winning_trades=win_count,
        losing_trades=loss_count,
        win_rate_pct=(win_count / total_trades * 100.0) if total_trades else 0.0,
        max_drawdown_pct=max_drawdown_pct,
        profit_factor=profit_factor,
        expectancy_pct=expectancy_pct,
        average_trade_pct=average_trade_pct,
        average_win_pct=average_win_pct,
        average_loss_pct=average_loss_pct,
        largest_win_pct=max((trade.pnl_pct for trade in trades), default=0.0),
        largest_loss_pct=min((trade.pnl_pct for trade in trades), default=0.0),
        long_trades=sum(1 for trade in trades if trade.side == "LONG"),
        long_wins=sum(1 for trade in trades if trade.side == "LONG" and trade.pnl_pct > 0),
        short_trades=sum(1 for trade in trades if trade.side == "SHORT"),
        short_wins=sum(1 for trade in trades if trade.side == "SHORT" and trade.pnl_pct > 0),
        trades=trades,
    )


def _ts(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 1, 1, hour, minute, tzinfo=UTC)


def test_profitable_strategy_passes_reasonable_thresholds():
    trades = [
        _trade(pnl_pct=10.0, entry_time=_ts(9, 0), exit_time=_ts(9, 30)),
        _trade(pnl_pct=12.0, entry_time=_ts(10, 0), exit_time=_ts(10, 30)),
        _trade(pnl_pct=-4.0, entry_time=_ts(11, 0), exit_time=_ts(11, 30)),
        _trade(pnl_pct=8.0, entry_time=_ts(12, 0), exit_time=_ts(12, 30)),
        _trade(pnl_pct=15.0, entry_time=_ts(13, 0), exit_time=_ts(13, 30)),
    ]
    result = _result(trades, max_drawdown_pct=18.0)

    evaluation = StrategyEvaluator().evaluate(
        result,
        policy=StrategyEvaluationPolicy(
            minimum_total_return_pct=5.0,
            minimum_profit_factor=1.2,
            minimum_win_rate_pct=40.0,
            maximum_drawdown_pct=25.0,
            minimum_trade_count=3,
            minimum_sharpe_ratio=0.0,
            minimum_sortino_ratio=0.0,
        ),
    )

    assert evaluation.decision == "PASS"
    assert evaluation.passed
    assert not evaluation.failed_criteria


def test_excessive_drawdown_fails():
    trades = [
        _trade(pnl_pct=20.0, entry_time=_ts(9, 0), exit_time=_ts(9, 30)),
        _trade(pnl_pct=25.0, entry_time=_ts(10, 0), exit_time=_ts(10, 30)),
        _trade(pnl_pct=-30.0, entry_time=_ts(11, 0), exit_time=_ts(11, 30)),
    ]
    result = _result(trades, max_drawdown_pct=35.0)

    evaluation = StrategyEvaluator().evaluate(
        result,
        policy=StrategyEvaluationPolicy(
            minimum_total_return_pct=0.0,
            minimum_profit_factor=1.0,
            minimum_win_rate_pct=0.0,
            maximum_drawdown_pct=20.0,
            minimum_trade_count=1,
        ),
    )

    assert evaluation.decision == "FAIL"
    assert "max_drawdown_pct" in evaluation.failed_criteria


def test_insufficient_trade_count_fails():
    result = _result([
        _trade(pnl_pct=10.0, entry_time=_ts(9, 0), exit_time=_ts(9, 30)),
    ])

    evaluation = StrategyEvaluator().evaluate(
        result,
        policy=StrategyEvaluationPolicy(
            minimum_total_return_pct=0.0,
            minimum_profit_factor=1.0,
            minimum_win_rate_pct=0.0,
            maximum_drawdown_pct=100.0,
            minimum_trade_count=2,
        ),
    )

    assert evaluation.decision == "FAIL"
    assert "minimum_trade_count" in evaluation.failed_criteria


def test_poor_profit_factor_fails():
    trades = [
        _trade(pnl_pct=10.0, entry_time=_ts(9, 0), exit_time=_ts(9, 30)),
        _trade(pnl_pct=-25.0, entry_time=_ts(10, 0), exit_time=_ts(10, 30)),
        _trade(pnl_pct=-15.0, entry_time=_ts(11, 0), exit_time=_ts(11, 30)),
    ]
    result = _result(trades)

    evaluation = StrategyEvaluator().evaluate(
        result,
        policy=StrategyEvaluationPolicy(
            minimum_total_return_pct=0.0,
            minimum_profit_factor=2.0,
            minimum_win_rate_pct=0.0,
            maximum_drawdown_pct=100.0,
            minimum_trade_count=1,
        ),
    )

    assert evaluation.decision == "FAIL"
    assert "profit_factor" in evaluation.failed_criteria


def test_poor_sharpe_fails_when_configured():
    trades = [
        _trade(pnl_pct=2.0, entry_time=_ts(9, 0), exit_time=_ts(9, 30)),
        _trade(pnl_pct=-2.0, entry_time=_ts(10, 0), exit_time=_ts(10, 30)),
        _trade(pnl_pct=-1.0, entry_time=_ts(11, 0), exit_time=_ts(11, 30)),
        _trade(pnl_pct=1.0, entry_time=_ts(12, 0), exit_time=_ts(12, 30)),
    ]
    result = _result(trades)

    evaluation = StrategyEvaluator().evaluate(
        result,
        policy=StrategyEvaluationPolicy(
            minimum_total_return_pct=0.0,
            minimum_profit_factor=1.0,
            minimum_win_rate_pct=0.0,
            maximum_drawdown_pct=100.0,
            minimum_trade_count=1,
            minimum_sharpe_ratio=0.5,
            minimum_sortino_ratio=0.0,
        ),
    )

    assert evaluation.decision == "FAIL"
    assert "sharpe_ratio" in evaluation.failed_criteria


def test_poor_sortino_fails_when_configured():
    trades = [
        _trade(pnl_pct=6.0, entry_time=_ts(9, 0), exit_time=_ts(9, 30)),
        _trade(pnl_pct=-14.0, entry_time=_ts(10, 0), exit_time=_ts(10, 30)),
        _trade(pnl_pct=-13.0, entry_time=_ts(11, 0), exit_time=_ts(11, 30)),
        _trade(pnl_pct=5.0, entry_time=_ts(12, 0), exit_time=_ts(12, 30)),
    ]
    result = _result(trades)

    evaluation = StrategyEvaluator().evaluate(
        result,
        policy=StrategyEvaluationPolicy(
            minimum_total_return_pct=0.0,
            minimum_profit_factor=1.0,
            minimum_win_rate_pct=0.0,
            maximum_drawdown_pct=100.0,
            minimum_trade_count=1,
            minimum_sharpe_ratio=0.0,
            minimum_sortino_ratio=0.5,
        ),
    )

    assert evaluation.decision == "FAIL"
    assert "sortino_ratio" in evaluation.failed_criteria


def test_multiple_failed_criteria_are_all_reported():
    result = _result([
        _trade(pnl_pct=5.0, entry_time=_ts(9, 0), exit_time=_ts(9, 30)),
        _trade(pnl_pct=-10.0, entry_time=_ts(10, 0), exit_time=_ts(10, 30)),
    ], max_drawdown_pct=40.0)

    evaluation = StrategyEvaluator().evaluate(
        result,
        policy=StrategyEvaluationPolicy(
            minimum_total_return_pct=10.0,
            minimum_profit_factor=5.0,
            minimum_win_rate_pct=75.0,
            maximum_drawdown_pct=20.0,
            minimum_trade_count=3,
            minimum_sharpe_ratio=1.0,
            minimum_sortino_ratio=1.0,
        ),
    )

    assert evaluation.decision == "FAIL"
    assert len(evaluation.failed_criteria) >= 3
    assert {"total_return_pct", "profit_factor", "max_drawdown_pct", "minimum_trade_count"}.intersection(evaluation.failed_criteria)


def test_no_trade_backtest_is_handled_deterministically():
    result = _result([], max_drawdown_pct=0.0)

    evaluation = StrategyEvaluator().evaluate(
        result,
        policy=StrategyEvaluationPolicy(),
    )

    assert evaluation.decision == "FAIL"
    assert evaluation.failed_criteria == ["no_trades"]
    assert "No trades were executed" in evaluation.reasons[0]


def test_zero_volatility_return_series_does_not_produce_nan_or_inf():
    trades = [
        _trade(pnl_pct=0.0, entry_time=_ts(9, 0), exit_time=_ts(9, 30)),
        _trade(pnl_pct=0.0, entry_time=_ts(10, 0), exit_time=_ts(10, 30)),
        _trade(pnl_pct=0.0, entry_time=_ts(11, 0), exit_time=_ts(11, 30)),
    ]
    result = _result(trades)

    evaluation = StrategyEvaluator().evaluate(
        result,
        policy=StrategyEvaluationPolicy(
            minimum_total_return_pct=0.0,
            minimum_profit_factor=1.0,
            minimum_win_rate_pct=0.0,
            maximum_drawdown_pct=100.0,
            minimum_trade_count=1,
            minimum_sharpe_ratio=0.1,
            minimum_sortino_ratio=0.1,
        ),
    )

    assert evaluation.metrics["sharpe_ratio"] == pytest.approx(0.0)
    assert evaluation.metrics["sortino_ratio"] == pytest.approx(0.0)
    assert evaluation.decision in {"FAIL", "REVIEW"}


def test_metrics_are_documented_and_deterministic():
    result = _result([
        _trade(pnl_pct=12.0, entry_time=_ts(9, 0), exit_time=_ts(9, 30)),
        _trade(pnl_pct=8.0, entry_time=_ts(10, 0), exit_time=_ts(10, 30)),
        _trade(pnl_pct=-4.0, entry_time=_ts(11, 0), exit_time=_ts(11, 30)),
        _trade(pnl_pct=10.0, entry_time=_ts(12, 0), exit_time=_ts(12, 30)),
    ])

    evaluation = StrategyEvaluator().evaluate(
        result,
        policy=StrategyEvaluationPolicy(
            minimum_total_return_pct=0.0,
            minimum_profit_factor=1.0,
            minimum_win_rate_pct=0.0,
            maximum_drawdown_pct=100.0,
            minimum_trade_count=1,
        ),
    )

    assert "sharpe_ratio" in evaluation.metric_definitions
    assert "sortino_ratio" in evaluation.metric_definitions
    assert evaluation.metric_definitions["sharpe_ratio"].startswith("Non-annualized")
    assert evaluation.metric_definitions["sortino_ratio"].startswith("Non-annualized")

    evaluation_again = StrategyEvaluator().evaluate(result, policy=StrategyEvaluationPolicy())
    assert evaluation_again.decision == evaluation.decision
    assert evaluation_again.metrics == evaluation.metrics
