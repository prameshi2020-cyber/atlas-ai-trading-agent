"""
Deterministic tests for BacktestEngine, using synthetic OHLCV data and
a scripted strategy double instead of live data or the real
StrategyEngine. This isolates the backtest engine's execution
mechanics (entry-bar SL/TP, PnL wiring, causal slicing) from
StrategyEngine's own scoring logic, which this milestone explicitly
does not touch.
"""

import pandas as pd
import pytest

from app.analysis.backtest import BacktestEngine
from app.analysis.strategy import Signal
from app.analysis.indicators import REQUIRED_FEATURE_COLUMNS


# ---------------------------------------------------------------------
# Test doubles / helpers
# ---------------------------------------------------------------------


class ScriptedStrategy:
    """
    Returns a pre-scripted sequence of signal actions, one per call,
    regardless of feature content. Used to drive BacktestEngine
    through specific, known execution scenarios without depending on
    (or exercising) StrategyEngine's own scoring rules.
    """

    def __init__(self, actions: list[str]) -> None:
        self._actions = actions
        self.calls: list[pd.DataFrame] = []

    def generate_signal(self, symbol: str, features: pd.DataFrame) -> Signal:
        self.calls.append(features)
        action = self._actions[len(self.calls) - 1]
        return Signal(
            symbol=symbol,
            action=action,
            score=0.0,
            confidence=0.0,
            reasons=["scripted"],
        )


def make_row(ts, o, h, l, c, atr, v=10.0) -> dict:
    """
    One synthetic OHLCV + feature row with every column
    BacktestEngine/StrategyEngine.generate_signal could touch filled
    with a valid (non-NaN) dummy value, so drop_warmup_rows() never
    strips it out.
    """
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


TS = pd.date_range("2026-01-01", periods=10, freq="1min", tz="UTC")


# ---------------------------------------------------------------------
# B: entry-bar stop-loss trigger
# ---------------------------------------------------------------------


def test_entry_bar_stop_loss_triggers_on_the_same_bar_as_entry():
    rows = [
        make_row(TS[0], 100, 100, 100, 100, atr=2.0),
        # Entry bar: opens at 100, low dips to 95 -> breaches
        # stop_price = 100 - 2*1.5 = 97 within THIS same bar.
        make_row(TS[1], 100, 101, 95, 99, atr=2.0),
        make_row(TS[2], 99, 99, 99, 99, atr=2.0),
    ]
    features = make_df(rows)

    strategy = ScriptedStrategy(["BUY", "HOLD"])
    engine = BacktestEngine(
        strategy=strategy,
        initial_capital=10_000.0,
        fee_pct=0.0,
        slippage_pct=0.0,
        stop_loss_atr=1.5,
        take_profit_atr=3.0,
    )

    result = engine.run(symbol="TEST", features=features)

    assert result.total_trades == 1
    trade = result.trades[0]

    assert trade.side == "LONG"
    assert trade.exit_reason == "STOP_LOSS"
    # The bug being fixed: entry and exit both happened on the SAME
    # bar (the bar the position was opened on), not one bar later.
    assert trade.entry_time == trade.exit_time == TS[1]
    assert trade.entry_price == pytest.approx(100.0)
    assert trade.exit_price == pytest.approx(97.0)
    assert trade.pnl_pct == pytest.approx(-3.0)
    assert result.final_capital == pytest.approx(9_700.0)


# ---------------------------------------------------------------------
# C: entry-bar take-profit trigger
# ---------------------------------------------------------------------


def test_entry_bar_take_profit_triggers_on_the_same_bar_as_entry():
    rows = [
        make_row(TS[0], 100, 100, 100, 100, atr=2.0),
        # Entry bar: opens at 100, high spikes to 110 -> breaches
        # target_price = 100 + 2*3 = 106 within THIS same bar. Low
        # only dips to 99, so stop_price=97 is never touched.
        make_row(TS[1], 100, 110, 99, 105, atr=2.0),
        make_row(TS[2], 105, 105, 105, 105, atr=2.0),
    ]
    features = make_df(rows)

    strategy = ScriptedStrategy(["BUY", "HOLD"])
    engine = BacktestEngine(
        strategy=strategy,
        initial_capital=10_000.0,
        fee_pct=0.0,
        slippage_pct=0.0,
        stop_loss_atr=1.5,
        take_profit_atr=3.0,
    )

    result = engine.run(symbol="TEST", features=features)

    assert result.total_trades == 1
    trade = result.trades[0]

    assert trade.side == "LONG"
    assert trade.exit_reason == "TAKE_PROFIT"
    assert trade.entry_time == trade.exit_time == TS[1]
    assert trade.exit_price == pytest.approx(106.0)
    assert trade.pnl_pct == pytest.approx(6.0)


# ---------------------------------------------------------------------
# D: ambiguous same-bar SL + TP -> conservative STOP_LOSS priority
# ---------------------------------------------------------------------


def test_ambiguous_same_bar_sl_and_tp_resolves_to_stop_loss():
    rows = [
        make_row(TS[0], 100, 100, 100, 100, atr=2.0),
        # Entry bar's range covers BOTH stop_price=97 (low=90) AND
        # target_price=106 (high=110). OHLC data cannot tell us which
        # was touched first -- the engine must not assume the
        # favorable (TAKE_PROFIT) outcome.
        make_row(TS[1], 100, 110, 90, 100, atr=2.0),
        make_row(TS[2], 100, 100, 100, 100, atr=2.0),
    ]
    features = make_df(rows)

    strategy = ScriptedStrategy(["BUY", "HOLD"])
    engine = BacktestEngine(
        strategy=strategy,
        initial_capital=10_000.0,
        fee_pct=0.0,
        slippage_pct=0.0,
        stop_loss_atr=1.5,
        take_profit_atr=3.0,
    )

    result = engine.run(symbol="TEST", features=features)

    assert result.total_trades == 1
    trade = result.trades[0]

    assert trade.exit_reason == "STOP_LOSS", (
        "Ambiguous same-bar SL+TP must conservatively resolve to "
        "STOP_LOSS, never TAKE_PROFIT."
    )
    assert trade.exit_price == pytest.approx(97.0)
    assert trade.pnl_pct == pytest.approx(-3.0)


# Same ambiguity check, but for a SHORT position, to make sure the
# conservative policy isn't accidentally LONG-only.
def test_ambiguous_same_bar_sl_and_tp_short_resolves_to_stop_loss():
    rows = [
        make_row(TS[0], 100, 100, 100, 100, atr=2.0),
        # SHORT entry at 100: stop_price = 100 + 2*1.5 = 103,
        # target_price = 100 - 2*3 = 94. This bar's range covers
        # both (high=110 >= 103, low=90 <= 94).
        make_row(TS[1], 100, 110, 90, 100, atr=2.0),
        make_row(TS[2], 100, 100, 100, 100, atr=2.0),
    ]
    features = make_df(rows)

    strategy = ScriptedStrategy(["SELL", "HOLD"])
    engine = BacktestEngine(
        strategy=strategy,
        initial_capital=10_000.0,
        fee_pct=0.0,
        slippage_pct=0.0,
        stop_loss_atr=1.5,
        take_profit_atr=3.0,
    )

    result = engine.run(symbol="TEST", features=features)

    assert result.total_trades == 1
    trade = result.trades[0]

    assert trade.side == "SHORT"
    assert trade.exit_reason == "STOP_LOSS"
    assert trade.exit_price == pytest.approx(103.0)


# ---------------------------------------------------------------------
# G: long PnL correctness (fees + slippage wired correctly)
# ---------------------------------------------------------------------


def test_long_pnl_uses_correct_entry_exit_prices_and_formula():
    rows = [
        make_row(TS[0], 100, 100, 100, 100, atr=2.0),
        make_row(TS[1], 100, 101, 99, 100.5, atr=2.0),   # entry bar
        make_row(TS[2], 100.5, 101, 100, 100.5, atr=2.0),  # hold bar
        make_row(TS[3], 110, 111, 109, 110, atr=2.0),      # exit bar (reversal)
    ]
    features = make_df(rows)

    strategy = ScriptedStrategy(["BUY", "HOLD", "SELL"])
    engine = BacktestEngine(
        strategy=strategy,
        initial_capital=10_000.0,
        fee_pct=0.10,
        slippage_pct=0.05,
        stop_loss_atr=None,   # isolate signal-driven PnL from SL/TP
        take_profit_atr=None,
    )

    result = engine.run(symbol="TEST", features=features)

    long_trades = [t for t in result.trades if t.side == "LONG"]
    assert len(long_trades) == 1
    trade = long_trades[0]

    expected_entry = engine._apply_entry_slippage("LONG", 100.0)   # row1 open
    expected_exit = engine._apply_exit_slippage("LONG", 110.0)     # row3 open
    expected_pnl = engine._calculate_pnl("LONG", expected_entry, expected_exit)

    assert trade.entry_price == pytest.approx(expected_entry)
    assert trade.exit_price == pytest.approx(expected_exit)
    assert trade.pnl_pct == pytest.approx(expected_pnl)
    assert trade.exit_reason == "SIGNAL"
    assert expected_pnl > 0  # sanity: price rose 100 -> 110, LONG should profit


# ---------------------------------------------------------------------
# H: short PnL correctness
# ---------------------------------------------------------------------


def test_short_pnl_uses_correct_entry_exit_prices_and_formula():
    rows = [
        make_row(TS[0], 100, 100, 100, 100, atr=2.0),
        make_row(TS[1], 100, 101, 99, 100.5, atr=2.0),     # entry bar
        make_row(TS[2], 100.5, 101, 100, 100.5, atr=2.0),  # hold bar
        make_row(TS[3], 90, 91, 89, 90, atr=2.0),           # exit bar (reversal)
    ]
    features = make_df(rows)

    strategy = ScriptedStrategy(["SELL", "HOLD", "BUY"])
    engine = BacktestEngine(
        strategy=strategy,
        initial_capital=10_000.0,
        fee_pct=0.10,
        slippage_pct=0.05,
        stop_loss_atr=None,
        take_profit_atr=None,
    )

    result = engine.run(symbol="TEST", features=features)

    short_trades = [t for t in result.trades if t.side == "SHORT"]
    assert len(short_trades) == 1
    trade = short_trades[0]

    expected_entry = engine._apply_entry_slippage("SHORT", 100.0)  # row1 open
    expected_exit = engine._apply_exit_slippage("SHORT", 90.0)     # row3 open
    expected_pnl = engine._calculate_pnl("SHORT", expected_entry, expected_exit)

    assert trade.entry_price == pytest.approx(expected_entry)
    assert trade.exit_price == pytest.approx(expected_exit)
    assert trade.pnl_pct == pytest.approx(expected_pnl)
    assert trade.exit_reason == "SIGNAL"
    assert expected_pnl > 0  # sanity: price fell 100 -> 90, SHORT should profit


# ---------------------------------------------------------------------
# I: no look-ahead regression
# ---------------------------------------------------------------------


def test_strategy_never_sees_rows_beyond_the_current_closed_candle():
    rows = [
        make_row(TS[i], 100 + i, 101 + i, 99 + i, 100 + i, atr=2.0)
        for i in range(6)
    ]
    features = make_df(rows)

    strategy = ScriptedStrategy(["HOLD"] * 5)
    engine = BacktestEngine(
        strategy=strategy,
        initial_capital=10_000.0,
        fee_pct=0.0,
        slippage_pct=0.0,
        stop_loss_atr=1.5,
        take_profit_atr=3.0,
    )

    engine.run(symbol="TEST", features=features)

    # 6 rows -> 5 iterations -> 5 calls to generate_signal.
    assert len(strategy.calls) == 5

    for i, seen in enumerate(strategy.calls):
        # At iteration i, the strategy must see exactly rows 0..i
        # (i+1 rows total) and never anything from row i+1 onward.
        assert len(seen) == i + 1
        assert seen["timestamp"].iloc[-1] == TS[i]
        assert (seen["timestamp"] <= TS[i]).all()
