from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.database.connection import get_connection
from app.analysis.analytics import AnalyticsEngine
from app.trading.journal import initialize_trade_journal, save_trade


UTC = timezone.utc


def setup_function():
    initialize_trade_journal()
    connection = get_connection()
    try:
        connection.execute("DELETE FROM trade_journal")
        connection.commit()
    finally:
        connection.close()


def _ts(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 1, 1, hour, minute, tzinfo=UTC)


def test_empty_journal_has_zero_metrics():
    report = AnalyticsEngine().build_report()

    assert report.total_trades == 0
    assert report.winning_trades == 0
    assert report.losing_trades == 0
    assert report.win_rate_pct == 0.0
    assert report.gross_profit == 0.0
    assert report.gross_loss == 0.0
    assert report.net_pnl == 0.0
    assert report.profit_factor == 0.0
    assert report.expectancy == 0.0
    assert report.average_win == 0.0
    assert report.average_loss == 0.0
    assert report.largest_win == 0.0
    assert report.largest_loss == 0.0
    assert report.long_trades == 0
    assert report.long_wins == 0
    assert report.short_trades == 0
    assert report.short_wins == 0
    assert report.partial_exit_count == 0
    assert report.average_holding_time_seconds == 0.0
    assert report.total_fees == 0.0
    assert report.trades == []


def test_single_winning_full_close_trade():
    save_trade(
        trade_id="ABC",
        symbol="BTCUSDT",
        side="LONG",
        quantity=10.0,
        entry_price=100.0,
        exit_price=110.0,
        entry_time=_ts(9, 0),
        exit_time=_ts(10, 0),
        pnl=100.0,
        fee=1.0,
    )

    report = AnalyticsEngine().build_report()

    assert report.total_trades == 1
    assert report.winning_trades == 1
    assert report.losing_trades == 0
    assert report.win_rate_pct == pytest.approx(100.0)
    assert report.gross_profit == pytest.approx(100.0)
    assert report.gross_loss == pytest.approx(0.0)
    assert report.net_pnl == pytest.approx(100.0)
    assert report.profit_factor == pytest.approx(float("inf"))
    assert report.expectancy == pytest.approx(100.0)
    assert report.average_win == pytest.approx(100.0)
    assert report.average_loss == pytest.approx(0.0)
    assert report.largest_win == pytest.approx(100.0)
    assert report.largest_loss == pytest.approx(0.0)
    assert report.long_trades == 1
    assert report.long_wins == 1
    assert report.short_trades == 0
    assert report.short_wins == 0
    assert report.partial_exit_count == 0
    assert report.total_fees == pytest.approx(1.0)
    assert report.trades[0].trade_id == "ABC"


def test_single_losing_trade():
    save_trade(
        trade_id="DEF",
        symbol="ETHUSDT",
        side="SHORT",
        quantity=5.0,
        entry_price=2000.0,
        exit_price=2100.0,
        entry_time=_ts(8, 0),
        exit_time=_ts(9, 30),
        pnl=-50.0,
        fee=2.0,
    )

    report = AnalyticsEngine().build_report()

    assert report.total_trades == 1
    assert report.winning_trades == 0
    assert report.losing_trades == 1
    assert report.win_rate_pct == pytest.approx(0.0)
    assert report.gross_profit == pytest.approx(0.0)
    assert report.gross_loss == pytest.approx(50.0)
    assert report.net_pnl == pytest.approx(-50.0)
    assert report.profit_factor == pytest.approx(0.0)
    assert report.expectancy == pytest.approx(-50.0)
    assert report.average_win == pytest.approx(0.0)
    assert report.average_loss == pytest.approx(-50.0)
    assert report.largest_win == pytest.approx(0.0)
    assert report.largest_loss == pytest.approx(-50.0)
    assert report.short_trades == 1
    assert report.short_wins == 0
    assert report.total_fees == pytest.approx(2.0)


def test_mixed_winners_and_losers():
    save_trade(
        trade_id="WIN1",
        symbol="BTCUSDT",
        side="LONG",
        quantity=2.0,
        entry_price=100.0,
        exit_price=110.0,
        entry_time=_ts(9, 0),
        exit_time=_ts(10, 0),
        pnl=20.0,
        fee=1.0,
    )
    save_trade(
        trade_id="LOSS1",
        symbol="SOLUSDT",
        side="LONG",
        quantity=3.0,
        entry_price=50.0,
        exit_price=45.0,
        entry_time=_ts(11, 0),
        exit_time=_ts(12, 0),
        pnl=-15.0,
        fee=1.5,
    )

    report = AnalyticsEngine().build_report()

    assert report.total_trades == 2
    assert report.winning_trades == 1
    assert report.losing_trades == 1
    assert report.win_rate_pct == pytest.approx(50.0)
    assert report.gross_profit == pytest.approx(20.0)
    assert report.gross_loss == pytest.approx(15.0)
    assert report.net_pnl == pytest.approx(5.0)
    assert report.profit_factor == pytest.approx(20.0 / 15.0)
    assert report.expectancy == pytest.approx(2.5)
    assert report.average_win == pytest.approx(20.0)
    assert report.average_loss == pytest.approx(-15.0)
    assert report.total_fees == pytest.approx(2.5)


def test_long_and_short_trade_counts_and_wins():
    save_trade(
        trade_id="LONG_WIN",
        symbol="BTCUSDT",
        side="LONG",
        quantity=1.0,
        entry_price=100.0,
        exit_price=105.0,
        entry_time=_ts(9, 0),
        exit_time=_ts(9, 30),
        pnl=5.0,
        fee=0.5,
    )
    save_trade(
        trade_id="SHORT_WIN",
        symbol="ETHUSDT",
        side="SHORT",
        quantity=2.0,
        entry_price=1000.0,
        exit_price=950.0,
        entry_time=_ts(10, 0),
        exit_time=_ts(11, 0),
        pnl=25.0,
        fee=1.0,
    )
    save_trade(
        trade_id="SHORT_LOSS",
        symbol="ADAUSDT",
        side="SHORT",
        quantity=3.0,
        entry_price=200.0,
        exit_price=210.0,
        entry_time=_ts(12, 0),
        exit_time=_ts(12, 30),
        pnl=-10.0,
        fee=0.25,
    )

    report = AnalyticsEngine().build_report()

    assert report.long_trades == 1
    assert report.long_wins == 1
    assert report.short_trades == 2
    assert report.short_wins == 1
    assert report.winning_trades == 2
    assert report.losing_trades == 1


def test_partial_exits_share_one_trade_lifecycle():
    save_trade(
        trade_id="ABC",
        symbol="BTCUSDT",
        side="LONG",
        quantity=10.0,
        entry_price=100.0,
        exit_price=110.0,
        entry_time=_ts(9, 0),
        exit_time=_ts(9, 10),
        pnl=50.0,
        fee=1.0,
    )
    save_trade(
        trade_id="ABC",
        symbol="BTCUSDT",
        side="LONG",
        quantity=10.0,
        entry_price=100.0,
        exit_price=120.0,
        entry_time=_ts(9, 0),
        exit_time=_ts(9, 20),
        pnl=40.0,
        fee=1.0,
    )

    report = AnalyticsEngine().build_report()

    assert report.total_trades == 1
    assert report.winning_trades == 1
    assert report.net_pnl == pytest.approx(90.0)
    assert report.total_fees == pytest.approx(2.0)
    assert report.trades[0].exit_count == 2
    assert report.partial_exit_count == 1
    assert report.win_rate_pct == pytest.approx(100.0)
    assert report.gross_profit == pytest.approx(90.0)


def test_lifecycle_holding_time_and_fee_summary():
    save_trade(
        trade_id="HOLD1",
        symbol="BTCUSDT",
        side="LONG",
        quantity=2.0,
        entry_price=100.0,
        exit_price=108.0,
        entry_time=_ts(10, 0),
        exit_time=_ts(12, 15),
        pnl=16.0,
        fee=0.5,
    )
    save_trade(
        trade_id="HOLD2",
        symbol="ETHUSDT",
        side="SHORT",
        quantity=3.0,
        entry_price=2000.0,
        exit_price=1900.0,
        entry_time=_ts(11, 0),
        exit_time=_ts(11, 30),
        pnl=-40.0,
        fee=2.0,
    )

    report = AnalyticsEngine().build_report()

    assert len(report.trades) == 2
    holding_times = [trade.holding_time_seconds for trade in report.trades]
    assert holding_times[0] == pytest.approx(8100.0)
    assert holding_times[1] == pytest.approx(1800.0)
    assert report.average_holding_time_seconds == pytest.approx(4950.0)
    assert report.total_fees == pytest.approx(2.5)


def test_zero_gross_loss_produces_infinite_profit_factor():
    save_trade(
        trade_id="ONLY_WIN",
        symbol="BTCUSDT",
        side="LONG",
        quantity=1.0,
        entry_price=100.0,
        exit_price=110.0,
        entry_time=_ts(9, 0),
        exit_time=_ts(9, 30),
        pnl=10.0,
        fee=0.5,
    )

    report = AnalyticsEngine().build_report()

    assert report.gross_loss == pytest.approx(0.0)
    assert report.profit_factor == pytest.approx(float("inf"))


def test_legacy_null_trade_id_is_treated_as_its_own_lifecycle():
    save_trade(
        trade_id=None,
        symbol="XRPUSDT",
        side="LONG",
        quantity=10.0,
        entry_price=1.0,
        exit_price=1.2,
        entry_time=_ts(8, 0),
        exit_time=_ts(8, 30),
        pnl=2.0,
        fee=0.1,
    )
    save_trade(
        trade_id=None,
        symbol="DOGEUSDT",
        side="SHORT",
        quantity=20.0,
        entry_price=0.5,
        exit_price=0.4,
        entry_time=_ts(9, 0),
        exit_time=_ts(9, 15),
        pnl=-3.0,
        fee=0.2,
    )

    report = AnalyticsEngine().build_report()

    assert report.total_trades == 2
    assert report.partial_exit_count == 0
    assert report.net_pnl == pytest.approx(-1.0)
    assert report.winning_trades == 1
    assert report.losing_trades == 1


def test_invalid_journal_data_is_rejected():
    save_trade(
        trade_id="BAD",
        symbol="",
        side="LONG",
        quantity=0.0,
        entry_price=0.0,
        exit_price=10.0,
        entry_time=_ts(9, 0),
        exit_time=_ts(9, 15),
        pnl=5.0,
        fee=0.5,
    )

    with pytest.raises(ValueError):
        AnalyticsEngine().build_report()


def test_journal_row_count_does_not_equal_lifecycle_count():
    save_trade(
        trade_id="ROWGROUP",
        symbol="BTCUSDT",
        side="LONG",
        quantity=10.0,
        entry_price=100.0,
        exit_price=105.0,
        entry_time=_ts(9, 0),
        exit_time=_ts(9, 5),
        pnl=20.0,
        fee=1.0,
    )
    save_trade(
        trade_id="ROWGROUP",
        symbol="BTCUSDT",
        side="LONG",
        quantity=10.0,
        entry_price=100.0,
        exit_price=110.0,
        entry_time=_ts(9, 0),
        exit_time=_ts(9, 15),
        pnl=30.0,
        fee=1.0,
    )

    report = AnalyticsEngine().build_report()

    assert len(report.trades) == 1
    assert report.total_trades == 1
    assert report.partial_exit_count == 1
    assert report.net_pnl == pytest.approx(50.0)
