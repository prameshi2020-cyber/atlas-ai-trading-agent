from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.analysis.analytics import AnalyticsEngine
from app.database.connection import get_connection
from app.market.models import Candle
from app.strategy.engine import StrategyEngine
from app.trading.journal import initialize_trade_journal
from app.risk.manager import RiskManager
from app.risk.models import RiskConfig
from app.strategy.stop_policy import StopPolicy
from app.trading.broker import PaperBroker
from app.trading.execution import ExecutionEngine
from app.trading.position_manager import PositionManager
from app.trading.runner import PaperTradingRunner


TIMESTAMP = datetime.now(timezone.utc)


def make_runner(cash: float = 10_000.0) -> PaperTradingRunner:
    broker = PaperBroker(
        initial_cash=cash,
        fee_rate=0.001,
    )

    risk_manager = RiskManager(
        initial_equity=cash,
        config=RiskConfig(
            risk_per_trade=0.01,
            maximum_position_pct=1.0,
            minimum_position_size=0.0,
            max_concurrent_positions=3,
            daily_loss_limit_pct=0.05,
            max_drawdown_pct=0.10,
        ),
    )

    position_manager = PositionManager()
    engine = ExecutionEngine(
        broker=broker,
        risk_manager=risk_manager,
        position_manager=position_manager,
    )

    initialize_trade_journal()
    return PaperTradingRunner(
        execution_engine=engine,
        stop_policy=StopPolicy(atr_multiplier=1.5),
    )


def setup_function():
    initialize_trade_journal()
    connection = get_connection()
    try:
        connection.execute("DELETE FROM trade_journal")
        connection.commit()
    finally:
        connection.close()


def _get_rows(symbol: str):
    connection = get_connection()
    try:
        rows = connection.execute(
            """
            SELECT
                id,
                trade_id,
                symbol,
                side,
                quantity,
                entry_price,
                exit_price,
                pnl,
                fee,
                status,
                exit_reason
            FROM trade_journal
            WHERE symbol = ?
            ORDER BY id ASC
            """,
            (symbol,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def test_runner_close_writes_exactly_one_journal_row_for_single_close():
    runner = make_runner()

    opened = runner.open(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    assert opened.execution is not None
    assert opened.execution.trade_id is not None

    closed = runner.close(
        symbol="BTCUSDT",
        exit_price=110.0,
        timestamp=TIMESTAMP,
        exit_reason="take_profit",
    )

    assert closed.execution is not None
    assert closed.execution.trade_id == opened.execution.trade_id

    rows = _get_rows("BTCUSDT")

    assert len(rows) == 1
    row = rows[0]
    assert row["trade_id"] == opened.execution.trade_id
    assert row["symbol"] == "BTCUSDT"
    assert row["side"] == "LONG"
    assert row["quantity"] > 0
    assert row["entry_price"] == pytest.approx(100.0)
    assert row["exit_price"] == pytest.approx(110.0)
    assert row["pnl"] > 0
    assert row["fee"] >= 0
    assert row["status"] == "CLOSED"
    assert row["exit_reason"] == "take_profit"


def test_runner_partial_close_writes_one_row_per_actual_exit():
    runner = make_runner()

    opened = runner.open(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )
    assert opened.execution is not None

    partial = runner.execution_engine.close_position(
        symbol="BTCUSDT",
        exit_price=110.0,
        timestamp=TIMESTAMP,
        quantity=10.0,
    )
    assert partial.result.success
    assert partial.trade_id == opened.execution.trade_id

    rows = _get_rows("BTCUSDT")

    assert len(rows) == 1
    assert rows[0]["trade_id"] == opened.execution.trade_id
    assert rows[0]["quantity"] == pytest.approx(10.0)
    assert rows[0]["entry_price"] == pytest.approx(100.0)
    assert rows[0]["exit_price"] == pytest.approx(110.0)

    report = AnalyticsEngine().build_report()
    assert report.total_trades == 1
    assert report.partial_exit_count == 0


def test_runner_reversal_does_not_duplicate_close_journal_entry():
    runner = make_runner()

    opened = runner.open(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )
    assert opened.execution is not None
    assert opened.execution.trade_id is not None

    candles = [
        Candle(
            symbol="BTCUSDT",
            timeframe="1m",
            open=100.0,
            high=110.0,
            low=90.0,
            close=100.0,
            volume=1000.0,
            timestamp=TIMESTAMP - timedelta(minutes=1),
        ),
        Candle(
            symbol="BTCUSDT",
            timeframe="1m",
            open=101.0,
            high=111.0,
            low=91.0,
            close=110.0,
            volume=1000.0,
            timestamp=TIMESTAMP,
        ),
        Candle(
            symbol="BTCUSDT",
            timeframe="1m",
            open=112.0,
            high=113.0,
            low=90.0,
            close=112.0,
            volume=1000.0,
            timestamp=TIMESTAMP + timedelta(minutes=1),
        ),
        Candle(
            symbol="BTCUSDT",
            timeframe="1m",
            open=115.0,
            high=116.0,
            low=92.0,
            close=115.0,
            volume=1000.0,
            timestamp=TIMESTAMP + timedelta(minutes=2),
        ),
        Candle(
            symbol="BTCUSDT",
            timeframe="1m",
            open=117.0,
            high=120.0,
            low=95.0,
            close=117.0,
            volume=1000.0,
            timestamp=TIMESTAMP + timedelta(minutes=3),
        ),
    ]

    strategy = StrategyEngine(
        fast_period=2,
        slow_period=4,
        atr_period=2,
    )
    signal = strategy.generate_signal(
        symbol="BTCUSDT",
        candles=candles,
    )
    assert signal.side in {"LONG", "SHORT", "FLAT"}

    result = runner.process_candles(
        symbol="BTCUSDT",
        candles=candles,
        strategy=strategy,
        timestamp=TIMESTAMP,
    )

    assert result.execution is None or result.execution.result.success or result.action in {"HOLD", "REJECT"}

    rows = _get_rows("BTCUSDT")
    assert len(rows) == 1
    assert rows[0]["trade_id"] == opened.execution.trade_id


def test_runner_close_uses_execution_trade_id_for_analytics_lifecycle_count():
    runner = make_runner()

    opened = runner.open(
        symbol="ETHUSDT",
        side="LONG",
        entry_price=2000.0,
        stop_price=1900.0,
        timestamp=TIMESTAMP,
    )
    assert opened.execution is not None

    closed = runner.close(
        symbol="ETHUSDT",
        exit_price=2100.0,
        timestamp=TIMESTAMP,
        exit_reason="take_profit",
    )
    assert closed.execution is not None
    assert closed.execution.trade_id == opened.execution.trade_id

    report = AnalyticsEngine().build_report()

    assert report.total_trades == 1
    assert report.winning_trades == 1
    assert report.trades[0].trade_id == opened.execution.trade_id
    assert report.trades[0].exit_count == 1


def test_runner_uses_latest_candle_close_for_entry_and_strategy_signal_contract():
    candles = []
    for index, close in enumerate([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0]):
        candles.append(
            Candle(
                symbol="BTCUSDT",
                timeframe="1m",
                open=close,
                high=close + 1.0,
                low=close - 1.0,
                close=close,
                volume=1000.0,
                timestamp=TIMESTAMP - timedelta(minutes=(len(candles) - index)),
            )
        )

    signal = StrategyEngine(
        fast_period=3,
        slow_period=5,
        atr_period=2,
    ).generate_signal(
        symbol="BTCUSDT",
        candles=candles,
    )

    assert signal.symbol == "BTCUSDT"
    assert signal.price == pytest.approx(106.0)
    assert signal.side in {"LONG", "SHORT", "FLAT"}
    assert signal.atr is not None

    runner = make_runner()
    result = runner.process_candles(
        symbol="BTCUSDT",
        candles=candles,
        strategy=StrategyEngine(
            fast_period=3,
            slow_period=5,
            atr_period=2,
        ),
        timestamp=TIMESTAMP,
    )

    assert result.action in {"OPEN", "HOLD", "REJECT"}
    assert result.execution is None or result.execution.result.success
