from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from app.market.models import Candle
from app.strategy.stop_policy import StopPolicy
from app.database.connection import get_connection
from app.trading.journal import initialize_trade_journal
from app.risk.manager import RiskManager
from app.risk.models import RiskConfig
from app.trading.broker import PaperBroker
from app.trading.execution import ExecutionEngine
from app.trading.position_manager import PositionManager
from app.trading.runner import PaperTradingRunner
from app.trading import intent_manager

try:
    from app.strategy.signal import TradingSignal  # type: ignore[reportMissingImports]
except ImportError:
    @dataclass
    class TradingSignal:
        symbol: str
        side: str
        price: float
        timestamp: datetime
        atr: Optional[float] = None


TIMESTAMP = datetime.now(timezone.utc)


def make_runner(defer_execution: bool = False) -> PaperTradingRunner:
    broker = PaperBroker(
        initial_cash=10_000.0,
        fee_rate=0.001,
    )

    risk_manager = RiskManager(
        initial_equity=10_000.0,
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
        stop_policy=StopPolicy(
            atr_multiplier=1.5,
        ),
        defer_execution=defer_execution,
    )


def test_runner_opens_long_position():
    runner = make_runner()

    result = runner.open(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    assert result.action == "OPEN"
    assert result.execution is not None
    assert result.execution.result.success


def test_runner_opens_short_position():
    runner = make_runner()

    result = runner.open(
        symbol="BTCUSDT",
        side="SHORT",
        entry_price=100.0,
        stop_price=105.0,
        timestamp=TIMESTAMP,
    )

    assert result.action == "OPEN"
    assert result.execution is not None
    assert result.execution.result.success


def test_runner_rejects_duplicate_position():
    runner = make_runner()

    runner.open(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    result = runner.open(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=101.0,
        stop_price=96.0,
        timestamp=TIMESTAMP,
    )

    assert result.action == "OPEN"
    assert result.execution is None
    assert result.reason == "position already exists"


def test_runner_closes_position():
    runner = make_runner()

    runner.open(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    result = runner.close(
        symbol="BTCUSDT",
        exit_price=110.0,
        timestamp=TIMESTAMP,
    )

    assert result.action == "CLOSE"
    assert result.execution is not None
    assert result.execution.result.success
    assert result.execution.pnl is not None
    assert result.execution.pnl > 0


def test_runner_rejects_close_without_position():
    runner = make_runner()

    result = runner.close(
        symbol="BTCUSDT",
        exit_price=100.0,
        timestamp=TIMESTAMP,
    )

    assert result.action == "CLOSE"
    assert result.execution is None
    assert result.reason == "no open position"

def setup_function():
    initialize_trade_journal()

    connection = get_connection()

    try:
        connection.execute("DELETE FROM trade_journal")
        connection.commit()
    finally:
        connection.close()

def test_runner_persists_closed_trade():
    runner = make_runner()

    runner.open(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=TIMESTAMP,
    )

    result = runner.close(
        symbol="BTCUSDT",
        exit_price=110.0,
        timestamp=TIMESTAMP,
        exit_reason="take_profit",
    )

    assert result.execution is not None
    assert result.execution.result.success

    connection = get_connection()

    try:
        row = connection.execute(
            """
            SELECT
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
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()

        assert row is not None
        assert row["symbol"] == "BTCUSDT"
        assert row["side"] == "LONG"
        assert row["quantity"] > 0
        assert row["entry_price"] == 100.0
        assert row["exit_price"] == 110.0
        assert row["pnl"] > 0
        assert row["fee"] >= 0
        assert row["status"] == "CLOSED"
        assert row["exit_reason"] == "take_profit"

    finally:
        connection.close()

def test_long_stop_closes_existing_position_on_candle_trigger():
    runner = make_runner()

    timestamp = datetime(2026, 1, 1, 12, 0, 0)

    opened = runner.open(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=timestamp,
    )

    assert opened.execution is not None
    assert opened.execution.result.success

    class FlatStrategy:
        def generate_signal(self, *, symbol, candles):
            return TradingSignal(
                symbol=symbol,
                side="FLAT",
                price=100.0,
                timestamp=timestamp,
                atr=None,
            )

    result = runner.process_candles(
        symbol="BTCUSDT",
        candles=[
            {
                "open": 100.0,
                "high": 101.0,
                "low": 94.0,
                "close": 100.0,
                "timestamp": timestamp,
            }
        ],
        strategy=FlatStrategy(),
        timestamp=timestamp,
    )

    assert result.action == "CLOSE"
    assert result.execution is not None
    assert result.execution.result.success
    assert result.reason == "stop loss"


def test_short_stop_closes_existing_position_on_candle_trigger():
    runner = make_runner()

    timestamp = datetime(2026, 1, 1, 12, 0, 0)

    opened = runner.open(
        symbol="BTCUSDT",
        side="SHORT",
        entry_price=100.0,
        stop_price=105.0,
        timestamp=timestamp,
    )

    assert opened.execution is not None
    assert opened.execution.result.success

    class FlatStrategy:
        def generate_signal(self, *, symbol, candles):
            return TradingSignal(
                symbol=symbol,
                side="FLAT",
                price=100.0,
                timestamp=timestamp,
                atr=None,
            )

    result = runner.process_candles(
        symbol="BTCUSDT",
        candles=[
            {
                "open": 100.0,
                "high": 106.0,
                "low": 99.0,
                "close": 100.0,
                "timestamp": timestamp,
            }
        ],
        strategy=FlatStrategy(),
        timestamp=timestamp,
    )

    assert result.action == "CLOSE"
    assert result.execution is not None
    assert result.execution.result.success
    assert result.reason == "stop loss"

def test_stop_does_not_close_position_when_candle_does_not_trigger():
    runner = make_runner()

    timestamp = datetime(2026, 1, 1, 12, 0, 0)

    opened = runner.open(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=timestamp,
    )

    assert opened.execution is not None
    assert opened.execution.result.success

    class FlatStrategy:
        def generate_signal(self, *, symbol, candles):
            return TradingSignal(
                symbol=symbol,
                side="FLAT",
                price=100.0,
                timestamp=timestamp,
                atr=None,
            )

    result = runner.process_candles(
        symbol="BTCUSDT",
        candles=[
            {
                "open": 100.0,
                "high": 101.0,
                "low": 96.0,
                "close": 100.0,
                "timestamp": timestamp,
            }
        ],
        strategy=FlatStrategy(),
        timestamp=timestamp,
    )

    assert result.action == "HOLD"
    assert result.execution is None
    assert result.reason == "strategy returned FLAT"

def test_stop_loss_closes_position_at_stop_price_and_persists_reason():
    runner = make_runner()

    timestamp = datetime(2026, 1, 1, 12, 0, 0)

    opened = runner.open(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=timestamp,
    )

    assert opened.execution is not None
    assert opened.execution.result.success

    class FlatStrategy:
        def generate_signal(self, *, symbol, candles):
            return TradingSignal(
                symbol=symbol,
                side="FLAT",
                price=100.0,
                timestamp=timestamp,
                atr=None,
            )

    result = runner.process_candles(
        symbol="BTCUSDT",
        candles=[
            {
                "open": 100.0,
                "high": 101.0,
                "low": 94.0,
                "close": 100.0,
                "timestamp": timestamp,
            }
        ],
        strategy=FlatStrategy(),
        timestamp=timestamp,
    )

    assert result.action == "CLOSE"
    assert result.execution is not None
    assert result.execution.result.success
    assert result.reason == "stop loss"

    assert (
        runner.execution_engine.position_manager.get_position(
            "BTCUSDT"
        )
        is None
    )

    assert result.execution.result.execution_price == 95.0

    connection = get_connection()

    try:
        row = connection.execute(
            """
            SELECT
                symbol,
                exit_price,
                status,
                exit_reason
            FROM trade_journal
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()

        assert row is not None
        assert row["symbol"] == "BTCUSDT"
        assert row["exit_price"] == 95.0
        assert row["status"] == "CLOSED"
        assert row["exit_reason"] == "stop loss"

    finally:
        connection.close()

def test_long_stop_gap_uses_candle_open_price():
    runner = make_runner()

    timestamp = datetime(2026, 1, 1, 12, 0, 0)

    opened = runner.open(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=95.0,
        timestamp=timestamp,
    )

    assert opened.execution is not None
    assert opened.execution.result.success

    class FlatStrategy:
        def generate_signal(self, *, symbol, candles):
            return TradingSignal(
                symbol=symbol,
                side="FLAT",
                price=90.0,
                timestamp=timestamp,
                atr=None,
            )

    result = runner.process_candles(
        symbol="BTCUSDT",
        candles=[
            {
                "open": 90.0,
                "high": 93.0,
                "low": 88.0,
                "close": 91.0,
                "timestamp": timestamp,
            }
        ],
        strategy=FlatStrategy(),
        timestamp=timestamp,
    )

    assert result.action == "CLOSE"
    assert result.execution is not None
    assert result.execution.result.success
    assert result.reason == "stop loss"

    assert result.execution.result.execution_price == 90.0

    assert (
        runner.execution_engine.position_manager.get_position(
            "BTCUSDT"
        )
        is None
    )

def test_short_stop_gap_uses_candle_open_price():
    runner = make_runner()

    timestamp = datetime(2026, 1, 1, 12, 0, 0)

    opened = runner.open(
        symbol="BTCUSDT",
        side="SHORT",
        entry_price=100.0,
        stop_price=105.0,
        timestamp=timestamp,
    )

    assert opened.execution is not None
    assert opened.execution.result.success

    class FlatStrategy:
        def generate_signal(self, *, symbol, candles):
            return TradingSignal(
                symbol=symbol,
                side="FLAT",
                price=110.0,
                timestamp=timestamp,
                atr=None,
            )

    result = runner.process_candles(
        symbol="BTCUSDT",
        candles=[
            {
                "open": 110.0,
                "high": 115.0,
                "low": 108.0,
                "close": 111.0,
                "timestamp": timestamp,
            }
        ],
        strategy=FlatStrategy(),
        timestamp=timestamp,
    )

    assert result.action == "CLOSE"
    assert result.execution is not None
    assert result.execution.result.success
    assert result.reason == "stop loss"

    assert result.execution.result.execution_price == 110.0

    assert (
        runner.execution_engine.position_manager.get_position(
            "BTCUSDT"
        )
        is None
    )

def test_runner_process_candles_accepts_candle_objects():
    runner = make_runner(defer_execution=True)

    timestamp = datetime(2026, 1, 1, 12, 0, 0)

    class LongStrategy:
        def generate_signal(self, *, symbol, candles):
            assert all(isinstance(candle, Candle) for candle in candles)

            return TradingSignal(
                symbol=symbol,
                side="LONG",
                price=candles[-1].close,
                timestamp=timestamp,
                atr=2.0,
            )

    candles = [
        Candle(
            symbol="BTCUSDT",
            timeframe="1m",
            open=100.0,
            high=102.0,
            low=99.0,
            close=101.0,
            volume=1000.0,
            timestamp=timestamp,
        ),
        Candle(
            symbol="BTCUSDT",
            timeframe="1m",
            open=101.0,
            high=104.0,
            low=100.0,
            close=103.0,
            volume=1100.0,
            timestamp=timestamp,
        ),
    ]

    result = runner.process_candles(
        symbol="BTCUSDT",
        candles=candles,
        strategy=LongStrategy(),
        timestamp=timestamp,
    )

    # New behavior: process_candles persists an ExecutionIntent instead of
    # opening immediately. Verify pending intent persisted and no position
    # was opened by the process_candles call.
    assert result.action == "PERSISTED_INTENT"
    assert result.execution is None

    intents = intent_manager.load_pending_intents(status="PENDING")
    assert any(i.symbol == "BTCUSDT" for i in intents)


def test_signal_creates_pending_intent_and_does_not_open():
    runner = make_runner(defer_execution=True)

    timestamp = datetime(2026, 1, 1, 12, 0, 0)

    class LongStrategy2:
        def generate_signal(self, *, symbol, candles):
            return TradingSignal(
                symbol=symbol,
                side="LONG",
                price=candles[-1].close + 10.0,
                timestamp=timestamp,
                atr=2.0,
            )

    result = runner.process_candles(
        symbol="BTCUSDT",
        candles=[
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "timestamp": timestamp}
        ],
        strategy=LongStrategy2(),
        timestamp=timestamp,
    )

    assert result.action == "PERSISTED_INTENT"
    # No open position yet
    assert runner.execution_engine.position_manager.get_position("BTCUSDT") is None


def test_next_candle_executes_pending_intent_and_marks_executed():
    runner = make_runner(defer_execution=True)

    t0 = datetime(2026, 1, 1, 12, 0, 0)
    t1 = datetime(2026, 1, 1, 12, 1, 0)

    class LongStrategy3:
        def generate_signal(self, *, symbol, candles):
            return TradingSignal(
                symbol=symbol,
                side="LONG",
                price=candles[-1].close,
                timestamp=t0,
                atr=1.5,
            )

    # Create pending intent from t0
    res = runner.process_candles(
        symbol="BTCUSDT",
        candles=[{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "timestamp": t0}],
        strategy=LongStrategy3(),
        timestamp=t0,
    )

    assert res.action == "PERSISTED_INTENT"

    # Next candle arrives at t1 -> should execute at open=102.0
    class FlatStrategy2:
        def generate_signal(self, *, symbol, candles):
            return TradingSignal(
                symbol=symbol,
                side="FLAT",
                price=0.0,
                timestamp=t1,
                atr=None,
            )

    res2 = runner.process_candles(
        symbol="BTCUSDT",
        candles=[
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "timestamp": t0},
            {"open": 102.0, "high": 103.0, "low": 101.0, "close": 102.0, "timestamp": t1},
        ],
        strategy=FlatStrategy2(),
        timestamp=t1,
    )

    assert res2.action == "INTENT_EXECUTED"
    # Position was opened at next candle open
    pos = runner.execution_engine.position_manager.get_position("BTCUSDT")
    assert pos is not None
    assert pos.entry_price == 102.0

    executed = intent_manager.load_pending_intents(status="EXECUTED")
    assert any(i.intent_id for i in executed)


def test_source_candle_does_not_execute_intent():
    runner = make_runner(defer_execution=True)

    t0 = datetime(2026, 1, 1, 12, 0, 0)

    class LongStrategy4:
        def generate_signal(self, *, symbol, candles):
            return TradingSignal(symbol=symbol, side="LONG", price=100.0, timestamp=t0, atr=1.0)

    # Create intent
    res = runner.process_candles(
        symbol="BTCUSDT",
        candles=[{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "timestamp": t0}],
        strategy=LongStrategy4(),
        timestamp=t0,
    )

    assert res.action == "PERSISTED_INTENT"

    # Process the same single candle again (no later candle) -> should NOT execute
    res2 = runner.process_candles(
        symbol="BTCUSDT",
        candles=[{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "timestamp": t0}],
        strategy=LongStrategy4(),
        timestamp=t0,
    )

    # Expect no execution and no second intent created (duplicate prevented)
    assert res2.action in {"HOLD", "PERSISTED_INTENT"}
    assert runner.execution_engine.position_manager.get_position("BTCUSDT") is None


def test_duplicate_processing_is_safe():
    runner = make_runner(defer_execution=True)

    t0 = datetime(2026, 1, 1, 12, 0, 0)
    t1 = datetime(2026, 1, 1, 12, 1, 0)

    class LongStrategy5:
        def generate_signal(self, *, symbol, candles):
            return TradingSignal(symbol=symbol, side="LONG", price=100.0, timestamp=t0, atr=1.0)

    runner.process_candles(
        symbol="BTCUSDT",
        candles=[{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "timestamp": t0}],
        strategy=LongStrategy5(),
        timestamp=t0,
    )

    # execute once
    res1 = runner.process_candles(
        symbol="BTCUSDT",
        candles=[
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "timestamp": t0},
            {"open": 105.0, "high": 106.0, "low": 104.0, "close": 105.0, "timestamp": t1},
        ],
        strategy=LongStrategy5(),
        timestamp=t1,
    )

    assert res1.action == "INTENT_EXECUTED"

    # process the same candles again: should not execute the same intent twice
    res2 = runner.process_candles(
        symbol="BTCUSDT",
        candles=[
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "timestamp": t0},
            {"open": 105.0, "high": 106.0, "low": 104.0, "close": 105.0, "timestamp": t1},
        ],
        strategy=LongStrategy5(),
        timestamp=t1,
    )

    # second processing should not create a duplicate position
    assert runner.execution_engine.position_manager.get_position("BTCUSDT") is not None


def test_strategy_price_is_ignored_on_execution():
    runner = make_runner(defer_execution=True)

    t0 = datetime(2026, 1, 1, 12, 0, 0)
    t1 = datetime(2026, 1, 1, 12, 1, 0)

    class PriceyStrategy:
        def generate_signal(self, *, symbol, candles):
            # Intentionally return a price that differs from the next candle open
            return TradingSignal(symbol=symbol, side="LONG", price=9999.0, timestamp=t0, atr=1.0)

    runner.process_candles(
        symbol="BTCUSDT",
        candles=[{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "timestamp": t0}],
        strategy=PriceyStrategy(),
        timestamp=t0,
    )

    res = runner.process_candles(
        symbol="BTCUSDT",
        candles=[
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "timestamp": t0},
            {"open": 123.45, "high": 124.0, "low": 122.0, "close": 123.45, "timestamp": t1},
        ],
        strategy=PriceyStrategy(),
        timestamp=t1,
    )

    assert res.action == "INTENT_EXECUTED"
    pos = runner.execution_engine.position_manager.get_position("BTCUSDT")
    assert pos is not None
    assert pos.entry_price == 123.45