from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from app.risk.models import RiskConfig
from app.database.models import load_candles
from app.market.base import MarketDataProvider
from app.market.collector import MarketCollector
from app.market.models import Candle, Ticker
from app.strategy.models import TradingSignal
from app.trading.bootstrap import create_paper_trading_engine
from app.trading.coordinator import TradingCoordinator
from app.trading.market_service import MarketTradingService
from app.trading.runner import PaperTradingRunner


TEST_SYMBOL = "INTEGRATIONBTC"

TIMESTAMP_1 = datetime(
    2099,
    1,
    1,
    11,
    58,
    0,
    tzinfo=timezone.utc,
)

TIMESTAMP_2 = datetime(
    2099,
    1,
    1,
    11,
    59,
    0,
    tzinfo=timezone.utc,
)

TIMESTAMP_3 = datetime(
    2099,
    1,
    1,
    12,
    0,
    0,
    tzinfo=timezone.utc,
)


class FakeMarketProvider(MarketDataProvider):
    def __init__(self) -> None:
        self.ticker = Ticker(
            symbol=TEST_SYMBOL,
            price=105.0,
            price_change_24h=1.5,
            high_24h=110.0,
            low_24h=95.0,
            volume_24h=1000.0,
            timestamp=TIMESTAMP_3,
        )

        self.candles = [
            Candle(
                symbol=TEST_SYMBOL,
                timeframe="1m",
                open=100.0,
                high=102.0,
                low=99.0,
                close=101.0,
                volume=10.0,
                timestamp=TIMESTAMP_1,
            ),
            Candle(
                symbol=TEST_SYMBOL,
                timeframe="1m",
                open=101.0,
                high=104.0,
                low=100.0,
                close=103.0,
                volume=12.0,
                timestamp=TIMESTAMP_2,
            ),
            Candle(
                symbol=TEST_SYMBOL,
                timeframe="1m",
                open=103.0,
                high=106.0,
                low=102.0,
                close=105.0,
                volume=15.0,
                timestamp=TIMESTAMP_3,
            ),
        ]

    async def get_ticker(self, symbol: str) -> Ticker:
        assert symbol == TEST_SYMBOL
        return self.ticker

    async def get_candle(
        self,
        symbol: str,
        timeframe: str = "1m",
    ) -> Candle:
        assert symbol == TEST_SYMBOL
        assert timeframe == "1m"
        return self.candles[-1]

    async def get_candles(
        self,
        symbols: list[str],
        timeframe: str = "1m",
    ) -> list[Candle]:
        assert symbols == [TEST_SYMBOL]
        assert timeframe == "1m"
        return self.candles


class FakeLongStrategy:
    def generate_signal(
        self,
        *,
        symbol: str,
        candles: list[Candle],
    ) -> TradingSignal:
        return TradingSignal(
            symbol=symbol,
            side="LONG",
            price=105.0,
            timestamp=candles[-1].timestamp,
            reason="integration test LONG signal",
            atr=2.0,
        )


def build_trading_service() -> tuple[
    object,
    PaperTradingRunner,
    TradingCoordinator,
    MarketTradingService,
]:
    engine = create_paper_trading_engine(
        initial_cash=10_000.0,
        fee_rate=0.001,
        risk_config=RiskConfig(
            risk_per_trade=0.01,
            daily_loss_limit_pct=0.03,
            max_drawdown_pct=0.10,
            max_concurrent_positions=100,
            maximum_position_pct=1.00,
            minimum_position_size=0.0,
        ),
    )

    from app.strategy.stop_policy import StopPolicy

    runner = PaperTradingRunner(
        execution_engine=engine,
        stop_policy=StopPolicy(
            atr_multiplier=1.5,
        ),
    )

    coordinator = TradingCoordinator(
        runner=runner,
    )

    service = MarketTradingService(
        coordinator=coordinator,
    )

    return engine, runner, coordinator, service


def test_market_data_flows_from_collector_to_database_and_trading():
    cleanup_integration_state()

    provider = FakeMarketProvider()

    collector = MarketCollector(
        provider=provider,
        symbols=[TEST_SYMBOL],
        interval_seconds=60,
    )

    asyncio.run(collector.collect_once())

    engine, _, _, service = build_trading_service()

    result = service.process_symbol(
        symbol=TEST_SYMBOL,
        strategy=FakeLongStrategy(),
        timestamp=TIMESTAMP_3,
    )

    assert result.action == "OPEN"
    assert result.execution is not None
    assert result.execution.result.success is True

    position = engine.position_manager.get_position(TEST_SYMBOL)

    assert position is not None
    assert position.side == "LONG"
    assert position.entry_price == pytest.approx(105.0)
    assert position.stop_price == pytest.approx(102.0)


def test_collector_persists_all_integration_candles():
    provider = FakeMarketProvider()

    collector = MarketCollector(
        provider=provider,
        symbols=[TEST_SYMBOL],
        interval_seconds=60,
    )

    asyncio.run(collector.collect_once())

    candles = load_candles(
        symbol=TEST_SYMBOL,
        timeframe="1m",
    )

    timestamps = {candle.timestamp for candle in candles}

    expected_timestamps = {
        candle.timestamp
        for candle in provider.candles
    }

    assert expected_timestamps.issubset(timestamps)


def test_full_pipeline_does_not_require_real_market_connection():
    provider = FakeMarketProvider()

    collector = MarketCollector(
        provider=provider,
        symbols=[TEST_SYMBOL],
        interval_seconds=60,
    )

    asyncio.run(collector.collect_once())

    candles = load_candles(
        symbol=TEST_SYMBOL,
        timeframe="1m",
    )

    assert len(candles) >= 3
    assert provider.ticker.symbol == TEST_SYMBOL

def cleanup_integration_state() -> None:
    from app.database.connection import get_connection

    connection = get_connection()

    try:
        connection.execute(
            "DELETE FROM market_candles WHERE symbol = ?",
            (TEST_SYMBOL,),
        )

        connection.execute(
            "DELETE FROM market_ticks WHERE symbol = ?",
            (TEST_SYMBOL,),
        )

        connection.execute(
            "DELETE FROM positions WHERE symbol = ?",
            (TEST_SYMBOL,),
        )

        connection.execute(
            "DELETE FROM trade_journal WHERE symbol = ?",
            (TEST_SYMBOL,),
        )

        connection.commit()
    finally:
        connection.close()