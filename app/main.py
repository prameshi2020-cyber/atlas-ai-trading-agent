import asyncio
import logging

from app.config.settings import settings
from app.database.models import (
    initialize_database,
    initialize_candle_database,
    initialize_position_database,
)
from app.market.binance import BinanceProvider
from app.market.collector import MarketCollector
from app.market.warmup import MarketWarmup
from app.strategy.engine import StrategyEngine
from app.strategy.stop_policy import StopPolicy
from app.trading.bootstrap import create_paper_trading_engine
from app.trading.coordinator import TradingCoordinator
from app.trading.market_service import MarketTradingService
from app.trading.orchestrator import TradingOrchestrator
from app.trading.runner import PaperTradingRunner


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


async def main() -> None:
    print("=" * 60)
    print("ATLAS AI Trading Agent")
    print(f"Environment: {settings.environment}")
    print("=" * 60)

    initialize_database()
    initialize_candle_database()
    initialize_position_database()

    # Paper trading engine only.
    engine = create_paper_trading_engine(
        initial_cash=10_000.0,
    )

    provider = BinanceProvider()

    warmup = MarketWarmup(
        provider=provider,
        symbols=settings.symbols,
        limit=100,
    )

    warmup_results = await warmup.warmup()

    for symbol, inserted in warmup_results.items():
        logging.info(
            "Historical warmup %s: inserted=%d candles",
            symbol,
            inserted,
        )

    collector = MarketCollector(
        provider=provider,
        symbols=settings.symbols,
        interval_seconds=settings.market_poll_interval_seconds,
    )

    runner = PaperTradingRunner(
        execution_engine=engine,
        stop_policy=StopPolicy(
            atr_multiplier=1.5,
        ),
    )

    coordinator = TradingCoordinator(
        runner=runner,
    )

    trading_service = MarketTradingService(
        coordinator=coordinator,
    )

    strategies = {
        symbol: StrategyEngine(
            fast_period=3,
            slow_period=5,
            atr_period=3,
            minimum_atr=0.0,
        )
        for symbol in settings.symbols
    }

    orchestrator = TradingOrchestrator(
        collector=collector,
        trading_service=trading_service,
        strategies=strategies,
        interval_seconds=settings.market_poll_interval_seconds,
    )

    await orchestrator.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nATLAS stopped.")
