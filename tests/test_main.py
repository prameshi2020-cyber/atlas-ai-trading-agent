import pytest

import app.main as main_module


@pytest.mark.anyio
async def test_main_wires_paper_trading_pipeline(monkeypatch):
    created = {}
    events = []

    class FakeEngine:
        pass

    class FakeProvider:
        pass

    class FakeWarmup:
        def __init__(
            self,
            *,
            provider,
            symbols,
            limit,
        ):
            created["warmup_provider"] = provider
            created["warmup_symbols"] = symbols
            created["warmup_limit"] = limit
            events.append("warmup_created")

        async def warmup(self):
            events.append("warmup_run")
            return {
                symbol: 100
                for symbol in main_module.settings.symbols
            }

    class FakeCollector:
        def __init__(
            self,
            *,
            provider,
            symbols,
            interval_seconds,
        ):
            created["provider"] = provider
            created["symbols"] = symbols
            created["collector_interval"] = interval_seconds
            events.append("collector_created")

        async def collect_once(self):
            pass

    class FakeRunner:
        def __init__(
            self,
            *,
            execution_engine,
            stop_policy,
        ):
            created["runner_engine"] = execution_engine
            created["runner_stop_policy"] = stop_policy
            events.append("runner_created")

    class FakeCoordinator:
        def __init__(self, *, runner):
            created["coordinator_runner"] = runner
            events.append("coordinator_created")

    class FakeTradingService:
        def __init__(self, *, coordinator):
            created["service_coordinator"] = coordinator
            events.append("service_created")

    class FakeStrategy:
        def __init__(
            self,
            *,
            fast_period,
            slow_period,
            atr_period,
            minimum_atr,
        ):
            created.setdefault("strategies", []).append(
                {
                    "fast_period": fast_period,
                    "slow_period": slow_period,
                    "atr_period": atr_period,
                    "minimum_atr": minimum_atr,
                }
            )
            events.append("strategy_created")

    class FakeOrchestrator:
        def __init__(
            self,
            *,
            collector,
            trading_service,
            strategies,
            interval_seconds,
        ):
            created["orchestrator_collector"] = collector
            created["orchestrator_service"] = trading_service
            created["orchestrator_strategies"] = strategies
            created["orchestrator_interval"] = interval_seconds
            events.append("orchestrator_created")

        async def run(self):
            events.append("orchestrator_run")
            created["orchestrator_run"] = True

    def fake_create_engine(
        *,
        initial_cash,
        fee_rate=0.001,
        risk_config=None,
    ):
        created["initial_cash"] = initial_cash
        created["fee_rate"] = fee_rate
        created["risk_config"] = risk_config
        engine = FakeEngine()
        created["engine"] = engine
        events.append("engine_created")
        return engine

    monkeypatch.setattr(
        main_module,
        "create_paper_trading_engine",
        fake_create_engine,
    )
    monkeypatch.setattr(
        main_module,
        "BinanceProvider",
        FakeProvider,
    )
    monkeypatch.setattr(
        main_module,
        "MarketWarmup",
        FakeWarmup,
    )
    monkeypatch.setattr(
        main_module,
        "MarketCollector",
        FakeCollector,
    )
    monkeypatch.setattr(
        main_module,
        "PaperTradingRunner",
        FakeRunner,
    )
    monkeypatch.setattr(
        main_module,
        "TradingCoordinator",
        FakeCoordinator,
    )
    monkeypatch.setattr(
        main_module,
        "MarketTradingService",
        FakeTradingService,
    )
    monkeypatch.setattr(
        main_module,
        "StrategyEngine",
        FakeStrategy,
    )
    monkeypatch.setattr(
        main_module,
        "TradingOrchestrator",
        FakeOrchestrator,
    )

    await main_module.main()

    assert created["initial_cash"] == 10_000.0

    assert isinstance(
        created["provider"],
        FakeProvider,
    )

    assert created["warmup_provider"] is created["provider"]
    assert created["warmup_symbols"] == main_module.settings.symbols
    assert created["warmup_limit"] == 100

    assert created["symbols"] == main_module.settings.symbols
    assert (
        created["collector_interval"]
        == main_module.settings.market_poll_interval_seconds
    )

    assert created["runner_engine"] is created["engine"]

    assert created["coordinator_runner"] is not None
    assert created["service_coordinator"] is not None

    assert (
        created["orchestrator_collector"] is not None
    )
    assert (
        created["orchestrator_service"] is not None
    )

    assert set(
        created["orchestrator_strategies"].keys()
    ) == set(main_module.settings.symbols)

    assert len(created["strategies"]) == len(
        main_module.settings.symbols
    )

    for strategy_config in created["strategies"]:
        assert strategy_config == {
            "fast_period": 3,
            "slow_period": 5,
            "atr_period": 3,
            "minimum_atr": 0.0,
        }

    assert (
        created["orchestrator_interval"]
        == main_module.settings.market_poll_interval_seconds
    )

    assert created["orchestrator_run"] is True

    assert events.index("warmup_run") < events.index(
        "orchestrator_created"
    )
    assert events.index("warmup_run") < events.index(
        "orchestrator_run"
    )


@pytest.mark.anyio
async def test_main_initializes_all_databases(monkeypatch):
    initialized = {}

    def fake_initialize_database():
        initialized["database"] = True

    def fake_initialize_candle_database():
        initialized["candles"] = True

    def fake_initialize_position_database():
        initialized["positions"] = True

    class FakeProvider:
        pass

    class FakeWarmup:
        def __init__(
            self,
            *,
            provider,
            symbols,
            limit,
        ):
            pass

        async def warmup(self):
            return {}

    class FakeCollector:
        def __init__(
            self,
            *,
            provider,
            symbols,
            interval_seconds,
        ):
            pass

    class FakeRunner:
        def __init__(
            self,
            *,
            execution_engine,
            stop_policy,
        ):
            pass

    class FakeCoordinator:
        def __init__(self, *, runner):
            pass

    class FakeTradingService:
        def __init__(self, *, coordinator):
            pass

    class FakeStrategy:
        def __init__(
            self,
            *,
            fast_period,
            slow_period,
            atr_period,
            minimum_atr,
        ):
            pass

    class FakeOrchestrator:
        def __init__(
            self,
            *,
            collector,
            trading_service,
            strategies,
            interval_seconds,
        ):
            pass

        async def run(self):
            pass

    def fake_create_engine(
        *,
        initial_cash,
        fee_rate=0.001,
        risk_config=None,
    ):
        return object()

    monkeypatch.setattr(
        main_module,
        "initialize_database",
        fake_initialize_database,
    )
    monkeypatch.setattr(
        main_module,
        "initialize_candle_database",
        fake_initialize_candle_database,
    )
    monkeypatch.setattr(
        main_module,
        "initialize_position_database",
        fake_initialize_position_database,
    )
    monkeypatch.setattr(
        main_module,
        "create_paper_trading_engine",
        fake_create_engine,
    )
    monkeypatch.setattr(
        main_module,
        "BinanceProvider",
        FakeProvider,
    )
    monkeypatch.setattr(
        main_module,
        "MarketWarmup",
        FakeWarmup,
    )
    monkeypatch.setattr(
        main_module,
        "MarketCollector",
        FakeCollector,
    )
    monkeypatch.setattr(
        main_module,
        "PaperTradingRunner",
        FakeRunner,
    )
    monkeypatch.setattr(
        main_module,
        "TradingCoordinator",
        FakeCoordinator,
    )
    monkeypatch.setattr(
        main_module,
        "MarketTradingService",
        FakeTradingService,
    )
    monkeypatch.setattr(
        main_module,
        "StrategyEngine",
        FakeStrategy,
    )
    monkeypatch.setattr(
        main_module,
        "TradingOrchestrator",
        FakeOrchestrator,
    )

    await main_module.main()

    assert initialized == {
        "database": True,
        "candles": True,
        "positions": True,
    }
