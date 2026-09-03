from __future__ import annotations

from app.risk.manager import RiskManager
from app.risk.models import RiskConfig
from app.trading.broker import PaperBroker
from app.trading.execution import ExecutionEngine
from app.trading.position_manager import PositionManager


def create_paper_trading_engine(
    *,
    initial_cash: float,
    fee_rate: float = 0.001,
    risk_config: RiskConfig | None = None,
) -> ExecutionEngine:
    if initial_cash <= 0:
        raise ValueError(
            "initial_cash must be greater than zero"
        )

    broker = PaperBroker(
        initial_cash=initial_cash,
        fee_rate=fee_rate,
    )

    broker.load_state()

    risk_manager = RiskManager(
        initial_equity=initial_cash,
        config=risk_config or RiskConfig(),
    )

    risk_manager.load_state()

    position_manager = PositionManager()
    position_manager.load_positions()

    risk_manager.restore_open_positions(
        position_manager.position_count()
    )

    return ExecutionEngine(
        broker=broker,
        risk_manager=risk_manager,
        position_manager=position_manager,
    )