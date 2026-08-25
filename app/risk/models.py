from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade: float = 0.01
    daily_loss_limit_pct: float = 0.03
    max_drawdown_pct: float = 0.10
    max_concurrent_positions: int = 1
    maximum_position_pct: float = 1.00
    minimum_position_size: float = 0.0


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    quantity: float
    risk_amount: float
    position_notional: float
    reason: str


@dataclass
class RiskState:
    equity_peak: float
    day_start_equity: float
    realized_daily_pnl: float = 0.0
    concurrent_positions: int = 0
    current_day: Optional[str] = None
    halted: bool = False
    halt_reason: Optional[str] = None