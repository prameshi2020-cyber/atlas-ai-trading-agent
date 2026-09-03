from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


OrderSide = Literal["BUY", "SELL"]
PositionSide = Literal["LONG", "SHORT"]


@dataclass(slots=True)
class OrderRequest:
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime


@dataclass(slots=True)
class Position:
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    entry_time: datetime
    entry_fee: float = 0.0
    stop_price: float | None = None


@dataclass(slots=True)
class ExecutionResult:
    success: bool
    symbol: str
    side: OrderSide
    quantity: float
    execution_price: float
    fee: float
    timestamp: datetime
    reason: str