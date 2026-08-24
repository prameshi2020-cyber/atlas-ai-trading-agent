from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class Ticker:
    symbol: str
    price: float
    price_change_24h: float
    high_24h: float
    low_24h: float
    volume_24h: float
    timestamp: datetime


@dataclass(slots=True)
class Candle:
    symbol: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime