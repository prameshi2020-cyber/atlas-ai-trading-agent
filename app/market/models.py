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