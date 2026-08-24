import os
from typing import Any

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Settings(BaseSettings):
        environment: str = "development"
        debug: bool = True
        market_poll_interval_seconds: int = 60
        symbols: list[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

    settings = Settings()
except Exception:
    # Fallback simple settings when pydantic-settings is not available
    class Settings:  # type: ignore[reportMissingTypeStubs]
        def __init__(self, **_: Any) -> None:
            self.environment = os.getenv("ENVIRONMENT", "development")
            self.debug = os.getenv("DEBUG", "1") in ("1", "true", "True")
            try:
                self.market_poll_interval_seconds = int(
                    os.getenv("MARKET_POLL_INTERVAL_SECONDS", "60")
                )
            except ValueError:
                self.market_poll_interval_seconds = 60

            symbols = os.getenv("SYMBOLS")
            if symbols:
                self.symbols = [s.strip().upper() for s in symbols.split(",") if s.strip()]
            else:
                self.symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    settings = Settings()