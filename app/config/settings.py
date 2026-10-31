"""Runtime configuration for ATLAS.

Two backends share one parsing contract (the ``parse_*`` helpers below):

- pydantic-settings when it is installed: reads the environment and ``.env``.
- a dependency-free fallback used *only* when pydantic-settings cannot be
  imported: reads the environment only; ``.env`` is not consulted.

Invalid configuration always fails fast with ValueError. It is never silently
replaced by a default: a trading system must not start on a configuration it
did not understand. Contract: tests/test_settings_contract.py.
"""
from __future__ import annotations

import os
import re
from typing import Annotated, Any

DEFAULT_ENVIRONMENT = "development"
DEFAULT_DEBUG = True
DEFAULT_MARKET_POLL_INTERVAL_SECONDS = 60
DEFAULT_SYMBOLS: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
DEFAULT_CLAUDE_MODEL = "claude-fable-5-1"

_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]+$")
_INTEGER_PATTERN = re.compile(r"^[+-]?\d+$")
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


# --- shared parsing contract -----------------------------------------------------


def parse_symbols(value: Any) -> list[str]:
    """Comma-separated symbols -> trimmed, upper-cased, de-duplicated list (order kept)."""
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, (list, tuple)):
        items = [str(item) for item in value]
    else:
        raise ValueError(f"symbols must be a comma-separated string, got {type(value).__name__}")

    symbols: list[str] = []
    for raw in items:
        symbol = raw.strip().upper()
        if not symbol:
            continue
        if not _SYMBOL_PATTERN.match(symbol):
            raise ValueError(f"invalid symbol {raw.strip()!r}; expected letters and digits only, e.g. BTCUSDT")
        if symbol not in symbols:
            symbols.append(symbol)
    if not symbols:
        raise ValueError("symbols cannot be empty")
    return symbols


def parse_debug(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(f"debug must be one of {sorted(_TRUE | _FALSE)}, got {value!r}")


def parse_poll_interval(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"market_poll_interval_seconds must be an integer > 0, got {value!r}")
    if isinstance(value, int):
        seconds = value
    else:
        text = str(value).strip()
        if not _INTEGER_PATTERN.match(text):
            raise ValueError(f"market_poll_interval_seconds must be an integer > 0, got {value!r}")
        seconds = int(text)
    if seconds <= 0:
        raise ValueError(f"market_poll_interval_seconds must be an integer > 0, got {value!r}")
    return seconds


def parse_claude_model(value: Any) -> str:
    model = str(value).strip()
    if not model:
        raise ValueError("claude_model cannot be empty")
    return model


def parse_environment(value: Any) -> str:
    environment = str(value).strip()
    if not environment:
        raise ValueError("environment cannot be empty")
    return environment


# --- backends ------------------------------------------------------------------------

try:
    from pydantic import field_validator
    from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
except ImportError:
    SETTINGS_BACKEND = "fallback"

    class Settings:
        """Environment-only settings used when pydantic-settings is not installed."""

        def __init__(self, **_: Any) -> None:
            self.environment = parse_environment(os.getenv("ENVIRONMENT", DEFAULT_ENVIRONMENT))
            self.debug = parse_debug(os.getenv("DEBUG", DEFAULT_DEBUG))
            self.market_poll_interval_seconds = parse_poll_interval(
                os.getenv("MARKET_POLL_INTERVAL_SECONDS", DEFAULT_MARKET_POLL_INTERVAL_SECONDS)
            )
            self.symbols = parse_symbols(os.getenv("SYMBOLS", list(DEFAULT_SYMBOLS)))
            self.claude_model = parse_claude_model(os.getenv("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL))

else:
    SETTINGS_BACKEND = "pydantic-settings"

    class Settings(BaseSettings):
        environment: str = DEFAULT_ENVIRONMENT
        debug: bool = DEFAULT_DEBUG
        market_poll_interval_seconds: int = DEFAULT_MARKET_POLL_INTERVAL_SECONDS
        # NoDecode: pydantic-settings must not JSON-decode SYMBOLS; the shared CSV parser owns it.
        symbols: Annotated[list[str], NoDecode] = list(DEFAULT_SYMBOLS)
        claude_model: str = DEFAULT_CLAUDE_MODEL

        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

        @field_validator("environment", mode="before")
        @classmethod
        def _validate_environment(cls, value: Any) -> str:
            return parse_environment(value)

        @field_validator("debug", mode="before")
        @classmethod
        def _validate_debug(cls, value: Any) -> bool:
            return parse_debug(value)

        @field_validator("market_poll_interval_seconds", mode="before")
        @classmethod
        def _validate_poll_interval(cls, value: Any) -> int:
            return parse_poll_interval(value)

        @field_validator("symbols", mode="before")
        @classmethod
        def _validate_symbols(cls, value: Any) -> list[str]:
            return parse_symbols(value)

        @field_validator("claude_model", mode="before")
        @classmethod
        def _validate_claude_model(cls, value: Any) -> str:
            return parse_claude_model(value)


settings = Settings()