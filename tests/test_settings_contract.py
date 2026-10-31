"""Settings contract: both configuration backends must agree.

app.config.settings has a pydantic-settings implementation and a dependency-free
fallback used only when pydantic-settings is not importable. This module loads
the source file under private module names -- once per backend -- so the
process-wide ``settings`` singleton is never touched, and asserts identical
behaviour from both.

Contract
- invalid configuration fails fast (ValueError); it is never silently replaced
  by a default, not even at import time
- SYMBOLS: comma separated, trimmed, upper-cased, de-duplicated (order kept)
- DEBUG: 1/0, true/false, yes/no, on/off, case-insensitive
- MARKET_POLL_INTERVAL_SECONDS: integer > 0
- CLAUDE_MODEL / ENVIRONMENT: stripped, non-empty
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "app" / "config" / "settings.py"
ENV_KEYS = ("ENVIRONMENT", "DEBUG", "MARKET_POLL_INTERVAL_SECONDS", "SYMBOLS", "CLAUDE_MODEL")
BACKENDS = ("pydantic-settings", "fallback")


def _load(name: str, *, backend: str, monkeypatch):
    if backend == "fallback":
        monkeypatch.setitem(sys.modules, "pydantic_settings", None)  # makes the import raise ImportError
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=BACKENDS, ids=BACKENDS)
def make_settings(request, monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    module = _load(f"_settings_under_test_{request.param.replace('-', '_')}", backend=request.param, monkeypatch=monkeypatch)

    def factory(**env: str):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return module.Settings(_env_file=None)  # never consult a developer's .env in tests

    factory.module = module
    factory.backend = request.param
    return factory


# --- backends -----------------------------------------------------------------


def test_real_module_uses_pydantic_backend_in_this_environment():
    import app.config.settings as real

    assert real.SETTINGS_BACKEND == "pydantic-settings"


def test_fixture_exercises_the_requested_backend(make_settings):
    assert make_settings.module.SETTINGS_BACKEND == make_settings.backend


def test_isolated_loads_never_replace_the_process_singleton(make_settings):
    import app.config.settings as real

    make_settings(SYMBOLS="XRPUSDT")

    assert sys.modules["app.config.settings"] is real
    assert make_settings.module is not real


def test_invalid_environment_fails_fast_at_import_instead_of_falling_back(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MARKET_POLL_INTERVAL_SECONDS", "not-a-number")

    with pytest.raises(ValueError):
        _load("_settings_fail_fast", backend="pydantic-settings", monkeypatch=monkeypatch)


# --- defaults -----------------------------------------------------------------


def test_defaults_are_identical_across_backends(make_settings):
    s = make_settings()

    assert s.environment == "development"
    assert s.debug is True
    assert s.market_poll_interval_seconds == 60
    assert s.symbols == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert s.claude_model == make_settings.module.DEFAULT_CLAUDE_MODEL


# --- SYMBOLS ------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        (" btcusdt, ETHUSDT ,,solusdt,BTCUSDT ", ["BTCUSDT", "ETHUSDT", "SOLUSDT"]),
        ("XRPUSDT", ["XRPUSDT"]),
    ],
)
def test_symbols_are_trimmed_uppercased_and_deduplicated(make_settings, raw, expected):
    assert make_settings(SYMBOLS=raw).symbols == expected


@pytest.mark.parametrize("raw", ["", "   ", ",,,", '["BTCUSDT","ETHUSDT"]', "BTC-USDT", "BTC USDT"])
def test_invalid_symbols_are_rejected_not_defaulted(make_settings, raw):
    with pytest.raises(ValueError):
        make_settings(SYMBOLS=raw)


# --- DEBUG ----------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "Yes", "on"])
def test_debug_truthy_forms(make_settings, raw):
    assert make_settings(DEBUG=raw).debug is True


@pytest.mark.parametrize("raw", ["0", "false", "No", "OFF"])
def test_debug_falsy_forms(make_settings, raw):
    assert make_settings(DEBUG=raw).debug is False


@pytest.mark.parametrize("raw", ["maybe", "2", ""])
def test_debug_rejects_unknown_forms(make_settings, raw):
    with pytest.raises(ValueError):
        make_settings(DEBUG=raw)


# --- MARKET_POLL_INTERVAL_SECONDS -------------------------------------------------


@pytest.mark.parametrize("raw, expected", [("30", 30), ("1", 1), (" 45 ", 45)])
def test_poll_interval_parses_positive_integers(make_settings, raw, expected):
    assert make_settings(MARKET_POLL_INTERVAL_SECONDS=raw).market_poll_interval_seconds == expected


@pytest.mark.parametrize("raw", ["0", "-5", "abc", "1.5", "", "true"])
def test_poll_interval_rejects_non_positive_or_non_integer(make_settings, raw):
    with pytest.raises(ValueError):
        make_settings(MARKET_POLL_INTERVAL_SECONDS=raw)


# --- CLAUDE_MODEL / ENVIRONMENT -----------------------------------------------------


def test_claude_model_is_stripped(make_settings):
    assert make_settings(CLAUDE_MODEL="  claude-x  ").claude_model == "claude-x"


@pytest.mark.parametrize("raw", ["", "   "])
def test_empty_claude_model_is_rejected(make_settings, raw):
    with pytest.raises(ValueError):
        make_settings(CLAUDE_MODEL=raw)


def test_environment_is_passed_through(make_settings):
    assert make_settings(ENVIRONMENT="production").environment == "production"


def test_empty_environment_is_rejected(make_settings):
    with pytest.raises(ValueError):
        make_settings(ENVIRONMENT="   ")