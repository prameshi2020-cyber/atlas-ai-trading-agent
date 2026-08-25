"""
Deterministic tests for BinanceProvider's closed-candle handling.

No live network calls are made. Pure-logic tests exercise
`_is_kline_closed` / `_closed_candles_from_klines` directly with an
injected `now`. The integration test uses httpx.MockTransport to
verify the full async `get_candle` / `get_historical_candles` methods
never return a still-forming candle, without touching the real
Binance API.
"""

from datetime import datetime, timedelta, timezone
from unittest import mock

import asyncio

import httpx

from app.market.binance import BinanceProvider


def _kline(open_time_ms: int, close_time_ms: int, *, close: float = 100.0) -> list:
    """Build a minimal synthetic Binance kline row."""
    return [
        open_time_ms,
        str(close),      # open
        str(close + 1),  # high
        str(close - 1),  # low
        str(close),       # close
        "10.0",           # volume
        close_time_ms,
        "0", 0, "0", "0", "0",
    ]


# ---------------------------------------------------------------------
# A1: pure filtering logic (no network, injected `now`)
# ---------------------------------------------------------------------


def test_is_kline_closed_true_when_close_time_in_the_past():
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    close_time_ms = int(now.timestamp() * 1000) - 1000  # 1s ago

    assert BinanceProvider._is_kline_closed(close_time_ms, now=now) is True


def test_is_kline_closed_false_when_close_time_in_the_future():
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    close_time_ms = int(now.timestamp() * 1000) + 1000  # 1s from now

    assert BinanceProvider._is_kline_closed(close_time_ms, now=now) is False


def test_is_kline_closed_false_exactly_at_boundary():
    # Candle is only closed once "now" has fully passed close_time,
    # not merely reached it.
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    close_time_ms = int(now.timestamp() * 1000)

    assert BinanceProvider._is_kline_closed(close_time_ms, now=now) is False


def test_closed_candles_from_klines_drops_forming_candle():
    now = datetime(2026, 1, 1, 12, 5, 30, tzinfo=timezone.utc)
    now_ms = int(now.timestamp() * 1000)

    closed_kline = _kline(1_000, now_ms - 5_000, close=100.5)
    forming_kline = _kline(61_000, now_ms + 55_000, close=101.0)

    candles = BinanceProvider._closed_candles_from_klines(
        "BTCUSDT", "1m", [closed_kline, forming_kline], now=now,
    )

    assert len(candles) == 1
    assert candles[0].close == 100.5
    assert candles[0].symbol == "BTCUSDT"
    assert candles[0].timeframe == "1m"


def test_closed_candles_from_klines_keeps_order():
    now = datetime(2026, 1, 1, 12, 5, 30, tzinfo=timezone.utc)
    now_ms = int(now.timestamp() * 1000)

    k1 = _kline(1_000, now_ms - 120_000, close=100.0)
    k2 = _kline(61_000, now_ms - 60_000, close=101.0)
    k3 = _kline(121_000, now_ms + 55_000, close=102.0)  # forming

    candles = BinanceProvider._closed_candles_from_klines(
        "BTCUSDT", "1m", [k1, k2, k3], now=now,
    )

    assert [c.close for c in candles] == [100.0, 101.0]


def test_closed_candles_from_klines_empty_when_all_forming():
    now = datetime(2026, 1, 1, 12, 5, 30, tzinfo=timezone.utc)
    now_ms = int(now.timestamp() * 1000)

    forming = _kline(1_000, now_ms + 55_000, close=100.0)

    candles = BinanceProvider._closed_candles_from_klines(
        "BTCUSDT", "1m", [forming], now=now,
    )

    assert candles == []


# ---------------------------------------------------------------------
# A2: end-to-end async methods via httpx.MockTransport (still no
# real network access -- all traffic is intercepted locally).
# ---------------------------------------------------------------------


def test_get_candle_never_returns_a_forming_candle():
    real_now = datetime.now(timezone.utc)
    now_ms = int(real_now.timestamp() * 1000)

    closed_kline = _kline(1_000, now_ms - 5_000, close=100.5)
    forming_kline = _kline(61_000, now_ms + 55_000, close=999.0)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v3/klines"
        assert request.url.params["limit"] == "2"
        return httpx.Response(200, json=[closed_kline, forming_kline])

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def fake_async_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    provider = BinanceProvider()

    with mock.patch("app.market.binance.httpx.AsyncClient", fake_async_client):
        candle = asyncio.run(provider.get_candle("BTCUSDT", "1m"))

    assert candle.close == 100.5  # the closed one, never the 999.0 forming one


def test_get_candle_raises_when_no_closed_candle_available():
    real_now = datetime.now(timezone.utc)
    now_ms = int(real_now.timestamp() * 1000)

    # Both klines still forming.
    forming_1 = _kline(1_000, now_ms + 10_000, close=100.0)
    forming_2 = _kline(61_000, now_ms + 70_000, close=101.0)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[forming_1, forming_2])

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def fake_async_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    provider = BinanceProvider()

    with mock.patch("app.market.binance.httpx.AsyncClient", fake_async_client):
        try:
            asyncio.run(provider.get_candle("BTCUSDT", "1m"))
            raised = False
        except ValueError:
            raised = True

    assert raised, "Expected get_candle to raise rather than silently return an incomplete candle"


def test_get_historical_candles_drops_trailing_forming_candle():
    real_now = datetime.now(timezone.utc)
    now_ms = int(real_now.timestamp() * 1000)

    k1 = _kline(1_000, now_ms - 180_000, close=100.0)
    k2 = _kline(61_000, now_ms - 120_000, close=101.0)
    k3 = _kline(121_000, now_ms - 60_000, close=102.0)
    forming = _kline(181_000, now_ms + 55_000, close=999.0)

    def handler(request: httpx.Request) -> httpx.Response:
        # limit=3 requested -> provider should ask for limit+1=4
        assert request.url.params["limit"] == "4"
        return httpx.Response(200, json=[k1, k2, k3, forming])

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def fake_async_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    provider = BinanceProvider()

    with mock.patch("app.market.binance.httpx.AsyncClient", fake_async_client):
        candles = asyncio.run(
            provider.get_historical_candles("BTCUSDT", "1m", limit=3)
        )

    assert len(candles) == 3
    assert [c.close for c in candles] == [100.0, 101.0, 102.0]
    assert 999.0 not in [c.close for c in candles]
