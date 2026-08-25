from datetime import datetime, timezone

import httpx

from app.market.base import MarketDataProvider
from app.market.models import Ticker, Candle


class BinanceProvider(MarketDataProvider):
    """
    Binance US market data provider.

    CLOSED-CANDLE GUARANTEE
    ------------------------
    Binance's /api/v3/klines endpoint, when queried without an
    explicit endTime, includes the currently-forming (not yet closed)
    interval as its most recent entry. If that in-progress candle
    were returned as-is, its high/low/close would keep changing after
    the fact -- it is not a real, reproducible closed bar, and using
    it for indicators or signals would be unsound.

    This provider parses each kline's close_time (the 7th field,
    index 6) and only ever returns candles whose close_time has
    fully elapsed relative to the current UTC wall-clock time. A
    still-forming candle is filtered out, never silently substituted
    or returned as if it were closed.
    """

    BASE_URL = "https://api.binance.us"

    # -------------------------------------------------------------
    # Closed-candle detection
    # -------------------------------------------------------------

    @staticmethod
    def _is_kline_closed(close_time_ms: int, *, now: datetime | None = None) -> bool:
        """
        A Binance kline's close_time is the last millisecond that is
        still inside that interval (open_time + interval_duration -
        1ms). The candle is only guaranteed final once wall-clock
        time has fully passed that instant.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        now_ms = int(now.timestamp() * 1000)

        return now_ms > close_time_ms

    @staticmethod
    def _kline_to_candle(symbol: str, timeframe: str, k: list) -> Candle:
        open_time_ms = int(k[0])

        return Candle(
            symbol=symbol,
            timeframe=timeframe,
            open=float(k[1]),
            high=float(k[2]),
            low=float(k[3]),
            close=float(k[4]),
            volume=float(k[5]),
            timestamp=datetime.fromtimestamp(
                open_time_ms / 1000.0,
                tz=timezone.utc,
            ),
        )

    @classmethod
    def _closed_candles_from_klines(
        cls,
        symbol: str,
        timeframe: str,
        klines: list,
        *,
        now: datetime | None = None,
    ) -> list[Candle]:
        """
        Convert raw Binance kline rows into Candle objects, dropping
        any kline whose interval has not fully closed yet. Order is
        preserved (Binance returns klines oldest-first).

        This is a pure function (no network I/O) so it can be unit
        tested deterministically without hitting the live API.
        """
        closed_klines = [
            k for k in klines
            if cls._is_kline_closed(int(k[6]), now=now)
        ]

        return [
            cls._kline_to_candle(symbol, timeframe, k)
            for k in closed_klines
        ]

    # -------------------------------------------------------------
    # Ticker
    # -------------------------------------------------------------

    async def get_ticker(self, symbol: str) -> Ticker:
        symbol = symbol.upper()

        async with httpx.AsyncClient(
            base_url=self.BASE_URL,
            timeout=10.0,
        ) as client:
            response = await client.get(
                "/api/v3/ticker/24hr",
                params={"symbol": symbol},
            )

            response.raise_for_status()
            data = response.json()

        return Ticker(
            symbol=data["symbol"],
            price=float(data["lastPrice"]),
            price_change_24h=float(data["priceChangePercent"]),
            high_24h=float(data["highPrice"]),
            low_24h=float(data["lowPrice"]),
            volume_24h=float(data["volume"]),
            timestamp=datetime.now(timezone.utc),
        )

    # -------------------------------------------------------------
    # Latest candle
    # -------------------------------------------------------------

    async def get_candle(
        self,
        symbol: str,
        timeframe: str = "1m",
    ) -> Candle:
        """
        Return the most recent CLOSED candle for `symbol`/`timeframe`.

        Requests 2 klines (not 1): Binance's most recent kline is
        typically still forming, so a buffer row is needed to reach
        the last genuinely closed one. Raises ValueError, rather than
        silently returning stale or incomplete data, if no closed
        candle is available in the response.
        """
        symbol = symbol.upper()

        async with httpx.AsyncClient(
            base_url=self.BASE_URL,
            timeout=10.0,
        ) as client:
            response = await client.get(
                "/api/v3/klines",
                params={
                    "symbol": symbol,
                    "interval": timeframe,
                    "limit": 2,
                },
            )

            response.raise_for_status()
            data = response.json()

        if not data:
            raise ValueError(
                f"No candle data returned for {symbol} {timeframe}"
            )

        candles = self._closed_candles_from_klines(symbol, timeframe, data)

        if not candles:
            raise ValueError(
                f"No closed candle available yet for {symbol} "
                f"{timeframe}; the most recent interval is still "
                f"forming."
            )

        return candles[-1]

    async def get_candles(
        self,
        symbols: list[str],
        timeframe: str = "1m",
    ) -> list[Candle]:
        results: list[Candle] = []

        for symbol in symbols:
            try:
                candle = await self.get_candle(symbol, timeframe)
                results.append(candle)
            except Exception:
                continue

        return results

    # -------------------------------------------------------------
    # Historical candles
    # -------------------------------------------------------------

    async def get_historical_candles(
        self,
        symbol: str,
        timeframe: str = "1m",
        limit: int = 500,
    ) -> list[Candle]:
        """
        Fetch up to `limit` historical, fully CLOSED OHLCV candles
        from Binance US.

        Binance allows a maximum of 1000 klines per request and its
        most recent kline is typically still forming. To still return
        up to `limit` closed candles, this requests one extra kline
        as a buffer and drops any still-forming trailing entry.

        If `limit` is already 1000 (Binance's hard per-request cap),
        there is no room for that buffer row; in that edge case this
        may return fewer than `limit` candles rather than ever
        returning an incomplete one.
        """
        symbol = symbol.upper()

        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")

        fetch_limit = min(limit + 1, 1000)

        async with httpx.AsyncClient(
            base_url=self.BASE_URL,
            timeout=15.0,
        ) as client:
            response = await client.get(
                "/api/v3/klines",
                params={
                    "symbol": symbol,
                    "interval": timeframe,
                    "limit": fetch_limit,
                },
            )

            response.raise_for_status()
            data = response.json()

        candles = self._closed_candles_from_klines(symbol, timeframe, data)

        # Keep only the most recent `limit` closed candles.
        return candles[-limit:]
