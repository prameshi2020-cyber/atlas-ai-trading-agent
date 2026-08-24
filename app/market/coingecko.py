from datetime import datetime, timezone
import asyncio
import json
import logging
import time
import urllib.parse
import urllib.request
import urllib.error
from email.utils import parsedate_to_datetime

from app.market.base import MarketDataProvider
from app.market.models import Ticker


logger = logging.getLogger(__name__)


class CoinGeckoProvider(MarketDataProvider):

    BASE_URL = "https://api.coingecko.com/api/v3"

    async def _fetch(self, path: str, params: dict) -> dict:
        # Handles HTTP 429 with respect to Retry-After header and
        # uses a small exponential backoff if needed. Tries up to
        # `max_attempts` times before raising.
        max_attempts = 3
        attempt = 0
        backoff_base = 1.0

        query = urllib.parse.urlencode(params)
        url = f"{self.BASE_URL}{path}?{query}"

        while attempt < max_attempts:
            attempt += 1
            try:
                # Preferred async path using httpx when available
                try:
                    import httpx  # type: ignore

                    async with httpx.AsyncClient(
                        base_url=self.BASE_URL,
                        timeout=10.0,
                    ) as client:

                        response = await client.get(path, params=params)
                        if response.status_code == 429:
                            ra = response.headers.get("Retry-After")
                            raise httpx.HTTPStatusError("429 Too Many Requests", request=response.request, response=response)

                        response.raise_for_status()
                        logger.debug("CoinGecko request: %s", url)
                        return response.json()

                except Exception as exc_httpx:
                    # Fall back to urllib for environments without httpx
                    if attempt == 1:
                        logger.debug("httpx unavailable or request failed, falling back to urllib: %s", exc_httpx)

                    try:
                        with urllib.request.urlopen(url, timeout=10) as r:
                            logger.debug("CoinGecko request (urllib): %s", url)
                            return json.load(r)
                    except urllib.error.HTTPError as e:
                        # Raise to outer handler
                        raise e

            except Exception as exc:
                # Determine if it's a 429 and whether Retry-After provided
                status = None
                headers = {}

                # httpx HTTPStatusError
                try:
                    import httpx  # type: ignore
                    if isinstance(exc, httpx.HTTPStatusError):
                        status = exc.response.status_code
                        headers = exc.response.headers
                except Exception:
                    pass

                # urllib HTTPError
                if status is None and isinstance(exc, urllib.error.HTTPError):
                    status = exc.code
                    headers = getattr(exc, 'headers', {}) or {}

                if status == 429:
                    retry_after = headers.get("Retry-After")
                    wait = None
                    if retry_after:
                        try:
                            # Retry-After can be seconds or HTTP-date
                            wait = int(retry_after)
                        except ValueError:
                            try:
                                dt = parsedate_to_datetime(retry_after)
                                wait = max(0, int((dt - datetime.now(timezone.utc)).total_seconds()))
                            except Exception:
                                wait = None

                    if wait is None:
                        wait = backoff_base * (2 ** (attempt - 1))

                    logger.warning("CoinGecko rate limited (429). Attempt %d/%d. Waiting %.1f seconds.", attempt, max_attempts, wait)
                    await asyncio.sleep(wait)
                    continue

                # Non-429 or exhausted attempts
                if attempt >= max_attempts:
                    logger.exception("CoinGecko request failed after %d attempts: %s", attempt, exc)
                    raise

                # small exponential backoff before retrying
                wait = backoff_base * (2 ** (attempt - 1))
                logger.info("CoinGecko request failed (attempt %d/%d), retrying in %.1f seconds...", attempt, max_attempts, wait)
                await asyncio.sleep(wait)

        # If we exit loop without returning, raise
        raise RuntimeError("CoinGecko request failed")

    async def get_ticker(self, symbol: str) -> Ticker:
        symbol = symbol.upper()

        symbol_map = {
            "BTCUSDT": "bitcoin",
            "ETHUSDT": "ethereum",
            "SOLUSDT": "solana",
        }

        if symbol not in symbol_map:
            raise ValueError(f"Unsupported symbol for CoinGecko: {symbol}")

        coin_id = symbol_map[symbol]

        params = {
            "ids": coin_id,
            "vs_currencies": "usd",
            "include_24hr_change": "true",
            "include_24hr_high": "true",
            "include_24hr_low": "true",
            "include_24hr_vol": "true",
        }

        data = await self._fetch("/simple/price", params)

        coin = data[coin_id]

        return Ticker(
            symbol=symbol,
            price=float(coin["usd"]),
            price_change_24h=float(coin.get("usd_24h_change", 0.0)),
            high_24h=float(coin.get("usd_24h_high", coin["usd"])),
            low_24h=float(coin.get("usd_24h_low", coin["usd"])),
            volume_24h=float(coin.get("usd_24h_vol", 0.0)),
            timestamp=datetime.now(timezone.utc),
        )

    async def get_tickers(self, symbols: list[str]) -> list[Ticker]:
        """Fetch multiple tickers in a single /simple/price request.

        Returns a list of `Ticker` objects for symbols that were
        successfully retrieved. Missing or unsupported symbols are
        skipped but logged.
        """
        # Map symbols to coin ids, filter unsupported
        symbol_map = {
            "BTCUSDT": "bitcoin",
            "ETHUSDT": "ethereum",
            "SOLUSDT": "solana",
        }

        seen_ids: list[str] = []
        requested: list[tuple[str, str]] = []  # (symbol, coin_id)
        for s in symbols:
            s_up = s.upper()
            if s_up in symbol_map:
                cid = symbol_map[s_up]
                requested.append((s_up, cid))
                if cid not in seen_ids:
                    seen_ids.append(cid)
            else:
                logger.warning("Unsupported symbol requested for batch: %s", s)

        if not seen_ids:
            return []

        params = {
            "ids": ",".join(seen_ids),
            "vs_currencies": "usd",
            "include_24hr_change": "true",
            "include_24hr_high": "true",
            "include_24hr_low": "true",
            "include_24hr_vol": "true",
        }

        logger.info("CoinGecko batch request ids=%s", params["ids"])

        data = await self._fetch("/simple/price", params)

        results: list[Ticker] = []
        for sym, cid in requested:
            coin = data.get(cid)
            if not coin:
                logger.warning("CoinGecko response missing data for %s (id=%s)", sym, cid)
                continue

            try:
                t = Ticker(
                    symbol=sym,
                    price=float(coin["usd"]),
                    price_change_24h=float(coin.get("usd_24h_change", 0.0)),
                    high_24h=float(coin.get("usd_24h_high", coin["usd"])),
                    low_24h=float(coin.get("usd_24h_low", coin["usd"])),
                    volume_24h=float(coin.get("usd_24h_vol", 0.0)),
                    timestamp=datetime.now(timezone.utc),
                )
                results.append(t)
            except Exception:
                logger.exception("Failed to parse CoinGecko data for %s", sym)

        return results