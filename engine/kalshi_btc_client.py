from __future__ import annotations

from dataclasses import dataclass

import httpx


KALSHI_MARKETS_URL = "https://api.elections.kalshi.com/trade-api/v2/markets"


@dataclass(slots=True)
class KalshiBtcMarket:
    ticker: str
    event_ticker: str
    strike: float
    yes_ask: float
    no_ask: float
    yes_ask_size: float
    no_ask_size: float
    status: str
    close_time: str | None
    expiration_time: str | None
    result: str
    subtitle: str


class KalshiBtcClient:
    def __init__(self, timeout_seconds: float = 20.0):
        self.client = httpx.AsyncClient(timeout=timeout_seconds)

    async def close(self) -> None:
        await self.client.aclose()

    async def get_event_markets(self, event_ticker: str) -> list[KalshiBtcMarket]:
        response = await self.client.get(
            KALSHI_MARKETS_URL,
            params={"event_ticker": event_ticker, "limit": 100},
        )
        response.raise_for_status()
        payload = response.json()
        raw_markets = payload.get("markets", [])
        markets: list[KalshiBtcMarket] = []
        for row in raw_markets:
            strike = row.get("floor_strike")
            if strike is None:
                continue
            markets.append(
                KalshiBtcMarket(
                    ticker=str(row.get("ticker") or ""),
                    event_ticker=str(row.get("event_ticker") or event_ticker),
                    strike=float(strike),
                    yes_ask=float(row.get("yes_ask_dollars") or 0.0),
                    no_ask=float(row.get("no_ask_dollars") or 0.0),
                    yes_ask_size=float(row.get("yes_ask_size_fp") or 0.0),
                    no_ask_size=float(row.get("no_ask_size_fp") or 0.0),
                    status=str(row.get("status") or ""),
                    close_time=row.get("close_time"),
                    expiration_time=row.get("expected_expiration_time") or row.get("expiration_time"),
                    result=str(row.get("result") or "").lower(),
                    subtitle=str(row.get("subtitle") or ""),
                )
            )
        markets.sort(key=lambda market: market.strike)
        return markets
