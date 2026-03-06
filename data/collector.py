"""
Data Collector
==============
Fetches market data from Polymarket Gamma API and CLOB API.
Read-only operations — no authentication needed.
"""

import httpx
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional
from data.models import Market, Event, Outcome, WhaleWallet

logger = logging.getLogger(__name__)


class PolymarketCollector:
    """Collects and normalizes data from Polymarket APIs."""

    def __init__(self, gamma_host: str, clob_host: str, data_host: str):
        self.gamma = gamma_host
        self.clob = clob_host
        self.data = data_host
        self.client = httpx.AsyncClient(timeout=30.0)
        self._market_cache: dict[str, Market] = {}
        self._event_cache: dict[str, Event] = {}

    async def close(self):
        await self.client.aclose()

    # ------------------------------------------------------------------
    # Gamma API — Market Discovery
    # ------------------------------------------------------------------

    async def get_active_markets(
        self, limit: int = 100, offset: int = 0, tag: str = ""
    ) -> list[Market]:
        """Fetch active markets from Gamma API."""
        params = {
            "limit": limit,
            "offset": offset,
            "active": "true",
            "closed": "false",
        }
        if tag:
            params["tag"] = tag

        try:
            resp = await self.client.get(f"{self.gamma}/markets", params=params)
            resp.raise_for_status()
            raw_markets = resp.json()
            markets = []
            for m in raw_markets:
                market = self._parse_gamma_market(m)
                if market:
                    markets.append(market)
                    self._market_cache[market.condition_id] = market
            return markets
        except Exception as e:
            logger.error(f"Failed to fetch markets: {e}")
            return []

    async def get_all_active_markets(self) -> list[Market]:
        """Paginate through all active markets."""
        all_markets = []
        offset = 0
        batch_size = 100
        pages = 0
        while True:
            batch = await self.get_active_markets(limit=batch_size, offset=offset)
            if not batch:
                break
            all_markets.extend(batch)
            pages += 1
            if len(batch) < batch_size:
                break
            offset += batch_size
            await asyncio.sleep(0.2)  # rate limit courtesy
        logger.info(
            "[COLLECTOR] Active markets refresh: %s markets across %s pages",
            len(all_markets),
            pages,
        )
        return all_markets

    async def get_events(self, limit: int = 50, offset: int = 0) -> list[Event]:
        """Fetch events (groups of related markets)."""
        params = {"limit": limit, "offset": offset, "active": "true", "closed": "false"}
        try:
            resp = await self.client.get(f"{self.gamma}/events", params=params)
            resp.raise_for_status()
            raw_events = resp.json()
            events = []
            for e in raw_events:
                event = self._parse_gamma_event(e)
                if event:
                    events.append(event)
                    self._event_cache[event.event_id] = event
            logger.info("[COLLECTOR] Events refresh: %s events", len(events))
            return events
        except Exception as e:
            logger.error(f"Failed to fetch events: {e}")
            return []

    async def get_market_by_slug(self, slug: str) -> Optional[Market]:
        """Get a specific market by slug."""
        try:
            resp = await self.client.get(f"{self.gamma}/markets", params={"slug": slug})
            resp.raise_for_status()
            data = resp.json()
            if data and len(data) > 0:
                return self._parse_gamma_market(data[0])
        except Exception as e:
            logger.error(f"Failed to fetch market {slug}: {e}")
        return None

    # ------------------------------------------------------------------
    # CLOB API — Order Book & Prices
    # ------------------------------------------------------------------

    async def get_orderbook(self, token_id: str) -> dict:
        """Get order book for a specific token."""
        try:
            resp = await self.client.get(f"{self.clob}/book", params={"token_id": token_id})
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch orderbook for {token_id}: {e}")
            return {}

    async def get_midpoint(self, token_id: str) -> Optional[float]:
        """Get midpoint price for a token."""
        try:
            resp = await self.client.get(f"{self.clob}/midpoint", params={"token_id": token_id})
            resp.raise_for_status()
            data = resp.json()
            return float(data.get("mid", 0))
        except Exception as e:
            logger.error(f"Failed to fetch midpoint for {token_id}: {e}")
            return None

    async def get_price(self, token_id: str, side: str = "BUY") -> Optional[float]:
        """Get best price for a token on a given side."""
        try:
            resp = await self.client.get(
                f"{self.clob}/price",
                params={"token_id": token_id, "side": side}
            )
            resp.raise_for_status()
            data = resp.json()
            return float(data.get("price", 0))
        except Exception as e:
            logger.error(f"Failed to fetch price for {token_id}: {e}")
            return None

    async def get_last_trade_price(self, token_id: str) -> Optional[float]:
        """Get last trade price for a token."""
        try:
            resp = await self.client.get(
                f"{self.clob}/last-trade-price",
                params={"token_id": token_id}
            )
            resp.raise_for_status()
            data = resp.json()
            return float(data.get("price", 0))
        except Exception as e:
            logger.error(f"Failed to fetch last trade for {token_id}: {e}")
            return None

    async def enrich_market_with_clob(self, market: Market) -> Market:
        """Add real-time CLOB data (orderbook, midpoint) to a market."""
        for outcome in market.outcomes:
            mid = await self.get_midpoint(outcome.token_id)
            if mid is not None:
                outcome.price = mid

            book = await self.get_orderbook(outcome.token_id)
            if book:
                bids = book.get("bids", [])
                asks = book.get("asks", [])
                outcome.book_bid = float(bids[0]["price"]) if bids else None
                outcome.book_ask = float(asks[0]["price"]) if asks else None

        # Recalculate spread and midpoint
        if len(market.outcomes) >= 2:
            yes = market.outcomes[0]
            no = market.outcomes[1]
            if yes.book_bid and yes.book_ask:
                market.spread = yes.book_ask - yes.book_bid
            market.midpoint = yes.price

        market.fetched_at = datetime.now(timezone.utc)
        return market

    # ------------------------------------------------------------------
    # Data API — Wallet / Activity tracking
    # ------------------------------------------------------------------

    async def get_wallet_positions(self, address: str) -> list[dict]:
        """Get positions for a specific wallet."""
        try:
            resp = await self.client.get(
                f"{self.data}/positions",
                params={"user": address, "sizeThreshold": 0}
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch positions for {address}: {e}")
            return []

    async def get_wallet_activity(
        self, address: str, limit: int = 50
    ) -> list[dict]:
        """Get recent activity for a wallet."""
        try:
            resp = await self.client.get(
                f"{self.data}/activity",
                params={"user": address, "limit": limit}
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch activity for {address}: {e}")
            return []

    async def get_leaderboard(self, limit: int = 50) -> list[dict]:
        """Get top traders from Polymarket leaderboard."""
        try:
            resp = await self.client.get(
                f"{self.data}/v1/leaderboard",
                params={"limit": limit, "window": "all"}
            )
            resp.raise_for_status()
            data = resp.json()
            # Handle different response formats (list vs paginated dict)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("results", data.get("data", data.get("leaderboard", [])))
            return []
        except Exception as e:
            logger.error(f"Failed to fetch leaderboard: {e}")
            return []

    # ------------------------------------------------------------------
    # Price History
    # ------------------------------------------------------------------

    async def get_price_history(
        self, token_id: str, interval: str = "1h", fidelity: int = 60
    ) -> list[dict]:
        """Get historical prices for a token (from CLOB timeseries)."""
        try:
            resp = await self.client.get(
                f"{self.clob}/prices-history",
                params={"market": token_id, "interval": interval, "fidelity": fidelity}
            )
            resp.raise_for_status()
            return resp.json().get("history", [])
        except Exception as e:
            logger.error(f"Failed to fetch price history for {token_id}: {e}")
            return []

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    def _parse_gamma_market(self, m: dict) -> Optional[Market]:
        """Parse a raw Gamma API market into our model."""
        try:
            outcomes = []
            tokens = m.get("clobTokenIds", "")
            prices = m.get("outcomePrices", "")

            if isinstance(tokens, str):
                tokens = [t.strip() for t in tokens.strip("[]").split(",") if t.strip()]
            if isinstance(prices, str):
                prices = [p.strip().strip('"') for p in prices.strip("[]").split(",") if p.strip()]

            outcome_names = ["Yes", "No"]
            for i, (tid, price) in enumerate(zip(tokens, prices)):
                tid = tid.strip().strip('"')
                try:
                    p = float(price)
                except (ValueError, TypeError):
                    p = 0.0
                outcomes.append(Outcome(
                    token_id=tid,
                    name=outcome_names[i] if i < 2 else f"Outcome {i+1}",
                    price=p,
                ))

            if not outcomes:
                return None

            # Parse reward info
            reward_pool = 0.0
            max_spread = 0.0
            min_shares = 0
            clob_rewards = m.get("clobRewards", [])
            if clob_rewards and len(clob_rewards) > 0:
                r = clob_rewards[0] if isinstance(clob_rewards[0], dict) else {}
                reward_pool = float(r.get("rewardsDailyRate", 0) or 0)
                max_spread = float(r.get("maxSpread", 0) or 0) or 0.05
                min_shares = int(r.get("minSize", 0) or 0)

            return Market(
                condition_id=m.get("conditionId", m.get("condition_id", "")),
                question=m.get("question", ""),
                slug=m.get("slug", ""),
                outcomes=outcomes,
                volume_24h=float(m.get("volume24hr", 0) or 0),
                volume_total=float(m.get("volumeNum", 0) or 0),
                liquidity=float(m.get("liquidity", 0) or 0),
                spread=float(m.get("spread", 0) or 0),
                end_date=m.get("endDate") or m.get("end_date_iso"),
                active=m.get("active", True),
                closed=m.get("closed", False),
                neg_risk=m.get("negRisk", False),
                tags=[t.get("label", "") for t in m.get("tags", []) if isinstance(t, dict)],
                reward_pool=reward_pool,
                max_spread_for_rewards=max_spread,
                min_shares_for_rewards=min_shares,
            )
        except Exception as e:
            logger.error(f"Failed to parse market: {e}")
            return None

    def _parse_gamma_event(self, e: dict) -> Optional[Event]:
        """Parse a raw Gamma API event."""
        try:
            markets = []
            for m in e.get("markets", []):
                market = self._parse_gamma_market(m)
                if market:
                    markets.append(market)
            return Event(
                event_id=str(e.get("id", "")),
                slug=e.get("slug", ""),
                title=e.get("title", ""),
                markets=markets,
                total_volume=float(e.get("volume", 0) or 0),
            )
        except Exception as e_err:
            logger.error(f"Failed to parse event: {e_err}")
            return None
