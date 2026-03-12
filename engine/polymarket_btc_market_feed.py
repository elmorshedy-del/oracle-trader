from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import websockets

UTC = timezone.utc
DEFAULT_POLYMARKET_MARKET_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

logger = logging.getLogger(__name__)


@dataclass
class PolymarketAssetQuote:
    asset_id: str
    best_bid: float | None
    best_ask: float | None
    midpoint: float | None
    spread: float | None
    updated_at: str
    event_type: str


class PolymarketBtcMarketFeed:
    def __init__(
        self,
        *,
        ws_url: str = DEFAULT_POLYMARKET_MARKET_WS_URL,
        ping_seconds: int = 10,
        quote_ttl_seconds: int = 25,
        max_watch_assets: int = 120,
        log_path: Path | None = None,
    ):
        self.ws_url = ws_url
        self.ping_seconds = max(5, int(ping_seconds))
        self.quote_ttl = timedelta(seconds=max(5, int(quote_ttl_seconds)))
        self.max_watch_assets = max(8, int(max_watch_assets))
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self._quotes: dict[str, PolymarketAssetQuote] = {}
        self._watched_assets: tuple[str, ...] = ()
        self._watch_revision = 0
        self._tasks: list[asyncio.Task] = []
        self._stop = asyncio.Event()
        self._started = False
        self._socket: Any = None
        self._stats: dict[str, Any] = {
            "connected": False,
            "reconnects": 0,
            "feed_errors": 0,
            "last_error": "",
            "watched_assets": 0,
            "quoted_assets": 0,
            "last_watch_update_at": None,
            "last_message_at": None,
            "last_quote_at": None,
            "last_pong_at": None,
            "quote_updates": 0,
            "book_updates": 0,
            "best_bid_ask_updates": 0,
            "heartbeat_sent": 0,
            "log_entries": 0,
            "last_log_at": None,
        }

    @property
    def stats(self) -> dict[str, Any]:
        stats = dict(self._stats)
        stats["watched_assets"] = len(self._watched_assets)
        stats["quoted_assets"] = len(self._fresh_quotes())
        return stats

    async def ensure_started(self) -> None:
        if self._started:
            return
        self._started = True
        self._tasks = [
            asyncio.create_task(self._ws_loop(), name="polymarket-btc-market-feed"),
        ]

    async def close(self) -> None:
        self._stop.set()
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    async def update_watchlist(self, asset_ids: list[str]) -> None:
        normalized = tuple(dict.fromkeys(asset for asset in asset_ids if asset))[: self.max_watch_assets]
        if normalized == self._watched_assets:
            return
        self._watched_assets = normalized
        self._watch_revision += 1
        self._stats["last_watch_update_at"] = datetime.now(UTC).isoformat()
        socket = self._socket
        if socket is not None and normalized:
            try:
                await self._send_subscribe(socket, normalized)
            except Exception as exc:
                self._stats["feed_errors"] += 1
                self._stats["last_error"] = f"watchlist:{type(exc).__name__}"
                logger.warning("[BTC_ML] Polymarket watchlist refresh failed: %s", exc)

    def quote(self, asset_id: str) -> PolymarketAssetQuote | None:
        quote = self._quotes.get(asset_id)
        if not quote:
            return None
        if datetime.now(UTC) - _parse_iso(quote.updated_at) > self.quote_ttl:
            return None
        return quote

    def append_log(self, payload: dict[str, Any]) -> None:
        if not self.log_path:
            return
        try:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, default=str) + "\n")
            self._stats["log_entries"] += 1
            self._stats["last_log_at"] = datetime.now(UTC).isoformat()
        except OSError as exc:
            self._stats["feed_errors"] += 1
            self._stats["last_error"] = f"log:{type(exc).__name__}"
            logger.warning("[BTC_ML] Failed to write Polymarket feed log: %s", exc)

    async def _ws_loop(self) -> None:
        while not self._stop.is_set():
            try:
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=None,
                    ping_timeout=None,
                    max_queue=20000,
                    close_timeout=5,
                ) as socket:
                    self._socket = socket
                    self._stats["connected"] = True
                    self._stats["last_error"] = ""
                    current_watch_revision = self._watch_revision
                    if self._watched_assets:
                        await self._send_subscribe(socket, self._watched_assets)
                    while not self._stop.is_set():
                        try:
                            raw_message = await asyncio.wait_for(socket.recv(), timeout=self.ping_seconds)
                        except asyncio.TimeoutError:
                            await socket.send("PING")
                            self._stats["heartbeat_sent"] += 1
                            continue
                        if current_watch_revision != self._watch_revision and self._watched_assets:
                            await self._send_subscribe(socket, self._watched_assets)
                            current_watch_revision = self._watch_revision
                        await self._handle_message(raw_message)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._socket = None
                self._stats["connected"] = False
                self._stats["feed_errors"] += 1
                self._stats["reconnects"] += 1
                self._stats["last_error"] = f"market_ws:{type(exc).__name__}"
                logger.warning("[BTC_ML] Polymarket market feed reconnect after error: %s", exc)
                await asyncio.sleep(2)

    async def _send_subscribe(self, socket: Any, asset_ids: tuple[str, ...]) -> None:
        if not asset_ids:
            return
        await socket.send(
            json.dumps(
                {
                    "type": "market",
                    "assets_ids": list(asset_ids),
                    "operation": "subscribe",
                    "custom_feature_enabled": True,
                }
            )
        )

    async def _handle_message(self, raw_message: Any) -> None:
        received_at = datetime.now(UTC)
        self._stats["last_message_at"] = received_at.isoformat()
        if isinstance(raw_message, bytes):
            raw_message = raw_message.decode("utf-8", errors="replace")
        if isinstance(raw_message, str):
            stripped = raw_message.strip()
            if stripped.upper() == "PONG":
                self._stats["last_pong_at"] = received_at.isoformat()
                return
            if not stripped:
                return
            payload = json.loads(stripped)
        else:
            payload = raw_message

        for item in _iter_payload_items(payload):
            quote = _quote_from_payload(item, received_at)
            if quote is None:
                continue
            self._quotes[quote.asset_id] = quote
            self._stats["quote_updates"] += 1
            self._stats["last_quote_at"] = quote.updated_at
            if quote.event_type == "best_bid_ask":
                self._stats["best_bid_ask_updates"] += 1
            else:
                self._stats["book_updates"] += 1
        self._trim_stale_quotes()

    def _trim_stale_quotes(self) -> None:
        fresh_ids = {
            asset_id
            for asset_id, quote in self._quotes.items()
            if datetime.now(UTC) - _parse_iso(quote.updated_at) <= self.quote_ttl
        }
        if len(fresh_ids) == len(self._quotes):
            return
        self._quotes = {
            asset_id: quote
            for asset_id, quote in self._quotes.items()
            if asset_id in fresh_ids
        }

    def _fresh_quotes(self) -> dict[str, PolymarketAssetQuote]:
        now = datetime.now(UTC)
        return {
            asset_id: quote
            for asset_id, quote in self._quotes.items()
            if now - _parse_iso(quote.updated_at) <= self.quote_ttl
        }


def _iter_payload_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("data", "events", "payload", "book", "books", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            return [value]
    return [payload]


def _quote_from_payload(payload: dict[str, Any], received_at: datetime) -> PolymarketAssetQuote | None:
    asset_id = _first_string(payload, "asset_id", "assetId", "asset", "token_id", "tokenId")
    if not asset_id:
        asset_id = _first_string(payload.get("market") or {}, "asset_id", "assetId")
    if not asset_id:
        return None

    event_type = (_first_string(payload, "event_type", "eventType", "type", "event") or "book").lower()
    best_bid = _first_price(payload, "best_bid", "bestBid", "bid", "best_bid_price")
    best_ask = _first_price(payload, "best_ask", "bestAsk", "ask", "best_ask_price")
    if best_bid is None or best_ask is None:
        bids = payload.get("bids") or payload.get("buy") or []
        asks = payload.get("asks") or payload.get("sell") or []
        if best_bid is None:
            best_bid = _best_side_price(bids, prefer="max")
        if best_ask is None:
            best_ask = _best_side_price(asks, prefer="min")

    midpoint = _first_price(payload, "mid", "midpoint")
    if midpoint is None and best_bid is not None and best_ask is not None:
        midpoint = (best_bid + best_ask) / 2.0
    spread = None
    if best_bid is not None and best_ask is not None:
        spread = max(best_ask - best_bid, 0.0)

    return PolymarketAssetQuote(
        asset_id=asset_id,
        best_bid=best_bid,
        best_ask=best_ask,
        midpoint=midpoint,
        spread=spread,
        updated_at=received_at.isoformat(),
        event_type="best_bid_ask" if "best_bid_ask" in event_type else event_type,
    )


def _first_string(payload: dict[str, Any], *keys: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _first_price(payload: dict[str, Any], *keys: str) -> float | None:
    if not isinstance(payload, dict):
        return None
    for key in keys:
        value = payload.get(key)
        try:
            if value is None:
                continue
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _best_side_price(levels: Any, *, prefer: str) -> float | None:
    prices: list[float] = []
    if not isinstance(levels, list):
        return None
    for level in levels:
        if isinstance(level, dict):
            raw = level.get("price") or level.get("p")
        elif isinstance(level, (list, tuple)) and level:
            raw = level[0]
        else:
            raw = None
        try:
            if raw is not None:
                prices.append(float(raw))
        except (TypeError, ValueError):
            continue
    if not prices:
        return None
    return max(prices) if prefer == "max" else min(prices)


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
