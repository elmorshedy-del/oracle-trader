from __future__ import annotations

import asyncio
import json
import logging
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

UTC = timezone.utc
NEWSAPI_URL = "https://newsapi.org/v2/everything"
GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
X_RULES_URL = "https://api.x.com/2/tweets/search/stream/rules"
X_STREAM_URL = "https://api.x.com/2/tweets/search/stream"
MAX_CONTEXT_ITEMS = 500
MAX_RECENT_SUMMARY_ITEMS = 6
BTC_RSS_KEYWORDS = (
    "bitcoin",
    " btc ",
    "btc ",
    " btc",
    "crypto",
    "cryptocurrency",
    "etf",
    "blockchain",
)

BTC_BULLISH_TERMS = {
    "approval", "adoption", "accumulate", "inflow", "reserve", "treasury",
    "buying", "bullish", "easing", "rate cut", "partnership", "breakout",
    "etf inflow", "sovereign", "institutional", "listing",
}
BTC_BEARISH_TERMS = {
    "hack", "exploit", "ban", "lawsuit", "charge", "outflow", "liquidation",
    "selloff", "dump", "bearish", "rate hike", "crackdown", "delay",
    "rejection", "bankruptcy", "outage", "fraud",
}
BTC_HIGH_IMPACT_TERMS = {
    "sec", "fed", "etf", "liquidation", "treasury", "reserve", "hack",
    "exploit", "approval", "rejection", "rate cut", "rate hike", "bankruptcy",
}

logger = logging.getLogger(__name__)


@dataclass
class ContextSnapshot:
    regime: str
    bias: str
    intensity: float
    recent_count: int
    provider_count: int
    healthy_provider_count: int
    last_item_at: str | None
    hold_profile: str
    notes: list[str]


class BitcoinContextFeed:
    def __init__(
        self,
        *,
        enabled: bool,
        query: str,
        shock_window_minutes: int,
        newsapi_key: str,
        newsapi_poll_seconds: int,
        newsapi_page_size: int,
        gdelt_enabled: bool,
        gdelt_poll_seconds: int,
        gdelt_max_records: int,
        rss_feeds: list[str] | None = None,
        rss_poll_seconds: int = 180,
        x_bearer_token: str,
        x_stream_enabled: bool,
        x_rule_tag: str,
        x_rule_value: str,
        log_path: Path | None = None,
    ):
        self.enabled = enabled
        self.query = query
        self.shock_window = timedelta(minutes=max(10, int(shock_window_minutes)))
        self.newsapi_key = newsapi_key.strip()
        self.newsapi_poll_seconds = max(60, int(newsapi_poll_seconds))
        self.newsapi_page_size = max(5, int(newsapi_page_size))
        self.gdelt_enabled = gdelt_enabled
        self.gdelt_poll_seconds = max(60, int(gdelt_poll_seconds))
        self.gdelt_max_records = max(5, int(gdelt_max_records))
        self.rss_feeds = tuple(feed.strip() for feed in (rss_feeds or []) if str(feed).strip())
        self.rss_poll_seconds = max(120, int(rss_poll_seconds))
        self.x_bearer_token = x_bearer_token.strip()
        self.x_stream_enabled = x_stream_enabled
        self.x_rule_tag = x_rule_tag.strip() or "oracle-btc-context"
        self.x_rule_value = x_rule_value.strip()
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.client = httpx.AsyncClient(timeout=20.0, headers={"User-Agent": "oracle-trader/1.0"})
        self._tasks: list[asyncio.Task] = []
        self._started = False
        self._stop = asyncio.Event()
        self._items: deque[dict[str, Any]] = deque(maxlen=MAX_CONTEXT_ITEMS)
        self._seen_ids: set[str] = set()
        self._stats: dict[str, Any] = {
            "enabled": enabled,
            "newsapi_configured": bool(self.newsapi_key),
            "gdelt_enabled": gdelt_enabled,
            "rss_enabled": bool(self.rss_feeds),
            "x_configured": bool(self.x_bearer_token and self.x_rule_value),
            "newsapi_ok": False,
            "gdelt_ok": False,
            "rss_ok": False,
            "x_connected": False,
            "recent_items": 0,
            "last_item_at": None,
            "last_error": "",
            "feed_errors": 0,
            "newsapi_polls": 0,
            "gdelt_polls": 0,
            "rss_polls": 0,
            "x_items": 0,
            "log_entries": 0,
            "last_log_at": None,
            "regime": "disabled" if not enabled else "normal_flow",
            "bias": "neutral",
            "intensity": 0.0,
            "provider_count": 0,
            "healthy_provider_count": 0,
        }

    @property
    def stats(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        stats = dict(self._stats)
        stats.update(
            {
                "regime": snapshot.regime,
                "bias": snapshot.bias,
                "intensity": snapshot.intensity,
                "recent_items": snapshot.recent_count,
                "provider_count": snapshot.provider_count,
                "healthy_provider_count": snapshot.healthy_provider_count,
                "last_item_at": snapshot.last_item_at,
                "hold_profile": snapshot.hold_profile,
                "notes": snapshot.notes,
            }
        )
        return stats

    async def ensure_started(self) -> None:
        if self._started or not self.enabled:
            return
        self._started = True
        if self.newsapi_key:
            self._tasks.append(asyncio.create_task(self._newsapi_loop(), name="btc-newsapi-context"))
        if self.gdelt_enabled:
            self._tasks.append(asyncio.create_task(self._gdelt_loop(), name="btc-gdelt-context"))
        if self.rss_feeds:
            self._tasks.append(asyncio.create_task(self._rss_loop(), name="btc-rss-context"))
        if self.x_stream_enabled and self.x_bearer_token and self.x_rule_value:
            self._tasks.append(asyncio.create_task(self._x_stream_loop(), name="btc-x-context"))

    async def close(self) -> None:
        self._stop.set()
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        await self.client.aclose()

    def snapshot(self) -> ContextSnapshot:
        provider_count = (
            int(bool(self.newsapi_key))
            + int(bool(self.gdelt_enabled))
            + int(bool(self.rss_feeds))
            + int(bool(self.x_bearer_token and self.x_rule_value and self.x_stream_enabled))
        )
        healthy_provider_count = (
            int(bool(self._stats.get("newsapi_ok")))
            + int(bool(self._stats.get("gdelt_ok")))
            + int(bool(self._stats.get("rss_ok")))
            + int(bool(self._stats.get("x_connected")))
        )
        if not self.enabled:
            return ContextSnapshot("disabled", "neutral", 0.0, 0, provider_count, healthy_provider_count, None, "normal", ["context disabled"])

        now = datetime.now(UTC)
        recent = [item for item in self._items if now - item["published_at"] <= self.shock_window]
        bull_score = 0.0
        bear_score = 0.0
        notes: list[str] = []
        last_item_at = recent[-1]["published_at"].isoformat() if recent else self._stats.get("last_item_at")
        for item in recent:
            age_minutes = max((now - item["published_at"]).total_seconds() / 60.0, 0.0)
            freshness_weight = max(0.25, 1.0 - (age_minutes / max(self.shock_window.total_seconds() / 60.0, 1.0)))
            magnitude = float(item.get("magnitude") or 0.0)
            weighted = freshness_weight * magnitude
            if item["bias"] == "bull":
                bull_score += weighted
            elif item["bias"] == "bear":
                bear_score += weighted

        total_score = bull_score + bear_score
        intensity = min(1.0, total_score / 3.5) if total_score > 0 else 0.0
        bias = "neutral"
        if bull_score >= max(bear_score * 1.25, 0.75):
            bias = "bull"
        elif bear_score >= max(bull_score * 1.25, 0.75):
            bias = "bear"

        if intensity >= 0.65 and bias != "neutral":
            regime = f"{bias}_news_shock"
            hold_profile = "extended"
        elif intensity >= 0.30 or len(recent) >= 3:
            regime = "elevated_news_flow"
            hold_profile = "normal"
        else:
            regime = "normal_flow"
            hold_profile = "normal"

        for item in recent[-MAX_RECENT_SUMMARY_ITEMS:]:
            notes.append(f"{item['source']}:{item['bias']}:{item['title'][:80]}")

        return ContextSnapshot(
            regime=regime,
            bias=bias,
            intensity=round(intensity, 4),
            recent_count=len(recent),
            provider_count=provider_count,
            healthy_provider_count=healthy_provider_count,
            last_item_at=last_item_at,
            hold_profile=hold_profile,
            notes=notes,
        )

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
            logger.warning("[BTC_ML] Failed to write BTC context log: %s", exc)

    async def _newsapi_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self._poll_newsapi()
                self._stats["newsapi_ok"] = True
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._stats["newsapi_ok"] = False
                self._stats["feed_errors"] += 1
                self._stats["last_error"] = f"newsapi:{type(exc).__name__}"
                logger.warning("[BTC_ML] NewsAPI context poll failed: %s", exc)
            await asyncio.sleep(self.newsapi_poll_seconds)

    async def _gdelt_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self._poll_gdelt()
                self._stats["gdelt_ok"] = True
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._stats["gdelt_ok"] = False
                self._stats["feed_errors"] += 1
                self._stats["last_error"] = f"gdelt:{type(exc).__name__}"
                logger.warning("[BTC_ML] GDELT context poll failed: %s", exc)
            await asyncio.sleep(self.gdelt_poll_seconds)

    async def _rss_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self._poll_rss()
                self._stats["rss_ok"] = True
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._stats["rss_ok"] = False
                self._stats["feed_errors"] += 1
                self._stats["last_error"] = f"rss:{type(exc).__name__}"
                logger.warning("[BTC_ML] RSS context poll failed: %s", exc)
            await asyncio.sleep(self.rss_poll_seconds)

    async def _x_stream_loop(self) -> None:
        headers = {"Authorization": f"Bearer {self.x_bearer_token}"}
        while not self._stop.is_set():
            try:
                await self._ensure_x_rule(headers)
                params = {"tweet.fields": "created_at,lang,public_metrics"}
                async with self.client.stream("GET", X_STREAM_URL, headers=headers, params=params) as response:
                    response.raise_for_status()
                    self._stats["x_connected"] = True
                    async for line in response.aiter_lines():
                        if self._stop.is_set():
                            break
                        if not line:
                            continue
                        payload = json.loads(line)
                        tweet = payload.get("data") or {}
                        text = (tweet.get("text") or "").strip()
                        if not text:
                            continue
                        published_at = _parse_datetime(tweet.get("created_at"))
                        self._ingest_item(
                            item_id=f"x:{tweet.get('id')}",
                            source="x",
                            title=text,
                            url=f"https://x.com/i/web/status/{tweet.get('id')}",
                            published_at=published_at,
                        )
                        self._stats["x_items"] += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._stats["x_connected"] = False
                self._stats["feed_errors"] += 1
                self._stats["last_error"] = f"x:{type(exc).__name__}"
                logger.warning("[BTC_ML] X stream reconnect after error: %s", exc)
                await asyncio.sleep(5)

    async def _poll_newsapi(self) -> None:
        params = {
            "q": self.query,
            "sortBy": "publishedAt",
            "language": "en",
            "searchIn": "title,description",
            "pageSize": self.newsapi_page_size,
        }
        headers = {"X-Api-Key": self.newsapi_key}
        response = await self.client.get(NEWSAPI_URL, params=params, headers=headers)
        response.raise_for_status()
        payload = response.json()
        for article in payload.get("articles") or []:
            title = (article.get("title") or "").strip()
            if not title:
                continue
            published_at = _parse_datetime(article.get("publishedAt"))
            self._ingest_item(
                item_id=f"newsapi:{article.get('url') or title}",
                source="newsapi",
                title=title,
                url=article.get("url") or "",
                published_at=published_at,
            )
        self._stats["newsapi_polls"] += 1

    async def _poll_gdelt(self) -> None:
        params = {
            "query": self.query,
            "mode": "ArtList",
            "sort": "DateDesc",
            "maxrecords": self.gdelt_max_records,
            "format": "json",
        }
        response = await self.client.get(GDELT_DOC_URL, params=params)
        response.raise_for_status()
        payload = response.json()
        for article in payload.get("articles") or []:
            title = (article.get("title") or "").strip()
            if not title:
                continue
            published_at = _parse_datetime(article.get("seendate"))
            self._ingest_item(
                item_id=f"gdelt:{article.get('url') or title}",
                source="gdelt",
                title=title,
                url=article.get("url") or "",
                published_at=published_at,
            )
        self._stats["gdelt_polls"] += 1

    async def _poll_rss(self) -> None:
        seen_feed = False
        for feed_url in self.rss_feeds:
            response = await self.client.get(feed_url)
            response.raise_for_status()
            seen_feed = True
            for item in _parse_rss_items(response.text):
                title = item["title"]
                summary = item.get("summary") or ""
                if not _looks_like_btc_story(title, summary):
                    continue
                self._ingest_item(
                    item_id=f"rss:{item['url'] or title}",
                    source=item["source"],
                    title=title,
                    url=item["url"],
                    published_at=item["published_at"],
                )
        if seen_feed:
            self._stats["rss_polls"] += 1

    async def _ensure_x_rule(self, headers: dict[str, str]) -> None:
        response = await self.client.get(X_RULES_URL, headers=headers)
        response.raise_for_status()
        payload = response.json()
        existing = payload.get("data") or []
        matching = [rule for rule in existing if rule.get("tag") == self.x_rule_tag]
        if any(rule.get("value") == self.x_rule_value for rule in matching):
            return
        if matching:
            await self.client.post(
                X_RULES_URL,
                headers=headers,
                json={"delete": {"ids": [rule["id"] for rule in matching if rule.get("id")]}}
            )
        add_response = await self.client.post(
            X_RULES_URL,
            headers=headers,
            json={"add": [{"value": self.x_rule_value, "tag": self.x_rule_tag}]},
        )
        add_response.raise_for_status()

    def _ingest_item(self, *, item_id: str, source: str, title: str, url: str, published_at: datetime) -> None:
        if item_id in self._seen_ids:
            return
        bias, magnitude = _classify_btc_context(title, source)
        item = {
            "id": item_id,
            "source": source,
            "title": title,
            "url": url,
            "published_at": published_at,
            "bias": bias,
            "magnitude": magnitude,
        }
        self._items.append(item)
        self._seen_ids.add(item_id)
        while len(self._seen_ids) > MAX_CONTEXT_ITEMS * 2:
            stale = self._items[0]["id"] if self._items else None
            if stale:
                self._seen_ids.discard(stale)
                self._items.popleft()
            else:
                break
        self._stats["last_item_at"] = published_at.isoformat()
        snapshot = self.snapshot()
        self._stats["regime"] = snapshot.regime
        self._stats["bias"] = snapshot.bias
        self._stats["intensity"] = snapshot.intensity
        self.append_log(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "source": source,
                "title": title,
                "bias": bias,
                "magnitude": magnitude,
                "regime": snapshot.regime,
                "intensity": snapshot.intensity,
            }
        )


def _classify_btc_context(text: str, source: str) -> tuple[str, float]:
    lowered = text.lower()
    bull_hits = sum(1 for term in BTC_BULLISH_TERMS if term in lowered)
    bear_hits = sum(1 for term in BTC_BEARISH_TERMS if term in lowered)
    impact_hits = sum(1 for term in BTC_HIGH_IMPACT_TERMS if term in lowered)
    source_weight = 0.8 if source == "x" else 1.0
    magnitude = min(1.0, source_weight * ((bull_hits + bear_hits) * 0.30 + impact_hits * 0.18 + 0.18))
    if bull_hits > bear_hits:
        return "bull", magnitude
    if bear_hits > bull_hits:
        return "bear", magnitude
    return "neutral", 0.18 if impact_hits else 0.0


def _looks_like_btc_story(title: str, summary: str) -> bool:
    text = f" {title.lower()} {summary.lower()} "
    return any(keyword in text for keyword in BTC_RSS_KEYWORDS)


def _parse_rss_items(xml_text: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    root = ET.fromstring(xml_text)
    for item in root.iter("item"):
        title_el = item.find("title")
        link_el = item.find("link")
        desc_el = item.find("description")
        pub_el = item.find("pubDate")
        title = (title_el.text or "").strip() if title_el is not None and title_el.text else ""
        if not title:
            continue
        url = (link_el.text or "").strip() if link_el is not None and link_el.text else ""
        summary = (desc_el.text or "").strip() if desc_el is not None and desc_el.text else ""
        published_at = _parse_datetime(pub_el.text if pub_el is not None else None)
        source = "rss"
        if url:
            try:
                source = url.split("/")[2]
            except IndexError:
                source = "rss"
        items.append({
            "title": title,
            "url": url,
            "summary": summary,
            "published_at": published_at,
            "source": source,
        })
    return items


def _parse_datetime(raw: Any) -> datetime:
    if isinstance(raw, datetime):
        return raw.astimezone(UTC)
    if raw is None:
        return datetime.now(UTC)
    text = str(raw).strip()
    if not text:
        return datetime.now(UTC)
    for fmt in (None, "%Y%m%dT%H%M%SZ"):
        try:
            if fmt is None:
                return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)
            return datetime.strptime(text, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return datetime.now(UTC)
