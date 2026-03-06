"""
Strategy: News-to-Price Latency (Optional LLM Layer)
====================================================
Ingests breaking news, classifies relevance + direction via Claude API,
and generates signals before the market fully prices in the information.

Requires ANTHROPIC_API_KEY environment variable.
"""

import httpx
import asyncio
import hashlib
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from data.models import Market, Event, Signal, SignalSource, SignalAction, NewsHeadline
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class NewsLatencyStrategy(BaseStrategy):
    name = "news_latency"
    description = "Classify breaking news with LLM and trade before market reacts"

    def __init__(self, config):
        super().__init__(config)
        self.cfg = config.news
        self.client = httpx.AsyncClient(timeout=15.0)
        self._seen_headlines: set[str] = set()
        self._recent_headlines: list[NewsHeadline] = []
        self._api_calls_this_hour: int = 0
        self._hour_start: datetime = datetime.now(timezone.utc)
        self._market_index: dict[str, str] = {}  # keyword -> market_slug

    async def scan(self, markets: list[Market], events: list[Event]) -> list[Signal]:
        if not self.cfg.enabled or not self.cfg.anthropic_api_key:
            return []

        self._stats["scans_completed"] += 1

        # Build keyword index of active markets
        self._build_market_index(markets)

        # Fetch new headlines
        headlines = await self._fetch_headlines()
        new_headlines = [h for h in headlines if self._is_new(h)]

        if not new_headlines:
            return []

        # Pre-filter: only send relevant headlines to Claude (save API budget)
        relevant_headlines = self._keyword_prefilter(new_headlines)
        relevant_count = len(relevant_headlines)
        if not relevant_headlines:
            # If prefilter finds nothing, send top 3 as fallback
            relevant_headlines = new_headlines[:3]
        else:
            # Cap at 10 most relevant
            relevant_headlines = relevant_headlines[:10]

        signals = []
        for headline in relevant_headlines:
            if not self._within_rate_limit():
                logger.warning("[NEWS] Rate limit reached, skipping remaining headlines")
                break

            classification = await self._classify_headline(headline, markets)
            if classification and classification.get("confidence", 0) >= self.cfg.min_confidence:
                signal = self._build_signal(headline, classification, markets)
                if signal:
                    signals.append(signal)
                    self._stats["signals_generated"] += 1

        logger.info(
            "[NEWS] Scan: %s new headlines | %s relevant | %s classified | %s signals",
            len(new_headlines),
            relevant_count,
            len(relevant_headlines),
            len(signals),
        )

        return signals

    def get_recent_headlines(self) -> list[NewsHeadline]:
        return self._recent_headlines[-50:]

    # ------------------------------------------------------------------
    # News Fetching
    # ------------------------------------------------------------------

    async def _fetch_headlines(self) -> list[NewsHeadline]:
        """Fetch headlines from configured RSS feeds."""
        headlines = []
        for feed_url in self.cfg.rss_feeds:
            try:
                resp = await self.client.get(feed_url)
                resp.raise_for_status()
                items = self._parse_rss(resp.text, feed_url)
                headlines.extend(items)
            except Exception as e:
                logger.error(f"[NEWS] Failed to fetch {feed_url}: {e}")
        return headlines

    def _parse_rss(self, xml_text: str, source_url: str) -> list[NewsHeadline]:
        """Parse RSS XML into NewsHeadline objects."""
        headlines = []
        try:
            root = ET.fromstring(xml_text)
            for item in root.iter("item"):
                title_el = item.find("title")
                link_el = item.find("link")
                pub_el = item.find("pubDate")

                if title_el is None or title_el.text is None:
                    continue

                headlines.append(NewsHeadline(
                    title=title_el.text.strip(),
                    source=source_url.split("/")[2] if "/" in source_url else source_url,
                    url=link_el.text.strip() if link_el is not None and link_el.text else "",
                    published=None,  # parse pubDate if needed
                ))
        except ET.ParseError as e:
            logger.error(f"[NEWS] RSS parse error: {e}")
        return headlines

    def _is_new(self, headline: NewsHeadline) -> bool:
        """Check if we've already seen this headline."""
        h = hashlib.md5(headline.title.encode()).hexdigest()
        if h in self._seen_headlines:
            return False
        self._seen_headlines.add(h)
        self._recent_headlines.append(headline)
        # Cap memory
        if len(self._seen_headlines) > 10000:
            self._seen_headlines = set(list(self._seen_headlines)[-5000:])
        if len(self._recent_headlines) > 200:
            self._recent_headlines = self._recent_headlines[-100:]
        return True

    # ------------------------------------------------------------------
    # Keyword Pre-Filter (saves API calls)
    # ------------------------------------------------------------------

    def _build_market_index(self, markets: list[Market]):
        """Build a keyword -> market mapping for fast filtering."""
        self._market_index.clear()
        for m in markets:
            keywords = m.question.lower().split()
            for kw in keywords:
                if len(kw) > 3:  # skip short words
                    self._market_index[kw] = m.slug

    def _keyword_prefilter(self, headlines: list[NewsHeadline]) -> list[NewsHeadline]:
        """Only keep headlines that match keywords from active markets."""
        relevant = []
        market_keywords = set(self._market_index.keys())
        for h in headlines:
            title_lower = h.title.lower()
            matched = False
            for kw in market_keywords:
                # Substring match: "iran" matches keyword "iranian" and vice versa
                if kw in title_lower or any(w in kw for w in title_lower.split() if len(w) > 3):
                    matched = True
                    break
            if matched:
                relevant.append(h)
        return relevant

    # ------------------------------------------------------------------
    # LLM Classification
    # ------------------------------------------------------------------

    async def _classify_headline(
        self, headline: NewsHeadline, markets: list[Market]
    ) -> dict | None:
        """Use Claude API to classify a headline against active markets."""
        market_list = "\n".join(
            f"- {m.slug}: \"{m.question}\" (current YES: {m.outcomes[0].price:.2f})"
            for m in markets[:50]  # limit context size
            if m.outcomes
        )

        prompt = f"""You are a prediction market analyst. Given this breaking headline and list of active prediction markets, determine:

1. Which market (if any) is most affected by this headline?
2. Is the headline BULLISH (increases probability of YES) or BEARISH (decreases)?
3. Confidence level (0.0 to 1.0) — how strongly does this headline affect the market?
4. Expected price impact in cents (how much should YES price move?)

HEADLINE: "{headline.title}"
SOURCE: {headline.source}

ACTIVE MARKETS:
{market_list}

Respond ONLY with JSON (no markdown, no backticks):
{{"market_slug": "slug-here-or-null", "direction": "bullish|bearish|neutral", "confidence": 0.0, "expected_impact_cents": 0, "reasoning": "brief explanation"}}

If no market is relevant, return {{"market_slug": null, "direction": "neutral", "confidence": 0.0, "expected_impact_cents": 0, "reasoning": "not relevant"}}"""

        try:
            resp = await self.client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.cfg.anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.cfg.model,
                    "max_tokens": 300,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            resp.raise_for_status()
            self._api_calls_this_hour += 1

            data = resp.json()
            text = data["content"][0]["text"].strip()

            # Strip markdown code blocks if Claude wrapped the JSON
            if text.startswith("```"):
                lines = text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\n".join(lines).strip()

            import json
            classification = json.loads(text)
            headline.classification = classification
            logger.debug(
                f"[NEWS] Classified: '{headline.title[:60]}' → "
                f"market={classification.get('market_slug', 'none')}, "
                f"dir={classification.get('direction', '?')}, "
                f"conf={classification.get('confidence', 0)}"
            )
            return classification

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"[NEWS] LLM classification failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Signal Building
    # ------------------------------------------------------------------

    def _build_signal(
        self, headline: NewsHeadline, classification: dict, markets: list[Market]
    ) -> Signal | None:
        """Convert a classification into a tradeable signal."""
        slug = classification.get("market_slug")
        if not slug:
            return None

        # Find the market
        market = next((m for m in markets if m.slug == slug), None)
        if not market or not market.outcomes:
            return None

        direction = classification.get("direction", "neutral")
        if direction == "neutral":
            return None

        confidence = float(classification.get("confidence", 0))
        impact = float(classification.get("expected_impact_cents", 0))

        action = SignalAction.BUY_YES if direction == "bullish" else SignalAction.BUY_NO

        return Signal(
            source=SignalSource.NEWS,
            action=action,
            market_slug=slug,
            condition_id=market.condition_id,
            token_id=market.outcomes[0].token_id if direction == "bullish" else (
                market.outcomes[1].token_id if len(market.outcomes) > 1 else None
            ),
            confidence=confidence,
            expected_edge=impact,
            reasoning=(
                f"NEWS: \"{headline.title}\" → {direction} for {slug} | "
                f"Impact: {impact}¢ | {classification.get('reasoning', '')}"
            ),
            suggested_size_usd=min(
                self.config.risk.max_position_usd * confidence,
                self.config.risk.max_position_usd,
            ),
        )

    # ------------------------------------------------------------------
    # Rate Limiting
    # ------------------------------------------------------------------

    def _within_rate_limit(self) -> bool:
        now = datetime.now(timezone.utc)
        if (now - self._hour_start).total_seconds() > 3600:
            self._api_calls_this_hour = 0
            self._hour_start = now
        return self._api_calls_this_hour < self.cfg.max_calls_per_hour
