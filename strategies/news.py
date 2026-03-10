"""
Strategy: News-to-Price Latency (Optional LLM Layer)
====================================================
Ingests breaking news, classifies relevance + direction via an LLM,
and generates signals before the market fully prices in the information.

Uses cheap-first routing for background scans:
- primary: Fireworks / GLM-5
- optional fallback: Anthropic / Sonnet
"""

import hashlib
import json
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta

import httpx
from data.models import Market, Event, Signal, SignalSource, SignalAction, NewsHeadline
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
FIREWORKS_API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
ANTHROPIC_BILLING_RETRY = timedelta(minutes=5)
ANTHROPIC_TRANSIENT_RETRY = timedelta(minutes=2)
MAX_PENDING_HEADLINES = 200
LEGACY_NEWS_MAX_TOKENS = 300


class NewsLatencyStrategy(BaseStrategy):
    name = "news_latency"
    description = "Classify breaking news with LLM and trade before market reacts"

    def __init__(self, config):
        super().__init__(config)
        self.cfg = config.news
        self.client = httpx.AsyncClient(timeout=15.0)
        self._processed_headlines: dict[str, None] = {}
        self._pending_headlines: dict[str, dict] = {}
        self._recent_headlines: list[NewsHeadline] = []
        self._api_calls_this_hour: int = 0
        self._hour_start: datetime = datetime.now(timezone.utc)
        self._anthropic_pause_until: datetime | None = None
        self._anthropic_last_error: str = ""
        self._market_index: dict[str, set[str]] = {}  # keyword -> market_slugs
        self._market_lookup: dict[str, Market] = {}
        self._indexed_keywords: tuple[str, ...] = ()
        self._stats.update({
            "matched_headlines": 0,
            "classified_headlines": 0,
            "last_context_markets": 0,
            "pending_headlines": 0,
            "anthropic_pause_until": "",
            "llm_primary_provider": self.cfg.primary_provider,
        })

    async def scan(self, markets: list[Market], events: list[Event]) -> list[Signal]:
        if not self.cfg.enabled or not self._available_llm_providers():
            return []

        self._stats["scans_completed"] += 1

        # Build keyword index of active markets
        self._build_market_index(markets)

        # Fetch new headlines
        headlines = await self._fetch_headlines()
        for headline in headlines:
            self._register_headline(headline)

        if self._anthropic_pause_until and datetime.now(timezone.utc) >= self._anthropic_pause_until:
            self._anthropic_pause_until = None
            self._anthropic_last_error = ""

        ready_headlines = self._ready_headlines()
        self._stats["pending_headlines"] = len(self._pending_headlines)
        self._stats["anthropic_pause_until"] = (
            self._anthropic_pause_until.isoformat() if self._anthropic_pause_until else ""
        )

        if not ready_headlines:
            return []

        if self._anthropic_pause_until and datetime.now(timezone.utc) < self._anthropic_pause_until:
            logger.warning(
                "[NEWS] Anthropic paused until %s | pending=%s | last_error=%s",
                self._anthropic_pause_until.isoformat(),
                len(self._pending_headlines),
                self._anthropic_last_error or "temporary provider issue",
            )
            return []

        # Pre-filter and cache candidate markets so classification stays focused.
        headline_context = {
            headline.title: self._candidate_markets_for_headline(headline)
            for headline in ready_headlines
        }
        relevant_headlines = [
            headline for headline in ready_headlines
            if headline_context.get(headline.title)
        ]
        relevant_count = len(relevant_headlines)
        if not relevant_headlines:
            return []

        # Keep background spend bounded; manual consults are where richer models belong.
        relevant_headlines = relevant_headlines[: self.cfg.max_headlines_per_scan]

        signals = []
        classified_count = 0
        max_context_markets = 0
        for headline in relevant_headlines:
            headline_id = self._headline_id(headline)
            if not self._within_rate_limit():
                logger.warning("[NEWS] Rate limit reached, skipping remaining headlines")
                break

            candidate_markets = headline_context.get(headline.title) or self._fallback_market_context(markets)
            max_context_markets = max(max_context_markets, len(candidate_markets))
            classification = await self._classify_headline(headline, candidate_markets)
            if self._anthropic_pause_until and datetime.now(timezone.utc) < self._anthropic_pause_until:
                break
            if classification:
                self._mark_headline_processed(headline_id)
                classified_count += 1
            if classification and classification.get("confidence", 0) >= self.cfg.min_confidence:
                signal = self._build_signal(headline, classification)
                if signal:
                    signals.append(signal)
                    self._stats["signals_generated"] += 1

        self._stats["matched_headlines"] = relevant_count
        self._stats["classified_headlines"] = classified_count
        self._stats["last_context_markets"] = max_context_markets

        logger.info(
            "[NEWS] Scan: %s new headlines | %s relevant | %s classified | %s signals | context<=%s markets",
            len(ready_headlines),
            relevant_count,
            classified_count,
            len(signals),
            max_context_markets,
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

    def _headline_id(self, headline: NewsHeadline) -> str:
        key = f"{headline.source}|{headline.title}"
        return hashlib.md5(key.encode()).hexdigest()

    def _register_headline(self, headline: NewsHeadline):
        headline_id = self._headline_id(headline)
        if headline_id in self._processed_headlines or headline_id in self._pending_headlines:
            return

        self._pending_headlines[headline_id] = {
            "headline": headline,
            "retry_after": datetime.now(timezone.utc),
            "attempts": 0,
        }
        self._recent_headlines.append(headline)
        if len(self._recent_headlines) > 200:
            self._recent_headlines = self._recent_headlines[-100:]
        while len(self._pending_headlines) > MAX_PENDING_HEADLINES:
            oldest_id = next(iter(self._pending_headlines))
            self._pending_headlines.pop(oldest_id, None)

    def _ready_headlines(self) -> list[NewsHeadline]:
        now = datetime.now(timezone.utc)
        ready = []
        for item in self._pending_headlines.values():
            retry_after = item.get("retry_after") or now
            if retry_after <= now:
                ready.append(item["headline"])
        return ready

    def _mark_headline_processed(self, headline_id: str):
        self._pending_headlines.pop(headline_id, None)
        self._processed_headlines[headline_id] = None
        while len(self._processed_headlines) > 5000:
            oldest_id = next(iter(self._processed_headlines))
            self._processed_headlines.pop(oldest_id, None)

    def _defer_headline(self, headline: NewsHeadline, delay: timedelta):
        headline_id = self._headline_id(headline)
        entry = self._pending_headlines.get(headline_id)
        if not entry:
            self._register_headline(headline)
            entry = self._pending_headlines.get(headline_id)
        if not entry:
            return
        entry["retry_after"] = datetime.now(timezone.utc) + delay
        entry["attempts"] = int(entry.get("attempts", 0)) + 1

    def _pause_anthropic(self, delay: timedelta, message: str):
        pause_until = datetime.now(timezone.utc) + delay
        if not self._anthropic_pause_until or pause_until > self._anthropic_pause_until:
            self._anthropic_pause_until = pause_until
        self._anthropic_last_error = message[:200]
        self._stats["anthropic_pause_until"] = self._anthropic_pause_until.isoformat()

    def _anthropic_retry_delay(self, exc: Exception) -> timedelta:
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            body = exc.response.text.lower()
            if "credit balance is too low" in body or "billing" in body or "upgrade or purchase credits" in body:
                return ANTHROPIC_BILLING_RETRY
            if status in {408, 409, 429, 500, 502, 503, 504, 529}:
                return ANTHROPIC_TRANSIENT_RETRY
        if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
            return ANTHROPIC_TRANSIENT_RETRY
        return timedelta()

    # ------------------------------------------------------------------
    # Keyword Pre-Filter (saves API calls)
    # ------------------------------------------------------------------

    def _build_market_index(self, markets: list[Market]):
        """Build a keyword -> market mapping for fast filtering."""
        self._market_index.clear()
        self._market_lookup.clear()
        for m in markets:
            if m.closed or not m.active or not m.outcomes:
                continue
            self._market_lookup[m.slug] = m
            keywords = self._tokenize(f"{m.question} {m.slug} {' '.join(m.tags)}")
            for kw in keywords:
                self._market_index.setdefault(kw, set()).add(m.slug)
        self._indexed_keywords = tuple(self._market_index.keys())

    def _candidate_markets_for_headline(
        self, headline: NewsHeadline, limit: int = 15
    ) -> list[Market]:
        """Rank candidate markets by keyword overlap with the headline."""
        tokens = self._tokenize(headline.title)
        if not tokens:
            return []

        scores: dict[str, int] = {}
        for token in tokens:
            for slug in self._market_index.get(token, set()):
                scores[slug] = scores.get(slug, 0) + 4

        # Allow loose prefix matches so "iran" can still find "iranian" markets.
        fuzzy_tokens = [token for token in tokens if len(token) >= 4]
        if fuzzy_tokens:
            for indexed in self._indexed_keywords:
                if len(indexed) < 4:
                    continue
                if not any(
                    token.startswith(indexed)
                    or indexed.startswith(token)
                    for token in fuzzy_tokens
                ):
                    continue
                for slug in self._market_index.get(indexed, set()):
                    scores[slug] = scores.get(slug, 0) + 1

        ranked = sorted(
            (
                self._market_lookup[slug]
                for slug in scores
                if slug in self._market_lookup
            ),
            key=lambda market: (
                scores.get(market.slug, 0),
                market.volume_24h,
                market.liquidity,
            ),
            reverse=True,
        )
        return ranked[:limit]

    def _fallback_market_context(self, markets: list[Market], limit: int = 15) -> list[Market]:
        """Fallback to the most active markets when a headline has no direct keyword match."""
        active = [
            market for market in markets
            if market.active and not market.closed and market.outcomes
        ]
        return sorted(
            active,
            key=lambda market: (market.volume_24h, market.liquidity),
            reverse=True,
        )[:limit]

    def _tokenize(self, text: str) -> set[str]:
        tokens = set()
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            if len(token) < 4:
                continue
            tokens.add(token)
            if token.endswith("s") and len(token) > 4:
                tokens.add(token[:-1])
        return tokens

    # ------------------------------------------------------------------
    # LLM Classification
    # ------------------------------------------------------------------

    async def _classify_headline(
        self, headline: NewsHeadline, markets: list[Market]
    ) -> dict | None:
        """Use cheap-first LLM routing to classify a headline against active markets."""
        market_list = "\n".join(
            f"- {m.slug}: \"{m.question}\" (current YES: {m.outcomes[0].price:.2f})"
            for m in markets[:15]
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

        last_error: Exception | None = None
        for provider in self._available_llm_providers():
            try:
                classification = await self._call_llm(provider, prompt)
                self._api_calls_this_hour += 1
                headline.classification = classification
                logger.debug(
                    "[NEWS] Classified via %s: '%s' → market=%s dir=%s conf=%s",
                    provider,
                    headline.title[:60],
                    classification.get("market_slug", "none"),
                    classification.get("direction", "?"),
                    classification.get("confidence", 0),
                )
                return classification
            except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
                self._stats["errors"] += 1
                delay = self._anthropic_retry_delay(exc)
                if provider == "anthropic" and delay > timedelta():
                    self._defer_headline(headline, delay)
                    message = exc.response.text if isinstance(exc, httpx.HTTPStatusError) else str(exc)
                    self._pause_anthropic(delay, message)
                    logger.warning(
                        "[NEWS] Anthropic unavailable, retrying after %ss | pending=%s | %s",
                        int(delay.total_seconds()),
                        len(self._pending_headlines),
                        message[:180],
                    )
                    return None
                logger.warning("[NEWS] %s classification failed: %s", provider, exc)
                continue
            except Exception as exc:
                last_error = exc
                self._stats["errors"] += 1
                logger.warning("[NEWS] %s classification failed: %s", provider, exc)
                continue

        self._defer_headline(headline, ANTHROPIC_TRANSIENT_RETRY)
        if last_error is not None:
            logger.error("[NEWS] All LLM providers failed: %s", last_error)
        return None

    def _available_llm_providers(self) -> list[str]:
        providers: list[str] = []
        primary = (self.cfg.primary_provider or "").strip().lower()
        fallback = (self.cfg.fallback_provider or "").strip().lower()
        if primary == "fireworks" and self.cfg.fireworks_api_key:
            providers.append("fireworks")
        elif primary == "anthropic" and self.cfg.anthropic_api_key:
            providers.append("anthropic")
        if fallback == "fireworks" and self.cfg.fireworks_api_key and "fireworks" not in providers:
            providers.append("fireworks")
        elif fallback == "anthropic" and self.cfg.anthropic_api_key and "anthropic" not in providers:
            providers.append("anthropic")
        if not providers:
            if self.cfg.fireworks_api_key:
                providers.append("fireworks")
            if self.cfg.anthropic_api_key:
                providers.append("anthropic")
        return providers

    async def _call_llm(self, provider: str, prompt: str) -> dict:
        if provider == "fireworks":
            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "authorization": f"Bearer {self.cfg.fireworks_api_key}",
            }
            request_payload = {
                "model": self.cfg.fireworks_model,
                "max_tokens": LEGACY_NEWS_MAX_TOKENS,
                "temperature": 0.1,
                "reasoning_effort": "low",
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": "Return only one JSON object matching the requested schema.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            }
            response = await self.client.post(
                FIREWORKS_API_URL,
                headers=headers,
                json=request_payload,
            )
            # Some Fireworks model routes reject advanced request knobs with HTTP 412.
            if response.status_code == 412:
                relaxed_payload = dict(request_payload)
                relaxed_payload.pop("reasoning_effort", None)
                relaxed_payload.pop("response_format", None)
                response = await self.client.post(
                    FIREWORKS_API_URL,
                    headers=headers,
                    json=relaxed_payload,
                )
            response.raise_for_status()
            payload = self._decode_json_response(response, "fireworks")
            message = (payload.get("choices") or [{}])[0].get("message", {}) or {}
            text = self._extract_chat_content(message.get("content")).strip()
            if not text:
                text = self._extract_chat_content(message.get("reasoning_content")).strip()
            if not text:
                text = self._extract_chat_content((payload.get("choices") or [{}])[0].get("text")).strip()
            if not text:
                raise RuntimeError("fireworks_empty_response")
            return self._parse_classification_json(text)

        if provider == "anthropic":
            response = await self.client.post(
                ANTHROPIC_API_URL,
                headers={
                    "x-api-key": self.cfg.anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.cfg.model,
                    "max_tokens": LEGACY_NEWS_MAX_TOKENS,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            response.raise_for_status()
            payload = response.json()
            text = payload["content"][0]["text"].strip()
            return self._parse_classification_json(text)

        raise RuntimeError(f"unsupported_provider:{provider}")

    def _parse_classification_json(self, text: str) -> dict:
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise
            return json.loads(text[start:end + 1])

    @staticmethod
    def _decode_json_response(response: httpx.Response, provider: str) -> dict:
        try:
            payload = response.json()
        except Exception as exc:
            snippet = (response.text or "").strip().replace("\n", " ")
            if len(snippet) > 240:
                snippet = f"{snippet[:240]}..."
            raise RuntimeError(
                f"{provider}_non_json_response status={response.status_code} body={snippet or '<empty>'}"
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"{provider}_invalid_json_payload")
        return payload

    @staticmethod
    def _extract_chat_content(content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts).strip()
        if isinstance(content, dict):
            text = content.get("text") or content.get("content")
            if isinstance(text, str):
                return text
        return ""

    # ------------------------------------------------------------------
    # Signal Building
    # ------------------------------------------------------------------

    def _build_signal(self, headline: NewsHeadline, classification: dict) -> Signal | None:
        """Convert a classification into a tradeable signal."""
        slug = classification.get("market_slug")
        if not slug:
            return None

        # Find the market
        market = self._market_lookup.get(slug)
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
