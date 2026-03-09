from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from config import PipelineConfig
from data.models import Market
from strategies.crypto_arb import (
    CryptoTemporalArbStrategy,
    MAX_SIGNAL_CONFIDENCE,
    MIN_THRESHOLD_EDGE,
    TEMPORAL_CONFIDENCE_SCALE,
)
from strategies.news import NewsLatencyStrategy
from strategies.weather import WEATHER_FEE_BUFFER, WeatherForecastStrategy

from .config import LLMConfig
from .contracts import EnrichmentResult, MarketContext, NormalizedMarket, utc_now
from .llm import MultiagentLLMRouter


@dataclass(frozen=True)
class ProviderCard:
    name: str
    status: str
    detail: str
    updated_at: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


class ProviderState:
    def __init__(self, name: str) -> None:
        self.name = name
        self.results: dict[str, EnrichmentResult] = {}
        self.last_error: str | None = None
        self.last_refresh: datetime | None = None
        self.last_count: int = 0
        self.last_stats: dict[str, Any] = {}
        self.recent_items: list[dict[str, Any]] = []

    def set_results(
        self,
        results: dict[str, EnrichmentResult],
        *,
        stats: dict[str, Any] | None = None,
        recent_items: list[dict[str, Any]] | None = None,
    ) -> None:
        self.results = results
        self.last_count = len(results)
        self.last_refresh = utc_now()
        self.last_error = None
        self.last_stats = dict(stats or {})
        self.last_stats["enriched_markets"] = len(results)
        self.recent_items = list((recent_items or [])[-40:])

    def set_error(
        self,
        error: Exception | str,
        *,
        stats: dict[str, Any] | None = None,
        recent_items: list[dict[str, Any]] | None = None,
    ) -> None:
        self.last_error = str(error)
        self.last_refresh = utc_now()
        self.last_stats = dict(stats or {})
        self.recent_items = list((recent_items or [])[-40:])

    def provider_card(self) -> ProviderCard:
        if self.last_error:
            status = "failed"
            detail = self._detail_text() or self.last_error
        elif self.last_count > 0:
            status = "healthy"
            detail = self._detail_text() or f"{self.last_count} enriched markets available"
        else:
            status = "degraded"
            detail = self._detail_text() or "No enrichments produced in the last refresh"
        return ProviderCard(
            name=self.name,
            status=status,
            detail=detail,
            updated_at=self.last_refresh.isoformat() if self.last_refresh else None,
            metrics=dict(self.last_stats),
        )

    def recent_activity(self, limit: int = 20) -> list[dict[str, Any]]:
        return list(self.recent_items[-limit:])

    def _detail_text(self) -> str:
        if self.name != "news":
            if not self.last_stats:
                return ""
            parts: list[str] = []
            if self.last_stats.get("enriched_markets") is not None:
                parts.append(f"{int(self.last_stats.get('enriched_markets', 0) or 0)} enriched markets available")
            return " | ".join(parts)

        parts: list[str] = []
        fetched = int(self.last_stats.get("fetched_headlines", 0) or 0)
        matched = int(self.last_stats.get("matched_headlines", 0) or 0)
        llm_attempted = int(self.last_stats.get("llm_attempted", 0) or 0)
        llm_succeeded = int(self.last_stats.get("llm_succeeded", 0) or 0)
        enriched = int(self.last_stats.get("enriched_markets", 0) or 0)
        if fetched:
            parts.append(f"{fetched} fetched")
        if matched or fetched:
            parts.append(f"{matched} matched")
        if llm_attempted or matched:
            parts.append(f"{llm_attempted} LLM attempts")
        if llm_succeeded or llm_attempted:
            parts.append(f"{llm_succeeded} LLM parsed")
        parts.append(f"{enriched} enrichments")
        llm_failed = int(self.last_stats.get("llm_failed", 0) or 0)
        if llm_failed:
            parts.append(f"{llm_failed} llm_failed")
        no_match = int(self.last_stats.get("prefilter_no_match", 0) or 0)
        if no_match:
            parts.append(f"{no_match} no_match")
        return " | ".join(parts)


class WeatherEnrichmentProvider:
    name = "weather"

    def __init__(self, pipeline_config: PipelineConfig) -> None:
        self.helper = WeatherForecastStrategy(pipeline_config, collector=None, state_path=None)
        self.state = ProviderState(self.name)

    async def refresh(self, raw_markets: list[Market]) -> None:
        now_ts = time.time()
        try:
            if now_ts - self.helper._last_forecast_fetch > self.helper.cfg.forecast_refresh_secs:
                await self.helper._fetch_forecasts()
                self.helper._last_forecast_fetch = now_ts
            self.helper._match_weather_markets(raw_markets)
            if not self.helper._matched_markets:
                supplemental = await self.helper.get_supplemental_markets()
                if supplemental:
                    self.helper._match_weather_markets(supplemental)
            self.helper._record_market_prices()
            results: dict[str, EnrichmentResult] = {}
            now = utc_now()

            for match in self.helper._matched_markets:
                market = match["market"]
                if len(market.outcomes) < 2:
                    continue
                horizon_days = self.helper._forecast_horizon_days(match["target_date"])
                if horizon_days is None:
                    continue
                context = self.helper._build_forecast_context(
                    city=match["city"],
                    target_date=match["target_date"],
                    temp_range=match["temp_range"],
                    range_kind=match["range_kind"],
                    horizon_days=horizon_days,
                )
                if not context:
                    continue

                yes_series = self.helper._price_series(
                    market.condition_id,
                    "YES",
                    self.helper.cfg.swing_lookback_minutes,
                )
                no_series = self.helper._price_series(
                    market.condition_id,
                    "NO",
                    self.helper.cfg.swing_lookback_minutes,
                )

                results[market.condition_id] = EnrichmentResult(
                    provider_name=self.name,
                    data={
                        "city": match["city"],
                        "target_date": match["target_date"],
                        "range_kind": match["range_kind"],
                        "temp_range": match["temp_range"],
                        "current_prob": context["current_prob"],
                        "previous_prob": context["previous_prob"],
                        "current_spread_f": context["current_spread_f"],
                        "changed_models": context["changed_models"],
                        "current_consensus": context["current_consensus"],
                        "previous_consensus": context["previous_consensus"],
                        "current_temps": context["current_temps"],
                        "previous_temps": context["previous_temps"],
                        "yes_price": market.outcomes[0].price,
                        "no_price": market.outcomes[1].price,
                        "yes_history_points": len(yes_series),
                        "no_history_points": len(no_series),
                        "recent_yes_peak": max(yes_series) if yes_series else None,
                        "recent_no_peak": max(no_series) if no_series else None,
                        "fee_buffer": WEATHER_FEE_BUFFER,
                    },
                    fetched_at=now,
                    staleness_seconds=0.0,
                )
            self.state.set_results(results)
        except Exception as exc:
            self.state.set_error(exc)

    def result_for(self, market_id: str) -> EnrichmentResult | None:
        return self.state.results.get(market_id)

    def supplemental_markets(self) -> list[Market]:
        return list(getattr(self.helper, "_supplemental_markets", []) or [])

    async def close(self) -> None:
        await self.helper.client.aclose()


class CryptoEnrichmentProvider:
    name = "crypto"

    def __init__(self, pipeline_config: PipelineConfig) -> None:
        self.helper = CryptoTemporalArbStrategy(pipeline_config)
        self.state = ProviderState(self.name)

    async def refresh(self, raw_markets: list[Market]) -> None:
        try:
            prices = await self.helper._fetch_spot_prices()
            if not prices:
                self.state.set_results({})
                return

            now_ts = time.time()
            for symbol, price in prices.items():
                self.helper._price_history[symbol].append({"price": price, "time": now_ts})
                cutoff = now_ts - 1200
                self.helper._price_history[symbol] = [
                    item for item in self.helper._price_history[symbol] if item["time"] > cutoff
                ]

            if now_ts - self.helper._last_market_scan > 300 or not self.helper._matched_markets:
                self.helper._match_crypto_markets(raw_markets)
                self.helper._last_market_scan = now_ts

            results: dict[str, EnrichmentResult] = {}
            now = utc_now()
            for symbol, price in prices.items():
                move = self.helper._calculate_move(symbol)
                move_pct = move[0] if move else 0.0
                move_direction = move[1] if move else "flat"
                for match in self.helper._matched_markets.get(symbol, []):
                    market = match["market"]
                    payload: dict[str, Any] = {
                        "symbol": symbol,
                        "spot_price": price,
                        "move_pct": move_pct,
                        "move_direction": move_direction,
                        "kind": match["kind"],
                    }
                    if match["kind"] == "temporal":
                        payload.update(
                            {
                                "up_index": match["up_index"],
                                "down_index": match["down_index"],
                                "up_price": market.outcomes[match["up_index"]].price,
                                "down_price": market.outcomes[match["down_index"]].price,
                            }
                        )
                    else:
                        yes_price = market.outcomes[match["yes_index"]].price
                        no_price = market.outcomes[match["no_index"]].price
                        payload.update(
                            {
                                "barrier_price": match["barrier_price"],
                                "years_left": match["years_left"],
                                "yes_index": match["yes_index"],
                                "no_index": match["no_index"],
                                "yes_price": yes_price,
                                "no_price": no_price,
                                "modeled_yes": self.helper._estimate_barrier_probability(
                                    symbol=symbol,
                                    spot_price=price,
                                    barrier_price=match["barrier_price"],
                                    years_left=match["years_left"],
                                ),
                                "min_threshold_edge": MIN_THRESHOLD_EDGE,
                            }
                        )

                    results[market.condition_id] = EnrichmentResult(
                        provider_name=self.name,
                        data=payload,
                        fetched_at=now,
                        staleness_seconds=0.0,
                    )
            self.state.set_results(results)
        except Exception as exc:
            self.state.set_error(exc)

    def result_for(self, market_id: str) -> EnrichmentResult | None:
        return self.state.results.get(market_id)

    async def close(self) -> None:
        await self.helper.client.aclose()


class NewsEnrichmentProvider:
    name = "news"

    def __init__(self, pipeline_config: PipelineConfig, llm_router: MultiagentLLMRouter) -> None:
        self.helper = NewsLatencyStrategy(pipeline_config)
        self.llm_router = llm_router
        self.state = ProviderState(self.name)

    async def refresh(self, raw_markets: list[Market]) -> None:
        refresh_stats: dict[str, Any] = {
            "fetched_headlines": 0,
            "ready_headlines": 0,
            "matched_headlines": 0,
            "prefilter_no_match": 0,
            "llm_attempted": 0,
            "llm_succeeded": 0,
            "llm_failed": 0,
            "llm_no_market": 0,
            "llm_bad_market_slug": 0,
            "enriched_markets": 0,
        }
        activity_rows: list[dict[str, Any]] = []
        try:
            self.helper._build_market_index(raw_markets)
            headlines = await self.helper._fetch_headlines()
            refresh_stats["fetched_headlines"] = len(headlines)
            for headline in headlines:
                self.helper._register_headline(headline)

            ready = self.helper._ready_headlines()[:15]
            refresh_stats["ready_headlines"] = len(ready)
            if not ready:
                self.state.set_results({}, stats=refresh_stats, recent_items=activity_rows)
                return

            results: dict[str, EnrichmentResult] = {}
            for headline in ready:
                headline_id = self.helper._headline_id(headline)
                candidate_markets = self.helper._candidate_markets_for_headline(headline, limit=10)
                activity_base = {
                    "headline": headline.title,
                    "source": headline.source,
                    "fetched_at": utc_now().isoformat(),
                    "matched_candidate_count": len(candidate_markets),
                    "llm_assisted": False,
                    "llm_provider": None,
                    "llm_model": None,
                    "llm_error": None,
                    "confidence": 0.0,
                    "expected_impact_cents": 0.0,
                    "impact_cents": 0.0,
                    "direction": "neutral",
                    "market_slug": None,
                    "reasoning": "",
                    "status": "prefilter_pending",
                    "stage": "prefilter",
                    "llm_attempts": [],
                }
                if not candidate_markets:
                    refresh_stats["prefilter_no_match"] += 1
                    activity_rows.append(
                        {
                            **activity_base,
                            "status": "no_candidate_matches",
                            "stage": "prefilter",
                        }
                    )
                    self.helper._mark_headline_processed(headline_id)
                    continue
                refresh_stats["matched_headlines"] += 1

                refresh_stats["llm_attempted"] += 1
                llm_result, attempts = await self.llm_router.complete_json(
                    task_name="news_relevance",
                    system_prompt=(
                        "You are labeling prediction-market news relevance. "
                        "You must choose exactly one market_slug from the provided candidate list or return an empty string if none are relevant. "
                        "Classify direction as bullish, bearish, or neutral. "
                        "Set confidence from 0.0 to 1.0. "
                        "Estimate a plausible expected market move in cents from 0 to 25. "
                        "Only assign confidence above 0.35 when the headline is directly relevant to the chosen market. "
                        "Keep reasoning short and concrete. "
                        "Return exactly one JSON object with keys: market_slug, direction, confidence, expected_impact_cents, reasoning."
                    ),
                    user_payload={
                        "headline": headline.title,
                        "source": headline.source,
                        "candidates": [
                            {
                                "market_slug": market.slug,
                                "question": market.question,
                                "description": getattr(market, "description", "")[:180],
                                "tags": list(getattr(market, "tags", [])[:6]),
                                "yes_price": market.outcomes[0].price if market.outcomes else None,
                                "no_price": market.outcomes[1].price if len(market.outcomes) > 1 else None,
                            }
                            for market in candidate_markets[:6]
                        ],
                    },
                    required_keys=("market_slug", "direction", "confidence", "expected_impact_cents", "reasoning"),
                )

                attempt_rows = [
                    {
                        "provider": attempt.provider,
                        "model": attempt.model,
                        "role": attempt.role,
                        "success": attempt.success,
                        "parsed": attempt.parsed,
                        "latency_ms": attempt.latency_ms,
                        "error": attempt.error,
                    }
                    for attempt in attempts
                ]
                failure_attempt = llm_result if llm_result is not None and llm_result.parsed_json is None else None
                if failure_attempt is not None:
                    attempt_rows.insert(
                        0,
                        {
                            "provider": failure_attempt.provider,
                            "model": failure_attempt.model,
                            "role": failure_attempt.role,
                            "success": failure_attempt.success,
                            "parsed": failure_attempt.parsed,
                            "latency_ms": failure_attempt.latency_ms,
                            "error": failure_attempt.error,
                        },
                    )

                if llm_result is None or llm_result.parsed_json is None:
                    refresh_stats["llm_failed"] += 1
                    activity_rows.append(
                        {
                            **activity_base,
                            "status": "llm_failed",
                            "stage": "llm",
                            "llm_assisted": False,
                            "llm_provider": getattr(failure_attempt, "provider", None),
                            "llm_model": getattr(failure_attempt, "model", None),
                            "llm_error": _summarize_attempt_errors(attempt_rows),
                            "llm_attempts": attempt_rows,
                        }
                    )
                    self.helper._mark_headline_processed(headline_id)
                    continue

                parsed = llm_result.parsed_json
                refresh_stats["llm_succeeded"] += 1
                market_slug = str(parsed.get("market_slug", "") or "").strip()
                if not market_slug:
                    refresh_stats["llm_no_market"] += 1
                    activity_rows.append(
                        {
                            **activity_base,
                            "status": "llm_no_market",
                            "stage": "llm",
                            "llm_assisted": True,
                            "llm_provider": llm_result.provider,
                            "llm_model": llm_result.model,
                            "llm_error": None,
                            "llm_attempts": attempt_rows,
                            "reasoning": str(parsed.get("reasoning", "") or ""),
                        }
                    )
                    self.helper._mark_headline_processed(headline_id)
                    continue
                chosen_market = self.helper._market_lookup.get(market_slug) if market_slug else None
                if chosen_market is None:
                    refresh_stats["llm_bad_market_slug"] += 1
                    activity_rows.append(
                        {
                            **activity_base,
                            "status": "llm_bad_market_slug",
                            "stage": "llm",
                            "llm_assisted": True,
                            "llm_provider": llm_result.provider,
                            "llm_model": llm_result.model,
                            "llm_error": f"unknown_market_slug:{market_slug}",
                            "llm_attempts": attempt_rows,
                            "reasoning": str(parsed.get("reasoning", "") or ""),
                            "market_slug": market_slug,
                        }
                    )
                    self.helper._mark_headline_processed(headline_id)
                    continue

                existing = results.get(chosen_market.condition_id)
                payload = {
                    "headline": headline.title,
                    "source": headline.source,
                    "market_slug": chosen_market.slug,
                    "direction": parsed.get("direction"),
                    "confidence": float(parsed.get("confidence", 0.0) or 0.0),
                    "expected_impact_cents": float(parsed.get("expected_impact_cents", 0.0) or 0.0),
                    "reasoning": parsed.get("reasoning", ""),
                    "llm_provider": llm_result.provider,
                    "llm_model": llm_result.model,
                    "llm_attempts": attempt_rows,
                    "llm_error": None,
                    "candidate_market_slugs": [market.slug for market in candidate_markets[:6]],
                }
                if existing is None or payload["confidence"] > float(existing.data.get("confidence", 0.0) or 0.0):
                    results[chosen_market.condition_id] = EnrichmentResult(
                        provider_name=self.name,
                        data=payload,
                        llm_assisted=True,
                        llm_model=llm_result.model,
                        llm_prompt_hash=llm_result.prompt_hash,
                        fetched_at=utc_now(),
                        staleness_seconds=0.0,
                    )
                activity_rows.append(
                    {
                        **activity_base,
                        "status": "enriched",
                        "stage": "enriched",
                        "llm_assisted": True,
                        "llm_provider": llm_result.provider,
                        "llm_model": llm_result.model,
                        "llm_error": None,
                        "llm_attempts": attempt_rows,
                        "confidence": payload["confidence"],
                        "expected_impact_cents": payload["expected_impact_cents"],
                        "impact_cents": payload["expected_impact_cents"],
                        "direction": payload["direction"],
                        "market_slug": payload["market_slug"],
                        "reasoning": payload["reasoning"],
                    }
                )
                self.helper._mark_headline_processed(headline_id)
            refresh_stats["enriched_markets"] = len(results)
            self.state.set_results(results, stats=refresh_stats, recent_items=activity_rows)
        except Exception as exc:
            self.state.set_error(exc, stats=refresh_stats, recent_items=activity_rows)

    def result_for(self, market_id: str) -> EnrichmentResult | None:
        return self.state.results.get(market_id)

    async def close(self) -> None:
        await self.helper.client.aclose()


class MultiProviderEnricher:
    def __init__(self, providers: list[Any]) -> None:
        self.providers = providers

    async def refresh(self, raw_markets: list[Market]) -> None:
        for provider in self.providers:
            await provider.refresh(raw_markets)

    def enrich(self, markets: list[NormalizedMarket]) -> list[MarketContext]:
        enriched: list[MarketContext] = []
        for market in markets:
            enrichments: dict[str, EnrichmentResult] = {}
            for provider in self.providers:
                result = provider.result_for(market.market_id)
                if result is not None:
                    enrichments[provider.name] = _with_staleness(result)
            completeness = len(enrichments) / max(len(self.providers), 1)
            enriched.append(
                MarketContext(
                    market_id=market.market_id,
                    question=market.question,
                    category=market.category,
                    outcomes=market.outcomes,
                    volume_24h=market.volume_24h,
                    total_volume=market.total_volume,
                    liquidity=market.liquidity,
                    created_date=market.created_date,
                    source_url=market.source_url,
                    resolution_date=market.resolution_date,
                    description=market.description,
                    tags=market.tags,
                    enrichments=enrichments,
                    enrichment_completeness=completeness,
                )
            )
        return enriched

    def provider_cards(self) -> list[ProviderCard]:
        return [provider.state.provider_card() for provider in self.providers]

    def supplemental_markets(self) -> list[Market]:
        markets: list[Market] = []
        seen: set[str] = set()
        for provider in self.providers:
            getter = getattr(provider, "supplemental_markets", None)
            if getter is None:
                continue
            try:
                extra_markets = getter() or []
            except Exception:
                continue
            for market in extra_markets:
                market_id = getattr(market, "condition_id", None)
                if market_id and market_id not in seen:
                    seen.add(market_id)
                    markets.append(market)
        return markets

    async def close(self) -> None:
        for provider in self.providers:
            close = getattr(provider, "close", None)
            if close is not None:
                await close()


def _with_staleness(result: EnrichmentResult) -> EnrichmentResult:
    age = (utc_now() - result.fetched_at).total_seconds()
    return EnrichmentResult(
        provider_name=result.provider_name,
        data=result.data,
        llm_assisted=result.llm_assisted,
        llm_model=result.llm_model,
        llm_prompt_hash=result.llm_prompt_hash,
        fetched_at=result.fetched_at,
        is_stale=age > 0,
        staleness_seconds=max(age, 0.0),
        error=result.error,
    )


def _summarize_attempt_errors(attempt_rows: list[dict[str, Any]]) -> str:
    errors = [
        f"{row.get('provider')}:{row.get('error')}"
        for row in attempt_rows
        if row.get("error")
    ]
    if not errors:
        return "llm_failed_without_error"
    return " | ".join(errors[:3])
