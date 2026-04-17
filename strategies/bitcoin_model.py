"""
Strategy: BTC Futures ML Catch-Up Sleeve
========================================
Comparison-book only sleeve that uses the frozen BTC futures impulse bundle as
the fast-market timing layer, then maps that direction into Polymarket BTC
barrier contracts that still look underreacted.

This sleeve is isolated from the legacy portfolio and other sleeves. It has its
own budget, state, and logs.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data.models import Event, Market, Signal, SignalAction, SignalSource
from engine.binance_btc_feature_feed import BinanceBtcFeatureFeed, FeedSnapshot
from engine.bitcoin_context_feed import BitcoinContextFeed, ContextSnapshot
from engine.polymarket_btc_market_feed import PolymarketBtcMarketFeed
from runtime_paths import LOG_DIR
from strategies.base import BaseStrategy
from strategies.crypto_arb import CryptoTemporalArbStrategy

logger = logging.getLogger(__name__)

try:
    from catboost import CatBoostClassifier, Pool
except Exception:  # pragma: no cover - runtime dependency gate
    CatBoostClassifier = None
    Pool = None


BASE_SIGNAL_CONFIDENCE = 0.58
MAX_SIGNAL_CONFIDENCE = 0.96
EDGE_CONFIDENCE_BONUS_CAP = 0.18
EDGE_CONFIDENCE_MULTIPLIER = 0.75
EDGE_SIZE_USD_MULTIPLIER = 220.0
SCORE_SIZE_USD_MULTIPLIER = 180.0
SCORED_MARKET_LIMIT = 16
DEGRADED_SCORE_ONLY_THRESHOLD = 0.60
DEGRADED_SCORE_ONLY_MARGIN = 0.01
DEGRADED_BULL_FALLBACK_THRESHOLD = 0.52
MARKET_FALLBACK_SCORE_THRESHOLD = 0.36
MARKET_FALLBACK_MAX_SCORE_GAP = 0.10
MARKET_INVENTORY_FALLBACK_SCORE_THRESHOLD = 0.36


class BitcoinModelBundle:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.metadata: dict[str, Any] = {}
        self.models: dict[str, CatBoostClassifier] = {}
        self.feature_names: dict[str, list[str]] = {}
        self.ready = False
        self.version = "unloaded"
        self.load_error: str | None = None
        self._load()

    def _load(self) -> None:
        if CatBoostClassifier is None or Pool is None:
            self.load_error = "catboost_not_installed"
            logger.warning("[BTC_ML] catboost not installed; BTC ML sleeve disabled")
            return

        metadata_path = self.model_dir / "metadata.json"
        if not metadata_path.exists():
            self.load_error = f"missing_metadata:{metadata_path}"
            logger.warning("[BTC_ML] Missing metadata at %s", metadata_path)
            return

        try:
            self.metadata = json.loads(metadata_path.read_text())
            for side, filename in (self.metadata.get("models") or {}).items():
                model_path = self.model_dir / str(filename)
                if not model_path.exists():
                    continue
                model = CatBoostClassifier()
                model.load_model(str(model_path))
                self.models[side] = model
                self.feature_names[side] = list(model.feature_names_ or [])
            self.ready = "long" in self.models and "short" in self.models
            self.version = self.metadata.get("bundle_version", "unknown")
            if not self.ready:
                self.load_error = "long_or_short_model_missing"
        except Exception as exc:
            self.ready = False
            self.load_error = str(exc)
            logger.warning("[BTC_ML] Failed to load BTC model bundle: %s", exc)

    def predict_scores(self, row: dict[str, float]) -> dict[str, float] | None:
        if not self.ready:
            return None

        scores: dict[str, float] = {}
        for side in ("long", "short"):
            model = self.models.get(side)
            feature_names = self.feature_names.get(side) or []
            if not model or not feature_names:
                return None
            values = [float(row.get(name, 0.0) or 0.0) for name in feature_names]
            pool = Pool([values], feature_names=feature_names)
            scores[side] = float(model.predict_proba(pool)[0][1])
        return scores


class BitcoinModelStrategy(BaseStrategy):
    name = "bitcoin_model"
    description = "BTC futures impulse catch-up sleeve for Polymarket BTC barrier markets"

    def __init__(self, config, crypto_strategy: CryptoTemporalArbStrategy):
        super().__init__(config)
        self.cfg = config.bitcoin_model
        self.crypto_strategy = crypto_strategy
        self.bundle = BitcoinModelBundle(self.cfg.model_dir)
        self.enabled = bool(self.cfg.enabled and self.bundle.ready)
        self.log_path = LOG_DIR / "bitcoin_model_sleeve.jsonl"
        self.market_log_path = LOG_DIR / "bitcoin_model_market_feed.jsonl"
        self.context_log_path = LOG_DIR / "bitcoin_model_context.jsonl"
        self.feed = BinanceBtcFeatureFeed(
            symbol=self.cfg.symbol,
            bucket_seconds=self.cfg.bucket_seconds,
            horizon_seconds=self.cfg.horizon_seconds,
            cost_bps=self.cfg.cost_bps,
            min_signed_ratio=self.cfg.min_signed_ratio,
            min_depth_imbalance=self.cfg.min_depth_imbalance,
            min_trade_z=self.cfg.min_trade_z,
            min_directional_efficiency=self.cfg.min_directional_efficiency,
            warmup_buckets=self.cfg.warmup_buckets,
            depth_poll_seconds=self.cfg.depth_poll_seconds,
            metrics_poll_seconds=self.cfg.metrics_poll_seconds,
            funding_poll_seconds=self.cfg.funding_poll_seconds,
            book_ticker_enabled=self.cfg.book_ticker_enabled,
            max_trade_age_buckets=self.cfg.max_trade_age_buckets,
            max_depth_age_buckets=self.cfg.max_depth_age_buckets,
            max_metrics_age_buckets=self.cfg.max_metrics_age_buckets,
            max_funding_age_buckets=self.cfg.max_funding_age_buckets,
            log_path=self.log_path,
        )
        self.market_feed = PolymarketBtcMarketFeed(
            ws_url=self.cfg.polymarket_market_ws_url,
            ping_seconds=self.cfg.polymarket_ping_seconds,
            quote_ttl_seconds=self.cfg.polymarket_quote_ttl_seconds,
            recent_quote_grace_seconds=self.cfg.polymarket_recent_quote_grace_seconds,
            max_watch_assets=self.cfg.polymarket_max_watch_assets,
            log_path=self.market_log_path,
        )
        self.context_feed = BitcoinContextFeed(
            enabled=self.cfg.context_enabled,
            query=self.cfg.context_query,
            shock_window_minutes=self.cfg.context_shock_window_minutes,
            newsapi_key=self.cfg.newsapi_key,
            newsapi_poll_seconds=self.cfg.newsapi_poll_seconds,
            newsapi_page_size=self.cfg.newsapi_page_size,
            gdelt_enabled=self.cfg.gdelt_enabled,
            gdelt_poll_seconds=self.cfg.gdelt_poll_seconds,
            gdelt_max_records=self.cfg.gdelt_max_records,
            rss_feeds=list(getattr(getattr(config, "news", None), "rss_feeds", []) or []),
            x_bearer_token=self.cfg.x_bearer_token,
            x_stream_enabled=self.cfg.x_stream_enabled,
            x_rule_tag=self.cfg.x_rule_tag,
            x_rule_value=self.cfg.x_rule_value,
            log_path=self.context_log_path,
        )
        self._stats.update(
            {
                "bundle_ready": self.bundle.ready,
                "bundle_version": self.bundle.version,
                "bundle_error": self.bundle.load_error,
                "model_dir": str(self.bundle.model_dir),
                "feature_count": len(self.bundle.feature_names.get("long") or []),
                "min_source_fresh_score": self.cfg.min_source_fresh_score,
                "degraded_threshold": self.cfg.degraded_threshold,
                "degraded_direction_margin": self.cfg.degraded_direction_margin,
                "candidate_markets": 0,
                "matched_markets": 0,
                "scored_markets": 0,
                "last_scan_at": None,
                "last_bucket_at": None,
                "last_price": 0.0,
                "last_long_score": 0.0,
                "last_short_score": 0.0,
                "last_direction": "neutral",
                "last_source_fresh_score": 0.0,
                "last_effective_source_fresh_score": 0.0,
                "last_long_candidate": False,
                "last_short_candidate": False,
                "degraded_live_mode": False,
                "effective_long_threshold": self.cfg.long_threshold,
                "effective_short_threshold": self.cfg.short_threshold,
                "effective_direction_margin": self.cfg.min_direction_margin,
                "last_direction_mode": "neutral",
                "metrics_supported": True,
                "funding_supported": True,
                "supported_source_count": 4,
                "last_signals": 0,
                "last_live_quote_assets": 0,
                "last_recent_quote_assets": 0,
                "last_live_lagged_markets": 0,
                "last_context_regime": "disabled",
                "last_context_bias": "neutral",
                "last_context_intensity": 0.0,
                "last_context_hold_profile": "normal",
                "log_entries": 0,
                "last_log_at": None,
                "feed_stats": self.feed.stats,
                "market_feed_stats": self.market_feed.stats,
                "context_stats": self.context_feed.stats,
            }
        )

    @property
    def stats(self) -> dict:
        return {
            "name": self.name,
            "enabled": self.enabled,
            **self._stats,
            "log_path": str(self.log_path.resolve()),
        }

    async def close(self) -> None:
        await self.feed.close()
        await self.market_feed.close()
        await self.context_feed.close()

    async def scan(self, markets: list[Market], events: list[Event]) -> list[Signal]:
        del events
        self._stats["scans_completed"] += 1
        self._stats["last_scan_at"] = datetime.now(timezone.utc).isoformat()
        self._stats["bundle_ready"] = self.bundle.ready
        self._stats["bundle_error"] = self.bundle.load_error

        if not self.enabled:
            self._stats["last_signals"] = 0
            self._sync_feed_stats()
            self._append_scan_log(snapshot=None, scores=None, direction="disabled", signals=[], context_snapshot=None)
            return []

        await self.feed.ensure_started()
        await self.market_feed.ensure_started()
        await self.context_feed.ensure_started()
        await self.market_feed.update_watchlist(self._candidate_asset_ids(markets))
        snapshot = await self.feed.snapshot()
        self._apply_snapshot_stats(snapshot)
        context_snapshot = self.context_feed.snapshot()
        self._apply_context_stats(context_snapshot)

        scores: dict[str, float] | None = None
        direction = "neutral"
        signals: list[Signal] = []
        effective_fresh = snapshot.effective_source_fresh_score or snapshot.source_fresh_score
        degraded_live_mode = self._is_degraded_live_mode(snapshot)
        self._stats["degraded_live_mode"] = degraded_live_mode

        if snapshot.ready and snapshot.feature_row and effective_fresh >= self.cfg.min_source_fresh_score:
            scores = self.bundle.predict_scores(snapshot.feature_row)
            if scores:
                self._stats["last_long_score"] = round(scores["long"], 6)
                self._stats["last_short_score"] = round(scores["short"], 6)
                direction = self._choose_direction(snapshot, scores)
                if direction in {"bull", "bear"} and snapshot.last_price:
                    signals = self._build_signals(
                        markets=markets,
                        spot_price=float(snapshot.last_price),
                        scores=scores,
                        direction=direction,
                        context_snapshot=context_snapshot,
                    )
                if (
                    not signals
                    and snapshot.last_price
                    and degraded_live_mode
                    and direction in {"bull", "bear"}
                ):
                    alternate_direction = "bear" if direction == "bull" else "bull"
                    if self._should_use_market_direction_fallback(
                        scores=scores,
                        chosen_direction=direction,
                        alternate_direction=alternate_direction,
                    ):
                        alternate_signals = self._build_signals(
                            markets=markets,
                            spot_price=float(snapshot.last_price),
                            scores=scores,
                            direction=alternate_direction,
                            context_snapshot=context_snapshot,
                        )
                        if alternate_signals:
                            direction = alternate_direction
                            signals = alternate_signals
                            self._stats["last_direction_mode"] = f"{alternate_direction}_market_fallback"
                if (
                    not signals
                    and snapshot.last_price
                    and degraded_live_mode
                    and self._should_use_bull_catchup_fallback(scores=scores, snapshot=snapshot)
                ):
                    fallback_signals = self._build_signals(
                        markets=markets,
                        spot_price=float(snapshot.last_price),
                        scores=scores,
                        direction="bull",
                        context_snapshot=context_snapshot,
                    )
                    if fallback_signals:
                        direction = "bull"
                        signals = fallback_signals
                        self._stats["last_direction_mode"] = "bull_catchup_fallback"
                if not signals and snapshot.last_price and degraded_live_mode:
                    fallback_direction, fallback_signals = self._inventory_direction_fallback(
                        markets=markets,
                        spot_price=float(snapshot.last_price),
                        scores=scores,
                        context_snapshot=context_snapshot,
                    )
                    if fallback_signals:
                        direction = fallback_direction
                        signals = fallback_signals
                        self._stats["last_direction_mode"] = "market_inventory_fallback"

        self._stats["last_direction"] = direction
        self._stats["last_signals"] = len(signals)
        self._stats["signals_generated"] += len(signals)
        self._sync_feed_stats()
        self._append_scan_log(snapshot=snapshot, scores=scores, direction=direction, signals=signals, context_snapshot=context_snapshot)
        return signals

    def _apply_snapshot_stats(self, snapshot: FeedSnapshot) -> None:
        self._stats["last_bucket_at"] = snapshot.last_bucket_at
        self._stats["last_price"] = round(float(snapshot.last_price or 0.0), 2)
        self._stats["last_source_fresh_score"] = round(float(snapshot.source_fresh_score or 0.0), 4)
        self._stats["last_effective_source_fresh_score"] = round(float(snapshot.effective_source_fresh_score or 0.0), 4)
        self._stats["last_long_candidate"] = bool(snapshot.long_candidate)
        self._stats["last_short_candidate"] = bool(snapshot.short_candidate)
        row = snapshot.feature_row or {}
        self._stats["last_signed_ratio_12"] = round(float(row.get("signed_ratio_12", 0.0) or 0.0), 4)
        self._stats["last_depth_imbalance_1pct"] = round(float(row.get("depth_imbalance_1pct", 0.0) or 0.0), 4)
        self._stats["last_trade_count_z_12"] = round(float(row.get("trade_count_z_12", 0.0) or 0.0), 4)
        self._stats["last_directional_efficiency_12"] = round(float(row.get("directional_efficiency_12", 0.0) or 0.0), 4)
        self._stats["last_impulse_alignment_12"] = round(float(row.get("impulse_alignment_12", 0.0) or 0.0), 4)

    def _apply_context_stats(self, context_snapshot: ContextSnapshot) -> None:
        self._stats["last_context_regime"] = context_snapshot.regime
        self._stats["last_context_bias"] = context_snapshot.bias
        self._stats["last_context_intensity"] = round(float(context_snapshot.intensity or 0.0), 4)
        self._stats["last_context_hold_profile"] = context_snapshot.hold_profile

    def _sync_feed_stats(self) -> None:
        feed_stats = self.feed.stats
        market_feed_stats = self.market_feed.stats
        context_stats = self.context_feed.stats
        self._stats["feed_stats"] = feed_stats
        self._stats["market_feed_stats"] = market_feed_stats
        self._stats["context_stats"] = context_stats
        self._stats["log_entries"] = int(feed_stats.get("log_entries") or 0)
        self._stats["last_log_at"] = feed_stats.get("last_log_at")
        self._stats["metrics_supported"] = bool(feed_stats.get("metrics_supported", True))
        self._stats["funding_supported"] = bool(feed_stats.get("funding_supported", True))
        self._stats["supported_source_count"] = int(feed_stats.get("supported_source_count") or 4)
        self._stats["last_live_quote_assets"] = int(market_feed_stats.get("quoted_assets") or 0)
        self._stats["last_recent_quote_assets"] = int(market_feed_stats.get("recent_quoted_assets") or 0)

    def _is_degraded_live_mode(self, snapshot: FeedSnapshot) -> bool:
        return (
            snapshot.effective_source_fresh_score > snapshot.source_fresh_score
            or snapshot.source_fresh_score < 0.75
            or int(self.feed.stats.get("supported_source_count") or 4) < 4
        )

    def _choose_direction(self, snapshot: FeedSnapshot, scores: dict[str, float]) -> str:
        degraded_live_mode = self._is_degraded_live_mode(snapshot)
        long_threshold = self.cfg.degraded_threshold if degraded_live_mode else self.cfg.long_threshold
        short_threshold = self.cfg.degraded_threshold if degraded_live_mode else self.cfg.short_threshold
        direction_margin = self.cfg.degraded_direction_margin if degraded_live_mode else self.cfg.min_direction_margin
        self._stats["effective_long_threshold"] = long_threshold
        self._stats["effective_short_threshold"] = short_threshold
        self._stats["effective_direction_margin"] = direction_margin

        long_score = float(scores["long"])
        short_score = float(scores["short"])
        self._stats["last_direction_mode"] = "neutral"
        bull_mode = None
        if snapshot.long_candidate:
            bull_mode = "candidate"
        elif degraded_live_mode and self._degraded_impulse_alignment(snapshot, "bull"):
            bull_mode = "degraded_impulse"
        elif degraded_live_mode and self._degraded_score_breakout(snapshot, scores, "bull"):
            bull_mode = "degraded_breakout"
        elif degraded_live_mode and self._degraded_score_only(scores, "bull"):
            bull_mode = "degraded_score_only"
        bull_margin = DEGRADED_SCORE_ONLY_MARGIN if bull_mode == "degraded_score_only" else direction_margin
        if bull_mode and long_score >= long_threshold and (long_score - short_score) >= bull_margin:
            self._stats["last_direction_mode"] = bull_mode
            return "bull"
        bear_mode = None
        if snapshot.short_candidate:
            bear_mode = "candidate"
        elif degraded_live_mode and self._degraded_impulse_alignment(snapshot, "bear"):
            bear_mode = "degraded_impulse"
        elif degraded_live_mode and self._degraded_score_breakout(snapshot, scores, "bear"):
            bear_mode = "degraded_breakout"
        elif degraded_live_mode and self._degraded_score_only(scores, "bear"):
            bear_mode = "degraded_score_only"
        bear_margin = DEGRADED_SCORE_ONLY_MARGIN if bear_mode == "degraded_score_only" else direction_margin
        if bear_mode and short_score >= short_threshold and (short_score - long_score) >= bear_margin:
            self._stats["last_direction_mode"] = bear_mode
            return "bear"
        return "neutral"

    def _degraded_impulse_alignment(self, snapshot: FeedSnapshot, direction: str) -> bool:
        row = snapshot.feature_row or {}
        signed_ratio = float(row.get("signed_ratio_12", 0.0) or 0.0)
        depth_imbalance = float(row.get("depth_imbalance_1pct", 0.0) or 0.0)
        trade_z = float(row.get("trade_count_z_12", 0.0) or 0.0)
        directional_efficiency = float(row.get("directional_efficiency_12", 0.0) or 0.0)
        flow_accel = float(row.get("flow_accel_3v12", 0.0) or 0.0)
        impulse_alignment = float(row.get("impulse_alignment_12", 0.0) or 0.0)
        signed_threshold = self.cfg.min_signed_ratio * 0.5
        depth_threshold = self.cfg.min_depth_imbalance * 0.5
        efficiency_threshold = self.cfg.min_directional_efficiency * 0.5
        if direction == "bull":
            return (
                signed_ratio >= signed_threshold
                and depth_imbalance >= depth_threshold
                and trade_z >= 0.0
                and directional_efficiency >= efficiency_threshold
                and flow_accel >= -0.10
                and impulse_alignment >= 0.0
            )
        return (
            signed_ratio <= -signed_threshold
            and depth_imbalance <= -depth_threshold
            and trade_z >= 0.0
            and directional_efficiency >= efficiency_threshold
            and flow_accel <= 0.10
            and impulse_alignment >= 0.0
        )

    def _degraded_score_breakout(
        self,
        snapshot: FeedSnapshot,
        scores: dict[str, float],
        direction: str,
    ) -> bool:
        row = snapshot.feature_row or {}
        signed_ratio = float(row.get("signed_ratio_12", 0.0) or 0.0)
        directional_efficiency = float(row.get("directional_efficiency_12", 0.0) or 0.0)
        flow_accel = float(row.get("flow_accel_3v12", 0.0) or 0.0)
        long_score = float(scores["long"])
        short_score = float(scores["short"])
        breakout_gap = max(self.cfg.degraded_direction_margin, 0.03)
        if direction == "bull":
            return (
                long_score >= self.cfg.degraded_threshold
                and (long_score - short_score) >= breakout_gap
                and signed_ratio >= max(self.cfg.min_signed_ratio * 5.0, 0.20)
                and directional_efficiency >= max(self.cfg.min_directional_efficiency * 2.0, 0.50)
                and flow_accel >= -0.15
            )
        return (
            short_score >= self.cfg.degraded_threshold
            and (short_score - long_score) >= breakout_gap
            and signed_ratio <= -max(self.cfg.min_signed_ratio * 5.0, 0.20)
            and directional_efficiency >= max(self.cfg.min_directional_efficiency * 2.0, 0.50)
            and flow_accel <= 0.15
        )

    def _degraded_score_only(self, scores: dict[str, float], direction: str) -> bool:
        long_score = float(scores["long"])
        short_score = float(scores["short"])
        if direction == "bull":
            return (
                long_score >= DEGRADED_SCORE_ONLY_THRESHOLD
                and (long_score - short_score) >= DEGRADED_SCORE_ONLY_MARGIN
            )
        return (
            short_score >= DEGRADED_SCORE_ONLY_THRESHOLD
            and (short_score - long_score) >= DEGRADED_SCORE_ONLY_MARGIN
        )

    def _should_use_bull_catchup_fallback(
        self,
        *,
        scores: dict[str, float],
        snapshot: FeedSnapshot,
    ) -> bool:
        row = snapshot.feature_row or {}
        long_score = float(scores["long"])
        directional_efficiency = float(row.get("directional_efficiency_12", 0.0) or 0.0)
        signed_ratio = float(row.get("signed_ratio_12", 0.0) or 0.0)
        return (
            long_score >= DEGRADED_BULL_FALLBACK_THRESHOLD
            and directional_efficiency >= max(self.cfg.min_directional_efficiency * 2.0, 0.50)
            and signed_ratio >= -0.10
        )

    def _should_use_market_direction_fallback(
        self,
        *,
        scores: dict[str, float],
        chosen_direction: str,
        alternate_direction: str,
    ) -> bool:
        chosen_score = float(scores["long"] if chosen_direction == "bull" else scores["short"])
        alternate_score = float(scores["long"] if alternate_direction == "bull" else scores["short"])
        return (
            alternate_score >= MARKET_FALLBACK_SCORE_THRESHOLD
            and (chosen_score - alternate_score) <= MARKET_FALLBACK_MAX_SCORE_GAP
        )

    def _candidate_asset_ids(self, markets: list[Market]) -> list[str]:
        asset_ids: list[str] = []
        for market in markets:
            text = f"{market.question} {market.slug}".lower()
            if self.crypto_strategy._match_symbol(text) != "BTC":
                continue
            match = self.crypto_strategy._match_barrier_market("BTC", market, text)
            if not match:
                continue
            for outcome in market.outcomes[:2]:
                if outcome.token_id:
                    asset_ids.append(outcome.token_id)
        return asset_ids

    def _inventory_direction_fallback(
        self,
        *,
        markets: list[Market],
        spot_price: float,
        scores: dict[str, float],
        context_snapshot: ContextSnapshot,
    ) -> tuple[str, list[Signal]]:
        long_score = float(scores["long"])
        short_score = float(scores["short"])
        bull_signals = self._build_signals(
            markets=markets,
            spot_price=spot_price,
            scores=scores,
            direction="bull",
            context_snapshot=context_snapshot,
        )
        bear_signals = self._build_signals(
            markets=markets,
            spot_price=spot_price,
            scores=scores,
            direction="bear",
            context_snapshot=context_snapshot,
        )
        if bull_signals and not bear_signals and long_score >= MARKET_INVENTORY_FALLBACK_SCORE_THRESHOLD:
            return "bull", bull_signals
        if bear_signals and not bull_signals and short_score >= MARKET_INVENTORY_FALLBACK_SCORE_THRESHOLD:
            return "bear", bear_signals
        return "neutral", []

    def _build_signals(
        self,
        *,
        markets: list[Market],
        spot_price: float,
        scores: dict[str, float],
        direction: str,
        context_snapshot: ContextSnapshot,
    ) -> list[Signal]:
        best_by_group: dict[str, Signal] = {}
        candidate_markets = 0
        matched_markets = 0
        scored_markets = 0

        for market in markets:
            text = f"{market.question} {market.slug}".lower()
            if self.crypto_strategy._match_symbol(text) != "BTC":
                continue

            candidate_markets += 1
            match = self.crypto_strategy._match_barrier_market("BTC", market, text)
            if not match:
                continue
            matched_markets += 1

            if not self._market_is_in_scope(match=match, spot_price=spot_price, direction=direction):
                continue

            signal = self._build_barrier_signal(
                spot_price=spot_price,
                match=match,
                scores=scores,
                direction=direction,
                context_snapshot=context_snapshot,
            )
            if not signal:
                continue
            scored_markets += 1
            existing = best_by_group.get(signal.group_key or signal.market_slug)
            if existing is None or (signal.expected_edge, signal.confidence) > (
                existing.expected_edge,
                existing.confidence,
            ):
                best_by_group[signal.group_key or signal.market_slug] = signal

            if scored_markets >= SCORED_MARKET_LIMIT:
                break

        self._stats["candidate_markets"] = candidate_markets
        self._stats["matched_markets"] = matched_markets
        self._stats["scored_markets"] = scored_markets
        self._stats["last_live_quote_assets"] = int((self.market_feed.stats or {}).get("quoted_assets") or 0)
        self._stats["last_live_lagged_markets"] = scored_markets
        ranked = sorted(
            best_by_group.values(),
            key=lambda signal: (signal.expected_edge, signal.confidence),
            reverse=True,
        )
        return ranked[: self.cfg.max_signals_per_scan]

    def _market_is_in_scope(self, *, match: dict[str, Any], spot_price: float, direction: str) -> bool:
        horizon_days = float(match["years_left"]) * 365.25
        if horizon_days > self.cfg.max_resolution_days:
            return False

        barrier_price = float(match["barrier_price"])
        if spot_price <= 0.0 or barrier_price <= 0.0:
            return False

        barrier_distance_pct = abs(barrier_price - spot_price) / spot_price
        if barrier_distance_pct > self.cfg.max_barrier_distance_pct:
            return False

        kind = str(match["kind"])
        if direction == "bull":
            return kind in {"reach", "ath", "dip"}
        if direction == "bear":
            return kind in {"reach", "ath", "dip"}
        return False

    def _resolve_live_quotes(self, *, yes_token_id: str, no_token_id: str) -> dict[str, float | None]:
        yes_quote = self.market_feed.quote(yes_token_id, allow_recent_stale=True)
        no_quote = self.market_feed.quote(no_token_id, allow_recent_stale=True)
        return {
            "yes_mid": None if not yes_quote else yes_quote.midpoint,
            "yes_bid": None if not yes_quote else yes_quote.best_bid,
            "yes_ask": None if not yes_quote else yes_quote.best_ask,
            "yes_spread": None if not yes_quote else yes_quote.spread,
            "no_mid": None if not no_quote else no_quote.midpoint,
            "no_bid": None if not no_quote else no_quote.best_bid,
            "no_ask": None if not no_quote else no_quote.best_ask,
            "no_spread": None if not no_quote else no_quote.spread,
        }

    def _context_adjustment(self, *, direction: str, context_snapshot: ContextSnapshot) -> dict[str, Any]:
        if context_snapshot.regime in {"disabled", "normal_flow", "elevated_news_flow"} or context_snapshot.bias == "neutral":
            return {
                "block": False,
                "size_multiplier": 1.0,
                "confidence_delta": 0.0,
                "hold_profile": context_snapshot.hold_profile,
                "note": context_snapshot.regime,
            }

        aligned = (
            (direction == "bull" and context_snapshot.bias == "bull")
            or (direction == "bear" and context_snapshot.bias == "bear")
        )
        if aligned:
            size_multiplier = 1.0 + ((self.cfg.context_aligned_size_multiplier - 1.0) * context_snapshot.intensity)
            confidence_delta = self.cfg.context_aligned_confidence_bonus * context_snapshot.intensity
            return {
                "block": False,
                "size_multiplier": size_multiplier,
                "confidence_delta": confidence_delta,
                "hold_profile": "extended",
                "note": f"{context_snapshot.regime}:aligned",
            }

        block = context_snapshot.intensity >= self.cfg.context_block_intensity and context_snapshot.regime.endswith("news_shock")
        size_multiplier = max(0.25, self.cfg.context_opposing_size_multiplier)
        confidence_delta = -self.cfg.context_opposing_confidence_penalty * context_snapshot.intensity
        return {
            "block": block,
            "size_multiplier": size_multiplier,
            "confidence_delta": confidence_delta,
            "hold_profile": "shortened",
            "note": f"{context_snapshot.regime}:opposed",
        }

    def _build_barrier_signal(
        self,
        *,
        spot_price: float,
        match: dict[str, Any],
        scores: dict[str, float],
        direction: str,
        context_snapshot: ContextSnapshot,
    ) -> Signal | None:
        market = match["market"]
        yes_outcome = market.outcomes[match["yes_index"]]
        no_outcome = market.outcomes[match["no_index"]]
        if yes_outcome.price <= 0.0 or no_outcome.price <= 0.0:
            return None

        modeled_yes = self.crypto_strategy._estimate_barrier_probability(
            symbol="BTC",
            spot_price=spot_price,
            barrier_price=float(match["barrier_price"]),
            years_left=float(match["years_left"]),
        )
        if modeled_yes is None:
            return None

        live_quotes = self._resolve_live_quotes(
            yes_token_id=yes_outcome.token_id,
            no_token_id=no_outcome.token_id,
        )
        live_yes_mid = float(live_quotes["yes_mid"] or yes_outcome.price)
        live_no_mid = float(live_quotes["no_mid"] or no_outcome.price)
        fair_no = 1.0 - modeled_yes

        action: SignalAction | None = None
        target_outcome = None
        entry_price = 0.0
        edge = 0.0
        live_spread = 0.0
        kind = str(match["kind"])
        bullish_barrier = kind in {"reach", "ath"}

        if direction == "bull":
            if bullish_barrier:
                entry_price = float(live_quotes["yes_ask"] or live_yes_mid)
                live_spread = float(live_quotes["yes_spread"] or 0.0)
                edge = modeled_yes - entry_price
                if edge >= self.cfg.min_live_quote_edge and entry_price < self.cfg.max_entry_price:
                    action = SignalAction.BUY_YES
                    target_outcome = yes_outcome
            else:
                entry_price = float(live_quotes["no_ask"] or live_no_mid)
                live_spread = float(live_quotes["no_spread"] or 0.0)
                edge = fair_no - entry_price
                if edge >= self.cfg.min_live_quote_edge and entry_price < self.cfg.max_entry_price:
                    action = SignalAction.BUY_NO
                    target_outcome = no_outcome
        elif direction == "bear":
            if kind == "dip":
                entry_price = float(live_quotes["yes_ask"] or live_yes_mid)
                live_spread = float(live_quotes["yes_spread"] or 0.0)
                edge = modeled_yes - entry_price
                if edge >= self.cfg.min_live_quote_edge and entry_price < self.cfg.max_entry_price:
                    action = SignalAction.BUY_YES
                    target_outcome = yes_outcome
            else:
                entry_price = float(live_quotes["no_ask"] or live_no_mid)
                live_spread = float(live_quotes["no_spread"] or 0.0)
                edge = fair_no - entry_price
                if edge >= self.cfg.min_live_quote_edge and entry_price < self.cfg.max_entry_price:
                    action = SignalAction.BUY_NO
                    target_outcome = no_outcome

        if not action or not target_outcome:
            return None

        spread_pct = (live_spread / entry_price) if entry_price > 0.0 else 0.0
        if spread_pct > self.cfg.max_polymarket_quote_spread:
            return None

        context_adjustment = self._context_adjustment(direction=direction, context_snapshot=context_snapshot)
        if context_adjustment["block"]:
            return None

        direction_score = float(scores["long"] if direction == "bull" else scores["short"])
        degraded_live_mode = bool(self._stats.get("degraded_live_mode"))
        threshold = float(
            self.cfg.degraded_threshold if degraded_live_mode else (
                self.cfg.long_threshold if direction == "bull" else self.cfg.short_threshold
            )
        )
        score_excess = max(direction_score - threshold, 0.0)
        side_probability = modeled_yes if action == SignalAction.BUY_YES else fair_no
        confidence = min(
            MAX_SIGNAL_CONFIDENCE,
            max(direction_score, side_probability, BASE_SIGNAL_CONFIDENCE)
            + min(edge * EDGE_CONFIDENCE_MULTIPLIER, EDGE_CONFIDENCE_BONUS_CAP)
            + float(context_adjustment["confidence_delta"]),
        )
        suggested_size_usd = min(
            self.cfg.max_size_usd,
            max(
                self.cfg.min_size_usd,
                (
                    self.cfg.min_size_usd
                    + (edge * EDGE_SIZE_USD_MULTIPLIER)
                    + (score_excess * SCORE_SIZE_USD_MULTIPLIER)
                ) * float(context_adjustment["size_multiplier"]),
            ),
        )

        return Signal(
            source=SignalSource.BITCOIN_MODEL,
            action=action,
            market_slug=market.slug,
            condition_id=market.condition_id,
            token_id=target_outcome.token_id,
            confidence=confidence,
            expected_edge=edge * 100.0,
            group_key=self._barrier_group_key(match=match, action=action),
            reasoning=(
                f"BTC ML CATCH-UP: {direction.upper()} impulse | long={scores['long']:.3f} short={scores['short']:.3f} | "
                f"spot=${spot_price:,.0f} | {kind} ${float(match['barrier_price']):,.0f} | "
                f"model YES={modeled_yes:.1%} | entry={entry_price:.1%} | ctx={context_adjustment['note']}"
            ),
            suggested_size_usd=suggested_size_usd,
        )

    def _barrier_group_key(self, *, match: dict[str, Any], action: SignalAction) -> str:
        market = match["market"]
        expiry_bucket = (market.end_date or "unknown")[:10]
        barrier_bucket = int(round(float(match["barrier_price"])))
        kind = str(match["kind"])
        bullish = (
            (kind in {"reach", "ath"} and action == SignalAction.BUY_YES)
            or (kind == "dip" and action == SignalAction.BUY_NO)
        )
        thesis = "bull" if bullish else "bear"
        return f"btcml:BTC:{kind}:{thesis}:{expiry_bucket}:{barrier_bucket}"

    def _append_scan_log(
        self,
        *,
        snapshot: FeedSnapshot | None,
        scores: dict[str, float] | None,
        direction: str,
        signals: list[Signal],
        context_snapshot: ContextSnapshot | None,
    ) -> None:
        market_stats = self.market_feed.stats
        context_stats = self.context_feed.stats
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": self.name,
            "enabled": self.enabled,
            "bundle_ready": self.bundle.ready,
            "bundle_version": self.bundle.version,
            "bundle_error": self.bundle.load_error,
            "candidate_markets": self._stats.get("candidate_markets", 0),
            "matched_markets": self._stats.get("matched_markets", 0),
            "scored_markets": self._stats.get("scored_markets", 0),
            "direction": direction,
            "long_score": None if not scores else round(float(scores["long"]), 6),
            "short_score": None if not scores else round(float(scores["short"]), 6),
            "last_price": round(float(snapshot.last_price or 0.0), 2) if snapshot else None,
            "source_fresh_score": round(float(snapshot.source_fresh_score or 0.0), 4) if snapshot else 0.0,
            "effective_source_fresh_score": round(float(snapshot.effective_source_fresh_score or 0.0), 4) if snapshot else 0.0,
            "bucket_ready": bool(snapshot.ready) if snapshot else False,
            "bucket_at": snapshot.last_bucket_at if snapshot else None,
            "long_candidate": bool(snapshot.long_candidate) if snapshot else False,
            "short_candidate": bool(snapshot.short_candidate) if snapshot else False,
            "degraded_live_mode": bool(self._stats.get("degraded_live_mode")),
            "direction_mode": self._stats.get("last_direction_mode"),
            "effective_long_threshold": self._stats.get("effective_long_threshold"),
            "effective_short_threshold": self._stats.get("effective_short_threshold"),
            "effective_direction_margin": self._stats.get("effective_direction_margin"),
            "fast_lane_ready": bool(self._stats.get("feed_stats", {}).get("fast_lane_ready")),
            "live_quote_assets": int(market_stats.get("quoted_assets") or 0),
            "recent_quote_assets": int(market_stats.get("recent_quoted_assets") or 0),
            "lagged_markets": int(self._stats.get("last_live_lagged_markets") or 0),
            "context_regime": None if not context_snapshot else context_snapshot.regime,
            "context_bias": None if not context_snapshot else context_snapshot.bias,
            "context_intensity": 0.0 if not context_snapshot else round(float(context_snapshot.intensity or 0.0), 4),
            "context_hold_profile": None if not context_snapshot else context_snapshot.hold_profile,
            "signals": [
                {
                    "market": signal.market_slug,
                    "action": signal.action.value,
                    "confidence": round(signal.confidence, 4),
                    "edge": round(signal.expected_edge, 3),
                    "size_usd": round(signal.suggested_size_usd, 2),
                }
                for signal in signals
            ],
            "feed": self.feed.stats,
            "market_feed": {
                "connected": bool(market_stats.get("connected")),
                "watched_assets": int(market_stats.get("watched_assets") or 0),
                "quoted_assets": int(market_stats.get("quoted_assets") or 0),
                "recent_quoted_assets": int(market_stats.get("recent_quoted_assets") or 0),
                "quote_updates": int(market_stats.get("quote_updates") or 0),
                "last_quote_at": market_stats.get("last_quote_at"),
                "last_error": market_stats.get("last_error"),
            },
            "context": {
                "enabled": bool(context_stats.get("enabled")),
                "healthy_provider_count": int(context_stats.get("healthy_provider_count") or 0),
                "provider_count": int(context_stats.get("provider_count") or 0),
                "recent_items": int(context_stats.get("recent_item_count") or 0),
                "last_item_at": context_stats.get("last_item_at"),
                "last_error": context_stats.get("last_error"),
            },
        }
        self.feed.append_log(payload)
