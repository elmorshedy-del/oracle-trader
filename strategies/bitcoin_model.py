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
            max_trade_age_buckets=self.cfg.max_trade_age_buckets,
            max_depth_age_buckets=self.cfg.max_depth_age_buckets,
            max_metrics_age_buckets=self.cfg.max_metrics_age_buckets,
            max_funding_age_buckets=self.cfg.max_funding_age_buckets,
            log_path=self.log_path,
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
                "log_entries": 0,
                "last_log_at": None,
                "feed_stats": self.feed.stats,
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

    async def scan(self, markets: list[Market], events: list[Event]) -> list[Signal]:
        del events
        self._stats["scans_completed"] += 1
        self._stats["last_scan_at"] = datetime.now(timezone.utc).isoformat()
        self._stats["bundle_ready"] = self.bundle.ready
        self._stats["bundle_error"] = self.bundle.load_error

        if not self.enabled:
            self._stats["last_signals"] = 0
            self._sync_feed_stats()
            self._append_scan_log(snapshot=None, scores=None, direction="disabled", signals=[])
            return []

        await self.feed.ensure_started()
        snapshot = await self.feed.snapshot()
        self._apply_snapshot_stats(snapshot)

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
                    )
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
                    )
                    if fallback_signals:
                        direction = "bull"
                        signals = fallback_signals
                        self._stats["last_direction_mode"] = "bull_catchup_fallback"

        self._stats["last_direction"] = direction
        self._stats["last_signals"] = len(signals)
        self._stats["signals_generated"] += len(signals)
        self._sync_feed_stats()
        self._append_scan_log(snapshot=snapshot, scores=scores, direction=direction, signals=signals)
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

    def _sync_feed_stats(self) -> None:
        feed_stats = self.feed.stats
        self._stats["feed_stats"] = feed_stats
        self._stats["log_entries"] = int(feed_stats.get("log_entries") or 0)
        self._stats["last_log_at"] = feed_stats.get("last_log_at")
        self._stats["metrics_supported"] = bool(feed_stats.get("metrics_supported", True))
        self._stats["funding_supported"] = bool(feed_stats.get("funding_supported", True))
        self._stats["supported_source_count"] = int(feed_stats.get("supported_source_count") or 4)

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

    def _build_signals(
        self,
        *,
        markets: list[Market],
        spot_price: float,
        scores: dict[str, float],
        direction: str,
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

    def _build_barrier_signal(
        self,
        *,
        spot_price: float,
        match: dict[str, Any],
        scores: dict[str, float],
        direction: str,
    ) -> Signal | None:
        market = match["market"]
        yes_outcome = market.outcomes[match["yes_index"]]
        no_outcome = market.outcomes[match["no_index"]]
        yes_price = float(yes_outcome.price)
        no_price = float(no_outcome.price)
        if yes_price <= 0.0 or no_price <= 0.0:
            return None

        modeled_yes = self.crypto_strategy._estimate_barrier_probability(
            symbol="BTC",
            spot_price=spot_price,
            barrier_price=float(match["barrier_price"]),
            years_left=float(match["years_left"]),
        )
        if modeled_yes is None:
            return None

        action: SignalAction | None = None
        target_outcome = None
        edge = 0.0
        kind = str(match["kind"])

        bullish_barrier = kind in {"reach", "ath"}
        if direction == "bull":
            if bullish_barrier:
                edge = modeled_yes - yes_price
                if edge >= self.cfg.min_barrier_edge and yes_price < self.cfg.max_entry_price:
                    action = SignalAction.BUY_YES
                    target_outcome = yes_outcome
            else:
                edge = yes_price - modeled_yes
                if edge >= self.cfg.min_barrier_edge and no_price < self.cfg.max_entry_price:
                    action = SignalAction.BUY_NO
                    target_outcome = no_outcome
        elif direction == "bear":
            if kind == "dip":
                edge = modeled_yes - yes_price
                if edge >= self.cfg.min_barrier_edge and yes_price < self.cfg.max_entry_price:
                    action = SignalAction.BUY_YES
                    target_outcome = yes_outcome
            else:
                edge = yes_price - modeled_yes
                if edge >= self.cfg.min_barrier_edge and no_price < self.cfg.max_entry_price:
                    action = SignalAction.BUY_NO
                    target_outcome = no_outcome

        if not action or not target_outcome:
            return None

        direction_score = float(scores["long"] if direction == "bull" else scores["short"])
        degraded_live_mode = bool(self._stats.get("degraded_live_mode"))
        threshold = float(
            self.cfg.degraded_threshold if degraded_live_mode else (
                self.cfg.long_threshold if direction == "bull" else self.cfg.short_threshold
            )
        )
        score_excess = max(direction_score - threshold, 0.0)
        side_probability = modeled_yes if action == SignalAction.BUY_YES else (1.0 - modeled_yes)
        confidence = min(
            MAX_SIGNAL_CONFIDENCE,
            max(direction_score, side_probability, BASE_SIGNAL_CONFIDENCE)
            + min(edge * EDGE_CONFIDENCE_MULTIPLIER, EDGE_CONFIDENCE_BONUS_CAP),
        )
        suggested_size_usd = min(
            self.cfg.max_size_usd,
            max(
                self.cfg.min_size_usd,
                self.cfg.min_size_usd
                + (edge * EDGE_SIZE_USD_MULTIPLIER)
                + (score_excess * SCORE_SIZE_USD_MULTIPLIER),
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
            group_key=self.crypto_strategy._barrier_group_key("BTC", market, kind, action),
            reasoning=(
                f"BTC ML CATCH-UP: {direction.upper()} impulse | long={scores['long']:.3f} short={scores['short']:.3f} | "
                f"spot=${spot_price:,.0f} | {kind} ${float(match['barrier_price']):,.0f} | "
                f"model YES={modeled_yes:.1%} vs market YES={yes_price:.1%}"
            ),
            suggested_size_usd=suggested_size_usd,
        )

    def _append_scan_log(
        self,
        *,
        snapshot: FeedSnapshot | None,
        scores: dict[str, float] | None,
        direction: str,
        signals: list[Signal],
    ) -> None:
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
        }
        self.feed.append_log(payload)
