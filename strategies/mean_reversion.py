"""
Strategy: Mean Reversion (Competing Signal — test against News)
==============================================================
Detects overreactions: when a market's price drops or spikes
significantly in a short window, bet on reversion toward the baseline.
"""

import logging
from datetime import datetime, timezone, timedelta
from data.models import Market, Event, Signal, SignalSource, SignalAction
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    name = "mean_reversion"
    description = "Buy dips / sell spikes, betting on price reversion"

    def __init__(self, config, collector=None):
        super().__init__(config)
        self.cfg = config.mean_reversion
        self.collector = collector
        # Store price baselines: token_id -> {baseline, prices, updated}
        self._baselines: dict[str, dict] = {}

    async def scan(self, markets: list[Market], events: list[Event]) -> list[Signal]:
        if not self.cfg.enabled:
            return []

        self._stats["scans_completed"] += 1
        signals = []

        # Limit to top 20 markets by volume to avoid flooding API
        sorted_markets = sorted([m for m in markets if m.active and not m.closed and m.outcomes and m.liquidity >= self.config.risk.min_liquidity_usd], key=lambda x: x.volume_24h, reverse=True)[:20]
        for market in sorted_markets:

            token_id = market.outcomes[0].token_id
            current_price = market.outcomes[0].price

            if current_price <= 0.05 or current_price >= 0.95:
                continue  # skip extreme prices

            # Resolution-aware: skip markets within 30 days of close
            if market.end_date:
                try:
                    from datetime import datetime
                    end = datetime.fromisoformat(market.end_date.replace("Z", "+00:00"))
                    days_left = (end - datetime.now(timezone.utc)).days
                    if days_left < 7:
                        continue
                except Exception:
                    pass

            # Update baseline
            baseline = await self._get_baseline(token_id, current_price)
            if baseline is None:
                continue

            # Check for significant drop
            drop_pct = (baseline - current_price) / baseline if baseline > 0 else 0
            spike_pct = (current_price - baseline) / baseline if baseline > 0 else 0

            signal = None

            if drop_pct >= self.cfg.drop_threshold_pct:
                # Price dropping — momentum says ride it down (buy NO)
                # Closer to resolution = stronger signal
                days_multiplier = min(30 / max(days_left, 1), 3.0) if days_left else 1.0
                confidence = min(drop_pct * 2 * days_multiplier, 0.9)
                edge = drop_pct * 0.5  # expect trend to continue ~50% further

                signal = Signal(
                    source=SignalSource.MEAN_REVERSION,
                    action=SignalAction.BUY_NO,
                    market_slug=market.slug,
                    condition_id=market.condition_id,
                    token_id=market.outcomes[1].token_id if len(market.outcomes) > 1 else None,
                    confidence=confidence,
                    expected_edge=edge * 100,
                    reasoning=(
                        f"MOMENTUM (drop): Price fell {drop_pct:.1%} "
                        f"from baseline {baseline:.3f} → {current_price:.3f} | "
                        f"Riding trend down | Days left: {days_left or '?'}"
                    ),
                    suggested_size_usd=self.config.risk.max_position_usd * 0.2 * confidence,
                )

            elif spike_pct >= self.cfg.drop_threshold_pct:
                # Price spiking — momentum says ride it up (buy YES)
                days_multiplier = min(30 / max(days_left, 1), 3.0) if days_left else 1.0
                confidence = min(spike_pct * 2 * days_multiplier, 0.9)
                edge = spike_pct * 0.5

                signal = Signal(
                    source=SignalSource.MEAN_REVERSION,
                    action=SignalAction.BUY_YES,
                    market_slug=market.slug,
                    condition_id=market.condition_id,
                    token_id=token_id,
                    confidence=confidence,
                    expected_edge=edge * 100,
                    reasoning=(
                        f"MOMENTUM (spike): Price rose {spike_pct:.1%} "
                        f"from baseline {baseline:.3f} → {current_price:.3f} | "
                        f"Riding trend up | Days left: {days_left or '?'}"
                    ),
                    suggested_size_usd=self.config.risk.max_position_usd * 0.2 * confidence,
                )

            if signal:
                signals.append(signal)
                self._stats["signals_generated"] += 1
                logger.info(f"[MEAN_REV] {signal.reasoning}")

        return signals

    async def _get_baseline(self, token_id: str, current_price: float) -> float | None:
        """
        Get the baseline price for a token over the lookback window.
        Uses price history if collector is available, otherwise maintains a running average.
        """
        if token_id in self._baselines:
            entry = self._baselines[token_id]
            entry["prices"].append(current_price)
            # Keep only lookback window worth of prices
            max_samples = self.cfg.lookback_hours * 2  # assume ~2 scans per hour
            if len(entry["prices"]) > max_samples:
                entry["prices"] = entry["prices"][-max_samples:]
            entry["baseline"] = sum(entry["prices"]) / len(entry["prices"])
            entry["updated"] = datetime.now(timezone.utc)
            return entry["baseline"]

        # First time seeing this token — try to fetch history
        if self.collector:
            try:
                history = await self.collector.get_price_history(
                    token_id, interval="1h", fidelity=self.cfg.lookback_hours
                )
                if history:
                    prices = [float(h.get("p", 0)) for h in history if h.get("p")]
                    if prices:
                        baseline = sum(prices) / len(prices)
                        self._baselines[token_id] = {
                            "baseline": baseline,
                            "prices": prices,
                            "updated": datetime.now(timezone.utc),
                        }
                        return baseline
            except Exception as e:
                logger.debug(f"[MEAN_REV] No history for {token_id}: {e}")

        # Fallback: start tracking from now
        self._baselines[token_id] = {
            "baseline": current_price,
            "prices": [current_price],
            "updated": datetime.now(timezone.utc),
        }
        return None  # not enough data yet
