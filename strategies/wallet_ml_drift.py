"""
Strategy: Wallet ML 30m Drift Shadow
=====================================
Detects directional price drift over 30-minute windows using an
EMA-smoothed z-score model.  When a token's recent price trajectory
deviates significantly from its rolling mean, a signal is emitted.

This is a *shadow* strategy — it runs in all modes and logs signals
for analysis, but the pipeline treats it like any other source.
"""

import logging
import math
from datetime import datetime, timezone
from data.models import Market, Event, Signal, SignalSource, SignalAction
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class WalletMLDriftStrategy(BaseStrategy):
    name = "wallet_ml_30m_drift_shadow"
    description = "ML-inspired drift detection on 30-min price windows"

    def __init__(self, config):
        super().__init__(config)
        self.cfg = config.wallet_ml_drift
        # token_id -> list of {price, timestamp}
        self._price_windows: dict[str, list[dict]] = {}
        # token_id -> EMA of price
        self._ema: dict[str, float] = {}

    async def scan(self, markets: list[Market], events: list[Event]) -> list[Signal]:
        if not self.cfg.enabled:
            return []

        self._stats["scans_completed"] += 1
        signals = []

        # Focus on top markets by volume to keep scan fast
        eligible = sorted(
            [m for m in markets if m.active and not m.closed and m.outcomes
             and m.liquidity >= self.config.risk.min_liquidity_usd],
            key=lambda x: x.volume_24h, reverse=True,
        )[:30]

        now = datetime.now(timezone.utc)

        for market in eligible:
            token_id = market.outcomes[0].token_id
            price = market.outcomes[0].price

            if price <= 0.05 or price >= 0.95:
                continue

            # Update rolling window
            window = self._price_windows.setdefault(token_id, [])
            window.append({"price": price, "ts": now})

            # Trim to window_size samples
            if len(window) > self.cfg.window_size:
                window[:] = window[-self.cfg.window_size:]

            # Need at least 3 samples to compute meaningful stats
            if len(window) < 3:
                continue

            # Update EMA
            if token_id in self._ema:
                self._ema[token_id] = (
                    self.cfg.ema_alpha * price
                    + (1 - self.cfg.ema_alpha) * self._ema[token_id]
                )
            else:
                self._ema[token_id] = price
                continue  # first observation, skip

            # Compute drift z-score
            prices = [s["price"] for s in window]
            mean_p = sum(prices) / len(prices)
            var_p = sum((p - mean_p) ** 2 for p in prices) / len(prices)
            std_p = math.sqrt(var_p) if var_p > 0 else 0

            if std_p < 1e-6:
                continue

            drift = price - self._ema[token_id]
            zscore = drift / std_p

            if abs(zscore) < self.cfg.min_drift_zscore:
                continue

            # Confidence proportional to z-score strength, capped at 0.90
            confidence = min(abs(zscore) / 4.0, 0.90)
            if confidence < self.cfg.min_confidence:
                continue

            edge = abs(drift) / price if price > 0 else 0

            if zscore > 0:
                # Upward drift — ride momentum (buy YES)
                action = SignalAction.BUY_YES
                tid = token_id
                direction = "UP"
            else:
                # Downward drift — ride momentum (buy NO)
                action = SignalAction.BUY_NO
                tid = market.outcomes[1].token_id if len(market.outcomes) > 1 else None
                direction = "DOWN"

            signal = Signal(
                source=SignalSource.WALLET_ML_DRIFT,
                action=action,
                market_slug=market.slug,
                condition_id=market.condition_id,
                token_id=tid,
                confidence=confidence,
                expected_edge=edge * 100,
                reasoning=(
                    f"ML DRIFT: {direction} drift z={zscore:.2f} on 30m window | "
                    f"EMA={self._ema[token_id]:.4f} price={price:.4f} | "
                    f"{market.slug}"
                ),
                suggested_size_usd=self.config.risk.max_position_usd * 0.15 * confidence,
            )
            signals.append(signal)
            self._stats["signals_generated"] += 1
            logger.info(f"[ML_DRIFT] {signal.reasoning}")

        if not signals and self._stats["scans_completed"] % 50 == 0:
            logger.info(
                f"[ML_DRIFT] Tracking {len(self._price_windows)} tokens, "
                f"{len(self._ema)} EMAs"
            )

        return signals
