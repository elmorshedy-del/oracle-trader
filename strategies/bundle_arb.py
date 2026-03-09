"""
Strategy: Strict Bundle Arbitrage (comparison-book only)
=======================================================
Scans only the safest event bundles:
- explicit multi-outcome event partitions
- neg-risk event groups when available

This stays separate from the main legacy arbitrage strategy so we can compare
bundle-style opportunities without changing the winning legacy arb path.
"""

from __future__ import annotations

import logging
import re

from data.models import Event, Market, Signal, SignalAction, SignalSource
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

BASE_FEE_RATE = 0.02
PLACEHOLDER_RE = re.compile(r"\b(other|others|none of the above|field|any other)\b", re.IGNORECASE)


class BundleArbitrageStrategy(BaseStrategy):
    name = "bundle_arb_strict"
    description = "Strict bundle arbitrage on neg-risk and explicit event partitions"

    def __init__(self, config):
        super().__init__(config)
        self.cfg = config.bundle_arb
        self._stats.update(
            {
                "eligible_events": 0,
                "neg_risk_events": 0,
                "placeholder_skips": 0,
                "low_edge_skips": 0,
            }
        )

    async def scan(self, markets: list[Market], events: list[Event]) -> list[Signal]:
        self._stats["scans_completed"] += 1
        self._stats["eligible_events"] = 0
        self._stats["neg_risk_events"] = 0
        self._stats["placeholder_skips"] = 0
        self._stats["low_edge_skips"] = 0

        signals: list[Signal] = []
        for event in events:
            signal = self._check_event_bundle(event)
            if signal:
                signals.append(signal)

        self._stats["signals_generated"] += len(signals)
        return signals

    def _check_event_bundle(self, event: Event) -> Signal | None:
        markets = [
            market
            for market in event.markets
            if market.active and not market.closed and len(market.outcomes) >= 1
        ]
        if len(markets) < 3 or len(markets) > self.cfg.max_outcomes:
            return None
        if event.total_volume < self.cfg.min_event_volume_usd:
            return None
        if any(m.liquidity < self.cfg.min_liquidity_usd for m in markets):
            return None

        if any(self._is_placeholder_market(m) for m in markets):
            self._stats["placeholder_skips"] += 1
            return None

        all_neg_risk = all(m.neg_risk for m in markets)
        if self.cfg.require_neg_risk and not all_neg_risk:
            return None

        total_cost = 0.0
        total_fees = 0.0
        outcome_tokens: list[str] = []
        weakest_liquidity = min(m.liquidity for m in markets)

        for market in markets:
            outcome = market.outcomes[0]
            yes_ask = outcome.book_ask or outcome.price
            if yes_ask <= 0 or yes_ask >= 1:
                return None

            exec_price = yes_ask * 1.005
            fee = BASE_FEE_RATE * min(exec_price, 1 - exec_price)
            total_cost += exec_price
            total_fees += fee
            outcome_tokens.append(outcome.token_id)

        gross_profit = 1.0 - total_cost
        net_profit = gross_profit - total_fees
        if net_profit < self.cfg.min_profit_cents / 100:
            self._stats["low_edge_skips"] += 1
            return None

        self._stats["eligible_events"] += 1
        if all_neg_risk:
            self._stats["neg_risk_events"] += 1

        confidence = min(
            0.55
            + min(net_profit * 8, 0.25)
            + (0.05 if all_neg_risk else 0.0)
            + (0.03 if len(markets) <= 6 else 0.0),
            0.97,
        )

        bundle_type = "neg-risk bundle" if all_neg_risk else "strict event bundle"
        return Signal(
            source=SignalSource.BUNDLE_ARB,
            action=SignalAction.ARB_ALL,
            market_slug=event.slug,
            condition_id=event.markets[0].condition_id,
            group_key=f"bundle:{event.slug}",
            confidence=confidence,
            expected_edge=net_profit * 100,
            reasoning=(
                f"{bundle_type}: {len(markets)} legs | cost={total_cost:.3f} | "
                f"gross={gross_profit:.3f} | fees={total_fees:.3f} | net={net_profit:.3f}"
            ),
            arb_outcomes=outcome_tokens,
            arb_total_cost=total_cost,
            arb_guaranteed_payout=1.0,
            suggested_size_usd=min(self.cfg.max_position_usd, weakest_liquidity * 0.01),
        )

    @staticmethod
    def _is_placeholder_market(market: Market) -> bool:
        haystacks = [market.question or "", market.slug or ""]
        return any(PLACEHOLDER_RE.search(text) for text in haystacks)
