"""
Strategy: Multi-Outcome Arbitrage (Layer 2 — The Bonus)
======================================================
Scans events with multiple outcomes (e.g., "Who will win?")
and detects when the sum of all YES prices ≠ $1.00.

If sum < $1.00 → buy all outcomes → guaranteed profit at resolution.
If sum > $1.00 → flag as overpriced (can short if possible).
"""

import logging
from data.models import Market, Event, Signal, SignalSource, SignalAction
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

# Polymarket fee: baseRate * min(price, 1-price) * size
# baseRate is typically ~2% for standard markets
BASE_FEE_RATE = 0.02


class ArbitrageStrategy(BaseStrategy):
    name = "multi_outcome_arbitrage"
    description = "Detect and exploit mispricing across multi-outcome events"

    def __init__(self, config):
        super().__init__(config)
        self.cfg = config.arbitrage

    async def scan(self, markets: list[Market], events: list[Event]) -> list[Signal]:
        signals = []
        self._stats["scans_completed"] += 1

        # --- Binary market mispricing (YES + NO ≠ $1.00) ---
        for market in markets:
            signal = self._check_binary_mispricing(market)
            if signal:
                signals.append(signal)

        # --- Multi-outcome event mispricing (all YES prices ≠ $1.00) ---
        for event in events:
            if len(event.markets) < 3:
                continue  # need 3+ outcomes for multi-outcome arb
            signal = self._check_multi_outcome_arb(event)
            if signal:
                signals.append(signal)

        return signals

    def _check_binary_mispricing(self, market: Market) -> Signal | None:
        """Check if YES + NO deviates from $1.00 in a binary market."""
        if len(market.outcomes) != 2:
            return None
        if market.closed or not market.active:
            return None
        if market.liquidity < self.config.risk.min_liquidity_usd:
            return None

        yes_price = market.outcomes[0].price
        no_price = market.outcomes[1].price

        if yes_price <= 0 or no_price <= 0:
            return None

        total = yes_price + no_price
        deviation = abs(total - 1.0)

        # Estimate fees for buying both sides
        fee_yes = BASE_FEE_RATE * min(yes_price, 1 - yes_price)
        fee_no = BASE_FEE_RATE * min(no_price, 1 - no_price)
        total_fees = fee_yes + fee_no

        # Net profit after fees
        if total < 1.0:
            gross_profit = 1.0 - total
            net_profit = gross_profit - total_fees
        else:
            # Overpriced — no direct arb unless we can short
            return None

        if net_profit < self.cfg.min_profit_cents / 100:
            return None

        self._stats["signals_generated"] += 1

        return Signal(
            source=SignalSource.ARBITRAGE,
            action=SignalAction.ARB_ALL_OUTCOMES,
            market_slug=market.slug,
            condition_id=market.condition_id,
            confidence=min(net_profit * 20, 1.0),  # higher profit = higher confidence
            expected_edge=net_profit * 100,  # in cents
            reasoning=(
                f"Binary mispricing: YES={yes_price:.3f} + NO={no_price:.3f} = "
                f"{total:.3f} | Gross: {gross_profit:.3f} | Fees: {total_fees:.3f} | "
                f"Net: {net_profit:.3f}"
            ),
            arb_outcomes=[o.token_id for o in market.outcomes],
            arb_total_cost=total,
            arb_guaranteed_payout=1.0,
            suggested_size_usd=min(
                self.config.risk.max_position_usd,
                market.liquidity * 0.01  # don't take more than 1% of liquidity
            ),
        )

    def _check_multi_outcome_arb(self, event: Event) -> Signal | None:
        """
        Check if buying YES on every outcome in a multi-outcome event
        costs less than $1.00 (guaranteed profit since exactly one resolves to $1).
        """
        if len(event.markets) > self.cfg.max_outcomes:
            return None

        total_cost = 0.0
        total_fees = 0.0
        outcome_tokens = []
        all_valid = True

        for market in event.markets:
            if not market.outcomes or market.outcomes[0].price <= 0:
                all_valid = False
                break

            yes_price = market.outcomes[0].price
            if market.liquidity < self.cfg.min_liquidity_usd:
                all_valid = False
                break

            # Use ask price if available (more realistic execution)
            exec_price = market.outcomes[0].book_ask or yes_price
            fee = BASE_FEE_RATE * min(exec_price, 1 - exec_price)

            total_cost += exec_price
            total_fees += fee
            outcome_tokens.append(market.outcomes[0].token_id)

        if not all_valid:
            return None

        gross_profit = 1.0 - total_cost
        net_profit = gross_profit - total_fees

        if net_profit < self.cfg.min_profit_cents / 100:
            return None

        self._stats["signals_generated"] += 1
        logger.info(
            f"[ARB] Multi-outcome arb on {event.title}: "
            f"cost={total_cost:.3f}, net_profit={net_profit:.3f}"
        )

        return Signal(
            source=SignalSource.ARBITRAGE,
            action=SignalAction.ARB_ALL_OUTCOMES,
            market_slug=event.slug,
            condition_id=event.markets[0].condition_id if event.markets else "",
            confidence=min(net_profit * 10, 1.0),
            expected_edge=net_profit * 100,
            reasoning=(
                f"Multi-outcome arb ({len(event.markets)} outcomes): "
                f"Total cost: ${total_cost:.3f} | Payout: $1.00 | "
                f"Fees: ${total_fees:.3f} | Net: ${net_profit:.3f}"
            ),
            arb_outcomes=outcome_tokens,
            arb_total_cost=total_cost,
            arb_guaranteed_payout=1.0,
            suggested_size_usd=min(
                self.config.risk.max_position_usd,
                min(m.liquidity for m in event.markets) * 0.01
            ),
        )
