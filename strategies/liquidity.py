"""
Strategy: Hedged Liquidity Provision (Layer 1 — The Salary)
==========================================================
Based on the "Optimal Liquidity Provision on Prediction Markets" paper.

Places resting limit orders on BOTH sides of a market to:
1. Earn Polymarket's daily Liquidity Rewards
2. Maintain a hedged (risk-bounded) position

Key formulas from the paper:
  - Scoring: S(s) = ((v - s) / v)^2 * b
  - Risk-free hedge: π = 1.00 - (p_YES + p_NO)
  - Kelly sizing: f* = (p*b - q) / b
  - Max overpayment: Δ_max = R / (n * T)
"""

import logging
from data.models import Market, Event, Signal, SignalSource, SignalAction
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class HedgedLiquidityStrategy(BaseStrategy):
    name = "hedged_liquidity"
    description = "Earn rewards by placing hedged limit orders on both sides"

    def __init__(self, config):
        super().__init__(config)
        self.cfg = config.liquidity

    async def scan(self, markets: list[Market], events: list[Event]) -> list[Signal]:
        signals = []
        self._stats["scans_completed"] += 1

        for market in markets:
            if not self._is_eligible(market):
                continue

            score = self._calculate_reward_score(market)
            hedge_cost = self._calculate_hedge_cost(market)
            expected_reward = self._estimate_daily_reward(market, score)
            edge = expected_reward - hedge_cost

            if edge > 0:
                # Kelly sizing
                kelly_fraction = self._kelly_fraction(market, expected_reward, hedge_cost)
                suggested_size = min(
                    kelly_fraction * self.config.risk.max_total_exposure_usd,
                    self.config.risk.max_position_usd,
                )

                signal = Signal(
                    source=SignalSource.LIQUIDITY,
                    action=SignalAction.HEDGE_BOTH,
                    market_slug=market.slug,
                    condition_id=market.condition_id,
                    confidence=min(score / 100, 1.0),
                    expected_edge=edge,
                    reasoning=(
                        f"Reward score: {score:.1f} | "
                        f"Est. daily reward: ${expected_reward:.3f} | "
                        f"Hedge cost: ${hedge_cost:.3f} | "
                        f"Net edge: ${edge:.3f} | "
                        f"Kelly: {kelly_fraction:.2%}"
                    ),
                    suggested_size_usd=suggested_size,
                )
                signals.append(signal)
                self._stats["signals_generated"] += 1
                logger.info(f"[LIQUIDITY] Signal: {market.slug} — edge ${edge:.3f}")

        return signals

    def _is_eligible(self, market: Market) -> bool:
        """Check market selection criteria from Table 1 of the paper."""
        if market.closed or not market.active:
            return False
        if len(market.outcomes) < 2:
            return False
        if market.reward_pool <= 0:
            return False
        if market.liquidity < self.config.risk.min_liquidity_usd:
            return False

        yes_price = market.outcomes[0].price
        no_price = market.outcomes[1].price if len(market.outcomes) > 1 else 0

        # Prefer markets near 50/50 (symmetric pricing reduces hedge skew)
        if self.cfg.prefer_price_near_50:
            if yes_price < 0.25 or yes_price > 0.75:
                return False

        # Max spread check
        if market.spread > self.cfg.max_spread_cents / 100:
            return False

        return True

    def _calculate_reward_score(self, market: Market) -> float:
        """
        Calculate expected reward score per order.
        S(s) = ((v - s) / v)^2 * b

        Where:
        - v = max spread (qualifying distance from mid)
        - s = our distance from mid (we target close to 0)
        - b = market multiplier
        """
        v = market.max_spread_for_rewards
        if v <= 0:
            return 0.0

        s = self.cfg.target_distance_cents / 100  # our target distance from mid
        b = market.reward_pool  # using pool as proxy for multiplier

        if s >= v:
            return 0.0

        score = ((v - s) / v) ** 2 * b
        # Double for two-sided placement (YES + NO)
        return score * 2

    def _calculate_hedge_cost(self, market: Market) -> float:
        """
        Calculate cost of the hedge.
        π = 1.00 - (p_YES + p_NO)

        Negative π = risk-free profit (rare).
        Positive π = cost of hedging.
        We want |π| < Δ_max.
        """
        if len(market.outcomes) < 2:
            return float('inf')

        p_yes = market.outcomes[0].price
        p_no = market.outcomes[1].price

        total = p_yes + p_no
        # Cost per hedge cycle: how much above $1.00 we pay
        hedge_cost = max(0, total - 1.0)

        # Factor in spread — we'll likely buy at ask prices
        if market.outcomes[0].book_ask and market.outcomes[1].book_ask:
            actual_cost = market.outcomes[0].book_ask + market.outcomes[1].book_ask
            hedge_cost = max(0, actual_cost - 1.0)

        return hedge_cost

    def _estimate_daily_reward(self, market: Market, score: float) -> float:
        """
        Estimate our share of the daily reward pool.
        R_i = (S_i / Σ_j S_j) * P

        We approximate Σ_j S_j based on competition level.
        """
        if score <= 0:
            return 0.0

        # Rough estimate: assume we're 1 of ~5-20 providers
        estimated_competitors = 10
        estimated_total_score = score * estimated_competitors
        our_share = score / estimated_total_score
        daily_reward = our_share * market.reward_pool

        return daily_reward

    def _kelly_fraction(
        self, market: Market, expected_reward: float, hedge_cost: float
    ) -> float:
        """
        Kelly criterion for position sizing.
        f* = (p * b - q) / b

        Where:
        - p = probability hedge completes (both sides fill within Δ_max)
        - q = 1 - p
        - b = net reward-to-risk ratio
        """
        if hedge_cost <= 0:
            hedge_cost = 0.001  # avoid division by zero

        # Estimate probability of successful hedge based on spread & volatility
        p = 0.7  # base assumption: 70% of hedges complete
        if market.spread < 0.01:
            p = 0.85  # tight spread = easier to hedge
        elif market.spread > 0.03:
            p = 0.5  # wide spread = harder

        b = expected_reward / hedge_cost if hedge_cost > 0 else 0
        q = 1 - p

        if b <= 0:
            return 0.0

        kelly = (p * b - q) / b
        # Cap at configured maximum
        return max(0, min(kelly, self.cfg.kelly_fraction_cap))
