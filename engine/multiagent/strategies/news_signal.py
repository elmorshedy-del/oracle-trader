from __future__ import annotations

from typing import Any

from ..contracts import MarketContext, PortfolioSnapshot, SignalCandidate
from ..enums import SignalDirection


class NewsSignalStrategy:
    name = "news_signal"

    def generate(
        self,
        markets: list[MarketContext],
        portfolio: PortfolioSnapshot,
        config: Any,
    ) -> list[SignalCandidate]:
        cfg = config or {}
        min_confidence = float(cfg.get("min_confidence", 0.58))
        min_edge = float(cfg.get("min_edge", 0.03))
        max_entry_price = float(cfg.get("max_entry_price", 0.75))
        candidates: list[SignalCandidate] = []

        for market in markets:
            enrichment = market.get_enrichment("news")
            if enrichment is None or enrichment.error:
                continue
            data = enrichment.data
            confidence = float(data.get("confidence", 0.0) or 0.0)
            impact_cents = float(data.get("expected_impact_cents", 0.0) or 0.0)
            direction = (data.get("direction") or "neutral").lower()
            if confidence < min_confidence or impact_cents <= 0:
                continue

            if direction == "bullish":
                current_price = market.yes_price
                fair = min(0.99, (current_price or 0.0) + (impact_cents / 100.0))
                trade_direction = SignalDirection.BUY_YES
                outcome = "YES"
            elif direction == "bearish":
                current_price = market.no_price
                fair = min(0.99, (current_price or 0.0) + (impact_cents / 100.0))
                trade_direction = SignalDirection.BUY_NO
                outcome = "NO"
            else:
                continue

            if current_price is None or current_price <= 0 or current_price >= max_entry_price:
                continue
            edge = fair - current_price
            if edge < min_edge:
                continue

            candidates.append(
                SignalCandidate(
                    market_id=market.market_id,
                    strategy_name=self.name,
                    direction=trade_direction,
                    outcome=outcome,
                    current_price=current_price,
                    estimated_fair_value=fair,
                    edge_estimate=edge,
                    edge_basis="news_llm_relevance",
                    reasoning=(
                        f"NEWS SIGNAL: {data.get('headline')} | {direction} on {data.get('market_slug')} | "
                        f"confidence {confidence:.0%} | impact {impact_cents:.1f}c"
                    ),
                    evidence=(
                        f"source={data.get('source')}",
                        f"llm_provider={data.get('llm_provider')}",
                        f"llm_model={data.get('llm_model')}",
                    ),
                    llm_involved=True,
                    market_snapshot=market,
                    metadata={
                        "headline": data.get("headline"),
                        "source": data.get("source"),
                        "llm_provider": data.get("llm_provider"),
                        "llm_model": data.get("llm_model"),
                    },
                )
            )

        candidates.sort(key=lambda item: item.edge_estimate, reverse=True)
        return candidates[: int(cfg.get("max_candidates_per_cycle", 4))]
