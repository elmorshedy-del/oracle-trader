from __future__ import annotations

from typing import Any

from strategies.crypto_arb import MAX_SIGNAL_CONFIDENCE, TEMPORAL_CONFIDENCE_SCALE

from ..contracts import MarketContext, PortfolioSnapshot, SignalCandidate
from ..enums import SignalDirection


class CryptoLatencyStrategy:
    name = "crypto_latency"

    def generate(
        self,
        markets: list[MarketContext],
        portfolio: PortfolioSnapshot,
        config: Any,
    ) -> list[SignalCandidate]:
        cfg = config or {}
        temporal_min_move = float(cfg.get("temporal_min_move_pct", 0.003))
        temporal_max_entry = float(cfg.get("temporal_max_entry_price", 0.75))
        barrier_min_edge = float(cfg.get("barrier_min_edge", 0.04))
        max_entry = float(cfg.get("max_entry_price", 0.80))

        candidates: list[SignalCandidate] = []
        for market in markets:
            enrichment = market.get_enrichment("crypto")
            if enrichment is None or enrichment.error:
                continue
            data = enrichment.data
            kind = data.get("kind")
            move_pct = float(data.get("move_pct", 0.0) or 0.0)
            move_direction = data.get("move_direction", "flat")
            symbol = data.get("symbol")

            if kind == "temporal":
                if abs(move_pct) < temporal_min_move:
                    continue
                if move_direction == "up":
                    current_price = float(data.get("up_price", 0.0) or 0.0)
                    fair = min(0.99, max(current_price, 0.01) + abs(move_pct))
                    direction = SignalDirection.BUY_YES
                    outcome = "YES"
                else:
                    current_price = float(data.get("down_price", 0.0) or 0.0)
                    fair = min(0.99, max(current_price, 0.01) + abs(move_pct))
                    direction = SignalDirection.BUY_NO
                    outcome = "NO"
                if current_price <= 0 or current_price >= temporal_max_entry:
                    continue
                edge = fair - current_price
                confidence = min(abs(move_pct) / TEMPORAL_CONFIDENCE_SCALE, MAX_SIGNAL_CONFIDENCE)
                candidates.append(
                    SignalCandidate(
                        market_id=market.market_id,
                        strategy_name=self.name,
                        direction=direction,
                        outcome=outcome,
                        current_price=current_price,
                        estimated_fair_value=fair,
                        edge_estimate=edge,
                        edge_basis="crypto_spot_latency",
                        reasoning=(
                            f"CRYPTO LATENCY: {symbol} moved {move_pct:+.2%} and the temporal market "
                            f"has not fully absorbed the move."
                        ),
                        evidence=(
                            f"move_pct={move_pct:.4f}",
                            f"move_direction={move_direction}",
                        ),
                        llm_involved=False,
                        market_snapshot=market,
                        metadata={"confidence_hint": confidence, "symbol": symbol, "kind": kind},
                    )
                )
                continue

            modeled_yes = data.get("modeled_yes")
            if modeled_yes is None:
                continue
            yes_price = float(data.get("yes_price", 0.0) or 0.0)
            no_price = float(data.get("no_price", 0.0) or 0.0)
            modeled_yes = float(modeled_yes)
            if modeled_yes - yes_price >= barrier_min_edge and yes_price < max_entry:
                candidates.append(
                    SignalCandidate(
                        market_id=market.market_id,
                        strategy_name=self.name,
                        direction=SignalDirection.BUY_YES,
                        outcome="YES",
                        current_price=yes_price,
                        estimated_fair_value=min(0.99, modeled_yes),
                        edge_estimate=modeled_yes - yes_price,
                        edge_basis="crypto_barrier_latency",
                        reasoning=(
                            f"CRYPTO LATENCY: {symbol} spot ${float(data.get('spot_price', 0.0)):,.2f} "
                            f"supports YES fair {modeled_yes:.1%} vs market YES {yes_price:.1%}."
                        ),
                        evidence=(
                            f"kind={kind}",
                            f"barrier_price={float(data.get('barrier_price', 0.0) or 0.0):.0f}",
                        ),
                        market_snapshot=market,
                        metadata={"symbol": symbol, "kind": kind},
                    )
                )
            elif yes_price - modeled_yes >= barrier_min_edge and no_price < max_entry:
                candidates.append(
                    SignalCandidate(
                        market_id=market.market_id,
                        strategy_name=self.name,
                        direction=SignalDirection.BUY_NO,
                        outcome="NO",
                        current_price=no_price,
                        estimated_fair_value=min(0.99, 1.0 - modeled_yes),
                        edge_estimate=yes_price - modeled_yes,
                        edge_basis="crypto_barrier_latency",
                        reasoning=(
                            f"CRYPTO LATENCY: {symbol} spot ${float(data.get('spot_price', 0.0)):,.2f} "
                            f"supports NO fair {1.0 - modeled_yes:.1%} vs market NO {no_price:.1%}."
                        ),
                        evidence=(
                            f"kind={kind}",
                            f"barrier_price={float(data.get('barrier_price', 0.0) or 0.0):.0f}",
                        ),
                        market_snapshot=market,
                        metadata={"symbol": symbol, "kind": kind},
                    )
                )

        candidates.sort(key=lambda item: item.edge_estimate, reverse=True)
        return candidates[: int(cfg.get("max_candidates_per_cycle", 8))]
