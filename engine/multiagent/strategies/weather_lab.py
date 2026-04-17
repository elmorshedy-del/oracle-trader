from __future__ import annotations

from typing import Any

from ..context import PipelineContext
from ..contracts import MarketContext, PortfolioSnapshot, SignalCandidate
from ..enums import SignalDirection


def _fee_buffer(price: float, base: float) -> float:
    return base * min(max(price, 0.01), 1.0 - max(price, 0.01))


class WeatherSniperStrategy:
    name = "weather_sniper"

    def generate(
        self,
        markets: list[MarketContext],
        portfolio: PortfolioSnapshot,
        context: PipelineContext,
        config: Any,
    ) -> list[SignalCandidate]:
        min_edge = float((config or {}).get("min_edge", 0.08))
        max_yes_price = float((config or {}).get("max_yes_price", 0.08))
        min_probability = float((config or {}).get("min_probability", 0.92))
        min_models = int((config or {}).get("min_models", 3))
        candidates: list[SignalCandidate] = []

        for market in markets:
            enrichment = market.get_enrichment("weather")
            if enrichment is None or enrichment.error:
                continue
            data = enrichment.data
            current_temps = data.get("current_temps", {})
            current_prob = float(data.get("current_prob", 0.0) or 0.0)
            yes_price = market.yes_price
            if yes_price is None:
                continue
            city = data.get("city")
            target_date = data.get("target_date")
            family_key = f"weather:sniper:{city}:{target_date}"
            if context.family_positions(family_key) > 0 or context.seen_family(family_key, 6.0):
                continue
            if len(current_temps) < min_models or current_prob < min_probability or yes_price > max_yes_price:
                continue
            edge = current_prob - yes_price - _fee_buffer(yes_price, float(data.get("fee_buffer", 0.012)))
            if edge < min_edge:
                continue
            candidates.append(
                SignalCandidate(
                    market_id=market.market_id,
                    strategy_name=self.name,
                    direction=SignalDirection.BUY_YES,
                    outcome="YES",
                    current_price=yes_price,
                    estimated_fair_value=min(0.99, current_prob),
                    edge_estimate=edge,
                    edge_basis="weather_consensus_sniper",
                    reasoning=(
                        f"WEATHER SNIPER: {data.get('city')} {data.get('target_date')} | "
                        f"consensus YES {current_prob:.1%} vs market YES {yes_price:.1%} | "
                        f"spread {float(data.get('current_spread_f', 0.0)):.1f}F"
                    ),
                    evidence=(
                        f"city={data.get('city')}",
                        f"target_date={data.get('target_date')}",
                        f"current_prob={current_prob:.4f}",
                    ),
                    market_snapshot=market,
                    metadata={
                        "variant": "sniper",
                        "city": city,
                        "target_date": target_date,
                        "family_key": family_key,
                        "theme_key": f"weather:{str(city).lower()}",
                    },
                )
            )
        return candidates


class WeatherLatencyStrategy:
    name = "weather_latency"

    def generate(
        self,
        markets: list[MarketContext],
        portfolio: PortfolioSnapshot,
        context: PipelineContext,
        config: Any,
    ) -> list[SignalCandidate]:
        min_edge = float((config or {}).get("min_edge", 0.04))
        min_shift = float((config or {}).get("min_probability_shift", 0.07))
        max_entry_price = float((config or {}).get("max_entry_price", 0.62))
        candidates: list[SignalCandidate] = []

        for market in markets:
            enrichment = market.get_enrichment("weather")
            if enrichment is None or enrichment.error:
                continue
            data = enrichment.data
            previous_prob = data.get("previous_prob")
            current_prob = float(data.get("current_prob", 0.0) or 0.0)
            if previous_prob is None:
                continue
            shift = current_prob - float(previous_prob)
            if abs(shift) < min_shift or not data.get("changed_models"):
                continue
            city = data.get("city")
            target_date = data.get("target_date")
            family_key = f"weather:latency:{city}:{target_date}"
            if context.family_positions(family_key) > 0 or context.seen_family(family_key, 3.0):
                continue

            if shift > 0:
                direction = SignalDirection.BUY_YES
                current_price = market.yes_price
                fair = current_prob
                outcome = "YES"
            else:
                direction = SignalDirection.BUY_NO
                current_price = market.no_price
                fair = 1.0 - current_prob
                outcome = "NO"
            if current_price is None or current_price > max_entry_price:
                continue
            edge = fair - current_price - _fee_buffer(current_price, float(data.get("fee_buffer", 0.012)))
            if edge < min_edge:
                continue
            candidates.append(
                SignalCandidate(
                    market_id=market.market_id,
                    strategy_name=self.name,
                    direction=direction,
                    outcome=outcome,
                    current_price=current_price,
                    estimated_fair_value=min(0.99, fair),
                    edge_estimate=edge,
                    edge_basis="weather_probability_shift",
                    reasoning=(
                        f"WEATHER LATENCY: {data.get('city')} {data.get('target_date')} | "
                        f"forecast moved {float(previous_prob):.1%}->{current_prob:.1%} while token is {current_price:.1%}"
                    ),
                    evidence=(
                        f"probability_shift={shift:.4f}",
                        f"changed_models={','.join(sorted((data.get('changed_models') or {}).keys()))}",
                    ),
                    market_snapshot=market,
                    metadata={
                        "variant": "latency",
                        "probability_shift": shift,
                        "city": city,
                        "target_date": target_date,
                        "family_key": family_key,
                        "theme_key": f"weather:{str(city).lower()}",
                    },
                )
            )
        return candidates


class WeatherSwingStrategy:
    name = "weather_swing"

    def generate(
        self,
        markets: list[MarketContext],
        portfolio: PortfolioSnapshot,
        context: PipelineContext,
        config: Any,
    ) -> list[SignalCandidate]:
        min_edge = float((config or {}).get("min_edge", 0.04))
        swing_min_prob = float((config or {}).get("swing_min_prob", 0.58))
        swing_max_prob = float((config or {}).get("swing_max_prob", 0.42))
        min_token_dip = float((config or {}).get("min_token_dip", 0.05))
        min_history_points = int((config or {}).get("min_history_points", 3))
        candidates: list[SignalCandidate] = []

        for market in markets:
            enrichment = market.get_enrichment("weather")
            if enrichment is None or enrichment.error:
                continue
            data = enrichment.data
            current_prob = float(data.get("current_prob", 0.0) or 0.0)
            previous_prob = data.get("previous_prob")
            if previous_prob is not None and abs(current_prob - float(previous_prob)) > 0.08:
                continue

            if current_prob >= swing_min_prob:
                side = "YES"
                current_price = market.yes_price
                fair = current_prob
                recent_peak = data.get("recent_yes_peak")
                history_points = int(data.get("yes_history_points", 0) or 0)
                direction = SignalDirection.BUY_YES
            elif current_prob <= swing_max_prob:
                side = "NO"
                current_price = market.no_price
                fair = 1.0 - current_prob
                recent_peak = data.get("recent_no_peak")
                history_points = int(data.get("no_history_points", 0) or 0)
                direction = SignalDirection.BUY_NO
            else:
                continue

            if current_price is None or recent_peak is None or history_points < min_history_points:
                continue
            city = data.get("city")
            target_date = data.get("target_date")
            family_key = f"weather:swing:{city}:{target_date}:{side}"
            if context.family_positions(family_key) > 0 or context.seen_family(family_key, 3.0):
                continue
            dip = float(recent_peak) - current_price
            if dip < min_token_dip:
                continue
            edge = fair - current_price - _fee_buffer(current_price, float(data.get("fee_buffer", 0.012)))
            if edge < min_edge:
                continue
            candidates.append(
                SignalCandidate(
                    market_id=market.market_id,
                    strategy_name=self.name,
                    direction=direction,
                    outcome=side,
                    current_price=current_price,
                    estimated_fair_value=min(0.99, fair),
                    edge_estimate=edge,
                    edge_basis="weather_price_dip_reversion",
                    reasoning=(
                        f"WEATHER SWING: {data.get('city')} {data.get('target_date')} | "
                        f"{side} dipped from {float(recent_peak):.1%} to {current_price:.1%} while fair stays {fair:.1%}"
                    ),
                    evidence=(
                        f"history_points={history_points}",
                        f"dip={dip:.4f}",
                    ),
                    market_snapshot=market,
                    metadata={
                        "variant": "swing",
                        "dip": dip,
                        "city": city,
                        "target_date": target_date,
                        "family_key": family_key,
                        "theme_key": f"weather:{str(city).lower()}",
                    },
                )
            )
        return candidates
