from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

from ..context import PipelineContext
from ..contracts import MarketContext, PortfolioSnapshot, SignalCandidate
from ..enums import MarketCategory, SignalDirection


QUESTION_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")
UPPER_BARRIER_PATTERNS = (
    re.compile(r"(?:hit|reach|above)\s*\$?([\d,.]+)\s*([km]?)", re.IGNORECASE),
)
LOWER_BARRIER_PATTERNS = (
    re.compile(r"(?:dip\s+to|below)\s*\$?([\d,.]+)\s*([km]?)", re.IGNORECASE),
)
EXPIRY_SLUG_RE = re.compile(r"by-([a-z]+)-(\d{1,2})-(\d{4})")
CRYPTO_SYMBOL_PATTERNS = {
    "BTC": re.compile(r"\b(?:btc|bitcoin)\b", re.IGNORECASE),
    "ETH": re.compile(r"\b(?:eth|ethereum)\b", re.IGNORECASE),
    "SOL": re.compile(r"\b(?:sol|solana)\b", re.IGNORECASE),
}
ALL_TIME_HIGH_USD = {
    "BTC": 109358.0,
    "ETH": 4891.7,
    "SOL": 294.33,
}
ATH_IMPLICATION_BUFFER_USD = 250.0


@dataclass(frozen=True)
class ParsedBarrier:
    market: MarketContext
    symbol: str
    kind: str
    expiry: str
    barrier_price: float
    yes_price: float


class RelationshipArbitrageStrategy:
    name = "relationship_arbitrage"

    def generate(
        self,
        markets: list[MarketContext],
        portfolio: PortfolioSnapshot,
        context: PipelineContext,
        config: Any,
    ) -> list[SignalCandidate]:
        cfg = config or {}
        candidates: list[SignalCandidate] = []

        candidates.extend(self._build_duplicate_candidates(markets, context, cfg))
        candidates.extend(self._build_crypto_structure_candidates(markets, context, cfg))

        best_by_market: dict[str, SignalCandidate] = {}
        for candidate in candidates:
            existing = best_by_market.get(candidate.market_id)
            if existing is None or candidate.edge_estimate > existing.edge_estimate:
                best_by_market[candidate.market_id] = candidate

        ordered = sorted(
            best_by_market.values(),
            key=lambda item: (item.edge_estimate, item.market_snapshot.volume_24h if item.market_snapshot else 0.0),
            reverse=True,
        )
        max_candidates = int(cfg.get("max_candidates_per_cycle", 8))
        fresh = [item for item in ordered if not context.has_open_market(item.market_id)]
        if fresh:
            return fresh[:max_candidates]
        return ordered[:max_candidates]

    def _build_duplicate_candidates(
        self,
        markets: list[MarketContext],
        context: PipelineContext,
        cfg: dict[str, Any],
    ) -> list[SignalCandidate]:
        min_edge = float(cfg.get("duplicate_min_edge", 0.045))
        max_entry = float(cfg.get("max_entry_price", 0.82))
        min_volume = float(cfg.get("min_volume_focus", 10000.0))

        groups: dict[str, list[MarketContext]] = {}
        for market in markets:
            if market.yes_price is None or market.yes_price <= 0 or market.yes_price >= max_entry:
                continue
            if market.volume_24h < min_volume or len(market.outcomes) < 2:
                continue
            key = _canonical_question_key(market.question)
            groups.setdefault(key, []).append(market)

        candidates: list[SignalCandidate] = []
        for key, items in groups.items():
            if len(items) < 2:
                continue

            ordered = sorted(items, key=lambda item: item.yes_price or 0.0)
            cheapest = ordered[0]
            richest = ordered[-1]
            cheap_yes = cheapest.yes_price or 0.0
            rich_yes = richest.yes_price or 0.0
            edge = rich_yes - cheap_yes
            if edge < min_edge:
                continue
            family_key = f"duplicate:{key}"
            if context.family_positions(family_key) > 0 or context.seen_family(family_key, 3.0):
                continue

            candidates.append(
                SignalCandidate(
                    market_id=cheapest.market_id,
                    strategy_name=self.name,
                    direction=SignalDirection.BUY_YES,
                    outcome="YES",
                    current_price=cheap_yes,
                    estimated_fair_value=min(0.99, rich_yes),
                    edge_estimate=edge,
                    edge_basis="equivalence_duplicate_gap",
                    reasoning=(
                        "Equivalent markets should clear near the same YES price. "
                        f"The cheapest copy is trading at {cheap_yes:.1%} while the richest is {rich_yes:.1%}."
                    ),
                    evidence=(
                        f"canonical_key={key}",
                        f"cheap_market={cheapest.market_id}:{cheap_yes:.3f}",
                        f"rich_market={richest.market_id}:{rich_yes:.3f}",
                    ),
                    market_snapshot=cheapest,
                    metadata={
                        "opportunity_type": "duplicate_equivalence",
                        "family_key": family_key,
                        "theme_key": f"equivalence:{key}",
                        "peer_market_id": richest.market_id,
                        "peer_price": rich_yes,
                    },
                )
            )

        return candidates

    def _build_crypto_structure_candidates(
        self,
        markets: list[MarketContext],
        context: PipelineContext,
        cfg: dict[str, Any],
    ) -> list[SignalCandidate]:
        parsed = [item for item in (_parse_barrier_market(market) for market in markets) if item is not None]
        if not parsed:
            return []

        min_ladder_edge = float(cfg.get("ladder_min_edge", 0.04))
        min_imp_edge = float(cfg.get("implication_min_edge", 0.05))
        max_entry = float(cfg.get("max_entry_price", 0.82))

        buckets: dict[tuple[str, str, str], list[ParsedBarrier]] = {}
        for item in parsed:
            if item.yes_price <= 0 or item.yes_price >= max_entry:
                continue
            buckets.setdefault((item.symbol, item.kind, item.expiry), []).append(item)

        candidates: list[SignalCandidate] = []
        for (symbol, kind, expiry), items in buckets.items():
            items.sort(key=lambda item: item.barrier_price)
            candidates.extend(
                self._build_duplicate_barrier_candidates(symbol, kind, expiry, items, context, cfg)
            )

            if kind == "reach":
                for easier, harder in zip(items, items[1:]):
                    violation = harder.yes_price - easier.yes_price
                    if violation < min_ladder_edge:
                        continue
                    candidates.append(
                        self._candidate_from_market(
                            market=easier.market,
                            current=easier.yes_price,
                            fair=min(0.99, harder.yes_price),
                            edge=violation,
                            basis="ladder_monotonicity_break",
                            reasoning=(
                                f"{symbol} reach ladder by {expiry} is inverted: the harder "
                                f"${harder.barrier_price:,.0f} market is {harder.yes_price:.1%} while "
                                f"the easier ${easier.barrier_price:,.0f} market is only {easier.yes_price:.1%}."
                            ),
                            evidence=(
                                f"easier={easier.market.market_id}:{easier.yes_price:.3f}",
                                f"harder={harder.market.market_id}:{harder.yes_price:.3f}",
                            ),
                            metadata={
                                "opportunity_type": "crypto_ladder",
                                "symbol": symbol,
                                "expiry": expiry,
                                "family_key": f"ladder:{symbol}:{expiry}",
                                "theme_key": f"crypto:{symbol.lower()}",
                                "harder_market_id": harder.market.market_id,
                            },
                        )
                    )
            elif kind == "dip":
                for harder, easier in zip(items, items[1:]):
                    violation = harder.yes_price - easier.yes_price
                    if violation < min_ladder_edge:
                        continue
                    candidates.append(
                        self._candidate_from_market(
                            market=easier.market,
                            current=easier.yes_price,
                            fair=min(0.99, harder.yes_price),
                            edge=violation,
                            basis="dip_ladder_monotonicity_break",
                            reasoning=(
                                f"{symbol} dip ladder by {expiry} is inverted: the harder "
                                f"${harder.barrier_price:,.0f} dip trades {harder.yes_price:.1%} while "
                                f"the easier ${easier.barrier_price:,.0f} dip trades {easier.yes_price:.1%}."
                            ),
                            evidence=(
                                f"easier={easier.market.market_id}:{easier.yes_price:.3f}",
                                f"harder={harder.market.market_id}:{harder.yes_price:.3f}",
                            ),
                            metadata={
                                "opportunity_type": "crypto_dip_ladder",
                                "symbol": symbol,
                                "expiry": expiry,
                                "family_key": f"dip_ladder:{symbol}:{expiry}",
                                "theme_key": f"crypto:{symbol.lower()}",
                                "harder_market_id": harder.market.market_id,
                            },
                        )
                    )

        for symbol, ath_value in ALL_TIME_HIGH_USD.items():
            for (bucket_symbol, kind, expiry), ath_items in buckets.items():
                if bucket_symbol != symbol or kind != "ath" or not ath_items:
                    continue
                ath_market = ath_items[0]
                for reach in buckets.get((symbol, "reach", expiry), []):
                    if reach.barrier_price < ath_value + ATH_IMPLICATION_BUFFER_USD:
                        continue
                    violation = reach.yes_price - ath_market.yes_price
                    if violation < min_imp_edge:
                        continue
                    candidates.append(
                        self._candidate_from_market(
                            market=ath_market.market,
                            current=ath_market.yes_price,
                            fair=min(0.99, reach.yes_price),
                            edge=violation,
                            basis="ath_implication_break",
                            reasoning=(
                                f"{symbol} reaching ${reach.barrier_price:,.0f} by {expiry} implies a new all-time high first, "
                                f"but the reach market trades {reach.yes_price:.1%} while ATH trades only {ath_market.yes_price:.1%}."
                            ),
                            evidence=(
                                f"ath_market={ath_market.market.market_id}:{ath_market.yes_price:.3f}",
                                f"reach_market={reach.market.market_id}:{reach.yes_price:.3f}",
                            ),
                            metadata={
                                "opportunity_type": "ath_implication",
                                "symbol": symbol,
                                "expiry": expiry,
                                "family_key": f"ath_implication:{symbol}:{expiry}",
                                "theme_key": f"crypto:{symbol.lower()}",
                                "peer_market_id": reach.market.market_id,
                            },
                        )
                    )

        return candidates

    def _build_duplicate_barrier_candidates(
        self,
        symbol: str,
        kind: str,
        expiry: str,
        items: list[ParsedBarrier],
        context: PipelineContext,
        cfg: dict[str, Any],
    ) -> list[SignalCandidate]:
        min_edge = float(cfg.get("duplicate_min_edge", 0.045))
        grouped: dict[int, list[ParsedBarrier]] = {}
        for item in items:
            grouped.setdefault(int(round(item.barrier_price)), []).append(item)

        candidates: list[SignalCandidate] = []
        for barrier, dupes in grouped.items():
            if len(dupes) < 2:
                continue
            ordered = sorted(dupes, key=lambda item: item.yes_price)
            cheapest = ordered[0]
            richest = ordered[-1]
            edge = richest.yes_price - cheapest.yes_price
            if edge < min_edge:
                continue
            family_key = f"crypto_duplicate:{symbol}:{kind}:{expiry}:{barrier}"
            if context.family_positions(family_key) > 0 or context.seen_family(family_key, 3.0):
                continue
            candidates.append(
                self._candidate_from_market(
                    market=cheapest.market,
                    current=cheapest.yes_price,
                    fair=min(0.99, richest.yes_price),
                    edge=edge,
                    basis="barrier_duplicate_gap",
                    reasoning=(
                        f"Equivalent {symbol} {kind} ${barrier:,.0f} markets by {expiry} disagree materially: "
                        f"{cheapest.yes_price:.1%} vs {richest.yes_price:.1%}."
                    ),
                    evidence=(
                        f"cheap_market={cheapest.market.market_id}:{cheapest.yes_price:.3f}",
                        f"rich_market={richest.market.market_id}:{richest.yes_price:.3f}",
                    ),
                    metadata={
                        "opportunity_type": "crypto_duplicate",
                        "symbol": symbol,
                        "expiry": expiry,
                        "family_key": family_key,
                        "theme_key": f"crypto:{symbol.lower()}",
                        "peer_market_id": richest.market.market_id,
                    },
                )
            )
        return candidates

    def _candidate_from_market(
        self,
        *,
        market: MarketContext,
        current: float,
        fair: float,
        edge: float,
        basis: str,
        reasoning: str,
        evidence: tuple[str, ...],
        metadata: dict[str, Any],
    ) -> SignalCandidate:
        return SignalCandidate(
            market_id=market.market_id,
            strategy_name=self.name,
            direction=SignalDirection.BUY_YES,
            outcome="YES",
            current_price=current,
            estimated_fair_value=fair,
            edge_estimate=edge,
            edge_basis=basis,
            reasoning=reasoning,
            evidence=evidence,
            market_snapshot=market,
            metadata=metadata,
        )


def _canonical_question_key(question: str) -> str:
    normalized = question.lower()
    normalized = normalized.replace("btc", "bitcoin").replace("eth", "ethereum").replace("sol", "solana")
    normalized = QUESTION_NORMALIZE_RE.sub(" ", normalized).strip()
    return " ".join(normalized.split())


def _parse_barrier_market(market: MarketContext) -> ParsedBarrier | None:
    if market.category != MarketCategory.CRYPTO:
        return None

    text = f"{market.question} {_slug_from_market(market)}"
    symbol = _extract_symbol(text)
    if symbol is None or market.yes_price is None:
        return None

    expiry = _extract_expiry(market)
    if expiry is None:
        return None

    lowered = text.lower()
    if "all-time-high" in lowered or "all time high" in lowered or "ath" in lowered:
        return ParsedBarrier(
            market=market,
            symbol=symbol,
            kind="ath",
            expiry=expiry,
            barrier_price=ALL_TIME_HIGH_USD.get(symbol, 0.0),
            yes_price=market.yes_price,
        )

    for pattern in UPPER_BARRIER_PATTERNS:
        match = pattern.search(text)
        if match:
            return ParsedBarrier(
                market=market,
                symbol=symbol,
                kind="reach",
                expiry=expiry,
                barrier_price=_parse_barrier_value(match.group(1), match.group(2)),
                yes_price=market.yes_price,
            )

    for pattern in LOWER_BARRIER_PATTERNS:
        match = pattern.search(text)
        if match:
            return ParsedBarrier(
                market=market,
                symbol=symbol,
                kind="dip",
                expiry=expiry,
                barrier_price=_parse_barrier_value(match.group(1), match.group(2)),
                yes_price=market.yes_price,
            )

    return None


def _extract_symbol(text: str) -> str | None:
    for symbol, pattern in CRYPTO_SYMBOL_PATTERNS.items():
        if pattern.search(text):
            return symbol
    return None


def _extract_expiry(market: MarketContext) -> str | None:
    slug = _slug_from_market(market)
    match = EXPIRY_SLUG_RE.search(slug)
    if match:
        month, day, year = match.groups()
        return f"{month}-{day}-{year}".lower()
    return None


def _slug_from_market(market: MarketContext) -> str:
    return market.source_url.rstrip("/").rsplit("/", 1)[-1].lower()


def _parse_barrier_value(number: str, suffix: str) -> float:
    value = float(number.replace(",", ""))
    suffix = suffix.lower()
    if suffix == "k":
        value *= 1000
    elif suffix == "m":
        value *= 1_000_000
    return value
