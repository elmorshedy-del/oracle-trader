from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from data.models import Market, Outcome as LegacyOutcome, Portfolio as LegacyPortfolio, Position as LegacyPosition

from .config import RiskLimits
from .contracts import MarketContext, NormalizedMarket, Outcome, PortfolioSnapshot, PositionState
from .enums import MarketCategory, PositionStatus, SignalDirection


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def categorize_market(question: str, tags: Iterable[str]) -> MarketCategory:
    haystack = f"{question} {' '.join(tags)}".lower()
    if any(token in haystack for token in ("temperature", "forecast", "weather", "rain", "snow", "wind")):
        return MarketCategory.WEATHER
    if any(token in haystack for token in ("bitcoin", "btc", "ethereum", "eth", "solana", "sol")):
        return MarketCategory.CRYPTO
    if any(token in haystack for token in ("election", "president", "trump", "biden", "senate", "government")):
        return MarketCategory.POLITICS
    if any(token in haystack for token in ("nba", "nfl", "mlb", "score", "match", "goal", "championship")):
        return MarketCategory.SPORTS
    if any(token in haystack for token in ("gdp", "inflation", "fed", "recession", "rates")):
        return MarketCategory.ECONOMICS
    if any(token in haystack for token in ("album", "movie", "oscar", "grammy", "celebrity", "gta")):
        return MarketCategory.ENTERTAINMENT
    return MarketCategory.OTHER


def normalized_outcomes_from_market(market: Market) -> tuple[Outcome, ...]:
    return tuple(
        Outcome(
            name=outcome.name,
            token_id=outcome.token_id,
            current_price=outcome.price,
        )
        for outcome in market.outcomes
    )


def normalized_market_from_legacy(market: Market) -> NormalizedMarket:
    return NormalizedMarket(
        market_id=market.condition_id,
        question=market.question,
        category=categorize_market(market.question, market.tags),
        outcomes=normalized_outcomes_from_market(market),
        volume_24h=market.volume_24h,
        total_volume=market.volume_total,
        liquidity=market.liquidity,
        created_date=market.fetched_at,
        source_url=f"https://polymarket.com/event/{market.slug}",
        scanned_at=utc_now(),
        resolution_date=_parse_iso_datetime(market.end_date),
        description="",
        tags=tuple(market.tags),
        end_date_source="api" if market.end_date else "unknown",
        raw_metadata={
            "spread": market.spread,
            "midpoint": market.midpoint,
            "reward_pool": market.reward_pool,
            "active": market.active,
            "closed": market.closed,
        },
    )


def bare_context_from_legacy(market: Market) -> MarketContext:
    normalized = normalized_market_from_legacy(market)
    return MarketContext(
        market_id=normalized.market_id,
        question=normalized.question,
        category=normalized.category,
        outcomes=normalized.outcomes,
        volume_24h=normalized.volume_24h,
        total_volume=normalized.total_volume,
        liquidity=normalized.liquidity,
        created_date=normalized.created_date,
        source_url=normalized.source_url,
        resolution_date=normalized.resolution_date,
        description=normalized.description,
        tags=normalized.tags,
        enrichments={},
        enrichment_completeness=0.0,
    )


def portfolio_snapshot_from_legacy(
    portfolio: LegacyPortfolio,
    risk_limits: RiskLimits | None = None,
) -> PortfolioSnapshot:
    total_capital = portfolio.total_value
    reserved_capital = total_capital * (risk_limits.min_reserve_pct if risk_limits else 0.0)
    positions = tuple(position_state_from_legacy(position) for position in portfolio.positions)
    exposure_by_strategy: dict[str, float] = {}
    exposure_by_category: dict[str, float] = {}
    positions_by_strategy: dict[str, int] = {}
    open_market_ids: set[str] = set()

    for position in positions:
        market_value = position.current_price * position.shares
        exposure_by_strategy[position.strategy_name] = (
            exposure_by_strategy.get(position.strategy_name, 0.0) + market_value
        )
        exposure_by_category[position.category.value] = (
            exposure_by_category.get(position.category.value, 0.0) + market_value
        )
        positions_by_strategy[position.strategy_name] = (
            positions_by_strategy.get(position.strategy_name, 0) + 1
        )
        open_market_ids.add(position.market_id)

    return PortfolioSnapshot(
        total_capital=total_capital,
        available_capital=portfolio.cash,
        deployed_capital=portfolio.positions_value,
        reserved_capital=reserved_capital,
        total_unrealized_pnl=sum(position.unrealized_pnl for position in positions),
        total_realized_pnl=portfolio.total_realized_pnl,
        positions=positions,
        position_count=len(positions),
        capital_utilization_pct=(portfolio.positions_value / total_capital) if total_capital else 0.0,
        open_market_ids=frozenset(open_market_ids),
        exposure_by_strategy=exposure_by_strategy,
        exposure_by_category=exposure_by_category,
        positions_by_strategy=positions_by_strategy,
        recently_closed={},
    )


def position_state_from_legacy(position: LegacyPosition) -> PositionState:
    direction = SignalDirection.BUY_NO if position.side.upper() == "NO" else SignalDirection.BUY_YES
    return PositionState(
        position_id=f"{position.condition_id}:{position.token_id}",
        market_id=position.condition_id,
        market_question=position.market_slug,
        strategy_name=position.source.value,
        category=categorize_market(position.market_slug, []),
        direction=direction,
        outcome=position.side,
        entry_price=position.avg_entry_price,
        current_price=position.current_price,
        shares=position.shares,
        cost_basis=position.avg_entry_price * position.shares,
        unrealized_pnl=position.unrealized_pnl,
        realized_pnl=0.0,
        status=PositionStatus.OPEN,
        opened_at=position.opened_at,
        signal_id=f"legacy:{position.condition_id}:{position.token_id}",
    )


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value.replace("Z", "+00:00")
        return datetime.fromisoformat(value)
    except ValueError:
        return None
