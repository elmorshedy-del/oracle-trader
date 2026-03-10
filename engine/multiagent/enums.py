from __future__ import annotations

from enum import Enum


class MarketCategory(str, Enum):
    POLITICS = "politics"
    CRYPTO = "crypto"
    WEATHER = "weather"
    SPORTS = "sports"
    ENTERTAINMENT = "entertainment"
    ECONOMICS = "economics"
    OTHER = "other"


class SignalDirection(str, Enum):
    BUY_YES = "buy_yes"
    BUY_NO = "buy_no"
    SELL_YES = "sell_yes"
    SELL_NO = "sell_no"


class ExecutionMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class PositionStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    EXPIRED = "expired"


class ModuleStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"


class RejectionReason(str, Enum):
    VOLUME_BELOW_MINIMUM = "volume_below_minimum"
    LIQUIDITY_BELOW_MINIMUM = "liquidity_below_minimum"
    RESOLUTION_TOO_SOON = "resolution_too_soon"
    MARKET_TOO_NEW = "market_too_new"
    EDGE_BELOW_THRESHOLD = "edge_below_threshold"
    MISSING_REASONING = "missing_reasoning"
    MISSING_EVIDENCE = "missing_evidence"
    MISSING_EDGE_BASIS = "missing_edge_basis"
    DUPLICATE_POSITION = "duplicate_position"
    RECENT_SIGNAL_DUPLICATE = "recent_signal_duplicate"
    RECENT_HEADLINE_DUPLICATE = "recent_headline_duplicate"
    FAMILY_REENTRY_COOLDOWN = "family_reentry_cooldown"
    MAX_MARKETS_PER_CATEGORY = "max_markets_per_category"
    ENRICHMENT_STALE = "enrichment_stale"
    ENRICHMENT_MISSING_CRITICAL = "enrichment_missing_critical"
    VALIDATION_ERROR = "validation_error"


class AllocationRejectionReason(str, Enum):
    CAPITAL_INSUFFICIENT = "capital_insufficient"
    RISK_BUDGET_EXCEEDED = "risk_budget_exceeded"
    POSITION_SIZE_EXCEEDS_MAX = "position_size_exceeds_max"
    EXECUTION_SLIPPAGE_LIMIT = "execution_slippage_limit"
    STRATEGY_CAP_REACHED = "strategy_cap_reached"
    FAMILY_CAP_REACHED = "family_cap_reached"
    THEME_CAP_REACHED = "theme_cap_reached"
    CATEGORY_CAP_REACHED = "category_cap_reached"
    MARKET_CAP_REACHED = "market_cap_reached"
    RESERVE_VIOLATED = "reserve_violated"
    CORRELATION_LIMIT = "correlation_limit"
    REENTRY_COOLDOWN = "reentry_cooldown"
