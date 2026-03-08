from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import uuid

from .enums import (
    AllocationRejectionReason,
    ExecutionMode,
    MarketCategory,
    ModuleStatus,
    PositionStatus,
    RejectionReason,
    SignalDirection,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class RawMarket:
    data: dict[str, Any]
    fetched_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class Outcome:
    name: str
    token_id: str
    current_price: float


@dataclass(frozen=True)
class NormalizedMarket:
    market_id: str
    question: str
    category: MarketCategory
    outcomes: tuple[Outcome, ...]
    volume_24h: float
    total_volume: float
    liquidity: float
    created_date: datetime
    source_url: str
    scanned_at: datetime
    resolution_date: Optional[datetime] = None
    description: str = ""
    tags: tuple[str, ...] = ()
    end_date_source: str = ""
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def hours_to_resolution(self) -> Optional[float]:
        if self.resolution_date is None:
            return None
        delta = self.resolution_date - utc_now()
        return max(0.0, delta.total_seconds() / 3600)

    @property
    def yes_price(self) -> Optional[float]:
        for outcome in self.outcomes:
            if outcome.name.lower() in ("yes", "true"):
                return outcome.current_price
        return self.outcomes[0].current_price if self.outcomes else None

    @property
    def no_price(self) -> Optional[float]:
        for outcome in self.outcomes:
            if outcome.name.lower() in ("no", "false"):
                return outcome.current_price
        return self.outcomes[1].current_price if len(self.outcomes) > 1 else None


@dataclass(frozen=True)
class EnrichmentResult:
    provider_name: str
    data: dict[str, Any]
    llm_assisted: bool = False
    llm_model: Optional[str] = None
    llm_prompt_hash: Optional[str] = None
    fetched_at: datetime = field(default_factory=utc_now)
    is_stale: bool = False
    staleness_seconds: float = 0.0
    error: Optional[str] = None


@dataclass(frozen=True)
class MarketContext:
    market_id: str
    question: str
    category: MarketCategory
    outcomes: tuple[Outcome, ...]
    volume_24h: float
    total_volume: float
    liquidity: float
    created_date: datetime
    source_url: str
    resolution_date: Optional[datetime] = None
    description: str = ""
    tags: tuple[str, ...] = ()
    enrichments: dict[str, EnrichmentResult] = field(default_factory=dict)
    related_market_ids: tuple[str, ...] = ()
    enrichment_timestamp: datetime = field(default_factory=utc_now)
    enrichment_completeness: float = 0.0

    def get_enrichment(self, provider: str) -> Optional[EnrichmentResult]:
        return self.enrichments.get(provider)

    def has_enrichment(self, provider: str) -> bool:
        enrichment = self.enrichments.get(provider)
        return enrichment is not None and enrichment.error is None

    @property
    def hours_to_resolution(self) -> Optional[float]:
        if self.resolution_date is None:
            return None
        delta = self.resolution_date - utc_now()
        return max(0.0, delta.total_seconds() / 3600)

    @property
    def yes_price(self) -> Optional[float]:
        for outcome in self.outcomes:
            if outcome.name.lower() in ("yes", "true"):
                return outcome.current_price
        return self.outcomes[0].current_price if self.outcomes else None

    @property
    def no_price(self) -> Optional[float]:
        for outcome in self.outcomes:
            if outcome.name.lower() in ("no", "false"):
                return outcome.current_price
        return self.outcomes[1].current_price if len(self.outcomes) > 1 else None


@dataclass(frozen=True)
class SignalCandidate:
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    market_id: str = ""
    strategy_name: str = ""
    direction: SignalDirection = SignalDirection.BUY_YES
    outcome: str = ""
    current_price: float = 0.0
    estimated_fair_value: float = 0.0
    edge_estimate: float = 0.0
    edge_basis: str = ""
    reasoning: str = ""
    evidence: tuple[str, ...] = ()
    llm_involved: bool = False
    market_snapshot: Optional[MarketContext] = None
    generated_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationCheck:
    rule_name: str
    passed: bool
    blocking: bool = True
    threshold: Optional[float] = None
    actual_value: Optional[float] = None
    reason: str = ""
    rejection_code: Optional[RejectionReason] = None


@dataclass(frozen=True)
class ValidatedSignal:
    signal: SignalCandidate
    checks: tuple[ValidationCheck, ...]
    warnings: tuple[str, ...] = ()
    validated_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class RejectedSignal:
    signal: SignalCandidate
    checks: tuple[ValidationCheck, ...]
    blocking_rules: tuple[str, ...]
    rejection_codes: tuple[RejectionReason, ...]
    rejected_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class ExecutionIntent:
    intent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    signal: Optional[ValidatedSignal] = None
    position_size_usd: float = 0.0
    estimated_shares: float = 0.0
    max_slippage_pct: float = 0.02
    order_type: str = "market"
    limit_price: Optional[float] = None
    risk_budget_consumed_pct: float = 0.0
    portfolio_weight_pct: float = 0.0
    allocation_priority: int = 0
    allocated_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class AllocationRejection:
    signal: ValidatedSignal
    reason: AllocationRejectionReason
    constraint_name: str
    constraint_limit: float
    current_utilization: float
    shortfall: float = 0.0
    rejected_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class ExecutionResult:
    result_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    intent: Optional[ExecutionIntent] = None
    executed: bool = False
    execution_mode: ExecutionMode = ExecutionMode.PAPER
    fill_price: Optional[float] = None
    shares_filled: float = 0.0
    slippage_pct: float = 0.0
    fees: float = 0.0
    error: Optional[str] = None
    executed_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class MarketResolution:
    market_id: str
    resolved_outcome: str
    resolution_price: float
    resolved_at: datetime = field(default_factory=utc_now)


@dataclass
class PositionState:
    position_id: str
    market_id: str
    market_question: str
    strategy_name: str
    category: MarketCategory
    direction: SignalDirection
    outcome: str
    entry_price: float
    current_price: float
    shares: float
    cost_basis: float
    unrealized_pnl: float
    realized_pnl: float
    status: PositionStatus
    opened_at: datetime
    signal_id: str
    closed_at: Optional[datetime] = None
    close_reason: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class PortfolioSnapshot:
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    total_capital: float = 0.0
    available_capital: float = 0.0
    deployed_capital: float = 0.0
    reserved_capital: float = 0.0
    total_unrealized_pnl: float = 0.0
    total_realized_pnl: float = 0.0
    positions: tuple[PositionState, ...] = ()
    position_count: int = 0
    capital_utilization_pct: float = 0.0
    snapshot_at: datetime = field(default_factory=utc_now)
    open_market_ids: frozenset[str] = field(default_factory=frozenset)
    exposure_by_strategy: dict[str, float] = field(default_factory=dict)
    exposure_by_category: dict[str, float] = field(default_factory=dict)
    positions_by_strategy: dict[str, int] = field(default_factory=dict)
    recently_closed: dict[str, datetime] = field(default_factory=dict)

    def has_position_in(self, market_id: str) -> bool:
        return market_id in self.open_market_ids

    def strategy_capital_deployed(self, strategy: str) -> float:
        return self.exposure_by_strategy.get(strategy, 0.0)

    def category_capital_deployed(self, category: str) -> float:
        return self.exposure_by_category.get(category, 0.0)


@dataclass(frozen=True)
class ModuleHealth:
    module_name: str
    status: ModuleStatus
    last_run_at: Optional[datetime] = None
    last_duration_seconds: float = 0.0
    last_error: Optional[str] = None
    items_in: int = 0
    items_out: int = 0
    consecutive_failures: int = 0


@dataclass
class ScanCycleReport:
    cycle_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: datetime = field(default_factory=utc_now)
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    success: bool = True
    markets_scanned: int = 0
    markets_dropped_prefilter: int = 0
    prefilter_reasons: dict[str, int] = field(default_factory=dict)
    markets_after_filter: int = 0
    markets_enriched: int = 0
    markets_fully_enriched: int = 0
    markets_partially_enriched: int = 0
    enrichment_provider_status: dict[str, str] = field(default_factory=dict)
    candidates_generated: int = 0
    candidates_per_strategy: dict[str, int] = field(default_factory=dict)
    candidates_validated: int = 0
    candidates_rejected: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    intents_created: int = 0
    allocation_rejections: int = 0
    allocation_rejection_reasons: dict[str, int] = field(default_factory=dict)
    executions_attempted: int = 0
    executions_succeeded: int = 0
    executions_failed: int = 0
    capital_total: float = 0.0
    capital_deployed: float = 0.0
    capital_available: float = 0.0
    capital_utilization_pct: float = 0.0
    open_position_count: int = 0
    unrealized_pnl: float = 0.0
    module_health: dict[str, ModuleHealth] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rejected_signals_detail: list[RejectedSignal] = field(default_factory=list)
    allocation_rejections_detail: list[AllocationRejection] = field(default_factory=list)
    execution_results_detail: list[ExecutionResult] = field(default_factory=list)

    @property
    def top_rejection_reason(self) -> Optional[str]:
        if not self.rejection_reasons:
            return None
        return max(self.rejection_reasons, key=self.rejection_reasons.get)

    @property
    def top_allocation_rejection_reason(self) -> Optional[str]:
        if not self.allocation_rejection_reasons:
            return None
        return max(
            self.allocation_rejection_reasons,
            key=self.allocation_rejection_reasons.get,
        )

    @property
    def zero_trade_explanation(self) -> str:
        if self.executions_succeeded > 0:
            return ""
        parts: list[str] = []
        if self.markets_scanned == 0:
            parts.append("Scanner returned 0 markets (API issue?)")
        elif self.markets_after_filter == 0:
            parts.append(
                f"All {self.markets_scanned} markets filtered out. "
                f"Top filter: {self._top_prefilter()}"
            )
        elif self.candidates_generated == 0:
            parts.append(
                f"{self.markets_after_filter} markets passed filters but "
                "0 strategies generated candidates"
            )
        elif self.candidates_validated == 0:
            parts.append(
                f"All {self.candidates_generated} candidates rejected by validation. "
                f"Top reason: {self.top_rejection_reason}"
            )
        elif self.intents_created == 0:
            parts.append(
                f"All {self.candidates_validated} validated signals rejected by allocation. "
                f"Top reason: {self.top_allocation_rejection_reason}"
            )
        elif self.executions_succeeded == 0:
            parts.append(f"All {self.intents_created} intents failed execution")
        return " | ".join(parts)

    def _top_prefilter(self) -> Optional[str]:
        if not self.prefilter_reasons:
            return None
        return max(self.prefilter_reasons, key=self.prefilter_reasons.get)


def dataclass_to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {
            item.name: dataclass_to_dict(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): dataclass_to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [dataclass_to_dict(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    return value
