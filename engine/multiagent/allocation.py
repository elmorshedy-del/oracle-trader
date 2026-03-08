from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from .config import ExecutionConfig, RiskLimits, SizingConfig
from .contracts import (
    AllocationRejection,
    ExecutionIntent,
    ExecutionResult,
    PortfolioSnapshot,
    ValidatedSignal,
)
from .enums import AllocationRejectionReason, ExecutionMode, SignalDirection


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def fixed_fractional(
    total_capital: float,
    base_fraction: float,
    edge: float,
    edge_normalization: float,
) -> float:
    edge_scalar = min(abs(edge) / edge_normalization, 2.0)
    return total_capital * base_fraction * edge_scalar


def kelly_fraction(edge: float, price: float) -> float:
    fair_value = max(0.01, min(0.99, price + edge))
    q = 1 - fair_value
    payout_odds = (1 / price) - 1 if price > 0 else 0.0
    if payout_odds <= 0:
        return 0.0
    kelly = (fair_value * payout_odds - q) / payout_odds
    return max(0.0, kelly)


class SlippageModel:
    def __init__(self, impact_factor: float = 0.1, min_slippage: float = 0.001):
        self.impact_factor = impact_factor
        self.min_slippage = min_slippage

    def compute(self, price: float, size_usd: float, liquidity: float) -> float:
        if liquidity <= 0:
            return self.min_slippage
        raw = (size_usd / liquidity) * self.impact_factor
        return max(self.min_slippage, raw)


class LiveExecutor(Protocol):
    def fill(self, intent: ExecutionIntent) -> ExecutionResult:
        ...


class PaperExecutor:
    def __init__(self, slippage_model: SlippageModel):
        self.slippage_model = slippage_model

    def fill(self, intent: ExecutionIntent) -> ExecutionResult:
        signal = intent.signal.signal if intent.signal is not None else None
        if signal is None or signal.market_snapshot is None:
            return ExecutionResult(
                intent=intent,
                executed=False,
                execution_mode=ExecutionMode.PAPER,
                error="intent missing validated signal or market snapshot",
            )

        price = signal.current_price
        slippage = self.slippage_model.compute(
            price=price,
            size_usd=intent.position_size_usd,
            liquidity=signal.market_snapshot.liquidity,
        )

        if signal.direction in (SignalDirection.BUY_YES, SignalDirection.BUY_NO):
            fill_price = price * (1 + slippage)
        else:
            fill_price = price * (1 - slippage)

        fill_price = max(0.001, min(0.999, fill_price))
        actual_slippage = abs(fill_price - price) / price if price > 0 else 0.0
        if actual_slippage > intent.max_slippage_pct:
            return ExecutionResult(
                intent=intent,
                executed=False,
                execution_mode=ExecutionMode.PAPER,
                error=f"slippage {actual_slippage:.4f} exceeds max {intent.max_slippage_pct:.4f}",
            )

        shares = intent.position_size_usd / fill_price if fill_price > 0 else 0.0
        return ExecutionResult(
            intent=intent,
            executed=True,
            execution_mode=ExecutionMode.PAPER,
            fill_price=fill_price,
            shares_filled=shares,
            slippage_pct=actual_slippage,
            fees=0.0,
        )


class Executor:
    def __init__(
        self,
        mode: ExecutionMode,
        paper: PaperExecutor,
        live: LiveExecutor | None = None,
    ):
        self.mode = mode
        self.paper = paper
        self.live = live

    @classmethod
    def from_config(
        cls,
        config: ExecutionConfig,
        live: LiveExecutor | None = None,
    ) -> "Executor":
        paper = PaperExecutor(
            SlippageModel(
                impact_factor=config.paper.impact_factor,
                min_slippage=config.paper.min_slippage_pct,
            )
        )
        return cls(
            mode=ExecutionMode(config.mode),
            paper=paper,
            live=live,
        )

    def execute(self, intents: list[ExecutionIntent]) -> list[ExecutionResult]:
        results: list[ExecutionResult] = []
        for intent in intents:
            if self.mode == ExecutionMode.LIVE and self.live is not None:
                results.append(self.live.fill(intent))
            else:
                results.append(self.paper.fill(intent))
        return results


@dataclass
class Allocator:
    limits: RiskLimits
    sizing_config: SizingConfig

    def allocate(
        self,
        signals: list[ValidatedSignal],
        portfolio: PortfolioSnapshot,
    ) -> tuple[list[ExecutionIntent], list[AllocationRejection]]:
        intents: list[ExecutionIntent] = []
        rejections: list[AllocationRejection] = []

        if self._is_drawdown_breached(portfolio):
            current_drawdown = self._current_drawdown(portfolio)
            for signal in signals:
                rejections.append(
                    AllocationRejection(
                        signal=signal,
                        reason=AllocationRejectionReason.RISK_BUDGET_EXCEEDED,
                        constraint_name="max_drawdown",
                        constraint_limit=self.limits.max_drawdown_pct,
                        current_utilization=current_drawdown,
                    )
                )
            return [], rejections

        ranked = self._rank_signals(signals)
        running_deployed = portfolio.deployed_capital
        running_strategy_exposure = dict(portfolio.exposure_by_strategy)
        running_category_exposure = dict(portfolio.exposure_by_category)
        running_strategy_positions = dict(portfolio.positions_by_strategy)
        running_market_ids = set(portfolio.open_market_ids)

        for signal in ranked:
            candidate = signal.signal
            snapshot = candidate.market_snapshot
            if snapshot is None:
                rejections.append(
                    AllocationRejection(
                        signal=signal,
                        reason=AllocationRejectionReason.CAPITAL_INSUFFICIENT,
                        constraint_name="missing_market_snapshot",
                        constraint_limit=0.0,
                        current_utilization=0.0,
                    )
                )
                continue

            market_id = candidate.market_id
            strategy = candidate.strategy_name
            category = snapshot.category.value
            strategy_cap = self.limits.strategy_caps.get(strategy)

            if market_id in portfolio.recently_closed:
                closed_at = portfolio.recently_closed[market_id]
                hours_since = (utc_now() - closed_at).total_seconds() / 3600
                if hours_since < self.limits.reentry_cooldown_hours:
                    rejections.append(
                        AllocationRejection(
                            signal=signal,
                            reason=AllocationRejectionReason.REENTRY_COOLDOWN,
                            constraint_name="reentry_cooldown",
                            constraint_limit=self.limits.reentry_cooldown_hours,
                            current_utilization=hours_since,
                        )
                    )
                    continue

            if strategy_cap is not None:
                current_positions = running_strategy_positions.get(strategy, 0)
                if current_positions >= strategy_cap.max_positions:
                    rejections.append(
                        AllocationRejection(
                            signal=signal,
                            reason=AllocationRejectionReason.STRATEGY_CAP_REACHED,
                            constraint_name=f"strategy_max_positions.{strategy}",
                            constraint_limit=float(strategy_cap.max_positions),
                            current_utilization=float(current_positions),
                        )
                    )
                    continue

            raw_size = self._compute_position_size(signal, portfolio)
            max_global_pct_usd = portfolio.total_capital * self.limits.max_single_position_pct
            max_strategy_usd = (
                strategy_cap.max_single_position_usd
                if strategy_cap is not None
                else self.limits.max_single_position_usd
            )
            capped_size = min(
                raw_size,
                self.limits.max_single_position_usd,
                max_global_pct_usd,
                max_strategy_usd,
            )

            if capped_size <= 0:
                rejections.append(
                    AllocationRejection(
                        signal=signal,
                        reason=AllocationRejectionReason.POSITION_SIZE_EXCEEDS_MAX,
                        constraint_name="position_size_zero_after_caps",
                        constraint_limit=0.0,
                        current_utilization=raw_size,
                    )
                )
                continue

            if strategy_cap is not None:
                strategy_limit_usd = portfolio.total_capital * strategy_cap.max_capital_pct
                strategy_deployed = running_strategy_exposure.get(strategy, 0.0)
                if strategy_deployed + capped_size > strategy_limit_usd:
                    remaining = strategy_limit_usd - strategy_deployed
                    if remaining < self.sizing_config.min_position_usd:
                        rejections.append(
                            AllocationRejection(
                                signal=signal,
                                reason=AllocationRejectionReason.STRATEGY_CAP_REACHED,
                                constraint_name=f"strategy_capital.{strategy}",
                                constraint_limit=strategy_limit_usd,
                                current_utilization=strategy_deployed,
                            )
                        )
                        continue
                    capped_size = remaining

            category_limit_pct = self.limits.category_caps.get(category, 0.10)
            category_limit_usd = portfolio.total_capital * category_limit_pct
            category_deployed = running_category_exposure.get(category, 0.0)
            if category_deployed + capped_size > category_limit_usd:
                remaining = category_limit_usd - category_deployed
                if remaining < self.sizing_config.min_position_usd:
                    rejections.append(
                        AllocationRejection(
                            signal=signal,
                            reason=AllocationRejectionReason.CATEGORY_CAP_REACHED,
                            constraint_name=f"category_capital.{category}",
                            constraint_limit=category_limit_usd,
                            current_utilization=category_deployed,
                        )
                    )
                    continue
                capped_size = remaining

            max_deployed = portfolio.total_capital * self.limits.max_portfolio_utilization_pct
            if running_deployed + capped_size > max_deployed:
                remaining = max_deployed - running_deployed
                if remaining < self.sizing_config.min_position_usd:
                    rejections.append(
                        AllocationRejection(
                            signal=signal,
                            reason=AllocationRejectionReason.CAPITAL_INSUFFICIENT,
                            constraint_name="max_portfolio_utilization",
                            constraint_limit=max_deployed,
                            current_utilization=running_deployed,
                        )
                    )
                    continue
                capped_size = remaining

            available_after = portfolio.total_capital - running_deployed - capped_size
            min_reserve = portfolio.total_capital * self.limits.min_reserve_pct
            if available_after < min_reserve:
                max_possible = portfolio.total_capital - running_deployed - min_reserve
                if max_possible < self.sizing_config.min_position_usd:
                    rejections.append(
                        AllocationRejection(
                            signal=signal,
                            reason=AllocationRejectionReason.RESERVE_VIOLATED,
                            constraint_name="min_reserve",
                            constraint_limit=min_reserve,
                            current_utilization=portfolio.total_capital - running_deployed,
                        )
                    )
                    continue
                capped_size = max_possible

            related_ids = set(snapshot.related_market_ids)
            overlap = related_ids & running_market_ids
            if len(overlap) >= self.limits.max_correlated_positions:
                rejections.append(
                    AllocationRejection(
                        signal=signal,
                        reason=AllocationRejectionReason.CORRELATION_LIMIT,
                        constraint_name="max_correlated_positions",
                        constraint_limit=float(self.limits.max_correlated_positions),
                        current_utilization=float(len(overlap)),
                    )
                )
                continue

            price = candidate.current_price
            shares = capped_size / price if price > 0 else 0.0
            intent = ExecutionIntent(
                signal=signal,
                position_size_usd=capped_size,
                estimated_shares=shares,
                max_slippage_pct=self.sizing_config.max_slippage_pct,
                risk_budget_consumed_pct=capped_size / portfolio.total_capital if portfolio.total_capital else 0.0,
                portfolio_weight_pct=capped_size / portfolio.total_capital if portfolio.total_capital else 0.0,
                allocation_priority=self._priority_score(signal),
            )
            intents.append(intent)

            running_deployed += capped_size
            running_strategy_exposure[strategy] = running_strategy_exposure.get(strategy, 0.0) + capped_size
            running_category_exposure[category] = running_category_exposure.get(category, 0.0) + capped_size
            running_strategy_positions[strategy] = running_strategy_positions.get(strategy, 0) + 1
            running_market_ids.add(market_id)

        return intents, rejections

    def _rank_signals(self, signals: list[ValidatedSignal]) -> list[ValidatedSignal]:
        return sorted(
            signals,
            key=lambda item: (
                abs(item.signal.edge_estimate),
                item.signal.market_snapshot.volume_24h if item.signal.market_snapshot else 0.0,
            ),
            reverse=True,
        )

    def _compute_position_size(
        self,
        signal: ValidatedSignal,
        portfolio: PortfolioSnapshot,
    ) -> float:
        edge = abs(signal.signal.edge_estimate)
        if self.sizing_config.method == "kelly":
            price = max(signal.signal.current_price, 0.01)
            fraction = kelly_fraction(signal.signal.edge_estimate, price)
            return portfolio.total_capital * fraction
        return fixed_fractional(
            total_capital=portfolio.total_capital,
            base_fraction=self.sizing_config.base_fraction,
            edge=edge,
            edge_normalization=self.sizing_config.edge_normalization,
        )

    def _is_drawdown_breached(self, portfolio: PortfolioSnapshot) -> bool:
        if portfolio.total_capital <= 0:
            return True
        drawdown = -portfolio.total_unrealized_pnl / portfolio.total_capital
        return drawdown > self.limits.max_drawdown_pct

    def _current_drawdown(self, portfolio: PortfolioSnapshot) -> float:
        if portfolio.total_capital <= 0:
            return 1.0
        return -portfolio.total_unrealized_pnl / portfolio.total_capital

    @staticmethod
    def _priority_score(signal: ValidatedSignal) -> int:
        return int(abs(signal.signal.edge_estimate) * 10000)
