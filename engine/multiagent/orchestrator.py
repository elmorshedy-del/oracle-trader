from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import time
import traceback
from typing import Any, Protocol

from .allocation import Allocator, Executor
from .audit import ScanCycleTracer, SnapshotStore
from .config import OrchestratorConfig
from .contracts import (
    MarketContext,
    ModuleHealth,
    NormalizedMarket,
    PortfolioSnapshot,
    ScanCycleReport,
)
from .enums import ModuleStatus
from .validation import Validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Strategy(Protocol):
    name: str

    def generate(
        self,
        markets: list[MarketContext],
        portfolio: PortfolioSnapshot,
        config: Any,
    ) -> list[Any]:
        ...


class StrategyRegistry(Protocol):
    def active_strategies(self) -> list[Strategy]:
        ...


class Scanner(Protocol):
    def discover(self, config: Any) -> list[Any]:
        ...

    def normalize(self, raw: list[Any]) -> list[NormalizedMarket]:
        ...

    def pre_filter(
        self,
        markets: list[NormalizedMarket],
        config: Any,
    ) -> tuple[list[NormalizedMarket], list[Any]]:
        ...


class Enricher(Protocol):
    def enrich(self, markets: list[NormalizedMarket]) -> list[MarketContext]:
        ...


class StateManager(Protocol):
    def snapshot(self) -> PortfolioSnapshot:
        ...

    def apply_executions(self, results: list[Any]) -> None:
        ...

    def save(self) -> None:
        ...


@dataclass
class Orchestrator:
    scanner: Scanner
    enricher: Enricher
    strategy_registry: StrategyRegistry
    validator: Validator
    allocator: Allocator
    executor: Executor
    state: StateManager
    tracer: ScanCycleTracer
    snapshot_store: SnapshotStore
    config: OrchestratorConfig = field(default_factory=OrchestratorConfig)

    def run_scan_cycle(self) -> ScanCycleReport:
        self.tracer.start_cycle()
        portfolio = self.state.snapshot()

        try:
            raw_markets = self._timed(
                "scanner.discover",
                lambda: self.scanner.discover(self.config.scanner),
            )
            normalized = self.scanner.normalize(raw_markets)
            passed, dropped = self.scanner.pre_filter(normalized, self.config.filters)
            self.tracer.record_discovery(
                scanned=len(normalized),
                filtered=len(passed),
                dropped=dropped,
            )
        except Exception as exc:
            self.tracer.record_module_failure("scanner", exc)
            return self._finalize_cycle(portfolio, abort_reason="scanner_failed")

        try:
            enriched = self._timed("enricher", lambda: self.enricher.enrich(passed))
            self.tracer.record_enrichment(enriched)
        except Exception as exc:
            self.tracer.record_module_failure("enricher", exc)
            enriched = self._to_bare_contexts(passed)
            self.tracer.add_warning(
                "enrichment_pipeline_crashed_using_bare_contexts"
            )

        all_candidates: list[Any] = []
        for strategy in self.strategy_registry.active_strategies():
            try:
                strategy_config = self.config.strategies.get(strategy.name)
                candidates = self._timed(
                    f"strategy.{strategy.name}",
                    lambda strat=strategy, cfg=strategy_config: strat.generate(
                        enriched, portfolio, cfg
                    ),
                )
                all_candidates.extend(candidates)
                self.tracer.record_strategy_output(strategy.name, len(candidates))
            except Exception as exc:
                self.tracer.record_module_failure(f"strategy.{strategy.name}", exc)

        self.tracer.record_signals(all_candidates)

        try:
            validated, rejected = self._timed(
                "validator",
                lambda: self.validator.validate(all_candidates, portfolio),
            )
            self.tracer.record_validation(validated, rejected)
        except Exception as exc:
            self.tracer.record_module_failure("validator", exc)
            return self._finalize_cycle(portfolio, abort_reason="validator_failed")

        try:
            intents, allocation_rejected = self._timed(
                "allocator",
                lambda: self.allocator.allocate(validated, portfolio),
            )
            self.tracer.record_allocation(intents, allocation_rejected)
        except Exception as exc:
            self.tracer.record_module_failure("allocator", exc)
            return self._finalize_cycle(portfolio, abort_reason="allocator_failed")

        results: list[Any] = []
        if intents:
            try:
                results = self._timed(
                    "executor",
                    lambda: self.executor.execute(intents),
                )
                self.tracer.record_execution(results)
            except Exception as exc:
                self.tracer.record_module_failure("executor", exc)

        if results:
            try:
                self.state.apply_executions(results)
                self.state.save()
            except Exception as exc:
                self.tracer.record_module_failure("state_manager", exc)
                self.tracer.add_error(
                    f"STATE_APPLY_FAILED: executions may not be recorded: {exc}"
                )

        return self._finalize_cycle(self.state.snapshot())

    def _finalize_cycle(
        self,
        portfolio: PortfolioSnapshot,
        abort_reason: str | None = None,
    ) -> ScanCycleReport:
        if abort_reason:
            self.tracer.add_error(f"CYCLE_ABORTED: {abort_reason}")
        report = self.tracer.finalize(portfolio)
        try:
            self.snapshot_store.save(report)
        except Exception:
            pass
        return report

    def _timed(self, module_name: str, callback):
        started = time.monotonic()
        try:
            result = callback()
            duration = time.monotonic() - started
            count_in, count_out = _infer_io_counts(result)
            self.tracer.record_health(
                module_name,
                ModuleHealth(
                    module_name=module_name,
                    status=ModuleStatus.HEALTHY,
                    last_run_at=utc_now(),
                    last_duration_seconds=duration,
                    items_in=count_in,
                    items_out=count_out,
                ),
            )
            return result
        except Exception:
            duration = time.monotonic() - started
            self.tracer.record_health(
                module_name,
                ModuleHealth(
                    module_name=module_name,
                    status=ModuleStatus.FAILED,
                    last_run_at=utc_now(),
                    last_duration_seconds=duration,
                    last_error=traceback.format_exc(),
                ),
            )
            raise

    @staticmethod
    def _to_bare_contexts(markets: list[NormalizedMarket]) -> list[MarketContext]:
        return [
            MarketContext(
                market_id=market.market_id,
                question=market.question,
                category=market.category,
                outcomes=market.outcomes,
                volume_24h=market.volume_24h,
                total_volume=market.total_volume,
                liquidity=market.liquidity,
                created_date=market.created_date,
                source_url=market.source_url,
                resolution_date=market.resolution_date,
                description=market.description,
                tags=market.tags,
                enrichments={},
                enrichment_completeness=0.0,
            )
            for market in markets
        ]


def _infer_io_counts(result: Any) -> tuple[int, int]:
    if isinstance(result, tuple):
        counts = [len(item) for item in result if hasattr(item, "__len__")]
        if len(counts) >= 2:
            return counts[0], counts[1]
        if counts:
            return counts[0], counts[0]
    if hasattr(result, "__len__"):
        size = len(result)
        return size, size
    return 0, 0
