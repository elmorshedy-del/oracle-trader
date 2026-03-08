from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import time

from .config import AuditConfig
from .contracts import (
    AllocationRejection,
    ExecutionIntent,
    ExecutionResult,
    MarketContext,
    ModuleHealth,
    RejectedSignal,
    ScanCycleReport,
    ValidatedSignal,
    dataclass_to_dict,
    utc_now,
)
from .enums import ModuleStatus


def compute_module_status(
    succeeded: bool,
    items_in: int,
    items_out: int,
    error: str | None,
    consecutive_failures: int,
) -> ModuleStatus:
    if not succeeded or error:
        return ModuleStatus.FAILED
    if items_in > 0 and items_out == 0:
        return ModuleStatus.DEGRADED
    if consecutive_failures > 0:
        return ModuleStatus.DEGRADED
    return ModuleStatus.HEALTHY


@dataclass
class ScanCycleTracer:
    report: ScanCycleReport | None = None

    def start_cycle(self) -> str:
        self.report = ScanCycleReport(started_at=utc_now())
        return self.report.cycle_id

    def record_discovery(self, scanned: int, filtered: int, dropped: list[Any]) -> None:
        report = self._require_report()
        report.markets_scanned = scanned
        report.markets_after_filter = filtered
        report.markets_dropped_prefilter = len(dropped)
        reasons: dict[str, int] = {}
        for item in dropped:
            reason = getattr(item, "reason", "unknown")
            reasons[reason] = reasons.get(reason, 0) + 1
        report.prefilter_reasons = reasons

    def record_enrichment(self, enriched: list[MarketContext]) -> None:
        report = self._require_report()
        report.markets_enriched = len(enriched)
        report.markets_fully_enriched = sum(
            1 for market in enriched if market.enrichment_completeness >= 0.99
        )
        report.markets_partially_enriched = sum(
            1 for market in enriched if 0.0 < market.enrichment_completeness < 0.99
        )
        provider_status: dict[str, str] = {}
        for market in enriched:
            for provider, enrichment in market.enrichments.items():
                provider_status[provider] = "error" if enrichment.error else "ok"
        report.enrichment_provider_status = provider_status

    def record_strategy_output(self, strategy_name: str, count: int) -> None:
        report = self._require_report()
        report.candidates_per_strategy[strategy_name] = count

    def record_signals(self, candidates: list[Any]) -> None:
        report = self._require_report()
        report.candidates_generated = len(candidates)
        report.candidates_detail = list(candidates[:40])

    def record_validation(
        self,
        validated: list[ValidatedSignal],
        rejected: list[RejectedSignal],
    ) -> None:
        report = self._require_report()
        report.candidates_validated = len(validated)
        report.candidates_rejected = len(rejected)
        reasons: dict[str, int] = {}
        for item in rejected:
            for code in item.rejection_codes:
                key = code.value
                reasons[key] = reasons.get(key, 0) + 1
        report.rejection_reasons = reasons
        report.rejected_signals_detail = rejected

    def record_allocation(
        self,
        intents: list[ExecutionIntent],
        rejected: list[AllocationRejection],
    ) -> None:
        report = self._require_report()
        report.intents_created = len(intents)
        report.allocation_rejections = len(rejected)
        reasons: dict[str, int] = {}
        for item in rejected:
            key = item.reason.value
            reasons[key] = reasons.get(key, 0) + 1
        report.allocation_rejection_reasons = reasons
        report.allocation_rejections_detail = rejected

    def record_execution(self, results: list[ExecutionResult]) -> None:
        report = self._require_report()
        report.executions_attempted = len(results)
        report.executions_succeeded = sum(1 for result in results if result.executed)
        report.executions_failed = sum(1 for result in results if not result.executed)
        report.execution_results_detail = results

    def record_health(self, module_name: str, health: ModuleHealth) -> None:
        report = self._require_report()
        report.module_health[module_name] = health

    def record_module_failure(self, module_name: str, error: Exception) -> None:
        report = self._require_report()
        report.success = False
        report.errors.append(f"{module_name}: {error}")

    def add_warning(self, message: str) -> None:
        self._require_report().warnings.append(message)

    def add_error(self, message: str) -> None:
        report = self._require_report()
        report.success = False
        report.errors.append(message)

    def finalize(self, portfolio: Any) -> ScanCycleReport:
        report = self._require_report()
        report.completed_at = utc_now()
        report.duration_seconds = (
            report.completed_at - report.started_at
        ).total_seconds()
        report.capital_total = getattr(portfolio, "total_capital", 0.0)
        report.capital_deployed = getattr(portfolio, "deployed_capital", 0.0)
        report.capital_available = getattr(portfolio, "available_capital", 0.0)
        report.capital_utilization_pct = getattr(portfolio, "capital_utilization_pct", 0.0)
        report.open_position_count = getattr(portfolio, "position_count", 0)
        report.unrealized_pnl = getattr(portfolio, "total_unrealized_pnl", 0.0)
        return report

    def _require_report(self) -> ScanCycleReport:
        if self.report is None:
            self.start_cycle()
        return self.report  # type: ignore[return-value]


@dataclass
class SnapshotStore:
    config: AuditConfig

    def __post_init__(self) -> None:
        self.config.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def save(self, report: ScanCycleReport) -> None:
        filename = self.config.snapshot_dir / f"cycle_{report.cycle_id}.json"
        filename.write_text(json.dumps(dataclass_to_dict(report), indent=2))
        self._prune()

    def load(self, cycle_id: str) -> dict[str, Any] | None:
        filename = self.config.snapshot_dir / f"cycle_{cycle_id}.json"
        if not filename.exists():
            return None
        return json.loads(filename.read_text())

    def list_recent(self, n: int = 100) -> list[dict[str, Any]]:
        files = sorted(
            self.config.snapshot_dir.glob("cycle_*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        return [json.loads(path.read_text()) for path in files[:n]]

    def _prune(self) -> None:
        files = sorted(
            self.config.snapshot_dir.glob("cycle_*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        for path in files[self.config.max_snapshots :]:
            path.unlink(missing_ok=True)
