from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from config import PipelineConfig
from data.collector import PolymarketCollector
from runtime_paths import LOG_DIR

from .allocation import Allocator, Executor
from .audit import ScanCycleTracer, SnapshotStore
from .config import OrchestratorConfig
from .contracts import (
    MarketContext,
    NormalizedMarket,
    PortfolioSnapshot,
    PositionState,
    ScanCycleReport,
    dataclass_to_dict,
    utc_now,
)
from .enrichment import (
    CryptoEnrichmentProvider,
    MultiProviderEnricher,
    NewsEnrichmentProvider,
    WeatherEnrichmentProvider,
)
from .enums import ModuleStatus, PositionStatus
from .llm import MultiagentLLMRouter
from .orchestrator import Orchestrator
from .strategies import (
    CryptoLatencyStrategy,
    NewsSignalStrategy,
    RelationshipArbitrageStrategy,
    StaticStrategyRegistry,
    WeatherLatencyStrategy,
    WeatherSniperStrategy,
    WeatherSwingStrategy,
)
from .store import RuntimeMetricsStore
from .validation import Validator
from .adapters import normalized_market_from_legacy


logger = logging.getLogger(__name__)

DEFAULT_MULTIAGENT_STARTING_CAPITAL = 2000.0
DEFAULT_MULTIAGENT_SCAN_INTERVAL_SECS = 60
MAX_MARKET_PREVIEW = 12
DEFAULT_CLOB_PRIORITY_MARKETS = 36
DEFAULT_CLOB_CONCURRENCY = 8
DEFAULT_EXIT_CONVERGENCE_RATIO = 0.96
DEFAULT_EXIT_MAX_HOLD_HOURS = 8.0
DEFAULT_EXIT_STOP_LOSS_PCT = -0.18
METRICS_LOG_PATH = LOG_DIR / "multiagent_metrics.jsonl"
METRICS_DB_PATH = LOG_DIR / "multiagent_runtime.sqlite"
COMPARISON_VIEW_ORDER = [
    ("all", "All Opus", "all"),
    ("relationship_arbitrage", "Arbitrage", "relationship_arbitrage"),
    ("weather_sniper", "Weather Sniper", "weather_sniper"),
    ("weather_latency", "Weather Latency", "weather_latency"),
    ("weather_swing", "Weather Swing", "weather_swing"),
    ("crypto_latency", "Crypto Latency", "crypto_latency"),
    ("news_signal", "News", "news_signal"),
]


@dataclass(frozen=True)
class PrefilterDrop:
    market_id: str
    reason: str


class CollectorScanner:
    def __init__(self) -> None:
        self._raw_markets: list[Any] = []

    def set_raw_markets(self, raw_markets: list[Any]) -> None:
        self._raw_markets = list(raw_markets)

    def discover(self, config: Any) -> list[Any]:
        return list(self._raw_markets)

    def normalize(self, raw: list[Any]) -> list[NormalizedMarket]:
        return [normalized_market_from_legacy(market) for market in raw]

    def pre_filter(
        self,
        markets: list[NormalizedMarket],
        config: Any,
    ) -> tuple[list[NormalizedMarket], list[PrefilterDrop]]:
        passed: list[NormalizedMarket] = []
        dropped: list[PrefilterDrop] = []

        for market in markets:
            reason = None
            if not market.outcomes:
                reason = "missing_outcomes"
            elif market.raw_metadata.get("closed"):
                reason = "closed"
            elif market.liquidity <= 0:
                reason = "no_liquidity"
            elif market.hours_to_resolution == 0:
                reason = "resolving_now"

            if reason:
                dropped.append(PrefilterDrop(market_id=market.market_id, reason=reason))
            else:
                passed.append(market)

        return passed, dropped


class BareContextEnricher:
    def enrich(self, markets: list[NormalizedMarket]) -> list[MarketContext]:
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


class EmptyStrategyRegistry:
    def active_strategies(self) -> list[Any]:
        return []


class InMemoryStateManager:
    def __init__(
        self,
        *,
        starting_capital: float,
        reserve_pct: float,
    ) -> None:
        self._starting_capital = starting_capital
        self._reserve_pct = reserve_pct
        self._positions: list[PositionState] = []
        self._closed_positions: list[dict[str, Any]] = []
        self._closed_event_queue: list[dict[str, Any]] = []
        self._recently_closed: dict[str, datetime] = {}
        self._cash_balance = starting_capital
        self._realized_pnl = 0.0
        self._snapshot = self._recompute_snapshot()

    def snapshot(self) -> PortfolioSnapshot:
        return self._snapshot

    def apply_executions(self, results: list[Any]) -> None:
        if not results:
            return

        positions = list(self._positions)
        cash_balance = self._cash_balance
        open_market_ids = set(self._snapshot.open_market_ids)
        exposure_by_strategy = dict(self._snapshot.exposure_by_strategy)
        exposure_by_category = dict(self._snapshot.exposure_by_category)
        positions_by_strategy = dict(self._snapshot.positions_by_strategy)

        for result in results:
            if not result.executed or result.intent is None or result.intent.signal is None:
                continue

            signal = result.intent.signal.signal
            market_snapshot = signal.market_snapshot
            if market_snapshot is None or result.fill_price is None:
                continue

            cost_basis = result.fill_price * result.shares_filled
            position = PositionState(
                position_id=result.result_id,
                market_id=signal.market_id,
                market_question=market_snapshot.question,
                strategy_name=signal.strategy_name,
                category=market_snapshot.category,
                direction=signal.direction,
                outcome=signal.outcome,
                entry_price=result.fill_price,
                current_price=result.fill_price,
                shares=result.shares_filled,
                cost_basis=cost_basis,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                status=PositionStatus.OPEN,
                opened_at=result.executed_at,
                signal_id=signal.signal_id,
                metadata={
                    "target_price": signal.estimated_fair_value,
                    **signal.metadata,
                },
            )
            positions.append(position)
            cash_balance -= cost_basis
            open_market_ids.add(signal.market_id)
            exposure_by_strategy[signal.strategy_name] = (
                exposure_by_strategy.get(signal.strategy_name, 0.0) + cost_basis
            )
            exposure_by_category[market_snapshot.category.value] = (
                exposure_by_category.get(market_snapshot.category.value, 0.0) + cost_basis
            )
            positions_by_strategy[signal.strategy_name] = (
                positions_by_strategy.get(signal.strategy_name, 0) + 1
            )

        self._positions = positions
        self._cash_balance = max(0.0, cash_balance)
        self._snapshot = self._recompute_snapshot()

    def save(self) -> None:
        return

    def refresh_prices(self, markets: list[NormalizedMarket]) -> None:
        if not self._positions:
            self._snapshot = self._recompute_snapshot()
            return

        market_map = {market.market_id: market for market in markets}
        now = utc_now()
        open_positions: list[PositionState] = []

        for position in self._positions:
            market = market_map.get(position.market_id)
            if market:
                latest_price = market.yes_price if position.outcome.upper() == "YES" else market.no_price
                if latest_price is not None:
                    position.current_price = latest_price
                position.unrealized_pnl = (position.current_price - position.entry_price) * position.shares
                position.last_updated = now

            exit_reason = self._exit_reason(position, now)
            if exit_reason:
                realized = position.unrealized_pnl
                self._cash_balance += position.current_price * position.shares
                self._realized_pnl += realized
                position.realized_pnl = realized
                position.status = PositionStatus.CLOSED
                position.close_reason = exit_reason
                position.closed_at = now
                self._recently_closed[position.market_id] = now
                event = {
                    "market_id": position.market_id,
                    "market_question": position.market_question,
                    "strategy_name": position.strategy_name,
                    "entry_price": position.entry_price,
                    "exit_price": position.current_price,
                    "shares": position.shares,
                    "realized_pnl": realized,
                    "opened_at": position.opened_at.isoformat(),
                    "closed_at": now.isoformat(),
                    "hold_hours": round((now - position.opened_at).total_seconds() / 3600, 2),
                    "close_reason": exit_reason,
                }
                self._closed_positions.append(event)
                self._closed_event_queue.append(event)
                self._closed_positions = self._closed_positions[-200:]
                continue

            open_positions.append(position)

        self._positions = open_positions
        self._cleanup_recently_closed(now)
        self._snapshot = self._recompute_snapshot()

    def _recompute_snapshot(self) -> PortfolioSnapshot:
        deployed_capital = sum(pos.current_price * pos.shares for pos in self._positions)
        total_unrealized = sum(pos.unrealized_pnl for pos in self._positions)
        total_capital = self._cash_balance + deployed_capital
        exposure_by_strategy: dict[str, float] = {}
        exposure_by_category: dict[str, float] = {}
        positions_by_strategy: dict[str, int] = {}
        open_market_ids: set[str] = set()

        for pos in self._positions:
            market_value = pos.current_price * pos.shares
            exposure_by_strategy[pos.strategy_name] = exposure_by_strategy.get(pos.strategy_name, 0.0) + market_value
            exposure_by_category[pos.category.value] = exposure_by_category.get(pos.category.value, 0.0) + market_value
            positions_by_strategy[pos.strategy_name] = positions_by_strategy.get(pos.strategy_name, 0) + 1
            open_market_ids.add(pos.market_id)

        return PortfolioSnapshot(
            total_capital=total_capital,
            available_capital=max(0.0, self._cash_balance),
            deployed_capital=deployed_capital,
            reserved_capital=total_capital * self._reserve_pct,
            total_unrealized_pnl=total_unrealized,
            total_realized_pnl=self._realized_pnl,
            positions=tuple(self._positions),
            position_count=len(self._positions),
            capital_utilization_pct=(deployed_capital / total_capital) if total_capital else 0.0,
            open_market_ids=frozenset(open_market_ids),
            exposure_by_strategy=exposure_by_strategy,
            exposure_by_category=exposure_by_category,
            positions_by_strategy=positions_by_strategy,
            recently_closed=dict(self._recently_closed),
        )

    def _exit_reason(self, position: PositionState, now: datetime) -> str | None:
        target_price = float(position.metadata.get("target_price", 0.0) or 0.0)
        if target_price > 0 and position.current_price >= target_price * DEFAULT_EXIT_CONVERGENCE_RATIO:
            return "target_converged"

        pnl_pct = 0.0
        if position.entry_price > 0:
            pnl_pct = (position.current_price - position.entry_price) / position.entry_price
        if pnl_pct <= DEFAULT_EXIT_STOP_LOSS_PCT:
            return "stop_loss"

        held_hours = (now - position.opened_at).total_seconds() / 3600
        if held_hours >= DEFAULT_EXIT_MAX_HOLD_HOURS:
            return "stale_rotation"

        return None

    def _cleanup_recently_closed(self, now: datetime) -> None:
        cutoff = now - timedelta(hours=24)
        self._recently_closed = {
            market_id: closed_at
            for market_id, closed_at in self._recently_closed.items()
            if closed_at >= cutoff
        }

    def closed_positions(self, limit: int = 50) -> list[dict[str, Any]]:
        return list(self._closed_positions[-limit:])

    def drain_closed_events(self) -> list[dict[str, Any]]:
        events = list(self._closed_event_queue)
        self._closed_event_queue = []
        return events

    def performance_summary(self) -> dict[str, Any]:
        closed = self._closed_positions
        wins = sum(1 for item in closed if item["realized_pnl"] > 0)
        losses = sum(1 for item in closed if item["realized_pnl"] < 0)
        hold_hours = [item["hold_hours"] for item in closed]
        close_reasons: dict[str, int] = {}
        for item in closed:
            reason = item["close_reason"]
            close_reasons[reason] = close_reasons.get(reason, 0) + 1
        total_closed = len(closed)
        return {
            "closed_positions": total_closed,
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / total_closed * 100) if total_closed else 0.0,
            "avg_hold_hours": (sum(hold_hours) / len(hold_hours)) if hold_hours else 0.0,
            "close_reasons": close_reasons,
            "recent_closed": self.closed_positions(12),
        }


class MultiagentRuntime:
    def __init__(
        self,
        *,
        pipeline_config: PipelineConfig | None = None,
        orchestrator_config: OrchestratorConfig | None = None,
        starting_capital: float = DEFAULT_MULTIAGENT_STARTING_CAPITAL,
        scan_interval_secs: int = DEFAULT_MULTIAGENT_SCAN_INTERVAL_SECS,
    ) -> None:
        self.pipeline_config = pipeline_config or PipelineConfig()
        self.config = orchestrator_config or OrchestratorConfig()
        self.starting_capital = starting_capital
        self.scan_interval_secs = scan_interval_secs

        self.collector = PolymarketCollector(
            gamma_host=self.pipeline_config.api.gamma_host,
            clob_host=self.pipeline_config.api.clob_host,
            data_host=self.pipeline_config.api.data_host,
        )
        self.scanner = CollectorScanner()
        self.llm_router = MultiagentLLMRouter(self.config.llm)
        self.enricher = MultiProviderEnricher(
            providers=[
                WeatherEnrichmentProvider(self.pipeline_config),
                CryptoEnrichmentProvider(self.pipeline_config),
                NewsEnrichmentProvider(self.pipeline_config, self.llm_router),
            ]
        )
        self.strategy_registry = StaticStrategyRegistry(
            strategies=[
                RelationshipArbitrageStrategy(),
                WeatherSniperStrategy(),
                WeatherLatencyStrategy(),
                WeatherSwingStrategy(),
                CryptoLatencyStrategy(),
                NewsSignalStrategy(),
            ]
        )
        self.state = InMemoryStateManager(
            starting_capital=self.starting_capital,
            reserve_pct=self.config.risk_limits.min_reserve_pct,
        )
        self.tracer = ScanCycleTracer()
        self.snapshot_store = SnapshotStore(self.config.audit)
        self.metrics_store = RuntimeMetricsStore(METRICS_DB_PATH)
        self.orchestrator = Orchestrator(
            scanner=self.scanner,
            enricher=self.enricher,
            strategy_registry=self.strategy_registry,
            validator=Validator(self.config.validation),
            allocator=Allocator(self.config.risk_limits, self.config.sizing),
            executor=Executor.from_config(self.config.execution),
            state=self.state,
            tracer=self.tracer,
            snapshot_store=self.snapshot_store,
            config=self.config,
        )

        self._running = False
        self._task: asyncio.Task | None = None
        self._last_report: ScanCycleReport | None = None
        self._last_markets: list[NormalizedMarket] = []
        self._scan_count = 0
        self._recent_errors: list[str] = []
        self._total_candidates = 0
        self._total_validated = 0
        self._total_executed = 0
        self._quiet_cycles = 0

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        logger.info("[MULTIAGENT] Starting isolated Opus runtime")
        while self._running:
            try:
                await self._run_cycle()
            except Exception as exc:
                logger.exception("[MULTIAGENT] Cycle failed: %s", exc)
                self._recent_errors.append(str(exc))
                self._recent_errors = self._recent_errors[-20:]
            await asyncio.sleep(self.scan_interval_secs)

    async def stop(self) -> None:
        self._running = False
        await self.enricher.close()
        await self.collector.close()

    def get_status(self) -> dict[str, Any]:
        report = self._last_report
        portfolio = self.state.snapshot()
        state_perf = self.state.performance_summary()
        blockers = self._build_blockers(report)
        comparison_views = self._build_comparison_views(report, portfolio, state_perf)
        return {
            "bridge": {
                "mode": "isolated_runtime",
                "state": "running" if self._running else "stopped",
                "next_step": "expand_opus_strategies",
            },
            "summary": {
                "scan_count": self._scan_count,
                "active_markets": len(self._last_markets),
                "open_positions": portfolio.position_count,
                "top_blocker": blockers[0]["title"] if blockers else "No dominant blocker detected",
            },
            "defaults": dataclass_to_dict(self.config),
            "portfolio": dataclass_to_dict(portfolio),
            "performance": self._build_performance(report),
            "health": self._build_health(report),
            "diagnostics": self._build_diagnostics(report),
            "market_mix": self._build_market_mix(),
            "market_preview": self._build_market_preview(),
            "module_cards": self._build_module_cards(report),
            "strategy_cards": self._build_strategy_cards(report),
            "comparison_views": comparison_views,
            "recent_trades": self._recent_trade_rows(limit=20),
            "recent_closed": state_perf["recent_closed"],
            "export": {
                "logs_url": "/api/multiagent/logs/export",
                "metrics_log_path": str(METRICS_LOG_PATH),
                "metrics_db_path": str(METRICS_DB_PATH),
                "snapshot_dir": str(self.snapshot_store.config.snapshot_dir),
            },
            "blockers": blockers,
        }

    async def _run_cycle(self) -> None:
        raw_markets = await self.collector.get_all_active_markets()
        raw_markets = await self._enrich_priority_markets(raw_markets)
        self.llm_router.begin_cycle()
        await self.enricher.refresh(raw_markets)
        raw_markets = self._merge_supplemental_markets(raw_markets, self.enricher.supplemental_markets())
        self.scanner.set_raw_markets(raw_markets)
        self._last_report = self.orchestrator.run_scan_cycle()
        self._last_markets = self.scanner.normalize(raw_markets)
        self.state.refresh_prices(self._last_markets)
        self._scan_count += 1
        self._total_candidates += self._last_report.candidates_generated
        self._total_validated += self._last_report.candidates_validated
        self._total_executed += self._last_report.executions_succeeded
        if self._last_report.executions_succeeded == 0:
            self._quiet_cycles += 1
        self.metrics_store.record_cycle(
            scan_id=self._scan_count,
            report=self._last_report,
            portfolio=self.state.snapshot(),
            closed_events=self.state.drain_closed_events(),
        )
        self._append_metrics_log(self._last_report)
        logger.info(
            "[MULTIAGENT] Scan #%s: %s markets | %s candidates | %s executed",
            self._scan_count,
            self._last_report.markets_after_filter,
            self._last_report.candidates_generated,
            self._last_report.executions_succeeded,
        )

    @staticmethod
    def _merge_supplemental_markets(raw_markets: list[Any], supplemental_markets: list[Any]) -> list[Any]:
        if not supplemental_markets:
            return raw_markets

        merged = list(raw_markets)
        seen = {getattr(market, "condition_id", None) for market in raw_markets}
        for market in supplemental_markets:
            market_id = getattr(market, "condition_id", None)
            if market_id and market_id not in seen:
                merged.append(market)
                seen.add(market_id)
        return merged

    async def _enrich_priority_markets(self, raw_markets: list[Any]) -> list[Any]:
        if not raw_markets:
            return raw_markets

        priority: list[Any] = []
        seen: set[str] = set()
        sorted_by_volume = sorted(raw_markets, key=lambda item: item.volume_24h, reverse=True)

        for market in sorted_by_volume:
            slug_text = f"{market.question} {market.slug}".lower()
            if "bitcoin" in slug_text or "btc" in slug_text or "ethereum" in slug_text or "eth" in slug_text or "solana" in slug_text or "sol" in slug_text:
                if market.condition_id not in seen:
                    priority.append(market)
                    seen.add(market.condition_id)

        for market in sorted_by_volume:
            if len(priority) >= DEFAULT_CLOB_PRIORITY_MARKETS:
                break
            if market.condition_id not in seen:
                priority.append(market)
                seen.add(market.condition_id)

        semaphore = asyncio.Semaphore(DEFAULT_CLOB_CONCURRENCY)

        async def enrich_one(market: Any) -> Any:
            async with semaphore:
                try:
                    return await self.collector.enrich_market_with_clob(market)
                except Exception as exc:
                    self._recent_errors.append(f"clob_enrich_failed:{market.slug}:{exc}")
                    self._recent_errors = self._recent_errors[-20:]
                    return market

        enriched_priority = await asyncio.gather(*(enrich_one(market) for market in priority))
        enriched_map = {market.condition_id: market for market in enriched_priority}
        return [enriched_map.get(market.condition_id, market) for market in raw_markets]

    def _build_health(self, report: ScanCycleReport | None) -> dict[str, Any]:
        if report is None:
            return {
                "overall_status": "booting",
                "scan": {"status": "booting", "last_scan": None, "avg_duration_secs": 0.0},
                "recent_errors": list(self._recent_errors),
            }

        statuses = [health.status.value for health in report.module_health.values()]
        if not statuses:
            overall = ModuleStatus.DEGRADED.value
        elif ModuleStatus.FAILED.value in statuses:
            overall = ModuleStatus.FAILED.value
        elif ModuleStatus.DEGRADED.value in statuses or report.errors:
            overall = ModuleStatus.DEGRADED.value
        else:
            overall = ModuleStatus.HEALTHY.value

        return {
            "overall_status": overall,
            "scan": {
                "status": overall,
                "last_scan": report.completed_at.isoformat() if report.completed_at else None,
                "avg_duration_secs": round(report.duration_seconds, 2),
            },
            "recent_errors": list(self._recent_errors) + report.errors[-5:],
        }

    def _build_diagnostics(self, report: ScanCycleReport | None) -> dict[str, Any]:
        if report is None:
            return {
                "scan": 0,
                "markets_total": 0,
                "markets_tradeable": 0,
                "signals_by_strategy": {},
                "filtered_signals": {},
                "executed": 0,
                "exits": 0,
                "resolved": 0,
            }

        return {
            "scan": self._scan_count,
            "markets_total": report.markets_scanned,
            "markets_tradeable": report.markets_after_filter,
            "signals_by_strategy": report.candidates_per_strategy,
            "filtered_signals": report.rejection_reasons,
            "executed": report.executions_succeeded,
            "exits": 0,
            "resolved": 0,
        }

    def _build_module_cards(self, report: ScanCycleReport | None) -> list[dict[str, Any]]:
        if report is None:
            return [
                {
                    "name": "opus_runtime",
                    "label": "Opus runtime",
                    "status": "degraded",
                    "detail": "Waiting for first independent scan cycle",
                }
            ]

        cards = []
        for provider_card in self.enricher.provider_cards():
            cards.append(
                {
                    "name": f"provider.{provider_card.name}",
                    "label": f"{provider_card.name} provider",
                    "status": provider_card.status,
                    "detail": provider_card.detail,
                }
            )
        for name, health in report.module_health.items():
            cards.append(
                {
                    "name": name,
                    "label": name,
                    "status": health.status.value,
                    "detail": (
                        f"items in {health.items_in} | items out {health.items_out} | "
                        f"{health.last_duration_seconds:.2f}s"
                    ),
                }
            )
        return cards

    def _build_strategy_cards(self, report: ScanCycleReport | None) -> list[dict[str, Any]]:
        strategy_names = [strategy.name for strategy in self.strategy_registry.active_strategies()]
        if report is None or not strategy_names:
            return [
                {
                    "name": "no_strategies_migrated",
                    "status": "degraded",
                    "runs": self._scan_count,
                    "errors": 0,
                    "signals": 0,
                    "last_error": "The isolated Opus runtime is scanning markets, but no strategy modules have been migrated into it yet.",
                }
            ]

        cards = []
        for name in strategy_names:
            count = report.candidates_per_strategy.get(name, 0)
            status = "healthy" if count > 0 else "degraded"
            last_error = None
            if count == 0:
                last_error = "Strategy scanned successfully but did not emit candidates in the latest cycle."
            cards.append(
                {
                    "name": name,
                    "status": status,
                    "runs": self._scan_count,
                    "errors": 0,
                    "signals": count,
                    "last_error": last_error,
                }
            )
        return cards

    def _build_market_mix(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for market in self._last_markets:
            counts[market.category.value] = counts.get(market.category.value, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))

    def _build_market_preview(self) -> list[dict[str, Any]]:
        preview = []
        for market in sorted(
            self._last_markets,
            key=lambda item: item.volume_24h,
            reverse=True,
        )[:MAX_MARKET_PREVIEW]:
            preview.append(
                {
                    "market_id": market.market_id,
                    "question": market.question,
                    "category": market.category.value,
                    "yes_price": market.yes_price,
                    "no_price": market.no_price,
                    "liquidity": market.liquidity,
                    "volume_24h": market.volume_24h,
                    "hours_to_resolution": market.hours_to_resolution,
                    "tags": list(market.tags[:4]),
                }
            )
        return preview

    def _build_comparison_views(
        self,
        report: ScanCycleReport | None,
        portfolio: PortfolioSnapshot,
        state_perf: dict[str, Any],
    ) -> list[dict[str, Any]]:
        candidates = list(report.candidates_detail) if report else []
        positions = list(portfolio.positions)
        closed = list(self.state.closed_positions(120))
        trades = self._recent_trade_rows(limit=60)
        views: list[dict[str, Any]] = []

        for key, label, source in COMPARISON_VIEW_ORDER:
            if key == "all":
                view_positions = positions
                view_closed = closed
                view_trades = trades
                view_signals = candidates
                realized_pnl = portfolio.total_realized_pnl
                unrealized_pnl = portfolio.total_unrealized_pnl
                exposure = portfolio.deployed_capital
                win_rate = state_perf.get("win_rate", 0.0)
                view_portfolio = {
                    "scope": "aggregate",
                    "starting_capital": self.starting_capital,
                    "total_value": portfolio.total_capital,
                    "cash": portfolio.available_capital,
                    "positions_value": portfolio.deployed_capital,
                    "total_pnl": realized_pnl + unrealized_pnl,
                    "total_pnl_pct": ((realized_pnl + unrealized_pnl) / self.starting_capital * 100)
                    if self.starting_capital
                    else 0.0,
                    "win_rate": win_rate,
                    "total_trades": portfolio.position_count + state_perf.get("closed_positions", 0),
                    "open_positions": portfolio.position_count,
                    "positions": [self._serialize_position(position) for position in view_positions],
                }
            else:
                view_positions = [position for position in positions if position.strategy_name == source]
                view_closed = [event for event in closed if event.get("strategy_name") == source]
                view_trades = [trade for trade in trades if trade.get("source") == source]
                view_signals = [candidate for candidate in candidates if candidate.strategy_name == source]
                realized_pnl = sum(float(event.get("realized_pnl", 0.0) or 0.0) for event in view_closed)
                unrealized_pnl = sum(position.unrealized_pnl for position in view_positions)
                exposure = sum(position.current_price * position.shares for position in view_positions)
                wins = sum(1 for event in view_closed if float(event.get("realized_pnl", 0.0) or 0.0) > 0)
                win_rate = (wins / len(view_closed) * 100) if view_closed else 0.0
                view_portfolio = {
                    "scope": "strategy_filter",
                    "starting_capital": None,
                    "total_value": exposure + realized_pnl,
                    "cash": None,
                    "positions_value": exposure,
                    "total_pnl": realized_pnl + unrealized_pnl,
                    "total_pnl_pct": 0.0,
                    "win_rate": win_rate,
                    "total_trades": len(view_positions) + len(view_closed),
                    "open_positions": len(view_positions),
                    "positions": [self._serialize_position(position) for position in view_positions],
                }

            views.append(
                {
                    "key": key,
                    "label": label,
                    "source": source,
                    "portfolio": view_portfolio,
                    "signals": [self._serialize_candidate(candidate) for candidate in view_signals[:18]],
                    "trades": view_trades[:25],
                    "performance": {
                        "realized_pnl": realized_pnl,
                        "unrealized_pnl": unrealized_pnl,
                        "total_pnl": realized_pnl + unrealized_pnl,
                        "win_rate": win_rate,
                        "total_trades": len(view_positions) + len(view_closed),
                        "signals": len(view_signals),
                        "open_positions": len(view_positions),
                        "exposure": exposure,
                        "closed_positions": len(view_closed),
                    },
                }
            )

        return views

    def _serialize_position(self, position: PositionState) -> dict[str, Any]:
        return {
            "market": position.market_question,
            "side": position.outcome,
            "shares": position.shares,
            "entry": position.entry_price,
            "current": position.current_price,
            "pnl": position.unrealized_pnl,
            "source": position.strategy_name,
            "opened_at": position.opened_at.isoformat(),
        }

    def _serialize_candidate(self, candidate: Any) -> dict[str, Any]:
        return {
            "source": candidate.strategy_name,
            "market": candidate.market_snapshot.question if candidate.market_snapshot else candidate.market_id,
            "action": candidate.direction.value,
            "confidence": min(max(float(candidate.edge_estimate or 0.0) * 4.0, 0.05), 0.99),
            "edge": float(candidate.edge_estimate or 0.0) * 100.0,
            "size": 0.0,
            "whale": False,
            "reasoning": candidate.reasoning,
        }

    def _recent_trade_rows(self, limit: int = 20) -> list[dict[str, Any]]:
        summary = self.metrics_store.llm_summary(recent_scans=12, recent_closes=40)
        fills = summary.get("recent_fills", [])
        question_map = self._market_question_map()
        open_position_map = {
            (position.market_id, position.strategy_name): position
            for position in self.state.snapshot().positions
        }
        closed_map: dict[tuple[str, str], dict[str, Any]] = {}
        for event in self.state.closed_positions(120):
            closed_map[(event.get("market_id", ""), event.get("strategy_name", ""))] = event

        rows: list[dict[str, Any]] = []
        for fill in fills:
            if not fill.get("executed"):
                continue
            market_id = fill.get("market_id") or ""
            strategy_name = fill.get("strategy_name") or ""
            open_position = open_position_map.get((market_id, strategy_name))
            closed_event = closed_map.get((market_id, strategy_name))
            pnl = None
            if closed_event is not None:
                pnl = float(closed_event.get("realized_pnl", 0.0) or 0.0)
            elif open_position is not None:
                pnl = float(open_position.unrealized_pnl)

            fill_price = float(fill.get("fill_price", 0.0) or 0.0)
            shares = float(fill.get("shares", 0.0) or 0.0)
            rows.append(
                {
                    "time": fill.get("timestamp"),
                    "source": strategy_name,
                    "market": fill.get("market_question") or question_map.get(market_id) or market_id,
                    "side": (fill.get("direction") or fill.get("outcome") or "").replace("_", " ").upper(),
                    "price": fill_price,
                    "usd": fill_price * shares,
                    "pnl": pnl,
                    "edge": float(fill.get("edge_estimate", 0.0) or 0.0),
                    "rationale": fill.get("rationale") or "",
                }
            )

        return rows[:limit]

    def _market_question_map(self) -> dict[str, str]:
        question_map = {market.market_id: market.question for market in self._last_markets}
        for position in self.state.snapshot().positions:
            question_map[position.market_id] = position.market_question
        for event in self.state.closed_positions(120):
            market_id = event.get("market_id")
            question = event.get("market_question")
            if market_id and question:
                question_map[market_id] = question
        return question_map

    def _build_performance(self, report: ScanCycleReport | None) -> dict[str, Any]:
        portfolio = self.state.snapshot()
        state_perf = self.state.performance_summary()
        return {
            "scan_count": self._scan_count,
            "quiet_cycles": self._quiet_cycles,
            "quiet_cycle_rate": (self._quiet_cycles / self._scan_count * 100) if self._scan_count else 0.0,
            "total_candidates": self._total_candidates,
            "total_validated": self._total_validated,
            "total_executed": self._total_executed,
            "avg_candidates_per_scan": (self._total_candidates / self._scan_count) if self._scan_count else 0.0,
            "execution_rate": (self._total_executed / self._total_validated * 100) if self._total_validated else 0.0,
            "open_positions": portfolio.position_count,
            "available_capital": portfolio.available_capital,
            "deployed_capital": portfolio.deployed_capital,
            "unrealized_pnl": portfolio.total_unrealized_pnl,
            "realized_pnl": portfolio.total_realized_pnl,
            **state_perf,
            "recent_cycle": {
                "markets_scanned": report.markets_scanned if report else 0,
                "markets_after_filter": report.markets_after_filter if report else 0,
                "candidates_generated": report.candidates_generated if report else 0,
                "executions_succeeded": report.executions_succeeded if report else 0,
            },
        }

    def _build_blockers(self, report: ScanCycleReport | None) -> list[dict[str, str]]:
        if report is None:
            return [
                {
                    "title": "Opus runtime booting",
                    "detail": "The isolated runtime has not finished its first scan yet.",
                    "severity": "warn",
                }
            ]

        blockers: list[dict[str, str]] = []
        if report.zero_trade_explanation:
            historical_trade_count = self.state.snapshot().position_count + self.state.performance_summary().get(
                "closed_positions", 0
            )
            blockers.append(
                {
                    "title": "No new Opus trades this scan"
                    if historical_trade_count > 0
                    else "No Opus trades yet",
                    "detail": report.zero_trade_explanation,
                    "severity": "warn",
                }
            )
        if report.errors:
            blockers.append(
                {
                    "title": "Runtime error",
                    "detail": report.errors[-1],
                    "severity": "bad",
                }
            )
        if report.candidates_generated == 0 and not report.errors:
            blockers.append(
                {
                    "title": "No current candidates",
                    "detail": "The isolated runtime completed its last scan, but no relationship opportunities cleared the current thresholds.",
                    "severity": "warn",
                }
            )
        if not blockers:
            blockers.append(
                {
                    "title": "No dominant blocker detected",
                    "detail": "The isolated runtime completed its last scan without a surfaced blocker.",
                    "severity": "good",
                }
            )
        return blockers

    def _append_metrics_log(self, report: ScanCycleReport) -> None:
        METRICS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": utc_now().isoformat(),
            "scan": self._scan_count,
            "markets_scanned": report.markets_scanned,
            "markets_after_filter": report.markets_after_filter,
            "candidates_generated": report.candidates_generated,
            "candidates_validated": report.candidates_validated,
            "executions_succeeded": report.executions_succeeded,
            "executions_failed": report.executions_failed,
            "top_rejection_reason": report.top_rejection_reason,
            "top_allocation_rejection_reason": report.top_allocation_rejection_reason,
            "zero_trade_explanation": report.zero_trade_explanation,
            "capital_total": self.state.snapshot().total_capital,
            "capital_available": self.state.snapshot().available_capital,
            "open_positions": self.state.snapshot().position_count,
        }
        try:
            with METRICS_LOG_PATH.open("a") as handle:
                handle.write(json.dumps(entry) + "\n")
        except OSError as exc:
            self._recent_errors.append(f"metrics_log_failed:{exc}")
            self._recent_errors = self._recent_errors[-20:]

    def llm_context(self) -> dict[str, Any]:
        return {
            "summary": self.get_status()["summary"],
            "performance": self._build_performance(self._last_report),
            "health": self._build_health(self._last_report),
            "diagnostics": self._build_diagnostics(self._last_report),
            "provider_cards": [card.__dict__ for card in self.enricher.provider_cards()],
            "blockers": self._build_blockers(self._last_report),
            "compact_metrics_store": self.metrics_store.llm_summary(recent_scans=24, recent_closes=20),
            "recent_closed_positions": self.state.closed_positions(20),
            "recent_errors": list(self._recent_errors[-12:]),
            "snapshot_reports": self.snapshot_store.list_recent(8),
            "metrics_log_path": str(METRICS_LOG_PATH),
            "metrics_db_path": str(METRICS_DB_PATH),
        }
