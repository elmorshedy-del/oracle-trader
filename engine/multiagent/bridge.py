from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from .adapters import (
    bare_context_from_legacy,
    normalized_market_from_legacy,
    portfolio_snapshot_from_legacy,
)
from .config import OrchestratorConfig
from .contracts import dataclass_to_dict
from .enums import ModuleStatus

if TYPE_CHECKING:
    from engine.pipeline import Pipeline


MAX_MARKET_PREVIEW = 12
MAX_BLOCKERS = 6
DOMINANT_SOURCE_EXPOSURE_PCT = 0.60
LOW_TRADEABLE_RATIO = 0.50


def build_legacy_bridge_status(
    pipeline: "Pipeline | None",
    config: OrchestratorConfig | None = None,
) -> dict[str, Any]:
    config = config or OrchestratorConfig()
    defaults = dataclass_to_dict(config)

    if pipeline is None:
        return {
            "bridge": {
                "mode": "legacy_readonly",
                "state": "booting",
                "next_step": "wait_for_pipeline_startup",
            },
            "summary": {
                "scan_count": 0,
                "active_markets": 0,
                "open_positions": 0,
                "top_blocker": "Pipeline is still starting",
            },
            "defaults": defaults,
            "portfolio": {},
            "health": {},
            "diagnostics": {},
            "market_mix": {},
            "market_preview": [],
            "module_cards": [],
            "strategy_cards": [],
            "comparison_views": [],
            "blockers": [
                {
                    "title": "Pipeline booting",
                    "detail": "The legacy Oracle runtime is not attached yet, so the bridge is waiting for the background pipeline to start.",
                    "severity": "warn",
                }
            ],
        }

    state = _safe_pipeline_state(pipeline)
    health = _safe_health(pipeline, state)
    diagnostics = state.get("diagnostics") or getattr(pipeline, "_latest_diagnostics", {}) or {}
    portfolio = portfolio_snapshot_from_legacy(
        pipeline.trader.portfolio,
        config.risk_limits,
    )
    blockers = _build_blockers(state, health, diagnostics, portfolio)
    comparison_views = _build_comparison_summaries(state.get("comparison_views", {}))

    return {
        "bridge": {
            "mode": "legacy_readonly",
            "state": "attached",
            "next_step": "migrate_weather_and_crypto_strategies",
        },
        "summary": {
            "scan_count": state.get("scan_count", 0),
            "active_markets": state.get("active_markets", 0),
            "open_positions": portfolio.position_count,
            "top_blocker": blockers[0]["title"] if blockers else "No dominant blocker detected",
        },
        "defaults": defaults,
        "portfolio": dataclass_to_dict(portfolio),
        "health": health,
        "diagnostics": diagnostics,
        "market_mix": _build_market_mix(getattr(pipeline, "_markets", [])),
        "market_preview": _build_market_preview(getattr(pipeline, "_markets", [])),
        "module_cards": _build_module_cards(state, health, diagnostics, portfolio),
        "strategy_cards": _build_strategy_cards(state.get("strategies", {}), health),
        "comparison_views": comparison_views,
        "blockers": blockers,
    }


def _safe_pipeline_state(pipeline: "Pipeline") -> dict[str, Any]:
    try:
        return pipeline.get_state()
    except Exception as exc:
        return {
            "scan_count": 0,
            "active_markets": 0,
            "uptime_human": "error",
            "strategies": {},
            "comparison_views": {},
            "diagnostics": {},
            "errors": [f"state_read_failed: {exc}"],
        }


def _safe_health(pipeline: "Pipeline", state: dict[str, Any]) -> dict[str, Any]:
    try:
        if hasattr(pipeline, "health"):
            return pipeline.health.get_health_report()
    except Exception as exc:
        return {
            "overall_status": "failed",
            "recent_errors": [f"health_read_failed: {exc}"],
        }
    return state.get("health", {})


def _build_market_mix(markets: list[Any]) -> dict[str, int]:
    category_counts = Counter()
    for market in markets:
        normalized = normalized_market_from_legacy(market)
        category_counts[normalized.category.value] += 1
    return dict(sorted(category_counts.items(), key=lambda item: item[1], reverse=True))


def _build_market_preview(markets: list[Any]) -> list[dict[str, Any]]:
    preview: list[dict[str, Any]] = []
    ranked = sorted(markets, key=lambda item: item.volume_24h, reverse=True)[:MAX_MARKET_PREVIEW]
    for market in ranked:
        normalized = normalized_market_from_legacy(market)
        context = bare_context_from_legacy(market)
        preview.append(
            {
                "market_id": normalized.market_id,
                "question": normalized.question,
                "category": normalized.category.value,
                "yes_price": normalized.yes_price,
                "no_price": normalized.no_price,
                "liquidity": normalized.liquidity,
                "volume_24h": normalized.volume_24h,
                "hours_to_resolution": context.hours_to_resolution,
                "tags": list(normalized.tags[:4]),
            }
        )
    return preview


def _build_module_cards(
    state: dict[str, Any],
    health: dict[str, Any],
    diagnostics: dict[str, Any],
    portfolio: Any,
) -> list[dict[str, Any]]:
    active_markets = state.get("active_markets", 0)
    scan_count = state.get("scan_count", 0)
    gamma_status = ((health.get("apis") or {}).get("gamma") or {}).get("status", "unknown")
    scan_status = ((health.get("scan") or {}).get("status")) or "unknown"
    diag_status = ModuleStatus.HEALTHY.value if diagnostics else ModuleStatus.DEGRADED.value

    cards = [
        {
            "name": "legacy_pipeline",
            "label": "Legacy pipeline",
            "status": _merge_statuses(health.get("overall_status"), scan_status),
            "detail": f"{scan_count} scans completed | {state.get('uptime_human', 'unknown uptime')}",
            "items_in": active_markets,
            "items_out": portfolio.position_count,
        },
        {
            "name": "scanner",
            "label": "Market scanner",
            "status": _status_or_degraded(gamma_status, active_markets > 0),
            "detail": f"{active_markets} active markets in working set",
            "items_in": active_markets,
            "items_out": active_markets,
        },
        {
            "name": "portfolio_state",
            "label": "Portfolio state",
            "status": ModuleStatus.HEALTHY.value,
            "detail": (
                f"${portfolio.available_capital:.2f} free | "
                f"{portfolio.position_count} open positions | "
                f"{portfolio.capital_utilization_pct * 100:.1f}% utilized"
            ),
            "items_in": portfolio.position_count,
            "items_out": portfolio.position_count,
        },
        {
            "name": "diagnostics",
            "label": "Cycle diagnostics",
            "status": diag_status,
            "detail": _diagnostic_detail(diagnostics),
            "items_in": diagnostics.get("markets_total", 0),
            "items_out": diagnostics.get("executed", 0),
        },
    ]
    return cards


def _build_strategy_cards(
    strategies: dict[str, Any],
    health: dict[str, Any],
) -> list[dict[str, Any]]:
    strategy_health = health.get("strategies") or {}
    cards: list[dict[str, Any]] = []
    for name, stats in sorted(strategies.items()):
        health_entry = strategy_health.get(name) or {}
        cards.append(
            {
                "name": name,
                "status": _status_or_degraded(
                    health_entry.get("status", "unknown"),
                    (health_entry.get("runs", 0) or stats.get("scans_completed", 0)) > 0,
                ),
                "runs": health_entry.get("runs", stats.get("scans_completed", 0)),
                "errors": health_entry.get("errors", stats.get("errors", 0)),
                "signals": health_entry.get("total_signals", stats.get("signals_generated", 0)),
                "last_error": health_entry.get("last_error"),
            }
        )
    return cards


def _build_comparison_summaries(comparison_views: dict[str, Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for key, view in comparison_views.items():
        portfolio = view.get("portfolio", {})
        summaries.append(
            {
                "key": key,
                "label": view.get("label", key),
                "source": view.get("source", "unknown"),
                "total_value": portfolio.get("total_value", 0.0),
                "total_pnl": portfolio.get("total_pnl", 0.0),
                "cash": portfolio.get("cash", 0.0),
                "open_positions": len(portfolio.get("positions", [])),
                "total_trades": portfolio.get("total_trades", 0),
                "win_rate": portfolio.get("win_rate", 0.0),
            }
        )
    return sorted(summaries, key=lambda item: item["label"])


def _build_blockers(
    state: dict[str, Any],
    health: dict[str, Any],
    diagnostics: dict[str, Any],
    portfolio: Any,
) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []

    if (health.get("scan") or {}).get("status") in {"degraded", "unhealthy"}:
        blockers.append(
            {
                "title": "Scan cadence degraded",
                "detail": f"Scan health is {health['scan']['status']}. The pipeline may be behind or not cycling fast enough.",
                "severity": "bad",
            }
        )

    stale_data = health.get("stale_data") or []
    if stale_data:
        stale_summary = ", ".join(
            f"{item['type']} ({item['age_minutes']}m)" for item in stale_data[:3]
        )
        blockers.append(
            {
                "title": "Stale data detected",
                "detail": f"Freshness checks flagged: {stale_summary}. Signals may be blocked before validation.",
                "severity": "bad",
            }
        )

    if diagnostics:
        total_signals = sum((diagnostics.get("signals_by_strategy") or {}).values())
        if diagnostics.get("executed", 0) == 0:
            if total_signals == 0:
                blockers.append(
                    {
                        "title": "No strategies emitted candidates",
                        "detail": (
                            f"{diagnostics.get('markets_tradeable', 0)} tradeable markets were scanned but no strategy produced a candidate in the last cycle."
                        ),
                        "severity": "warn",
                    }
                )
            elif diagnostics.get("filtered_signals"):
                top_reason, count = _top_entry(diagnostics["filtered_signals"])
                blockers.append(
                    {
                        "title": "Signals blocked before execution",
                        "detail": f"{count} signals were filtered on `{top_reason}` in the last scan.",
                        "severity": "warn",
                    }
                )

            tradeable_ratio = _tradeable_ratio(diagnostics)
            if tradeable_ratio < LOW_TRADEABLE_RATIO:
                blockers.append(
                    {
                        "title": "Working set is saturated by held markets",
                        "detail": (
                            f"Only {diagnostics.get('markets_tradeable', 0)} of {diagnostics.get('markets_total', 0)} markets were tradeable in the last scan."
                        ),
                        "severity": "warn",
                    }
                )

        dominant_source = _dominant_source(diagnostics)
        if dominant_source:
            source, pct = dominant_source
            blockers.append(
                {
                    "title": f"Exposure dominated by {source}",
                    "detail": f"{pct:.0%} of deployed exposure currently belongs to `{source}`. Other sleeves may be starved.",
                    "severity": "warn",
                }
            )

    if portfolio.available_capital < portfolio.reserved_capital:
        blockers.append(
            {
                "title": "Reserve cushion is already consumed",
                "detail": (
                    f"Available capital is ${portfolio.available_capital:.2f}, below the configured reserve target of ${portfolio.reserved_capital:.2f}."
                ),
                "severity": "bad",
            }
        )

    recent_errors = health.get("recent_errors") or state.get("errors") or []
    if recent_errors:
        blockers.append(
            {
                "title": "Runtime errors present",
                "detail": str(recent_errors[-1])[:220],
                "severity": "bad",
            }
        )

    return blockers[:MAX_BLOCKERS]


def _diagnostic_detail(diagnostics: dict[str, Any]) -> str:
    if not diagnostics:
        return "No compact diagnostic cycle has been written yet."
    return (
        f"Scan {diagnostics.get('scan', 0)} | "
        f"{diagnostics.get('executed', 0)} executed | "
        f"{diagnostics.get('exits', 0)} exits | "
        f"{diagnostics.get('resolved', 0)} resolved"
    )


def _status_or_degraded(raw_status: str | None, has_signal: bool) -> str:
    normalized = _normalize_status(raw_status)
    if normalized == ModuleStatus.HEALTHY.value and has_signal:
        return normalized
    if normalized == ModuleStatus.FAILED.value:
        return normalized
    return ModuleStatus.DEGRADED.value


def _merge_statuses(*raw_statuses: str | None) -> str:
    normalized = [_normalize_status(item) for item in raw_statuses]
    if ModuleStatus.FAILED.value in normalized:
        return ModuleStatus.FAILED.value
    if ModuleStatus.DEGRADED.value in normalized:
        return ModuleStatus.DEGRADED.value
    return ModuleStatus.HEALTHY.value


def _normalize_status(raw_status: str | None) -> str:
    if raw_status in {ModuleStatus.HEALTHY.value, "healthy"}:
        return ModuleStatus.HEALTHY.value
    if raw_status in {None, ModuleStatus.DEGRADED.value, "degraded", "unknown"}:
        return ModuleStatus.DEGRADED.value
    return ModuleStatus.FAILED.value


def _tradeable_ratio(diagnostics: dict[str, Any]) -> float:
    total = diagnostics.get("markets_total", 0)
    if not total:
        return 0.0
    return diagnostics.get("markets_tradeable", 0) / total


def _dominant_source(diagnostics: dict[str, Any]) -> tuple[str, float] | None:
    exposure = ((diagnostics.get("portfolio") or {}).get("exposure_by_source")) or {}
    if not exposure:
        return None
    total = sum(exposure.values())
    if total <= 0:
        return None
    source, amount = max(exposure.items(), key=lambda item: item[1])
    pct = amount / total
    if pct < DOMINANT_SOURCE_EXPOSURE_PCT:
        return None
    return source, pct


def _top_entry(payload: dict[str, int]) -> tuple[str, int]:
    if not payload:
        return "unknown", 0
    return max(payload.items(), key=lambda item: item[1])
