#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.weather_edge_backtest import (
    _group_summary,
    _rolling_window_summary,
    _simulate_binary_backtest,
    _split_half_summary,
)
from engine.weather_edge_config import DEFAULT_BANKROLL_USD, KELLY_FRACTION, MAX_POSITION_FRACTION


CURRENT_ALLOWED_METRICS = ("temperature",)
CURRENT_ALLOWED_LEAD_TIMES_HOURS = (12,)
CURRENT_ALLOWED_REGIONS_BY_LEAD_TIME = {12: ("coastal",)}
CURRENT_MIN_EDGE = 0.05
CURRENT_MIN_MODEL_AGREEMENT = 0.70
CURRENT_MIN_MARKET_VOLUME = 0.0
CURRENT_SELECT_TOP_CONTRACT_PER_EVENT_HORIZON = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a separate current-regime weather-edge backtest from an existing scored report.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--source-report-json", required=True)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--bankroll-usd", type=float, default=DEFAULT_BANKROLL_USD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root)
    research_root = repo_root / "research" / "weather"
    output_root = Path(args.output_root) if args.output_root else repo_root / "output" / "weather_edge_current_regime_v1"
    source_report_json = Path(args.source_report_json).expanduser().resolve()
    source_report = json.loads(source_report_json.read_text())

    scored_rows = [_normalize_row(row) for row in source_report.get("rows") or []]
    eligible_rows = [row for row in scored_rows if _passes_current_regime_filter(row)]
    selected_rows = _select_trade_rows(eligible_rows)
    backtest = _simulate_binary_backtest(selected_rows, bankroll_usd=float(args.bankroll_usd))

    summary = {
        "totals": {
            "rows_scored": len(scored_rows),
            "rows_eligible": len(eligible_rows),
            "markets_scored": len({row["market_id"] for row in scored_rows}),
            "markets_eligible": len({row["market_id"] for row in eligible_rows}),
            "trade_rows_selected": len(selected_rows),
        },
        "rules": {
            "allowed_metrics": list(CURRENT_ALLOWED_METRICS),
            "allowed_lead_times_hours": list(CURRENT_ALLOWED_LEAD_TIMES_HOURS),
            "allowed_regions_by_lead_time": {str(key): list(value) for key, value in CURRENT_ALLOWED_REGIONS_BY_LEAD_TIME.items()},
            "min_edge": CURRENT_MIN_EDGE,
            "min_model_agreement": CURRENT_MIN_MODEL_AGREEMENT,
            "min_market_volume": CURRENT_MIN_MARKET_VOLUME,
            "select_top_contract_per_event_horizon": CURRENT_SELECT_TOP_CONTRACT_PER_EVENT_HORIZON,
            "kelly_fraction": KELLY_FRACTION,
            "max_position_fraction": MAX_POSITION_FRACTION,
            "bankroll_usd": float(args.bankroll_usd),
        },
        "breakdowns": {
            "metric_type": _group_summary(scored_rows, "metric_type"),
            "lead_time_hours": _group_summary(scored_rows, "lead_time_hours"),
            "region": _group_summary(scored_rows, "region"),
            "season": _group_summary(scored_rows, "season"),
        },
        "backtest": backtest,
        "split_half": _split_half_summary(selected_rows, bankroll_usd=float(args.bankroll_usd)),
        "quarters_15d": _rolling_window_summary(selected_rows, bankroll_usd=float(args.bankroll_usd)),
        "notes": _build_notes(scored_rows=scored_rows, eligible_rows=eligible_rows, selected_rows=selected_rows, source_report_json=source_report_json),
    }

    run_id = f"weather_edge_current_regime_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_v1"
    run_root = output_root / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    report_json = run_root / "weather_edge_current_regime_report.json"
    report_md = run_root / "weather_edge_current_regime_report.md"
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_report_json": str(source_report_json),
        "source_generated_at": source_report.get("metadata", {}).get("generated_at"),
        "source_lookback_days": source_report.get("metadata", {}).get("lookback_days"),
        "source_market_offset": source_report.get("metadata", {}).get("market_offset"),
        "source_markets_loaded": source_report.get("metadata", {}).get("markets_loaded"),
        "bundle_dir": source_report.get("metadata", {}).get("bundle_dir"),
        "bankroll_usd": float(args.bankroll_usd),
    }
    report_json.write_text(json.dumps({"metadata": metadata, "summary": _json_ready(summary), "rows": _json_ready(selected_rows)}, indent=2))
    report_md.write_text(_render_markdown(metadata=metadata, summary=summary))
    checkpoint_id = _record_checkpoint(research_root=research_root, report_json=report_json, report_md=report_md, metadata=metadata, summary=summary)
    print(json.dumps({"run_root": str(run_root), "report_json": str(report_json), "report_md": str(report_md), "checkpoint_id": checkpoint_id}, indent=2))
    return 0


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    value = normalized.get("resolution_time")
    if isinstance(value, str):
        normalized["resolution_time"] = datetime.fromisoformat(value)
    return normalized


def _passes_current_regime_filter(row: dict[str, Any]) -> bool:
    if row.get("metric_type") not in CURRENT_ALLOWED_METRICS:
        return False
    lead_time_hours = int(row.get("lead_time_hours") or 0)
    if lead_time_hours not in CURRENT_ALLOWED_LEAD_TIMES_HOURS:
        return False
    allowed_regions = CURRENT_ALLOWED_REGIONS_BY_LEAD_TIME.get(lead_time_hours)
    if allowed_regions and row.get("region") not in allowed_regions:
        return False
    if abs(float(row.get("raw_edge") or 0.0)) < CURRENT_MIN_EDGE:
        return False
    if float(row.get("model_agreement") or 0.0) < CURRENT_MIN_MODEL_AGREEMENT:
        return False
    if float(row.get("volume_clob") or 0.0) < CURRENT_MIN_MARKET_VOLUME:
        return False
    return True


def _select_trade_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not CURRENT_SELECT_TOP_CONTRACT_PER_EVENT_HORIZON:
        return list(rows)
    selected_by_event: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        event_key = (
            str(row.get("city") or ""),
            str(row.get("target_date") or ""),
            int(row.get("lead_time_hours") or 0),
        )
        current = selected_by_event.get(event_key)
        if current is None or abs(float(row.get("raw_edge") or 0.0)) > abs(float(current.get("raw_edge") or 0.0)):
            selected_by_event[event_key] = row
    return sorted(
        selected_by_event.values(),
        key=lambda row: (row["resolution_time"], row["city"], row["lead_time_hours"], row["market_id"]),
    )


def _build_notes(
    *,
    scored_rows: list[dict[str, Any]],
    eligible_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    source_report_json: Path,
) -> list[str]:
    notes = [
        f"Derived from source report: {source_report_json}",
        "This is a separate current-regime lane; the legacy 24h/48h weather-edge backtest remains frozen.",
        "Current-regime rule set is designed around the lead times the newest markets actually expose, instead of the historical 24h/48h assumption.",
    ]
    metric_types = sorted({row["metric_type"] for row in scored_rows})
    notes.append(f"Scored metric types present: {', '.join(metric_types) or 'none'}.")
    if not eligible_rows:
        notes.append("No rows passed the current-regime rule filter.")
    elif len(selected_rows) < len(eligible_rows):
        notes.append(
            f"Current-regime filter produced {len(eligible_rows)} candidate rows, narrowed to {len(selected_rows)} event-level trades after contract competition."
        )
    return notes


def _record_checkpoint(
    *,
    research_root: Path,
    report_json: Path,
    report_md: Path,
    metadata: dict[str, Any],
    summary: dict[str, Any],
) -> str:
    checkpoints_root = research_root / "checkpoints"
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    index_path = checkpoints_root / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else []

    checkpoint_id = f"weather-edge-current-regime-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    manifest_dir = checkpoints_root / checkpoint_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint_id": checkpoint_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "weather_edge_current_regime",
        "category": "weather_edge",
        "summary": "Separate current-regime weather-edge backtest lane using the frozen legacy model and updated lead-time rules.",
        "artifacts": {
            "report_json": str(report_json),
            "report_md": str(report_md),
            "source_report_json": metadata["source_report_json"],
        },
        "metrics": {
            "rows_scored": summary["totals"]["rows_scored"],
            "rows_eligible": summary["totals"]["rows_eligible"],
            "trade_rows_selected": summary["totals"]["trade_rows_selected"],
            "trade_count": summary["backtest"]["trade_count"],
            "win_rate": summary["backtest"]["win_rate"],
            "total_net_bps": summary["backtest"]["total_net_bps"],
            "max_drawdown": summary["backtest"]["max_drawdown"],
        },
    }
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    index.append(manifest)
    index_path.write_text(json.dumps(index, indent=2))

    diary_path = research_root / "diary.md"
    with diary_path.open("a") as handle:
        handle.write(
            "\n\n"
            f"## {datetime.now(timezone.utc).strftime('%Y-%m-%d')} - Weather edge current-regime lane\n\n"
            f"- Checkpoint: `{checkpoint_id}`\n"
            f"- Source report: `{metadata['source_report_json']}`\n"
            f"- Source market offset: `{metadata['source_market_offset']}`\n"
            f"- Source markets loaded: `{metadata['source_markets_loaded']}`\n"
            f"- Allowed lead times: `{CURRENT_ALLOWED_LEAD_TIMES_HOURS}`\n"
            f"- Allowed regions by lead time: `{CURRENT_ALLOWED_REGIONS_BY_LEAD_TIME}`\n"
            f"- Rows scored: `{summary['totals']['rows_scored']}`\n"
            f"- Eligible rows: `{summary['totals']['rows_eligible']}`\n"
            f"- Event-level trades selected: `{summary['totals']['trade_rows_selected']}`\n"
            f"- Trades: `{summary['backtest']['trade_count']}`\n"
            f"- Win rate: `{summary['backtest']['win_rate']}`\n"
            f"- Total net bps: `{summary['backtest']['total_net_bps']}`\n"
            f"- Total net USD: `{summary['backtest']['total_net_usd']}`\n"
            f"- Max drawdown: `{summary['backtest']['max_drawdown']}`\n"
            f"- Report: `{report_md}`\n"
        )
    return checkpoint_id


def _render_markdown(*, metadata: dict[str, Any], summary: dict[str, Any]) -> str:
    lines = [
        "# Weather Edge Current-Regime Report",
        "",
        f"Generated at: `{metadata['generated_at']}`",
        f"Source report: `{metadata['source_report_json']}`",
        f"Source market offset: `{metadata['source_market_offset']}`",
        f"Source markets loaded: `{metadata['source_markets_loaded']}`",
        "",
        "## Rules",
        "",
        f"- Allowed lead times: `{CURRENT_ALLOWED_LEAD_TIMES_HOURS}`",
        f"- Allowed regions by lead time: `{CURRENT_ALLOWED_REGIONS_BY_LEAD_TIME}`",
        f"- Min edge: `{CURRENT_MIN_EDGE}`",
        f"- Min model agreement: `{CURRENT_MIN_MODEL_AGREEMENT}`",
        f"- Kelly fraction: `{KELLY_FRACTION}`",
        f"- Max position fraction: `{MAX_POSITION_FRACTION}`",
        "",
        "## Totals",
        "",
        f"- Rows scored: `{summary['totals']['rows_scored']}`",
        f"- Rows eligible: `{summary['totals']['rows_eligible']}`",
        f"- Markets scored: `{summary['totals']['markets_scored']}`",
        f"- Markets eligible: `{summary['totals']['markets_eligible']}`",
        f"- Event-level trades selected: `{summary['totals']['trade_rows_selected']}`",
        "",
        "## Backtest",
        "",
        f"- Trades: `{summary['backtest']['trade_count']}`",
        f"- Win count / loss count: `{summary['backtest']['win_count']}` / `{summary['backtest']['loss_count']}`",
        f"- Win rate: `{summary['backtest']['win_rate']}`",
        f"- Total net bps: `{summary['backtest']['total_net_bps']}`",
        f"- Total net USD: `{summary['backtest']['total_net_usd']}`",
        f"- Max drawdown: `{summary['backtest']['max_drawdown']}`",
        "",
        "## Breakdowns",
        "",
    ]
    for label, rows in summary["breakdowns"].items():
        lines.append(f"### {label}")
        lines.append("")
        if not rows:
            lines.append("- none")
            lines.append("")
            continue
        for row in rows:
            lines.append(
                f"- `{row['group']}` | rows `{row['rows']}` | avg edge `{row['avg_absolute_edge']}` | "
                f"avg realized `{row['avg_realized_value']}` | positive share `{row['positive_share']}`"
            )
        lines.append("")

    lines.extend(["## Notes", ""])
    for note in summary["notes"]:
        lines.append(f"- {note}")
    return "\n".join(lines)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(current) for key, current in value.items()}
    if isinstance(value, list):
        return [_json_ready(current) for current in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
