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
    parser = argparse.ArgumentParser(description="Aggregate the current-regime weather-edge lane across multiple scored source reports.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--label", required=True)
    parser.add_argument("--source-report-json", action="append", required=True)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--bankroll-usd", type=float, default=DEFAULT_BANKROLL_USD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root)
    research_root = repo_root / "research" / "weather"
    output_root = Path(args.output_root) if args.output_root else repo_root / "output" / "weather_edge_current_regime_multislice"
    source_paths = [Path(path).expanduser().resolve() for path in args.source_report_json]

    per_source = []
    combined_scored_rows: list[dict[str, Any]] = []
    combined_eligible_rows: list[dict[str, Any]] = []
    for source_path in source_paths:
        source_report = json.loads(source_path.read_text())
        metadata = source_report.get("metadata", {})
        scored_rows = [_normalize_row(row) for row in source_report.get("rows") or []]
        eligible_rows = [row for row in scored_rows if _passes_current_regime_filter(row)]
        selected_rows = _select_trade_rows(eligible_rows)
        backtest = _simulate_binary_backtest(selected_rows, bankroll_usd=float(args.bankroll_usd))
        per_source.append(
            {
                "source_report_json": str(source_path),
                "source_market_offset": metadata.get("market_offset"),
                "source_markets_loaded": metadata.get("markets_loaded"),
                "rows_scored": len(scored_rows),
                "rows_eligible": len(eligible_rows),
                "trade_rows_selected": len(selected_rows),
                "backtest": backtest,
            }
        )
        combined_scored_rows.extend(scored_rows)
        combined_eligible_rows.extend(eligible_rows)

    combined_selected_rows = _select_trade_rows(combined_eligible_rows)
    combined_backtest = _simulate_binary_backtest(combined_selected_rows, bankroll_usd=float(args.bankroll_usd))

    summary = {
        "label": args.label,
        "source_count": len(source_paths),
        "source_reports": per_source,
        "totals": {
            "rows_scored": len(combined_scored_rows),
            "rows_eligible": len(combined_eligible_rows),
            "markets_scored": len({row["market_id"] for row in combined_scored_rows}),
            "markets_eligible": len({row["market_id"] for row in combined_eligible_rows}),
            "trade_rows_selected": len(combined_selected_rows),
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
            "metric_type": _group_summary(combined_scored_rows, "metric_type"),
            "lead_time_hours": _group_summary(combined_scored_rows, "lead_time_hours"),
            "region": _group_summary(combined_scored_rows, "region"),
            "season": _group_summary(combined_scored_rows, "season"),
        },
        "backtest": combined_backtest,
        "split_half": _split_half_summary(combined_selected_rows, bankroll_usd=float(args.bankroll_usd)),
        "quarters_15d": _rolling_window_summary(combined_selected_rows, bankroll_usd=float(args.bankroll_usd)),
        "notes": _build_notes(source_paths=source_paths, combined_selected_rows=combined_selected_rows),
    }

    run_id = f"weather_edge_current_regime_multislice_{args.label}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_v1"
    run_root = output_root / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    report_json = run_root / "weather_edge_current_regime_multislice_report.json"
    report_md = run_root / "weather_edge_current_regime_multislice_report.md"
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "source_report_jsons": [str(path) for path in source_paths],
        "bankroll_usd": float(args.bankroll_usd),
    }
    report_json.write_text(json.dumps({"metadata": metadata, "summary": _json_ready(summary), "rows": _json_ready(combined_selected_rows)}, indent=2))
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


def _build_notes(*, source_paths: list[Path], combined_selected_rows: list[dict[str, Any]]) -> list[str]:
    notes = [
        f"Derived from {len(source_paths)} source reports.",
        "This is a multi-slice aggregate for the separate current-regime lane; the legacy 24h/48h weather-edge backtest remains frozen.",
        "Current-regime rule set is temperature + 12h + coastal only, with the frozen edge/agreement/Kelly settings kept unchanged.",
    ]
    if len(combined_selected_rows) < 100:
        notes.append(f"Trade count remains below the requested 100-trade bar under frozen current-regime rules: {len(combined_selected_rows)}.")
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

    checkpoint_id = f"weather-edge-current-regime-multislice-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    manifest_dir = checkpoints_root / checkpoint_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint_id": checkpoint_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "weather_edge_current_regime_multislice",
        "category": "weather_edge",
        "summary": "Aggregate current-regime weather-edge backtest across multiple scored report slices using the frozen current-regime filter.",
        "artifacts": {
            "report_json": str(report_json),
            "report_md": str(report_md),
            "source_report_jsons": metadata["source_report_jsons"],
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
            f"## {datetime.now(timezone.utc).strftime('%Y-%m-%d')} - Weather edge current-regime multislice aggregate\n\n"
            f"- Checkpoint: `{checkpoint_id}`\n"
            f"- Label: `{metadata['label']}`\n"
            f"- Source reports: `{len(metadata['source_report_jsons'])}`\n"
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
        "# Weather Edge Current-Regime Multi-Slice Report",
        "",
        f"Generated at: `{metadata['generated_at']}`",
        f"Label: `{metadata['label']}`",
        f"Source reports: `{len(metadata['source_report_jsons'])}`",
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
        "## Aggregate Totals",
        "",
        f"- Rows scored: `{summary['totals']['rows_scored']}`",
        f"- Rows eligible: `{summary['totals']['rows_eligible']}`",
        f"- Markets scored: `{summary['totals']['markets_scored']}`",
        f"- Markets eligible: `{summary['totals']['markets_eligible']}`",
        f"- Event-level trades selected: `{summary['totals']['trade_rows_selected']}`",
        "",
        "## Aggregate Backtest",
        "",
        f"- Trades: `{summary['backtest']['trade_count']}`",
        f"- Win count / loss count: `{summary['backtest']['win_count']}` / `{summary['backtest']['loss_count']}`",
        f"- Win rate: `{summary['backtest']['win_rate']}`",
        f"- Total net bps: `{summary['backtest']['total_net_bps']}`",
        f"- Total net USD: `{summary['backtest']['total_net_usd']}`",
        f"- Max drawdown: `{summary['backtest']['max_drawdown']}`",
        "",
        "## Per-Source Summary",
        "",
    ]
    for source in summary["source_reports"]:
        lines.extend(
            [
                f"- `{source['source_report_json']}`",
                f"  - market offset: `{source['source_market_offset']}`",
                f"  - markets loaded: `{source['source_markets_loaded']}`",
                f"  - rows scored: `{source['rows_scored']}`",
                f"  - rows eligible: `{source['rows_eligible']}`",
                f"  - selected trades: `{source['trade_rows_selected']}`",
                f"  - total net bps: `{source['backtest']['total_net_bps']}`",
            ]
        )
    lines.extend(["", "## Breakdowns", ""])
    for group_name, rows in summary["breakdowns"].items():
        lines.append(f"### {group_name}")
        lines.append("")
        if not rows:
            lines.append("- none")
        else:
            for row in rows:
                lines.append(
                    f"- `{row['group']}` | rows `{row['rows']}` | avg edge `{row['avg_absolute_edge']}` | avg realized `{row['avg_realized_value']}` | positive share `{row['positive_share']}`"
                )
        lines.append("")
    lines.extend(["## Notes", ""])
    for note in summary["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _json_ready(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
