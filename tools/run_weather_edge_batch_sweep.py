#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.weather_edge_backtest import summarize_weather_edge
from engine.weather_edge_config import DEFAULT_LOOKBACK_DAYS, default_weather_edge_root, default_weather_research_root


DEFAULT_BATCH_SIZE = 250
DEFAULT_MAX_BATCHES = 0
PIPELINE_SCRIPT = REPO_ROOT / "tools" / "run_weather_edge_pipeline.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep the standalone weather edge pipeline in batches and aggregate the results.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-batches", type=int, default=DEFAULT_MAX_BATCHES, help="0 means keep running until an empty batch is returned.")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--cache-root", default="")
    parser.add_argument("--history-sources", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root)
    output_root = Path(args.output_root) if args.output_root else default_weather_edge_root(repo_root)
    cache_root = Path(args.cache_root) if args.cache_root else output_root / "cache"
    research_root = default_weather_research_root(repo_root)

    batch_results: list[dict] = []
    batch_index = 0
    while True:
        if args.max_batches and batch_index >= args.max_batches:
            break
        market_offset = batch_index * args.batch_size
        result = _run_batch(
            repo_root=repo_root,
            lookback_days=args.lookback_days,
            batch_size=args.batch_size,
            market_offset=market_offset,
            allow_network=bool(args.allow_network),
            output_root=output_root,
            cache_root=cache_root,
            history_sources=args.history_sources,
        )
        batch_results.append(result)
        if result["summary"]["totals"]["rows_scored"] == 0:
            break
        batch_index += 1

    non_empty_batches = [result for result in batch_results if result["summary"]["totals"]["rows_scored"] > 0]
    aggregate = _aggregate_results(non_empty_batches)
    aggregate_root = output_root / "aggregates" / f"weather_edge_full_sweep_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_v1"
    aggregate_root.mkdir(parents=True, exist_ok=True)
    report_json = aggregate_root / "weather_edge_report.json"
    report_md = aggregate_root / "weather_edge_report.md"

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lookback_days": args.lookback_days,
        "batch_size": args.batch_size,
        "batch_count": len(non_empty_batches),
        "cache_root": str(cache_root),
        "source_runs": [result["run_root"] for result in non_empty_batches],
    }
    report_json.write_text(json.dumps({"metadata": metadata, "summary": aggregate["summary"]}, indent=2, default=_json_default))
    report_md.write_text(_render_markdown(metadata=metadata, summary=aggregate["summary"]))
    checkpoint_id = _record_aggregate_checkpoint(
        research_root=research_root,
        report_json=report_json,
        report_md=report_md,
        metadata=metadata,
        summary=aggregate["summary"],
    )
    print(json.dumps({"report_json": str(report_json), "report_md": str(report_md), "checkpoint_id": checkpoint_id}, indent=2))
    return 0


def _run_batch(
    *,
    repo_root: Path,
    lookback_days: int,
    batch_size: int,
    market_offset: int,
    allow_network: bool,
    output_root: Path,
    cache_root: Path,
    history_sources: str,
) -> dict:
    command = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "--repo-root",
        str(repo_root),
        "--lookback-days",
        str(lookback_days),
        "--max-markets",
        str(batch_size),
        "--market-offset",
        str(market_offset),
        "--output-root",
        str(output_root),
        "--cache-root",
        str(cache_root),
    ]
    if allow_network:
        command.append("--allow-network")
    if history_sources:
        command.extend(["--history-sources", history_sources])
    process = subprocess.run(command, capture_output=True, text=True, check=True, cwd=str(repo_root))
    payload = json.loads(process.stdout)
    report_json = Path(payload["report_json"])
    report = json.loads(report_json.read_text())
    return {
        "run_root": payload["run_root"],
        "report_json": payload["report_json"],
        "summary": report["summary"],
        "rows": report.get("rows") or [],
    }


def _aggregate_results(batch_results: list[dict]) -> dict:
    from datetime import datetime

    rows = []
    for result in batch_results:
        for row in result["rows"]:
            copied = dict(row)
            if isinstance(copied.get("resolution_time"), str):
                copied["resolution_time"] = datetime.fromisoformat(copied["resolution_time"])
            rows.append(copied)
    summary = summarize_weather_edge(scored_rows=rows)
    return {"summary": summary}


def _render_markdown(*, metadata: dict, summary: dict) -> str:
    lines = [
        "# Weather Edge Batch Sweep",
        "",
        f"Generated at: `{metadata['generated_at']}`",
        f"Batch size: `{metadata['batch_size']}`",
        f"Batch count: `{metadata['batch_count']}`",
        "",
        "## Totals",
        "",
        f"- Rows scored: `{summary['totals']['rows_scored']}`",
        f"- Rows eligible: `{summary['totals']['rows_eligible']}`",
        f"- Event-level trades selected: `{summary['totals']['trade_rows_selected']}`",
        "",
        "## Backtest",
        "",
        f"- Trades: `{summary['backtest']['trade_count']}`",
        f"- Win rate: `{summary['backtest']['win_rate']}`",
        f"- Total net bps: `{summary['backtest']['total_net_bps']}`",
        f"- Total net USD: `{summary['backtest']['total_net_usd']}`",
        f"- Max drawdown: `{summary['backtest']['max_drawdown']}`",
        "",
        "## Lead Time Breakdown",
        "",
    ]
    for row in summary["breakdowns"]["lead_time_hours"]:
        lines.append(
            f"- `{row['group']}`h | rows `{row['rows']}` | avg realized `{row['avg_realized_value']}` | positive share `{row['positive_share']}`"
        )
    lines.extend(["", "## Notes", ""])
    for note in summary["notes"]:
        lines.append(f"- {note}")
    return "\n".join(lines)


def _record_aggregate_checkpoint(
    *,
    research_root: Path,
    report_json: Path,
    report_md: Path,
    metadata: dict,
    summary: dict,
) -> str:
    checkpoints_root = research_root / "checkpoints"
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    index_path = checkpoints_root / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else []

    checkpoint_id = f"weather-edge-batch-sweep-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    manifest_dir = checkpoints_root / checkpoint_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint_id": checkpoint_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "weather_edge_batch_sweep",
        "category": "weather_edge",
        "summary": "Batch-swept standalone weather edge summary across multiple pipeline runs.",
        "artifacts": {
            "report_json": str(report_json),
            "report_md": str(report_md),
            "source_runs": metadata["source_runs"],
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
            f"## {datetime.now(timezone.utc).strftime('%Y-%m-%d')} - Standalone weather edge batch sweep\n\n"
            f"- Checkpoint: `{checkpoint_id}`\n"
            f"- Batch count: `{metadata['batch_count']}`\n"
            f"- Rows scored: `{summary['totals']['rows_scored']}`\n"
            f"- Event-level trades selected: `{summary['totals']['trade_rows_selected']}`\n"
            f"- Trades: `{summary['backtest']['trade_count']}`\n"
            f"- Win rate: `{summary['backtest']['win_rate']}`\n"
            f"- Total net bps: `{summary['backtest']['total_net_bps']}`\n"
            f"- Max drawdown: `{summary['backtest']['max_drawdown']}`\n"
            f"- Report: `{report_md}`\n"
        )
    return checkpoint_id


def _json_default(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Unsupported type: {type(value)!r}")


if __name__ == "__main__":
    raise SystemExit(main())
