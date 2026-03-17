#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.weather_edge_backtest import score_market_horizons, summarize_weather_edge, write_weather_edge_report
from engine.weather_edge_baseline import FrozenWeatherModelBundle
from engine.weather_edge_config import DEFAULT_LOOKBACK_DAYS, default_weather_edge_root, default_weather_research_root
from engine.weather_edge_replay import (
    OpenMeteoProxyClient,
    WeatherPriceHistoryClient,
    WeatherReplayStore,
    build_market_horizon_rows,
    load_history_sources,
)


DEFAULT_REPO_ROOT = REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the standalone weather edge replay/backtest lane.")
    parser.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT))
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--max-markets", type=int, default=0)
    parser.add_argument("--market-offset", type=int, default=0)
    parser.add_argument("--allow-network", action="store_true", help="Allow Polymarket/Open-Meteo cache misses to fetch from the network.")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--cache-root", default="")
    parser.add_argument("--history-sources", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root)
    history_sources_path = Path(args.history_sources) if args.history_sources else repo_root / "models" / "weather_ml" / "weather_v2_history_sources.json"
    output_base = Path(args.output_root) if args.output_root else default_weather_edge_root(repo_root)
    cache_root = Path(args.cache_root) if args.cache_root else output_base / "cache"
    research_root = default_weather_research_root(repo_root)
    run_id = f"weather_edge_pipeline_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_v1"
    run_root = output_base / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    history_sources = load_history_sources(history_sources_path)
    replay_store = WeatherReplayStore(history_sources)
    bundle = FrozenWeatherModelBundle(history_sources.baseline_bundle_frozen)
    if not bundle.ready:
        raise SystemExit(f"Frozen weather bundle is not ready: {bundle.load_error}")

    markets = replay_store.load_recent_resolved_markets(lookback_days=args.lookback_days)
    if args.market_offset:
        markets = markets[args.market_offset :]
    if args.max_markets:
        markets = markets[: args.max_markets]

    price_history_client = WeatherPriceHistoryClient(cache_root / "price_history")
    open_meteo_client = OpenMeteoProxyClient(cache_root / "open_meteo")
    market_horizon_rows = build_market_horizon_rows(
        markets=markets,
        price_history_client=price_history_client,
        open_meteo_client=open_meteo_client,
        allow_network=bool(args.allow_network),
    )
    scored_rows = score_market_horizons(rows=market_horizon_rows, bundle=bundle)
    summary = summarize_weather_edge(scored_rows=scored_rows)

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "lookback_days": args.lookback_days,
        "market_offset": args.market_offset,
        "allow_network": bool(args.allow_network),
        "cache_root": str(cache_root),
        "bundle_dir": str(history_sources.baseline_bundle_frozen),
        "history_sources": str(history_sources_path),
        "markets_loaded": len(markets),
        "market_horizon_rows": len(market_horizon_rows),
        "scored_rows": len(scored_rows),
    }

    _write_jsonl(run_root / "resolved_markets.jsonl", markets)
    _write_jsonl(run_root / "market_horizon_rows.jsonl", market_horizon_rows)
    _write_csv(run_root / "market_horizon_rows.csv", market_horizon_rows)
    _write_jsonl(run_root / "scored_rows.jsonl", scored_rows)
    report_json, report_md = write_weather_edge_report(
        output_root=run_root,
        scored_rows=scored_rows,
        summary=summary,
        metadata=metadata,
    )
    checkpoint_id = _record_checkpoint(
        repo_root=repo_root,
        research_root=research_root,
        run_root=run_root,
        report_json=report_json,
        report_md=report_md,
        metadata=metadata,
        summary=summary,
    )

    print(json.dumps({"run_root": str(run_root), "report_json": str(report_json), "report_md": str(report_md), "checkpoint_id": checkpoint_id}, indent=2))
    return 0


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(_serialize_value(row), ensure_ascii=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flattened_rows = [_flatten_for_csv(row) for row in rows]
    if not flattened_rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in flattened_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened_rows)


def _flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, dict):
            flattened[key] = json.dumps(_serialize_value(value), ensure_ascii=True)
        elif isinstance(value, datetime):
            flattened[key] = value.isoformat()
        else:
            flattened[key] = value
    return flattened


def _serialize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _serialize_value(current) for key, current in value.items()}
    if isinstance(value, list):
        return [_serialize_value(current) for current in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _record_checkpoint(
    *,
    repo_root: Path,
    research_root: Path,
    run_root: Path,
    report_json: Path,
    report_md: Path,
    metadata: dict[str, Any],
    summary: dict[str, Any],
) -> str:
    checkpoints_root = research_root / "checkpoints"
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    index_path = checkpoints_root / "index.json"
    if not index_path.exists():
        index_path.write_text("[]\n")

    checkpoint_id = f"weather-edge-v1-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    manifest_dir = checkpoints_root / checkpoint_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.json"

    manifest = {
        "checkpoint_id": checkpoint_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "standalone_weather_edge",
        "category": "weather_edge",
        "summary": "Standalone weather edge replay/backtest lane built around the frozen legacy CatBoost weather bundle.",
        "artifacts": {
            "run_root": str(run_root),
            "report_json": str(report_json),
            "report_md": str(report_md),
        },
        "metrics": {
            "markets_loaded": metadata["markets_loaded"],
            "market_horizon_rows": metadata["market_horizon_rows"],
            "scored_rows": metadata["scored_rows"],
            "eligible_rows": summary["totals"]["rows_eligible"],
            "backtest_trades": summary["backtest"]["trade_count"],
            "backtest_total_net_bps": summary["backtest"]["total_net_bps"],
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    index = json.loads(index_path.read_text())
    index.append(manifest)
    index_path.write_text(json.dumps(index, indent=2))

    diary_path = research_root / "diary.md"
    diary_path.parent.mkdir(parents=True, exist_ok=True)
    if not diary_path.exists():
        diary_path.write_text("# Weather Research Diary\n\nThis diary is append-only.\nEach new weather replay, calibration, or execution experiment gets a new entry.\n")
    with diary_path.open("a") as handle:
        handle.write(
            "\n\n"
            f"## {datetime.now(timezone.utc).strftime('%Y-%m-%d')} - Standalone weather edge v1\n\n"
            f"- Checkpoint: `{checkpoint_id}`\n"
            f"- Run root: `{run_root}`\n"
            f"- Markets loaded: `{metadata['markets_loaded']}`\n"
            f"- Horizon rows: `{metadata['market_horizon_rows']}`\n"
            f"- Scored rows: `{metadata['scored_rows']}`\n"
            f"- Eligible rows: `{summary['totals']['rows_eligible']}`\n"
            f"- Backtest trades: `{summary['backtest']['trade_count']}`\n"
            f"- Backtest total net bps: `{summary['backtest']['total_net_bps']}`\n"
            f"- Report: `{report_md}`\n"
        )

    project_root = research_root / "projects" / "weather-edge-v1"
    project_root.mkdir(parents=True, exist_ok=True)
    plan_path = project_root / "plan.md"
    if not plan_path.exists():
        plan_path.write_text(
            "# Weather Edge v1\n\n"
            "Standalone replay/backtest lane built on the frozen legacy CatBoost weather model.\n\n"
            "## Scope\n\n"
            "- Load resolved weather markets from the preserved history dataset.\n"
            "- Sample Polymarket odds at 48h / 24h / 12h / 6h / 2h before resolution.\n"
            "- Replay the frozen weather model against those market snapshots.\n"
            "- Compute raw edge, rule-based filter results, Kelly sizing, split-half, and 15-day summaries.\n"
            "- Keep the live baseline weather sleeves untouched.\n"
        )
    return checkpoint_id


if __name__ == "__main__":
    raise SystemExit(main())
