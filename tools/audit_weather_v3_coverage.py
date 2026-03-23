#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.weather_edge_config import ENTRY_HORIZON_HOURS
from engine.weather_edge_replay import (
    WeatherReplayStore,
    WeatherPriceHistoryClient,
    _parse_datetime,
    load_history_sources,
)

POOLED_MIN_DAYS = 150
BUCKETED_MIN_DAYS = 100
SEPARATE_MIN_DAYS = 300
BUCKETS = {
    "early_48_24": (48, 24),
    "mid_12_6": (12, 6),
    "late_2": (2,),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit whether Weather Edge V3 has enough horizon-aligned historical coverage to justify pooled, bucketed, or separate model tests.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--history-sources", default="")
    parser.add_argument("--lookback-days", type=int, default=90)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--cache-root", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root)
    history_sources_path = Path(args.history_sources) if args.history_sources else repo_root / "models" / "weather_ml" / "weather_v2_history_sources.json"
    output_root = Path(args.output_root) if args.output_root else repo_root / "output" / "weather_edge_v3"

    history_sources = load_history_sources(history_sources_path)
    replay_store = WeatherReplayStore(history_sources)
    price_client = WeatherPriceHistoryClient(output_root / "cache" / "price_history")
    cache_roots = _resolve_cache_roots(repo_root=repo_root, extra_cache_roots=args.cache_root)

    all_markets = _load_markets(replay_store=replay_store, source_path=history_sources.source_dataset)
    recent_markets = _filter_lookback(all_markets, lookback_days=args.lookback_days)

    all_summary = _build_summary(markets=all_markets, price_client=price_client, cache_roots=cache_roots)
    lookback_summary = _build_summary(markets=recent_markets, price_client=price_client, cache_roots=cache_roots)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "history_sources": str(history_sources_path),
        "source_counts": {
            "source_rows": _count_lines(history_sources.source_dataset),
            "forecast_feature_rows": _count_lines(history_sources.forecast_features),
            "multimodel_feature_rows": _count_lines(history_sources.multimodel_features),
        },
        "cache_roots": [str(path) for path in cache_roots],
        "all_time": all_summary,
        "lookback": {
            "days": args.lookback_days,
            **lookback_summary,
        },
    }

    run_id = f"weather_v3_coverage_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    run_root = output_root / "coverage_audits" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    report_json = run_root / "weather_v3_coverage_report.json"
    report_md = run_root / "weather_v3_coverage_report.md"
    report_json.write_text(json.dumps(report, indent=2))
    report_md.write_text(_render_markdown(report))
    _append_diary(repo_root=repo_root, report=report, report_md=report_md)

    print(json.dumps({"run_root": str(run_root), "report_json": str(report_json), "report_md": str(report_md)}, indent=2))
    return 0


def _load_markets(*, replay_store: WeatherReplayStore, source_path: Path) -> list[dict[str, Any]]:
    markets: list[dict[str, Any]] = []
    with source_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            normalized = replay_store._normalize_market(row)
            if normalized is None:
                continue
            markets.append(normalized)
    markets.sort(key=lambda row: (row["resolution_time"], row["market_id"]))
    return markets


def _filter_lookback(markets: list[dict[str, Any]], *, lookback_days: int) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    return [market for market in markets if market["resolution_time"] >= cutoff]


def _build_summary(
    *,
    markets: list[dict[str, Any]],
    price_client: WeatherPriceHistoryClient,
    cache_roots: list[Path],
) -> dict[str, Any]:
    if not markets:
        return {
            "normalized_markets": 0,
            "resolution_days": 0,
            "forecast_artifact_markets": 0,
            "multimodel_artifact_markets": 0,
            "all_five_horizon_markets": 0,
            "horizon_market_counts": {},
            "horizon_day_counts": {},
            "bucket_day_counts": {},
            "config_viability": _config_viability({}, {}),
        }

    horizon_markets: dict[int, int] = defaultdict(int)
    horizon_days: dict[int, set[str]] = defaultdict(set)
    bucket_days: dict[str, set[str]] = defaultdict(set)
    forecast_artifact_markets = 0
    multimodel_artifact_markets = 0
    all_five_horizon_markets = 0
    cached_price_history_markets = 0
    resolution_days = {market["resolution_time"].date().isoformat() for market in markets}

    for market in markets:
        if market.get("local_forecast"):
            forecast_artifact_markets += 1
        if market.get("local_multimodel"):
            multimodel_artifact_markets += 1

        price_history = _load_price_history_for_market(market=market, cache_roots=cache_roots)
        if price_history:
            cached_price_history_markets += 1
        elif market.get("local_yes_price_history"):
            price_history = market["local_yes_price_history"]
        price_rows = price_client.horizon_probabilities(
            price_history=price_history,
            resolution_time=market["resolution_time"],
        )
        available_horizons = sorted(price_rows.keys())
        if len(available_horizons) == len(ENTRY_HORIZON_HOURS):
            all_five_horizon_markets += 1

        day_key = market["resolution_time"].date().isoformat()
        for horizon in available_horizons:
            horizon_markets[horizon] += 1
            horizon_days[horizon].add(day_key)
        for bucket_name, bucket_horizons in BUCKETS.items():
            if any(horizon in price_rows for horizon in bucket_horizons):
                bucket_days[bucket_name].add(day_key)

    horizon_market_counts = {str(horizon): horizon_markets.get(horizon, 0) for horizon in ENTRY_HORIZON_HOURS}
    horizon_day_counts = {str(horizon): len(horizon_days.get(horizon, set())) for horizon in ENTRY_HORIZON_HOURS}
    bucket_day_counts = {name: len(days) for name, days in bucket_days.items()}

    return {
        "normalized_markets": len(markets),
        "resolution_days": len(resolution_days),
        "forecast_artifact_markets": forecast_artifact_markets,
        "forecast_artifact_share": round(forecast_artifact_markets / len(markets), 6),
        "multimodel_artifact_markets": multimodel_artifact_markets,
        "multimodel_artifact_share": round(multimodel_artifact_markets / len(markets), 6),
        "cached_price_history_markets": cached_price_history_markets,
        "cached_price_history_share": round(cached_price_history_markets / len(markets), 6),
        "all_five_horizon_markets": all_five_horizon_markets,
        "all_five_horizon_share": round(all_five_horizon_markets / len(markets), 6),
        "horizon_market_counts": horizon_market_counts,
        "horizon_day_counts": horizon_day_counts,
        "bucket_day_counts": bucket_day_counts,
        "config_viability": _config_viability(horizon_day_counts, bucket_day_counts),
    }


def _config_viability(horizon_day_counts: dict[str, int], bucket_day_counts: dict[str, int]) -> dict[str, Any]:
    max_horizon_days = max(horizon_day_counts.values(), default=0)
    pooled_pass = max_horizon_days >= POOLED_MIN_DAYS
    bucketed_pass = all(bucket_day_counts.get(name, 0) >= BUCKETED_MIN_DAYS for name in BUCKETS)
    separate_pass = all(horizon_day_counts.get(str(horizon), 0) >= SEPARATE_MIN_DAYS for horizon in ENTRY_HORIZON_HOURS)
    return {
        "pooled": {
            "min_days_target": POOLED_MIN_DAYS,
            "best_horizon_day_count": max_horizon_days,
            "passes": pooled_pass,
        },
        "bucketed": {
            "min_days_target_per_bucket": BUCKETED_MIN_DAYS,
            "bucket_day_counts": bucket_day_counts,
            "passes": bucketed_pass,
        },
        "separate": {
            "min_days_target_per_horizon": SEPARATE_MIN_DAYS,
            "horizon_day_counts": horizon_day_counts,
            "passes": separate_pass,
        },
    }


def _count_lines(path: Path) -> int:
    with path.open() as handle:
        return sum(1 for _ in handle if _.strip())


def _resolve_cache_roots(*, repo_root: Path, extra_cache_roots: list[str]) -> list[Path]:
    cache_roots: list[Path] = []
    default_root = repo_root / "output" / "weather_edge_v1" / "cache" / "price_history"
    if default_root.exists():
        cache_roots.append(default_root)
    for raw_path in extra_cache_roots:
        path = Path(raw_path).expanduser().resolve()
        if path.exists() and path not in cache_roots:
            cache_roots.append(path)
    return cache_roots


def _load_price_history_for_market(*, market: dict[str, Any], cache_roots: list[Path]) -> list[dict[str, Any]]:
    token_id = str(market.get("yes_token_id") or "").strip()
    if not token_id:
        return []
    filename = f"{token_id}.json"
    for cache_root in cache_roots:
        candidate = cache_root / filename
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text())
            history = payload.get("history") or []
            if history:
                return history
        except Exception:
            continue
    return []


def _render_markdown(report: dict[str, Any]) -> str:
    all_time = report["all_time"]
    lookback = report["lookback"]
    return "\n".join(
        [
            "# Weather V3 Coverage Audit",
            "",
            f"Generated at: `{report['generated_at']}`",
            f"History sources: `{report['history_sources']}`",
            "",
            "## Source Counts",
            "",
            f"- Source market rows: `{report['source_counts']['source_rows']}`",
            f"- Forecast artifact rows: `{report['source_counts']['forecast_feature_rows']}`",
            f"- Multimodel artifact rows: `{report['source_counts']['multimodel_feature_rows']}`",
            f"- Price history cache roots: `{report['cache_roots']}`",
            "",
            "## All Time",
            "",
            f"- Normalized markets: `{all_time['normalized_markets']}`",
            f"- Resolution days: `{all_time['resolution_days']}`",
            f"- Forecast artifact share: `{all_time['forecast_artifact_share']}`",
            f"- Multimodel artifact share: `{all_time['multimodel_artifact_share']}`",
            f"- Cached price history share: `{all_time['cached_price_history_share']}`",
            f"- All five horizons share: `{all_time['all_five_horizon_share']}`",
            f"- Horizon day counts: `{all_time['horizon_day_counts']}`",
            f"- Bucket day counts: `{all_time['bucket_day_counts']}`",
            f"- Config viability: `{all_time['config_viability']}`",
            "",
            f"## Lookback {lookback['days']} Days",
            "",
            f"- Normalized markets: `{lookback['normalized_markets']}`",
            f"- Resolution days: `{lookback['resolution_days']}`",
            f"- Forecast artifact share: `{lookback['forecast_artifact_share']}`",
            f"- Multimodel artifact share: `{lookback['multimodel_artifact_share']}`",
            f"- Cached price history share: `{lookback['cached_price_history_share']}`",
            f"- All five horizons share: `{lookback['all_five_horizon_share']}`",
            f"- Horizon day counts: `{lookback['horizon_day_counts']}`",
            f"- Bucket day counts: `{lookback['bucket_day_counts']}`",
            f"- Config viability: `{lookback['config_viability']}`",
        ]
    )


def _append_diary(*, repo_root: Path, report: dict[str, Any], report_md: Path) -> None:
    diary_path = repo_root / "research" / "weather" / "diary.md"
    lookback = report["lookback"]
    all_time = report["all_time"]
    with diary_path.open("a") as handle:
        handle.write(
            "\n\n"
            f"## {datetime.now(timezone.utc).strftime('%Y-%m-%d')} - Weather V3 coverage audit\n\n"
            f"- Source market rows: `{report['source_counts']['source_rows']}`\n"
            f"- Forecast artifact rows: `{report['source_counts']['forecast_feature_rows']}`\n"
            f"- Multimodel artifact rows: `{report['source_counts']['multimodel_feature_rows']}`\n"
            f"- Cache roots: `{report['cache_roots']}`\n"
            f"- All-time normalized markets: `{all_time['normalized_markets']}`\n"
            f"- All-time resolution days: `{all_time['resolution_days']}`\n"
            f"- All-time cached price history share: `{all_time['cached_price_history_share']}`\n"
            f"- All-time horizon day counts: `{all_time['horizon_day_counts']}`\n"
            f"- All-time bucket day counts: `{all_time['bucket_day_counts']}`\n"
            f"- All-time config viability: `{all_time['config_viability']}`\n"
            f"- Lookback days: `{lookback['days']}`\n"
            f"- Lookback normalized markets: `{lookback['normalized_markets']}`\n"
            f"- Lookback resolution days: `{lookback['resolution_days']}`\n"
            f"- Lookback cached price history share: `{lookback['cached_price_history_share']}`\n"
            f"- Lookback horizon day counts: `{lookback['horizon_day_counts']}`\n"
            f"- Lookback bucket day counts: `{lookback['bucket_day_counts']}`\n"
            f"- Lookback config viability: `{lookback['config_viability']}`\n"
            f"- Report: `{report_md}`\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
