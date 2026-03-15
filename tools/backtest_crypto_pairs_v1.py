#!/usr/bin/env python3
"""Backtest the rule-based crypto pairs strategy on one pair or a basket."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.crypto_pairs.backtester import load_pair_price_data, run_basket_backtest, run_single_pair_backtest
from engine.crypto_pairs.config import ExecutionConfig, RiskConfig, SignalConfig
from engine.crypto_pairs.discovery import build_runtime_configs, load_discovery_report, resolve_discovery_report_path


UTC = timezone.utc
DEFAULT_RAW_ROOT = Path("output/crypto_pairs/raw/spot_klines_1h")
DEFAULT_OUTPUT_ROOT = Path("output/crypto_pairs/backtests")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest the crypto pairs V1 z-score strategy.")
    parser.add_argument("--discovery-report", default=None)
    parser.add_argument("--pair", default=None, help="Pair key like LINK/SOL; defaults to the best pair")
    parser.add_argument("--top-pairs", type=int, default=1, help="Use the top N discovered pairs")
    parser.add_argument("--basket", action="store_true", help="Run a basket backtest across the selected pairs")
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.0)
    parser.add_argument("--stop-z", type=float, default=4.0)
    parser.add_argument("--max-hold-seconds", type=int, default=21_600)
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--capital-per-pair-pct", type=float, default=0.20)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-total-exposure-pct", type=float, default=0.80)
    parser.add_argument("--max-daily-loss-pct", type=float, default=0.03)
    parser.add_argument("--max-correlation-overlap", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    discovery_path = resolve_discovery_report_path(args.discovery_report)
    discovery_report = load_discovery_report(discovery_path)
    raw_root = resolve_path(args.raw_root)
    output_root = resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    available_pairs = max(len(list(discovery_report.get("tradeable_pairs", []))), 1)
    pair_limit = available_pairs if args.pair else max(args.top_pairs, 1)
    _, pair_configs, _ = build_runtime_configs(discovery_report, top_pairs=pair_limit)
    start_date = datetime.fromisoformat(str(discovery_report["start_date"])).date()
    end_date = datetime.fromisoformat(str(discovery_report["end_date"])).date()
    signal_config = SignalConfig(
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        stop_z=args.stop_z,
        max_hold_seconds=args.max_hold_seconds,
    )
    execution_config = ExecutionConfig()

    if args.basket or args.top_pairs > 1:
        selected_pairs = pair_configs[:pair_limit]
        data_by_pair = {
            pair_config.pair_key: load_pair_price_data(
                raw_root=raw_root,
                pair_config=pair_config,
                start_date=start_date,
                end_date=end_date,
            )
            for pair_config in selected_pairs
        }
        report = run_basket_backtest(
            pair_configs=selected_pairs,
            signal_config=signal_config,
            execution_config=execution_config,
            risk_config=RiskConfig(
                max_positions=args.max_positions,
                max_capital_per_pair_pct=args.capital_per_pair_pct,
                max_total_exposure_pct=args.max_total_exposure_pct,
                max_daily_loss_pct=args.max_daily_loss_pct,
                max_correlation_overlap=args.max_correlation_overlap,
            ),
            capital=args.capital,
            data_by_pair=data_by_pair,
        )
    else:
        pair_config = select_pair(pair_configs, args.pair)
        data_a, data_b = load_pair_price_data(
            raw_root=raw_root,
            pair_config=pair_config,
            start_date=start_date,
            end_date=end_date,
        )
        report = run_single_pair_backtest(
            pair_config=pair_config,
            signal_config=signal_config,
            execution_config=execution_config,
            capital=args.capital,
            capital_per_pair_pct=args.capital_per_pair_pct,
            data_a=data_a,
            data_b=data_b,
        )

    run_name = f"crypto_pairs_backtest_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}_v1"
    run_root = output_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    report_path = run_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report_json": str(report_path), "summary": report["summary"]}, indent=2))


def resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def select_pair(pair_configs, pair_key: str | None):
    if pair_key is None:
        return pair_configs[0]
    for config in pair_configs:
        if config.pair_key == pair_key:
            return config
    raise ValueError(f"Pair {pair_key} not found in discovery report")


if __name__ == "__main__":
    main()
