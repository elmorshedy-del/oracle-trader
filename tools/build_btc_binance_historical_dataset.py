#!/usr/bin/env python3
"""
Build a Binance-only historical BTC dataset from official bulk archives.

This is a distinct track from the live multivenue dataset because Coinbase
historical L2 is not part of Binance's bulk archive.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.btc_binance_historical_shared import build_binance_historical_dataset


DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_LOOKBACK_DAYS = 7
DEFAULT_BUCKET_SECONDS = 1
DEFAULT_RAW_ROOT = Path("output/btc_binance_historical/raw")
DEFAULT_OUTPUT_ROOT = Path("output/btc_binance_historical/datasets")
UTC = timezone.utc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Binance-only BTC historical dataset from archive zips.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Binance symbol, default BTCUSDT")
    parser.add_argument("--start-date", help="Inclusive start date YYYY-MM-DD")
    parser.add_argument("--end-date", help="Inclusive end date YYYY-MM-DD")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS, help="Used if start-date is omitted")
    parser.add_argument("--bucket-seconds", type=int, default=DEFAULT_BUCKET_SECONDS, help="Bucket size in seconds")
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT), help="Root of downloaded Binance historical archives")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Destination root for built datasets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root).resolve()
    output_root = Path(args.output_root).resolve()
    end_date = parse_iso_date(args.end_date) if args.end_date else (datetime.now(UTC).date() - timedelta(days=1))
    start_date = parse_iso_date(args.start_date) if args.start_date else (end_date - timedelta(days=args.lookback_days - 1))
    if start_date > end_date:
        raise SystemExit("start-date must be on or before end-date")

    dataset = build_binance_historical_dataset(
        raw_root=raw_root,
        symbol=args.symbol.upper(),
        start_date=start_date,
        end_date=end_date,
        bucket_seconds=args.bucket_seconds,
    )
    if dataset.empty:
        raise SystemExit("No dataset rows built from the requested Binance archive range.")

    run_name = f"btc_binance_historical_{args.bucket_seconds}s_{start_date:%Y%m%d}_{end_date:%Y%m%d}_v1"
    run_root = output_root / run_name
    dataset_root = run_root / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)

    dataset_path = dataset_root / "features.csv.gz"
    dataset.to_csv(dataset_path, compression="gzip")

    metadata = {
        "symbol": args.symbol.upper(),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "bucket_seconds": args.bucket_seconds,
        "raw_root": str(raw_root),
        "row_count": int(len(dataset)),
        "column_count": int(dataset.shape[1]),
        "columns": list(dataset.columns),
        "start_time": dataset.index.min().isoformat(),
        "end_time": dataset.index.max().isoformat(),
        "output_path": str(dataset_path),
    }
    (dataset_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


if __name__ == "__main__":
    main()
