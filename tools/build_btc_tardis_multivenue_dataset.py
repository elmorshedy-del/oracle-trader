#!/usr/bin/env python3
"""
Build a frozen-model-compatible BTC multivenue dataset from Tardis CSV files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.btc_tardis_multivenue_shared import build_tardis_multivenue_dataset
from engine.tardis_historical_archive import iter_dates, parse_date


DEFAULT_RAW_ROOT = Path("output/btc_tardis_multivenue/raw")
DEFAULT_OUTPUT_ROOT = Path("output/btc_tardis_multivenue/datasets")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BTC multivenue dataset from Tardis CSV files.")
    parser.add_argument("--start-date", required=True, help="Inclusive start date YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="Inclusive end date YYYY-MM-DD")
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT), help="Directory containing downloaded Tardis CSV files")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Dataset output root")
    parser.add_argument("--bucket-seconds", type=int, default=1, help="Bucket size in seconds")
    parser.add_argument("--levels", type=int, default=20, help="Depth levels to aggregate from snapshot_25 files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    if end_date < start_date:
        raise SystemExit("end-date must be on or after start-date")

    raw_root = Path(args.raw_root).resolve()
    output_root = Path(args.output_root).resolve()
    dates = [day.isoformat() for day in iter_dates(start_date, end_date)]

    futures_snapshot_paths = collect_tardis_files(raw_root, exchange="binance-futures", data_type="book_snapshot_25", dates=dates, symbol="BTCUSDT")
    futures_trade_paths = collect_tardis_files(raw_root, exchange="binance-futures", data_type="trades", dates=dates, symbol="BTCUSDT")
    spot_trade_paths = collect_tardis_files(raw_root, exchange="binance", data_type="trades", dates=dates, symbol="BTCUSDT")
    coinbase_snapshot_paths = collect_tardis_files(raw_root, exchange="coinbase", data_type="book_snapshot_25", dates=dates, symbol="BTC-USD")
    coinbase_trade_paths = collect_tardis_files(raw_root, exchange="coinbase", data_type="trades", dates=dates, symbol="BTC-USD")

    missing = []
    for label, paths in {
        "futures_snapshot": futures_snapshot_paths,
        "futures_trades": futures_trade_paths,
        "spot_trades": spot_trade_paths,
        "coinbase_snapshot": coinbase_snapshot_paths,
        "coinbase_trades": coinbase_trade_paths,
    }.items():
        if not paths:
            missing.append(label)
    if missing:
        raise SystemExit(f"Missing required Tardis files: {', '.join(missing)}")

    dataset = build_tardis_multivenue_dataset(
        futures_snapshot_paths=futures_snapshot_paths,
        futures_trade_paths=futures_trade_paths,
        spot_trade_paths=spot_trade_paths,
        coinbase_snapshot_paths=coinbase_snapshot_paths,
        coinbase_trade_paths=coinbase_trade_paths,
        bucket_seconds=args.bucket_seconds,
        levels=args.levels,
    )

    date_min = dataset.index.min()
    date_max = dataset.index.max()
    run_name = f"btc_tardis_multivenue_{args.bucket_seconds}s_{start_date:%Y%m%d}_{end_date:%Y%m%d}_v1"
    run_root = output_root / run_name / "dataset"
    run_root.mkdir(parents=True, exist_ok=True)
    dataset_path = run_root / "features.csv.gz"
    dataset.to_csv(dataset_path, index=True, compression="gzip")

    metadata = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "bucket_seconds": args.bucket_seconds,
        "levels": args.levels,
        "raw_root": str(raw_root),
        "row_count": int(len(dataset)),
        "column_count": int(dataset.shape[1]),
        "columns": list(dataset.columns),
        "time_start": date_min.isoformat(),
        "time_end": date_max.isoformat(),
        "output_path": str(dataset_path.resolve()),
        "futures_snapshot_paths": [str(path) for path in futures_snapshot_paths],
        "futures_trade_paths": [str(path) for path in futures_trade_paths],
        "spot_trade_paths": [str(path) for path in spot_trade_paths],
        "coinbase_snapshot_paths": [str(path) for path in coinbase_snapshot_paths],
        "coinbase_trade_paths": [str(path) for path in coinbase_trade_paths],
    }
    (run_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def collect_tardis_files(
    raw_root: Path,
    *,
    exchange: str,
    data_type: str,
    dates: list[str],
    symbol: str,
) -> list[Path]:
    paths = [raw_root / f"{exchange}_{data_type}_{day}_{symbol}.csv.gz" for day in dates]
    return [path.resolve() for path in paths if path.exists()]


if __name__ == "__main__":
    main()
