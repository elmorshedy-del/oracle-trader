#!/usr/bin/env python3
"""
Download official Binance historical BTC archives for the separate Binance-only
historical backtest track.
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

from engine.binance_historical_archive import ArchiveJob, download_archive_jobs, iter_days


DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_LOOKBACK_DAYS = 7
DEFAULT_OUTPUT_ROOT = Path("output/btc_binance_historical/raw")
DEFAULT_MAX_DOWNLOAD_WORKERS = 4
UTC = timezone.utc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download official Binance BTC spot/futures historical archives.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Binance symbol, default BTCUSDT")
    parser.add_argument("--start-date", help="Inclusive start date YYYY-MM-DD")
    parser.add_argument("--end-date", help="Inclusive end date YYYY-MM-DD")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS, help="Used if start-date is omitted")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Root for archived downloads")
    parser.add_argument("--max-download-workers", type=int, default=DEFAULT_MAX_DOWNLOAD_WORKERS, help="Parallel download workers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    end_date = parse_iso_date(args.end_date) if args.end_date else (datetime.now(UTC).date() - timedelta(days=1))
    start_date = parse_iso_date(args.start_date) if args.start_date else (end_date - timedelta(days=args.lookback_days - 1))
    if start_date > end_date:
        raise SystemExit("start-date must be on or before end-date")

    jobs: list[ArchiveJob] = []
    for day in iter_days(start_date, end_date):
        jobs.extend(
            [
                ArchiveJob(
                    market="futures_um",
                    dataset="aggTrades",
                    symbol=args.symbol.upper(),
                    day=day,
                    target_path=output_root / "futures_um" / "aggTrades" / f"{args.symbol.upper()}-aggTrades-{day:%Y-%m-%d}.zip",
                ),
                ArchiveJob(
                    market="futures_um",
                    dataset="bookDepth",
                    symbol=args.symbol.upper(),
                    day=day,
                    target_path=output_root / "futures_um" / "bookDepth" / f"{args.symbol.upper()}-bookDepth-{day:%Y-%m-%d}.zip",
                ),
                ArchiveJob(
                    market="spot",
                    dataset="aggTrades",
                    symbol=args.symbol.upper(),
                    day=day,
                    target_path=output_root / "spot" / "aggTrades" / f"{args.symbol.upper()}-aggTrades-{day:%Y-%m-%d}.zip",
                ),
            ]
        )

    files = download_archive_jobs(jobs, max_workers=args.max_download_workers)
    report = {
        "symbol": args.symbol.upper(),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "output_root": str(output_root),
        "job_count": len(jobs),
        "downloaded_file_count": len(files),
        "files": files,
    }
    print(json.dumps(report, indent=2))


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


if __name__ == "__main__":
    main()
