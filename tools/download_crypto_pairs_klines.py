#!/usr/bin/env python3
"""
Download Binance spot 1h klines for crypto pairs discovery.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.binance_spot_kline_archive import KlineArchiveJob, download_jobs, iter_days


DEFAULT_OUTPUT_ROOT = Path("output/crypto_pairs/raw/spot_klines_1h")
DEFAULT_INTERVAL = "1h"
DEFAULT_SYMBOLS = [
    "ETHUSDT", "BTCUSDT", "SOLUSDT", "AVAXUSDT", "NEARUSDT",
    "DOTUSDT", "ATOMUSDT", "ADAUSDT", "MATICUSDT", "SUIUSDT",
    "UNIUSDT", "AAVEUSDT", "LINKUSDT", "MKRUSDT",
    "ARBUSDT", "OPUSDT", "DOGEUSDT", "SHIBUSDT",
]
UTC = timezone.utc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Binance spot 1h klines for crypto pairs discovery.")
    parser.add_argument("--start-date", help="Inclusive start date YYYY-MM-DD")
    parser.add_argument("--end-date", help="Inclusive end date YYYY-MM-DD")
    parser.add_argument("--lookback-days", type=int, default=30, help="Used when dates are omitted; default 30 full days ending yesterday UTC")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, help="Binance kline interval, default 1h")
    parser.add_argument("--symbol", action="append", default=[], help="Specific symbol to include. Can be repeated.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--max-download-workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    else:
        end_date = datetime.now(UTC).date() - timedelta(days=1)
        start_date = end_date - timedelta(days=args.lookback_days - 1)
    if end_date < start_date:
        raise SystemExit("end-date must be on or after start-date")

    symbols = [symbol.upper() for symbol in (args.symbol or DEFAULT_SYMBOLS)]
    output_root = Path(args.output_root).resolve()

    jobs = [
        KlineArchiveJob(
            symbol=symbol,
            interval=args.interval,
            day=day,
            target_path=output_root / symbol / args.interval / f"{symbol}-{args.interval}-{day:%Y-%m-%d}.zip",
        )
        for symbol in symbols
        for day in iter_days(start_date, end_date)
    ]
    report = download_jobs(jobs, max_workers=args.max_download_workers)
    result = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "interval": args.interval,
        "symbols": symbols,
        "download_count": len(report),
        "output_root": str(output_root),
        "downloads_preview": report[:20],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
