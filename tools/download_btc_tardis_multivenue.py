#!/usr/bin/env python3
"""
Download historical BTC multivenue data from Tardis.

This downloader is intentionally narrow: it only fetches the minimum channels
needed to replay the frozen March 13 BTC mean-reversion model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.tardis_historical_archive import (
    BTC_TARDIS_REQUESTS,
    download_requests,
    ensure_free_date_compatibility,
    iter_dates,
    load_tardis_api_key,
    parse_date,
)


DEFAULT_OUTPUT_ROOT = Path("output/btc_tardis_multivenue/raw")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download BTC multivenue Tardis history.")
    parser.add_argument("--start-date", required=True, help="Inclusive start date YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="Inclusive end date YYYY-MM-DD")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Destination directory")
    parser.add_argument("--concurrency", type=int, default=2, help="Per-request downloader concurrency")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    if end_date < start_date:
        raise SystemExit("end-date must be on or after start-date")

    api_key = load_tardis_api_key()
    requested_dates = list(iter_dates(start_date, end_date))
    ensure_free_date_compatibility(dates=requested_dates, api_key=api_key)
    output_root = Path(args.output_root).resolve()
    downloaded_files = download_requests(
        requests=BTC_TARDIS_REQUESTS,
        start_date=start_date,
        end_date=end_date,
        output_root=output_root,
        api_key=api_key,
        concurrency=args.concurrency,
    )
    result = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "api_key_present": bool(api_key),
        "output_root": str(output_root),
        "downloaded_file_count": len(downloaded_files),
        "downloaded_files_preview": downloaded_files[:20],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
