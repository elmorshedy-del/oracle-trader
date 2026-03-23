#!/usr/bin/env python3
"""
Build sequence-ready LOB snapshots from captured Binance partial-depth streams.

Input:
- depth20@100ms JSONL captured by binance_futures_live_capture.py

Output:
- fixed-interval snapshot matrix in NPZ form
- metadata JSON with column names and run stats

This is the proper data substrate for future DeepLOB-style work, unlike the
historical percentage-depth summaries in the archive bundle.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CAPTURE_ROOT = Path("output/futures_ml_live")
DEFAULT_OUTPUT_ROOT = Path("output/futures_lob_dataset")
DEFAULT_LEVELS = 20
DEFAULT_SAMPLE_MS = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed-interval LOB tensors from captured Binance partial depth streams.")
    parser.add_argument("--capture-root", default=str(DEFAULT_CAPTURE_ROOT), help="Root containing symbol/date capture folders")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Destination root for built LOB datasets")
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol folder to read, default BTCUSDT")
    parser.add_argument("--date", help="Single capture date folder YYYY-MM-DD")
    parser.add_argument("--dates", help="Comma-separated capture dates YYYY-MM-DD,YYYY-MM-DD")
    parser.add_argument("--all-dates", action="store_true", help="Consume all available capture date folders for the symbol")
    parser.add_argument("--levels", type=int, default=DEFAULT_LEVELS, help="Depth levels per side expected in the capture")
    parser.add_argument("--sample-ms", type=int, default=DEFAULT_SAMPLE_MS, help="Fixed snapshot interval in milliseconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbol = args.symbol.upper()
    capture_dirs = resolve_capture_dirs(
        capture_root=Path(args.capture_root).resolve(),
        symbol=symbol,
        date=args.date,
        dates_csv=args.dates,
        all_dates=args.all_dates,
    )
    depth_paths = [capture_dir / "depth.jsonl" for capture_dir in capture_dirs]
    missing = [path for path in depth_paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing depth capture file(s): {missing}")
    date_label = capture_date_label(capture_dirs)

    output_dir = Path(args.output_root).resolve() / symbol / date_label
    output_dir.mkdir(parents=True, exist_ok=True)

    bucket_records: dict[int, dict[str, Any]] = {}
    depth_messages = 0
    skipped_messages = 0
    for depth_path in depth_paths:
        with depth_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                depth_messages += 1
                record = json.loads(raw_line)
                event = record.get("data") or {}
                bids = event.get("b")
                asks = event.get("a")
                if not bids or not asks:
                    skipped_messages += 1
                    continue
                event_ms = int(event.get("E") or iso_to_ms(record.get("captured_at")))
                bucket_ms = (event_ms // args.sample_ms) * args.sample_ms
                snapshot = build_snapshot(event_ms=event_ms, bids=bids, asks=asks, levels=args.levels)
                if snapshot is None:
                    skipped_messages += 1
                    continue
                bucket_records[bucket_ms] = snapshot

    if not bucket_records:
        raise SystemExit("No usable partial-depth snapshots found in capture")

    ordered_buckets = sorted(bucket_records)
    timestamps_ms = np.asarray(ordered_buckets, dtype=np.int64)
    features = np.asarray([bucket_records[bucket]["features"] for bucket in ordered_buckets], dtype=np.float32)
    mid_prices = np.asarray([bucket_records[bucket]["mid_price"] for bucket in ordered_buckets], dtype=np.float64)
    spreads_bps = np.asarray([bucket_records[bucket]["spread_bps"] for bucket in ordered_buckets], dtype=np.float32)
    columns = feature_columns(args.levels)

    npz_path = output_dir / f"lob_{args.levels}lvl_{args.sample_ms}ms.npz"
    metadata_path = output_dir / f"lob_{args.levels}lvl_{args.sample_ms}ms.metadata.json"
    np.savez_compressed(
        npz_path,
        timestamps_ms=timestamps_ms,
        features=features,
        mid_price=mid_prices,
        spread_bps=spreads_bps,
    )
    metadata = {
        "symbol": symbol,
        "capture_dates": [path.name for path in capture_dirs],
        "levels": args.levels,
        "sample_ms": args.sample_ms,
        "source_capture_dirs": [str(path) for path in capture_dirs],
        "source_depth_paths": [str(path) for path in depth_paths],
        "snapshots": int(len(timestamps_ms)),
        "feature_count": int(features.shape[1]),
        "columns": columns,
        "depth_messages": depth_messages,
        "skipped_messages": skipped_messages,
        "first_bucket_ms": int(timestamps_ms[0]),
        "last_bucket_ms": int(timestamps_ms[-1]),
        "npz_path": str(npz_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def resolve_capture_dirs(
    *,
    capture_root: Path,
    symbol: str,
    date: str | None,
    dates_csv: str | None,
    all_dates: bool,
) -> list[Path]:
    symbol_root = capture_root / symbol
    if not symbol_root.exists():
        raise SystemExit(f"Missing symbol capture directory: {symbol_root}")
    selected_dates: list[str]
    if all_dates:
        selected_dates = sorted(path.name for path in symbol_root.iterdir() if path.is_dir())
    elif dates_csv:
        selected_dates = sorted(part.strip() for part in dates_csv.split(",") if part.strip())
    elif date:
        selected_dates = [date]
    else:
        raise SystemExit("Provide one of --date, --dates, or --all-dates")
    if not selected_dates:
        raise SystemExit("No capture dates resolved")
    capture_dirs = [symbol_root / value for value in selected_dates]
    missing = [path for path in capture_dirs if not path.exists()]
    if missing:
        raise SystemExit(f"Missing capture directories: {missing}")
    return capture_dirs


def capture_date_label(capture_dirs: list[Path]) -> str:
    dates = [path.name for path in capture_dirs]
    if len(dates) == 1:
        return dates[0]
    return f"{dates[0]}_{dates[-1]}_{len(dates)}d"


def build_snapshot(*, event_ms: int, bids: list[list[str]], asks: list[list[str]], levels: int) -> dict[str, Any] | None:
    parsed_bids = parse_side(bids, levels)
    parsed_asks = parse_side(asks, levels)
    if parsed_bids is None or parsed_asks is None:
        return None

    best_bid = parsed_bids[0][0]
    best_ask = parsed_asks[0][0]
    if best_bid <= 0.0 or best_ask <= 0.0 or best_ask <= best_bid:
        return None
    mid_price = (best_bid + best_ask) / 2.0
    spread_bps = ((best_ask - best_bid) / mid_price) * 10000.0

    feature_values: list[float] = []
    for prices, sizes, side in ((parsed_bids[0], parsed_bids[1], "bid"), (parsed_asks[0], parsed_asks[1], "ask")):
        for price in prices:
            price_bps = ((price / mid_price) - 1.0) * 10000.0
            feature_values.append(price_bps)
        for size in sizes:
            feature_values.append(float(np.log1p(size)))
    feature_values.append(float(spread_bps))
    return {
        "event_ms": event_ms,
        "mid_price": mid_price,
        "spread_bps": spread_bps,
        "features": feature_values,
    }


def parse_side(levels_data: list[list[str]], levels: int) -> tuple[list[float], list[float]] | None:
    prices: list[float] = []
    sizes: list[float] = []
    for level in levels_data[:levels]:
        if len(level) < 2:
            return None
        prices.append(float(level[0]))
        sizes.append(float(level[1]))
    if len(prices) < levels:
        prices.extend([prices[-1]] * (levels - len(prices)))
        sizes.extend([0.0] * (levels - len(sizes)))
    return prices, sizes


def feature_columns(levels: int) -> list[str]:
    columns: list[str] = []
    for side in ("bid", "ask"):
        for level in range(1, levels + 1):
            columns.append(f"{side}_price_bps_{level}")
        for level in range(1, levels + 1):
            columns.append(f"{side}_log_size_{level}")
    columns.append("spread_bps")
    return columns


def iso_to_ms(value: str | None) -> int:
    if not value:
        raise SystemExit("Missing captured_at timestamp in depth record")
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


if __name__ == "__main__":
    main()
