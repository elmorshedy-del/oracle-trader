#!/usr/bin/env python3
"""
Build an aligned BTC multivenue feature table from captured live sessions.

Sources:
- Binance futures BTCUSDT
- Binance spot BTCUSDT
- Coinbase BTC-USD

This is the first shared dataset for the new multi-venue continuation and
mean-reversion tracks. It is append-safe because it writes to a separate output
root and never mutates prior futures-only runs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from engine.btc_multivenue_shared import (
    add_cross_venue_features,
    add_future_targets,
    build_binance_agg_trade_frame,
    build_binance_book_ticker_frame,
    build_binance_partial_depth_frame,
    build_coinbase_level2_frame,
    build_coinbase_ticker_frame,
)


DEFAULT_CAPTURE_ROOT = Path("output/btc_multivenue_capture/sessions")
DEFAULT_OUTPUT_ROOT = Path("output/btc_multivenue_dataset")
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_PRODUCT_ID = "BTC-USD"
DEFAULT_BUCKET_SECONDS = 1
DEFAULT_LEVELS = 20
UTC = timezone.utc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an aligned BTC multivenue dataset from captured sessions.")
    parser.add_argument("--capture-root", default=str(DEFAULT_CAPTURE_ROOT), help="Root containing multivenue capture sessions")
    parser.add_argument(
        "--session-dir",
        action="append",
        default=[],
        help="Specific session directory to include. Can be passed multiple times. If omitted, all sessions under capture-root are used.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Destination root for built datasets")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Binance symbol, default BTCUSDT")
    parser.add_argument("--product-id", default=DEFAULT_PRODUCT_ID, help="Coinbase product id, default BTC-USD")
    parser.add_argument("--bucket-seconds", type=int, default=DEFAULT_BUCKET_SECONDS, help="Bucket size in seconds")
    parser.add_argument("--levels", type=int, default=DEFAULT_LEVELS, help="Top depth levels to aggregate for book features")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    capture_root = Path(args.capture_root).resolve()
    output_root = Path(args.output_root).resolve()
    session_dirs = resolve_session_dirs(capture_root=capture_root, session_args=args.session_dir)
    if not session_dirs:
        raise SystemExit(f"No capture sessions found under {capture_root}")

    futures_book = load_binance_book_ticker(
        collect_paths(session_dirs, "binance_futures", args.symbol, "bookTicker.jsonl"),
        bucket_seconds=args.bucket_seconds,
        prefix="fut_",
    )
    futures_trades = load_binance_agg_trades(
        collect_paths(session_dirs, "binance_futures", args.symbol, "aggTrade.jsonl"),
        bucket_seconds=args.bucket_seconds,
        prefix="fut_",
    )
    futures_depth = load_binance_partial_depth(
        collect_paths(session_dirs, "binance_futures", args.symbol, "depth.jsonl"),
        bucket_seconds=args.bucket_seconds,
        levels=args.levels,
        prefix="fut_",
    )

    spot_book = load_binance_book_ticker(
        collect_paths(session_dirs, "binance_spot", args.symbol, "bookTicker.jsonl"),
        bucket_seconds=args.bucket_seconds,
        prefix="spot_",
    )
    spot_trades = load_binance_agg_trades(
        collect_paths(session_dirs, "binance_spot", args.symbol, "aggTrade.jsonl"),
        bucket_seconds=args.bucket_seconds,
        prefix="spot_",
    )
    spot_depth = load_binance_partial_depth(
        collect_paths(session_dirs, "binance_spot", args.symbol, "depth.jsonl"),
        bucket_seconds=args.bucket_seconds,
        levels=args.levels,
        prefix="spot_",
    )

    coinbase_book = load_coinbase_level2(
        collect_paths(session_dirs, "coinbase", args.product_id, "level2.jsonl"),
        bucket_seconds=args.bucket_seconds,
        levels=args.levels,
        prefix="cb_",
    )
    coinbase_ticker = load_coinbase_ticker(
        collect_paths(session_dirs, "coinbase", args.product_id, "ticker.jsonl"),
        bucket_seconds=args.bucket_seconds,
        prefix="cb_",
    )

    dataset = pd.concat(
        [futures_book, futures_trades, futures_depth, spot_book, spot_trades, spot_depth, coinbase_book, coinbase_ticker],
        axis=1,
    ).sort_index()
    dataset = dataset.loc[~dataset.index.duplicated(keep="last")]
    dataset = add_cross_venue_features(dataset)
    dataset = add_future_targets(dataset)
    dataset = dataset.dropna(subset=["fut_mid_price"])

    date_min = dataset.index.min()
    date_max = dataset.index.max()
    run_name = (
        f"btc_multivenue_{args.bucket_seconds}s_"
        f"{date_min:%Y%m%dT%H%M%S}_{date_max:%Y%m%dT%H%M%S}_"
        f"{len(session_dirs)}sessions_v1"
    )
    run_root = output_root / run_name
    dataset_root = run_root / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)

    dataset_path = dataset_root / "features.csv.gz"
    dataset.to_csv(dataset_path, index=True, compression="gzip")

    metadata = {
        "symbol": args.symbol,
        "product_id": args.product_id,
        "bucket_seconds": args.bucket_seconds,
        "levels": args.levels,
        "sessions": [str(path) for path in session_dirs],
        "row_count": int(len(dataset)),
        "column_count": int(dataset.shape[1]),
        "columns": list(dataset.columns),
        "start_time": date_min.isoformat(),
        "end_time": date_max.isoformat(),
        "output_path": str(dataset_path),
    }
    (dataset_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def resolve_session_dirs(*, capture_root: Path, session_args: list[str]) -> list[Path]:
    if session_args:
        session_dirs: list[Path] = []
        for raw in session_args:
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = capture_root / path
            path = path.resolve()
            if not path.exists() or not path.is_dir():
                raise SystemExit(f"Missing session directory: {path}")
            session_dirs.append(path)
        return sorted(session_dirs)
    return sorted(path for path in capture_root.iterdir() if path.is_dir())


def collect_paths(session_dirs: list[Path], venue: str, symbol: str, file_name: str) -> list[Path]:
    paths: list[Path] = []
    for session_dir in session_dirs:
        venue_root = session_dir / venue / symbol
        if not venue_root.exists():
            continue
        for date_dir in sorted(path for path in venue_root.iterdir() if path.is_dir()):
            path = date_dir / file_name
            if path.exists():
                paths.append(path)
    return paths


def iter_jsonl(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def load_binance_book_ticker(paths: list[Path], *, bucket_seconds: int, prefix: str) -> pd.DataFrame:
    return build_binance_book_ticker_frame(iter_jsonl(paths), bucket_seconds=bucket_seconds, prefix=prefix)


def load_binance_agg_trades(paths: list[Path], *, bucket_seconds: int, prefix: str) -> pd.DataFrame:
    return build_binance_agg_trade_frame(iter_jsonl(paths), bucket_seconds=bucket_seconds, prefix=prefix)


def load_binance_partial_depth(paths: list[Path], *, bucket_seconds: int, levels: int, prefix: str) -> pd.DataFrame:
    return build_binance_partial_depth_frame(
        iter_jsonl(paths),
        bucket_seconds=bucket_seconds,
        levels=levels,
        prefix=prefix,
    )


def load_coinbase_level2(paths: list[Path], *, bucket_seconds: int, levels: int, prefix: str) -> pd.DataFrame:
    return build_coinbase_level2_frame(
        iter_jsonl(paths),
        bucket_seconds=bucket_seconds,
        levels=levels,
        prefix=prefix,
    )

def load_coinbase_ticker(paths: list[Path], *, bucket_seconds: int, prefix: str) -> pd.DataFrame:
    return build_coinbase_ticker_frame(iter_jsonl(paths), bucket_seconds=bucket_seconds, prefix=prefix)


if __name__ == "__main__":
    main()
