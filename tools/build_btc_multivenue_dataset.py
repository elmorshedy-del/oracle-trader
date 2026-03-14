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
from zipfile import ZipFile

import numpy as np
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
DEFAULT_FUTURES_METRICS_ROOT = Path("output/futures_ml/binance_btcusdt_raw/metrics")
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
    parser.add_argument(
        "--include-futures-open-interest",
        action="store_true",
        help="Enrich the futures venue with historical Binance open-interest metrics aligned to the session buckets",
    )
    parser.add_argument(
        "--futures-metrics-root",
        default=str(DEFAULT_FUTURES_METRICS_ROOT),
        help="Directory containing Binance futures metrics CSV/ZIP files",
    )
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
    metrics_paths: list[str] = []
    if args.include_futures_open_interest:
        metrics_frame, metrics_paths = load_binance_futures_metrics(
            metrics_root=Path(args.futures_metrics_root).resolve(),
            symbol=args.symbol,
            bucket_index=dataset.index,
            prefix="fut_",
        )
        dataset = pd.concat([dataset, metrics_frame], axis=1).sort_index()
    dataset = add_cross_venue_features(dataset)
    dataset = add_future_targets(dataset)
    dataset = dataset.dropna(subset=["fut_mid_price"])

    date_min = dataset.index.min()
    date_max = dataset.index.max()
    run_suffix = "oi_v1" if args.include_futures_open_interest else "v1"
    run_name = (
        f"btc_multivenue_{args.bucket_seconds}s_"
        f"{date_min:%Y%m%dT%H%M%S}_{date_max:%Y%m%dT%H%M%S}_"
        f"{len(session_dirs)}sessions_{run_suffix}"
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
        "include_futures_open_interest": bool(args.include_futures_open_interest),
        "futures_metrics_root": str(Path(args.futures_metrics_root).resolve()),
        "futures_metrics_paths": metrics_paths,
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


def load_binance_futures_metrics(
    *,
    metrics_root: Path,
    symbol: str,
    bucket_index: pd.DatetimeIndex,
    prefix: str,
) -> tuple[pd.DataFrame, list[str]]:
    if bucket_index.empty:
        return pd.DataFrame(), []

    metric_paths: list[Path] = []
    for date in pd.date_range(bucket_index.min().normalize(), bucket_index.max().normalize(), freq="D", tz=UTC):
        day = date.strftime("%Y-%m-%d")
        zip_path = metrics_root / f"{symbol}-metrics-{day}.zip"
        csv_path = metrics_root / f"{symbol}-metrics-{day}.csv"
        if zip_path.exists():
            metric_paths.append(zip_path)
        elif csv_path.exists():
            metric_paths.append(csv_path)

    if not metric_paths:
        raise SystemExit(
            f"Requested futures open-interest enrichment, but no metrics files were found in {metrics_root} "
            f"for {bucket_index.min():%Y-%m-%d} to {bucket_index.max():%Y-%m-%d}"
        )

    frames: list[pd.DataFrame] = []
    raw_columns = [
        "sum_open_interest",
        "sum_open_interest_value",
        "count_toptrader_long_short_ratio",
        "sum_toptrader_long_short_ratio",
        "count_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
    ]
    for path in metric_paths:
        raw = read_binance_metrics_file(path)
        if raw.empty or "create_time" not in raw.columns:
            continue
        keep = [column for column in raw_columns if column in raw.columns]
        if not keep:
            continue
        frame = raw[["create_time", *keep]].copy()
        frame["bucket"] = pd.to_datetime(frame["create_time"], utc=True)
        frame = frame.drop(columns=["create_time"]).set_index("bucket").sort_index()
        for column in keep:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frames.append(frame)

    if not frames:
        raise SystemExit(f"Metrics files were found in {metrics_root}, but none contained usable OI columns.")

    metrics = pd.concat(frames).sort_index()
    metrics = metrics.loc[~metrics.index.duplicated(keep="last")]
    metrics["sum_open_interest_delta_5m"] = metrics["sum_open_interest"].diff()
    metrics["sum_open_interest_delta_5m_bps"] = pct_change_bps(metrics["sum_open_interest"])
    metrics["sum_open_interest_value_delta_5m"] = metrics["sum_open_interest_value"].diff()
    metrics["sum_open_interest_value_delta_5m_bps"] = pct_change_bps(metrics["sum_open_interest_value"])
    metrics = metrics.rename(columns={column: f"{prefix}{column}" for column in metrics.columns})
    aligned = metrics.reindex(bucket_index.union(metrics.index)).sort_index().ffill().reindex(bucket_index)
    aligned.index.name = "bucket"
    return aligned, [str(path) for path in metric_paths]


def read_binance_metrics_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
            if not csv_names:
                return pd.DataFrame()
            with archive.open(csv_names[0]) as handle:
                return pd.read_csv(handle)
    return pd.read_csv(path)


def pct_change_bps(series: pd.Series) -> pd.Series:
    prior = series.shift(1).replace(0.0, np.nan)
    return ((series - prior) / prior) * 10000.0


if __name__ == "__main__":
    main()
