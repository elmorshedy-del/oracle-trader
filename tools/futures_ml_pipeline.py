#!/usr/bin/env python3
"""
Download Binance USD-M BTCUSDT futures data and train a first-pass CatBoost bundle.

The pipeline uses only official Binance public data archive files:
- aggTrades
- bookDepth
- metrics
- fundingRate

Optional live enrichment can be joined from captured official Binance streams:
- bookTicker

It builds 5-second order-flow + depth + context features and trains two binary models:
- long_60s_net: future return exceeds cost threshold
- short_60s_net: future downside exceeds cost threshold
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd

try:
    from catboost import CatBoostClassifier, Pool
except Exception as exc:  # pragma: no cover - runtime dependency gate
    raise SystemExit(f"catboost is required for this pipeline: {exc}")


ARCHIVE_BASE_URL = "https://data.binance.vision/data/futures/um"
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_LOOKBACK_DAYS = 14
DEFAULT_BUCKET_SECONDS = 5
DEFAULT_HORIZON_SECONDS = 60
DEFAULT_COST_BPS = 4.0
DEFAULT_OUTPUT_ROOT = Path("output/futures_ml")
UTC = timezone.utc
DEFAULT_LABEL_MODE = "broad"
DEFAULT_MIN_SOURCE_COMPLETENESS = 1.0
DEFAULT_CANDIDATE_MIN_SIGNED_RATIO = 0.08
DEFAULT_CANDIDATE_MIN_DEPTH_IMBALANCE = 0.02
DEFAULT_CANDIDATE_MIN_TRADE_Z = 0.75
DEFAULT_MAX_TRADE_AGE_BUCKETS = 12
DEFAULT_MAX_DEPTH_AGE_BUCKETS = 12
DEFAULT_MAX_METRICS_AGE_BUCKETS = 120
DEFAULT_MAX_FUNDING_AGE_BUCKETS = 5760
DEFAULT_PRICE_CONTEXT_INTERVAL = "1m"
DEFAULT_LIVE_BOOK_TICKER_ROOT = ""

RAW_DATASETS = ("aggTrades", "bookDepth", "metrics")
KLINE_CONTEXT_DATASETS = ("markPriceKlines", "indexPriceKlines", "premiumIndexKlines")
FEATURE_DEPTH_LEVELS = (1, 2, 5)
TARGET_COLUMNS = {"timestamp", "future_return", "long_label", "short_label"}


@dataclass(frozen=True)
class SplitMetrics:
    auc: float
    base_rate: float
    precision_at_top_decile: float
    top_decile_threshold: float
    samples: int


@dataclass(frozen=True)
class ModelSummary:
    label_name: str
    positive_rate_train: float
    positive_rate_valid: float
    positive_rate_test: float
    train: SplitMetrics
    valid: SplitMetrics
    test: SplitMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Binance futures data and train CatBoost direction models.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Binance futures symbol, default BTCUSDT")
    parser.add_argument("--start-date", help="Inclusive start date YYYY-MM-DD")
    parser.add_argument("--end-date", help="Inclusive end date YYYY-MM-DD")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS, help="Used if start-date is omitted")
    parser.add_argument("--bucket-seconds", type=int, default=DEFAULT_BUCKET_SECONDS, help="Feature aggregation bucket")
    parser.add_argument("--horizon-seconds", type=int, default=DEFAULT_HORIZON_SECONDS, help="Prediction horizon")
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS, help="Net move threshold in basis points")
    parser.add_argument(
        "--label-mode",
        choices=("broad", "continuation"),
        default=DEFAULT_LABEL_MODE,
        help="Broad predicts every bucket. Continuation only trains on plausible follow-through setups.",
    )
    parser.add_argument("--candidate-min-signed-ratio", type=float, default=DEFAULT_CANDIDATE_MIN_SIGNED_RATIO, help="Minimum signed flow ratio for continuation setup")
    parser.add_argument("--candidate-min-depth-imbalance", type=float, default=DEFAULT_CANDIDATE_MIN_DEPTH_IMBALANCE, help="Minimum top-depth imbalance for continuation setup")
    parser.add_argument("--candidate-min-trade-z", type=float, default=DEFAULT_CANDIDATE_MIN_TRADE_Z, help="Minimum trade-burst z-score for continuation setup")
    parser.add_argument(
        "--min-source-completeness",
        type=float,
        default=DEFAULT_MIN_SOURCE_COMPLETENESS,
        help="Minimum fraction of source families that must be fresh for a row to be kept",
    )
    parser.add_argument("--max-trade-age-buckets", type=int, default=DEFAULT_MAX_TRADE_AGE_BUCKETS, help="Maximum trade staleness in buckets")
    parser.add_argument("--max-depth-age-buckets", type=int, default=DEFAULT_MAX_DEPTH_AGE_BUCKETS, help="Maximum depth staleness in buckets")
    parser.add_argument("--max-metrics-age-buckets", type=int, default=DEFAULT_MAX_METRICS_AGE_BUCKETS, help="Maximum metrics staleness in buckets")
    parser.add_argument("--max-funding-age-buckets", type=int, default=DEFAULT_MAX_FUNDING_AGE_BUCKETS, help="Maximum funding staleness in buckets")
    parser.add_argument("--price-context-interval", default=DEFAULT_PRICE_CONTEXT_INTERVAL, help="Interval for mark/index/premium price context klines")
    parser.add_argument(
        "--live-book-ticker-root",
        default=DEFAULT_LIVE_BOOK_TICKER_ROOT,
        help="Optional root containing captured live bookTicker JSONL files from binance_futures_live_capture.py",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Local output directory")
    parser.add_argument("--skip-download", action="store_true", help="Reuse existing downloaded archives")
    parser.add_argument("--skip-train", action="store_true", help="Only download data")
    parser.add_argument("--max-download-workers", type=int, default=1, help="Reserved for future parallel downloads")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    symbol = args.symbol.upper()
    end_date = parse_iso_date(args.end_date) if args.end_date else (datetime.now(UTC).date() - timedelta(days=1))
    start_date = parse_iso_date(args.start_date) if args.start_date else (end_date - timedelta(days=args.lookback_days - 1))
    if start_date > end_date:
        raise SystemExit("start-date must be on or before end-date")

    completeness_tag = int(round(args.min_source_completeness * 100))
    date_tag = f"{start_date:%Y%m%d}_{end_date:%Y%m%d}"
    quote_tag = "_qtlive" if args.live_book_ticker_root else ""
    run_name = f"binance_{symbol.lower()}_{args.bucket_seconds}s_{args.horizon_seconds}s_{args.label_mode}_c{completeness_tag:03d}_{date_tag}{quote_tag}_v1"
    run_root = output_root / run_name
    raw_root = resolve_raw_root(output_root=output_root, symbol=symbol, skip_download=args.skip_download)
    live_book_ticker_root = Path(args.live_book_ticker_root).resolve() if args.live_book_ticker_root else None
    dataset_root = run_root / "dataset"
    model_root = run_root / "models"
    report_root = run_root / "reports"
    for path in (dataset_root, model_root, report_root):
        path.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "symbol": symbol,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "bucket_seconds": args.bucket_seconds,
        "horizon_seconds": args.horizon_seconds,
        "cost_bps": args.cost_bps,
        "label_mode": args.label_mode,
        "price_context_interval": args.price_context_interval,
        "live_book_ticker_root": str(live_book_ticker_root) if live_book_ticker_root else None,
        "run_root": str(run_root),
        "raw_root": str(raw_root),
        "downloaded_at": datetime.now(UTC).isoformat(),
        "raw_files": [],
    }

    if not args.skip_download:
        manifest["raw_files"] = download_archives(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            raw_root=raw_root,
            max_download_workers=args.max_download_workers,
            price_context_interval=args.price_context_interval,
        )
        (report_root / "download_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

    if args.skip_train:
        return

    dataset = build_feature_dataset(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        raw_root=raw_root,
        bucket_seconds=args.bucket_seconds,
        horizon_seconds=args.horizon_seconds,
        cost_bps=args.cost_bps,
        label_mode=args.label_mode,
        candidate_min_signed_ratio=args.candidate_min_signed_ratio,
        candidate_min_depth_imbalance=args.candidate_min_depth_imbalance,
        candidate_min_trade_z=args.candidate_min_trade_z,
        min_source_completeness=args.min_source_completeness,
        max_trade_age_buckets=args.max_trade_age_buckets,
        max_depth_age_buckets=args.max_depth_age_buckets,
        max_metrics_age_buckets=args.max_metrics_age_buckets,
        max_funding_age_buckets=args.max_funding_age_buckets,
        price_context_interval=args.price_context_interval,
        live_book_ticker_root=live_book_ticker_root,
    )
    if dataset.empty:
        raise SystemExit("No dataset rows were built. Check downloaded archive coverage.")

    dataset_path = dataset_root / "features.csv.gz"
    dataset.to_csv(dataset_path, index=False, compression="gzip")

    feature_columns = feature_columns_for_dataset(dataset)
    model_report = train_models(
        dataset=dataset,
        feature_columns=feature_columns,
        model_root=model_root,
        bucket_seconds=args.bucket_seconds,
        horizon_seconds=args.horizon_seconds,
        cost_bps=args.cost_bps,
        label_mode=args.label_mode,
    )

    metadata = {
        "bundle_version": run_name,
        "created_at": datetime.now(UTC).isoformat(),
        "symbol": symbol,
        "date_range": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
        },
        "bucket_seconds": args.bucket_seconds,
        "horizon_seconds": args.horizon_seconds,
        "cost_bps": args.cost_bps,
        "label_mode": args.label_mode,
        "price_context_interval": args.price_context_interval,
        "live_book_ticker_root": str(live_book_ticker_root) if live_book_ticker_root else None,
        "candidate_thresholds": {
            "min_signed_ratio": args.candidate_min_signed_ratio,
            "min_depth_imbalance": args.candidate_min_depth_imbalance,
            "min_trade_z": args.candidate_min_trade_z,
        },
        "source_completeness": {
            "min_source_completeness": args.min_source_completeness,
            "max_age_buckets": {
                "trade": args.max_trade_age_buckets,
                "depth": args.max_depth_age_buckets,
                "metrics": args.max_metrics_age_buckets,
                "funding": args.max_funding_age_buckets,
            },
            "coverage_summary": summarise_source_coverage(dataset),
        },
        "rows": int(len(dataset)),
        "features": feature_columns,
        "dataset_path": str(dataset_path),
        "models": model_report,
    }
    (model_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (report_root / "train_report.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (report_root / "train_report.md").write_text(render_report_markdown(metadata), encoding="utf-8")

    print(f"Dataset rows: {len(dataset):,}")
    print(f"Feature file: {dataset_path}")
    print(f"Model bundle: {model_root}")
    print(f"Report: {report_root / 'train_report.md'}")


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def resolve_raw_root(*, output_root: Path, symbol: str, skip_download: bool) -> Path:
    shared_raw_root = output_root / f"binance_{symbol.lower()}_raw"
    if raw_archive_count(shared_raw_root) > 0 or not skip_download:
        return shared_raw_root

    legacy_candidates = sorted(path for path in output_root.glob(f"binance_{symbol.lower()}_*") if path.is_dir())
    best_candidate: Path | None = None
    best_count = -1
    for candidate in legacy_candidates:
        legacy_raw = candidate / "raw"
        count = raw_archive_count(legacy_raw)
        if count > best_count:
            best_candidate = legacy_raw
            best_count = count
    if best_candidate and best_count > 0:
        return best_candidate
    return shared_raw_root


def raw_archive_count(raw_root: Path) -> int:
    if not raw_root.exists():
        return 0
    return sum(1 for _ in raw_root.rglob("*.zip"))


def iter_days(start_date: date, end_date: date) -> list[date]:
    days: list[date] = []
    cursor = start_date
    while cursor <= end_date:
        days.append(cursor)
        cursor += timedelta(days=1)
    return days


def iter_month_starts(start_date: date, end_date: date) -> list[date]:
    months: list[date] = []
    year = start_date.year
    month = start_date.month
    while (year, month) <= (end_date.year, end_date.month):
        months.append(date(year, month, 1))
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
    return months


def archive_url(dataset: str, symbol: str, stamp: date, *, interval: str | None = None) -> str:
    if dataset == "fundingRate":
        return f"{ARCHIVE_BASE_URL}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{stamp:%Y-%m}.zip"
    if dataset in KLINE_CONTEXT_DATASETS:
        if not interval:
            raise ValueError(f"interval required for dataset: {dataset}")
        return f"{ARCHIVE_BASE_URL}/daily/{dataset}/{symbol}/{interval}/{symbol}-{interval}-{stamp:%Y-%m-%d}.zip"
    if dataset not in RAW_DATASETS:
        raise ValueError(f"unsupported dataset: {dataset}")
    return f"{ARCHIVE_BASE_URL}/daily/{dataset}/{symbol}/{symbol}-{dataset}-{stamp:%Y-%m-%d}.zip"


def download_archives(
    *,
    symbol: str,
    start_date: date,
    end_date: date,
    raw_root: Path,
    max_download_workers: int,
    price_context_interval: str,
) -> list[dict[str, Any]]:
    download_jobs: list[tuple[str, Path]] = []
    for dataset in RAW_DATASETS:
        target_dir = raw_root / dataset
        target_dir.mkdir(parents=True, exist_ok=True)
        for day in iter_days(start_date, end_date):
            url = archive_url(dataset, symbol, day)
            target = target_dir / Path(url).name
            download_jobs.append((url, target))

    funding_dir = raw_root / "fundingRate"
    funding_dir.mkdir(parents=True, exist_ok=True)
    for month_start in iter_month_starts(start_date, end_date):
        url = archive_url("fundingRate", symbol, month_start)
        target = funding_dir / Path(url).name
        download_jobs.append((url, target))

    for dataset in KLINE_CONTEXT_DATASETS:
        target_dir = raw_root / dataset / price_context_interval
        target_dir.mkdir(parents=True, exist_ok=True)
        for day in iter_days(start_date, end_date):
            url = archive_url(dataset, symbol, day, interval=price_context_interval)
            target = target_dir / Path(url).name
            download_jobs.append((url, target))

    files: list[dict[str, Any]] = []
    if max_download_workers <= 1:
        for url, target in download_jobs:
            info = download_file(url, target)
            if info:
                files.append(info)
        return files

    with ThreadPoolExecutor(max_workers=max_download_workers) as executor:
        future_map = {
            executor.submit(download_file, url, target): (url, target)
            for url, target in download_jobs
        }
        for future in as_completed(future_map):
            info = future.result()
            if info:
                files.append(info)
    files.sort(key=lambda item: (item["dataset"], item["path"]))
    return files


def download_file(url: str, target: Path) -> dict[str, Any] | None:
    if target.exists() and target.stat().st_size > 0:
        return {
            "dataset": target.parent.name,
            "path": str(target),
            "url": url,
            "bytes": target.stat().st_size,
            "status": "cached",
        }
    try:
        with urlopen(url, timeout=120) as response:
            payload = response.read()
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    except URLError as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    target.write_bytes(payload)
    return {
        "dataset": target.parent.name,
        "path": str(target),
        "url": url,
        "bytes": len(payload),
        "status": "downloaded",
    }


def build_feature_dataset(
    *,
    symbol: str,
    start_date: date,
    end_date: date,
    raw_root: Path,
    bucket_seconds: int,
    horizon_seconds: int,
    cost_bps: float,
    label_mode: str,
    candidate_min_signed_ratio: float,
    candidate_min_depth_imbalance: float,
    candidate_min_trade_z: float,
    min_source_completeness: float,
    max_trade_age_buckets: int,
    max_depth_age_buckets: int,
    max_metrics_age_buckets: int,
    max_funding_age_buckets: int,
    price_context_interval: str,
    live_book_ticker_root: Path | None = None,
) -> pd.DataFrame:
    trades = load_trades(raw_root / "aggTrades", symbol, start_date, end_date, bucket_seconds)
    depth = load_depth(raw_root / "bookDepth", symbol, start_date, end_date, bucket_seconds)
    metrics = load_metrics(raw_root / "metrics", symbol, start_date, end_date, bucket_seconds)
    funding = load_funding(raw_root / "fundingRate", symbol, start_date, end_date, bucket_seconds)
    price_context = load_price_context(raw_root, symbol, start_date, end_date, bucket_seconds, price_context_interval)
    live_book_ticker = load_live_book_ticker(
        live_book_ticker_root,
        symbol,
        start_date,
        end_date,
        bucket_seconds,
    )

    combined = trades.join(depth, how="left")
    combined = combined.join(metrics, how="left")
    combined = combined.join(funding, how="left")
    combined = combined.join(price_context, how="left")
    combined = combined.join(live_book_ticker, how="left")
    combined = combined.sort_index()

    combined = add_source_coverage(
        combined,
        max_age_buckets={
            "trade": max_trade_age_buckets,
            "depth": max_depth_age_buckets,
            "metrics": max_metrics_age_buckets,
            "funding": max_funding_age_buckets,
        },
    )

    observed_columns = [column for column in combined.columns if column.endswith("_observed")]
    fill_zero_columns = [
        column for column in combined.columns
        if column.startswith(("buy_", "sell_", "signed_", "trade_", "quote_", "depth_", "notional_"))
        or column in observed_columns
    ]
    if fill_zero_columns:
        combined[fill_zero_columns] = combined[fill_zero_columns].fillna(0.0)

    ffill_columns = [column for column in combined.columns if column not in fill_zero_columns]
    if ffill_columns:
        combined[ffill_columns] = combined[ffill_columns].ffill()

    if "book_bid_price" in combined.columns:
        price_fallback_columns = [column for column in ("book_bid_price", "book_ask_price", "book_mid_price", "book_microprice") if column in combined.columns]
        if price_fallback_columns:
            combined[price_fallback_columns] = combined[price_fallback_columns].apply(
                lambda series: series.fillna(combined["price_last"])
            )
        neutral_quote_columns = [
            column
            for column in (
                "book_bid_qty",
                "book_ask_qty",
                "book_bid_notional",
                "book_ask_notional",
                "book_spread_abs",
                "book_spread_bps",
                "book_size_imbalance",
                "book_notional_imbalance",
                "book_micro_gap_bps",
            )
            if column in combined.columns
        ]
        if neutral_quote_columns:
            combined[neutral_quote_columns] = combined[neutral_quote_columns].fillna(0.0)

    combined = combined.dropna(subset=["price_last"])
    combined = add_derived_features(
        combined,
        bucket_seconds=bucket_seconds,
        horizon_seconds=horizon_seconds,
        cost_bps=cost_bps,
        candidate_min_signed_ratio=candidate_min_signed_ratio,
        candidate_min_depth_imbalance=candidate_min_depth_imbalance,
        candidate_min_trade_z=candidate_min_trade_z,
    )
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna()
    combined = combined[combined["source_fresh_score"] >= min_source_completeness]
    if label_mode == "continuation":
        combined = combined[combined["setup_active"] == 1]
    combined = combined.reset_index(names="timestamp")
    return combined


def load_trades(directory: Path, symbol: str, start_date: date, end_date: date, bucket_seconds: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for day in iter_days(start_date, end_date):
        file_path = directory / f"{symbol}-aggTrades-{day:%Y-%m-%d}.zip"
        if not file_path.exists():
            continue
        frame = pd.read_csv(file_path, compression="zip")
        frame["timestamp"] = pd.to_datetime(frame["transact_time"], unit="ms", utc=True)
        frame["bucket"] = frame["timestamp"].dt.floor(f"{bucket_seconds}s")
        frame["quantity"] = frame["quantity"].astype(float)
        frame["price"] = frame["price"].astype(float)
        frame["quote_qty"] = frame["price"] * frame["quantity"]
        frame["buy_qty"] = np.where(frame["is_buyer_maker"], 0.0, frame["quantity"])
        frame["sell_qty"] = np.where(frame["is_buyer_maker"], frame["quantity"], 0.0)
        frame["buy_quote_qty"] = np.where(frame["is_buyer_maker"], 0.0, frame["quote_qty"])
        frame["sell_quote_qty"] = np.where(frame["is_buyer_maker"], frame["quote_qty"], 0.0)
        frame["signed_qty"] = np.where(frame["is_buyer_maker"], -frame["quantity"], frame["quantity"])
        frame["signed_quote_qty"] = np.where(frame["is_buyer_maker"], -frame["quote_qty"], frame["quote_qty"])

        grouped = frame.groupby("bucket").agg(
            price_last=("price", "last"),
            price_first=("price", "first"),
            price_high=("price", "max"),
            price_low=("price", "min"),
            quantity_total=("quantity", "sum"),
            quote_qty_total=("quote_qty", "sum"),
            buy_qty=("buy_qty", "sum"),
            sell_qty=("sell_qty", "sum"),
            buy_quote_qty=("buy_quote_qty", "sum"),
            sell_quote_qty=("sell_quote_qty", "sum"),
            signed_qty=("signed_qty", "sum"),
            signed_quote_qty=("signed_quote_qty", "sum"),
            trade_count=("agg_trade_id", "count"),
        )
        grouped["price_vwap"] = grouped["quote_qty_total"] / grouped["quantity_total"].replace(0.0, np.nan)
        grouped["avg_trade_size"] = grouped["quantity_total"] / grouped["trade_count"].replace(0.0, np.nan)
        frames.append(grouped)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames).sort_index()
    full_index = pd.date_range(combined.index.min(), combined.index.max(), freq=f"{bucket_seconds}s", tz=UTC)
    combined = combined.groupby(level=0).last().reindex(full_index)
    combined["trade_observed"] = combined["price_last"].notna().astype(float)
    quantity_columns = [
        "quantity_total",
        "quote_qty_total",
        "buy_qty",
        "sell_qty",
        "buy_quote_qty",
        "sell_quote_qty",
        "signed_qty",
        "signed_quote_qty",
        "trade_count",
    ]
    combined[quantity_columns] = combined[quantity_columns].fillna(0.0)
    combined["price_last"] = combined["price_last"].ffill()
    combined["price_first"] = combined["price_first"].fillna(combined["price_last"])
    combined["price_high"] = combined["price_high"].fillna(combined["price_last"])
    combined["price_low"] = combined["price_low"].fillna(combined["price_last"])
    combined["price_vwap"] = combined["price_vwap"].fillna(combined["price_last"])
    combined["avg_trade_size"] = combined["avg_trade_size"].fillna(0.0)
    return combined


def load_depth(directory: Path, symbol: str, start_date: date, end_date: date, bucket_seconds: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for day in iter_days(start_date, end_date):
        file_path = directory / f"{symbol}-bookDepth-{day:%Y-%m-%d}.zip"
        if not file_path.exists():
            continue
        frame = pd.read_csv(file_path, compression="zip")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame["bucket"] = frame["timestamp"].dt.floor(f"{bucket_seconds}s")
        frame["percentage"] = frame["percentage"].astype(float)
        frame["depth"] = frame["depth"].astype(float)
        frame["notional"] = frame["notional"].astype(float)
        subset = frame[frame["percentage"].isin({-5.0, -2.0, -1.0, 1.0, 2.0, 5.0})]
        if subset.empty:
            continue
        pivot_depth = subset.pivot_table(index="bucket", columns="percentage", values="depth", aggfunc="last")
        pivot_notional = subset.pivot_table(index="bucket", columns="percentage", values="notional", aggfunc="last")
        features = pd.DataFrame(index=pivot_depth.index.union(pivot_notional.index))
        for level in FEATURE_DEPTH_LEVELS:
            bid_col = -float(level)
            ask_col = float(level)
            bid_depth = pivot_depth.get(bid_col, pd.Series(index=features.index, dtype=float))
            ask_depth = pivot_depth.get(ask_col, pd.Series(index=features.index, dtype=float))
            bid_notional = pivot_notional.get(bid_col, pd.Series(index=features.index, dtype=float))
            ask_notional = pivot_notional.get(ask_col, pd.Series(index=features.index, dtype=float))
            features[f"depth_bid_{level}pct"] = bid_depth
            features[f"depth_ask_{level}pct"] = ask_depth
            features[f"notional_bid_{level}pct"] = bid_notional
            features[f"notional_ask_{level}pct"] = ask_notional
            features[f"depth_imbalance_{level}pct"] = safe_divide(bid_depth - ask_depth, bid_depth + ask_depth)
            features[f"notional_imbalance_{level}pct"] = safe_divide(bid_notional - ask_notional, bid_notional + ask_notional)
        frames.append(features)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames).sort_index()
    combined = combined.groupby(level=0).last()
    combined["depth_observed"] = 1.0
    return combined


def load_metrics(directory: Path, symbol: str, start_date: date, end_date: date, bucket_seconds: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for day in iter_days(start_date, end_date):
        file_path = directory / f"{symbol}-metrics-{day:%Y-%m-%d}.zip"
        if not file_path.exists():
            continue
        frame = pd.read_csv(file_path, compression="zip")
        frame["timestamp"] = pd.to_datetime(frame["create_time"], utc=True)
        frame = frame.drop(columns=["create_time", "symbol"], errors="ignore")
        frame = frame.set_index("timestamp").sort_index()
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames).sort_index()
    full_index = pd.date_range(combined.index.min(), combined.index.max(), freq=f"{bucket_seconds}s", tz=UTC)
    combined["metrics_observed"] = 1.0
    combined = combined.groupby(level=0).last().reindex(full_index)
    combined["metrics_observed"] = combined["metrics_observed"].fillna(0.0)
    metric_columns = [column for column in combined.columns if column != "metrics_observed"]
    combined[metric_columns] = combined[metric_columns].ffill()
    return combined


def load_funding(directory: Path, symbol: str, start_date: date, end_date: date, bucket_seconds: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for month_start in iter_month_starts(start_date, end_date):
        file_path = directory / f"{symbol}-fundingRate-{month_start:%Y-%m}.zip"
        if not file_path.exists():
            continue
        frame = pd.read_csv(file_path, compression="zip")
        frame["timestamp"] = pd.to_datetime(frame["calc_time"], unit="ms", utc=True)
        frame["timestamp"] = frame["timestamp"].dt.floor(f"{bucket_seconds}s")
        frame = frame.drop(columns=["calc_time"], errors="ignore")
        frame = frame.set_index("timestamp").sort_index()
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames).sort_index()
    full_index = pd.date_range(combined.index.min(), combined.index.max(), freq=f"{bucket_seconds}s", tz=UTC)
    combined["funding_observed"] = 1.0
    combined = combined.groupby(level=0).last().reindex(full_index)
    combined["funding_observed"] = combined["funding_observed"].fillna(0.0)
    funding_columns = [column for column in combined.columns if column != "funding_observed"]
    combined[funding_columns] = combined[funding_columns].ffill()
    return combined


def load_live_book_ticker(
    root: Path | None,
    symbol: str,
    start_date: date,
    end_date: date,
    bucket_seconds: int,
) -> pd.DataFrame:
    if root is None:
        return pd.DataFrame()

    symbol_root = root / symbol
    if not symbol_root.exists():
        return pd.DataFrame()

    records: list[dict[str, float | pd.Timestamp]] = []
    for day in iter_days(start_date, end_date):
        file_path = symbol_root / f"{day:%Y-%m-%d}" / "bookTicker.jsonl"
        if not file_path.exists():
            continue
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                data = payload.get("data") or {}
                bid_price = _float_or_none(data.get("b") or data.get("bidPrice"))
                ask_price = _float_or_none(data.get("a") or data.get("askPrice"))
                bid_qty = _float_or_none(data.get("B") or data.get("bidQty"))
                ask_qty = _float_or_none(data.get("A") or data.get("askQty"))
                if None in (bid_price, ask_price, bid_qty, ask_qty):
                    continue

                timestamp_ms = data.get("E") or data.get("T")
                if timestamp_ms is not None:
                    timestamp = pd.to_datetime(int(timestamp_ms), unit="ms", utc=True)
                else:
                    captured_at = payload.get("captured_at")
                    if not captured_at:
                        continue
                    timestamp = pd.to_datetime(captured_at, utc=True)

                bid_notional = bid_price * bid_qty
                ask_notional = ask_price * ask_qty
                mid_price = (bid_price + ask_price) / 2.0
                spread_abs = max(ask_price - bid_price, 0.0)
                spread_bps = ((spread_abs / mid_price) * 10000.0) if mid_price > 0.0 else 0.0
                size_denom = bid_qty + ask_qty
                size_imbalance = ((bid_qty - ask_qty) / size_denom) if size_denom > 0.0 else 0.0
                notional_denom = bid_notional + ask_notional
                notional_imbalance = ((bid_notional - ask_notional) / notional_denom) if notional_denom > 0.0 else 0.0
                microprice = ((ask_price * bid_qty) + (bid_price * ask_qty)) / size_denom if size_denom > 0.0 else mid_price
                micro_gap_bps = (((microprice - mid_price) / mid_price) * 10000.0) if mid_price > 0.0 else 0.0

                records.append(
                    {
                        "bucket": timestamp.floor(f"{bucket_seconds}s"),
                        "book_bid_price": bid_price,
                        "book_ask_price": ask_price,
                        "book_bid_qty": bid_qty,
                        "book_ask_qty": ask_qty,
                        "book_bid_notional": bid_notional,
                        "book_ask_notional": ask_notional,
                        "book_mid_price": mid_price,
                        "book_spread_abs": spread_abs,
                        "book_spread_bps": spread_bps,
                        "book_size_imbalance": size_imbalance,
                        "book_notional_imbalance": notional_imbalance,
                        "book_microprice": microprice,
                        "book_micro_gap_bps": micro_gap_bps,
                        "book_ticker_observed": 1.0,
                    }
                )

    if not records:
        return pd.DataFrame()

    frame = pd.DataFrame.from_records(records)
    combined = frame.groupby("bucket").last().sort_index()
    return combined


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def load_price_context(
    raw_root: Path,
    symbol: str,
    start_date: date,
    end_date: date,
    bucket_seconds: int,
    interval: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    dataset_map = {
        "markPriceKlines": ("mark_ctx_open", "mark_ctx_close"),
        "indexPriceKlines": ("index_ctx_open", "index_ctx_close"),
        "premiumIndexKlines": ("premium_ctx_open", "premium_ctx_close"),
    }
    for dataset_name, (open_name, close_name) in dataset_map.items():
        directory = raw_root / dataset_name / interval
        frame = load_kline_context_dataset(
            directory,
            symbol,
            start_date,
            end_date,
            bucket_seconds,
            open_name=open_name,
            close_name=close_name,
            interval=interval,
        )
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=1).sort_index()
    combined = combined.groupby(level=0).last()
    combined["price_context_observed"] = (
        combined.filter(regex=r"_(open|close)$").notna().any(axis=1)
    ).astype(float)
    context_columns = [column for column in combined.columns if column != "price_context_observed"]
    combined[context_columns] = combined[context_columns].ffill()
    return combined


def load_kline_context_dataset(
    directory: Path,
    symbol: str,
    start_date: date,
    end_date: date,
    bucket_seconds: int,
    *,
    open_name: str,
    close_name: str,
    interval: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for day in iter_days(start_date, end_date):
        file_path = directory / f"{symbol}-{interval}-{day:%Y-%m-%d}.zip"
        if not file_path.exists():
            continue
        frame = pd.read_csv(file_path, compression="zip", header=None)
        if frame.empty:
            continue
        if str(frame.iloc[0, 0]).lower() == "open_time":
            frame = frame.iloc[1:].copy()
        if frame.empty:
            continue
        frame = frame.iloc[:, :12].copy()
        frame.columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "count",
            "taker_buy_volume",
            "taker_buy_quote_volume",
            "ignore",
        ]
        frame["open_timestamp"] = pd.to_datetime(frame["open_time"].astype("int64"), unit="ms", utc=True)
        frame["close_timestamp"] = pd.to_datetime(frame["close_time"].astype("int64"), unit="ms", utc=True) + pd.Timedelta(milliseconds=1)
        frame[open_name] = frame["open"].astype(float)
        frame[close_name] = frame["close"].astype(float)

        open_grouped = (
            frame.assign(bucket=frame["open_timestamp"].dt.floor(f"{bucket_seconds}s"))
            .groupby("bucket")
            .agg(**{open_name: (open_name, "first")})
        )
        close_grouped = (
            frame.assign(bucket=frame["close_timestamp"].dt.floor(f"{bucket_seconds}s"))
            .groupby("bucket")
            .agg(**{close_name: (close_name, "last")})
        )
        grouped = open_grouped.join(close_grouped, how="outer")
        frames.append(grouped)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames).sort_index()
    full_index = pd.date_range(combined.index.min(), combined.index.max(), freq=f"{bucket_seconds}s", tz=UTC)
    combined = combined.groupby(level=0).last().reindex(full_index)
    return combined


def add_source_coverage(frame: pd.DataFrame, *, max_age_buckets: dict[str, int]) -> pd.DataFrame:
    out = frame.copy()
    source_flags = {
        "trade": out.get("trade_observed", pd.Series(0.0, index=out.index)),
        "depth": out.get("depth_observed", pd.Series(0.0, index=out.index)),
        "metrics": out.get("metrics_observed", pd.Series(0.0, index=out.index)),
        "funding": out.get("funding_observed", pd.Series(0.0, index=out.index)),
    }

    fresh_columns: list[str] = []
    for source_name, raw_flag in source_flags.items():
        flag = raw_flag.fillna(0.0).astype(float)
        out[f"{source_name}_observed"] = flag
        age = buckets_since_last_seen(flag.astype(bool))
        out[f"{source_name}_age_buckets"] = age
        freshness = (age <= float(max_age_buckets[source_name])).astype(float)
        out[f"{source_name}_fresh"] = freshness
        fresh_columns.append(f"{source_name}_fresh")

    out["source_fresh_count"] = out[fresh_columns].sum(axis=1)
    out["source_fresh_score"] = out["source_fresh_count"] / float(len(fresh_columns))
    out["source_all_fresh"] = (out["source_fresh_score"] >= 0.999).astype(float)
    return out


def buckets_since_last_seen(flags: pd.Series) -> pd.Series:
    values = flags.to_numpy(dtype=bool)
    ages = np.full(len(values), np.inf, dtype=float)
    last_seen = -1
    for idx, is_seen in enumerate(values):
        if is_seen:
            last_seen = idx
            ages[idx] = 0.0
        elif last_seen >= 0:
            ages[idx] = float(idx - last_seen)
    return pd.Series(ages, index=flags.index, dtype=float)


def add_derived_features(
    frame: pd.DataFrame,
    *,
    bucket_seconds: int,
    horizon_seconds: int,
    cost_bps: float,
    candidate_min_signed_ratio: float,
    candidate_min_depth_imbalance: float,
    candidate_min_trade_z: float,
) -> pd.DataFrame:
    bucket_horizon = max(1, int(horizon_seconds / bucket_seconds))
    out = frame.copy()

    out["buy_sell_ratio_1"] = safe_divide(out["buy_qty"], out["sell_qty"].replace(0.0, np.nan))
    out["signed_qty_ratio_1"] = safe_divide(out["signed_qty"], out["quantity_total"].replace(0.0, np.nan))
    out["signed_quote_ratio_1"] = safe_divide(out["signed_quote_qty"], out["quote_qty_total"].replace(0.0, np.nan))

    for window in (1, 3, 12, 36):
        out[f"ret_{window}"] = out["price_last"].pct_change(window)
        out[f"trade_count_sum_{window}"] = out["trade_count"].rolling(window).sum()
        out[f"quantity_sum_{window}"] = out["quantity_total"].rolling(window).sum()
        out[f"signed_qty_sum_{window}"] = out["signed_qty"].rolling(window).sum()
        out[f"signed_quote_sum_{window}"] = out["signed_quote_qty"].rolling(window).sum()
        out[f"buy_qty_sum_{window}"] = out["buy_qty"].rolling(window).sum()
        out[f"sell_qty_sum_{window}"] = out["sell_qty"].rolling(window).sum()
        out[f"buy_quote_sum_{window}"] = out["buy_quote_qty"].rolling(window).sum()
        out[f"sell_quote_sum_{window}"] = out["sell_quote_qty"].rolling(window).sum()
        out[f"buy_ratio_{window}"] = safe_divide(
            out[f"buy_qty_sum_{window}"],
            out[f"quantity_sum_{window}"].replace(0.0, np.nan),
        )
        out[f"signed_ratio_{window}"] = safe_divide(
            out[f"signed_qty_sum_{window}"],
            out[f"quantity_sum_{window}"].replace(0.0, np.nan),
        )

    out["flow_accel_3v12"] = out["signed_ratio_3"] - out["signed_ratio_12"]
    out["flow_accel_12v36"] = out["signed_ratio_12"] - out["signed_ratio_36"]

    out["range_1"] = safe_divide(out["price_high"] - out["price_low"], out["price_last"])
    out["vwap_gap"] = safe_divide(out["price_last"] - out["price_vwap"], out["price_vwap"])
    out["vol_12"] = out["ret_1"].rolling(12).std()
    out["vol_36"] = out["ret_1"].rolling(36).std()

    out["trade_count_mean_72"] = out["trade_count_sum_12"].rolling(72).mean()
    out["trade_count_std_72"] = out["trade_count_sum_12"].rolling(72).std()
    out["trade_count_z_12"] = safe_divide(
        out["trade_count_sum_12"] - out["trade_count_mean_72"],
        out["trade_count_std_72"].replace(0.0, np.nan),
    )
    out["signed_quote_mean_72"] = out["signed_quote_sum_12"].rolling(72).mean()
    out["signed_quote_std_72"] = out["signed_quote_sum_12"].rolling(72).std()
    out["signed_quote_z_12"] = safe_divide(
        out["signed_quote_sum_12"] - out["signed_quote_mean_72"],
        out["signed_quote_std_72"].replace(0.0, np.nan),
    )

    for level in FEATURE_DEPTH_LEVELS:
        out[f"depth_imbalance_{level}pct_diff_3"] = out[f"depth_imbalance_{level}pct"].diff(3)
        out[f"notional_imbalance_{level}pct_diff_3"] = out[f"notional_imbalance_{level}pct"].diff(3)
        out[f"flow_depth_align_{level}pct_12"] = out["signed_ratio_12"] * out[f"depth_imbalance_{level}pct"]

    for column in (
        "sum_open_interest",
        "sum_open_interest_value",
        "count_toptrader_long_short_ratio",
        "sum_toptrader_long_short_ratio",
        "count_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
        "last_funding_rate",
    ):
        if column in out.columns:
            out[f"{column}_diff_1"] = out[column].diff()
            out[f"{column}_diff_12"] = out[column].diff(12)

    if "sum_open_interest_diff_12" in out.columns:
        out["oi_flow_align_12"] = out["sum_open_interest_diff_12"] * out["signed_quote_sum_12"]
    if "last_funding_rate" in out.columns:
        out["funding_flow_align_12"] = out["last_funding_rate"] * out["signed_ratio_12"]
    if "count_long_short_ratio" in out.columns and "sum_taker_long_short_vol_ratio" in out.columns:
        out["crowding_pressure"] = out["count_long_short_ratio"] * out["sum_taker_long_short_vol_ratio"]
    if "book_mid_price" in out.columns:
        out["book_mid_trade_gap_bps"] = safe_divide(out["price_last"] - out["book_mid_price"], out["book_mid_price"]) * 10000.0
        out["book_mid_ret_1"] = out["book_mid_price"].pct_change(1)
        out["book_mid_ret_3"] = out["book_mid_price"].pct_change(3)
        out["book_mid_ret_12"] = out["book_mid_price"].pct_change(12)
    if "book_microprice" in out.columns:
        out["book_micro_trade_gap_bps"] = safe_divide(out["price_last"] - out["book_microprice"], out["book_microprice"]) * 10000.0
        out["book_micro_ret_1"] = out["book_microprice"].pct_change(1)
        out["book_micro_ret_3"] = out["book_microprice"].pct_change(3)
        out["book_micro_ret_12"] = out["book_microprice"].pct_change(12)
    if "book_spread_bps" in out.columns:
        out["book_spread_change_1"] = out["book_spread_bps"].diff(1)
        out["book_spread_change_3"] = out["book_spread_bps"].diff(3)
    if "book_size_imbalance" in out.columns:
        out["book_size_imbalance_change_1"] = out["book_size_imbalance"].diff(1)
        out["book_size_imbalance_change_3"] = out["book_size_imbalance"].diff(3)
        out["book_flow_align_12"] = out["book_size_imbalance"] * out["signed_ratio_12"]
    if "book_notional_imbalance" in out.columns:
        out["book_notional_imbalance_change_1"] = out["book_notional_imbalance"].diff(1)
        out["book_notional_imbalance_change_3"] = out["book_notional_imbalance"].diff(3)
        out["book_notional_flow_align_12"] = out["book_notional_imbalance"] * out["signed_quote_ratio_1"]
    if "book_bid_qty" in out.columns and "book_ask_qty" in out.columns:
        out["book_bid_qty_change_1"] = out["book_bid_qty"].diff(1)
        out["book_ask_qty_change_1"] = out["book_ask_qty"].diff(1)
        out["book_bid_qty_change_3"] = out["book_bid_qty"].diff(3)
        out["book_ask_qty_change_3"] = out["book_ask_qty"].diff(3)
    if "book_bid_notional" in out.columns and "book_ask_notional" in out.columns:
        out["book_bid_notional_change_1"] = out["book_bid_notional"].diff(1)
        out["book_ask_notional_change_1"] = out["book_ask_notional"].diff(1)
        out["book_bid_notional_change_3"] = out["book_bid_notional"].diff(3)
        out["book_ask_notional_change_3"] = out["book_ask_notional"].diff(3)
    book_fill_zero_columns = [
        column
        for column in (
            "book_mid_trade_gap_bps",
            "book_mid_ret_1",
            "book_mid_ret_3",
            "book_mid_ret_12",
            "book_micro_trade_gap_bps",
            "book_micro_ret_1",
            "book_micro_ret_3",
            "book_micro_ret_12",
            "book_spread_change_1",
            "book_spread_change_3",
            "book_size_imbalance_change_1",
            "book_size_imbalance_change_3",
            "book_flow_align_12",
            "book_notional_imbalance_change_1",
            "book_notional_imbalance_change_3",
            "book_notional_flow_align_12",
            "book_bid_qty_change_1",
            "book_ask_qty_change_1",
            "book_bid_qty_change_3",
            "book_ask_qty_change_3",
            "book_bid_notional_change_1",
            "book_ask_notional_change_1",
            "book_bid_notional_change_3",
            "book_ask_notional_change_3",
        )
        if column in out.columns
    ]
    if book_fill_zero_columns:
        out[book_fill_zero_columns] = out[book_fill_zero_columns].fillna(0.0)
    if "mark_ctx_close" in out.columns and "index_ctx_close" in out.columns:
        out["mark_index_basis_bps"] = safe_divide(out["mark_ctx_close"] - out["index_ctx_close"], out["index_ctx_close"]) * 10000.0
        out["mark_index_basis_change_12"] = out["mark_index_basis_bps"].diff(12)
        out["mark_ctx_ret_12"] = out["mark_ctx_close"].pct_change(12)
        out["index_ctx_ret_12"] = out["index_ctx_close"].pct_change(12)
    if "mark_ctx_close" in out.columns:
        out["trade_mark_gap_bps"] = safe_divide(out["price_last"] - out["mark_ctx_close"], out["mark_ctx_close"]) * 10000.0
        out["trade_mark_gap_change_12"] = out["trade_mark_gap_bps"].diff(12)
    if "index_ctx_close" in out.columns:
        out["trade_index_gap_bps"] = safe_divide(out["price_last"] - out["index_ctx_close"], out["index_ctx_close"]) * 10000.0
        out["trade_index_gap_change_12"] = out["trade_index_gap_bps"].diff(12)
    if "premium_ctx_close" in out.columns:
        out["premium_ctx_change_12"] = out["premium_ctx_close"].diff(12)
        out["premium_ctx_abs"] = out["premium_ctx_close"].abs()

    out["future_return"] = out["price_last"].shift(-bucket_horizon) / out["price_last"] - 1.0
    threshold = cost_bps / 10000.0
    out["long_label"] = (out["future_return"] > threshold).astype(int)
    out["short_label"] = (out["future_return"] < -threshold).astype(int)
    out["long_candidate"] = (
        (out["signed_ratio_12"] >= candidate_min_signed_ratio)
        & (out["depth_imbalance_1pct"] >= candidate_min_depth_imbalance)
        & (out["trade_count_z_12"] >= candidate_min_trade_z)
        & (out["flow_accel_3v12"] >= 0.0)
    ).astype(int)
    out["short_candidate"] = (
        (out["signed_ratio_12"] <= -candidate_min_signed_ratio)
        & (out["depth_imbalance_1pct"] <= -candidate_min_depth_imbalance)
        & (out["trade_count_z_12"] >= candidate_min_trade_z)
        & (out["flow_accel_3v12"] <= 0.0)
    ).astype(int)
    out["setup_active"] = ((out["long_candidate"] == 1) | (out["short_candidate"] == 1)).astype(int)
    return out


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.astype(float) / denominator.astype(float)


def summarise_source_coverage(dataset: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows": int(len(dataset)),
        "source_fresh_score_mean": float(dataset["source_fresh_score"].mean()),
        "all_fresh_rate": float(dataset["source_all_fresh"].mean()),
    }
    for source_name in ("trade", "depth", "metrics", "funding"):
        age_series = dataset[f"{source_name}_age_buckets"].replace([np.inf, -np.inf], np.nan).dropna()
        summary[source_name] = {
            "fresh_rate": float(dataset[f"{source_name}_fresh"].mean()),
            "observed_rate": float(dataset[f"{source_name}_observed"].mean()),
            "age_buckets_mean": float(age_series.mean()) if not age_series.empty else None,
            "age_buckets_p95": float(age_series.quantile(0.95)) if not age_series.empty else None,
        }
    if "price_context_observed" in dataset.columns:
        summary["price_context"] = {
            "observed_rate": float(dataset["price_context_observed"].mean()),
        }
    return summary


def feature_columns_for_dataset(dataset: pd.DataFrame) -> list[str]:
    return [column for column in dataset.columns if column not in TARGET_COLUMNS]


def candidate_flag_for_label(label_name: str) -> str:
    if label_name == "long_label":
        return "long_candidate"
    if label_name == "short_label":
        return "short_candidate"
    raise ValueError(f"unsupported label: {label_name}")


def select_label_frame(frame: pd.DataFrame, *, label_mode: str, label_name: str) -> pd.DataFrame:
    if label_mode == "broad":
        return frame.copy()
    candidate_flag = candidate_flag_for_label(label_name)
    return frame[frame[candidate_flag] == 1].copy()


def build_classifier() -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        iterations=350,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=False,
    )


def top_feature_importances(model: CatBoostClassifier, feature_columns: list[str], pool: Pool, *, limit: int = 20) -> list[dict[str, float | str]]:
    importances = model.get_feature_importance(pool)
    return [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in sorted(
            zip(feature_columns, importances, strict=False),
            key=lambda item: item[1],
            reverse=True,
        )[:limit]
    ]


def train_models(
    *,
    dataset: pd.DataFrame,
    feature_columns: list[str],
    model_root: Path,
    bucket_seconds: int,
    horizon_seconds: int,
    cost_bps: float,
    label_mode: str,
) -> dict[str, Any]:
    train_frame, valid_frame, test_frame = chronological_split(dataset)
    model_root.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "labels": {},
        "bucket_seconds": bucket_seconds,
        "horizon_seconds": horizon_seconds,
        "cost_bps": cost_bps,
        "label_mode": label_mode,
        "rows": {
            "train": int(len(train_frame)),
            "valid": int(len(valid_frame)),
            "test": int(len(test_frame)),
        },
    }

    for label_name in ("long_label", "short_label"):
        candidate_flag = candidate_flag_for_label(label_name)
        label_train = select_label_frame(train_frame, label_mode=label_mode, label_name=label_name)
        label_valid = select_label_frame(valid_frame, label_mode=label_mode, label_name=label_name)
        label_test = select_label_frame(test_frame, label_mode=label_mode, label_name=label_name)
        if min(len(label_train), len(label_valid), len(label_test)) == 0:
            raise SystemExit(f"No rows available for {label_name} under label_mode={label_mode}")
        model = build_classifier()
        train_pool = Pool(label_train[feature_columns], label=label_train[label_name])
        valid_pool = Pool(label_valid[feature_columns], label=label_valid[label_name])
        test_pool = Pool(label_test[feature_columns], label=label_test[label_name])
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

        model_file = model_root / f"{label_name}.cbm"
        model.save_model(str(model_file))

        summary = ModelSummary(
            label_name=label_name,
            positive_rate_train=float(label_train[label_name].mean()),
            positive_rate_valid=float(label_valid[label_name].mean()),
            positive_rate_test=float(label_test[label_name].mean()),
            train=evaluate_split(model, train_pool, label_train[label_name]),
            valid=evaluate_split(model, valid_pool, label_valid[label_name]),
            test=evaluate_split(model, test_pool, label_test[label_name]),
        )
        top_features = top_feature_importances(model, feature_columns, train_pool)

        report["labels"][label_name] = {
            **asdict(summary),
            "best_iteration": int(model.get_best_iteration()),
            "candidate_rows": {
                "train": int(len(label_train)),
                "valid": int(len(label_valid)),
                "test": int(len(label_test)),
            },
            "candidate_rate": {
                "train": float(train_frame[candidate_flag].mean()),
                "valid": float(valid_frame[candidate_flag].mean()),
                "test": float(test_frame[candidate_flag].mean()),
            },
            "top_features": top_features,
        }
    return report


def chronological_split(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = dataset.sort_values("timestamp").reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)
    train = ordered.iloc[:train_end].copy()
    valid = ordered.iloc[train_end:valid_end].copy()
    test = ordered.iloc[valid_end:].copy()
    return train, valid, test


def evaluate_split(model: CatBoostClassifier, pool: Pool, labels: pd.Series) -> SplitMetrics:
    labels_array = labels.to_numpy(dtype=float)
    probs = model.predict_proba(pool)[:, 1]
    if len(np.unique(labels_array)) < 2:
        auc = 0.5
    else:
        auc_scores = model.eval_metrics(pool, ["AUC"], ntree_start=0, ntree_end=model.tree_count_, eval_period=model.tree_count_)
        auc = float(auc_scores["AUC"][-1])
    threshold = float(np.quantile(probs, 0.90))
    selected = probs >= threshold
    precision = float(labels_array[selected].mean()) if np.any(selected) else 0.0
    return SplitMetrics(
        auc=auc,
        base_rate=float(labels_array.mean()),
        precision_at_top_decile=precision,
        top_decile_threshold=threshold,
        samples=int(len(labels_array)),
    )


def render_report_markdown(metadata: dict[str, Any]) -> str:
    source_coverage = metadata.get("source_completeness", {})
    lines = [
        "# Futures ML Training Report",
        "",
        f"- Symbol: `{metadata['symbol']}`",
        f"- Date range: `{metadata['date_range']['start']}` to `{metadata['date_range']['end']}`",
        f"- Bucket: `{metadata['bucket_seconds']}s`",
        f"- Horizon: `{metadata['horizon_seconds']}s`",
        f"- Cost threshold: `{metadata['cost_bps']}` bps",
        f"- Label mode: `{metadata['label_mode']}`",
        f"- Rows: `{metadata['rows']:,}`",
        "",
        "## Source Coverage",
        "",
        f"- Minimum source completeness: `{source_coverage.get('min_source_completeness', 'n/a')}`",
        f"- Mean fresh score: `{source_coverage.get('coverage_summary', {}).get('source_fresh_score_mean', 0.0):.4f}`",
        f"- All-fresh row rate: `{source_coverage.get('coverage_summary', {}).get('all_fresh_rate', 0.0):.4f}`",
        "",
        "## Models",
        "",
    ]
    for label_name, summary in metadata["models"]["labels"].items():
        lines.extend(
            [
                f"### `{label_name}`",
                "",
                f"- Best iteration: `{summary['best_iteration']}`",
                f"- Train AUC: `{summary['train']['auc']:.4f}`",
                f"- Valid AUC: `{summary['valid']['auc']:.4f}`",
                f"- Test AUC: `{summary['test']['auc']:.4f}`",
                f"- Test precision at top decile: `{summary['test']['precision_at_top_decile']:.4f}`",
                f"- Test base rate: `{summary['test']['base_rate']:.4f}`",
                "",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
