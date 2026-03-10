#!/usr/bin/env python3
"""
Download Binance USD-M BTCUSDT futures data and train a first-pass CatBoost bundle.

The pipeline uses only official Binance public data archive files:
- aggTrades
- bookDepth
- metrics
- fundingRate

It builds 5-second order-flow + depth + context features and trains two binary models:
- long_60s_net: future return exceeds cost threshold
- short_60s_net: future downside exceeds cost threshold
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
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

RAW_DATASETS = ("aggTrades", "bookDepth", "metrics")
FEATURE_DEPTH_LEVELS = (1, 2, 5)


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

    run_name = f"binance_{symbol.lower()}_{args.bucket_seconds}s_{args.horizon_seconds}s_v1"
    run_root = output_root / run_name
    raw_root = run_root / "raw"
    dataset_root = run_root / "dataset"
    model_root = run_root / "models"
    report_root = run_root / "reports"
    for path in (raw_root, dataset_root, model_root, report_root):
        path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "symbol": symbol,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "bucket_seconds": args.bucket_seconds,
        "horizon_seconds": args.horizon_seconds,
        "cost_bps": args.cost_bps,
        "run_root": str(run_root),
        "downloaded_at": datetime.now(UTC).isoformat(),
        "raw_files": [],
    }

    if not args.skip_download:
        manifest["raw_files"] = download_archives(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            raw_root=raw_root,
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
    )
    if dataset.empty:
        raise SystemExit("No dataset rows were built. Check downloaded archive coverage.")

    dataset_path = dataset_root / "features.csv.gz"
    dataset.to_csv(dataset_path, index=False, compression="gzip")

    feature_columns = [
        column for column in dataset.columns
        if column not in {"timestamp", "future_return", "long_label", "short_label"}
    ]
    model_report = train_models(
        dataset=dataset,
        feature_columns=feature_columns,
        model_root=model_root,
        bucket_seconds=args.bucket_seconds,
        horizon_seconds=args.horizon_seconds,
        cost_bps=args.cost_bps,
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


def archive_url(dataset: str, symbol: str, stamp: date) -> str:
    if dataset == "fundingRate":
        return f"{ARCHIVE_BASE_URL}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{stamp:%Y-%m}.zip"
    if dataset not in RAW_DATASETS:
        raise ValueError(f"unsupported dataset: {dataset}")
    return f"{ARCHIVE_BASE_URL}/daily/{dataset}/{symbol}/{symbol}-{dataset}-{stamp:%Y-%m-%d}.zip"


def download_archives(
    *,
    symbol: str,
    start_date: date,
    end_date: date,
    raw_root: Path,
) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for dataset in RAW_DATASETS:
        target_dir = raw_root / dataset
        target_dir.mkdir(parents=True, exist_ok=True)
        for day in iter_days(start_date, end_date):
            url = archive_url(dataset, symbol, day)
            target = target_dir / Path(url).name
            info = download_file(url, target)
            if info:
                files.append(info)

    funding_dir = raw_root / "fundingRate"
    funding_dir.mkdir(parents=True, exist_ok=True)
    for month_start in iter_month_starts(start_date, end_date):
        url = archive_url("fundingRate", symbol, month_start)
        target = funding_dir / Path(url).name
        info = download_file(url, target)
        if info:
            files.append(info)
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
) -> pd.DataFrame:
    trades = load_trades(raw_root / "aggTrades", symbol, start_date, end_date, bucket_seconds)
    depth = load_depth(raw_root / "bookDepth", symbol, start_date, end_date, bucket_seconds)
    metrics = load_metrics(raw_root / "metrics", symbol, start_date, end_date, bucket_seconds)
    funding = load_funding(raw_root / "fundingRate", symbol, start_date, end_date, bucket_seconds)

    combined = trades.join(depth, how="left")
    combined = combined.join(metrics, how="left")
    combined = combined.join(funding, how="left")
    combined = combined.sort_index()

    fill_zero_columns = [column for column in combined.columns if column.startswith(("buy_", "sell_", "signed_", "trade_", "quote_", "depth_", "notional_"))]
    if fill_zero_columns:
        combined[fill_zero_columns] = combined[fill_zero_columns].fillna(0.0)

    ffill_columns = [column for column in combined.columns if column not in fill_zero_columns]
    if ffill_columns:
        combined[ffill_columns] = combined[ffill_columns].ffill()

    combined = combined.dropna(subset=["price_last"])
    combined = add_derived_features(combined, bucket_seconds=bucket_seconds, horizon_seconds=horizon_seconds, cost_bps=cost_bps)
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna()
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
    combined = combined.groupby(level=0).last().reindex(full_index).ffill()
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
    combined = combined.groupby(level=0).last().reindex(full_index).ffill()
    return combined


def add_derived_features(
    frame: pd.DataFrame,
    *,
    bucket_seconds: int,
    horizon_seconds: int,
    cost_bps: float,
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

    out["range_1"] = safe_divide(out["price_high"] - out["price_low"], out["price_last"])
    out["vwap_gap"] = safe_divide(out["price_last"] - out["price_vwap"], out["price_vwap"])
    out["vol_12"] = out["ret_1"].rolling(12).std()
    out["vol_36"] = out["ret_1"].rolling(36).std()

    for level in FEATURE_DEPTH_LEVELS:
        out[f"depth_imbalance_{level}pct_diff_3"] = out[f"depth_imbalance_{level}pct"].diff(3)
        out[f"notional_imbalance_{level}pct_diff_3"] = out[f"notional_imbalance_{level}pct"].diff(3)

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

    out["future_return"] = out["price_last"].shift(-bucket_horizon) / out["price_last"] - 1.0
    threshold = cost_bps / 10000.0
    out["long_label"] = (out["future_return"] > threshold).astype(int)
    out["short_label"] = (out["future_return"] < -threshold).astype(int)
    return out


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.astype(float) / denominator.astype(float)


def train_models(
    *,
    dataset: pd.DataFrame,
    feature_columns: list[str],
    model_root: Path,
    bucket_seconds: int,
    horizon_seconds: int,
    cost_bps: float,
) -> dict[str, Any]:
    train_frame, valid_frame, test_frame = chronological_split(dataset)
    model_root.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "labels": {},
        "bucket_seconds": bucket_seconds,
        "horizon_seconds": horizon_seconds,
        "cost_bps": cost_bps,
        "rows": {
            "train": int(len(train_frame)),
            "valid": int(len(valid_frame)),
            "test": int(len(test_frame)),
        },
    }

    for label_name in ("long_label", "short_label"):
        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            iterations=350,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=False,
        )
        train_pool = Pool(train_frame[feature_columns], label=train_frame[label_name])
        valid_pool = Pool(valid_frame[feature_columns], label=valid_frame[label_name])
        test_pool = Pool(test_frame[feature_columns], label=test_frame[label_name])
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

        model_file = model_root / f"{label_name}.cbm"
        model.save_model(str(model_file))

        summary = ModelSummary(
            label_name=label_name,
            positive_rate_train=float(train_frame[label_name].mean()),
            positive_rate_valid=float(valid_frame[label_name].mean()),
            positive_rate_test=float(test_frame[label_name].mean()),
            train=evaluate_split(model, train_pool, train_frame[label_name]),
            valid=evaluate_split(model, valid_pool, valid_frame[label_name]),
            test=evaluate_split(model, test_pool, test_frame[label_name]),
        )

        importances = model.get_feature_importance(train_pool)
        top_features = [
            {"feature": feature, "importance": float(importance)}
            for feature, importance in sorted(
                zip(feature_columns, importances, strict=False),
                key=lambda item: item[1],
                reverse=True,
            )[:20]
        ]

        report["labels"][label_name] = {
            **asdict(summary),
            "best_iteration": int(model.get_best_iteration()),
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
    lines = [
        "# Futures ML Training Report",
        "",
        f"- Symbol: `{metadata['symbol']}`",
        f"- Date range: `{metadata['date_range']['start']}` to `{metadata['date_range']['end']}`",
        f"- Bucket: `{metadata['bucket_seconds']}s`",
        f"- Horizon: `{metadata['horizon_seconds']}s`",
        f"- Cost threshold: `{metadata['cost_bps']}` bps",
        f"- Rows: `{metadata['rows']:,}`",
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
