#!/usr/bin/env python3
"""
Controlled comparison for an impulse-followthrough futures model.

This experiment keeps the existing core futures data family, adds impulse-quality features,
and relabels outcomes using simple profit/stop barriers over the future horizon.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import Pool

from futures_ml_pipeline import (
    DEFAULT_BUCKET_SECONDS,
    DEFAULT_CANDIDATE_MIN_DEPTH_IMBALANCE,
    DEFAULT_CANDIDATE_MIN_SIGNED_RATIO,
    DEFAULT_CANDIDATE_MIN_TRADE_Z,
    DEFAULT_COST_BPS,
    DEFAULT_HORIZON_SECONDS,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MAX_DEPTH_AGE_BUCKETS,
    DEFAULT_MAX_FUNDING_AGE_BUCKETS,
    DEFAULT_MAX_METRICS_AGE_BUCKETS,
    DEFAULT_MAX_TRADE_AGE_BUCKETS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PRICE_CONTEXT_INTERVAL,
    DEFAULT_SYMBOL,
    build_classifier,
    build_feature_dataset,
    download_archives,
    evaluate_split,
    parse_iso_date,
    resolve_raw_root,
    summarise_source_coverage,
    top_feature_importances,
)


DEFAULT_COMPARE_OUTPUT_ROOT = Path("output/futures_ml_impulse_compare")
CORE_CONTEXT_PREFIXES = (
    "mark_ctx_",
    "index_ctx_",
    "premium_ctx_",
    "trade_mark_",
    "trade_index_",
    "mark_index_",
    "price_context_",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run impulse-followthrough futures ML comparison.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Binance futures symbol, default BTCUSDT")
    parser.add_argument("--start-date", help="Inclusive start date YYYY-MM-DD")
    parser.add_argument("--end-date", help="Inclusive end date YYYY-MM-DD")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS, help="Used if start-date is omitted")
    parser.add_argument("--bucket-seconds", type=int, default=DEFAULT_BUCKET_SECONDS, help="Feature aggregation bucket")
    parser.add_argument("--horizon-seconds", type=int, default=DEFAULT_HORIZON_SECONDS, help="Barrier horizon")
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS, help="Reference cost threshold")
    parser.add_argument("--profit-bps", type=float, default=8.0, help="Take-profit barrier in basis points")
    parser.add_argument("--stop-bps", type=float, default=6.0, help="Stop-loss barrier in basis points")
    parser.add_argument("--train-thresholds", default="0.75,1.0", help="Comma-separated source completeness thresholds used for training variants")
    parser.add_argument("--eval-thresholds", default="0.75,1.0", help="Comma-separated source completeness thresholds used for fixed test slices")
    parser.add_argument("--anchor-threshold", type=float, help="Threshold whose timeline defines the shared train/valid/test windows. Defaults to the highest threshold.")
    parser.add_argument("--min-signed-ratio", type=float, default=DEFAULT_CANDIDATE_MIN_SIGNED_RATIO / 2.0, help="Minimum signed flow ratio for impulse candidates")
    parser.add_argument("--min-depth-imbalance", type=float, default=DEFAULT_CANDIDATE_MIN_DEPTH_IMBALANCE / 2.0, help="Minimum top-depth imbalance for impulse candidates")
    parser.add_argument("--min-trade-z", type=float, default=DEFAULT_CANDIDATE_MIN_TRADE_Z / 3.0, help="Minimum trade burst z-score for impulse candidates")
    parser.add_argument("--min-directional-efficiency", type=float, default=0.15, help="Minimum directional efficiency for impulse candidates")
    parser.add_argument("--max-trade-age-buckets", type=int, default=DEFAULT_MAX_TRADE_AGE_BUCKETS, help="Maximum trade staleness in buckets")
    parser.add_argument("--max-depth-age-buckets", type=int, default=DEFAULT_MAX_DEPTH_AGE_BUCKETS, help="Maximum depth staleness in buckets")
    parser.add_argument("--max-metrics-age-buckets", type=int, default=DEFAULT_MAX_METRICS_AGE_BUCKETS, help="Maximum metrics staleness in buckets")
    parser.add_argument("--max-funding-age-buckets", type=int, default=DEFAULT_MAX_FUNDING_AGE_BUCKETS, help="Maximum funding staleness in buckets")
    parser.add_argument(
        "--price-context-interval",
        default="",
        help="Optional context interval for the base dataset builder. Blank keeps the impulse experiment core-only.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_COMPARE_OUTPUT_ROOT), help="Comparison output directory")
    parser.add_argument("--raw-output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Location of shared futures raw cache")
    parser.add_argument("--skip-download", action="store_true", help="Reuse existing downloaded archives")
    parser.add_argument("--max-download-workers", type=int, default=1, help="Parallel archive download workers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    raw_output_root = Path(args.raw_output_root).resolve()
    symbol = args.symbol.upper()
    end_date = parse_iso_date(args.end_date) if args.end_date else (datetime.now(UTC).date() - timedelta(days=1))
    start_date = parse_iso_date(args.start_date) if args.start_date else (end_date - timedelta(days=args.lookback_days - 1))
    if start_date > end_date:
        raise SystemExit("start-date must be on or before end-date")

    train_thresholds = parse_thresholds(args.train_thresholds)
    eval_thresholds = parse_thresholds(args.eval_thresholds)
    anchor_threshold = args.anchor_threshold if args.anchor_threshold is not None else max(train_thresholds + eval_thresholds)
    train_tags = "-".join(threshold_tag(value) for value in train_thresholds)
    eval_tags = "-".join(threshold_tag(value) for value in eval_thresholds)
    date_tag = f"{start_date:%Y%m%d}_{end_date:%Y%m%d}"
    run_name = (
        f"binance_{symbol.lower()}_{args.bucket_seconds}s_impulse_{args.horizon_seconds}s_"
        f"tp{int(round(args.profit_bps))}_sl{int(round(args.stop_bps))}_{date_tag}_"
        f"compare_train-{train_tags}_eval-{eval_tags}_v1"
    )
    run_root = output_root / run_name
    dataset_root = run_root / "dataset"
    model_root = run_root / "models"
    report_root = run_root / "reports"
    for path in (dataset_root, model_root, report_root):
        path.mkdir(parents=True, exist_ok=True)

    raw_root = resolve_raw_root(output_root=raw_output_root, symbol=symbol, skip_download=args.skip_download)
    raw_root.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        download_archives(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            raw_root=raw_root,
            max_download_workers=args.max_download_workers,
            price_context_interval=args.price_context_interval,
        )

    base_dataset = build_feature_dataset(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        raw_root=raw_root,
        bucket_seconds=args.bucket_seconds,
        horizon_seconds=args.horizon_seconds,
        cost_bps=args.cost_bps,
        label_mode="broad",
        candidate_min_signed_ratio=DEFAULT_CANDIDATE_MIN_SIGNED_RATIO,
        candidate_min_depth_imbalance=DEFAULT_CANDIDATE_MIN_DEPTH_IMBALANCE,
        candidate_min_trade_z=DEFAULT_CANDIDATE_MIN_TRADE_Z,
        min_source_completeness=0.0,
        max_trade_age_buckets=args.max_trade_age_buckets,
        max_depth_age_buckets=args.max_depth_age_buckets,
        max_metrics_age_buckets=args.max_metrics_age_buckets,
        max_funding_age_buckets=args.max_funding_age_buckets,
        price_context_interval=args.price_context_interval,
    )
    if base_dataset.empty:
        raise SystemExit("Base dataset is empty. Check raw archive coverage.")

    master_dataset = prepare_impulse_dataset(
        base_dataset,
        bucket_seconds=args.bucket_seconds,
        horizon_seconds=args.horizon_seconds,
        profit_bps=args.profit_bps,
        stop_bps=args.stop_bps,
        min_signed_ratio=args.min_signed_ratio,
        min_depth_imbalance=args.min_depth_imbalance,
        min_trade_z=args.min_trade_z,
        min_directional_efficiency=args.min_directional_efficiency,
    )
    if master_dataset.empty:
        raise SystemExit("Impulse dataset is empty after feature and label preparation.")

    master_dataset_path = dataset_root / "master_features.csv.gz"
    master_dataset.to_csv(master_dataset_path, index=False, compression="gzip")

    train_frame, valid_frame, test_frame = anchored_time_split(master_dataset, anchor_threshold)
    feature_columns = impulse_feature_columns(master_dataset)

    report = {
        "bundle_version": run_name,
        "created_at": datetime.now(UTC).isoformat(),
        "symbol": symbol,
        "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        "bucket_seconds": args.bucket_seconds,
        "horizon_seconds": args.horizon_seconds,
        "profit_bps": args.profit_bps,
        "stop_bps": args.stop_bps,
        "cost_bps": args.cost_bps,
        "train_thresholds": train_thresholds,
        "eval_thresholds": eval_thresholds,
        "anchor_threshold": anchor_threshold,
        "raw_root": str(raw_root),
        "master_dataset_path": str(master_dataset_path),
        "master_rows": int(len(master_dataset)),
        "master_coverage_summary": summarise_source_coverage(master_dataset),
        "candidate_summary": {
            "long_rate": float(master_dataset["long_impulse_candidate"].mean()),
            "short_rate": float(master_dataset["short_impulse_candidate"].mean()),
        },
        "time_split": {
            "train_rows": int(len(train_frame)),
            "valid_rows": int(len(valid_frame)),
            "test_rows": int(len(test_frame)),
            "train_start": str(train_frame["timestamp"].min()),
            "train_end": str(train_frame["timestamp"].max()),
            "valid_start": str(valid_frame["timestamp"].min()),
            "valid_end": str(valid_frame["timestamp"].max()),
            "test_start": str(test_frame["timestamp"].min()),
            "test_end": str(test_frame["timestamp"].max()),
        },
        "test_slices": {},
        "labels": {},
    }

    for eval_threshold in eval_thresholds:
        eval_key = threshold_tag(eval_threshold)
        eval_base = threshold_frame(test_frame, eval_threshold)
        report["test_slices"][eval_key] = {
            "base_rows": int(len(eval_base)),
            "coverage_summary": summarise_source_coverage(eval_base) if len(eval_base) else None,
        }

    for label_name, candidate_column in (
        ("long_followthrough_label", "long_impulse_candidate"),
        ("short_followthrough_label", "short_impulse_candidate"),
    ):
        label_report: dict[str, Any] = {"candidate_column": candidate_column, "models": {}}
        for train_threshold in train_thresholds:
            variant_key = threshold_tag(train_threshold)
            train_base = threshold_frame(train_frame, train_threshold)
            valid_base = threshold_frame(valid_frame, train_threshold)
            label_train = train_base[train_base[candidate_column] == 1].copy()
            label_valid = valid_base[valid_base[candidate_column] == 1].copy()
            if min(len(label_train), len(label_valid)) == 0:
                raise SystemExit(f"No rows available for {label_name} under train threshold {train_threshold}")

            model = build_classifier()
            train_pool = Pool(label_train[feature_columns], label=label_train[label_name])
            valid_pool = Pool(label_valid[feature_columns], label=label_valid[label_name])
            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

            variant_model_root = model_root / variant_key
            variant_model_root.mkdir(parents=True, exist_ok=True)
            model_file = variant_model_root / f"{label_name}.cbm"
            model.save_model(str(model_file))

            evaluations: dict[str, Any] = {}
            for eval_threshold in eval_thresholds:
                eval_key = threshold_tag(eval_threshold)
                eval_base = threshold_frame(test_frame, eval_threshold)
                label_eval = eval_base[eval_base[candidate_column] == 1].copy()
                if len(label_eval) == 0:
                    evaluations[eval_key] = None
                    continue
                eval_pool = Pool(label_eval[feature_columns], label=label_eval[label_name])
                evaluations[eval_key] = {
                    "base_rows": int(len(eval_base)),
                    "label_rows": int(len(label_eval)),
                    "positive_rate": float(label_eval[label_name].mean()),
                    "metrics": asdict(evaluate_split(model, eval_pool, label_eval[label_name])),
                }

            label_report["models"][variant_key] = {
                "train_threshold": train_threshold,
                "train_base_rows": int(len(train_base)),
                "valid_base_rows": int(len(valid_base)),
                "train_label_rows": int(len(label_train)),
                "valid_label_rows": int(len(label_valid)),
                "train_positive_rate": float(label_train[label_name].mean()),
                "valid_positive_rate": float(label_valid[label_name].mean()),
                "best_iteration": int(model.get_best_iteration()),
                "top_features": top_feature_importances(model, feature_columns, train_pool),
                "evaluations": evaluations,
            }
        report["labels"][label_name] = label_report

    metadata_path = report_root / "comparison_report.json"
    markdown_path = report_root / "comparison_report.md"
    metadata_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")

    print(f"Master rows: {len(master_dataset):,}")
    print(f"Master dataset: {master_dataset_path}")
    print(f"Comparison report: {markdown_path}")


def parse_thresholds(value: str) -> list[float]:
    values = sorted({round(float(item.strip()), 4) for item in value.split(",") if item.strip()})
    if not values:
        raise SystemExit("At least one threshold is required")
    for threshold in values:
        if threshold < 0.0 or threshold > 1.0:
            raise SystemExit(f"Threshold must be between 0 and 1: {threshold}")
    return values


def threshold_tag(threshold: float) -> str:
    return f"c{int(round(threshold * 100)):03d}"


def threshold_frame(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return frame[frame["source_fresh_score"] >= threshold].copy()


def anchored_time_split(dataset: pd.DataFrame, anchor_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reference_frame = threshold_frame(dataset, anchor_threshold)
    if len(reference_frame) == 0:
        raise SystemExit(f"No rows available for anchor threshold {anchor_threshold}")
    ref_train, ref_valid, ref_test = chronological_split(reference_frame)
    if min(len(ref_train), len(ref_valid), len(ref_test)) == 0:
        raise SystemExit(f"Anchor threshold {anchor_threshold} produced an empty split")

    base = dataset.sort_values("timestamp").reset_index(drop=True)
    train_frame = base[(base["timestamp"] >= ref_train["timestamp"].min()) & (base["timestamp"] <= ref_train["timestamp"].max())].copy()
    valid_frame = base[(base["timestamp"] >= ref_valid["timestamp"].min()) & (base["timestamp"] <= ref_valid["timestamp"].max())].copy()
    test_frame = base[(base["timestamp"] >= ref_test["timestamp"].min()) & (base["timestamp"] <= ref_test["timestamp"].max())].copy()
    if min(len(train_frame), len(valid_frame), len(test_frame)) == 0:
        raise SystemExit("Anchored time split produced an empty master window")
    return train_frame, valid_frame, test_frame


def chronological_split(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = dataset.sort_values("timestamp").reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)
    return ordered.iloc[:train_end].copy(), ordered.iloc[train_end:valid_end].copy(), ordered.iloc[valid_end:].copy()


def prepare_impulse_dataset(
    base_dataset: pd.DataFrame,
    *,
    bucket_seconds: int,
    horizon_seconds: int,
    profit_bps: float,
    stop_bps: float,
    min_signed_ratio: float,
    min_depth_imbalance: float,
    min_trade_z: float,
    min_directional_efficiency: float,
) -> pd.DataFrame:
    frame = base_dataset.copy()
    core_columns = [column for column in frame.columns if not column.startswith(CORE_CONTEXT_PREFIXES)]
    frame = frame[core_columns].copy()

    frame["avg_trade_notional_1"] = safe_divide(frame["quote_qty_total"], frame["trade_count"].replace(0.0, np.nan))
    frame["avg_trade_notional_mean_72"] = frame["avg_trade_notional_1"].rolling(72).mean()
    frame["avg_trade_notional_std_72"] = frame["avg_trade_notional_1"].rolling(72).std()
    frame["avg_trade_notional_z_12"] = safe_divide(
        frame["avg_trade_notional_1"] - frame["avg_trade_notional_mean_72"],
        frame["avg_trade_notional_std_72"].replace(0.0, np.nan),
    )
    frame["abs_signed_ratio_12"] = frame["signed_ratio_12"].abs()
    frame["abs_signed_ratio_3"] = frame["signed_ratio_3"].abs()
    frame["directional_efficiency_12"] = safe_divide(frame["ret_12"].abs(), frame["vol_12"].replace(0.0, np.nan))
    frame["directional_efficiency_36"] = safe_divide(frame["ret_36"].abs(), frame["vol_36"].replace(0.0, np.nan))
    frame["quote_conviction_12"] = safe_divide(frame["signed_quote_sum_12"].abs(), (frame["buy_quote_sum_12"] + frame["sell_quote_sum_12"]).replace(0.0, np.nan))
    frame["depth_pressure_change_12"] = frame["depth_imbalance_1pct"].diff(12)
    frame["depth_pressure_abs_12"] = frame["depth_imbalance_1pct"].abs()
    frame["range_vol_ratio_12"] = safe_divide(frame["range_1"], frame["vol_12"].replace(0.0, np.nan))
    frame["impulse_alignment_12"] = frame["signed_ratio_12"] * frame["depth_imbalance_1pct"]
    frame["impulse_alignment_36"] = frame["signed_ratio_36"] * frame["depth_imbalance_1pct"]

    frame["long_impulse_candidate"] = (
        (frame["signed_ratio_12"] >= min_signed_ratio)
        & (frame["depth_imbalance_1pct"] >= min_depth_imbalance)
        & (frame["trade_count_z_12"] >= min_trade_z)
        & (frame["directional_efficiency_12"] >= min_directional_efficiency)
        & (frame["impulse_alignment_12"] > 0.0)
    ).astype(int)
    frame["short_impulse_candidate"] = (
        (frame["signed_ratio_12"] <= -min_signed_ratio)
        & (frame["depth_imbalance_1pct"] <= -min_depth_imbalance)
        & (frame["trade_count_z_12"] >= min_trade_z)
        & (frame["directional_efficiency_12"] >= min_directional_efficiency)
        & (frame["impulse_alignment_12"] > 0.0)
    ).astype(int)

    long_labels, short_labels = compute_barrier_labels(
        frame["price_last"].to_numpy(dtype=float),
        horizon_buckets=max(1, int(horizon_seconds / bucket_seconds)),
        profit_bps=profit_bps,
        stop_bps=stop_bps,
    )
    frame["long_followthrough_label"] = long_labels
    frame["short_followthrough_label"] = short_labels

    frame = frame.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return frame


def compute_barrier_labels(prices: np.ndarray, *, horizon_buckets: int, profit_bps: float, stop_bps: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(prices)
    long_labels = np.zeros(n, dtype=int)
    short_labels = np.zeros(n, dtype=int)
    up_barrier = profit_bps / 10000.0
    down_barrier = stop_bps / 10000.0
    for idx in range(n):
        entry = prices[idx]
        if not np.isfinite(entry) or entry <= 0.0:
            continue
        window = prices[idx + 1 : idx + 1 + horizon_buckets]
        if len(window) == 0:
            continue
        rel = window / entry - 1.0
        long_up_hit = np.where(rel >= up_barrier)[0]
        long_down_hit = np.where(rel <= -down_barrier)[0]
        short_down_hit = np.where(rel <= -up_barrier)[0]
        short_up_hit = np.where(rel >= down_barrier)[0]

        long_up_idx = int(long_up_hit[0]) if len(long_up_hit) else None
        long_down_idx = int(long_down_hit[0]) if len(long_down_hit) else None
        short_down_idx = int(short_down_hit[0]) if len(short_down_hit) else None
        short_up_idx = int(short_up_hit[0]) if len(short_up_hit) else None

        if long_up_idx is not None and (long_down_idx is None or long_up_idx < long_down_idx):
            long_labels[idx] = 1
        if short_down_idx is not None and (short_up_idx is None or short_down_idx < short_up_idx):
            short_labels[idx] = 1
    return long_labels, short_labels


def impulse_feature_columns(dataset: pd.DataFrame) -> list[str]:
    excluded = {
        "timestamp",
        "future_return",
        "long_label",
        "short_label",
        "long_followthrough_label",
        "short_followthrough_label",
    }
    return [column for column in dataset.columns if column not in excluded]


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.astype(float) / denominator.astype(float)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Futures ML Impulse Comparison",
        "",
        f"- Symbol: `{report['symbol']}`",
        f"- Date range: `{report['date_range']['start']}` to `{report['date_range']['end']}`",
        f"- Profit barrier: `{report['profit_bps']}` bps",
        f"- Stop barrier: `{report['stop_bps']}` bps",
        f"- Master rows: `{report['master_rows']:,}`",
        f"- Long candidate rate: `{report['candidate_summary']['long_rate']:.4f}`",
        f"- Short candidate rate: `{report['candidate_summary']['short_rate']:.4f}`",
        "",
    ]
    for label_name, label_report in report["labels"].items():
        lines.append(f"## `{label_name}`")
        lines.append("")
        for model_key, model_info in label_report["models"].items():
            lines.append(f"- Train `{model_key}` rows: `{model_info['train_label_rows']:,}`")
            for eval_key, evaluation in model_info["evaluations"].items():
                if evaluation is None:
                    lines.append(f"  - Eval `{eval_key}`: no rows")
                    continue
                metrics = evaluation["metrics"]
                lines.append(
                    f"  - Eval `{eval_key}`: AUC `{metrics['auc']:.4f}`, precision@top-decile `{metrics['precision_at_top_decile']:.4f}`, base rate `{metrics['base_rate']:.4f}`, rows `{evaluation['label_rows']:,}`"
                )
            lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
