#!/usr/bin/env python3
"""
Train completeness-threshold variants on the same master dataset and score them on the same fixed test slices.

This avoids the sloppy comparison where each model is trained and tested on a different row population.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from catboost import Pool

from futures_ml_pipeline import (
    DEFAULT_BUCKET_SECONDS,
    DEFAULT_CANDIDATE_MIN_DEPTH_IMBALANCE,
    DEFAULT_CANDIDATE_MIN_SIGNED_RATIO,
    DEFAULT_CANDIDATE_MIN_TRADE_Z,
    DEFAULT_COST_BPS,
    DEFAULT_HORIZON_SECONDS,
    DEFAULT_LABEL_MODE,
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
    candidate_flag_for_label,
    chronological_split,
    download_archives,
    evaluate_split,
    feature_columns_for_dataset,
    parse_iso_date,
    resolve_raw_root,
    select_label_frame,
    summarise_source_coverage,
    top_feature_importances,
)


DEFAULT_COMPARE_OUTPUT_ROOT = Path("output/futures_ml_compare")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a controlled futures ML completeness comparison.")
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
    parser.add_argument("--max-trade-age-buckets", type=int, default=DEFAULT_MAX_TRADE_AGE_BUCKETS, help="Maximum trade staleness in buckets")
    parser.add_argument("--max-depth-age-buckets", type=int, default=DEFAULT_MAX_DEPTH_AGE_BUCKETS, help="Maximum depth staleness in buckets")
    parser.add_argument("--max-metrics-age-buckets", type=int, default=DEFAULT_MAX_METRICS_AGE_BUCKETS, help="Maximum metrics staleness in buckets")
    parser.add_argument("--max-funding-age-buckets", type=int, default=DEFAULT_MAX_FUNDING_AGE_BUCKETS, help="Maximum funding staleness in buckets")
    parser.add_argument("--price-context-interval", default=DEFAULT_PRICE_CONTEXT_INTERVAL, help="Interval for mark/index/premium price context klines")
    parser.add_argument("--live-book-ticker-root", default="", help="Optional root containing captured live bookTicker JSONL files")
    parser.add_argument(
        "--train-thresholds",
        default="0.75,1.0",
        help="Comma-separated source completeness thresholds used for training variants",
    )
    parser.add_argument(
        "--eval-thresholds",
        default="0.75,1.0",
        help="Comma-separated source completeness thresholds used for identical test slices",
    )
    parser.add_argument(
        "--anchor-threshold",
        type=float,
        help="Threshold whose timeline defines the shared train/valid/test windows. Defaults to the highest threshold.",
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
    quote_tag = "_qtlive" if args.live_book_ticker_root else ""
    run_name = f"binance_{symbol.lower()}_{args.bucket_seconds}s_{args.horizon_seconds}s_{args.label_mode}_{date_tag}{quote_tag}_compare_train-{train_tags}_eval-{eval_tags}_v1"
    run_root = output_root / run_name
    dataset_root = run_root / "dataset"
    model_root = run_root / "models"
    report_root = run_root / "reports"
    for path in (dataset_root, model_root, report_root):
        path.mkdir(parents=True, exist_ok=True)

    raw_root = resolve_raw_root(output_root=raw_output_root, symbol=symbol, skip_download=args.skip_download)
    live_book_ticker_root = Path(args.live_book_ticker_root).resolve() if args.live_book_ticker_root else None
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

    master_dataset = build_feature_dataset(
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
        min_source_completeness=0.0,
        max_trade_age_buckets=args.max_trade_age_buckets,
        max_depth_age_buckets=args.max_depth_age_buckets,
        max_metrics_age_buckets=args.max_metrics_age_buckets,
        max_funding_age_buckets=args.max_funding_age_buckets,
        price_context_interval=args.price_context_interval,
        live_book_ticker_root=live_book_ticker_root,
    )
    if master_dataset.empty:
        raise SystemExit("Master dataset is empty. Check raw archive coverage.")

    master_dataset_path = dataset_root / "master_features.csv.gz"
    master_dataset.to_csv(master_dataset_path, index=False, compression="gzip")

    feature_columns = feature_columns_for_dataset(master_dataset)
    train_frame, valid_frame, test_frame = anchored_time_split(master_dataset, anchor_threshold)

    report = {
        "bundle_version": run_name,
        "created_at": datetime.now(UTC).isoformat(),
        "symbol": symbol,
        "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        "bucket_seconds": args.bucket_seconds,
        "horizon_seconds": args.horizon_seconds,
        "cost_bps": args.cost_bps,
        "label_mode": args.label_mode,
        "price_context_interval": args.price_context_interval,
        "live_book_ticker_root": str(live_book_ticker_root) if live_book_ticker_root else None,
        "train_thresholds": train_thresholds,
        "eval_thresholds": eval_thresholds,
        "anchor_threshold": anchor_threshold,
        "raw_root": str(raw_root),
        "master_dataset_path": str(master_dataset_path),
        "master_rows": int(len(master_dataset)),
        "master_coverage_summary": summarise_source_coverage(master_dataset),
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
        threshold_key = threshold_tag(eval_threshold)
        eval_base = threshold_frame(test_frame, eval_threshold)
        report["test_slices"][threshold_key] = {
            "base_rows": int(len(eval_base)),
            "coverage_summary": summarise_source_coverage(eval_base) if len(eval_base) else None,
        }

    for label_name in ("long_label", "short_label"):
        label_report: dict[str, Any] = {
            "candidate_flag": candidate_flag_for_label(label_name),
            "models": {},
        }
        for train_threshold in train_thresholds:
            variant_key = threshold_tag(train_threshold)
            train_base = threshold_frame(train_frame, train_threshold)
            valid_base = threshold_frame(valid_frame, train_threshold)
            label_train = select_label_frame(train_base, label_mode=args.label_mode, label_name=label_name)
            label_valid = select_label_frame(valid_base, label_mode=args.label_mode, label_name=label_name)
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
                label_eval = select_label_frame(eval_base, label_mode=args.label_mode, label_name=label_name)
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


def threshold_frame(frame: Any, threshold: float) -> Any:
    return frame[frame["source_fresh_score"] >= threshold].copy()


def anchored_time_split(dataset: Any, anchor_threshold: float) -> tuple[Any, Any, Any]:
    reference_frame = threshold_frame(dataset, anchor_threshold)
    if len(reference_frame) == 0:
        raise SystemExit(f"No rows available for anchor threshold {anchor_threshold}")
    ref_train, ref_valid, ref_test = chronological_split(reference_frame)
    if min(len(ref_train), len(ref_valid), len(ref_test)) == 0:
        raise SystemExit(f"Anchor threshold {anchor_threshold} produced an empty split")

    train_start = ref_train["timestamp"].min()
    train_end = ref_train["timestamp"].max()
    valid_start = ref_valid["timestamp"].min()
    valid_end = ref_valid["timestamp"].max()
    test_start = ref_test["timestamp"].min()
    test_end = ref_test["timestamp"].max()

    base = dataset.sort_values("timestamp").reset_index(drop=True)
    train_frame = base[(base["timestamp"] >= train_start) & (base["timestamp"] <= train_end)].copy()
    valid_frame = base[(base["timestamp"] >= valid_start) & (base["timestamp"] <= valid_end)].copy()
    test_frame = base[(base["timestamp"] >= test_start) & (base["timestamp"] <= test_end)].copy()
    if min(len(train_frame), len(valid_frame), len(test_frame)) == 0:
        raise SystemExit("Anchored time split produced an empty master window")
    return train_frame, valid_frame, test_frame


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Controlled Futures ML Comparison",
        "",
        f"- Symbol: `{report['symbol']}`",
        f"- Date range: `{report['date_range']['start']}` to `{report['date_range']['end']}`",
        f"- Label mode: `{report['label_mode']}`",
        f"- Anchor threshold: `{report['anchor_threshold']}`",
        f"- Master rows: `{report['master_rows']:,}`",
        f"- Test window: `{report['time_split']['test_start']}` to `{report['time_split']['test_end']}`",
        "",
        "## Test Slices",
        "",
    ]
    for eval_key, slice_info in report["test_slices"].items():
        lines.append(f"- `{eval_key}` base rows: `{slice_info['base_rows']:,}`")
    lines.extend(["", "## Label Results", ""])

    for label_name, label_report in report["labels"].items():
        lines.append(f"### `{label_name}`")
        lines.append("")
        for model_key, model_info in label_report["models"].items():
            lines.append(f"- Train `{model_key}` rows: `{model_info['train_label_rows']:,}` | best iteration `{model_info['best_iteration']}`")
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
