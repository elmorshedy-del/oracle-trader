#!/usr/bin/env python3
"""
Run an XGBoost comparison on the same 5-second BTCUSDT impulse dataset used for CatBoost.

This stays isolated from the frozen CatBoost baseline:
- same raw Binance data family
- same impulse labeling
- same chronological split discipline
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from futures_ml_impulse_compare import (
    anchored_time_split,
    impulse_feature_columns,
    prepare_impulse_dataset,
    threshold_frame,
)
from futures_ml_pipeline import (
    DEFAULT_BUCKET_SECONDS,
    DEFAULT_COST_BPS,
    DEFAULT_HORIZON_SECONDS,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MAX_DEPTH_AGE_BUCKETS,
    DEFAULT_MAX_FUNDING_AGE_BUCKETS,
    DEFAULT_MAX_METRICS_AGE_BUCKETS,
    DEFAULT_MAX_TRADE_AGE_BUCKETS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SYMBOL,
    build_feature_dataset,
    download_archives,
    parse_iso_date,
    resolve_raw_root,
    summarise_source_coverage,
)


DEFAULT_COMPARE_OUTPUT_ROOT = Path("output/futures_ml_xgboost_compare")
UTC = timezone.utc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare XGBoost on the 5-second BTCUSDT impulse dataset.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Binance futures symbol, default BTCUSDT")
    parser.add_argument("--start-date", help="Inclusive start date YYYY-MM-DD")
    parser.add_argument("--end-date", help="Inclusive end date YYYY-MM-DD")
    parser.add_argument("--lookback-days", type=int, default=90, help="Used if start-date is omitted")
    parser.add_argument("--bucket-seconds", type=int, default=DEFAULT_BUCKET_SECONDS, help="Feature aggregation bucket")
    parser.add_argument("--horizon-seconds", type=int, default=DEFAULT_HORIZON_SECONDS, help="Barrier horizon")
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS, help="Reference feature threshold cost")
    parser.add_argument("--profit-bps", type=float, default=8.0, help="Take-profit barrier in basis points")
    parser.add_argument("--stop-bps", type=float, default=6.0, help="Stop-loss barrier in basis points")
    parser.add_argument("--source-threshold", type=float, default=1.0, help="Minimum source freshness kept in the dataset")
    parser.add_argument("--min-signed-ratio", type=float, default=0.04, help="Minimum signed flow ratio for impulse candidates")
    parser.add_argument("--min-depth-imbalance", type=float, default=0.01, help="Minimum top-depth imbalance for impulse candidates")
    parser.add_argument("--min-trade-z", type=float, default=0.25, help="Minimum trade burst z-score for impulse candidates")
    parser.add_argument("--min-directional-efficiency", type=float, default=0.15, help="Minimum directional efficiency for impulse candidates")
    parser.add_argument("--max-trade-age-buckets", type=int, default=DEFAULT_MAX_TRADE_AGE_BUCKETS, help="Maximum trade staleness in buckets")
    parser.add_argument("--max-depth-age-buckets", type=int, default=DEFAULT_MAX_DEPTH_AGE_BUCKETS, help="Maximum depth staleness in buckets")
    parser.add_argument("--max-metrics-age-buckets", type=int, default=DEFAULT_MAX_METRICS_AGE_BUCKETS, help="Maximum metrics staleness in buckets")
    parser.add_argument("--max-funding-age-buckets", type=int, default=DEFAULT_MAX_FUNDING_AGE_BUCKETS, help="Maximum funding staleness in buckets")
    parser.add_argument("--n-estimators", type=int, default=400, help="XGBoost tree count")
    parser.add_argument("--max-depth", type=int, default=6, help="XGBoost max depth")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="XGBoost learning rate")
    parser.add_argument("--subsample", type=float, default=0.9, help="XGBoost subsample")
    parser.add_argument("--colsample-bytree", type=float, default=0.8, help="XGBoost colsample_bytree")
    parser.add_argument("--min-child-weight", type=float, default=1.0, help="XGBoost min_child_weight")
    parser.add_argument("--reg-lambda", type=float, default=1.0, help="XGBoost L2 regularization")
    parser.add_argument("--output-root", default=str(DEFAULT_COMPARE_OUTPUT_ROOT), help="Comparison output directory")
    parser.add_argument("--raw-output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Location of shared futures raw cache")
    parser.add_argument("--skip-download", action="store_true", help="Reuse existing downloaded archives")
    parser.add_argument(
        "--skip-dataset-export",
        action="store_true",
        help="Do not write the full master feature table to disk; faster for repeated compare runs.",
    )
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

    run_name = (
        f"binance_{symbol.lower()}_{args.bucket_seconds}s_impulse_xgboost_"
        f"{start_date:%Y%m%d}_{end_date:%Y%m%d}_"
        f"tp{int(round(args.profit_bps))}_sl{int(round(args.stop_bps))}_"
        f"sig{int(round(args.min_signed_ratio * 100)):03d}_"
        f"dep{int(round(args.min_depth_imbalance * 100)):03d}_"
        f"tz{int(round(args.min_trade_z * 100)):03d}_"
        f"eff{int(round(args.min_directional_efficiency * 100)):03d}_"
        f"src{int(round(args.source_threshold * 100)):03d}_v1"
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
            price_context_interval="",
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
        candidate_min_signed_ratio=0.08,
        candidate_min_depth_imbalance=0.02,
        candidate_min_trade_z=0.75,
        min_source_completeness=0.0,
        max_trade_age_buckets=args.max_trade_age_buckets,
        max_depth_age_buckets=args.max_depth_age_buckets,
        max_metrics_age_buckets=args.max_metrics_age_buckets,
        max_funding_age_buckets=args.max_funding_age_buckets,
        price_context_interval="",
        live_book_ticker_root=None,
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
    master_dataset = threshold_frame(master_dataset, args.source_threshold)
    if master_dataset.empty:
        raise SystemExit("Impulse dataset is empty after source-threshold filtering.")

    dataset_path = dataset_root / "master_features.csv.gz"
    if not args.skip_dataset_export:
        master_dataset.to_csv(dataset_path, index=False, compression="gzip")

    train_frame, valid_frame, test_frame = anchored_time_split(master_dataset, args.source_threshold)
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
        "source_threshold": args.source_threshold,
        "raw_root": str(raw_root),
        "master_dataset_path": str(dataset_path) if not args.skip_dataset_export else None,
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
        "xgboost_params": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "min_child_weight": args.min_child_weight,
            "reg_lambda": args.reg_lambda,
        },
        "labels": {},
    }

    for label_name, candidate_column in (
        ("long_followthrough_label", "long_impulse_candidate"),
        ("short_followthrough_label", "short_impulse_candidate"),
    ):
        label_train = train_frame[train_frame[candidate_column] == 1].copy()
        label_valid = valid_frame[valid_frame[candidate_column] == 1].copy()
        label_test = test_frame[test_frame[candidate_column] == 1].copy()
        if min(len(label_train), len(label_valid), len(label_test)) == 0:
            raise SystemExit(f"No rows available for {label_name}")

        model = build_xgboost(args)
        model.fit(
            label_train[feature_columns],
            label_train[label_name],
            eval_set=[(label_valid[feature_columns], label_valid[label_name])],
            verbose=False,
        )
        model_file = model_root / f"{label_name}.json"
        model.save_model(str(model_file))

        train_metrics = evaluate_split_xgb(model, label_train[feature_columns], label_train[label_name])
        valid_metrics = evaluate_split_xgb(model, label_valid[feature_columns], label_valid[label_name])
        test_metrics = evaluate_split_xgb(model, label_test[feature_columns], label_test[label_name])
        feature_gain = model.get_booster().get_score(importance_type="gain")
        top_features = [
            {"feature": feature, "importance": float(importance)}
            for feature, importance in sorted(feature_gain.items(), key=lambda item: item[1], reverse=True)[:20]
        ]

        report["labels"][label_name] = {
            "candidate_rows": {
                "train": int(len(label_train)),
                "valid": int(len(label_valid)),
                "test": int(len(label_test)),
            },
            "positive_rate": {
                "train": float(label_train[label_name].mean()),
                "valid": float(label_valid[label_name].mean()),
                "test": float(label_test[label_name].mean()),
            },
            "metrics": {
                "train": train_metrics,
                "valid": valid_metrics,
                "test": test_metrics,
            },
            "top_features": top_features,
        }

    json_path = report_root / "comparison_report.json"
    md_path = report_root / "comparison_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"XGBoost report: {md_path}")


def build_xgboost(args: argparse.Namespace) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        reg_lambda=args.reg_lambda,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=42,
        n_jobs=8,
    )


def evaluate_split_xgb(model: XGBClassifier, features: pd.DataFrame, labels: pd.Series) -> dict[str, float | int]:
    probs = model.predict_proba(features)[:, 1]
    labels_array = labels.to_numpy(dtype=float)
    auc = 0.5 if len(np.unique(labels_array)) < 2 else float(roc_auc_score(labels_array, probs))
    threshold = float(np.quantile(probs, 0.90))
    selected = probs >= threshold
    precision = float(labels_array[selected].mean()) if np.any(selected) else 0.0
    return {
        "auc": auc,
        "base_rate": float(labels_array.mean()),
        "precision_at_top_decile": precision,
        "top_decile_threshold": threshold,
        "samples": int(len(labels_array)),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Futures ML XGBoost Comparison",
        "",
        f"- Symbol: `{report['symbol']}`",
        f"- Date range: `{report['date_range']['start']}` to `{report['date_range']['end']}`",
        f"- Bucket: `{report['bucket_seconds']}`s",
        f"- Profit barrier: `{report['profit_bps']}` bps",
        f"- Stop barrier: `{report['stop_bps']}` bps",
        f"- Master rows: `{report['master_rows']:,}`",
        f"- Long candidate rate: `{report['candidate_summary']['long_rate']:.4f}`",
        f"- Short candidate rate: `{report['candidate_summary']['short_rate']:.4f}`",
        "",
    ]
    for label_name, info in report["labels"].items():
        lines.append(f"## `{label_name}`")
        lines.append("")
        lines.append(f"- Train rows: `{info['candidate_rows']['train']:,}`")
        lines.append(f"- Valid rows: `{info['candidate_rows']['valid']:,}`")
        lines.append(f"- Test rows: `{info['candidate_rows']['test']:,}`")
        for split_name in ("train", "valid", "test"):
            metrics = info["metrics"][split_name]
            lines.append(
                f"- {split_name.title()} AUC `{metrics['auc']:.4f}` | precision@top-decile `{metrics['precision_at_top_decile']:.4f}` | base rate `{metrics['base_rate']:.4f}`"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
