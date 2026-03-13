#!/usr/bin/env python3
"""
Train first-pass BTC multivenue CatBoost baselines.

Models:
- continuation_long_30s
- continuation_short_30s
- meanrev_after_upshock_30s
- meanrev_after_downshock_30s

The goal is not to replace any prior BTC champion. This creates a fresh,
versioned baseline inside the multivenue track.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


DEFAULT_OUTPUT_ROOT = Path("output/btc_multivenue_models")
DEFAULT_CONTINUATION_HORIZON_SECONDS = 30
DEFAULT_PROFIT_BPS = 8.0
DEFAULT_MEANREV_SHOCK_WINDOW_SECONDS = 5
DEFAULT_MEANREV_SHOCK_BPS = 10.0
DEFAULT_MEANREV_REVERT_BPS = 8.0
DEFAULT_MIN_ROWS = 200
UTC = timezone.utc


@dataclass
class ModelSummary:
    name: str
    status: str
    train_rows: int
    valid_rows: int
    test_rows: int
    positive_rate_train: float
    positive_rate_valid: float
    positive_rate_test: float
    valid_auc: float | None
    test_auc: float | None
    valid_precision_top_decile: float | None
    test_precision_top_decile: float | None
    model_path: str | None
    notes: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CatBoost baselines on the BTC multivenue dataset.")
    parser.add_argument("--dataset-path", required=True, help="Path to features.csv.gz from build_btc_multivenue_dataset.py")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output root for trained model runs")
    parser.add_argument("--continuation-horizon-seconds", type=int, default=DEFAULT_CONTINUATION_HORIZON_SECONDS, help="Future horizon for continuation labels")
    parser.add_argument("--profit-bps", type=float, default=DEFAULT_PROFIT_BPS, help="Positive continuation threshold in bps")
    parser.add_argument("--meanrev-shock-window-seconds", type=int, default=DEFAULT_MEANREV_SHOCK_WINDOW_SECONDS, help="Past return window used to define an overextension")
    parser.add_argument("--meanrev-shock-bps", type=float, default=DEFAULT_MEANREV_SHOCK_BPS, help="Minimum past impulse magnitude for a mean-reversion candidate")
    parser.add_argument("--meanrev-revert-bps", type=float, default=DEFAULT_MEANREV_REVERT_BPS, help="Required snapback magnitude in the future horizon")
    parser.add_argument("--min-rows", type=int, default=DEFAULT_MIN_ROWS, help="Minimum candidate rows required before a model is trained")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path).resolve()
    if not dataset_path.exists():
        raise SystemExit(f"Missing dataset: {dataset_path}")

    df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
    target_column = f"target_fut_ret_{args.continuation_horizon_seconds}s_bps"
    if target_column not in df.columns:
        raise SystemExit(f"Missing target column: {target_column}")

    feature_columns = [column for column in df.columns if not column.startswith("target_")]
    X = df[feature_columns].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    medians = X.median(numeric_only=True)
    X = X.fillna(medians).fillna(0.0)

    continuation_target = pd.to_numeric(df[target_column], errors="coerce")
    past_window_column = f"fut_ret_{args.meanrev_shock_window_seconds}s"
    if past_window_column not in df.columns:
        raise SystemExit(f"Missing past return feature required for mean reversion: {past_window_column}")
    past_return_bps = pd.to_numeric(df[past_window_column], errors="coerce") * 10000.0

    labels = {
        "continuation_long_30s": {
            "mask": continuation_target.notna(),
            "y": (continuation_target >= args.profit_bps).astype(int),
            "notes": f"Future {args.continuation_horizon_seconds}s return >= {args.profit_bps} bps",
        },
        "continuation_short_30s": {
            "mask": continuation_target.notna(),
            "y": (continuation_target <= -args.profit_bps).astype(int),
            "notes": f"Future {args.continuation_horizon_seconds}s return <= -{args.profit_bps} bps",
        },
        "meanrev_after_upshock_30s": {
            "mask": continuation_target.notna() & (past_return_bps >= args.meanrev_shock_bps),
            "y": (continuation_target <= -args.meanrev_revert_bps).astype(int),
            "notes": f"Past {args.meanrev_shock_window_seconds}s impulse >= {args.meanrev_shock_bps} bps then revert <= -{args.meanrev_revert_bps} bps in {args.continuation_horizon_seconds}s",
        },
        "meanrev_after_downshock_30s": {
            "mask": continuation_target.notna() & (past_return_bps <= -args.meanrev_shock_bps),
            "y": (continuation_target >= args.meanrev_revert_bps).astype(int),
            "notes": f"Past {args.meanrev_shock_window_seconds}s impulse <= -{args.meanrev_shock_bps} bps then revert >= {args.meanrev_revert_bps} bps in {args.continuation_horizon_seconds}s",
        },
    }

    started_at = datetime.now(UTC)
    run_name = f"btc_multivenue_catboost_{started_at.strftime('%Y%m%dT%H%M%S')}_v1"
    run_root = Path(args.output_root).resolve() / run_name
    model_root = run_root / "models"
    report_root = run_root / "reports"
    model_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    summaries: list[ModelSummary] = []
    for name, spec in labels.items():
        summary = train_one_model(
            name=name,
            X=X,
            y=spec["y"],
            mask=spec["mask"],
            min_rows=args.min_rows,
            model_path=model_root / f"{name}.cbm",
            notes=spec["notes"],
        )
        summaries.append(summary)

    report = {
        "run_name": run_name,
        "run_root": str(run_root),
        "model_root": str(model_root),
        "report_root": str(report_root),
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "dataset_path": str(dataset_path),
        "row_count": int(len(df)),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "models": [asdict(summary) for summary in summaries],
    }
    report_metadata_path = report_root / "metadata.json"
    report_markdown_path = report_root / "report.md"
    report["report_metadata_path"] = str(report_metadata_path)
    report["report_markdown_path"] = str(report_markdown_path)
    report_metadata_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_markdown_path.write_text(render_report(report), encoding="utf-8")
    print(json.dumps(report, indent=2))


def train_one_model(
    *,
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    mask: pd.Series,
    min_rows: int,
    model_path: Path,
    notes: str,
) -> ModelSummary:
    subset = X.loc[mask].copy()
    target = y.loc[mask].astype(int)
    subset = subset.loc[target.index]
    total_rows = len(subset)
    if total_rows < min_rows:
        return ModelSummary(
            name=name,
            status="skipped",
            train_rows=0,
            valid_rows=0,
            test_rows=0,
            positive_rate_train=0.0,
            positive_rate_valid=0.0,
            positive_rate_test=0.0,
            valid_auc=None,
            test_auc=None,
            valid_precision_top_decile=None,
            test_precision_top_decile=None,
            model_path=None,
            notes=f"Not enough candidate rows ({total_rows} < {min_rows}). {notes}",
        )

    train_end = int(total_rows * 0.6)
    valid_end = int(total_rows * 0.8)
    X_train = subset.iloc[:train_end]
    y_train = target.iloc[:train_end]
    X_valid = subset.iloc[train_end:valid_end]
    y_valid = target.iloc[train_end:valid_end]
    X_test = subset.iloc[valid_end:]
    y_test = target.iloc[valid_end:]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_valid)) < 2 or len(np.unique(y_test)) < 2:
        return ModelSummary(
            name=name,
            status="skipped",
            train_rows=len(X_train),
            valid_rows=len(X_valid),
            test_rows=len(X_test),
            positive_rate_train=float(y_train.mean()) if len(y_train) else 0.0,
            positive_rate_valid=float(y_valid.mean()) if len(y_valid) else 0.0,
            positive_rate_test=float(y_test.mean()) if len(y_test) else 0.0,
            valid_auc=None,
            test_auc=None,
            valid_precision_top_decile=None,
            test_precision_top_decile=None,
            model_path=None,
            notes=f"Split lost class diversity. {notes}",
        )

    model = CatBoostClassifier(
        iterations=400,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)

    valid_pred = model.predict_proba(X_valid)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]
    model.save_model(str(model_path))

    return ModelSummary(
        name=name,
        status="trained",
        train_rows=len(X_train),
        valid_rows=len(X_valid),
        test_rows=len(X_test),
        positive_rate_train=float(y_train.mean()),
        positive_rate_valid=float(y_valid.mean()),
        positive_rate_test=float(y_test.mean()),
        valid_auc=binary_roc_auc(y_valid.to_numpy(), valid_pred),
        test_auc=binary_roc_auc(y_test.to_numpy(), test_pred),
        valid_precision_top_decile=precision_at_top_quantile(y_valid.to_numpy(), valid_pred, 0.9),
        test_precision_top_decile=precision_at_top_quantile(y_test.to_numpy(), test_pred, 0.9),
        model_path=str(model_path),
        notes=notes,
    )


def binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    scores = np.asarray(y_score, dtype=float)
    positives = y == 1
    negatives = y == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    rank = 1.0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (rank + (rank + (j - i) - 1)) / 2.0
        ranks[order[i:j]] = avg_rank
        rank += (j - i)
        i = j
    pos_ranks = ranks[positives]
    auc = (pos_ranks.sum() - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def precision_at_top_quantile(y_true: np.ndarray, y_score: np.ndarray, quantile: float) -> float | None:
    if len(y_true) == 0:
        return None
    threshold = float(np.quantile(y_score, quantile))
    mask = y_score >= threshold
    if mask.sum() == 0:
        return None
    return float(np.mean(y_true[mask]))


def render_report(report: dict[str, object]) -> str:
    lines = [
        "# BTC Multivenue CatBoost Baselines",
        "",
        f"- Dataset: `{report['dataset_path']}`",
        f"- Rows: `{report['row_count']}`",
        f"- Features: `{report['feature_count']}`",
        "",
    ]
    for model in report["models"]:
        lines.append(f"## `{model['name']}`")
        lines.append("")
        lines.append(f"- Status: `{model['status']}`")
        lines.append(f"- Train / valid / test: `{model['train_rows']}` / `{model['valid_rows']}` / `{model['test_rows']}`")
        lines.append(f"- Valid AUC: `{model['valid_auc']}`")
        lines.append(f"- Test AUC: `{model['test_auc']}`")
        lines.append(f"- Valid precision@top-decile: `{model['valid_precision_top_decile']}`")
        lines.append(f"- Test precision@top-decile: `{model['test_precision_top_decile']}`")
        lines.append(f"- Notes: {model['notes']}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


if __name__ == "__main__":
    main()
