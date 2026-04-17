#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


UTC = timezone.utc
DEFAULT_OUTPUT_ROOT = Path("output/wallet_copy_ml")
DEFAULT_API_BASE_URL = "https://just-grace-production-a401.up.railway.app"
DEFAULT_PAGE_SIZE = 500
DEFAULT_MIN_ROWS = 2000
DEFAULT_RANDOM_SEED = 42

EXPORT_COLUMNS = [
    "id",
    "trade_key",
    "timestamp",
    "detected_at",
    "wallet_address",
    "market_condition_id",
    "asset_token_id",
    "market_title",
    "market_slug",
    "outcome",
    "price",
    "size_shares",
    "size_usd",
    "token_best_bid",
    "token_best_ask",
    "token_best_bid_size",
    "token_best_ask_size",
    "token_depth_within_2pct",
    "market_yes_bid",
    "market_yes_ask",
    "market_no_bid",
    "market_no_ask",
    "market_spread",
    "market_volume_24h",
    "market_volume_total",
    "market_liquidity",
    "market_midpoint",
    "market_reward_pool",
    "market_primary_tag",
    "market_seconds_to_expiry",
    "market_category",
    "wallet_leaderboard_rank",
    "wallet_leaderboard_profit",
    "wallet_leaderboard_win_rate",
    "wallet_open_positions",
    "wallet_trade_count_24h",
    "is_adding_to_position",
    "size_vs_wallet_avg",
    "detection_delay_seconds",
    "hour_of_day",
    "day_of_week",
    "btc_price",
    "btc_momentum_60s",
    "resolution_timestamp",
    "resolution_return",
    "is_win_resolution",
    "winning_outcome",
]

CATEGORICAL_FEATURES = ["market_category", "market_primary_tag"]
NUMERIC_FEATURES = [
    "price",
    "price_distance_from_50",
    "size_shares",
    "size_usd",
    "log_size_shares",
    "log_size_usd",
    "token_best_bid",
    "token_best_ask",
    "token_best_bid_size",
    "token_best_ask_size",
    "token_depth_within_2pct",
    "token_spread",
    "token_midpoint",
    "price_vs_token_mid",
    "market_yes_bid",
    "market_yes_ask",
    "market_no_bid",
    "market_no_ask",
    "market_spread",
    "spread_frac_of_price",
    "market_volume_24h",
    "market_volume_total",
    "market_liquidity",
    "market_midpoint",
    "price_vs_market_mid",
    "market_reward_pool",
    "hours_to_expiry",
    "wallet_leaderboard_rank",
    "wallet_leaderboard_rank_recip",
    "wallet_leaderboard_profit",
    "wallet_leaderboard_profit_log",
    "wallet_leaderboard_win_rate_norm",
    "wallet_open_positions",
    "wallet_trade_count_24h",
    "wallet_trade_rate_per_hour",
    "is_adding_to_position",
    "size_vs_wallet_avg",
    "detection_delay_seconds",
    "hour_of_day",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "btc_price",
    "btc_momentum_60s",
    "observed_market_age_hours",
    "prior_wallet_resolution_trades",
    "prior_wallet_resolution_win_rate",
    "prior_wallet_avg_resolution_return",
    "prior_wallet_avg_hold_hours",
]

SQL_SELECT = """
SELECT
    id,
    trade_key,
    timestamp,
    detected_at,
    wallet_address,
    market_condition_id,
    asset_token_id,
    market_title,
    market_slug,
    outcome,
    price,
    size_shares,
    size_usd,
    token_best_bid,
    token_best_ask,
    token_best_bid_size,
    token_best_ask_size,
    token_depth_within_2pct,
    market_yes_bid,
    market_yes_ask,
    market_no_bid,
    market_no_ask,
    market_spread,
    market_volume_24h,
    market_volume_total,
    market_liquidity,
    market_midpoint,
    market_reward_pool,
    market_primary_tag,
    market_seconds_to_expiry,
    market_category,
    wallet_leaderboard_rank,
    wallet_leaderboard_profit,
    wallet_leaderboard_win_rate,
    wallet_open_positions,
    wallet_trade_count_24h,
    is_adding_to_position,
    size_vs_wallet_avg,
    detection_delay_seconds,
    hour_of_day,
    day_of_week,
    btc_price,
    btc_momentum_60s,
    resolution_timestamp,
    resolution_return,
    is_win_resolution,
    winning_outcome
FROM wallet_trades
WHERE id > ?
  AND side = 'BUY'
  AND COALESCE(price, 0) > 0
  AND COALESCE(size_usd, 0) > 0
  AND COALESCE(market_closed, 0) = 0
  AND (market_seconds_to_expiry IS NULL OR market_seconds_to_expiry > 0)
  AND market_resolved = 1
ORDER BY id ASC
LIMIT ?
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a v0 wallet-copy ML probe.")
    parser.add_argument("--db-path", default="wallet_copy_research.sqlite", help="Optional local wallet-copy SQLite path")
    parser.add_argument("--rows-json", default="", help="Optional pre-exported JSON file with resolved training rows")
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL, help="Base URL for paginated wallet-copy export")
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE, help="Rows to request per API page")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output root for model artifacts and reports")
    parser.add_argument("--min-rows", type=int, default=DEFAULT_MIN_ROWS, help="Minimum rows required to train the probe")
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed")
    parser.add_argument("--catboost-iterations", type=int, default=500, help="CatBoost training iterations")
    parser.add_argument("--lightgbm-estimators", type=int, default=500, help="LightGBM estimator count")
    parser.add_argument(
        "--target-mode",
        choices=("resolution_win", "forward_abs_move"),
        default="resolution_win",
        help="Prediction target to train against",
    )
    parser.add_argument(
        "--forward-column",
        default="price_5min_after",
        help="Forward checkpoint column for forward_abs_move mode",
    )
    parser.add_argument(
        "--forward-threshold",
        type=float,
        default=0.01,
        help="Minimum absolute price improvement required in forward_abs_move mode",
    )
    parser.add_argument(
        "--model-output-dir",
        default="",
        help="Optional directory to write the fitted CatBoost model and deployment metadata",
    )
    return parser.parse_args()


def target_slug(args: argparse.Namespace) -> str:
    if args.target_mode == "resolution_win":
        return "resolution_win"
    column = str(args.forward_column).replace("_after", "").replace("price_", "")
    threshold_bps = int(round(float(args.forward_threshold) * 10000))
    return f"{column}_plus_{threshold_bps}bps"


def normalize_rate(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return None
    if rate > 1.0:
        rate /= 100.0
    if rate < 0.0:
        return None
    return min(rate, 1.0)


def log1p_nonnegative(series: pd.Series) -> pd.Series:
    return np.log1p(pd.to_numeric(series, errors="coerce").clip(lower=0.0))


def query_local_resolution_rows(db_path: Path, *, page_size: int) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows: list[dict[str, Any]] = []
    last_id = 0
    try:
        while True:
            page = conn.execute(SQL_SELECT, (last_id, page_size)).fetchall()
            if not page:
                break
            payload = [dict(row) for row in page]
            rows.extend(payload)
            last_id = int(payload[-1]["id"])
            if len(payload) < page_size:
                break
    finally:
        conn.close()
    return rows


def fetch_api_resolution_rows(*, base_url: str, page_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    next_after_id = 0
    endpoint = base_url.rstrip("/") + "/api/wallet_resolution_rows"
    with httpx.Client(timeout=30.0) as client:
        while True:
            response = client.get(endpoint, params={"after_id": next_after_id, "limit": page_size})
            response.raise_for_status()
            payload = response.json()
            page = payload.get("rows") or []
            if not page:
                break
            rows.extend(page)
            next_after_id = int(payload.get("next_after_id") or page[-1]["id"])
            if not payload.get("has_more"):
                break
    return rows


def prepare_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["timestamp", "id"], kind="mergesort").reset_index(drop=True)

    numeric_columns = [
        "timestamp",
        "detected_at",
        "price",
        "size_shares",
        "size_usd",
        "token_best_bid",
        "token_best_ask",
        "token_best_bid_size",
        "token_best_ask_size",
        "token_depth_within_2pct",
        "market_yes_bid",
        "market_yes_ask",
        "market_no_bid",
        "market_no_ask",
        "market_spread",
        "market_volume_24h",
        "market_volume_total",
        "market_liquidity",
        "market_midpoint",
        "market_reward_pool",
        "market_seconds_to_expiry",
        "wallet_leaderboard_rank",
        "wallet_leaderboard_profit",
        "wallet_open_positions",
        "wallet_trade_count_24h",
        "is_adding_to_position",
        "size_vs_wallet_avg",
        "detection_delay_seconds",
        "hour_of_day",
        "day_of_week",
        "btc_price",
        "btc_momentum_60s",
        "resolution_timestamp",
        "resolution_return",
        "is_win_resolution",
        "price_1min_after",
        "price_5min_after",
        "price_30min_after",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["wallet_leaderboard_win_rate_norm"] = df["wallet_leaderboard_win_rate"].map(normalize_rate)
    df["price_distance_from_50"] = (df["price"] - 0.5).abs()
    df["log_size_shares"] = log1p_nonnegative(df["size_shares"])
    df["log_size_usd"] = log1p_nonnegative(df["size_usd"])
    df["token_spread"] = df["token_best_ask"] - df["token_best_bid"]
    df["token_midpoint"] = (df["token_best_bid"] + df["token_best_ask"]) / 2.0
    df["price_vs_token_mid"] = df["price"] - df["token_midpoint"]
    df["price_vs_market_mid"] = df["price"] - df["market_midpoint"]
    df["spread_frac_of_price"] = df["market_spread"] / df["price"].replace(0, np.nan)
    df["wallet_leaderboard_rank_recip"] = 1.0 / (df["wallet_leaderboard_rank"].clip(lower=0.0) + 1.0)
    df["wallet_leaderboard_profit_log"] = np.sign(df["wallet_leaderboard_profit"].fillna(0.0)) * np.log1p(
        df["wallet_leaderboard_profit"].abs().fillna(0.0)
    )
    df["wallet_trade_rate_per_hour"] = df["wallet_trade_count_24h"] / 24.0
    df["hours_to_expiry"] = df["market_seconds_to_expiry"] / 3600.0
    radians = 2.0 * np.pi * df["hour_of_day"].fillna(0.0) / 24.0
    df["hour_sin"] = np.sin(radians)
    df["hour_cos"] = np.cos(radians)
    first_seen_by_market = df.groupby("market_condition_id")["timestamp"].transform("min")
    df["observed_market_age_hours"] = (df["timestamp"] - first_seen_by_market) / 3600.0
    df["market_category"] = df["market_category"].fillna("unknown").astype(str)
    df["market_primary_tag"] = df["market_primary_tag"].fillna("unknown").astype(str)

    prior_trade_counts: list[float] = []
    prior_win_rates: list[float] = []
    prior_avg_returns: list[float] = []
    prior_avg_holds: list[float] = []
    wallet_stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "wins": 0.0, "return_sum": 0.0, "hold_sum": 0.0}
    )
    resolution_rows = (
        df[["wallet_address", "resolution_timestamp", "is_win_resolution", "resolution_return", "timestamp"]]
        .dropna(subset=["resolution_timestamp"])
        .sort_values(["resolution_timestamp", "timestamp"], kind="mergesort")
        .to_dict("records")
    )
    resolution_idx = 0
    for row in df.to_dict("records"):
        current_ts = float(row.get("timestamp") or 0.0)
        while resolution_idx < len(resolution_rows):
            completed = resolution_rows[resolution_idx]
            completed_ts = float(completed.get("resolution_timestamp") or 0.0)
            if completed_ts >= current_ts:
                break
            wallet_address = str(completed.get("wallet_address") or "")
            if wallet_address:
                stats = wallet_stats[wallet_address]
                stats["count"] += 1.0
                stats["wins"] += float(completed.get("is_win_resolution") or 0.0)
                stats["return_sum"] += float(completed.get("resolution_return") or 0.0)
                hold_hours = max(completed_ts - float(completed.get("timestamp") or 0.0), 0.0) / 3600.0
                stats["hold_sum"] += hold_hours
            resolution_idx += 1
        stats = wallet_stats[str(row.get("wallet_address") or "")]
        count = stats["count"]
        prior_trade_counts.append(count)
        prior_win_rates.append((stats["wins"] / count) if count else np.nan)
        prior_avg_returns.append((stats["return_sum"] / count) if count else np.nan)
        prior_avg_holds.append((stats["hold_sum"] / count) if count else np.nan)
    df["prior_wallet_resolution_trades"] = prior_trade_counts
    df["prior_wallet_resolution_win_rate"] = prior_win_rates
    df["prior_wallet_avg_resolution_return"] = prior_avg_returns
    df["prior_wallet_avg_hold_hours"] = prior_avg_holds

    return df


def apply_target(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    target_label = "target"
    frame = df.copy()
    if args.target_mode == "resolution_win":
        frame = frame.dropna(subset=["is_win_resolution"]).copy()
        frame[target_label] = frame["is_win_resolution"].astype(int)
        return frame
    if args.target_mode == "forward_abs_move":
        forward_column = args.forward_column
        if forward_column not in frame.columns:
            raise ValueError(f"missing forward column: {forward_column}")
        frame = frame.dropna(subset=[forward_column, "price"]).copy()
        frame["forward_abs_move"] = pd.to_numeric(frame[forward_column], errors="coerce") - pd.to_numeric(frame["price"], errors="coerce")
        frame = frame.dropna(subset=["forward_abs_move"]).copy()
        frame[target_label] = (frame["forward_abs_move"] >= float(args.forward_threshold)).astype(int)
        return frame
    raise ValueError(f"unsupported target mode: {args.target_mode}")


def ensure_class_diversity(name: str, y: pd.Series) -> None:
    if y.nunique(dropna=True) < 2:
        raise ValueError(f"{name} split lost class diversity")


def build_splits(df: pd.DataFrame, *, gap_seconds: float = 60.0) -> dict[str, pd.DataFrame]:
    if len(df) < 10:
        raise ValueError("not enough rows for temporal split")
    valid_boundary = max(1, int(len(df) * 0.7))
    test_boundary = max(valid_boundary + 1, int(len(df) * 0.8))
    valid_start_ts = float(df.iloc[valid_boundary]["timestamp"])
    test_start_ts = float(df.iloc[test_boundary]["timestamp"])

    train = df[df["timestamp"] < (valid_start_ts - gap_seconds)].copy()
    valid = df[(df["timestamp"] >= valid_start_ts) & (df["timestamp"] < (test_start_ts - gap_seconds))].copy()
    test = df[df["timestamp"] >= test_start_ts].copy()

    if train.empty or valid.empty or test.empty:
        train = df.iloc[:valid_boundary].copy()
        valid = df.iloc[valid_boundary:test_boundary].copy()
        test = df.iloc[test_boundary:].copy()

    ensure_class_diversity("train", train["target"])
    ensure_class_diversity("valid", valid["target"])
    ensure_class_diversity("test", test["target"])
    return {"train": train, "valid": valid, "test": test}


def build_tabular_matrices(splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
    ordered_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    combined = pd.concat([splits["train"], splits["valid"], splits["test"]], axis=0, ignore_index=True)
    model_frame = combined[ordered_columns].copy()
    numeric_frame = model_frame[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    train_numeric = numeric_frame.iloc[: len(splits["train"])]
    medians = train_numeric.median(numeric_only=True)
    numeric_frame = numeric_frame.fillna(medians).fillna(0.0)
    categorical_frame = pd.get_dummies(model_frame[CATEGORICAL_FEATURES].fillna("unknown").astype(str), dummy_na=False)
    tabular = pd.concat([numeric_frame, categorical_frame], axis=1)

    train_end = len(splits["train"])
    valid_end = train_end + len(splits["valid"])
    return {
        "train": tabular.iloc[:train_end].copy(),
        "valid": tabular.iloc[train_end:valid_end].copy(),
        "test": tabular.iloc[valid_end:].copy(),
        "feature_names": list(tabular.columns),
    }


def train_catboost_subset(
    *,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    y_test: pd.Series,
    feature_columns: list[str],
    iterations: int,
    random_seed: int,
) -> dict[str, Any]:
    train_x = train[feature_columns].copy()
    valid_x = valid[feature_columns].copy()
    test_x = test[feature_columns].copy()
    cat_indices = [idx for idx, column in enumerate(feature_columns) if column in CATEGORICAL_FEATURES]
    model = CatBoostClassifier(
        iterations=iterations,
        depth=6,
        learning_rate=0.03,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=random_seed,
        verbose=False,
        allow_writing_files=False,
        auto_class_weights="Balanced",
    )
    model.fit(
        train_x,
        y_train,
        cat_features=cat_indices,
        eval_set=(valid_x, y_valid),
        use_best_model=True,
    )
    valid_score = model.predict_proba(valid_x)[:, 1]
    test_score = model.predict_proba(test_x)[:, 1]
    return {
        "valid": evaluate_predictions(y_valid, valid_score),
        "test": evaluate_predictions(y_test, test_score),
    }


def precision_at_top_decile(y_true: pd.Series, y_score: np.ndarray) -> float | None:
    if len(y_true) < 10:
        return None
    cutoff = max(int(np.ceil(len(y_true) * 0.1)), 1)
    order = np.argsort(y_score)[::-1][:cutoff]
    return float(np.asarray(y_true)[order].mean())


def score_cutoff_at_top_fraction(y_score: np.ndarray, fraction: float) -> float | None:
    score = np.asarray(y_score, dtype=float)
    if len(score) < 10:
        return None
    cutoff = max(int(np.ceil(len(score) * fraction)), 1)
    order = np.argsort(score)[::-1][:cutoff]
    return float(score[order[-1]])


def evaluate_predictions(y_true: pd.Series, y_score: np.ndarray) -> dict[str, Any]:
    labels = y_true.to_numpy(dtype=int)
    score = np.asarray(y_score, dtype=float)
    pred = (score >= 0.5).astype(int)
    metrics: dict[str, Any] = {
        "rows": int(len(labels)),
        "positive_rate": float(labels.mean()) if len(labels) else None,
        "auc": None,
        "average_precision": None,
        "logloss": None,
        "brier": None,
        "accuracy": float(accuracy_score(labels, pred)) if len(labels) else None,
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)) if len(labels) else None,
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "precision_top_decile": precision_at_top_decile(y_true, score),
        "score_top_decile_cutoff": score_cutoff_at_top_fraction(score, 0.10),
        "score_top_5pct_cutoff": score_cutoff_at_top_fraction(score, 0.05),
    }
    if len(np.unique(labels)) >= 2:
        metrics["auc"] = float(roc_auc_score(labels, score))
        metrics["average_precision"] = float(average_precision_score(labels, score))
        metrics["logloss"] = float(log_loss(labels, np.clip(score, 1e-6, 1 - 1e-6)))
        metrics["brier"] = float(brier_score_loss(labels, score))
    return metrics


def evaluate_slice_set(
    *,
    name: str,
    frame: pd.DataFrame,
    y_score: np.ndarray,
    target_col: str = "is_win_resolution",
) -> dict[str, Any]:
    if frame.empty:
        return {"name": name, "rows": 0}
    metrics = evaluate_predictions(frame[target_col], y_score)
    return {"name": name, **metrics}


def build_slice_reports(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    test_scores: np.ndarray,
) -> list[dict[str, Any]]:
    reports = [evaluate_slice_set(name="overall_test", frame=test, y_score=test_scores, target_col="target")]
    seen_markets = set(train["market_condition_id"].astype(str))
    seen_wallets = set(train["wallet_address"].astype(str))
    unseen_markets = test.loc[~test["market_condition_id"].astype(str).isin(seen_markets)]
    unseen_wallets = test.loc[~test["wallet_address"].astype(str).isin(seen_wallets)]
    if not unseen_markets.empty:
        reports.append(
            evaluate_slice_set(
                name="unseen_markets",
                frame=unseen_markets,
                y_score=test_scores[unseen_markets.index.to_numpy()],
                target_col="target",
            )
        )
    if not unseen_wallets.empty:
        reports.append(
            evaluate_slice_set(
                name="unseen_wallets",
                frame=unseen_wallets,
                y_score=test_scores[unseen_wallets.index.to_numpy()],
                target_col="target",
            )
        )
    for category, category_frame in test.groupby("market_category"):
        if len(category_frame) < 100:
            continue
        reports.append(
            evaluate_slice_set(
                name=f"category:{category}",
                frame=category_frame,
                y_score=test_scores[category_frame.index.to_numpy()],
                target_col="target",
            )
        )
    return reports


def train_probe(args: argparse.Namespace) -> dict[str, Any]:
    local_db_path = Path(args.db_path).expanduser().resolve()
    rows_json_path = Path(args.rows_json).expanduser().resolve() if args.rows_json else None
    source = "api"
    if rows_json_path and rows_json_path.exists():
        rows = json.loads(rows_json_path.read_text(encoding="utf-8"))
        source = f"json:{rows_json_path}"
    else:
        rows = query_local_resolution_rows(local_db_path, page_size=args.page_size)
        if rows:
            source = f"sqlite:{local_db_path}"
        else:
            rows = fetch_api_resolution_rows(base_url=args.api_base_url, page_size=args.page_size)
            source = f"api:{args.api_base_url}"
    if len(rows) < args.min_rows:
        raise SystemExit(f"Need at least {args.min_rows} resolved rows, got {len(rows)}")

    df = prepare_frame(rows)
    df = apply_target(df, args)
    splits = build_splits(df)
    train = splits["train"].reset_index(drop=True)
    valid = splits["valid"].reset_index(drop=True)
    test = splits["test"].reset_index(drop=True)
    splits = {"train": train, "valid": valid, "test": test}

    y_train = train["target"].astype(int)
    y_valid = valid["target"].astype(int)
    y_test = test["target"].astype(int)

    tabular = build_tabular_matrices(splits)
    X_train_tab = tabular["train"]
    X_valid_tab = tabular["valid"]
    X_test_tab = tabular["test"]

    catboost_features = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    catboost_valid = valid[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    catboost_test = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    cat_feature_indices = [len(NUMERIC_FEATURES) + idx for idx, _ in enumerate(CATEGORICAL_FEATURES)]
    cat_model = CatBoostClassifier(
        iterations=args.catboost_iterations,
        depth=6,
        learning_rate=0.03,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=args.random_seed,
        verbose=False,
        allow_writing_files=False,
        auto_class_weights="Balanced",
    )
    cat_model.fit(
        catboost_features,
        y_train,
        cat_features=cat_feature_indices,
        eval_set=(catboost_valid, y_valid),
        use_best_model=True,
    )
    cat_valid_score = cat_model.predict_proba(catboost_valid)[:, 1]
    cat_test_score = cat_model.predict_proba(catboost_test)[:, 1]

    lgb_model = LGBMClassifier(
        n_estimators=args.lightgbm_estimators,
        max_depth=6,
        learning_rate=0.03,
        class_weight="balanced",
        random_state=args.random_seed,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_samples=40,
        reg_lambda=1.0,
        verbose=-1,
    )
    lgb_model.fit(X_train_tab, y_train, eval_set=[(X_valid_tab, y_valid)], eval_metric="auc")
    lgb_valid_score = lgb_model.predict_proba(X_valid_tab)[:, 1]
    lgb_test_score = lgb_model.predict_proba(X_test_tab)[:, 1]

    logistic = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=args.random_seed,
    )
    logistic.fit(X_train_tab, y_train)
    logistic_valid_score = logistic.predict_proba(X_valid_tab)[:, 1]
    logistic_test_score = logistic.predict_proba(X_test_tab)[:, 1]

    dummy = DummyClassifier(strategy="prior")
    dummy.fit(X_train_tab, y_train)
    dummy_valid_score = dummy.predict_proba(X_valid_tab)[:, 1]
    dummy_test_score = dummy.predict_proba(X_test_tab)[:, 1]

    ensemble_valid_score = (cat_valid_score + lgb_valid_score + logistic_valid_score) / 3.0
    ensemble_test_score = (cat_test_score + lgb_test_score + logistic_test_score) / 3.0

    subset_models = {
        "price_only_catboost": train_catboost_subset(
            train=train,
            valid=valid,
            test=test,
            y_train=y_train,
            y_valid=y_valid,
            y_test=y_test,
            feature_columns=["price", "price_distance_from_50", "token_midpoint", "market_midpoint"],
            iterations=args.catboost_iterations,
            random_seed=args.random_seed,
        ),
        "market_only_catboost": train_catboost_subset(
            train=train,
            valid=valid,
            test=test,
            y_train=y_train,
            y_valid=y_valid,
            y_test=y_test,
            feature_columns=[
                "price",
                "price_distance_from_50",
                "token_best_bid",
                "token_best_ask",
                "token_depth_within_2pct",
                "market_yes_bid",
                "market_yes_ask",
                "market_no_bid",
                "market_no_ask",
                "market_spread",
                "market_volume_24h",
                "market_volume_total",
                "market_liquidity",
                "market_midpoint",
                "market_reward_pool",
                "hours_to_expiry",
                "observed_market_age_hours",
                "market_category",
                "market_primary_tag",
            ],
            iterations=args.catboost_iterations,
            random_seed=args.random_seed,
        ),
        "wallet_only_catboost": train_catboost_subset(
            train=train,
            valid=valid,
            test=test,
            y_train=y_train,
            y_valid=y_valid,
            y_test=y_test,
            feature_columns=[
                "size_shares",
                "size_usd",
                "log_size_shares",
                "log_size_usd",
                "wallet_leaderboard_rank",
                "wallet_leaderboard_rank_recip",
                "wallet_leaderboard_profit_log",
                "wallet_leaderboard_win_rate_norm",
                "wallet_open_positions",
                "wallet_trade_count_24h",
                "wallet_trade_rate_per_hour",
                "is_adding_to_position",
                "size_vs_wallet_avg",
                "detection_delay_seconds",
                "hour_of_day",
                "hour_sin",
                "hour_cos",
                "day_of_week",
                "btc_price",
                "btc_momentum_60s",
                "prior_wallet_resolution_trades",
                "prior_wallet_resolution_win_rate",
                "prior_wallet_avg_resolution_return",
                "prior_wallet_avg_hold_hours",
            ],
            iterations=args.catboost_iterations,
            random_seed=args.random_seed,
        ),
    }

    models = {
        "dummy_prior": {"valid": evaluate_predictions(y_valid, dummy_valid_score), "test": evaluate_predictions(y_test, dummy_test_score)},
        "catboost": {"valid": evaluate_predictions(y_valid, cat_valid_score), "test": evaluate_predictions(y_test, cat_test_score)},
        "lightgbm": {"valid": evaluate_predictions(y_valid, lgb_valid_score), "test": evaluate_predictions(y_test, lgb_test_score)},
        "logistic": {"valid": evaluate_predictions(y_valid, logistic_valid_score), "test": evaluate_predictions(y_test, logistic_test_score)},
        "ensemble_mean": {"valid": evaluate_predictions(y_valid, ensemble_valid_score), "test": evaluate_predictions(y_test, ensemble_test_score)},
        **subset_models,
    }

    top_features = []
    importances = cat_model.get_feature_importance()
    for name, importance in sorted(
        zip(NUMERIC_FEATURES + CATEGORICAL_FEATURES, importances, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )[:20]:
        top_features.append({"feature": name, "importance": float(importance)})

    slice_reports = build_slice_reports(train=train, test=test, test_scores=ensemble_test_score)

    report = {
        "source": source,
        "target_mode": args.target_mode,
        "forward_column": args.forward_column if args.target_mode == "forward_abs_move" else None,
        "forward_threshold": args.forward_threshold if args.target_mode == "forward_abs_move" else None,
        "row_count": int(len(df)),
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "test_rows": int(len(test)),
        "train_positive_rate": float(y_train.mean()),
        "valid_positive_rate": float(y_valid.mean()),
        "test_positive_rate": float(y_test.mean()),
        "feature_count_catboost": len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES),
        "feature_count_tabular": int(X_train_tab.shape[1]),
        "feature_columns": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "models": models,
        "top_catboost_features": top_features,
        "slice_reports": slice_reports,
        "dataset_mix": {
            "categories": {
                str(key): int(value)
                for key, value in train.groupby("market_category").size().sort_values(ascending=False).items()
            },
            "wallets": int(train["wallet_address"].nunique()),
            "markets": int(train["market_condition_id"].nunique()),
        },
        "test_mix": {
            "categories": {
                str(key): int(value)
                for key, value in test.groupby("market_category").size().sort_values(ascending=False).items()
            },
            "wallets": int(test["wallet_address"].nunique()),
            "markets": int(test["market_condition_id"].nunique()),
        },
        "incremental_signal": {
            "ensemble_auc_minus_price_only_auc": (
                models["ensemble_mean"]["test"]["auc"] - models["price_only_catboost"]["test"]["auc"]
                if models["ensemble_mean"]["test"]["auc"] is not None and models["price_only_catboost"]["test"]["auc"] is not None
                else None
            ),
            "ensemble_ap_minus_price_only_ap": (
                models["ensemble_mean"]["test"]["average_precision"] - models["price_only_catboost"]["test"]["average_precision"]
                if models["ensemble_mean"]["test"]["average_precision"] is not None
                and models["price_only_catboost"]["test"]["average_precision"] is not None
                else None
            ),
            "ensemble_auc_minus_market_only_auc": (
                models["ensemble_mean"]["test"]["auc"] - models["market_only_catboost"]["test"]["auc"]
                if models["ensemble_mean"]["test"]["auc"] is not None and models["market_only_catboost"]["test"]["auc"] is not None
                else None
            ),
            "ensemble_ap_minus_market_only_ap": (
                models["ensemble_mean"]["test"]["average_precision"] - models["market_only_catboost"]["test"]["average_precision"]
                if models["ensemble_mean"]["test"]["average_precision"] is not None
                and models["market_only_catboost"]["test"]["average_precision"] is not None
                else None
            ),
        },
    }
    if args.model_output_dir:
        model_output_dir = Path(args.model_output_dir).expanduser().resolve()
        model_output_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_output_dir / "catboost.cbm"
        metadata_path = model_output_dir / "metadata.json"
        cat_model.save_model(str(model_path))
        deployment_metadata = sanitize_json(
            {
                "model_type": "catboost",
                "model_path": str(model_path),
                "target_slug": target_slug(args),
                "target_mode": report["target_mode"],
                "forward_column": report["forward_column"],
                "forward_threshold": report["forward_threshold"],
                "feature_columns": report["feature_columns"],
                "categorical_features": CATEGORICAL_FEATURES,
                "numeric_features": NUMERIC_FEATURES,
                "recommended_score_threshold": report["models"]["catboost"]["valid"].get("score_top_decile_cutoff"),
                "valid_metrics": report["models"]["catboost"]["valid"],
                "test_metrics": report["models"]["catboost"]["test"],
                "top_catboost_features": report["top_catboost_features"],
                "trained_at": datetime.now(UTC).isoformat(),
                "source": report["source"],
                "row_count": report["row_count"],
            }
        )
        metadata_path.write_text(json.dumps(deployment_metadata, indent=2), encoding="utf-8")
        report["model_artifacts"] = {
            "catboost_model_path": str(model_path),
            "metadata_path": str(metadata_path),
        }
    return report


def render_report(report: dict[str, Any]) -> str:
    ensemble_test = report["models"]["ensemble_mean"]["test"]
    cat_test = report["models"]["catboost"]["test"]
    lines = [
        "# Wallet Copy ML Probe",
        "",
        f"- Source: `{report['source']}`",
        f"- Target mode: `{report.get('target_mode')}`",
        f"- Forward column: `{report.get('forward_column')}`",
        f"- Forward threshold: `{report.get('forward_threshold')}`",
        f"- Rows: `{report['row_count']}`",
        f"- Split: train `{report['train_rows']}`, valid `{report['valid_rows']}`, test `{report['test_rows']}`",
        f"- Test positive rate: `{report['test_positive_rate']:.4f}`",
        "",
        "## Test Metrics",
        "",
        f"- Ensemble AUC: `{ensemble_test.get('auc')}`",
        f"- Ensemble average precision: `{ensemble_test.get('average_precision')}`",
        f"- Ensemble logloss: `{ensemble_test.get('logloss')}`",
        f"- CatBoost AUC: `{cat_test.get('auc')}`",
        "",
        "## Slice Reports",
        "",
    ]
    for item in report["slice_reports"]:
        if item.get("rows", 0) <= 0:
            continue
        lines.append(
            f"- `{item['name']}`: rows={item['rows']}, auc={item.get('auc')}, "
            f"ap={item.get('average_precision')}, precision_top_decile={item.get('precision_top_decile')}"
        )
    lines.extend(["", "## Top CatBoost Features", ""])
    for item in report["top_catboost_features"]:
        lines.append(f"- `{item['feature']}`: {item['importance']:.4f}")
    return "\n".join(lines) + "\n"


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    return value


def main() -> None:
    args = parse_args()
    started_at = datetime.now(UTC)
    report = train_probe(args)
    run_name = f"wallet_copy_probe_{target_slug(args)}_{started_at.strftime('%Y%m%dT%H%M%S%f')}_v1"
    run_root = Path(args.output_root).resolve() / run_name
    report_root = run_root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)

    report["run_name"] = run_name
    report["run_root"] = str(run_root)
    report["started_at"] = started_at.isoformat()
    report["finished_at"] = datetime.now(UTC).isoformat()

    report_json_path = report_root / "metadata.json"
    report_md_path = report_root / "report.md"
    report["report_json_path"] = str(report_json_path)
    report["report_md_path"] = str(report_md_path)
    sanitized = sanitize_json(report)
    report_json_path.write_text(json.dumps(sanitized, indent=2), encoding="utf-8")
    report_md_path.write_text(render_report(sanitized), encoding="utf-8")
    print(json.dumps(sanitized, indent=2))


if __name__ == "__main__":
    main()
