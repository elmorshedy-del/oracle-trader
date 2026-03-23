#!/usr/bin/env python3
"""
Replay the BTCUSDT impulse model on actual Binance aggTrades + bookDepth.

The workflow is strict:
- train models on the train split
- choose the replay setup only on the validation split
- report the chosen setup on the held-out test split

Execution assumptions stay explicit and configurable:
- signal latency
- depth-limited size
- depth-derived impact
- per-side taker fees
- cooldown and max concurrency
- take-profit / stop-loss / timeout exits
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import Pool

from futures_ml_impulse_compare import (
    anchored_time_split,
    impulse_feature_columns,
    parse_thresholds,
    prepare_impulse_dataset,
    threshold_frame,
    threshold_tag,
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
    build_classifier,
    build_feature_dataset,
    download_archives,
    parse_iso_date,
    resolve_raw_root,
)


DEFAULT_REPLAY_OUTPUT_ROOT = Path("output/futures_ml_replay")
DEFAULT_SOURCE_THRESHOLD = 1.0
DEFAULT_MIN_SIGNED_RATIO = 0.04
DEFAULT_MIN_DEPTH_IMBALANCE = 0.01
DEFAULT_MIN_TRADE_Z = 0.25
DEFAULT_MIN_DIRECTIONAL_EFFICIENCY = 0.15


@dataclass(frozen=True)
class ReplayConfig:
    long_quantile: float
    short_quantile: float
    position_notional_usd: float
    max_concurrent_positions: int
    cooldown_seconds: int
    latency_seconds: int
    timeout_seconds: int
    depth_fraction_limit: float


@dataclass(frozen=True)
class ReplayTrade:
    side: str
    signal_timestamp: str
    entry_timestamp: str
    exit_timestamp: str
    signal_prob: float
    threshold: float
    requested_notional_usd: float
    filled_notional_usd: float
    quantity: float
    entry_price_raw: float
    entry_price_filled: float
    exit_price_raw: float
    exit_price_filled: float
    entry_impact_bps: float
    exit_impact_bps: float
    gross_pnl_usd: float
    fees_usd: float
    net_pnl_usd: float
    hold_seconds: float
    exit_reason: str
    depth_snapshot_age_seconds_entry: float
    depth_snapshot_age_seconds_exit: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay impulse-model BTC futures setups on historical market data.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Binance futures symbol, default BTCUSDT")
    parser.add_argument("--start-date", help="Inclusive start date YYYY-MM-DD")
    parser.add_argument("--end-date", help="Inclusive end date YYYY-MM-DD")
    parser.add_argument("--lookback-days", type=int, default=90, help="Used if start-date is omitted")
    parser.add_argument("--bucket-seconds", type=int, default=DEFAULT_BUCKET_SECONDS, help="Feature aggregation bucket")
    parser.add_argument("--horizon-seconds", type=int, default=DEFAULT_HORIZON_SECONDS, help="Feature horizon")
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS, help="Reference feature threshold cost")
    parser.add_argument("--profit-bps", type=float, default=8.0, help="Take-profit barrier in basis points")
    parser.add_argument("--stop-bps", type=float, default=6.0, help="Stop-loss barrier in basis points")
    parser.add_argument("--source-threshold", type=float, default=DEFAULT_SOURCE_THRESHOLD, help="Minimum source freshness used for the replay dataset")
    parser.add_argument("--min-signed-ratio", type=float, default=DEFAULT_MIN_SIGNED_RATIO, help="Impulse signed-flow threshold")
    parser.add_argument("--min-depth-imbalance", type=float, default=DEFAULT_MIN_DEPTH_IMBALANCE, help="Impulse depth threshold")
    parser.add_argument("--min-trade-z", type=float, default=DEFAULT_MIN_TRADE_Z, help="Impulse trade-burst threshold")
    parser.add_argument("--min-directional-efficiency", type=float, default=DEFAULT_MIN_DIRECTIONAL_EFFICIENCY, help="Impulse directional-efficiency threshold")
    parser.add_argument("--max-trade-age-buckets", type=int, default=DEFAULT_MAX_TRADE_AGE_BUCKETS, help="Maximum trade staleness in buckets")
    parser.add_argument("--max-depth-age-buckets", type=int, default=DEFAULT_MAX_DEPTH_AGE_BUCKETS, help="Maximum depth staleness in buckets")
    parser.add_argument("--max-metrics-age-buckets", type=int, default=DEFAULT_MAX_METRICS_AGE_BUCKETS, help="Maximum metrics staleness in buckets")
    parser.add_argument("--max-funding-age-buckets", type=int, default=DEFAULT_MAX_FUNDING_AGE_BUCKETS, help="Maximum funding staleness in buckets")
    parser.add_argument("--fee-bps-per-side", type=float, default=2.0, help="Taker fee charged on each side of the trade")
    parser.add_argument("--max-depth-snapshot-age-seconds", type=int, default=30, help="Maximum age of the last depth snapshot used for execution")
    parser.add_argument("--min-fill-notional-usd", type=float, default=250.0, help="Minimum size after depth cap to count a fill")
    parser.add_argument("--long-quantiles", default="0.90,0.93,0.96,0.98", help="Validation quantiles to test for long signals")
    parser.add_argument("--short-quantiles", default="0.90,0.93,0.96,0.98", help="Validation quantiles to test for short signals")
    parser.add_argument("--position-notionals", default="1000,2500,5000", help="Requested position notionals in USD")
    parser.add_argument("--max-concurrent-grid", default="1,2,4", help="Maximum simultaneous open positions")
    parser.add_argument("--cooldown-seconds-grid", default="0,30,60", help="Cooldown after each fill")
    parser.add_argument("--latency-seconds-grid", default="1,5", help="Signal-to-entry latency assumptions")
    parser.add_argument("--timeout-seconds-grid", default="60,90", help="Timeout exits to test")
    parser.add_argument("--depth-fraction-grid", default="0.01,0.02,0.05", help="Max fraction of same-side 0.20%% depth that can be consumed")
    parser.add_argument("--leaderboard-limit", type=int, default=20, help="Number of validation setups kept in the report")
    parser.add_argument("--minimum-validation-trades", type=int, default=20, help="Validation trades required before a setup can win")
    parser.add_argument("--output-root", default=str(DEFAULT_REPLAY_OUTPUT_ROOT), help="Local output directory")
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

    run_name = (
        f"binance_{symbol.lower()}_{args.bucket_seconds}s_replay_"
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
        raise SystemExit("Base dataset is empty. Check archive coverage.")

    impulse_dataset = prepare_impulse_dataset(
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
    impulse_dataset = threshold_frame(impulse_dataset, args.source_threshold)
    if impulse_dataset.empty:
        raise SystemExit("Impulse dataset is empty after source-threshold filtering.")

    dataset_path = dataset_root / "master_features.csv.gz"
    impulse_dataset.to_csv(dataset_path, index=False, compression="gzip")

    train_frame, valid_frame, test_frame = anchored_time_split(impulse_dataset, args.source_threshold)
    feature_columns = impulse_feature_columns(impulse_dataset)

    models, signal_frames, model_metadata = train_and_score(
        train_frame=train_frame,
        valid_frame=valid_frame,
        test_frame=test_frame,
        feature_columns=feature_columns,
        model_root=model_root,
    )

    replay_market = ReplayMarket(
        raw_root=raw_root,
        symbol=symbol,
        max_depth_snapshot_age_seconds=args.max_depth_snapshot_age_seconds,
    )

    configs = list(generate_configs(args))
    leaderboard: list[dict[str, Any]] = []
    best_config: ReplayConfig | None = None
    best_validation_summary: dict[str, Any] | None = None

    for config in configs:
        long_threshold = float(signal_frames["valid"]["long"]["prob"].quantile(config.long_quantile))
        short_threshold = float(signal_frames["valid"]["short"]["prob"].quantile(config.short_quantile))
        validation_result = replay_signals(
            valid_long=signal_frames["valid"]["long"],
            valid_short=signal_frames["valid"]["short"],
            market=replay_market,
            config=config,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            profit_bps=args.profit_bps,
            stop_bps=args.stop_bps,
            fee_bps_per_side=args.fee_bps_per_side,
            min_fill_notional_usd=args.min_fill_notional_usd,
        )
        summary = summarise_replay(validation_result)
        leaderboard.append(
            {
                "config": asdict(config),
                "long_threshold": long_threshold,
                "short_threshold": short_threshold,
                "validation": summary,
            }
        )
        if summary["trades"] < args.minimum_validation_trades:
            continue
        if best_validation_summary is None or validation_key(summary) > validation_key(best_validation_summary):
            best_config = config
            best_validation_summary = {**summary, "long_threshold": long_threshold, "short_threshold": short_threshold}

    if best_config is None or best_validation_summary is None:
        raise SystemExit("No replay setup met the minimum-validation-trades requirement.")

    test_result = replay_signals(
        valid_long=signal_frames["test"]["long"],
        valid_short=signal_frames["test"]["short"],
        market=replay_market,
        config=best_config,
        long_threshold=float(best_validation_summary["long_threshold"]),
        short_threshold=float(best_validation_summary["short_threshold"]),
        profit_bps=args.profit_bps,
        stop_bps=args.stop_bps,
        fee_bps_per_side=args.fee_bps_per_side,
        min_fill_notional_usd=args.min_fill_notional_usd,
    )
    test_summary = summarise_replay(test_result)

    report = {
        "bundle_version": run_name,
        "created_at": datetime.now(UTC).isoformat(),
        "symbol": symbol,
        "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        "dataset_path": str(dataset_path),
        "raw_root": str(raw_root),
        "profit_bps": args.profit_bps,
        "stop_bps": args.stop_bps,
        "fee_bps_per_side": args.fee_bps_per_side,
        "max_depth_snapshot_age_seconds": args.max_depth_snapshot_age_seconds,
        "min_fill_notional_usd": args.min_fill_notional_usd,
        "source_threshold": args.source_threshold,
        "impulse_thresholds": {
            "min_signed_ratio": args.min_signed_ratio,
            "min_depth_imbalance": args.min_depth_imbalance,
            "min_trade_z": args.min_trade_z,
            "min_directional_efficiency": args.min_directional_efficiency,
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
        "model_metadata": model_metadata,
        "search_space": {
            "long_quantiles": parse_float_list(args.long_quantiles),
            "short_quantiles": parse_float_list(args.short_quantiles),
            "position_notionals": parse_float_list(args.position_notionals),
            "max_concurrent_grid": parse_int_list(args.max_concurrent_grid),
            "cooldown_seconds_grid": parse_int_list(args.cooldown_seconds_grid),
            "latency_seconds_grid": parse_int_list(args.latency_seconds_grid),
            "timeout_seconds_grid": parse_int_list(args.timeout_seconds_grid),
            "depth_fraction_grid": parse_float_list(args.depth_fraction_grid),
        },
        "validation_objective": "max net_pnl_usd with minimum trades; ties break on lower max_drawdown_usd then higher profit_factor",
        "leaderboard": sorted(leaderboard, key=lambda item: validation_key(item["validation"]), reverse=True)[: args.leaderboard_limit],
        "chosen_setup": {
            "config": asdict(best_config),
            "long_threshold": best_validation_summary["long_threshold"],
            "short_threshold": best_validation_summary["short_threshold"],
            "validation": best_validation_summary,
            "test": test_summary,
        },
        "sample_test_trades": [asdict(trade) for trade in test_result["trades"][:25]],
    }

    json_path = report_root / "replay_report.json"
    md_path = report_root / "replay_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    print(f"Replay report: {md_path}")
    print(f"Validation best: {best_validation_summary['net_pnl_usd']:.2f} USD across {best_validation_summary['trades']} trades")
    print(f"Test result: {test_summary['net_pnl_usd']:.2f} USD across {test_summary['trades']} trades")


def train_and_score(
    *,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
    model_root: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, pd.DataFrame]], dict[str, Any]]:
    models: dict[str, Any] = {}
    signals: dict[str, dict[str, pd.DataFrame]] = {"valid": {}, "test": {}}
    metadata: dict[str, Any] = {}
    for label_name, candidate_column, side in (
        ("long_followthrough_label", "long_impulse_candidate", "long"),
        ("short_followthrough_label", "short_impulse_candidate", "short"),
    ):
        label_train = train_frame[train_frame[candidate_column] == 1].copy()
        label_valid = valid_frame[valid_frame[candidate_column] == 1].copy()
        label_test = test_frame[test_frame[candidate_column] == 1].copy()
        if min(len(label_train), len(label_valid), len(label_test)) == 0:
            raise SystemExit(f"No rows available for {label_name}")

        model = build_classifier()
        train_pool = Pool(label_train[feature_columns], label=label_train[label_name])
        valid_pool = Pool(label_valid[feature_columns], label=label_valid[label_name])
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        model_file = model_root / f"{side}.cbm"
        model.save_model(str(model_file))

        valid_probs = model.predict_proba(label_valid[feature_columns])[:, 1]
        test_probs = model.predict_proba(label_test[feature_columns])[:, 1]
        signals["valid"][side] = build_signal_frame(label_valid, side, valid_probs)
        signals["test"][side] = build_signal_frame(label_test, side, test_probs)
        models[side] = model
        metadata[side] = {
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
            "best_iteration": int(model.get_best_iteration()),
        }
    return models, signals, metadata


def build_signal_frame(frame: pd.DataFrame, side: str, probs: np.ndarray) -> pd.DataFrame:
    out = frame[
        [
            "timestamp",
            "price_last",
            "price_first",
            "price_high",
            "price_low",
            "depth_imbalance_1pct",
            "notional_bid_1pct",
            "notional_ask_1pct",
            "long_followthrough_label",
            "short_followthrough_label",
        ]
    ].copy()
    out["side"] = side
    out["prob"] = probs
    out["label"] = out["long_followthrough_label"] if side == "long" else out["short_followthrough_label"]
    out = out.drop(columns=["long_followthrough_label", "short_followthrough_label"])
    out = out.sort_values(["timestamp", "prob"], ascending=[True, False]).reset_index(drop=True)
    return out


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(float(item.strip())) for item in value.split(",") if item.strip()]


def generate_configs(args: argparse.Namespace) -> list[ReplayConfig]:
    long_quantiles = parse_float_list(args.long_quantiles)
    short_quantiles = parse_float_list(args.short_quantiles)
    notionals = parse_float_list(args.position_notionals)
    max_concurrent_grid = parse_int_list(args.max_concurrent_grid)
    cooldown_grid = parse_int_list(args.cooldown_seconds_grid)
    latency_grid = parse_int_list(args.latency_seconds_grid)
    timeout_grid = parse_int_list(args.timeout_seconds_grid)
    depth_fraction_grid = parse_float_list(args.depth_fraction_grid)
    configs = [
        ReplayConfig(
            long_quantile=long_quantile,
            short_quantile=short_quantile,
            position_notional_usd=position_notional_usd,
            max_concurrent_positions=max_concurrent_positions,
            cooldown_seconds=cooldown_seconds,
            latency_seconds=latency_seconds,
            timeout_seconds=timeout_seconds,
            depth_fraction_limit=depth_fraction_limit,
        )
        for long_quantile, short_quantile, position_notional_usd, max_concurrent_positions, cooldown_seconds, latency_seconds, timeout_seconds, depth_fraction_limit in product(
            long_quantiles,
            short_quantiles,
            notionals,
            max_concurrent_grid,
            cooldown_grid,
            latency_grid,
            timeout_grid,
            depth_fraction_grid,
        )
    ]
    return configs


class ReplayMarket:
    def __init__(self, *, raw_root: Path, symbol: str, max_depth_snapshot_age_seconds: int) -> None:
        self.raw_root = raw_root
        self.symbol = symbol
        self.max_depth_snapshot_age_ms = max_depth_snapshot_age_seconds * 1_000
        self._trade_day_cache: dict[date, pd.DataFrame] = {}
        self._depth_day_cache: dict[date, pd.DataFrame] = {}

    def load_trade_day(self, day: date) -> pd.DataFrame:
        cached = self._trade_day_cache.get(day)
        if cached is not None:
            return cached
        file_path = self.raw_root / "aggTrades" / f"{self.symbol}-aggTrades-{day:%Y-%m-%d}.zip"
        if not file_path.exists():
            frame = pd.DataFrame(columns=["timestamp_ms", "timestamp", "price", "quantity", "quote_qty"])
            self._trade_day_cache[day] = frame
            return frame
        frame = pd.read_csv(file_path, compression="zip", usecols=["transact_time", "price", "quantity"])
        frame["timestamp"] = pd.to_datetime(frame["transact_time"], unit="ms", utc=True)
        frame["timestamp_ms"] = timestamp_to_ms(frame["timestamp"])
        frame["price"] = frame["price"].astype(float)
        frame["quantity"] = frame["quantity"].astype(float)
        frame["quote_qty"] = frame["price"] * frame["quantity"]
        out = frame[["timestamp_ms", "timestamp", "price", "quantity", "quote_qty"]].sort_values("timestamp_ms").reset_index(drop=True)
        self._trade_day_cache[day] = out
        return out

    def load_depth_day(self, day: date) -> pd.DataFrame:
        cached = self._depth_day_cache.get(day)
        if cached is not None:
            return cached
        file_path = self.raw_root / "bookDepth" / f"{self.symbol}-bookDepth-{day:%Y-%m-%d}.zip"
        if not file_path.exists():
            frame = pd.DataFrame(columns=["timestamp_ms", "timestamp"])
            self._depth_day_cache[day] = frame
            return frame
        frame = pd.read_csv(file_path, compression="zip")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame["percentage"] = frame["percentage"].astype(float)
        frame["notional"] = frame["notional"].astype(float)
        subset = frame[frame["percentage"].isin({-1.0, -0.2, 0.2, 1.0})]
        if subset.empty:
            out = pd.DataFrame(columns=["timestamp_ms", "timestamp"])
            self._depth_day_cache[day] = out
            return out
        pivot = subset.pivot_table(index="timestamp", columns="percentage", values="notional", aggfunc="last")
        out = pd.DataFrame(index=pivot.index)
        out["bid_notional_0p2"] = pivot.get(-0.2, pd.Series(index=out.index, dtype=float))
        out["ask_notional_0p2"] = pivot.get(0.2, pd.Series(index=out.index, dtype=float))
        out["bid_notional_1p0"] = pivot.get(-1.0, pd.Series(index=out.index, dtype=float))
        out["ask_notional_1p0"] = pivot.get(1.0, pd.Series(index=out.index, dtype=float))
        out = out.sort_index().reset_index()
        out["timestamp_ms"] = timestamp_to_ms(out["timestamp"])
        self._depth_day_cache[day] = out
        return out

    def first_trade_after(self, timestamp: pd.Timestamp) -> dict[str, Any] | None:
        target_ms = int(timestamp.value // 1_000_000)
        for day in (timestamp.date(), timestamp.date() + timedelta(days=1)):
            frame = self.load_trade_day(day)
            if frame.empty:
                continue
            idx = int(np.searchsorted(frame["timestamp_ms"].to_numpy(), target_ms, side="left"))
            if idx < len(frame):
                row = frame.iloc[idx]
                return {
                    "timestamp": row["timestamp"],
                    "timestamp_ms": int(row["timestamp_ms"]),
                    "price": float(row["price"]),
                }
        return None

    def trade_window(self, start_timestamp: pd.Timestamp, end_timestamp: pd.Timestamp) -> pd.DataFrame:
        start_ms = int(start_timestamp.value // 1_000_000)
        end_ms = int(end_timestamp.value // 1_000_000)
        frames: list[pd.DataFrame] = []
        cursor = start_timestamp.date()
        while cursor <= end_timestamp.date():
            frame = self.load_trade_day(cursor)
            if not frame.empty:
                left = int(np.searchsorted(frame["timestamp_ms"].to_numpy(), start_ms, side="left"))
                right = int(np.searchsorted(frame["timestamp_ms"].to_numpy(), end_ms, side="right"))
                if right > left:
                    frames.append(frame.iloc[left:right])
            cursor += timedelta(days=1)
        if not frames:
            return pd.DataFrame(columns=["timestamp_ms", "timestamp", "price", "quantity", "quote_qty"])
        return pd.concat(frames, ignore_index=True)

    def depth_snapshot(self, timestamp: pd.Timestamp) -> dict[str, Any] | None:
        target_ms = int(timestamp.value // 1_000_000)
        candidates: list[pd.DataFrame] = []
        for day in (timestamp.date() - timedelta(days=1), timestamp.date()):
            frame = self.load_depth_day(day)
            if not frame.empty:
                candidates.append(frame)
        if not candidates:
            return None
        combined = pd.concat(candidates, ignore_index=True).sort_values("timestamp_ms").reset_index(drop=True)
        idx = int(np.searchsorted(combined["timestamp_ms"].to_numpy(), target_ms, side="right")) - 1
        if idx < 0:
            return None
        row = combined.iloc[idx]
        age_ms = target_ms - int(row["timestamp_ms"])
        if age_ms < 0 or age_ms > self.max_depth_snapshot_age_ms:
            return None
        return {
            "timestamp": row["timestamp"],
            "age_seconds": age_ms / 1_000.0,
            "bid_notional_0p2": float(row.get("bid_notional_0p2", np.nan)),
            "ask_notional_0p2": float(row.get("ask_notional_0p2", np.nan)),
            "bid_notional_1p0": float(row.get("bid_notional_1p0", np.nan)),
            "ask_notional_1p0": float(row.get("ask_notional_1p0", np.nan)),
        }


def replay_signals(
    *,
    valid_long: pd.DataFrame,
    valid_short: pd.DataFrame,
    market: ReplayMarket,
    config: ReplayConfig,
    long_threshold: float,
    short_threshold: float,
    profit_bps: float,
    stop_bps: float,
    fee_bps_per_side: float,
    min_fill_notional_usd: float,
) -> dict[str, Any]:
    long_selected = valid_long[valid_long["prob"] >= long_threshold].copy()
    short_selected = valid_short[valid_short["prob"] >= short_threshold].copy()
    long_selected["threshold"] = long_threshold
    short_selected["threshold"] = short_threshold
    signals = pd.concat([long_selected, short_selected], ignore_index=True)
    if signals.empty:
        return {"trades": [], "skip_reasons": {"below_threshold": int(len(valid_long) + len(valid_short))}}
    signals = signals.sort_values(["timestamp", "prob"], ascending=[True, False]).reset_index(drop=True)

    trades: list[ReplayTrade] = []
    skip_reasons: dict[str, int] = {}
    open_exit_times: list[pd.Timestamp] = []
    next_allowed = {"long": pd.Timestamp.min.tz_localize(UTC), "short": pd.Timestamp.min.tz_localize(UTC)}

    for signal in signals.itertuples(index=False):
        signal_ts = pd.Timestamp(signal.timestamp)
        open_exit_times = [value for value in open_exit_times if value > signal_ts]
        side = str(signal.side)
        if signal_ts < next_allowed[side]:
            bump_reason(skip_reasons, "cooldown")
            continue
        if len(open_exit_times) >= config.max_concurrent_positions:
            bump_reason(skip_reasons, "max_concurrent")
            continue
        trade = simulate_trade(
            signal=signal,
            market=market,
            config=config,
            profit_bps=profit_bps,
            stop_bps=stop_bps,
            fee_bps_per_side=fee_bps_per_side,
            min_fill_notional_usd=min_fill_notional_usd,
        )
        if trade is None:
            continue
        if isinstance(trade, str):
            bump_reason(skip_reasons, trade)
            continue
        trades.append(trade)
        open_exit_times.append(pd.Timestamp(trade.exit_timestamp))
        next_allowed[side] = signal_ts + pd.Timedelta(seconds=config.cooldown_seconds)

    return {"trades": trades, "skip_reasons": skip_reasons}


def simulate_trade(
    *,
    signal: Any,
    market: ReplayMarket,
    config: ReplayConfig,
    profit_bps: float,
    stop_bps: float,
    fee_bps_per_side: float,
    min_fill_notional_usd: float,
) -> ReplayTrade | str | None:
    signal_ts = pd.Timestamp(signal.timestamp)
    entry_target_ts = signal_ts + pd.Timedelta(seconds=config.latency_seconds)
    entry_trade = market.first_trade_after(entry_target_ts)
    if entry_trade is None:
        return "no_entry_trade"

    side = str(signal.side)
    entry_depth = market.depth_snapshot(pd.Timestamp(entry_trade["timestamp"]))
    if entry_depth is None:
        return "no_entry_depth"

    same_side_key = "ask" if side == "long" else "bid"
    depth_0p2 = max(0.0, float(entry_depth[f"{same_side_key}_notional_0p2"]))
    depth_1p0 = max(depth_0p2, float(entry_depth[f"{same_side_key}_notional_1p0"]))
    if depth_0p2 <= 0.0 or depth_1p0 <= 0.0:
        return "no_same_side_depth"

    fill_cap_usd = min(config.position_notional_usd, config.depth_fraction_limit * depth_0p2)
    if fill_cap_usd < min_fill_notional_usd:
        return "insufficient_depth"

    entry_impact_bps = impact_bps(fill_cap_usd, depth_0p2, depth_1p0)
    entry_price_raw = float(entry_trade["price"])
    entry_price_filled = apply_impact(entry_price_raw, side=side, impact_bps=entry_impact_bps, is_entry=True)
    quantity = fill_cap_usd / entry_price_filled

    deadline = pd.Timestamp(entry_trade["timestamp"]) + pd.Timedelta(seconds=config.timeout_seconds)
    path = market.trade_window(pd.Timestamp(entry_trade["timestamp"]) + pd.Timedelta(milliseconds=1), deadline)
    if path.empty:
        return "no_exit_trade"

    take_level = entry_price_filled * (1.0 + profit_bps / 10000.0) if side == "long" else entry_price_filled * (1.0 - profit_bps / 10000.0)
    stop_level = entry_price_filled * (1.0 - stop_bps / 10000.0) if side == "long" else entry_price_filled * (1.0 + stop_bps / 10000.0)

    prices = path["price"].to_numpy(dtype=float)
    timestamps = path["timestamp"].to_numpy()
    if side == "long":
        take_hits = np.where(prices >= take_level)[0]
        stop_hits = np.where(prices <= stop_level)[0]
    else:
        take_hits = np.where(prices <= take_level)[0]
        stop_hits = np.where(prices >= stop_level)[0]

    hit_index: int | None = None
    exit_reason = "timeout"
    if len(take_hits) and len(stop_hits):
        hit_index = int(min(take_hits[0], stop_hits[0]))
        exit_reason = "take_profit" if take_hits[0] < stop_hits[0] else "stop_loss"
    elif len(take_hits):
        hit_index = int(take_hits[0])
        exit_reason = "take_profit"
    elif len(stop_hits):
        hit_index = int(stop_hits[0])
        exit_reason = "stop_loss"

    if hit_index is None:
        hit_index = len(path) - 1
    exit_timestamp = pd.Timestamp(timestamps[hit_index])
    exit_price_raw = float(prices[hit_index])
    exit_depth = market.depth_snapshot(exit_timestamp)
    if exit_depth is None:
        return "no_exit_depth"

    opposite_key = "bid" if side == "long" else "ask"
    exit_depth_0p2 = max(0.0, float(exit_depth[f"{opposite_key}_notional_0p2"]))
    exit_depth_1p0 = max(exit_depth_0p2, float(exit_depth[f"{opposite_key}_notional_1p0"]))
    if exit_depth_0p2 <= 0.0 or exit_depth_1p0 <= 0.0:
        return "no_exit_side_depth"

    exit_impact_bps = impact_bps(fill_cap_usd, exit_depth_0p2, exit_depth_1p0)
    exit_price_filled = apply_impact(exit_price_raw, side=side, impact_bps=exit_impact_bps, is_entry=False)

    if side == "long":
        gross_pnl_usd = quantity * (exit_price_filled - entry_price_filled)
    else:
        gross_pnl_usd = quantity * (entry_price_filled - exit_price_filled)
    exit_notional_usd = quantity * exit_price_filled
    fees_usd = fill_cap_usd * fee_bps_per_side / 10000.0 + exit_notional_usd * fee_bps_per_side / 10000.0
    net_pnl_usd = gross_pnl_usd - fees_usd

    return ReplayTrade(
        side=side,
        signal_timestamp=signal_ts.isoformat(),
        entry_timestamp=pd.Timestamp(entry_trade["timestamp"]).isoformat(),
        exit_timestamp=exit_timestamp.isoformat(),
        signal_prob=float(signal.prob),
        threshold=float(signal.threshold),
        requested_notional_usd=float(config.position_notional_usd),
        filled_notional_usd=float(fill_cap_usd),
        quantity=float(quantity),
        entry_price_raw=entry_price_raw,
        entry_price_filled=float(entry_price_filled),
        exit_price_raw=exit_price_raw,
        exit_price_filled=float(exit_price_filled),
        entry_impact_bps=float(entry_impact_bps),
        exit_impact_bps=float(exit_impact_bps),
        gross_pnl_usd=float(gross_pnl_usd),
        fees_usd=float(fees_usd),
        net_pnl_usd=float(net_pnl_usd),
        hold_seconds=float((exit_timestamp - pd.Timestamp(entry_trade["timestamp"])).total_seconds()),
        exit_reason=exit_reason,
        depth_snapshot_age_seconds_entry=float(entry_depth["age_seconds"]),
        depth_snapshot_age_seconds_exit=float(exit_depth["age_seconds"]),
    )


def impact_bps(notional_usd: float, depth_0p2_usd: float, depth_1p0_usd: float) -> float:
    """
    Approximate impact from observed cumulative depth bands.

    Up to the 0.20% book, average impact rises linearly from 0 to 10 bps.
    Between 0.20% and 1.00%, it rises from 10 bps to 50 bps.
    """

    if depth_0p2_usd <= 0.0 or depth_1p0_usd <= 0.0:
        return 50.0
    if notional_usd <= depth_0p2_usd:
        share = max(0.0, min(1.0, notional_usd / depth_0p2_usd))
        return 10.0 * share
    if depth_1p0_usd <= depth_0p2_usd:
        overflow_share = max(0.0, (notional_usd - depth_0p2_usd) / max(depth_0p2_usd, 1.0))
        return min(50.0, 10.0 + 40.0 * overflow_share)
    overflow_share = max(0.0, min(1.0, (notional_usd - depth_0p2_usd) / (depth_1p0_usd - depth_0p2_usd)))
    return 10.0 + 40.0 * overflow_share


def apply_impact(price: float, *, side: str, impact_bps: float, is_entry: bool) -> float:
    impact = impact_bps / 10000.0
    if side == "long":
        return price * (1.0 + impact if is_entry else 1.0 - impact)
    return price * (1.0 - impact if is_entry else 1.0 + impact)


def timestamp_to_ms(series: pd.Series) -> pd.Series:
    naive_utc = series.dt.tz_convert("UTC").dt.tz_localize(None)
    return (naive_utc.astype("datetime64[ns]").astype("int64") // 1_000_000).astype(np.int64)


def bump_reason(counter: dict[str, int], reason: str) -> None:
    counter[reason] = counter.get(reason, 0) + 1


def validation_key(summary: dict[str, Any]) -> tuple[float, float, float]:
    profit_factor = float(summary.get("profit_factor", 0.0))
    return (
        float(summary.get("net_pnl_usd", -1e18)),
        -float(summary.get("max_drawdown_usd", 1e18)),
        profit_factor,
    )


def summarise_replay(result: dict[str, Any]) -> dict[str, Any]:
    trades: list[ReplayTrade] = result["trades"]
    if not trades:
        return {
            "trades": 0,
            "wins": 0,
            "win_rate": 0.0,
            "gross_pnl_usd": 0.0,
            "fees_usd": 0.0,
            "net_pnl_usd": 0.0,
            "avg_hold_seconds": 0.0,
            "avg_filled_notional_usd": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_usd": 0.0,
            "skip_reasons": result.get("skip_reasons", {}),
            "exit_reasons": {},
        }

    pnl = np.array([trade.net_pnl_usd for trade in trades], dtype=float)
    gross = np.array([trade.gross_pnl_usd for trade in trades], dtype=float)
    fees = np.array([trade.fees_usd for trade in trades], dtype=float)
    holds = np.array([trade.hold_seconds for trade in trades], dtype=float)
    fills = np.array([trade.filled_notional_usd for trade in trades], dtype=float)
    cumulative = np.cumsum(pnl)
    running_peak = np.maximum.accumulate(np.maximum(cumulative, 0.0))
    drawdowns = running_peak - cumulative
    positive = pnl[pnl > 0.0].sum()
    negative = -pnl[pnl < 0.0].sum()
    exit_reasons: dict[str, int] = {}
    for trade in trades:
        exit_reasons[trade.exit_reason] = exit_reasons.get(trade.exit_reason, 0) + 1
    return {
        "trades": int(len(trades)),
        "wins": int(np.sum(pnl > 0.0)),
        "win_rate": float(np.mean(pnl > 0.0)),
        "gross_pnl_usd": float(gross.sum()),
        "fees_usd": float(fees.sum()),
        "net_pnl_usd": float(pnl.sum()),
        "avg_hold_seconds": float(holds.mean()),
        "avg_filled_notional_usd": float(fills.mean()),
        "profit_factor": float(positive / negative) if negative > 0.0 else float("inf"),
        "max_drawdown_usd": float(drawdowns.max()) if len(drawdowns) else 0.0,
        "skip_reasons": result.get("skip_reasons", {}),
        "exit_reasons": exit_reasons,
    }


def render_markdown(report: dict[str, Any]) -> str:
    chosen = report["chosen_setup"]
    validation = chosen["validation"]
    test = chosen["test"]
    lines = [
        "# Futures Execution Replay",
        "",
        "## Setup",
        "",
        f"- Symbol: `{report['symbol']}`",
        f"- Date range: `{report['date_range']['start']}` to `{report['date_range']['end']}`",
        f"- Profit / stop: `{report['profit_bps']}` / `{report['stop_bps']}` bps",
        f"- Fee per side: `{report['fee_bps_per_side']}` bps",
        f"- Source threshold: `{report['source_threshold']}`",
        "",
        "## Chosen Validation Setup",
        "",
        f"- Config: `{json.dumps(chosen['config'], sort_keys=True)}`",
        f"- Long threshold: `{chosen['long_threshold']:.6f}`",
        f"- Short threshold: `{chosen['short_threshold']:.6f}`",
        f"- Validation trades: `{validation['trades']}`",
        f"- Validation net PnL: `{validation['net_pnl_usd']:.2f}` USD",
        f"- Validation max drawdown: `{validation['max_drawdown_usd']:.2f}` USD",
        f"- Validation win rate: `{validation['win_rate']:.3f}`",
        "",
        "## Held-Out Test",
        "",
        f"- Trades: `{test['trades']}`",
        f"- Net PnL: `{test['net_pnl_usd']:.2f}` USD",
        f"- Gross PnL: `{test['gross_pnl_usd']:.2f}` USD",
        f"- Fees: `{test['fees_usd']:.2f}` USD",
        f"- Win rate: `{test['win_rate']:.3f}`",
        f"- Profit factor: `{test['profit_factor']}`",
        f"- Max drawdown: `{test['max_drawdown_usd']:.2f}` USD",
        "",
        "## Leaderboard",
        "",
    ]
    for idx, row in enumerate(report["leaderboard"], start=1):
        summary = row["validation"]
        lines.append(
            f"{idx}. Net `{summary['net_pnl_usd']:.2f}` USD | DD `{summary['max_drawdown_usd']:.2f}` | Trades `{summary['trades']}` | Config `{json.dumps(row['config'], sort_keys=True)}`"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
