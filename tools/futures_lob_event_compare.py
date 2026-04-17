#!/usr/bin/env python3
"""
Compare DeepLOB-style vs XGBoost on captured BTC futures LOB data.

Targets:
- next N events mid-price move (regression)
- next T seconds mid-price move (regression, default 10s)

Evaluation:
- directional AUC on the held-out split
- quote-based post-cost PnL, with thresholds chosen on validation and applied to test

This is the proper comparison layer for the live-captured depth20/bookTicker path.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, roc_auc_score
from torch import nn
from xgboost import XGBRegressor


DEFAULT_CAPTURE_ROOT = Path("output/futures_ml_live_deeplob")
DEFAULT_OUTPUT_ROOT = Path("output/futures_lob_compare")
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_LEVELS = 20
DEFAULT_SEQUENCE_LENGTH = 50
DEFAULT_EVENT_HORIZON = 50
DEFAULT_TIME_HORIZON_SECONDS = 10.0
DEFAULT_LONGER_EVENT_HORIZONS = (20, 50, 100)
DEFAULT_LONGER_TIME_HORIZONS_SECONDS = (10.0, 30.0, 60.0)
DEFAULT_BATCH_SIZE = 256
DEFAULT_EPOCHS = 12
DEFAULT_PATIENCE = 3
DEFAULT_CONV_CHANNELS = 32
DEFAULT_LSTM_HIDDEN = 48
DEFAULT_DROPOUT = 0.20
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_FEE_BPS_PER_SIDE = 1.0
DEFAULT_MIN_VALIDATION_TRADES = 20
DEFAULT_XGB_ESTIMATORS = 120
DEFAULT_XGB_MAX_DEPTH = 5
DEFAULT_XGB_N_JOBS = 4
THRESHOLD_GRID = (0.80, 0.90, 0.95, 0.98)
TRADE_WINDOWS_MS = (250, 1000, 5000)
QUOTE_WINDOWS_MS = (250, 1000, 5000)
DEFAULT_ENTRY_TIMEOUT_SECONDS = 2.0
DEFAULT_EXIT_TIMEOUT_SECONDS = 2.0
DEFAULT_ORDER_NOTIONAL_USD = 1000.0
DEFAULT_QUEUE_AHEAD_MULTIPLIER = 1.0
DEFAULT_MAX_QUOTE_AGE_MS = 1000
UTC = timezone.utc
RANDOM_SEED = 42


@dataclass
class RegressionMetrics:
    mae_bps: float
    directional_auc: float
    directional_base_rate: float


@dataclass
class TradeMetrics:
    signals_considered: int
    trades: int
    fill_rate: float
    win_rate: float
    gross_pnl_bps: float
    net_pnl_bps: float
    avg_net_pnl_bps: float
    avg_hold_seconds: float
    profit_factor: float
    passive_exit_rate: float
    forced_taker_exit_rate: float
    chosen_quantile: float | None
    threshold_abs_bps: float | None


@dataclass
class TargetSpec:
    report_key: str
    target_column: str
    future_spread_column: str
    hold_column: str


@dataclass
class ReplayConfig:
    maker_fee_bps_per_side: float
    taker_fee_bps_per_side: float
    order_notional_usd: float
    entry_timeout_seconds: float
    exit_timeout_seconds: float
    queue_ahead_multiplier: float
    max_quote_age_ms: int


@dataclass
class DeepTrainingSummary:
    best_epoch: int
    best_valid_mae_bps: float
    train_rows: int
    valid_rows: int
    test_rows: int


class DeepLOBRegressor(nn.Module):
    def __init__(self, input_features: int, conv_channels: int, lstm_hidden: int, dropout: float) -> None:
        super().__init__()
        kernel = 3 if input_features >= 3 else 1
        pad = kernel // 2
        branch_channels = conv_channels
        merged_channels = branch_channels * 3
        self.stem = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=(1, kernel), padding=(0, pad), bias=False),
            nn.BatchNorm2d(conv_channels),
            nn.LeakyReLU(0.01),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(conv_channels),
            nn.LeakyReLU(0.01),
        )
        self.branch_short = nn.Sequential(
            nn.Conv2d(conv_channels, branch_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.LeakyReLU(0.01),
        )
        self.branch_medium = nn.Sequential(
            nn.Conv2d(conv_channels, branch_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.LeakyReLU(0.01),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.LeakyReLU(0.01),
        )
        self.branch_long = nn.Sequential(
            nn.Conv2d(conv_channels, branch_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.LeakyReLU(0.01),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=(5, 1), padding=(2, 0), bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.LeakyReLU(0.01),
        )
        self.dropout = nn.Dropout(dropout)
        self.sequence_model = nn.LSTM(
            input_size=merged_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.stem(x)
        short = self.branch_short(x)
        medium = self.branch_medium(x)
        long = self.branch_long(x)
        x = torch.cat((short, medium, long), dim=1)
        x = x.mean(dim=-1)
        x = x.transpose(1, 2)
        seq_out, _ = self.sequence_model(self.dropout(x))
        return self.head(seq_out[:, -1, :]).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare DeepLOB-style and XGBoost on captured BTC futures LOB data.")
    parser.add_argument("--capture-root", default=str(DEFAULT_CAPTURE_ROOT), help="Root containing symbol/date capture folders")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output root for reports and models")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Symbol folder to read")
    parser.add_argument("--date", help="Single capture date folder YYYY-MM-DD")
    parser.add_argument("--dates", help="Comma-separated capture dates YYYY-MM-DD,YYYY-MM-DD")
    parser.add_argument("--all-dates", action="store_true", help="Consume all available capture date folders for the symbol")
    parser.add_argument("--levels", type=int, default=DEFAULT_LEVELS, help="Number of depth levels per side to keep")
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH, help="History length in depth events")
    parser.add_argument("--event-horizon", type=int, default=DEFAULT_EVENT_HORIZON, help="Future event count for the event-based target")
    parser.add_argument(
        "--event-horizons",
        default=",".join(str(value) for value in DEFAULT_LONGER_EVENT_HORIZONS),
        help="Comma-separated future event counts to score, e.g. 20,50,100",
    )
    parser.add_argument("--time-horizon-seconds", type=float, default=DEFAULT_TIME_HORIZON_SECONDS, help="Future wall-clock horizon in seconds")
    parser.add_argument(
        "--time-horizons-seconds",
        default=",".join(str(int(value)) for value in DEFAULT_LONGER_TIME_HORIZONS_SECONDS),
        help="Comma-separated future wall-clock horizons in seconds, e.g. 10,30,60",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Deep model batch size")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Maximum deep model epochs")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Deep model early-stop patience")
    parser.add_argument("--conv-channels", type=int, default=DEFAULT_CONV_CHANNELS, help="DeepLOB-style conv channel count")
    parser.add_argument("--lstm-hidden", type=int, default=DEFAULT_LSTM_HIDDEN, help="DeepLOB-style LSTM hidden size")
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT, help="DeepLOB-style dropout")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR, help="AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="AdamW weight decay")
    parser.add_argument("--fee-bps-per-side", type=float, default=DEFAULT_FEE_BPS_PER_SIDE, help="Passive maker fee charged per side in bps")
    parser.add_argument("--taker-fee-bps-per-side", type=float, help="Forced taker-exit fee in bps; defaults to fee-bps-per-side")
    parser.add_argument("--order-notional-usd", type=float, default=DEFAULT_ORDER_NOTIONAL_USD, help="Per-signal order notional used in queue-aware replay")
    parser.add_argument("--entry-timeout-seconds", type=float, default=DEFAULT_ENTRY_TIMEOUT_SECONDS, help="Maximum passive entry wait")
    parser.add_argument("--exit-timeout-seconds", type=float, default=DEFAULT_EXIT_TIMEOUT_SECONDS, help="Maximum passive exit wait before forcing taker exit")
    parser.add_argument("--queue-ahead-multiplier", type=float, default=DEFAULT_QUEUE_AHEAD_MULTIPLIER, help="Multiplier on visible top-of-book queue ahead of our passive order")
    parser.add_argument("--max-quote-age-ms", type=int, default=DEFAULT_MAX_QUOTE_AGE_MS, help="Maximum age of the nearest quote snapshot used for replay")
    parser.add_argument("--min-validation-trades", type=int, default=DEFAULT_MIN_VALIDATION_TRADES, help="Minimum validation trades before a quantile can win")
    parser.add_argument("--xgb-estimators", type=int, default=DEFAULT_XGB_ESTIMATORS, help="XGBoost tree count")
    parser.add_argument("--xgb-max-depth", type=int, default=DEFAULT_XGB_MAX_DEPTH, help="XGBoost max depth")
    parser.add_argument("--xgb-n-jobs", type=int, default=DEFAULT_XGB_N_JOBS, help="XGBoost worker threads")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"), help="Torch device selection")
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not parsed:
        raise SystemExit("Expected at least one integer horizon")
    return parsed


def parse_float_list(value: str) -> list[float]:
    parsed = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not parsed:
        raise SystemExit("Expected at least one float horizon")
    return parsed


def horizon_tag(event_horizons: list[int], time_horizons: list[float]) -> str:
    event_tag = "-".join(f"{value:03d}" for value in event_horizons)
    time_tag = "-".join(f"{int(round(value)):02d}" for value in time_horizons)
    return f"ev{event_tag}_t{time_tag}s"


def build_target_specs(event_horizons: list[int], time_horizons: list[float]) -> list[TargetSpec]:
    specs: list[TargetSpec] = []
    for horizon in event_horizons:
        specs.append(
            TargetSpec(
                report_key=f"event_{horizon:03d}",
                target_column=f"target_event_{horizon:03d}_bps",
                future_spread_column=f"future_spread_event_{horizon:03d}_bps",
                hold_column=f"hold_event_{horizon:03d}_seconds",
            )
        )
    for horizon in time_horizons:
        seconds_tag = int(round(horizon))
        specs.append(
            TargetSpec(
                report_key=f"time_{seconds_tag:02d}s",
                target_column=f"target_time_{seconds_tag:02d}s_bps",
                future_spread_column=f"future_spread_time_{seconds_tag:02d}s_bps",
                hold_column=f"hold_time_{seconds_tag:02d}s_seconds",
            )
        )
    return specs


def main() -> None:
    args = parse_args()
    set_random_seed(RANDOM_SEED)
    event_horizons = sorted(set(parse_int_list(args.event_horizons) + [args.event_horizon]))
    time_horizons = sorted(set(parse_float_list(args.time_horizons_seconds) + [args.time_horizon_seconds]))
    target_specs = build_target_specs(event_horizons, time_horizons)
    taker_fee_bps_per_side = args.taker_fee_bps_per_side if args.taker_fee_bps_per_side is not None else args.fee_bps_per_side
    replay_config = ReplayConfig(
        maker_fee_bps_per_side=args.fee_bps_per_side,
        taker_fee_bps_per_side=taker_fee_bps_per_side,
        order_notional_usd=args.order_notional_usd,
        entry_timeout_seconds=args.entry_timeout_seconds,
        exit_timeout_seconds=args.exit_timeout_seconds,
        queue_ahead_multiplier=args.queue_ahead_multiplier,
        max_quote_age_ms=args.max_quote_age_ms,
    )

    symbol = args.symbol.upper()
    capture_dirs = resolve_capture_dirs(
        capture_root=Path(args.capture_root).resolve(),
        symbol=symbol,
        date=args.date,
        dates_csv=args.dates,
        all_dates=args.all_dates,
    )
    depth_paths = [capture_dir / "depth.jsonl" for capture_dir in capture_dirs]
    agg_trade_paths = [capture_dir / "aggTrade.jsonl" for capture_dir in capture_dirs]
    missing = [path for path in depth_paths + agg_trade_paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing capture file(s): {missing}")
    date_label = capture_date_label(capture_dirs)

    run_name = (
        f"lob_compare_{args.symbol.lower()}_{date_label}_"
        f"lvl{args.levels:02d}_seq{args.sequence_length:03d}_"
        f"{horizon_tag(event_horizons, time_horizons)}_v2"
    )
    run_root = Path(args.output_root).resolve() / run_name
    dataset_root = run_root / "dataset"
    model_root = run_root / "models"
    report_root = run_root / "reports"
    for path in (dataset_root, model_root, report_root):
        path.mkdir(parents=True, exist_ok=True)

    print(f"[lob] building dataset from {len(depth_paths)} depth file(s) and {len(agg_trade_paths)} aggTrade file(s)", flush=True)
    dataset, trade_frame = build_event_dataset(
        capture_dirs=capture_dirs,
        levels=args.levels,
        event_horizons=event_horizons,
        time_horizons_seconds=time_horizons,
    )
    dataset_path = dataset_root / "lob_event_dataset.csv.gz"
    dataset.to_csv(dataset_path, index=False, compression="gzip")
    print(f"[lob] dataset rows={len(dataset):,} saved={dataset_path}", flush=True)

    feature_columns = resolve_feature_columns(dataset)
    split_idx = split_dataset(dataset)
    scaler = build_scaler(dataset.iloc[: split_idx["train_end"]], feature_columns)
    device = resolve_device(args.device)
    print(f"[lob] feature_count={len(feature_columns)} device={device.type}", flush=True)

    report: dict[str, Any] = {
        "bundle_version": run_name,
        "created_at": datetime.now(UTC).isoformat(),
        "capture_dirs": [str(path) for path in capture_dirs],
        "capture_dates": [path.name for path in capture_dirs],
        "dataset_path": str(dataset_path),
        "rows": int(len(dataset)),
        "feature_count": int(len(feature_columns)),
        "levels": args.levels,
        "sequence_length": args.sequence_length,
        "event_horizons": event_horizons,
        "time_horizons_seconds": time_horizons,
        "maker_fee_bps_per_side": args.fee_bps_per_side,
        "taker_fee_bps_per_side": taker_fee_bps_per_side,
        "entry_timeout_seconds": args.entry_timeout_seconds,
        "exit_timeout_seconds": args.exit_timeout_seconds,
        "order_notional_usd": args.order_notional_usd,
        "queue_ahead_multiplier": args.queue_ahead_multiplier,
        "time_split": {
            "train_rows": split_idx["train_end"],
            "valid_rows": split_idx["valid_end"] - split_idx["train_end"],
            "test_rows": len(dataset) - split_idx["valid_end"],
        },
        "targets": {},
    }

    for target_spec in target_specs:
        print(f"[lob] preparing samples for {target_spec.report_key}", flush=True)
        sequences, flat_features, labels, sample_idx = build_samples(
            dataset=dataset,
            feature_columns=feature_columns,
            target_spec=target_spec,
            sequence_length=args.sequence_length,
            scaler=scaler,
        )
        print(f"[lob] {target_spec.report_key} samples={len(sequences):,}", flush=True)
        seq_split = split_samples(len(sequences))
        if min(seq_split["train_end"], seq_split["valid_end"] - seq_split["train_end"], len(sequences) - seq_split["valid_end"]) <= 0:
            raise SystemExit(f"Not enough sequence samples for {target_spec.report_key}")

        X_train, X_valid, X_test = split_array(flat_features, seq_split)
        S_train, S_valid, S_test = split_array(sequences, seq_split)
        y_train, y_valid, y_test = split_array(labels, seq_split)
        sample_idx_train, sample_idx_valid, sample_idx_test = split_array(sample_idx, seq_split)

        print(f"[lob] fitting xgboost for {target_spec.report_key}", flush=True)
        xgb = build_xgboost_regressor(
            n_estimators=args.xgb_estimators,
            max_depth=args.xgb_max_depth,
            n_jobs=args.xgb_n_jobs,
        )
        xgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        xgb_preds = {
            "train": xgb.predict(X_train),
            "valid": xgb.predict(X_valid),
            "test": xgb.predict(X_test),
        }
        print(f"[lob] xgboost complete for {target_spec.report_key}", flush=True)

        deep_model = DeepLOBRegressor(
            input_features=sequences.shape[-1],
            conv_channels=args.conv_channels,
            lstm_hidden=args.lstm_hidden,
            dropout=args.dropout,
        ).to(device)
        print(f"[lob] fitting deeplob-style for {target_spec.report_key}", flush=True)
        deep_summary = train_deep_regressor(
            model=deep_model,
            train_sequences=S_train,
            train_labels=y_train,
            valid_sequences=S_valid,
            valid_labels=y_valid,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=device,
        )
        print(f"[lob] deeplob-style complete for {target_spec.report_key}", flush=True)
        torch.save(deep_model.state_dict(), model_root / f"{target_spec.report_key}_deeplob.pt")
        deep_preds = {
            "train": predict_deep(deep_model, S_train, device=device, batch_size=args.batch_size),
            "valid": predict_deep(deep_model, S_valid, device=device, batch_size=args.batch_size),
            "test": predict_deep(deep_model, S_test, device=device, batch_size=args.batch_size),
        }

        target_report: dict[str, Any] = {"models": {}}
        for model_name, preds in (("xgboost", xgb_preds), ("deeplob_style", deep_preds)):
            validation_trade = choose_trade_quantile(
                predictions=preds["valid"],
                labels=y_valid,
                sample_indices=sample_idx_valid,
                dataset=dataset,
                trade_frame=trade_frame,
                replay_config=replay_config,
                min_validation_trades=args.min_validation_trades,
            )
            test_trade = evaluate_trade_policy(
                predictions=preds["test"],
                labels=y_test,
                sample_indices=sample_idx_test,
                dataset=dataset,
                trade_frame=trade_frame,
                replay_config=replay_config,
                quantile=validation_trade.chosen_quantile,
                threshold_abs_bps=validation_trade.threshold_abs_bps,
            )
            target_report["models"][model_name] = {
                "train_metrics": asdict(evaluate_regression(y_train, preds["train"])),
                "valid_metrics": asdict(evaluate_regression(y_valid, preds["valid"])),
                "test_metrics": asdict(evaluate_regression(y_test, preds["test"])),
                "validation_trade": asdict(validation_trade),
                "test_trade": asdict(test_trade),
                "training": asdict(deep_summary) if model_name == "deeplob_style" else None,
            }
        report["targets"][target_spec.report_key] = target_report

    report_path_json = report_root / "comparison_report.json"
    report_path_md = report_root / "comparison_report.md"
    report_path_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_path_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"LOB compare report: {report_path_md}")


def build_event_dataset(
    *,
    capture_dirs: list[Path],
    levels: int,
    event_horizons: list[int],
    time_horizons_seconds: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for capture_dir in capture_dirs:
        depth_path = capture_dir / "depth.jsonl"
        with depth_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                record = json.loads(raw_line)
                event = record.get("data") or {}
                bids = event.get("b")
                asks = event.get("a")
                if not bids or not asks:
                    continue
                built = build_depth_row(event=event, bids=bids, asks=asks, levels=levels)
                if built is not None:
                    rows.append(built)
    if not rows:
        raise SystemExit("No usable depth events in capture")

    frame = pd.DataFrame(rows).sort_values("timestamp_ms").drop_duplicates("timestamp_ms", keep="last").reset_index(drop=True)
    frame = add_quote_shape_features(frame)
    frame = add_depth_ofi_features(frame)
    trade_frame = load_trade_frame(capture_dirs)
    frame = add_trade_flow_features(frame, trade_frame)

    for horizon in event_horizons:
        frame[f"target_event_{horizon:03d}_bps"] = compute_event_horizon_target(frame["mid_price"].to_numpy(dtype=float), horizon)
        frame[f"future_spread_event_{horizon:03d}_bps"] = shift_forward(frame["spread_bps"].to_numpy(dtype=float), horizon)
        hold_col = f"hold_event_{horizon:03d}_seconds"
        frame[hold_col] = np.nan
        if horizon < len(frame):
            frame.loc[: len(frame) - horizon - 1, hold_col] = (frame["timestamp_ms"].shift(-horizon) - frame["timestamp_ms"]) / 1000.0

    for horizon_seconds in time_horizons_seconds:
        seconds_tag = int(round(horizon_seconds))
        target, future_spread, hold_time_seconds = compute_time_horizon_targets(frame, horizon_seconds)
        frame[f"target_time_{seconds_tag:02d}s_bps"] = target
        frame[f"future_spread_time_{seconds_tag:02d}s_bps"] = future_spread
        frame[f"hold_time_{seconds_tag:02d}s_seconds"] = hold_time_seconds

    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=resolve_feature_columns(frame)).reset_index(drop=True)
    return frame, trade_frame


def load_trade_frame(capture_dirs: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for capture_dir in capture_dirs:
        trade_path = capture_dir / "aggTrade.jsonl"
        with trade_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                record = json.loads(raw_line)
                event = record.get("data") or {}
                timestamp_ms = int(event.get("T") or event.get("E") or 0)
                price = float(event.get("p") or 0.0)
                qty = float(event.get("q") or 0.0)
                if timestamp_ms <= 0 or price <= 0.0 or qty <= 0.0:
                    continue
                aggressor = -1.0 if bool(event.get("m")) else 1.0
                rows.append(
                    {
                        "timestamp_ms": timestamp_ms,
                        "price": price,
                        "qty": qty,
                        "signed_qty": qty * aggressor,
                        "notional": qty * price,
                        "signed_notional": qty * price * aggressor,
                        "buy_qty": qty if aggressor > 0 else 0.0,
                        "sell_qty": qty if aggressor < 0 else 0.0,
                        "buy_count": 1.0 if aggressor > 0 else 0.0,
                        "sell_count": 1.0 if aggressor < 0 else 0.0,
                        "is_buy_aggressor": aggressor > 0,
                    }
                )
    if not rows:
        raise SystemExit("No usable aggTrade events in capture")
    return pd.DataFrame(rows).sort_values("timestamp_ms").reset_index(drop=True)


def resolve_feature_columns(dataset: pd.DataFrame) -> list[str]:
    return sorted(
        column
        for column in dataset.columns
        if column.startswith(("bid_price_bps_", "bid_log_size_", "ask_price_bps_", "ask_log_size_", "feat_"))
        or column == "spread_bps"
    )


def add_quote_shape_features(frame: pd.DataFrame) -> pd.DataFrame:
    bid_qty = frame["best_bid_qty"].to_numpy(dtype=float)
    ask_qty = frame["best_ask_qty"].to_numpy(dtype=float)
    top5_bid = frame[[f"bid_size_{idx}" for idx in range(1, 6)]].sum(axis=1).to_numpy(dtype=float)
    top5_ask = frame[[f"ask_size_{idx}" for idx in range(1, 6)]].sum(axis=1).to_numpy(dtype=float)
    top10_bid = frame[[f"bid_size_{idx}" for idx in range(1, 11)]].sum(axis=1).to_numpy(dtype=float)
    top10_ask = frame[[f"ask_size_{idx}" for idx in range(1, 11)]].sum(axis=1).to_numpy(dtype=float)
    denom_top1 = np.maximum(bid_qty + ask_qty, 1e-9)
    denom_top5 = np.maximum(top5_bid + top5_ask, 1e-9)
    denom_top10 = np.maximum(top10_bid + top10_ask, 1e-9)
    microprice = ((frame["best_ask"].to_numpy(dtype=float) * bid_qty) + (frame["best_bid"].to_numpy(dtype=float) * ask_qty)) / denom_top1
    frame["feat_microprice_offset_bps"] = ((microprice / frame["mid_price"].to_numpy(dtype=float)) - 1.0) * 10000.0
    frame["feat_top1_imbalance"] = (bid_qty - ask_qty) / denom_top1
    frame["feat_top5_imbalance"] = (top5_bid - top5_ask) / denom_top5
    frame["feat_top10_imbalance"] = (top10_bid - top10_ask) / denom_top10
    frame["feat_top1_depth_log"] = np.log1p(denom_top1)
    frame["feat_top5_depth_log"] = np.log1p(denom_top5)
    frame["feat_top10_depth_log"] = np.log1p(denom_top10)
    return frame


def add_depth_ofi_features(frame: pd.DataFrame) -> pd.DataFrame:
    bid = frame["best_bid"].to_numpy(dtype=float)
    ask = frame["best_ask"].to_numpy(dtype=float)
    bid_qty = frame["best_bid_qty"].to_numpy(dtype=float)
    ask_qty = frame["best_ask_qty"].to_numpy(dtype=float)
    prev_bid = np.roll(bid, 1)
    prev_ask = np.roll(ask, 1)
    prev_bid_qty = np.roll(bid_qty, 1)
    prev_ask_qty = np.roll(ask_qty, 1)
    event_ofi = np.zeros(len(frame), dtype=np.float64)
    if len(frame) > 1:
        event_ofi[1:] = (
            np.where(bid[1:] >= prev_bid[1:], bid_qty[1:], 0.0)
            - np.where(bid[1:] <= prev_bid[1:], prev_bid_qty[1:], 0.0)
            - np.where(ask[1:] <= prev_ask[1:], ask_qty[1:], 0.0)
            + np.where(ask[1:] >= prev_ask[1:], prev_ask_qty[1:], 0.0)
        )
    frame["feat_event_ofi"] = event_ofi
    frame["feat_bid_size_delta"] = bid_qty - prev_bid_qty
    frame["feat_ask_size_delta"] = ask_qty - prev_ask_qty
    frame.loc[0, ["feat_bid_size_delta", "feat_ask_size_delta"]] = 0.0
    timestamps = frame["timestamp_ms"].to_numpy(dtype=np.int64)
    for window_ms in QUOTE_WINDOWS_MS:
        window_label = f"{window_ms}ms"
        summed_ofi = rolling_sum_by_time(timestamps, event_ofi, window_ms)
        frame[f"feat_ofi_{window_label}"] = summed_ofi
        frame[f"feat_ofi_ratio_{window_label}"] = summed_ofi / np.maximum(frame["feat_top1_depth_log"].to_numpy(dtype=float), 1e-6)
        frame[f"feat_mid_return_{window_label}_bps"] = trailing_mid_return_bps(frame, window_ms)
    return frame


def add_trade_flow_features(frame: pd.DataFrame, trade_frame: pd.DataFrame) -> pd.DataFrame:
    event_ts = frame["timestamp_ms"].to_numpy(dtype=np.int64)
    trade_ts = trade_frame["timestamp_ms"].to_numpy(dtype=np.int64)
    total_qty = trade_frame["qty"].to_numpy(dtype=np.float64)
    signed_qty = trade_frame["signed_qty"].to_numpy(dtype=np.float64)
    total_notional = trade_frame["notional"].to_numpy(dtype=np.float64)
    signed_notional = trade_frame["signed_notional"].to_numpy(dtype=np.float64)
    buy_qty = trade_frame["buy_qty"].to_numpy(dtype=np.float64)
    sell_qty = trade_frame["sell_qty"].to_numpy(dtype=np.float64)
    buy_count = trade_frame["buy_count"].to_numpy(dtype=np.float64)
    sell_count = trade_frame["sell_count"].to_numpy(dtype=np.float64)
    total_count = buy_count + sell_count

    for window_ms in TRADE_WINDOWS_MS:
        window_label = f"{window_ms}ms"
        qty_total = rolling_window_sum(trade_ts, total_qty, event_ts, window_ms)
        qty_signed = rolling_window_sum(trade_ts, signed_qty, event_ts, window_ms)
        notional_total = rolling_window_sum(trade_ts, total_notional, event_ts, window_ms)
        notional_signed = rolling_window_sum(trade_ts, signed_notional, event_ts, window_ms)
        buy_qty_sum = rolling_window_sum(trade_ts, buy_qty, event_ts, window_ms)
        sell_qty_sum = rolling_window_sum(trade_ts, sell_qty, event_ts, window_ms)
        trade_count = rolling_window_sum(trade_ts, total_count, event_ts, window_ms)
        buy_count_sum = rolling_window_sum(trade_ts, buy_count, event_ts, window_ms)
        sell_count_sum = rolling_window_sum(trade_ts, sell_count, event_ts, window_ms)
        duration_seconds = window_ms / 1000.0
        frame[f"feat_trade_count_{window_label}"] = trade_count
        frame[f"feat_trade_intensity_{window_label}"] = trade_count / duration_seconds
        frame[f"feat_trade_qty_log_{window_label}"] = np.log1p(qty_total)
        frame[f"feat_trade_signed_qty_{window_label}"] = qty_signed
        frame[f"feat_trade_signed_ratio_{window_label}"] = qty_signed / np.maximum(qty_total, 1e-9)
        frame[f"feat_trade_buy_share_{window_label}"] = buy_qty_sum / np.maximum(buy_qty_sum + sell_qty_sum, 1e-9)
        frame[f"feat_trade_buy_count_share_{window_label}"] = buy_count_sum / np.maximum(buy_count_sum + sell_count_sum, 1e-9)
        frame[f"feat_trade_signed_notional_{window_label}"] = notional_signed
        frame[f"feat_trade_signed_notional_ratio_{window_label}"] = notional_signed / np.maximum(notional_total, 1e-9)
        frame[f"feat_trade_avg_size_{window_label}"] = qty_total / np.maximum(trade_count, 1.0)
    return frame


def resolve_capture_dirs(
    *,
    capture_root: Path,
    symbol: str,
    date: str | None,
    dates_csv: str | None,
    all_dates: bool,
) -> list[Path]:
    symbol_root = capture_root / symbol
    if not symbol_root.exists():
        raise SystemExit(f"Missing symbol capture directory: {symbol_root}")
    selected_dates: list[str]
    if all_dates:
        selected_dates = sorted(path.name for path in symbol_root.iterdir() if path.is_dir())
    elif dates_csv:
        selected_dates = sorted(part.strip() for part in dates_csv.split(",") if part.strip())
    elif date:
        selected_dates = [date]
    else:
        raise SystemExit("Provide one of --date, --dates, or --all-dates")
    if not selected_dates:
        raise SystemExit("No capture dates resolved")
    capture_dirs = [symbol_root / value for value in selected_dates]
    missing = [path for path in capture_dirs if not path.exists()]
    if missing:
        raise SystemExit(f"Missing capture directories: {missing}")
    return capture_dirs


def capture_date_label(capture_dirs: list[Path]) -> str:
    dates = [path.name for path in capture_dirs]
    if len(dates) == 1:
        return dates[0]
    return f"{dates[0]}_{dates[-1]}_{len(dates)}d"


def build_depth_row(*, event: dict[str, Any], bids: list[list[str]], asks: list[list[str]], levels: int) -> dict[str, Any] | None:
    parsed_bids = parse_side(bids, levels)
    parsed_asks = parse_side(asks, levels)
    if parsed_bids is None or parsed_asks is None:
        return None
    bid_prices, bid_sizes = parsed_bids
    ask_prices, ask_sizes = parsed_asks
    best_bid = bid_prices[0]
    best_ask = ask_prices[0]
    if best_bid <= 0.0 or best_ask <= 0.0 or best_ask <= best_bid:
        return None
    mid = (best_bid + best_ask) / 2.0
    spread_bps = ((best_ask - best_bid) / mid) * 10000.0
    row: dict[str, Any] = {
        "timestamp_ms": int(event.get("E") or event.get("T") or 0),
        "mid_price": mid,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "best_bid_qty": bid_sizes[0],
        "best_ask_qty": ask_sizes[0],
        "spread_bps": spread_bps,
    }
    for idx, price in enumerate(bid_prices, start=1):
        row[f"bid_price_bps_{idx}"] = ((price / mid) - 1.0) * 10000.0
        row[f"bid_size_{idx}"] = bid_sizes[idx - 1]
    for idx, size in enumerate(bid_sizes, start=1):
        row[f"bid_log_size_{idx}"] = math.log1p(size)
    for idx, price in enumerate(ask_prices, start=1):
        row[f"ask_price_bps_{idx}"] = ((price / mid) - 1.0) * 10000.0
        row[f"ask_size_{idx}"] = ask_sizes[idx - 1]
    for idx, size in enumerate(ask_sizes, start=1):
        row[f"ask_log_size_{idx}"] = math.log1p(size)
    return row


def parse_side(levels_data: list[list[str]], levels: int) -> tuple[list[float], list[float]] | None:
    prices: list[float] = []
    sizes: list[float] = []
    for level in levels_data[:levels]:
        if len(level) < 2:
            return None
        prices.append(float(level[0]))
        sizes.append(float(level[1]))
    if len(prices) < levels:
        prices.extend([prices[-1]] * (levels - len(prices)))
        sizes.extend([0.0] * (levels - len(sizes)))
    return prices, sizes


def compute_event_horizon_target(mid_prices: np.ndarray, horizon: int) -> np.ndarray:
    out = np.full(len(mid_prices), np.nan, dtype=np.float32)
    if horizon <= 0:
        return out
    future = shift_forward(mid_prices, horizon)
    valid = np.isfinite(mid_prices) & np.isfinite(future) & (mid_prices > 0.0)
    out[valid] = ((future[valid] / mid_prices[valid]) - 1.0) * 10000.0
    return out


def shift_forward(values: np.ndarray, horizon: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float64)
    if horizon < len(values):
        out[:-horizon] = values[horizon:]
    return out


def rolling_sum_by_time(timestamps: np.ndarray, values: np.ndarray, window_ms: int) -> np.ndarray:
    return rolling_window_sum(timestamps, values, timestamps, window_ms)


def rolling_window_sum(source_timestamps: np.ndarray, values: np.ndarray, target_timestamps: np.ndarray, window_ms: int) -> np.ndarray:
    padded = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    end_pos = np.searchsorted(source_timestamps, target_timestamps, side="right")
    start_pos = np.searchsorted(source_timestamps, target_timestamps - window_ms, side="left")
    return padded[end_pos] - padded[start_pos]


def trailing_mid_return_bps(frame: pd.DataFrame, window_ms: int) -> np.ndarray:
    timestamps = frame["timestamp_ms"].to_numpy(dtype=np.int64)
    mid = frame["mid_price"].to_numpy(dtype=np.float64)
    prior_pos = np.searchsorted(timestamps, timestamps - window_ms, side="left")
    prior_pos = np.clip(prior_pos, 0, len(frame) - 1)
    prior_mid = mid[prior_pos]
    out = np.zeros(len(frame), dtype=np.float64)
    valid = prior_mid > 0.0
    out[valid] = ((mid[valid] / prior_mid[valid]) - 1.0) * 10000.0
    out[prior_pos == np.arange(len(frame))] = 0.0
    return out


def compute_time_horizon_targets(frame: pd.DataFrame, horizon_seconds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps = frame["timestamp_ms"].to_numpy(dtype=np.int64)
    mid_prices = frame["mid_price"].to_numpy(dtype=np.float64)
    spreads = frame["spread_bps"].to_numpy(dtype=np.float64)
    out_target = np.full(len(frame), np.nan, dtype=np.float32)
    out_spread = np.full(len(frame), np.nan, dtype=np.float32)
    out_hold = np.full(len(frame), np.nan, dtype=np.float32)
    horizon_ms = int(round(horizon_seconds * 1000.0))
    future_idx = np.searchsorted(timestamps, timestamps + horizon_ms, side="left")
    valid_mask = future_idx < len(frame)
    current_idx = np.flatnonzero(valid_mask)
    target_idx = future_idx[valid_mask]
    out_target[current_idx] = ((mid_prices[target_idx] / mid_prices[current_idx]) - 1.0) * 10000.0
    out_spread[current_idx] = spreads[target_idx]
    out_hold[current_idx] = (timestamps[target_idx] - timestamps[current_idx]) / 1000.0
    return out_target, out_spread, out_hold


def split_dataset(frame: pd.DataFrame) -> dict[str, int]:
    n = len(frame)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)
    return {"train_end": train_end, "valid_end": valid_end}


def split_samples(n: int) -> dict[str, int]:
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)
    return {"train_end": train_end, "valid_end": valid_end}


def build_scaler(frame: pd.DataFrame, feature_columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    mean = frame[feature_columns].mean().to_numpy(dtype=np.float32)
    std = frame[feature_columns].std().replace(0.0, 1.0).fillna(1.0).to_numpy(dtype=np.float32)
    std[std == 0.0] = 1.0
    return mean, std


def build_samples(
    *,
    dataset: pd.DataFrame,
    feature_columns: list[str],
    target_spec: TargetSpec,
    sequence_length: int,
    scaler: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = dataset[feature_columns].to_numpy(dtype=np.float32)
    mean, std = scaler
    features = (features - mean) / std
    labels = dataset[target_spec.target_column].to_numpy(dtype=np.float32)
    hold_seconds = dataset[target_spec.hold_column].to_numpy(dtype=np.float32)

    sequences: list[np.ndarray] = []
    flattened: list[np.ndarray] = []
    sequence_labels: list[float] = []
    sample_meta: list[tuple[float, float]] = []
    for idx in range(sequence_length - 1, len(dataset)):
        if not np.isfinite(labels[idx]) or not np.isfinite(hold_seconds[idx]):
            continue
        seq = features[idx - sequence_length + 1 : idx + 1]
        sequences.append(seq)
        flattened.append(seq.reshape(-1))
        sequence_labels.append(labels[idx])
        sample_meta.append((float(idx), float(hold_seconds[idx])))
    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(flattened, dtype=np.float32),
        np.asarray(sequence_labels, dtype=np.float32),
        np.asarray(sample_meta, dtype=np.float32),
    )


def split_array(values: np.ndarray, split_idx: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        values[: split_idx["train_end"]],
        values[split_idx["train_end"] : split_idx["valid_end"]],
        values[split_idx["valid_end"] :],
    )


def build_xgboost_regressor(*, n_estimators: int, max_depth: int, n_jobs: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=n_jobs,
        verbosity=0,
        random_state=RANDOM_SEED,
    )


def evaluate_regression(labels: np.ndarray, predictions: np.ndarray) -> RegressionMetrics:
    mae = float(mean_absolute_error(labels, predictions))
    direction = (labels > 0.0).astype(int)
    pred_direction = predictions
    base_rate = float(direction.mean()) if len(direction) else float("nan")
    auc = 0.5
    if len(np.unique(direction)) > 1:
        auc = float(roc_auc_score(direction, pred_direction))
    return RegressionMetrics(mae_bps=mae, directional_auc=auc, directional_base_rate=base_rate)


def choose_trade_quantile(
    *,
    predictions: np.ndarray,
    labels: np.ndarray,
    sample_indices: np.ndarray,
    dataset: pd.DataFrame,
    trade_frame: pd.DataFrame,
    replay_config: ReplayConfig,
    min_validation_trades: int,
) -> TradeMetrics:
    best: TradeMetrics | None = None
    for quantile in THRESHOLD_GRID:
        threshold_abs = float(np.quantile(np.abs(predictions), quantile))
        metrics = evaluate_trade_policy(
            predictions=predictions,
            labels=labels,
            sample_indices=sample_indices,
            dataset=dataset,
            trade_frame=trade_frame,
            replay_config=replay_config,
            quantile=quantile,
            threshold_abs_bps=threshold_abs,
        )
        if metrics.trades < min_validation_trades:
            continue
        if best is None or trade_key(metrics) > trade_key(best):
            best = metrics
    if best is None:
        return empty_trade_metrics(0, None, None)
    return best


def evaluate_trade_policy(
    *,
    predictions: np.ndarray,
    labels: np.ndarray,
    sample_indices: np.ndarray,
    dataset: pd.DataFrame,
    trade_frame: pd.DataFrame,
    replay_config: ReplayConfig,
    quantile: float | None,
    threshold_abs_bps: float | None,
) -> TradeMetrics:
    if threshold_abs_bps is None or len(predictions) == 0:
        return empty_trade_metrics(0, quantile, threshold_abs_bps)
    selected = np.abs(predictions) >= threshold_abs_bps
    selected_signals = int(selected.sum())
    if not np.any(selected):
        return empty_trade_metrics(selected_signals, quantile, threshold_abs_bps)
    replay_rows = simulate_trade_replay(
        predictions=predictions[selected],
        sample_meta=sample_indices[selected],
        dataset=dataset,
        trade_frame=trade_frame,
        replay_config=replay_config,
    )
    if not replay_rows:
        return empty_trade_metrics(selected_signals, quantile, threshold_abs_bps)
    gross = np.asarray([row["gross_pnl_bps"] for row in replay_rows], dtype=np.float64)
    net = np.asarray([row["net_pnl_bps"] for row in replay_rows], dtype=np.float64)
    wins = net > 0.0
    gross_total = float(gross.sum())
    net_total = float(net.sum())
    losses = float(-net[net < 0.0].sum())
    gains = float(net[net > 0.0].sum())
    profit_factor = float(gains / losses) if losses > 0.0 else float("inf") if gains > 0 else 0.0
    passive_exits = sum(1 for row in replay_rows if row["exit_mode"] == "passive")
    forced_exits = sum(1 for row in replay_rows if row["exit_mode"] == "forced_taker")
    return TradeMetrics(
        signals_considered=selected_signals,
        trades=len(replay_rows),
        fill_rate=float(len(replay_rows) / max(selected_signals, 1)),
        win_rate=float(wins.mean()) if len(wins) else 0.0,
        gross_pnl_bps=gross_total,
        net_pnl_bps=net_total,
        avg_net_pnl_bps=float(net.mean()) if len(net) else 0.0,
        avg_hold_seconds=float(np.mean([row["hold_seconds"] for row in replay_rows])) if replay_rows else 0.0,
        profit_factor=float(profit_factor),
        passive_exit_rate=float(passive_exits / len(replay_rows)) if replay_rows else 0.0,
        forced_taker_exit_rate=float(forced_exits / len(replay_rows)) if replay_rows else 0.0,
        chosen_quantile=quantile,
        threshold_abs_bps=threshold_abs_bps,
    )


def trade_key(metrics: TradeMetrics) -> tuple[float, float, float]:
    return (metrics.net_pnl_bps, metrics.profit_factor, metrics.win_rate)


def empty_trade_metrics(signals_considered: int, quantile: float | None, threshold_abs_bps: float | None) -> TradeMetrics:
    return TradeMetrics(
        signals_considered=signals_considered,
        trades=0,
        fill_rate=0.0,
        win_rate=0.0,
        gross_pnl_bps=0.0,
        net_pnl_bps=0.0,
        avg_net_pnl_bps=0.0,
        avg_hold_seconds=0.0,
        profit_factor=0.0,
        passive_exit_rate=0.0,
        forced_taker_exit_rate=0.0,
        chosen_quantile=quantile,
        threshold_abs_bps=threshold_abs_bps,
    )


def simulate_trade_replay(
    *,
    predictions: np.ndarray,
    sample_meta: np.ndarray,
    dataset: pd.DataFrame,
    trade_frame: pd.DataFrame,
    replay_config: ReplayConfig,
) -> list[dict[str, float | str]]:
    depth_ts = dataset["timestamp_ms"].to_numpy(dtype=np.int64)
    trade_ts = trade_frame["timestamp_ms"].to_numpy(dtype=np.int64)
    trade_price = trade_frame["price"].to_numpy(dtype=np.float64)
    trade_qty = trade_frame["qty"].to_numpy(dtype=np.float64)
    trade_is_buy = trade_frame["is_buy_aggressor"].to_numpy(dtype=bool)
    replay_rows: list[dict[str, float | str]] = []
    next_available_ts = 0
    ordered = sorted(
        zip(sample_meta.tolist(), predictions.tolist()),
        key=lambda value: int(dataset.iloc[int(value[0][0])]["timestamp_ms"]),
    )
    for meta, prediction in ordered:
        sample_idx = int(meta[0])
        hold_seconds = float(meta[1])
        row = dataset.iloc[sample_idx]
        signal_ts = int(row["timestamp_ms"])
        if signal_ts < next_available_ts:
            continue
        side = "long" if prediction > 0.0 else "short"
        entry_quote_idx = find_quote_snapshot_idx(depth_ts, signal_ts, replay_config.max_quote_age_ms)
        if entry_quote_idx is None:
            continue
        if side == "long":
            entry_price = float(dataset.iloc[entry_quote_idx]["best_bid"])
            queue_ahead_qty = float(dataset.iloc[entry_quote_idx]["best_bid_qty"]) * replay_config.queue_ahead_multiplier
        else:
            entry_price = float(dataset.iloc[entry_quote_idx]["best_ask"])
            queue_ahead_qty = float(dataset.iloc[entry_quote_idx]["best_ask_qty"]) * replay_config.queue_ahead_multiplier
        order_qty = replay_config.order_notional_usd / max(float(dataset.iloc[entry_quote_idx]["mid_price"]), 1e-9)
        entry_fill = match_passive_fill(
            trade_ts=trade_ts,
            trade_price=trade_price,
            trade_qty=trade_qty,
            trade_is_buy=trade_is_buy,
            side=side,
            limit_price=entry_price,
            queue_ahead_qty=queue_ahead_qty,
            order_qty=order_qty,
            start_ts=signal_ts,
            end_ts=signal_ts + int(round(replay_config.entry_timeout_seconds * 1000.0)),
        )
        if entry_fill is None:
            continue
        fill_ts = entry_fill
        intended_exit_ts = fill_ts + int(round(hold_seconds * 1000.0))
        exit_quote_idx = find_quote_snapshot_idx(depth_ts, intended_exit_ts, replay_config.max_quote_age_ms)
        if exit_quote_idx is None:
            continue
        if side == "long":
            passive_exit_price = float(dataset.iloc[exit_quote_idx]["best_ask"])
            exit_queue_qty = float(dataset.iloc[exit_quote_idx]["best_ask_qty"]) * replay_config.queue_ahead_multiplier
        else:
            passive_exit_price = float(dataset.iloc[exit_quote_idx]["best_bid"])
            exit_queue_qty = float(dataset.iloc[exit_quote_idx]["best_bid_qty"]) * replay_config.queue_ahead_multiplier
        passive_exit_fill = match_passive_fill(
            trade_ts=trade_ts,
            trade_price=trade_price,
            trade_qty=trade_qty,
            trade_is_buy=trade_is_buy,
            side="short" if side == "long" else "long",
            limit_price=passive_exit_price,
            queue_ahead_qty=exit_queue_qty,
            order_qty=order_qty,
            start_ts=intended_exit_ts,
            end_ts=intended_exit_ts + int(round(replay_config.exit_timeout_seconds * 1000.0)),
        )
        exit_mode = "passive"
        exit_fee_bps = replay_config.maker_fee_bps_per_side
        if passive_exit_fill is None:
            forced_idx = find_quote_snapshot_idx(
                depth_ts,
                intended_exit_ts + int(round(replay_config.exit_timeout_seconds * 1000.0)),
                replay_config.max_quote_age_ms,
            )
            if forced_idx is None:
                continue
            passive_exit_fill = int(depth_ts[forced_idx])
            if side == "long":
                passive_exit_price = float(dataset.iloc[forced_idx]["best_bid"])
            else:
                passive_exit_price = float(dataset.iloc[forced_idx]["best_ask"])
            exit_mode = "forced_taker"
            exit_fee_bps = replay_config.taker_fee_bps_per_side
        gross_pnl_bps = (
            ((passive_exit_price / entry_price) - 1.0) * 10000.0
            if side == "long"
            else ((entry_price / passive_exit_price) - 1.0) * 10000.0
        )
        net_pnl_bps = gross_pnl_bps - replay_config.maker_fee_bps_per_side - exit_fee_bps
        replay_rows.append(
            {
                "side": side,
                "gross_pnl_bps": gross_pnl_bps,
                "net_pnl_bps": net_pnl_bps,
                "hold_seconds": max((passive_exit_fill - fill_ts) / 1000.0, 0.0),
                "exit_mode": exit_mode,
            }
        )
        next_available_ts = passive_exit_fill
    return replay_rows


def find_quote_snapshot_idx(depth_ts: np.ndarray, target_ts: int, max_quote_age_ms: int) -> int | None:
    idx = int(np.searchsorted(depth_ts, target_ts, side="right") - 1)
    if idx < 0:
        return None
    if target_ts - int(depth_ts[idx]) > max_quote_age_ms:
        return None
    return idx


def match_passive_fill(
    *,
    trade_ts: np.ndarray,
    trade_price: np.ndarray,
    trade_qty: np.ndarray,
    trade_is_buy: np.ndarray,
    side: str,
    limit_price: float,
    queue_ahead_qty: float,
    order_qty: float,
    start_ts: int,
    end_ts: int,
) -> int | None:
    start_idx = int(np.searchsorted(trade_ts, start_ts, side="right"))
    end_idx = int(np.searchsorted(trade_ts, end_ts, side="right"))
    if end_idx <= start_idx:
        return None
    required_qty = queue_ahead_qty + order_qty
    consumed_qty = 0.0
    for idx in range(start_idx, end_idx):
        if side == "long":
            qualifies = (not trade_is_buy[idx]) and trade_price[idx] <= limit_price
        else:
            qualifies = trade_is_buy[idx] and trade_price[idx] >= limit_price
        if not qualifies:
            continue
        consumed_qty += float(trade_qty[idx])
        if consumed_qty >= required_qty:
            return int(trade_ts[idx])
    return None


def train_deep_regressor(
    *,
    model: nn.Module,
    train_sequences: np.ndarray,
    train_labels: np.ndarray,
    valid_sequences: np.ndarray,
    valid_labels: np.ndarray,
    batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> DeepTrainingSummary:
    train_tensor = torch.from_numpy(train_sequences)
    train_target_tensor = torch.from_numpy(train_labels)
    valid_tensor = torch.from_numpy(valid_sequences)
    if device.type != "cpu":
        train_tensor = train_tensor.to(device)
        train_target_tensor = train_target_tensor.to(device)
        valid_tensor = valid_tensor.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        foreach=False,
    )
    loss_fn = nn.HuberLoss()
    best_mae = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(len(train_sequences))
        train_loss_total = 0.0
        train_batches = 0
        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]
            if device.type == "cpu":
                features = train_tensor[batch_idx].to(device)
                targets = train_target_tensor[batch_idx].to(device)
            else:
                index = torch.as_tensor(batch_idx, device=device)
                features = train_tensor.index_select(0, index)
                targets = train_target_tensor.index_select(0, index)
            optimizer.zero_grad(set_to_none=True)
            preds = model(features)
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss_total += float(loss.detach().cpu())
            train_batches += 1
        valid_preds = predict_deep_tensor(model, valid_tensor, device=device, batch_size=batch_size)
        mae = float(mean_absolute_error(valid_labels, valid_preds))
        avg_train_loss = train_loss_total / max(train_batches, 1)
        print(
            f"[lob][deep] epoch={epoch}/{epochs} train_loss={avg_train_loss:.6f} valid_mae={mae:.6f}",
            flush=True,
        )
        if mae < best_mae:
            best_mae = mae
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise SystemExit("DeepLOB regressor failed to produce a checkpoint")
    model.load_state_dict(best_state)
    return DeepTrainingSummary(
        best_epoch=best_epoch,
        best_valid_mae_bps=best_mae,
        train_rows=int(len(train_sequences)),
        valid_rows=int(len(valid_sequences)),
        test_rows=0,
    )


def predict_deep(model: nn.Module, sequences: np.ndarray, *, device: torch.device, batch_size: int) -> np.ndarray:
    tensor = torch.from_numpy(sequences)
    if device.type != "cpu":
        tensor = tensor.to(device)
    return predict_deep_tensor(model, tensor, device=device, batch_size=batch_size)


def predict_deep_tensor(model: nn.Module, sequences: torch.Tensor, *, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            if device.type == "cpu":
                batch = sequences[start : start + batch_size].to(device)
            else:
                batch = sequences[start : start + batch_size]
            preds = model(batch).detach().cpu().numpy()
            outputs.append(preds)
    return np.concatenate(outputs).astype(np.float32) if outputs else np.empty((0,), dtype=np.float32)


def resolve_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise SystemExit("MPS requested but not available")
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Futures LOB Model Comparison",
        "",
        f"- Rows: `{report['rows']:,}`",
        f"- Features: `{report['feature_count']}`",
        f"- Sequence length: `{report['sequence_length']}` events",
        f"- Event horizons: `{report['event_horizons']}`",
        f"- Time horizons: `{report['time_horizons_seconds']}`",
        f"- Maker fee per side: `{report['maker_fee_bps_per_side']}` bps",
        f"- Taker fee per side: `{report['taker_fee_bps_per_side']}` bps",
        f"- Passive entry timeout: `{report['entry_timeout_seconds']}` s",
        f"- Passive exit timeout: `{report['exit_timeout_seconds']}` s",
        "",
    ]
    for target_name, target_info in report["targets"].items():
        lines.append(f"## `{target_name}`")
        lines.append("")
        for model_name, model_info in target_info["models"].items():
            test_metrics = model_info["test_metrics"]
            test_trade = model_info["test_trade"]
            lines.append(
                f"- `{model_name}`: test MAE `{test_metrics['mae_bps']:.4f}` bps, test AUC `{test_metrics['directional_auc']:.4f}`, test net PnL `{test_trade['net_pnl_bps']:.4f}` bps across `{test_trade['trades']}` fills (fill rate `{test_trade['fill_rate']:.2%}`, passive exits `{test_trade['passive_exit_rate']:.2%}`)"
            )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
