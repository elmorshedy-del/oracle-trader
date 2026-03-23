#!/usr/bin/env python3
"""
Run an adapted DeepLOB-style comparison on the same 5-second BTCUSDT impulse dataset.

This is intentionally isolated from the frozen CatBoost baseline:
- same raw Binance data family
- same impulse labeling
- same anchored chronological split

It is "DeepLOB-style" rather than a literal paper reproduction because the current
research bundle is built from engineered 5-second rows, not raw multi-level LOB tensors.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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


DEFAULT_COMPARE_OUTPUT_ROOT = Path("output/futures_ml_deeplob_compare")
UTC = timezone.utc
DEFAULT_SEQUENCE_LENGTH = 20
DEFAULT_BATCH_SIZE = 256
DEFAULT_EPOCHS = 12
DEFAULT_PATIENCE = 3
DEFAULT_CONV_CHANNELS = 32
DEFAULT_LSTM_HIDDEN = 48
DEFAULT_DROPOUT = 0.20
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42


@dataclass
class SplitMetrics:
    auc: float
    precision_at_top_decile: float
    base_rate: float


@dataclass
class TrainingSummary:
    best_epoch: int
    best_valid_auc: float
    train_rows: int
    valid_rows: int
    test_rows: int


class DeepLOBStyleModel(nn.Module):
    def __init__(
        self,
        *,
        input_features: int,
        conv_channels: int,
        lstm_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        feature_kernel = 3 if input_features >= 3 else 1
        feature_padding = feature_kernel // 2
        branch_channels = conv_channels
        merged_channels = branch_channels * 3

        self.stem = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=(1, feature_kernel), padding=(0, feature_padding), bias=False),
            nn.BatchNorm2d(conv_channels),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(conv_channels),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.branch_short = nn.Sequential(
            nn.Conv2d(conv_channels, branch_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.branch_medium = nn.Sequential(
            nn.Conv2d(conv_channels, branch_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.branch_long = nn.Sequential(
            nn.Conv2d(conv_channels, branch_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=(5, 1), padding=(2, 0), bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.LeakyReLU(negative_slope=0.01),
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
            nn.LeakyReLU(negative_slope=0.01),
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
        sequence_out, _ = self.sequence_model(self.dropout(x))
        logits = self.head(sequence_out[:, -1, :]).squeeze(-1)
        return logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare a DeepLOB-style sequence model on the 5-second BTCUSDT impulse dataset.")
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
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH, help="Number of 5-second rows fed into each sequence sample")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Maximum training epochs per label")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early-stop patience on validation AUC")
    parser.add_argument("--conv-channels", type=int, default=DEFAULT_CONV_CHANNELS, help="Channel count in the DeepLOB-style conv stem")
    parser.add_argument("--lstm-hidden", type=int, default=DEFAULT_LSTM_HIDDEN, help="Hidden size in the bidirectional LSTM head")
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT, help="Dropout used in the sequence head")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE, help="AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="AdamW weight decay")
    parser.add_argument("--output-root", default=str(DEFAULT_COMPARE_OUTPUT_ROOT), help="Comparison output directory")
    parser.add_argument("--raw-output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Location of shared futures raw cache")
    parser.add_argument("--skip-download", action="store_true", help="Reuse existing downloaded archives")
    parser.add_argument(
        "--skip-dataset-export",
        action="store_true",
        help="Do not write the full master feature table to disk; faster for repeated compare runs.",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"), help="Torch device selection")
    parser.add_argument("--max-download-workers", type=int, default=1, help="Parallel archive download workers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seed(RANDOM_SEED)

    output_root = Path(args.output_root).resolve()
    raw_output_root = Path(args.raw_output_root).resolve()
    symbol = args.symbol.upper()
    end_date = parse_iso_date(args.end_date) if args.end_date else (datetime.now(UTC).date() - timedelta(days=1))
    start_date = parse_iso_date(args.start_date) if args.start_date else (end_date - timedelta(days=args.lookback_days - 1))
    if start_date > end_date:
        raise SystemExit("start-date must be on or before end-date")

    run_name = (
        f"binance_{symbol.lower()}_{args.bucket_seconds}s_impulse_deeplob_"
        f"{start_date:%Y%m%d}_{end_date:%Y%m%d}_"
        f"tp{int(round(args.profit_bps))}_sl{int(round(args.stop_bps))}_"
        f"sig{int(round(args.min_signed_ratio * 100)):03d}_"
        f"dep{int(round(args.min_depth_imbalance * 100)):03d}_"
        f"tz{int(round(args.min_trade_z * 100)):03d}_"
        f"eff{int(round(args.min_directional_efficiency * 100)):03d}_"
        f"src{int(round(args.source_threshold * 100)):03d}_"
        f"seq{args.sequence_length:03d}_v1"
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
    scaler = build_scaler(train_frame, feature_columns)
    device = resolve_device(args.device)

    report: dict[str, Any] = {
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
        "deeplob_style_params": {
            "sequence_length": args.sequence_length,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "conv_channels": args.conv_channels,
            "lstm_hidden": args.lstm_hidden,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "device": device.type,
        },
        "labels": {},
    }

    for label_name, candidate_column in (
        ("long_followthrough_label", "long_impulse_candidate"),
        ("short_followthrough_label", "short_impulse_candidate"),
    ):
        train_sequences, train_labels = build_sequence_samples(
            frame=train_frame,
            candidate_column=candidate_column,
            label_name=label_name,
            feature_columns=feature_columns,
            sequence_length=args.sequence_length,
            scaler=scaler,
        )
        valid_sequences, valid_labels = build_sequence_samples(
            frame=valid_frame,
            candidate_column=candidate_column,
            label_name=label_name,
            feature_columns=feature_columns,
            sequence_length=args.sequence_length,
            scaler=scaler,
        )
        test_sequences, test_labels = build_sequence_samples(
            frame=test_frame,
            candidate_column=candidate_column,
            label_name=label_name,
            feature_columns=feature_columns,
            sequence_length=args.sequence_length,
            scaler=scaler,
        )
        if min(len(train_sequences), len(valid_sequences), len(test_sequences)) == 0:
            raise SystemExit(f"No sequence rows available for {label_name}")

        model = DeepLOBStyleModel(
            input_features=train_sequences.shape[-1],
            conv_channels=args.conv_channels,
            lstm_hidden=args.lstm_hidden,
            dropout=args.dropout,
        ).to(device)
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(train_sequences), torch.from_numpy(train_labels)),
            batch_size=args.batch_size,
            shuffle=True,
        )
        valid_loader = DataLoader(
            TensorDataset(torch.from_numpy(valid_sequences), torch.from_numpy(valid_labels)),
            batch_size=args.batch_size,
            shuffle=False,
        )

        model_summary = train_model(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            valid_sequences=valid_sequences,
            valid_labels=valid_labels,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        model_file = model_root / f"{label_name}.pt"
        torch.save(model.state_dict(), model_file)

        train_probs = predict_probabilities(model, train_sequences, device=device, batch_size=args.batch_size)
        valid_probs = predict_probabilities(model, valid_sequences, device=device, batch_size=args.batch_size)
        test_probs = predict_probabilities(model, test_sequences, device=device, batch_size=args.batch_size)

        report["labels"][label_name] = {
            "candidate_rows": {
                "train": int(len(train_sequences)),
                "valid": int(len(valid_sequences)),
                "test": int(len(test_sequences)),
            },
            "positive_rate": {
                "train": float(train_labels.mean()),
                "valid": float(valid_labels.mean()),
                "test": float(test_labels.mean()),
            },
            "metrics": {
                "train": asdict(evaluate_probabilities(train_labels, train_probs)),
                "valid": asdict(evaluate_probabilities(valid_labels, valid_probs)),
                "test": asdict(evaluate_probabilities(test_labels, test_probs)),
            },
            "training": asdict(model_summary),
        }

    json_path = report_root / "comparison_report.json"
    md_path = report_root / "comparison_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    print(f"DeepLOB-style report: {md_path}")


def build_scaler(frame: pd.DataFrame, feature_columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    mean = frame[feature_columns].mean().to_numpy(dtype=np.float32)
    std = frame[feature_columns].std().replace(0.0, 1.0).fillna(1.0).to_numpy(dtype=np.float32)
    std[std == 0.0] = 1.0
    return mean, std


def build_sequence_samples(
    *,
    frame: pd.DataFrame,
    candidate_column: str,
    label_name: str,
    feature_columns: list[str],
    sequence_length: int,
    scaler: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    ordered = frame.sort_values("timestamp").reset_index(drop=True)
    features = ordered[feature_columns].to_numpy(dtype=np.float32)
    mean, std = scaler
    features = (features - mean) / std

    candidate_mask = ordered[candidate_column].to_numpy(dtype=np.int8) == 1
    labels = ordered[label_name].to_numpy(dtype=np.float32)

    sequences: list[np.ndarray] = []
    sequence_labels: list[float] = []
    for idx in np.flatnonzero(candidate_mask):
        start = idx - sequence_length + 1
        if start < 0:
            continue
        sequences.append(features[start : idx + 1])
        sequence_labels.append(labels[idx])

    if not sequences:
        return (
            np.empty((0, sequence_length, len(feature_columns)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    return np.stack(sequences).astype(np.float32), np.asarray(sequence_labels, dtype=np.float32)


def evaluate_probabilities(labels: np.ndarray, probabilities: np.ndarray) -> SplitMetrics:
    labels = np.asarray(labels, dtype=np.float32)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if len(labels) == 0:
        return SplitMetrics(auc=float("nan"), precision_at_top_decile=float("nan"), base_rate=float("nan"))
    if len(np.unique(labels)) < 2:
        auc = 0.5
    else:
        auc = float(roc_auc_score(labels, probabilities))
    top_n = max(1, int(np.ceil(len(probabilities) * 0.10)))
    top_indices = np.argsort(probabilities)[-top_n:]
    precision_at_top_decile = float(labels[top_indices].mean()) if len(top_indices) else float("nan")
    return SplitMetrics(
        auc=auc,
        precision_at_top_decile=precision_at_top_decile,
        base_rate=float(labels.mean()),
    )


def predict_probabilities(
    model: nn.Module,
    sequences: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = torch.from_numpy(sequences[start : start + batch_size]).to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            outputs.append(probs)
    if not outputs:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(outputs).astype(np.float32)


def train_model(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    valid_sequences: np.ndarray,
    valid_labels: np.ndarray,
    device: torch.device,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
) -> TrainingSummary:
    train_targets = train_loader.dataset.tensors[1].numpy()
    positive_rate = float(train_targets.mean())
    pos_weight_value = (1.0 - positive_rate) / max(positive_rate, 1e-6)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_auc = float("-inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()

        valid_probs = predict_probabilities(model, valid_sequences, device=device, batch_size=valid_loader.batch_size or DEFAULT_BATCH_SIZE)
        valid_auc = evaluate_probabilities(valid_labels, valid_probs).auc
        if valid_auc > best_auc:
            best_auc = valid_auc
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise SystemExit("DeepLOB-style training never produced a valid checkpoint")
    model.load_state_dict(best_state)
    return TrainingSummary(
        best_epoch=best_epoch,
        best_valid_auc=float(best_auc),
        train_rows=int(len(train_loader.dataset)),
        valid_rows=int(len(valid_loader.dataset)),
        test_rows=0,
    )


def resolve_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise SystemExit("MPS requested but not available")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Futures ML DeepLOB-Style Comparison",
        "",
        f"- Symbol: `{report['symbol']}`",
        f"- Date range: `{report['date_range']['start']}` to `{report['date_range']['end']}`",
        f"- Bucket: `{report['bucket_seconds']}`s",
        f"- Profit barrier: `{report['profit_bps']}` bps",
        f"- Stop barrier: `{report['stop_bps']}` bps",
        f"- Sequence length: `{report['deeplob_style_params']['sequence_length']}`",
        f"- Device: `{report['deeplob_style_params']['device']}`",
        f"- Master rows: `{report['master_rows']:,}`",
        f"- Long candidate rate: `{report['candidate_summary']['long_rate']:.4f}`",
        f"- Short candidate rate: `{report['candidate_summary']['short_rate']:.4f}`",
        "",
    ]
    for label_name, label_report in report["labels"].items():
        lines.append(f"## `{label_name}`")
        lines.append("")
        lines.append(
            f"- Train rows: `{label_report['candidate_rows']['train']:,}` | Valid rows: `{label_report['candidate_rows']['valid']:,}` | Test rows: `{label_report['candidate_rows']['test']:,}`"
        )
        lines.append(
            f"- Best epoch: `{label_report['training']['best_epoch']}` | Best valid AUC `{label_report['training']['best_valid_auc']:.4f}`"
        )
        for split_name in ("train", "valid", "test"):
            metrics = label_report["metrics"][split_name]
            lines.append(
                f"- {split_name.title()} AUC `{metrics['auc']:.4f}` | precision@top-decile `{metrics['precision_at_top_decile']:.4f}` | base rate `{metrics['base_rate']:.4f}`"
            )
        lines.append("")
    return "\n".join(lines)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


if __name__ == "__main__":
    main()
