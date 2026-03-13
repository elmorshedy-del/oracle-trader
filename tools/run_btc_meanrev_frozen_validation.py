#!/usr/bin/env python3
"""
Run frozen out-of-sample validation for the BTC downshock mean-reversion winner.

This runner does not search parameters. It applies the locked validation spec to
any new multivenue dataset and reports overall plus day-level performance.
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


UTC = timezone.utc
DEFAULT_OUTPUT_ROOT = Path("output/btc_meanrev_validation")


@dataclass
class ValidationResult:
    trades: int
    wins: int
    win_rate: float
    total_net_bps: float
    avg_net_bps: float
    median_net_bps: float
    profit_factor: float | None
    take_profit_exits: int
    stop_loss_exits: int
    timeout_exits: int
    no_path_skips: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen validation for BTC downshock mean reversion.")
    parser.add_argument("--spec-path", required=True, help="Path to frozen validation spec JSON")
    parser.add_argument("--dataset-path", required=True, help="Path to multivenue features.csv.gz")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output root for validation runs")
    parser.add_argument("--bootstrap-samples", type=int, default=5000, help="Bootstrap samples for day-level outcomes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = datetime.now(UTC)
    spec_path = Path(args.spec_path).resolve()
    dataset_path = Path(args.dataset_path).resolve()
    if not spec_path.exists():
        raise SystemExit(f"Missing spec: {spec_path}")
    if not dataset_path.exists():
        raise SystemExit(f"Missing dataset: {dataset_path}")

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    model_path = Path(spec["model_path"]).resolve()
    if not model_path.exists():
        raise SystemExit(f"Missing model from spec: {model_path}")

    signal = spec["signal_definition"]
    execution = spec["execution_definition"]

    df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
    feature_columns = [column for column in df.columns if not column.startswith("target_")]
    X = df[feature_columns].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

    past_window_column = f"fut_ret_{int(signal['shock_window_seconds'])}s"
    if past_window_column not in df.columns:
        raise SystemExit(f"Missing required impulse feature: {past_window_column}")
    past_return_bps = pd.to_numeric(df[past_window_column], errors="coerce") * 10000.0
    candidate_mask = past_return_bps <= -float(signal["shock_bps"])

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    scores = pd.Series(np.nan, index=df.index, dtype=float)
    if int(candidate_mask.sum()) > 0:
        scores.loc[candidate_mask] = model.predict_proba(X.loc[candidate_mask])[:, 1]

    price_series = pd.to_numeric(df["fut_mid_price"], errors="coerce")
    timestamps = df.index.to_numpy()
    prices = price_series.to_numpy(dtype=float)
    score_values = scores.to_numpy(dtype=float)
    candidate_positions = np.flatnonzero(
        candidate_mask.to_numpy()
        & np.isfinite(score_values)
        & np.isfinite(prices)
        & (score_values >= float(signal["score_threshold"]))
    )

    overall_result, trade_records = simulate_params(
        timestamps=timestamps,
        prices=prices,
        score_values=score_values,
        candidate_positions=candidate_positions,
        score_threshold=float(signal["score_threshold"]),
        take_profit_bps=float(execution["take_profit_bps"]),
        stop_loss_bps=float(execution["stop_loss_bps"]),
        max_hold_seconds=int(execution["max_hold_seconds"]),
        cooldown_seconds=int(execution["cooldown_seconds"]),
        fee_bps_per_side=float(execution["fee_bps_per_side"]),
        entry_slippage_bps=float(execution["entry_slippage_bps"]),
        exit_slippage_bps=float(execution["exit_slippage_bps"]),
    )

    day_rows = summarize_days(trade_records)
    day_bootstrap = bootstrap_day_totals(day_rows, samples=args.bootstrap_samples, seed=args.seed)

    run_name = f"btc_meanrev_validation_{started_at.strftime('%Y%m%dT%H%M%S')}_v1"
    run_root = Path(args.output_root).resolve() / run_name
    report_root = run_root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)

    report = {
        "run_name": run_name,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "spec_path": str(spec_path),
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "candidate_event_count": int(len(candidate_positions)),
        "spec": spec,
        "overall_result": asdict(overall_result) if overall_result is not None else None,
        "day_rows": day_rows,
        "day_count": len(day_rows),
        "positive_day_share": (
            float(np.mean([row["total_net_bps"] > 0.0 for row in day_rows])) if day_rows else None
        ),
        "day_bootstrap": day_bootstrap,
        "trade_records_preview": make_trade_records_json_safe(trade_records[:25]),
    }

    metadata_path = report_root / "metadata.json"
    markdown_path = report_root / "report.md"
    metadata_path.write_text(json.dumps(to_json_safe(report), indent=2), encoding="utf-8")
    markdown_path.write_text(render_report(report), encoding="utf-8")
    print(json.dumps(to_json_safe(report), indent=2))


def simulate_params(
    *,
    timestamps: np.ndarray,
    prices: np.ndarray,
    score_values: np.ndarray,
    candidate_positions: np.ndarray,
    score_threshold: float,
    take_profit_bps: float,
    stop_loss_bps: float,
    max_hold_seconds: int,
    cooldown_seconds: int,
    fee_bps_per_side: float,
    entry_slippage_bps: float,
    exit_slippage_bps: float,
) -> tuple[ValidationResult | None, list[dict[str, object]]]:
    trade_records: list[dict[str, object]] = []
    next_allowed_time = None
    take_profit_exits = 0
    stop_loss_exits = 0
    timeout_exits = 0
    no_path_skips = 0

    for pos in candidate_positions:
        score = score_values[pos]
        if score < score_threshold:
            continue
        ts = timestamps[pos]
        if next_allowed_time is not None and ts < next_allowed_time:
            continue

        trade = simulate_trade(
            entry_pos=int(pos),
            timestamps=timestamps,
            prices=prices,
            take_profit_bps=take_profit_bps,
            stop_loss_bps=stop_loss_bps,
            max_hold_seconds=max_hold_seconds,
            fee_bps_per_side=fee_bps_per_side,
            entry_slippage_bps=entry_slippage_bps,
            exit_slippage_bps=exit_slippage_bps,
        )
        if trade is None:
            no_path_skips += 1
            continue

        trade["score"] = float(score)
        trade_records.append(trade)
        next_allowed_time = trade["exit_time"] + np.timedelta64(cooldown_seconds, "s")
        if trade["exit_reason"] == "take_profit":
            take_profit_exits += 1
        elif trade["exit_reason"] == "stop_loss":
            stop_loss_exits += 1
        else:
            timeout_exits += 1

    if not trade_records:
        return None, trade_records

    net_bps = np.array([float(item["net_bps"]) for item in trade_records], dtype=float)
    wins = int(np.sum(net_bps > 0.0))
    gross_wins = float(net_bps[net_bps > 0.0].sum())
    gross_losses = float(-net_bps[net_bps < 0.0].sum())
    profit_factor = None if gross_losses <= 0.0 else gross_wins / gross_losses

    validation = ValidationResult(
        trades=len(trade_records),
        wins=wins,
        win_rate=float(wins / len(trade_records)),
        total_net_bps=float(net_bps.sum()),
        avg_net_bps=float(net_bps.mean()),
        median_net_bps=float(np.median(net_bps)),
        profit_factor=profit_factor,
        take_profit_exits=take_profit_exits,
        stop_loss_exits=stop_loss_exits,
        timeout_exits=timeout_exits,
        no_path_skips=no_path_skips,
    )
    return validation, trade_records


def simulate_trade(
    *,
    entry_pos: int,
    timestamps: np.ndarray,
    prices: np.ndarray,
    take_profit_bps: float,
    stop_loss_bps: float,
    max_hold_seconds: int,
    fee_bps_per_side: float,
    entry_slippage_bps: float,
    exit_slippage_bps: float,
) -> dict[str, object] | None:
    entry_mid = float(prices[entry_pos])
    if not np.isfinite(entry_mid) or entry_mid <= 0.0:
        return None

    entry_time = timestamps[entry_pos]
    entry_price = entry_mid * (1.0 + entry_slippage_bps / 10000.0)

    last_pos = entry_pos
    exit_reason = "timeout"
    exit_mid = None
    for pos in range(entry_pos + 1, len(prices)):
        step_gap = (timestamps[pos] - timestamps[pos - 1]) / np.timedelta64(1, "s")
        if float(step_gap) > 1.5:
            break
        seconds_forward = (timestamps[pos] - entry_time) / np.timedelta64(1, "s")
        if float(seconds_forward) > max_hold_seconds:
            break
        last_pos = pos
        mid = float(prices[pos])
        if not np.isfinite(mid) or mid <= 0.0:
            continue
        gross_mid_bps = (mid / entry_price - 1.0) * 10000.0
        if gross_mid_bps >= take_profit_bps:
            exit_reason = "take_profit"
            exit_mid = mid
            break
        if gross_mid_bps <= -stop_loss_bps:
            exit_reason = "stop_loss"
            exit_mid = mid
            break

    if last_pos == entry_pos:
        return None

    if exit_mid is None:
        exit_mid = float(prices[last_pos])
    exit_time = timestamps[last_pos]
    exit_price = exit_mid * (1.0 - exit_slippage_bps / 10000.0)
    gross_bps = (exit_price / entry_price - 1.0) * 10000.0
    net_bps = gross_bps - (2.0 * fee_bps_per_side)
    return {
        "entry_time": str(entry_time),
        "exit_time": exit_time,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_bps": gross_bps,
        "net_bps": net_bps,
        "exit_reason": exit_reason,
        "hold_seconds": float((exit_time - entry_time) / np.timedelta64(1, "s")),
    }


def summarize_days(trade_records: list[dict[str, object]]) -> list[dict[str, object]]:
    if not trade_records:
        return []
    frame = pd.DataFrame.from_records(trade_records)
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True)
    frame["day"] = frame["entry_time"].dt.strftime("%Y-%m-%d")
    rows: list[dict[str, object]] = []
    for day, day_frame in frame.groupby("day"):
        net_bps = day_frame["net_bps"].to_numpy(dtype=float)
        wins = int(np.sum(net_bps > 0.0))
        gross_wins = float(net_bps[net_bps > 0.0].sum())
        gross_losses = float(-net_bps[net_bps < 0.0].sum())
        profit_factor = None if gross_losses <= 0.0 else gross_wins / gross_losses
        rows.append(
            {
                "day": day,
                "trades": int(len(day_frame)),
                "wins": wins,
                "win_rate": float(wins / len(day_frame)),
                "total_net_bps": float(net_bps.sum()),
                "avg_net_bps": float(net_bps.mean()),
                "median_net_bps": float(np.median(net_bps)),
                "profit_factor": profit_factor,
            }
        )
    return rows


def bootstrap_day_totals(day_rows: list[dict[str, object]], *, samples: int, seed: int) -> dict[str, object] | None:
    if not day_rows:
        return None
    rng = np.random.default_rng(seed)
    totals = np.array([float(row["total_net_bps"]) for row in day_rows], dtype=float)
    boot_totals = np.empty(samples, dtype=float)
    positive_shares = np.empty(samples, dtype=float)
    for idx in range(samples):
        draw = rng.choice(totals, size=len(totals), replace=True)
        boot_totals[idx] = float(draw.sum())
        positive_shares[idx] = float(np.mean(draw > 0.0))
    return {
        "samples": samples,
        "day_count": int(len(totals)),
        "mean_total_net_bps": float(np.mean(boot_totals)),
        "median_total_net_bps": float(np.median(boot_totals)),
        "p05_total_net_bps": float(np.quantile(boot_totals, 0.05)),
        "p25_total_net_bps": float(np.quantile(boot_totals, 0.25)),
        "p75_total_net_bps": float(np.quantile(boot_totals, 0.75)),
        "p95_total_net_bps": float(np.quantile(boot_totals, 0.95)),
        "mean_positive_day_share": float(np.mean(positive_shares)),
        "p05_positive_day_share": float(np.quantile(positive_shares, 0.05)),
    }


def make_trade_records_json_safe(trade_records: list[dict[str, object]]) -> list[dict[str, object]]:
    safe: list[dict[str, object]] = []
    for record in trade_records:
        converted: dict[str, object] = {}
        for key, value in record.items():
            if isinstance(value, np.generic):
                converted[key] = value.item()
            elif isinstance(value, pd.Timestamp):
                converted[key] = value.isoformat()
            else:
                converted[key] = value
        safe.append(converted)
    return safe


def to_json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [to_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def render_report(report: dict[str, object]) -> str:
    lines = [
        "# BTC Mean Reversion Frozen Validation",
        "",
        f"- Spec: `{report['spec_path']}`",
        f"- Dataset: `{report['dataset_path']}`",
        f"- Model: `{report['model_path']}`",
        f"- Candidate events: `{report['candidate_event_count']}`",
        "",
    ]
    overall = report.get("overall_result")
    if overall:
        lines.extend(
            [
                "## Overall",
                "",
                f"- Trades / win rate: `{overall['trades']}` / `{overall['win_rate']:.3f}`",
                f"- Total / avg / median net bps: `{overall['total_net_bps']:.2f}` / `{overall['avg_net_bps']:.2f}` / `{overall['median_net_bps']:.2f}`",
                f"- Profit factor: `{overall['profit_factor']}`",
                "",
            ]
        )
    lines.extend(["## By Day", ""])
    for row in report["day_rows"]:
        lines.append(
            f"- `{row['day']}` | trades `{row['trades']}` | win rate `{row['win_rate']:.3f}` | total net `{row['total_net_bps']:.2f}` bps | profit factor `{row['profit_factor']}`"
        )
    lines.append("")
    if report.get("day_bootstrap"):
        boot = report["day_bootstrap"]
        lines.extend(
            [
                "## Day-Level Bootstrap",
                "",
                f"- Day count: `{boot['day_count']}`",
                f"- Mean / median total net bps: `{boot['mean_total_net_bps']:.2f}` / `{boot['median_total_net_bps']:.2f}`",
                f"- P05 / P95 total net bps: `{boot['p05_total_net_bps']:.2f}` / `{boot['p95_total_net_bps']:.2f}`",
                f"- Mean positive-day share: `{boot['mean_positive_day_share']:.3f}`",
                f"- P05 positive-day share: `{boot['p05_positive_day_share']:.3f}`",
                "",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
