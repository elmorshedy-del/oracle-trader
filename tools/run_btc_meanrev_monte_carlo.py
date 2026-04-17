#!/usr/bin/env python3
"""
Monte Carlo entry/exit simulation for BTC multivenue mean reversion.

This script is intentionally path-based rather than AUC-based:
- score candidate downshock events with a trained CatBoost model
- simulate long mean-reversion trades on the actual 1-second futures mid-price path
- randomize entry threshold, TP/SL, hold, cooldown, fees, and slippage
- report the best parameterizations and bootstrap their trade outcomes
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
DEFAULT_OUTPUT_ROOT = Path("output/btc_multivenue_monte_carlo")


@dataclass
class SimulationResult:
    score_threshold: float
    take_profit_bps: float
    stop_loss_bps: float
    max_hold_seconds: int
    cooldown_seconds: int
    fee_bps_per_side: float
    entry_slippage_bps: float
    exit_slippage_bps: float
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
    parser = argparse.ArgumentParser(description="Run Monte Carlo entry/exit simulation for a BTC mean-reversion model.")
    parser.add_argument("--dataset-path", required=True, help="Path to multivenue features.csv.gz")
    parser.add_argument("--model-path", required=True, help="Path to trained CatBoost mean-reversion model")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output root for Monte Carlo runs")
    parser.add_argument("--iterations", type=int, default=500, help="Number of random parameter sets to test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--shock-window-seconds", type=int, default=5, help="Past return window used to define the downshock candidate")
    parser.add_argument("--shock-bps", type=float, default=5.0, help="Minimum 5s down impulse in bps")
    parser.add_argument("--min-trades", type=int, default=15, help="Minimum trades required for a simulation to count")
    parser.add_argument("--bootstrap-samples", type=int, default=2000, help="Bootstrap samples for the best configuration")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path).resolve()
    model_path = Path(args.model_path).resolve()
    if not dataset_path.exists():
        raise SystemExit(f"Missing dataset: {dataset_path}")
    if not model_path.exists():
        raise SystemExit(f"Missing model: {model_path}")

    df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
    feature_columns = [column for column in df.columns if not column.startswith("target_")]
    X = df[feature_columns].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

    past_window_column = f"fut_ret_{args.shock_window_seconds}s"
    if past_window_column not in df.columns:
        raise SystemExit(f"Missing required impulse feature: {past_window_column}")
    past_return_bps = pd.to_numeric(df[past_window_column], errors="coerce") * 10000.0
    candidate_mask = past_return_bps <= -args.shock_bps

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    scores = pd.Series(np.nan, index=df.index, dtype=float)
    if int(candidate_mask.sum()) > 0:
        scores.loc[candidate_mask] = model.predict_proba(X.loc[candidate_mask])[:, 1]

    price_series = pd.to_numeric(df["fut_mid_price"], errors="coerce")
    timestamps = df.index.to_numpy()
    prices = price_series.to_numpy(dtype=float)
    score_values = scores.to_numpy(dtype=float)
    candidate_positions = np.flatnonzero(candidate_mask.to_numpy() & np.isfinite(score_values) & np.isfinite(prices))

    rng = np.random.default_rng(args.seed)
    simulation_results: list[SimulationResult] = []
    trade_records_by_rank: dict[int, list[dict[str, object]]] = {}
    for iteration in range(args.iterations):
        params = sample_params(rng)
        simulation, trade_records = simulate_params(
            timestamps=timestamps,
            prices=prices,
            score_values=score_values,
            candidate_positions=candidate_positions,
            min_trades=args.min_trades,
            **params,
        )
        if simulation is not None:
            simulation_results.append(simulation)
            trade_records_by_rank[len(simulation_results) - 1] = trade_records

    started_at = datetime.now(UTC)
    run_name = f"btc_meanrev_monte_carlo_{started_at.strftime('%Y%m%dT%H%M%S')}_v1"
    run_root = Path(args.output_root).resolve() / run_name
    report_root = run_root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)

    ranked = sorted(simulation_results, key=lambda item: (item.total_net_bps, item.avg_net_bps, item.win_rate), reverse=True)
    top_results = [asdict(item) for item in ranked[:25]]

    best_bootstrap = None
    best_trade_records: list[dict[str, object]] = []
    if ranked:
        best_index = simulation_results.index(ranked[0])
        best_trade_records = trade_records_by_rank.get(best_index, [])
        best_bootstrap = bootstrap_trade_outcomes(best_trade_records, samples=args.bootstrap_samples, seed=args.seed + 1)

    report = {
        "run_name": run_name,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "candidate_event_count": int(len(candidate_positions)),
        "iterations": args.iterations,
        "min_trades": args.min_trades,
        "shock_window_seconds": args.shock_window_seconds,
        "shock_bps": args.shock_bps,
        "best_result": top_results[0] if top_results else None,
        "top_results": top_results,
        "best_trade_count": len(best_trade_records),
        "best_trade_records_preview": best_trade_records[:20],
        "best_bootstrap": best_bootstrap,
    }

    metadata_path = report_root / "metadata.json"
    markdown_path = report_root / "report.md"
    metadata_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(render_report(report), encoding="utf-8")
    print(json.dumps(report, indent=2))


def sample_params(rng: np.random.Generator) -> dict[str, object]:
    return {
        "score_threshold": float(rng.uniform(0.35, 0.9)),
        "take_profit_bps": float(rng.choice([3.0, 4.0, 5.0, 6.0, 8.0])),
        "stop_loss_bps": float(rng.choice([3.0, 4.0, 5.0, 6.0, 8.0, 10.0])),
        "max_hold_seconds": int(rng.choice([10, 15, 20, 30, 45, 60])),
        "cooldown_seconds": int(rng.choice([0, 1, 3, 5, 10, 20])),
        "fee_bps_per_side": float(rng.choice([0.5, 0.75, 1.0, 1.25, 1.5])),
        "entry_slippage_bps": float(rng.choice([0.0, 0.25, 0.5, 0.75, 1.0])),
        "exit_slippage_bps": float(rng.choice([0.0, 0.25, 0.5, 0.75, 1.0])),
    }


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
    min_trades: int,
) -> tuple[SimulationResult | None, list[dict[str, object]]]:
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

    if len(trade_records) < min_trades:
        return None, trade_records

    net_bps = np.array([float(item["net_bps"]) for item in trade_records], dtype=float)
    wins = int(np.sum(net_bps > 0.0))
    gross_wins = float(net_bps[net_bps > 0.0].sum())
    gross_losses = float(-net_bps[net_bps < 0.0].sum())
    profit_factor = None if gross_losses <= 0.0 else gross_wins / gross_losses

    simulation = SimulationResult(
        score_threshold=score_threshold,
        take_profit_bps=take_profit_bps,
        stop_loss_bps=stop_loss_bps,
        max_hold_seconds=max_hold_seconds,
        cooldown_seconds=cooldown_seconds,
        fee_bps_per_side=fee_bps_per_side,
        entry_slippage_bps=entry_slippage_bps,
        exit_slippage_bps=exit_slippage_bps,
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
    return simulation, trade_records


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


def bootstrap_trade_outcomes(trade_records: list[dict[str, object]], *, samples: int, seed: int) -> dict[str, object] | None:
    if not trade_records:
        return None
    rng = np.random.default_rng(seed)
    outcomes = np.array([float(item["net_bps"]) for item in trade_records], dtype=float)
    totals = np.empty(samples, dtype=float)
    for idx in range(samples):
        draw = rng.choice(outcomes, size=len(outcomes), replace=True)
        totals[idx] = float(draw.sum())
    return {
        "samples": samples,
        "trade_count": int(len(outcomes)),
        "mean_total_net_bps": float(np.mean(totals)),
        "median_total_net_bps": float(np.median(totals)),
        "p05_total_net_bps": float(np.quantile(totals, 0.05)),
        "p25_total_net_bps": float(np.quantile(totals, 0.25)),
        "p75_total_net_bps": float(np.quantile(totals, 0.75)),
        "p95_total_net_bps": float(np.quantile(totals, 0.95)),
        "positive_total_share": float(np.mean(totals > 0.0)),
    }


def render_report(report: dict[str, object]) -> str:
    lines = [
        "# BTC Downshock Mean Reversion Monte Carlo",
        "",
        f"- Dataset: `{report['dataset_path']}`",
        f"- Model: `{report['model_path']}`",
        f"- Candidate events: `{report['candidate_event_count']}`",
        f"- Monte Carlo iterations: `{report['iterations']}`",
        "",
    ]
    best = report.get("best_result")
    if best:
        lines.extend(
            [
                "## Best Configuration",
                "",
                f"- Score threshold: `{best['score_threshold']:.4f}`",
                f"- Take profit: `{best['take_profit_bps']}` bps",
                f"- Stop loss: `{best['stop_loss_bps']}` bps",
                f"- Max hold: `{best['max_hold_seconds']}` s",
                f"- Cooldown: `{best['cooldown_seconds']}` s",
                f"- Fee / side: `{best['fee_bps_per_side']}` bps",
                f"- Entry / exit slippage: `{best['entry_slippage_bps']}` / `{best['exit_slippage_bps']}` bps",
                f"- Trades: `{best['trades']}`",
                f"- Win rate: `{best['win_rate']:.3f}`",
                f"- Total / avg net bps: `{best['total_net_bps']:.2f}` / `{best['avg_net_bps']:.2f}`",
                "",
            ]
        )
    bootstrap = report.get("best_bootstrap")
    if bootstrap:
        lines.extend(
            [
                "## Bootstrap",
                "",
                f"- Positive total share: `{bootstrap['positive_total_share']:.3f}`",
                f"- Mean / median total net bps: `{bootstrap['mean_total_net_bps']:.2f}` / `{bootstrap['median_total_net_bps']:.2f}`",
                f"- P05 / P95 total net bps: `{bootstrap['p05_total_net_bps']:.2f}` / `{bootstrap['p95_total_net_bps']:.2f}`",
                "",
            ]
        )
    lines.append("## Top Results")
    lines.append("")
    for row in report["top_results"][:10]:
        lines.append(
            f"- threshold `{row['score_threshold']:.4f}` | tp `{row['take_profit_bps']}` | sl `{row['stop_loss_bps']}` | hold `{row['max_hold_seconds']}`s | trades `{row['trades']}` | total net `{row['total_net_bps']:.2f}` bps | win `{row['win_rate']:.3f}` | pf `{row['profit_factor']}`"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
