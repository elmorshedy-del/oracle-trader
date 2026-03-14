#!/usr/bin/env python3
"""
Hybrid search for BTC impulse-conditioned replay candidates.

This runner does:
1. exhaustive grid search over core structural parameters
2. Monte Carlo execution uncertainty only on the top core candidates
3. bootstrap on the best stressed configuration
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


UTC = timezone.utc
DEFAULT_OUTPUT_ROOT = Path("output/btc_multivenue_hybrid_search")


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
    parser = argparse.ArgumentParser(description="Run hybrid search for BTC impulse-conditioned replay candidates.")
    parser.add_argument("--dataset-path", required=True, help="Path to multivenue features.csv.gz")
    parser.add_argument("--model-path", required=True, help="Path to trained CatBoost impulse-conditioned model")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output root for search runs")
    parser.add_argument("--shock-window-seconds", type=int, default=5, help="Past return window used to define the impulse candidate")
    parser.add_argument("--shock-bps", type=float, default=5.0, help="Minimum absolute impulse magnitude in bps")
    parser.add_argument("--candidate-direction", choices=("down", "up"), default="down", help="Which shock direction defines candidate events")
    parser.add_argument("--trade-direction", choices=("long", "short"), default="long", help="Whether the replay enters long or short after a qualifying candidate event")
    parser.add_argument("--min-trades", type=int, default=15, help="Minimum trades required for a simulation to count")
    parser.add_argument("--top-core-count", type=int, default=20, help="Number of best core parameter sets to stress with execution Monte Carlo")
    parser.add_argument("--execution-draws", type=int, default=250, help="Execution uncertainty draws per top core configuration")
    parser.add_argument("--bootstrap-samples", type=int, default=2000, help="Bootstrap samples for the best stressed configuration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = datetime.now(UTC)
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
    if args.candidate_direction == "down":
        candidate_mask = past_return_bps <= -args.shock_bps
    else:
        candidate_mask = past_return_bps >= args.shock_bps

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

    core_results: list[SimulationResult] = []
    for params in build_core_grid():
        simulation, _ = simulate_params(
            timestamps=timestamps,
            prices=prices,
            score_values=score_values,
            candidate_positions=candidate_positions,
            score_threshold=float(params["score_threshold"]),
            take_profit_bps=float(params["take_profit_bps"]),
            stop_loss_bps=float(params["stop_loss_bps"]),
            max_hold_seconds=int(params["max_hold_seconds"]),
            cooldown_seconds=3,
            fee_bps_per_side=1.0,
            entry_slippage_bps=0.5,
            exit_slippage_bps=0.5,
            min_trades=args.min_trades,
            trade_direction=args.trade_direction,
        )
        if simulation is not None:
            core_results.append(simulation)

    ranked_core = sorted(core_results, key=lambda item: (item.total_net_bps, item.avg_net_bps, item.win_rate), reverse=True)
    top_core = ranked_core[: args.top_core_count]

    rng = np.random.default_rng(args.seed)
    stressed_results: list[dict[str, object]] = []
    best_trade_records: list[dict[str, object]] = []
    best_bootstrap = None
    if top_core:
        for core in top_core:
            stressed_results.append(
                stress_core_config(
                    core=core,
                    timestamps=timestamps,
                    prices=prices,
                    score_values=score_values,
                    candidate_positions=candidate_positions,
                    min_trades=args.min_trades,
                    draws=args.execution_draws,
                    rng=rng,
                    trade_direction=args.trade_direction,
                )
            )
        stressed_results.sort(
            key=lambda item: (
                item["mean_total_net_bps"],
                item["positive_total_share"],
                item["p05_total_net_bps"],
            ),
            reverse=True,
        )
        best_stress = stressed_results[0]
        best_trade_records = best_stress["best_trade_records"]
        best_bootstrap = bootstrap_trade_outcomes(best_trade_records, samples=args.bootstrap_samples, seed=args.seed + 1)
    else:
        best_stress = None

    run_name = f"btc_meanrev_hybrid_search_{started_at.strftime('%Y%m%dT%H%M%S')}_v1"
    run_root = Path(args.output_root).resolve() / run_name
    report_root = run_root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)

    report = {
        "run_name": run_name,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "candidate_event_count": int(len(candidate_positions)),
        "mode": "hybrid_grid_plus_execution_monte_carlo",
        "shock_window_seconds": args.shock_window_seconds,
        "shock_bps": args.shock_bps,
        "candidate_direction": args.candidate_direction,
        "trade_direction": args.trade_direction,
        "core_grid_size": len(build_core_grid()),
        "top_core_count": args.top_core_count,
        "execution_draws": args.execution_draws,
        "best_core_result": asdict(ranked_core[0]) if ranked_core else None,
        "top_core_results": [asdict(item) for item in ranked_core[:25]],
        "best_stressed_result": best_stress,
        "top_stressed_results": stressed_results[:10],
        "best_trade_count": len(best_trade_records),
        "best_trade_records_preview": make_trade_records_json_safe(best_trade_records[:20]),
        "best_bootstrap": best_bootstrap,
    }

    metadata_path = report_root / "metadata.json"
    markdown_path = report_root / "report.md"
    metadata_path.write_text(json.dumps(to_json_safe(report), indent=2), encoding="utf-8")
    markdown_path.write_text(render_report(report), encoding="utf-8")
    print(json.dumps(to_json_safe(report), indent=2))


def build_core_grid() -> list[dict[str, object]]:
    thresholds = [0.35, 0.45, 0.55, 0.65, 0.75]
    take_profits = [3.0, 4.0, 5.0, 6.0, 8.0]
    stop_losses = [3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    holds = [10, 15, 20, 30, 45, 60]
    return [
        {
            "score_threshold": float(threshold),
            "take_profit_bps": float(tp),
            "stop_loss_bps": float(sl),
            "max_hold_seconds": int(hold),
        }
        for threshold, tp, sl, hold in itertools.product(thresholds, take_profits, stop_losses, holds)
    ]


def stress_core_config(
    *,
    core: SimulationResult,
    timestamps: np.ndarray,
    prices: np.ndarray,
    score_values: np.ndarray,
    candidate_positions: np.ndarray,
    min_trades: int,
    draws: int,
    rng: np.random.Generator,
    trade_direction: str,
) -> dict[str, object]:
    stressed: list[SimulationResult] = []
    trade_records_by_idx: dict[int, list[dict[str, object]]] = {}
    for _ in range(draws):
        simulation, trade_records = simulate_params(
            timestamps=timestamps,
            prices=prices,
            score_values=score_values,
            candidate_positions=candidate_positions,
            score_threshold=core.score_threshold,
            take_profit_bps=core.take_profit_bps,
            stop_loss_bps=core.stop_loss_bps,
            max_hold_seconds=core.max_hold_seconds,
            cooldown_seconds=int(rng.choice([0, 1, 3, 5, 10, 20])),
            fee_bps_per_side=float(rng.choice([0.5, 0.75, 1.0, 1.25, 1.5])),
            entry_slippage_bps=float(rng.choice([0.0, 0.25, 0.5, 0.75, 1.0])),
            exit_slippage_bps=float(rng.choice([0.0, 0.25, 0.5, 0.75, 1.0])),
            min_trades=min_trades,
            trade_direction=trade_direction,
        )
        if simulation is not None:
            stressed.append(simulation)
            trade_records_by_idx[len(stressed) - 1] = trade_records

    if not stressed:
        return {
            "core": asdict(core),
            "draws": draws,
            "successful_draws": 0,
            "mean_total_net_bps": None,
            "median_total_net_bps": None,
            "p05_total_net_bps": None,
            "p25_total_net_bps": None,
            "p75_total_net_bps": None,
            "p95_total_net_bps": None,
            "positive_total_share": None,
            "best_draw": None,
            "best_trade_records": [],
        }

    totals = np.array([item.total_net_bps for item in stressed], dtype=float)
    positive = np.array([item.total_net_bps > 0.0 for item in stressed], dtype=bool)
    best_idx = int(np.argmax(totals))
    return {
        "core": asdict(core),
        "draws": draws,
        "successful_draws": len(stressed),
        "mean_total_net_bps": float(np.mean(totals)),
        "median_total_net_bps": float(np.median(totals)),
        "p05_total_net_bps": float(np.quantile(totals, 0.05)),
        "p25_total_net_bps": float(np.quantile(totals, 0.25)),
        "p75_total_net_bps": float(np.quantile(totals, 0.75)),
        "p95_total_net_bps": float(np.quantile(totals, 0.95)),
        "positive_total_share": float(np.mean(positive)),
        "best_draw": asdict(stressed[best_idx]),
        "best_trade_records": trade_records_by_idx.get(best_idx, []),
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
    trade_direction: str,
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
            trade_direction=trade_direction,
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
    trade_direction: str,
) -> dict[str, object] | None:
    entry_mid = float(prices[entry_pos])
    if not np.isfinite(entry_mid) or entry_mid <= 0.0:
        return None

    entry_time = timestamps[entry_pos]
    if trade_direction == "long":
        entry_price = entry_mid * (1.0 + entry_slippage_bps / 10000.0)
    else:
        entry_price = entry_mid * (1.0 - entry_slippage_bps / 10000.0)

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
        if trade_direction == "long":
            gross_mid_bps = (mid / entry_price - 1.0) * 10000.0
        else:
            gross_mid_bps = (entry_price / mid - 1.0) * 10000.0
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
    if trade_direction == "long":
        exit_price = exit_mid * (1.0 - exit_slippage_bps / 10000.0)
        gross_bps = (exit_price / entry_price - 1.0) * 10000.0
    else:
        exit_price = exit_mid * (1.0 + exit_slippage_bps / 10000.0)
        gross_bps = (entry_price / exit_price - 1.0) * 10000.0
    net_bps = gross_bps - (2.0 * fee_bps_per_side)
    return {
        "entry_time": str(entry_time),
        "exit_time": exit_time,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "trade_direction": trade_direction,
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
    candidate_direction = report.get("candidate_direction", "down")
    trade_direction = report.get("trade_direction", "long")
    lines = [
        "# BTC Impulse Replay Hybrid Search",
        "",
        f"- Dataset: `{report['dataset_path']}`",
        f"- Model: `{report['model_path']}`",
        f"- Candidate direction: `{candidate_direction}`",
        f"- Trade direction: `{trade_direction}`",
        f"- Candidate events: `{report['candidate_event_count']}`",
        f"- Core grid size: `{report['core_grid_size']}`",
        f"- Execution draws per top core: `{report['execution_draws']}`",
        "",
    ]
    best = report.get("best_stressed_result")
    if best:
        core = best["core"]
        best_draw = best["best_draw"]
        lines.extend(
            [
                "## Best Robust Configuration",
                "",
                f"- Core threshold / tp / sl / hold: `{core['score_threshold']:.4f}` / `{core['take_profit_bps']}` / `{core['stop_loss_bps']}` / `{core['max_hold_seconds']}`s",
                f"- Mean / median total net bps across execution draws: `{best['mean_total_net_bps']:.2f}` / `{best['median_total_net_bps']:.2f}`",
                f"- P05 / P95 total net bps: `{best['p05_total_net_bps']:.2f}` / `{best['p95_total_net_bps']:.2f}`",
                f"- Positive total share: `{best['positive_total_share']:.3f}`",
                f"- Best draw trades / total net / win rate: `{best_draw['trades']}` / `{best_draw['total_net_bps']:.2f}` / `{best_draw['win_rate']:.3f}`",
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
    lines.append("## Top Robust Results")
    lines.append("")
    for row in report["top_stressed_results"][:10]:
        core = row["core"]
        lines.append(
            f"- threshold `{core['score_threshold']:.4f}` | tp `{core['take_profit_bps']}` | sl `{core['stop_loss_bps']}` | hold `{core['max_hold_seconds']}`s | mean total `{row['mean_total_net_bps']:.2f}` bps | p05 `{row['p05_total_net_bps']:.2f}` | positive share `{row['positive_total_share']:.3f}`"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
