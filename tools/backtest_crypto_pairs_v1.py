#!/usr/bin/env python3
"""Backtest the V1 crypto pairs strategy on Binance spot archive data."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.crypto_pairs.config import DEFAULT_DISCOVERY_PROJECT_ROOT, ExecutionConfig, SignalConfig
from engine.crypto_pairs.discovery import build_runtime_configs, load_discovery_report, resolve_discovery_report_path
from engine.crypto_pairs.historical import load_binance_spot_klines


UTC = timezone.utc
DEFAULT_RAW_ROOT = Path("output/crypto_pairs/raw/spot_klines_1h")
DEFAULT_OUTPUT_ROOT = Path("output/crypto_pairs/backtests")
BAR_INTERVAL_SECONDS = 3600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest the crypto pairs V1 z-score strategy.")
    parser.add_argument("--discovery-report", default=None)
    parser.add_argument("--pair", default=None, help="Pair key like LINK/SOL; defaults to the best pair")
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.0)
    parser.add_argument("--stop-z", type=float, default=4.0)
    parser.add_argument("--max-hold-seconds", type=int, default=21_600)
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--capital-per-pair-pct", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    discovery_path = resolve_discovery_report_path(args.discovery_report)
    discovery_report = load_discovery_report(discovery_path)
    _, pair_configs, _ = build_runtime_configs(discovery_report, top_pairs=5)
    pair_config = select_pair(pair_configs, args.pair)
    raw_root = resolve_path(args.raw_root)
    output_root = resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    start_date = datetime.fromisoformat(str(discovery_report["start_date"])).date()
    end_date = datetime.fromisoformat(str(discovery_report["end_date"])).date()
    data_a = load_binance_spot_klines(
        raw_root=raw_root,
        symbol=pair_config.token_a,
        interval="1h",
        start_date=start_date,
        end_date=end_date,
    )
    data_b = load_binance_spot_klines(
        raw_root=raw_root,
        symbol=pair_config.token_b,
        interval="1h",
        start_date=start_date,
        end_date=end_date,
    )
    if data_a is None or data_b is None:
        raise SystemExit(f"Missing archive data for {pair_config.token_a} or {pair_config.token_b}")

    report = run_backtest(
        pair_config=pair_config,
        signal_config=SignalConfig(
            entry_z=args.entry_z,
            exit_z=args.exit_z,
            stop_z=args.stop_z,
            max_hold_seconds=args.max_hold_seconds,
        ),
        execution_config=ExecutionConfig(),
        capital=args.capital,
        capital_per_pair_pct=args.capital_per_pair_pct,
        data_a=data_a,
        data_b=data_b,
    )
    run_name = f"crypto_pairs_backtest_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}_v1"
    run_root = output_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    report_path = run_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report_json": str(report_path), "summary": report["summary"]}, indent=2))


def select_pair(pair_configs, pair_key: str | None):
    if pair_key is None:
        return pair_configs[0]
    for config in pair_configs:
        if config.pair_key == pair_key:
            return config
    raise ValueError(f"Pair {pair_key} not found in discovery report")


def run_backtest(*, pair_config, signal_config: SignalConfig, execution_config: ExecutionConfig, capital: float, capital_per_pair_pct: float, data_a: pd.DataFrame, data_b: pd.DataFrame) -> dict[str, object]:
    merged = pd.merge(
        data_a[["close"]].rename(columns={"close": "price_a"}),
        data_b[["close"]].rename(columns={"close": "price_b"}),
        left_index=True,
        right_index=True,
        how="inner",
    ).dropna()
    merged["ratio"] = np.log(merged["price_a"] / merged["price_b"])

    lookback_bars = max(2, int(round(pair_config.lookback_seconds / BAR_INTERVAL_SECONDS)))
    capital_per_leg = capital * capital_per_pair_pct / 2
    position = None
    trades: list[dict[str, object]] = []

    for index in range(lookback_bars, len(merged)):
        window = merged["ratio"].iloc[index - lookback_bars:index]
        current_ratio = float(merged["ratio"].iloc[index])
        rolling_mean = float(window.mean())
        rolling_std = float(window.std())
        if rolling_std <= 0:
            continue
        zscore = (current_ratio - rolling_mean) / rolling_std
        timestamp = merged.index[index]
        price_a = float(merged["price_a"].iloc[index])
        price_b = float(merged["price_b"].iloc[index])

        if position is None:
            if zscore >= signal_config.entry_z:
                position = build_position("SHORT_A_LONG_B", zscore, timestamp, price_a, price_b, capital_per_leg)
            elif zscore <= -signal_config.entry_z:
                position = build_position("LONG_A_SHORT_B", zscore, timestamp, price_a, price_b, capital_per_leg)
            continue

        hold_seconds = (timestamp - position["entry_time"]).total_seconds()
        should_exit = False
        reason = "hold"
        if position["direction"] == "SHORT_A_LONG_B" and zscore <= signal_config.exit_z:
            should_exit = True
            reason = "take_profit_mean_reversion"
        elif position["direction"] == "LONG_A_SHORT_B" and zscore >= -signal_config.exit_z:
            should_exit = True
            reason = "take_profit_mean_reversion"
        elif abs(zscore) >= signal_config.stop_z:
            should_exit = True
            reason = "stop_loss"
        elif hold_seconds >= signal_config.max_hold_seconds:
            should_exit = True
            reason = "max_hold_timeout"

        if should_exit:
            trades.append(close_position(position, timestamp, price_a, price_b, execution_config, reason))
            position = None

    total_pnl = sum(float(row["pnl_usd"]) for row in trades)
    total_bps = sum(float(row["pnl_bps"]) for row in trades)
    wins = sum(1 for row in trades if float(row["pnl_usd"]) > 0)
    summary = {
        "pair": pair_config.pair_key,
        "trades": len(trades),
        "win_rate": wins / len(trades) if trades else 0.0,
        "total_pnl_usd": round(total_pnl, 6),
        "total_pnl_bps": round(total_bps, 4),
        "lookback_bars": lookback_bars,
        "entry_z": signal_config.entry_z,
        "exit_z": signal_config.exit_z,
        "stop_z": signal_config.stop_z,
    }
    return {"summary": summary, "trades": trades}


def build_position(direction: str, zscore: float, timestamp, price_a: float, price_b: float, capital_per_leg: float) -> dict[str, object]:
    return {
        "direction": direction,
        "entry_time": timestamp,
        "entry_zscore": zscore,
        "entry_price_a": price_a,
        "entry_price_b": price_b,
        "capital_per_leg": capital_per_leg,
        "qty_a": capital_per_leg / price_a,
        "qty_b": capital_per_leg / price_b,
    }


def close_position(position: dict[str, object], timestamp, price_a: float, price_b: float, execution_config: ExecutionConfig, reason: str) -> dict[str, object]:
    if position["direction"] == "LONG_A_SHORT_B":
        fill_a_entry = position["entry_price_a"] * (1 + execution_config.slippage_bps / 10_000)
        fill_b_entry = position["entry_price_b"] * (1 - execution_config.slippage_bps / 10_000)
        fill_a_exit = price_a * (1 - execution_config.slippage_bps / 10_000)
        fill_b_exit = price_b * (1 + execution_config.slippage_bps / 10_000)
        pnl_a = (fill_a_exit - fill_a_entry) * position["qty_a"]
        pnl_b = (fill_b_entry - fill_b_exit) * position["qty_b"]
    else:
        fill_a_entry = position["entry_price_a"] * (1 - execution_config.slippage_bps / 10_000)
        fill_b_entry = position["entry_price_b"] * (1 + execution_config.slippage_bps / 10_000)
        fill_a_exit = price_a * (1 + execution_config.slippage_bps / 10_000)
        fill_b_exit = price_b * (1 - execution_config.slippage_bps / 10_000)
        pnl_a = (fill_a_entry - fill_a_exit) * position["qty_a"]
        pnl_b = (fill_b_exit - fill_b_entry) * position["qty_b"]

    total_fee = float(position["capital_per_leg"]) * 4 * execution_config.fee_bps / 10_000
    total_pnl = pnl_a + pnl_b - total_fee
    total_bps = total_pnl / (float(position["capital_per_leg"]) * 2) * 10_000 if position["capital_per_leg"] else 0.0
    return {
        "direction": position["direction"],
        "entry_time": position["entry_time"].isoformat(),
        "exit_time": timestamp.isoformat(),
        "entry_zscore": position["entry_zscore"],
        "hold_seconds": (timestamp - position["entry_time"]).total_seconds(),
        "reason": reason,
        "pnl_usd": round(total_pnl, 6),
        "pnl_bps": round(total_bps, 4),
    }


def resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


if __name__ == "__main__":
    main()
