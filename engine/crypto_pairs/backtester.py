"""Backtesting helpers for the rule-based crypto pairs lane."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExecutionConfig, RiskConfig, SignalConfig
from .historical import load_binance_spot_klines
from .position_manager import PositionManager


BAR_INTERVAL_SECONDS = 3600


def load_pair_price_data(*, raw_root: Path, pair_config, start_date, end_date) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        raise FileNotFoundError(f"Missing archive data for {pair_config.token_a} or {pair_config.token_b}")
    return data_a, data_b


def prepare_pair_series(*, pair_config, data_a: pd.DataFrame, data_b: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(
        data_a[["close"]].rename(columns={"close": "price_a"}),
        data_b[["close"]].rename(columns={"close": "price_b"}),
        left_index=True,
        right_index=True,
        how="inner",
    ).dropna()
    merged["ratio"] = np.log(merged["price_a"] / merged["price_b"])
    lookback_bars = max(2, int(round(pair_config.lookback_seconds / BAR_INTERVAL_SECONDS)))
    rolling_mean = merged["ratio"].rolling(window=lookback_bars).mean()
    rolling_std = merged["ratio"].rolling(window=lookback_bars).std()
    merged["zscore"] = (merged["ratio"] - rolling_mean) / rolling_std
    return merged.dropna(subset=["zscore"]).assign(lookback_bars=lookback_bars)


def run_single_pair_backtest(
    *,
    pair_config,
    signal_config: SignalConfig,
    execution_config: ExecutionConfig,
    capital: float,
    capital_per_pair_pct: float,
    data_a: pd.DataFrame,
    data_b: pd.DataFrame,
) -> dict[str, object]:
    series = prepare_pair_series(pair_config=pair_config, data_a=data_a, data_b=data_b)
    capital_per_leg = capital * capital_per_pair_pct / 2
    position = None
    trades: list[dict[str, object]] = []

    for timestamp, row in series.iterrows():
        zscore = float(row["zscore"])
        price_a = float(row["price_a"])
        price_b = float(row["price_b"])

        if position is None:
            direction = evaluate_entry_signal(zscore, signal_config)
            if direction is not None:
                position = build_position(direction, zscore, timestamp, price_a, price_b, capital_per_leg)
            continue

        exit_reason = evaluate_exit_signal(
            direction=position["direction"],
            zscore=zscore,
            hold_seconds=(timestamp - position["entry_time"]).total_seconds(),
            signal_config=signal_config,
        )
        if exit_reason is not None:
            trades.append(close_position(position, timestamp, price_a, price_b, execution_config, exit_reason))
            position = None

    return {"summary": build_trade_summary(trades, pair_config.pair_key, int(series["lookback_bars"].iloc[0])), "trades": trades}


def run_basket_backtest(
    *,
    pair_configs: list,
    signal_config: SignalConfig,
    execution_config: ExecutionConfig,
    risk_config: RiskConfig,
    capital: float,
    data_by_pair: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> dict[str, object]:
    series_by_pair = {
        pair_config.pair_key: prepare_pair_series(pair_config=pair_config, data_a=data_by_pair[pair_config.pair_key][0], data_b=data_by_pair[pair_config.pair_key][1])
        for pair_config in pair_configs
    }
    union_timestamps = sorted({timestamp for series in series_by_pair.values() for timestamp in series.index})
    rows_by_pair = {pair_key: series.iterrows() for pair_key, series in series_by_pair.items()}
    current_rows: dict[str, tuple[pd.Timestamp, pd.Series] | None] = {pair_key: next(iterator, None) for pair_key, iterator in rows_by_pair.items()}

    position_manager = PositionManager(total_capital=capital, risk_config=risk_config)
    trades: list[dict[str, object]] = []
    per_pair_trades: dict[str, list[dict[str, object]]] = defaultdict(list)

    for timestamp in union_timestamps:
        row_map: dict[str, pd.Series] = {}
        for pair_key, current in list(current_rows.items()):
            if current is None:
                continue
            current_timestamp, row = current
            if current_timestamp == timestamp:
                row_map[pair_key] = row
                current_rows[pair_key] = next(rows_by_pair[pair_key], None)

        for pair_key in list(position_manager.positions):
            row = row_map.get(pair_key)
            if row is None:
                continue
            position = position_manager.positions[pair_key]
            exit_reason = evaluate_exit_signal(
                direction=position.direction,
                zscore=float(row["zscore"]),
                hold_seconds=(timestamp.to_pydatetime() - pd.Timestamp(position.entry_time_ms, unit="ms", tz="UTC")).total_seconds(),
                signal_config=signal_config,
            )
            if exit_reason is None:
                continue
            exit_trade = close_position(
                {
                    "direction": position.direction,
                    "entry_time": pd.Timestamp(position.entry_time_ms, unit="ms", tz="UTC"),
                    "entry_zscore": position.entry_zscore,
                    "entry_price_a": position.entry_trade["entry_price_a"],
                    "entry_price_b": position.entry_trade["entry_price_b"],
                    "capital_per_leg": position.entry_trade["capital_per_leg"],
                    "qty_a": position.entry_trade["qty_a"],
                    "qty_b": position.entry_trade["qty_b"],
                    "pair": pair_key,
                },
                timestamp.to_pydatetime(),
                float(row["price_a"]),
                float(row["price_b"]),
                execution_config,
                exit_reason,
            )
            position_manager.close_position(pair_key=pair_key, exit_trade=exit_trade)
            trades.append(exit_trade)
            per_pair_trades[pair_key].append(exit_trade)

        for pair_key, row in row_map.items():
            if pair_key in position_manager.positions:
                continue
            direction = evaluate_entry_signal(float(row["zscore"]), signal_config)
            if direction is None:
                continue
            can_open, _ = position_manager.can_open(pair_key)
            if not can_open:
                continue
            capital_per_leg = position_manager.get_position_size_per_leg()
            entry_trade = build_position(
                direction,
                float(row["zscore"]),
                timestamp.to_pydatetime(),
                float(row["price_a"]),
                float(row["price_b"]),
                capital_per_leg,
            )
            position_manager.open_position(
                pair_key=pair_key,
                direction=direction,
                entry_trade={
                    "pair": pair_key,
                    "entry_price_a": entry_trade["entry_price_a"],
                    "entry_price_b": entry_trade["entry_price_b"],
                    "capital_per_leg": entry_trade["capital_per_leg"],
                    "qty_a": entry_trade["qty_a"],
                    "qty_b": entry_trade["qty_b"],
                },
                zscore=float(row["zscore"]),
                max_hold_seconds=signal_config.max_hold_seconds,
                entry_time_ms=int(timestamp.timestamp() * 1000),
            )

    for pair_key, position in list(position_manager.positions.items()):
        series = series_by_pair[pair_key]
        last_row = series.iloc[-1]
        last_timestamp = series.index[-1].to_pydatetime()
        exit_trade = close_position(
            {
                "direction": position.direction,
                "entry_time": pd.Timestamp(position.entry_time_ms, unit="ms", tz="UTC").to_pydatetime(),
                "entry_zscore": position.entry_zscore,
                "entry_price_a": position.entry_trade["entry_price_a"],
                "entry_price_b": position.entry_trade["entry_price_b"],
                "capital_per_leg": position.entry_trade["capital_per_leg"],
                "qty_a": position.entry_trade["qty_a"],
                "qty_b": position.entry_trade["qty_b"],
                "pair": pair_key,
            },
            last_timestamp,
            float(last_row["price_a"]),
            float(last_row["price_b"]),
            execution_config,
            "end_of_backtest",
        )
        position_manager.close_position(pair_key=pair_key, exit_trade=exit_trade)
        trades.append(exit_trade)
        per_pair_trades[pair_key].append(exit_trade)

    summary = build_basket_summary(
        trades=trades,
        per_pair_trades=per_pair_trades,
        pair_configs=pair_configs,
        signal_config=signal_config,
    )
    return {"summary": summary, "trades": trades, "per_pair": summary["per_pair"]}


def evaluate_entry_signal(zscore: float, signal_config: SignalConfig) -> str | None:
    if zscore >= signal_config.entry_z:
        return "SHORT_A_LONG_B"
    if zscore <= -signal_config.entry_z:
        return "LONG_A_SHORT_B"
    return None


def evaluate_exit_signal(*, direction: str, zscore: float, hold_seconds: float, signal_config: SignalConfig) -> str | None:
    if direction == "SHORT_A_LONG_B" and zscore <= signal_config.exit_z:
        return "take_profit_mean_reversion"
    if direction == "LONG_A_SHORT_B" and zscore >= -signal_config.exit_z:
        return "take_profit_mean_reversion"
    if abs(zscore) >= signal_config.stop_z:
        return "stop_loss"
    if hold_seconds >= signal_config.max_hold_seconds:
        return "max_hold_timeout"
    return None


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
        "pair": position.get("pair"),
        "direction": position["direction"],
        "entry_time": position["entry_time"].isoformat(),
        "exit_time": timestamp.isoformat(),
        "entry_zscore": position["entry_zscore"],
        "hold_seconds": (timestamp - position["entry_time"]).total_seconds(),
        "reason": reason,
        "pnl_usd": round(total_pnl, 6),
        "pnl_bps": round(total_bps, 4),
    }


def build_trade_summary(trades: list[dict[str, object]], pair_key: str, lookback_bars: int) -> dict[str, object]:
    total_pnl = sum(float(row["pnl_usd"]) for row in trades)
    total_bps = sum(float(row["pnl_bps"]) for row in trades)
    wins = sum(1 for row in trades if float(row["pnl_usd"]) > 0)
    return {
        "pair": pair_key,
        "trades": len(trades),
        "win_rate": wins / len(trades) if trades else 0.0,
        "total_pnl_usd": round(total_pnl, 6),
        "total_pnl_bps": round(total_bps, 4),
        "lookback_bars": lookback_bars,
    }


def build_basket_summary(*, trades: list[dict[str, object]], per_pair_trades: dict[str, list[dict[str, object]]], pair_configs: list, signal_config: SignalConfig) -> dict[str, object]:
    total_pnl = sum(float(row["pnl_usd"]) for row in trades)
    total_bps = sum(float(row["pnl_bps"]) for row in trades)
    wins = sum(1 for row in trades if float(row["pnl_usd"]) > 0)
    per_pair = []
    for pair_config in pair_configs:
        pair_trades = per_pair_trades.get(pair_config.pair_key, [])
        pair_summary = build_trade_summary(pair_trades, pair_config.pair_key, max(2, int(round(pair_config.lookback_seconds / BAR_INTERVAL_SECONDS))))
        per_pair.append(pair_summary)
    return {
        "mode": "basket",
        "pairs": [pair_config.pair_key for pair_config in pair_configs],
        "trades": len(trades),
        "win_rate": wins / len(trades) if trades else 0.0,
        "total_pnl_usd": round(total_pnl, 6),
        "total_pnl_bps": round(total_bps, 4),
        "entry_z": signal_config.entry_z,
        "exit_z": signal_config.exit_z,
        "stop_z": signal_config.stop_z,
        "per_pair": per_pair,
    }
