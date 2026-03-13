#!/usr/bin/env python3
"""
Build an aligned BTC multivenue feature table from captured live sessions.

Sources:
- Binance futures BTCUSDT
- Binance spot BTCUSDT
- Coinbase BTC-USD

This is the first shared dataset for the new multi-venue continuation and
mean-reversion tracks. It is append-safe because it writes to a separate output
root and never mutates prior futures-only runs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_CAPTURE_ROOT = Path("output/btc_multivenue_capture/sessions")
DEFAULT_OUTPUT_ROOT = Path("output/btc_multivenue_dataset")
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_PRODUCT_ID = "BTC-USD"
DEFAULT_BUCKET_SECONDS = 1
DEFAULT_LEVELS = 20
UTC = timezone.utc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an aligned BTC multivenue dataset from captured sessions.")
    parser.add_argument("--capture-root", default=str(DEFAULT_CAPTURE_ROOT), help="Root containing multivenue capture sessions")
    parser.add_argument(
        "--session-dir",
        action="append",
        default=[],
        help="Specific session directory to include. Can be passed multiple times. If omitted, all sessions under capture-root are used.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Destination root for built datasets")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Binance symbol, default BTCUSDT")
    parser.add_argument("--product-id", default=DEFAULT_PRODUCT_ID, help="Coinbase product id, default BTC-USD")
    parser.add_argument("--bucket-seconds", type=int, default=DEFAULT_BUCKET_SECONDS, help="Bucket size in seconds")
    parser.add_argument("--levels", type=int, default=DEFAULT_LEVELS, help="Top depth levels to aggregate for book features")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    capture_root = Path(args.capture_root).resolve()
    output_root = Path(args.output_root).resolve()
    session_dirs = resolve_session_dirs(capture_root=capture_root, session_args=args.session_dir)
    if not session_dirs:
        raise SystemExit(f"No capture sessions found under {capture_root}")

    futures_book = load_binance_book_ticker(
        collect_paths(session_dirs, "binance_futures", args.symbol, "bookTicker.jsonl"),
        bucket_seconds=args.bucket_seconds,
        prefix="fut_",
    )
    futures_trades = load_binance_agg_trades(
        collect_paths(session_dirs, "binance_futures", args.symbol, "aggTrade.jsonl"),
        bucket_seconds=args.bucket_seconds,
        prefix="fut_",
    )
    futures_depth = load_binance_partial_depth(
        collect_paths(session_dirs, "binance_futures", args.symbol, "depth.jsonl"),
        bucket_seconds=args.bucket_seconds,
        levels=args.levels,
        prefix="fut_",
    )

    spot_book = load_binance_book_ticker(
        collect_paths(session_dirs, "binance_spot", args.symbol, "bookTicker.jsonl"),
        bucket_seconds=args.bucket_seconds,
        prefix="spot_",
    )
    spot_trades = load_binance_agg_trades(
        collect_paths(session_dirs, "binance_spot", args.symbol, "aggTrade.jsonl"),
        bucket_seconds=args.bucket_seconds,
        prefix="spot_",
    )
    spot_depth = load_binance_partial_depth(
        collect_paths(session_dirs, "binance_spot", args.symbol, "depth.jsonl"),
        bucket_seconds=args.bucket_seconds,
        levels=args.levels,
        prefix="spot_",
    )

    coinbase_book = load_coinbase_level2(
        collect_paths(session_dirs, "coinbase", args.product_id, "level2.jsonl"),
        bucket_seconds=args.bucket_seconds,
        levels=args.levels,
        prefix="cb_",
    )
    coinbase_ticker = load_coinbase_ticker(
        collect_paths(session_dirs, "coinbase", args.product_id, "ticker.jsonl"),
        bucket_seconds=args.bucket_seconds,
        prefix="cb_",
    )

    dataset = pd.concat(
        [futures_book, futures_trades, futures_depth, spot_book, spot_trades, spot_depth, coinbase_book, coinbase_ticker],
        axis=1,
    ).sort_index()
    dataset = dataset.loc[~dataset.index.duplicated(keep="last")]
    dataset = add_cross_venue_features(dataset)
    dataset = add_future_targets(dataset)
    dataset = dataset.dropna(subset=["fut_mid_price"])

    date_min = dataset.index.min()
    date_max = dataset.index.max()
    run_name = (
        f"btc_multivenue_{args.bucket_seconds}s_"
        f"{date_min:%Y%m%dT%H%M%S}_{date_max:%Y%m%dT%H%M%S}_"
        f"{len(session_dirs)}sessions_v1"
    )
    run_root = output_root / run_name
    dataset_root = run_root / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)

    dataset_path = dataset_root / "features.csv.gz"
    dataset.to_csv(dataset_path, index=True, compression="gzip")

    metadata = {
        "symbol": args.symbol,
        "product_id": args.product_id,
        "bucket_seconds": args.bucket_seconds,
        "levels": args.levels,
        "sessions": [str(path) for path in session_dirs],
        "row_count": int(len(dataset)),
        "column_count": int(dataset.shape[1]),
        "columns": list(dataset.columns),
        "start_time": date_min.isoformat(),
        "end_time": date_max.isoformat(),
        "output_path": str(dataset_path),
    }
    (dataset_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def resolve_session_dirs(*, capture_root: Path, session_args: list[str]) -> list[Path]:
    if session_args:
        session_dirs: list[Path] = []
        for raw in session_args:
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = capture_root / path
            path = path.resolve()
            if not path.exists() or not path.is_dir():
                raise SystemExit(f"Missing session directory: {path}")
            session_dirs.append(path)
        return sorted(session_dirs)
    return sorted(path for path in capture_root.iterdir() if path.is_dir())


def collect_paths(session_dirs: list[Path], venue: str, symbol: str, file_name: str) -> list[Path]:
    paths: list[Path] = []
    for session_dir in session_dirs:
        venue_root = session_dir / venue / symbol
        if not venue_root.exists():
            continue
        for date_dir in sorted(path for path in venue_root.iterdir() if path.is_dir()):
            path = date_dir / file_name
            if path.exists():
                paths.append(path)
    return paths


def iter_jsonl(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def load_binance_book_ticker(paths: list[Path], *, bucket_seconds: int, prefix: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for payload in iter_jsonl(paths):
        data = payload.get("data") or {}
        bid = _float_or_none(data.get("b"))
        ask = _float_or_none(data.get("a"))
        bid_qty = _float_or_none(data.get("B"))
        ask_qty = _float_or_none(data.get("A"))
        event_ms = data.get("E") or data.get("T")
        if None in (bid, ask, bid_qty, ask_qty) or event_ms is None:
            continue
        mid = (bid + ask) / 2.0
        spread_bps = ((ask - bid) / mid) * 10000.0 if mid > 0 else 0.0
        total_qty = bid_qty + ask_qty
        size_imbalance = ((bid_qty - ask_qty) / total_qty) if total_qty > 0 else 0.0
        records.append(
            {
                "bucket": pd.to_datetime(int(event_ms), unit="ms", utc=True).floor(f"{bucket_seconds}s"),
                f"{prefix}bid_price": bid,
                f"{prefix}ask_price": ask,
                f"{prefix}mid_price": mid,
                f"{prefix}spread_bps": spread_bps,
                f"{prefix}bid_qty": bid_qty,
                f"{prefix}ask_qty": ask_qty,
                f"{prefix}size_imbalance": size_imbalance,
            }
        )
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame.from_records(records)
    return frame.groupby("bucket").last().sort_index()


def load_binance_agg_trades(paths: list[Path], *, bucket_seconds: int, prefix: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for payload in iter_jsonl(paths):
        data = payload.get("data") or {}
        event_ms = data.get("E") or data.get("T")
        qty = _float_or_none(data.get("q"))
        price = _float_or_none(data.get("p"))
        is_buyer_maker = bool(data.get("m"))
        if None in (event_ms, qty, price):
            continue
        signed_qty = -qty if is_buyer_maker else qty
        records.append(
            {
                "bucket": pd.to_datetime(int(event_ms), unit="ms", utc=True).floor(f"{bucket_seconds}s"),
                f"{prefix}trade_count": 1.0,
                f"{prefix}trade_qty": qty,
                f"{prefix}signed_trade_qty": signed_qty,
                f"{prefix}trade_notional": qty * price,
                f"{prefix}signed_trade_notional": signed_qty * price,
                f"{prefix}last_trade_price": price,
            }
        )
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame.from_records(records)
    grouped = frame.groupby("bucket").agg(
        {
            f"{prefix}trade_count": "sum",
            f"{prefix}trade_qty": "sum",
            f"{prefix}signed_trade_qty": "sum",
            f"{prefix}trade_notional": "sum",
            f"{prefix}signed_trade_notional": "sum",
            f"{prefix}last_trade_price": "last",
        }
    )
    qty = grouped[f"{prefix}trade_qty"].replace(0.0, np.nan)
    grouped[f"{prefix}signed_trade_ratio"] = (grouped[f"{prefix}signed_trade_qty"] / qty).fillna(0.0)
    return grouped.sort_index()


def load_binance_partial_depth(paths: list[Path], *, bucket_seconds: int, levels: int, prefix: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for payload in iter_jsonl(paths):
        data = payload.get("data") or {}
        event_ms = data.get("E") or data.get("T")
        bids = data.get("b") or []
        asks = data.get("a") or []
        if event_ms is None or not bids or not asks:
            continue
        bid_prices, bid_sizes = parse_side_levels(bids, levels)
        ask_prices, ask_sizes = parse_side_levels(asks, levels)
        bid_qty = float(np.sum(bid_sizes))
        ask_qty = float(np.sum(ask_sizes))
        bid_notional = float(np.sum(np.asarray(bid_prices) * np.asarray(bid_sizes)))
        ask_notional = float(np.sum(np.asarray(ask_prices) * np.asarray(ask_sizes)))
        qty_denom = bid_qty + ask_qty
        notional_denom = bid_notional + ask_notional
        records.append(
            {
                "bucket": pd.to_datetime(int(event_ms), unit="ms", utc=True).floor(f"{bucket_seconds}s"),
                f"{prefix}depth_bid_qty": bid_qty,
                f"{prefix}depth_ask_qty": ask_qty,
                f"{prefix}depth_bid_notional": bid_notional,
                f"{prefix}depth_ask_notional": ask_notional,
                f"{prefix}depth_qty_imbalance": ((bid_qty - ask_qty) / qty_denom) if qty_denom > 0 else 0.0,
                f"{prefix}depth_notional_imbalance": ((bid_notional - ask_notional) / notional_denom) if notional_denom > 0 else 0.0,
                f"{prefix}depth_best_bid": bid_prices[0],
                f"{prefix}depth_best_ask": ask_prices[0],
            }
        )
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame.from_records(records)
    return frame.groupby("bucket").last().sort_index()


def load_coinbase_level2(paths: list[Path], *, bucket_seconds: int, levels: int, prefix: str) -> pd.DataFrame:
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    records: list[dict[str, Any]] = []

    for payload in iter_jsonl(paths):
        data = payload.get("data") or {}
        message_type = data.get("type")
        if message_type == "snapshot":
            bids = parse_coinbase_book(data.get("bids") or [])
            asks = parse_coinbase_book(data.get("asks") or [])
        elif message_type == "l2update":
            for side, price_raw, size_raw in data.get("changes") or []:
                price = float(price_raw)
                size = float(size_raw)
                book = bids if side == "buy" else asks
                if size <= 0.0:
                    book.pop(price, None)
                else:
                    book[price] = size
        else:
            continue

        if not bids or not asks:
            continue

        timestamp = data.get("time") or payload.get("captured_at")
        if not timestamp:
            continue
        bucket = pd.to_datetime(timestamp, utc=True).floor(f"{bucket_seconds}s")

        top_bids = sorted(bids.items(), key=lambda item: item[0], reverse=True)[:levels]
        top_asks = sorted(asks.items(), key=lambda item: item[0])[:levels]
        if not top_bids or not top_asks:
            continue

        best_bid = top_bids[0][0]
        best_ask = top_asks[0][0]
        if best_ask <= best_bid:
            continue
        mid = (best_bid + best_ask) / 2.0
        spread_bps = ((best_ask - best_bid) / mid) * 10000.0 if mid > 0 else 0.0
        bid_qty = float(sum(size for _, size in top_bids))
        ask_qty = float(sum(size for _, size in top_asks))
        bid_notional = float(sum(price * size for price, size in top_bids))
        ask_notional = float(sum(price * size for price, size in top_asks))
        qty_denom = bid_qty + ask_qty
        notional_denom = bid_notional + ask_notional

        records.append(
            {
                "bucket": bucket,
                f"{prefix}bid_price": best_bid,
                f"{prefix}ask_price": best_ask,
                f"{prefix}mid_price": mid,
                f"{prefix}spread_bps": spread_bps,
                f"{prefix}depth_bid_qty": bid_qty,
                f"{prefix}depth_ask_qty": ask_qty,
                f"{prefix}depth_bid_notional": bid_notional,
                f"{prefix}depth_ask_notional": ask_notional,
                f"{prefix}depth_qty_imbalance": ((bid_qty - ask_qty) / qty_denom) if qty_denom > 0 else 0.0,
                f"{prefix}depth_notional_imbalance": ((bid_notional - ask_notional) / notional_denom) if notional_denom > 0 else 0.0,
            }
        )

    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame.from_records(records)
    return frame.groupby("bucket").last().sort_index()


def parse_coinbase_book(levels: list[list[str]]) -> dict[float, float]:
    book: dict[float, float] = {}
    for price_raw, size_raw in levels:
        price = float(price_raw)
        size = float(size_raw)
        if size > 0.0:
            book[price] = size
    return book


def load_coinbase_ticker(paths: list[Path], *, bucket_seconds: int, prefix: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for payload in iter_jsonl(paths):
        data = payload.get("data") or {}
        if data.get("type") != "ticker":
            continue
        timestamp = data.get("time") or payload.get("captured_at")
        price = _float_or_none(data.get("price"))
        best_bid = _float_or_none(data.get("best_bid"))
        best_ask = _float_or_none(data.get("best_ask"))
        if not timestamp or price is None:
            continue
        record = {
            "bucket": pd.to_datetime(timestamp, utc=True).floor(f"{bucket_seconds}s"),
            f"{prefix}last_trade_price": price,
        }
        if best_bid is not None:
            record[f"{prefix}ticker_best_bid"] = best_bid
        if best_ask is not None:
            record[f"{prefix}ticker_best_ask"] = best_ask
        records.append(record)
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame.from_records(records)
    return frame.groupby("bucket").last().sort_index()


def add_cross_venue_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()

    if "fut_mid_price" in out.columns and "spot_mid_price" in out.columns:
        out["fut_spot_gap_bps"] = safe_divide(out["fut_mid_price"] - out["spot_mid_price"], out["spot_mid_price"]) * 10000.0
        out["fut_spot_gap_change_5"] = out["fut_spot_gap_bps"].diff(5)
    if "cb_mid_price" in out.columns and "spot_mid_price" in out.columns:
        out["cb_spot_gap_bps"] = safe_divide(out["cb_mid_price"] - out["spot_mid_price"], out["spot_mid_price"]) * 10000.0
        out["cb_spot_gap_change_5"] = out["cb_spot_gap_bps"].diff(5)
    if "cb_mid_price" in out.columns and "fut_mid_price" in out.columns:
        out["cb_fut_gap_bps"] = safe_divide(out["cb_mid_price"] - out["fut_mid_price"], out["fut_mid_price"]) * 10000.0
        out["cb_fut_gap_change_5"] = out["cb_fut_gap_bps"].diff(5)

    for prefix in ("fut_", "spot_", "cb_"):
        if f"{prefix}mid_price" in out.columns:
            out[f"{prefix}ret_1s"] = out[f"{prefix}mid_price"].pct_change(1)
            out[f"{prefix}ret_5s"] = out[f"{prefix}mid_price"].pct_change(5)
            out[f"{prefix}ret_30s"] = out[f"{prefix}mid_price"].pct_change(30)
        if f"{prefix}spread_bps" in out.columns:
            out[f"{prefix}spread_change_1"] = out[f"{prefix}spread_bps"].diff(1)
            out[f"{prefix}spread_change_5"] = out[f"{prefix}spread_bps"].diff(5)
        if f"{prefix}depth_qty_imbalance" in out.columns:
            out[f"{prefix}depth_qty_imbalance_change_1"] = out[f"{prefix}depth_qty_imbalance"].diff(1)
            out[f"{prefix}depth_qty_imbalance_change_5"] = out[f"{prefix}depth_qty_imbalance"].diff(5)
        if f"{prefix}signed_trade_ratio" in out.columns:
            out[f"{prefix}signed_trade_ratio_change_1"] = out[f"{prefix}signed_trade_ratio"].diff(1)
            out[f"{prefix}signed_trade_ratio_change_5"] = out[f"{prefix}signed_trade_ratio"].diff(5)

    numeric_cols = out.select_dtypes(include=["number"]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return out


def add_future_targets(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "fut_mid_price" not in out.columns:
        return out
    for horizon in (5, 10, 30, 60, 90):
        future_mid = out["fut_mid_price"].shift(-horizon)
        out[f"target_fut_ret_{horizon}s_bps"] = safe_divide(future_mid - out["fut_mid_price"], out["fut_mid_price"]) * 10000.0
    return out


def parse_side_levels(levels_data: list[list[str]], levels: int) -> tuple[list[float], list[float]]:
    prices: list[float] = []
    sizes: list[float] = []
    for level in levels_data[:levels]:
        if len(level) < 2:
            continue
        prices.append(float(level[0]))
        sizes.append(float(level[1]))
    if not prices:
        return [0.0] * levels, [0.0] * levels
    if len(prices) < levels:
        prices.extend([prices[-1]] * (levels - len(prices)))
        sizes.extend([0.0] * (levels - len(sizes)))
    return prices, sizes


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.divide(denominator.replace(0.0, np.nan))


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
