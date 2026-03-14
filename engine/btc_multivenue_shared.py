"""
Shared BTC multivenue frame builders used by both research and runtime.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


def build_binance_book_ticker_frame(
    payloads: Iterable[dict[str, Any]],
    *,
    bucket_seconds: int,
    prefix: str,
) -> pd.DataFrame:
    records_by_bucket: dict[pd.Timestamp, dict[str, Any]] = {}
    for payload in payloads:
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
        bucket = pd.to_datetime(int(event_ms), unit="ms", utc=True).floor(f"{bucket_seconds}s")
        records_by_bucket[bucket] = {
            f"{prefix}bid_price": bid,
            f"{prefix}ask_price": ask,
            f"{prefix}mid_price": mid,
            f"{prefix}spread_bps": spread_bps,
            f"{prefix}bid_qty": bid_qty,
            f"{prefix}ask_qty": ask_qty,
            f"{prefix}size_imbalance": size_imbalance,
        }
    if not records_by_bucket:
        return pd.DataFrame()
    frame = pd.DataFrame.from_dict(records_by_bucket, orient="index")
    frame.index.name = "bucket"
    return frame.sort_index()


def build_binance_agg_trade_frame(
    payloads: Iterable[dict[str, Any]],
    *,
    bucket_seconds: int,
    prefix: str,
) -> pd.DataFrame:
    buckets: dict[pd.Timestamp, dict[str, Any]] = {}
    for payload in payloads:
        data = payload.get("data") or {}
        event_ms = data.get("E") or data.get("T")
        qty = _float_or_none(data.get("q"))
        price = _float_or_none(data.get("p"))
        is_buyer_maker = bool(data.get("m"))
        if None in (event_ms, qty, price):
            continue
        signed_qty = -qty if is_buyer_maker else qty
        bucket = pd.to_datetime(int(event_ms), unit="ms", utc=True).floor(f"{bucket_seconds}s")
        entry = buckets.setdefault(
            bucket,
            {
                f"{prefix}trade_count": 0.0,
                f"{prefix}trade_qty": 0.0,
                f"{prefix}signed_trade_qty": 0.0,
                f"{prefix}trade_notional": 0.0,
                f"{prefix}signed_trade_notional": 0.0,
                f"{prefix}last_trade_price": price,
            },
        )
        entry[f"{prefix}trade_count"] += 1.0
        entry[f"{prefix}trade_qty"] += qty
        entry[f"{prefix}signed_trade_qty"] += signed_qty
        entry[f"{prefix}trade_notional"] += qty * price
        entry[f"{prefix}signed_trade_notional"] += signed_qty * price
        entry[f"{prefix}last_trade_price"] = price
    if not buckets:
        return pd.DataFrame()
    grouped = pd.DataFrame.from_dict(buckets, orient="index")
    grouped.index.name = "bucket"
    qty = grouped[f"{prefix}trade_qty"].replace(0.0, np.nan)
    grouped[f"{prefix}signed_trade_ratio"] = (grouped[f"{prefix}signed_trade_qty"] / qty).fillna(0.0)
    return grouped.sort_index()


def build_binance_partial_depth_frame(
    payloads: Iterable[dict[str, Any]],
    *,
    bucket_seconds: int,
    levels: int,
    prefix: str,
) -> pd.DataFrame:
    records_by_bucket: dict[pd.Timestamp, dict[str, Any]] = {}
    for payload in payloads:
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
        bucket = pd.to_datetime(int(event_ms), unit="ms", utc=True).floor(f"{bucket_seconds}s")
        records_by_bucket[bucket] = {
            f"{prefix}depth_bid_qty": bid_qty,
            f"{prefix}depth_ask_qty": ask_qty,
            f"{prefix}depth_bid_notional": bid_notional,
            f"{prefix}depth_ask_notional": ask_notional,
            f"{prefix}depth_qty_imbalance": ((bid_qty - ask_qty) / qty_denom) if qty_denom > 0 else 0.0,
            f"{prefix}depth_notional_imbalance": ((bid_notional - ask_notional) / notional_denom) if notional_denom > 0 else 0.0,
            f"{prefix}depth_best_bid": bid_prices[0],
            f"{prefix}depth_best_ask": ask_prices[0],
        }
    if not records_by_bucket:
        return pd.DataFrame()
    frame = pd.DataFrame.from_dict(records_by_bucket, orient="index")
    frame.index.name = "bucket"
    return frame.sort_index()


def build_coinbase_level2_frame(
    payloads: Iterable[dict[str, Any]],
    *,
    bucket_seconds: int,
    levels: int,
    prefix: str,
) -> pd.DataFrame:
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    records_by_bucket: dict[pd.Timestamp, dict[str, Any]] = {}
    active_bucket: pd.Timestamp | None = None

    def snapshot_bucket(bucket: pd.Timestamp) -> None:
        if not bids or not asks:
            return
        top_bids = sorted(bids.items(), key=lambda item: item[0], reverse=True)[:levels]
        top_asks = sorted(asks.items(), key=lambda item: item[0])[:levels]
        if not top_bids or not top_asks:
            return

        best_bid = top_bids[0][0]
        best_ask = top_asks[0][0]
        if best_ask <= best_bid:
            return
        mid = (best_bid + best_ask) / 2.0
        spread_bps = ((best_ask - best_bid) / mid) * 10000.0 if mid > 0 else 0.0
        bid_qty = float(sum(size for _, size in top_bids))
        ask_qty = float(sum(size for _, size in top_asks))
        bid_notional = float(sum(price * size for price, size in top_bids))
        ask_notional = float(sum(price * size for price, size in top_asks))
        qty_denom = bid_qty + ask_qty
        notional_denom = bid_notional + ask_notional
        records_by_bucket[bucket] = {
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

    for payload in payloads:
        data = payload.get("data") or {}
        message_type = data.get("type")
        timestamp = data.get("time") or payload.get("captured_at")
        if not timestamp:
            continue
        bucket = pd.to_datetime(timestamp, utc=True).floor(f"{bucket_seconds}s")
        if active_bucket is not None and bucket != active_bucket:
            snapshot_bucket(active_bucket)

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
        active_bucket = bucket

    if active_bucket is not None:
        snapshot_bucket(active_bucket)

    if not records_by_bucket:
        return pd.DataFrame()
    frame = pd.DataFrame.from_dict(records_by_bucket, orient="index")
    frame.index.name = "bucket"
    return frame.sort_index()


def build_coinbase_ticker_frame(
    payloads: Iterable[dict[str, Any]],
    *,
    bucket_seconds: int,
    prefix: str,
) -> pd.DataFrame:
    records_by_bucket: dict[pd.Timestamp, dict[str, Any]] = {}
    for payload in payloads:
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
            f"{prefix}last_trade_price": price,
        }
        if best_bid is not None:
            record[f"{prefix}ticker_best_bid"] = best_bid
        if best_ask is not None:
            record[f"{prefix}ticker_best_ask"] = best_ask
        bucket = pd.to_datetime(timestamp, utc=True).floor(f"{bucket_seconds}s")
        records_by_bucket[bucket] = record
    if not records_by_bucket:
        return pd.DataFrame()
    frame = pd.DataFrame.from_dict(records_by_bucket, orient="index")
    frame.index.name = "bucket"
    return frame.sort_index()


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


def normalize_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    numeric_cols = out.select_dtypes(include=["number"]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    medians = out[numeric_cols].median(numeric_only=True)
    out[numeric_cols] = out[numeric_cols].fillna(medians).fillna(0.0)
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


def parse_coinbase_book(levels: list[list[str]]) -> dict[float, float]:
    book: dict[float, float] = {}
    for price_raw, size_raw in levels:
        price = float(price_raw)
        size = float(size_raw)
        if size > 0.0:
            book[price] = size
    return book


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.divide(denominator.replace(0.0, np.nan))


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
