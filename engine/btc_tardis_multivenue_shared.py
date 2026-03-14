"""
Shared Tardis CSV builders for the frozen BTC multivenue validation track.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from engine.btc_multivenue_shared import add_cross_venue_features, add_future_targets


def build_tardis_trade_frame(
    paths: list[Path],
    *,
    bucket_seconds: int,
    prefix: str,
) -> pd.DataFrame:
    records_by_bucket: dict[pd.Timestamp, dict[str, float]] = {}
    usecols = ["timestamp", "side", "price", "amount"]
    for path in paths:
        for chunk in pd.read_csv(path, compression="gzip", usecols=usecols, chunksize=250_000):
            if chunk.empty:
                continue
            chunk["timestamp"] = pd.to_numeric(chunk["timestamp"], errors="coerce")
            chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce")
            chunk["amount"] = pd.to_numeric(chunk["amount"], errors="coerce")
            chunk = chunk.dropna(subset=["timestamp", "price", "amount"])
            if chunk.empty:
                continue
            chunk["bucket"] = pd.to_datetime(chunk["timestamp"].astype("int64"), unit="us", utc=True).dt.floor(f"{bucket_seconds}s")
            signed_multiplier = np.where(chunk["side"].astype(str).str.lower() == "buy", 1.0, -1.0)
            chunk["signed_amount"] = chunk["amount"] * signed_multiplier
            chunk["notional"] = chunk["amount"] * chunk["price"]
            chunk["signed_notional"] = chunk["signed_amount"] * chunk["price"]
            grouped = chunk.groupby("bucket", sort=True).agg(
                trade_count=("amount", "count"),
                trade_qty=("amount", "sum"),
                signed_trade_qty=("signed_amount", "sum"),
                trade_notional=("notional", "sum"),
                signed_trade_notional=("signed_notional", "sum"),
                last_trade_price=("price", "last"),
            )
            qty = grouped["trade_qty"].replace(0.0, np.nan)
            grouped["signed_trade_ratio"] = (grouped["signed_trade_qty"] / qty).fillna(0.0)
            grouped = grouped.rename(columns={column: f"{prefix}{column}" for column in grouped.columns})
            records_by_bucket.update(grouped.to_dict(orient="index"))
    if not records_by_bucket:
        return pd.DataFrame()
    frame = pd.DataFrame.from_dict(records_by_bucket, orient="index")
    frame.index.name = "bucket"
    return frame.sort_index()


def build_tardis_snapshot_frame(
    paths: list[Path],
    *,
    bucket_seconds: int,
    prefix: str,
    levels: int,
    include_top_size_fields: bool = False,
    include_ticker_fields: bool = False,
) -> pd.DataFrame:
    records_by_bucket: dict[pd.Timestamp, dict[str, float]] = {}
    usecols = ["timestamp"]
    for level in range(levels):
        usecols.extend(
            [
                f"asks[{level}].price",
                f"asks[{level}].amount",
                f"bids[{level}].price",
                f"bids[{level}].amount",
            ]
        )

    for path in paths:
        for chunk in pd.read_csv(path, compression="gzip", usecols=usecols, chunksize=50_000):
            if chunk.empty:
                continue
            chunk["timestamp"] = pd.to_numeric(chunk["timestamp"], errors="coerce")
            chunk = chunk.dropna(subset=["timestamp"])
            if chunk.empty:
                continue
            bids_price = []
            bids_amount = []
            asks_price = []
            asks_amount = []
            for level in range(levels):
                bids_price.append(pd.to_numeric(chunk[f"bids[{level}].price"], errors="coerce").to_numpy(dtype=float))
                bids_amount.append(pd.to_numeric(chunk[f"bids[{level}].amount"], errors="coerce").fillna(0.0).to_numpy(dtype=float))
                asks_price.append(pd.to_numeric(chunk[f"asks[{level}].price"], errors="coerce").to_numpy(dtype=float))
                asks_amount.append(pd.to_numeric(chunk[f"asks[{level}].amount"], errors="coerce").fillna(0.0).to_numpy(dtype=float))

            bid_prices = np.column_stack(bids_price)
            ask_prices = np.column_stack(asks_price)
            bid_sizes = np.column_stack(bids_amount)
            ask_sizes = np.column_stack(asks_amount)

            best_bid = bid_prices[:, 0]
            best_ask = ask_prices[:, 0]
            valid = np.isfinite(best_bid) & np.isfinite(best_ask) & (best_ask > best_bid)
            if not valid.any():
                continue

            bid_qty = bid_sizes.sum(axis=1)
            ask_qty = ask_sizes.sum(axis=1)
            bid_notional = np.nansum(bid_prices * bid_sizes, axis=1)
            ask_notional = np.nansum(ask_prices * ask_sizes, axis=1)
            mid = (best_bid + best_ask) / 2.0
            spread_bps = ((best_ask - best_bid) / mid) * 10000.0

            bucket_values = pd.to_datetime(
                chunk.loc[valid, "timestamp"].to_numpy(dtype="int64"),
                unit="us",
                utc=True,
            ).floor(f"{bucket_seconds}s")
            metrics = pd.DataFrame(
                {
                    "bucket": bucket_values,
                    f"{prefix}bid_price": best_bid[valid],
                    f"{prefix}ask_price": best_ask[valid],
                    f"{prefix}mid_price": mid[valid],
                    f"{prefix}spread_bps": spread_bps[valid],
                    f"{prefix}depth_bid_qty": bid_qty[valid],
                    f"{prefix}depth_ask_qty": ask_qty[valid],
                    f"{prefix}depth_bid_notional": bid_notional[valid],
                    f"{prefix}depth_ask_notional": ask_notional[valid],
                    f"{prefix}depth_best_bid": best_bid[valid],
                    f"{prefix}depth_best_ask": best_ask[valid],
                }
            )
            qty_denom = metrics[f"{prefix}depth_bid_qty"] + metrics[f"{prefix}depth_ask_qty"]
            notional_denom = metrics[f"{prefix}depth_bid_notional"] + metrics[f"{prefix}depth_ask_notional"]
            metrics[f"{prefix}depth_qty_imbalance"] = np.where(
                qty_denom > 0.0,
                (metrics[f"{prefix}depth_bid_qty"] - metrics[f"{prefix}depth_ask_qty"]) / qty_denom,
                0.0,
            )
            metrics[f"{prefix}depth_notional_imbalance"] = np.where(
                notional_denom > 0.0,
                (metrics[f"{prefix}depth_bid_notional"] - metrics[f"{prefix}depth_ask_notional"]) / notional_denom,
                0.0,
            )
            if include_top_size_fields:
                top_bid_qty = bid_sizes[:, 0][valid]
                top_ask_qty = ask_sizes[:, 0][valid]
                total_top_qty = top_bid_qty + top_ask_qty
                metrics[f"{prefix}bid_qty"] = top_bid_qty
                metrics[f"{prefix}ask_qty"] = top_ask_qty
                metrics[f"{prefix}size_imbalance"] = np.where(
                    total_top_qty > 0.0,
                    (top_bid_qty - top_ask_qty) / total_top_qty,
                    0.0,
                )
            if include_ticker_fields:
                metrics[f"{prefix}ticker_best_bid"] = metrics[f"{prefix}bid_price"]
                metrics[f"{prefix}ticker_best_ask"] = metrics[f"{prefix}ask_price"]
            grouped = metrics.groupby("bucket", sort=True).last()
            records_by_bucket.update(grouped.to_dict(orient="index"))
    if not records_by_bucket:
        return pd.DataFrame()
    frame = pd.DataFrame.from_dict(records_by_bucket, orient="index")
    frame.index.name = "bucket"
    return frame.sort_index()


def build_tardis_multivenue_dataset(
    *,
    futures_snapshot_paths: list[Path],
    futures_trade_paths: list[Path],
    spot_trade_paths: list[Path],
    coinbase_snapshot_paths: list[Path],
    coinbase_trade_paths: list[Path],
    bucket_seconds: int,
    levels: int,
) -> pd.DataFrame:
    fut_snapshot = build_tardis_snapshot_frame(
        futures_snapshot_paths,
        bucket_seconds=bucket_seconds,
        prefix="fut_",
        levels=levels,
        include_top_size_fields=True,
    )
    fut_trades = build_tardis_trade_frame(futures_trade_paths, bucket_seconds=bucket_seconds, prefix="fut_")
    spot_trades = build_tardis_trade_frame(spot_trade_paths, bucket_seconds=bucket_seconds, prefix="spot_")
    cb_snapshot = build_tardis_snapshot_frame(
        coinbase_snapshot_paths,
        bucket_seconds=bucket_seconds,
        prefix="cb_",
        levels=levels,
        include_ticker_fields=True,
    )
    cb_trades = build_tardis_trade_frame(coinbase_trade_paths, bucket_seconds=bucket_seconds, prefix="cb_")

    dataset = pd.concat([fut_snapshot, fut_trades, spot_trades, cb_snapshot, cb_trades], axis=1).sort_index()
    dataset = dataset.loc[~dataset.index.duplicated(keep="last")]
    dataset = add_cross_venue_features(dataset)
    dataset = add_future_targets(dataset)
    dataset = dataset.dropna(subset=["fut_mid_price"])
    return dataset
