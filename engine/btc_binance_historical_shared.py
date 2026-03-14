"""
Shared builders for Binance-only historical BTC archive datasets.

This track is intentionally separate from the live multivenue BTC model because
the live model depends on Coinbase features that do not exist in Binance's bulk
archive.
"""

from __future__ import annotations

from datetime import date, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from engine.binance_historical_archive import iter_days
from engine.btc_multivenue_shared import add_cross_venue_features, add_future_targets


UTC = timezone.utc
SPOT_AGG_COLUMNS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
    "is_best_match",
]
HISTORICAL_DEPTH_PERCENTAGES = (-5.0, -2.0, -1.0, -0.2, 0.2, 1.0, 2.0, 5.0)
TOP_DEPTH_PERCENTAGE = 0.2


def build_binance_historical_dataset(
    *,
    raw_root: Path,
    symbol: str,
    start_date: date,
    end_date: date,
    bucket_seconds: int,
) -> pd.DataFrame:
    futures_root = raw_root / "futures_um"
    spot_root = raw_root / "spot"

    futures_trades = load_futures_agg_trades(
        futures_root / "aggTrades",
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        bucket_seconds=bucket_seconds,
        prefix="fut_",
    )
    futures_depth = load_futures_book_depth(
        futures_root / "bookDepth",
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        bucket_seconds=bucket_seconds,
        prefix="fut_",
    )
    spot_trades = load_spot_agg_trades(
        spot_root / "aggTrades",
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        bucket_seconds=bucket_seconds,
        prefix="spot_",
    )

    frames = [frame for frame in (futures_trades, futures_depth, spot_trades) if not frame.empty]
    if not frames:
        return pd.DataFrame()

    dataset = pd.concat(frames, axis=1).sort_index()
    dataset = dataset.loc[~dataset.index.duplicated(keep="last")]

    min_index = dataset.index.min()
    max_index = dataset.index.max()
    full_index = pd.date_range(min_index, max_index, freq=f"{bucket_seconds}s", tz=UTC)
    dataset = dataset.reindex(full_index)
    dataset.index.name = "bucket"

    zero_fill_columns = [
        column
        for column in dataset.columns
        if column.endswith(("trade_count", "trade_qty", "signed_trade_qty", "trade_notional", "signed_trade_notional"))
        or column.endswith(("_observed", "signed_trade_ratio"))
    ]
    if zero_fill_columns:
        dataset[zero_fill_columns] = dataset[zero_fill_columns].fillna(0.0)

    forward_fill_columns = [column for column in dataset.columns if column not in zero_fill_columns]
    if forward_fill_columns:
        dataset[forward_fill_columns] = dataset[forward_fill_columns].ffill()

    if "fut_last_trade_price" in dataset.columns:
        dataset["fut_mid_price"] = dataset["fut_last_trade_price"]
    if "spot_last_trade_price" in dataset.columns:
        dataset["spot_mid_price"] = dataset["spot_last_trade_price"]

    dataset = add_cross_venue_features(dataset)
    dataset = add_future_targets(dataset)
    dataset = dataset.dropna(subset=["fut_mid_price"])

    numeric_columns = dataset.select_dtypes(include=["number"]).columns
    dataset[numeric_columns] = dataset[numeric_columns].replace([np.inf, -np.inf], np.nan)
    return dataset


def load_futures_agg_trades(
    directory: Path,
    *,
    symbol: str,
    start_date: date,
    end_date: date,
    bucket_seconds: int,
    prefix: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for day in iter_days(start_date, end_date):
        file_path = directory / f"{symbol}-aggTrades-{day:%Y-%m-%d}.zip"
        if not file_path.exists():
            continue
        frame = pd.read_csv(file_path, compression="zip")
        frames.append(_aggregate_trade_frame(frame, bucket_seconds=bucket_seconds, prefix=prefix, timestamp_unit="ms"))
    return _combine_trade_frames(frames, bucket_seconds=bucket_seconds)


def load_spot_agg_trades(
    directory: Path,
    *,
    symbol: str,
    start_date: date,
    end_date: date,
    bucket_seconds: int,
    prefix: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for day in iter_days(start_date, end_date):
        file_path = directory / f"{symbol}-aggTrades-{day:%Y-%m-%d}.zip"
        if not file_path.exists():
            continue
        frame = pd.read_csv(file_path, compression="zip", header=None, names=SPOT_AGG_COLUMNS)
        timestamp_unit = infer_timestamp_unit(frame["transact_time"])
        frames.append(_aggregate_trade_frame(frame, bucket_seconds=bucket_seconds, prefix=prefix, timestamp_unit=timestamp_unit))
    return _combine_trade_frames(frames, bucket_seconds=bucket_seconds)


def _aggregate_trade_frame(frame: pd.DataFrame, *, bucket_seconds: int, prefix: str, timestamp_unit: str) -> pd.DataFrame:
    local = frame.copy()
    local["timestamp"] = pd.to_datetime(local["transact_time"], unit=timestamp_unit, utc=True)
    local["bucket"] = local["timestamp"].dt.floor(f"{bucket_seconds}s")
    local["quantity"] = pd.to_numeric(local["quantity"], errors="coerce")
    local["price"] = pd.to_numeric(local["price"], errors="coerce")
    local["is_buyer_maker"] = local["is_buyer_maker"].astype(str).str.lower().eq("true")
    local = local.dropna(subset=["bucket", "quantity", "price"])
    local["trade_notional"] = local["price"] * local["quantity"]
    local["signed_trade_qty"] = np.where(local["is_buyer_maker"], -local["quantity"], local["quantity"])
    local["signed_trade_notional"] = np.where(local["is_buyer_maker"], -local["trade_notional"], local["trade_notional"])

    grouped = local.groupby("bucket").agg(
        {
            "quantity": "sum",
            "trade_notional": "sum",
            "signed_trade_qty": "sum",
            "signed_trade_notional": "sum",
            "price": "last",
            "bucket": "count",
        }
    )
    grouped = grouped.rename(
        columns={
            "quantity": f"{prefix}trade_qty",
            "trade_notional": f"{prefix}trade_notional",
            "signed_trade_qty": f"{prefix}signed_trade_qty",
            "signed_trade_notional": f"{prefix}signed_trade_notional",
            "price": f"{prefix}last_trade_price",
            "bucket": f"{prefix}trade_count",
        }
    )
    qty = grouped[f"{prefix}trade_qty"].replace(0.0, np.nan)
    grouped[f"{prefix}signed_trade_ratio"] = (grouped[f"{prefix}signed_trade_qty"] / qty).fillna(0.0)
    grouped[f"{prefix}trade_observed"] = 1.0
    return grouped.sort_index()


def _combine_trade_frames(frames: list[pd.DataFrame], *, bucket_seconds: int) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames).sort_index()
    combined = combined.groupby(level=0).last()
    full_index = pd.date_range(combined.index.min(), combined.index.max(), freq=f"{bucket_seconds}s", tz=UTC)
    combined = combined.reindex(full_index)

    zero_fill_columns = [
        column
        for column in combined.columns
        if column.endswith(("trade_count", "trade_qty", "signed_trade_qty", "trade_notional", "signed_trade_notional", "trade_observed"))
    ]
    if zero_fill_columns:
        combined[zero_fill_columns] = combined[zero_fill_columns].fillna(0.0)

    last_price_columns = [column for column in combined.columns if column.endswith("last_trade_price")]
    if last_price_columns:
        combined[last_price_columns] = combined[last_price_columns].ffill()

    signed_ratio_columns = [column for column in combined.columns if column.endswith("signed_trade_ratio")]
    if signed_ratio_columns:
        combined[signed_ratio_columns] = combined[signed_ratio_columns].fillna(0.0)
    return combined


def load_futures_book_depth(
    directory: Path,
    *,
    symbol: str,
    start_date: date,
    end_date: date,
    bucket_seconds: int,
    prefix: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for day in iter_days(start_date, end_date):
        file_path = directory / f"{symbol}-bookDepth-{day:%Y-%m-%d}.zip"
        if not file_path.exists():
            continue
        frame = pd.read_csv(file_path, compression="zip")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame["bucket"] = frame["timestamp"].dt.floor(f"{bucket_seconds}s")
        frame["percentage"] = pd.to_numeric(frame["percentage"], errors="coerce")
        frame["depth"] = pd.to_numeric(frame["depth"], errors="coerce")
        frame["notional"] = pd.to_numeric(frame["notional"], errors="coerce")
        frame = frame.dropna(subset=["bucket", "percentage", "depth", "notional"])
        subset = frame[frame["percentage"].isin(HISTORICAL_DEPTH_PERCENTAGES)]
        if subset.empty:
            continue
        pivot_depth = subset.pivot_table(index="bucket", columns="percentage", values="depth", aggfunc="last")
        pivot_notional = subset.pivot_table(index="bucket", columns="percentage", values="notional", aggfunc="last")
        features = pd.DataFrame(index=pivot_depth.index.union(pivot_notional.index))

        for pct in (TOP_DEPTH_PERCENTAGE, 1.0, 2.0, 5.0):
            bid_col = -float(pct)
            ask_col = float(pct)
            bid_depth = pivot_depth.get(bid_col, pd.Series(index=features.index, dtype=float))
            ask_depth = pivot_depth.get(ask_col, pd.Series(index=features.index, dtype=float))
            bid_notional = pivot_notional.get(bid_col, pd.Series(index=features.index, dtype=float))
            ask_notional = pivot_notional.get(ask_col, pd.Series(index=features.index, dtype=float))
            suffix = pct_suffix(pct)

            features[f"{prefix}depth_bid_qty_{suffix}"] = bid_depth
            features[f"{prefix}depth_ask_qty_{suffix}"] = ask_depth
            features[f"{prefix}depth_bid_notional_{suffix}"] = bid_notional
            features[f"{prefix}depth_ask_notional_{suffix}"] = ask_notional
            qty_denom = bid_depth + ask_depth
            notional_denom = bid_notional + ask_notional
            features[f"{prefix}depth_qty_imbalance_{suffix}"] = safe_divide(bid_depth - ask_depth, qty_denom)
            features[f"{prefix}depth_notional_imbalance_{suffix}"] = safe_divide(bid_notional - ask_notional, notional_denom)

            if pct == TOP_DEPTH_PERCENTAGE:
                features[f"{prefix}depth_bid_qty"] = bid_depth
                features[f"{prefix}depth_ask_qty"] = ask_depth
                features[f"{prefix}depth_bid_notional"] = bid_notional
                features[f"{prefix}depth_ask_notional"] = ask_notional
                features[f"{prefix}depth_qty_imbalance"] = safe_divide(bid_depth - ask_depth, qty_denom)
                features[f"{prefix}depth_notional_imbalance"] = safe_divide(bid_notional - ask_notional, notional_denom)

        features[f"{prefix}depth_observed"] = 1.0
        frames.append(features.sort_index())

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames).sort_index()
    combined = combined.groupby(level=0).last()
    return combined


def infer_timestamp_unit(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return "ms"
    return "us" if float(values.max()) >= 1_000_000_000_000_000 else "ms"


def pct_suffix(pct: float) -> str:
    if abs(pct - TOP_DEPTH_PERCENTAGE) < 1e-9:
        return "0p2pct"
    if float(pct).is_integer():
        return f"{int(pct)}pct"
    return f"{str(pct).replace('.', 'p')}pct"


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.divide(denominator.replace(0.0, np.nan))
