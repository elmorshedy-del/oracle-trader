"""Historical Binance spot archive helpers for the crypto pairs lane."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


MICROSECOND_EPOCH_THRESHOLD = 10**15


def load_binance_spot_klines(
    *,
    raw_root: Path,
    symbol: str,
    interval: str,
    start_date,
    end_date,
) -> pd.DataFrame | None:
    frames: list[pd.DataFrame] = []
    day = start_date
    while day <= end_date:
        zip_path = raw_root / symbol / interval / f"{symbol}-{interval}-{day:%Y-%m-%d}.zip"
        if zip_path.exists():
            with ZipFile(zip_path) as archive:
                member = archive.namelist()[0]
                with archive.open(member) as handle:
                    frame = pd.read_csv(
                        handle,
                        header=None,
                        usecols=[0, 1, 2, 3, 4, 5],
                        names=["open_time", "open", "high", "low", "close", "volume"],
                    )
            frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
            frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
            timestamp_unit = detect_binance_time_unit(frame["open_time"])
            frame["timestamp"] = pd.to_datetime(frame["open_time"], unit=timestamp_unit, utc=True)
            frames.append(frame[["timestamp", "close", "volume"]])
        day += timedelta(days=1)
    if not frames:
        return None
    merged = (
        pd.concat(frames, ignore_index=True)
        .dropna(subset=["close"])
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
    )
    return merged.set_index("timestamp")


def detect_binance_time_unit(values: pd.Series) -> str:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return "ms"
    return "us" if int(numeric.iloc[0]) >= MICROSECOND_EPOCH_THRESHOLD else "ms"
