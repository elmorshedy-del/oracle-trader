"""
Helpers for downloading historical BTC multivenue data from Tardis.

This track is separate from the existing live capture and Binance-only archive
paths. It exists to test the frozen March 13 BTC mean-reversion model on
historical multivenue days without mutating prior checkpoints.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

from tardis_dev import datasets


@dataclass(frozen=True)
class TardisDownloadRequest:
    exchange: str
    data_types: tuple[str, ...]
    symbols: tuple[str, ...]


BTC_TARDIS_REQUESTS: tuple[TardisDownloadRequest, ...] = (
    TardisDownloadRequest(
        exchange="binance-futures",
        data_types=("trades", "book_snapshot_25"),
        symbols=("BTCUSDT",),
    ),
    TardisDownloadRequest(
        exchange="binance",
        data_types=("trades",),
        symbols=("BTCUSDT",),
    ),
    TardisDownloadRequest(
        exchange="coinbase",
        data_types=("trades", "book_snapshot_25"),
        symbols=("BTC-USD",),
    ),
)


def parse_date(raw: str) -> date:
    return datetime.strptime(raw, "%Y-%m-%d").date()


def iter_dates(start_date: date, end_date: date) -> Iterable[date]:
    cursor = start_date
    while cursor <= end_date:
        yield cursor
        cursor += timedelta(days=1)


def ensure_free_date_compatibility(*, dates: Iterable[date], api_key: str) -> None:
    if api_key:
        return
    invalid = [day.isoformat() for day in dates if day.day != 1]
    if invalid:
        joined = ", ".join(invalid[:10])
        raise SystemExit(
            "Tardis free access only covers the first day of each month. "
            f"Set TARDIS_API_KEY for other dates. Invalid dates: {joined}"
        )


def download_requests(
    *,
    requests: Iterable[TardisDownloadRequest],
    start_date: date,
    end_date: date,
    output_root: Path,
    api_key: str = "",
    concurrency: int = 2,
) -> list[str]:
    output_root.mkdir(parents=True, exist_ok=True)
    downloaded_files: list[str] = []
    for request in requests:
        datasets.download(
            exchange=request.exchange,
            data_types=list(request.data_types),
            symbols=list(request.symbols),
            from_date=start_date.isoformat(),
            to_date=end_date.isoformat(),
            api_key=api_key,
            download_dir=str(output_root),
            concurrency=concurrency,
        )
        for data_type in request.data_types:
            for symbol in request.symbols:
                for day in iter_dates(start_date, end_date):
                    file_name = f"{request.exchange}_{data_type}_{day.isoformat()}_{symbol}.csv.gz"
                    path = output_root / file_name
                    if path.exists():
                        downloaded_files.append(str(path.resolve()))
    return downloaded_files


def load_tardis_api_key() -> str:
    return os.environ.get("TARDIS_API_KEY", "").strip()
