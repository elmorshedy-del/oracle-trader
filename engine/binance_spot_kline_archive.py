"""
Official Binance spot kline archive helpers for crypto pairs research.
"""

from __future__ import annotations

import ssl
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SPOT_BASE_URL = "https://data.binance.vision/data/spot/daily/klines"
DEFAULT_TIMEOUT_SECONDS = 180
DEFAULT_USER_AGENT = "oracle-trader-crypto-pairs/1.0"


@dataclass(frozen=True)
class KlineArchiveJob:
    symbol: str
    interval: str
    day: date
    target_path: Path

    @property
    def url(self) -> str:
        return archive_url(self.symbol, self.interval, self.day)


def iter_days(start_date: date, end_date: date) -> Iterable[date]:
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def archive_url(symbol: str, interval: str, day: date) -> str:
    return f"{SPOT_BASE_URL}/{symbol}/{interval}/{symbol}-{interval}-{day:%Y-%m-%d}.zip"


def download_jobs(
    jobs: Iterable[KlineArchiveJob],
    *,
    max_workers: int = 6,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    ordered_jobs = list(jobs)
    if max_workers <= 1:
        results = [_download_one(job, timeout_seconds=timeout_seconds) for job in ordered_jobs]
        return [result for result in results if result is not None]

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_download_one, job, timeout_seconds=timeout_seconds): job for job in ordered_jobs}
        for future in as_completed(future_map):
            try:
                result = future.result()
            except Exception as exc:
                job = future_map[future]
                result = {
                    "symbol": job.symbol,
                    "interval": job.interval,
                    "date": job.day.isoformat(),
                    "path": str(job.target_path),
                    "url": job.url,
                    "bytes": 0,
                    "status": "error",
                    "error": str(exc),
                }
            if result is not None:
                results.append(result)
    return sorted(results, key=lambda item: (item["symbol"], item["date"]))


def _download_one(job: KlineArchiveJob, *, timeout_seconds: int) -> dict[str, Any] | None:
    target = job.target_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return {
            "symbol": job.symbol,
            "interval": job.interval,
            "date": job.day.isoformat(),
            "path": str(target),
            "url": job.url,
            "bytes": target.stat().st_size,
            "status": "cached",
        }

    payload = _download_payload(job.url, timeout_seconds=timeout_seconds)
    if payload is None:
        return None
    target.write_bytes(payload)
    return {
        "symbol": job.symbol,
        "interval": job.interval,
        "date": job.day.isoformat(),
        "path": str(target),
        "url": job.url,
        "bytes": len(payload),
        "status": "downloaded",
    }


def _download_payload(url: str, *, timeout_seconds: int) -> bytes | None:
    try:
        return _download_with_python(url, timeout_seconds=timeout_seconds)
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    except URLError:
        return _download_with_curl(url, timeout_seconds=timeout_seconds)
    except ssl.SSLError:
        return _download_with_curl(url, timeout_seconds=timeout_seconds)


def _download_with_python(url: str, *, timeout_seconds: int) -> bytes:
    request = Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
    with urlopen(request, timeout=timeout_seconds, context=_ssl_context()) as response:
        return response.read()


def _download_with_curl(url: str, *, timeout_seconds: int) -> bytes:
    command = [
        "curl",
        "-L",
        "--fail",
        "--silent",
        "--show-error",
        "--max-time",
        str(timeout_seconds),
        url,
    ]
    return subprocess.check_output(command)


def _ssl_context() -> ssl.SSLContext | None:
    try:
        import certifi
    except Exception:
        return None
    return ssl.create_default_context(cafile=certifi.where())
