"""
Live multivenue BTC feed for the frozen mean-reversion shadow sleeve.

This feed mirrors the research capture schema:
- Binance futures BTCUSDT aggTrade/bookTicker/depth20@100ms
- Binance spot BTCUSDT aggTrade/bookTicker/depth20@100ms
- Coinbase BTC-USD level2_batch/ticker

It archives raw future data for later controlled tests, but it does not train or
update any model weights.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import pandas as pd
import websockets

from engine.btc_multivenue_shared import (
    add_cross_venue_features,
    build_binance_agg_trade_frame,
    build_binance_book_ticker_frame,
    build_binance_partial_depth_frame,
    build_coinbase_level2_frame,
    build_coinbase_ticker_frame,
    normalize_feature_frame,
)


UTC = timezone.utc
BINANCE_FUTURES_STREAM_BASE = "wss://fstream.binance.com/stream?streams="
BINANCE_SPOT_PRIMARY_STREAM_BASE = "wss://stream.binance.com:9443/stream?streams="
BINANCE_SPOT_FALLBACK_STREAM_BASE = "wss://data-stream.binance.vision/stream?streams="
COINBASE_WS_URL = "wss://ws-feed.exchange.coinbase.com"
DEFAULT_DEPTH_LEVELS = 20
MAX_BOOK_RECORDS = 12_000
MAX_DEPTH_RECORDS = 12_000
MAX_TICKER_RECORDS = 6_000
MAX_TRADE_RECORDS = 120_000
MAX_FEATURE_FRAME_ROWS = 360

logger = logging.getLogger(__name__)


@dataclass
class MultiVenueFeedSnapshot:
    ready: bool
    bucket_at: datetime | None
    feature_row: dict[str, float] | None
    fut_mid_price: float | None
    diagnostics: dict[str, Any]


class SessionArchiveWriter:
    def __init__(self, *, capture_root: Path, session_label: str):
        self.capture_root = capture_root.resolve()
        self.capture_root.mkdir(parents=True, exist_ok=True)
        self.started_at = datetime.now(UTC)
        self.session_root = self.capture_root / "sessions" / f"{self.started_at.strftime('%Y%m%d_%H%M%S')}_{session_label}"
        self.session_root.mkdir(parents=True, exist_ok=False)
        self._handles: dict[Path, TextIO] = {}
        self._counts: defaultdict[str, int] = defaultdict(int)
        self._last_write_at: str | None = None

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "session_root": str(self.session_root),
            "archive_counts": dict(self._counts),
            "archive_entries": int(sum(self._counts.values())),
            "last_archive_write_at": self._last_write_at,
        }

    def write_binance(self, *, venue: str, symbol: str, bucket: str, record: dict[str, Any]) -> None:
        self._write_record(relative_path=self._relative_path(venue=venue, symbol=symbol, bucket=bucket, captured_at=record["captured_at"]), record=record)

    def write_coinbase(self, *, product_id: str, bucket: str, record: dict[str, Any]) -> None:
        self._write_record(relative_path=self._relative_path(venue="coinbase", symbol=product_id, bucket=bucket, captured_at=record["captured_at"]), record=record)

    def close(self) -> None:
        ended_at = datetime.now(UTC).isoformat()
        summary = {
            "started_at": self.started_at.isoformat(),
            "ended_at": ended_at,
            "session_root": str(self.session_root),
            "archive_counts": dict(self._counts),
            "archive_entries": int(sum(self._counts.values())),
            "last_archive_write_at": self._last_write_at,
        }
        (self.session_root / "session_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def _write_record(self, *, relative_path: Path, record: dict[str, Any]) -> None:
        path = self.session_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = self._handles.get(path)
        if handle is None:
            handle = path.open("a", encoding="utf-8")
            self._handles[path] = handle
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        handle.flush()
        key = str(relative_path)
        self._counts[key] += 1
        self._last_write_at = record.get("captured_at")

    @staticmethod
    def _relative_path(*, venue: str, symbol: str, bucket: str, captured_at: str) -> Path:
        date_part = captured_at[:10]
        return Path(venue) / symbol / date_part / f"{bucket}.jsonl"


class BinanceVenueStream:
    def __init__(
        self,
        *,
        venue_name: str,
        symbol: str,
        ws_urls: list[str],
        writer: SessionArchiveWriter,
    ):
        self.venue_name = venue_name
        self.symbol = symbol.upper()
        self.ws_urls = ws_urls
        self.writer = writer
        self.trade_records: deque[dict[str, Any]] = deque(maxlen=MAX_TRADE_RECORDS)
        self.book_records: deque[dict[str, Any]] = deque(maxlen=MAX_BOOK_RECORDS)
        self.depth_records: deque[dict[str, Any]] = deque(maxlen=MAX_DEPTH_RECORDS)
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._stats: dict[str, Any] = {
            "connected": False,
            "reconnects": 0,
            "errors": 0,
            "last_error": "",
            "agg_trade_events": 0,
            "book_ticker_events": 0,
            "depth_events": 0,
            "last_trade_at": None,
            "last_book_ticker_at": None,
            "last_depth_at": None,
        }

    @property
    def stats(self) -> dict[str, Any]:
        return dict(self._stats)

    async def ensure_started(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name=f"{self.venue_name}-multivenue")

    async def close(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def _run(self) -> None:
        stream_path = "/".join(
            [
                f"{self.symbol.lower()}@aggTrade",
                f"{self.symbol.lower()}@bookTicker",
                f"{self.symbol.lower()}@depth{DEFAULT_DEPTH_LEVELS}@100ms",
            ]
        )
        while not self._stop.is_set():
            attempt_succeeded = False
            for base_url in self.ws_urls:
                try:
                    async with websockets.connect(
                        base_url + stream_path,
                        ping_interval=20,
                        ping_timeout=20,
                        max_queue=10000,
                    ) as websocket:
                        self._stats["connected"] = True
                        attempt_succeeded = True
                        async for raw_message in websocket:
                            if self._stop.is_set():
                                break
                            payload = json.loads(raw_message)
                            stream_name = str(payload.get("stream") or "")
                            data = payload.get("data") or {}
                            record = {
                                "captured_at": datetime.now(UTC).isoformat(),
                                "stream": stream_name,
                                "data": data,
                            }
                            self._route_record(record)
                        self._stats["connected"] = False
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._stats["connected"] = False
                    self._stats["errors"] += 1
                    self._stats["reconnects"] += 1
                    self._stats["last_error"] = str(exc)
                    logger.warning("[BTC_SHADOW] %s stream reconnect after error: %s", self.venue_name, exc)
                    await asyncio.sleep(2)
            if not attempt_succeeded:
                await asyncio.sleep(2)

    def _route_record(self, record: dict[str, Any]) -> None:
        stream_name = str(record.get("stream") or "").lower()
        bucket = "raw"
        if "@aggtrade" in stream_name:
            bucket = "aggTrade"
            self.trade_records.append(record)
            self._stats["agg_trade_events"] += 1
            self._stats["last_trade_at"] = record["captured_at"]
        elif "@bookticker" in stream_name:
            bucket = "bookTicker"
            self.book_records.append(record)
            self._stats["book_ticker_events"] += 1
            self._stats["last_book_ticker_at"] = record["captured_at"]
        elif "@depth" in stream_name:
            bucket = "depth"
            self.depth_records.append(record)
            self._stats["depth_events"] += 1
            self._stats["last_depth_at"] = record["captured_at"]
        self.writer.write_binance(venue=self.venue_name, symbol=self.symbol, bucket=bucket, record=record)
        self.writer.write_binance(venue=self.venue_name, symbol=self.symbol, bucket="raw", record=record)


class CoinbaseVenueStream:
    def __init__(
        self,
        *,
        product_id: str,
        writer: SessionArchiveWriter,
    ):
        self.product_id = product_id.upper()
        self.writer = writer
        self.level2_records: deque[dict[str, Any]] = deque(maxlen=MAX_DEPTH_RECORDS)
        self.ticker_records: deque[dict[str, Any]] = deque(maxlen=MAX_TICKER_RECORDS)
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._stats: dict[str, Any] = {
            "connected": False,
            "reconnects": 0,
            "errors": 0,
            "last_error": "",
            "level2_events": 0,
            "ticker_events": 0,
            "last_level2_at": None,
            "last_ticker_at": None,
        }

    @property
    def stats(self) -> dict[str, Any]:
        return dict(self._stats)

    async def ensure_started(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="coinbase-multivenue")

    async def close(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def _run(self) -> None:
        subscribe_payload = {
            "type": "subscribe",
            "product_ids": [self.product_id],
            "channels": ["level2_batch", "ticker", "heartbeat"],
        }
        while not self._stop.is_set():
            try:
                async with websockets.connect(
                    COINBASE_WS_URL,
                    ping_interval=20,
                    ping_timeout=20,
                    max_queue=10000,
                    max_size=None,
                ) as websocket:
                    self._stats["connected"] = True
                    await websocket.send(json.dumps(subscribe_payload))
                    async for raw_message in websocket:
                        if self._stop.is_set():
                            break
                        payload = json.loads(raw_message)
                        record = {
                            "captured_at": datetime.now(UTC).isoformat(),
                            "product_id": self.product_id,
                            "data": payload,
                        }
                        self._route_record(record)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._stats["connected"] = False
                self._stats["errors"] += 1
                self._stats["reconnects"] += 1
                self._stats["last_error"] = str(exc)
                logger.warning("[BTC_SHADOW] Coinbase reconnect after error: %s", exc)
                await asyncio.sleep(2)

    def _route_record(self, record: dict[str, Any]) -> None:
        payload = record.get("data") or {}
        message_type = str(payload.get("type") or "").lower()
        bucket = "raw"
        if message_type in {"snapshot", "l2update"}:
            bucket = "level2"
            self.level2_records.append(record)
            self._stats["level2_events"] += 1
            self._stats["last_level2_at"] = record["captured_at"]
        elif message_type == "ticker":
            bucket = "ticker"
            self.ticker_records.append(record)
            self._stats["ticker_events"] += 1
            self._stats["last_ticker_at"] = record["captured_at"]
        elif message_type == "heartbeat":
            bucket = "heartbeat"
        elif message_type == "subscriptions":
            bucket = "subscriptions"
        self.writer.write_coinbase(product_id=self.product_id, bucket=bucket, record=record)
        self.writer.write_coinbase(product_id=self.product_id, bucket="raw", record=record)


class BtcMultivenueFeatureFeed:
    def __init__(
        self,
        *,
        symbol: str,
        product_id: str,
        capture_root: Path,
        session_label: str,
        bucket_seconds: int,
        levels: int,
        warmup_buckets: int,
    ):
        self.symbol = symbol.upper()
        self.product_id = product_id.upper()
        self.bucket_seconds = bucket_seconds
        self.levels = levels
        self.warmup_buckets = warmup_buckets
        self.writer = SessionArchiveWriter(capture_root=capture_root, session_label=session_label)
        self.futures = BinanceVenueStream(
            venue_name="binance_futures",
            symbol=self.symbol,
            ws_urls=[BINANCE_FUTURES_STREAM_BASE],
            writer=self.writer,
        )
        self.spot = BinanceVenueStream(
            venue_name="binance_spot",
            symbol=self.symbol,
            ws_urls=[BINANCE_SPOT_PRIMARY_STREAM_BASE, BINANCE_SPOT_FALLBACK_STREAM_BASE],
            writer=self.writer,
        )
        self.coinbase = CoinbaseVenueStream(product_id=self.product_id, writer=self.writer)
        self._stats: dict[str, Any] = {
            "session_root": str(self.writer.session_root),
            "warmup_ready": False,
            "last_snapshot_at": None,
            "last_bucket_at": None,
            "archive_entries": 0,
            "archive_last_write_at": None,
        }

    @property
    def stats(self) -> dict[str, Any]:
        return {
            **self._stats,
            "archive": self.writer.stats,
            "futures": self.futures.stats,
            "spot": self.spot.stats,
            "coinbase": self.coinbase.stats,
        }

    async def ensure_started(self) -> None:
        await self.futures.ensure_started()
        await self.spot.ensure_started()
        await self.coinbase.ensure_started()

    async def close(self) -> None:
        await self.futures.close()
        await self.spot.close()
        await self.coinbase.close()
        self.writer.close()

    async def snapshot(self) -> MultiVenueFeedSnapshot:
        frame = self._build_feature_frame()
        self._stats["last_snapshot_at"] = datetime.now(UTC).isoformat()
        self._stats["archive_entries"] = self.writer.stats["archive_entries"]
        self._stats["archive_last_write_at"] = self.writer.stats["last_archive_write_at"]

        if frame.empty or "fut_mid_price" not in frame.columns:
            return MultiVenueFeedSnapshot(
                ready=False,
                bucket_at=None,
                feature_row=None,
                fut_mid_price=None,
                diagnostics=self.stats,
            )

        frame = frame.dropna(subset=["fut_mid_price"])
        if frame.empty:
            return MultiVenueFeedSnapshot(
                ready=False,
                bucket_at=None,
                feature_row=None,
                fut_mid_price=None,
                diagnostics=self.stats,
            )

        normalized = normalize_feature_frame(frame.tail(MAX_FEATURE_FRAME_ROWS))
        latest = normalized.iloc[-1]
        bucket_at = pd.Timestamp(normalized.index[-1]).to_pydatetime()
        self._stats["warmup_ready"] = len(normalized) >= self.warmup_buckets
        self._stats["last_bucket_at"] = bucket_at.isoformat()
        return MultiVenueFeedSnapshot(
            ready=bool(len(normalized) >= self.warmup_buckets),
            bucket_at=bucket_at,
            feature_row={key: float(latest[key]) for key in normalized.columns},
            fut_mid_price=float(latest["fut_mid_price"]),
            diagnostics=self.stats,
        )

    def _build_feature_frame(self) -> pd.DataFrame:
        pieces = [
            build_binance_book_ticker_frame(self.futures.book_records, bucket_seconds=self.bucket_seconds, prefix="fut_"),
            build_binance_agg_trade_frame(self.futures.trade_records, bucket_seconds=self.bucket_seconds, prefix="fut_"),
            build_binance_partial_depth_frame(self.futures.depth_records, bucket_seconds=self.bucket_seconds, levels=self.levels, prefix="fut_"),
            build_binance_book_ticker_frame(self.spot.book_records, bucket_seconds=self.bucket_seconds, prefix="spot_"),
            build_binance_agg_trade_frame(self.spot.trade_records, bucket_seconds=self.bucket_seconds, prefix="spot_"),
            build_binance_partial_depth_frame(self.spot.depth_records, bucket_seconds=self.bucket_seconds, levels=self.levels, prefix="spot_"),
            build_coinbase_level2_frame(self.coinbase.level2_records, bucket_seconds=self.bucket_seconds, levels=self.levels, prefix="cb_"),
            build_coinbase_ticker_frame(self.coinbase.ticker_records, bucket_seconds=self.bucket_seconds, prefix="cb_"),
        ]
        non_empty = [piece for piece in pieces if not piece.empty]
        if not non_empty:
            return pd.DataFrame()
        frame = pd.concat(non_empty, axis=1).sort_index()
        frame = frame.loc[~frame.index.duplicated(keep="last")]
        return add_cross_venue_features(frame)
