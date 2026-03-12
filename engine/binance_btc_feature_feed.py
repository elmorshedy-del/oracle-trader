"""
Live BTC futures feature feed for the legacy comparison-book ML sleeve.

The feed keeps a small rolling window of Binance USD-M BTCUSDT market data and
rebuilds the same 5-second feature family the frozen impulse model was trained
on:
- aggTrade flow
- 1/2/5% depth summary from live order book snapshots
- crowding / open-interest metrics
- funding
"""

from __future__ import annotations

import asyncio
import json
import logging
import warnings
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import websockets
from pandas.errors import PerformanceWarning

UTC = timezone.utc
BINANCE_FUTURES_REST = "https://fapi.binance.com"
BINANCE_FUTURES_WS = "wss://fstream.binance.com/ws"
FEATURE_DEPTH_LEVELS = (1, 2, 5)
LOG_RETENTION_MINUTES = 40
DEPTH_COLUMNS = [
    *(f"depth_bid_{level}pct" for level in FEATURE_DEPTH_LEVELS),
    *(f"depth_ask_{level}pct" for level in FEATURE_DEPTH_LEVELS),
    *(f"notional_bid_{level}pct" for level in FEATURE_DEPTH_LEVELS),
    *(f"notional_ask_{level}pct" for level in FEATURE_DEPTH_LEVELS),
    *(f"depth_imbalance_{level}pct" for level in FEATURE_DEPTH_LEVELS),
    *(f"notional_imbalance_{level}pct" for level in FEATURE_DEPTH_LEVELS),
]
METRICS_COLUMNS = [
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
]
FUNDING_COLUMNS = [
    "funding_interval_hours",
    "last_funding_rate",
]

logger = logging.getLogger(__name__)


@dataclass
class FeedSnapshot:
    ready: bool
    feature_row: dict[str, float] | None
    last_price: float | None
    last_bucket_at: str | None
    source_fresh_score: float
    effective_source_fresh_score: float
    long_candidate: bool
    short_candidate: bool
    diagnostics: dict[str, Any]


class BinanceBtcFeatureFeed:
    def __init__(
        self,
        *,
        symbol: str,
        bucket_seconds: int,
        horizon_seconds: int,
        cost_bps: float,
        min_signed_ratio: float,
        min_depth_imbalance: float,
        min_trade_z: float,
        min_directional_efficiency: float,
        warmup_buckets: int,
        depth_poll_seconds: int,
        metrics_poll_seconds: int,
        funding_poll_seconds: int,
        book_ticker_enabled: bool,
        max_trade_age_buckets: int,
        max_depth_age_buckets: int,
        max_metrics_age_buckets: int,
        max_funding_age_buckets: int,
        log_path: Path,
    ):
        self.symbol = symbol.upper()
        self.bucket_seconds = bucket_seconds
        self.horizon_seconds = horizon_seconds
        self.cost_bps = cost_bps
        self.min_signed_ratio = min_signed_ratio
        self.min_depth_imbalance = min_depth_imbalance
        self.min_trade_z = min_trade_z
        self.min_directional_efficiency = min_directional_efficiency
        self.warmup_buckets = warmup_buckets
        self.depth_poll_seconds = depth_poll_seconds
        self.metrics_poll_seconds = metrics_poll_seconds
        self.funding_poll_seconds = funding_poll_seconds
        self.book_ticker_enabled = book_ticker_enabled
        self.max_trade_age_buckets = max_trade_age_buckets
        self.max_depth_age_buckets = max_depth_age_buckets
        self.max_metrics_age_buckets = max_metrics_age_buckets
        self.max_funding_age_buckets = max_funding_age_buckets
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.client = httpx.AsyncClient(timeout=10.0, headers={"User-Agent": "oracle-trader/1.0"})
        self.trade_events: deque[dict[str, Any]] = deque(maxlen=120_000)
        self.depth_snapshots: deque[dict[str, Any]] = deque(maxlen=2_000)
        self._latest_book_ticker: dict[str, Any] = {}
        self.metrics_snapshots: deque[dict[str, Any]] = deque(maxlen=400)
        self.funding_snapshots: deque[dict[str, Any]] = deque(maxlen=120)
        self._tasks: list[asyncio.Task] = []
        self._started = False
        self._stop = asyncio.Event()
        self._last_price: float | None = None
        self._stats: dict[str, Any] = {
            "stream_connected": False,
            "depth_stream_connected": False,
            "book_ticker_connected": False,
            "metrics_supported": True,
            "funding_supported": True,
            "supported_source_count": 4,
            "reconnects": 0,
            "feed_errors": 0,
            "last_error": "",
            "trade_events": 0,
            "depth_snapshots": 0,
            "book_ticker_updates": 0,
            "metric_updates": 0,
            "funding_updates": 0,
            "last_trade_at": None,
            "last_depth_at": None,
            "last_book_ticker_at": None,
            "last_metrics_at": None,
            "last_funding_at": None,
            "last_snapshot_at": None,
            "last_source_fresh_score": 0.0,
            "last_effective_source_fresh_score": 0.0,
            "warmup_ready": False,
            "fast_lane_ready": False,
            "last_best_bid": 0.0,
            "last_best_ask": 0.0,
            "last_inside_spread_bps": 0.0,
            "last_inside_imbalance": 0.0,
            "last_microprice_gap_bps": 0.0,
            "log_entries": 0,
            "last_log_at": None,
        }

    @property
    def stats(self) -> dict[str, Any]:
        return dict(self._stats)

    async def ensure_started(self) -> None:
        if self._started:
            return
        self._started = True
        self._tasks = [
            asyncio.create_task(self._agg_trade_loop(), name="btc-agg-trade"),
            asyncio.create_task(self._depth_loop(), name="btc-depth"),
            *( [asyncio.create_task(self._book_ticker_loop(), name="btc-book-ticker")] if self.book_ticker_enabled else [] ),
            asyncio.create_task(self._poll_snapshots_loop(), name="btc-rest-poller"),
        ]

    async def close(self) -> None:
        self._stop.set()
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        await self.client.aclose()

    async def snapshot(self) -> FeedSnapshot:
        frame = self._build_live_feature_frame()
        if frame.empty:
            return FeedSnapshot(
                ready=False,
                feature_row=None,
                last_price=self._last_price,
                last_bucket_at=None,
                source_fresh_score=0.0,
                effective_source_fresh_score=0.0,
                long_candidate=False,
                short_candidate=False,
                diagnostics=self.stats,
            )

        latest = frame.iloc[-1]
        effective_fresh = self._effective_source_fresh_score(latest)
        self._stats["last_snapshot_at"] = datetime.now(UTC).isoformat()
        self._stats["last_source_fresh_score"] = float(latest.get("source_fresh_score", 0.0))
        self._stats["last_effective_source_fresh_score"] = effective_fresh
        self._stats["warmup_ready"] = len(frame) >= self.warmup_buckets
        self._stats["supported_source_count"] = self._supported_source_count()
        self._stats["fast_lane_ready"] = bool(
            self._stats.get("stream_connected")
            and self._stats.get("depth_stream_connected")
            and (self._stats.get("book_ticker_connected") or not self.book_ticker_enabled)
            and self._stats.get("warmup_ready")
        )

        feature_row = {key: float(latest[key]) for key in frame.columns if key != "timestamp"}
        return FeedSnapshot(
            ready=bool(len(frame) >= self.warmup_buckets),
            feature_row=feature_row,
            last_price=float(latest["price_last"]) if pd.notna(latest["price_last"]) else self._last_price,
            last_bucket_at=str(latest["timestamp"]),
            source_fresh_score=float(latest.get("source_fresh_score", 0.0)),
            effective_source_fresh_score=effective_fresh,
            long_candidate=bool(latest.get("long_impulse_candidate", 0.0) >= 1.0),
            short_candidate=bool(latest.get("short_impulse_candidate", 0.0) >= 1.0),
            diagnostics=self.stats,
        )

    async def _agg_trade_loop(self) -> None:
        stream = f"{self.symbol.lower()}@aggTrade"
        url = f"{BINANCE_FUTURES_WS}/{stream}"
        while not self._stop.is_set():
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20, max_queue=20000) as socket:
                    self._stats["stream_connected"] = True
                    async for raw_message in socket:
                        if self._stop.is_set():
                            break
                        payload = json.loads(raw_message)
                        ts = pd.to_datetime(int(payload["E"]), unit="ms", utc=True)
                        price = float(payload["p"])
                        quantity = float(payload["q"])
                        is_buyer_maker = bool(payload["m"])
                        quote_qty = price * quantity
                        self.trade_events.append(
                            {
                                "timestamp": ts,
                                "price": price,
                                "quantity": quantity,
                                "quote_qty": quote_qty,
                                "buy_qty": 0.0 if is_buyer_maker else quantity,
                                "sell_qty": quantity if is_buyer_maker else 0.0,
                                "buy_quote_qty": 0.0 if is_buyer_maker else quote_qty,
                                "sell_quote_qty": quote_qty if is_buyer_maker else 0.0,
                                "signed_qty": -quantity if is_buyer_maker else quantity,
                                "signed_quote_qty": -quote_qty if is_buyer_maker else quote_qty,
                                "agg_trade_id": int(payload["a"]),
                            }
                        )
                        self._last_price = price
                        self._stats["trade_events"] += 1
                        self._stats["last_trade_at"] = ts.isoformat()
                        self._trim_old_buffers()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._stats["stream_connected"] = False
                self._stats["feed_errors"] += 1
                self._stats["reconnects"] += 1
                self._stats["last_error"] = f"agg_trade:{type(exc).__name__}"
                logger.warning("[BTC_ML] aggTrade reconnect after error: %s", exc)
                await asyncio.sleep(2)

    async def _poll_snapshots_loop(self) -> None:
        next_metrics = 0.0
        next_funding = 0.0
        loop = asyncio.get_running_loop()
        while not self._stop.is_set():
            try:
                now = loop.time()
                calls = []
                if now >= next_metrics:
                    calls.append(self._fetch_metrics_snapshot())
                    next_metrics = now + self.metrics_poll_seconds
                if now >= next_funding:
                    calls.append(self._fetch_funding_snapshot())
                    next_funding = now + self.funding_poll_seconds
                if calls:
                    await asyncio.gather(*calls)
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._stats["feed_errors"] += 1
                self._stats["last_error"] = f"poll:{type(exc).__name__}"
                logger.warning("[BTC_ML] polling error: %s", exc)
                await asyncio.sleep(2)

    async def _depth_loop(self) -> None:
        stream = f"{self.symbol.lower()}@depth20@100ms"
        url = f"{BINANCE_FUTURES_WS}/{stream}"
        while not self._stop.is_set():
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20, max_queue=10000) as socket:
                    self._stats["depth_stream_connected"] = True
                    async for raw_message in socket:
                        if self._stop.is_set():
                            break
                        payload = json.loads(raw_message)
                        recorded_at = pd.to_datetime(int(payload.get("E") or payload.get("T") or 0), unit="ms", utc=True)
                        if pd.isna(recorded_at):
                            recorded_at = datetime.now(UTC)
                        snapshot = self._compute_depth_snapshot(payload, recorded_at)
                        self.depth_snapshots.append(snapshot)
                        self._stats["depth_snapshots"] += 1
                        self._stats["last_depth_at"] = recorded_at.isoformat()
                        self._trim_old_buffers()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._stats["depth_stream_connected"] = False
                self._stats["feed_errors"] += 1
                self._stats["reconnects"] += 1
                self._stats["last_error"] = f"depth:{type(exc).__name__}"
                logger.warning("[BTC_ML] depth reconnect after error: %s", exc)
                await asyncio.sleep(2)

    async def _book_ticker_loop(self) -> None:
        stream = f"{self.symbol.lower()}@bookTicker"
        url = f"{BINANCE_FUTURES_WS}/{stream}"
        while not self._stop.is_set():
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20, max_queue=10000) as socket:
                    self._stats["book_ticker_connected"] = True
                    async for raw_message in socket:
                        if self._stop.is_set():
                            break
                        payload = json.loads(raw_message)
                        recorded_at = pd.to_datetime(int(payload.get("E") or 0), unit="ms", utc=True)
                        if pd.isna(recorded_at):
                            recorded_at = datetime.now(UTC)
                        snapshot = self._compute_book_ticker_snapshot(payload, recorded_at)
                        self._latest_book_ticker = snapshot
                        self._stats["book_ticker_updates"] += 1
                        self._stats["last_book_ticker_at"] = recorded_at.isoformat()
                        self._stats["last_best_bid"] = snapshot["best_bid"]
                        self._stats["last_best_ask"] = snapshot["best_ask"]
                        self._stats["last_inside_spread_bps"] = snapshot["inside_spread_bps"]
                        self._stats["last_inside_imbalance"] = snapshot["inside_book_imbalance"]
                        self._stats["last_microprice_gap_bps"] = snapshot["microprice_gap_bps"]
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._stats["book_ticker_connected"] = False
                self._stats["feed_errors"] += 1
                self._stats["reconnects"] += 1
                self._stats["last_error"] = f"book_ticker:{type(exc).__name__}"
                logger.warning("[BTC_ML] bookTicker reconnect after error: %s", exc)
                await asyncio.sleep(2)

    async def _fetch_metrics_snapshot(self) -> None:
        recorded_at = datetime.now(UTC)
        results = await asyncio.gather(
            self._get_json("/futures/data/openInterestHist", {"symbol": self.symbol, "period": "5m", "limit": 1}),
            self._get_json("/futures/data/topLongShortAccountRatio", {"symbol": self.symbol, "period": "5m", "limit": 1}),
            self._get_json("/futures/data/topLongShortPositionRatio", {"symbol": self.symbol, "period": "5m", "limit": 1}),
            self._get_json("/futures/data/globalLongShortAccountRatio", {"symbol": self.symbol, "period": "5m", "limit": 1}),
            self._get_json("/futures/data/takerlongshortRatio", {"symbol": self.symbol, "period": "5m", "limit": 1}),
            return_exceptions=True,
        )
        open_interest_hist, top_account, top_position, global_account, taker = [
            {} if isinstance(result, Exception) else result
            for result in results
        ]
        failures = [result for result in results if isinstance(result, Exception)]
        if failures:
            failure_names = [
                type(result).__name__
                for result in results
                if isinstance(result, Exception)
            ]
            self._stats["last_error"] = f"metrics_partial:{'/'.join(failure_names[:3])}"
            if (
                len(failures) == len(results)
                and all(self._is_restricted_location_error(result) for result in failures)
            ):
                self._stats["metrics_supported"] = False
                self._stats["last_error"] = "metrics_restricted"
        record = {
            "timestamp": recorded_at,
            "sum_open_interest": self._last_float(open_interest_hist, "sumOpenInterest", "sum_open_interest"),
            "sum_open_interest_value": self._last_float(open_interest_hist, "sumOpenInterestValue", "sum_open_interest_value"),
            "count_toptrader_long_short_ratio": self._last_float(top_account, "longShortRatio", "count_toptrader_long_short_ratio"),
            "sum_toptrader_long_short_ratio": self._last_float(top_position, "longShortRatio", "sum_toptrader_long_short_ratio"),
            "count_long_short_ratio": self._last_float(global_account, "longShortRatio", "count_long_short_ratio"),
            "sum_taker_long_short_vol_ratio": self._last_float(taker, "buySellRatio", "longShortRatio", "sum_taker_long_short_vol_ratio"),
            "metrics_observed": 1.0 if any(not isinstance(result, Exception) for result in results) else 0.0,
        }
        self.metrics_snapshots.append(record)
        self._stats["metric_updates"] += 1
        self._stats["last_metrics_at"] = recorded_at.isoformat()
        self._trim_old_buffers()

    async def _fetch_funding_snapshot(self) -> None:
        recorded_at = datetime.now(UTC)
        try:
            funding = await self._get_json("/fapi/v1/fundingRate", {"symbol": self.symbol, "limit": 1})
        except Exception as exc:
            funding = []
            if self._is_restricted_location_error(exc):
                self._stats["funding_supported"] = False
                self._stats["last_error"] = "funding_restricted"
            else:
                self._stats["last_error"] = f"funding_partial:{type(exc).__name__}"
        latest = funding[-1] if isinstance(funding, list) and funding else {}
        record = {
            "timestamp": recorded_at,
            "funding_interval_hours": float(latest.get("fundingIntervalHours") or 8.0),
            "last_funding_rate": float(latest.get("fundingRate") or latest.get("lastFundingRate") or 0.0),
            "funding_observed": 1.0 if latest else 0.0,
        }
        self.funding_snapshots.append(record)
        self._stats["funding_updates"] += 1
        self._stats["last_funding_at"] = recorded_at.isoformat()
        self._trim_old_buffers()

    @staticmethod
    def _is_restricted_location_error(exc: Exception) -> bool:
        if not isinstance(exc, httpx.HTTPStatusError):
            return False
        return exc.response is not None and exc.response.status_code == 451

    def _supported_source_count(self) -> int:
        return sum(
            1
            for supported in (
                True,
                True,
                bool(self._stats.get("metrics_supported", True)),
                bool(self._stats.get("funding_supported", True)),
            )
            if supported
        )

    def _effective_source_fresh_score(self, latest: pd.Series) -> float:
        source_states = {
            "trade": (True, float(latest.get("trade_fresh", 0.0) or 0.0)),
            "depth": (True, float(latest.get("depth_fresh", 0.0) or 0.0)),
            "metrics": (bool(self._stats.get("metrics_supported", True)), float(latest.get("metrics_fresh", 0.0) or 0.0)),
            "funding": (bool(self._stats.get("funding_supported", True)), float(latest.get("funding_fresh", 0.0) or 0.0)),
        }
        active_scores = [fresh for supported, fresh in source_states.values() if supported]
        if not active_scores:
            return 0.0
        return float(sum(active_scores) / len(active_scores))

    async def _get_json(self, path: str, params: dict[str, Any]) -> Any:
        response = await self.client.get(f"{BINANCE_FUTURES_REST}{path}", params=params)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _last_float(payload: Any, *keys: str) -> float:
        if isinstance(payload, list):
            payload = payload[-1] if payload else {}
        if not isinstance(payload, dict):
            return 0.0
        for key in keys:
            value = payload.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return 0.0

    def _compute_book_ticker_snapshot(self, payload: dict[str, Any], recorded_at: datetime) -> dict[str, Any]:
        best_bid = float(payload.get("b") or payload.get("bidPrice") or 0.0)
        best_ask = float(payload.get("a") or payload.get("askPrice") or 0.0)
        best_bid_qty = float(payload.get("B") or payload.get("bidQty") or 0.0)
        best_ask_qty = float(payload.get("A") or payload.get("askQty") or 0.0)
        inside_mid = (best_bid + best_ask) / 2.0 if best_bid > 0.0 and best_ask > 0.0 else max(best_bid, best_ask, self._last_price or 0.0)
        microprice = inside_mid
        if best_bid_qty > 0.0 and best_ask_qty > 0.0 and best_bid > 0.0 and best_ask > 0.0:
            microprice = ((best_ask * best_bid_qty) + (best_bid * best_ask_qty)) / (best_bid_qty + best_ask_qty)
        inside_spread = max(best_ask - best_bid, 0.0) if best_bid > 0.0 and best_ask > 0.0 else 0.0
        spread_bps = _safe_scalar_divide(inside_spread, inside_mid) * 10_000.0 if inside_mid > 0.0 else 0.0
        inside_book_imbalance = _safe_scalar_divide(best_bid_qty - best_ask_qty, best_bid_qty + best_ask_qty)
        microprice_gap_bps = _safe_scalar_divide(microprice - inside_mid, inside_mid) * 10_000.0 if inside_mid > 0.0 else 0.0
        return {
            "timestamp": recorded_at,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "best_bid_qty": best_bid_qty,
            "best_ask_qty": best_ask_qty,
            "inside_mid": inside_mid,
            "inside_spread": inside_spread,
            "inside_spread_bps": spread_bps,
            "inside_book_imbalance": inside_book_imbalance,
            "microprice": microprice,
            "microprice_gap_bps": microprice_gap_bps,
        }

    def _compute_depth_snapshot(self, payload: dict[str, Any], recorded_at: datetime) -> dict[str, Any]:
        bids = [(float(price), float(size)) for price, size in payload.get("bids", [])]
        asks = [(float(price), float(size)) for price, size in payload.get("asks", [])]
        best_bid = bids[0][0] if bids else self._last_price or 0.0
        best_ask = asks[0][0] if asks else self._last_price or 0.0
        mid = (best_bid + best_ask) / 2.0 if best_bid > 0.0 and best_ask > 0.0 else max(best_bid, best_ask, self._last_price or 0.0)
        record: dict[str, Any] = {"timestamp": recorded_at, "depth_observed": 1.0}
        for level in FEATURE_DEPTH_LEVELS:
            pct = level / 100.0
            bid_qty = sum(size for price, size in bids if price >= mid * (1.0 - pct))
            ask_qty = sum(size for price, size in asks if price <= mid * (1.0 + pct))
            bid_notional = sum(price * size for price, size in bids if price >= mid * (1.0 - pct))
            ask_notional = sum(price * size for price, size in asks if price <= mid * (1.0 + pct))
            record[f"depth_bid_{level}pct"] = bid_qty
            record[f"depth_ask_{level}pct"] = ask_qty
            record[f"notional_bid_{level}pct"] = bid_notional
            record[f"notional_ask_{level}pct"] = ask_notional
            record[f"depth_imbalance_{level}pct"] = _safe_scalar_divide(bid_qty - ask_qty, bid_qty + ask_qty)
            record[f"notional_imbalance_{level}pct"] = _safe_scalar_divide(bid_notional - ask_notional, bid_notional + ask_notional)
        return record

    def _trim_old_buffers(self) -> None:
        cutoff = datetime.now(UTC) - timedelta(minutes=LOG_RETENTION_MINUTES)
        for bucket in (self.trade_events, self.depth_snapshots, self.metrics_snapshots, self.funding_snapshots):
            while bucket and bucket[0]["timestamp"] < cutoff:
                bucket.popleft()

    def _build_live_feature_frame(self) -> pd.DataFrame:
        trades = self._trade_frame()
        if trades.empty:
            return pd.DataFrame()

        combined = trades.join(self._depth_frame(), how="left")
        combined = combined.join(self._metrics_frame(), how="left")
        combined = combined.join(self._funding_frame(), how="left")
        combined = combined.sort_index()
        for column in DEPTH_COLUMNS:
            if column not in combined.columns:
                combined[column] = 0.0
        if "depth_observed" not in combined.columns:
            combined["depth_observed"] = 0.0
        for column in METRICS_COLUMNS:
            if column not in combined.columns:
                combined[column] = 0.0
        if "metrics_observed" not in combined.columns:
            combined["metrics_observed"] = 0.0
        for column in FUNDING_COLUMNS:
            if column not in combined.columns:
                combined[column] = 0.0
        if "funding_observed" not in combined.columns:
            combined["funding_observed"] = 0.0

        combined = _add_source_coverage(
            combined,
            max_age_buckets={
                "trade": self.max_trade_age_buckets,
                "depth": self.max_depth_age_buckets,
                "metrics": self.max_metrics_age_buckets,
                "funding": self.max_funding_age_buckets,
            },
        )

        observed_columns = [column for column in combined.columns if column.endswith("_observed")]
        fill_zero_columns = [
            column for column in combined.columns
            if column.startswith(("buy_", "sell_", "signed_", "trade_", "quote_", "depth_", "notional_"))
            or column in observed_columns
        ]
        if fill_zero_columns:
            combined[fill_zero_columns] = combined[fill_zero_columns].fillna(0.0)

        ffill_columns = [column for column in combined.columns if column not in fill_zero_columns]
        if ffill_columns:
            combined[ffill_columns] = combined[ffill_columns].ffill()

        combined = combined.dropna(subset=["price_last"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PerformanceWarning)
            combined = _add_live_derived_features(
                combined,
                bucket_seconds=self.bucket_seconds,
                min_signed_ratio=self.min_signed_ratio,
                min_depth_imbalance=self.min_depth_imbalance,
                min_trade_z=self.min_trade_z,
                min_directional_efficiency=self.min_directional_efficiency,
            )
        combined = combined.replace([np.inf, -np.inf], np.nan).fillna(0.0).reset_index(names="timestamp")
        return combined

    def _trade_frame(self) -> pd.DataFrame:
        if not self.trade_events:
            return pd.DataFrame()
        frame = pd.DataFrame.from_records(self.trade_events)
        frame["bucket"] = frame["timestamp"].dt.floor(f"{self.bucket_seconds}s")
        grouped = frame.groupby("bucket").agg(
            price_last=("price", "last"),
            price_first=("price", "first"),
            price_high=("price", "max"),
            price_low=("price", "min"),
            quantity_total=("quantity", "sum"),
            quote_qty_total=("quote_qty", "sum"),
            buy_qty=("buy_qty", "sum"),
            sell_qty=("sell_qty", "sum"),
            buy_quote_qty=("buy_quote_qty", "sum"),
            sell_quote_qty=("sell_quote_qty", "sum"),
            signed_qty=("signed_qty", "sum"),
            signed_quote_qty=("signed_quote_qty", "sum"),
            trade_count=("agg_trade_id", "count"),
        )
        grouped["price_vwap"] = grouped["quote_qty_total"] / grouped["quantity_total"].replace(0.0, np.nan)
        grouped["avg_trade_size"] = grouped["quantity_total"] / grouped["trade_count"].replace(0.0, np.nan)
        full_index = pd.date_range(grouped.index.min(), grouped.index.max(), freq=f"{self.bucket_seconds}s", tz=UTC)
        grouped = grouped.reindex(full_index)
        grouped["trade_observed"] = grouped["price_last"].notna().astype(float)
        quantity_columns = [
            "quantity_total",
            "quote_qty_total",
            "buy_qty",
            "sell_qty",
            "buy_quote_qty",
            "sell_quote_qty",
            "signed_qty",
            "signed_quote_qty",
            "trade_count",
        ]
        grouped[quantity_columns] = grouped[quantity_columns].fillna(0.0)
        grouped["price_last"] = grouped["price_last"].ffill()
        grouped["price_first"] = grouped["price_first"].fillna(grouped["price_last"])
        grouped["price_high"] = grouped["price_high"].fillna(grouped["price_last"])
        grouped["price_low"] = grouped["price_low"].fillna(grouped["price_last"])
        grouped["price_vwap"] = grouped["price_vwap"].fillna(grouped["price_last"])
        grouped["avg_trade_size"] = grouped["avg_trade_size"].fillna(0.0)
        return grouped

    def _depth_frame(self) -> pd.DataFrame:
        if not self.depth_snapshots:
            return pd.DataFrame()
        frame = pd.DataFrame.from_records(self.depth_snapshots)
        frame["bucket"] = frame["timestamp"].dt.floor(f"{self.bucket_seconds}s")
        return frame.drop(columns=["timestamp"]).groupby("bucket").last().sort_index()

    def _metrics_frame(self) -> pd.DataFrame:
        if not self.metrics_snapshots:
            return pd.DataFrame()
        frame = pd.DataFrame.from_records(self.metrics_snapshots)
        frame["bucket"] = frame["timestamp"].dt.floor(f"{self.bucket_seconds}s")
        return frame.drop(columns=["timestamp"]).groupby("bucket").last().sort_index()

    def _funding_frame(self) -> pd.DataFrame:
        if not self.funding_snapshots:
            return pd.DataFrame()
        frame = pd.DataFrame.from_records(self.funding_snapshots)
        frame["bucket"] = frame["timestamp"].dt.floor(f"{self.bucket_seconds}s")
        return frame.drop(columns=["timestamp"]).groupby("bucket").last().sort_index()

    def append_log(self, payload: dict[str, Any]) -> None:
        try:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, default=str) + "\n")
            self._stats["log_entries"] += 1
            self._stats["last_log_at"] = datetime.now(UTC).isoformat()
        except OSError as exc:
            self._stats["feed_errors"] += 1
            self._stats["last_error"] = f"log:{type(exc).__name__}"
            logger.warning("[BTC_ML] failed to write log: %s", exc)


def _add_source_coverage(frame: pd.DataFrame, *, max_age_buckets: dict[str, int]) -> pd.DataFrame:
    out = frame.copy()
    source_flags = {
        "trade": out.get("trade_observed", pd.Series(0.0, index=out.index)),
        "depth": out.get("depth_observed", pd.Series(0.0, index=out.index)),
        "metrics": out.get("metrics_observed", pd.Series(0.0, index=out.index)),
        "funding": out.get("funding_observed", pd.Series(0.0, index=out.index)),
    }
    fresh_columns: list[str] = []
    for source_name, raw_flag in source_flags.items():
        flag = raw_flag.fillna(0.0).astype(float)
        out[f"{source_name}_observed"] = flag
        age = _buckets_since_last_seen(flag.astype(bool))
        out[f"{source_name}_age_buckets"] = age
        freshness = (age <= float(max_age_buckets[source_name])).astype(float)
        out[f"{source_name}_fresh"] = freshness
        fresh_columns.append(f"{source_name}_fresh")
    out["source_fresh_count"] = out[fresh_columns].sum(axis=1)
    out["source_fresh_score"] = out["source_fresh_count"] / float(len(fresh_columns))
    out["source_all_fresh"] = (out["source_fresh_score"] >= 0.999).astype(float)
    return out


def _buckets_since_last_seen(flags: pd.Series) -> pd.Series:
    values = flags.to_numpy(dtype=bool)
    ages = np.full(len(values), np.inf, dtype=float)
    last_seen = -1
    for idx, is_seen in enumerate(values):
        if is_seen:
            last_seen = idx
            ages[idx] = 0.0
        elif last_seen >= 0:
            ages[idx] = float(idx - last_seen)
    return pd.Series(ages, index=flags.index, dtype=float)


def _add_live_derived_features(
    frame: pd.DataFrame,
    *,
    bucket_seconds: int,
    min_signed_ratio: float,
    min_depth_imbalance: float,
    min_trade_z: float,
    min_directional_efficiency: float,
) -> pd.DataFrame:
    out = frame.copy()
    out["buy_sell_ratio_1"] = _safe_divide(out["buy_qty"], out["sell_qty"].replace(0.0, np.nan))
    out["signed_qty_ratio_1"] = _safe_divide(out["signed_qty"], out["quantity_total"].replace(0.0, np.nan))
    out["signed_quote_ratio_1"] = _safe_divide(out["signed_quote_qty"], out["quote_qty_total"].replace(0.0, np.nan))

    for window in (1, 3, 12, 36):
        out[f"ret_{window}"] = out["price_last"].pct_change(window)
        out[f"trade_count_sum_{window}"] = out["trade_count"].rolling(window).sum()
        out[f"quantity_sum_{window}"] = out["quantity_total"].rolling(window).sum()
        out[f"signed_qty_sum_{window}"] = out["signed_qty"].rolling(window).sum()
        out[f"signed_quote_sum_{window}"] = out["signed_quote_qty"].rolling(window).sum()
        out[f"buy_qty_sum_{window}"] = out["buy_qty"].rolling(window).sum()
        out[f"sell_qty_sum_{window}"] = out["sell_qty"].rolling(window).sum()
        out[f"buy_quote_sum_{window}"] = out["buy_quote_qty"].rolling(window).sum()
        out[f"sell_quote_sum_{window}"] = out["sell_quote_qty"].rolling(window).sum()
        out[f"buy_ratio_{window}"] = _safe_divide(
            out[f"buy_qty_sum_{window}"],
            out[f"quantity_sum_{window}"].replace(0.0, np.nan),
        )
        out[f"signed_ratio_{window}"] = _safe_divide(
            out[f"signed_qty_sum_{window}"],
            out[f"quantity_sum_{window}"].replace(0.0, np.nan),
        )

    out["flow_accel_3v12"] = out["signed_ratio_3"] - out["signed_ratio_12"]
    out["flow_accel_12v36"] = out["signed_ratio_12"] - out["signed_ratio_36"]
    out["range_1"] = _safe_divide(out["price_high"] - out["price_low"], out["price_last"])
    out["vwap_gap"] = _safe_divide(out["price_last"] - out["price_vwap"], out["price_vwap"])
    out["vol_12"] = out["ret_1"].rolling(12).std()
    out["vol_36"] = out["ret_1"].rolling(36).std()
    out["trade_count_mean_72"] = out["trade_count_sum_12"].rolling(72).mean()
    out["trade_count_std_72"] = out["trade_count_sum_12"].rolling(72).std()
    out["trade_count_z_12"] = _safe_divide(
        out["trade_count_sum_12"] - out["trade_count_mean_72"],
        out["trade_count_std_72"].replace(0.0, np.nan),
    )
    out["signed_quote_mean_72"] = out["signed_quote_sum_12"].rolling(72).mean()
    out["signed_quote_std_72"] = out["signed_quote_sum_12"].rolling(72).std()
    out["signed_quote_z_12"] = _safe_divide(
        out["signed_quote_sum_12"] - out["signed_quote_mean_72"],
        out["signed_quote_std_72"].replace(0.0, np.nan),
    )

    for level in FEATURE_DEPTH_LEVELS:
        out[f"depth_imbalance_{level}pct_diff_3"] = out[f"depth_imbalance_{level}pct"].diff(3)
        out[f"notional_imbalance_{level}pct_diff_3"] = out[f"notional_imbalance_{level}pct"].diff(3)
        out[f"flow_depth_align_{level}pct_12"] = out["signed_ratio_12"] * out[f"depth_imbalance_{level}pct"]

    for column in (
        "sum_open_interest",
        "sum_open_interest_value",
        "count_toptrader_long_short_ratio",
        "sum_toptrader_long_short_ratio",
        "count_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
        "last_funding_rate",
    ):
        if column in out.columns:
            out[f"{column}_diff_1"] = out[column].diff()
            out[f"{column}_diff_12"] = out[column].diff(12)

    if "sum_open_interest_diff_12" in out.columns:
        out["oi_flow_align_12"] = out["sum_open_interest_diff_12"] * out["signed_quote_sum_12"]
    if "last_funding_rate" in out.columns:
        out["funding_flow_align_12"] = out["last_funding_rate"] * out["signed_ratio_12"]
    if "count_long_short_ratio" in out.columns and "sum_taker_long_short_vol_ratio" in out.columns:
        out["crowding_pressure"] = out["count_long_short_ratio"] * out["sum_taker_long_short_vol_ratio"]

    out["avg_trade_notional_1"] = _safe_divide(out["quote_qty_total"], out["trade_count"].replace(0.0, np.nan))
    out["avg_trade_notional_mean_72"] = out["avg_trade_notional_1"].rolling(72).mean()
    out["avg_trade_notional_std_72"] = out["avg_trade_notional_1"].rolling(72).std()
    out["avg_trade_notional_z_12"] = _safe_divide(
        out["avg_trade_notional_1"] - out["avg_trade_notional_mean_72"],
        out["avg_trade_notional_std_72"].replace(0.0, np.nan),
    )
    out["abs_signed_ratio_12"] = out["signed_ratio_12"].abs()
    out["abs_signed_ratio_3"] = out["signed_ratio_3"].abs()
    out["directional_efficiency_12"] = _safe_divide(out["ret_12"].abs(), out["vol_12"].replace(0.0, np.nan))
    out["directional_efficiency_36"] = _safe_divide(out["ret_36"].abs(), out["vol_36"].replace(0.0, np.nan))
    out["quote_conviction_12"] = _safe_divide(
        out["signed_quote_sum_12"].abs(),
        (out["buy_quote_sum_12"] + out["sell_quote_sum_12"]).replace(0.0, np.nan),
    )
    out["depth_pressure_change_12"] = out["depth_imbalance_1pct"].diff(12)
    out["depth_pressure_abs_12"] = out["depth_imbalance_1pct"].abs()
    out["range_vol_ratio_12"] = _safe_divide(out["range_1"], out["vol_12"].replace(0.0, np.nan))
    out["impulse_alignment_12"] = out["signed_ratio_12"] * out["depth_imbalance_1pct"]
    out["impulse_alignment_36"] = out["signed_ratio_36"] * out["depth_imbalance_1pct"]

    out["long_candidate"] = (
        (out["signed_ratio_12"] >= 0.08)
        & (out["depth_imbalance_1pct"] >= 0.02)
        & (out["trade_count_z_12"] >= 0.75)
        & (out["flow_accel_3v12"] >= 0.0)
    ).astype(float)
    out["short_candidate"] = (
        (out["signed_ratio_12"] <= -0.08)
        & (out["depth_imbalance_1pct"] <= -0.02)
        & (out["trade_count_z_12"] >= 0.75)
        & (out["flow_accel_3v12"] <= 0.0)
    ).astype(float)
    out["setup_active"] = ((out["long_candidate"] == 1.0) | (out["short_candidate"] == 1.0)).astype(float)
    out["long_impulse_candidate"] = (
        (out["signed_ratio_12"] >= min_signed_ratio)
        & (out["depth_imbalance_1pct"] >= min_depth_imbalance)
        & (out["trade_count_z_12"] >= min_trade_z)
        & (out["directional_efficiency_12"] >= min_directional_efficiency)
        & (out["impulse_alignment_12"] > 0.0)
    ).astype(float)
    out["short_impulse_candidate"] = (
        (out["signed_ratio_12"] <= -min_signed_ratio)
        & (out["depth_imbalance_1pct"] <= -min_depth_imbalance)
        & (out["trade_count_z_12"] >= min_trade_z)
        & (out["directional_efficiency_12"] >= min_directional_efficiency)
        & (out["impulse_alignment_12"] > 0.0)
    ).astype(float)
    return out


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.astype(float) / denominator.astype(float)


def _safe_scalar_divide(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-12:
        return 0.0
    return float(numerator / denominator)
