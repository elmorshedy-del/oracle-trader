from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

from config import PipelineConfig, WalletCopyResearchConfig
from data.collector import PolymarketCollector
from data.models import Market

from .wallet_copy_store import WalletCopyResearchStore


logger = logging.getLogger(__name__)

UTC = timezone.utc
CATEGORY_VERSION = "wallet_copy_category_v1"
MAX_RECENT_TRADE_KEYS = 20000
TOKEN_DEPTH_BAND_PCT = 0.02
HEARTBEAT_PROGRESS_INTERVAL = 5


class WalletTrackerService:
    def __init__(
        self,
        *,
        pipeline_config: PipelineConfig,
        collector: PolymarketCollector,
        store: WalletCopyResearchStore,
    ) -> None:
        self.pipeline_config = pipeline_config
        self.cfg: WalletCopyResearchConfig = pipeline_config.wallet_copy_research
        self.collector = collector
        self.store = store
        self._stop_event = asyncio.Event()
        self._tracked_wallets: list[dict[str, Any]] = []
        self._last_leaderboard_refresh_at = 0.0
        self._last_positions_refresh_at = 0.0
        self._market_cache: dict[str, tuple[float, dict[str, Any], Market | None]] = {}
        self._orderbook_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._latest_positions: dict[str, dict[str, Any]] = {}
        self._latest_trade_timestamp_by_wallet: dict[str, float] = {}
        self._recent_trade_keys: deque[str] = deque()
        self._recent_trade_key_set: set[str] = set()
        self._btc_samples: deque[tuple[float, float]] = deque()
        self._last_btc_sample_at = 0.0

    async def start(self) -> None:
        if not self.cfg.enabled:
            logger.info("[WALLET_COPY] tracker disabled")
            return
        logger.info("[WALLET_COPY] tracker started")
        while not self._stop_event.is_set():
            cycle_started = time.time()
            try:
                self.store.mark_heartbeat("tracker", cycle_started)
                await self._sample_btc_context(cycle_started)
                await self._refresh_wallets_if_needed(cycle_started)
                await self._refresh_positions_if_needed(cycle_started)
                await self._poll_wallet_activity(cycle_started)
            except Exception:
                logger.exception("[WALLET_COPY] tracker cycle failed")
            sleep_for = max(self.cfg.wallet_activity_poll_seconds - (time.time() - cycle_started), 0.5)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=sleep_for)
            except asyncio.TimeoutError:
                continue

    async def stop(self) -> None:
        self._stop_event.set()

    def tracked_wallet_count(self) -> int:
        return len(self._tracked_wallets)

    async def _refresh_wallets_if_needed(self, now_ts: float) -> None:
        if self._tracked_wallets and now_ts - self._last_leaderboard_refresh_at < self.cfg.leaderboard_refresh_seconds:
            return
        tasks = [
            self.collector.get_leaderboard(limit=self.cfg.leaderboard_all_limit, period="all", sort_by="profit"),
            self.collector.get_leaderboard(limit=self.cfg.leaderboard_30d_profit_limit, period="30d", sort_by="profit"),
            self.collector.get_leaderboard(limit=self.cfg.leaderboard_30d_volume_limit, period="30d", sort_by="volume"),
        ]
        leaderboard_all, leaderboard_30d_profit, leaderboard_30d_volume = await asyncio.gather(*tasks, return_exceptions=False)
        merged = self._merge_leaderboards(
            leaderboard_all=leaderboard_all,
            leaderboard_30d_profit=leaderboard_30d_profit,
            leaderboard_30d_volume=leaderboard_30d_volume,
            snapshot_at=now_ts,
        )
        self._tracked_wallets = merged[: self.cfg.tracked_wallet_limit]
        self._last_leaderboard_refresh_at = now_ts
        self.store.replace_tracked_wallets(self._tracked_wallets, snapshot_at=now_ts)
        logger.info("[WALLET_COPY] refreshed %s tracked wallets", len(self._tracked_wallets))

    def _merge_leaderboards(
        self,
        *,
        leaderboard_all: list[dict[str, Any]],
        leaderboard_30d_profit: list[dict[str, Any]],
        leaderboard_30d_volume: list[dict[str, Any]],
        snapshot_at: float,
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}

        def ingest(rows: list[dict[str, Any]], *, source_flag: str, rank_key: str) -> None:
            for index, row in enumerate(rows, start=1):
                address = self._wallet_address(row)
                if not address:
                    continue
                payload = merged.setdefault(
                    address,
                    {
                        "address": address,
                        "leaderboard_rank_all": None,
                        "leaderboard_rank_30d": None,
                        "leaderboard_rank_30d_volume": None,
                        "leaderboard_profit": None,
                        "leaderboard_volume": None,
                        "leaderboard_num_trades": None,
                        "leaderboard_win_rate": None,
                        "leaderboard_source_flags": [],
                        "raw_leaderboard_json": {},
                        "snapshot_at": snapshot_at,
                    },
                )
                payload[rank_key] = min(
                    index,
                    payload[rank_key] if isinstance(payload[rank_key], int) else index,
                )
                payload["leaderboard_profit"] = self._coalesce_numeric(
                    payload["leaderboard_profit"],
                    row.get("pnl"),
                    row.get("cashPnl"),
                    row.get("totalPnl"),
                )
                payload["leaderboard_volume"] = self._coalesce_numeric(
                    payload["leaderboard_volume"],
                    row.get("vol"),
                    row.get("volume"),
                )
                payload["leaderboard_num_trades"] = self._coalesce_int(
                    payload["leaderboard_num_trades"],
                    row.get("numTrades"),
                    row.get("trades"),
                    row.get("tradeCount"),
                )
                payload["leaderboard_win_rate"] = self._coalesce_numeric(
                    payload["leaderboard_win_rate"],
                    row.get("winRate"),
                    row.get("win_rate"),
                )
                if source_flag not in payload["leaderboard_source_flags"]:
                    payload["leaderboard_source_flags"].append(source_flag)
                payload["raw_leaderboard_json"][source_flag] = row

        ingest(leaderboard_all, source_flag="all_profit", rank_key="leaderboard_rank_all")
        ingest(leaderboard_30d_profit, source_flag="30d_profit", rank_key="leaderboard_rank_30d")
        ingest(leaderboard_30d_volume, source_flag="30d_volume", rank_key="leaderboard_rank_30d_volume")

        def sort_key(row: dict[str, Any]) -> tuple[int, int, int]:
            fallback = 10**9
            return (
                int(row["leaderboard_rank_30d"] or fallback),
                int(row["leaderboard_rank_all"] or fallback),
                int(row["leaderboard_rank_30d_volume"] or fallback),
            )

        return sorted(merged.values(), key=sort_key)

    async def _refresh_positions_if_needed(self, now_ts: float) -> None:
        if not self._tracked_wallets:
            return
        if self._last_positions_refresh_at and now_ts - self._last_positions_refresh_at < self.cfg.positions_refresh_seconds:
            return
        self.store.set_meta("positions_last_started_at", now_ts, updated_at=now_ts)
        self.store.set_meta("positions_last_progress_at", now_ts, updated_at=now_ts)
        semaphore = asyncio.Semaphore(self.cfg.positions_concurrency)

        async def fetch(wallet: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
            async with semaphore:
                rows = await self.collector.get_wallet_positions(wallet["address"])
                return wallet["address"], rows

        results = await asyncio.gather(*(fetch(wallet) for wallet in self._tracked_wallets), return_exceptions=True)
        for index, result in enumerate(results, start=1):
            if isinstance(result, Exception):
                logger.warning("[WALLET_COPY] positions fetch error: %s", result)
                self.store.set_meta("positions_last_progress_at", time.time(), updated_at=time.time())
                if index == 1 or index % HEARTBEAT_PROGRESS_INTERVAL == 0:
                    self.store.mark_heartbeat("tracker", time.time())
                continue
            address, rows = result
            held_condition_ids = sorted({str(row.get("conditionId") or "") for row in rows if row.get("conditionId")})
            held_asset_ids = sorted({str(row.get("asset") or "") for row in rows if row.get("asset")})
            snapshot = {
                "wallet_address": address,
                "snapshot_at": now_ts,
                "open_positions": len(rows),
                "held_condition_ids": held_condition_ids,
                "held_asset_ids": held_asset_ids,
                "raw_positions": rows,
            }
            self._latest_positions[address] = snapshot
            self.store.record_positions_snapshot(
                wallet_address=address,
                snapshot_at=now_ts,
                open_positions=len(rows),
                held_condition_ids=held_condition_ids,
                held_asset_ids=held_asset_ids,
                raw_positions=rows,
            )
            if index == 1 or index % HEARTBEAT_PROGRESS_INTERVAL == 0:
                progress_at = time.time()
                self.store.set_meta("positions_last_progress_at", progress_at, updated_at=progress_at)
                self.store.mark_heartbeat("tracker", progress_at)
        completed_at = time.time()
        self._last_positions_refresh_at = completed_at
        self.store.set_meta("positions_last_refreshed_at", completed_at, updated_at=completed_at)
        self.store.set_meta("positions_last_completed_at", completed_at, updated_at=completed_at)

    async def _poll_wallet_activity(self, now_ts: float) -> None:
        if not self._tracked_wallets:
            return
        semaphore = asyncio.Semaphore(self.cfg.activity_concurrency)

        async def fetch(wallet: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            async with semaphore:
                rows = await self.collector.get_wallet_activity(
                    wallet["address"],
                    limit=self.cfg.wallet_activity_limit,
                    activity_type="TRADE",
                )
                return wallet, rows

        results = await asyncio.gather(*(fetch(wallet) for wallet in self._tracked_wallets), return_exceptions=True)
        for index, result in enumerate(results, start=1):
            if isinstance(result, Exception):
                logger.warning("[WALLET_COPY] activity fetch error: %s", result)
                if index == 1 or index % HEARTBEAT_PROGRESS_INTERVAL == 0:
                    self.store.mark_heartbeat("tracker", time.time())
                continue
            wallet, rows = result
            rows_sorted = sorted(rows or [], key=lambda item: int(item.get("timestamp") or 0))
            latest_ts = self._latest_trade_timestamp_by_wallet.get(wallet["address"], 0.0)
            for row in rows_sorted:
                if str(row.get("type") or "").upper() != "TRADE":
                    continue
                trade_key = self._build_trade_key(wallet["address"], row)
                event_ts = float(row.get("timestamp") or 0.0)
                if trade_key in self._recent_trade_key_set:
                    continue
                if latest_ts and event_ts and event_ts < float(latest_ts):
                    continue
                trade = await self._build_trade_row(wallet=wallet, row=row, now_ts=now_ts)
                if trade is None:
                    continue
                inserted = self.store.insert_wallet_trade(trade)
                self._remember_trade_key(trade_key)
                if inserted and event_ts:
                    latest_ts = max(latest_ts, event_ts)
            if latest_ts:
                self._latest_trade_timestamp_by_wallet[wallet["address"]] = latest_ts
            if index == 1 or index % HEARTBEAT_PROGRESS_INTERVAL == 0:
                self.store.mark_heartbeat("tracker", time.time())

    async def _build_trade_row(self, *, wallet: dict[str, Any], row: dict[str, Any], now_ts: float) -> dict[str, Any] | None:
        condition_id = str(row.get("conditionId") or "")
        asset_token_id = str(row.get("asset") or "")
        if not condition_id or not asset_token_id:
            return None
        raw_market, parsed_market = await self._get_market(condition_id)
        if parsed_market is None:
            return None
        raw_orderbook = await self._get_orderbook(asset_token_id)
        token_best_bid, token_best_ask, token_best_bid_size, token_best_ask_size = self._best_quotes(raw_orderbook)
        token_depth = self._depth_within_band(raw_orderbook, side=str(row.get("side") or "").upper())
        binary_quotes = await self._binary_market_quotes(raw_market)
        latest_positions = self._latest_positions.get(wallet["address"]) or self.store.latest_position_snapshot(wallet["address"])
        size_usd = self._as_float(row.get("usdcSize"))
        metrics = self.store.wallet_metrics_before(
            wallet_address=wallet["address"],
            trade_timestamp=float(row.get("timestamp") or now_ts),
            condition_id=condition_id,
            asset_token_id=asset_token_id,
            trade_size_usd=size_usd,
            latest_positions_snapshot=latest_positions,
        )
        btc_price, btc_momentum_60s = self._current_btc_context(now_ts)
        event_ts = float(row.get("timestamp") or now_ts)
        event_dt = datetime.fromtimestamp(event_ts, tz=UTC)
        tags = self._market_tags(raw_market)
        market_category = self._categorize_market(
            slug=str(parsed_market.slug or ""),
            title=str(parsed_market.question or row.get("title") or ""),
            tags=tags,
        )
        return {
            "trade_key": self._build_trade_key(wallet["address"], row),
            "tx_hash": str(row.get("transactionHash") or ""),
            "timestamp": event_ts,
            "detected_at": now_ts,
            "snapshot_taken_at": now_ts,
            "wallet_address": wallet["address"],
            "market_condition_id": condition_id,
            "asset_token_id": asset_token_id,
            "market_title": str(parsed_market.question or row.get("title") or ""),
            "market_slug": str(parsed_market.slug or row.get("slug") or ""),
            "side": str(row.get("side") or "").upper(),
            "outcome": str(row.get("outcome") or self._outcome_name(parsed_market, asset_token_id)),
            "price": self._as_float(row.get("price")),
            "size_shares": self._as_float(row.get("size")),
            "size_usd": size_usd,
            "token_best_bid": token_best_bid,
            "token_best_ask": token_best_ask,
            "token_best_bid_size": token_best_bid_size,
            "token_best_ask_size": token_best_ask_size,
            "token_depth_within_2pct": token_depth,
            "market_yes_bid": binary_quotes.get("yes_bid"),
            "market_yes_ask": binary_quotes.get("yes_ask"),
            "market_no_bid": binary_quotes.get("no_bid"),
            "market_no_ask": binary_quotes.get("no_ask"),
            "market_spread": self._as_float(parsed_market.spread),
            "market_volume_24h": self._as_float(parsed_market.volume_24h),
            "market_volume_total": self._as_float(parsed_market.volume_total),
            "market_liquidity": self._as_float(parsed_market.liquidity),
            "market_midpoint": self._as_float(parsed_market.midpoint),
            "market_reward_pool": self._as_float(parsed_market.reward_pool),
            "market_primary_tag": tags[0] if tags else None,
            "market_tags_json": json.dumps(tags, sort_keys=True),
            "market_end_date": parsed_market.end_date,
            "market_seconds_to_expiry": self._seconds_to_expiry(parsed_market.end_date, now_ts),
            "market_category": market_category,
            "market_category_version": CATEGORY_VERSION,
            "market_active": 1 if parsed_market.active else 0,
            "market_closed": 1 if parsed_market.closed else 0,
            "wallet_leaderboard_rank": metrics["wallet_leaderboard_rank"],
            "wallet_leaderboard_profit": metrics["wallet_leaderboard_profit"],
            "wallet_leaderboard_win_rate": metrics["wallet_leaderboard_win_rate"],
            "wallet_open_positions": metrics["wallet_open_positions"],
            "wallet_trade_count_24h": metrics["wallet_trade_count_24h"],
            "is_adding_to_position": metrics["is_adding_to_position"],
            "size_vs_wallet_avg": metrics["size_vs_wallet_avg"],
            "detection_delay_seconds": max(now_ts - event_ts, 0.0),
            "hour_of_day": event_dt.hour + (event_dt.minute / 60.0),
            "day_of_week": event_dt.weekday(),
            "btc_price": btc_price,
            "btc_momentum_60s": btc_momentum_60s,
            "raw_activity_json": json.dumps(row, sort_keys=True),
            "raw_market_json": json.dumps(raw_market, sort_keys=True),
            "raw_orderbook_json": json.dumps(raw_orderbook, sort_keys=True),
            "raw_positions_json": metrics["raw_positions_json"],
            "collector_version": self.cfg.collector_version,
            "schema_version": self.cfg.schema_version,
        }

    async def _get_market(self, condition_id: str) -> tuple[dict[str, Any], Market | None]:
        cached = self._market_cache.get(condition_id)
        now_ts = time.time()
        if cached and cached[0] > now_ts:
            return cached[1], cached[2]
        raw_market = await self.collector.get_gamma_market_payload_by_condition_id(condition_id)
        parsed_market = self.collector._parse_gamma_market(raw_market) if raw_market else None
        if raw_market is None:
            raw_market = {}
        self._market_cache[condition_id] = (now_ts + self.cfg.market_cache_ttl_seconds, raw_market, parsed_market)
        return raw_market, parsed_market

    async def _get_orderbook(self, asset_token_id: str) -> dict[str, Any]:
        cached = self._orderbook_cache.get(asset_token_id)
        now_ts = time.time()
        if cached and cached[0] > now_ts:
            return cached[1]
        orderbook = await self.collector.get_orderbook(asset_token_id)
        self._orderbook_cache[asset_token_id] = (now_ts + self.cfg.orderbook_cache_ttl_seconds, orderbook)
        return orderbook

    async def _binary_market_quotes(self, raw_market: dict[str, Any]) -> dict[str, float | None]:
        tokens = self._parse_string_list(raw_market.get("clobTokenIds"))
        outcome_names = self._parse_string_list(raw_market.get("outcomes"))
        if len(tokens) < 2 or len(outcome_names) < 2:
            return {"yes_bid": None, "yes_ask": None, "no_bid": None, "no_ask": None}
        mapping = {outcome_names[index].strip().lower(): tokens[index] for index in range(min(len(tokens), len(outcome_names)))}
        yes_token = mapping.get("yes")
        no_token = mapping.get("no")
        if not yes_token or not no_token:
            return {"yes_bid": None, "yes_ask": None, "no_bid": None, "no_ask": None}
        yes_book, no_book = await asyncio.gather(self._get_orderbook(yes_token), self._get_orderbook(no_token))
        yes_bid, yes_ask, _, _ = self._best_quotes(yes_book)
        no_bid, no_ask, _, _ = self._best_quotes(no_book)
        return {"yes_bid": yes_bid, "yes_ask": yes_ask, "no_bid": no_bid, "no_ask": no_ask}

    async def _sample_btc_context(self, now_ts: float) -> None:
        if not self.cfg.btc_context_enabled:
            return
        if self._last_btc_sample_at and now_ts - self._last_btc_sample_at < self.cfg.btc_price_poll_seconds:
            return
        try:
            response = await self.collector.client.get("https://api.coinbase.com/v2/prices/BTC-USD/spot")
            response.raise_for_status()
            payload = response.json()
            price = self._as_float(payload.get("data", {}).get("amount"))
            if price > 0:
                self._btc_samples.append((now_ts, price))
                while len(self._btc_samples) > 720:
                    self._btc_samples.popleft()
                while self._btc_samples and now_ts - self._btc_samples[0][0] > 7200:
                    self._btc_samples.popleft()
                self._last_btc_sample_at = now_ts
                self.store.set_meta("btc_last_sample_at", now_ts, updated_at=now_ts)
        except Exception:
            logger.warning("[WALLET_COPY] BTC context sample failed", exc_info=True)

    def _current_btc_context(self, now_ts: float) -> tuple[float | None, float | None]:
        if not self._btc_samples:
            return None, None
        latest_ts, latest_price = self._btc_samples[-1]
        reference_price = None
        target_ts = now_ts - 60.0
        for sample_ts, sample_price in reversed(self._btc_samples):
            if sample_ts <= target_ts:
                reference_price = sample_price
                break
        if reference_price is None:
            reference_price = self._btc_samples[0][1]
        momentum = ((latest_price - reference_price) / reference_price) if reference_price else None
        return latest_price, momentum

    @staticmethod
    def _wallet_address(row: dict[str, Any]) -> str:
        return str(row.get("proxyWallet") or row.get("address") or row.get("wallet") or row.get("user") or "").strip()

    @staticmethod
    def _coalesce_numeric(current: Any, *values: Any) -> float | None:
        if current is not None:
            return float(current)
        for value in values:
            if value is None or value == "":
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _coalesce_int(current: Any, *values: Any) -> int | None:
        if current is not None:
            return int(current)
        for value in values:
            if value is None or value == "":
                continue
            try:
                return int(float(value))
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _as_float(value: Any) -> float | None:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _build_trade_key(wallet_address: str, row: dict[str, Any]) -> str:
        return "|".join(
            [
                wallet_address,
                str(row.get("transactionHash") or ""),
                str(row.get("conditionId") or ""),
                str(row.get("asset") or ""),
                str(row.get("side") or ""),
                str(row.get("timestamp") or ""),
                str(row.get("type") or ""),
            ]
        )

    def _remember_trade_key(self, trade_key: str) -> None:
        if trade_key in self._recent_trade_key_set:
            return
        self._recent_trade_key_set.add(trade_key)
        self._recent_trade_keys.append(trade_key)
        while len(self._recent_trade_keys) > MAX_RECENT_TRADE_KEYS:
            oldest = self._recent_trade_keys.popleft()
            self._recent_trade_key_set.discard(oldest)

    @staticmethod
    def _best_quotes(orderbook: dict[str, Any]) -> tuple[float | None, float | None, float | None, float | None]:
        bids = orderbook.get("bids") or []
        asks = orderbook.get("asks") or []
        best_bid = WalletTrackerService._as_float(bids[0].get("price")) if bids else None
        best_bid_size = WalletTrackerService._as_float(bids[0].get("size")) if bids else None
        best_ask = WalletTrackerService._as_float(asks[0].get("price")) if asks else None
        best_ask_size = WalletTrackerService._as_float(asks[0].get("size")) if asks else None
        return best_bid, best_ask, best_bid_size, best_ask_size

    @staticmethod
    def _depth_within_band(orderbook: dict[str, Any], *, side: str) -> float | None:
        bids = orderbook.get("bids") or []
        asks = orderbook.get("asks") or []
        if side == "BUY" and asks:
            best_ask = WalletTrackerService._as_float(asks[0].get("price"))
            if best_ask is None:
                return None
            max_price = best_ask * (1.0 + TOKEN_DEPTH_BAND_PCT)
            depth = 0.0
            for level in asks:
                level_price = WalletTrackerService._as_float(level.get("price"))
                level_size = WalletTrackerService._as_float(level.get("size")) or 0.0
                if level_price is None or level_price > max_price:
                    break
                depth += level_size
            return depth
        if side == "SELL" and bids:
            best_bid = WalletTrackerService._as_float(bids[0].get("price"))
            if best_bid is None:
                return None
            min_price = best_bid * (1.0 - TOKEN_DEPTH_BAND_PCT)
            depth = 0.0
            for level in bids:
                level_price = WalletTrackerService._as_float(level.get("price"))
                level_size = WalletTrackerService._as_float(level.get("size")) or 0.0
                if level_price is None or level_price < min_price:
                    break
                depth += level_size
            return depth
        return None

    @staticmethod
    def _parse_string_list(raw_value: Any) -> list[str]:
        if isinstance(raw_value, list):
            return [str(item) for item in raw_value]
        if raw_value is None:
            return []
        try:
            parsed = json.loads(str(raw_value))
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            pass
        value = str(raw_value).strip().strip("[]")
        if not value:
            return []
        return [part.strip().strip('"') for part in value.split(",") if part.strip()]

    def _market_tags(self, raw_market: dict[str, Any]) -> list[str]:
        tags = raw_market.get("tags") or []
        normalized: list[str] = []
        for tag in tags:
            if isinstance(tag, dict):
                label = str(tag.get("label") or tag.get("slug") or "").strip()
            else:
                label = str(tag).strip()
            if label:
                normalized.append(label)
        return normalized

    @staticmethod
    def _categorize_market(*, slug: str, title: str, tags: list[str]) -> str:
        haystack = " ".join([slug.lower(), title.lower(), " ".join(tag.lower() for tag in tags)])
        if any(keyword in haystack for keyword in ("nhl", "nba", "nfl", "mlb", "soccer", "football", "tennis", "ufc", "fifa", "golf")):
            return "sports"
        if any(keyword in haystack for keyword in ("bitcoin", "btc", "ethereum", "eth", "sol", "crypto")):
            if any(keyword in haystack for keyword in ("5m", "5 min", "15m", "15 min", "30m", "30 min")):
                return "crypto_short"
            if any(keyword in haystack for keyword in ("hour", "daily", "week")):
                return "crypto_hourly"
            return "crypto"
        if any(keyword in haystack for keyword in ("election", "president", "senate", "house", "governor", "mayor", "parliament")):
            return "politics"
        if any(keyword in haystack for keyword in ("fed", "rates", "cpi", "inflation", "recession", "gdp", "tariff", "economy")):
            return "macro"
        if any(keyword in haystack for keyword in ("policy", "law", "court", "supreme court", "executive order")):
            return "policy"
        return "other"

    @staticmethod
    def _seconds_to_expiry(end_date: str | None, now_ts: float) -> float | None:
        if not end_date:
            return None
        try:
            end_at = datetime.fromisoformat(str(end_date).replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
        return end_at - now_ts

    @staticmethod
    def _outcome_name(market: Market, asset_token_id: str) -> str:
        for outcome in market.outcomes:
            if outcome.token_id == asset_token_id:
                return outcome.name
        return ""
