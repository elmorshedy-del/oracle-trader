from __future__ import annotations

import asyncio
import logging
import time
from bisect import bisect_left
from datetime import datetime, timezone

from config import PipelineConfig, WalletCopyResearchConfig
from data.collector import PolymarketCollector

from .wallet_copy_store import WalletCopyResearchStore


logger = logging.getLogger(__name__)

UTC = timezone.utc


class WalletLabelerService:
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

    async def start(self) -> None:
        if not self.cfg.enabled:
            logger.info("[WALLET_COPY] labeler disabled")
            return
        logger.info("[WALLET_COPY] labeler started")
        while not self._stop_event.is_set():
            cycle_started = time.time()
            try:
                self.store.mark_heartbeat("labeler", cycle_started)
                await self._apply_wallet_sell_labels(cycle_started)
                await self._apply_market_resolution_labels(cycle_started)
                await self._backfill_price_checkpoints(cycle_started)
            except Exception:
                logger.exception("[WALLET_COPY] labeler cycle failed")
            sleep_for = max(self.cfg.labeler_poll_seconds - (time.time() - cycle_started), 1.0)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=sleep_for)
            except asyncio.TimeoutError:
                continue

    async def stop(self) -> None:
        self._stop_event.set()

    async def _apply_wallet_sell_labels(self, now_ts: float) -> None:
        labeled = 0
        for buy_row in self.store.get_unlabeled_buy_trades(limit=250):
            sell_row = self.store.find_first_later_sell(buy_row)
            if sell_row is None:
                continue
            self.store.apply_wallet_sell_label(
                buy_trade_id=int(buy_row["id"]),
                sell_row=sell_row,
                match_quality="first_later_sell",
            )
            labeled += 1
        if labeled:
            self.store.set_meta("labeler_last_wallet_sell_label_at", now_ts, updated_at=now_ts)
            logger.info("[WALLET_COPY] applied %s wallet-sell labels", labeled)

    async def _apply_market_resolution_labels(self, now_ts: float) -> None:
        updated_total = 0
        for market_row in self.store.unresolved_markets(limit=250):
            market = await self.collector.get_market_by_condition_id(str(market_row["market_condition_id"]))
            if market is None or not market.closed:
                continue
            outcome_prices = {str(outcome.name): float(outcome.price or 0.0) for outcome in market.outcomes}
            winning_outcome = None
            if market.outcomes:
                winning = max(market.outcomes, key=lambda item: float(item.price or 0.0))
                winning_outcome = str(winning.name) if float(winning.price or 0.0) > 0.5 else None
            resolution_timestamp = self._resolution_timestamp(market)
            updated_total += self.store.apply_market_resolution(
                condition_id=str(market_row["market_condition_id"]),
                resolution_timestamp=resolution_timestamp,
                winning_outcome=winning_outcome,
                outcome_prices=outcome_prices,
                resolution_source="gamma_market_closed",
            )
        if updated_total:
            self.store.set_meta("labeler_last_resolution_label_at", now_ts, updated_at=now_ts)
            logger.info("[WALLET_COPY] applied %s resolution labels", updated_total)

    async def _backfill_price_checkpoints(self, now_ts: float) -> None:
        updated = 0
        cache: dict[str, list[dict[str, float]]] = {}
        for trade_row in self.store.trades_missing_price_checkpoints(limit=250):
            trade_ts = float(trade_row["timestamp"] or 0.0)
            if now_ts - trade_ts < 1800:
                continue
            asset_token_id = str(trade_row["asset_token_id"])
            history = cache.get(asset_token_id)
            if history is None:
                history = await self.collector.get_price_history(
                    asset_token_id,
                    interval=self.cfg.price_history_interval,
                    fidelity=self.cfg.price_history_fidelity,
                )
                cache[asset_token_id] = history
            if not history:
                continue
            p1 = self._history_price_at(history, trade_ts + 60.0)
            p5 = self._history_price_at(history, trade_ts + 300.0)
            p30 = self._history_price_at(history, trade_ts + 1800.0)
            self.store.update_price_checkpoints(
                trade_id=int(trade_row["id"]),
                price_1min_after=p1,
                price_5min_after=p5,
                price_30min_after=p30,
            )
            updated += 1
        if updated:
            self.store.set_meta("labeler_last_price_backfill_at", now_ts, updated_at=now_ts)
            logger.info("[WALLET_COPY] backfilled price checkpoints for %s trades", updated)

    @staticmethod
    def _history_price_at(history: list[dict[str, float]], target_ts: float) -> float | None:
        if not history:
            return None
        timestamps = [float(row.get("t") or 0.0) for row in history]
        idx = bisect_left(timestamps, target_ts)
        candidates: list[tuple[float, float]] = []
        if idx < len(history):
            candidates.append((abs(timestamps[idx] - target_ts), float(history[idx].get("p") or 0.0)))
        if idx > 0:
            candidates.append((abs(timestamps[idx - 1] - target_ts), float(history[idx - 1].get("p") or 0.0)))
        if not candidates:
            return None
        best_distance, best_price = min(candidates, key=lambda item: item[0])
        if best_distance > 900:
            return None
        return best_price

    @staticmethod
    def _resolution_timestamp(market) -> float:
        for raw_value in (getattr(market, "fetched_at", None), getattr(market, "end_date", None)):
            if raw_value is None:
                continue
            if isinstance(raw_value, datetime):
                return raw_value.astimezone(UTC).timestamp()
            try:
                return datetime.fromisoformat(str(raw_value).replace("Z", "+00:00")).astimezone(UTC).timestamp()
            except ValueError:
                continue
        return time.time()
