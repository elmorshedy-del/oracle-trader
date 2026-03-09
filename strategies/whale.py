"""
Strategy: Whale Wallet Tracking (Layer 3 — The Advisor)
======================================================
Tracks top-performing wallets and uses their activity as a
confirmation/rejection filter for signals from other strategies.

Does NOT generate standalone signals — instead it provides a
confidence multiplier.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from data.models import Market, Event, Signal, WhaleWallet, SignalSource, SignalAction
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class WhaleTrackingStrategy(BaseStrategy):
    name = "whale_tracking"
    description = "Track smart money wallets for signal confirmation"

    def __init__(self, config, collector=None):
        super().__init__(config)
        self.cfg = config.whale
        self.collector = collector
        self.whale_wallets: list[WhaleWallet] = []
        self.whale_activity: dict[str, list[dict]] = {}  # market_slug -> recent whale trades
        self._last_refresh: datetime | None = None
        self._last_activity_refresh: datetime | None = None
        self._market_sentiment_cache: dict[str, dict] = {}
        self._stats.update(
            {
                "activity_markets": 0,
                "overlay_matches": 0,
                "standalone_candidates": 0,
            }
        )

    async def scan(self, markets: list[Market], events: list[Event]) -> list[Signal]:
        """Whale tracking doesn't generate signals — it enriches other signals."""
        self._stats["scans_completed"] += 1

        # Refresh whale rankings periodically
        if self._should_refresh():
            await self.refresh_whales()
        if self._should_refresh_activity():
            await self.refresh_market_activity()

        return []

    async def refresh_whales(self):
        """Refresh the whale wallet rankings from the leaderboard."""
        if not self.collector:
            return

        try:
            leaderboard = await self.collector.get_leaderboard(limit=self.cfg.top_n_wallets)
            self.whale_wallets = []

            for entry in leaderboard:
                address = (
                    entry.get("proxyWallet") or
                    entry.get("address") or
                    entry.get("wallet") or
                    entry.get("user") or ""
                )
                name = entry.get("name") or entry.get("pseudonym") or entry.get("username") or "anon"
                total_pnl = float(
                    entry.get("cashPnl") or entry.get("pnl") or
                    entry.get("totalPnl") or entry.get("profit") or 0
                )
                win_rate = float(entry.get("winRate") or entry.get("win_rate") or 0)
                total_trades = int(
                    entry.get("numTrades") or entry.get("totalTrades") or
                    entry.get("trades") or entry.get("numTraded") or 0
                )
                wallet = WhaleWallet(
                    address=address,
                    name=name,
                    total_pnl=total_pnl,
                    win_rate=win_rate,
                    total_trades=total_trades,
                )

                # API doesn't return win_rate, filter on PnL only
                if wallet.total_pnl >= self.cfg.min_pnl_usd:
                    self.whale_wallets.append(wallet)

            self._last_refresh = datetime.now(timezone.utc)
            logger.info(
                f"[WHALE] Refreshed: {len(self.whale_wallets)} whales loaded from "
                f"{len(leaderboard)} entries"
            )
            self._stats["whale_count"] = len(self.whale_wallets)
            logger.info(f"[WHALE] Refreshed: tracking {len(self.whale_wallets)} whales")

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"[WHALE] Failed to refresh: {e}")

    async def refresh_market_activity(self):
        """Build a cached condition_id -> whale sentiment map from recent whale trades."""
        if not self.collector or not self.whale_wallets:
            self._market_sentiment_cache = {}
            self._stats["activity_markets"] = 0
            return

        wallets = self.whale_wallets[: self.cfg.activity_wallet_limit]
        semaphore = asyncio.Semaphore(4)

        async def fetch_wallet_activity(wallet: WhaleWallet):
            async with semaphore:
                return wallet.address, await self.collector.get_wallet_activity(
                    wallet.address,
                    limit=self.cfg.activity_trades_per_wallet,
                )

        try:
            results = await asyncio.gather(
                *(fetch_wallet_activity(wallet) for wallet in wallets),
                return_exceptions=True,
            )
            aggregated: dict[str, dict] = {}
            for result in results:
                if isinstance(result, Exception):
                    continue
                address, activity = result
                for trade in activity or []:
                    condition_id = (
                        trade.get("conditionId")
                        or trade.get("condition_id")
                        or trade.get("marketConditionId")
                        or ""
                    )
                    if not condition_id:
                        continue
                    row = aggregated.setdefault(
                        condition_id,
                        {
                            "condition_id": condition_id,
                            "market_slug": (
                                trade.get("slug")
                                or trade.get("marketSlug")
                                or trade.get("market_slug")
                                or ""
                            ),
                            "buy_size": 0.0,
                            "sell_size": 0.0,
                            "total_size": 0.0,
                            "trade_count": 0,
                            "wallets": set(),
                            "last_seen": None,
                        },
                    )
                    side = str(trade.get("side", "") or "").upper()
                    size = self._coerce_float(
                        trade.get("size")
                        or trade.get("amount")
                        or trade.get("notional")
                        or trade.get("matchedAmount")
                        or 0
                    )
                    if side == "BUY":
                        row["buy_size"] += size
                    else:
                        row["sell_size"] += size
                    row["total_size"] += size
                    row["trade_count"] += 1
                    row["wallets"].add(address)
                    seen_at = self._parse_trade_time(
                        trade.get("timestamp")
                        or trade.get("createdAt")
                        or trade.get("time")
                        or trade.get("created_at")
                    )
                    if row["last_seen"] is None or seen_at > row["last_seen"]:
                        row["last_seen"] = seen_at
                    if not row["market_slug"]:
                        row["market_slug"] = (
                            trade.get("slug")
                            or trade.get("marketSlug")
                            or trade.get("market_slug")
                            or ""
                        )

            sentiment_cache: dict[str, dict] = {}
            for condition_id, row in aggregated.items():
                buy_size = row["buy_size"]
                sell_size = row["sell_size"]
                total_size = max(row["total_size"], 0.0)
                whale_count = len(row["wallets"])
                if total_size <= 0 or whale_count <= 0:
                    continue
                buy_ratio = buy_size / total_size
                if buy_ratio > 0.65:
                    direction = "bullish"
                elif buy_ratio < 0.35:
                    direction = "bearish"
                else:
                    direction = "neutral"
                sentiment_cache[condition_id] = {
                    "condition_id": condition_id,
                    "market_slug": row["market_slug"],
                    "whale_count": whale_count,
                    "trade_count": row["trade_count"],
                    "buy_ratio": buy_ratio,
                    "net_direction": direction,
                    "total_volume": total_size,
                    "confidence_multiplier": 1.0
                    + min(abs(buy_ratio - 0.5) * 0.8, max(self.cfg.confirmation_boost - 1.0, 0.0)),
                    "last_seen": row["last_seen"],
                }
            self._market_sentiment_cache = sentiment_cache
            self._last_activity_refresh = datetime.now(timezone.utc)
            self._stats["activity_markets"] = len(sentiment_cache)
            logger.info("[WHALE] Activity cache refreshed: %s markets", len(sentiment_cache))
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"[WHALE] Failed to refresh activity cache: {e}")

    async def get_whale_sentiment(self, condition_id: str) -> dict:
        """
        Check if whales are active in a specific market.
        Returns: {
            "whale_count": int,
            "net_direction": "bullish" | "bearish" | "neutral",
            "total_volume": float,
            "confidence_multiplier": float,
        }
        """
        if not self.collector or not self.whale_wallets:
            return {
                "whale_count": 0,
                "net_direction": "neutral",
                "total_volume": 0.0,
                "confidence_multiplier": 1.0,
            }

        buys = 0
        sells = 0
        whale_count = 0
        total_volume = 0.0

        for wallet in self.whale_wallets[:20]:  # check top 20 to limit API calls
            try:
                activity = await self.collector.get_wallet_activity(wallet.address, limit=10)
                for trade in activity:
                    trade_condition = trade.get("conditionId", "")
                    if trade_condition == condition_id:
                        whale_count += 1
                        size = float(trade.get("size", 0) or 0)
                        total_volume += size
                        if trade.get("side") == "BUY":
                            buys += size
                        else:
                            sells += size
            except Exception:
                continue

        if buys + sells == 0:
            direction = "neutral"
            multiplier = 1.0
        else:
            buy_ratio = buys / (buys + sells)
            if buy_ratio > 0.65:
                direction = "bullish"
                multiplier = 1.0 + min((buy_ratio - 0.5) * 0.75, 0.3)
            elif buy_ratio < 0.35:
                direction = "bearish"
                multiplier = 1.0 + min((0.5 - buy_ratio) * 0.75, 0.3)
            else:
                direction = "neutral"
                multiplier = 1.0

        return {
            "whale_count": whale_count,
            "net_direction": direction,
            "total_volume": total_volume,
            "confidence_multiplier": multiplier,
        }

    def get_cached_whale_sentiment(self, condition_id: str) -> dict | None:
        sentiment = self._market_sentiment_cache.get(condition_id)
        if not sentiment:
            return None
        if not self._is_fresh(sentiment.get("last_seen")):
            return None
        return sentiment

    def apply_cached_confirmation(self, signal: Signal) -> tuple[Signal, bool]:
        sentiment = self.get_cached_whale_sentiment(signal.condition_id)
        cloned = signal.model_copy(deep=True)
        if not sentiment or not self._eligible_sentiment(
            sentiment,
            min_whales=self.cfg.overlay_min_whales,
            min_total_size=self.cfg.overlay_min_total_size,
        ):
            return cloned, False
        self._stats["overlay_matches"] += 1
        return self.confirm_signal(cloned, sentiment), True

    def build_standalone_signals(self, markets: list[Market]) -> list[Signal]:
        """Create a standalone whale-follow experiment from the cached activity map."""
        if not self.cfg.standalone_enabled:
            self._stats["standalone_candidates"] = 0
            return []

        market_by_condition = {market.condition_id: market for market in markets}
        signals: list[Signal] = []
        for condition_id, sentiment in self._market_sentiment_cache.items():
            if not self._eligible_sentiment(
                sentiment,
                min_whales=self.cfg.standalone_min_whales,
                min_total_size=self.cfg.standalone_min_total_size,
            ):
                continue
            market = market_by_condition.get(condition_id)
            if not market or not market.active or market.closed or len(market.outcomes) < 2:
                continue

            bullish = sentiment["net_direction"] == "bullish"
            outcome = market.outcomes[0] if bullish else market.outcomes[1]
            entry_price = outcome.price
            if entry_price <= 0 or entry_price > self.cfg.standalone_max_entry_price:
                continue

            confidence = min(
                0.55
                + abs(sentiment["buy_ratio"] - 0.5) * 0.9
                + min(sentiment["whale_count"], 4) * 0.03,
                0.85,
            )
            if confidence < self.cfg.standalone_min_confidence:
                continue

            edge = max(4.0, min((confidence - max(entry_price, 0.01)) * 100.0, 35.0))
            size_usd = min(
                self.cfg.standalone_max_size_usd,
                max(
                    self.cfg.standalone_min_size_usd,
                    15.0 + sentiment["whale_count"] * 5.0 + min(sentiment["total_volume"] / 150.0, 20.0),
                ),
            )

            signals.append(
                Signal(
                    source=SignalSource.WHALE,
                    action=SignalAction.BUY_YES if bullish else SignalAction.BUY_NO,
                    market_slug=market.slug,
                    condition_id=market.condition_id,
                    token_id=outcome.token_id,
                    confidence=confidence,
                    expected_edge=edge,
                    reasoning=(
                        f"WHALE FOLLOW: {sentiment['whale_count']} whales {sentiment['net_direction']} "
                        f"| buy_ratio={sentiment['buy_ratio']:.2f} | size={sentiment['total_volume']:.1f}"
                    ),
                    suggested_size_usd=size_usd,
                    group_key=f"whale:{market.slug}",
                )
            )

        self._stats["signals_generated"] += len(signals)
        self._stats["standalone_candidates"] = len(signals)
        return signals

    def confirm_signal(self, signal: Signal, whale_data: dict) -> Signal:
        """Apply whale confirmation to a signal."""
        if whale_data["net_direction"] == "neutral":
            return signal

        # Check if whale direction matches signal direction
        is_bullish_signal = signal.action in ("buy_yes", "hedge_both")
        whale_agrees = (
            (is_bullish_signal and whale_data["net_direction"] == "bullish") or
            (not is_bullish_signal and whale_data["net_direction"] == "bearish")
        )

        if whale_agrees:
            signal.whale_confirmed = True
            signal.confidence = min(
                signal.confidence * whale_data["confidence_multiplier"], 1.0
            )
            signal.reasoning += (
                f" | Whale confirmed: {whale_data['whale_count']} whales "
                f"{whale_data['net_direction']} (vol: ${whale_data['total_volume']:.0f})"
            )
        else:
            penalty = 1.0 / max(whale_data["confidence_multiplier"], 1.01)
            signal.confidence *= max(penalty, 0.4)
            signal.reasoning += (
                f" | Whale caution: {whale_data['whale_count']} whales "
                f"{whale_data['net_direction']} — opposing (penalty: {penalty:.2f}x)"
            )

        return signal

    def _should_refresh(self) -> bool:
        if self._last_refresh is None:
            return True
        elapsed = datetime.now(timezone.utc) - self._last_refresh
        return elapsed > timedelta(hours=self.cfg.refresh_interval_hours)

    def _should_refresh_activity(self) -> bool:
        if self._last_activity_refresh is None:
            return True
        elapsed = datetime.now(timezone.utc) - self._last_activity_refresh
        return elapsed > timedelta(minutes=self.cfg.activity_refresh_minutes)

    def _eligible_sentiment(self, sentiment: dict, *, min_whales: int, min_total_size: float) -> bool:
        if sentiment.get("net_direction") not in {"bullish", "bearish"}:
            return False
        if sentiment.get("whale_count", 0) < min_whales:
            return False
        if float(sentiment.get("total_volume", 0.0) or 0.0) < min_total_size:
            return False
        return self._is_fresh(sentiment.get("last_seen"))

    def _is_fresh(self, seen_at: datetime | None) -> bool:
        if seen_at is None:
            return False
        age = datetime.now(timezone.utc) - seen_at
        return age <= timedelta(minutes=self.cfg.signal_ttl_minutes)

    @staticmethod
    def _coerce_float(value) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _parse_trade_time(value) -> datetime:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if value in (None, ""):
            return datetime.now(timezone.utc)
        text = str(value).strip()
        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            return datetime.fromisoformat(text)
        except ValueError:
            return datetime.now(timezone.utc)
