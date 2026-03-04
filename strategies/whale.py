"""
Strategy: Whale Wallet Tracking (Layer 3 — The Advisor)
======================================================
Tracks top-performing wallets and uses their activity as a
confirmation/rejection filter for signals from other strategies.

Does NOT generate standalone signals — instead it provides a
confidence multiplier.
"""

import logging
from datetime import datetime, timezone, timedelta
from data.models import Market, Event, Signal, WhaleWallet, SignalSource
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

    async def scan(self, markets: list[Market], events: list[Event]) -> list[Signal]:
        """Whale tracking doesn't generate signals — it enriches other signals."""
        self._stats["scans_completed"] += 1

        # Refresh whale rankings periodically
        if self._should_refresh():
            await self.refresh_whales()

        return []

    async def refresh_whales(self):
        """Refresh the whale wallet rankings from the leaderboard."""
        if not self.collector:
            return

        try:
            leaderboard = await self.collector.get_leaderboard(limit=self.cfg.top_n_wallets)
            self.whale_wallets = []

            for entry in leaderboard:
                wallet = WhaleWallet(
                    address=entry.get("proxyWallet", entry.get("address", "")),
                    name=entry.get("name") or entry.get("pseudonym"),
                    total_pnl=float(entry.get("cashPnl", 0) or 0),
                    win_rate=float(entry.get("winRate", 0) or 0),
                    total_trades=int(entry.get("numTrades", 0) or 0),
                )

                if (wallet.total_pnl >= self.cfg.min_pnl_usd and
                        wallet.win_rate >= self.cfg.min_win_rate):
                    self.whale_wallets.append(wallet)

            self._last_refresh = datetime.now(timezone.utc)
            self._stats["whale_count"] = len(self.whale_wallets)
            logger.info(f"[WHALE] Refreshed: tracking {len(self.whale_wallets)} whales")

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"[WHALE] Failed to refresh: {e}")

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
