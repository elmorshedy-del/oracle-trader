from __future__ import annotations

import logging
from typing import Any

from config import BitcoinLatencyShadowConfig
from data.models import Event, Market, Signal, SignalSource
from strategies.base import BaseStrategy
from strategies.crypto_arb import CryptoTemporalArbStrategy


logger = logging.getLogger(__name__)

BTC_HOURLY_SLUG_HINT = "bitcoin-up-or-down"
BTC_TEXT_HINTS = ("bitcoin", "btc")


class BitcoinLatencyShadowStrategy(BaseStrategy):
    name = "bitcoin_latency_shadow"
    description = "Dedicated BTC hourly dislocation sleeve with its own paper budget"

    def __init__(self, config):
        super().__init__(config)
        self.cfg: BitcoinLatencyShadowConfig = config.bitcoin_latency_shadow
        self.inner = CryptoTemporalArbStrategy(config)
        self.inner.cfg.enabled = bool(self.cfg.enabled)
        self.inner.cfg.min_move_pct = float(self.cfg.min_move_pct)
        self.inner.cfg.lookback_seconds = int(self.cfg.lookback_seconds)
        self.inner.cfg.max_entry_price = float(self.cfg.max_entry_price)
        self.enabled = bool(self.cfg.enabled)
        self._stats.update(
            {
                "view_key": self.cfg.view_key,
                "source": self.cfg.source,
                "budget_usd": self.cfg.budget_usd,
                "min_move_pct": self.cfg.min_move_pct,
                "lookback_seconds": self.cfg.lookback_seconds,
                "max_entry_price": self.cfg.max_entry_price,
                "candidate_signals": 0,
                "emitted_signals": 0,
            }
        )

    async def scan(self, markets: list[Market], events: list[Event]) -> list[Signal]:
        if not self.enabled:
            return []

        self._stats["scans_completed"] += 1
        btc_markets = [
            market for market in markets
            if any(token in market.slug.lower() or token in market.question.lower() for token in BTC_TEXT_HINTS)
        ]
        raw_signals = await self.inner.scan(btc_markets, events)
        filtered = [
            signal for signal in raw_signals
            if signal.source == SignalSource.CRYPTO_ARB
            and BTC_HOURLY_SLUG_HINT in signal.market_slug.lower()
        ]
        self._stats["candidate_signals"] = len(filtered)

        emitted: list[Signal] = []
        for signal in filtered:
            size_usd = min(max(signal.suggested_size_usd, self.cfg.min_trade_usd), self.cfg.max_trade_usd)
            emitted.append(
                signal.model_copy(
                    update={
                        "source": SignalSource.BITCOIN_LATENCY_SHADOW,
                        "suggested_size_usd": size_usd,
                        "reasoning": f"BTC LATENCY SHADOW: {signal.reasoning}",
                    }
                )
            )

        self._stats["signals_generated"] += len(emitted)
        self._stats["emitted_signals"] = len(emitted)
        return emitted

    @property
    def stats(self) -> dict[str, Any]:
        inner_stats = self.inner.stats
        return {
            "name": self.name,
            "enabled": self.enabled,
            **self._stats,
            "inner_scans_completed": inner_stats.get("scans_completed", 0),
            "inner_matched_markets": inner_stats.get("matched_markets", 0),
            "inner_price_provider_count": inner_stats.get("last_price_provider_count", 0),
            "inner_last_price_error": inner_stats.get("last_price_error", ""),
        }

    async def close(self) -> None:
        await self.inner.client.aclose()
