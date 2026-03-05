"""
Strategy: Crypto Temporal Arbitrage
====================================
Watches BTC/ETH/SOL spot prices on Binance and trades Polymarket's
15-minute up/down markets when exchange price confirms direction
but Polymarket hasn't repriced yet.

This is the strategy that turned $313 → $438K (0x8dxd bot).
Our version uses REST polling (slower) but still captures lagging markets.

No API key needed — Binance public API is free.
"""

import httpx
import asyncio
import logging
import time
from datetime import datetime, timezone
from data.models import Market, Event, Signal, SignalSource, SignalAction
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

# Binance spot price endpoints (no auth needed)
BINANCE_PRICES = {
    "BTC": "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
    "ETH": "https://api.binance.com/api/v3/ticker/price?symbol=ETHUSDT",
    "SOL": "https://api.binance.com/api/v3/ticker/price?symbol=SOLUSDT",
}

# Fallback: CoinGecko (if Binance is blocked)
COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,solana&vs_currencies=usd"

# Keywords to match Polymarket 15-min crypto markets
CRYPTO_MARKET_PATTERNS = {
    "BTC": ["btc", "bitcoin"],
    "ETH": ["eth", "ethereum"],
    "SOL": ["sol", "solana"],
}

# Time-window keywords (these markets are short-duration)
TIME_PATTERNS = ["15-minute", "15-min", "15m", "1-hour", "1-hr", "hourly", "30-min"]
DIRECTION_UP = ["up", "higher", "increase", "above", "rise"]
DIRECTION_DOWN = ["down", "lower", "decrease", "below", "drop", "fall"]


class CryptoTemporalArbStrategy(BaseStrategy):
    name = "crypto_temporal_arb"
    description = "Exploit exchange-to-Polymarket price lag on crypto 15-min markets"

    def __init__(self, config):
        super().__init__(config)
        self.cfg = config.crypto_arb
        self.client = httpx.AsyncClient(timeout=10.0)

        # Price tracking: symbol -> list of {price, timestamp}
        self._price_history: dict[str, list[dict]] = {
            "BTC": [], "ETH": [], "SOL": []
        }
        # Cache matched markets: symbol -> list of market data
        self._matched_markets: dict[str, list[dict]] = {}
        self._last_market_scan: float = 0

    async def scan(self, markets: list[Market], events: list[Event]) -> list[Signal]:
        if not self.cfg.enabled:
            return []

        self._stats["scans_completed"] += 1
        signals = []

        # Step 1: Fetch current exchange prices
        prices = await self._fetch_spot_prices()
        if not prices:
            return []

        # Step 2: Update price history
        now = time.time()
        for symbol, price in prices.items():
            self._price_history[symbol].append({"price": price, "time": now})
            # Keep last 20 minutes of data
            cutoff = now - 1200
            self._price_history[symbol] = [
                p for p in self._price_history[symbol] if p["time"] > cutoff
            ]

        # Step 3: Match Polymarket markets (refresh every 5 minutes)
        if now - self._last_market_scan > 300:
            self._match_crypto_markets(markets)
            self._last_market_scan = now

        # Step 4: For each crypto, check if exchange price confirms a direction
        for symbol, price in prices.items():
            move = self._calculate_move(symbol)
            if move is None:
                continue

            move_pct, direction = move

            # Need minimum move to be confident
            if abs(move_pct) < self.cfg.min_move_pct:
                continue

            # Step 5: Find matching Polymarket markets where price hasn't adjusted
            matched = self._matched_markets.get(symbol, [])
            for match in matched:
                market = match["market"]
                market_direction = match["direction"]  # "up" or "down"

                if not market.outcomes or len(market.outcomes) < 2:
                    continue

                yes_price = market.outcomes[0].price

                # The edge: exchange says UP but Polymarket YES-up is still cheap
                if direction == "up" and market_direction == "up" and yes_price < self.cfg.max_entry_price:
                    edge = (1.0 - yes_price) * abs(move_pct) / self.cfg.min_move_pct
                    confidence = min(abs(move_pct) / 0.01, 0.95)  # 1% move = max confidence

                    signal = Signal(
                        source=SignalSource.CRYPTO_ARB,
                        action=SignalAction.BUY_YES,
                        market_slug=market.slug,
                        condition_id=market.condition_id,
                        token_id=market.outcomes[0].token_id,
                        confidence=confidence,
                        expected_edge=edge * 100,
                        reasoning=(
                            f"CRYPTO ARB: {symbol} moved +{move_pct:.3%} on Binance | "
                            f"Polymarket '{market.slug}' YES={yes_price:.3f} (underpriced) | "
                            f"Exchange confirms UP"
                        ),
                        suggested_size_usd=min(
                            self.config.risk.max_position_usd * confidence,
                            self.config.risk.max_position_usd,
                        ),
                    )
                    signals.append(signal)
                    self._stats["signals_generated"] += 1
                    logger.info(f"[CRYPTO] {symbol} +{move_pct:.3%} → BUY YES on {market.slug} @ {yes_price:.3f}")

                elif direction == "down" and market_direction == "down" and yes_price < self.cfg.max_entry_price:
                    edge = (1.0 - yes_price) * abs(move_pct) / self.cfg.min_move_pct
                    confidence = min(abs(move_pct) / 0.01, 0.95)

                    signal = Signal(
                        source=SignalSource.CRYPTO_ARB,
                        action=SignalAction.BUY_YES,
                        market_slug=market.slug,
                        condition_id=market.condition_id,
                        token_id=market.outcomes[0].token_id,
                        confidence=confidence,
                        expected_edge=edge * 100,
                        reasoning=(
                            f"CRYPTO ARB: {symbol} moved {move_pct:.3%} on Binance | "
                            f"Polymarket '{market.slug}' YES={yes_price:.3f} (underpriced) | "
                            f"Exchange confirms DOWN"
                        ),
                        suggested_size_usd=min(
                            self.config.risk.max_position_usd * confidence,
                            self.config.risk.max_position_usd,
                        ),
                    )
                    signals.append(signal)
                    self._stats["signals_generated"] += 1
                    logger.info(f"[CRYPTO] {symbol} {move_pct:.3%} → BUY YES on {market.slug} @ {yes_price:.3f}")

                # Also: if exchange says UP but market is "down" market, buy NO (it won't go down)
                elif direction == "up" and market_direction == "down":
                    no_price = market.outcomes[1].price if len(market.outcomes) > 1 else None
                    if no_price and no_price < self.cfg.max_entry_price:
                        edge = (1.0 - no_price) * abs(move_pct) / self.cfg.min_move_pct
                        confidence = min(abs(move_pct) / 0.01, 0.95)

                        signal = Signal(
                            source=SignalSource.CRYPTO_ARB,
                            action=SignalAction.BUY_NO,
                            market_slug=market.slug,
                            condition_id=market.condition_id,
                            token_id=market.outcomes[1].token_id if len(market.outcomes) > 1 else None,
                            confidence=confidence,
                            expected_edge=edge * 100,
                            reasoning=(
                                f"CRYPTO ARB: {symbol} +{move_pct:.3%} on Binance | "
                                f"Polymarket '{market.slug}' is DOWN market, buying NO | "
                                f"Exchange confirms UP (won't go down)"
                            ),
                            suggested_size_usd=min(
                                self.config.risk.max_position_usd * confidence,
                                self.config.risk.max_position_usd,
                            ),
                        )
                        signals.append(signal)
                        self._stats["signals_generated"] += 1

                elif direction == "down" and market_direction == "up":
                    no_price = market.outcomes[1].price if len(market.outcomes) > 1 else None
                    if no_price and no_price < self.cfg.max_entry_price:
                        edge = (1.0 - no_price) * abs(move_pct) / self.cfg.min_move_pct
                        confidence = min(abs(move_pct) / 0.01, 0.95)

                        signal = Signal(
                            source=SignalSource.CRYPTO_ARB,
                            action=SignalAction.BUY_NO,
                            market_slug=market.slug,
                            condition_id=market.condition_id,
                            token_id=market.outcomes[1].token_id if len(market.outcomes) > 1 else None,
                            confidence=confidence,
                            expected_edge=edge * 100,
                            reasoning=(
                                f"CRYPTO ARB: {symbol} {move_pct:.3%} on Binance | "
                                f"Polymarket '{market.slug}' is UP market, buying NO | "
                                f"Exchange confirms DOWN (won't go up)"
                            ),
                            suggested_size_usd=min(
                                self.config.risk.max_position_usd * confidence,
                                self.config.risk.max_position_usd,
                            ),
                        )
                        signals.append(signal)
                        self._stats["signals_generated"] += 1

        if signals:
            logger.info(f"[CRYPTO] Generated {len(signals)} crypto arb signals")
        elif self._stats["scans_completed"] % 20 == 0:
            # Diagnostic logging
            for sym in ["BTC", "ETH", "SOL"]:
                n = len(self._price_history[sym])
                matched_n = len(self._matched_markets.get(sym, []))
                if n > 0:
                    latest = self._price_history[sym][-1]["price"]
                    logger.info(f"[CRYPTO] {sym}: ${latest:,.2f} | {n} price points | {matched_n} matched markets")

        return signals

    async def _fetch_spot_prices(self) -> dict[str, float]:
        """Fetch current spot prices from Binance (primary) or CoinGecko (fallback)."""
        prices = {}

        # Try Binance first (faster, more reliable)
        for symbol, url in BINANCE_PRICES.items():
            try:
                resp = await self.client.get(url)
                resp.raise_for_status()
                data = resp.json()
                prices[symbol] = float(data["price"])
            except Exception:
                pass

        if prices:
            return prices

        # Fallback to CoinGecko
        try:
            resp = await self.client.get(COINGECKO_URL)
            resp.raise_for_status()
            data = resp.json()
            if "bitcoin" in data:
                prices["BTC"] = float(data["bitcoin"]["usd"])
            if "ethereum" in data:
                prices["ETH"] = float(data["ethereum"]["usd"])
            if "solana" in data:
                prices["SOL"] = float(data["solana"]["usd"])
        except Exception as e:
            logger.error(f"[CRYPTO] Failed to fetch prices: {e}")
            self._stats["errors"] += 1

        return prices

    def _calculate_move(self, symbol: str) -> tuple[float, str] | None:
        """
        Calculate price move over the lookback window.
        Returns (move_pct, direction) or None if insufficient data.
        """
        history = self._price_history.get(symbol, [])
        if len(history) < 2:
            return None

        now = time.time()
        lookback = self.cfg.lookback_seconds

        # Find price from lookback_seconds ago
        old_prices = [p for p in history if p["time"] < now - lookback + 5]
        if not old_prices:
            return None

        old_price = old_prices[-1]["price"]
        current_price = history[-1]["price"]

        if old_price <= 0:
            return None

        move_pct = (current_price - old_price) / old_price
        direction = "up" if move_pct > 0 else "down"

        return (move_pct, direction)

    def _match_crypto_markets(self, markets: list[Market]):
        """Find Polymarket markets that match crypto 15-min up/down patterns."""
        self._matched_markets = {"BTC": [], "ETH": [], "SOL": []}

        for market in markets:
            if market.closed or not market.active:
                continue

            slug_lower = market.slug.lower()
            question_lower = market.question.lower()
            text = f"{slug_lower} {question_lower}"

            # Check if it's a time-windowed market
            is_time_market = any(tp in text for tp in TIME_PATTERNS)
            if not is_time_market:
                # Also match "updown" style slugs
                if "updown" not in slug_lower and "up-down" not in slug_lower:
                    continue

            # Match to crypto symbol
            for symbol, patterns in CRYPTO_MARKET_PATTERNS.items():
                if any(p in text for p in patterns):
                    # Determine if this is an "up" or "down" market
                    is_up = any(d in text for d in DIRECTION_UP)
                    is_down = any(d in text for d in DIRECTION_DOWN)

                    if is_up and not is_down:
                        self._matched_markets[symbol].append({
                            "market": market,
                            "direction": "up",
                        })
                    elif is_down and not is_up:
                        self._matched_markets[symbol].append({
                            "market": market,
                            "direction": "down",
                        })
                    else:
                        # Generic crypto market — treat as "up" if question implies it
                        self._matched_markets[symbol].append({
                            "market": market,
                            "direction": "up",
                        })

        total = sum(len(v) for v in self._matched_markets.values())
        if total > 0:
            logger.info(
                f"[CRYPTO] Matched {total} crypto markets: "
                f"BTC={len(self._matched_markets['BTC'])}, "
                f"ETH={len(self._matched_markets['ETH'])}, "
                f"SOL={len(self._matched_markets['SOL'])}"
            )
        else:
            logger.info("[CRYPTO] No 15-min crypto markets found on Polymarket right now")
