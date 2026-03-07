"""
Strategy: Crypto Market Dislocation
===================================
Watches BTC/ETH/SOL spot prices across public exchange feeds and trades
Polymarket crypto markets that look underpriced versus current spot context.

The strategy still supports short-window "Up/Down" contracts when they are
available, but it also covers the BTC/ETH/SOL barrier markets that dominate
the live book today: "reach $X", "dip to $X", and "all time high".

No API key needed — all configured quote sources are public.
"""

import httpx
import asyncio
import logging
import math
import re
import statistics
import time
from datetime import datetime, timezone
from data.models import Market, Event, Signal, SignalSource, SignalAction
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

# Public quote endpoints (no auth needed)
BINANCE_US_PRICES = {
    "BTC": "https://api.binance.us/api/v3/ticker/price?symbol=BTCUSDT",
    "ETH": "https://api.binance.us/api/v3/ticker/price?symbol=ETHUSDT",
    "SOL": "https://api.binance.us/api/v3/ticker/price?symbol=SOLUSDT",
}
COINBASE_PRICES = {
    "BTC": "https://api.coinbase.com/v2/prices/BTC-USD/spot",
    "ETH": "https://api.coinbase.com/v2/prices/ETH-USD/spot",
    "SOL": "https://api.coinbase.com/v2/prices/SOL-USD/spot",
}
KRAKEN_PAIRS = {
    "BTC": "XBTUSD",
    "ETH": "ETHUSD",
    "SOL": "SOLUSD",
}
COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,solana&vs_currencies=usd"
COINGECKO_IDS = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana"}
REQUEST_HEADERS = {"User-Agent": "oracle-trader/1.0"}

# Regex keyword matchers for Polymarket crypto markets
CRYPTO_MARKET_PATTERNS = {
    "BTC": re.compile(r"\b(?:btc|bitcoin)\b", re.IGNORECASE),
    "ETH": re.compile(r"\b(?:eth|ethereum)\b", re.IGNORECASE),
    "SOL": re.compile(r"\b(?:sol|solana)\b", re.IGNORECASE),
}

# Time-window keywords for short-duration crypto contracts
TIME_PATTERNS = ["5-minute", "5-min", "5m", "15-minute", "15-min", "15m", "1-hour", "1-hr", "hourly", "30-min"]
TEMPORAL_MARKET_HINTS = ("up or down", "updown", "up-down")
DIRECTION_UP = ("up", "higher", "increase", "above", "rise")
DIRECTION_DOWN = ("down", "lower", "decrease", "below", "drop", "fall")

# Barrier market parsing
UPPER_BARRIER_PATTERNS = (
    re.compile(r"(?:hit|reach|above)\s*\$?([\d,.]+)\s*([km]?)", re.IGNORECASE),
)
LOWER_BARRIER_PATTERNS = (
    re.compile(r"(?:dip\s+to|below)\s*\$?([\d,.]+)\s*([km]?)", re.IGNORECASE),
)
ANNUALIZED_VOLATILITY = {
    "BTC": 0.65,
    "ETH": 0.85,
    "SOL": 1.05,
}
ALL_TIME_HIGH_USD = {
    "BTC": 109358.0,
    "ETH": 4891.7,
    "SOL": 294.33,
}
TEMPORAL_CONFIDENCE_SCALE = 0.01
MAX_SIGNAL_CONFIDENCE = 0.95
MIN_THRESHOLD_EDGE = 0.12
MIN_THRESHOLD_HORIZON_HOURS = 6
MAX_THRESHOLD_HORIZON_DAYS = 400
MAX_MODELED_PROBABILITY = 0.98
MIN_THRESHOLD_CONFIDENCE = 0.55
MOVE_ALIGNMENT_BONUS = 0.08


class CryptoTemporalArbStrategy(BaseStrategy):
    name = "crypto_temporal_arb"
    description = "Exploit crypto spot-to-Polymarket dislocations across temporal and barrier markets"

    def __init__(self, config):
        super().__init__(config)
        self.cfg = config.crypto_arb
        self.client = httpx.AsyncClient(timeout=10.0)
        self._stats.update({
            "matched_markets": 0,
            "last_price_provider_count": 0,
            "last_price_providers": [],
            "last_price_error": "",
        })

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
        best_signals: dict[str, Signal] = {}

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

        # Step 4: For each crypto, evaluate matched Polymarket markets.
        for symbol, price in prices.items():
            move = self._calculate_move(symbol)
            move_pct = move[0] if move else 0.0
            direction = move[1] if move else "flat"

            # Step 5: Find matching Polymarket markets where price hasn't adjusted
            matched = self._matched_markets.get(symbol, [])
            for match in matched:
                signal = None
                if match["kind"] == "temporal":
                    signal = self._build_temporal_signal(
                        symbol=symbol,
                        match=match,
                        move_pct=move_pct,
                        direction=direction,
                    )
                else:
                    signal = self._build_barrier_signal(
                        symbol=symbol,
                        spot_price=price,
                        match=match,
                        move_pct=move_pct,
                        direction=direction,
                    )

                if signal:
                    self._track_best_signal(best_signals, signal)

        signals = sorted(
            best_signals.values(),
            key=lambda signal: (signal.expected_edge, signal.confidence),
            reverse=True,
        )
        self._stats["signals_generated"] += len(signals)

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
        """Fetch current spot prices from multiple public providers and aggregate."""
        providers = {
            "binance_us": self._fetch_binance_us_prices,
            "coinbase": self._fetch_coinbase_prices,
            "kraken": self._fetch_kraken_prices,
            "coingecko": self._fetch_coingecko_prices,
        }
        results = await asyncio.gather(
            *(fetcher() for fetcher in providers.values()),
            return_exceptions=True,
        )

        provider_names = list(providers.keys())
        prices_by_symbol = {symbol: [] for symbol in self.cfg.symbols}
        healthy_providers: list[str] = []
        failures: list[str] = []

        for provider_name, result in zip(provider_names, results):
            if isinstance(result, Exception):
                failures.append(f"{provider_name}:{type(result).__name__}")
                continue
            if not result:
                failures.append(f"{provider_name}:empty")
                continue
            healthy_providers.append(provider_name)
            for symbol, price in result.items():
                if symbol in prices_by_symbol and price > 0:
                    prices_by_symbol[symbol].append(price)

        prices = {
            symbol: round(statistics.median(values), 8)
            for symbol, values in prices_by_symbol.items()
            if values
        }

        self._stats["last_price_provider_count"] = len(healthy_providers)
        self._stats["last_price_providers"] = healthy_providers
        self._stats["last_price_error"] = "; ".join(failures[:4])

        if not prices:
            self._stats["errors"] += 1
            logger.error(
                "[CRYPTO] No spot prices available | %s",
                self._stats["last_price_error"] or "all providers unavailable",
            )
            return {}

        if self._stats["scans_completed"] % 10 == 0:
            logger.info(
                "[CRYPTO] Spot prices ready from %s providers | %s",
                len(healthy_providers),
                ", ".join(f"{symbol}=${price:,.2f}" for symbol, price in prices.items()),
            )

        return prices

    async def _fetch_binance_us_prices(self) -> dict[str, float]:
        prices = {}
        for symbol, url in BINANCE_US_PRICES.items():
            try:
                resp = await self.client.get(url, headers=REQUEST_HEADERS)
                resp.raise_for_status()
                prices[symbol] = float(resp.json()["price"])
            except Exception:
                continue
        return prices

    async def _fetch_coinbase_prices(self) -> dict[str, float]:
        prices = {}
        for symbol, url in COINBASE_PRICES.items():
            try:
                resp = await self.client.get(url, headers=REQUEST_HEADERS)
                resp.raise_for_status()
                prices[symbol] = float(resp.json()["data"]["amount"])
            except Exception:
                continue
        return prices

    async def _fetch_kraken_prices(self) -> dict[str, float]:
        prices = {}
        for symbol, pair in KRAKEN_PAIRS.items():
            try:
                resp = await self.client.get(
                    "https://api.kraken.com/0/public/Ticker",
                    params={"pair": pair},
                    headers=REQUEST_HEADERS,
                )
                resp.raise_for_status()
                data = resp.json().get("result", {})
                if not data:
                    continue
                first_pair = next(iter(data.values()))
                prices[symbol] = float(first_pair["c"][0])
            except Exception:
                continue
        return prices

    async def _fetch_coingecko_prices(self) -> dict[str, float]:
        prices = {}
        try:
            resp = await self.client.get(COINGECKO_URL, headers=REQUEST_HEADERS)
            resp.raise_for_status()
            data = resp.json()
            for symbol, coin_id in COINGECKO_IDS.items():
                usd = data.get(coin_id, {}).get("usd")
                if usd is not None:
                    prices[symbol] = float(usd)
        except Exception:
            return {}
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
        """Find active crypto markets that the strategy knows how to price."""
        self._matched_markets = {symbol: [] for symbol in self.cfg.symbols}
        now = datetime.now(timezone.utc)

        for market in markets:
            if market.closed or not market.active or self._is_market_expired(market, now):
                continue

            slug_lower = market.slug.lower()
            question_lower = market.question.lower()
            text = f"{slug_lower} {question_lower}"

            symbol = self._match_symbol(text)
            if not symbol:
                continue

            temporal_match = self._match_temporal_market(market, text)
            if temporal_match:
                self._matched_markets[symbol].append(temporal_match)
                continue

            barrier_match = self._match_barrier_market(symbol, market, text)
            if barrier_match:
                self._matched_markets[symbol].append(barrier_match)

        total = sum(len(v) for v in self._matched_markets.values())
        self._stats["matched_markets"] = total
        if total > 0:
            logger.info(
                f"[CRYPTO] Matched {total} crypto markets: "
                f"BTC={len(self._matched_markets['BTC'])}, "
                f"ETH={len(self._matched_markets['ETH'])}, "
                f"SOL={len(self._matched_markets['SOL'])}"
            )
        else:
            logger.info("[CRYPTO] No crypto markets matched the current scan")

    def _match_symbol(self, text: str) -> str | None:
        for symbol, pattern in CRYPTO_MARKET_PATTERNS.items():
            if pattern.search(text):
                return symbol
        return None

    def _match_temporal_market(self, market: Market, text: str) -> dict | None:
        is_time_market = any(pattern in text for pattern in TIME_PATTERNS) or any(
            hint in text for hint in TEMPORAL_MARKET_HINTS
        )
        if not is_time_market or len(market.outcomes) < 2:
            return None

        up_index, down_index = self._resolve_temporal_indices(market)
        up_price = market.outcomes[up_index].price
        down_price = market.outcomes[down_index].price
        if up_price <= 0 or down_price <= 0:
            return None

        return {
            "kind": "temporal",
            "market": market,
            "up_index": up_index,
            "down_index": down_index,
        }

    def _match_barrier_market(self, symbol: str, market: Market, text: str) -> dict | None:
        if len(market.outcomes) < 2 or " or " in text:
            return None

        yes_index, no_index = self._resolve_yes_no_indices(market)
        if yes_index is None or no_index is None:
            return None

        years_left = self._years_until_expiry(market.end_date)
        if years_left is None:
            return None

        horizon_hours = years_left * 365.25 * 24
        horizon_days = years_left * 365.25
        if horizon_hours < MIN_THRESHOLD_HORIZON_HOURS or horizon_days > MAX_THRESHOLD_HORIZON_DAYS:
            return None

        barrier_price = None
        kind = None

        if "all time high" in text:
            barrier_price = ALL_TIME_HIGH_USD.get(symbol)
            kind = "ath"
        else:
            barrier_price = self._extract_price_level(text, UPPER_BARRIER_PATTERNS)
            if barrier_price:
                kind = "reach"
            else:
                barrier_price = self._extract_price_level(text, LOWER_BARRIER_PATTERNS)
                if barrier_price:
                    kind = "dip"

        if not kind or not barrier_price:
            return None

        return {
            "kind": kind,
            "market": market,
            "barrier_price": barrier_price,
            "years_left": years_left,
            "yes_index": yes_index,
            "no_index": no_index,
        }

    def _resolve_temporal_indices(self, market: Market) -> tuple[int, int]:
        names = [outcome.name.lower() for outcome in market.outcomes[:2]]
        if "down" in names[0] and "up" in names[1]:
            return 1, 0
        return 0, 1

    def _resolve_yes_no_indices(self, market: Market) -> tuple[int, int] | tuple[None, None]:
        if len(market.outcomes) < 2:
            return None, None

        names = [outcome.name.lower() for outcome in market.outcomes[:2]]
        if names[0] == "yes" and names[1] == "no":
            return 0, 1
        if names[0] == "no" and names[1] == "yes":
            return 1, 0
        if len(market.outcomes) == 2:
            return 0, 1
        return None, None

    def _build_temporal_signal(
        self,
        *,
        symbol: str,
        match: dict,
        move_pct: float,
        direction: str,
    ) -> Signal | None:
        if abs(move_pct) < self.cfg.min_move_pct:
            return None

        market = match["market"]
        target_index = match["up_index"] if direction == "up" else match["down_index"]
        target_outcome = market.outcomes[target_index]
        target_price = target_outcome.price
        if target_price <= 0 or target_price >= self.cfg.max_entry_price:
            return None

        action = SignalAction.BUY_YES if target_index == 0 else SignalAction.BUY_NO
        edge = (1.0 - target_price) * abs(move_pct) / self.cfg.min_move_pct
        confidence = min(abs(move_pct) / TEMPORAL_CONFIDENCE_SCALE, MAX_SIGNAL_CONFIDENCE)
        signed_move = f"{move_pct:+.3%}"

        logger.info(
            "[CRYPTO] %s temporal %s -> %s on %s @ %.3f",
            symbol,
            signed_move,
            action.value,
            market.slug,
            target_price,
        )

        return Signal(
            source=SignalSource.CRYPTO_ARB,
            action=action,
            market_slug=market.slug,
            condition_id=market.condition_id,
            token_id=target_outcome.token_id,
            confidence=confidence,
            expected_edge=edge * 100,
            group_key=self._temporal_group_key(symbol, market, direction),
            reasoning=(
                f"CRYPTO TEMPORAL: {symbol} moved {signed_move} across spot feeds | "
                f"buying {target_outcome.name.upper()} at {target_price:.3f} on '{market.slug}'"
            ),
            suggested_size_usd=min(
                self.config.risk.max_position_usd * confidence,
                self.config.risk.max_position_usd,
            ),
        )

    def _build_barrier_signal(
        self,
        *,
        symbol: str,
        spot_price: float,
        match: dict,
        move_pct: float,
        direction: str,
    ) -> Signal | None:
        market = match["market"]
        yes_outcome = market.outcomes[match["yes_index"]]
        no_outcome = market.outcomes[match["no_index"]]
        yes_price = yes_outcome.price
        no_price = no_outcome.price
        if yes_price <= 0 or no_price <= 0:
            return None

        modeled_yes = self._estimate_barrier_probability(
            symbol=symbol,
            spot_price=spot_price,
            barrier_price=match["barrier_price"],
            years_left=match["years_left"],
        )
        if modeled_yes is None:
            return None

        action = None
        target_outcome = None
        side_probability = 0.0
        edge = 0.0
        if modeled_yes - yes_price >= MIN_THRESHOLD_EDGE and yes_price < self.cfg.max_entry_price:
            action = SignalAction.BUY_YES
            target_outcome = yes_outcome
            side_probability = modeled_yes
            edge = modeled_yes - yes_price
        elif yes_price - modeled_yes >= MIN_THRESHOLD_EDGE and no_price < self.cfg.max_entry_price:
            action = SignalAction.BUY_NO
            target_outcome = no_outcome
            side_probability = 1.0 - modeled_yes
            edge = yes_price - modeled_yes

        if not action or not target_outcome:
            return None

        aligns_with_move = (
            (match["kind"] in {"reach", "ath"} and direction == "up")
            or (match["kind"] == "dip" and direction == "down")
        )
        confidence = max(side_probability, MIN_THRESHOLD_CONFIDENCE)
        if aligns_with_move:
            confidence += MOVE_ALIGNMENT_BONUS
        confidence = min(confidence, MAX_SIGNAL_CONFIDENCE)

        logger.info(
            "[CRYPTO] %s %s %s -> %s on %s | model=%.1f%% market_yes=%.1f%%",
            symbol,
            match["kind"],
            f"${match['barrier_price']:,.0f}",
            action.value,
            market.slug,
            modeled_yes * 100,
            yes_price * 100,
        )

        return Signal(
            source=SignalSource.CRYPTO_ARB,
            action=action,
            market_slug=market.slug,
            condition_id=market.condition_id,
            token_id=target_outcome.token_id,
            confidence=confidence,
            expected_edge=edge * 100,
            group_key=self._barrier_group_key(symbol, market, match["kind"], action),
            reasoning=(
                f"CRYPTO BARRIER: {symbol} spot ${spot_price:,.2f} vs {match['kind']} "
                f"${match['barrier_price']:,.0f} on '{market.slug}' | "
                f"model={modeled_yes:.1%}, market YES={yes_price:.1%}, move={move_pct:+.3%}"
            ),
            suggested_size_usd=min(
                self.config.risk.max_position_usd * confidence,
                self.config.risk.max_position_usd,
            ),
        )

    def _estimate_barrier_probability(
        self,
        *,
        symbol: str,
        spot_price: float,
        barrier_price: float,
        years_left: float,
    ) -> float | None:
        if spot_price <= 0 or barrier_price <= 0 or years_left <= 0:
            return None

        volatility = ANNUALIZED_VOLATILITY.get(symbol)
        if not volatility:
            return None

        log_distance = abs(math.log(barrier_price / spot_price))
        scaled_vol = volatility * math.sqrt(years_left)
        if scaled_vol <= 0:
            return None

        z_score = log_distance / scaled_vol
        normal_cdf = 0.5 * (1 + math.erf(z_score / math.sqrt(2)))
        probability = 2 * (1 - normal_cdf)
        return max(0.0, min(MAX_MODELED_PROBABILITY, probability))

    def _extract_price_level(self, text: str, patterns: tuple[re.Pattern, ...]) -> float | None:
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                amount = float(match.group(1).replace(",", ""))
                suffix = match.group(2).lower()
                if suffix == "k":
                    amount *= 1_000
                elif suffix == "m":
                    amount *= 1_000_000
                return amount
        return None

    def _is_market_expired(self, market: Market, now: datetime) -> bool:
        if not market.end_date:
            return False
        try:
            end_dt = datetime.fromisoformat(market.end_date.replace("Z", "+00:00"))
        except ValueError:
            return False
        return end_dt <= now

    def _years_until_expiry(self, end_date: str | None) -> float | None:
        if not end_date:
            return None
        try:
            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        except ValueError:
            return None
        seconds_left = (end_dt - datetime.now(timezone.utc)).total_seconds()
        if seconds_left <= 0:
            return None
        return seconds_left / (365.25 * 24 * 3600)

    def _track_best_signal(self, best_signals: dict[str, Signal], signal: Signal):
        group_key = signal.group_key or signal.condition_id
        existing = best_signals.get(group_key)
        candidate_rank = (signal.expected_edge, signal.confidence)
        existing_rank = (
            (existing.expected_edge, existing.confidence)
            if existing
            else (-1.0, -1.0)
        )
        if existing is None or candidate_rank > existing_rank:
            best_signals[group_key] = signal

    def _temporal_group_key(self, symbol: str, market: Market, direction: str) -> str:
        expiry_bucket = (market.end_date or "unknown")[:10]
        return f"crypto:{symbol}:temporal:{direction}:{expiry_bucket}"

    def _barrier_group_key(
        self,
        symbol: str,
        market: Market,
        kind: str,
        action: SignalAction,
    ) -> str:
        expiry_bucket = (market.end_date or "unknown")[:10]
        bullish = (
            (kind in {"reach", "ath"} and action == SignalAction.BUY_YES)
            or (kind == "dip" and action == SignalAction.BUY_NO)
        )
        thesis = "bull" if bullish else "bear"
        return f"crypto:{symbol}:barrier:{thesis}:{expiry_bucket}"
