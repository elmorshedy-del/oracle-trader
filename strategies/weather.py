"""
Strategy: Weather Forecast Arbitrage
=====================================
Compares official NOAA weather forecasts to Polymarket weather/temperature
markets. When the scientific forecast disagrees with market prices, trades
on the forecast.

Proven strategy: one bot turned $1K → $24K trading London weather alone.

No API key needed — NOAA Weather API (api.weather.gov) is free.
"""

import asyncio
import httpx
import re
import logging
from datetime import datetime, timezone, timedelta
from data.models import Market, Event, Outcome, Signal, SignalSource, SignalAction
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

# NOAA forecast endpoints for cities Polymarket typically covers
# These return 7-day forecasts with temperature ranges
CITY_FORECAST_URLS = {
    "new-york": "https://api.weather.gov/gridpoints/OKX/33,37/forecast",
    "nyc": "https://api.weather.gov/gridpoints/OKX/33,37/forecast",
    "chicago": "https://api.weather.gov/gridpoints/LOT/75,72/forecast",
    "los-angeles": "https://api.weather.gov/gridpoints/LOX/154,44/forecast",
    "miami": "https://api.weather.gov/gridpoints/MFL/111,49/forecast",
    "london": None,  # NOAA doesn't cover London, would need UK Met Office
    "seoul": None,    # Would need KMA API
}

CITY_KEYWORDS = {
    "new-york": ["new york", "nyc", "new york city", "manhattan"],
    "chicago": ["chicago"],
    "los-angeles": ["los angeles"],
    "miami": ["miami"],
    "london": ["london"],
    "seoul": ["seoul"],
}

SUPPORTED_WEATHER_MARKET = re.compile(r"\bhighest temperature in\b", re.IGNORECASE)
TEMP_RANGE_PATTERNS = (
    re.compile(r'(\d+)\s*[-–]\s*(\d+)\s*(?:°|degrees|fahrenheit|celsius|f\b|c\b)', re.IGNORECASE),
    re.compile(r'(\d+)\s+to\s+(\d+)\s*(?:°|degrees|fahrenheit|celsius|f\b|c\b)', re.IGNORECASE),
    re.compile(r'between\s+(\d+)\s+and\s+(\d+)', re.IGNORECASE),
    re.compile(r'(\d+)\s*[-–]\s*(\d+)', re.IGNORECASE),
)
TEMP_EXACT_PATTERN = re.compile(r'be\s+(\d+)\s*(?:°|degrees)?\s*([fc])\b', re.IGNORECASE)
TEMP_ABOVE_PATTERNS = (
    re.compile(r'(?:above|over|exceed|higher than)\s+(\d+)', re.IGNORECASE),
    re.compile(r'(\d+)\s*(?:°|degrees)?\s*[fc]?\s*or higher', re.IGNORECASE),
)
TEMP_BELOW_PATTERNS = (
    re.compile(r'(?:below|under|lower than)\s+(\d+)', re.IGNORECASE),
    re.compile(r'(\d+)\s*(?:°|degrees)?\s*[fc]?\s*or below', re.IGNORECASE),
)
WEATHER_DISCOVERY_REFRESH_SECS = 1800
WEATHER_MARKET_LOOKAHEAD_DAYS = 4


class WeatherForecastStrategy(BaseStrategy):
    name = "weather_forecast"
    description = "Compare NOAA forecasts to Polymarket weather markets"

    def __init__(self, config, collector=None):
        super().__init__(config)
        self.cfg = config.weather
        self.collector = collector
        self.client = httpx.AsyncClient(
            timeout=15.0,
            headers={"User-Agent": "oracle-trader/1.0 (weather-bot)"}  # NOAA requires User-Agent
        )

        # Cache forecasts: city -> {forecast_data, fetched_at}
        self._forecasts: dict[str, dict] = {}
        self._matched_markets: list[dict] = []
        self._supplemental_markets: list[Market] = []
        self._last_forecast_fetch: float = 0
        self._last_market_scan: float = 0
        self._last_supplemental_fetch: float = 0
        self._stats.update({
            "matched_markets": 0,
            "supplemental_markets": 0,
            "forecast_cities": 0,
        })

    async def scan(self, markets: list[Market], events: list[Event]) -> list[Signal]:
        if not self.cfg.enabled:
            return []

        self._stats["scans_completed"] += 1
        best_signals: dict[str, Signal] = {}

        now = datetime.now(timezone.utc).timestamp()

        # Step 1: Refresh forecasts every 30 minutes (forecasts don't change fast)
        if now - self._last_forecast_fetch > 1800:
            await self._fetch_forecasts()
            self._last_forecast_fetch = now

        # Step 2: Match weather markets (every 5 minutes)
        if now - self._last_market_scan > 300:
            self._match_weather_markets(markets)
            self._last_market_scan = now

        # Step 3: Compare forecasts to market prices
        for match in self._matched_markets:
            market = match["market"]
            city = match["city"]
            temp_range = match.get("temp_range")  # e.g., (40, 45) for "40-45°F"

            if not market.outcomes or len(market.outcomes) < 2:
                continue

            forecast = self._forecasts.get(city)
            if not forecast:
                continue

            # Get forecast temperature for the matching day
            forecast_temp = self._get_forecast_temp(forecast, match.get("target_date"))
            if forecast_temp is None:
                continue

            yes_price = market.outcomes[0].price

            # Calculate forecast probability that temperature falls in this range
            if temp_range:
                forecast_prob = self._temp_in_range_probability(
                    forecast_temp, temp_range[0], temp_range[1]
                )
            else:
                # Generic temperature market — just compare direction
                continue

            # The edge: forecast says 80% probability but market says 30%
            edge = forecast_prob - yes_price

            if edge > self.cfg.min_edge:
                # Forecast says more likely than market — buy YES
                confidence = min(abs(edge) * 2, 0.95)

                signal = Signal(
                    source=SignalSource.WEATHER,
                    action=SignalAction.BUY_YES,
                    market_slug=market.slug,
                    condition_id=market.condition_id,
                    token_id=market.outcomes[0].token_id,
                    confidence=confidence,
                    expected_edge=edge * 100,
                    group_key=self._weather_group_key(city, match.get("target_date")),
                    reasoning=(
                        f"WEATHER: NOAA forecast {forecast_temp:.0f}°F for {city} | "
                        f"Range {temp_range[0]}-{temp_range[1]}°F: forecast prob={forecast_prob:.0%} "
                        f"vs market={yes_price:.0%} | Edge: {edge:.0%}"
                    ),
                    suggested_size_usd=min(
                        self.config.risk.max_position_usd * confidence,
                        self.config.risk.max_position_usd,
                    ),
                )
                self._track_best_signal(best_signals, signal)
                logger.debug(
                    f"[WEATHER] {city}: NOAA={forecast_temp:.0f}°F, "
                    f"range={temp_range}, prob={forecast_prob:.0%} vs market={yes_price:.0%} → BUY YES"
                )

            elif edge < -self.cfg.min_edge:
                # Forecast says less likely than market — buy NO
                no_price = market.outcomes[1].price if len(market.outcomes) > 1 else None
                if no_price is None:
                    continue

                confidence = min(abs(edge) * 2, 0.95)

                signal = Signal(
                    source=SignalSource.WEATHER,
                    action=SignalAction.BUY_NO,
                    market_slug=market.slug,
                    condition_id=market.condition_id,
                    token_id=market.outcomes[1].token_id if len(market.outcomes) > 1 else None,
                    confidence=confidence,
                    expected_edge=abs(edge) * 100,
                    group_key=self._weather_group_key(city, match.get("target_date")),
                    reasoning=(
                        f"WEATHER: NOAA forecast {forecast_temp:.0f}°F for {city} | "
                        f"Range {temp_range[0]}-{temp_range[1]}°F: forecast prob={forecast_prob:.0%} "
                        f"vs market={yes_price:.0%} | Market overpriced → BUY NO"
                    ),
                    suggested_size_usd=min(
                        self.config.risk.max_position_usd * confidence,
                        self.config.risk.max_position_usd,
                    ),
                )
                self._track_best_signal(best_signals, signal)
                logger.debug(
                    f"[WEATHER] {city}: NOAA={forecast_temp:.0f}°F, "
                    f"range={temp_range}, market overpriced → BUY NO"
                )

        signals = sorted(
            best_signals.values(),
            key=lambda signal: (signal.expected_edge, signal.confidence),
            reverse=True,
        )
        self._stats["signals_generated"] += len(signals)

        if signals:
            logger.info(
                "[WEATHER] Generated %s signals from %s matched markets",
                len(signals),
                len(self._matched_markets),
            )
        if not signals and self._stats["scans_completed"] % 30 == 0:
            logger.info(
                f"[WEATHER] Status: {len(self._forecasts)} city forecasts, "
                f"{len(self._matched_markets)} matched markets, 0 signals"
            )

        return signals

    def _weather_group_key(self, city: str, target_date: str | None) -> str:
        return f"weather:{city}:{target_date or 'unknown'}"

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

    async def get_supplemental_markets(self) -> list[Market]:
        """Fetch weather markets that the generic active-market feed misses."""
        if not self.cfg.enabled:
            return []

        now = datetime.now(timezone.utc).timestamp()
        if (
            self._supplemental_markets
            and now - self._last_supplemental_fetch < WEATHER_DISCOVERY_REFRESH_SECS
        ):
            return list(self._supplemental_markets)

        discovered = await self._discover_weather_markets()
        self._supplemental_markets = discovered
        self._last_supplemental_fetch = now
        self._stats["supplemental_markets"] = len(discovered)
        return list(discovered)

    async def _fetch_forecasts(self):
        """Fetch weather forecasts from NOAA for all configured cities."""
        for city in self.cfg.cities:
            url = CITY_FORECAST_URLS.get(city)
            if url is None:
                continue

            try:
                resp = await self.client.get(url)
                resp.raise_for_status()
                data = resp.json()
                periods = data.get("properties", {}).get("periods", [])

                if periods:
                    daily_highs: dict[str, float] = {}
                    for period in periods:
                        start = period.get("startTime", "")
                        temp = period.get("temperature")
                        if temp is None or not start:
                            continue
                        date_key = start[:10]
                        temp_f = float(temp)
                        if period.get("temperatureUnit", "F") == "C":
                            temp_f = (temp_f * 9 / 5) + 32
                        daily_highs[date_key] = max(daily_highs.get(date_key, -999), temp_f)

                    self._forecasts[city] = {
                        "periods": periods,
                        "daily_highs": daily_highs,
                        "fetched_at": datetime.now(timezone.utc),
                    }
                    today = periods[0] if periods else {}
                    logger.debug(
                        f"[WEATHER] {city}: {today.get('name', '?')} "
                        f"temp={today.get('temperature', '?')}°{today.get('temperatureUnit', 'F')} "
                        f"({today.get('shortForecast', '?')})"
                    )
            except Exception as e:
                logger.debug(f"[WEATHER] Failed to fetch forecast for {city}: {e}")

        self._stats["forecast_cities"] = len(self._forecasts)
        logger.info(f"[WEATHER] Fetched forecasts for {len(self._forecasts)} cities")

    def _match_weather_markets(self, markets: list[Market]):
        """Find Polymarket markets related to weather/temperature."""
        self._matched_markets = []

        for market in markets:
            if market.closed or not market.active:
                continue

            text = f"{market.slug} {market.question}".lower()

            if not self._is_weather_market(text):
                continue

            matched_city = self._match_city(text)
            if not matched_city:
                continue

            # Try to extract temperature range from the market question
            temp_range = self._extract_temp_range(market.question)

            # Try to extract target date
            target_date = self._extract_date(market.question)

            if temp_range is None or target_date is None:
                continue

            self._matched_markets.append({
                "market": market,
                "city": matched_city,
                "temp_range": temp_range,
                "target_date": target_date,
            })

        self._stats["matched_markets"] = len(self._matched_markets)
        if self._matched_markets:
            logger.info(f"[WEATHER] Matched {len(self._matched_markets)} weather markets")
        else:
            logger.info("[WEATHER] No weather markets found on Polymarket right now")

    def _extract_temp_range(self, question: str) -> tuple[float, float] | None:
        """Extract temperature range from market question."""
        # Match patterns like "40-45", "40 to 45", "between 40 and 45"
        question_lower = question.lower()
        for pattern in TEMP_RANGE_PATTERNS:
            match = pattern.search(question_lower)
            if match:
                low = float(match.group(1))
                high = float(match.group(2))
                if 0 <= low <= 150 and 0 <= high <= 150:  # sanity check
                    return (low, high)

        for pattern in TEMP_ABOVE_PATTERNS:
            above_match = pattern.search(question_lower)
            if above_match:
                threshold = float(above_match.group(1))
                return (threshold, threshold + 100)  # open-ended range

        for pattern in TEMP_BELOW_PATTERNS:
            below_match = pattern.search(question_lower)
            if below_match:
                threshold = float(below_match.group(1))
                return (-50, threshold)  # open-ended range

        exact_match = TEMP_EXACT_PATTERN.search(question_lower)
        if exact_match:
            exact = float(exact_match.group(1))
            return (exact - 0.5, exact + 0.5)

        return None

    def _extract_date(self, question: str) -> str | None:
        """Try to extract a target date from the market question."""
        # Match patterns like "March 7", "March 7th", "3/7"
        months = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
        }

        for month_name, month_num in months.items():
            match = re.search(
                rf'{month_name}\s+(\d{{1,2}})',
                question.lower()
            )
            if match:
                day = int(match.group(1))
                year = datetime.now().year
                try:
                    return datetime(year, month_num, day).strftime("%Y-%m-%d")
                except ValueError:
                    pass

        # Match "tomorrow", "today"
        if "tomorrow" in question.lower():
            return (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        if "today" in question.lower():
            return datetime.now().strftime("%Y-%m-%d")

        return None

    def _get_forecast_temp(self, forecast: dict, target_date: str | None = None) -> float | None:
        """Get the forecasted temperature for a specific day."""
        if target_date:
            daily_highs = forecast.get("daily_highs", {})
            if target_date in daily_highs:
                return float(daily_highs[target_date])

        periods = forecast.get("periods", [])
        if not periods:
            return None

        if target_date:
            # Try to match the target date
            for period in periods:
                start = period.get("startTime", "")
                if target_date in start:
                    temp = period.get("temperature")
                    if temp is not None:
                        return float(temp)

        # Default: return the next daytime period's temperature
        for period in periods:
            if period.get("isDaytime", True):
                temp = period.get("temperature")
                if temp is not None:
                    return float(temp)

        # Last resort: first period
        if periods:
            temp = periods[0].get("temperature")
            if temp is not None:
                return float(temp)

        return None

    def _temp_in_range_probability(
        self, forecast_temp: float, range_low: float, range_high: float
    ) -> float:
        """
        Estimate probability that actual temperature falls in range,
        given NOAA forecast temperature.

        NOAA forecasts are typically within ±3°F for 1-day forecasts,
        ±5°F for 2-day forecasts. We model this as a normal distribution.
        """
        import math

        # Standard deviation of NOAA forecast error (°F)
        std_dev = 3.0  # for 1-day forecast

        # Center of the range
        center = forecast_temp

        # Cumulative normal distribution (approximation)
        def norm_cdf(x):
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

        # P(range_low <= T <= range_high) where T ~ N(forecast_temp, std_dev²)
        z_low = (range_low - center) / std_dev
        z_high = (range_high - center) / std_dev

        probability = norm_cdf(z_high) - norm_cdf(z_low)

        return max(0.01, min(0.99, probability))

    async def _discover_weather_markets(self) -> list[Market]:
        """Page active events and extract upcoming temperature markets."""
        gamma_host = self.config.api.gamma_host.rstrip("/")
        supported_cities = {
            city for city in self.cfg.cities if CITY_FORECAST_URLS.get(city)
        }
        discovered: dict[str, Market] = {}
        offset = 0
        limit = 100
        now = datetime.now(timezone.utc)
        latest_end = now + timedelta(days=WEATHER_MARKET_LOOKAHEAD_DAYS)

        while True:
            try:
                resp = await self.client.get(
                    f"{gamma_host}/events",
                    params={
                        "limit": limit,
                        "offset": offset,
                        "active": "true",
                        "closed": "false",
                    },
                )
                resp.raise_for_status()
                raw_events = resp.json()
            except Exception as e:
                logger.debug("[WEATHER] Supplemental event fetch failed at offset=%s: %s", offset, e)
                break

            if not raw_events:
                break

            for raw_event in raw_events:
                title = raw_event.get("title", "")
                slug = raw_event.get("slug", "")
                event_text = f"{slug} {title}".lower()
                if not self._is_weather_market(event_text):
                    continue

                matched_city = self._match_city(event_text)
                if not matched_city or matched_city not in supported_cities:
                    continue

                for raw_market in raw_event.get("markets", []):
                    market = self._parse_raw_market(raw_market)
                    if not market or market.condition_id in discovered:
                        continue
                    if market.closed or not market.active:
                        continue
                    if not self._is_weather_market(f"{market.slug} {market.question}".lower()):
                        continue
                    if self._match_city(f"{market.slug} {market.question}".lower()) != matched_city:
                        continue
                    if not market.end_date:
                        continue
                    try:
                        end_dt = datetime.fromisoformat(market.end_date.replace("Z", "+00:00"))
                    except ValueError:
                        continue
                    if end_dt < now or end_dt > latest_end:
                        continue
                    discovered[market.condition_id] = market

            if len(raw_events) < limit:
                break
            offset += limit
            await asyncio.sleep(0.05)

        if discovered:
            logger.info("[WEATHER] Supplemental discovery found %s active weather markets", len(discovered))
        else:
            logger.info("[WEATHER] Supplemental discovery found no upcoming weather markets")

        return list(discovered.values())

    def _parse_raw_market(self, raw_market: dict) -> Market | None:
        """Parse a raw Gamma market payload into our internal Market model."""
        tokens = self._coerce_str_list(raw_market.get("clobTokenIds"))
        prices = self._coerce_str_list(raw_market.get("outcomePrices"))
        outcome_names = self._coerce_str_list(raw_market.get("outcomes"))

        if not tokens:
            return None

        outcomes: list[Outcome] = []
        for index, token_id in enumerate(tokens):
            price = 0.0
            if index < len(prices):
                try:
                    price = float(prices[index])
                except (TypeError, ValueError):
                    price = 0.0
            name = outcome_names[index] if index < len(outcome_names) else f"Outcome {index + 1}"
            outcomes.append(
                Outcome(
                    token_id=token_id,
                    name=name,
                    price=price,
                )
            )

        if not outcomes:
            return None

        return Market(
            condition_id=raw_market.get("conditionId", raw_market.get("condition_id", "")),
            question=raw_market.get("question", ""),
            slug=raw_market.get("slug", ""),
            outcomes=outcomes,
            volume_24h=float(raw_market.get("volume24hr", 0) or 0),
            volume_total=float(raw_market.get("volumeNum", 0) or 0),
            liquidity=float(raw_market.get("liquidity", 0) or 0),
            spread=float(raw_market.get("spread", 0) or 0),
            end_date=raw_market.get("endDate") or raw_market.get("end_date_iso"),
            active=raw_market.get("active", True),
            closed=raw_market.get("closed", False),
            neg_risk=raw_market.get("negRisk", False),
            tags=[],
        )

    def _coerce_str_list(self, value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip().strip('"') for item in value if str(item).strip()]
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("[") and text.endswith("]"):
                text = text[1:-1]
            parts = [part.strip().strip('"') for part in text.split(",") if part.strip()]
            return parts
        return []

    def _is_weather_market(self, text: str) -> bool:
        return bool(SUPPORTED_WEATHER_MARKET.search(text))

    def _match_city(self, text: str) -> str | None:
        for city, keywords in CITY_KEYWORDS.items():
            for keyword in keywords:
                if re.search(rf"\b{re.escape(keyword)}\b", text):
                    return city
        return None
