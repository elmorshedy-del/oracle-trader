"""
Strategy: Weather Forecast Arbitrage
=====================================
Compares official NOAA weather forecasts to Polymarket weather/temperature
markets. When the scientific forecast disagrees with market prices, trades
on the forecast.

Proven strategy: one bot turned $1K → $24K trading London weather alone.

No API key needed — NOAA Weather API (api.weather.gov) is free.
"""

import httpx
import re
import logging
from datetime import datetime, timezone, timedelta
from data.models import Market, Event, Signal, SignalSource, SignalAction
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

# Keywords to find weather markets on Polymarket
WEATHER_KEYWORDS = [
    "temperature", "weather", "degrees", "fahrenheit", "celsius",
    "high temp", "low temp", "hot", "cold", "freeze", "snow", "rain",
]

CITY_KEYWORDS = {
    "new-york": ["new york", "nyc", "manhattan"],
    "chicago": ["chicago"],
    "los-angeles": ["los angeles", "la"],
    "miami": ["miami"],
    "london": ["london"],
    "seoul": ["seoul"],
}


class WeatherForecastStrategy(BaseStrategy):
    name = "weather_forecast"
    description = "Compare NOAA forecasts to Polymarket weather markets"

    def __init__(self, config):
        super().__init__(config)
        self.cfg = config.weather
        self.client = httpx.AsyncClient(
            timeout=15.0,
            headers={"User-Agent": "oracle-trader/1.0 (weather-bot)"}  # NOAA requires User-Agent
        )

        # Cache forecasts: city -> {forecast_data, fetched_at}
        self._forecasts: dict[str, dict] = {}
        self._matched_markets: list[dict] = []
        self._last_forecast_fetch: float = 0
        self._last_market_scan: float = 0

    async def scan(self, markets: list[Market], events: list[Event]) -> list[Signal]:
        if not self.cfg.enabled:
            return []

        self._stats["scans_completed"] += 1
        signals = []

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
                signals.append(signal)
                self._stats["signals_generated"] += 1
                logger.info(
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
                signals.append(signal)
                self._stats["signals_generated"] += 1
                logger.info(
                    f"[WEATHER] {city}: NOAA={forecast_temp:.0f}°F, "
                    f"range={temp_range}, market overpriced → BUY NO"
                )

        if not signals and self._stats["scans_completed"] % 30 == 0:
            logger.info(
                f"[WEATHER] Status: {len(self._forecasts)} city forecasts, "
                f"{len(self._matched_markets)} matched markets, 0 signals"
            )

        return signals

    async def _fetch_forecasts(self):
        """Fetch weather forecasts from NOAA for all configured cities."""
        for city, url in CITY_FORECAST_URLS.items():
            if url is None:
                continue

            try:
                resp = await self.client.get(url)
                resp.raise_for_status()
                data = resp.json()
                periods = data.get("properties", {}).get("periods", [])

                if periods:
                    self._forecasts[city] = {
                        "periods": periods,
                        "fetched_at": datetime.now(timezone.utc),
                    }
                    # Log the next day's forecast
                    today = periods[0] if periods else {}
                    logger.info(
                        f"[WEATHER] {city}: {today.get('name', '?')} "
                        f"temp={today.get('temperature', '?')}°{today.get('temperatureUnit', 'F')} "
                        f"({today.get('shortForecast', '?')})"
                    )
            except Exception as e:
                logger.debug(f"[WEATHER] Failed to fetch forecast for {city}: {e}")

        logger.info(f"[WEATHER] Fetched forecasts for {len(self._forecasts)} cities")

    def _match_weather_markets(self, markets: list[Market]):
        """Find Polymarket markets related to weather/temperature."""
        self._matched_markets = []

        for market in markets:
            if market.closed or not market.active:
                continue

            text = f"{market.slug} {market.question}".lower()

            # Check if this is a weather market
            is_weather = any(kw in text for kw in WEATHER_KEYWORDS)
            if not is_weather:
                continue

            # Match to a city
            matched_city = None
            for city, keywords in CITY_KEYWORDS.items():
                if any(kw in text for kw in keywords):
                    matched_city = city
                    break

            if not matched_city:
                continue

            # Try to extract temperature range from the market question
            # Patterns like "40-45", "40 to 45", "between 40 and 45"
            temp_range = self._extract_temp_range(market.question)

            # Try to extract target date
            target_date = self._extract_date(market.question)

            self._matched_markets.append({
                "market": market,
                "city": matched_city,
                "temp_range": temp_range,
                "target_date": target_date,
            })

        if self._matched_markets:
            logger.info(f"[WEATHER] Matched {len(self._matched_markets)} weather markets")
        else:
            logger.info("[WEATHER] No weather markets found on Polymarket right now")

    def _extract_temp_range(self, question: str) -> tuple[float, float] | None:
        """Extract temperature range from market question."""
        # Match patterns like "40-45", "40 to 45", "between 40 and 45"
        patterns = [
            r'(\d+)\s*[-–]\s*(\d+)\s*(?:°|degrees|fahrenheit|celsius|f\b|c\b)',
            r'(\d+)\s+to\s+(\d+)\s*(?:°|degrees|fahrenheit|celsius|f\b|c\b)',
            r'between\s+(\d+)\s+and\s+(\d+)',
            r'(\d+)\s*[-–]\s*(\d+)',  # fallback: any range
        ]

        for pattern in patterns:
            match = re.search(pattern, question.lower())
            if match:
                low = float(match.group(1))
                high = float(match.group(2))
                if 0 <= low <= 150 and 0 <= high <= 150:  # sanity check
                    return (low, high)

        # Match "above X" or "over X"
        above_match = re.search(r'(?:above|over|exceed|higher than)\s+(\d+)', question.lower())
        if above_match:
            threshold = float(above_match.group(1))
            return (threshold, threshold + 100)  # open-ended range

        # Match "below X" or "under X"
        below_match = re.search(r'(?:below|under|lower than)\s+(\d+)', question.lower())
        if below_match:
            threshold = float(below_match.group(1))
            return (-50, threshold)  # open-ended range

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
