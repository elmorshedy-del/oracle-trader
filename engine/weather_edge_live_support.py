from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any

from data.models import Market
from engine.weather_edge_config import CITY_REGION_OVERRIDES, WEATHER_MODEL_IDS


UTC = timezone.utc
WEATHER_EDGE_LIVE_ALLOWED_LEAD_TIMES_HOURS = (48, 24)
WEATHER_EDGE_LIVE_SLOT_TOLERANCE_HOURS = 3.0
WEATHER_EDGE_LIVE_MIN_RESOLVED_PRICE = 0.99
WEATHER_EDGE_LIVE_METRIC = "temperature"
WEATHER_EDGE_SNAPSHOT_FORECAST_DAYS = 7
WEATHER_EDGE_SNAPSHOT_CALL_DELAY_SECONDS = 0.5
WEATHER_EDGE_SNAPSHOT_ALERT_FAILURE_RATIO = 0.20
WEATHER_EDGE_SNAPSHOT_MODELS = tuple(WEATHER_MODEL_IDS.values())
WEATHER_EDGE_SNAPSHOT_DAILY_FIELDS = (
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "snowfall_sum",
)
WEATHER_EDGE_SNAPSHOT_FALLBACK_CITIES = (
    "New York City",
    "Los Angeles",
    "Chicago",
    "Houston",
    "Phoenix",
    "Philadelphia",
    "San Antonio",
    "San Diego",
    "Dallas",
    "Miami",
    "Atlanta",
    "Boston",
    "San Francisco",
    "Seattle",
    "Denver",
    "Nashville",
    "Portland",
    "Las Vegas",
    "Detroit",
    "Minneapolis",
    "Tampa",
    "Orlando",
    "Sacramento",
    "Cleveland",
    "Kansas City",
    "Salt Lake City",
    "Honolulu",
    "Anchorage",
    "New Orleans",
    "Charlotte",
)

WEATHER_CITY_CAPTURE_PATTERN = re.compile(r"highest temperature in\s+(.+?)\s+on\b", re.IGNORECASE)
WEATHER_CITY_NORMALIZATION = {
    "nyc": "New York City",
    "new york": "New York City",
    "new york city": "New York City",
    "la": "Los Angeles",
}


def parse_market_end_datetime(value: Any) -> datetime | None:
    if value in (None, "", "null"):
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def determine_live_lead_time_hours(
    *,
    resolution_time: datetime | None,
    now: datetime,
) -> int | None:
    if resolution_time is None:
        return None
    hours_to_resolution = (resolution_time - now).total_seconds() / 3600.0
    best_slot = None
    best_distance = math.inf
    for lead_time_hours in WEATHER_EDGE_LIVE_ALLOWED_LEAD_TIMES_HOURS:
        distance = abs(hours_to_resolution - lead_time_hours)
        if distance <= WEATHER_EDGE_LIVE_SLOT_TOLERANCE_HOURS and distance < best_distance:
            best_slot = lead_time_hours
            best_distance = distance
    return best_slot


def compute_binary_kelly_fraction(*, your_probability: float, market_probability: float, max_fraction: float) -> float:
    if market_probability <= 0 or market_probability >= 1:
        return 0.0
    payout_ratio = (1.0 / market_probability) - 1.0
    lose_probability = 1.0 - your_probability
    raw_fraction = ((payout_ratio * your_probability) - lose_probability) / payout_ratio
    return min(max_fraction, max(0.0, raw_fraction * 0.25))


def canonical_weather_city_name(raw_city: str) -> str:
    compact = " ".join(str(raw_city or "").strip().split())
    normalized = WEATHER_CITY_NORMALIZATION.get(compact.lower())
    if normalized:
        return normalized
    return compact.title()


def region_for_city_name(city: str) -> str:
    return CITY_REGION_OVERRIDES.get(city, "unknown")


def extract_weather_city_name(question: str) -> str | None:
    match = WEATHER_CITY_CAPTURE_PATTERN.search(str(question or ""))
    if not match:
        return None
    return canonical_weather_city_name(match.group(1))


def collect_active_weather_cities(markets: list[Market]) -> list[str]:
    cities: list[str] = []
    seen: set[str] = set()
    for market in markets:
        city = extract_weather_city_name(market.question)
        if not city or city in seen:
            continue
        seen.add(city)
        cities.append(city)
    return cities


def best_resolution_yes_outcome(market: Market) -> bool | None:
    if len(market.outcomes) < 2:
        return None
    yes_price = float(market.outcomes[0].price or 0.0)
    no_price = float(market.outcomes[1].price or 0.0)
    if yes_price >= WEATHER_EDGE_LIVE_MIN_RESOLVED_PRICE:
        return True
    if no_price >= WEATHER_EDGE_LIVE_MIN_RESOLVED_PRICE:
        return False
    if market.closed and yes_price != no_price:
        return yes_price > no_price
    return None
