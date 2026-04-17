from __future__ import annotations

from pathlib import Path


DEFAULT_LOOKBACK_DAYS = 90
ENTRY_HORIZON_HOURS = (48, 24, 12, 6, 2)
REVISION_LOOKBACK_HOURS = 24
EXTENDED_HORIZON_HOURS = tuple(sorted(set(ENTRY_HORIZON_HOURS + (72,))))

WEATHER_MODEL_IDS = {
    "gfs": "gfs_seamless",
    "ecmwf": "ecmwf_ifs025",
    "icon": "icon_seamless",
    "gem": "gem_seamless",
    "jma": "jma_seamless",
}

OPEN_METEO_TEMPERATURE_DAILY_VARIABLE = "temperature_2m_max"
OPEN_METEO_ALLOWED_HOSTS = {
    "clob.polymarket.com",
    "geocoding-api.open-meteo.com",
    "historical-forecast-api.open-meteo.com",
    "previous-runs-api.open-meteo.com",
}

NETWORK_TIMEOUT_SECONDS = 30
PRICE_HISTORY_FIDELITY_MINUTES = 5
PRICE_HISTORY_INTERVAL = "max"

RULE_MIN_EDGE = 0.05
RULE_MIN_MODEL_AGREEMENT = 0.70
RULE_MIN_MARKET_VOLUME = 0.0
RULE_ALLOWED_METRICS = ("temperature",)
RULE_ALLOWED_LEAD_TIMES_HOURS = (48, 24)
RULE_ALLOWED_REGIONS_BY_LEAD_TIME = {
    24: ("coastal",),
    48: ("coastal", "inland", "unknown"),
}
RULE_SELECT_TOP_CONTRACT_PER_EVENT_HORIZON = True
KELLY_FRACTION = 0.25
MAX_POSITION_FRACTION = 0.02
DEFAULT_BANKROLL_USD = 1000.0

SPLIT_HALF_DAYS = 45
QUARTER_WINDOW_DAYS = 15
MIN_GROUP_ROWS = 10

CITY_REGION_OVERRIDES = {
    "Ankara": "inland",
    "Atlanta": "inland",
    "Buenos Aires": "coastal",
    "Chicago": "inland",
    "Dallas": "inland",
    "London": "coastal",
    "Lucknow": "inland",
    "Miami": "coastal",
    "Munich": "inland",
    "New York City": "coastal",
    "Paris": "inland",
    "Sao Paulo": "inland",
    "Seattle": "coastal",
    "Seoul": "coastal",
    "Toronto": "inland",
    "Wellington": "coastal",
}

SEASON_BY_MONTH = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "fall",
    10: "fall",
    11: "fall",
}


def default_weather_edge_root(repo_root: Path) -> Path:
    return repo_root / "output" / "weather_edge_v1"


def default_weather_research_root(repo_root: Path) -> Path:
    return repo_root / "research" / "weather"
