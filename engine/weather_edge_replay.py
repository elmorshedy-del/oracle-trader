from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from engine.weather_edge_baseline import probability_from_temperature
from engine.weather_edge_config import (
    CITY_REGION_OVERRIDES,
    ENTRY_HORIZON_HOURS,
    NETWORK_TIMEOUT_SECONDS,
    OPEN_METEO_ALLOWED_HOSTS,
    OPEN_METEO_TEMPERATURE_DAILY_VARIABLE,
    PRICE_HISTORY_FIDELITY_MINUTES,
    PRICE_HISTORY_INTERVAL,
    REVISION_LOOKBACK_HOURS,
    SEASON_BY_MONTH,
    WEATHER_MODEL_IDS,
)


WEATHER_METRIC_TEMPERATURE = "temperature"
QUESTION_NUMBER_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)")
GEOCODE_ENDPOINT = "https://geocoding-api.open-meteo.com/v1/search"
POLYMARKET_PRICE_HISTORY_ENDPOINT = "https://clob.polymarket.com/prices-history"
OPEN_METEO_HISTORICAL_FORECAST_ENDPOINT = "https://historical-forecast-api.open-meteo.com/v1/forecast"


@dataclass
class WeatherHistorySources:
    baseline_bundle_frozen: Path
    source_dataset: Path
    market_metadata: Path
    forecast_features: Path
    multimodel_features: Path


class WeatherReplayStore:
    def __init__(self, history_sources: WeatherHistorySources):
        self.history_sources = history_sources
        self.metadata_by_market_id = _index_jsonl(history_sources.market_metadata, "market_id")
        self.forecast_by_key = _index_jsonl(
            history_sources.forecast_features,
            key_fn=lambda row: _forecast_key(row["city"], row["target_date"], row["temp_unit"]),
        )
        self.multimodel_by_key = _index_jsonl(
            history_sources.multimodel_features,
            key_fn=lambda row: _forecast_key(row["city"], row["target_date"], row["temp_unit"]),
        )

    def load_recent_resolved_markets(self, *, lookback_days: int) -> list[dict[str, Any]]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        markets: list[dict[str, Any]] = []
        with self.history_sources.source_dataset.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                end_date = _parse_datetime(row.get("end_date"))
                if end_date is None or end_date < cutoff:
                    continue

                normalized = self._normalize_market(row)
                if normalized is None:
                    continue
                markets.append(normalized)

        # Recent means most recently resolved first so sampled smoke runs hit the newest markets.
        markets.sort(key=lambda row: (row["resolution_time"], row["market_id"]), reverse=True)
        return markets

    def _normalize_market(self, row: dict[str, Any]) -> dict[str, Any] | None:
        market_id = str(row.get("market_id") or "").strip()
        city = str(row.get("city") or "").strip()
        target_date = str(row.get("target_date") or "").strip()
        temp_unit = str(row.get("temp_unit") or "C").upper()
        yes_token_id = str(row.get("yes_token_id") or "").strip()
        question = str(row.get("question") or "").strip()
        if not market_id or not city or not target_date or not yes_token_id or not question:
            return None

        temp_kind, temp_range_low, temp_range_high = _normalize_market_thresholds(
            question=question,
            temp_kind=row.get("temp_kind"),
            temp_range_low=row.get("temp_range_low"),
            temp_range_high=row.get("temp_range_high"),
        )
        if temp_kind is None or temp_range_low is None or temp_range_high is None:
            return None

        metadata = self.metadata_by_market_id.get(market_id, {})
        resolution_time = _parse_datetime(metadata.get("closed_time")) or _parse_datetime(row.get("end_date"))
        if resolution_time is None:
            return None

        key = _forecast_key(city, target_date, temp_unit)
        archive_weather = row.get("archive_weather") or {}
        return {
            "market_id": market_id,
            "question": question,
            "description": row.get("description") or "",
            "slug": row.get("slug") or "",
            "city": city,
            "metric_type": WEATHER_METRIC_TEMPERATURE,
            "target_date": target_date,
            "temp_unit": temp_unit,
            "temp_kind": temp_kind,
            "temp_range_low": temp_range_low,
            "temp_range_high": temp_range_high,
            "threshold_display": _threshold_display(temp_kind, temp_range_low, temp_range_high, temp_unit),
            "resolved_yes": int(row.get("resolved_yes") or 0),
            "yes_token_id": yes_token_id,
            "best_ask": _safe_float(row.get("best_ask")),
            "best_bid": _safe_float(row.get("best_bid")),
            "volume_clob": _safe_float(row.get("volume_clob")),
            "liquidity_clob": _safe_float(row.get("liquidity_clob")),
            "resolution_time": resolution_time,
            "season": SEASON_BY_MONTH.get(int(target_date[5:7]), "unknown"),
            "region": CITY_REGION_OVERRIDES.get(city, "unknown"),
            "created_at": metadata.get("created_at"),
            "closed_time": metadata.get("closed_time"),
            "lead_hours": _safe_float(metadata.get("lead_hours")),
            "spread": _safe_float(metadata.get("spread")),
            "market_metadata": metadata,
            "forecast_key": key,
            "local_forecast": self.forecast_by_key.get(key, {}),
            "local_multimodel": self.multimodel_by_key.get(key, {}),
            "archive_weather": archive_weather,
            "local_yes_price_history": row.get("yes_price_history") or [],
        }


class WeatherPriceHistoryClient:
    def __init__(self, cache_root: Path):
        self.cache_root = cache_root
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def load_price_history(
        self,
        *,
        token_id: str,
        allow_network: bool,
    ) -> tuple[list[dict[str, Any]], str]:
        cache_path = self.cache_root / f"{token_id}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text())
            return payload.get("history", []), payload.get("source", "cache")

        if not allow_network:
            return [], "unavailable_offline"

        try:
            params = {
                "market": token_id,
                "interval": PRICE_HISTORY_INTERVAL,
                "fidelity": PRICE_HISTORY_FIDELITY_MINUTES,
            }
            url = f"{POLYMARKET_PRICE_HISTORY_ENDPOINT}?{urllib.parse.urlencode(params)}"
            payload = _fetch_json(url)
            history = payload.get("history") or []
            cache_path.write_text(json.dumps({"source": "network", "history": history}))
            return history, "network"
        except Exception as exc:
            return [], f"network_error:{type(exc).__name__}"

    def horizon_probabilities(
        self,
        *,
        price_history: list[dict[str, Any]],
        resolution_time: datetime,
        horizons: tuple[int, ...] = ENTRY_HORIZON_HOURS,
    ) -> dict[int, dict[str, Any]]:
        points = _normalize_price_history(price_history)
        result: dict[int, dict[str, Any]] = {}
        for horizon in horizons:
            anchor = resolution_time - timedelta(hours=horizon)
            point = _last_point_at_or_before(points, anchor)
            if point is None:
                continue
            result[horizon] = {
                "timestamp": point["timestamp"].isoformat(),
                "market_yes_probability": point["price"],
            }
        return result


class OpenMeteoProxyClient:
    def __init__(self, cache_root: Path):
        self.cache_root = cache_root
        self.geocode_cache_dir = self.cache_root / "geocodes"
        self.forecast_cache_dir = self.cache_root / "historical_daily"
        self.geocode_cache_dir.mkdir(parents=True, exist_ok=True)
        self.forecast_cache_dir.mkdir(parents=True, exist_ok=True)

    def enrich_market(
        self,
        *,
        market: dict[str, Any],
        allow_network: bool,
    ) -> dict[str, Any]:
        local_forecast = market.get("local_forecast") or {}
        local_multimodel = market.get("local_multimodel") or {}
        enriched_models = self._load_local_models(local_multimodel)
        geocode = self._geocode_city(market["city"], allow_network=allow_network)
        historical_best = None
        if geocode and allow_network:
            try:
                historical_best = self._fetch_historical_daily_temperature(
                    latitude=geocode["latitude"],
                    longitude=geocode["longitude"],
                    target_date=market["target_date"],
                    temp_unit=market["temp_unit"],
                )
            except Exception:
                historical_best = None

        if historical_best is None:
            historical_best = _safe_float(local_forecast.get("forecast_temp_max"), default=None)

        consensus_probability, model_agreement, model_spread = _consensus_from_models(
            model_temperatures=enriched_models,
            temp_kind=market["temp_kind"],
            temp_range_low=market["temp_range_low"],
            temp_range_high=market["temp_range_high"],
        )

        return {
            "historical_temp_max": historical_best,
            "model_temperatures": enriched_models,
            "consensus_probability": consensus_probability,
            "model_agreement": model_agreement,
            "model_spread": model_spread,
            "revision_24h": None,
            "revision_basis_hours": None,
            "revision_direction_matches": None,
            "proxy_source": "local_artifacts" if geocode is None or historical_best == _safe_float(local_forecast.get("forecast_temp_max"), default=None) else "open_meteo_public",
        }

    def _geocode_city(self, city: str, *, allow_network: bool) -> dict[str, Any] | None:
        cache_path = self.geocode_cache_dir / f"{_slugify(city)}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())
        if not allow_network:
            return None
        try:
            url = f"{GEOCODE_ENDPOINT}?{urllib.parse.urlencode({'name': city, 'count': 1, 'language': 'en', 'format': 'json'})}"
            payload = _fetch_json(url)
            result = (payload.get("results") or [None])[0]
            if not result:
                return None
            cache_path.write_text(json.dumps(result))
            return result
        except Exception:
            return None

    def _fetch_historical_daily_temperature(
        self,
        *,
        latitude: float,
        longitude: float,
        target_date: str,
        temp_unit: str,
    ) -> float | None:
        cache_key = f"{latitude}_{longitude}_{target_date}_{temp_unit}.json".replace(".", "_")
        cache_path = self.forecast_cache_dir / cache_key
        if cache_path.exists():
            payload = json.loads(cache_path.read_text())
        else:
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": target_date,
                "end_date": target_date,
                "daily": OPEN_METEO_TEMPERATURE_DAILY_VARIABLE,
                "temperature_unit": "celsius" if temp_unit.upper() == "C" else "fahrenheit",
                "timezone": "UTC",
            }
            url = f"{OPEN_METEO_HISTORICAL_FORECAST_ENDPOINT}?{urllib.parse.urlencode(params)}"
            payload = _fetch_json(url)
            cache_path.write_text(json.dumps(payload))
        values = (payload.get("daily") or {}).get(OPEN_METEO_TEMPERATURE_DAILY_VARIABLE) or []
        if not values:
            return None
        return _safe_float(values[0], default=None)

    @staticmethod
    def _load_local_models(local_multimodel: dict[str, Any]) -> dict[str, float | None]:
        models: dict[str, float | None] = {}
        for open_meteo_model in WEATHER_MODEL_IDS.values():
            key = f"{open_meteo_model}_temp_max"
            if key in local_multimodel:
                models[open_meteo_model] = _safe_float(local_multimodel.get(key), default=None)
        if "best_match_temp_max" in local_multimodel:
            models["best_match"] = _safe_float(local_multimodel.get("best_match_temp_max"), default=None)
        return models


def load_history_sources(path: Path) -> WeatherHistorySources:
    payload = json.loads(path.read_text())
    artifacts = payload.get("artifacts") or {}
    return WeatherHistorySources(
        baseline_bundle_frozen=Path(payload["baseline_bundle_frozen"]),
        source_dataset=Path(payload["source_dataset"]),
        market_metadata=Path(artifacts["market_metadata"]),
        forecast_features=Path(artifacts["forecast_features"]),
        multimodel_features=Path(artifacts["multimodel_features"]),
    )


def build_market_horizon_rows(
    *,
    markets: list[dict[str, Any]],
    price_history_client: WeatherPriceHistoryClient,
    open_meteo_client: OpenMeteoProxyClient,
    allow_network: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for market in markets:
        price_history, price_history_source = price_history_client.load_price_history(
            token_id=market["yes_token_id"],
            allow_network=allow_network,
        )
        if not price_history and market.get("local_yes_price_history"):
            price_history = market["local_yes_price_history"]
            price_history_source = "local_market_artifact"
        price_rows = price_history_client.horizon_probabilities(
            price_history=price_history,
            resolution_time=market["resolution_time"],
        )
        if not price_rows:
            continue

        weather_proxy = open_meteo_client.enrich_market(market=market, allow_network=allow_network)
        for horizon, price_row in price_rows.items():
            rows.append(
                {
                    **market,
                    "lead_time_hours": horizon,
                    "market_yes_probability": price_row["market_yes_probability"],
                    "market_probability_timestamp": price_row["timestamp"],
                    "price_history_source": price_history_source,
                    "historical_temp_max": weather_proxy["historical_temp_max"],
                    "model_temperatures": weather_proxy["model_temperatures"],
                    "consensus_probability": weather_proxy["consensus_probability"],
                    "model_agreement": weather_proxy["model_agreement"],
                    "model_spread": weather_proxy["model_spread"],
                    "revision_24h": weather_proxy["revision_24h"],
                    "revision_basis_hours": weather_proxy["revision_basis_hours"],
                    "revision_direction_matches": weather_proxy["revision_direction_matches"],
                    "weather_proxy_source": weather_proxy["proxy_source"],
                }
            )
    return rows


def _fetch_json(url: str) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in OPEN_METEO_ALLOWED_HOSTS:
        raise ValueError(f"Refusing to fetch non-allowlisted URL: {url}")
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (compatible; oracle-weather-edge/1.0)",
        },
    )
    with urllib.request.urlopen(request, timeout=NETWORK_TIMEOUT_SECONDS) as response:
        return json.loads(response.read())


def _index_jsonl(path: Path, key: str | None = None, key_fn=None) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row_key = row.get(key) if key else key_fn(row)
            if row_key is None:
                continue
            rows[str(row_key)] = row
    return rows


def _forecast_key(city: str, target_date: str, temp_unit: str) -> str:
    return f"{city}::{target_date}::{temp_unit.upper()}"


def _parse_datetime(value: Any) -> datetime | None:
    if value in (None, "", "null"):
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except Exception:
            return None


def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value in (None, "", "null"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _normalize_market_thresholds(
    *,
    question: str,
    temp_kind: Any,
    temp_range_low: Any,
    temp_range_high: Any,
) -> tuple[str | None, float | None, float | None]:
    kind = str(temp_kind).strip().lower() if temp_kind not in (None, "") else None
    low = _safe_float(temp_range_low, default=None)
    high = _safe_float(temp_range_high, default=None)

    if kind and low is not None and high is not None:
        return kind, float(low), float(high)

    normalized_question = question.lower()
    numbers = [float(match) for match in QUESTION_NUMBER_PATTERN.findall(normalized_question)]
    if "or below" in normalized_question and numbers:
        threshold = numbers[-1]
        return "below", -50.0, threshold
    if "or higher" in normalized_question and numbers:
        threshold = numbers[-1]
        return "above", threshold, threshold + 100.0
    if "between" in normalized_question and len(numbers) >= 2:
        return "bounded", numbers[-2], numbers[-1]
    if numbers:
        threshold = numbers[-1]
        return "exact", threshold - 0.5, threshold + 0.5
    return None, None, None


def _threshold_display(temp_kind: str, low: float, high: float, unit: str) -> str:
    if temp_kind == "above":
        return f">{high:g}{unit}"
    if temp_kind == "below":
        return f"<={high:g}{unit}"
    if temp_kind == "bounded":
        return f"{low:g}-{high:g}{unit}"
    return f"{(low + high) / 2.0:g}{unit}"


def _normalize_price_history(price_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for item in price_history:
        timestamp = _parse_datetime(item.get("t"))
        price = _safe_float(item.get("p"), default=None)
        if timestamp is None or price is None:
            continue
        points.append({"timestamp": timestamp, "price": min(max(float(price), 0.0), 1.0)})
    points.sort(key=lambda row: row["timestamp"])
    return points


def _last_point_at_or_before(points: list[dict[str, Any]], anchor: datetime) -> dict[str, Any] | None:
    best = None
    for point in points:
        if point["timestamp"] <= anchor:
            best = point
            continue
        break
    return best


def _consensus_from_models(
    *,
    model_temperatures: dict[str, float | None],
    temp_kind: str,
    temp_range_low: float,
    temp_range_high: float,
) -> tuple[float | None, float | None, float | None]:
    valid_temperatures = [value for value in model_temperatures.values() if value is not None]
    if not valid_temperatures:
        return None, None, None

    model_votes = [
        probability_from_temperature(
            temp_kind=temp_kind,
            temp_range_low=temp_range_low,
            temp_range_high=temp_range_high,
            forecast_temp=value,
        )
        for value in valid_temperatures
    ]
    model_votes = [vote for vote in model_votes if vote is not None]
    if not model_votes:
        return None, None, None
    consensus_probability = sum(model_votes) / len(model_votes)
    model_agreement = max(consensus_probability, 1.0 - consensus_probability)
    model_spread = max(valid_temperatures) - min(valid_temperatures)
    return round(consensus_probability, 6), round(model_agreement, 6), round(model_spread, 6)
