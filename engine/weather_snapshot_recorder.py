from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import APIConfig
from data.collector import PolymarketCollector
from engine.https_support import urlopen_with_default_context
from engine.telegram_notifier import TelegramNotifier
from engine.weather_edge_live_support import (
    WEATHER_EDGE_SNAPSHOT_ALERT_FAILURE_RATIO,
    WEATHER_EDGE_SNAPSHOT_CALL_DELAY_SECONDS,
    WEATHER_EDGE_SNAPSHOT_DAILY_FIELDS,
    WEATHER_EDGE_SNAPSHOT_FALLBACK_CITIES,
    WEATHER_EDGE_SNAPSHOT_FORECAST_DAYS,
    WEATHER_EDGE_SNAPSHOT_MODELS,
    canonical_weather_city_name,
    collect_active_weather_cities,
)
from runtime_paths import DATA_DIR


logger = logging.getLogger(__name__)

UTC = timezone.utc
OPEN_METEO_FORECAST_ENDPOINT = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEOCODING_ENDPOINT = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_ALLOWED_HOSTS = {
    "api.open-meteo.com",
    "geocoding-api.open-meteo.com",
}
WEATHER_SNAPSHOT_REQUEST_TIMEOUT_SECONDS = 30
WEATHER_SNAPSHOT_FALLBACK_ROOT = Path("logs") / "weather_snapshots"


@dataclass(slots=True)
class WeatherSnapshotRecorderConfig:
    storage_root: Path = DATA_DIR / "weather_snapshots"
    models: tuple[str, ...] = WEATHER_EDGE_SNAPSHOT_MODELS
    forecast_days: int = WEATHER_EDGE_SNAPSHOT_FORECAST_DAYS
    delay_seconds: float = WEATHER_EDGE_SNAPSHOT_CALL_DELAY_SECONDS
    failure_alert_ratio: float = WEATHER_EDGE_SNAPSHOT_ALERT_FAILURE_RATIO


class WeatherSnapshotRecorder:
    def __init__(
        self,
        *,
        collector: PolymarketCollector,
        notifier: TelegramNotifier,
        config: WeatherSnapshotRecorderConfig | None = None,
    ):
        self.collector = collector
        self.notifier = notifier
        self.config = config or WeatherSnapshotRecorderConfig()
        self.config.storage_root = self._resolve_storage_root(self.config.storage_root)
        self.config.storage_root.mkdir(parents=True, exist_ok=True)
        self.geocode_cache_dir = self.config.storage_root / "geocodes"
        self.error_log_path = self.config.storage_root / "errors.jsonl"
        self.geocode_cache_dir.mkdir(parents=True, exist_ok=True)

    async def run(self) -> dict[str, Any]:
        fetched_at = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
        output_path = self.config.storage_root / f"snapshots_{fetched_at.date().isoformat()}.jsonl"
        cities = await self._active_or_fallback_cities()

        total_fetches = 0
        failed_fetches = 0
        rows_written = 0
        for city in cities:
            geocode = self._geocode_city(city)
            if geocode is None:
                total_fetches += len(self.config.models)
                failed_fetches += len(self.config.models)
                self._log_error(
                    {
                        "timestamp": fetched_at.isoformat(),
                        "city": city,
                        "error": "geocode_failed",
                    }
                )
                continue

            for model in self.config.models:
                total_fetches += 1
                try:
                    payload = self._fetch_model_snapshot(
                        latitude=float(geocode["latitude"]),
                        longitude=float(geocode["longitude"]),
                        model=model,
                    )
                    snapshot_row = {
                        "fetched_at": fetched_at.isoformat().replace("+00:00", "Z"),
                        "city": city,
                        "latitude": float(geocode["latitude"]),
                        "longitude": float(geocode["longitude"]),
                        "model": model,
                        "forecasts": self._forecast_rows(payload),
                    }
                    with output_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(snapshot_row, sort_keys=True))
                        handle.write("\n")
                    rows_written += 1
                except Exception as exc:
                    failed_fetches += 1
                    self._log_error(
                        {
                            "timestamp": fetched_at.isoformat(),
                            "city": city,
                            "model": model,
                            "error": str(exc),
                        }
                    )
                time.sleep(self.config.delay_seconds)

        failure_ratio = (failed_fetches / total_fetches) if total_fetches else 0.0
        summary = {
            "fetched_at": fetched_at.isoformat(),
            "output_path": str(output_path),
            "cities": len(cities),
            "rows_written": rows_written,
            "total_fetches": total_fetches,
            "failed_fetches": failed_fetches,
            "failure_ratio": round(failure_ratio, 6),
            "storage_root": str(self.config.storage_root),
        }
        if total_fetches and failure_ratio > self.config.failure_alert_ratio:
            self.notifier.send_message(
                "\n".join(
                    [
                        "Weather Snapshot Recorder Alert",
                        f"Fetched at: {fetched_at.isoformat()}",
                        f"Failures: {failed_fetches}/{total_fetches} ({failure_ratio:.1%})",
                        f"Output: {output_path}",
                    ]
                )
            )
        return summary

    async def _active_or_fallback_cities(self) -> list[str]:
        markets = await self.collector.get_all_active_markets()
        cities = collect_active_weather_cities(markets)
        if cities:
            return cities
        return list(WEATHER_EDGE_SNAPSHOT_FALLBACK_CITIES)

    def _geocode_city(self, city: str) -> dict[str, Any] | None:
        cache_path = self.geocode_cache_dir / f"{self._slugify(city)}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())
        params = urllib.parse.urlencode(
            {
                "name": canonical_weather_city_name(city),
                "count": 1,
                "language": "en",
                "format": "json",
            }
        )
        payload = self._fetch_json(f"{OPEN_METEO_GEOCODING_ENDPOINT}?{params}")
        result = (payload.get("results") or [None])[0]
        if not result:
            return None
        cache_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def _fetch_model_snapshot(
        self,
        *,
        latitude: float,
        longitude: float,
        model: str,
    ) -> dict[str, Any]:
        params = urllib.parse.urlencode(
            {
                "latitude": latitude,
                "longitude": longitude,
                "models": model,
                "daily": ",".join(WEATHER_EDGE_SNAPSHOT_DAILY_FIELDS),
                "forecast_days": self.config.forecast_days,
                "timezone": "UTC",
                "temperature_unit": "fahrenheit",
                "precipitation_unit": "inch",
                "wind_speed_unit": "mph",
            }
        )
        return self._fetch_json(f"{OPEN_METEO_FORECAST_ENDPOINT}?{params}")

    def _forecast_rows(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        daily = payload.get("daily") or {}
        dates = daily.get("time") or []
        temp_max = daily.get("temperature_2m_max") or []
        temp_min = daily.get("temperature_2m_min") or []
        precip = daily.get("precipitation_sum") or []
        snow = daily.get("snowfall_sum") or []
        rows: list[dict[str, Any]] = []
        for index, date_value in enumerate(dates):
            rows.append(
                {
                    "date": date_value,
                    "temp_max": temp_max[index] if index < len(temp_max) else None,
                    "temp_min": temp_min[index] if index < len(temp_min) else None,
                    "precip_sum": precip[index] if index < len(precip) else None,
                    "snow_sum": snow[index] if index < len(snow) else None,
                }
            )
        return rows

    def _fetch_json(self, url: str) -> dict[str, Any]:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in OPEN_METEO_ALLOWED_HOSTS:
            raise ValueError(f"Refusing to fetch non-allowlisted URL: {url}")
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "oracle-trader/weather-snapshot-recorder",
            },
        )
        with urlopen_with_default_context(
            request,
            timeout=WEATHER_SNAPSHOT_REQUEST_TIMEOUT_SECONDS,
        ) as response:
            return json.loads(response.read())

    def _log_error(self, payload: dict[str, Any]) -> None:
        with self.error_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")

    @staticmethod
    def _slugify(value: str) -> str:
        return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")

    @staticmethod
    def _resolve_storage_root(preferred: Path) -> Path:
        for candidate in (preferred, WEATHER_SNAPSHOT_FALLBACK_ROOT):
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                probe = candidate / ".write-test"
                probe.write_text("ok", encoding="utf-8")
                probe.unlink(missing_ok=True)
                return candidate
            except OSError:
                continue
        return WEATHER_SNAPSHOT_FALLBACK_ROOT


async def build_default_snapshot_recorder() -> WeatherSnapshotRecorder:
    api = APIConfig()
    collector = PolymarketCollector(
        gamma_host=api.gamma_host,
        clob_host=api.clob_host,
        data_host=api.data_host,
    )
    return WeatherSnapshotRecorder(
        collector=collector,
        notifier=TelegramNotifier.from_env(),
    )
