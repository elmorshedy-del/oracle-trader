"""
Enhanced Weather Feature Builder v2
====================================
Adds ~15 new features on top of the existing 20 that the weather ML model uses.
Designed to be called instead of the original _build_feature_row().

NEW FEATURE FAMILIES:
  1. Forecast Drift    — how the forecast changed since last fetch (data already exists!)
  2. ECMWF Ensemble    — spread/uncertainty from 51-member ensemble via Open-Meteo
  3. Climatology       — historical normal temps, anomaly detection
  4. Time Features     — hours to resolution, day of week, season
  5. Market Momentum   — price change velocity from market_price_history

INTEGRATION:
  In weather_model_v2.py, call build_v2_feature_row(candidate, context) instead of
  the original _build_feature_row(candidate). The v2 row contains all original 20
  features PLUS the new ones. The existing CatBoost models will only use the original
  20 — new features are collected for retraining.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

import httpx
import numpy as np

logger = logging.getLogger(__name__)

UTC = timezone.utc

# -------------------------------------------------------------------------
# ECMWF Ensemble endpoint (free via Open-Meteo)
# -------------------------------------------------------------------------
ECMWF_ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
GFS_ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"

# -------------------------------------------------------------------------
# Historical climatology normals (avg high temps °F by city/month)
# Source: NOAA 30-year normals (1991-2020)
# -------------------------------------------------------------------------
CLIMATOLOGY_NORMALS_F = {
    "new-york": {
        1: 39, 2: 42, 3: 50, 4: 62, 5: 72, 6: 80,
        7: 85, 8: 84, 9: 76, 10: 65, 11: 54, 12: 43,
    },
    "chicago": {
        1: 32, 2: 36, 3: 47, 4: 59, 5: 70, 6: 80,
        7: 84, 8: 82, 9: 75, 10: 62, 11: 48, 12: 35,
    },
    "los-angeles": {
        1: 68, 2: 69, 3: 70, 4: 72, 5: 74, 6: 78,
        7: 84, 8: 85, 9: 83, 10: 79, 11: 73, 12: 68,
    },
    "miami": {
        1: 76, 2: 78, 3: 80, 4: 83, 5: 87, 6: 90,
        7: 91, 8: 91, 9: 89, 10: 86, 11: 82, 12: 78,
    },
    "london": {
        1: 46, 2: 47, 3: 52, 4: 57, 5: 63, 6: 69,
        7: 73, 8: 73, 9: 67, 10: 59, 11: 51, 12: 46,
    },
    "seoul": {
        1: 34, 2: 39, 3: 50, 4: 62, 5: 72, 6: 80,
        7: 84, 8: 85, 9: 78, 10: 66, 11: 52, 12: 38,
    },
}


class WeatherFeatureBuilderV2:
    """
    Enhanced weather feature builder that adds ~15 new features
    on top of the existing 20.
    """

    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=10.0,
            headers={"User-Agent": "oracle-trader-weather-v2/1.0"},
        )
        # Cache for ensemble data (city+date → ensemble stats)
        self._ensemble_cache: dict[str, dict[str, float]] = {}
        self._ensemble_cache_ttl: dict[str, datetime] = {}
        self._stats = {
            "ensemble_fetches": 0,
            "ensemble_errors": 0,
            "features_built": 0,
        }

    async def close(self):
        await self.client.aclose()

    @property
    def stats(self) -> dict[str, Any]:
        return dict(self._stats)

    def build_v2_feature_row(
        self,
        candidate: dict,
        context: dict | None = None,
        market_price_history: list[dict] | None = None,
        ensemble_data: dict | None = None,
    ) -> dict | None:
        """
        Build the full v2 feature row from a weather market candidate.

        Returns a dict with ALL features — the original 20 plus ~15 new ones.
        Existing CatBoost models will just ignore the new columns.

        Args:
            candidate: Market candidate dict from weather strategy
            context: Forecast context from _build_forecast_context()
            market_price_history: List of {timestamp, yes_price} dicts
            ensemble_data: Pre-fetched ECMWF ensemble stats (optional)
        """
        # --- Original 20 features ---
        base_row = self._build_base_features(candidate, context)
        if base_row is None:
            return None

        # --- Forecast Drift features (data already in context!) ---
        drift_features = self._build_forecast_drift_features(context)
        base_row.update(drift_features)

        # --- Climatology features ---
        clim_features = self._build_climatology_features(candidate)
        base_row.update(clim_features)

        # --- Time features ---
        time_features = self._build_time_features(candidate)
        base_row.update(time_features)

        # --- Market Momentum features ---
        momentum_features = self._build_market_momentum_features(
            candidate, market_price_history
        )
        base_row.update(momentum_features)

        # --- ECMWF Ensemble features (if pre-fetched) ---
        if ensemble_data:
            base_row.update(ensemble_data)
        else:
            base_row.update(self._empty_ensemble_features())

        self._stats["features_built"] += 1
        return base_row

    # ------------------------------------------------------------------
    # Original 20 features (exact same logic as weather_model.py)
    # ------------------------------------------------------------------

    def _build_base_features(self, candidate: dict, context: dict | None) -> dict | None:
        """Replicate the original _build_feature_row() from weather_model.py."""
        ctx = context or candidate.get("context") or {}
        raw_temps = ctx.get("current_temps") or {}
        if len(raw_temps) < 2:
            return None

        temp_unit = candidate.get("temp_unit") or "F"
        temps = {
            model_name: self._convert_temperature(value, temp_unit)
            for model_name, value in raw_temps.items()
            if value is not None
        }
        if len(temps) < 2:
            return None

        low, high = candidate["temp_range"]
        range_center = (low + high) / 2.0
        target_date = candidate.get("target_date")
        try:
            target_dt = datetime.fromisoformat(target_date) if target_date else None
        except ValueError:
            target_dt = None

        temp_values = list(temps.values())
        temp_mean = statistics.mean(temp_values)
        temp_spread = max(temp_values) - min(temp_values)
        temp_std = statistics.pstdev(temp_values) if len(temp_values) >= 2 else 0.0

        return {
            "city": candidate["city"],
            "temp_unit": temp_unit,
            "temp_kind": candidate["range_kind"],
            "temp_range_low": float(low),
            "temp_range_high": float(high),
            "range_width": float(high - low),
            "range_center": float(range_center),
            "target_month": target_dt.month if target_dt else None,
            "target_day": target_dt.day if target_dt else None,
            "first_yes_price": float(candidate["yes_price"]),
            "forecast_available": 1.0,
            "forecast_temp_max": float(temp_mean),
            "model_count_available": float(len(temp_values)),
            "temp_max_mean": float(temp_mean),
            "temp_max_spread": float(temp_spread),
            "temp_max_std": float(temp_std),
            "gfs_seamless_temp_max": self._maybe_float(temps.get("gfs")),
            "icon_seamless_temp_max": self._maybe_float(temps.get("icon")),
            "temp_max_bucket_gap": float(temp_mean - range_center),
            "temp_max_in_bucket": float(low <= temp_mean <= high),
        }

    # ------------------------------------------------------------------
    # NEW: Forecast Drift Features
    # ------------------------------------------------------------------

    def _build_forecast_drift_features(self, context: dict | None) -> dict[str, float]:
        """
        How much has the forecast changed since the previous fetch?

        The data already exists in context['previous_temps'] and
        context['changed_models'] — we just weren't feeding it to the model!
        """
        if not context:
            return {
                "forecast_shift_mean": 0.0,
                "forecast_shift_max": 0.0,
                "forecast_shift_direction": 0.0,
                "forecast_consensus_shift": 0.0,
                "forecast_stability": 1.0,
                "models_changed_count": 0.0,
                "prob_shift": 0.0,
            }

        current_temps = context.get("current_temps") or {}
        previous_temps = context.get("previous_temps") or {}
        changed_models = context.get("changed_models") or {}

        features: dict[str, float] = {}

        if previous_temps and current_temps:
            # Per-model shifts
            shifts = []
            for model, current_val in current_temps.items():
                prev_val = previous_temps.get(model)
                if prev_val is not None:
                    shifts.append(current_val - prev_val)

            if shifts:
                features["forecast_shift_mean"] = float(statistics.mean(shifts))
                features["forecast_shift_max"] = float(max(shifts, key=abs))
                # Direction: +1 warming, -1 cooling, 0 stable
                mean_shift = statistics.mean(shifts)
                features["forecast_shift_direction"] = (
                    1.0 if mean_shift > 0.5 else (-1.0 if mean_shift < -0.5 else 0.0)
                )
            else:
                features["forecast_shift_mean"] = 0.0
                features["forecast_shift_max"] = 0.0
                features["forecast_shift_direction"] = 0.0

            # Consensus shift
            current_consensus = context.get("current_consensus")
            previous_consensus = context.get("previous_consensus")
            if current_consensus is not None and previous_consensus is not None:
                features["forecast_consensus_shift"] = float(current_consensus - previous_consensus)
            else:
                features["forecast_consensus_shift"] = 0.0

            # Stability: low shift AND low spread = stable forecast
            spread = context.get("current_spread_f", 0.0)
            shift_magnitude = abs(features["forecast_shift_mean"])
            features["forecast_stability"] = max(0.0, 1.0 - (shift_magnitude / 3.0) - (spread / 5.0))
        else:
            features["forecast_shift_mean"] = 0.0
            features["forecast_shift_max"] = 0.0
            features["forecast_shift_direction"] = 0.0
            features["forecast_consensus_shift"] = 0.0
            features["forecast_stability"] = 1.0

        features["models_changed_count"] = float(len(changed_models))

        # Probability shift
        current_prob = context.get("current_prob")
        previous_prob = context.get("previous_prob")
        if current_prob is not None and previous_prob is not None:
            features["prob_shift"] = float(current_prob - previous_prob)
        else:
            features["prob_shift"] = 0.0

        return features

    # ------------------------------------------------------------------
    # NEW: Climatology Features
    # ------------------------------------------------------------------

    def _build_climatology_features(self, candidate: dict) -> dict[str, float]:
        """
        How does the forecast compare to historical normals?

        A forecast that's far from the climatological normal is:
        - Less reliable (models struggle with extremes)
        - More likely to revert toward normal
        """
        city = candidate.get("city", "")
        target_date = candidate.get("target_date")
        try:
            target_dt = datetime.fromisoformat(target_date) if target_date else None
        except ValueError:
            target_dt = None

        normals = CLIMATOLOGY_NORMALS_F.get(city)
        if not normals or not target_dt:
            return {
                "climatology_normal": 0.0,
                "climatology_anomaly": 0.0,
                "climatology_anomaly_abs": 0.0,
                "is_extreme_forecast": 0.0,
            }

        normal_high = normals.get(target_dt.month, 60)

        # Get forecast temp (use context if available)
        context = candidate.get("context") or {}
        current_temps = context.get("current_temps") or {}
        if current_temps:
            temp_unit = candidate.get("temp_unit", "F")
            temps_f = [
                v if temp_unit == "F" else v * 9 / 5 + 32
                for v in current_temps.values()
                if v is not None
            ]
            forecast_high = statistics.mean(temps_f) if temps_f else normal_high
        else:
            forecast_high = normal_high

        anomaly = forecast_high - normal_high

        return {
            "climatology_normal": float(normal_high),
            "climatology_anomaly": float(anomaly),
            "climatology_anomaly_abs": float(abs(anomaly)),
            "is_extreme_forecast": float(abs(anomaly) > 15),  # >15°F from normal
        }

    # ------------------------------------------------------------------
    # NEW: Time Features
    # ------------------------------------------------------------------

    def _build_time_features(self, candidate: dict) -> dict[str, float]:
        """
        Time-based features: hours to resolution, season, day of week.
        """
        target_date = candidate.get("target_date")
        now = datetime.now(UTC)

        try:
            target_dt = datetime.fromisoformat(target_date) if target_date else None
        except ValueError:
            target_dt = None

        if target_dt:
            # Make target_dt timezone-aware if it isn't
            if target_dt.tzinfo is None:
                target_dt = target_dt.replace(tzinfo=UTC)

            # Hours until the market resolves
            hours_to_resolution = max(0, (target_dt - now).total_seconds() / 3600)

            # Season bucket (0=winter, 1=spring, 2=summer, 3=fall)
            month = target_dt.month
            season = (
                0 if month in (12, 1, 2) else
                1 if month in (3, 4, 5) else
                2 if month in (6, 7, 8) else 3
            )

            # Day of week (0=Mon, 6=Sun)
            day_of_week = target_dt.weekday()

            return {
                "hours_to_resolution": float(hours_to_resolution),
                "is_same_day": float(hours_to_resolution < 24),
                "is_next_day": float(24 <= hours_to_resolution < 48),
                "season": float(season),
                "day_of_week": float(day_of_week),
            }
        else:
            return {
                "hours_to_resolution": 48.0,
                "is_same_day": 0.0,
                "is_next_day": 0.0,
                "season": 0.0,
                "day_of_week": 0.0,
            }

    # ------------------------------------------------------------------
    # NEW: Market Momentum Features
    # ------------------------------------------------------------------

    def _build_market_momentum_features(
        self,
        candidate: dict,
        price_history: list[dict] | None,
    ) -> dict[str, float]:
        """
        How is the market price moving? Fast-moving markets may indicate
        information that the model hasn't yet captured.
        """
        if not price_history or len(price_history) < 2:
            return {
                "market_price_change_1h": 0.0,
                "market_price_change_3h": 0.0,
                "market_price_velocity": 0.0,
                "market_price_vs_model": 0.0,
                "market_volume_signal": 0.0,
            }

        features: dict[str, float] = {}

        # Sort by timestamp
        sorted_history = sorted(price_history, key=lambda x: x.get("timestamp", ""))
        prices = [h.get("yes_price", h.get("price", 0)) for h in sorted_history]
        current_price = float(candidate.get("yes_price", prices[-1]))

        # Price change over last N entries (each entry is ~1 scan cycle)
        if len(prices) >= 3:
            features["market_price_change_1h"] = float(current_price - prices[-3])
        else:
            features["market_price_change_1h"] = 0.0

        if len(prices) >= 6:
            features["market_price_change_3h"] = float(current_price - prices[-6])
        else:
            features["market_price_change_3h"] = features["market_price_change_1h"]

        # Velocity: rate of price change per entry
        if len(prices) >= 3:
            recent_changes = [prices[i] - prices[i - 1] for i in range(-3, 0)]
            features["market_price_velocity"] = float(statistics.mean(recent_changes))
        else:
            features["market_price_velocity"] = 0.0

        # Model-vs-market disagreement
        context = candidate.get("context") or {}
        model_prob = context.get("current_prob")
        if model_prob is not None:
            features["market_price_vs_model"] = float(model_prob - current_price)
        else:
            features["market_price_vs_model"] = 0.0

        # Volume signal: large price swings suggest informed traders
        if len(prices) >= 5:
            price_std = statistics.pstdev(prices[-5:])
            features["market_volume_signal"] = float(min(price_std / 0.05, 1.0))
        else:
            features["market_volume_signal"] = 0.0

        return features

    # ------------------------------------------------------------------
    # ECMWF Ensemble Features
    # ------------------------------------------------------------------

    async def fetch_ensemble_data(
        self,
        city: str,
        target_date: str,
        lat: float,
        lon: float,
    ) -> dict[str, float]:
        """
        Fetch ECMWF ensemble (51-member) temperature forecast spread.

        This gives us UNCERTAINTY — something no single-model forecast provides.
        """
        cache_key = f"{city}:{target_date}"
        now = datetime.now(UTC)

        # Check cache (refresh every 3 hours)
        if cache_key in self._ensemble_cache:
            cache_time = self._ensemble_cache_ttl.get(cache_key)
            if cache_time and (now - cache_time).total_seconds() < 10800:
                return self._ensemble_cache[cache_key]

        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max",
                "start_date": target_date,
                "end_date": target_date,
                "models": "ecmwf_ifs025",
            }
            response = await self.client.get(ECMWF_ENSEMBLE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            daily = data.get("daily", {})
            # Ensemble members come as temperature_2m_max_member01, ...member50
            member_temps = []
            for key, values in daily.items():
                if key.startswith("temperature_2m_max") and values:
                    val = values[0]  # first (only) date
                    if val is not None:
                        member_temps.append(float(val))

            if len(member_temps) >= 10:
                features = {
                    "ensemble_mean": float(statistics.mean(member_temps)),
                    "ensemble_std": float(statistics.pstdev(member_temps)),
                    "ensemble_spread": float(max(member_temps) - min(member_temps)),
                    "ensemble_p10": float(sorted(member_temps)[len(member_temps) // 10]),
                    "ensemble_p90": float(sorted(member_temps)[9 * len(member_temps) // 10]),
                    "ensemble_members": float(len(member_temps)),
                    "ensemble_skew": float(self._skewness(member_temps)),
                }
            else:
                features = self._empty_ensemble_features()

            self._ensemble_cache[cache_key] = features
            self._ensemble_cache_ttl[cache_key] = now
            self._stats["ensemble_fetches"] += 1
            return features

        except Exception as exc:
            logger.warning("[WEATHER_V2] Ensemble fetch failed for %s: %s", cache_key, exc)
            self._stats["ensemble_errors"] += 1
            return self._empty_ensemble_features()

    @staticmethod
    def _empty_ensemble_features() -> dict[str, float]:
        return {
            "ensemble_mean": 0.0,
            "ensemble_std": 0.0,
            "ensemble_spread": 0.0,
            "ensemble_p10": 0.0,
            "ensemble_p90": 0.0,
            "ensemble_members": 0.0,
            "ensemble_skew": 0.0,
        }

    @staticmethod
    def _skewness(values: list[float]) -> float:
        """Simple skewness calculation."""
        n = len(values)
        if n < 3:
            return 0.0
        mean = statistics.mean(values)
        std = statistics.pstdev(values)
        if std == 0:
            return 0.0
        return sum(((v - mean) / std) ** 3 for v in values) / n

    @staticmethod
    def _convert_temperature(value_f: float, target_unit: str) -> float:
        if target_unit == "C":
            return (float(value_f) - 32.0) * 5.0 / 9.0
        return float(value_f)

    @staticmethod
    def _maybe_float(value) -> Optional[float]:
        if value is None:
            return None
        return float(value)
