"""
Strategy: Weather Forecast Variants
===================================
Three weather trading styles built on top of the same free forecast data:

- Sniper: tiny YES-only mispricing bets when GFS/ECMWF/ICON tightly agree.
- Latency Hunter: reacts when the forecast consensus shifts before the market catches up.
- Swing Trader: buys dips in the favored bucket when the forecast stays stable.

Forecast source: Open-Meteo model endpoints (GFS / ECMWF / DWD ICON).
"""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone, timedelta
import httpx
import json
import logging
from pathlib import Path
import re
import statistics

from data.models import Event, Market, Outcome, Signal, SignalAction, SignalSource
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

OPEN_METEO_MODEL_ENDPOINTS = {
    "gfs": "https://api.open-meteo.com/v1/gfs",
    "ecmwf": "https://api.open-meteo.com/v1/ecmwf",
    "icon": "https://api.open-meteo.com/v1/dwd-icon",
}

CITY_COORDINATES = {
    "new-york": {"lat": 40.7128, "lon": -74.0060},
    "chicago": {"lat": 41.8781, "lon": -87.6298},
    "los-angeles": {"lat": 34.0522, "lon": -118.2437},
    "miami": {"lat": 25.7617, "lon": -80.1918},
    "london": {"lat": 51.5072, "lon": -0.1276},
    "seoul": {"lat": 37.5665, "lon": 126.9780},
}

CITY_KEYWORDS = {
    "new-york": ["new york", "nyc", "new york city", "manhattan"],
    "chicago": ["chicago"],
    "los-angeles": ["los angeles", "la"],
    "miami": ["miami"],
    "london": ["london"],
    "seoul": ["seoul"],
}

SUPPORTED_WEATHER_MARKET = re.compile(r"\bhighest temperature in\b", re.IGNORECASE)
TEMP_RANGE_PATTERNS = (
    re.compile(r"(\d+)\s*[-–]\s*(\d+)\s*(?:°|degrees|fahrenheit|celsius|f\b|c\b)", re.IGNORECASE),
    re.compile(r"(\d+)\s+to\s+(\d+)\s*(?:°|degrees|fahrenheit|celsius|f\b|c\b)", re.IGNORECASE),
    re.compile(r"between\s+(\d+)\s+and\s+(\d+)", re.IGNORECASE),
    re.compile(r"(\d+)\s*[-–]\s*(\d+)", re.IGNORECASE),
)
TEMP_EXACT_PATTERN = re.compile(r"be\s+(\d+)\s*(?:°|degrees)?\s*([fc])\b", re.IGNORECASE)
TEMP_ABOVE_PATTERNS = (
    re.compile(r"(?:above|over|exceed|higher than)\s+(\d+)", re.IGNORECASE),
    re.compile(r"(\d+)\s*(?:°|degrees)?\s*[fc]?\s*or higher", re.IGNORECASE),
)
TEMP_BELOW_PATTERNS = (
    re.compile(r"(?:below|under|lower than)\s+(\d+)", re.IGNORECASE),
    re.compile(r"(\d+)\s*(?:°|degrees)?\s*[fc]?\s*or below", re.IGNORECASE),
)

WEATHER_DISCOVERY_REFRESH_SECS = 1800
WEATHER_MARKET_LOOKAHEAD_DAYS = 4
WEATHER_MAX_SIGNAL_HORIZON_DAYS = 2
WEATHER_PRICE_HISTORY_LIMIT = 240
WEATHER_STATE_RETENTION_HOURS = 72
WEATHER_BASE_STD_DEV_F = 2.6
WEATHER_STD_DEV_PER_DAY_F = 0.8
WEATHER_MODEL_SPREAD_MULTIPLIER = 0.35
WEATHER_NARROW_RANGE_STD_DEV_F = 0.35
WEATHER_BOUNDARY_BUFFER_F = 2.0
WEATHER_OPEN_THRESHOLD_BUFFER_F = 3.0
WEATHER_FEE_BUFFER = 0.012
WEATHER_MIN_CLOSED_RANGE_YES_PROB = 0.18
WEATHER_MIN_CLOSED_RANGE_NO_PROB = 0.08
WEATHER_CITY_ALIASES = {
    "nyc": "new-york",
}

WEATHER_VARIANTS = {
    "sniper": {
        "source": SignalSource.WEATHER_SNIPER,
        "label": "Sniper",
    },
    "latency": {
        "source": SignalSource.WEATHER_LATENCY,
        "label": "Latency Hunter",
    },
    "swing": {
        "source": SignalSource.WEATHER_SWING,
        "label": "Swing Trader",
    },
}


class WeatherForecastStrategy(BaseStrategy):
    name = "weather_forecast"
    description = "Open-Meteo weather model consensus vs Polymarket temperature markets"

    def __init__(self, config, collector=None, state_path: str | None = None):
        super().__init__(config)
        self.cfg = config.weather
        self.collector = collector
        self.state_path = Path(state_path) if state_path else None
        self.client = httpx.AsyncClient(
            timeout=15.0,
            headers={"User-Agent": "oracle-trader/1.0 (weather-lab)"},
        )

        self._model_forecasts: dict[str, dict[str, dict]] = {}
        self._matched_markets: list[dict] = []
        self._supplemental_markets: list[Market] = []
        self._market_price_history: dict[str, deque] = {}
        self._last_forecast_fetch: float = 0
        self._last_market_scan: float = 0
        self._last_supplemental_fetch: float = 0
        self._variant_stats = {
            name: {
                "label": meta["label"],
                "source": meta["source"].value,
                "signals_generated": 0,
                "last_signals": 0,
                "last_candidates": 0,
                "last_model_spread_f": 0.0,
                "previous_ready_markets": 0,
                "history_ready_markets": 0,
            }
            for name, meta in WEATHER_VARIANTS.items()
        }
        self._stats.update({
            "matched_markets": 0,
            "supplemental_markets": 0,
            "forecast_cities": 0,
            "models_ready": 0,
            "last_forecast_refresh": None,
            "state_loaded": False,
        })
        self._load_state()

    @property
    def stats(self) -> dict:
        return {
            "name": self.name,
            "enabled": self.enabled,
            **self._stats,
            "variants": self._variant_stats,
        }

    async def scan(self, markets: list[Market], events: list[Event]) -> list[Signal]:
        if not self.cfg.enabled:
            return []

        self._stats["scans_completed"] += 1
        for stats in self._variant_stats.values():
            stats["last_signals"] = 0
            stats["last_candidates"] = 0
            stats["last_model_spread_f"] = 0.0
            stats["previous_ready_markets"] = 0
            stats["history_ready_markets"] = 0
        now = datetime.now(timezone.utc)
        now_ts = now.timestamp()

        if now_ts - self._last_forecast_fetch > self.cfg.forecast_refresh_secs:
            await self._fetch_forecasts()
            self._last_forecast_fetch = now_ts

        if now_ts - self._last_market_scan > self.cfg.market_refresh_secs:
            self._match_weather_markets(markets)
            self._last_market_scan = now_ts

        self._record_market_prices()
        self._stats["state_loaded"] = bool(self._model_forecasts or self._market_price_history)

        per_variant_best = {name: {} for name in WEATHER_VARIANTS}

        for match in self._matched_markets:
            market = match["market"]
            if len(market.outcomes) < 2:
                continue

            target_date = match.get("target_date")
            horizon_days = self._forecast_horizon_days(target_date)
            if horizon_days is None or horizon_days > WEATHER_MAX_SIGNAL_HORIZON_DAYS:
                continue

            context = self._build_forecast_context(
                city=match["city"],
                target_date=target_date,
                temp_range=match["temp_range"],
                range_kind=match["range_kind"],
                horizon_days=horizon_days,
            )
            if not context:
                continue

            yes_price = market.outcomes[0].price
            no_price = market.outcomes[1].price

            sniper = self._build_sniper_signal(
                match=match,
                context=context,
                yes_price=yes_price,
                no_price=no_price,
            )
            if sniper:
                self._track_best_signal(per_variant_best["sniper"], sniper)

            latency = self._build_latency_signal(
                match=match,
                context=context,
                yes_price=yes_price,
                no_price=no_price,
            )
            if latency:
                self._track_best_signal(per_variant_best["latency"], latency)

            swing = self._build_swing_signal(
                match=match,
                context=context,
                yes_price=yes_price,
                no_price=no_price,
            )
            if swing:
                self._track_best_signal(per_variant_best["swing"], swing)

        signals: list[Signal] = []
        total_generated = 0
        for variant_name, best_map in per_variant_best.items():
            variant_signals = sorted(
                best_map.values(),
                key=lambda signal: (signal.expected_edge, signal.confidence),
                reverse=True,
            )
            self._variant_stats[variant_name]["last_signals"] = len(variant_signals)
            self._variant_stats[variant_name]["signals_generated"] += len(variant_signals)
            total_generated += len(variant_signals)
            signals.extend(variant_signals)

        self._stats["signals_generated"] += total_generated
        self._stats["matched_markets"] = len(self._matched_markets)

        if signals:
            logger.info(
                "[WEATHER] Generated %s signals across variants (sniper=%s latency=%s swing=%s)",
                len(signals),
                self._variant_stats["sniper"]["last_signals"],
                self._variant_stats["latency"]["last_signals"],
                self._variant_stats["swing"]["last_signals"],
            )
        elif self._stats["scans_completed"] % 20 == 0:
            logger.info(
                "[WEATHER] Status: %s cities ready | %s matched markets | 0 signals",
                self._stats["forecast_cities"],
                len(self._matched_markets),
            )

        return signals

    def get_model_candidates(self) -> list[dict]:
        """Expose enriched weather matches for external-only ML sleeves."""
        candidates: list[dict] = []
        for match in self._matched_markets:
            market = match["market"]
            if len(market.outcomes) < 2:
                continue

            target_date = match.get("target_date")
            horizon_days = self._forecast_horizon_days(target_date)
            if horizon_days is None or horizon_days > WEATHER_MAX_SIGNAL_HORIZON_DAYS:
                continue

            context = self._build_forecast_context(
                city=match["city"],
                target_date=target_date,
                temp_range=match["temp_range"],
                range_kind=match["range_kind"],
                horizon_days=horizon_days,
            )
            if not context:
                continue

            candidates.append(
                {
                    "market": market,
                    "city": match["city"],
                    "temp_range": match["temp_range"],
                    "range_kind": match["range_kind"],
                    "target_date": target_date,
                    "temp_unit": match.get("temp_unit", "F"),
                    "context": context,
                    "yes_price": market.outcomes[0].price,
                    "no_price": market.outcomes[1].price,
                }
            )
        return candidates

    def _weather_group_key(self, city: str, target_date: str | None) -> str:
        canonical_city = WEATHER_CITY_ALIASES.get(city, city)
        return f"weather:{canonical_city}:{target_date or 'unknown'}"

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

    def _record_market_prices(self):
        now = datetime.now(timezone.utc)
        for match in self._matched_markets:
            market = match["market"]
            if len(market.outcomes) < 2:
                continue
            history = self._market_price_history.setdefault(
                market.condition_id,
                deque(maxlen=WEATHER_PRICE_HISTORY_LIMIT),
            )
            history.append({
                "time": now,
                "yes": market.outcomes[0].price,
                "no": market.outcomes[1].price,
            })

    def _price_series(self, condition_id: str, side: str, lookback_minutes: int) -> list[float]:
        history = self._market_price_history.get(condition_id)
        if not history:
            return []
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
        key = "yes" if side == "YES" else "no"
        return [entry[key] for entry in history if entry["time"] >= cutoff]

    def _load_state(self):
        if not self.state_path or not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text())
            self._model_forecasts = self._deserialize_model_forecasts(
                data.get("model_forecasts", {})
            )
            self._market_price_history = self._deserialize_price_history(
                data.get("market_price_history", {})
            )
            self._stats["state_loaded"] = bool(self._model_forecasts or self._market_price_history)
        except Exception as exc:
            logger.warning("[WEATHER] Failed to load weather state: %s", exc)

    def save_state(self):
        if not self.state_path:
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "model_forecasts": self._serialize_model_forecasts(),
                "market_price_history": self._serialize_price_history(),
            }
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload))
            tmp.rename(self.state_path)
        except Exception as exc:
            logger.warning("[WEATHER] Failed to save weather state: %s", exc)

    def reset_state(self):
        self._model_forecasts = {}
        self._matched_markets = []
        self._supplemental_markets = []
        self._market_price_history = {}
        self._last_forecast_fetch = 0
        self._last_market_scan = 0
        self._last_supplemental_fetch = 0
        self._stats["state_loaded"] = False
        for stats in self._variant_stats.values():
            stats["last_signals"] = 0
            stats["last_candidates"] = 0
            stats["last_model_spread_f"] = 0.0
            stats["previous_ready_markets"] = 0
            stats["history_ready_markets"] = 0
        if self.state_path:
            try:
                self.state_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("[WEATHER] Failed to delete weather state: %s", exc)

    def _serialize_model_forecasts(self) -> dict:
        payload = {}
        for city, models in self._model_forecasts.items():
            payload[city] = {}
            for model_name, state in models.items():
                payload[city][model_name] = {
                    "current": state.get("current", {}),
                    "previous": state.get("previous", {}),
                    "fetched_at": self._datetime_to_iso(state.get("fetched_at")),
                    "last_changed_at": self._datetime_to_iso(state.get("last_changed_at")),
                }
        return payload

    def _deserialize_model_forecasts(self, payload: dict) -> dict:
        restored = {}
        for city, models in payload.items():
            restored[city] = {}
            for model_name, state in models.items():
                restored[city][model_name] = {
                    "current": state.get("current", {}),
                    "previous": state.get("previous", {}),
                    "fetched_at": self._iso_to_datetime(state.get("fetched_at")),
                    "last_changed_at": self._iso_to_datetime(state.get("last_changed_at")),
                }
        return restored

    def _serialize_price_history(self) -> dict:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=WEATHER_STATE_RETENTION_HOURS)
        payload = {}
        for condition_id, history in self._market_price_history.items():
            recent = [
                {
                    "time": self._datetime_to_iso(entry.get("time")),
                    "yes": entry.get("yes"),
                    "no": entry.get("no"),
                }
                for entry in history
                if isinstance(entry.get("time"), datetime) and entry["time"] >= cutoff
            ]
            if recent:
                payload[condition_id] = recent[-WEATHER_PRICE_HISTORY_LIMIT:]
        return payload

    def _deserialize_price_history(self, payload: dict) -> dict:
        restored = {}
        for condition_id, entries in payload.items():
            history = deque(maxlen=WEATHER_PRICE_HISTORY_LIMIT)
            for entry in entries:
                ts = self._iso_to_datetime(entry.get("time"))
                if ts is None:
                    continue
                history.append({
                    "time": ts,
                    "yes": float(entry.get("yes", 0.0) or 0.0),
                    "no": float(entry.get("no", 0.0) or 0.0),
                })
            if history:
                restored[condition_id] = history
        return restored

    @staticmethod
    def _datetime_to_iso(value: datetime | None) -> str | None:
        if not isinstance(value, datetime):
            return None
        return value.isoformat()

    @staticmethod
    def _iso_to_datetime(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    def _build_forecast_context(
        self,
        *,
        city: str,
        target_date: str,
        temp_range: tuple[float, float],
        range_kind: str,
        horizon_days: int,
    ) -> dict | None:
        city_models = self._model_forecasts.get(city, {})
        current_temps = {}
        previous_temps = {}
        model_changes = {}

        for model_name, state in city_models.items():
            current = state.get("current", {}).get(target_date)
            previous = state.get("previous", {}).get(target_date)
            if current is not None:
                current_temps[model_name] = current
            if previous is not None:
                previous_temps[model_name] = previous
                if current is not None:
                    model_changes[model_name] = current - previous

        if len(current_temps) < 2:
            return None

        settlement_range = self._settlement_temp_range(temp_range, range_kind)
        current_consensus = statistics.median(current_temps.values())
        current_spread = max(current_temps.values()) - min(current_temps.values())
        current_prob = self._temp_in_range_probability(
            current_consensus,
            settlement_range[0],
            settlement_range[1],
            horizon_days=horizon_days,
            model_spread_f=current_spread,
        )

        previous_prob = None
        previous_consensus = None
        if len(previous_temps) >= 2:
            previous_consensus = statistics.median(previous_temps.values())
            previous_spread = max(previous_temps.values()) - min(previous_temps.values())
            previous_prob = self._temp_in_range_probability(
                previous_consensus,
                settlement_range[0],
                settlement_range[1],
                horizon_days=horizon_days,
                model_spread_f=previous_spread,
            )

        changed_models = {
            name: delta
            for name, delta in model_changes.items()
            if abs(delta) >= 0.5
        }

        return {
            "city": city,
            "target_date": target_date,
            "settlement_range": settlement_range,
            "range_kind": range_kind,
            "horizon_days": horizon_days,
            "current_temps": current_temps,
            "current_consensus": current_consensus,
            "current_spread_f": current_spread,
            "current_prob": current_prob,
            "previous_temps": previous_temps,
            "previous_consensus": previous_consensus,
            "previous_prob": previous_prob,
            "changed_models": changed_models,
        }

    def _build_sniper_signal(
        self,
        *,
        match: dict,
        context: dict,
        yes_price: float,
        no_price: float,
    ) -> Signal | None:
        self._variant_stats["sniper"]["last_candidates"] += 1
        self._variant_stats["sniper"]["last_model_spread_f"] = round(context["current_spread_f"], 2)

        if len(context["current_temps"]) < 3:
            return None
        if context["current_spread_f"] > self.cfg.model_agreement_max_spread_f:
            return None
        if context["current_prob"] < self.cfg.sniper_min_prob:
            return None
        if yes_price > self.cfg.sniper_max_yes_price:
            return None

        edge = context["current_prob"] - yes_price - self._fee_buffer(yes_price)
        if edge < self.cfg.min_edge:
            return None

        size_usd = min(
            self.cfg.sniper_max_size_usd,
            max(self.cfg.sniper_min_size_usd, 1.0 + (edge * 8.0)),
        )
        confidence = min(0.99, 0.75 + edge)
        reasoning = (
            f"SNIPER: {match['city']} {match['target_date']} | "
            f"GFS/ECMWF/ICON agree within {context['current_spread_f']:.1f}F | "
            f"fair YES={context['current_prob']:.0%} vs market YES={yes_price:.0%}"
        )
        return self._directional_signal(
            variant="sniper",
            market=match["market"],
            city=match["city"],
            target_date=match["target_date"],
            action=SignalAction.BUY_YES,
            token_price=yes_price,
            confidence=confidence,
            edge=edge,
            size_usd=size_usd,
            reasoning=reasoning,
        )

    def _build_latency_signal(
        self,
        *,
        match: dict,
        context: dict,
        yes_price: float,
        no_price: float,
    ) -> Signal | None:
        self._variant_stats["latency"]["last_candidates"] += 1
        self._variant_stats["latency"]["last_model_spread_f"] = round(context["current_spread_f"], 2)

        previous_prob = context["previous_prob"]
        if previous_prob is None:
            return None
        self._variant_stats["latency"]["previous_ready_markets"] += 1

        probability_shift = context["current_prob"] - previous_prob
        if abs(probability_shift) < self.cfg.latency_min_probability_shift:
            return None
        if not context["changed_models"]:
            return None

        if probability_shift > 0:
            token_price = yes_price
            token_fair = context["current_prob"]
            action = SignalAction.BUY_YES
        else:
            token_price = no_price
            token_fair = 1.0 - context["current_prob"]
            action = SignalAction.BUY_NO

        if token_price > self.cfg.latency_max_entry_price:
            return None

        edge = token_fair - token_price - self._fee_buffer(token_price)
        if edge < self.cfg.latency_min_edge:
            return None

        size_usd = min(
            self.cfg.latency_max_size_usd,
            max(
                self.cfg.latency_min_size_usd,
                self.cfg.latency_min_size_usd + (abs(probability_shift) * 80.0),
            ),
        )
        confidence = min(0.97, 0.60 + abs(probability_shift) + edge)
        reasoning = (
            f"LATENCY: {match['city']} {match['target_date']} | "
            f"fair YES moved {previous_prob:.0%} -> {context['current_prob']:.0%} | "
            f"market token still {token_price:.0%} | models changed: "
            + ", ".join(
                f"{name}:{delta:+.1f}F" for name, delta in sorted(context["changed_models"].items())
            )
        )
        return self._directional_signal(
            variant="latency",
            market=match["market"],
            city=match["city"],
            target_date=match["target_date"],
            action=action,
            token_price=token_price,
            confidence=confidence,
            edge=edge,
            size_usd=size_usd,
            reasoning=reasoning,
        )

    def _build_swing_signal(
        self,
        *,
        match: dict,
        context: dict,
        yes_price: float,
        no_price: float,
    ) -> Signal | None:
        self._variant_stats["swing"]["last_candidates"] += 1
        self._variant_stats["swing"]["last_model_spread_f"] = round(context["current_spread_f"], 2)

        previous_prob = context["previous_prob"]
        if previous_prob is not None and abs(context["current_prob"] - previous_prob) > 0.08:
            return None

        if context["current_prob"] >= self.cfg.swing_min_prob:
            side = "YES"
            token_price = yes_price
            token_fair = context["current_prob"]
            action = SignalAction.BUY_YES
        elif context["current_prob"] <= self.cfg.swing_max_prob:
            side = "NO"
            token_price = no_price
            token_fair = 1.0 - context["current_prob"]
            action = SignalAction.BUY_NO
        else:
            return None

        series = self._price_series(
            match["market"].condition_id,
            side,
            self.cfg.swing_lookback_minutes,
        )
        if len(series) >= 3:
            self._variant_stats["swing"]["history_ready_markets"] += 1
        if len(series) < 3:
            return None

        recent_peak = max(series)
        dip = recent_peak - token_price
        if dip < self.cfg.swing_min_token_dip:
            return None

        edge = token_fair - token_price - self._fee_buffer(token_price)
        if edge < self.cfg.swing_min_edge:
            return None

        size_usd = min(
            self.cfg.swing_max_size_usd,
            max(self.cfg.swing_min_size_usd, self.cfg.swing_min_size_usd + (dip * 60.0)),
        )
        confidence = min(0.95, 0.55 + edge + min(dip, 0.15))
        reasoning = (
            f"SWING: {match['city']} {match['target_date']} | "
            f"favored {side} token dipped from {recent_peak:.0%} to {token_price:.0%} | "
            f"fair token={token_fair:.0%} with steady forecast"
        )
        return self._directional_signal(
            variant="swing",
            market=match["market"],
            city=match["city"],
            target_date=match["target_date"],
            action=action,
            token_price=token_price,
            confidence=confidence,
            edge=edge,
            size_usd=size_usd,
            reasoning=reasoning,
        )

    def _directional_signal(
        self,
        *,
        variant: str,
        market: Market,
        city: str,
        target_date: str | None,
        action: SignalAction,
        token_price: float,
        confidence: float,
        edge: float,
        size_usd: float,
        reasoning: str,
    ) -> Signal | None:
        outcome_index = 0 if action == SignalAction.BUY_YES else 1
        if len(market.outcomes) <= outcome_index:
            return None
        token_id = market.outcomes[outcome_index].token_id
        source = WEATHER_VARIANTS[variant]["source"]
        return Signal(
            source=source,
            action=action,
            market_slug=market.slug,
            condition_id=market.condition_id,
            token_id=token_id,
            confidence=max(0.05, min(confidence, 0.99)),
            expected_edge=edge * 100,
            group_key=self._weather_group_key(city, target_date),
            reasoning=reasoning,
            suggested_size_usd=size_usd,
        )

    def _fee_buffer(self, price: float) -> float:
        price = max(0.01, min(0.99, price))
        return WEATHER_FEE_BUFFER * min(price, 1.0 - price)

    def _settlement_temp_range(
        self,
        temp_range: tuple[float, float],
        range_kind: str,
    ) -> tuple[float, float]:
        low, high = temp_range
        if range_kind == "above":
            return (low - 0.5, high)
        if range_kind == "below":
            return (low, high + 0.5)
        if range_kind == "exact":
            return (low, high)
        return (low - 0.5, high + 0.5)

    def _forecast_horizon_days(self, target_date: str | None) -> int | None:
        if not target_date:
            return None
        try:
            target = datetime.fromisoformat(target_date).date()
        except ValueError:
            return None
        today = datetime.now(timezone.utc).date()
        return (target - today).days

    async def get_supplemental_markets(self) -> list[Market]:
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
        cities = [
            WEATHER_CITY_ALIASES.get(city, city)
            for city in self.cfg.cities
            if WEATHER_CITY_ALIASES.get(city, city) in CITY_COORDINATES
        ]
        if not cities:
            return

        latitudes = ",".join(f"{CITY_COORDINATES[city]['lat']:.4f}" for city in cities)
        longitudes = ",".join(f"{CITY_COORDINATES[city]['lon']:.4f}" for city in cities)
        now = datetime.now(timezone.utc)

        ready_models = 0
        for model_name, endpoint in OPEN_METEO_MODEL_ENDPOINTS.items():
            try:
                resp = await self.client.get(
                    endpoint,
                    params={
                        "latitude": latitudes,
                        "longitude": longitudes,
                        "daily": "temperature_2m_max,temperature_2m_min",
                        "forecast_days": self.cfg.forecast_days,
                        "timezone": "auto",
                        "temperature_unit": "fahrenheit",
                    },
                )
                resp.raise_for_status()
                payloads = resp.json()
                if not isinstance(payloads, list):
                    payloads = [payloads]
            except Exception as exc:
                logger.debug("[WEATHER] %s forecast fetch failed: %s", model_name, exc)
                continue

            for city, payload in zip(cities, payloads):
                daily = payload.get("daily", {})
                times = daily.get("time") or []
                highs = daily.get("temperature_2m_max") or []
                daily_highs = {}
                for date_key, temp in zip(times, highs):
                    if temp is None:
                        continue
                    daily_highs[str(date_key)] = float(temp)

                if not daily_highs:
                    continue

                city_state = self._model_forecasts.setdefault(city, {})
                model_state = city_state.get(model_name, {
                    "current": {},
                    "previous": {},
                    "fetched_at": None,
                    "last_changed_at": None,
                })
                previous_current = model_state.get("current", {})
                if previous_current:
                    model_state["previous"] = previous_current
                if daily_highs != previous_current:
                    model_state["current"] = daily_highs
                    model_state["last_changed_at"] = now
                else:
                    model_state["current"] = daily_highs
                model_state["fetched_at"] = now
                city_state[model_name] = model_state
            ready_models += 1
            await asyncio.sleep(0.05)

        self._stats["models_ready"] = ready_models
        self._stats["forecast_cities"] = sum(
            1
            for city, models in self._model_forecasts.items()
            if len([name for name, state in models.items() if state.get("current")]) >= 2
        )
        self._stats["last_forecast_refresh"] = now.isoformat()
        logger.info(
            "[WEATHER] Refreshed Open-Meteo forecasts for %s cities across %s models",
            self._stats["forecast_cities"],
            ready_models,
        )

    def _match_weather_markets(self, markets: list[Market]):
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

            temp_range = self._extract_temp_range(market.question)
            range_kind = self._temp_range_kind(market.question)
            target_date = self._extract_date(market.question)
            temp_unit = self._extract_temp_unit(market.question)

            if temp_range is None or target_date is None or range_kind is None:
                continue

            self._matched_markets.append({
                "market": market,
                "city": matched_city,
                "temp_range": temp_range,
                "range_kind": range_kind,
                "target_date": target_date,
                "temp_unit": temp_unit,
            })

        self._stats["matched_markets"] = len(self._matched_markets)
        if self._matched_markets:
            logger.info("[WEATHER] Matched %s active weather markets", len(self._matched_markets))
        else:
            logger.info("[WEATHER] No weather markets found on Polymarket right now")

    def _extract_temp_range(self, question: str) -> tuple[float, float] | None:
        question_lower = question.lower()
        for pattern in TEMP_RANGE_PATTERNS:
            match = pattern.search(question_lower)
            if match:
                low = float(match.group(1))
                high = float(match.group(2))
                if 0 <= low <= 150 and 0 <= high <= 150:
                    return (low, high)

        for pattern in TEMP_ABOVE_PATTERNS:
            above_match = pattern.search(question_lower)
            if above_match:
                threshold = float(above_match.group(1))
                return (threshold, threshold + 100)

        for pattern in TEMP_BELOW_PATTERNS:
            below_match = pattern.search(question_lower)
            if below_match:
                threshold = float(below_match.group(1))
                return (-50, threshold)

        exact_match = TEMP_EXACT_PATTERN.search(question_lower)
        if exact_match:
            exact = float(exact_match.group(1))
            return (exact - 0.5, exact + 0.5)

        return None

    def _extract_temp_unit(self, question: str) -> str:
        question_lower = question.lower()
        if "celsius" in question_lower:
            return "C"
        if "fahrenheit" in question_lower:
            return "F"
        if re.search(r"\b\d+\s*°?\s*c\b", question_lower):
            return "C"
        if re.search(r"\b\d+\s*°?\s*f\b", question_lower):
            return "F"
        return "F"

    def _temp_range_kind(self, question: str) -> str | None:
        question_lower = question.lower()
        for pattern in TEMP_ABOVE_PATTERNS:
            if pattern.search(question_lower):
                return "above"
        for pattern in TEMP_BELOW_PATTERNS:
            if pattern.search(question_lower):
                return "below"
        if TEMP_EXACT_PATTERN.search(question_lower):
            return "exact"
        for pattern in TEMP_RANGE_PATTERNS:
            if pattern.search(question_lower):
                return "bounded"
        return None

    def _extract_date(self, question: str) -> str | None:
        months = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
        }

        for month_name, month_num in months.items():
            match = re.search(rf"{month_name}\s+(\d{{1,2}})", question.lower())
            if match:
                day = int(match.group(1))
                year = datetime.now().year
                try:
                    return datetime(year, month_num, day).strftime("%Y-%m-%d")
                except ValueError:
                    return None

        if "tomorrow" in question.lower():
            return (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        if "today" in question.lower():
            return datetime.now().strftime("%Y-%m-%d")
        return None

    def _temp_in_range_probability(
        self,
        forecast_temp: float,
        range_low: float,
        range_high: float,
        *,
        horizon_days: int = 0,
        model_spread_f: float = 0.0,
    ) -> float:
        import math

        std_dev = WEATHER_BASE_STD_DEV_F + (max(horizon_days, 0) * WEATHER_STD_DEV_PER_DAY_F)
        std_dev += max(model_spread_f, 0.0) * WEATHER_MODEL_SPREAD_MULTIPLIER
        if (range_high - range_low) <= 1.5:
            std_dev += WEATHER_NARROW_RANGE_STD_DEV_F

        def norm_cdf(x: float) -> float:
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

        z_low = (range_low - forecast_temp) / std_dev
        z_high = (range_high - forecast_temp) / std_dev
        probability = norm_cdf(z_high) - norm_cdf(z_low)
        return max(0.0001, min(0.9999, probability))

    async def _discover_weather_markets(self) -> list[Market]:
        gamma_host = self.config.api.gamma_host.rstrip("/")
        supported_cities = {
            WEATHER_CITY_ALIASES.get(city, city)
            for city in self.cfg.cities
            if WEATHER_CITY_ALIASES.get(city, city) in CITY_COORDINATES
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
            except Exception as exc:
                logger.debug("[WEATHER] Supplemental event fetch failed at offset=%s: %s", offset, exc)
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
            outcomes.append(Outcome(token_id=token_id, name=name, price=price))

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
            return [part.strip().strip('"') for part in text.split(",") if part.strip()]
        return []

    def _is_weather_market(self, text: str) -> bool:
        return bool(SUPPORTED_WEATHER_MARKET.search(text))

    def _match_city(self, text: str) -> str | None:
        for city, keywords in CITY_KEYWORDS.items():
            for keyword in keywords:
                if re.search(rf"\b{re.escape(keyword)}\b", text):
                    return WEATHER_CITY_ALIASES.get(city, city)
        return None
