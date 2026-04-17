"""
Strategy: External-Only Weather ML Sleeves
==========================================
Two comparison-book only sleeves driven by a frozen external-history CatBoost bundle:

- trader: broader standalone ML trader
- signal: stricter standalone ML signal trader

These sleeves do not affect the main legacy portfolio.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

from data.models import Event, Market, Signal, SignalAction, SignalSource
from runtime_paths import LOG_DIR
from strategies.base import BaseStrategy
from strategies.weather import WeatherForecastStrategy

logger = logging.getLogger(__name__)

try:
    from catboost import CatBoostClassifier, Pool
except Exception:  # pragma: no cover - runtime dependency gate
    CatBoostClassifier = None
    Pool = None


WEATHER_MODEL_VARIANTS = {
    "trader": {
        "label": "Weather Model Trader",
        "source": SignalSource.WEATHER_MODEL_TRADER,
    },
    "signal": {
        "label": "Weather Model Signal",
        "source": SignalSource.WEATHER_MODEL_SIGNAL,
    },
}

KIND_EDGE_FLOORS = {
    "trader": {"above": 0.05, "below": 0.05, "exact": 0.08, "bounded": 0.10, "default": 0.10},
    "signal": {"above": 0.08, "below": 0.08, "exact": 0.11, "bounded": 0.13, "default": 0.12},
}

KIND_PROB_DISTANCE_FLOORS = {
    "trader": {"above": 0.12, "below": 0.12, "exact": 0.14, "bounded": 0.16, "default": 0.15},
    "signal": {"above": 0.18, "below": 0.18, "exact": 0.20, "bounded": 0.22, "default": 0.20},
}


class WeatherModelBundle:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.metadata: dict = {}
        self.models: dict[str, CatBoostClassifier] = {}
        self.ready = False
        self.version = "unloaded"
        self.load_error: str | None = None
        self._load()

    def _load(self):
        if CatBoostClassifier is None or Pool is None:
            self.load_error = "catboost_not_installed"
            logger.warning("[WEATHER_ML] catboost not installed; weather model sleeves disabled")
            return
        metadata_path = self.model_dir / "metadata.json"
        if not metadata_path.exists():
            self.load_error = f"missing_metadata:{metadata_path}"
            logger.warning("[WEATHER_ML] Missing metadata at %s", metadata_path)
            return
        try:
            self.metadata = json.loads(metadata_path.read_text())
            for name in self.metadata.get("models", {}):
                model_path = self.model_dir / f"{name}.cbm"
                if not model_path.exists():
                    continue
                model = CatBoostClassifier()
                model.load_model(str(model_path))
                self.models[name] = model
            self.ready = "overall" in self.models
            self.version = self.metadata.get("bundle_version", "unknown")
            if not self.ready:
                self.load_error = "overall_model_missing"
        except Exception as exc:
            logger.warning("[WEATHER_ML] Failed to load model bundle: %s", exc)
            self.ready = False
            self.load_error = str(exc)

    def available_models(self) -> list[str]:
        return sorted(self.models.keys())

    def predict_yes_probability(self, row: dict, *, temp_kind: str | None) -> dict | None:
        if not self.ready:
            return None
        model_name = temp_kind if temp_kind in self.models else "overall"
        model = self.models.get(model_name) or self.models.get("overall")
        spec = (self.metadata.get("models") or {}).get(model_name) or (self.metadata.get("models") or {}).get("overall")
        if not model or not spec:
            return None

        feature_names = spec["features"]
        categorical = set(spec.get("categorical_features") or [])
        medians = spec.get("medians") or {}
        values = []
        cat_indices = []
        for idx, feature_name in enumerate(feature_names):
            value = row.get(feature_name)
            if feature_name in categorical:
                values.append("__missing__" if value in (None, "") else str(value))
                cat_indices.append(idx)
            else:
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    value = medians.get(feature_name, 0.0)
                values.append(float(value))

        pool = Pool([values], feature_names=feature_names, cat_features=cat_indices)
        prob = float(model.predict_proba(pool)[0][1])
        return {
            "prob_yes": prob,
            "model_name": model_name,
            "model_auc": float(spec.get("roc_auc") or 0.0),
        }


class WeatherModelStrategy(BaseStrategy):
    name = "weather_model"
    description = "External-only CatBoost weather sleeves (comparison-book only)"

    def __init__(self, config, weather_strategy: WeatherForecastStrategy):
        super().__init__(config)
        self.cfg = config.weather_model
        self.weather_strategy = weather_strategy
        self.bundle = WeatherModelBundle(self.cfg.model_dir)
        self.enabled = bool(self.cfg.enabled and self.bundle.ready)
        self.log_path = LOG_DIR / "weather_model_sleeves.jsonl"
        self._stats.update(
            {
                "bundle_ready": self.bundle.ready,
                "bundle_version": self.bundle.version,
                "bundle_error": self.bundle.load_error,
                "model_dir": str(self.bundle.model_dir),
                "models_loaded": self.bundle.available_models(),
                "candidate_markets": 0,
                "scored_markets": 0,
                "last_scan_at": None,
                "last_log_at": None,
                "log_entries": 0,
            }
        )
        self._variant_stats = {
            name: {
                "label": meta["label"],
                "source": meta["source"].value,
                "signals_generated": 0,
                "last_signals": 0,
                "last_candidates": 0,
                "min_edge": getattr(self.cfg, f"{name}_min_edge"),
                "min_prob_distance": getattr(self.cfg, f"{name}_min_prob_distance"),
                "max_token_price": getattr(self.cfg, f"{name}_max_token_price"),
            }
            for name, meta in WEATHER_MODEL_VARIANTS.items()
        }

    async def scan(self, markets: list[Market], events: list[Event]) -> list[Signal]:
        return []

    @property
    def stats(self) -> dict:
        return {
            "name": self.name,
            "enabled": self.enabled,
            **self._stats,
            "log_path": str(self.log_path.resolve()),
            "variants": self._variant_stats,
        }

    def scan_variants(self) -> dict[str, list[Signal]]:
        self._stats["scans_completed"] += 1
        self._stats["last_scan_at"] = datetime.now(timezone.utc).isoformat()
        self._stats["bundle_ready"] = self.bundle.ready
        self._stats["bundle_error"] = self.bundle.load_error
        self._stats["models_loaded"] = self.bundle.available_models()

        if not self.enabled:
            for variant in self._variant_stats.values():
                variant["last_signals"] = 0
                variant["last_candidates"] = 0
            outputs = {"weather_model_trader": [], "weather_model_signal": []}
            self._append_scan_log(outputs)
            return outputs

        candidates = self.weather_strategy.get_model_candidates()
        self._stats["candidate_markets"] = len(candidates)
        self._stats["scored_markets"] = 0

        outputs = {"weather_model_trader": [], "weather_model_signal": []}
        for variant in self._variant_stats.values():
            variant["last_signals"] = 0
            variant["last_candidates"] = len(candidates)

        for candidate in candidates:
            feature_row = self._build_feature_row(candidate)
            if not feature_row:
                continue
            prediction = self.bundle.predict_yes_probability(
                feature_row,
                temp_kind=str(feature_row.get("temp_kind") or ""),
            )
            if not prediction:
                continue
            self._stats["scored_markets"] += 1

            for variant_key in ("trader", "signal"):
                signal = self._build_signal(candidate, feature_row, prediction, variant_key=variant_key)
                if not signal:
                    continue
                outputs[f"weather_model_{variant_key}"].append(signal)
                self._variant_stats[variant_key]["last_signals"] += 1
                self._variant_stats[variant_key]["signals_generated"] += 1

        self._stats["signals_generated"] += sum(
            len(signals) for signals in outputs.values()
        )
        self._append_scan_log(outputs)
        return outputs

    def _append_scan_log(self, outputs: dict[str, list[Signal]]) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": self.name,
            "enabled": self.enabled,
            "bundle_ready": self.bundle.ready,
            "bundle_version": self.bundle.version,
            "bundle_error": self.bundle.load_error,
            "candidate_markets": self._stats.get("candidate_markets", 0),
            "scored_markets": self._stats.get("scored_markets", 0),
            "models_loaded": self.bundle.available_models(),
            "variants": {
                variant_key: {
                    "signals": len(outputs.get(f"weather_model_{variant_key}", [])),
                    "last_signals": stats.get("last_signals", 0),
                    "last_candidates": stats.get("last_candidates", 0),
                }
                for variant_key, stats in self._variant_stats.items()
            },
        }
        try:
            with open(self.log_path, "a") as handle:
                handle.write(json.dumps(entry) + "\n")
            self._stats["log_entries"] += 1
            self._stats["last_log_at"] = entry["timestamp"]
        except Exception as exc:
            logger.warning("[WEATHER_ML] Failed to write weather model log: %s", exc)

    def _build_feature_row(self, candidate: dict) -> dict | None:
        context = candidate.get("context") or {}
        raw_temps = context.get("current_temps") or {}
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

    def _build_signal(
        self,
        candidate: dict,
        feature_row: dict,
        prediction: dict,
        *,
        variant_key: str,
    ) -> Signal | None:
        market = candidate["market"]
        yes_price = float(candidate["yes_price"])
        no_price = float(candidate["no_price"])
        prob_yes = prediction["prob_yes"]
        prob_no = 1.0 - prob_yes
        prob_distance = abs(prob_yes - 0.5)
        temp_kind = str(feature_row.get("temp_kind") or "default")

        min_edge = max(
            getattr(self.cfg, f"{variant_key}_min_edge"),
            KIND_EDGE_FLOORS[variant_key].get(temp_kind, KIND_EDGE_FLOORS[variant_key]["default"]),
        )
        min_prob_distance = max(
            getattr(self.cfg, f"{variant_key}_min_prob_distance"),
            KIND_PROB_DISTANCE_FLOORS[variant_key].get(
                temp_kind,
                KIND_PROB_DISTANCE_FLOORS[variant_key]["default"],
            ),
        )
        max_token_price = getattr(self.cfg, f"{variant_key}_max_token_price")

        yes_edge = prob_yes - yes_price - self.weather_strategy._fee_buffer(yes_price)
        no_edge = prob_no - no_price - self.weather_strategy._fee_buffer(no_price)

        if yes_edge >= no_edge:
            action = SignalAction.BUY_YES
            token_price = yes_price
            edge = yes_edge
            token_prob = prob_yes
            token_id = market.outcomes[0].token_id if market.outcomes else None
        else:
            action = SignalAction.BUY_NO
            token_price = no_price
            edge = no_edge
            token_prob = prob_no
            token_id = market.outcomes[1].token_id if len(market.outcomes) > 1 else None

        if edge < min_edge or prob_distance < min_prob_distance or token_price > max_token_price:
            return None
        if not token_id:
            return None

        size_usd = self._size_for_variant(
            variant_key=variant_key,
            edge=edge,
            prob_distance=prob_distance,
            model_auc=prediction.get("model_auc") or 0.0,
        )
        confidence = min(
            0.99,
            0.55 + (prob_distance * 1.1) + min(edge, 0.20) + max(prediction.get("model_auc", 0.5) - 0.5, 0.0),
        )

        reasoning = (
            f"WEATHER_ML_{variant_key.upper()}: {candidate['city']} {candidate['target_date']} "
            f"| kind={temp_kind} | model={prediction['model_name']} auc={prediction.get('model_auc', 0.0):.3f} "
            f"| yes_prob={prob_yes:.0%} | market_yes={yes_price:.0%} | market_no={no_price:.0%} "
            f"| picked={'YES' if action == SignalAction.BUY_YES else 'NO'} edge={edge:.1%}"
        )

        return Signal(
            source=WEATHER_MODEL_VARIANTS[variant_key]["source"],
            action=action,
            market_slug=market.slug,
            condition_id=market.condition_id,
            token_id=token_id,
            confidence=max(0.05, min(confidence, 0.99)),
            expected_edge=edge * 100.0,
            group_key=self.weather_strategy._weather_group_key(candidate["city"], candidate["target_date"]),
            reasoning=reasoning,
            suggested_size_usd=size_usd,
        )

    def _size_for_variant(self, *, variant_key: str, edge: float, prob_distance: float, model_auc: float) -> float:
        min_size = getattr(self.cfg, f"{variant_key}_min_size_usd")
        max_size = getattr(self.cfg, f"{variant_key}_max_size_usd")
        auc_bonus = max(model_auc - 0.70, 0.0) * 120.0
        raw_size = min_size + (edge * 220.0) + (prob_distance * 140.0) + auc_bonus
        return min(max_size, max(min_size, raw_size))

    @staticmethod
    def _convert_temperature(value_f: float, target_unit: str) -> float:
        if target_unit == "C":
            return (float(value_f) - 32.0) * 5.0 / 9.0
        return float(value_f)

    @staticmethod
    def _maybe_float(value):
        if value is None:
            return None
        return float(value)
