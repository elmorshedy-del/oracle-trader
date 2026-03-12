"""
Strategy: External-Only Weather ML Sleeves v2
=============================================
Side-by-side experiment that builds on the frozen weather ML baseline while
adding richer v2 features and optional learner stacking.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data.models import Event, Market, Signal, SignalAction, SignalSource
from runtime_paths import LOG_DIR
from strategies.base import BaseStrategy
from strategies.weather import WeatherForecastStrategy
from engine.weather_feature_builder_v2 import WeatherFeatureBuilderV2

logger = logging.getLogger(__name__)

try:
    from catboost import CatBoostClassifier, Pool
except Exception:  # pragma: no cover - runtime dependency gate
    CatBoostClassifier = None
    Pool = None


WEATHER_MODEL_V2_VARIANTS = {
    "trader": {
        "label": "Weather Model V2 Trader",
        "source": SignalSource.WEATHER_MODEL_V2_TRADER,
    },
    "signal": {
        "label": "Weather Model V2 Signal",
        "source": SignalSource.WEATHER_MODEL_V2_SIGNAL,
    },
}


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _stable_string_number(value: Any) -> float:
    text = str(value)
    return float(sum(ord(ch) for ch in text) % 10000)


class StackedModelBundle:
    def __init__(self, model_dir: str | Path, fallback_model_dir: str | Path | None = None):
        self.requested_model_dir = Path(model_dir)
        self.fallback_model_dir = Path(fallback_model_dir) if fallback_model_dir else None
        self.active_model_dir = self.requested_model_dir
        self.metadata: dict = {}
        self.models: dict[str, dict[str, Any]] = {}
        self.ready = False
        self.version = "unloaded"
        self.load_error: str | None = None
        self.using_fallback_bundle = False
        self._load()

    def _load(self):
        if self._load_from_dir(self.requested_model_dir):
            return
        if self.fallback_model_dir and self._load_from_dir(self.fallback_model_dir):
            self.using_fallback_bundle = True
            self.active_model_dir = self.fallback_model_dir
            return
        if not self.load_error:
            self.load_error = f"missing_bundle:{self.requested_model_dir}"

    def _load_from_dir(self, model_dir: Path) -> bool:
        metadata_path = model_dir / "metadata.json"
        if not metadata_path.exists():
            self.load_error = f"missing_metadata:{metadata_path}"
            return False
        try:
            metadata = json.loads(metadata_path.read_text())
        except Exception as exc:  # pragma: no cover - defensive
            self.load_error = f"metadata_parse_error:{exc}"
            return False

        loaded_models: dict[str, dict[str, Any]] = {}
        for kind in (metadata.get("models") or {}):
            loaded_models[kind] = {}
            spec = (metadata.get("models") or {}).get(kind) or {}

            if CatBoostClassifier is not None and Pool is not None:
                cbm_path = model_dir / f"{kind}.cbm"
                if not cbm_path.exists():
                    alt = spec.get("file")
                    if alt:
                        cbm_path = model_dir / alt
                if cbm_path.exists():
                    try:
                        model = CatBoostClassifier()
                        model.load_model(str(cbm_path))
                        loaded_models[kind]["catboost"] = model
                    except Exception as exc:
                        logger.warning("[WEATHER_ML_V2] Failed to load CatBoost %s from %s: %s", kind, cbm_path, exc)

            lgb_path = model_dir / f"{kind}_lgb.txt"
            if lgb_path.exists() and _module_available("lightgbm"):
                try:
                    import lightgbm as lgb  # type: ignore
                    loaded_models[kind]["lightgbm"] = lgb.Booster(model_file=str(lgb_path))
                except Exception as exc:
                    logger.warning("[WEATHER_ML_V2] Failed to load LightGBM %s: %s", kind, exc)

            xgb_path = model_dir / f"{kind}_xgb.json"
            if xgb_path.exists() and _module_available("xgboost"):
                try:
                    import xgboost as xgb  # type: ignore
                    model = xgb.Booster()
                    model.load_model(str(xgb_path))
                    loaded_models[kind]["xgboost"] = model
                except Exception as exc:
                    logger.warning("[WEATHER_ML_V2] Failed to load XGBoost %s: %s", kind, exc)

        if not any(loaded_models.values()):
            self.load_error = f"no_models_loaded:{model_dir}"
            return False

        self.metadata = metadata
        self.models = loaded_models
        self.ready = "overall" in self.models and bool(self.models["overall"])
        self.version = metadata.get("bundle_version", model_dir.name)
        self.load_error = None if self.ready else "overall_model_missing"
        self.active_model_dir = model_dir
        return self.ready

    def available_models(self) -> dict[str, list[str]]:
        return {kind: sorted(learners.keys()) for kind, learners in self.models.items() if learners}

    def loaded_model_count(self) -> int:
        return sum(len(v) for v in self.models.values())

    def predict_yes_probability(self, row: dict, *, temp_kind: str | None) -> dict | None:
        if not self.ready:
            return None

        requested_kind = temp_kind if temp_kind in self.models and self.models.get(temp_kind) else "overall"
        model_kind = requested_kind if self.models.get(requested_kind) else "overall"
        kind_models = self.models.get(model_kind) or self.models.get("overall")
        spec = (self.metadata.get("models") or {}).get(model_kind) or (self.metadata.get("models") or {}).get("overall")
        if not kind_models or not spec:
            return None

        feature_names = spec.get("features") or []
        categorical = set(spec.get("categorical_features") or [])
        medians = spec.get("medians") or {}
        values = []
        cat_indices = []
        row_data: dict[str, Any] = {}
        for idx, feature_name in enumerate(feature_names):
            value = row.get(feature_name)
            if feature_name in categorical:
                value = "__missing__" if value in (None, "") else str(value)
                cat_indices.append(idx)
            else:
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    value = medians.get(feature_name, 0.0)
                value = float(value)
            values.append(value)
            row_data[feature_name] = value

        predictions: dict[str, float] = {}
        catboost_model = kind_models.get("catboost")
        if catboost_model is not None and Pool is not None:
            try:
                pool = Pool([values], feature_names=feature_names, cat_features=cat_indices)
                predictions["catboost"] = float(catboost_model.predict_proba(pool)[0][1])
            except Exception as exc:
                logger.debug("[WEATHER_ML_V2] CatBoost predict failed for %s: %s", model_kind, exc)

        numeric_row = {
            name: (_stable_string_number(value) if name in categorical else float(value))
            for name, value in row_data.items()
        }

        lgb_model = kind_models.get("lightgbm")
        if lgb_model is not None:
            try:
                import pandas as pd  # type: ignore
                predictions["lightgbm"] = float(lgb_model.predict(pd.DataFrame([numeric_row]))[0])
            except Exception as exc:
                logger.debug("[WEATHER_ML_V2] LightGBM predict failed for %s: %s", model_kind, exc)

        xgb_model = kind_models.get("xgboost")
        if xgb_model is not None:
            try:
                import pandas as pd  # type: ignore
                import xgboost as xgb  # type: ignore
                dmat = xgb.DMatrix(pd.DataFrame([numeric_row]))
                predictions["xgboost"] = float(xgb_model.predict(dmat)[0])
            except Exception as exc:
                logger.debug("[WEATHER_ML_V2] XGBoost predict failed for %s: %s", model_kind, exc)

        if not predictions:
            return None

        return {
            "prob_yes": float(statistics.mean(predictions.values())),
            "model_name": model_kind,
            "model_auc": float(spec.get("roc_auc") or 0.0),
            "individual_scores": predictions,
            "stacked": len(predictions) > 1,
            "num_models": len(predictions),
        }


class WeatherModelStrategyV2(BaseStrategy):
    name = "weather_model_v2"
    description = "Stacked weather ML v2 sleeves (comparison-book only)"

    def __init__(self, config, weather_strategy: WeatherForecastStrategy):
        super().__init__(config)
        self.cfg = config.weather_model_v2
        self.weather_strategy = weather_strategy
        self.feature_builder = WeatherFeatureBuilderV2()
        self.bundle = StackedModelBundle(
            self.cfg.model_dir,
            fallback_model_dir=self.cfg.fallback_model_dir,
        )
        self.enabled = bool(self.cfg.enabled and self.bundle.ready)
        self.log_path = LOG_DIR / "weather_model_v2_sleeves.jsonl"
        self._stats.update(
            {
                "bundle_ready": self.bundle.ready,
                "bundle_version": self.bundle.version,
                "bundle_error": self.bundle.load_error,
                "requested_model_dir": str(self.bundle.requested_model_dir),
                "active_model_dir": str(self.bundle.active_model_dir),
                "using_fallback_bundle": self.bundle.using_fallback_bundle,
                "models_loaded": self.bundle.available_models(),
                "loaded_model_count": self.bundle.loaded_model_count(),
                "history_manifest_path": self.cfg.history_manifest_path,
                "candidate_markets": 0,
                "scored_markets": 0,
                "v2_features_count": 0,
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
            for name, meta in WEATHER_MODEL_V2_VARIANTS.items()
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
            "feature_builder": self.feature_builder.stats,
            "variants": self._variant_stats,
        }

    async def close(self):
        await self.feature_builder.close()

    def scan_variants(self) -> dict[str, list[Signal]]:
        self._stats["scans_completed"] += 1
        self._stats["last_scan_at"] = datetime.now(timezone.utc).isoformat()
        self._stats["bundle_ready"] = self.bundle.ready
        self._stats["bundle_error"] = self.bundle.load_error
        self._stats["models_loaded"] = self.bundle.available_models()
        self._stats["loaded_model_count"] = self.bundle.loaded_model_count()
        self._stats["active_model_dir"] = str(self.bundle.active_model_dir)
        self._stats["using_fallback_bundle"] = self.bundle.using_fallback_bundle

        outputs = {
            "weather_model_v2_trader": [],
            "weather_model_v2_signal": [],
        }
        if not self.enabled:
            for variant in self._variant_stats.values():
                variant["last_signals"] = 0
                variant["last_candidates"] = 0
            self._append_scan_log(outputs)
            return outputs

        candidates = self.weather_strategy.get_model_candidates()
        self._stats["candidate_markets"] = len(candidates)
        self._stats["scored_markets"] = 0
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
            self._stats["v2_features_count"] = max(self._stats["v2_features_count"], len(feature_row))

            for variant_key in ("trader", "signal"):
                signal = self._build_signal(candidate, feature_row, prediction, variant_key=variant_key)
                if not signal:
                    continue
                outputs[f"weather_model_v2_{variant_key}"].append(signal)
                self._variant_stats[variant_key]["last_signals"] += 1
                self._variant_stats[variant_key]["signals_generated"] += 1

        self._stats["signals_generated"] += sum(len(signals) for signals in outputs.values())
        self._append_scan_log(outputs)
        return outputs

    def _build_feature_row(self, candidate: dict) -> dict | None:
        market = candidate.get("market")
        condition_id = getattr(market, "condition_id", None)
        price_history = []
        if condition_id:
            history = self.weather_strategy._market_price_history.get(condition_id) or []
            price_history = [
                {
                    "timestamp": entry.get("time").isoformat() if entry.get("time") else None,
                    "yes_price": float(entry.get("yes", 0.0) or 0.0),
                    "no_price": float(entry.get("no", 0.0) or 0.0),
                }
                for entry in history
            ]
        return self.feature_builder.build_v2_feature_row(
            candidate=candidate,
            context=candidate.get("context"),
            market_price_history=price_history,
        )

    def _append_scan_log(self, outputs: dict[str, list[Signal]]) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": self.name,
            "enabled": self.enabled,
            "bundle_ready": self.bundle.ready,
            "bundle_version": self.bundle.version,
            "bundle_error": self.bundle.load_error,
            "requested_model_dir": str(self.bundle.requested_model_dir),
            "active_model_dir": str(self.bundle.active_model_dir),
            "using_fallback_bundle": self.bundle.using_fallback_bundle,
            "candidate_markets": self._stats.get("candidate_markets", 0),
            "scored_markets": self._stats.get("scored_markets", 0),
            "v2_features_count": self._stats.get("v2_features_count", 0),
            "models_loaded": self.bundle.available_models(),
            "feature_builder": self.feature_builder.stats,
            "variants": {
                variant_key: {
                    "signals": len(outputs.get(f"weather_model_v2_{variant_key}", [])),
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
            logger.warning("[WEATHER_ML_V2] Failed to write scan log: %s", exc)

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

        ensemble_std = float(feature_row.get("ensemble_std") or 0.0)
        forecast_stability = float(feature_row.get("forecast_stability") or 0.0)
        if ensemble_std > 0 and ensemble_std < 1.5:
            edge *= 1.05
        if forecast_stability > 0.8:
            edge *= 1.03

        min_edge = getattr(self.cfg, f"{variant_key}_min_edge")
        min_prob_distance = getattr(self.cfg, f"{variant_key}_min_prob_distance")
        max_token_price = getattr(self.cfg, f"{variant_key}_max_token_price")

        if edge < min_edge or prob_distance < min_prob_distance or token_price > max_token_price:
            return None
        if not token_id:
            return None

        size_usd = self._size_for_variant(variant_key=variant_key, edge=edge, prob_distance=prob_distance, prediction=prediction)
        confidence = min(
            0.99,
            0.55
            + (prob_distance * 1.1)
            + min(edge, 0.20)
            + max((prediction.get("model_auc") or 0.5) - 0.5, 0.0)
            + (0.02 if prediction.get("stacked") else 0.0),
        )

        individual_scores = prediction.get("individual_scores") or {}
        stacked_info = ""
        if individual_scores:
            stacked_info = " | " + ", ".join(
                f"{name}={score:.2f}" for name, score in sorted(individual_scores.items())
            )

        reasoning = (
            f"WEATHER_ML_V2_{variant_key.upper()}: {candidate['city']} {candidate['target_date']} "
            f"| kind={feature_row.get('temp_kind')} | model={prediction.get('model_name')} "
            f"auc={prediction.get('model_auc', 0.0):.3f} | yes_prob={prob_yes:.0%} "
            f"| market_yes={yes_price:.0%} | market_no={no_price:.0%} "
            f"| picked={'YES' if action == SignalAction.BUY_YES else 'NO'} edge={edge:.1%} "
            f"| stability={forecast_stability:.2f} | ensemble_std={ensemble_std:.2f}{stacked_info}"
        )

        return Signal(
            source=WEATHER_MODEL_V2_VARIANTS[variant_key]["source"],
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

    def _size_for_variant(self, *, variant_key: str, edge: float, prob_distance: float, prediction: dict) -> float:
        min_size = getattr(self.cfg, f"{variant_key}_min_size_usd")
        max_size = getattr(self.cfg, f"{variant_key}_max_size_usd")
        auc_bonus = max((prediction.get("model_auc") or 0.70) - 0.70, 0.0) * 120.0
        stacking_bonus = 10.0 if prediction.get("stacked") else 0.0
        raw_size = min_size + (edge * 220.0) + (prob_distance * 140.0) + auc_bonus + stacking_bonus
        return min(max_size, max(min_size, raw_size))
