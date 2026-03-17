from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

try:
    from catboost import CatBoostClassifier, Pool
except Exception:  # pragma: no cover - runtime dependency gate
    CatBoostClassifier = None
    Pool = None


class FrozenWeatherModelBundle:
    def __init__(self, bundle_dir: str | Path):
        self.bundle_dir = Path(bundle_dir)
        self.metadata: dict = {}
        self.models: dict[str, CatBoostClassifier] = {}
        self.ready = False
        self.load_error: str | None = None
        self._load()

    def _load(self) -> None:
        metadata_path = self.bundle_dir / "metadata.json"
        if CatBoostClassifier is None or Pool is None:
            self.load_error = "catboost_not_installed"
            return
        if not metadata_path.exists():
            self.load_error = f"missing_metadata:{metadata_path}"
            return
        try:
            self.metadata = json.loads(metadata_path.read_text())
            for model_name in (self.metadata.get("models") or {}):
                model_path = self.bundle_dir / f"{model_name}.cbm"
                if not model_path.exists():
                    continue
                model = CatBoostClassifier()
                model.load_model(str(model_path))
                self.models[model_name] = model
            self.ready = "overall" in self.models
            if not self.ready:
                self.load_error = "overall_model_missing"
        except Exception as exc:  # pragma: no cover - defensive load surface
            self.load_error = str(exc)
            self.ready = False

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
        values: list[object] = []
        cat_indices: list[int] = []
        for idx, feature_name in enumerate(feature_names):
            value = row.get(feature_name)
            if feature_name in categorical:
                values.append("__missing__" if value in (None, "") else str(value))
                cat_indices.append(idx)
                continue
            if value is None or (isinstance(value, float) and math.isnan(value)):
                value = medians.get(feature_name, 0.0)
            values.append(float(value))

        pool = Pool([values], feature_names=feature_names, cat_features=cat_indices)
        probability_yes = float(model.predict_proba(pool)[0][1])
        return {
            "prob_yes": probability_yes,
            "model_name": model_name,
            "model_auc": float(spec.get("roc_auc") or 0.0),
        }


def build_baseline_feature_row(
    *,
    city: str,
    temp_unit: str,
    temp_kind: str,
    temp_range_low: float,
    temp_range_high: float,
    target_month: int | None,
    target_day: int | None,
    market_yes_price: float,
    forecast_temp_max: float | None,
    model_temperatures: dict[str, float | None],
) -> dict:
    valid_model_temperatures = {
        model_name: float(value)
        for model_name, value in model_temperatures.items()
        if value is not None
    }
    temp_values = list(valid_model_temperatures.values())
    forecast_available = 1.0 if forecast_temp_max is not None else 0.0
    temp_mean = statistics.mean(temp_values) if temp_values else None
    temp_spread = (max(temp_values) - min(temp_values)) if len(temp_values) >= 2 else 0.0
    temp_std = statistics.pstdev(temp_values) if len(temp_values) >= 2 else 0.0
    range_center = (float(temp_range_low) + float(temp_range_high)) / 2.0
    forecast_reference = float(forecast_temp_max) if forecast_temp_max is not None else temp_mean
    if forecast_reference is None:
        raise ValueError("At least one model temperature or a forecast_temp_max value is required.")

    return {
        "city": city,
        "temp_unit": temp_unit,
        "temp_kind": temp_kind,
        "temp_range_low": float(temp_range_low),
        "temp_range_high": float(temp_range_high),
        "range_width": float(temp_range_high - temp_range_low),
        "range_center": float(range_center),
        "target_month": target_month,
        "target_day": target_day,
        "first_yes_price": float(market_yes_price),
        "forecast_available": forecast_available,
        "forecast_temp_max": float(forecast_reference),
        "model_count_available": float(len(temp_values)),
        "temp_max_mean": float(temp_mean if temp_mean is not None else forecast_reference),
        "temp_max_spread": float(temp_spread),
        "temp_max_std": float(temp_std),
        "gfs_seamless_temp_max": _maybe_float(valid_model_temperatures.get("gfs_seamless")),
        "icon_seamless_temp_max": _maybe_float(valid_model_temperatures.get("icon_seamless")),
        "temp_max_bucket_gap": float((temp_mean if temp_mean is not None else forecast_reference) - range_center),
        "temp_max_in_bucket": float(temp_range_low <= (temp_mean if temp_mean is not None else forecast_reference) <= temp_range_high),
    }


def probability_from_temperature(
    *,
    temp_kind: str,
    temp_range_low: float,
    temp_range_high: float,
    forecast_temp: float | None,
) -> float | None:
    if forecast_temp is None:
        return None

    forecast_temp = float(forecast_temp)
    if temp_kind == "above":
        return 1.0 if forecast_temp > temp_range_high else 0.0
    if temp_kind == "below":
        return 1.0 if forecast_temp < temp_range_low else 0.0
    if temp_kind == "exact":
        return 1.0 if temp_range_low <= forecast_temp <= temp_range_high else 0.0
    if temp_kind == "bounded":
        return 1.0 if temp_range_low <= forecast_temp <= temp_range_high else 0.0
    return None


def _maybe_float(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)
