"""
Retrain weather v2 models from the preserved weather history manifest.

Builds on the existing structured weather calibration dataset and historical
forecast artifacts, then emits a CatBoost-first v2 bundle with optional
LightGBM/XGBoost challengers if those libraries are available locally.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
logger = logging.getLogger('retrain_weather_v2')

MODEL_KINDS = ('overall', 'above', 'below', 'bounded', 'exact')
CATEGORICAL_FEATURES = ['city', 'temp_unit', 'temp_kind']
BASE_FEATURES = [
    'city', 'temp_unit', 'temp_kind',
    'temp_range_low', 'temp_range_high', 'range_width', 'range_center',
    'target_month', 'target_day',
    'first_yes_price', 'forecast_available',
    'forecast_temp_max', 'model_count_available',
    'temp_max_mean', 'temp_max_spread', 'temp_max_std',
    'gfs_seamless_temp_max', 'icon_seamless_temp_max',
    'temp_max_bucket_gap', 'temp_max_in_bucket',
]
V2_FEATURES = [
    'forecast_shift_mean', 'forecast_shift_max', 'forecast_shift_direction',
    'forecast_consensus_shift', 'forecast_stability', 'models_changed_count',
    'prob_shift', 'climatology_normal', 'climatology_anomaly',
    'climatology_anomaly_abs', 'is_extreme_forecast', 'hours_to_resolution',
    'is_same_day', 'is_next_day', 'season', 'day_of_week',
    'market_price_change_1h', 'market_price_change_3h', 'market_price_velocity',
    'market_price_vs_model', 'market_volume_signal', 'ensemble_mean', 'ensemble_std',
    'ensemble_spread', 'ensemble_p10', 'ensemble_p90', 'ensemble_skew',
]
ALL_FEATURES = BASE_FEATURES + V2_FEATURES
CLIMATOLOGY_NORMALS_F = {
    'new-york': {1: 39, 2: 42, 3: 50, 4: 62, 5: 72, 6: 80, 7: 85, 8: 84, 9: 76, 10: 65, 11: 54, 12: 43},
    'chicago': {1: 32, 2: 36, 3: 47, 4: 59, 5: 70, 6: 80, 7: 84, 8: 82, 9: 75, 10: 62, 11: 48, 12: 35},
    'los-angeles': {1: 68, 2: 69, 3: 70, 4: 72, 5: 74, 6: 78, 7: 84, 8: 85, 9: 83, 10: 79, 11: 73, 12: 68},
    'miami': {1: 76, 2: 78, 3: 80, 4: 83, 5: 87, 6: 90, 7: 91, 8: 91, 9: 89, 10: 86, 11: 82, 12: 78},
    'london': {1: 46, 2: 47, 3: 52, 4: 57, 5: 63, 6: 69, 7: 73, 8: 73, 9: 67, 10: 59, 11: 51, 12: 46},
    'seoul': {1: 34, 2: 39, 3: 50, 4: 62, 5: 72, 6: 80, 7: 84, 8: 85, 9: 78, 10: 66, 11: 52, 12: 38},
}
CITY_ALIASES = {'new york': 'new-york', 'los angeles': 'los-angeles'}


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def canonical_city(value: str) -> str:
    normalized = str(value or '').strip().lower().replace('_', '-').replace(' ', '-')
    return CITY_ALIASES.get(normalized, normalized)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_price_history(history: list[dict]) -> tuple[float, float, float, float]:
    if not history:
        return 0.0, 0.0, 0.0, 0.0
    points = [(float(item.get('t', 0) or 0), float(item.get('p', 0) or 0)) for item in history]
    points.sort(key=lambda item: item[0])
    first_price = points[0][1]
    last_price = points[-1][1]
    now_ts = points[-1][0]
    one_hour_cut = now_ts - 3600
    three_hour_cut = now_ts - 10800
    one_hour_start = next((p for t, p in points if t >= one_hour_cut), first_price)
    three_hour_start = next((p for t, p in points if t >= three_hour_cut), first_price)
    duration = max(now_ts - points[0][0], 1)
    velocity = (last_price - first_price) / duration
    return first_price, last_price - one_hour_start, last_price - three_hour_start, velocity


def build_dataset(manifest: dict) -> pd.DataFrame:
    markets = pd.DataFrame(load_jsonl(Path(manifest['source_dataset'])))
    metadata = pd.DataFrame(load_jsonl(Path(manifest['artifacts']['market_metadata'])))
    forecast = pd.DataFrame(load_jsonl(Path(manifest['artifacts']['forecast_features'])))
    multimodel = pd.DataFrame(load_jsonl(Path(manifest['artifacts']['multimodel_features'])))

    markets['city'] = markets['city'].map(canonical_city)
    forecast['city'] = forecast['city'].map(canonical_city)
    multimodel['city'] = multimodel['city'].map(canonical_city)

    base = markets.merge(metadata[['market_id', 'created_at', 'lead_hours', 'volume_clob']], on='market_id', how='left')
    base = base.merge(forecast[['city', 'target_date', 'temp_unit', 'forecast_available', 'forecast_temp_max']], on=['city', 'target_date', 'temp_unit'], how='left')
    base = base.merge(multimodel, on=['city', 'target_date', 'temp_unit'], how='left', suffixes=('', '_multi'))

    first_prices = []
    change_1h = []
    change_3h = []
    velocity = []
    for history in base['yes_price_history'].tolist():
        fp, c1, c3, vel = parse_price_history(history or [])
        first_prices.append(fp)
        change_1h.append(c1)
        change_3h.append(c3)
        velocity.append(vel)

    base['first_yes_price'] = np.where(np.array(first_prices) > 0, first_prices, pd.to_numeric(base['outcome_prices'].apply(lambda xs: (xs or ['0'])[0]), errors='coerce').fillna(0.0))
    base['market_price_change_1h'] = change_1h
    base['market_price_change_3h'] = change_3h
    base['market_price_velocity'] = velocity
    base['market_volume_signal'] = pd.to_numeric(base.get('volume_clob'), errors='coerce').fillna(0.0)

    base['temp_range_low'] = pd.to_numeric(base['temp_range_low'], errors='coerce').fillna(0.0)
    base['temp_range_high'] = pd.to_numeric(base['temp_range_high'], errors='coerce').fillna(0.0)
    base['range_width'] = base['temp_range_high'] - base['temp_range_low']
    base['range_center'] = (base['temp_range_high'] + base['temp_range_low']) / 2.0
    base['target_month'] = pd.to_datetime(base['target_date'], errors='coerce').dt.month.fillna(0).astype(int)
    base['target_day'] = pd.to_datetime(base['target_date'], errors='coerce').dt.day.fillna(0).astype(int)
    base['forecast_available'] = base['forecast_available'].fillna(True).astype(float)
    base['model_count_available'] = pd.to_numeric(base.get('model_count_available'), errors='coerce').fillna(0.0)
    base['temp_max_mean'] = pd.to_numeric(base.get('temp_max_mean'), errors='coerce').fillna(pd.to_numeric(base.get('forecast_temp_max'), errors='coerce').fillna(0.0))
    base['temp_max_spread'] = pd.to_numeric(base.get('temp_max_spread'), errors='coerce').fillna(0.0)
    base['temp_max_std'] = pd.to_numeric(base.get('temp_max_std'), errors='coerce').fillna(0.0)
    base['gfs_seamless_temp_max'] = pd.to_numeric(base.get('gfs_seamless_temp_max'), errors='coerce').fillna(base['temp_max_mean'])
    base['icon_seamless_temp_max'] = pd.to_numeric(base.get('icon_seamless_temp_max'), errors='coerce').fillna(base['temp_max_mean'])
    base['temp_max_bucket_gap'] = base['temp_max_mean'] - base['range_center']
    base['temp_max_in_bucket'] = ((base['temp_max_mean'] >= base['temp_range_low']) & (base['temp_max_mean'] <= base['temp_range_high'])).astype(float)

    base['climatology_normal'] = base.apply(lambda row: CLIMATOLOGY_NORMALS_F.get(row['city'], {}).get(int(row['target_month'] or 0), 60), axis=1)
    mean_f = np.where(base['temp_unit'].astype(str).str.upper() == 'C', (base['temp_max_mean'] * 9.0 / 5.0) + 32.0, base['temp_max_mean'])
    base['climatology_anomaly'] = mean_f - base['climatology_normal']
    base['climatology_anomaly_abs'] = base['climatology_anomaly'].abs()
    base['is_extreme_forecast'] = (base['climatology_anomaly_abs'] > 15).astype(float)
    base['hours_to_resolution'] = pd.to_numeric(base.get('lead_hours'), errors='coerce').fillna(0.0)
    base['is_same_day'] = (base['hours_to_resolution'] <= 24).astype(float)
    base['is_next_day'] = ((base['hours_to_resolution'] > 24) & (base['hours_to_resolution'] <= 48)).astype(float)
    base['season'] = base['target_month'].apply(lambda m: 0 if m in (12, 1, 2) else (1 if m in (3, 4, 5) else (2 if m in (6, 7, 8) else 3)))
    base['day_of_week'] = pd.to_datetime(base['target_date'], errors='coerce').dt.dayofweek.fillna(0).astype(int)

    base['forecast_shift_mean'] = 0.0
    base['forecast_shift_max'] = 0.0
    base['forecast_shift_direction'] = 0.0
    base['forecast_consensus_shift'] = 0.0
    base['forecast_stability'] = 1.0
    base['models_changed_count'] = 0.0
    base['prob_shift'] = 0.0

    base['market_price_vs_model'] = base['first_yes_price'] - base['resolved_yes'].astype(float)
    base['ensemble_mean'] = base['temp_max_mean']
    base['ensemble_std'] = base['temp_max_std']
    base['ensemble_spread'] = base['temp_max_spread']
    base['ensemble_p10'] = base['temp_max_mean'] - base['temp_max_std']
    base['ensemble_p90'] = base['temp_max_mean'] + base['temp_max_std']
    base['ensemble_skew'] = 0.0

    for feature in ALL_FEATURES:
        if feature not in base.columns:
            base[feature] = 0.0

    base['resolved_yes'] = base['resolved_yes'].astype(int)
    base = base.dropna(subset=['resolved_yes'])
    base = base.sort_values(['target_date', 'market_id']).reset_index(drop=True)
    return base


def split_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unique_dates = sorted(df['target_date'].dropna().unique().tolist())
    if len(unique_dates) < 6:
        raise ValueError('Need at least 6 target dates for train/validation/test split')
    train_cut = int(len(unique_dates) * 0.70)
    val_cut = int(len(unique_dates) * 0.85)
    train_dates = set(unique_dates[:train_cut])
    val_dates = set(unique_dates[train_cut:val_cut])
    test_dates = set(unique_dates[val_cut:])
    return (
        df[df['target_date'].isin(train_dates)].copy(),
        df[df['target_date'].isin(val_dates)].copy(),
        df[df['target_date'].isin(test_dates)].copy(),
    )


def train_catboost(X_train, y_train, X_val, y_val, cat_features):
    import catboost
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    cat_indices = [X_train.columns.tolist().index(col) for col in cat_features if col in X_train.columns]
    model = catboost.CatBoostClassifier(
        iterations=600,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        eval_metric='AUC',
        cat_features=cat_indices,
        verbose=False,
        early_stopping_rounds=60,
        auto_class_weights='Balanced',
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    preds = model.predict_proba(X_val)[:, 1]
    return model, {
        'roc_auc': float(roc_auc_score(y_val, preds)),
        'log_loss': float(log_loss(y_val, preds)),
        'brier_score': float(brier_score_loss(y_val, preds)),
        'rows': int(len(X_train) + len(X_val)),
        'positive_rate': float(y_train.mean()),
    }


def train_lightgbm(X_train, y_train, X_val, y_val, cat_features):
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    X_train = X_train.copy()
    X_val = X_val.copy()
    for col in cat_features:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype('category')
            X_val[col] = X_val[col].astype('category')
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=[c for c in cat_features if c in X_train.columns])
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    model = lgb.train(
        {
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'is_unbalance': True,
        },
        train_data,
        num_boost_round=600,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(60), lgb.log_evaluation(0)],
    )
    preds = model.predict(X_val)
    return model, {'roc_auc': float(roc_auc_score(y_val, preds))}


def train_xgboost(X_train, y_train, X_val, y_val):
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    X_train = X_train.select_dtypes(include=[np.number])
    X_val = X_val.select_dtypes(include=[np.number])
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    model = xgb.train(
        {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1.0),
        },
        dtrain,
        num_boost_round=600,
        evals=[(dval, 'val')],
        early_stopping_rounds=60,
        verbose_eval=False,
    )
    preds = model.predict(dval)
    return model, {'roc_auc': float(roc_auc_score(y_val, preds))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='models/weather_ml/weather_v2_history_sources.json')
    parser.add_argument('--model-dir', default=None)
    parser.add_argument('--stacking', action='store_true')
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    model_dir = Path(args.model_dir or manifest['v2_bundle_target']).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataset(manifest)
    train_df, val_df, test_df = split_df(df)
    logger.info('built dataset rows=%s train=%s val=%s test=%s', len(df), len(train_df), len(val_df), len(test_df))

    metadata = {
        'bundle_version': 'legacy-weather-ml-v2',
        'source_dataset': manifest['source_dataset'],
        'history_manifest': str(manifest_path),
        'features': ALL_FEATURES,
        'categorical_features': CATEGORICAL_FEATURES,
        'models': {},
    }

    for kind in MODEL_KINDS:
        kind_train = train_df if kind == 'overall' else train_df[train_df['temp_kind'] == kind]
        kind_val = val_df if kind == 'overall' else val_df[val_df['temp_kind'] == kind]
        if len(kind_train) < 50 or len(kind_val) < 20:
            logger.info('skip %s because dataset too small train=%s val=%s', kind, len(kind_train), len(kind_val))
            continue
        X_train = kind_train[ALL_FEATURES].copy()
        X_val = kind_val[ALL_FEATURES].copy()
        y_train = kind_train['resolved_yes']
        y_val = kind_val['resolved_yes']

        cb_model, cb_metrics = train_catboost(X_train, y_train, X_val, y_val, CATEGORICAL_FEATURES)
        cb_model.save_model(str(model_dir / f'{kind}.cbm'))
        kind_meta = {
            **cb_metrics,
            'accuracy': None,
            'features': ALL_FEATURES,
            'categorical_features': CATEGORICAL_FEATURES,
            'medians': {col: float(pd.to_numeric(X_train[col], errors='coerce').median()) for col in ALL_FEATURES if col not in CATEGORICAL_FEATURES},
        }

        if args.stacking and module_available('lightgbm'):
            try:
                lgb_model, lgb_metrics = train_lightgbm(X_train, y_train, X_val, y_val, CATEGORICAL_FEATURES)
                lgb_model.save_model(str(model_dir / f'{kind}_lgb.txt'))
                kind_meta['lightgbm_roc_auc'] = lgb_metrics['roc_auc']
            except Exception as exc:
                logger.warning('lightgbm %s failed: %s', kind, exc)

        if args.stacking and module_available('xgboost'):
            try:
                xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_val, y_val)
                xgb_model.save_model(str(model_dir / f'{kind}_xgb.json'))
                kind_meta['xgboost_roc_auc'] = xgb_metrics['roc_auc']
            except Exception as exc:
                logger.warning('xgboost %s failed: %s', kind, exc)

        metadata['models'][kind] = kind_meta
        logger.info('trained %s catboost_auc=%.4f', kind, kind_meta['roc_auc'])

    (model_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    logger.info('saved v2 weather bundle to %s', model_dir)


if __name__ == '__main__':
    main()
