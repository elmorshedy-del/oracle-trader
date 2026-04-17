#!/usr/bin/env python3
"""
Sweep held-out execution policies for the BTC futures LOB research path.

Purpose:
- reuse the captured LOB dataset + model training path
- compare horizons on post-cost PnL, not only AUC
- search passive execution policies on validation, then report held-out test results

Default behavior keeps the run practical:
- xgboost only
- same captured depth/trade inputs as the LOB compare runner
- multiple event/time horizons
- validation-selected execution policy
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch

from futures_lob_event_compare import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CAPTURE_ROOT,
    DEFAULT_CONV_CHANNELS,
    DEFAULT_DROPOUT,
    DEFAULT_ENTRY_TIMEOUT_SECONDS,
    DEFAULT_EVENT_HORIZON,
    DEFAULT_EXIT_TIMEOUT_SECONDS,
    DEFAULT_FEE_BPS_PER_SIDE,
    DEFAULT_LEVELS,
    DEFAULT_LONGER_EVENT_HORIZONS,
    DEFAULT_LONGER_TIME_HORIZONS_SECONDS,
    DEFAULT_LR,
    DEFAULT_LSTM_HIDDEN,
    DEFAULT_MAX_QUOTE_AGE_MS,
    DEFAULT_MIN_VALIDATION_TRADES,
    DEFAULT_ORDER_NOTIONAL_USD,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PATIENCE,
    DEFAULT_QUEUE_AHEAD_MULTIPLIER,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_SYMBOL,
    DEFAULT_TIME_HORIZON_SECONDS,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_XGB_ESTIMATORS,
    DEFAULT_XGB_MAX_DEPTH,
    DEFAULT_XGB_N_JOBS,
    THRESHOLD_GRID,
    DeepLOBRegressor,
    ReplayConfig,
    build_event_dataset,
    build_samples,
    build_scaler,
    build_target_specs,
    build_xgboost_regressor,
    capture_date_label,
    choose_trade_quantile,
    empty_trade_metrics,
    evaluate_regression,
    evaluate_trade_policy,
    horizon_tag,
    parse_float_list,
    parse_int_list,
    predict_deep,
    resolve_capture_dirs,
    resolve_device,
    resolve_feature_columns,
    set_random_seed,
    split_array,
    split_dataset,
    split_samples,
    trade_key,
    train_deep_regressor,
)


UTC = timezone.utc
RANDOM_SEED = 42
DEFAULT_MAKER_FEES_BPS = (0.5, 1.0)
DEFAULT_TAKER_FEES_BPS = (1.0, 2.0)
DEFAULT_ENTRY_TIMEOUTS_SECONDS = (1.0, 2.0, 5.0)
DEFAULT_EXIT_TIMEOUTS_SECONDS = (1.0, 2.0, 5.0)
DEFAULT_QUEUE_MULTIPLIERS = (0.0, 0.5, 1.0)
DEFAULT_MODELS = ('xgboost',)
MODEL_CHOICES = ('xgboost', 'deeplob_style')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Sweep BTC LOB execution policies on held-out replay.')
    parser.add_argument('--capture-root', default=str(DEFAULT_CAPTURE_ROOT), help='Root containing symbol/date capture folders')
    parser.add_argument('--output-root', default=str(Path(DEFAULT_OUTPUT_ROOT).parent / 'futures_lob_policy_sweep'), help='Output root for reports and models')
    parser.add_argument('--symbol', default=DEFAULT_SYMBOL, help='Symbol folder to read')
    parser.add_argument('--date', help='Single capture date folder YYYY-MM-DD')
    parser.add_argument('--dates', help='Comma-separated capture dates YYYY-MM-DD,YYYY-MM-DD')
    parser.add_argument('--all-dates', action='store_true', help='Consume all available capture date folders for the symbol')
    parser.add_argument('--levels', type=int, default=DEFAULT_LEVELS, help='Number of depth levels per side to keep')
    parser.add_argument('--sequence-length', type=int, default=DEFAULT_SEQUENCE_LENGTH, help='History length in depth events')
    parser.add_argument('--event-horizon', type=int, default=DEFAULT_EVENT_HORIZON, help='Future event count for the event-based target')
    parser.add_argument('--event-horizons', default=','.join(str(value) for value in DEFAULT_LONGER_EVENT_HORIZONS), help='Comma-separated future event counts to score, e.g. 20,50,100')
    parser.add_argument('--time-horizon-seconds', type=float, default=DEFAULT_TIME_HORIZON_SECONDS, help='Future wall-clock horizon in seconds')
    parser.add_argument('--time-horizons-seconds', default=','.join(str(int(value)) for value in DEFAULT_LONGER_TIME_HORIZONS_SECONDS), help='Comma-separated future wall-clock horizons in seconds, e.g. 10,30,60')
    parser.add_argument('--models', default=','.join(DEFAULT_MODELS), help='Comma-separated models: xgboost,deeplob_style')
    parser.add_argument('--maker-fees-bps', default=','.join(str(value) for value in DEFAULT_MAKER_FEES_BPS), help='Comma-separated maker fees in bps')
    parser.add_argument('--taker-fees-bps', default=','.join(str(value) for value in DEFAULT_TAKER_FEES_BPS), help='Comma-separated taker fees in bps')
    parser.add_argument('--entry-timeouts-seconds', default=','.join(str(value) for value in DEFAULT_ENTRY_TIMEOUTS_SECONDS), help='Comma-separated passive entry timeouts in seconds')
    parser.add_argument('--exit-timeouts-seconds', default=','.join(str(value) for value in DEFAULT_EXIT_TIMEOUTS_SECONDS), help='Comma-separated passive exit timeouts in seconds')
    parser.add_argument('--queue-multipliers', default=','.join(str(value) for value in DEFAULT_QUEUE_MULTIPLIERS), help='Comma-separated queue-ahead multipliers')
    parser.add_argument('--order-notional-usd', type=float, default=DEFAULT_ORDER_NOTIONAL_USD, help='Per-signal order notional used in replay')
    parser.add_argument('--max-quote-age-ms', type=int, default=DEFAULT_MAX_QUOTE_AGE_MS, help='Maximum age of the nearest quote snapshot used for replay')
    parser.add_argument('--min-validation-trades', type=int, default=DEFAULT_MIN_VALIDATION_TRADES, help='Minimum validation fills before a policy can win')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='Deep model batch size')
    parser.add_argument('--epochs', type=int, default=12, help='Maximum deep model epochs')
    parser.add_argument('--patience', type=int, default=DEFAULT_PATIENCE, help='Deep model early-stop patience')
    parser.add_argument('--conv-channels', type=int, default=DEFAULT_CONV_CHANNELS, help='DeepLOB-style conv channel count')
    parser.add_argument('--lstm-hidden', type=int, default=DEFAULT_LSTM_HIDDEN, help='DeepLOB-style LSTM hidden size')
    parser.add_argument('--dropout', type=float, default=DEFAULT_DROPOUT, help='DeepLOB-style dropout')
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_LR, help='AdamW learning rate')
    parser.add_argument('--weight-decay', type=float, default=DEFAULT_WEIGHT_DECAY, help='AdamW weight decay')
    parser.add_argument('--xgb-estimators', type=int, default=DEFAULT_XGB_ESTIMATORS, help='XGBoost tree count')
    parser.add_argument('--xgb-max-depth', type=int, default=DEFAULT_XGB_MAX_DEPTH, help='XGBoost max depth')
    parser.add_argument('--xgb-n-jobs', type=int, default=DEFAULT_XGB_N_JOBS, help='XGBoost worker threads')
    parser.add_argument('--device', default='auto', choices=('auto', 'cpu', 'mps'), help='Torch device selection')
    return parser.parse_args()


def parse_model_list(value: str) -> list[str]:
    models = [part.strip() for part in value.split(',') if part.strip()]
    unknown = [model for model in models if model not in MODEL_CHOICES]
    if unknown:
        raise SystemExit(f'Unsupported model(s): {unknown}')
    if not models:
        raise SystemExit('Expected at least one model')
    return models


def build_policy_grid(args: argparse.Namespace) -> list[ReplayConfig]:
    maker_fees = parse_float_list(args.maker_fees_bps)
    taker_fees = parse_float_list(args.taker_fees_bps)
    entry_timeouts = parse_float_list(args.entry_timeouts_seconds)
    exit_timeouts = parse_float_list(args.exit_timeouts_seconds)
    queue_multipliers = parse_float_list(args.queue_multipliers)
    configs: list[ReplayConfig] = []
    for maker_fee, taker_fee, entry_timeout, exit_timeout, queue_mult in product(
        maker_fees,
        taker_fees,
        entry_timeouts,
        exit_timeouts,
        queue_multipliers,
    ):
        configs.append(
            ReplayConfig(
                maker_fee_bps_per_side=maker_fee,
                taker_fee_bps_per_side=taker_fee,
                order_notional_usd=args.order_notional_usd,
                entry_timeout_seconds=entry_timeout,
                exit_timeout_seconds=exit_timeout,
                queue_ahead_multiplier=queue_mult,
                max_quote_age_ms=args.max_quote_age_ms,
            )
        )
    return configs


def policy_key(result: dict[str, Any]) -> tuple[float, float, float, float]:
    trade_metrics = result['validation_trade']
    return (
        float(trade_metrics['net_pnl_bps']),
        float(trade_metrics['profit_factor']),
        float(trade_metrics['win_rate']),
        -float(result['policy']['maker_fee_bps_per_side']),
    )


def render_policy_markdown(report: dict[str, Any]) -> str:
    lines = [
        '# Futures LOB Policy Sweep',
        '',
        f"- Rows: `{report['rows']:,}`",
        f"- Features: `{report['feature_count']}`",
        f"- Sequence length: `{report['sequence_length']}` events",
        f"- Event horizons: `{report['event_horizons']}`",
        f"- Time horizons: `{report['time_horizons_seconds']}`",
        f"- Models: `{report['models']}`",
        f"- Policies tested per model/target: `{report['policy_grid_size']}`",
        '',
    ]
    for target_name, target_info in report['targets'].items():
        lines.append(f"## `{target_name}`")
        lines.append('')
        for model_name, model_info in target_info['models'].items():
            test_metrics = model_info['test_metrics']
            test_trade = model_info['selected_test_trade']
            policy = model_info['selected_policy']
            lines.append(
                f"- `{model_name}`: test AUC `{test_metrics['directional_auc']:.4f}`, test net PnL `{test_trade['net_pnl_bps']:.4f}` bps across `{test_trade['trades']}` fills, chosen validation quantile `{test_trade['chosen_quantile']}`, maker/taker `{policy['maker_fee_bps_per_side']}/{policy['taker_fee_bps_per_side']}` bps, entry/exit timeouts `{policy['entry_timeout_seconds']}/{policy['exit_timeout_seconds']}` s, queue mult `{policy['queue_ahead_multiplier']}`"
            )
        lines.append('')
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    set_random_seed(RANDOM_SEED)
    event_horizons = sorted(set(parse_int_list(args.event_horizons) + [args.event_horizon]))
    time_horizons = sorted(set(parse_float_list(args.time_horizons_seconds) + [args.time_horizon_seconds]))
    target_specs = build_target_specs(event_horizons, time_horizons)
    models = parse_model_list(args.models)
    policy_grid = build_policy_grid(args)

    symbol = args.symbol.upper()
    capture_dirs = resolve_capture_dirs(
        capture_root=Path(args.capture_root).resolve(),
        symbol=symbol,
        date=args.date,
        dates_csv=args.dates,
        all_dates=args.all_dates,
    )
    date_label = capture_date_label(capture_dirs)
    run_name = (
        f"lob_policy_{args.symbol.lower()}_{date_label}_"
        f"lvl{args.levels:02d}_seq{args.sequence_length:03d}_"
        f"{horizon_tag(event_horizons, time_horizons)}_v1"
    )
    run_root = Path(args.output_root).resolve() / run_name
    dataset_root = run_root / 'dataset'
    model_root = run_root / 'models'
    report_root = run_root / 'reports'
    for path in (dataset_root, model_root, report_root):
        path.mkdir(parents=True, exist_ok=True)

    print(f'[policy] building dataset from {len(capture_dirs)} capture date(s)', flush=True)
    dataset, trade_frame = build_event_dataset(
        capture_dirs=capture_dirs,
        levels=args.levels,
        event_horizons=event_horizons,
        time_horizons_seconds=time_horizons,
    )
    dataset_path = dataset_root / 'lob_event_dataset.csv.gz'
    dataset.to_csv(dataset_path, index=False, compression='gzip')

    feature_columns = resolve_feature_columns(dataset)
    split_idx = split_dataset(dataset)
    scaler = build_scaler(dataset.iloc[: split_idx['train_end']], feature_columns)
    device = resolve_device(args.device)

    report: dict[str, Any] = {
        'bundle_version': run_name,
        'created_at': datetime.now(UTC).isoformat(),
        'capture_dirs': [str(path) for path in capture_dirs],
        'capture_dates': [path.name for path in capture_dirs],
        'dataset_path': str(dataset_path),
        'rows': int(len(dataset)),
        'feature_count': int(len(feature_columns)),
        'sequence_length': args.sequence_length,
        'event_horizons': event_horizons,
        'time_horizons_seconds': time_horizons,
        'models': models,
        'policy_grid_size': int(len(policy_grid) * len(THRESHOLD_GRID)),
        'targets': {},
    }

    for target_spec in target_specs:
        print(f'[policy] preparing {target_spec.report_key}', flush=True)
        sequences, flat_features, labels, sample_idx = build_samples(
            dataset=dataset,
            feature_columns=feature_columns,
            target_spec=target_spec,
            sequence_length=args.sequence_length,
            scaler=scaler,
        )
        seq_split = split_samples(len(sequences))
        X_train, X_valid, X_test = split_array(flat_features, seq_split)
        S_train, S_valid, S_test = split_array(sequences, seq_split)
        y_train, y_valid, y_test = split_array(labels, seq_split)
        _sample_idx_train, sample_idx_valid, sample_idx_test = split_array(sample_idx, seq_split)

        target_report: dict[str, Any] = {'models': {}}
        for model_name in models:
            print(f'[policy] fitting {model_name} for {target_spec.report_key}', flush=True)
            deep_summary = None
            if model_name == 'xgboost':
                model = build_xgboost_regressor(
                    n_estimators=args.xgb_estimators,
                    max_depth=args.xgb_max_depth,
                    n_jobs=args.xgb_n_jobs,
                )
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
                preds_train = model.predict(X_train)
                preds_valid = model.predict(X_valid)
                preds_test = model.predict(X_test)
            else:
                deep_model = DeepLOBRegressor(
                    input_features=sequences.shape[-1],
                    conv_channels=args.conv_channels,
                    lstm_hidden=args.lstm_hidden,
                    dropout=args.dropout,
                ).to(device)
                deep_summary = train_deep_regressor(
                    model=deep_model,
                    train_sequences=S_train,
                    train_labels=y_train,
                    valid_sequences=S_valid,
                    valid_labels=y_valid,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    patience=args.patience,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    device=device,
                )
                torch.save(deep_model.state_dict(), model_root / f'{target_spec.report_key}_{model_name}.pt')
                preds_train = predict_deep(deep_model, S_train, device=device, batch_size=args.batch_size)
                preds_valid = predict_deep(deep_model, S_valid, device=device, batch_size=args.batch_size)
                preds_test = predict_deep(deep_model, S_test, device=device, batch_size=args.batch_size)

            best_result: dict[str, Any] | None = None
            for replay_config in policy_grid:
                validation_trade = choose_trade_quantile(
                    predictions=preds_valid,
                    labels=y_valid,
                    sample_indices=sample_idx_valid,
                    dataset=dataset,
                    trade_frame=trade_frame,
                    replay_config=replay_config,
                    min_validation_trades=args.min_validation_trades,
                )
                if validation_trade.trades < args.min_validation_trades:
                    continue
                selected_test_trade = evaluate_trade_policy(
                    predictions=preds_test,
                    labels=y_test,
                    sample_indices=sample_idx_test,
                    dataset=dataset,
                    trade_frame=trade_frame,
                    replay_config=replay_config,
                    quantile=validation_trade.chosen_quantile,
                    threshold_abs_bps=validation_trade.threshold_abs_bps,
                )
                candidate = {
                    'policy': asdict(replay_config),
                    'validation_trade': asdict(validation_trade),
                    'selected_test_trade': asdict(selected_test_trade),
                }
                if best_result is None or policy_key(candidate) > policy_key(best_result):
                    best_result = candidate

            if best_result is None:
                best_result = {
                    'policy': asdict(policy_grid[0]),
                    'validation_trade': asdict(empty_trade_metrics(0, None, None)),
                    'selected_test_trade': asdict(empty_trade_metrics(0, None, None)),
                }

            model_report: dict[str, Any] = {
                'train_metrics': asdict(evaluate_regression(y_train, preds_train)),
                'valid_metrics': asdict(evaluate_regression(y_valid, preds_valid)),
                'test_metrics': asdict(evaluate_regression(y_test, preds_test)),
                'selected_policy': best_result['policy'],
                'validation_trade': best_result['validation_trade'],
                'selected_test_trade': best_result['selected_test_trade'],
            }
            if deep_summary is not None:
                model_report['training'] = asdict(deep_summary)
            target_report['models'][model_name] = model_report
        report['targets'][target_spec.report_key] = target_report

    report_path_json = report_root / 'policy_sweep_report.json'
    report_path_md = report_root / 'policy_sweep_report.md'
    report_path_json.write_text(json.dumps(report, indent=2), encoding='utf-8')
    report_path_md.write_text(render_policy_markdown(report), encoding='utf-8')
    print(f'LOB policy sweep report: {report_path_md}')


if __name__ == '__main__':
    main()
