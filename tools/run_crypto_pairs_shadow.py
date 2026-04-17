#!/usr/bin/env python3
"""Run the crypto pairs architecture in paper/shadow mode from the latest discovery output."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.crypto_pairs.config import ExecutionConfig, RiskConfig, ShadowRunnerConfig, SignalConfig
from engine.crypto_pairs.discovery import build_runtime_configs, load_discovery_report, resolve_discovery_report_path
from engine.crypto_pairs.execution_engine import ExecutionEngine
from engine.crypto_pairs.logger import CryptoPairsLogger
from engine.crypto_pairs.position_manager import PositionManager
from engine.crypto_pairs.price_streamer import PriceStreamer
from engine.crypto_pairs.ratio_engine import RatioEngine
from engine.crypto_pairs.signal_engine_v1 import Signal, SignalEngineV1


UTC = timezone.utc
DEFAULT_SESSION_ROOT = Path("output/crypto_pairs/sessions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run crypto pairs paper/shadow runtime.")
    parser.add_argument("--discovery-report", default=None, help="Path to a frozen pair_discovery_results.json")
    parser.add_argument("--top-pairs", type=int, default=5)
    parser.add_argument("--pair-key", action="append", default=[], help="Explicit pair key like AAVE/DOGE; can be passed multiple times")
    parser.add_argument("--runtime-seconds", type=int, default=None)
    parser.add_argument("--total-capital", type=float, default=10_000.0)
    parser.add_argument("--session-root", default=str(DEFAULT_SESSION_ROOT))
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.0)
    parser.add_argument("--stop-z", type=float, default=4.0)
    parser.add_argument("--max-hold-seconds", type=int, default=21_600)
    parser.add_argument("--cooldown-seconds", type=int, default=60)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--capital-per-pair-pct", type=float, default=0.20)
    parser.add_argument("--max-total-exposure-pct", type=float, default=0.80)
    parser.add_argument("--max-daily-loss-pct", type=float, default=0.03)
    parser.add_argument("--max-correlation-overlap", type=int, default=2)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    discovery_path = resolve_discovery_report_path(args.discovery_report)
    discovery_report = load_discovery_report(discovery_path)
    symbols, pair_configs, active_pairs = build_runtime_configs(
        discovery_report,
        top_pairs=args.top_pairs,
        pair_keys=args.pair_key,
    )

    session_id = f"crypto_pairs_shadow_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}_v1"
    session_root = resolve_path(args.session_root) / session_id
    logger = CryptoPairsLogger(session_root)
    logger.flush_summary(
        discovery_report=str(discovery_path),
        active_pairs=[row["pair"] for row in active_pairs],
        requested_pair_keys=args.pair_key,
    )

    streamer = PriceStreamer(symbols)
    ratio_engine = RatioEngine(pair_configs)
    signal_engine = SignalEngineV1(
        SignalConfig(
            entry_z=args.entry_z,
            exit_z=args.exit_z,
            stop_z=args.stop_z,
            max_hold_seconds=args.max_hold_seconds,
            cooldown_seconds=args.cooldown_seconds,
        )
    )
    executor = ExecutionEngine(streamer.latest_prices.get, ExecutionConfig(paper_trade=True))
    position_manager = PositionManager(
        total_capital=args.total_capital,
        risk_config=RiskConfig(
            max_positions=args.max_positions,
            max_capital_per_pair_pct=args.capital_per_pair_pct,
            max_total_exposure_pct=args.max_total_exposure_pct,
            max_daily_loss_pct=args.max_daily_loss_pct,
            max_correlation_overlap=args.max_correlation_overlap,
        ),
    )

    def on_bar(bar) -> None:
        updated_pairs = ratio_engine.on_price_bar(bar)
        for pair_key in updated_pairs:
            state = ratio_engine.get_state(pair_key)
            if state is None:
                continue
            logger.log_ratio_tick(
                {
                    "pair": pair_key,
                    "timestamp_ms": state.last_ratio_timestamp_ms,
                    "ratio": state.current_ratio,
                    "zscore": state.current_zscore,
                    "rolling_mean": state.rolling_mean,
                    "rolling_std": state.rolling_std,
                    "ready": state.ready,
                    "features": state.features if state.ready else {},
                }
            )
            if not state.ready:
                continue
            decision = signal_engine.evaluate(
                pair_key=pair_key,
                zscore=state.current_zscore,
                features=state.features,
                timestamp_ms=state.last_ratio_timestamp_ms,
            )
            logger.log_signal(
                {
                    "pair": pair_key,
                    "timestamp_ms": state.last_ratio_timestamp_ms,
                    "signal": decision.signal.value,
                    "reason": decision.reason,
                    "zscore": state.current_zscore,
                }
            )
            if decision.signal in (Signal.LONG_A_SHORT_B, Signal.SHORT_A_LONG_B):
                can_open, reason = position_manager.can_open(pair_key)
                if not can_open:
                    logger.log_pair_health(
                        {
                            "pair": pair_key,
                            "timestamp_ms": state.last_ratio_timestamp_ms,
                            "status": "skip_entry",
                            "reason": reason,
                            "zscore": state.current_zscore,
                        }
                    )
                    continue
                capital = position_manager.get_position_size_per_leg()
                entry_trade = executor.execute_entry(
                    pair_key=pair_key,
                    direction=decision.signal,
                    token_a=state.token_a,
                    token_b=state.token_b,
                    capital_per_leg_usdt=capital,
                )
                position_manager.open_position(
                    pair_key=pair_key,
                    direction=decision.signal.value,
                    entry_trade=entry_trade,
                    zscore=state.current_zscore,
                    max_hold_seconds=signal_engine.config.max_hold_seconds,
                )
                signal_engine.confirm_entry(
                    pair_key=pair_key,
                    signal=decision.signal,
                    zscore=state.current_zscore,
                    timestamp_ms=state.last_ratio_timestamp_ms,
                )
                logger.log_trade_event({"event": "entry", **entry_trade, "entry_zscore": state.current_zscore})
            elif decision.signal == Signal.EXIT and pair_key in position_manager.positions:
                exit_trade = executor.execute_exit(position_manager.positions[pair_key].entry_trade)
                position_manager.close_position(pair_key=pair_key, exit_trade=exit_trade)
                signal_engine.confirm_exit(pair_key=pair_key, timestamp_ms=state.last_ratio_timestamp_ms)
                logger.log_trade_event({"event": "exit", **exit_trade, "reason": decision.reason})

        current_time_ms = bar.timestamp_ms
        for pair_key in position_manager.check_stale_positions(current_time_ms):
            if pair_key not in position_manager.positions:
                continue
            exit_trade = executor.execute_exit(position_manager.positions[pair_key].entry_trade)
            position_manager.close_position(pair_key=pair_key, exit_trade=exit_trade)
            signal_engine.confirm_exit(pair_key=pair_key, timestamp_ms=current_time_ms)
            logger.log_trade_event({"event": "timeout_exit", **exit_trade, "reason": "stale_position"})

    streamer.on_bar(on_bar)
    await streamer.run_forever(runtime_seconds=args.runtime_seconds)

    for pair_key in list(position_manager.positions):
        exit_trade = executor.execute_exit(position_manager.positions[pair_key].entry_trade)
        position_manager.close_position(pair_key=pair_key, exit_trade=exit_trade)
        logger.log_trade_event({"event": "shutdown_exit", **exit_trade, "reason": "runner_shutdown"})

    logger.flush_summary(
        session_id=session_id,
        symbols=symbols,
        open_positions=len(position_manager.positions),
        closed_trades=len(position_manager.closed_trades),
        daily_pnl_usd=position_manager.daily_pnl_usd,
        streamer_stats=streamer.stats,
        ratio_engine_stats=ratio_engine.stats,
    )
    print(json.dumps({"session_root": str(session_root), "summary": logger.summary}, indent=2))


def resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


if __name__ == "__main__":
    asyncio.run(main())
