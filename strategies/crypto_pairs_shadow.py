"""Oracle-integrated crypto-pairs shadow sleeve."""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data.models import Event, Market
from engine.crypto_pairs.audit import CryptoPairsAudit
from engine.crypto_pairs.config import ExecutionConfig, PriceStreamerConfig, RiskConfig, SignalConfig
from engine.crypto_pairs.discovery import build_runtime_configs, load_discovery_report, resolve_discovery_report_path
from engine.crypto_pairs.execution_engine import ExecutionEngine
from engine.crypto_pairs.position_manager import PositionManager
from engine.crypto_pairs.price_streamer import PriceBar, PriceStreamer
from engine.crypto_pairs.ratio_engine import PairState, RatioEngine
from engine.crypto_pairs.signal_engine_v1 import Signal, SignalEngineV1
from runtime_paths import LOG_DIR
from strategies.base import BaseStrategy
from config import CryptoPairsShadowProfileConfig

logger = logging.getLogger(__name__)

try:  # pragma: no cover - dependency gate for runtime
    from statsmodels.tsa.stattools import coint
except Exception:  # pragma: no cover - runtime dependency gate
    coint = None


UTC = timezone.utc
MAX_RECENT_SIGNALS = 40
MAX_RECENT_TRADES = 80
MAX_COINTEGRATION_SAMPLES = 10_000


@dataclass(slots=True)
class LivePairPosition:
    pair_key: str
    direction: str
    entry_timestamp: datetime
    entry_timestamp_ms: int
    entry_zscore: float
    entry_ratio: float
    entry_price_a: float
    entry_price_b: float
    position_size_per_leg_usdt: float
    entry_trade: dict[str, Any]


class CryptoPairsShadowStrategy(BaseStrategy):
    name = "crypto_pairs_shadow"
    description = "Frozen crypto pairs shadow sleeve"

    def __init__(self, config, profile: CryptoPairsShadowProfileConfig):
        super().__init__(config)
        self.cfg = config.crypto_pairs_shadow
        self.profile = profile
        self.name = profile.strategy_key
        self.description = f"Frozen {profile.pair_key} crypto pairs shadow sleeve"
        self.discovery_path = resolve_discovery_report_path(self.cfg.discovery_report)
        self.discovery_report = load_discovery_report(self.discovery_path)
        symbols, pair_configs, active_pairs = build_runtime_configs(
            self.discovery_report,
            top_pairs=self.cfg.top_pairs,
            pair_keys=[self.profile.pair_key],
        )
        if len(pair_configs) != 1:
            raise ValueError(f"Expected exactly one crypto pairs runtime config, got {len(pair_configs)}")

        self.pair_config = pair_configs[0]
        self.active_pair = active_pairs[0]
        self.streamer = PriceStreamer(
            symbols,
            PriceStreamerConfig(
                ws_urls=tuple(self.cfg.ws_urls),
                bar_interval_seconds=self.cfg.bar_interval_seconds,
                reconnect_delay_seconds=self.cfg.reconnect_delay_seconds,
            ),
        )
        self.ratio_engine = RatioEngine(pair_configs, max_leg_lag_ms=self.cfg.max_leg_lag_ms)
        self.signal_engine = SignalEngineV1(
            SignalConfig(
                entry_z=self.cfg.entry_z,
                exit_z=self.cfg.exit_z,
                stop_z=self.cfg.stop_z,
                max_hold_seconds=self.cfg.max_hold_seconds,
                cooldown_seconds=self.cfg.cooldown_seconds,
            )
        )
        self.executor = ExecutionEngine(
            self.streamer.latest_prices.get,
            ExecutionConfig(
                fee_bps=self.cfg.fee_bps,
                slippage_bps=self.cfg.slippage_bps,
                quantity_precision=self.cfg.quantity_precision,
                paper_trade=True,
            ),
        )
        self.position_manager = PositionManager(
            total_capital=self.cfg.budget_usd,
            risk_config=RiskConfig(
                max_positions=1,
                max_capital_per_pair_pct=self.cfg.capital_per_pair_pct,
                max_total_exposure_pct=self.cfg.max_total_exposure_pct,
                max_daily_loss_pct=self.cfg.max_daily_loss_pct,
                max_correlation_overlap=1,
            ),
        )

        self.audit_root = self._resolve_audit_root()
        self.audit = CryptoPairsAudit(self.audit_root)
        self.audit.write_metadata(
            {
                "started_at": datetime.now(UTC).isoformat(),
                "strategy": self.name,
                "label": self.profile.label,
                "view_key": self.profile.view_key,
                "source": self.profile.source,
                "pair_key": self.pair_config.pair_key,
                "token_a": self.pair_config.token_a,
                "token_b": self.pair_config.token_b,
                "discovery_report": str(self.discovery_path),
                "entry_z": self.cfg.entry_z,
                "exit_z": self.cfg.exit_z,
                "stop_z": self.cfg.stop_z,
                "max_hold_seconds": self.cfg.max_hold_seconds,
                "cooldown_seconds": self.cfg.cooldown_seconds,
                "budget_usd": self.cfg.budget_usd,
                "capital_per_pair_pct": self.cfg.capital_per_pair_pct,
                "fee_bps": self.cfg.fee_bps,
                "slippage_bps": self.cfg.slippage_bps,
                "max_leg_lag_ms": self.cfg.max_leg_lag_ms,
            }
        )

        self._stream_thread: threading.Thread | None = None
        self._stream_loop: asyncio.AbstractEventLoop | None = None
        self._stop = False
        self._last_hourly_check_at: datetime | None = None
        self._last_daily_day_key: str | None = None
        self._last_trade_at: datetime | None = None
        self._last_ratio_tick_at: datetime | None = None
        self._recent_signals: list[dict[str, Any]] = []
        self._recent_trades: list[dict[str, Any]] = []
        self._open_position: LivePairPosition | None = None
        self._daily_records: dict[str, dict[str, Any]] = {}
        self._daily_samples: dict[str, dict[str, list[float]]] = {}
        self._stats.update(
            {
                "pair_key": self.pair_config.pair_key,
                "view_key": self.profile.view_key,
                "source": self.profile.source,
                "token_a": self.pair_config.token_a,
                "token_b": self.pair_config.token_b,
                "budget_usd": self.cfg.budget_usd,
                "entry_z": self.cfg.entry_z,
                "exit_z": self.cfg.exit_z,
                "stop_z": self.cfg.stop_z,
                "max_hold_seconds": self.cfg.max_hold_seconds,
                "cooldown_seconds": self.cfg.cooldown_seconds,
                "ratio_updates": 0,
                "entry_signals": 0,
                "blocked_entry_signals": 0,
                "entries": 0,
                "closed_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "realized_net_bps": 0.0,
                "realized_pnl_usd": 0.0,
                "unrealized_net_bps": 0.0,
                "unrealized_pnl_usd": 0.0,
                "total_net_bps": 0.0,
                "total_pnl_usd": 0.0,
                "current_cointegration_pvalue": None,
                "last_ratio_tick_at": None,
                "last_trade_at": None,
                "last_entry_reason": None,
                "last_exit_reason": None,
                "last_block_reason": None,
                "last_hourly_check_at": None,
                "last_daily_summary_at": None,
                "open_position": False,
                "open_direction": None,
                "open_hold_seconds": 0.0,
                "audit_root": str(self.audit_root),
                "ratio_ticks_csv": str(self.audit.paths["ratio_ticks_csv"]),
                "trade_ledger_csv": str(self.audit.paths["trade_ledger_csv"]),
                "daily_summary_path": str(self.audit.paths["daily_summary_jsonl"]),
                "hourly_checks_path": str(self.audit.paths["hourly_checks_jsonl"]),
                "streamer_stats": self.streamer.stats,
                "ratio_engine_stats": self.ratio_engine.stats,
            }
        )
        self.enabled = bool(self.cfg.enabled)
        self.streamer.on_bar(self._on_bar)

    async def ensure_started(self) -> None:
        if not self.enabled or self._stream_thread is not None:
            return
        thread = threading.Thread(
            target=self._run_streamer_thread,
            name="crypto-pairs-shadow-stream",
            daemon=True,
        )
        self._stream_thread = thread
        thread.start()

    async def close(self) -> None:
        self._stop = True
        self.streamer.stop()
        if self._stream_loop is not None:
            try:
                self._stream_loop.call_soon_threadsafe(lambda: None)
            except RuntimeError:
                pass
        if self._stream_thread is not None:
            await asyncio.to_thread(self._stream_thread.join, 5.0)
            self._stream_thread = None
        self._finalize_all_daily_records(finalize_open_position=True)
        self._write_runtime_state()

    async def scan(self, markets: list[Market], events: list[Event]) -> list:
        del markets, events
        self._stats["scans_completed"] += 1
        self._stats["streamer_stats"] = self.streamer.stats
        self._stats["ratio_engine_stats"] = self.ratio_engine.stats
        await self.ensure_started()
        self._write_runtime_state()
        return []

    @property
    def stats(self) -> dict:
        return {
            "name": self.name,
            "enabled": self.enabled,
            **self._stats,
        }

    def serialize_view(self) -> dict[str, Any]:
        unrealized_usd, unrealized_bps = self._current_unrealized()
        cash = self.cfg.budget_usd + float(self._stats["realized_pnl_usd"])
        positions_value = 0.0
        positions = []
        if self._open_position is not None:
            deployed = self._open_position.position_size_per_leg_usdt * 2.0
            cash -= deployed
            positions_value = deployed + unrealized_usd
            pair_state = self.ratio_engine.get_state(self.pair_config.pair_key)
            current_ratio = pair_state.current_ratio if pair_state is not None else self._open_position.entry_ratio
            positions.append(
                {
                    "market": self.pair_config.pair_key,
                    "side": self._human_direction(self._open_position.direction),
                    "shares": round(self._open_position.position_size_per_leg_usdt, 2),
                    "entry": round(self._open_position.entry_ratio, 6),
                    "current": round(float(current_ratio), 6),
                    "pnl": round(unrealized_usd, 2),
                    "source": self.profile.source,
                }
            )
        total_value = cash + positions_value
        total_pnl = total_value - self.cfg.budget_usd
        total_pnl_pct = (total_pnl / self.cfg.budget_usd * 100.0) if self.cfg.budget_usd else 0.0
        return {
            "key": self.profile.view_key,
            "label": self.profile.label,
            "source": self.profile.source,
            "portfolio": {
                "starting_capital": round(self.cfg.budget_usd, 2),
                "total_value": round(total_value, 2),
                "cash": round(cash, 2),
                "positions_value": round(positions_value, 2),
                "total_pnl": round(total_pnl, 2),
                "total_pnl_pct": round(total_pnl_pct, 2),
                "total_trades": int(self._stats["closed_trades"]),
                "win_rate": round(float(self._stats["win_rate"]) * 100.0, 1),
                "max_drawdown": round(float(self._current_day_record().get("max_drawdown_pct") or 0.0) * 100.0, 2),
                "total_fees": round(float(self._total_fees_paid()), 2),
                "positions": positions,
            },
            "signals": list(self._recent_signals[:30]),
            "trades": list(self._recent_trades[:30]),
            "performance": {
                "total_pnl": round(total_pnl, 2),
                "total_pnl_pct": round(total_pnl_pct, 2),
                "win_rate": float(self._stats["win_rate"]),
                "total_trades": int(self._stats["closed_trades"]),
                "realized_pnl": round(float(self._stats["realized_pnl_usd"]), 2),
                "realized_net_bps": round(float(self._stats["realized_net_bps"]), 4),
                "unrealized_pnl": round(unrealized_usd, 2),
                "unrealized_net_bps": round(unrealized_bps, 4),
                "entry_signals": int(self._stats["entry_signals"]),
                "blocked_entry_signals": int(self._stats["blocked_entry_signals"]),
            },
        }

    def _run_streamer_thread(self) -> None:
        loop = asyncio.new_event_loop()
        self._stream_loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.streamer.run_forever())
        except Exception as exc:
            logger.exception("[CRYPTO_PAIRS_SHADOW] Stream thread crashed: %s", exc)
            self._stats["errors"] = int(self._stats["errors"]) + 1
            self._stats["last_block_reason"] = f"stream_thread_error:{type(exc).__name__}"
        finally:
            self._stream_loop = None
            self._stream_thread = None
            loop.close()

    def _resolve_audit_root(self) -> Path:
        base = Path(self.cfg.audit_root).resolve() if self.cfg.audit_root else (LOG_DIR / "comparison" / self.profile.session_label)
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _on_bar(self, bar: PriceBar) -> None:
        if self._stop:
            return
        updated_pairs = self.ratio_engine.on_price_bar(bar)
        self._stats["streamer_stats"] = self.streamer.stats
        self._stats["ratio_engine_stats"] = self.ratio_engine.stats
        now = datetime.fromtimestamp(bar.timestamp_ms / 1000.0, tz=UTC)
        day_key = now.strftime("%Y-%m-%d")
        self._ensure_day_record(day_key)

        for pair_key in updated_pairs:
            state = self.ratio_engine.get_state(pair_key)
            if state is None:
                continue
            self._process_ratio_state(now=now, state=state)

        self._maybe_finalize_previous_days(current_day_key=day_key)
        self._maybe_write_hourly_check(now=now)
        self._write_runtime_state()

    def _process_ratio_state(self, *, now: datetime, state: PairState) -> None:
        day_key = now.strftime("%Y-%m-%d")
        record = self._ensure_day_record(day_key)
        record["ratio_tick_count"] += 1
        record["last_ratio_tick_at"] = now.isoformat()
        self._daily_samples[day_key]["a"].append(float(state.last_price_a))
        self._daily_samples[day_key]["b"].append(float(state.last_price_b))
        self._daily_samples[day_key]["timestamps"].append(now.isoformat())
        self._last_ratio_tick_at = now
        self._stats["ratio_updates"] = int(self._stats["ratio_updates"]) + 1
        self._stats["last_ratio_tick_at"] = now.isoformat()
        self.audit.log_ratio_tick(
            {
                "timestamp": now.isoformat(),
                "pair": state.pair_key,
                "price_a": round(float(state.last_price_a), 6),
                "price_b": round(float(state.last_price_b), 6),
                "ratio": round(float(state.current_ratio), 8),
                "zscore": round(float(state.current_zscore), 6),
                "rolling_mean": round(float(state.rolling_mean), 8),
                "rolling_std": round(float(state.rolling_std), 8),
                "ready": bool(state.ready),
            }
        )

        self._update_mark_to_market(now=now, state=state)

        if not state.ready:
            return

        decision = self.signal_engine.evaluate(
            pair_key=state.pair_key,
            zscore=float(state.current_zscore),
            features=state.features,
            timestamp_ms=state.last_ratio_timestamp_ms,
        )
        if decision.signal in (Signal.LONG_A_SHORT_B, Signal.SHORT_A_LONG_B):
            self._handle_entry_signal(now=now, state=state, decision=decision)
        elif decision.signal == Signal.EXIT:
            self._handle_exit_signal(now=now, state=state, exit_reason=decision.reason)

        for stale_pair_key in self.position_manager.check_stale_positions(state.last_ratio_timestamp_ms):
            if stale_pair_key in self.position_manager.positions:
                stale_state = self.ratio_engine.get_state(stale_pair_key)
                if stale_state is not None:
                    self._handle_exit_signal(now=now, state=stale_state, exit_reason="max_hold_timeout")

    def _handle_entry_signal(self, *, now: datetime, state: PairState, decision) -> None:
        day_key = now.strftime("%Y-%m-%d")
        record = self._ensure_day_record(day_key)
        record["signal_count"] += 1
        self._stats["entry_signals"] = int(self._stats["entry_signals"]) + 1
        can_open, reason = self.position_manager.can_open(state.pair_key)
        signal_payload = {
            "id": f"{state.pair_key}-{now.strftime('%H%M%S')}",
            "time": now.isoformat(),
            "source": self.profile.source,
            "action": decision.signal.value,
            "market": state.pair_key,
            "confidence": round(min(abs(float(state.current_zscore)) / max(self.cfg.entry_z, 1e-6), 1.0), 2),
            "edge": round(abs(float(state.current_zscore)), 4),
            "size": round(self.position_manager.get_position_size_per_leg() * 2.0, 2),
            "whale": False,
            "reasoning": decision.reason,
        }
        self.audit.log_signal(
            {
                "timestamp": now.isoformat(),
                "pair": state.pair_key,
                "action": decision.signal.value,
                "reason": decision.reason,
                "zscore": round(float(state.current_zscore), 6),
                "ratio": round(float(state.current_ratio), 8),
                "blocked": not can_open,
                "block_reason": None if can_open else reason,
            }
        )
        self._recent_signals.insert(0, signal_payload)
        self._recent_signals = self._recent_signals[:MAX_RECENT_SIGNALS]
        self._stats["signals_generated"] = int(self._stats["signals_generated"]) + 1
        if not can_open:
            record["blocked_signal_count"] += 1
            self._stats["blocked_entry_signals"] = int(self._stats["blocked_entry_signals"]) + 1
            self._stats["last_block_reason"] = reason
            return

        capital_per_leg = self.position_manager.get_position_size_per_leg()
        entry_trade = self.executor.execute_entry(
            pair_key=state.pair_key,
            direction=decision.signal,
            token_a=state.token_a,
            token_b=state.token_b,
            capital_per_leg_usdt=capital_per_leg,
        )
        self.position_manager.open_position(
            pair_key=state.pair_key,
            direction=decision.signal.value,
            entry_trade=entry_trade,
            zscore=float(state.current_zscore),
            max_hold_seconds=self.cfg.max_hold_seconds,
            entry_time_ms=state.last_ratio_timestamp_ms,
        )
        self.signal_engine.confirm_entry(
            pair_key=state.pair_key,
            signal=decision.signal,
            zscore=float(state.current_zscore),
            timestamp_ms=state.last_ratio_timestamp_ms,
        )
        self._open_position = LivePairPosition(
            pair_key=state.pair_key,
            direction=decision.signal.value,
            entry_timestamp=now,
            entry_timestamp_ms=state.last_ratio_timestamp_ms,
            entry_zscore=float(state.current_zscore),
            entry_ratio=float(state.current_ratio),
            entry_price_a=float(entry_trade["fill_a"]),
            entry_price_b=float(entry_trade["fill_b"]),
            position_size_per_leg_usdt=capital_per_leg,
            entry_trade=entry_trade,
        )
        record["entries_taken"] += 1
        self._stats["entries"] = int(self._stats["entries"]) + 1
        self._stats["open_position"] = True
        self._stats["open_direction"] = self._human_direction(decision.signal.value)
        self._stats["last_entry_reason"] = decision.reason
        self.audit.log_trade_event(
            {
                "timestamp": now.isoformat(),
                "event": "entry",
                "pair": state.pair_key,
                "direction": self._human_direction(decision.signal.value),
                "entry_zscore": round(float(state.current_zscore), 6),
                "entry_ratio": round(float(state.current_ratio), 8),
                "entry_price_a": round(float(entry_trade["fill_a"]), 6),
                "entry_price_b": round(float(entry_trade["fill_b"]), 6),
                "position_size_per_leg_usdt": round(capital_per_leg, 2),
            }
        )

    def _handle_exit_signal(self, *, now: datetime, state: PairState, exit_reason: str) -> None:
        if self._open_position is None or state.pair_key not in self.position_manager.positions:
            return
        position = self.position_manager.positions[state.pair_key]
        exit_trade = self.executor.execute_exit(position.entry_trade)
        self.position_manager.close_position(pair_key=state.pair_key, exit_trade=exit_trade)
        self.signal_engine.confirm_exit(pair_key=state.pair_key, timestamp_ms=state.last_ratio_timestamp_ms)
        ledger_row = self._build_trade_ledger_row(now=now, state=state, exit_trade=exit_trade, exit_reason=exit_reason)
        self.audit.log_trade_event(
            {
                "timestamp": now.isoformat(),
                "event": "exit",
                "pair": state.pair_key,
                "exit_reason": ledger_row["exit_reason"],
                "net_pnl_bps": ledger_row["net_pnl_bps"],
                "net_pnl_usdt": ledger_row["net_pnl_usdt"],
                "hold_seconds": ledger_row["hold_seconds"],
            }
        )
        self.audit.log_trade_ledger(ledger_row)

        trade_view = {
            "id": f"{state.pair_key}-trade-{now.strftime('%H%M%S')}",
            "time": ledger_row["exit_timestamp"],
            "source": self.profile.source,
            "market": state.pair_key,
            "side": ledger_row["direction"],
            "price": ledger_row["entry_ratio"],
            "usd": round(self._open_position.position_size_per_leg_usdt * 2.0, 2),
            "pnl": ledger_row["net_pnl_usdt"],
        }
        self._recent_trades.insert(0, trade_view)
        self._recent_trades = self._recent_trades[:MAX_RECENT_TRADES]
        self._last_trade_at = now
        self._stats["last_trade_at"] = now.isoformat()
        self._stats["last_exit_reason"] = ledger_row["exit_reason"]
        self._stats["closed_trades"] = int(self._stats["closed_trades"]) + 1
        self._stats["realized_net_bps"] = round(float(self._stats["realized_net_bps"]) + float(ledger_row["net_pnl_bps"]), 4)
        self._stats["realized_pnl_usd"] = round(float(self._stats["realized_pnl_usd"]) + float(ledger_row["net_pnl_usdt"]), 6)
        if float(ledger_row["net_pnl_usdt"]) > 0:
            self._stats["wins"] = int(self._stats["wins"]) + 1
        else:
            self._stats["losses"] = int(self._stats["losses"]) + 1
        total_closed = int(self._stats["closed_trades"])
        self._stats["win_rate"] = (int(self._stats["wins"]) / total_closed) if total_closed else 0.0
        self._stats["open_position"] = False
        self._stats["open_direction"] = None
        self._open_position = None
        self._update_mark_to_market(now=now, state=state)

        day_key = now.strftime("%Y-%m-%d")
        record = self._ensure_day_record(day_key)
        record["trade_count"] += 1
        if float(ledger_row["net_pnl_usdt"]) > 0:
            record["win_count"] += 1
        else:
            record["loss_count"] += 1
        record["total_net_bps"] += float(ledger_row["net_pnl_bps"])
        record["total_net_usdt"] += float(ledger_row["net_pnl_usdt"])
        record["exit_reason_counts"][ledger_row["exit_reason"]] = record["exit_reason_counts"].get(ledger_row["exit_reason"], 0) + 1
        record["last_trade_at"] = now.isoformat()
        self._write_daily_summary(day_key, finalize=False)

    def _build_trade_ledger_row(self, *, now: datetime, state: PairState, exit_trade: dict[str, Any], exit_reason: str) -> dict[str, Any]:
        if self._open_position is None:
            raise RuntimeError("Cannot build trade ledger row without an open position")
        human_reason = {
            "take_profit_mean_reversion": "mean_reversion",
            "stop_loss": "stop_loss",
            "max_hold_timeout": "timeout",
        }.get(exit_reason, exit_reason)
        return {
            "pair": state.pair_key,
            "entry_timestamp": self._open_position.entry_timestamp.isoformat(),
            "entry_zscore": round(float(self._open_position.entry_zscore), 6),
            "entry_price_a": round(float(self._open_position.entry_price_a), 6),
            "entry_price_b": round(float(self._open_position.entry_price_b), 6),
            "entry_ratio": round(float(self._open_position.entry_ratio), 8),
            "direction": self._human_direction(self._open_position.direction),
            "position_size_per_leg_usdt": round(float(self._open_position.position_size_per_leg_usdt), 2),
            "exit_timestamp": now.isoformat(),
            "exit_zscore": round(float(state.current_zscore), 6),
            "exit_price_a": round(float(exit_trade["fill_a"]), 6),
            "exit_price_b": round(float(exit_trade["fill_b"]), 6),
            "exit_ratio": round(float(state.current_ratio), 8),
            "exit_reason": human_reason,
            "gross_pnl_bps": round(float(exit_trade["gross_pnl_bps"]), 4),
            "fees_usd": round(float(exit_trade["fees_usd"]), 6),
            "slippage_usd": round(float(exit_trade["slippage_usd"]), 6),
            "slippage_bps": round(float(exit_trade["slippage_bps"]), 4),
            "net_pnl_bps": round(float(exit_trade["pnl_bps"]), 4),
            "net_pnl_usdt": round(float(exit_trade["pnl_usd"]), 6),
            "hold_seconds": round((now - self._open_position.entry_timestamp).total_seconds(), 2),
        }

    def _current_unrealized(self) -> tuple[float, float]:
        state = self.ratio_engine.get_state(self.pair_config.pair_key)
        if self._open_position is None or state is None:
            return 0.0, 0.0
        entry_trade = self._open_position.entry_trade
        qty_a = float(entry_trade["qty_a"])
        qty_b = float(entry_trade["qty_b"])
        capital_per_leg = float(entry_trade["capital_per_leg"])
        current_price_a = float(state.last_price_a)
        current_price_b = float(state.last_price_b)
        if self._open_position.direction == Signal.LONG_A_SHORT_B.value:
            fill_a_exit = current_price_a * (1 - self.cfg.slippage_bps / 10_000)
            fill_b_exit = current_price_b * (1 + self.cfg.slippage_bps / 10_000)
            pnl_a = (fill_a_exit - float(entry_trade["fill_a"])) * qty_a
            pnl_b = (float(entry_trade["fill_b"]) - fill_b_exit) * qty_b
        else:
            fill_a_exit = current_price_a * (1 + self.cfg.slippage_bps / 10_000)
            fill_b_exit = current_price_b * (1 - self.cfg.slippage_bps / 10_000)
            pnl_a = (float(entry_trade["fill_a"]) - fill_a_exit) * qty_a
            pnl_b = (fill_b_exit - float(entry_trade["fill_b"])) * qty_b
        gross = pnl_a + pnl_b
        fees = capital_per_leg * 4 * self.cfg.fee_bps / 10_000
        net = gross - fees
        net_bps = net / (capital_per_leg * 2.0) * 10_000 if capital_per_leg > 0 else 0.0
        return round(net, 6), round(net_bps, 4)

    def _update_mark_to_market(self, *, now: datetime, state: PairState) -> None:
        unrealized_usd, unrealized_bps = self._current_unrealized()
        self._stats["unrealized_pnl_usd"] = unrealized_usd
        self._stats["unrealized_net_bps"] = unrealized_bps
        self._stats["total_pnl_usd"] = round(float(self._stats["realized_pnl_usd"]) + unrealized_usd, 6)
        self._stats["total_net_bps"] = round(float(self._stats["realized_net_bps"]) + unrealized_bps, 4)
        if self._open_position is not None:
            self._stats["open_hold_seconds"] = round((now - self._open_position.entry_timestamp).total_seconds(), 2)
        else:
            self._stats["open_hold_seconds"] = 0.0

        day_key = now.strftime("%Y-%m-%d")
        record = self._ensure_day_record(day_key)
        equity = self.cfg.budget_usd + float(self._stats["realized_pnl_usd"]) + unrealized_usd
        record["peak_equity_usd"] = max(record["peak_equity_usd"], equity)
        if record["peak_equity_usd"] > 0:
            drawdown_usd = record["peak_equity_usd"] - equity
            drawdown_pct = drawdown_usd / record["peak_equity_usd"]
            record["max_drawdown_usd"] = max(record["max_drawdown_usd"], drawdown_usd)
            record["max_drawdown_pct"] = max(record["max_drawdown_pct"], drawdown_pct)
        record["current_equity_usd"] = equity

    def _ensure_day_record(self, day_key: str) -> dict[str, Any]:
        if day_key not in self._daily_records:
            self._daily_records[day_key] = {
                "date": day_key,
                "trade_count": 0,
                "win_count": 0,
                "loss_count": 0,
                "total_net_bps": 0.0,
                "total_net_usdt": 0.0,
                "signal_count": 0,
                "entries_taken": 0,
                "blocked_signal_count": 0,
                "ratio_tick_count": 0,
                "exit_reason_counts": {},
                "peak_equity_usd": self.cfg.budget_usd + float(self._stats["realized_pnl_usd"]),
                "current_equity_usd": self.cfg.budget_usd + float(self._stats["realized_pnl_usd"]),
                "max_drawdown_usd": 0.0,
                "max_drawdown_pct": 0.0,
                "last_ratio_tick_at": None,
                "last_trade_at": None,
            }
        if day_key not in self._daily_samples:
            self._daily_samples[day_key] = {"a": [], "b": [], "timestamps": []}
        self._last_daily_day_key = day_key
        return self._daily_records[day_key]

    def _maybe_finalize_previous_days(self, *, current_day_key: str) -> None:
        for day_key in list(self._daily_records.keys()):
            if day_key >= current_day_key:
                continue
            self._write_daily_summary(day_key, finalize=True)
            self._daily_records.pop(day_key, None)
            self._daily_samples.pop(day_key, None)

    def _finalize_all_daily_records(self, *, finalize_open_position: bool) -> None:
        if finalize_open_position and self._open_position is not None:
            state = self.ratio_engine.get_state(self.pair_config.pair_key)
            if state is not None:
                now = datetime.now(UTC)
                self._handle_exit_signal(now=now, state=state, exit_reason="shutdown_exit")
        for day_key in list(self._daily_records.keys()):
            self._write_daily_summary(day_key, finalize=True)
            self._daily_records.pop(day_key, None)
            self._daily_samples.pop(day_key, None)

    def _write_daily_summary(self, day_key: str, *, finalize: bool) -> None:
        record = self._daily_records.get(day_key)
        if record is None:
            return
        pvalue = self._compute_cointegration_pvalue(day_key)
        summary = {
            "date": day_key,
            "pair": self.pair_config.pair_key,
            "trade_count": int(record["trade_count"]),
            "win_count": int(record["win_count"]),
            "loss_count": int(record["loss_count"]),
            "win_rate": round((record["win_count"] / record["trade_count"]) if record["trade_count"] else 0.0, 6),
            "total_net_bps": round(float(record["total_net_bps"]), 4),
            "total_net_usdt": round(float(record["total_net_usdt"]), 6),
            "max_drawdown_usd": round(float(record["max_drawdown_usd"]), 6),
            "max_drawdown_pct": round(float(record["max_drawdown_pct"]), 6),
            "signals_generated": int(record["signal_count"]),
            "trades_taken": int(record["entries_taken"]),
            "blocked_signals": int(record["blocked_signal_count"]),
            "ratio_tick_count": int(record["ratio_tick_count"]),
            "cointegration_pvalue": pvalue,
            "exit_reason_counts": record["exit_reason_counts"],
            "last_ratio_tick_at": record["last_ratio_tick_at"],
            "last_trade_at": record["last_trade_at"],
            "finalized": finalize,
        }
        self.audit.log_daily_summary(summary)
        self._stats["current_cointegration_pvalue"] = pvalue
        self._stats["last_daily_summary_at"] = datetime.now(UTC).isoformat()

    def _maybe_write_hourly_check(self, *, now: datetime) -> None:
        if self._last_hourly_check_at is not None and (now - self._last_hourly_check_at).total_seconds() < self.cfg.hourly_check_seconds:
            return
        current_day = now.strftime("%Y-%m-%d")
        current_record = self._ensure_day_record(current_day)
        pvalue = self._compute_cointegration_pvalue(current_day)
        check = {
            "timestamp": now.isoformat(),
            "pair": self.pair_config.pair_key,
            "ratio_ticks_total": int(self._stats["ratio_updates"]),
            "signals_total": int(self._stats["entry_signals"]),
            "entries_total": int(self._stats["entries"]),
            "closed_trades_total": int(self._stats["closed_trades"]),
            "last_ratio_tick_at": self._stats["last_ratio_tick_at"],
            "last_trade_at": self._stats["last_trade_at"],
            "current_day_ratio_ticks": int(current_record["ratio_tick_count"]),
            "current_day_signals": int(current_record["signal_count"]),
            "current_day_entries": int(current_record["entries_taken"]),
            "current_day_trades": int(current_record["trade_count"]),
            "current_day_net_bps": round(float(current_record["total_net_bps"]), 4),
            "current_day_net_usdt": round(float(current_record["total_net_usdt"]), 6),
            "current_day_cointegration_pvalue": pvalue,
            "recording_alive": self._last_ratio_tick_at is not None,
            "streamer_messages": int(self.streamer.stats.get("messages") or 0),
            "bars_emitted": int(self.streamer.stats.get("bars_emitted") or 0),
        }
        self.audit.log_hourly_check(check)
        self._last_hourly_check_at = now
        self._stats["last_hourly_check_at"] = now.isoformat()
        self._stats["current_cointegration_pvalue"] = pvalue
        self._write_daily_summary(current_day, finalize=False)

    def _compute_cointegration_pvalue(self, day_key: str) -> float | None:
        if coint is None:
            return None
        sample = self._daily_samples.get(day_key)
        if not sample:
            return None
        a = sample["a"]
        b = sample["b"]
        if len(a) < 50 or len(b) < 50:
            return None
        if len(a) > MAX_COINTEGRATION_SAMPLES:
            step = max(1, len(a) // MAX_COINTEGRATION_SAMPLES)
            a = a[::step]
            b = b[::step]
        try:
            _, pvalue, _ = coint(a, b)
            return round(float(pvalue), 6)
        except Exception as exc:
            logger.warning("[CRYPTO_PAIRS_SHADOW] Failed to compute cointegration p-value: %s", exc)
            return None

    def _current_day_record(self) -> dict[str, Any]:
        day_key = self._last_daily_day_key or datetime.now(UTC).strftime("%Y-%m-%d")
        return self._ensure_day_record(day_key)

    def _write_runtime_state(self) -> None:
        self.audit.write_runtime_state(
            {
                "updated_at": datetime.now(UTC).isoformat(),
                "strategy": self.name,
                "label": self.profile.label,
                "view_key": self.profile.view_key,
                "source": self.profile.source,
                "pair": self.pair_config.pair_key,
                "stats": self.stats,
                "recent_trades": self._recent_trades[:10],
                "recent_signals": self._recent_signals[:10],
            }
        )

    def _total_fees_paid(self) -> float:
        closed = int(self._stats["closed_trades"])
        capital_per_leg = self.cfg.budget_usd * self.cfg.capital_per_pair_pct / 2.0
        return capital_per_leg * 4 * self.cfg.fee_bps / 10_000 * closed

    def _human_direction(self, direction: str) -> str:
        if direction == Signal.LONG_A_SHORT_B.value:
            return f"LONG_{self.pair_config.token_a}_SHORT_{self.pair_config.token_b}"
        if direction == Signal.SHORT_A_LONG_B.value:
            return f"SHORT_{self.pair_config.token_a}_LONG_{self.pair_config.token_b}"
        return direction
