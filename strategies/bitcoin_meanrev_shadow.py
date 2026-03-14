"""
Frozen BTC multivenue mean-reversion shadow sleeve.

This runs the preregistered downshock mean-reversion model live inside Oracle,
but only in shadow mode:
- uses live futures + spot + Coinbase L2
- archives every future event for later controlled tests
- never retrains or updates model weights online
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from data.models import Event, Market
from engine.btc_multivenue_feature_feed import BtcMultivenueFeatureFeed, MultiVenueFeedSnapshot
from runtime_paths import DATA_DIR, LOG_DIR
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

try:
    from catboost import CatBoostClassifier, Pool
except Exception:  # pragma: no cover - runtime dependency gate
    CatBoostClassifier = None
    Pool = None


UTC = timezone.utc
MAX_RECENT_TRADES = 120
MAX_RECENT_SIGNALS = 40


@dataclass
class ShadowPosition:
    opened_at: datetime
    entry_mid: float
    entry_price: float
    score: float
    signal_bucket: str


class FrozenMeanRevBundle:
    def __init__(self, spec_path: str | Path):
        self.spec_path = Path(spec_path).resolve()
        self.spec: dict[str, Any] = {}
        self.model: CatBoostClassifier | None = None
        self.feature_names: list[str] = []
        self.ready = False
        self.load_error: str | None = None
        self.model_path: Path | None = None
        self.strategy_id = "unloaded"
        self.score_threshold = 0.0
        self.shock_window_seconds = 5
        self.shock_bps = 0.0
        self.take_profit_bps = 0.0
        self.stop_loss_bps = 0.0
        self.max_hold_seconds = 0
        self.cooldown_seconds = 0
        self.fee_bps_per_side = 0.0
        self.entry_slippage_bps = 0.0
        self.exit_slippage_bps = 0.0
        self.past_window_column = "fut_ret_5s"
        self._load()

    def _load(self) -> None:
        if CatBoostClassifier is None or Pool is None:
            self.load_error = "catboost_not_installed"
            return
        if not self.spec_path.exists():
            self.load_error = f"missing_spec:{self.spec_path}"
            return
        try:
            self.spec = json.loads(self.spec_path.read_text(encoding="utf-8"))
            self.strategy_id = str(self.spec.get("strategy_id") or "btc-meanrev-shadow")
            signal = self.spec.get("signal_definition") or {}
            execution = self.spec.get("execution_definition") or {}
            self.score_threshold = float(signal.get("score_threshold") or 0.0)
            self.shock_window_seconds = int(signal.get("shock_window_seconds") or 5)
            self.shock_bps = float(signal.get("shock_bps") or 0.0)
            self.take_profit_bps = float(execution.get("take_profit_bps") or 0.0)
            self.stop_loss_bps = float(execution.get("stop_loss_bps") or 0.0)
            self.max_hold_seconds = int(execution.get("max_hold_seconds") or 0)
            self.cooldown_seconds = int(execution.get("cooldown_seconds") or 0)
            self.fee_bps_per_side = float(execution.get("fee_bps_per_side") or 0.0)
            self.entry_slippage_bps = float(execution.get("entry_slippage_bps") or 0.0)
            self.exit_slippage_bps = float(execution.get("exit_slippage_bps") or 0.0)
            self.past_window_column = f"fut_ret_{self.shock_window_seconds}s"
            self.model_path = Path(self.spec["model_path"]).resolve()
            if not self.model_path.exists():
                self.load_error = f"missing_model:{self.model_path}"
                return
            self.model = CatBoostClassifier()
            self.model.load_model(str(self.model_path))
            self.feature_names = list(self.model.feature_names_ or [])
            self.ready = bool(self.feature_names)
            if not self.ready:
                self.load_error = "missing_feature_names"
        except Exception as exc:
            self.load_error = str(exc)
            self.ready = False

    def predict_score(self, feature_row: dict[str, float]) -> float | None:
        if not self.ready or self.model is None or Pool is None:
            return None
        values = [float(feature_row.get(name, 0.0) or 0.0) for name in self.feature_names]
        pool = Pool([values], feature_names=self.feature_names)
        return float(self.model.predict_proba(pool)[0][1])


class BitcoinMeanRevShadowStrategy(BaseStrategy):
    name = "bitcoin_meanrev_shadow"
    description = "Frozen BTC multivenue downshock mean-reversion shadow sleeve"

    def __init__(self, config):
        super().__init__(config)
        self.cfg = config.bitcoin_meanrev_shadow
        self.bundle = FrozenMeanRevBundle(self.cfg.spec_path)
        capture_root = Path(self.cfg.capture_root).resolve() if self.cfg.capture_root else (DATA_DIR / "btc_multivenue_capture")
        self.feed = BtcMultivenueFeatureFeed(
            symbol=self.cfg.symbol,
            product_id=self.cfg.product_id,
            capture_root=capture_root,
            session_label=self.cfg.session_label,
            bucket_seconds=self.cfg.bucket_seconds,
            levels=self.cfg.levels,
            warmup_buckets=self.cfg.warmup_buckets,
        )
        self.enabled = bool(self.cfg.enabled and self.bundle.ready)
        self.log_path = LOG_DIR / "bitcoin_meanrev_shadow.jsonl"
        self._loop_task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._last_scored_bucket: str | None = None
        self._next_allowed_time: datetime | None = None
        self._open_position: ShadowPosition | None = None
        self._recent_trades: list[dict[str, Any]] = []
        self._recent_signals: list[dict[str, Any]] = []
        self._day_totals_bps: dict[str, float] = {}
        self._day_trade_counts: dict[str, int] = {}
        self._stats.update(
            {
                "bundle_ready": self.bundle.ready,
                "bundle_error": self.bundle.load_error,
                "strategy_id": self.bundle.strategy_id,
                "spec_path": str(self.bundle.spec_path),
                "model_path": str(self.bundle.model_path) if self.bundle.model_path else None,
                "feature_count": len(self.bundle.feature_names),
                "frozen_model_only": True,
                "online_learning_enabled": False,
                "capture_root": str(capture_root),
                "session_root": str(self.feed.writer.session_root),
                "archive_future_data": True,
                "warmup_buckets": self.cfg.warmup_buckets,
                "trade_notional_usd": self.cfg.trade_notional_usd,
                "score_threshold": self.bundle.score_threshold,
                "shock_bps": self.bundle.shock_bps,
                "take_profit_bps": self.bundle.take_profit_bps,
                "stop_loss_bps": self.bundle.stop_loss_bps,
                "max_hold_seconds": self.bundle.max_hold_seconds,
                "cooldown_seconds": self.bundle.cooldown_seconds,
                "last_loop_at": None,
                "last_bucket_at": None,
                "last_price": 0.0,
                "last_score": 0.0,
                "candidate_events": 0,
                "qualified_candidates": 0,
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
                "open_position": False,
                "open_hold_seconds": 0.0,
                "last_exit_reason": None,
                "last_trade_day": None,
                "positive_days": 0,
                "negative_days": 0,
                "day_count": 0,
                "feed_stats": self.feed.stats,
                "log_entries": 0,
                "last_log_at": None,
            }
        )

    @property
    def stats(self) -> dict:
        return {
            "name": self.name,
            "enabled": self.enabled,
            **self._stats,
            "log_path": str(self.log_path.resolve()),
        }

    async def ensure_started(self) -> None:
        if not self.enabled:
            return
        await self.feed.ensure_started()
        if self._loop_task is None:
            self._loop_task = asyncio.create_task(self._shadow_loop(), name="btc-meanrev-shadow-loop")

    async def close(self) -> None:
        self._stop.set()
        if self._loop_task is not None:
            self._loop_task.cancel()
            await asyncio.gather(self._loop_task, return_exceptions=True)
            self._loop_task = None
        await self.feed.close()

    async def scan(self, markets: list[Market], events: list[Event]) -> list:
        del markets, events
        self._stats["scans_completed"] += 1
        self._stats["feed_stats"] = self.feed.stats
        await self.ensure_started()
        return []

    def serialize_view(self) -> dict[str, Any]:
        open_positions = []
        cash = self.cfg.budget_usd + float(self._stats["realized_pnl_usd"])
        positions_value = 0.0
        if self._open_position is not None:
            positions_value = self.cfg.trade_notional_usd + float(self._stats["unrealized_pnl_usd"])
            cash -= self.cfg.trade_notional_usd
            open_positions.append(
                {
                    "market": "btc-meanrev-shadow",
                    "side": "LONG",
                    "shares": round(self.cfg.trade_notional_usd / max(self._open_position.entry_price, 1.0), 6),
                    "entry": round(self._open_position.entry_price, 2),
                    "current": round(float(self._stats["last_price"] or self._open_position.entry_price), 2),
                    "pnl": round(float(self._stats["unrealized_pnl_usd"]), 2),
                    "source": "bitcoin_meanrev_shadow",
                }
            )

        total_value = cash + positions_value
        starting_capital = self.cfg.budget_usd
        total_pnl = total_value - starting_capital
        total_pnl_pct = (total_pnl / starting_capital * 100.0) if starting_capital > 0 else 0.0

        return {
            "key": "bitcoin_meanrev_shadow",
            "label": "BTC MeanRev Shadow",
            "source": "bitcoin_meanrev_shadow",
            "portfolio": {
                "starting_capital": round(starting_capital, 2),
                "total_value": round(total_value, 2),
                "cash": round(cash, 2),
                "positions_value": round(positions_value, 2),
                "total_pnl": round(total_pnl, 2),
                "total_pnl_pct": round(total_pnl_pct, 2),
                "total_trades": int(self._stats["closed_trades"]),
                "win_rate": round(float(self._stats["win_rate"]) * 100.0, 1),
                "max_drawdown": 0.0,
                "total_fees": round(self.cfg.trade_notional_usd * (self.bundle.fee_bps_per_side / 10000.0) * 2.0 * int(self._stats["closed_trades"]), 2),
                "positions": open_positions,
            },
            "signals": list(self._recent_signals[:30]),
            "trades": list(self._recent_trades[:30]),
            "performance": {
                "total_pnl": round(total_pnl, 2),
                "total_pnl_pct": round(total_pnl_pct, 2),
                "win_rate": float(self._stats["win_rate"]),
                "total_trades": int(self._stats["closed_trades"]),
                "realized_pnl": round(float(self._stats["realized_pnl_usd"]), 2),
                "realized_net_bps": round(float(self._stats["realized_net_bps"]), 2),
            },
        }

    async def _shadow_loop(self) -> None:
        while not self._stop.is_set():
            try:
                snapshot = await self.feed.snapshot()
                self._stats["feed_stats"] = self.feed.stats
                self._stats["last_loop_at"] = datetime.now(UTC).isoformat()
                self._process_snapshot(snapshot)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._stats["errors"] += 1
                logger.warning("[BTC_SHADOW] shadow loop error: %s", exc)
            await asyncio.sleep(max(self.cfg.evaluation_interval_seconds, 0.2))

    def _process_snapshot(self, snapshot: MultiVenueFeedSnapshot) -> None:
        if not snapshot.bucket_at or snapshot.fut_mid_price is None:
            return

        now = snapshot.bucket_at
        self._stats["last_bucket_at"] = now.isoformat()
        self._stats["last_price"] = round(float(snapshot.fut_mid_price), 2)
        self._update_open_position(now=now, fut_mid_price=float(snapshot.fut_mid_price))

        if not snapshot.ready or snapshot.feature_row is None:
            return

        bucket_key = now.isoformat()
        if bucket_key == self._last_scored_bucket:
            return
        self._last_scored_bucket = bucket_key

        past_return = float(snapshot.feature_row.get(self.bundle.past_window_column, 0.0) or 0.0) * 10000.0
        if past_return > -self.bundle.shock_bps:
            return

        self._stats["candidate_events"] += 1
        score = self.bundle.predict_score(snapshot.feature_row)
        if score is None:
            return
        self._stats["last_score"] = round(score, 6)
        if score < self.bundle.score_threshold:
            return

        self._stats["qualified_candidates"] += 1
        if self._open_position is not None:
            return
        if self._next_allowed_time is not None and now < self._next_allowed_time:
            return

        entry_mid = float(snapshot.fut_mid_price)
        entry_price = entry_mid * (1.0 + self.bundle.entry_slippage_bps / 10000.0)
        self._open_position = ShadowPosition(
            opened_at=now,
            entry_mid=entry_mid,
            entry_price=entry_price,
            score=float(score),
            signal_bucket=bucket_key,
        )
        self._stats["entries"] += 1
        self._stats["open_position"] = True
        self._record_signal(
            {
                "id": f"btc-shadow-{now.strftime('%H%M%S')}",
                "time": now.isoformat(),
                "source": "bitcoin_meanrev_shadow",
                "action": "shadow_long",
                "market": "btc-meanrev-shadow",
                "confidence": round(float(score), 2),
                "edge": round(self.bundle.take_profit_bps, 2),
                "size": round(self.cfg.trade_notional_usd, 2),
                "whale": False,
                "reasoning": (
                    f"Frozen {self.bundle.strategy_id}: past {self.bundle.shock_window_seconds}s return "
                    f"{past_return:.2f}bps <= -{self.bundle.shock_bps:.1f}bps, score {score:.3f} >= "
                    f"{self.bundle.score_threshold:.2f}. No online learning."
                ),
            }
        )
        self._append_log("entry", now=now, extra={"score": score, "past_return_bps": past_return, "entry_price": entry_price})

    def _update_open_position(self, *, now: datetime, fut_mid_price: float) -> None:
        if self._open_position is None:
            self._stats["unrealized_net_bps"] = 0.0
            self._stats["unrealized_pnl_usd"] = 0.0
            self._stats["open_hold_seconds"] = 0.0
            self._stats["total_net_bps"] = round(float(self._stats["realized_net_bps"]), 4)
            self._stats["total_pnl_usd"] = round(float(self._stats["realized_pnl_usd"]), 4)
            return

        hold_seconds = max((now - self._open_position.opened_at).total_seconds(), 0.0)
        self._stats["open_hold_seconds"] = round(hold_seconds, 2)
        gross_mid_bps = (fut_mid_price / self._open_position.entry_price - 1.0) * 10000.0
        estimated_exit_price = fut_mid_price * (1.0 - self.bundle.exit_slippage_bps / 10000.0)
        unrealized_gross_bps = (estimated_exit_price / self._open_position.entry_price - 1.0) * 10000.0
        unrealized_net_bps = unrealized_gross_bps - (2.0 * self.bundle.fee_bps_per_side)
        self._stats["unrealized_net_bps"] = round(unrealized_net_bps, 4)
        self._stats["unrealized_pnl_usd"] = round(self.cfg.trade_notional_usd * unrealized_net_bps / 10000.0, 4)
        self._stats["total_net_bps"] = round(float(self._stats["realized_net_bps"]) + unrealized_net_bps, 4)
        self._stats["total_pnl_usd"] = round(float(self._stats["realized_pnl_usd"]) + float(self._stats["unrealized_pnl_usd"]), 4)

        exit_reason = None
        if gross_mid_bps >= self.bundle.take_profit_bps:
            exit_reason = "take_profit"
        elif gross_mid_bps <= -self.bundle.stop_loss_bps:
            exit_reason = "stop_loss"
        elif hold_seconds >= self.bundle.max_hold_seconds:
            exit_reason = "timeout"

        if exit_reason is None:
            return

        exit_price = estimated_exit_price
        net_bps = (exit_price / self._open_position.entry_price - 1.0) * 10000.0 - (2.0 * self.bundle.fee_bps_per_side)
        pnl_usd = self.cfg.trade_notional_usd * net_bps / 10000.0
        trade_day = now.strftime("%Y-%m-%d")
        self._day_totals_bps[trade_day] = self._day_totals_bps.get(trade_day, 0.0) + net_bps
        self._day_trade_counts[trade_day] = self._day_trade_counts.get(trade_day, 0) + 1
        self._stats["closed_trades"] += 1
        self._stats["realized_net_bps"] = round(float(self._stats["realized_net_bps"]) + net_bps, 4)
        self._stats["realized_pnl_usd"] = round(float(self._stats["realized_pnl_usd"]) + pnl_usd, 4)
        self._stats["last_exit_reason"] = exit_reason
        self._stats["last_trade_day"] = trade_day
        if net_bps > 0.0:
            self._stats["wins"] += 1
        else:
            self._stats["losses"] += 1
        total_closed = int(self._stats["closed_trades"])
        self._stats["win_rate"] = float(self._stats["wins"]) / total_closed if total_closed else 0.0
        positive_days = sum(1 for total in self._day_totals_bps.values() if total > 0.0)
        negative_days = sum(1 for total in self._day_totals_bps.values() if total <= 0.0)
        self._stats["positive_days"] = positive_days
        self._stats["negative_days"] = negative_days
        self._stats["day_count"] = len(self._day_totals_bps)
        self._stats["open_position"] = False

        record = {
            "id": f"btc-shadow-trade-{now.strftime('%H%M%S')}",
            "time": now.isoformat(),
            "source": "bitcoin_meanrev_shadow",
            "market": "btc-meanrev-shadow",
            "side": "BUY",
            "price": round(self._open_position.entry_price, 2),
            "shares": round(self.cfg.trade_notional_usd / max(self._open_position.entry_price, 1.0), 6),
            "usd": round(self.cfg.trade_notional_usd, 2),
            "pnl": round(pnl_usd, 4),
            "net_bps": round(net_bps, 4),
            "exit_reason": exit_reason,
            "hold_seconds": round(hold_seconds, 2),
        }
        self._recent_trades.insert(0, record)
        self._recent_trades = self._recent_trades[:MAX_RECENT_TRADES]
        self._next_allowed_time = now + timedelta(seconds=self.bundle.cooldown_seconds)
        self._append_log("exit", now=now, extra=record)
        self._open_position = None
        self._stats["unrealized_net_bps"] = 0.0
        self._stats["unrealized_pnl_usd"] = 0.0
        self._stats["total_net_bps"] = round(float(self._stats["realized_net_bps"]), 4)
        self._stats["total_pnl_usd"] = round(float(self._stats["realized_pnl_usd"]), 4)

    def _record_signal(self, signal: dict[str, Any]) -> None:
        self._recent_signals.insert(0, signal)
        self._recent_signals = self._recent_signals[:MAX_RECENT_SIGNALS]
        self._stats["signals_generated"] += 1

    def _append_log(self, event_type: str, *, now: datetime, extra: dict[str, Any]) -> None:
        entry = {
            "timestamp": now.isoformat(),
            "event": event_type,
            "strategy_id": self.bundle.strategy_id,
            "frozen_model_only": True,
            "online_learning_enabled": False,
            "session_root": str(self.feed.writer.session_root),
            **extra,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, separators=(",", ":")) + "\n")
        self._stats["log_entries"] += 1
        self._stats["last_log_at"] = now.isoformat()
