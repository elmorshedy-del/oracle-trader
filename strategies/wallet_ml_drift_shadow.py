from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from catboost import CatBoostClassifier

from config import WalletMLDriftShadowConfig
from data.models import Event, Market
from engine.shadow_sleeve_audit import ShadowSleeveAudit
from engine.wallet_copy_ml_features import build_wallet_copy_feature_frame
from engine.wallet_copy_store import WalletCopyResearchStore
from runtime_paths import LOG_DIR
from strategies.base import BaseStrategy


logger = logging.getLogger(__name__)

UTC = timezone.utc
MAX_RECENT_ITEMS = 40
ML_DECISION_LOG_KEY = "ml_decision_log"
ML_TRADE_LEDGER_FIELDS = [
    "logged_at",
    "trade_id",
    "decision_id",
    "source_trade_id",
    "wallet_address",
    "market_slug",
    "outcome",
    "entry_timestamp",
    "entry_price",
    "entry_size_usd",
    "exit_timestamp",
    "exit_price",
    "realized_pnl_usd",
    "close_reason",
    "model_score",
]
ML_DAILY_SUMMARY_FIELDS = [
    "logged_at",
    "date",
    "decisions_logged",
    "paper_trades",
    "open_positions",
    "closed_trades",
    "wins",
    "losses",
    "win_rate",
    "realized_pnl_usd",
    "cash_balance_usd",
    "portfolio_value_usd",
]


@dataclass(slots=True)
class MLDriftPosition:
    trade_id: str
    decision_id: int
    source_trade_id: int
    wallet_address: str
    condition_id: str
    market_slug: str
    token_id: str
    outcome_label: str
    shares: float
    entry_price: float
    entry_size_usd: float
    entry_timestamp: datetime
    model_score: float

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "MLDriftPosition":
        return cls(
            trade_id=str(payload["trade_id"]),
            decision_id=int(payload["decision_id"]),
            source_trade_id=int(payload["source_trade_id"]),
            wallet_address=str(payload["wallet_address"]),
            condition_id=str(payload["condition_id"]),
            market_slug=str(payload["market_slug"]),
            token_id=str(payload["token_id"]),
            outcome_label=str(payload["outcome_label"]),
            shares=float(payload["shares"]),
            entry_price=float(payload["entry_price"]),
            entry_size_usd=float(payload["entry_size_usd"]),
            entry_timestamp=datetime.fromisoformat(str(payload["entry_timestamp"])).astimezone(UTC),
            model_score=float(payload.get("model_score") or 0.0),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "decision_id": self.decision_id,
            "source_trade_id": self.source_trade_id,
            "wallet_address": self.wallet_address,
            "condition_id": self.condition_id,
            "market_slug": self.market_slug,
            "token_id": self.token_id,
            "outcome_label": self.outcome_label,
            "shares": self.shares,
            "entry_price": self.entry_price,
            "entry_size_usd": self.entry_size_usd,
            "entry_timestamp": self.entry_timestamp.isoformat(),
            "model_score": self.model_score,
        }


class WalletMLDriftShadowStrategy(BaseStrategy):
    name = "wallet_ml_drift_shadow"
    description = "Paper sleeve that trades top-decile wallet-copy ML drift scores with timed exits"

    def __init__(self, config, collector, store: WalletCopyResearchStore, profile: WalletMLDriftShadowConfig):
        super().__init__(config)
        self.collector = collector
        self.store = store
        self.profile = profile
        self.cash_balance = float(profile.budget_usd)
        self.open_positions: dict[str, MLDriftPosition] = {}
        self.realized_pnl_usd = 0.0
        self.wins = 0
        self.losses = 0
        self.closed_trades = 0
        self.decisions_logged = 0
        self.paper_trades = 0
        self.last_processed_trade_id = 0
        self._last_prices: dict[str, float] = {}
        self._recent_signals: list[dict[str, Any]] = []
        self._recent_trades: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {}
        self.model: CatBoostClassifier | None = None
        self.model_error: str | None = None
        self.feature_columns: list[str] = []
        self.categorical_features: list[str] = []
        self.score_threshold = float(profile.score_threshold or 0.0)
        self.audit_root = self._resolve_audit_root()
        self.audit = ShadowSleeveAudit(
            self.audit_root,
            lane_key=self.profile.source,
            label=self.profile.label,
            category="wallet_copy_ml",
            source=self.profile.source,
            description="Wallet-copy ML paper sleeve: top-decile 30-minute forward-drift filter.",
            trade_ledger_fields=ML_TRADE_LEDGER_FIELDS,
            daily_summary_fields=ML_DAILY_SUMMARY_FIELDS,
            extra_jsonl_keys=(ML_DECISION_LOG_KEY,),
        )
        self._load_model()
        self.enabled = bool(self.profile.enabled and self.model is not None)
        self._stats.update(
            {
                "view_key": self.profile.view_key,
                "source": self.profile.source,
                "model_loaded": self.model is not None,
                "model_error": self.model_error,
                "score_threshold": round(self.score_threshold, 6),
                "horizon_seconds": self.profile.horizon_seconds,
                "target_move_dollars": self.profile.target_move_dollars,
                "allowed_categories": list(self.profile.allowed_categories),
                "budget_usd": self.profile.budget_usd,
                "max_open_positions": self.profile.max_open_positions,
                "cash_balance_usd": self.cash_balance,
                "portfolio_value_usd": self.cash_balance,
                "decisions_logged": 0,
                "paper_trades": 0,
                "closed_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "realized_pnl_usd": 0.0,
                "last_processed_trade_id": 0,
                "runtime_state_path": str(self.audit.paths["runtime_state"]),
                "trade_ledger_csv": str(self.audit.paths["trade_ledger_csv"]),
            }
        )
        self.audit.write_metadata(
            {
                "started_at": datetime.now(UTC).isoformat(),
                "strategy": self.profile.strategy_key,
                "view_key": self.profile.view_key,
                "model_path": str(self._resolve_path(self.profile.model_path)),
                "metadata_path": str(self._resolve_path(self.profile.metadata_path)),
                "model_loaded": self.model is not None,
                "model_error": self.model_error,
                "score_threshold": self.score_threshold,
                "horizon_seconds": self.profile.horizon_seconds,
                "target_move_dollars": self.profile.target_move_dollars,
                "allowed_categories": list(self.profile.allowed_categories),
                "budget_usd": self.profile.budget_usd,
                "min_trade_usd": self.profile.min_trade_usd,
                "max_trade_usd": self.profile.max_trade_usd,
            },
            extra={"view_key": self.profile.view_key, "session_label": self.profile.session_label},
        )
        self._load_runtime_state()
        if self.last_processed_trade_id <= 0:
            self.last_processed_trade_id = self.store.latest_wallet_trade_id()
        self._stats["last_processed_trade_id"] = self.last_processed_trade_id
        self._stats["cash_balance_usd"] = round(self.cash_balance, 4)
        self._stats["portfolio_value_usd"] = round(self._current_equity(), 4)
        self._write_runtime_state()

    async def scan(self, markets: list[Market], events: list[Event]) -> list:
        del events
        self._stats["scans_completed"] += 1
        now = datetime.now(UTC)
        market_map = {market.condition_id: market for market in markets}
        self._refresh_last_prices(market_map)
        await self._close_due_positions(now=now, market_map=market_map)
        if self.enabled:
            await self._process_new_trades(now=now)
        self._roll_daily_summary(now)
        self._stats["cash_balance_usd"] = round(self.cash_balance, 4)
        self._stats["portfolio_value_usd"] = round(self._current_equity(), 4)
        self._stats["last_processed_trade_id"] = self.last_processed_trade_id
        self._write_runtime_state()
        return []

    @property
    def stats(self) -> dict:
        return {"name": self.profile.strategy_key, "enabled": self.enabled, **self._stats}

    def serialize_view(self) -> dict[str, Any]:
        positions = []
        current_value = 0.0
        for position in self.open_positions.values():
            mark = self._last_prices.get(position.token_id, position.entry_price)
            value = mark * position.shares
            current_value += value
            positions.append(
                {
                    "market": position.market_slug,
                    "side": position.outcome_label,
                    "shares": round(position.shares, 2),
                    "entry": round(position.entry_price, 4),
                    "current": round(mark, 4),
                    "pnl": round(value - position.entry_size_usd, 2),
                    "source": self.profile.source,
                }
            )
        total_value = self.cash_balance + current_value
        return {
            "key": self.profile.view_key,
            "label": self.profile.label,
            "source": self.profile.source,
            "portfolio": {
                "starting_capital": round(self.profile.budget_usd, 2),
                "total_value": round(total_value, 2),
                "cash": round(self.cash_balance, 2),
                "positions_value": round(current_value, 2),
                "total_pnl": round(total_value - self.profile.budget_usd, 2),
                "total_pnl_pct": round(((total_value / self.profile.budget_usd) - 1.0) * 100.0, 2) if self.profile.budget_usd else 0.0,
                "total_trades": self.closed_trades,
                "win_rate": round((self.wins / self.closed_trades) * 100.0, 1) if self.closed_trades else 0.0,
                "max_drawdown": 0.0,
                "positions": positions,
            },
            "signals": self._recent_signals[:MAX_RECENT_ITEMS],
            "trades": self._recent_trades[:MAX_RECENT_ITEMS],
            "performance": {
                "total_pnl": round(total_value - self.profile.budget_usd, 2),
                "win_rate": (self.wins / self.closed_trades) if self.closed_trades else 0.0,
                "total_trades": self.closed_trades,
                "cash": round(self.cash_balance, 2),
                "budget": round(self.profile.budget_usd, 2),
                "paper_trades": self.paper_trades,
                "score_threshold": round(self.score_threshold, 6),
            },
        }

    async def close(self) -> None:
        self._write_runtime_state()

    async def _process_new_trades(self, *, now: datetime) -> None:
        rows = self.store.list_wallet_trades_after_id(
            last_id=self.last_processed_trade_id,
            limit=self.profile.scan_trade_limit,
            side="BUY",
        )
        for row in rows:
            trade_id = int(row["id"])
            self.last_processed_trade_id = max(self.last_processed_trade_id, trade_id)
            if not self._should_score_trade(row):
                continue
            score = self._score_row(row)
            if score is None:
                continue
            entry_price = self._entry_price(row)
            suggested_size = self._suggested_size(score)
            should_copy = score >= self.score_threshold and suggested_size >= self.profile.min_trade_usd
            reasons = ["ml_top_decile_30m_plus_2c"] if should_copy else ["below_ml_threshold"]
            blockers: list[str] = []
            if should_copy:
                blockers = self._paper_trade_blockers(row=row, entry_price=entry_price, suggested_size_usd=suggested_size)
                if blockers:
                    should_copy = False
                    suggested_size = 0.0
                    reasons.extend(blockers)
            decision_payload = self._decision_payload(
                row=row,
                score=score,
                should_copy=should_copy,
                suggested_size_usd=suggested_size,
                reasons=reasons,
                entry_price=entry_price,
                now=now,
            )
            decision_id = self.store.insert_copy_decision(decision_payload)
            self.decisions_logged += 1
            self._stats["decisions_logged"] = self.decisions_logged
            self.audit.log_extra(ML_DECISION_LOG_KEY, {**decision_payload, "decision_id": decision_id})
            self._remember_signal(decision_payload, now)
            if should_copy:
                self._open_position(
                    decision_id=decision_id,
                    row=row,
                    score=score,
                    entry_price=entry_price,
                    size_usd=suggested_size,
                    now=now,
                )

    async def _close_due_positions(self, *, now: datetime, market_map: dict[str, Market]) -> None:
        for position in list(self.open_positions.values()):
            age_seconds = (now - position.entry_timestamp).total_seconds()
            if age_seconds < self.profile.horizon_seconds:
                continue
            exit_price = await self._mark_price(position=position, market_map=market_map)
            if exit_price is None:
                continue
            self._close_position(
                position,
                exit_price=exit_price,
                close_reason=f"horizon_{self.profile.horizon_seconds}s",
                now=now,
            )

    def _score_row(self, row: dict[str, Any]) -> float | None:
        if self.model is None:
            return None
        wallet_perf = self.store.observed_wallet_performance_before(
            wallet_address=str(row.get("wallet_address") or ""),
            before_timestamp=float(row.get("timestamp") or 0.0),
        )
        market_first_seen = self.store.market_first_seen_timestamp(
            condition_id=str(row.get("market_condition_id") or "")
        )
        features = build_wallet_copy_feature_frame(
            trade_row=row,
            wallet_performance=wallet_perf,
            market_first_seen_timestamp=market_first_seen,
            feature_columns=self.feature_columns,
            categorical_features=self.categorical_features,
        )
        return float(self.model.predict_proba(features)[0, 1])

    def _should_score_trade(self, row: dict[str, Any]) -> bool:
        if str(row.get("side") or "").upper() != "BUY":
            return False
        if row.get("market_resolved") or row.get("market_closed"):
            return False
        if not row.get("asset_token_id") or not row.get("market_condition_id"):
            return False
        category = str(row.get("market_category") or "unknown").lower()
        return category in set(self.profile.allowed_categories)

    def _paper_trade_blockers(self, *, row: dict[str, Any], entry_price: float, suggested_size_usd: float) -> list[str]:
        blockers: list[str] = []
        if entry_price <= 0:
            blockers.append("missing_entry_price")
        if len(self.open_positions) >= self.profile.max_open_positions:
            blockers.append("max_open_positions_reached")
        # Shadow/paper trader — no cash gate (same rationale as copy_trader_shadow)
        # if suggested_size_usd > self.cash_balance:
        #     blockers.append("insufficient_cash")
        position_key = self._position_key(
            condition_id=str(row.get("market_condition_id") or ""),
            token_id=str(row.get("asset_token_id") or ""),
        )
        if position_key in self._open_position_keys():
            blockers.append("already_open_position")
        spread = self._as_float(row.get("market_spread"))
        if spread is not None and spread > self.profile.max_market_spread:
            blockers.append("wide_spread")
        delay = self._as_float(row.get("detection_delay_seconds"))
        if delay is not None and delay > self.profile.max_detection_delay_seconds:
            blockers.append("stale_detection")
        depth_shares = self._as_float(row.get("token_depth_within_2pct"))
        depth_usd = (depth_shares or 0.0) * entry_price
        if suggested_size_usd > 0 and depth_usd < suggested_size_usd * self.profile.min_depth_to_size_multiple:
            blockers.append("insufficient_depth")
        return blockers

    def _open_position(
        self,
        *,
        decision_id: int,
        row: dict[str, Any],
        score: float,
        entry_price: float,
        size_usd: float,
        now: datetime,
    ) -> None:
        shares = size_usd / entry_price
        trade_id = f"{self.profile.strategy_key}:{decision_id}"
        position = MLDriftPosition(
            trade_id=trade_id,
            decision_id=decision_id,
            source_trade_id=int(row["id"]),
            wallet_address=str(row.get("wallet_address") or ""),
            condition_id=str(row.get("market_condition_id") or ""),
            market_slug=str(row.get("market_slug") or ""),
            token_id=str(row.get("asset_token_id") or ""),
            outcome_label=str(row.get("outcome") or ""),
            shares=shares,
            entry_price=entry_price,
            entry_size_usd=size_usd,
            entry_timestamp=now,
            model_score=score,
        )
        self.open_positions[trade_id] = position
        self.cash_balance -= size_usd
        self.paper_trades += 1
        self._stats["paper_trades"] = self.paper_trades
        self._stats["last_entry_at"] = now.isoformat()
        self.store.mark_copy_decision_executed(
            decision_id=decision_id,
            paper_trade_id=trade_id,
            paper_side=str(row.get("outcome") or ""),
            entry_timestamp=now.timestamp(),
            entry_price=entry_price,
            entry_size_usd=size_usd,
        )
        self._recent_trades.insert(
            0,
            {
                "id": trade_id,
                "time": now.isoformat(),
                "timestamp": now.isoformat(),
                "source": self.profile.source,
                "market": position.market_slug,
                "wallet": position.wallet_address,
                "side": position.outcome_label,
                "price": round(entry_price, 4),
                "usd": round(size_usd, 2),
                "pnl": None,
                "score": round(score, 3),
                "event": "entry",
            },
        )
        self._recent_trades = self._recent_trades[:MAX_RECENT_ITEMS]
        self.audit.log_trade_event(
            {
                "event_type": "entry",
                "trade_id": trade_id,
                "decision_id": decision_id,
                "wallet_address": position.wallet_address,
                "market_slug": position.market_slug,
                "outcome": position.outcome_label,
                "price": entry_price,
                "size_usd": size_usd,
                "model_score": round(score, 6),
            }
        )

    def _close_position(
        self,
        position: MLDriftPosition,
        *,
        exit_price: float,
        close_reason: str,
        now: datetime,
    ) -> None:
        current_value = position.shares * exit_price
        realized_pnl = current_value - position.entry_size_usd
        self.cash_balance += current_value
        self.realized_pnl_usd += realized_pnl
        self.closed_trades += 1
        if realized_pnl >= 0:
            self.wins += 1
        else:
            self.losses += 1
        self._stats["closed_trades"] = self.closed_trades
        self._stats["wins"] = self.wins
        self._stats["losses"] = self.losses
        self._stats["win_rate"] = self.wins / self.closed_trades if self.closed_trades else 0.0
        self._stats["realized_pnl_usd"] = round(self.realized_pnl_usd, 4)
        self._stats["last_exit_at"] = now.isoformat()
        self.store.close_copy_decision(
            decision_id=position.decision_id,
            exit_timestamp=now.timestamp(),
            exit_price=exit_price,
            realized_pnl_usd=realized_pnl,
            close_reason=close_reason,
        )
        ledger = {
            "trade_id": position.trade_id,
            "decision_id": position.decision_id,
            "source_trade_id": position.source_trade_id,
            "wallet_address": position.wallet_address,
            "market_slug": position.market_slug,
            "outcome": position.outcome_label,
            "entry_timestamp": position.entry_timestamp.isoformat(),
            "entry_price": position.entry_price,
            "entry_size_usd": position.entry_size_usd,
            "exit_timestamp": now.isoformat(),
            "exit_price": exit_price,
            "realized_pnl_usd": round(realized_pnl, 4),
            "close_reason": close_reason,
            "model_score": round(position.model_score, 6),
        }
        self.audit.log_trade_ledger(ledger)
        self.audit.log_trade_event({"event_type": "exit", **ledger})
        self._recent_trades.insert(
            0,
            {
                "id": position.trade_id,
                "time": now.isoformat(),
                "timestamp": now.isoformat(),
                "source": self.profile.source,
                "market": position.market_slug,
                "wallet": position.wallet_address,
                "side": position.outcome_label,
                "price": round(exit_price, 4),
                "usd": round(position.entry_size_usd, 2),
                "pnl": round(realized_pnl, 2),
                "entry": round(position.entry_price, 4),
                "exit": round(exit_price, 4),
                "score": round(position.model_score, 3),
                "event": "exit",
            },
        )
        self._recent_trades = self._recent_trades[:MAX_RECENT_ITEMS]
        self.open_positions.pop(position.trade_id, None)

    def _decision_payload(
        self,
        *,
        row: dict[str, Any],
        score: float,
        should_copy: bool,
        suggested_size_usd: float,
        reasons: list[str],
        entry_price: float,
        now: datetime,
    ) -> dict[str, Any]:
        context = {
            "model_score": round(score, 8),
            "score_threshold": round(self.score_threshold, 8),
            "target": "price_30min_after_minus_entry_price>=0.02",
            "horizon_seconds": self.profile.horizon_seconds,
            "entry_price": entry_price,
            "market_category": row.get("market_category"),
            "market_spread": row.get("market_spread"),
            "token_depth_within_2pct": row.get("token_depth_within_2pct"),
            "detection_delay_seconds": row.get("detection_delay_seconds"),
            "model_path": str(self._resolve_path(self.profile.model_path)),
        }
        return {
            "decision_key": f"{self.profile.strategy_key}:{int(row['id'])}",
            "strategy_key": self.profile.strategy_key,
            "profile_kind": "ml_30m_drift",
            "wallet_trade_id": int(row["id"]),
            "wallet_trade_key": str(row.get("trade_key") or ""),
            "tx_hash": str(row.get("tx_hash") or ""),
            "wallet_address": str(row.get("wallet_address") or ""),
            "market_condition_id": str(row.get("market_condition_id") or ""),
            "asset_token_id": str(row.get("asset_token_id") or ""),
            "market_slug": str(row.get("market_slug") or ""),
            "source_side": str(row.get("side") or ""),
            "source_outcome": str(row.get("outcome") or ""),
            "score": float(round(score, 6)),
            "should_copy": 1 if should_copy else 0,
            "suggested_size_usd": float(round(suggested_size_usd, 4)),
            "reasons_json": reasons,
            "context_json": context,
            "paper_executed": 0,
            "paper_trade_id": None,
            "paper_side": None,
            "paper_entry_timestamp": None,
            "paper_entry_price": None,
            "paper_entry_size_usd": None,
            "paper_exit_timestamp": None,
            "paper_exit_price": None,
            "paper_realized_pnl_usd": None,
            "paper_close_reason": None,
            "created_at": now.timestamp(),
            "updated_at": now.timestamp(),
        }

    def _remember_signal(self, decision_payload: dict[str, Any], now: datetime) -> None:
        self._recent_signals.insert(
            0,
            {
                "id": decision_payload["decision_key"],
                "time": now.isoformat(),
                "timestamp": now.isoformat(),
                "source": self.profile.source,
                "action": "copy" if decision_payload["should_copy"] else "skip",
                "market": decision_payload["market_slug"],
                "confidence": round(float(decision_payload["score"]), 3),
                "edge": round(self.profile.target_move_dollars, 4),
                "size": round(float(decision_payload["suggested_size_usd"] or 0.0), 2),
                "whale": True,
                "reasoning": ", ".join(decision_payload["reasons_json"]),
                "wallet": decision_payload["wallet_address"],
                "outcome": decision_payload["source_outcome"],
                "score": round(float(decision_payload["score"]), 3),
                "copy": bool(decision_payload["should_copy"]),
            },
        )
        self._recent_signals = self._recent_signals[:MAX_RECENT_ITEMS]
        self.audit.log_signal(self._recent_signals[0])

    def _entry_price(self, row: dict[str, Any]) -> float:
        outcome = str(row.get("outcome") or "").strip().lower()
        if outcome == "yes":
            return float(row.get("market_yes_ask") or row.get("token_best_ask") or row.get("price") or 0.0)
        if outcome == "no":
            return float(row.get("market_no_ask") or row.get("token_best_ask") or row.get("price") or 0.0)
        return float(row.get("token_best_ask") or row.get("price") or 0.0)

    def _suggested_size(self, score: float) -> float:
        if score < self.score_threshold:
            return 0.0
        margin = max(score - self.score_threshold, 0.0)
        scale = min(margin / 0.25, 1.0)
        size = self.profile.min_trade_usd + (self.profile.max_trade_usd - self.profile.min_trade_usd) * scale
        return round(min(size, self.profile.max_trade_usd), 4)

    async def _mark_price(self, *, position: MLDriftPosition, market_map: dict[str, Market]) -> float | None:
        market = market_map.get(position.condition_id)
        if market is None:
            market = await self.collector.get_market_by_slug(position.market_slug)
        if market is None:
            return None
        for outcome in market.outcomes:
            if outcome.token_id == position.token_id:
                return float(outcome.book_bid or outcome.price or 0.0)
        return None

    def _load_model(self) -> None:
        try:
            metadata_path = self._resolve_path(self.profile.metadata_path)
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.feature_columns = [str(column) for column in self.metadata.get("feature_columns") or []]
            self.categorical_features = [str(column) for column in self.metadata.get("categorical_features") or []]
            if not self.score_threshold:
                self.score_threshold = float(self.metadata.get("recommended_score_threshold") or 0.0)
            model_path = self._resolve_path(self.profile.model_path)
            model = CatBoostClassifier()
            model.load_model(str(model_path))
            self.model = model
            if not self.feature_columns:
                raise ValueError("model metadata has no feature_columns")
            if not self.score_threshold:
                raise ValueError("model metadata has no recommended_score_threshold")
        except Exception as exc:
            self.model = None
            self.model_error = str(exc)
            logger.warning("[WALLET_ML_DRIFT] model unavailable: %s", exc)

    def _load_runtime_state(self) -> None:
        path = self.audit.paths["runtime_state"]
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        self.cash_balance = float(payload.get("cash_balance") or self.cash_balance)
        self.realized_pnl_usd = float(payload.get("realized_pnl_usd") or 0.0)
        self.wins = int(payload.get("wins") or 0)
        self.losses = int(payload.get("losses") or 0)
        self.closed_trades = int(payload.get("closed_trades") or 0)
        self.decisions_logged = int(payload.get("decisions_logged") or 0)
        self.paper_trades = int(payload.get("paper_trades") or 0)
        self.last_processed_trade_id = int(payload.get("last_processed_trade_id") or 0)
        self.open_positions = {
            trade_id: MLDriftPosition.from_payload(row)
            for trade_id, row in (payload.get("open_positions") or {}).items()
        }
        self._recent_signals = list(payload.get("recent_signals") or [])[:MAX_RECENT_ITEMS]
        self._recent_trades = list(payload.get("recent_trades") or [])[:MAX_RECENT_ITEMS]
        self._stats.update(
            {
                "cash_balance_usd": round(self.cash_balance, 4),
                "portfolio_value_usd": round(self._current_equity(), 4),
                "decisions_logged": self.decisions_logged,
                "paper_trades": self.paper_trades,
                "closed_trades": self.closed_trades,
                "wins": self.wins,
                "losses": self.losses,
                "win_rate": self.wins / self.closed_trades if self.closed_trades else 0.0,
                "realized_pnl_usd": round(self.realized_pnl_usd, 4),
                "last_processed_trade_id": self.last_processed_trade_id,
            }
        )

    def _write_runtime_state(self) -> None:
        self.audit.write_runtime_state(
            {
                "cash_balance": self.cash_balance,
                "realized_pnl_usd": self.realized_pnl_usd,
                "wins": self.wins,
                "losses": self.losses,
                "closed_trades": self.closed_trades,
                "decisions_logged": self.decisions_logged,
                "paper_trades": self.paper_trades,
                "last_processed_trade_id": self.last_processed_trade_id,
                "open_positions": {
                    trade_id: position.to_payload() for trade_id, position in self.open_positions.items()
                },
                "recent_signals": self._recent_signals[:MAX_RECENT_ITEMS],
                "recent_trades": self._recent_trades[:MAX_RECENT_ITEMS],
            }
        )

    def _roll_daily_summary(self, now: datetime) -> None:
        day_key = now.strftime("%Y-%m-%d")
        existing = self.audit.paths["daily_summary_latest"]
        if existing.exists():
            try:
                last = json.loads(existing.read_text(encoding="utf-8"))
                if last.get("date") == day_key:
                    return
            except json.JSONDecodeError:
                pass
        self.audit.log_daily_summary(
            {
                "date": day_key,
                "decisions_logged": self.decisions_logged,
                "paper_trades": self.paper_trades,
                "open_positions": len(self.open_positions),
                "closed_trades": self.closed_trades,
                "wins": self.wins,
                "losses": self.losses,
                "win_rate": round(self.wins / self.closed_trades, 6) if self.closed_trades else 0.0,
                "realized_pnl_usd": round(self.realized_pnl_usd, 4),
                "cash_balance_usd": round(self.cash_balance, 4),
                "portfolio_value_usd": round(self._current_equity(), 4),
            }
        )

    def _refresh_last_prices(self, market_map: dict[str, Market]) -> None:
        for market in market_map.values():
            for outcome in market.outcomes:
                self._last_prices[outcome.token_id] = float(outcome.book_bid or outcome.price or 0.0)

    def _current_equity(self) -> float:
        return self.cash_balance + sum(
            position.shares * self._last_prices.get(position.token_id, position.entry_price)
            for position in self.open_positions.values()
        )

    def _open_position_keys(self) -> set[str]:
        return {
            self._position_key(condition_id=position.condition_id, token_id=position.token_id)
            for position in self.open_positions.values()
        }

    def _resolve_audit_root(self) -> Path:
        base = Path(self.profile.audit_root).resolve() if self.profile.audit_root else (LOG_DIR / "comparison" / self.profile.session_label)
        base.mkdir(parents=True, exist_ok=True)
        return base

    @staticmethod
    def _position_key(*, condition_id: str, token_id: str) -> str:
        return f"{condition_id}:{token_id}"

    @staticmethod
    def _as_float(value: Any) -> float | None:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _resolve_path(raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        if path.is_absolute():
            return path
        cwd_path = Path.cwd() / path
        if cwd_path.exists():
            return cwd_path
        return Path(__file__).resolve().parents[1] / path
