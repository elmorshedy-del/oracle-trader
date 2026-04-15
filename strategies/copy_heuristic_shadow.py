from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import CopyHeuristicShadowProfileConfig
from data.models import Event, Market
from engine.copy_heuristics import HeuristicEvaluation, build_decision_key, evaluate_trade
from engine.shadow_sleeve_audit import ShadowSleeveAudit
from engine.wallet_copy_store import WalletCopyResearchStore
from runtime_paths import LOG_DIR
from strategies.base import BaseStrategy


logger = logging.getLogger(__name__)

UTC = timezone.utc
MAX_RECENT_ITEMS = 40
HEURISTIC_DECISION_LOG_KEY = "decision_log"
HEURISTIC_TRADE_LEDGER_FIELDS = [
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
    "score",
]
HEURISTIC_DAILY_SUMMARY_FIELDS = [
    "logged_at",
    "date",
    "decisions_logged",
    "copied_trades",
    "open_positions",
    "resolved_trades",
    "wins",
    "losses",
    "win_rate",
    "realized_pnl_usd",
    "cash_balance_usd",
    "portfolio_value_usd",
]


@dataclass(slots=True)
class HeuristicPosition:
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
    score: float
    close_on_wallet_exit: bool

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "HeuristicPosition":
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
            score=float(payload.get("score") or 0.0),
            close_on_wallet_exit=bool(payload.get("close_on_wallet_exit", True)),
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
            "score": self.score,
            "close_on_wallet_exit": self.close_on_wallet_exit,
        }


class CopyHeuristicShadowStrategy(BaseStrategy):
    name = "copy_heuristic_shadow"
    description = "Paper sleeves that score real tracked-wallet trades with heuristic rules"

    def __init__(self, config, collector, store: WalletCopyResearchStore, profile: CopyHeuristicShadowProfileConfig):
        super().__init__(config)
        self.collector = collector
        self.store = store
        self.profile = profile
        self.enabled = True
        self.cash_balance = float(profile.budget_usd)
        self.open_positions: dict[str, HeuristicPosition] = {}
        self.realized_pnl_usd = 0.0
        self.wins = 0
        self.losses = 0
        self.resolved_trades = 0
        self.decisions_logged = 0
        self.copied_trades = 0
        self.last_processed_trade_id = 0
        self._last_prices: dict[str, float] = {}
        self._recent_signals: list[dict[str, Any]] = []
        self._recent_trades: list[dict[str, Any]] = []
        self.audit_root = self._resolve_audit_root()
        self.audit = ShadowSleeveAudit(
            self.audit_root,
            lane_key=self.profile.source,
            label=self.profile.label,
            category="copy_heuristic",
            source=self.profile.source,
            description=f"Wallet-copy heuristic paper sleeve: {self.profile.kind}.",
            trade_ledger_fields=HEURISTIC_TRADE_LEDGER_FIELDS,
            daily_summary_fields=HEURISTIC_DAILY_SUMMARY_FIELDS,
            extra_jsonl_keys=(HEURISTIC_DECISION_LOG_KEY,),
        )
        self.audit.write_metadata(
            {
                "started_at": datetime.now(UTC).isoformat(),
                "strategy": self.profile.strategy_key,
                "view_key": self.profile.view_key,
                "kind": self.profile.kind,
                "budget_usd": self.profile.budget_usd,
                "score_threshold": self.profile.score_threshold,
                "max_open_positions": self.profile.max_open_positions,
                "scan_trade_limit": self.profile.scan_trade_limit,
            },
            extra={"view_key": self.profile.view_key, "session_label": self.profile.session_label},
        )
        self._stats.update(
            {
                "view_key": self.profile.view_key,
                "source": self.profile.source,
                "kind": self.profile.kind,
                "budget_usd": self.profile.budget_usd,
                "max_open_positions": self.profile.max_open_positions,
                "cash_balance_usd": self.cash_balance,
                "portfolio_value_usd": self.cash_balance,
                "decisions_logged": 0,
                "copied_trades": 0,
                "resolved_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "realized_pnl_usd": 0.0,
                "last_processed_trade_id": 0,
                "runtime_state_path": str(self.audit.paths["runtime_state"]),
                "trade_ledger_csv": str(self.audit.paths["trade_ledger_csv"]),
            }
        )
        self._load_runtime_state()
        if self.last_processed_trade_id <= 0:
            self.last_processed_trade_id = self.store.latest_wallet_trade_id()
        self._write_runtime_state()

    async def scan(self, markets: list[Market], events: list[Event]) -> list:
        del events
        self._stats["scans_completed"] += 1
        now = datetime.now(UTC)
        market_map = {market.condition_id: market for market in markets}
        self._refresh_last_prices(market_map)
        await self._resolve_open_positions(now, market_map)
        await self._process_new_trades(now)
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
                "total_trades": self.resolved_trades,
                "win_rate": round((self.wins / self.resolved_trades) * 100.0, 1) if self.resolved_trades else 0.0,
                "max_drawdown": 0.0,
                "positions": positions,
            },
            "signals": list(self._recent_signals[:MAX_RECENT_ITEMS]),
            "trades": list(self._recent_trades[:MAX_RECENT_ITEMS]),
            "performance": {
                "total_pnl": round(total_value - self.profile.budget_usd, 2),
                "win_rate": (self.wins / self.resolved_trades) if self.resolved_trades else 0.0,
                "total_trades": self.resolved_trades,
                "cash": round(self.cash_balance, 2),
                "budget": round(self.profile.budget_usd, 2),
            },
        }

    async def close(self) -> None:
        self._write_runtime_state()

    def reset_state(self) -> None:
        self.cash_balance = float(self.profile.budget_usd)
        self.open_positions = {}
        self.realized_pnl_usd = 0.0
        self.wins = 0
        self.losses = 0
        self.resolved_trades = 0
        self.decisions_logged = 0
        self.copied_trades = 0
        self._recent_signals = []
        self._recent_trades = []
        self.last_processed_trade_id = self.store.latest_wallet_trade_id()
        self._stats.update(
            {
                "cash_balance_usd": self.cash_balance,
                "portfolio_value_usd": self.cash_balance,
                "decisions_logged": 0,
                "copied_trades": 0,
                "resolved_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "realized_pnl_usd": 0.0,
                "last_processed_trade_id": self.last_processed_trade_id,
            }
        )
        self._write_runtime_state()

    async def _process_new_trades(self, now: datetime) -> None:
        rows = self.store.list_wallet_trades_after_id(
            last_id=self.last_processed_trade_id,
            limit=self.profile.scan_trade_limit,
        )
        for row in rows:
            trade_id = int(row["id"])
            self.last_processed_trade_id = max(self.last_processed_trade_id, trade_id)
            if not self._should_consider_trade(row):
                continue
            evaluation = self._evaluate_row(row)
            if evaluation is None:
                continue
            if evaluation.should_copy:
                blocked_reason = self._paper_trade_blocker(row, evaluation)
                if blocked_reason:
                    evaluation.should_copy = False
                    evaluation.reasons.append(blocked_reason)
                    evaluation.suggested_size_usd = 0.0
            decision_payload = self._decision_payload(row=row, evaluation=evaluation, now=now)
            decision_id = self.store.insert_copy_decision(decision_payload)
            self.decisions_logged += 1
            self._stats["decisions_logged"] = self.decisions_logged
            self.audit.log_extra(HEURISTIC_DECISION_LOG_KEY, {**decision_payload, "decision_id": decision_id})
            self._remember_signal(decision_payload, now)
            if evaluation.should_copy:
                self._open_position(
                    decision_id=decision_id,
                    row=row,
                    evaluation=evaluation,
                    now=now,
                )

    def _evaluate_row(self, row: dict[str, Any]) -> HeuristicEvaluation | None:
        wallet_perf = self.store.observed_wallet_performance_before(
            wallet_address=str(row.get("wallet_address") or ""),
            before_timestamp=float(row.get("timestamp") or 0.0),
        )
        consensus_rows: list[dict[str, Any]] = []
        consensus_avg_win_rate = None
        if self.profile.kind == "whale_consensus":
            since_ts = float(row.get("timestamp") or 0.0) - self.profile.consensus_window_seconds
            consensus_rows = self.store.recent_same_side_trades(
                condition_id=str(row.get("market_condition_id") or ""),
                asset_token_id=str(row.get("asset_token_id") or ""),
                outcome=str(row.get("outcome") or ""),
                since_timestamp=since_ts,
                exclude_trade_id=int(row["id"]),
                limit=50,
            )
            win_rates: list[float] = []
            seen_wallets: set[str] = set()
            for peer in [row, *consensus_rows]:
                wallet_address = str(peer.get("wallet_address") or "")
                if not wallet_address or wallet_address in seen_wallets:
                    continue
                seen_wallets.add(wallet_address)
                peer_perf = self.store.observed_wallet_performance_before(
                    wallet_address=wallet_address,
                    before_timestamp=float(row.get("timestamp") or 0.0),
                )
                peer_rate = self._effective_peer_win_rate(peer, peer_perf)
                if peer_rate is not None:
                    peer["effective_wallet_win_rate"] = peer_rate
                    win_rates.append(peer_rate)
            if win_rates:
                consensus_avg_win_rate = sum(win_rates) / len(win_rates)
        market_first_seen_timestamp = None
        if self.profile.kind == "fresh_market":
            market_first_seen_timestamp = self.store.market_first_seen_timestamp(
                condition_id=str(row.get("market_condition_id") or "")
            )
        prior_buy_row = None
        if self.profile.kind == "contrarian_exit":
            prior_buy_row = self.store.find_prior_buy(
                wallet_address=str(row.get("wallet_address") or ""),
                condition_id=str(row.get("market_condition_id") or ""),
                asset_token_id=str(row.get("asset_token_id") or ""),
                before_timestamp=float(row.get("timestamp") or 0.0),
            )
        return evaluate_trade(
            profile=self.profile,
            trade_row=row,
            wallet_performance=wallet_perf,
            consensus_rows=consensus_rows,
            consensus_avg_win_rate=consensus_avg_win_rate,
            market_first_seen_timestamp=market_first_seen_timestamp,
            prior_buy_row=prior_buy_row,
            available_cash_usd=self.cash_balance,
        )

    async def _resolve_open_positions(self, now: datetime, market_map: dict[str, Market]) -> None:
        if not self.open_positions:
            return
        for position in list(self.open_positions.values()):
            source_row = self.store.get_wallet_trade(trade_id=position.source_trade_id)
            if source_row is None:
                continue
            if position.close_on_wallet_exit and source_row.get("wallet_sell_timestamp") and source_row.get("wallet_sell_price"):
                self._close_position(
                    position,
                    exit_price=float(source_row["wallet_sell_price"]),
                    close_reason="wallet_sell",
                    now=now,
                )
                continue
            market = market_map.get(position.condition_id)
            if market is None:
                market = await self.collector.get_market_by_slug(position.market_slug)
            if market is None:
                continue
            settlement_price = self._settlement_price(market, position.token_id)
            if settlement_price is None:
                continue
            self._close_position(
                position,
                exit_price=settlement_price,
                close_reason="resolved",
                now=now,
            )

    def _open_position(
        self,
        *,
        decision_id: int,
        row: dict[str, Any],
        evaluation: HeuristicEvaluation,
        now: datetime,
    ) -> None:
        entry_price = self._entry_price_for_trade(row, evaluation)
        if entry_price <= 0:
            return
        size_usd = float(evaluation.suggested_size_usd)
        if size_usd <= 0 or size_usd > self.cash_balance:
            return
        shares = size_usd / entry_price
        trade_id = f"{self.profile.strategy_key}:{decision_id}"
        position = HeuristicPosition(
            trade_id=trade_id,
            decision_id=decision_id,
            source_trade_id=int(row["id"]),
            wallet_address=str(row.get("wallet_address") or ""),
            condition_id=str(row.get("market_condition_id") or ""),
            market_slug=str(row.get("market_slug") or ""),
            token_id=str(evaluation.position_token_id or row.get("asset_token_id") or ""),
            outcome_label=str(evaluation.position_outcome or row.get("outcome") or ""),
            shares=shares,
            entry_price=entry_price,
            entry_size_usd=size_usd,
            entry_timestamp=now,
            score=float(evaluation.score),
            close_on_wallet_exit=self.profile.kind != "contrarian_exit",
        )
        self.open_positions[trade_id] = position
        self.cash_balance -= size_usd
        self.copied_trades += 1
        self._stats["copied_trades"] = self.copied_trades
        self._stats["last_entry_at"] = now.isoformat()
        self.store.mark_copy_decision_executed(
            decision_id=decision_id,
            paper_trade_id=trade_id,
            paper_side=str(evaluation.position_side or evaluation.position_outcome or ""),
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
                "score": round(position.score, 3),
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
                "score": round(position.score, 6),
            }
        )

    def _close_position(
        self,
        position: HeuristicPosition,
        *,
        exit_price: float,
        close_reason: str,
        now: datetime,
    ) -> None:
        current_value = position.shares * exit_price
        realized_pnl = current_value - position.entry_size_usd
        self.cash_balance += current_value
        self.realized_pnl_usd += realized_pnl
        self.resolved_trades += 1
        if realized_pnl >= 0:
            self.wins += 1
        else:
            self.losses += 1
        self._stats["resolved_trades"] = self.resolved_trades
        self._stats["wins"] = self.wins
        self._stats["losses"] = self.losses
        self._stats["win_rate"] = self.wins / self.resolved_trades if self.resolved_trades else 0.0
        self._stats["realized_pnl_usd"] = round(self.realized_pnl_usd, 4)
        self._stats["last_resolution_at"] = now.isoformat()
        self.store.close_copy_decision(
            decision_id=position.decision_id,
            exit_timestamp=now.timestamp(),
            exit_price=exit_price,
            realized_pnl_usd=realized_pnl,
            close_reason=close_reason,
        )
        self.audit.log_trade_ledger(
            {
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
                "score": round(position.score, 6),
            }
        )
        self.audit.log_trade_event(
            {
                "event_type": "exit",
                "trade_id": position.trade_id,
                "decision_id": position.decision_id,
                "wallet_address": position.wallet_address,
                "market_slug": position.market_slug,
                "outcome": position.outcome_label,
                "exit_price": exit_price,
                "realized_pnl_usd": round(realized_pnl, 4),
                "close_reason": close_reason,
                "score": round(position.score, 6),
            }
        )
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
                "score": round(position.score, 3),
                "event": "exit",
            },
        )
        self._recent_trades = self._recent_trades[:MAX_RECENT_ITEMS]
        self.open_positions.pop(position.trade_id, None)

    def _decision_payload(
        self,
        *,
        row: dict[str, Any],
        evaluation: HeuristicEvaluation,
        now: datetime,
    ) -> dict[str, Any]:
        return {
            "decision_key": build_decision_key(strategy_key=self.profile.strategy_key, trade_id=int(row["id"])),
            "strategy_key": self.profile.strategy_key,
            "profile_kind": self.profile.kind,
            "wallet_trade_id": int(row["id"]),
            "wallet_trade_key": str(row.get("trade_key") or ""),
            "tx_hash": str(row.get("tx_hash") or ""),
            "wallet_address": str(row.get("wallet_address") or ""),
            "market_condition_id": str(row.get("market_condition_id") or ""),
            "asset_token_id": str(evaluation.position_token_id or row.get("asset_token_id") or ""),
            "market_slug": str(row.get("market_slug") or ""),
            "source_side": str(row.get("side") or ""),
            "source_outcome": str(evaluation.position_outcome or row.get("outcome") or ""),
            "score": float(round(evaluation.score, 6)),
            "should_copy": 1 if evaluation.should_copy else 0,
            "suggested_size_usd": float(evaluation.suggested_size_usd or 0.0),
            "reasons_json": evaluation.reasons,
            "context_json": evaluation.context,
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
        reasoning = ", ".join(decision_payload["reasons_json"]) if decision_payload["reasons_json"] else "no_match"
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
                "edge": 0.0,
                "size": round(float(decision_payload["suggested_size_usd"] or 0.0), 2),
                "whale": True,
                "reasoning": reasoning,
                "wallet": decision_payload["wallet_address"],
                "outcome": decision_payload["source_outcome"],
                "score": round(float(decision_payload["score"]), 3),
                "copy": bool(decision_payload["should_copy"]),
            },
        )
        self._recent_signals = self._recent_signals[:MAX_RECENT_ITEMS]

    def _paper_trade_blocker(self, row: dict[str, Any], evaluation: HeuristicEvaluation) -> str | None:
        if row.get("market_resolved"):
            return "market_already_resolved"
        if self.profile.kind != "contrarian_exit" and row.get("wallet_sell_timestamp"):
            return "wallet_already_exited"
        position_key = self._position_key(
            condition_id=str(row.get("market_condition_id") or ""),
            token_id=str(evaluation.position_token_id or row.get("asset_token_id") or ""),
        )
        if position_key in self._open_position_keys():
            return "already_open_position"
        if len(self.open_positions) >= self.profile.max_open_positions:
            return "max_open_positions_reached"
        return None

    def _entry_price_for_trade(self, row: dict[str, Any], evaluation: HeuristicEvaluation) -> float:
        outcome = str(evaluation.position_outcome or row.get("outcome") or "").strip().lower()
        if outcome == "yes":
            return float(row.get("market_yes_ask") or row.get("token_best_ask") or row.get("price") or 0.0)
        if outcome == "no":
            return float(row.get("market_no_ask") or row.get("token_best_ask") or row.get("price") or 0.0)
        return float(row.get("token_best_ask") or row.get("price") or 0.0)

    def _settlement_price(self, market: Market, token_id: str) -> float | None:
        if market.active and not market.closed:
            return None
        for outcome in market.outcomes:
            if outcome.token_id == token_id:
                return float(outcome.price or 0.0)
        return None

    def _effective_peer_win_rate(self, row: dict[str, Any], peer_perf: dict[str, Any]) -> float | None:
        observed = peer_perf.get("win_rate")
        if observed is not None and int(peer_perf.get("labeled_trades") or 0) >= self.profile.min_wallet_labeled_trades:
            return float(observed)
        leaderboard = row.get("wallet_leaderboard_win_rate")
        if leaderboard is None:
            return None
        leaderboard_value = float(leaderboard)
        if leaderboard_value > 1.0 and leaderboard_value <= 100.0:
            leaderboard_value /= 100.0
        return leaderboard_value

    def _should_consider_trade(self, row: dict[str, Any]) -> bool:
        side = str(row.get("side") or "").upper()
        if self.profile.kind == "contrarian_exit":
            return side == "SELL"
        return side == "BUY"

    def _open_position_keys(self) -> set[str]:
        return {
            self._position_key(condition_id=position.condition_id, token_id=position.token_id)
            for position in self.open_positions.values()
        }

    @staticmethod
    def _position_key(*, condition_id: str, token_id: str) -> str:
        return f"{condition_id}:{token_id}"

    def _current_equity(self) -> float:
        positions_value = 0.0
        for position in self.open_positions.values():
            positions_value += position.shares * self._last_prices.get(position.token_id, position.entry_price)
        return self.cash_balance + positions_value

    def _refresh_last_prices(self, market_map: dict[str, Market]) -> None:
        for market in market_map.values():
            for outcome in market.outcomes:
                self._last_prices[outcome.token_id] = float(outcome.price or 0.0)

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
                "copied_trades": self.copied_trades,
                "open_positions": len(self.open_positions),
                "resolved_trades": self.resolved_trades,
                "wins": self.wins,
                "losses": self.losses,
                "win_rate": round(self.wins / self.resolved_trades, 6) if self.resolved_trades else 0.0,
                "realized_pnl_usd": round(self.realized_pnl_usd, 4),
                "cash_balance_usd": round(self.cash_balance, 4),
                "portfolio_value_usd": round(self._current_equity(), 4),
            }
        )

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
        self.resolved_trades = int(payload.get("resolved_trades") or 0)
        self.decisions_logged = int(payload.get("decisions_logged") or 0)
        self.copied_trades = int(payload.get("copied_trades") or 0)
        self.last_processed_trade_id = int(payload.get("last_processed_trade_id") or 0)
        self.open_positions = {
            trade_id: HeuristicPosition.from_payload(row)
            for trade_id, row in (payload.get("open_positions") or {}).items()
        }
        self._recent_signals = list(payload.get("recent_signals") or [])[:MAX_RECENT_ITEMS]
        self._recent_trades = list(payload.get("recent_trades") or [])[:MAX_RECENT_ITEMS]
        self._stats.update(
            {
                "cash_balance_usd": round(self.cash_balance, 4),
                "portfolio_value_usd": round(self._current_equity(), 4),
                "decisions_logged": self.decisions_logged,
                "copied_trades": self.copied_trades,
                "resolved_trades": self.resolved_trades,
                "wins": self.wins,
                "losses": self.losses,
                "win_rate": self.wins / self.resolved_trades if self.resolved_trades else 0.0,
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
                "resolved_trades": self.resolved_trades,
                "decisions_logged": self.decisions_logged,
                "copied_trades": self.copied_trades,
                "last_processed_trade_id": self.last_processed_trade_id,
                "open_positions": {
                    trade_id: position.to_payload() for trade_id, position in self.open_positions.items()
                },
                "recent_signals": self._recent_signals[:MAX_RECENT_ITEMS],
                "recent_trades": self._recent_trades[:MAX_RECENT_ITEMS],
            }
        )

    def _resolve_audit_root(self) -> Path:
        base = Path(self.profile.audit_root).resolve() if self.profile.audit_root else (LOG_DIR / "comparison" / self.profile.session_label)
        base.mkdir(parents=True, exist_ok=True)
        return base
