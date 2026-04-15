from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from config import CopyTraderShadowConfig
from data.models import Event, Market
from engine.shadow_sleeve_audit import ShadowSleeveAudit
from runtime_paths import LOG_DIR
from strategies.base import BaseStrategy


logger = logging.getLogger(__name__)

UTC = timezone.utc
MAX_RECENT_ITEMS = 40
MAX_SEEN_TRADE_KEYS = 5000
COPY_TRADER_BLOCKED_MAX_OPEN_POSITIONS = "max_open_positions_reached"
COPY_TRADER_TRADE_LEDGER_FIELDS = [
    "logged_at",
    "trade_id",
    "wallet_address",
    "wallet_name",
    "tx_hash",
    "condition_id",
    "market_slug",
    "outcome",
    "entry_timestamp",
    "entry_price",
    "entry_size_usd",
    "exit_timestamp",
    "exit_price",
    "realized_pnl_usd",
    "close_reason",
]
COPY_TRADER_DAILY_SUMMARY_FIELDS = [
    "logged_at",
    "date",
    "tracked_wallets",
    "open_positions",
    "resolved_trades",
    "wins",
    "losses",
    "win_rate",
    "realized_pnl_usd",
    "cash_balance_usd",
    "portfolio_value_usd",
]
COPY_TRADER_CANDIDATE_BUY_KEY = "candidate_buys"
COPY_TRADER_WALLET_SELL_EVENT_KEY = "wallet_sell_events"
COPY_TRADER_CANDIDATE_LABEL_KEY = "candidate_labels"


@dataclass(slots=True)
class CopyTradeWallet:
    address: str
    name: str
    pnl_usd: float
    rank: int = 0


@dataclass(slots=True)
class CopyTradePosition:
    trade_id: str
    wallet_address: str
    wallet_name: str
    tx_hash: str
    condition_id: str
    market_slug: str
    token_id: str
    outcome_label: str
    shares: float
    entry_price: float
    entry_size_usd: float
    entry_timestamp: datetime

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "CopyTradePosition":
        return cls(
            trade_id=str(payload["trade_id"]),
            wallet_address=str(payload["wallet_address"]),
            wallet_name=str(payload["wallet_name"]),
            tx_hash=str(payload["tx_hash"]),
            condition_id=str(payload["condition_id"]),
            market_slug=str(payload["market_slug"]),
            token_id=str(payload["token_id"]),
            outcome_label=str(payload["outcome_label"]),
            shares=float(payload["shares"]),
            entry_price=float(payload["entry_price"]),
            entry_size_usd=float(payload["entry_size_usd"]),
            entry_timestamp=datetime.fromisoformat(str(payload["entry_timestamp"])).astimezone(UTC),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "wallet_address": self.wallet_address,
            "wallet_name": self.wallet_name,
            "tx_hash": self.tx_hash,
            "condition_id": self.condition_id,
            "market_slug": self.market_slug,
            "token_id": self.token_id,
            "outcome_label": self.outcome_label,
            "shares": self.shares,
            "entry_price": self.entry_price,
            "entry_size_usd": self.entry_size_usd,
            "entry_timestamp": self.entry_timestamp.isoformat(),
        }


class CopyTraderShadowStrategy(BaseStrategy):
    name = "copy_trader_shadow"
    description = "Paper sleeve that mirrors top-wallet Polymarket trades"

    def __init__(self, config, collector):
        super().__init__(config)
        self.cfg: CopyTraderShadowConfig = config.copy_trader_shadow
        self.collector = collector
        self.enabled = bool(self.cfg.enabled)
        self.cash_balance = float(self.cfg.budget_usd)
        self.open_positions: dict[str, CopyTradePosition] = {}
        self.target_wallets: list[CopyTradeWallet] = []
        self.seen_trade_keys: list[str] = []
        self._seen_trade_key_set: set[str] = set()
        self._last_prices: dict[str, float] = {}
        self._recent_signals: list[dict[str, Any]] = []
        self._recent_trades: list[dict[str, Any]] = []
        self._last_wallet_refresh_at: datetime | None = None
        self._last_activity_refresh_at: datetime | None = None
        self.realized_pnl_usd = 0.0
        self.wins = 0
        self.losses = 0
        self.resolved_trades = 0
        self.audit_root = self._resolve_audit_root()
        self.audit = ShadowSleeveAudit(
            self.audit_root,
            lane_key=self.cfg.source,
            label=self.cfg.label,
            category="copy_trader",
            source=self.cfg.source,
            description="Top-wallet copy trading paper sleeve logs.",
            trade_ledger_fields=COPY_TRADER_TRADE_LEDGER_FIELDS,
            daily_summary_fields=COPY_TRADER_DAILY_SUMMARY_FIELDS,
            extra_jsonl_keys=(
                COPY_TRADER_CANDIDATE_BUY_KEY,
                COPY_TRADER_WALLET_SELL_EVENT_KEY,
                COPY_TRADER_CANDIDATE_LABEL_KEY,
            ),
        )
        self.audit.write_metadata(
            {
                "started_at": datetime.now(UTC).isoformat(),
                "strategy": self.name,
                "view_key": self.cfg.view_key,
                "budget_usd": self.cfg.budget_usd,
                "top_wallets": self.cfg.top_wallets,
                "min_wallet_pnl_usd": self.cfg.min_wallet_pnl_usd,
                "copy_size_multiplier": self.cfg.copy_size_multiplier,
                "min_trade_usd": self.cfg.min_trade_usd,
                "max_trade_usd": self.cfg.max_trade_usd,
                "max_entry_price": self.cfg.max_entry_price,
                "max_open_positions": self.cfg.max_open_positions,
                "tracked_wallets": self.cfg.tracked_wallets,
            },
            extra={"view_key": self.cfg.view_key, "session_label": self.cfg.session_label},
        )
        self._stats.update(
            {
                "view_key": self.cfg.view_key,
                "source": self.cfg.source,
                "budget_usd": self.cfg.budget_usd,
                "max_open_positions": self.cfg.max_open_positions,
                "cash_balance_usd": self.cash_balance,
                "portfolio_value_usd": self.cash_balance,
                "target_wallets": 0,
                "activity_events_seen": 0,
                "entries": 0,
                "resolved_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "realized_pnl_usd": 0.0,
                "last_wallet_refresh_at": None,
                "last_activity_refresh_at": None,
                "last_entry_at": None,
                "last_resolution_at": None,
                "runtime_state_path": str(self.audit.paths["runtime_state"]),
                "trade_ledger_csv": str(self.audit.paths["trade_ledger_csv"]),
            }
        )
        self._load_runtime_state()
        self._write_runtime_state()

    async def scan(self, markets: list[Market], events: list[Event]) -> list:
        del events
        self._stats["scans_completed"] += 1
        now = datetime.now(UTC)
        market_map = {market.condition_id: market for market in markets}
        self._refresh_last_prices(market_map)
        await self._resolve_positions(now)
        await self._refresh_wallets_if_needed(now)
        await self._process_activity(now, market_map)
        self._roll_daily_summary(now)
        self._stats["cash_balance_usd"] = round(self.cash_balance, 4)
        self._stats["portfolio_value_usd"] = round(self._current_equity(), 4)
        self._write_runtime_state()
        return []

    @property
    def stats(self) -> dict:
        return {"name": self.name, "enabled": self.enabled, **self._stats}

    def serialize_view(self) -> dict[str, Any]:
        positions = []
        current_value = 0.0
        for position in self.open_positions.values():
            mark = self._last_prices.get(position.token_id, position.entry_price)
            current_position_value = mark * position.shares
            current_value += current_position_value
            positions.append(
                {
                    "market": position.market_slug,
                    "side": position.outcome_label,
                    "shares": round(position.shares, 2),
                    "entry": round(position.entry_price, 3),
                    "current": round(mark, 3),
                    "pnl": round(current_position_value - position.entry_size_usd, 2),
                    "source": self.cfg.source,
                }
            )
        total_value = self.cash_balance + current_value
        return {
            "key": self.cfg.view_key,
            "label": self.cfg.label,
            "source": self.cfg.source,
            "portfolio": {
                "starting_capital": round(self.cfg.budget_usd, 2),
                "total_value": round(total_value, 2),
                "cash": round(self.cash_balance, 2),
                "positions_value": round(current_value, 2),
                "total_pnl": round(total_value - self.cfg.budget_usd, 2),
                "total_pnl_pct": round(((total_value / self.cfg.budget_usd) - 1.0) * 100.0, 2) if self.cfg.budget_usd else 0.0,
                "total_trades": self.resolved_trades,
                "win_rate": round((self.wins / self.resolved_trades) * 100.0, 1) if self.resolved_trades else 0.0,
                "max_drawdown": 0.0,
                "positions": positions,
            },
            "signals": list(self._recent_signals[:MAX_RECENT_ITEMS]),
            "trades": list(self._recent_trades[:MAX_RECENT_ITEMS]),
            "performance": {
                "total_pnl": round(total_value - self.cfg.budget_usd, 2),
                "win_rate": (self.wins / self.resolved_trades) if self.resolved_trades else 0.0,
                "total_trades": self.resolved_trades,
                "cash": round(self.cash_balance, 2),
                "budget": round(self.cfg.budget_usd, 2),
            },
        }

    async def close(self) -> None:
        self._write_runtime_state()

    def reset_state(self) -> None:
        self.cash_balance = float(self.cfg.budget_usd)
        self.open_positions = {}
        self.seen_trade_keys = []
        self._seen_trade_key_set = set()
        self._recent_signals = []
        self._recent_trades = []
        self.realized_pnl_usd = 0.0
        self.wins = 0
        self.losses = 0
        self.resolved_trades = 0
        self._stats.update(
            {
                "entries": 0,
                "resolved_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "realized_pnl_usd": 0.0,
            }
        )
        self._write_runtime_state()

    async def _refresh_wallets_if_needed(self, now: datetime) -> None:
        if self.cfg.tracked_wallets:
            self.target_wallets = [
                CopyTradeWallet(address=address, name=address[:10], pnl_usd=0.0, rank=index + 1)
                for index, address in enumerate(self.cfg.tracked_wallets)
            ]
            self._stats["target_wallets"] = len(self.target_wallets)
            return

        ttl = timedelta(minutes=self.cfg.leaderboard_refresh_minutes)
        if self._last_wallet_refresh_at and now - self._last_wallet_refresh_at < ttl:
            return

        leaderboard = await self.collector.get_leaderboard(limit=self.cfg.leaderboard_limit)
        wallets: list[CopyTradeWallet] = []
        for row in leaderboard:
            address = str(row.get("proxyWallet") or row.get("address") or row.get("wallet") or row.get("user") or "")
            if not address:
                continue
            pnl_usd = float(row.get("pnl") or row.get("cashPnl") or row.get("totalPnl") or 0.0)
            if pnl_usd < self.cfg.min_wallet_pnl_usd:
                continue
            name = str(row.get("userName") or row.get("name") or address[:10])
            wallets.append(CopyTradeWallet(address=address, name=name, pnl_usd=pnl_usd, rank=len(wallets) + 1))
            if len(wallets) >= self.cfg.top_wallets:
                break
        self.target_wallets = wallets
        self._last_wallet_refresh_at = now
        self._stats["target_wallets"] = len(wallets)
        self._stats["last_wallet_refresh_at"] = now.isoformat()

    async def _process_activity(self, now: datetime, market_map: dict[str, Market]) -> None:
        if not self.target_wallets:
            return
        if self._last_activity_refresh_at and (now - self._last_activity_refresh_at).total_seconds() < self.cfg.activity_refresh_seconds:
            return

        semaphore = asyncio.Semaphore(4)

        async def fetch(wallet: CopyTradeWallet):
            async with semaphore:
                return wallet, await self.collector.get_wallet_activity(wallet.address, limit=self.cfg.activity_trades_per_wallet)

        results = await asyncio.gather(*(fetch(wallet) for wallet in self.target_wallets), return_exceptions=True)
        rows: list[tuple[CopyTradeWallet, dict[str, Any]]] = []
        for result in results:
            if isinstance(result, Exception):
                continue
            wallet, activity = result
            for row in activity or []:
                rows.append((wallet, row))
        rows.sort(key=lambda item: int(item[1].get("timestamp") or 0))

        for wallet, row in rows:
            trade_key = self._build_trade_key(wallet.address, row)
            if trade_key in self._seen_trade_key_set:
                continue
            self._remember_trade_key(trade_key)
            if str(row.get("type") or "").upper() != "TRADE":
                continue
            condition_id = str(row.get("conditionId") or "")
            token_id = str(row.get("asset") or "")
            market_slug = str(row.get("slug") or "")
            if not condition_id or not token_id or not market_slug:
                continue
            wallet_side = str(row.get("side") or "").upper()
            market = market_map.get(condition_id)
            if market is None:
                market = await self.collector.get_market_by_slug(market_slug)
            if market is None:
                continue
            if wallet_side == "BUY":
                candidate = self._evaluate_buy_candidate(wallet, row, market, now)
                self.audit.log_extra(COPY_TRADER_CANDIDATE_BUY_KEY, candidate)
                if candidate["would_current_rules_take"]:
                    self._open_position_from_candidate(candidate, now)
            elif wallet_side == "SELL":
                self.audit.log_extra(
                    COPY_TRADER_WALLET_SELL_EVENT_KEY,
                    self._build_wallet_sell_event(wallet, row, market, now),
                )
                self._handle_sell(wallet, row, market, now)

        self._last_activity_refresh_at = now
        self._stats["last_activity_refresh_at"] = now.isoformat()

    def _evaluate_buy_candidate(self, wallet: CopyTradeWallet, row: dict[str, Any], market: Market, now: datetime) -> dict[str, Any]:
        wallet_usd = float(row.get("usdcSize") or row.get("size") or 0.0)
        token_id = str(row.get("asset") or "")
        outcome_label = str(row.get("outcome") or self._outcome_label(market, token_id))
        entry_price = float(row.get("price") or self._token_price(market, token_id) or 0.0)
        size_usd = min(wallet_usd * self.cfg.copy_size_multiplier, self.cfg.max_trade_usd, self.cash_balance)
        activity_at = self._parse_timestamp(row.get("timestamp"), now)
        trade_id = f"copy:{wallet.address[-6:]}:{row.get('transactionHash') or row.get('timestamp')}"
        token_price = self._token_price(market, token_id)
        token_bid, token_ask = self._token_book(market, token_id)
        blocked_reason = "eligible"
        if wallet_usd < self.cfg.min_trade_usd:
            blocked_reason = "below_min_wallet_trade_usd"
        elif not token_id:
            blocked_reason = "missing_token_id"
        elif entry_price <= 0:
            blocked_reason = "invalid_entry_price"
        elif entry_price > self.cfg.max_entry_price:
            blocked_reason = "above_max_entry_price"
        elif len(self.open_positions) >= self.cfg.max_open_positions:
            blocked_reason = COPY_TRADER_BLOCKED_MAX_OPEN_POSITIONS
        elif size_usd < self.cfg.min_trade_usd:
            blocked_reason = "sized_below_min_trade_usd"
        time_to_close_hours = self._time_to_close_hours(market, now)
        return {
            "candidate_id": self._build_candidate_id(wallet.address, row),
            "observed_at": now.isoformat(),
            "activity_timestamp": activity_at.isoformat(),
            "wallet_address": wallet.address,
            "wallet_name": wallet.name,
            "wallet_rank": wallet.rank,
            "wallet_pnl_usd": round(wallet.pnl_usd, 4),
            "tx_hash": str(row.get("transactionHash") or ""),
            "condition_id": str(row.get("conditionId") or ""),
            "market_slug": market.slug,
            "market_question": market.question,
            "market_primary_tag": market.tags[0] if market.tags else "",
            "market_tag_count": len(market.tags),
            "market_volume_24h": round(float(market.volume_24h or 0.0), 4),
            "market_volume_total": round(float(market.volume_total or 0.0), 4),
            "market_liquidity": round(float(market.liquidity or 0.0), 4),
            "market_spread": round(float(market.spread or 0.0), 6),
            "market_midpoint": round(float(market.midpoint or 0.0), 6),
            "market_reward_pool": round(float(market.reward_pool or 0.0), 4),
            "market_active": bool(market.active),
            "market_closed": bool(market.closed),
            "time_to_close_hours": round(time_to_close_hours, 4) if time_to_close_hours is not None else None,
            "token_id": token_id,
            "outcome": outcome_label,
            "wallet_side": "BUY",
            "wallet_trade_usd": round(wallet_usd, 4),
            "wallet_trade_price": round(entry_price, 6),
            "token_price": round(float(token_price or 0.0), 6),
            "token_book_bid": round(float(token_bid or 0.0), 6) if token_bid is not None else None,
            "token_book_ask": round(float(token_ask or 0.0), 6) if token_ask is not None else None,
            "copy_size_multiplier": self.cfg.copy_size_multiplier,
            "available_cash_usd": round(self.cash_balance, 4),
            "copy_size_usd": round(size_usd, 4),
            "would_current_rules_take": blocked_reason == "eligible",
            "blocked_reason": blocked_reason,
            "copied_trade_id": trade_id if blocked_reason == "eligible" else None,
        }

    def _open_position_from_candidate(self, candidate: dict[str, Any], now: datetime) -> None:
        if len(self.open_positions) >= self.cfg.max_open_positions:
            return
        trade_id = str(candidate["copied_trade_id"])
        entry_price = float(candidate["wallet_trade_price"])
        size_usd = float(candidate["copy_size_usd"])
        shares = size_usd / entry_price
        position = CopyTradePosition(
            trade_id=trade_id,
            wallet_address=str(candidate["wallet_address"]),
            wallet_name=str(candidate["wallet_name"]),
            tx_hash=str(candidate["tx_hash"]),
            condition_id=str(candidate["condition_id"]),
            market_slug=str(candidate["market_slug"]),
            token_id=str(candidate["token_id"]),
            outcome_label=str(candidate["outcome"]),
            shares=shares,
            entry_price=entry_price,
            entry_size_usd=size_usd,
            entry_timestamp=datetime.fromisoformat(str(candidate["activity_timestamp"])).astimezone(UTC),
        )
        self.open_positions[trade_id] = position
        self.cash_balance -= size_usd
        self._stats["entries"] = int(self._stats.get("entries") or 0) + 1
        self._stats["last_entry_at"] = now.isoformat()
        signal_view = {
            "id": str(candidate["copied_trade_id"]),
            "time": now.isoformat(),
            "timestamp": now.isoformat(),
            "source": self.cfg.source,
            "action": "copy_buy",
            "market": str(candidate["market_slug"]),
            "confidence": 0.5,
            "edge": 0.0,
            "size": round(size_usd, 2),
            "whale": True,
            "reasoning": f"Copied {candidate['wallet_name']} into {candidate['outcome']} at ${entry_price:.3f}",
            "wallet": str(candidate["wallet_name"]),
            "outcome": str(candidate["outcome"]),
            "price": round(entry_price, 4),
            "size_usd": round(size_usd, 2),
        }
        self._recent_signals.insert(0, signal_view)
        self._recent_signals = self._recent_signals[:MAX_RECENT_ITEMS]
        self._recent_trades.insert(
            0,
            {
                "id": str(candidate["copied_trade_id"]),
                "time": now.isoformat(),
                "timestamp": now.isoformat(),
                "source": self.cfg.source,
                "market": str(candidate["market_slug"]),
                "side": str(candidate["outcome"]),
                "price": round(entry_price, 4),
                "usd": round(size_usd, 2),
                "pnl": None,
                "wallet": str(candidate["wallet_name"]),
                "event": "entry",
            },
        )
        self._recent_trades = self._recent_trades[:MAX_RECENT_ITEMS]
        self.audit.log_signal(signal_view)
        self.audit.log_trade_event(
            {
                "event_type": "entry",
                "trade_id": str(candidate["copied_trade_id"]),
                "wallet_address": str(candidate["wallet_address"]),
                "market_slug": str(candidate["market_slug"]),
                "outcome": str(candidate["outcome"]),
                "price": entry_price,
                "size_usd": size_usd,
            }
        )

    def _handle_sell(self, wallet: CopyTradeWallet, row: dict[str, Any], market: Market, now: datetime) -> None:
        token_id = str(row.get("asset") or "")
        candidates = [
            position for position in self.open_positions.values()
            if position.wallet_address == wallet.address
            and position.condition_id == str(row.get("conditionId") or "")
            and position.token_id == token_id
        ]
        if not candidates:
            return
        position = min(candidates, key=lambda item: item.entry_timestamp)
        exit_price = float(row.get("price") or self._token_price(market, token_id) or 0.0)
        if exit_price <= 0:
            return
        self._close_position(position, exit_price=exit_price, close_reason="wallet_sell", now=now)

    def _build_wallet_sell_event(self, wallet: CopyTradeWallet, row: dict[str, Any], market: Market, now: datetime) -> dict[str, Any]:
        token_id = str(row.get("asset") or "")
        activity_at = self._parse_timestamp(row.get("timestamp"), now)
        matching_positions = [
            position.trade_id
            for position in self.open_positions.values()
            if position.wallet_address == wallet.address
            and position.condition_id == str(row.get("conditionId") or "")
            and position.token_id == token_id
        ]
        return {
            "event_id": self._build_candidate_id(wallet.address, row),
            "observed_at": now.isoformat(),
            "activity_timestamp": activity_at.isoformat(),
            "wallet_address": wallet.address,
            "wallet_name": wallet.name,
            "wallet_rank": wallet.rank,
            "wallet_pnl_usd": round(wallet.pnl_usd, 4),
            "tx_hash": str(row.get("transactionHash") or ""),
            "condition_id": str(row.get("conditionId") or ""),
            "market_slug": market.slug,
            "token_id": token_id,
            "outcome": str(row.get("outcome") or self._outcome_label(market, token_id)),
            "wallet_side": "SELL",
            "wallet_trade_usd": round(float(row.get("usdcSize") or row.get("size") or 0.0), 4),
            "wallet_trade_price": round(float(row.get("price") or self._token_price(market, token_id) or 0.0), 6),
            "matched_open_trade_ids": matching_positions,
            "matched_open_trade_count": len(matching_positions),
        }

    async def _resolve_positions(self, now: datetime) -> None:
        if not self.open_positions:
            return
        for position in list(self.open_positions.values()):
            market = await self.collector.get_market_by_slug(position.market_slug)
            if market is None:
                continue
            winning_token, settlement_price = self._final_outcome(market, position.token_id)
            if winning_token is None or settlement_price is None:
                continue
            self._close_position(
                position,
                exit_price=float(settlement_price),
                close_reason="resolved",
                now=now,
            )

    def _close_position(self, position: CopyTradePosition, *, exit_price: float, close_reason: str, now: datetime) -> None:
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
        self.audit.log_trade_ledger(
            {
                "trade_id": position.trade_id,
                "wallet_address": position.wallet_address,
                "wallet_name": position.wallet_name,
                "tx_hash": position.tx_hash,
                "condition_id": position.condition_id,
                "market_slug": position.market_slug,
                "outcome": position.outcome_label,
                "entry_timestamp": position.entry_timestamp.isoformat(),
                "entry_price": position.entry_price,
                "entry_size_usd": position.entry_size_usd,
                "exit_timestamp": now.isoformat(),
                "exit_price": exit_price,
                "realized_pnl_usd": round(realized_pnl, 4),
                "close_reason": close_reason,
            }
        )
        self.audit.log_trade_event(
            {
                "event_type": "exit",
                "trade_id": position.trade_id,
                "wallet_address": position.wallet_address,
                "market_slug": position.market_slug,
                "outcome": position.outcome_label,
                "exit_price": exit_price,
                "realized_pnl_usd": round(realized_pnl, 4),
                "close_reason": close_reason,
            }
        )
        self._recent_trades.insert(
            0,
            {
                "id": position.trade_id,
                "time": now.isoformat(),
                "timestamp": now.isoformat(),
                "source": self.cfg.source,
                "market": position.market_slug,
                "wallet": position.wallet_name,
                "side": position.outcome_label,
                "price": round(exit_price, 4),
                "usd": round(position.entry_size_usd, 2),
                "pnl": round(realized_pnl, 2),
                "entry": round(position.entry_price, 4),
                "exit": round(exit_price, 4),
                "event": "exit",
            },
        )
        self._recent_trades = self._recent_trades[:MAX_RECENT_ITEMS]
        self.open_positions.pop(position.trade_id, None)

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
        portfolio_value = self._current_equity()
        self.audit.log_daily_summary(
            {
                "date": day_key,
                "tracked_wallets": len(self.target_wallets),
                "open_positions": len(self.open_positions),
                "resolved_trades": self.resolved_trades,
                "wins": self.wins,
                "losses": self.losses,
                "win_rate": round(self.wins / self.resolved_trades, 6) if self.resolved_trades else 0.0,
                "realized_pnl_usd": round(self.realized_pnl_usd, 4),
                "cash_balance_usd": round(self.cash_balance, 4),
                "portfolio_value_usd": round(portfolio_value, 4),
            }
        )

    def _current_equity(self) -> float:
        positions_value = 0.0
        for position in self.open_positions.values():
            positions_value += position.shares * self._last_prices.get(position.token_id, position.entry_price)
        return self.cash_balance + positions_value

    def _refresh_last_prices(self, market_map: dict[str, Market]) -> None:
        for market in market_map.values():
            for outcome in market.outcomes:
                self._last_prices[outcome.token_id] = outcome.price

    def _outcome_label(self, market: Market, token_id: str) -> str:
        for outcome in market.outcomes:
            if outcome.token_id == token_id:
                return outcome.name
        return "Outcome"

    def _token_price(self, market: Market, token_id: str) -> float | None:
        for outcome in market.outcomes:
            if outcome.token_id == token_id:
                return outcome.price
        return None

    def _token_book(self, market: Market, token_id: str) -> tuple[float | None, float | None]:
        for outcome in market.outcomes:
            if outcome.token_id == token_id:
                return outcome.book_bid, outcome.book_ask
        return None, None

    @staticmethod
    def _time_to_close_hours(market: Market, now: datetime) -> float | None:
        if not market.end_date:
            return None
        try:
            end_at = datetime.fromisoformat(str(market.end_date).replace("Z", "+00:00"))
        except ValueError:
            return None
        return (end_at.astimezone(UTC) - now).total_seconds() / 3600.0

    @staticmethod
    def _parse_timestamp(raw_timestamp: Any, fallback: datetime) -> datetime:
        try:
            return datetime.fromtimestamp(int(raw_timestamp), tz=UTC)
        except Exception:
            return fallback

    @staticmethod
    def _build_trade_key(wallet_address: str, row: dict[str, Any]) -> str:
        return "|".join(
            [
                wallet_address,
                str(row.get("transactionHash") or ""),
                str(row.get("conditionId") or ""),
                str(row.get("asset") or ""),
                str(row.get("side") or ""),
                str(row.get("timestamp") or ""),
                str(row.get("type") or ""),
            ]
        )

    @staticmethod
    def _build_candidate_id(wallet_address: str, row: dict[str, Any]) -> str:
        return "|".join(
            [
                "copy_candidate",
                wallet_address,
                str(row.get("transactionHash") or ""),
                str(row.get("conditionId") or ""),
                str(row.get("asset") or ""),
                str(row.get("side") or ""),
                str(row.get("timestamp") or ""),
            ]
        )

    def _remember_trade_key(self, trade_key: str) -> None:
        if trade_key in self._seen_trade_key_set:
            return
        self._seen_trade_key_set.add(trade_key)
        self.seen_trade_keys.append(trade_key)
        if len(self.seen_trade_keys) > MAX_SEEN_TRADE_KEYS:
            oldest = self.seen_trade_keys.pop(0)
            self._seen_trade_key_set.discard(oldest)
        self._stats["activity_events_seen"] = len(self.seen_trade_keys)

    @staticmethod
    def _final_outcome(market: Market, token_id: str) -> tuple[str | None, float | None]:
        if market.active and not market.closed:
            return None, None
        winner = max(market.outcomes, key=lambda outcome: float(outcome.price or 0.0))
        if winner.price is None:
            return None, None
        if float(winner.price) < 0.5:
            return None, None
        settlement_price = 0.0
        for outcome in market.outcomes:
            if outcome.token_id == token_id:
                settlement_price = float(outcome.price or 0.0)
                break
        return winner.token_id, settlement_price

    def _runtime_state_payload(self) -> dict[str, Any]:
        return {
            "cash_balance": self.cash_balance,
            "realized_pnl_usd": self.realized_pnl_usd,
            "wins": self.wins,
            "losses": self.losses,
            "resolved_trades": self.resolved_trades,
            "target_wallets": [asdict(wallet) for wallet in self.target_wallets],
            "open_positions": {trade_id: position.to_payload() for trade_id, position in self.open_positions.items()},
            "seen_trade_keys": self.seen_trade_keys[-MAX_SEEN_TRADE_KEYS:],
            "recent_signals": self._recent_signals[:10],
            "recent_trades": self._recent_trades[:10],
            "last_wallet_refresh_at": self._last_wallet_refresh_at.isoformat() if self._last_wallet_refresh_at else None,
            "last_activity_refresh_at": self._last_activity_refresh_at.isoformat() if self._last_activity_refresh_at else None,
        }

    def _normalize_signal_view(self, row: dict[str, Any]) -> dict[str, Any]:
        price = float(row.get("price") or 0.0)
        size_usd = float(row.get("size") or row.get("size_usd") or 0.0)
        wallet = str(row.get("wallet") or row.get("wallet_name") or "")
        outcome = str(row.get("outcome") or row.get("side") or "Outcome")
        signal_id = str(row.get("id") or row.get("trade_id") or "")
        when = str(row.get("time") or row.get("timestamp") or datetime.now(UTC).isoformat())
        market = str(row.get("market") or row.get("market_slug") or "")
        return {
            "id": signal_id,
            "time": when,
            "timestamp": when,
            "source": str(row.get("source") or self.cfg.source),
            "action": str(row.get("action") or "copy_buy"),
            "market": market,
            "confidence": float(row.get("confidence") or 0.5),
            "edge": float(row.get("edge") or 0.0),
            "size": round(size_usd, 2),
            "whale": bool(row.get("whale", True)),
            "reasoning": str(row.get("reasoning") or f"Copied {wallet or 'wallet'} into {outcome} at ${price:.3f}"),
            "wallet": wallet,
            "outcome": outcome,
            "price": round(price, 4),
            "size_usd": round(size_usd, 2),
        }

    def _normalize_trade_view(self, row: dict[str, Any]) -> dict[str, Any]:
        entry = float(row.get("entry") or row.get("entry_price") or 0.0)
        exit_price = float(row.get("exit") or row.get("exit_price") or 0.0)
        price = float(row.get("price") or exit_price or entry)
        size_usd = float(row.get("usd") or row.get("size_usd") or row.get("entry_size_usd") or 0.0)
        when = str(row.get("time") or row.get("timestamp") or row.get("exit_timestamp") or row.get("entry_timestamp") or datetime.now(UTC).isoformat())
        return {
            "id": str(row.get("id") or row.get("trade_id") or ""),
            "time": when,
            "timestamp": when,
            "source": str(row.get("source") or self.cfg.source),
            "market": str(row.get("market") or row.get("market_slug") or ""),
            "wallet": str(row.get("wallet") or row.get("wallet_name") or ""),
            "side": str(row.get("side") or row.get("outcome") or "Outcome"),
            "price": round(price, 4),
            "usd": round(size_usd, 2),
            "pnl": row.get("pnl") if row.get("pnl") is not None else row.get("realized_pnl_usd"),
            "entry": round(entry, 4) if entry else 0.0,
            "exit": round(exit_price, 4) if exit_price else 0.0,
            "event": str(row.get("event") or ("exit" if row.get("pnl") is not None or row.get("realized_pnl_usd") is not None else "entry")),
        }

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
        self.target_wallets = [
            CopyTradeWallet(
                address=str(row.get("address") or ""),
                name=str(row.get("name") or ""),
                pnl_usd=float(row.get("pnl_usd") or 0.0),
                rank=int(row.get("rank") or 0),
            )
            for row in (payload.get("target_wallets") or [])
            if row.get("address")
        ]
        self.open_positions = {
            trade_id: CopyTradePosition.from_payload(row)
            for trade_id, row in (payload.get("open_positions") or {}).items()
        }
        self.seen_trade_keys = list(payload.get("seen_trade_keys") or [])
        self._seen_trade_key_set = set(self.seen_trade_keys)
        self._recent_signals = [self._normalize_signal_view(row) for row in (payload.get("recent_signals") or [])]
        self._recent_trades = [self._normalize_trade_view(row) for row in (payload.get("recent_trades") or [])]
        if payload.get("last_wallet_refresh_at"):
            self._last_wallet_refresh_at = datetime.fromisoformat(payload["last_wallet_refresh_at"]).astimezone(UTC)
        if payload.get("last_activity_refresh_at"):
            self._last_activity_refresh_at = datetime.fromisoformat(payload["last_activity_refresh_at"]).astimezone(UTC)
        self._stats["target_wallets"] = len(self.target_wallets)
        self._stats["entries"] = len(self.open_positions) + self.resolved_trades
        self._stats["resolved_trades"] = self.resolved_trades
        self._stats["wins"] = self.wins
        self._stats["losses"] = self.losses
        self._stats["win_rate"] = self.wins / self.resolved_trades if self.resolved_trades else 0.0
        self._stats["realized_pnl_usd"] = round(self.realized_pnl_usd, 4)

    def _write_runtime_state(self) -> None:
        self.audit.write_runtime_state(self._runtime_state_payload())

    def _resolve_audit_root(self) -> Path:
        base = Path(self.cfg.audit_root).resolve() if self.cfg.audit_root else (LOG_DIR / "comparison" / self.cfg.session_label)
        base.mkdir(parents=True, exist_ok=True)
        return base
