from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from config import KalshiBtcArbShadowConfig
from data.collector import PolymarketCollector
from data.models import Event, Market
from engine.btc_hourly_clock import kalshi_btc_event_ticker, polymarket_btc_hourly_slug
from engine.kalshi_btc_client import KalshiBtcClient, KalshiBtcMarket
from engine.shadow_sleeve_audit import ShadowSleeveAudit
from runtime_paths import LOG_DIR
from strategies.base import BaseStrategy


logger = logging.getLogger(__name__)

UTC = timezone.utc
MAX_RECENT_ITEMS = 40

KALSHI_ARB_TRADE_LEDGER_FIELDS = [
    "logged_at",
    "trade_id",
    "event_ticker",
    "poly_slug",
    "poly_leg",
    "poly_price_to_beat",
    "poly_entry_price",
    "kalshi_ticker",
    "kalshi_strike",
    "kalshi_leg",
    "kalshi_entry_price",
    "units",
    "entry_cost_usd",
    "guaranteed_pnl_usd",
    "exit_timestamp",
    "payout_usd",
    "realized_pnl_usd",
    "close_reason",
]

KALSHI_ARB_DAILY_SUMMARY_FIELDS = [
    "logged_at",
    "date",
    "open_positions",
    "candidate_opportunities",
    "entries",
    "resolved_trades",
    "wins",
    "losses",
    "win_rate",
    "realized_pnl_usd",
    "cash_balance_usd",
    "portfolio_value_usd",
]


@dataclass(slots=True)
class PolyBtcHourlyMarket:
    slug: str
    condition_id: str
    end_date: str | None
    active: bool
    closed: bool
    price_to_beat: float
    up_token_id: str
    down_token_id: str
    up_price: float
    down_price: float


@dataclass(slots=True)
class KalshiBtcArbOpportunity:
    event_ticker: str
    poly_slug: str
    poly_condition_id: str
    poly_price_to_beat: float
    poly_leg: str
    poly_token_id: str
    poly_entry_price: float
    kalshi_ticker: str
    kalshi_strike: float
    kalshi_leg: str
    kalshi_entry_price: float
    cost_per_unit: float
    net_margin_per_unit: float
    guaranteed_pnl_per_unit: float
    middle_low: float
    middle_high: float


@dataclass(slots=True)
class KalshiBtcArbPosition:
    trade_id: str
    event_ticker: str
    poly_slug: str
    poly_condition_id: str
    poly_price_to_beat: float
    poly_leg: str
    poly_token_id: str
    poly_entry_price: float
    kalshi_ticker: str
    kalshi_strike: float
    kalshi_leg: str
    kalshi_entry_price: float
    cost_per_unit: float
    guaranteed_pnl_per_unit: float
    middle_low: float
    middle_high: float
    units: float
    entry_cost_usd: float
    guaranteed_pnl_usd: float
    entry_timestamp: datetime

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "KalshiBtcArbPosition":
        return cls(
            trade_id=str(payload["trade_id"]),
            event_ticker=str(payload["event_ticker"]),
            poly_slug=str(payload["poly_slug"]),
            poly_condition_id=str(payload["poly_condition_id"]),
            poly_price_to_beat=float(payload["poly_price_to_beat"]),
            poly_leg=str(payload["poly_leg"]),
            poly_token_id=str(payload["poly_token_id"]),
            poly_entry_price=float(payload["poly_entry_price"]),
            kalshi_ticker=str(payload["kalshi_ticker"]),
            kalshi_strike=float(payload["kalshi_strike"]),
            kalshi_leg=str(payload["kalshi_leg"]),
            kalshi_entry_price=float(payload["kalshi_entry_price"]),
            cost_per_unit=float(payload["cost_per_unit"]),
            guaranteed_pnl_per_unit=float(payload["guaranteed_pnl_per_unit"]),
            middle_low=float(payload["middle_low"]),
            middle_high=float(payload["middle_high"]),
            units=float(payload["units"]),
            entry_cost_usd=float(payload["entry_cost_usd"]),
            guaranteed_pnl_usd=float(payload["guaranteed_pnl_usd"]),
            entry_timestamp=datetime.fromisoformat(str(payload["entry_timestamp"])).astimezone(UTC),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "event_ticker": self.event_ticker,
            "poly_slug": self.poly_slug,
            "poly_condition_id": self.poly_condition_id,
            "poly_price_to_beat": self.poly_price_to_beat,
            "poly_leg": self.poly_leg,
            "poly_token_id": self.poly_token_id,
            "poly_entry_price": self.poly_entry_price,
            "kalshi_ticker": self.kalshi_ticker,
            "kalshi_strike": self.kalshi_strike,
            "kalshi_leg": self.kalshi_leg,
            "kalshi_entry_price": self.kalshi_entry_price,
            "cost_per_unit": self.cost_per_unit,
            "guaranteed_pnl_per_unit": self.guaranteed_pnl_per_unit,
            "middle_low": self.middle_low,
            "middle_high": self.middle_high,
            "units": self.units,
            "entry_cost_usd": self.entry_cost_usd,
            "guaranteed_pnl_usd": self.guaranteed_pnl_usd,
            "entry_timestamp": self.entry_timestamp.isoformat(),
        }


class KalshiBtcArbShadowStrategy(BaseStrategy):
    name = "kalshi_btc_arb_shadow"
    description = "Self-managed Poly/Kalshi BTC overlap arb paper sleeve"

    def __init__(self, config, collector: PolymarketCollector):
        super().__init__(config)
        self.cfg: KalshiBtcArbShadowConfig = config.kalshi_btc_arb_shadow
        self.collector = collector
        self.kalshi = KalshiBtcClient()
        self.enabled = bool(self.cfg.enabled)
        self.cash_balance = float(self.cfg.budget_usd)
        self.open_positions: dict[str, KalshiBtcArbPosition] = {}
        self.completed_events: set[str] = set()
        self.realized_pnl_usd = 0.0
        self.wins = 0
        self.losses = 0
        self.resolved_trades = 0
        self._recent_signals: list[dict[str, Any]] = []
        self._recent_trades: list[dict[str, Any]] = []
        self.audit_root = self._resolve_audit_root()
        self.audit = ShadowSleeveAudit(
            self.audit_root,
            lane_key=self.cfg.source,
            label=self.cfg.label,
            category="kalshi_btc_arb",
            source=self.cfg.source,
            description="Cross-venue Poly/Kalshi BTC overlap arb paper sleeve.",
            trade_ledger_fields=KALSHI_ARB_TRADE_LEDGER_FIELDS,
            daily_summary_fields=KALSHI_ARB_DAILY_SUMMARY_FIELDS,
        )
        self.audit.write_metadata(
            {
                "started_at": datetime.now(UTC).isoformat(),
                "strategy": self.name,
                "view_key": self.cfg.view_key,
                "budget_usd": self.cfg.budget_usd,
                "min_trade_usd": self.cfg.min_trade_usd,
                "max_trade_usd": self.cfg.max_trade_usd,
                "min_net_margin_dollars": self.cfg.min_net_margin_dollars,
                "trade_fee_buffer_dollars": self.cfg.trade_fee_buffer_dollars,
                "max_open_positions": self.cfg.max_open_positions,
            },
            extra={"view_key": self.cfg.view_key, "session_label": self.cfg.session_label},
        )
        self._stats.update(
            {
                "view_key": self.cfg.view_key,
                "source": self.cfg.source,
                "budget_usd": self.cfg.budget_usd,
                "cash_balance_usd": self.cash_balance,
                "portfolio_value_usd": self.cash_balance,
                "candidate_opportunities": 0,
                "entries": 0,
                "resolved_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "realized_pnl_usd": 0.0,
                "last_price_to_beat": None,
                "last_best_margin_dollars": 0.0,
                "last_event_ticker": None,
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
        await self._resolve_positions(now)
        candidate_count = 0
        best = None
        if len(self.open_positions) < self.cfg.max_open_positions:
            best, candidate_count = await self._find_best_opportunity(markets)
        self._stats["candidate_opportunities"] = candidate_count
        if best and best.event_ticker not in self.open_positions and best.event_ticker not in self.completed_events:
            self._enter_position(best, now)
        self._roll_daily_summary(now)
        self._stats["cash_balance_usd"] = round(self.cash_balance, 4)
        self._stats["portfolio_value_usd"] = round(self._current_equity(), 4)
        self._write_runtime_state()
        return []

    @property
    def stats(self) -> dict[str, Any]:
        return {"name": self.name, "enabled": self.enabled, **self._stats}

    def serialize_view(self) -> dict[str, Any]:
        positions = []
        guaranteed_value = 0.0
        for position in self.open_positions.values():
            guaranteed_position_value = position.units
            guaranteed_value += guaranteed_position_value
            positions.append(
                {
                    "market": f"{position.poly_slug} | {position.kalshi_ticker}",
                    "side": f"POLY {position.poly_leg.upper()} + KALSHI {position.kalshi_leg.upper()}",
                    "shares": round(position.units, 2),
                    "entry": round(position.cost_per_unit, 3),
                    "current": 1.0,
                    "pnl": round(position.guaranteed_pnl_usd, 2),
                    "source": self.cfg.source,
                }
            )

        total_value = self.cash_balance + guaranteed_value
        return {
            "key": self.cfg.view_key,
            "label": self.cfg.label,
            "source": self.cfg.source,
            "portfolio": {
                "starting_capital": round(self.cfg.budget_usd, 2),
                "total_value": round(total_value, 2),
                "cash": round(self.cash_balance, 2),
                "positions_value": round(guaranteed_value, 2),
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
                "win_rate": round(self.wins / self.resolved_trades, 6) if self.resolved_trades else 0.0,
                "total_trades": self.resolved_trades,
                "cash": round(self.cash_balance, 2),
                "budget": round(self.cfg.budget_usd, 2),
            },
        }

    async def close(self) -> None:
        self._write_runtime_state()
        await self.kalshi.close()

    def reset_state(self) -> None:
        self.cash_balance = float(self.cfg.budget_usd)
        self.open_positions = {}
        self.completed_events = set()
        self.realized_pnl_usd = 0.0
        self.wins = 0
        self.losses = 0
        self.resolved_trades = 0
        self._recent_signals = []
        self._recent_trades = []
        self._stats.update(
            {
                "candidate_opportunities": 0,
                "entries": 0,
                "resolved_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "realized_pnl_usd": 0.0,
                "last_best_margin_dollars": 0.0,
                "last_price_to_beat": None,
                "last_event_ticker": None,
            }
        )
        self._write_runtime_state()

    async def _find_best_opportunity(self, markets: list[Market]) -> tuple[KalshiBtcArbOpportunity | None, int]:
        slug = polymarket_btc_hourly_slug()
        event_ticker = kalshi_btc_event_ticker()
        if event_ticker in self.open_positions or event_ticker in self.completed_events:
            return None, 0

        market = next((row for row in markets if row.slug == slug), None)
        if market is None:
            market = await self.collector.get_market_by_slug(slug)
        raw_market = await self.collector.get_gamma_market_payload_by_slug(slug)
        poly = self._parse_poly_market(raw_market)
        if market is None or poly is None:
            return None, 0
        if not poly.active or poly.closed:
            return None, 0

        kalshi_markets = await self.kalshi.get_event_markets(event_ticker)
        ranked_markets = sorted(
            kalshi_markets,
            key=lambda kalshi_market: abs(kalshi_market.strike - poly.price_to_beat),
        )[: max(4, self.cfg.kalshi_neighbor_count * 2)]
        self._stats["last_price_to_beat"] = round(poly.price_to_beat, 2)
        self._stats["last_event_ticker"] = event_ticker
        candidates: list[KalshiBtcArbOpportunity] = []
        for kalshi_market in ranked_markets:
            if kalshi_market.status.lower() != "active":
                continue
            if kalshi_market.strike < poly.price_to_beat:
                if poly.down_price <= 0 or kalshi_market.yes_ask <= 0 or kalshi_market.yes_ask >= 1:
                    continue
                cost = poly.down_price + kalshi_market.yes_ask + self.cfg.trade_fee_buffer_dollars
                if cost <= 1.0 - self.cfg.min_net_margin_dollars:
                    candidates.append(
                        KalshiBtcArbOpportunity(
                            event_ticker=event_ticker,
                            poly_slug=poly.slug,
                            poly_condition_id=poly.condition_id,
                            poly_price_to_beat=poly.price_to_beat,
                            poly_leg="down",
                            poly_token_id=poly.down_token_id,
                            poly_entry_price=poly.down_price,
                            kalshi_ticker=kalshi_market.ticker,
                            kalshi_strike=kalshi_market.strike,
                            kalshi_leg="yes",
                            kalshi_entry_price=kalshi_market.yes_ask,
                            cost_per_unit=cost,
                            net_margin_per_unit=1.0 - cost,
                            guaranteed_pnl_per_unit=1.0 - cost,
                            middle_low=kalshi_market.strike,
                            middle_high=poly.price_to_beat,
                        )
                    )
            elif kalshi_market.strike > poly.price_to_beat:
                if poly.up_price <= 0 or kalshi_market.no_ask <= 0 or kalshi_market.no_ask >= 1:
                    continue
                cost = poly.up_price + kalshi_market.no_ask + self.cfg.trade_fee_buffer_dollars
                if cost <= 1.0 - self.cfg.min_net_margin_dollars:
                    candidates.append(
                        KalshiBtcArbOpportunity(
                            event_ticker=event_ticker,
                            poly_slug=poly.slug,
                            poly_condition_id=poly.condition_id,
                            poly_price_to_beat=poly.price_to_beat,
                            poly_leg="up",
                            poly_token_id=poly.up_token_id,
                            poly_entry_price=poly.up_price,
                            kalshi_ticker=kalshi_market.ticker,
                            kalshi_strike=kalshi_market.strike,
                            kalshi_leg="no",
                            kalshi_entry_price=kalshi_market.no_ask,
                            cost_per_unit=cost,
                            net_margin_per_unit=1.0 - cost,
                            guaranteed_pnl_per_unit=1.0 - cost,
                            middle_low=poly.price_to_beat,
                            middle_high=kalshi_market.strike,
                        )
                    )

        if not candidates:
            self._stats["last_best_margin_dollars"] = 0.0
            return None, 0

        candidates.sort(
            key=lambda opportunity: (
                opportunity.net_margin_per_unit,
                opportunity.middle_high - opportunity.middle_low,
            ),
            reverse=True,
        )
        best = candidates[0]
        self._stats["last_best_margin_dollars"] = round(best.net_margin_per_unit, 4)
        return best, len(candidates)

    def _enter_position(self, opportunity: KalshiBtcArbOpportunity, now: datetime) -> None:
        spend = min(self.cash_balance, self.cfg.max_trade_usd)
        if spend < self.cfg.min_trade_usd or opportunity.cost_per_unit <= 0:
            return

        units = spend / opportunity.cost_per_unit
        entry_cost = units * opportunity.cost_per_unit
        if entry_cost < self.cfg.min_trade_usd:
            return

        trade_id = f"kalshi-arb:{opportunity.event_ticker}:{opportunity.kalshi_ticker}"
        position = KalshiBtcArbPosition(
            trade_id=trade_id,
            event_ticker=opportunity.event_ticker,
            poly_slug=opportunity.poly_slug,
            poly_condition_id=opportunity.poly_condition_id,
            poly_price_to_beat=opportunity.poly_price_to_beat,
            poly_leg=opportunity.poly_leg,
            poly_token_id=opportunity.poly_token_id,
            poly_entry_price=opportunity.poly_entry_price,
            kalshi_ticker=opportunity.kalshi_ticker,
            kalshi_strike=opportunity.kalshi_strike,
            kalshi_leg=opportunity.kalshi_leg,
            kalshi_entry_price=opportunity.kalshi_entry_price,
            cost_per_unit=opportunity.cost_per_unit,
            guaranteed_pnl_per_unit=opportunity.guaranteed_pnl_per_unit,
            middle_low=opportunity.middle_low,
            middle_high=opportunity.middle_high,
            units=units,
            entry_cost_usd=entry_cost,
            guaranteed_pnl_usd=units * opportunity.guaranteed_pnl_per_unit,
            entry_timestamp=now,
        )
        self.open_positions[position.event_ticker] = position
        self.cash_balance -= entry_cost
        self._stats["entries"] = int(self._stats.get("entries") or 0) + 1
        signal_view = {
            "timestamp": now.isoformat(),
            "event_ticker": position.event_ticker,
            "poly_slug": position.poly_slug,
            "bundle": f"POLY {position.poly_leg.upper()} + KALSHI {position.kalshi_leg.upper()}",
            "entry_cost_per_unit": round(position.cost_per_unit, 4),
            "guaranteed_pnl_per_unit": round(position.guaranteed_pnl_per_unit, 4),
            "units": round(position.units, 3),
        }
        self._recent_signals.insert(0, signal_view)
        self._recent_signals = self._recent_signals[:MAX_RECENT_ITEMS]
        self.audit.log_signal(signal_view)
        self.audit.log_trade_event(
            {
                "event_type": "entry",
                "trade_id": position.trade_id,
                "event_ticker": position.event_ticker,
                "poly_slug": position.poly_slug,
                "kalshi_ticker": position.kalshi_ticker,
                "cost_per_unit": position.cost_per_unit,
                "units": position.units,
                "entry_cost_usd": position.entry_cost_usd,
                "guaranteed_pnl_usd": position.guaranteed_pnl_usd,
            }
        )

    async def _resolve_positions(self, now: datetime) -> None:
        for position in list(self.open_positions.values()):
            close_time = self._position_close_time(position)
            if close_time and now < close_time + timedelta(minutes=self.cfg.resolution_grace_minutes):
                continue

            poly_payload = await self.collector.get_gamma_market_payload_by_slug(position.poly_slug)
            poly_result = self._resolve_poly_result(poly_payload)
            kalshi_markets = await self.kalshi.get_event_markets(position.event_ticker)
            kalshi_result = self._resolve_kalshi_result(kalshi_markets, position.kalshi_ticker)
            if poly_result is None or kalshi_result is None:
                continue

            poly_win = 1 if poly_result == position.poly_leg else 0
            kalshi_win = 1 if kalshi_result == position.kalshi_leg else 0
            payout_usd = position.units * (poly_win + kalshi_win)
            realized_pnl = payout_usd - position.entry_cost_usd
            self.cash_balance += payout_usd
            self.realized_pnl_usd += realized_pnl
            self.resolved_trades += 1
            self.completed_events.add(position.event_ticker)
            if realized_pnl >= 0:
                self.wins += 1
            else:
                self.losses += 1
            self._stats["resolved_trades"] = self.resolved_trades
            self._stats["wins"] = self.wins
            self._stats["losses"] = self.losses
            self._stats["win_rate"] = self.wins / self.resolved_trades if self.resolved_trades else 0.0
            self._stats["realized_pnl_usd"] = round(self.realized_pnl_usd, 4)
            self.audit.log_trade_ledger(
                {
                    "trade_id": position.trade_id,
                    "event_ticker": position.event_ticker,
                    "poly_slug": position.poly_slug,
                    "poly_leg": position.poly_leg,
                    "poly_price_to_beat": position.poly_price_to_beat,
                    "poly_entry_price": position.poly_entry_price,
                    "kalshi_ticker": position.kalshi_ticker,
                    "kalshi_strike": position.kalshi_strike,
                    "kalshi_leg": position.kalshi_leg,
                    "kalshi_entry_price": position.kalshi_entry_price,
                    "units": round(position.units, 6),
                    "entry_cost_usd": round(position.entry_cost_usd, 4),
                    "guaranteed_pnl_usd": round(position.guaranteed_pnl_usd, 4),
                    "exit_timestamp": now.isoformat(),
                    "payout_usd": round(payout_usd, 4),
                    "realized_pnl_usd": round(realized_pnl, 4),
                    "close_reason": "resolved",
                }
            )
            self.audit.log_trade_event(
                {
                    "event_type": "exit",
                    "trade_id": position.trade_id,
                    "event_ticker": position.event_ticker,
                    "poly_slug": position.poly_slug,
                    "kalshi_ticker": position.kalshi_ticker,
                    "payout_usd": round(payout_usd, 4),
                    "realized_pnl_usd": round(realized_pnl, 4),
                    "close_reason": "resolved",
                }
            )
            self._recent_trades.insert(
                0,
                {
                    "timestamp": now.isoformat(),
                    "event_ticker": position.event_ticker,
                    "bundle": f"POLY {position.poly_leg.upper()} + KALSHI {position.kalshi_leg.upper()}",
                    "entry_cost": round(position.entry_cost_usd, 2),
                    "payout": round(payout_usd, 2),
                    "pnl": round(realized_pnl, 2),
                },
            )
            self._recent_trades = self._recent_trades[:MAX_RECENT_ITEMS]
            self.open_positions.pop(position.event_ticker, None)

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
                "open_positions": len(self.open_positions),
                "candidate_opportunities": int(self._stats.get("candidate_opportunities") or 0),
                "entries": int(self._stats.get("entries") or 0),
                "resolved_trades": self.resolved_trades,
                "wins": self.wins,
                "losses": self.losses,
                "win_rate": round(self.wins / self.resolved_trades, 6) if self.resolved_trades else 0.0,
                "realized_pnl_usd": round(self.realized_pnl_usd, 4),
                "cash_balance_usd": round(self.cash_balance, 4),
                "portfolio_value_usd": round(self._current_equity(), 4),
            }
        )

    def _current_equity(self) -> float:
        return self.cash_balance + sum(position.units for position in self.open_positions.values())

    @staticmethod
    def _parse_poly_market(payload: dict[str, Any] | None) -> PolyBtcHourlyMarket | None:
        if not payload:
            return None
        outcomes = payload.get("outcomes") or []
        token_ids = payload.get("clobTokenIds") or []
        prices = payload.get("outcomePrices") or []
        if isinstance(outcomes, str):
            outcomes = [item.strip().strip('"') for item in outcomes.strip("[]").split(",") if item.strip()]
        if isinstance(token_ids, str):
            token_ids = [item.strip().strip('"') for item in token_ids.strip("[]").split(",") if item.strip()]
        if isinstance(prices, str):
            prices = [item.strip().strip('"') for item in prices.strip("[]").split(",") if item.strip()]
        if len(outcomes) < 2 or len(token_ids) < 2 or len(prices) < 2:
            return None

        mapping: dict[str, tuple[str, float]] = {}
        for name, token_id, price in zip(outcomes, token_ids, prices):
            try:
                mapping[str(name).lower()] = (str(token_id), float(price))
            except (TypeError, ValueError):
                continue
        if "up" not in mapping or "down" not in mapping:
            return None

        price_to_beat = None
        event_metadata = payload.get("eventMetadata") or {}
        if isinstance(event_metadata, dict):
            price_to_beat = event_metadata.get("priceToBeat")
        if price_to_beat is None:
            events = payload.get("events") or []
            if events and isinstance(events[0], dict):
                meta = events[0].get("eventMetadata") or {}
                if isinstance(meta, dict):
                    price_to_beat = meta.get("priceToBeat")
        if price_to_beat is None:
            return None

        up_token_id, up_price = mapping["up"]
        down_token_id, down_price = mapping["down"]
        return PolyBtcHourlyMarket(
            slug=str(payload.get("slug") or ""),
            condition_id=str(payload.get("conditionId") or payload.get("condition_id") or ""),
            end_date=payload.get("endDate") or payload.get("end_date_iso"),
            active=bool(payload.get("active", True)),
            closed=bool(payload.get("closed", False)),
            price_to_beat=float(price_to_beat),
            up_token_id=up_token_id,
            down_token_id=down_token_id,
            up_price=up_price,
            down_price=down_price,
        )

    @staticmethod
    def _resolve_poly_result(payload: dict[str, Any] | None) -> str | None:
        poly = KalshiBtcArbShadowStrategy._parse_poly_market(payload)
        if poly is None:
            return None
        if poly.active and not poly.closed:
            return None
        if poly.up_price >= 0.99:
            return "up"
        if poly.down_price >= 0.99:
            return "down"
        return None

    @staticmethod
    def _resolve_kalshi_result(markets: list[KalshiBtcMarket], ticker: str) -> str | None:
        for market in markets:
            if market.ticker != ticker:
                continue
            result = (market.result or "").lower()
            if "yes" in result:
                return "yes"
            if "no" in result:
                return "no"
        return None

    def _position_close_time(self, position: KalshiBtcArbPosition) -> datetime | None:
        try:
            market_time = datetime.fromisoformat(position.entry_timestamp.isoformat())
        except ValueError:
            market_time = position.entry_timestamp
        return market_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

    def _runtime_state_payload(self) -> dict[str, Any]:
        return {
            "cash_balance": self.cash_balance,
            "realized_pnl_usd": self.realized_pnl_usd,
            "wins": self.wins,
            "losses": self.losses,
            "resolved_trades": self.resolved_trades,
            "open_positions": {
                event_ticker: position.to_payload()
                for event_ticker, position in self.open_positions.items()
            },
            "completed_events": sorted(self.completed_events),
            "recent_signals": self._recent_signals[:10],
            "recent_trades": self._recent_trades[:10],
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
        self.open_positions = {
            event_ticker: KalshiBtcArbPosition.from_payload(row)
            for event_ticker, row in (payload.get("open_positions") or {}).items()
        }
        self.completed_events = set(payload.get("completed_events") or [])
        self._recent_signals = list(payload.get("recent_signals") or [])
        self._recent_trades = list(payload.get("recent_trades") or [])
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
