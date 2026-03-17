from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data.models import Event, Market
from engine.telegram_notifier import TelegramNotifier
from engine.weather_edge_backtest import _passes_rule_filter
from engine.weather_edge_baseline import FrozenWeatherModelBundle, build_baseline_feature_row
from engine.weather_edge_live_audit import WeatherEdgeLiveAudit
from engine.weather_edge_live_support import (
    WEATHER_EDGE_LIVE_METRIC,
    best_resolution_yes_outcome,
    compute_binary_kelly_fraction,
    determine_live_lead_time_hours,
    parse_market_end_datetime,
    region_for_city_name,
)
from engine.weather_edge_replay import _consensus_from_models
from runtime_paths import LOG_DIR
from strategies.base import BaseStrategy
from strategies.weather import WeatherForecastStrategy


logger = logging.getLogger(__name__)

UTC = timezone.utc
MAX_RECENT_ACTIVITY = 30


@dataclass(slots=True)
class WeatherEdgeOpenPosition:
    trade_id: str
    event_key: str
    market_id: str
    market_slug: str
    city: str
    metric: str
    lead_time_hours: int
    trade_side: str
    model_probability: float
    market_probability: float
    edge: float
    kelly_fraction: float
    position_size_usdc: float
    entry_timestamp: datetime
    entry_price: float
    shares: float
    token_id: str

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "WeatherEdgeOpenPosition":
        return cls(
            trade_id=str(payload["trade_id"]),
            event_key=str(payload["event_key"]),
            market_id=str(payload["market_id"]),
            market_slug=str(payload["market_slug"]),
            city=str(payload["city"]),
            metric=str(payload["metric"]),
            lead_time_hours=int(payload["lead_time_hours"]),
            trade_side=str(payload["trade_side"]),
            model_probability=float(payload["model_probability"]),
            market_probability=float(payload["market_probability"]),
            edge=float(payload["edge"]),
            kelly_fraction=float(payload["kelly_fraction"]),
            position_size_usdc=float(payload["position_size_usdc"]),
            entry_timestamp=datetime.fromisoformat(str(payload["entry_timestamp"])).astimezone(UTC),
            entry_price=float(payload["entry_price"]),
            shares=float(payload["shares"]),
            token_id=str(payload["token_id"]),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "event_key": self.event_key,
            "market_id": self.market_id,
            "market_slug": self.market_slug,
            "city": self.city,
            "metric": self.metric,
            "lead_time_hours": self.lead_time_hours,
            "trade_side": self.trade_side,
            "model_probability": self.model_probability,
            "market_probability": self.market_probability,
            "edge": self.edge,
            "kelly_fraction": self.kelly_fraction,
            "position_size_usdc": self.position_size_usdc,
            "entry_timestamp": self.entry_timestamp.isoformat(),
            "entry_price": self.entry_price,
            "shares": self.shares,
            "token_id": self.token_id,
        }


class WeatherEdgeLiveStrategy(BaseStrategy):
    name = "weather_edge_live"
    description = "Standalone live weather-edge-v1 lane"

    def __init__(self, config, weather_strategy: WeatherForecastStrategy):
        super().__init__(config)
        self.cfg = config.weather_edge_live
        self.weather_strategy = weather_strategy
        self.bundle = FrozenWeatherModelBundle(self.cfg.model_dir)
        self.notifier = TelegramNotifier.from_env()
        self.audit_root = self._resolve_audit_root()
        self.audit = WeatherEdgeLiveAudit(self.audit_root)
        self.enabled = bool(self.cfg.enabled and self.bundle.ready)

        self.cash_balance = float(self.cfg.starting_bankroll_usd)
        self.peak_equity = float(self.cfg.starting_bankroll_usd)
        self.max_drawdown_pct = 0.0
        self.open_positions: dict[str, WeatherEdgeOpenPosition] = {}
        self.completed_event_keys: set[str] = set()
        self.daily_records: dict[str, dict[str, Any]] = {}
        self.sent_daily_summaries: set[str] = set()
        self._recent_trade_events: list[dict[str, Any]] = []
        self._recent_signals: list[dict[str, Any]] = []
        self._last_position_prices: dict[str, float] = {}

        self.audit.write_metadata(
            {
                "started_at": datetime.now(UTC).isoformat(),
                "strategy": self.name,
                "label": "Weather Edge Live",
                "bundle_dir": str(Path(self.cfg.model_dir).resolve()),
                "starting_bankroll_usd": self.cfg.starting_bankroll_usd,
                "min_edge": self.cfg.min_edge,
                "max_position_fraction": self.cfg.max_position_fraction,
                "allowed_lead_times_hours": list(self.cfg.allowed_lead_times_hours),
                "daily_summary_hour_utc": self.cfg.daily_summary_hour_utc,
                "telegram_enabled": self.notifier.enabled,
            }
        )
        self._load_runtime_state()
        self._stats.update(
            {
                "bundle_ready": self.bundle.ready,
                "bundle_error": self.bundle.load_error,
                "bundle_models": self.bundle.available_models(),
                "bankroll_usd": round(self._current_equity(), 4),
                "cash_usd": round(self.cash_balance, 4),
                "max_drawdown_pct": round(self.max_drawdown_pct * 100.0, 4),
                "candidate_markets": 0,
                "eligible_markets": 0,
                "selected_markets": 0,
                "entries": 0,
                "resolved_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "realized_pnl_usd": 0.0,
                "last_scan_at": None,
                "last_entry_at": None,
                "last_resolution_at": None,
                "last_daily_summary_at": None,
                "telegram_enabled": self.notifier.enabled,
                "trade_ledger_csv": str(self.audit.paths["trade_ledger_csv"]),
                "daily_summary_path": str(self.audit.paths["daily_summary_jsonl"]),
                "runtime_state_path": str(self.audit.paths["runtime_state"]),
            }
        )
        self._write_runtime_state()

    async def scan(self, markets: list[Market], events: list[Event]) -> list:
        del events
        self._stats["scans_completed"] += 1
        self._stats["last_scan_at"] = datetime.now(UTC).isoformat()
        try:
            now = datetime.now(UTC)
            self._roll_daily_summary(now)
            market_map = {market.condition_id: market for market in markets}
            self._resolve_positions(market_map, now)
            opportunities = self._build_live_opportunities(now)
            selected = self._select_best_opportunities(opportunities)
            self._stats["candidate_markets"] = len(opportunities)
            self._stats["selected_markets"] = len(selected)
            for opportunity in selected:
                self._enter_position(opportunity, now)
            self._update_runtime_metrics(market_map)
            self._write_runtime_state()
        except Exception as exc:
            self._stats["errors"] += 1
            self._notify_error(f"weather_edge_live scan failure: {exc}")
            raise
        return []

    @property
    def stats(self) -> dict:
        return {
            "name": self.name,
            "enabled": self.enabled,
            **self._stats,
        }

    def serialize_view(self) -> dict[str, Any]:
        positions = []
        current_mark_to_market = 0.0
        for position in self.open_positions.values():
            current_probability = self._last_position_prices.get(position.market_id, position.entry_price)
            current_value = position.shares * current_probability
            current_mark_to_market += current_value
            positions.append(
                {
                    "market": position.market_slug,
                    "side": position.trade_side.upper(),
                    "shares": round(position.shares, 2),
                    "entry": round(position.entry_price, 3),
                    "current": round(current_probability, 3),
                    "pnl": round(current_value - position.position_size_usdc, 2),
                    "source": self.cfg.source,
                }
            )

        total_value = self.cash_balance + current_mark_to_market
        resolved = int(self._stats["resolved_trades"] or 0)
        wins = int(self._stats["wins"] or 0)
        return {
            "key": self.cfg.view_key,
            "label": self.cfg.label,
            "source": self.cfg.source,
            "portfolio": {
                "starting_capital": round(self.cfg.starting_bankroll_usd, 2),
                "total_value": round(total_value, 2),
                "cash": round(self.cash_balance, 2),
                "positions_value": round(current_mark_to_market, 2),
                "total_pnl": round(total_value - self.cfg.starting_bankroll_usd, 2),
                "total_pnl_pct": round(((total_value / self.cfg.starting_bankroll_usd) - 1.0) * 100.0, 2),
                "total_trades": resolved,
                "win_rate": round((wins / resolved) * 100.0, 1) if resolved else 0.0,
                "max_drawdown": round(self.max_drawdown_pct * 100.0, 2),
                "positions": positions,
            },
            "signals": list(self._recent_signals[-MAX_RECENT_ACTIVITY:]),
            "trades": list(self._recent_trade_events[-MAX_RECENT_ACTIVITY:]),
            "performance": {
                "total_pnl": round(total_value - self.cfg.starting_bankroll_usd, 2),
                "win_rate": round(wins / resolved, 6) if resolved else 0.0,
                "total_trades": resolved,
                "open_positions": len(self.open_positions),
            },
        }

    async def close(self) -> None:
        self._write_runtime_state()

    def _build_live_opportunities(self, now: datetime) -> list[dict[str, Any]]:
        if not self.enabled:
            return []

        opportunities: list[dict[str, Any]] = []
        eligible_count = 0
        for candidate in self.weather_strategy.get_model_candidates():
            market = candidate["market"]
            resolution_time = parse_market_end_datetime(market.end_date)
            lead_time_hours = determine_live_lead_time_hours(
                resolution_time=resolution_time,
                now=now,
            )
            if lead_time_hours is None:
                continue

            city = str(candidate["city"])
            region = region_for_city_name(city)
            context = candidate.get("context") or {}
            model_temperatures = dict(context.get("current_temps") or {})
            consensus_probability, model_agreement, model_spread = _consensus_from_models(
                model_temperatures=model_temperatures,
                temp_kind=str(candidate["range_kind"]),
                temp_range_low=float(candidate["temp_range"][0]),
                temp_range_high=float(candidate["temp_range"][1]),
            )
            if consensus_probability is None or model_agreement is None or model_spread is None:
                continue

            feature_row = build_baseline_feature_row(
                city=city,
                temp_unit=str(candidate.get("temp_unit") or "F"),
                temp_kind=str(candidate["range_kind"]),
                temp_range_low=float(candidate["temp_range"][0]),
                temp_range_high=float(candidate["temp_range"][1]),
                target_month=int(str(candidate["target_date"])[5:7]),
                target_day=int(str(candidate["target_date"])[8:10]),
                market_yes_price=float(candidate["yes_price"]),
                forecast_temp_max=float(context["current_consensus"]),
                model_temperatures=model_temperatures,
            )
            prediction = self.bundle.predict_yes_probability(
                feature_row,
                temp_kind=str(candidate["range_kind"]),
            )
            if not prediction:
                continue

            model_probability = float(prediction["prob_yes"])
            market_probability = float(candidate["yes_price"])
            edge = model_probability - market_probability
            trade_side = "yes" if edge >= 0 else "no"
            trade_market_probability = market_probability if trade_side == "yes" else 1.0 - market_probability
            trade_model_probability = model_probability if trade_side == "yes" else 1.0 - model_probability
            kelly_fraction = compute_binary_kelly_fraction(
                your_probability=trade_model_probability,
                market_probability=trade_market_probability,
                max_fraction=float(self.cfg.max_position_fraction),
            )

            row = {
                "market_id": market.condition_id,
                "market_slug": market.slug,
                "market": market,
                "city": city,
                "metric_type": WEATHER_EDGE_LIVE_METRIC,
                "lead_time_hours": lead_time_hours,
                "target_date": str(candidate["target_date"]),
                "region": region,
                "temp_unit": str(candidate.get("temp_unit") or "F"),
                "temp_kind": str(candidate["range_kind"]),
                "temp_range_low": float(candidate["temp_range"][0]),
                "temp_range_high": float(candidate["temp_range"][1]),
                "market_yes_probability": market_probability,
                "model_yes_probability": model_probability,
                "raw_edge": edge,
                "absolute_edge": abs(edge),
                "trade_side": trade_side,
                "model_probability": model_probability,
                "market_probability": market_probability,
                "edge": edge,
                "kelly_fraction": kelly_fraction,
                "model_agreement": model_agreement,
                "model_spread": model_spread,
                "volume_clob": float(getattr(market, "volume_total", 0.0) or 0.0),
                "resolution_time": resolution_time,
                "reasoning": (
                    f"Weather Edge Live {trade_side.upper()} | {city} {candidate['target_date']} | "
                    f"lead {lead_time_hours}h | model {model_probability:.1%} vs market {market_probability:.1%}"
                ),
            }
            if not _passes_rule_filter(row):
                continue
            if abs(edge) < float(self.cfg.min_edge):
                continue
            eligible_count += 1
            opportunities.append(row)

        self._stats["eligible_markets"] = eligible_count
        return opportunities

    def _select_best_opportunities(self, opportunities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        selected_by_event: dict[str, dict[str, Any]] = {}
        for row in opportunities:
            event_key = self._event_key(
                city=row["city"],
                target_date=row["target_date"],
                lead_time_hours=row["lead_time_hours"],
            )
            row["event_key"] = event_key
            if event_key in self.completed_event_keys or event_key in self.open_positions:
                continue
            current = selected_by_event.get(event_key)
            if current is None or abs(float(row["edge"])) > abs(float(current["edge"])):
                selected_by_event[event_key] = row
        return sorted(
            selected_by_event.values(),
            key=lambda row: (abs(float(row["edge"])), float(row["model_probability"])),
            reverse=True,
        )

    def _enter_position(self, opportunity: dict[str, Any], now: datetime) -> None:
        market = opportunity["market"]
        token_index = 0 if opportunity["trade_side"] == "yes" else 1
        token = market.outcomes[token_index]
        trade_market_probability = (
            float(opportunity["market_probability"])
            if opportunity["trade_side"] == "yes"
            else 1.0 - float(opportunity["market_probability"])
        )
        position_size_usdc = min(
            self.cash_balance,
            self._current_equity() * float(opportunity["kelly_fraction"]),
        )
        if position_size_usdc <= 0:
            return

        shares = position_size_usdc / max(trade_market_probability, 0.0001)
        position = WeatherEdgeOpenPosition(
            trade_id=str(uuid.uuid4())[:8],
            event_key=str(opportunity["event_key"]),
            market_id=str(opportunity["market_id"]),
            market_slug=str(opportunity["market_slug"]),
            city=str(opportunity["city"]),
            metric=WEATHER_EDGE_LIVE_METRIC,
            lead_time_hours=int(opportunity["lead_time_hours"]),
            trade_side=str(opportunity["trade_side"]),
            model_probability=float(opportunity["model_probability"]),
            market_probability=float(opportunity["market_probability"]),
            edge=float(opportunity["edge"]),
            kelly_fraction=float(opportunity["kelly_fraction"]),
            position_size_usdc=position_size_usdc,
            entry_timestamp=now,
            entry_price=trade_market_probability,
            shares=shares,
            token_id=str(token.token_id),
        )
        self.cash_balance -= position_size_usdc
        self.open_positions[position.event_key] = position
        self._stats["entries"] += 1
        self._stats["last_entry_at"] = now.isoformat()

        event_payload = {
            "trade_id": position.trade_id,
            "event": "entry",
            "market_id": position.market_id,
            "city": position.city,
            "metric": position.metric,
            "lead_time_hours": position.lead_time_hours,
            "model_probability": round(position.model_probability, 6),
            "market_probability": round(position.market_probability, 6),
            "edge": round(position.edge, 6),
            "kelly_fraction": round(position.kelly_fraction, 6),
            "position_size_usdc": round(position.position_size_usdc, 4),
            "entry_timestamp": position.entry_timestamp.isoformat(),
            "outcome": "PENDING",
            "pnl_usdc": 0.0,
            "cumulative_bankroll": round(self._current_equity(), 4),
        }
        self.audit.log_trade_event(event_payload)
        self.audit.log_trade_ledger(event_payload)
        self._push_recent_trade_event(event_payload)
        signal_payload = {
            "id": position.trade_id,
            "time": position.entry_timestamp.isoformat(),
            "source": self.cfg.source,
            "action": f"buy_{position.trade_side}",
            "market": position.market_slug,
            "confidence": round(abs(position.edge), 4),
            "edge": round(position.edge, 4),
            "size": round(position.position_size_usdc, 2),
            "reasoning": opportunity["reasoning"],
        }
        self._push_recent_signal(signal_payload)
        self.audit.write_runtime_state(self._runtime_state_payload())

        self.notifier.send_message(
            "\n".join(
                [
                    "Weather Edge Entry",
                    f"Side: BUY {position.trade_side.upper()}",
                    f"City: {position.city}",
                    f"Metric: {position.metric}",
                    f"Lead: {position.lead_time_hours}h",
                    f"Edge: {position.edge:+.2%}",
                    f"Size: ${position.position_size_usdc:.2f}",
                    f"Price: {position.entry_price:.3f}",
                ]
            )
        )

    def _resolve_positions(self, market_map: dict[str, Market], now: datetime) -> None:
        resolved_event_keys: list[str] = []
        for event_key, position in self.open_positions.items():
            market = market_map.get(position.market_id)
            if market is None:
                continue
            resolved_yes = best_resolution_yes_outcome(market)
            if resolved_yes is None:
                continue

            won = resolved_yes if position.trade_side == "yes" else not resolved_yes
            payout_usdc = position.shares if won else 0.0
            pnl_usdc = payout_usdc - position.position_size_usdc
            self.cash_balance += payout_usdc
            self._stats["resolved_trades"] += 1
            self._stats["wins"] += 1 if won else 0
            self._stats["losses"] += 0 if won else 1
            self._stats["realized_pnl_usd"] = round(float(self._stats["realized_pnl_usd"]) + pnl_usdc, 4)
            self._stats["last_resolution_at"] = now.isoformat()
            resolved_event_keys.append(event_key)
            self.completed_event_keys.add(event_key)
            equity_after_resolution = self.cash_balance + sum(
                other.shares * self._last_position_prices.get(other.market_id, other.entry_price)
                for other_event_key, other in self.open_positions.items()
                if other_event_key != event_key
            )

            exit_payload = {
                "trade_id": position.trade_id,
                "event": "resolution",
                "market_id": position.market_id,
                "city": position.city,
                "metric": position.metric,
                "lead_time_hours": position.lead_time_hours,
                "model_probability": round(position.model_probability, 6),
                "market_probability": round(position.market_probability, 6),
                "edge": round(position.edge, 6),
                "kelly_fraction": round(position.kelly_fraction, 6),
                "position_size_usdc": round(position.position_size_usdc, 4),
                "entry_timestamp": position.entry_timestamp.isoformat(),
                "outcome": "WIN" if won else "LOSS",
                "pnl_usdc": round(pnl_usdc, 4),
                "cumulative_bankroll": round(equity_after_resolution, 4),
                "exit_timestamp": now.isoformat(),
            }
            self.audit.log_trade_event(exit_payload)
            self.audit.log_trade_ledger(exit_payload)
            self._push_recent_trade_event(exit_payload)
            self._record_daily_resolution(
                date_key=now.date().isoformat(),
                pnl_usdc=pnl_usdc,
                won=won,
            )
            self.notifier.send_message(
                "\n".join(
                    [
                        "Weather Edge Resolution",
                        f"City: {position.city}",
                        f"Lead: {position.lead_time_hours}h",
                        f"Outcome: {'WIN' if won else 'LOSS'}",
                        f"PnL: ${pnl_usdc:+.2f}",
                        f"Bankroll: ${self._current_equity():.2f}",
                    ]
                )
            )

        for event_key in resolved_event_keys:
            self.open_positions.pop(event_key, None)

    def _roll_daily_summary(self, now: datetime) -> None:
        current_date = now.date()
        previous_date = current_date.fromordinal(current_date.toordinal() - 1).isoformat()
        if previous_date in self.sent_daily_summaries:
            return
        if now.hour < int(self.cfg.daily_summary_hour_utc):
            return
        summary = self._daily_summary_payload(previous_date)
        self.audit.log_daily_summary(summary)
        self.sent_daily_summaries.add(previous_date)
        self._stats["last_daily_summary_at"] = now.isoformat()
        self.notifier.send_message(
            "\n".join(
                [
                    "Weather Edge Daily Summary",
                    f"Date: {summary['date']}",
                    f"Trades: {summary['total_trades']}",
                    f"Win rate: {summary['win_rate']:.2%}",
                    f"Realized PnL: ${summary['realized_pnl_usdc']:+.2f}",
                    f"Unrealized PnL: ${summary['unrealized_pnl_usdc']:+.2f}",
                    f"Bankroll: ${summary['current_bankroll']:.2f}",
                ]
            )
        )

    def _record_daily_resolution(self, *, date_key: str, pnl_usdc: float, won: bool) -> None:
        record = self._ensure_daily_record(date_key)
        record["total_trades"] += 1
        record["wins"] += 1 if won else 0
        record["losses"] += 0 if won else 1
        record["realized_pnl_usdc"] += pnl_usdc
        record["current_bankroll"] = self._current_equity()

    def _ensure_daily_record(self, date_key: str) -> dict[str, Any]:
        record = self.daily_records.get(date_key)
        if record is None:
            record = {
                "date": date_key,
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "realized_pnl_usdc": 0.0,
                "unrealized_pnl_usdc": 0.0,
                "current_bankroll": self._current_equity(),
                "equity_peak": self._current_equity(),
                "max_drawdown_pct": 0.0,
                "pending_positions": 0,
            }
            self.daily_records[date_key] = record
        else:
            record.setdefault("realized_pnl_usdc", float(record.get("total_pnl_usdc", 0.0) or 0.0))
            record.setdefault("unrealized_pnl_usdc", 0.0)
            record.setdefault("pending_positions", 0)
        return record

    def _update_runtime_metrics(self, market_map: dict[str, Market]) -> None:
        equity = self._current_equity(market_map)
        unrealized_pnl = self._current_unrealized_pnl(market_map)
        self.peak_equity = max(self.peak_equity, equity)
        if self.peak_equity > 0:
            self.max_drawdown_pct = max(self.max_drawdown_pct, (self.peak_equity - equity) / self.peak_equity)
        current_record = self._ensure_daily_record(datetime.now(UTC).date().isoformat())
        current_record["current_bankroll"] = equity
        current_record["equity_peak"] = max(float(current_record["equity_peak"]), equity)
        current_record["pending_positions"] = len(self.open_positions)
        current_record["unrealized_pnl_usdc"] = unrealized_pnl
        peak = float(current_record["equity_peak"])
        if peak > 0:
            current_record["max_drawdown_pct"] = max(
                float(current_record["max_drawdown_pct"]),
                (peak - equity) / peak,
            )
        resolved = int(self._stats["resolved_trades"] or 0)
        wins = int(self._stats["wins"] or 0)
        self._stats["bankroll_usd"] = round(equity, 4)
        self._stats["cash_usd"] = round(self.cash_balance, 4)
        self._stats["max_drawdown_pct"] = round(self.max_drawdown_pct * 100.0, 4)
        self._stats["pending_positions"] = len(self.open_positions)
        self._stats["win_rate"] = round((wins / resolved), 6) if resolved else 0.0
        self.audit.write_runtime_state(self._runtime_state_payload())

    def _current_equity(self, market_map: dict[str, Market] | None = None) -> float:
        total = self.cash_balance
        for position in self.open_positions.values():
            current_probability = self._last_position_prices.get(position.market_id, position.entry_price)
            if market_map:
                market = market_map.get(position.market_id)
                if market is not None and len(market.outcomes) >= 2:
                    current_probability = float(
                        market.outcomes[0].price if position.trade_side == "yes" else market.outcomes[1].price
                    )
                    self._last_position_prices[position.market_id] = current_probability
            total += position.shares * current_probability
        return total

    def _current_unrealized_pnl(self, market_map: dict[str, Market] | None = None) -> float:
        unrealized = 0.0
        for position in self.open_positions.values():
            current_probability = self._last_position_prices.get(position.market_id, position.entry_price)
            if market_map:
                market = market_map.get(position.market_id)
                if market is not None and len(market.outcomes) >= 2:
                    current_probability = float(
                        market.outcomes[0].price if position.trade_side == "yes" else market.outcomes[1].price
                    )
                    self._last_position_prices[position.market_id] = current_probability
            unrealized += position.shares * current_probability - position.position_size_usdc
        return unrealized

    def _runtime_state_payload(self) -> dict[str, Any]:
        return {
            "saved_at": datetime.now(UTC).isoformat(),
            "cash_balance": self.cash_balance,
            "peak_equity": self.peak_equity,
            "max_drawdown_pct": self.max_drawdown_pct,
            "stats": self._stats,
            "open_positions": [position.to_payload() for position in self.open_positions.values()],
            "completed_event_keys": sorted(self.completed_event_keys),
            "daily_records": self.daily_records,
            "sent_daily_summaries": sorted(self.sent_daily_summaries),
            "recent_trade_events": self._recent_trade_events[-MAX_RECENT_ACTIVITY:],
            "recent_signals": self._recent_signals[-MAX_RECENT_ACTIVITY:],
            "last_position_prices": self._last_position_prices,
        }

    def _load_runtime_state(self) -> None:
        state_path = self.audit.paths["runtime_state"]
        if not state_path.exists():
            current_date = datetime.now(UTC).date()
            initial_previous_date = current_date.fromordinal(current_date.toordinal() - 1).isoformat()
            self.sent_daily_summaries.add(initial_previous_date)
            return
        try:
            payload = json.loads(state_path.read_text())
        except Exception as exc:
            logger.warning("[WEATHER_EDGE_LIVE] Failed to load runtime state: %s", exc)
            return
        self.cash_balance = float(payload.get("cash_balance", self.cash_balance))
        self.peak_equity = float(payload.get("peak_equity", self.peak_equity))
        self.max_drawdown_pct = float(payload.get("max_drawdown_pct", self.max_drawdown_pct))
        self.open_positions = {
            position["event_key"]: WeatherEdgeOpenPosition.from_payload(position)
            for position in payload.get("open_positions") or []
        }
        self.completed_event_keys = set(payload.get("completed_event_keys") or [])
        self.daily_records = payload.get("daily_records") or {}
        self.sent_daily_summaries = set(payload.get("sent_daily_summaries") or [])
        self._recent_trade_events = list(payload.get("recent_trade_events") or [])
        self._recent_signals = list(payload.get("recent_signals") or [])
        self._last_position_prices = {
            str(key): float(value)
            for key, value in (payload.get("last_position_prices") or {}).items()
        }
        stored_stats = payload.get("stats") or {}
        for key in ("entries", "resolved_trades", "wins", "losses", "realized_pnl_usd"):
            if key in stored_stats:
                self._stats[key] = stored_stats[key]

    def _write_runtime_state(self) -> None:
        self.audit.write_runtime_state(self._runtime_state_payload())

    def _daily_summary_payload(self, date_key: str) -> dict[str, Any]:
        record = self._ensure_daily_record(date_key)
        total_trades = int(record["total_trades"])
        wins = int(record["wins"])
        realized_pnl_usdc = round(float(record["realized_pnl_usdc"]), 4)
        unrealized_pnl_usdc = round(float(record["unrealized_pnl_usdc"]), 4)
        return {
            "date": date_key,
            "total_trades": total_trades,
            "wins": wins,
            "losses": int(record["losses"]),
            "win_rate": (wins / total_trades) if total_trades else 0.0,
            "total_pnl_usdc": round(realized_pnl_usdc + unrealized_pnl_usdc, 4),
            "realized_pnl_usdc": realized_pnl_usdc,
            "unrealized_pnl_usdc": unrealized_pnl_usdc,
            "max_drawdown_pct": round(float(record["max_drawdown_pct"]) * 100.0, 4),
            "current_bankroll": round(float(record["current_bankroll"]), 4),
            "pending_positions": int(record["pending_positions"]),
        }

    def _push_recent_trade_event(self, payload: dict[str, Any]) -> None:
        self._recent_trade_events.append(payload)
        self._recent_trade_events = self._recent_trade_events[-MAX_RECENT_ACTIVITY:]

    def _push_recent_signal(self, payload: dict[str, Any]) -> None:
        self._recent_signals.append(payload)
        self._recent_signals = self._recent_signals[-MAX_RECENT_ACTIVITY:]

    def _notify_error(self, message: str) -> None:
        self.audit.log_alert({"message": message})
        self.notifier.send_message(f"Weather Edge Error\n{message}")

    def _resolve_audit_root(self) -> Path:
        base = Path(self.cfg.audit_root).resolve() if self.cfg.audit_root else (LOG_DIR / "comparison" / self.cfg.session_label)
        return base

    @staticmethod
    def _event_key(*, city: str, target_date: str, lead_time_hours: int) -> str:
        return f"{city}:{target_date}:{lead_time_hours}"
