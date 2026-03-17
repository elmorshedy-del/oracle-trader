from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from engine.log_namespace import register_log_namespace


UTC = timezone.utc

WEATHER_EDGE_TRADE_LEDGER_FIELDS = [
    "logged_at",
    "trade_id",
    "market_id",
    "city",
    "metric",
    "lead_time_hours",
    "model_probability",
    "market_probability",
    "edge",
    "kelly_fraction",
    "position_size_usdc",
    "entry_timestamp",
    "outcome",
    "pnl_usdc",
    "cumulative_bankroll",
    "exit_timestamp",
]

WEATHER_EDGE_DAILY_SUMMARY_FIELDS = [
    "logged_at",
    "date",
    "total_trades",
    "wins",
    "losses",
    "win_rate",
    "total_pnl_usdc",
    "realized_pnl_usdc",
    "unrealized_pnl_usdc",
    "max_drawdown_pct",
    "current_bankroll",
    "pending_positions",
]


class WeatherEdgeLiveAudit:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.paths = {
            "metadata": self.root / "metadata.json",
            "runtime_state": self.root / "runtime_state.json",
            "trade_events": self.root / "trade_events.jsonl",
            "trade_ledger_jsonl": self.root / "trade_ledger.jsonl",
            "trade_ledger_csv": self.root / "trade_ledger.csv",
            "daily_summary_jsonl": self.root / "daily_summary.jsonl",
            "daily_summary_csv": self.root / "daily_summary.csv",
            "daily_summary_latest": self.root / "daily_summary_latest.json",
            "alerts_jsonl": self.root / "alerts.jsonl",
        }

    def write_metadata(self, payload: dict[str, object]) -> None:
        self.paths["metadata"].write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        register_log_namespace(
            root=self.root,
            lane_key=str(payload.get("strategy") or "weather_edge_live"),
            label=str(payload.get("label") or payload.get("strategy") or "Weather Edge Live"),
            category="weather_edge",
            source=str(payload.get("strategy") or "weather_edge_live"),
            description="Live weather-edge trade, summary, and runtime logs.",
            paths=self.paths,
            extra={
                "view_key": payload.get("view_key"),
                "pair_key": payload.get("pair_key"),
                "session_label": payload.get("session_label"),
            },
        )

    def write_runtime_state(self, payload: dict[str, object]) -> None:
        self.paths["runtime_state"].write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def log_trade_event(self, payload: dict[str, object]) -> None:
        self._append_jsonl(self.paths["trade_events"], payload)

    def log_trade_ledger(self, payload: dict[str, object]) -> None:
        self._append_jsonl(self.paths["trade_ledger_jsonl"], payload)
        self._append_csv(self.paths["trade_ledger_csv"], WEATHER_EDGE_TRADE_LEDGER_FIELDS, payload)

    def log_daily_summary(self, payload: dict[str, object]) -> None:
        self._append_jsonl(self.paths["daily_summary_jsonl"], payload)
        self._append_csv(self.paths["daily_summary_csv"], WEATHER_EDGE_DAILY_SUMMARY_FIELDS, payload)
        self.paths["daily_summary_latest"].write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def log_alert(self, payload: dict[str, object]) -> None:
        self._append_jsonl(self.paths["alerts_jsonl"], payload)

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
        row = {"logged_at": datetime.now(UTC).isoformat(), **payload}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=str))
            handle.write("\n")

    @staticmethod
    def _append_csv(path: Path, fieldnames: list[str], payload: dict[str, object]) -> None:
        write_header = not path.exists() or path.stat().st_size == 0
        row = {"logged_at": datetime.now(UTC).isoformat()}
        for field in fieldnames:
            if field == "logged_at":
                continue
            row[field] = payload.get(field)
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
