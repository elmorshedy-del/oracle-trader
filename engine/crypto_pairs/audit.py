"""Durable audit logging for live crypto-pairs shadow sleeves."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


UTC = timezone.utc


class CryptoPairsAudit:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.paths = {
            "metadata": self.root / "metadata.json",
            "runtime_state": self.root / "runtime_state.json",
            "signals": self.root / "signals.jsonl",
            "trade_events": self.root / "trade_events.jsonl",
            "trade_ledger_jsonl": self.root / "trade_ledger.jsonl",
            "trade_ledger_csv": self.root / "trade_ledger.csv",
            "ratio_ticks_jsonl": self.root / "ratio_ticks.jsonl",
            "ratio_ticks_csv": self.root / "ratio_ticks.csv",
            "daily_summary_jsonl": self.root / "daily_summary.jsonl",
            "daily_summary_latest": self.root / "daily_summary_latest.json",
            "hourly_checks_jsonl": self.root / "hourly_checks.jsonl",
            "hourly_checks_latest": self.root / "hourly_checks_latest.json",
        }

    def write_metadata(self, payload: dict[str, object]) -> None:
        self.paths["metadata"].write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def write_runtime_state(self, payload: dict[str, object]) -> None:
        self.paths["runtime_state"].write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def log_signal(self, payload: dict[str, object]) -> None:
        self._append_jsonl(self.paths["signals"], payload)

    def log_trade_event(self, payload: dict[str, object]) -> None:
        self._append_jsonl(self.paths["trade_events"], payload)

    def log_trade_ledger(self, payload: dict[str, object]) -> None:
        self._append_jsonl(self.paths["trade_ledger_jsonl"], payload)
        self._append_csv(
            self.paths["trade_ledger_csv"],
            [
                "logged_at",
                "pair",
                "entry_timestamp",
                "entry_zscore",
                "entry_price_a",
                "entry_price_b",
                "entry_ratio",
                "direction",
                "position_size_per_leg_usdt",
                "exit_timestamp",
                "exit_zscore",
                "exit_price_a",
                "exit_price_b",
                "exit_ratio",
                "exit_reason",
                "gross_pnl_bps",
                "fees_usd",
                "slippage_usd",
                "slippage_bps",
                "net_pnl_bps",
                "net_pnl_usdt",
                "hold_seconds",
            ],
            payload,
        )

    def log_ratio_tick(self, payload: dict[str, object]) -> None:
        self._append_jsonl(self.paths["ratio_ticks_jsonl"], payload)
        self._append_csv(
            self.paths["ratio_ticks_csv"],
            [
                "logged_at",
                "timestamp",
                "pair",
                "price_a",
                "price_b",
                "ratio",
                "zscore",
                "rolling_mean",
                "rolling_std",
                "ready",
            ],
            payload,
        )

    def log_daily_summary(self, payload: dict[str, object]) -> None:
        self._append_jsonl(self.paths["daily_summary_jsonl"], payload)
        self.paths["daily_summary_latest"].write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def log_hourly_check(self, payload: dict[str, object]) -> None:
        self._append_jsonl(self.paths["hourly_checks_jsonl"], payload)
        self.paths["hourly_checks_latest"].write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

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
