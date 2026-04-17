from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from engine.log_namespace import register_log_namespace


UTC = timezone.utc


class ShadowSleeveAudit:
    def __init__(
        self,
        root: Path,
        *,
        lane_key: str,
        label: str,
        category: str,
        source: str,
        description: str,
        trade_ledger_fields: list[str],
        daily_summary_fields: list[str],
        extra_jsonl_keys: tuple[str, ...] = (),
    ):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.lane_key = lane_key
        self.label = label
        self.category = category
        self.source = source
        self.description = description
        self.trade_ledger_fields = list(trade_ledger_fields)
        self.daily_summary_fields = list(daily_summary_fields)
        self.paths = {
            "metadata": self.root / "metadata.json",
            "runtime_state": self.root / "runtime_state.json",
            "signals": self.root / "signals.jsonl",
            "trade_events": self.root / "trade_events.jsonl",
            "trade_ledger_jsonl": self.root / "trade_ledger.jsonl",
            "trade_ledger_csv": self.root / "trade_ledger.csv",
            "daily_summary_jsonl": self.root / "daily_summary.jsonl",
            "daily_summary_csv": self.root / "daily_summary.csv",
            "daily_summary_latest": self.root / "daily_summary_latest.json",
        }
        for key in extra_jsonl_keys:
            self.paths[key] = self.root / f"{key}.jsonl"

    def write_metadata(self, payload: dict[str, object], *, extra: dict[str, object] | None = None) -> None:
        data = {
            "lane_key": self.lane_key,
            "label": self.label,
            "category": self.category,
            "source": self.source,
            "description": self.description,
            **payload,
        }
        self.paths["metadata"].write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        register_log_namespace(
            root=self.root,
            lane_key=self.lane_key,
            label=self.label,
            category=self.category,
            source=self.source,
            description=self.description,
            paths=self.paths,
            extra=extra or {},
        )

    def write_runtime_state(self, payload: dict[str, object]) -> None:
        self.paths["runtime_state"].write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def log_signal(self, payload: dict[str, object]) -> None:
        self._append_jsonl(self.paths["signals"], payload)

    def log_trade_event(self, payload: dict[str, object]) -> None:
        self._append_jsonl(self.paths["trade_events"], payload)

    def log_trade_ledger(self, payload: dict[str, object]) -> None:
        self._append_jsonl(self.paths["trade_ledger_jsonl"], payload)
        self._append_csv(self.paths["trade_ledger_csv"], self.trade_ledger_fields, payload)

    def log_daily_summary(self, payload: dict[str, object]) -> None:
        self._append_jsonl(self.paths["daily_summary_jsonl"], payload)
        self._append_csv(self.paths["daily_summary_csv"], self.daily_summary_fields, payload)
        self.paths["daily_summary_latest"].write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def log_extra(self, key: str, payload: dict[str, object]) -> None:
        path = self.paths.get(key)
        if path is None:
            raise KeyError(f"Unknown extra audit key: {key}")
        self._append_jsonl(path, payload)

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
