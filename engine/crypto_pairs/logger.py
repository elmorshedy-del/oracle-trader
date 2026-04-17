"""Structured JSON logging for crypto pairs runtime sessions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


UTC = timezone.utc


class CryptoPairsLogger:
    def __init__(self, session_root: Path):
        self.session_root = session_root
        self.session_root.mkdir(parents=True, exist_ok=True)
        self.paths = {
            "ratio_ticks": self.session_root / "ratio_ticks.jsonl",
            "signals": self.session_root / "signals.jsonl",
            "trade_events": self.session_root / "trade_events.jsonl",
            "pair_health": self.session_root / "pair_health.jsonl",
            "summary": self.session_root / "summary.json",
        }
        self.summary = {
            "started_at": datetime.now(UTC).isoformat(),
            "ratio_ticks": 0,
            "signals": 0,
            "trade_events": 0,
            "pair_health_events": 0,
        }
        self.flush_summary()

    def log_ratio_tick(self, payload: dict[str, object]) -> None:
        self._append("ratio_ticks", payload)
        self.summary["ratio_ticks"] += 1

    def log_signal(self, payload: dict[str, object]) -> None:
        self._append("signals", payload)
        self.summary["signals"] += 1

    def log_trade_event(self, payload: dict[str, object]) -> None:
        self._append("trade_events", payload)
        self.summary["trade_events"] += 1

    def log_pair_health(self, payload: dict[str, object]) -> None:
        self._append("pair_health", payload)
        self.summary["pair_health_events"] += 1

    def flush_summary(self, **extra: object) -> None:
        self.summary.update(extra)
        self.summary["updated_at"] = datetime.now(UTC).isoformat()
        self.paths["summary"].write_text(json.dumps(self.summary, indent=2), encoding="utf-8")

    def _append(self, key: str, payload: dict[str, object]) -> None:
        payload = {"logged_at": datetime.now(UTC).isoformat(), **payload}
        with self.paths[key].open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")

