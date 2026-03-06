"""
Runtime paths for persisted state and logs.
"""

from __future__ import annotations

import os
from pathlib import Path


def _resolve_dir(preferred: Path, fallback: Path) -> Path:
    for candidate in (preferred, fallback):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".codex-write-test"
            probe.write_text("ok")
            probe.unlink(missing_ok=True)
            return candidate
        except OSError:
            continue
    return fallback


def _resolve_state_path(preferred: Path, fallback: Path) -> Path:
    for candidate in (preferred, fallback):
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            probe = candidate.parent / ".codex-write-test"
            probe.write_text("ok")
            probe.unlink(missing_ok=True)
            return candidate
        except OSError:
            continue
    return fallback


DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
LOG_DIR = _resolve_dir(Path(os.getenv("LOG_DIR", str(DATA_DIR / "logs"))), Path("logs"))
STATE_PATH = _resolve_state_path(
    Path(os.getenv("STATE_PATH", str(DATA_DIR / "state.json"))),
    Path("state.json"),
)
