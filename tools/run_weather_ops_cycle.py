#!/usr/bin/env python3
"""Hourly ops cycle for weather and crypto shadow monitoring."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.log_namespace import register_log_namespace
from runtime_paths import LOG_DIR


UTC = timezone.utc
DEFAULT_BASE_URL = "https://just-grace-production-a401.up.railway.app"
DEFAULT_OUTPUT_ROOT = LOG_DIR / "comparison" / "ops_cycle_monitor"
SNAPSHOT_UTC_HOURS = {0, 6, 12, 18}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the hourly Oracle ops cycle without using Codex automation.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--skip-snapshot", action="store_true")
    parser.add_argument("--force-snapshot", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checked_at = datetime.now(UTC)
    should_run_snapshot = args.force_snapshot or (not args.skip_snapshot and checked_at.hour in SNAPSHOT_UTC_HOURS)
    output_root = resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    register_log_namespace(
        root=output_root,
        lane_key="weather_ops_cycle_monitor",
        label="Weather Ops Cycle Monitor",
        category="monitoring",
        source="weather_ops_cycle_monitor",
        description="Hourly Railway cron summaries covering weather-edge live, crypto-pairs live, and scheduled weather snapshot runs.",
        paths={
            "latest": output_root / "latest.json",
            "history": output_root / "history.jsonl",
        },
    )

    summary = {
        "checked_at": checked_at.isoformat(),
        "base_url": args.base_url,
        "weather_edge_live": run_json_command(
            "check_weather_edge_live_health.py",
            "--base-url",
            args.base_url,
        ),
        "crypto_pairs": run_json_command(
            "check_crypto_pairs_shadow_health.py",
            "--base-url",
            args.base_url,
        ),
        "snapshot_recorder_ran": should_run_snapshot,
        "snapshot_recorder": None,
    }

    if should_run_snapshot:
        summary["snapshot_recorder"] = run_json_command("run_weather_snapshot_recorder.py")

    latest_path = output_root / "latest.json"
    history_path = output_root / "history.jsonl"
    latest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, sort_keys=True))
        handle.write("\n")

    print(json.dumps(summary, indent=2))
    return 0


def run_json_command(script_name: str, *script_args: str) -> dict[str, object]:
    command = [sys.executable, str(REPO_ROOT / "tools" / script_name), *script_args]
    try:
        output = subprocess.check_output(command, text=True)
    except subprocess.CalledProcessError as exc:
        return {
            "status": "error",
            "script": script_name,
            "returncode": exc.returncode,
            "stdout": exc.stdout,
            "stderr": exc.stderr,
        }
    return json.loads(output)


def resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
