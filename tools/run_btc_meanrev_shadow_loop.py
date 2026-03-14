#!/usr/bin/env python3
"""
Forward shadow runner for the frozen BTC downshock mean-reversion candidate.

This keeps the research model out of the legacy trading sleeve while still
deploying it on future live data:
- each cycle runs the frozen validation capture + replay unchanged
- results append to the existing BTC validation history and checkpoint ledger
- no tuning or parameter changes happen inside this loop
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


UTC = timezone.utc
DEFAULT_ANALYSIS_PYTHON = Path("/Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14")
DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SLEEP_SECONDS = 23 * 60 * 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen BTC mean-reversion validation cycle repeatedly as a shadow forward monitor.")
    parser.add_argument(
        "--analysis-python",
        default=str(DEFAULT_ANALYSIS_PYTHON),
        help="Python executable used for the validation cycle",
    )
    parser.add_argument(
        "--repo-root",
        default=str(DEFAULT_REPO_ROOT),
        help="Oracle repository root",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Number of validation cycles to run. Use 0 for infinite loop.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=int,
        default=DEFAULT_SLEEP_SECONDS,
        help="Sleep interval between completed cycles",
    )
    parser.add_argument(
        "--capture-duration-seconds",
        type=int,
        default=3600,
        help="Length of each frozen validation capture session",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    python_bin = Path(args.analysis_python).resolve()
    cycle_script = repo_root / "tools" / "run_btc_meanrev_validation_cycle.py"
    if not cycle_script.exists():
        raise SystemExit(f"Missing validation cycle script: {cycle_script}")

    completed = 0
    while True:
        started_at = datetime.now(UTC).isoformat()
        command = [
            str(python_bin),
            str(cycle_script),
            "--capture-duration-seconds",
            str(args.capture_duration_seconds),
        ]
        print(f"[BTC_SHADOW] starting cycle at {started_at}", flush=True)
        subprocess.check_call(command, cwd=str(repo_root))
        completed += 1
        finished_at = datetime.now(UTC).isoformat()
        print(f"[BTC_SHADOW] completed cycle {completed} at {finished_at}", flush=True)

        if args.cycles > 0 and completed >= args.cycles:
            break

        time.sleep(max(args.sleep_seconds, 1))


if __name__ == "__main__":
    main()
