#!/usr/bin/env python3
"""
Start a separate BTC multivenue capture session.

This keeps the next-track data separate from all prior futures-only work.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_DURATION_SECONDS = 1800
DEFAULT_OUTPUT_ROOT = Path("output/btc_multivenue_capture")
UTC = timezone.utc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BTC multivenue capture suite in a separate frozen track.")
    parser.add_argument("--duration-seconds", type=int, default=DEFAULT_DURATION_SECONDS, help="Capture length for each child collector")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Root for capture sessions")
    parser.add_argument("--run-label", default="multivenue_v1", help="Human-readable run label")
    parser.add_argument("--symbol", default="BTCUSDT", help="Binance symbol")
    parser.add_argument("--product-id", default="BTC-USD", help="Coinbase product id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = datetime.now(UTC)
    session_name = f"{started_at.strftime('%Y%m%d_%H%M%S')}_{args.run_label}"
    session_root = Path(args.output_root).resolve() / "sessions" / session_name
    session_root.mkdir(parents=True, exist_ok=False)

    python_bin = sys.executable
    tools_root = Path(__file__).resolve().parent

    commands = {
        "binance_futures": [
            python_bin,
            str(tools_root / "binance_futures_live_capture.py"),
            "--symbol",
            args.symbol,
            "--duration-seconds",
            str(args.duration_seconds),
            "--output-root",
            str(session_root / "binance_futures"),
            "--depth-mode",
            "partial",
            "--depth-levels",
            "20",
        ],
        "binance_spot": [
            python_bin,
            str(tools_root / "binance_spot_live_capture.py"),
            "--symbol",
            args.symbol,
            "--duration-seconds",
            str(args.duration_seconds),
            "--output-root",
            str(session_root / "binance_spot"),
            "--depth-mode",
            "partial",
            "--depth-levels",
            "20",
        ],
        "coinbase": [
            python_bin,
            str(tools_root / "coinbase_l2_live_capture.py"),
            "--product-id",
            args.product_id,
            "--duration-seconds",
            str(args.duration_seconds),
            "--output-root",
            str(session_root / "coinbase"),
        ],
    }

    children: dict[str, subprocess.Popen[str]] = {}
    for name, command in commands.items():
        log_path = session_root / f"{name}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        children[name] = subprocess.Popen(
            command,
            cwd=str(tools_root.parent),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

    stop_requested = False

    def handle_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True
        for child in children.values():
            if child.poll() is None:
                child.terminate()

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    while True:
        alive = [child for child in children.values() if child.poll() is None]
        if not alive:
            break
        time.sleep(1)
        if stop_requested:
            break

    ended_at = datetime.now(UTC)
    summary = {
        "run_label": args.run_label,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "duration_seconds": args.duration_seconds,
        "session_root": str(session_root),
        "commands": commands,
        "children": {
            name: {
                "pid": child.pid,
                "returncode": child.poll(),
                "log_path": str(session_root / f"{name}.log"),
            }
            for name, child in children.items()
        },
    }
    (session_root / "session_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
