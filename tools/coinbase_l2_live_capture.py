#!/usr/bin/env python3
"""
Capture public Coinbase BTC-USD market data for cross-venue BTC research.

Channels:
- level2_batch
- ticker
- heartbeat
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import websockets


BASE_WS_URL = "wss://ws-feed.exchange.coinbase.com"
DEFAULT_OUTPUT_ROOT = Path("output/btc_multivenue_capture/coinbase")
DEFAULT_DURATION_SECONDS = 1800
UTC = timezone.utc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture public Coinbase market data to local JSONL files.")
    parser.add_argument("--product-id", default="BTC-USD", help="Coinbase product id, default BTC-USD")
    parser.add_argument("--duration-seconds", type=int, default=DEFAULT_DURATION_SECONDS, help="How long to capture before stopping")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Local output root")
    return parser.parse_args()


def file_handles(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    handles: dict[str, Any] = {}
    for key in ("level2", "ticker", "heartbeat", "subscriptions", "raw"):
        handles[key] = (output_dir / f"{key}.jsonl").open("a", encoding="utf-8")
    return handles


def route_message(payload: dict[str, Any]) -> str:
    message_type = str(payload.get("type") or "").lower()
    if message_type in ("snapshot", "l2update"):
        return "level2"
    if message_type == "ticker":
        return "ticker"
    if message_type == "heartbeat":
        return "heartbeat"
    if message_type == "subscriptions":
        return "subscriptions"
    return "raw"


async def capture(args: argparse.Namespace) -> None:
    product_id = args.product_id.upper()
    started_at = datetime.now(UTC)
    output_dir = Path(args.output_root).resolve() / product_id / started_at.strftime("%Y-%m-%d")
    handles = file_handles(output_dir)
    counts = defaultdict(int)
    stop_event = asyncio.Event()
    reconnects = 0
    last_error: str | None = None

    def _handle_stop(*_: Any) -> None:
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(sig, _handle_stop)
        except NotImplementedError:
            signal.signal(sig, lambda *_args: stop_event.set())

    subscribe_payload = {
        "type": "subscribe",
        "product_ids": [product_id],
        "channels": ["level2_batch", "ticker", "heartbeat"],
    }
    deadline = asyncio.get_running_loop().time() + args.duration_seconds

    try:
        while not stop_event.is_set():
            if asyncio.get_running_loop().time() >= deadline:
                break
            try:
                async with websockets.connect(
                    BASE_WS_URL,
                    ping_interval=20,
                    ping_timeout=20,
                    max_queue=10000,
                    max_size=None,
                ) as websocket:
                    await websocket.send(json.dumps(subscribe_payload))
                    while not stop_event.is_set() and asyncio.get_running_loop().time() < deadline:
                        raw_message = await asyncio.wait_for(websocket.recv(), timeout=30)
                        payload = json.loads(raw_message)
                        record = {
                            "captured_at": datetime.now(UTC).isoformat(),
                            "product_id": product_id,
                            "data": payload,
                        }
                        bucket = route_message(payload)
                        handles[bucket].write(json.dumps(record, separators=(",", ":")) + "\n")
                        handles["raw"].write(json.dumps(record, separators=(",", ":")) + "\n")
                        counts[bucket] += 1
            except Exception as exc:
                reconnects += 1
                last_error = str(exc)
                await asyncio.sleep(2)
    finally:
        ended_at = datetime.now(UTC)
        summary = {
            "product_id": product_id,
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_seconds": args.duration_seconds,
            "channels": subscribe_payload["channels"],
            "counts": dict(counts),
            "reconnects": reconnects,
            "last_error": last_error,
            "output_dir": str(output_dir),
        }
        (output_dir / "capture_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        for handle in handles.values():
            handle.close()
        print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    asyncio.run(capture(args))


if __name__ == "__main__":
    main()
