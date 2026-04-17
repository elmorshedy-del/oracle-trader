#!/usr/bin/env python3
"""
Capture live Binance USD-M futures market streams for local enrichment.

Streams collected:
- aggTrade
- bookTicker
- depth@100ms
- markPrice@1s
- forceOrder
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


BASE_WS_URL = "wss://fstream.binance.com/stream?streams="
DEFAULT_OUTPUT_ROOT = Path("output/futures_ml_live")
DEFAULT_DURATION_SECONDS = 1800
VALID_PARTIAL_DEPTH_LEVELS = (5, 10, 20)
VALID_DEPTH_MODES = ("diff", "partial")
UTC = timezone.utc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture live Binance USD-M futures streams to local JSONL files.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Futures symbol, default BTCUSDT")
    parser.add_argument("--duration-seconds", type=int, default=DEFAULT_DURATION_SECONDS, help="How long to capture before stopping")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Local output root")
    parser.add_argument("--include-agg-trade", action="store_true", default=True, help="Capture aggTrade stream")
    parser.add_argument("--include-book-ticker", action="store_true", default=True, help="Capture bookTicker stream")
    parser.add_argument("--include-depth", action="store_true", default=True, help="Capture diff depth stream at 100ms")
    parser.add_argument(
        "--depth-mode",
        choices=VALID_DEPTH_MODES,
        default="diff",
        help="Choose raw diff depth or partial top-of-book snapshots for the depth stream",
    )
    parser.add_argument(
        "--depth-levels",
        type=int,
        default=20,
        help="When depth-mode=partial, capture this many top levels per side",
    )
    parser.add_argument("--include-mark-price", action="store_true", default=True, help="Capture markPrice stream")
    parser.add_argument("--include-force-order", action="store_true", default=True, help="Capture liquidation stream")
    args = parser.parse_args()
    if args.depth_mode == "partial" and args.depth_levels not in VALID_PARTIAL_DEPTH_LEVELS:
        raise SystemExit(f"depth-levels must be one of {VALID_PARTIAL_DEPTH_LEVELS} when depth-mode=partial")
    return args


def selected_streams(args: argparse.Namespace, symbol: str) -> list[str]:
    base = symbol.lower()
    streams: list[str] = []
    if args.include_agg_trade:
        streams.append(f"{base}@aggTrade")
    if args.include_book_ticker:
        streams.append(f"{base}@bookTicker")
    if args.include_depth:
        if args.depth_mode == "partial":
            streams.append(f"{base}@depth{args.depth_levels}@100ms")
        else:
            streams.append(f"{base}@depth@100ms")
    if args.include_mark_price:
        streams.append(f"{base}@markPrice@1s")
    if args.include_force_order:
        streams.append(f"{base}@forceOrder")
    return streams


def file_handles(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    handles: dict[str, Any] = {}
    for key in ("aggTrade", "bookTicker", "depth", "markPrice", "forceOrder", "raw"):
        handles[key] = (output_dir / f"{key}.jsonl").open("a", encoding="utf-8")
    return handles


def route_stream(stream_name: str) -> str:
    lowered = stream_name.lower()
    if "@aggtrade" in lowered:
        return "aggTrade"
    if "@bookticker" in lowered:
        return "bookTicker"
    if "@depth" in lowered:
        return "depth"
    if "@markprice" in lowered:
        return "markPrice"
    if "@forceorder" in lowered:
        return "forceOrder"
    return "raw"


async def capture(args: argparse.Namespace) -> None:
    symbol = args.symbol.upper()
    streams = selected_streams(args, symbol)
    if not streams:
        raise SystemExit("No streams selected")

    started_at = datetime.now(UTC)
    output_dir = Path(args.output_root).resolve() / symbol / started_at.strftime("%Y-%m-%d")
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

    ws_url = BASE_WS_URL + "/".join(streams)
    deadline = asyncio.get_running_loop().time() + args.duration_seconds

    try:
        while not stop_event.is_set():
            if asyncio.get_running_loop().time() >= deadline:
                break
            try:
                async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20, max_queue=10000) as websocket:
                    while not stop_event.is_set() and asyncio.get_running_loop().time() < deadline:
                        raw_message = await asyncio.wait_for(websocket.recv(), timeout=30)
                        payload = json.loads(raw_message)
                        stream_name = str(payload.get("stream") or "")
                        data = payload.get("data") or {}
                        record = {
                            "captured_at": datetime.now(UTC).isoformat(),
                            "stream": stream_name,
                            "data": data,
                        }
                        bucket = route_stream(stream_name)
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
            "symbol": symbol,
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_seconds": args.duration_seconds,
            "streams": streams,
            "depth_mode": args.depth_mode if args.include_depth else None,
            "depth_levels": args.depth_levels if args.include_depth and args.depth_mode == "partial" else None,
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
