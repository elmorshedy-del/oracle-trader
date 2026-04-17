#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.weather_snapshot_recorder import build_default_snapshot_recorder


async def _run() -> int:
    recorder = await build_default_snapshot_recorder()
    try:
        summary = await recorder.run()
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        recorder.notifier.send_message(f"Weather Snapshot Recorder Fatal Error\n{exc}")
        raise
    finally:
        await recorder.collector.close()


def main() -> int:
    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
