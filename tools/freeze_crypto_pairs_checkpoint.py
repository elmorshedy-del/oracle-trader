#!/usr/bin/env python3
"""
Append-only checkpoint helper for crypto pairs research.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


RESEARCH_ROOT = Path("research/crypto_pairs")
CHECKPOINT_INDEX = RESEARCH_ROOT / "checkpoints" / "index.json"
CHECKPOINTS_ROOT = RESEARCH_ROOT / "checkpoints"
UTC = timezone.utc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze a crypto pairs research checkpoint into the bookkeeping ledger.")
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--metric", action="append", default=[])
    parser.add_argument("--note-file")
    return parser.parse_args()


def parse_pairs(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        key, sep, raw = value.partition("=")
        if not sep:
            raise SystemExit(f"Expected key=value pair, got: {value}")
        parsed[key.strip()] = raw.strip()
    return parsed


def parse_metrics(values: list[str]) -> dict[str, float | int | str]:
    metrics: dict[str, float | int | str] = {}
    for key, raw in parse_pairs(values).items():
        try:
            metrics[key] = float(raw) if "." in raw else int(raw)
        except ValueError:
            metrics[key] = raw
    return metrics


def main() -> None:
    args = parse_args()
    CHECKPOINTS_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = CHECKPOINTS_ROOT / args.checkpoint_id
    if checkpoint_dir.exists():
        raise SystemExit(f"Checkpoint already exists: {checkpoint_dir}")
    checkpoint_dir.mkdir(parents=True, exist_ok=False)

    note_copy = None
    if args.note_file:
        source = Path(args.note_file).resolve()
        if not source.exists():
            raise SystemExit(f"Missing note file: {source}")
        note_copy = checkpoint_dir / source.name
        shutil.copy2(source, note_copy)

    entry = {
        "checkpoint_id": args.checkpoint_id,
        "created_at": datetime.now(UTC).isoformat(),
        "status": args.status,
        "category": args.category,
        "summary": args.summary,
        "artifacts": parse_pairs(args.artifact),
        "metrics": parse_metrics(args.metric),
    }
    if note_copy is not None:
        entry["artifacts"]["note_copy"] = str(note_copy.resolve())

    manifest_path = checkpoint_dir / "manifest.json"
    manifest_path.write_text(json.dumps(entry, indent=2), encoding="utf-8")

    if CHECKPOINT_INDEX.exists():
        index = json.loads(CHECKPOINT_INDEX.read_text(encoding="utf-8"))
        if not isinstance(index, list):
            raise SystemExit(f"Checkpoint index is not a list: {CHECKPOINT_INDEX}")
    else:
        index = []
    index.append(entry)
    CHECKPOINT_INDEX.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
