#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import PipelineConfig
from data.collector import PolymarketCollector
from data.models import Market
from runtime_paths import LOG_DIR


UTC = timezone.utc
DEFAULT_AUDIT_ROOT = LOG_DIR / "comparison" / "copy_trader_shadow"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Async label job for copy-trader candidate buys. Writes separate candidate_labels.jsonl rows."
    )
    parser.add_argument(
        "--audit-root",
        default=str(DEFAULT_AUDIT_ROOT),
        help="Copy trader audit root containing candidate_buys.jsonl, wallet_sell_events.jsonl, and candidate_labels.jsonl",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of unlabeled candidates to inspect in this run",
    )
    return parser.parse_args()


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).astimezone(UTC)
    except ValueError:
        return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"logged_at": datetime.now(UTC).isoformat(), **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str))
        handle.write("\n")


def build_sell_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("wallet_address") or ""),
            str(row.get("condition_id") or ""),
            str(row.get("token_id") or ""),
        )
        if not all(key):
            continue
        grouped[key].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: parse_iso(str(row.get("activity_timestamp") or "")) or datetime.min.replace(tzinfo=UTC))
    return grouped


def first_sell_after(candidate: dict[str, Any], sell_index: dict[tuple[str, str, str], list[dict[str, Any]]]) -> dict[str, Any] | None:
    key = (
        str(candidate.get("wallet_address") or ""),
        str(candidate.get("condition_id") or ""),
        str(candidate.get("token_id") or ""),
    )
    rows = sell_index.get(key) or []
    if not rows:
        return None
    candidate_at = parse_iso(str(candidate.get("activity_timestamp") or ""))
    if candidate_at is None:
        return None
    sell_times = [parse_iso(str(row.get("activity_timestamp") or "")) or datetime.min.replace(tzinfo=UTC) for row in rows]
    idx = bisect_right(sell_times, candidate_at)
    return rows[idx] if idx < len(rows) else None


def resolved_exit_price(market: Market, token_id: str) -> float | None:
    if market.active and not market.closed:
        return None
    winner = max(market.outcomes, key=lambda outcome: float(outcome.price or 0.0), default=None)
    if winner is None or winner.price is None or float(winner.price) < 0.5:
        return None
    for outcome in market.outcomes:
        if outcome.token_id == token_id:
            return float(outcome.price or 0.0)
    return None


def build_label_row(
    candidate: dict[str, Any],
    *,
    exit_mode: str,
    exit_price: float,
    exit_timestamp: str,
    market_status: str,
) -> dict[str, Any]:
    entry_price = float(candidate.get("wallet_trade_price") or 0.0)
    if entry_price <= 0:
        raise ValueError("candidate has non-positive entry price")
    exit_return_pct = ((exit_price / entry_price) - 1.0) * 100.0
    candidate_at = parse_iso(str(candidate.get("activity_timestamp") or ""))
    exit_at = parse_iso(exit_timestamp)
    hold_minutes = None
    if candidate_at is not None and exit_at is not None:
        hold_minutes = (exit_at - candidate_at).total_seconds() / 60.0
    return {
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "wallet_address": str(candidate.get("wallet_address") or ""),
        "condition_id": str(candidate.get("condition_id") or ""),
        "token_id": str(candidate.get("token_id") or ""),
        "market_slug": str(candidate.get("market_slug") or ""),
        "entry_price": round(entry_price, 6),
        "exit_mode": exit_mode,
        "exit_price": round(exit_price, 6),
        "exit_timestamp": exit_timestamp,
        "hold_minutes": round(hold_minutes, 4) if hold_minutes is not None else None,
        "return_pct": round(exit_return_pct, 6),
        "pnl_per_100usd": round(exit_return_pct, 6),
        "is_win": exit_return_pct > 0,
        "market_status": market_status,
    }


async def run(args: argparse.Namespace) -> int:
    audit_root = Path(args.audit_root).expanduser().resolve()
    candidate_buys_path = audit_root / "candidate_buys.jsonl"
    wallet_sell_events_path = audit_root / "wallet_sell_events.jsonl"
    candidate_labels_path = audit_root / "candidate_labels.jsonl"

    candidates = load_jsonl(candidate_buys_path)
    wallet_sell_events = load_jsonl(wallet_sell_events_path)
    existing_labels = load_jsonl(candidate_labels_path)
    labeled_ids = {
        str(row.get("candidate_id") or "")
        for row in existing_labels
        if row.get("candidate_id")
    }

    sell_index = build_sell_index(wallet_sell_events)
    config = PipelineConfig()
    collector = PolymarketCollector(
        gamma_host=config.api.gamma_host,
        clob_host=config.api.clob_host,
        data_host=config.api.data_host,
    )
    market_cache: dict[str, Market | None] = {}

    labeled_now = 0
    wallet_sell_labels = 0
    resolved_labels = 0
    unresolved = 0
    inspected = 0

    try:
        for candidate in candidates:
            candidate_id = str(candidate.get("candidate_id") or "")
            if not candidate_id or candidate_id in labeled_ids:
                continue
            if str(candidate.get("wallet_side") or "").upper() != "BUY":
                continue
            if float(candidate.get("wallet_trade_price") or 0.0) <= 0:
                continue
            inspected += 1
            if args.limit and inspected > args.limit:
                break

            sell_event = first_sell_after(candidate, sell_index)
            if sell_event is not None:
                label = build_label_row(
                    candidate,
                    exit_mode="wallet_sell",
                    exit_price=float(sell_event.get("wallet_trade_price") or 0.0),
                    exit_timestamp=str(sell_event.get("activity_timestamp") or sell_event.get("observed_at") or ""),
                    market_status="wallet_sell",
                )
                append_jsonl(candidate_labels_path, label)
                labeled_ids.add(candidate_id)
                labeled_now += 1
                wallet_sell_labels += 1
                continue

            slug = str(candidate.get("market_slug") or "")
            market = market_cache.get(slug)
            if slug not in market_cache:
                market = await collector.get_market_by_slug(slug)
                market_cache[slug] = market
            if market is None:
                unresolved += 1
                continue
            exit_price = resolved_exit_price(market, str(candidate.get("token_id") or ""))
            if exit_price is None:
                unresolved += 1
                continue
            label = build_label_row(
                candidate,
                exit_mode="resolved",
                exit_price=exit_price,
                exit_timestamp=str(market.fetched_at.isoformat()),
                market_status="resolved",
            )
            append_jsonl(candidate_labels_path, label)
            labeled_ids.add(candidate_id)
            labeled_now += 1
            resolved_labels += 1
    finally:
        await collector.close()

    print(
        json.dumps(
            {
                "audit_root": str(audit_root),
                "candidate_rows": len(candidates),
                "existing_labels": len(existing_labels),
                "inspected_unlabeled_candidates": inspected,
                "new_labels_written": labeled_now,
                "wallet_sell_labels": wallet_sell_labels,
                "resolved_labels": resolved_labels,
                "still_unresolved": unresolved,
                "candidate_labels_path": str(candidate_labels_path),
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
