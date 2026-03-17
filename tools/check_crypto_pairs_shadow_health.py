#!/usr/bin/env python3
"""Check and record the live Oracle crypto-pairs shadow sleeve health."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.log_namespace import register_log_namespace
from runtime_paths import LOG_DIR


UTC = timezone.utc
DEFAULT_BASE_URL = "https://just-grace-production-a401.up.railway.app"
DEFAULT_OUTPUT_ROOT = LOG_DIR / "comparison" / "crypto_pairs_shadow_monitor"
RATIO_STALE_SECONDS = 20 * 60
HOURLY_STALE_SECONDS = 70 * 60
COINTEGRATION_FAIL_THRESHOLD = 0.05


@dataclass(frozen=True, slots=True)
class MonitorTarget:
    slug: str
    label: str
    view_key: str
    strategy_key: str


DEFAULT_TARGETS = (
    MonitorTarget(
        slug="aave_doge",
        label="AAVE/DOGE",
        view_key="crypto_pairs_aave_doge",
        strategy_key="crypto_pairs_shadow",
    ),
    MonitorTarget(
        slug="comp_floki",
        label="COMP/FLOKI",
        view_key="crypto_pairs_comp_floki",
        strategy_key="crypto_pairs_shadow_comp_floki",
    ),
    MonitorTarget(
        slug="comp_link",
        label="COMP/LINK",
        view_key="crypto_pairs_comp_link",
        strategy_key="crypto_pairs_shadow_comp_link",
    ),
)


@dataclass(slots=True)
class CheckSnapshot:
    checked_at: str
    base_url: str
    slug: str
    label: str
    view_key: str
    strategy_key: str
    health_status: str
    ratio_updates: int
    entry_signals: int
    blocked_entry_signals: int
    entries: int
    closed_trades: int
    realized_net_bps: float
    realized_pnl_usd: float
    current_cointegration_pvalue: float | None
    last_ratio_tick_at: str | None
    last_trade_at: str | None
    last_hourly_check_at: str | None
    ratio_ticks_csv: str | None
    trade_ledger_csv: str | None
    daily_summary_path: str | None
    hourly_checks_path: str | None
    positions_count: int
    alerts: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the Oracle crypto-pairs shadow sleeves and persist monitoring snapshots.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--view-key", help="Optional single-target comparison view key.")
    parser.add_argument("--strategy-key", help="Optional single-target strategy key.")
    parser.add_argument("--label", help="Optional single-target label.")
    parser.add_argument("--slug", help="Optional single-target output slug.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    register_log_namespace(
        root=output_root,
        lane_key="crypto_pairs_shadow_monitor",
        label="Crypto Pairs Shadow Monitor",
        category="monitoring",
        source="crypto_pairs_shadow_monitor",
        description="Hourly API-based monitoring snapshots for live crypto-pairs shadow sleeves.",
        paths={
            "latest": output_root / "latest.json",
            "history": output_root / "history.jsonl",
        },
    )

    health_payload = fetch_json(f"{args.base_url.rstrip('/')}/api/health", timeout_seconds=args.timeout_seconds)
    state_payload = fetch_json(f"{args.base_url.rstrip('/')}/api/state", timeout_seconds=args.timeout_seconds)
    snapshots = [
        build_snapshot(
            target=target,
            base_url=args.base_url,
            health_payload=health_payload,
            state_payload=state_payload,
        )
        for target in resolve_targets(args)
    ]

    for snapshot in snapshots:
        append_snapshot(output_root / snapshot_slug(snapshot), snapshot)

    combined_payload = {
        "checked_at": datetime.now(UTC).isoformat(),
        "base_url": args.base_url,
        "health_status": str(health_payload.get("status") or "unknown"),
        "targets": [asdict(snapshot) for snapshot in snapshots],
        "alerts": {snapshot.label: list(snapshot.alerts) for snapshot in snapshots if snapshot.alerts},
    }
    append_combined_snapshot(output_root, combined_payload)
    print(json.dumps(combined_payload, indent=2))


def resolve_targets(args: argparse.Namespace) -> tuple[MonitorTarget, ...]:
    if args.view_key or args.strategy_key or args.label or args.slug:
        return (
            MonitorTarget(
                slug=args.slug or normalize_slug(args.label or args.view_key or args.strategy_key or "crypto_pairs"),
                label=args.label or args.view_key or args.strategy_key or "Crypto Pairs",
                view_key=args.view_key or "crypto_pairs_aave_doge",
                strategy_key=args.strategy_key or "crypto_pairs_shadow",
            ),
        )
    return DEFAULT_TARGETS


def build_snapshot(
    *,
    target: MonitorTarget,
    base_url: str,
    health_payload: dict[str, object],
    state_payload: dict[str, object],
) -> CheckSnapshot:
    runtime = (((state_payload or {}).get("strategies") or {}).get(target.strategy_key) or {})
    view = (((state_payload or {}).get("comparison_views") or {}).get(target.view_key) or {})
    portfolio = (view.get("portfolio") or {})

    alerts: list[str] = []
    last_ratio_tick_at = runtime.get("last_ratio_tick_at")
    last_hourly_check_at = runtime.get("last_hourly_check_at")
    last_trade_at = runtime.get("last_trade_at")
    cointegration_pvalue = runtime.get("current_cointegration_pvalue")

    if not runtime:
        alerts.append("missing_runtime_state")
    if not view:
        alerts.append("missing_comparison_view")
    if health_payload.get("status") != "ok":
        alerts.append(f"health_not_ok:{health_payload.get('status')}")
    if timestamp_is_stale(last_ratio_tick_at, threshold_seconds=RATIO_STALE_SECONDS):
        alerts.append("ratio_ticks_stale")
    if timestamp_is_stale(last_hourly_check_at, threshold_seconds=HOURLY_STALE_SECONDS):
        alerts.append("hourly_check_stale")
    if cointegration_pvalue is not None and float(cointegration_pvalue) > COINTEGRATION_FAIL_THRESHOLD:
        alerts.append(f"cointegration_broken:{cointegration_pvalue}")

    return CheckSnapshot(
        checked_at=datetime.now(UTC).isoformat(),
        base_url=base_url,
        slug=target.slug,
        label=target.label,
        view_key=target.view_key,
        strategy_key=target.strategy_key,
        health_status=str(health_payload.get("status") or "unknown"),
        ratio_updates=int(runtime.get("ratio_updates") or 0),
        entry_signals=int(runtime.get("entry_signals") or 0),
        blocked_entry_signals=int(runtime.get("blocked_entry_signals") or 0),
        entries=int(runtime.get("entries") or 0),
        closed_trades=int(runtime.get("closed_trades") or 0),
        realized_net_bps=float(runtime.get("realized_net_bps") or 0.0),
        realized_pnl_usd=float(runtime.get("realized_pnl_usd") or 0.0),
        current_cointegration_pvalue=float(cointegration_pvalue) if cointegration_pvalue is not None else None,
        last_ratio_tick_at=last_ratio_tick_at,
        last_trade_at=last_trade_at,
        last_hourly_check_at=last_hourly_check_at,
        ratio_ticks_csv=runtime.get("ratio_ticks_csv"),
        trade_ledger_csv=runtime.get("trade_ledger_csv"),
        daily_summary_path=runtime.get("daily_summary_path"),
        hourly_checks_path=runtime.get("hourly_checks_path"),
        positions_count=len(portfolio.get("positions") or []),
        alerts=alerts,
    )


def resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def fetch_json(url: str, *, timeout_seconds: float) -> dict[str, object]:
    try:
        output = subprocess.check_output(
            ["curl", "-fsSL", "--max-time", str(int(timeout_seconds)), url],
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Network error fetching {url}: curl exited with {exc.returncode}") from exc
    return json.loads(output)


def append_snapshot(output_root: Path, snapshot: CheckSnapshot) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = asdict(snapshot)
    latest_path = output_root / "latest.json"
    history_path = output_root / "history.jsonl"
    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def append_combined_snapshot(output_root: Path, payload: dict[str, object]) -> None:
    latest_path = output_root / "latest.json"
    history_path = output_root / "history.jsonl"
    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def snapshot_slug(snapshot: CheckSnapshot) -> str:
    return snapshot.slug


def normalize_slug(raw: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in raw).strip("_")


def timestamp_is_stale(raw: str | None, *, threshold_seconds: int) -> bool:
    if not raw:
        return True
    try:
        timestamp = datetime.fromisoformat(raw)
    except ValueError:
        return True
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return (datetime.now(UTC) - timestamp.astimezone(UTC)).total_seconds() > threshold_seconds


if __name__ == "__main__":
    main()
