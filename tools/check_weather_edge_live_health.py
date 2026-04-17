#!/usr/bin/env python3
"""Check and record the live Oracle weather-edge sleeve health."""

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
DEFAULT_OUTPUT_ROOT = LOG_DIR / "comparison" / "weather_edge_live_monitor"
SCAN_STALE_SECONDS = 70 * 60
DAILY_SUMMARY_STALE_SECONDS = 30 * 60 * 60


@dataclass(slots=True)
class WeatherEdgeSnapshot:
    checked_at: str
    base_url: str
    health_status: str
    scans_completed: int
    candidate_markets: int
    eligible_markets: int
    selected_markets: int
    entries: int
    resolved_trades: int
    wins: int
    losses: int
    pending_positions: int
    bankroll_usd: float
    cash_usd: float
    total_value_usd: float
    total_pnl_usd: float
    total_pnl_pct: float
    max_drawdown_pct: float
    last_scan_at: str | None
    last_entry_at: str | None
    last_resolution_at: str | None
    last_daily_summary_at: str | None
    telegram_enabled: bool
    trade_ledger_csv: str | None
    daily_summary_path: str | None
    runtime_state_path: str | None
    alerts: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the Oracle weather-edge live sleeve and persist monitoring snapshots.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    register_log_namespace(
        root=output_root,
        lane_key="weather_edge_live_monitor",
        label="Weather Edge Live Monitor",
        category="monitoring",
        source="weather_edge_live_monitor",
        description="Hourly API-based monitoring snapshots for the live weather-edge sleeve.",
        paths={
            "latest": output_root / "latest.json",
            "history": output_root / "history.jsonl",
        },
    )

    health_payload = fetch_json(f"{args.base_url.rstrip('/')}/api/health", timeout_seconds=args.timeout_seconds)
    state_payload = fetch_json(f"{args.base_url.rstrip('/')}/api/state", timeout_seconds=args.timeout_seconds)
    snapshot = build_snapshot(
        base_url=args.base_url,
        health_payload=health_payload,
        state_payload=state_payload,
    )

    payload = asdict(snapshot)
    latest_path = output_root / "latest.json"
    history_path = output_root / "history.jsonl"
    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")
    print(json.dumps(payload, indent=2))


def build_snapshot(
    *,
    base_url: str,
    health_payload: dict[str, object],
    state_payload: dict[str, object],
) -> WeatherEdgeSnapshot:
    runtime = (((state_payload or {}).get("strategies") or {}).get("weather_edge_live") or {})
    view = (((state_payload or {}).get("comparison_views") or {}).get("weather_edge_live") or {})
    portfolio = (view.get("portfolio") or {})

    alerts: list[str] = []
    last_scan_at = runtime.get("last_scan_at")
    last_daily_summary_at = runtime.get("last_daily_summary_at")

    if not runtime:
        alerts.append("missing_runtime_state")
    if not view:
        alerts.append("missing_comparison_view")
    if health_payload.get("status") != "ok":
        alerts.append(f"health_not_ok:{health_payload.get('status')}")
    if timestamp_is_stale(last_scan_at, threshold_seconds=SCAN_STALE_SECONDS):
        alerts.append("scan_stale")
    if timestamp_is_stale(last_daily_summary_at, threshold_seconds=DAILY_SUMMARY_STALE_SECONDS):
        alerts.append("daily_summary_stale")
    if not bool(runtime.get("telegram_enabled")):
        alerts.append("telegram_disabled")

    return WeatherEdgeSnapshot(
        checked_at=datetime.now(UTC).isoformat(),
        base_url=base_url,
        health_status=str(health_payload.get("status") or "unknown"),
        scans_completed=int(runtime.get("scans_completed") or 0),
        candidate_markets=int(runtime.get("candidate_markets") or 0),
        eligible_markets=int(runtime.get("eligible_markets") or 0),
        selected_markets=int(runtime.get("selected_markets") or 0),
        entries=int(runtime.get("entries") or 0),
        resolved_trades=int(runtime.get("resolved_trades") or 0),
        wins=int(runtime.get("wins") or 0),
        losses=int(runtime.get("losses") or 0),
        pending_positions=int(runtime.get("pending_positions") or len(portfolio.get("positions") or [])),
        bankroll_usd=float(runtime.get("bankroll_usd") or portfolio.get("starting_capital") or 0.0),
        cash_usd=float(runtime.get("cash_usd") or portfolio.get("cash") or 0.0),
        total_value_usd=float(portfolio.get("total_value") or 0.0),
        total_pnl_usd=float(portfolio.get("total_pnl") or runtime.get("realized_pnl_usd") or 0.0),
        total_pnl_pct=float(portfolio.get("total_pnl_pct") or 0.0),
        max_drawdown_pct=float(runtime.get("max_drawdown_pct") or portfolio.get("max_drawdown") or 0.0),
        last_scan_at=last_scan_at,
        last_entry_at=runtime.get("last_entry_at"),
        last_resolution_at=runtime.get("last_resolution_at"),
        last_daily_summary_at=last_daily_summary_at,
        telegram_enabled=bool(runtime.get("telegram_enabled")),
        trade_ledger_csv=runtime.get("trade_ledger_csv"),
        daily_summary_path=runtime.get("daily_summary_path"),
        runtime_state_path=runtime.get("runtime_state_path"),
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
