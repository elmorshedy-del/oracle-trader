#!/usr/bin/env python3
"""Check and record the live Oracle AAVE/DOGE shadow sleeve health."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime_paths import LOG_DIR


UTC = timezone.utc
DEFAULT_BASE_URL = "https://just-grace-production-a401.up.railway.app"
DEFAULT_OUTPUT_ROOT = LOG_DIR / "comparison" / "crypto_pairs_aave_doge_monitor"
RATIO_STALE_SECONDS = 20 * 60
HOURLY_STALE_SECONDS = 70 * 60
COINTEGRATION_FAIL_THRESHOLD = 0.05


@dataclass(slots=True)
class CheckSnapshot:
    checked_at: str
    base_url: str
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
    parser = argparse.ArgumentParser(description="Check the Oracle AAVE/DOGE shadow sleeve and persist a monitoring snapshot.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    health_payload = fetch_json(f"{args.base_url.rstrip('/')}/api/health", timeout_seconds=args.timeout_seconds)
    state_payload = fetch_json(f"{args.base_url.rstrip('/')}/api/state", timeout_seconds=args.timeout_seconds)

    runtime = (((state_payload or {}).get("strategies") or {}).get("crypto_pairs_shadow") or {})
    view = (((state_payload or {}).get("comparison_views") or {}).get("crypto_pairs_aave_doge") or {})
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

    snapshot = CheckSnapshot(
        checked_at=datetime.now(UTC).isoformat(),
        base_url=args.base_url,
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
    append_snapshot(output_root, snapshot)
    print(json.dumps(asdict(snapshot), indent=2))


def resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def fetch_json(url: str, *, timeout_seconds: float) -> dict[str, object]:
    try:
        with urlopen(url, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise SystemExit(f"HTTP error fetching {url}: {exc.code}") from exc
    except URLError as exc:
        raise SystemExit(f"Network error fetching {url}: {exc.reason}") from exc


def append_snapshot(output_root: Path, snapshot: CheckSnapshot) -> None:
    payload = asdict(snapshot)
    latest_path = output_root / "latest.json"
    history_path = output_root / "history.jsonl"
    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


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
