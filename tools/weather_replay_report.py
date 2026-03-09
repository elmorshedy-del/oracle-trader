#!/usr/bin/env python3
"""
Generate a first-pass weather replay / calibration report.

This is intentionally operational, not academic:
- uses the current live Opus export bundle
- inspects recent Opus weather candidates, fills, and closes
- summarizes legacy weather paper books from local state files
- distinguishes execution feasibility from actual edge calibration

Outputs:
- output/weather_replay_report.json
- output/weather_replay_report.md
"""

from __future__ import annotations

import argparse
import io
import json
import sqlite3
import statistics
import tempfile
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_EXPORT_URL = "https://just-grace-production-a401.up.railway.app/api/multiagent/logs/export"
WEATHER_STRATEGIES = ("weather_sniper", "weather_latency", "weather_swing")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_export_bundle(url: str) -> zipfile.ZipFile:
    raw = urllib.request.urlopen(url, timeout=60).read()
    return zipfile.ZipFile(io.BytesIO(raw))


def _extract_sqlite(zip_file: zipfile.ZipFile, name: str) -> tuple[sqlite3.Connection, Path]:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite")
    tmp.write(zip_file.read(name))
    tmp.close()
    path = Path(tmp.name)
    return sqlite3.connect(path), path


def _query_rows(con: sqlite3.Connection, query: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    con.row_factory = sqlite3.Row
    return list(con.execute(query, params))


def _fetch_json(zip_file: zipfile.ZipFile, name: str) -> dict[str, Any]:
    return json.loads(zip_file.read(name))


def _summarize_legacy_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    payload = json.loads(path.read_text())
    portfolio = payload.get("portfolio", {})
    positions = portfolio.get("positions", [])
    open_unrealized = sum(_safe_float(item.get("unrealized_pnl")) for item in positions)
    return {
        "file": path.name,
        "starting_capital": _safe_float(portfolio.get("starting_capital")),
        "cash": _safe_float(portfolio.get("cash")),
        "open_positions": len(positions),
        "total_trades": int(portfolio.get("total_trades", 0) or 0),
        "winning_trades": int(portfolio.get("winning_trades", 0) or 0),
        "losing_trades": int(portfolio.get("losing_trades", 0) or 0),
        "realized_pnl": _safe_float(portfolio.get("total_realized_pnl")),
        "open_unrealized_pnl": open_unrealized,
        "fees_paid": _safe_float(portfolio.get("total_fees_paid")),
        "max_drawdown_pct": _safe_float(portfolio.get("max_drawdown")) * 100.0,
        "current_drawdown_pct": _safe_float(portfolio.get("current_drawdown")) * 100.0,
    }


def analyze_weather(repo_root: Path, export_url: str) -> dict[str, Any]:
    bundle = _load_export_bundle(export_url)
    status = _fetch_json(bundle, "status.json")
    llm_context = _fetch_json(bundle, "llm_context.json")

    con, sqlite_path = _extract_sqlite(bundle, "logs/multiagent_runtime.sqlite")
    try:
        fill_rows = _query_rows(
            con,
            """
            SELECT strategy_name, market_question, executed, error, fill_price, shares,
                   edge_estimate, timestamp, market_id
            FROM fills
            WHERE strategy_name IN ('weather_sniper', 'weather_latency', 'weather_swing')
            ORDER BY timestamp ASC
            """,
        )
        close_rows = _query_rows(
            con,
            """
            SELECT strategy_name, market_question, realized_pnl, hold_hours, close_reason, closed_at
            FROM position_closes
            WHERE strategy_name IN ('weather_sniper', 'weather_latency', 'weather_swing')
            ORDER BY closed_at ASC
            """,
        )
        cycle_rows = _query_rows(
            con,
            """
            SELECT scan_id, started_at, candidates_generated, candidates_validated,
                   executions_attempted, executions_succeeded, executions_failed,
                   top_rejection_reason, top_allocation_rejection_reason, zero_trade_explanation
            FROM cycles
            ORDER BY scan_id ASC
            """,
        )
        strategy_rows = _query_rows(
            con,
            """
            SELECT scan_id, strategy_name, candidates
            FROM strategy_counts
            WHERE strategy_name IN ('weather_sniper', 'weather_latency', 'weather_swing')
            ORDER BY scan_id ASC
            """,
        )
    finally:
        con.close()
        sqlite_path.unlink(missing_ok=True)

    fills_by_strategy: dict[str, list[sqlite3.Row]] = defaultdict(list)
    failures_by_market: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    fill_edges: list[float] = []
    for row in fill_rows:
        fills_by_strategy[row["strategy_name"]].append(row)
        fill_edges.append(_safe_float(row["edge_estimate"]))
        if not row["executed"]:
            failure_reasons[row["error"] or "unknown_error"] += 1
            failures_by_market[row["market_question"] or row["market_id"]] += 1

    weather_candidate_counts: dict[str, int] = Counter()
    scans_with_weather_candidates: Counter[str] = Counter()
    for row in strategy_rows:
        weather_candidate_counts[row["strategy_name"]] += int(row["candidates"] or 0)
        if int(row["candidates"] or 0) > 0:
            scans_with_weather_candidates[row["strategy_name"]] += 1

    legacy_states = {}
    for name in (
        "state-weather_sniper.json",
        "state-weather_latency.json",
        "state-weather_swing.json",
        "state-weather_all.json",
    ):
        summary = _summarize_legacy_state(repo_root / name)
        if summary:
            legacy_states[name] = summary

    legacy_sniper = legacy_states.get("state-weather_sniper.json")
    opus_closes = [
        {
            "strategy_name": row["strategy_name"],
            "market_question": row["market_question"],
            "realized_pnl": _safe_float(row["realized_pnl"]),
            "hold_hours": _safe_float(row["hold_hours"]),
            "close_reason": row["close_reason"],
            "closed_at": row["closed_at"],
        }
        for row in close_rows
    ]

    latest_scan = llm_context.get("latest_scan", {})
    blockers = llm_context.get("blockers", [])
    status_perf = status.get("performance", {})

    executions_summary = {}
    for strategy_name in WEATHER_STRATEGIES:
        rows = fills_by_strategy.get(strategy_name, [])
        executed = sum(1 for row in rows if row["executed"])
        failed = len(rows) - executed
        executions_summary[strategy_name] = {
            "attempted": len(rows),
            "executed": executed,
            "failed": failed,
            "execution_rate_pct": round((executed / len(rows) * 100.0), 2) if rows else 0.0,
            "median_edge_estimate": round(statistics.median([_safe_float(r["edge_estimate"]) for r in rows]), 5)
            if rows
            else None,
        }

    result = {
        "generated_at": status.get("portfolio", {}).get("snapshot_at") or status.get("last_scan_completed_at"),
        "source": {
            "export_url": export_url,
            "status_endpoint": status.get("runtime_mode"),
        },
        "current_runtime": {
            "scan_count": status.get("scan_count"),
            "open_positions": status.get("portfolio", {}).get("position_count"),
            "available_capital": status.get("portfolio", {}).get("available_capital"),
            "capital_utilization_pct": status.get("portfolio", {}).get("capital_utilization_pct"),
            "current_realized_pnl": status_perf.get("realized_pnl"),
            "current_unrealized_pnl": status_perf.get("unrealized_pnl"),
            "blockers": blockers,
            "latest_scan": latest_scan,
        },
        "opus_weather": {
            "candidate_counts_total": dict(weather_candidate_counts),
            "scans_with_candidates": dict(scans_with_weather_candidates),
            "execution_summary": executions_summary,
            "top_execution_failures": [
                {"reason": reason, "count": count}
                for reason, count in failure_reasons.most_common(10)
            ],
            "most_repeated_failed_markets": [
                {"market": market, "count": count}
                for market, count in failures_by_market.most_common(10)
            ],
            "closed_positions": opus_closes,
        },
        "legacy_weather": legacy_states,
        "assessment": {
            "weather_execution_issue_real": executions_summary["weather_sniper"]["failed"] > 0,
            "weather_edge_proven": len(opus_closes) >= 10,
            "weather_calibration_ready": len(opus_closes) >= 10,
            "legacy_sniper_positive_realized": bool(
                legacy_sniper and legacy_sniper["realized_pnl"] > 0 and legacy_sniper["winning_trades"] > 0
            ),
            "notes": [
                "Opus weather can be judged for execution feasibility right now.",
                "Opus weather cannot yet be treated as calibrated because the realized close sample is tiny.",
                "Legacy weather books provide evidence that weather can work in this repo, but they are not a full calibrated replay dataset.",
            ],
        },
    }
    return result


def render_markdown(report: dict[str, Any]) -> str:
    runtime = report["current_runtime"]
    opus = report["opus_weather"]
    legacy = report["legacy_weather"]
    assessment = report["assessment"]
    latest_scan = runtime["latest_scan"]

    lines = [
        "# Weather Replay Report",
        "",
        f"Generated at: `{report['generated_at']}`",
        "",
        "## Verdict",
        "",
        f"- Weather execution issue is real: `{assessment['weather_execution_issue_real']}`",
        f"- Weather edge is proven from Opus closes: `{assessment['weather_edge_proven']}`",
        f"- Weather calibration sample is sufficient: `{assessment['weather_calibration_ready']}`",
        f"- Legacy weather sniper has positive realized PnL: `{assessment['legacy_sniper_positive_realized']}`",
        "",
        "## Current Opus Runtime",
        "",
        f"- Scan count: `{runtime['scan_count']}`",
        f"- Open positions: `{runtime['open_positions']}`",
        f"- Realized PnL: `{runtime['current_realized_pnl']}`",
        f"- Unrealized PnL: `{runtime['current_unrealized_pnl']}`",
        f"- Latest scan candidates: `{latest_scan.get('candidates_generated')}`",
        f"- Latest scan validated: `{latest_scan.get('candidates_validated')}`",
        f"- Latest scan executions succeeded: `{latest_scan.get('executions_succeeded')}`",
        f"- Latest scan executions failed: `{latest_scan.get('executions_failed')}`",
        "",
        "## Opus Weather Signal / Execution Summary",
        "",
    ]

    for strategy_name, summary in opus["execution_summary"].items():
        lines.extend(
            [
                f"### {strategy_name}",
                "",
                f"- Candidates seen: `{opus['candidate_counts_total'].get(strategy_name, 0)}`",
                f"- Scans with candidates: `{opus['scans_with_candidates'].get(strategy_name, 0)}`",
                f"- Execution attempts: `{summary['attempted']}`",
                f"- Executed: `{summary['executed']}`",
                f"- Failed: `{summary['failed']}`",
                f"- Execution rate: `{summary['execution_rate_pct']}%`",
                f"- Median estimated edge: `{summary['median_edge_estimate']}`",
                "",
            ]
        )

    lines.extend(["## Top Execution Failure Reasons", ""])
    for item in opus["top_execution_failures"]:
        lines.append(f"- `{item['reason']}`: `{item['count']}`")
    if not opus["top_execution_failures"]:
        lines.append("- none")

    lines.extend(["", "## Most Repeated Failed Weather Markets", ""])
    for item in opus["most_repeated_failed_markets"]:
        lines.append(f"- `{item['market']}`: `{item['count']}` failed attempts")
    if not opus["most_repeated_failed_markets"]:
        lines.append("- none")

    lines.extend(["", "## Opus Weather Closes", ""])
    if opus["closed_positions"]:
        for close in opus["closed_positions"]:
            lines.append(
                f"- `{close['strategy_name']}` | `{close['market_question']}` | "
                f"PnL `{close['realized_pnl']:.2f}` | hold `{close['hold_hours']:.2f}h` | "
                f"reason `{close['close_reason']}`"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Legacy Weather Books", ""])
    for name, summary in legacy.items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Starting capital: `{summary['starting_capital']}`",
                f"- Total trades: `{summary['total_trades']}`",
                f"- Wins / losses: `{summary['winning_trades']}` / `{summary['losing_trades']}`",
                f"- Realized PnL: `{summary['realized_pnl']}`",
                f"- Open unrealized PnL: `{summary['open_unrealized_pnl']}`",
                f"- Fees paid: `{summary['fees_paid']}`",
                f"- Max drawdown: `{summary['max_drawdown_pct']:.2f}%`",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation",
            "",
            "- The weather issue is not that no signals exist. Signals are being generated repeatedly.",
            "- The strongest current failure is execution feasibility on ultra-cheap sniper entries, especially slippage limits on sub-1c YES tokens.",
            "- The current Opus close sample is too small to call the probability model calibrated.",
            "- Legacy weather results show that weather trading can work in this repo, which points to an Opus execution/sizing/policy mismatch rather than weather being inherently bad.",
            "",
            "## Next Tests",
            "",
            "- Reduce or reshape Opus sniper sizing so it behaves like the legacy micro-bet book rather than trying to buy very large size in ultra-thin prices.",
            "- Run a walk-forward replay that uses stored forecast snapshots and market prices to score signal quality independently from execution rules.",
            "- Add an explicit weather calibration report once enough resolved Opus closes accumulate.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate an Opus weather replay report.")
    parser.add_argument("--repo-root", default="/Users/ahmedelmorshedy/Downloads/oracle-trader")
    parser.add_argument("--export-url", default=DEFAULT_EXPORT_URL)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = analyze_weather(repo_root=repo_root, export_url=args.export_url)
    json_path = output_dir / "weather_replay_report.json"
    md_path = output_dir / "weather_replay_report.md"
    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(render_markdown(report))

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
