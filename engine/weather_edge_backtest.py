from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from engine.weather_edge_baseline import FrozenWeatherModelBundle, build_baseline_feature_row
from engine.weather_edge_config import (
    DEFAULT_BANKROLL_USD,
    KELLY_FRACTION,
    MAX_POSITION_FRACTION,
    MIN_GROUP_ROWS,
    QUARTER_WINDOW_DAYS,
    RULE_ALLOWED_METRICS,
    RULE_ALLOWED_LEAD_TIMES_HOURS,
    RULE_ALLOWED_REGIONS_BY_LEAD_TIME,
    RULE_MIN_EDGE,
    RULE_MIN_MARKET_VOLUME,
    RULE_MIN_MODEL_AGREEMENT,
    RULE_SELECT_TOP_CONTRACT_PER_EVENT_HORIZON,
    SPLIT_HALF_DAYS,
)


def score_market_horizons(
    *,
    rows: list[dict[str, Any]],
    bundle: FrozenWeatherModelBundle,
) -> list[dict[str, Any]]:
    scored_rows: list[dict[str, Any]] = []
    for row in rows:
        feature_row = build_baseline_feature_row(
            city=row["city"],
            temp_unit=row["temp_unit"],
            temp_kind=row["temp_kind"],
            temp_range_low=row["temp_range_low"],
            temp_range_high=row["temp_range_high"],
            target_month=int(str(row["target_date"])[5:7]),
            target_day=int(str(row["target_date"])[8:10]),
            market_yes_price=row["market_yes_probability"],
            forecast_temp_max=row.get("historical_temp_max"),
            model_temperatures=row.get("model_temperatures") or {},
        )
        prediction = bundle.predict_yes_probability(feature_row, temp_kind=row["temp_kind"])
        if not prediction:
            continue

        model_yes_probability = prediction["prob_yes"]
        model_no_probability = 1.0 - model_yes_probability
        market_yes_probability = float(row["market_yes_probability"])
        market_no_probability = 1.0 - market_yes_probability
        edge = model_yes_probability - market_yes_probability
        trade_side = "yes" if edge >= 0 else "no"
        absolute_edge = abs(edge)
        would_win = bool(row["resolved_yes"]) if trade_side == "yes" else not bool(row["resolved_yes"])
        realized_value = absolute_edge if would_win else -absolute_edge
        market_trade_probability = market_yes_probability if trade_side == "yes" else market_no_probability
        model_trade_probability = model_yes_probability if trade_side == "yes" else model_no_probability
        kelly_fraction = _kelly_fraction(
            your_probability=model_trade_probability,
            market_probability=market_trade_probability,
        )

        scored_rows.append(
            {
                **row,
                "model_yes_probability": round(model_yes_probability, 6),
                "model_no_probability": round(model_no_probability, 6),
                "model_name": prediction["model_name"],
                "model_auc": prediction["model_auc"],
                "raw_edge": round(edge, 6),
                "absolute_edge": round(absolute_edge, 6),
                "trade_side": trade_side,
                "would_win": would_win,
                "realized_value": round(realized_value, 6),
                "kelly_fraction": round(kelly_fraction, 6),
                "model_probability_gap": round(abs(model_yes_probability - 0.5), 6),
            }
        )
    return scored_rows


def summarize_weather_edge(
    *,
    scored_rows: list[dict[str, Any]],
    bankroll_usd: float = DEFAULT_BANKROLL_USD,
) -> dict[str, Any]:
    rules = {
        "allowed_metrics": RULE_ALLOWED_METRICS,
        "allowed_lead_times_hours": RULE_ALLOWED_LEAD_TIMES_HOURS,
        "allowed_regions_by_lead_time": RULE_ALLOWED_REGIONS_BY_LEAD_TIME,
        "min_edge": RULE_MIN_EDGE,
        "min_model_agreement": RULE_MIN_MODEL_AGREEMENT,
        "min_market_volume": RULE_MIN_MARKET_VOLUME,
        "select_top_contract_per_event_horizon": RULE_SELECT_TOP_CONTRACT_PER_EVENT_HORIZON,
        "kelly_fraction": KELLY_FRACTION,
        "max_position_fraction": MAX_POSITION_FRACTION,
        "bankroll_usd": bankroll_usd,
    }
    eligible_rows = [row for row in scored_rows if _passes_rule_filter(row)]
    selected_rows = _select_trade_rows(eligible_rows)
    trade_backtest = _simulate_binary_backtest(selected_rows, bankroll_usd=bankroll_usd)

    return {
        "totals": {
            "rows_scored": len(scored_rows),
            "rows_eligible": len(eligible_rows),
            "markets_scored": len({row["market_id"] for row in scored_rows}),
            "markets_eligible": len({row["market_id"] for row in eligible_rows}),
            "trade_rows_selected": len(selected_rows),
        },
        "rules": rules,
        "breakdowns": {
            "metric_type": _group_summary(scored_rows, "metric_type"),
            "lead_time_hours": _group_summary(scored_rows, "lead_time_hours"),
            "region": _group_summary(scored_rows, "region"),
            "season": _group_summary(scored_rows, "season"),
        },
        "backtest": trade_backtest,
        "split_half": _split_half_summary(selected_rows, bankroll_usd=bankroll_usd),
        "quarters_15d": _rolling_window_summary(selected_rows, bankroll_usd=bankroll_usd),
        "notes": _build_notes(scored_rows, eligible_rows, selected_rows),
    }


def write_weather_edge_report(
    *,
    output_root: Path,
    scored_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    report_json = output_root / "weather_edge_report.json"
    report_md = output_root / "weather_edge_report.md"
    report_json.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "summary": summary,
                "rows": _json_ready(scored_rows),
            },
            indent=2,
        )
    )
    report_md.write_text(_render_markdown_report(summary=summary, metadata=metadata))
    return report_json, report_md


def _passes_rule_filter(row: dict[str, Any]) -> bool:
    if row.get("metric_type") not in RULE_ALLOWED_METRICS:
        return False
    lead_time_hours = int(row.get("lead_time_hours") or 0)
    if RULE_ALLOWED_LEAD_TIMES_HOURS and lead_time_hours not in RULE_ALLOWED_LEAD_TIMES_HOURS:
        return False
    allowed_regions = RULE_ALLOWED_REGIONS_BY_LEAD_TIME.get(lead_time_hours)
    if allowed_regions and row.get("region") not in allowed_regions:
        return False
    if abs(float(row.get("raw_edge") or 0.0)) < RULE_MIN_EDGE:
        return False
    if (row.get("model_agreement") or 0.0) < RULE_MIN_MODEL_AGREEMENT:
        return False
    if float(row.get("volume_clob") or 0.0) < RULE_MIN_MARKET_VOLUME:
        return False
    return True


def _select_trade_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not RULE_SELECT_TOP_CONTRACT_PER_EVENT_HORIZON:
        return list(rows)
    selected_by_event: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        event_key = (
            str(row.get("city") or ""),
            str(row.get("target_date") or ""),
            int(row.get("lead_time_hours") or 0),
        )
        current = selected_by_event.get(event_key)
        if current is None or abs(float(row.get("raw_edge") or 0.0)) > abs(float(current.get("raw_edge") or 0.0)):
            selected_by_event[event_key] = row
    return sorted(
        selected_by_event.values(),
        key=lambda row: (row["resolution_time"], row["city"], row["lead_time_hours"], row["market_id"]),
    )


def _kelly_fraction(*, your_probability: float, market_probability: float) -> float:
    if market_probability <= 0 or market_probability >= 1:
        return 0.0
    payout_ratio = (1.0 / market_probability) - 1.0
    lose_probability = 1.0 - your_probability
    fraction = ((payout_ratio * your_probability) - lose_probability) / payout_ratio
    return min(MAX_POSITION_FRACTION, max(0.0, fraction * KELLY_FRACTION))


def _simulate_binary_backtest(rows: list[dict[str, Any]], *, bankroll_usd: float) -> dict[str, Any]:
    ordered_rows = sorted(rows, key=lambda row: (row["resolution_time"], row["market_id"], row["lead_time_hours"]))
    bankroll = bankroll_usd
    peak = bankroll
    max_drawdown = 0.0
    total_net_bps = 0.0
    win_count = 0
    loss_count = 0
    exit_reason_counts = defaultdict(int)

    for row in ordered_rows:
        stake = bankroll * float(row["kelly_fraction"])
        if stake <= 0:
            continue

        trade_probability = float(row["market_yes_probability"] if row["trade_side"] == "yes" else 1.0 - row["market_yes_probability"])
        gross_pnl_usd = _binary_contract_pnl(
            stake_usd=stake,
            market_probability=trade_probability,
            won=bool(row["would_win"]),
        )
        bankroll += gross_pnl_usd
        peak = max(peak, bankroll)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - bankroll) / peak)

        net_bps = (gross_pnl_usd / bankroll_usd) * 10_000.0
        total_net_bps += net_bps
        if gross_pnl_usd >= 0:
            win_count += 1
            exit_reason_counts["mean_reversion"] += 1
        else:
            loss_count += 1
            exit_reason_counts["loss"] += 1

    trade_count = win_count + loss_count
    return {
        "bankroll_start_usd": bankroll_usd,
        "bankroll_end_usd": round(bankroll, 4),
        "trade_count": trade_count,
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate": round((win_count / trade_count), 6) if trade_count else None,
        "total_net_bps": round(total_net_bps, 4),
        "total_net_usd": round(bankroll - bankroll_usd, 4),
        "max_drawdown": round(max_drawdown, 6),
        "exit_reason_counts": dict(exit_reason_counts),
    }


def _binary_contract_pnl(*, stake_usd: float, market_probability: float, won: bool) -> float:
    if market_probability <= 0 or market_probability >= 1:
        return 0.0
    if won:
        shares = stake_usd / market_probability
        return shares * (1.0 - market_probability)
    return -stake_usd


def _group_summary(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(group_key) or "unknown")].append(row)

    summaries = []
    for name, group_rows in groups.items():
        if len(group_rows) < MIN_GROUP_ROWS:
            continue
        avg_edge = sum(float(row["absolute_edge"]) for row in group_rows) / len(group_rows)
        avg_realized_value = sum(float(row["realized_value"]) for row in group_rows) / len(group_rows)
        summaries.append(
            {
                "group": name,
                "rows": len(group_rows),
                "avg_absolute_edge": round(avg_edge, 6),
                "avg_realized_value": round(avg_realized_value, 6),
                "positive_share": round(sum(1 for row in group_rows if row["realized_value"] > 0) / len(group_rows), 6),
            }
        )
    summaries.sort(key=lambda row: (row["avg_realized_value"], row["avg_absolute_edge"]), reverse=True)
    return summaries


def _split_half_summary(rows: list[dict[str, Any]], *, bankroll_usd: float) -> dict[str, Any]:
    if not rows:
        return {"available": False}
    ordered = sorted(rows, key=lambda row: row["resolution_time"])
    split_index = max(1, len(ordered) // 2)
    first_half = ordered[:split_index]
    second_half = ordered[split_index:]
    return {
        "available": True,
        "first_half": _simulate_binary_backtest(first_half, bankroll_usd=bankroll_usd),
        "second_half": _simulate_binary_backtest(second_half, bankroll_usd=bankroll_usd),
        "window_days": SPLIT_HALF_DAYS,
    }


def _rolling_window_summary(rows: list[dict[str, Any]], *, bankroll_usd: float) -> list[dict[str, Any]]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda row: row["resolution_time"])
    summaries: list[dict[str, Any]] = []
    window_start = ordered[0]["resolution_time"].date()
    final_date = ordered[-1]["resolution_time"].date()
    while window_start <= final_date:
        window_end = window_start + timedelta(days=QUARTER_WINDOW_DAYS - 1)
        window_rows = [
            row
            for row in ordered
            if window_start <= row["resolution_time"].date() <= window_end
        ]
        if window_rows:
            summaries.append(
                {
                    "start_date": window_start.isoformat(),
                    "end_date": window_end.isoformat(),
                    **_simulate_binary_backtest(window_rows, bankroll_usd=bankroll_usd),
                }
            )
        window_start = window_end + timedelta(days=1)
    return summaries


def _build_notes(
    scored_rows: list[dict[str, Any]],
    eligible_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
) -> list[str]:
    notes = []
    metric_types = sorted({row["metric_type"] for row in scored_rows})
    notes.append(f"Resolved market dataset is currently limited to metric types: {', '.join(metric_types) or 'none'}.")
    if any(row.get("weather_proxy_source") == "local_artifacts" for row in scored_rows):
        notes.append(
            "Model-side weather features are currently sourced from the preserved historical snapshot artifacts by default; this keeps the lane standalone, but exact historical sub-day forecast revisions still need richer archived forecast-run data."
        )
    if RULE_SELECT_TOP_CONTRACT_PER_EVENT_HORIZON:
        notes.append(
            "Backtest execution selects the single highest-edge contract per city/date/lead-time event to avoid taking multiple mutually exclusive temperature buckets at once."
        )
    if not eligible_rows:
        notes.append("No rows passed the default rule filter. Adjust thresholds only after reviewing the raw breakdowns.")
    elif len(selected_rows) < len(eligible_rows):
        notes.append(
            f"Rule filtering produced {len(eligible_rows)} candidate rows, narrowed to {len(selected_rows)} event-level trades after contract competition."
        )
    return notes


def _render_markdown_report(*, summary: dict[str, Any], metadata: dict[str, Any]) -> str:
    lines = [
        "# Weather Edge v1 Report",
        "",
        f"Generated at: `{metadata['generated_at']}`",
        f"Lookback days: `{metadata['lookback_days']}`",
        f"Bundle: `{metadata['bundle_dir']}`",
        "",
        "## Totals",
        "",
        f"- Rows scored: `{summary['totals']['rows_scored']}`",
        f"- Rows eligible: `{summary['totals']['rows_eligible']}`",
        f"- Markets scored: `{summary['totals']['markets_scored']}`",
        f"- Markets eligible: `{summary['totals']['markets_eligible']}`",
        f"- Event-level trades selected: `{summary['totals']['trade_rows_selected']}`",
        "",
        "## Backtest",
        "",
        f"- Trades: `{summary['backtest']['trade_count']}`",
        f"- Win count / loss count: `{summary['backtest']['win_count']}` / `{summary['backtest']['loss_count']}`",
        f"- Win rate: `{summary['backtest']['win_rate']}`",
        f"- Total net bps: `{summary['backtest']['total_net_bps']}`",
        f"- Total net USD: `{summary['backtest']['total_net_usd']}`",
        f"- Max drawdown: `{summary['backtest']['max_drawdown']}`",
        "",
        "## Breakdowns",
        "",
    ]

    for label, rows in summary["breakdowns"].items():
        lines.append(f"### {label}")
        lines.append("")
        if not rows:
            lines.append("- none")
            lines.append("")
            continue
        for row in rows:
            lines.append(
                f"- `{row['group']}` | rows `{row['rows']}` | avg edge `{row['avg_absolute_edge']}` | "
                f"avg realized `{row['avg_realized_value']}` | positive share `{row['positive_share']}`"
            )
        lines.append("")

    lines.extend(["## Notes", ""])
    for note in summary.get("notes", []):
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(current) for key, current in value.items()}
    if isinstance(value, list):
        return [_json_ready(current) for current in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value
