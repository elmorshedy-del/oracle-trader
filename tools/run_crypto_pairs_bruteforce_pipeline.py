#!/usr/bin/env python3
"""
Brute-force crypto pairs discovery and follow-up backtests on a broad token universe.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_PYTHON = Path("/Users/ahmedelmorshedy/.local/bin/oracle-btc-python")
DEFAULT_PROJECT_ROOT = Path("research/crypto_pairs/projects/crypto-pairs-bruteforce-v1")
DEFAULT_DIARY_PATH = Path("research/crypto_pairs/diary.md")
DEFAULT_BACKTEST_OUTPUT_ROOT = Path("output/crypto_pairs/backtests")
DEFAULT_SYMBOLS = [
    "AAVEUSDT",
    "UNIUSDT",
    "LINKUSDT",
    "DOGEUSDT",
    "SHIBUSDT",
    "SOLUSDT",
    "ETHUSDT",
    "BTCUSDT",
    "AVAXUSDT",
    "NEARUSDT",
    "DOTUSDT",
    "ATOMUSDT",
    "ADAUSDT",
    "MATICUSDT",
    "ARBUSDT",
    "OPUSDT",
    "FETUSDT",
    "RENDERUSDT",
    "SUIUSDT",
    "SEIUSDT",
    "INJUSDT",
    "TIAUSDT",
    "MKRUSDT",
    "SNXUSDT",
    "CRVUSDT",
    "COMPUSDT",
    "PEPEUSDT",
    "FLOKIUSDT",
    "WIFUSDT",
    "BONKUSDT",
    "FTMUSDT",
    "ALGOUSDT",
    "XLMUSDT",
    "HBARUSDT",
    "VETUSDT",
    "TAOUSDT",
    "RUNEUSDT",
    "GRTUSDT",
    "STXUSDT",
    "IMXUSDT",
]
THIRTY_DAY_WINDOW = 30
FIFTEEN_DAY_WINDOW = 15


@dataclass(frozen=True)
class BacktestSlice:
    name: str
    start_date: date
    end_date: date


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run brute-force crypto pairs discovery and filtered follow-up backtests.")
    parser.add_argument("--start-date", help="Inclusive start date YYYY-MM-DD")
    parser.add_argument("--end-date", help="Inclusive end date YYYY-MM-DD")
    parser.add_argument("--lookback-days", type=int, default=60)
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--diary-path", default=str(DEFAULT_DIARY_PATH))
    parser.add_argument("--backtest-output-root", default=str(DEFAULT_BACKTEST_OUTPUT_ROOT))
    parser.add_argument("--discovery-report", default=None, help="Existing discovery report path to resume from.")
    parser.add_argument("--symbol", action="append", default=[])
    parser.add_argument("--max-coint-pvalue", type=float, default=0.05)
    parser.add_argument("--min-correlation", type=float, default=0.80)
    parser.add_argument("--min-halflife-hours", type=float, default=1.0)
    parser.add_argument("--max-halflife-hours", type=float, default=72.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    end_date, start_date = resolve_dates(args.start_date, args.end_date, args.lookback_days)
    repo_root = REPO_ROOT
    python_exe = resolve_path(args.python)
    project_root = resolve_path(args.project_root)
    diary_path = resolve_path(args.diary_path)
    backtest_output_root = resolve_path(args.backtest_output_root)
    project_root.mkdir(parents=True, exist_ok=True)
    symbols = [symbol.upper() for symbol in (args.symbol or DEFAULT_SYMBOLS)]

    if args.discovery_report:
        discovery_report_path = resolve_path(args.discovery_report)
    else:
        discovery_payload = run_json(
            [
                str(python_exe),
                str(repo_root / "tools" / "run_crypto_pairs_discovery.py"),
                "--start-date",
                start_date.isoformat(),
                "--end-date",
                end_date.isoformat(),
                "--project-root",
                str(project_root),
                "--diary-path",
                str(diary_path),
                "--min-correlation",
                str(args.min_correlation),
                "--max-coint-pvalue",
                str(args.max_coint_pvalue),
                "--min-halflife-hours",
                str(args.min_halflife_hours),
                "--max-halflife-hours",
                str(args.max_halflife_hours),
                *repeat_flag("--symbol", symbols),
            ],
            cwd=repo_root,
        )
        discovery_report_path = Path(str(discovery_payload["report_json"])).resolve()
    discovery_report = json.loads(discovery_report_path.read_text(encoding="utf-8"))
    tradeable_pairs = list(discovery_report.get("tradeable_pairs", []))

    full_window = BacktestSlice("full_60d", start_date, end_date)
    split_windows = build_slices(start_date, end_date, THIRTY_DAY_WINDOW, "split_30d")
    quarter_windows = build_slices(start_date, end_date, FIFTEEN_DAY_WINDOW, "split_15d")

    analyzed_pairs = []
    for pair_row in tradeable_pairs:
        pair_key = str(pair_row["pair"])
        full_report = run_backtest(
            python_exe=python_exe,
            discovery_report_path=discovery_report_path,
            pair_key=pair_key,
            backtest_output_root=backtest_output_root,
            slice_window=full_window,
            cwd=repo_root,
        )
        split_reports = [
            run_backtest(
                python_exe=python_exe,
                discovery_report_path=discovery_report_path,
                pair_key=pair_key,
                backtest_output_root=backtest_output_root,
                slice_window=slice_window,
                cwd=repo_root,
            )
            for slice_window in split_windows
        ]
        profitable_both_halves = all(report["summary"]["total_pnl_bps"] > 0 for report in split_reports)
        quarter_reports = []
        if profitable_both_halves:
            quarter_reports = [
                run_backtest(
                    python_exe=python_exe,
                    discovery_report_path=discovery_report_path,
                    pair_key=pair_key,
                    backtest_output_root=backtest_output_root,
                    slice_window=slice_window,
                    cwd=repo_root,
                )
                for slice_window in quarter_windows
            ]

        analyzed_pairs.append(
            {
                "pair": pair_key,
                "discovery": pair_row,
                "full_window": summarize_report(full_report),
                "split_30d": [summarize_report(report) for report in split_reports],
                "passes_split_30d": profitable_both_halves,
                "split_15d": [summarize_report(report) for report in quarter_reports],
            }
        )

    survivors = [row for row in analyzed_pairs if row["passes_split_30d"]]
    run_name = f"crypto_pairs_bruteforce_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}_v1"
    run_root = project_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_name": run_name,
        "started_at": datetime.now(UTC).isoformat(),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "symbols_requested": symbols,
        "total_symbols": len(symbols),
        "total_pairs_possible": len(symbols) * (len(symbols) - 1) // 2,
        "discovery_report": str(discovery_report_path),
        "tradeable_pair_count": len(tradeable_pairs),
        "survivor_count": len(survivors),
        "tradeable_pairs": analyzed_pairs,
        "survivors": survivors,
    }
    summary_path = run_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    checkpoint_id = f"crypto-pairs-bruteforce-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
    subprocess.check_call(
        [
            str(python_exe),
            str(repo_root / "tools" / "freeze_crypto_pairs_checkpoint.py"),
            "--checkpoint-id",
            checkpoint_id,
            "--status",
            "bruteforce_screen_complete",
            "--category",
            "pair_discovery",
            "--summary",
            "Brute-force crypto pairs discovery on a 40-token Binance spot universe with 60-day backtests and split filters on all cointegrated pairs.",
            "--artifact",
            f"project_root={project_root}",
            "--artifact",
            f"discovery_report={discovery_report_path}",
            "--artifact",
            f"summary_json={summary_path}",
            "--metric",
            f"symbols_loaded={len(discovery_report.get('symbols_loaded', []))}",
            "--metric",
            f"pairs_tested={discovery_report.get('total_pairs_tested', 0)}",
            "--metric",
            f"cointegrated_pairs={len(tradeable_pairs)}",
            "--metric",
            f"survivors={len(survivors)}",
        ],
        cwd=str(repo_root),
    )

    write_plan(
        project_root=project_root,
        summary_path=summary_path,
        checkpoint_id=checkpoint_id,
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        survivors=survivors,
    )
    append_diary(
        diary_path=diary_path,
        checkpoint_id=checkpoint_id,
        summary_path=summary_path,
        start_date=start_date,
        end_date=end_date,
        symbols_loaded=len(discovery_report.get("symbols_loaded", [])),
        pairs_tested=int(discovery_report.get("total_pairs_tested", 0)),
        cointegrated_pairs=len(tradeable_pairs),
        survivors=survivors,
    )
    print(json.dumps({"checkpoint_id": checkpoint_id, "summary_json": str(summary_path)}, indent=2))


def resolve_dates(start_date_raw: str | None, end_date_raw: str | None, lookback_days: int) -> tuple[date, date]:
    if start_date_raw and end_date_raw:
        start_date = datetime.strptime(start_date_raw, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date_raw, "%Y-%m-%d").date()
    else:
        end_date = datetime.now(UTC).date() - timedelta(days=1)
        start_date = end_date - timedelta(days=lookback_days - 1)
    if end_date < start_date:
        raise SystemExit("end-date must be on or after start-date")
    return end_date, start_date


def resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def repeat_flag(flag: str, values: list[str]) -> list[str]:
    expanded: list[str] = []
    for value in values:
        expanded.extend([flag, value])
    return expanded


def run_json(command: list[str], *, cwd: Path) -> dict[str, object]:
    output = subprocess.check_output(command, cwd=str(cwd), text=True)
    return parse_last_json_object(output)


def parse_last_json_object(output: str) -> dict[str, object]:
    decoder = json.JSONDecoder()
    parsed: list[dict[str, object]] = []
    index = 0
    while index < len(output):
        next_open = output.find("{", index)
        if next_open == -1:
            break
        try:
            candidate, offset = decoder.raw_decode(output, next_open)
        except json.JSONDecodeError:
            index = next_open + 1
            continue
        if isinstance(candidate, dict):
            parsed.append(candidate)
        index = offset
    if not parsed:
        raise ValueError("No JSON object found in subprocess output")
    return parsed[-1]


def run_backtest(
    *,
    python_exe: Path,
    discovery_report_path: Path,
    pair_key: str,
    backtest_output_root: Path,
    slice_window: BacktestSlice,
    cwd: Path,
) -> dict[str, object]:
    payload = run_json(
        [
            str(python_exe),
            str(REPO_ROOT / "tools" / "backtest_crypto_pairs_v1.py"),
            "--discovery-report",
            str(discovery_report_path),
            "--pair",
            pair_key,
            "--start-date",
            slice_window.start_date.isoformat(),
            "--end-date",
            slice_window.end_date.isoformat(),
            "--output-root",
            str(backtest_output_root),
        ],
        cwd=cwd,
    )
    report_path = Path(str(payload["report_json"])).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["_artifact_path"] = str(report_path)
    report["_window_name"] = slice_window.name
    return report


def summarize_report(report: dict[str, object]) -> dict[str, object]:
    summary = dict(report["summary"])
    summary["report_json"] = report["_artifact_path"]
    summary["window_name"] = report["_window_name"]
    return summary


def build_slices(start_date: date, end_date: date, window_days: int, prefix: str) -> list[BacktestSlice]:
    slices: list[BacktestSlice] = []
    cursor = start_date
    index = 1
    while cursor <= end_date:
        slice_end = min(cursor + timedelta(days=window_days - 1), end_date)
        slices.append(BacktestSlice(f"{prefix}_{index}", cursor, slice_end))
        cursor = slice_end + timedelta(days=1)
        index += 1
    return slices


def write_plan(
    *,
    project_root: Path,
    summary_path: Path,
    checkpoint_id: str,
    start_date: date,
    end_date: date,
    symbols: list[str],
    survivors: list[dict[str, object]],
) -> None:
    preview = "\n".join(
        f"- `{row['pair']}` full `{row['full_window']['total_pnl_bps']} bps`"
        for row in survivors[:10]
    ) or "- none"
    text = f"""# Crypto Pairs Bruteforce v1

Brute-force crypto pairs discovery over a broad Binance spot universe, followed by full-window and split validation using the frozen V1 rule set.

- Date range: `{start_date}` to `{end_date}`
- Symbols: `{', '.join(symbols)}`
- Summary: `{summary_path}`
- Checkpoint: `{checkpoint_id}`

Survivors after the 30-day split filter:
{preview}
"""
    (project_root / "plan.md").write_text(text, encoding="utf-8")


def append_diary(
    *,
    diary_path: Path,
    checkpoint_id: str,
    summary_path: Path,
    start_date: date,
    end_date: date,
    symbols_loaded: int,
    pairs_tested: int,
    cointegrated_pairs: int,
    survivors: list[dict[str, object]],
) -> None:
    preview = "\n".join(
        f"  - `{row['pair']}` full `{row['full_window']['total_pnl_bps']} bps`"
        for row in survivors[:5]
    ) or "  - none"
    entry = f"""

## {datetime.now(UTC).date().isoformat()} - Brute-force crypto pairs screen

- Checkpoint: `{checkpoint_id}`
- Date range: `{start_date}` to `{end_date}`
- Symbols loaded: `{symbols_loaded}`
- Pairs tested: `{pairs_tested}`
- Cointegrated pairs: `{cointegrated_pairs}`
- Survivors after 30-day split: `{len(survivors)}`
- Summary: `{summary_path}`
- Survivors:
{preview}
"""
    with diary_path.open("a", encoding="utf-8") as handle:
        handle.write(entry)


if __name__ == "__main__":
    main()
