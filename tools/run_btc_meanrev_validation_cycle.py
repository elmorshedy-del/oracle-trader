#!/usr/bin/env python3
"""
Run one official out-of-sample validation cycle for the frozen BTC downshock
mean-reversion candidate.

This is intentionally append-only:
- optional delayed start to avoid crossing a UTC day boundary
- one fresh multivenue capture session
- dataset built from that session only
- frozen validation runner with no parameter changes
- checkpoint + diary + validation history update
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


UTC = timezone.utc
DEFAULT_CAPTURE_DURATION_SECONDS = 3600
DEFAULT_CAPTURE_PYTHON = Path("/Users/ahmedelmorshedy/.local/bin/oracle-btc-python")
DEFAULT_ANALYSIS_PYTHON = Path("/Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14")
DEFAULT_SPEC_PATH = Path("research/btc/projects/btc-meanrev-downshock30-v1/validation_spec.json")
DEFAULT_CAPTURE_ROOT = Path("output/btc_multivenue_capture")
DEFAULT_DATASET_ROOT = Path("output/btc_multivenue_dataset")
DEFAULT_VALIDATION_ROOT = Path("output/btc_meanrev_validation")
DEFAULT_HISTORY_PATH = Path("research/btc/projects/btc-meanrev-downshock30-v1/validation_history.json")
DEFAULT_SUMMARY_PATH = Path("research/btc/projects/btc-meanrev-downshock30-v1/validation_summary.json")
DEFAULT_DIARY_PATH = Path("research/btc/diary.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one frozen BTC mean-reversion validation cycle.")
    parser.add_argument("--spec-path", default=str(DEFAULT_SPEC_PATH), help="Frozen validation spec JSON")
    parser.add_argument("--capture-python", default=str(DEFAULT_CAPTURE_PYTHON), help="Python executable used for the capture session")
    parser.add_argument("--analysis-python", default=str(DEFAULT_ANALYSIS_PYTHON), help="Python executable used for build/validation/bookkeeping")
    parser.add_argument("--capture-root", default=str(DEFAULT_CAPTURE_ROOT), help="Root for multivenue capture sessions")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT), help="Root for built single-session datasets")
    parser.add_argument("--validation-root", default=str(DEFAULT_VALIDATION_ROOT), help="Root for validation reports")
    parser.add_argument("--history-path", default=str(DEFAULT_HISTORY_PATH), help="Validation history JSON")
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH), help="Aggregate validation summary JSON")
    parser.add_argument("--diary-path", default=str(DEFAULT_DIARY_PATH), help="BTC diary markdown")
    parser.add_argument("--capture-duration-seconds", type=int, default=DEFAULT_CAPTURE_DURATION_SECONDS, help="Length of the validation capture session")
    parser.add_argument("--delay-seconds", type=int, default=0, help="Optional wait before starting the capture")
    parser.add_argument("--run-label", default="meanrev_validation_v1", help="Capture session label")
    parser.add_argument("--bootstrap-samples", type=int, default=5000, help="Bootstrap samples for frozen validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--existing-session-dir", default=None, help="Use an already-finished session directory instead of starting a new capture")
    parser.add_argument("--existing-dataset-path", default=None, help="Use an already-built dataset path instead of rebuilding from the session")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    spec_path = (repo_root / args.spec_path).resolve() if not Path(args.spec_path).is_absolute() else Path(args.spec_path).resolve()
    history_path = (repo_root / args.history_path).resolve() if not Path(args.history_path).is_absolute() else Path(args.history_path).resolve()
    summary_path = (repo_root / args.summary_path).resolve() if not Path(args.summary_path).is_absolute() else Path(args.summary_path).resolve()
    diary_path = (repo_root / args.diary_path).resolve() if not Path(args.diary_path).is_absolute() else Path(args.diary_path).resolve()
    capture_root = (repo_root / args.capture_root).resolve() if not Path(args.capture_root).is_absolute() else Path(args.capture_root).resolve()
    dataset_root = (repo_root / args.dataset_root).resolve() if not Path(args.dataset_root).is_absolute() else Path(args.dataset_root).resolve()
    validation_root = (repo_root / args.validation_root).resolve() if not Path(args.validation_root).is_absolute() else Path(args.validation_root).resolve()
    capture_python = Path(args.capture_python).resolve()
    analysis_python = Path(args.analysis_python).resolve()

    if args.delay_seconds > 0:
        time.sleep(args.delay_seconds)

    if args.existing_session_dir:
        session_root = Path(args.existing_session_dir).resolve()
        if not session_root.exists():
            raise SystemExit(f"Missing existing session dir: {session_root}")
    else:
        capture_report = run_json_command(
            [
                str(capture_python),
                str(repo_root / "tools" / "start_btc_multivenue_capture.py"),
                "--duration-seconds",
                str(args.capture_duration_seconds),
                "--run-label",
                args.run_label,
                "--output-root",
                str(capture_root),
            ],
            cwd=repo_root,
        )
        session_root = Path(capture_report["session_root"]).resolve()

    if args.existing_dataset_path:
        dataset_path = Path(args.existing_dataset_path).resolve()
        if not dataset_path.exists():
            raise SystemExit(f"Missing existing dataset path: {dataset_path}")
    else:
        build_report = run_json_command(
            [
                str(analysis_python),
                str(repo_root / "tools" / "build_btc_multivenue_dataset.py"),
                "--capture-root",
                str(capture_root / "sessions"),
                "--session-dir",
                str(session_root),
                "--output-root",
                str(dataset_root),
            ],
            cwd=repo_root,
        )
        dataset_path = Path(build_report["output_path"]).resolve()

    validation_report = run_json_command(
        [
            str(analysis_python),
            str(repo_root / "tools" / "run_btc_meanrev_frozen_validation.py"),
            "--spec-path",
            str(spec_path),
            "--dataset-path",
            str(dataset_path),
            "--output-root",
            str(validation_root),
            "--bootstrap-samples",
            str(args.bootstrap_samples),
            "--seed",
            str(args.seed),
        ],
        cwd=repo_root,
    )

    checkpoint_id = f"btc-meanrev-downshock30-validation-oos-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
    freeze_cmd = [
        str(analysis_python),
        str(repo_root / "tools" / "freeze_btc_checkpoint.py"),
        "--checkpoint-id",
        checkpoint_id,
        "--status",
        "validation_oos",
        "--category",
        "mean_reversion_validation",
        "--summary",
        "Official out-of-sample validation run for the frozen BTC 30s downshock mean-reversion candidate.",
        "--artifact",
        f"spec_json={spec_path}",
        "--artifact",
        f"session_root={session_root}",
        "--artifact",
        f"dataset_path={dataset_path}",
        "--artifact",
        f"validation_report={validation_report_path(validation_report)}",
        "--metric",
        f"trades={safe_metric(validation_report, 'overall_result', 'trades', 0)}",
        "--metric",
        f"win_rate={safe_metric(validation_report, 'overall_result', 'win_rate', 0.0)}",
        "--metric",
        f"total_net_bps={safe_metric(validation_report, 'overall_result', 'total_net_bps', 0.0)}",
        "--metric",
        f"positive_day_share={validation_report.get('positive_day_share', 0.0) or 0.0}",
    ]
    day_bootstrap = validation_report.get("day_bootstrap") or {}
    if day_bootstrap.get("p05_total_net_bps") is not None:
        freeze_cmd.extend(["--metric", f"bootstrap_p05_total_net_bps={day_bootstrap['p05_total_net_bps']}"])
    subprocess.check_call(freeze_cmd, cwd=str(repo_root))

    history = load_json_list(history_path)
    history_entry = make_history_entry(
        checkpoint_id=checkpoint_id,
        session_root=session_root,
        dataset_path=dataset_path,
        validation_report=validation_report,
    )
    history.append(history_entry)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    summary = summarize_history(history)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    append_diary(
        diary_path=diary_path,
        checkpoint_id=checkpoint_id,
        session_root=session_root,
        dataset_path=dataset_path,
        validation_report=validation_report,
        summary=summary,
    )

    result = {
        "checkpoint_id": checkpoint_id,
        "session_root": str(session_root),
        "dataset_path": str(dataset_path),
        "validation_run": validation_report["run_name"],
        "validation_report": validation_report_path(validation_report),
        "history_path": str(history_path),
        "summary_path": str(summary_path),
        "aggregate_summary": summary,
    }
    print(json.dumps(result, indent=2))


def run_json_command(command: list[str], *, cwd: Path) -> dict[str, Any]:
    output = subprocess.check_output(command, cwd=str(cwd), text=True)
    return json.loads(output)


def validation_report_path(report: dict[str, Any]) -> str:
    run_name = report["run_name"]
    return str((DEFAULT_VALIDATION_ROOT / run_name / "reports" / "metadata.json").resolve())


def safe_metric(report: dict[str, Any], group: str, key: str, default: float | int) -> float | int:
    inner = report.get(group) or {}
    value = inner.get(key)
    return default if value is None else value


def load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def make_history_entry(*, checkpoint_id: str, session_root: Path, dataset_path: Path, validation_report: dict[str, Any]) -> dict[str, Any]:
    overall = validation_report.get("overall_result") or {}
    day_rows = validation_report.get("day_rows") or []
    return {
        "checkpoint_id": checkpoint_id,
        "captured_at": datetime.now(UTC).isoformat(),
        "session_root": str(session_root),
        "dataset_path": str(dataset_path),
        "validation_run": validation_report["run_name"],
        "validation_report": validation_report_path(validation_report),
        "overall_result": overall,
        "day_rows": day_rows,
        "positive_day_share": validation_report.get("positive_day_share"),
        "day_bootstrap": validation_report.get("day_bootstrap"),
    }


def summarize_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    day_rows: list[dict[str, Any]] = []
    total_trades = 0
    total_net_bps = 0.0
    run_ids: list[str] = []
    for item in history:
        run_ids.append(item["checkpoint_id"])
        overall = item.get("overall_result") or {}
        total_trades += int(overall.get("trades") or 0)
        total_net_bps += float(overall.get("total_net_bps") or 0.0)
        day_rows.extend(item.get("day_rows") or [])

    unique_days = sorted({row["day"] for row in day_rows if "day" in row})
    day_totals = np.array([float(row["total_net_bps"]) for row in day_rows], dtype=float) if day_rows else np.array([], dtype=float)
    positive_day_share = float(np.mean(day_totals > 0.0)) if len(day_totals) else None
    bootstrap = bootstrap_day_totals(day_totals, samples=5000, seed=42) if len(day_totals) else None

    return {
        "validation_run_count": len(history),
        "validation_checkpoints": run_ids,
        "unique_day_count": len(unique_days),
        "unique_days": unique_days,
        "total_trades": total_trades,
        "total_net_bps": total_net_bps,
        "positive_day_share": positive_day_share,
        "day_bootstrap": bootstrap,
    }


def bootstrap_day_totals(day_totals: np.ndarray, *, samples: int, seed: int) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    boot_totals = np.empty(samples, dtype=float)
    positive_shares = np.empty(samples, dtype=float)
    for idx in range(samples):
        draw = rng.choice(day_totals, size=len(day_totals), replace=True)
        boot_totals[idx] = float(draw.sum())
        positive_shares[idx] = float(np.mean(draw > 0.0))
    return {
        "samples": samples,
        "day_count": int(len(day_totals)),
        "mean_total_net_bps": float(np.mean(boot_totals)),
        "median_total_net_bps": float(np.median(boot_totals)),
        "p05_total_net_bps": float(np.quantile(boot_totals, 0.05)),
        "p25_total_net_bps": float(np.quantile(boot_totals, 0.25)),
        "p75_total_net_bps": float(np.quantile(boot_totals, 0.75)),
        "p95_total_net_bps": float(np.quantile(boot_totals, 0.95)),
        "mean_positive_day_share": float(np.mean(positive_shares)),
        "p05_positive_day_share": float(np.quantile(positive_shares, 0.05)),
    }


def append_diary(
    *,
    diary_path: Path,
    checkpoint_id: str,
    session_root: Path,
    dataset_path: Path,
    validation_report: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    overall = validation_report.get("overall_result") or {}
    lines = [
        "",
        f"## {datetime.now(UTC).date().isoformat()} - Frozen mean-reversion out-of-sample validation",
        "",
        f"- Checkpoint: `{checkpoint_id}`",
        f"- Capture session: `{session_root}`",
        f"- Dataset: `{dataset_path}`",
        f"- Validation run: `{validation_report['run_name']}`",
        f"- Trades: `{overall.get('trades')}`",
        f"- Win rate: `{overall.get('win_rate')}`",
        f"- Total net bps: `{overall.get('total_net_bps')}`",
        f"- Aggregate validation day count: `{summary.get('unique_day_count')}`",
        f"- Aggregate positive-day share: `{summary.get('positive_day_share')}`",
    ]
    day_bootstrap = summary.get("day_bootstrap") or {}
    if day_bootstrap.get("p05_total_net_bps") is not None:
        lines.append(f"- Aggregate bootstrap p05 total net bps: `{day_bootstrap['p05_total_net_bps']}`")
    with diary_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
