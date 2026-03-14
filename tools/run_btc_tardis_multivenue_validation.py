#!/usr/bin/env python3
"""
Download Tardis BTC multivenue history and run the frozen March 13 validator.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


DEFAULT_PYTHON = Path("/Users/ahmedelmorshedy/.local/bin/oracle-btc-python")
DEFAULT_RAW_ROOT = Path("output/btc_tardis_multivenue/raw")
DEFAULT_DATASET_ROOT = Path("output/btc_tardis_multivenue/datasets")
DEFAULT_VALIDATION_ROOT = Path("output/btc_tardis_multivenue/validation")
DEFAULT_PROJECT_ROOT = Path("research/btc/projects/btc-tardis-multivenue-validation-v1")
DEFAULT_DIARY_PATH = Path("research/btc/diary.md")
DEFAULT_SPEC_PATH = Path("research/btc/projects/btc-meanrev-downshock30-v1/validation_spec.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen BTC validation on Tardis multivenue history.")
    parser.add_argument("--start-date", required=True, help="Inclusive start date YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="Inclusive end date YYYY-MM-DD")
    parser.add_argument("--python", default=str(DEFAULT_PYTHON), help="Python executable")
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT), help="Raw Tardis download root")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT), help="Dataset output root")
    parser.add_argument("--validation-root", default=str(DEFAULT_VALIDATION_ROOT), help="Validation output root")
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT), help="Project bookkeeping root")
    parser.add_argument("--diary-path", default=str(DEFAULT_DIARY_PATH), help="BTC diary path")
    parser.add_argument("--spec-path", default=str(DEFAULT_SPEC_PATH), help="Frozen validation spec path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    python_exe = resolve_path(repo_root, args.python)
    raw_root = resolve_path(repo_root, args.raw_root)
    dataset_root = resolve_path(repo_root, args.dataset_root)
    validation_root = resolve_path(repo_root, args.validation_root)
    project_root = resolve_path(repo_root, args.project_root)
    diary_path = resolve_path(repo_root, args.diary_path)
    spec_path = resolve_path(repo_root, args.spec_path)
    project_root.mkdir(parents=True, exist_ok=True)

    download_report = run_json(
        [
            str(python_exe),
            str(repo_root / "tools" / "download_btc_tardis_multivenue.py"),
            "--start-date",
            args.start_date,
            "--end-date",
            args.end_date,
            "--output-root",
            str(raw_root),
        ],
        cwd=repo_root,
    )
    dataset_report = run_json(
        [
            str(python_exe),
            str(repo_root / "tools" / "build_btc_tardis_multivenue_dataset.py"),
            "--start-date",
            args.start_date,
            "--end-date",
            args.end_date,
            "--raw-root",
            str(raw_root),
            "--output-root",
            str(dataset_root),
        ],
        cwd=repo_root,
    )
    dataset_path = Path(str(dataset_report["output_path"])).resolve()
    validation_report = run_json(
        [
            str(python_exe),
            str(repo_root / "tools" / "run_btc_meanrev_frozen_validation.py"),
            "--spec-path",
            str(spec_path),
            "--dataset-path",
            str(dataset_path),
            "--output-root",
            str(validation_root),
        ],
        cwd=repo_root,
    )

    checkpoint_id = f"btc-tardis-multivenue-validation-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
    validation_metadata_path = validation_root / str(validation_report["run_name"]) / "reports" / "metadata.json"
    subprocess.check_call(
        [
            str(python_exe),
            str(repo_root / "tools" / "freeze_btc_checkpoint.py"),
            "--checkpoint-id",
            checkpoint_id,
            "--status",
            "historical_multivenue_validation",
            "--category",
            "tardis_multivenue_validation",
            "--summary",
            "Frozen March 13 BTC mean-reversion model validated on historical multivenue Tardis data with no new features.",
            "--artifact",
            f"project_root={project_root}",
            "--artifact",
            f"dataset_path={dataset_path}",
            "--artifact",
            f"spec_path={spec_path}",
            "--artifact",
            f"validation_report={validation_metadata_path.resolve()}",
            "--metric",
            f"row_count={dataset_report['row_count']}",
            "--metric",
            f"candidate_events={validation_report.get('candidate_event_count', 0)}",
            "--metric",
            f"total_net_bps={((validation_report.get('overall_result') or {}).get('total_net_bps', 0.0))}",
            "--metric",
            f"trade_count={((validation_report.get('overall_result') or {}).get('trades', 0))}",
        ],
        cwd=str(repo_root),
    )

    write_project_plan(
        project_root=project_root,
        args=args,
        dataset_path=dataset_path,
        validation_report=validation_report,
        checkpoint_id=checkpoint_id,
    )
    append_diary(
        diary_path=diary_path,
        args=args,
        dataset_path=dataset_path,
        validation_report=validation_report,
        checkpoint_id=checkpoint_id,
    )

    result = {
        "checkpoint_id": checkpoint_id,
        "dataset_path": str(dataset_path),
        "validation_report": str(validation_metadata_path.resolve()),
        "project_root": str(project_root),
        "downloaded_file_count": download_report.get("downloaded_file_count", 0),
    }
    print(json.dumps(result, indent=2))


def run_json(command: list[str], *, cwd: Path) -> dict[str, object]:
    output = subprocess.check_output(command, cwd=str(cwd), text=True)
    return json.loads(output)


def resolve_path(repo_root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def write_project_plan(
    *,
    project_root: Path,
    args: argparse.Namespace,
    dataset_path: Path,
    validation_report: dict[str, object],
    checkpoint_id: str,
) -> None:
    overall = validation_report.get("overall_result") or {}
    text = f"""# BTC Tardis Multivenue Validation v1

This project reuses the frozen March 13 BTC mean-reversion model/spec without retraining.

- Start date: `{args.start_date}`
- End date: `{args.end_date}`
- Dataset: `{dataset_path}`
- Validation run: `{validation_report['run_name']}`
- Frozen checkpoint: `{checkpoint_id}`
- Trades: `{overall.get('trades', 0)}`
- Total net bps: `{overall.get('total_net_bps', 0.0)}`
"""
    (project_root / "plan.md").write_text(text, encoding="utf-8")


def append_diary(
    *,
    diary_path: Path,
    args: argparse.Namespace,
    dataset_path: Path,
    validation_report: dict[str, object],
    checkpoint_id: str,
) -> None:
    overall = validation_report.get("overall_result") or {}
    entry = f"""

## {datetime.now(UTC).date().isoformat()} - Tardis multivenue frozen validation

- Checkpoint: `{checkpoint_id}`
- Date range: `{args.start_date}` to `{args.end_date}`
- Dataset: `{dataset_path}`
- Validation run: `{validation_report['run_name']}`
- Candidate events: `{validation_report.get('candidate_event_count', 0)}`
- Trades: `{overall.get('trades', 0)}`
- Win rate: `{overall.get('win_rate', 0.0):.4f}`
- Total net bps: `{overall.get('total_net_bps', 0.0):.2f}`
"""
    with diary_path.open("a", encoding="utf-8") as handle:
        handle.write(entry)


if __name__ == "__main__":
    main()
