#!/usr/bin/env python3
"""
Run a frozen, append-only Binance historical BTC mean-reversion backfill cycle.

This does not touch the existing live multivenue model. It creates a new
Binance-only research track using official bulk archives.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


UTC = timezone.utc
DEFAULT_PYTHON = Path("/Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14")
DEFAULT_RAW_ROOT = Path("output/btc_binance_historical/raw")
DEFAULT_DATASET_ROOT = Path("output/btc_binance_historical/datasets")
DEFAULT_MODEL_ROOT = Path("output/btc_binance_historical/models")
DEFAULT_HYBRID_ROOT = Path("output/btc_binance_historical/hybrid_search")
DEFAULT_PROJECT_ROOT = Path("research/btc/projects/btc-binance-historical-meanrev-v1")
DEFAULT_DIARY_PATH = Path("research/btc/diary.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Binance historical BTC mean-reversion backfill.")
    parser.add_argument("--start-date", required=True, help="Inclusive start date YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="Inclusive end date YYYY-MM-DD")
    parser.add_argument("--symbol", default="BTCUSDT", help="Binance symbol, default BTCUSDT")
    parser.add_argument("--python", default=str(DEFAULT_PYTHON), help="Python executable")
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT), help="Raw download root")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT), help="Dataset output root")
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT), help="Model output root")
    parser.add_argument("--hybrid-root", default=str(DEFAULT_HYBRID_ROOT), help="Hybrid replay output root")
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT), help="Project bookkeeping root")
    parser.add_argument("--diary-path", default=str(DEFAULT_DIARY_PATH), help="BTC diary path")
    parser.add_argument("--max-download-workers", type=int, default=4, help="Parallel download workers")
    parser.add_argument("--bucket-seconds", type=int, default=1, help="Dataset bucket size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    python_exe = Path(args.python).resolve()
    raw_root = resolve_path(repo_root, args.raw_root)
    dataset_root = resolve_path(repo_root, args.dataset_root)
    model_root = resolve_path(repo_root, args.model_root)
    hybrid_root = resolve_path(repo_root, args.hybrid_root)
    project_root = resolve_path(repo_root, args.project_root)
    diary_path = resolve_path(repo_root, args.diary_path)
    project_root.mkdir(parents=True, exist_ok=True)

    download_report = run_json(
        [
            str(python_exe),
            str(repo_root / "tools" / "download_btc_binance_historical.py"),
            "--symbol",
            args.symbol,
            "--start-date",
            args.start_date,
            "--end-date",
            args.end_date,
            "--output-root",
            str(raw_root),
            "--max-download-workers",
            str(args.max_download_workers),
        ],
        cwd=repo_root,
    )
    dataset_report = run_json(
        [
            str(python_exe),
            str(repo_root / "tools" / "build_btc_binance_historical_dataset.py"),
            "--symbol",
            args.symbol,
            "--start-date",
            args.start_date,
            "--end-date",
            args.end_date,
            "--bucket-seconds",
            str(args.bucket_seconds),
            "--raw-root",
            str(raw_root),
            "--output-root",
            str(dataset_root),
        ],
        cwd=repo_root,
    )
    dataset_path = Path(dataset_report["output_path"]).resolve()

    train_report = run_json(
        [
            str(python_exe),
            str(repo_root / "tools" / "train_btc_multivenue_catboost.py"),
            "--dataset-path",
            str(dataset_path),
            "--output-root",
            str(model_root),
            "--continuation-horizon-seconds",
            "30",
            "--profit-bps",
            "8",
            "--meanrev-shock-window-seconds",
            "5",
            "--meanrev-shock-bps",
            "5",
            "--meanrev-revert-bps",
            "4",
            "--min-rows",
            "200",
        ],
        cwd=repo_root,
    )
    model_path = pick_meanrev_model_path(train_report)

    hybrid_report = run_json(
        [
            str(python_exe),
            str(repo_root / "tools" / "run_btc_meanrev_hybrid_search.py"),
            "--dataset-path",
            str(dataset_path),
            "--model-path",
            str(model_path),
            "--output-root",
            str(hybrid_root),
            "--shock-window-seconds",
            "5",
            "--shock-bps",
            "5",
            "--candidate-direction",
            "down",
            "--trade-direction",
            "long",
        ],
        cwd=repo_root,
    )

    checkpoint_id = f"btc-binance-historical-meanrev-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
    hybrid = hybrid_report.get("best_stressed_result") or {}
    subprocess.check_call(
        [
            str(python_exe),
            str(repo_root / "tools" / "freeze_btc_checkpoint.py"),
            "--checkpoint-id",
            checkpoint_id,
            "--status",
            "historical_replay_candidate",
            "--category",
            "binance_historical_mean_reversion",
            "--summary",
            "Binance-only historical BTC downshock mean-reversion backfill using official bulk archives.",
            "--artifact",
            f"project_root={project_root}",
            "--artifact",
            f"dataset_path={dataset_path}",
            "--artifact",
            f"training_report={train_report['report_metadata_path']}",
            "--artifact",
            f"hybrid_report={hybrid_report_path(hybrid_root, hybrid_report)}",
            "--metric",
            f"row_count={dataset_report['row_count']}",
            "--metric",
            f"candidate_events={hybrid_report.get('candidate_event_count', 0)}",
            "--metric",
            f"best_stressed_mean_total_net_bps={hybrid.get('mean_total_net_bps', 0.0)}",
            "--metric",
            f"best_stressed_p05_total_net_bps={hybrid.get('p05_total_net_bps', 0.0)}",
        ],
        cwd=str(repo_root),
    )

    write_project_plan(
        project_root=project_root,
        args=args,
        dataset_path=dataset_path,
        train_report=train_report,
        hybrid_report=hybrid_report,
        checkpoint_id=checkpoint_id,
    )
    append_diary(
        diary_path=diary_path,
        checkpoint_id=checkpoint_id,
        dataset_path=dataset_path,
        train_report=train_report,
        hybrid_report=hybrid_report,
    )

    result = {
        "checkpoint_id": checkpoint_id,
        "dataset_path": str(dataset_path),
        "training_report": train_report["report_metadata_path"],
        "hybrid_report": hybrid_report_path(hybrid_root, hybrid_report),
        "project_root": str(project_root),
    }
    print(json.dumps(result, indent=2))


def run_json(command: list[str], *, cwd: Path) -> dict[str, object]:
    output = subprocess.check_output(command, cwd=str(cwd), text=True)
    return json.loads(output)


def resolve_path(repo_root: Path, raw: str) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def pick_meanrev_model_path(train_report: dict[str, object]) -> Path:
    for model in train_report.get("models", []):
        if model.get("name") == "meanrev_after_downshock_30s" and model.get("status") == "trained":
            return Path(str(model["model_path"])).resolve()
    raise SystemExit("Historical Binance backfill did not train meanrev_after_downshock_30s successfully.")


def hybrid_report_path(hybrid_root: Path, hybrid_report: dict[str, object]) -> str:
    run_name = str(hybrid_report["run_name"])
    return str((hybrid_root / run_name / "reports" / "metadata.json").resolve())


def write_project_plan(
    *,
    project_root: Path,
    args: argparse.Namespace,
    dataset_path: Path,
    train_report: dict[str, object],
    hybrid_report: dict[str, object],
    checkpoint_id: str,
) -> None:
    plan = f"""# BTC Binance Historical Mean Reversion v1

This project is separate from the live multivenue BTC shadow sleeve.

- Reason: the live frozen model depends on Coinbase features that do not exist in Binance's bulk archive.
- Goal: use official Binance bulk archives to expand historical BTC mean-reversion discovery and replay coverage.
- Symbol: `{args.symbol.upper()}`
- Date range: `{args.start_date}` to `{args.end_date}`
- Dataset: `{dataset_path}`
- Training run: `{train_report['run_name']}`
- Hybrid replay run: `{hybrid_report['run_name']}`
- Frozen checkpoint: `{checkpoint_id}`
"""
    (project_root / "plan.md").write_text(plan, encoding="utf-8")


def append_diary(
    *,
    diary_path: Path,
    checkpoint_id: str,
    dataset_path: Path,
    train_report: dict[str, object],
    hybrid_report: dict[str, object],
) -> None:
    stressed = hybrid_report.get("best_stressed_result") or {}
    entry = f"""

## {datetime.now(UTC).date().isoformat()} - Binance historical mean-reversion backfill

- Checkpoint: `{checkpoint_id}`
- Dataset: `{dataset_path}`
- Training run: `{train_report['run_name']}`
- Hybrid replay run: `{hybrid_report['run_name']}`
- Candidate events: `{hybrid_report.get('candidate_event_count', 0)}`
- Best stressed mean total net: `{stressed.get('mean_total_net_bps')}`
- Best stressed p05 total net: `{stressed.get('p05_total_net_bps')}`
- Read: this is the first Binance-only historical replay track using official bulk archive data, kept separate from the frozen multivenue live model.
"""
    with diary_path.open("a", encoding="utf-8") as handle:
        handle.write(entry)


if __name__ == "__main__":
    main()
