#!/usr/bin/env python3
"""
Wait for a BTC multivenue capture session to finish, then build and train.

This keeps the flow append-only:
- wait for session manifest
- build latest multivenue dataset
- train fresh CatBoost baselines
- freeze the training run into the checkpoint ledger
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


UTC = timezone.utc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a BTC multivenue capture session and launch training when it finishes.")
    parser.add_argument("--session-root", required=True, help="Session root created by start_btc_multivenue_capture.py")
    parser.add_argument("--poll-seconds", type=int, default=30, help="Polling interval while waiting for the session manifest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_root = Path(args.session_root).resolve()
    manifest_path = session_root / "session_manifest.json"
    tools_root = Path(__file__).resolve().parent
    repo_root = tools_root.parent
    python_bin = sys.executable

    while not manifest_path.exists():
        time.sleep(args.poll_seconds)

    build_cmd = [
        python_bin,
        str(tools_root / "build_btc_multivenue_dataset.py"),
        "--capture-root",
        str(repo_root / "output" / "btc_multivenue_capture" / "sessions"),
        "--output-root",
        str(repo_root / "output" / "btc_multivenue_dataset"),
    ]
    build_output = subprocess.check_output(build_cmd, cwd=str(repo_root), text=True)
    build_report = json.loads(build_output)
    dataset_path = build_report["output_path"]

    train_cmd = [
        python_bin,
        str(tools_root / "train_btc_multivenue_catboost.py"),
        "--dataset-path",
        dataset_path,
        "--output-root",
        str(repo_root / "output" / "btc_multivenue_models"),
    ]
    train_output = subprocess.check_output(train_cmd, cwd=str(repo_root), text=True)
    train_report = json.loads(train_output)

    trained = [model for model in train_report["models"] if model["status"] == "trained"]
    checkpoint_id = f"btc-multivenue-catboost-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
    summary = (
        "First CatBoost multivenue baselines built from Binance futures + spot + Coinbase level2."
    )
    freeze_cmd = [
        python_bin,
        str(tools_root / "freeze_btc_checkpoint.py"),
        "--checkpoint-id",
        checkpoint_id,
        "--status",
        "trained_baseline",
        "--category",
        "multivenue",
        "--summary",
        summary,
        "--artifact",
        f"session_root={session_root}",
        "--artifact",
        f"dataset_path={dataset_path}",
        "--artifact",
        f"training_run_root={train_report['run_root']}",
        "--artifact",
        f"training_report={train_report['report_metadata_path']}",
    ]
    for model in trained:
        if model["model_path"]:
            freeze_cmd.extend(["--artifact", f"{model['name']}={model['model_path']}"])
        if model["test_auc"] is not None:
            freeze_cmd.extend(["--metric", f"{model['name']}_test_auc={model['test_auc']}"])
    subprocess.check_call(freeze_cmd, cwd=str(repo_root))

    append_diary(
        repo_root / "research" / "btc" / "diary.md",
        checkpoint_id=checkpoint_id,
        session_root=session_root,
        dataset_path=Path(dataset_path),
        train_report=train_report,
    )

    result = {
        "session_root": str(session_root),
        "dataset_path": dataset_path,
        "training_run": train_report["run_name"],
        "checkpoint_id": checkpoint_id,
    }
    print(json.dumps(result, indent=2))


def append_diary(diary_path: Path, *, checkpoint_id: str, session_root: Path, dataset_path: Path, train_report: dict[str, object]) -> None:
    lines = [
        "",
        f"## {datetime.now(UTC).date().isoformat()} - Auto-trained multivenue CatBoost baselines",
        "",
        f"- Capture session: `{session_root}`",
        f"- Dataset: `{dataset_path}`",
        f"- Training run: `{train_report['run_name']}`",
        f"- Checkpoint: `{checkpoint_id}`",
    ]
    for model in train_report["models"]:
        lines.append(
            f"- {model['name']}: status `{model['status']}`, test AUC `{model['test_auc']}`"
        )
    with diary_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
