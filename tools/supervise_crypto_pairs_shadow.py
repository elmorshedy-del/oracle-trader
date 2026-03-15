#!/usr/bin/env python3
"""Detached external supervisor for the crypto pairs shadow runner."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.crypto_pairs.discovery import resolve_discovery_report_path


UTC = timezone.utc
DEFAULT_STATE_ROOT = Path("output/crypto_pairs/shadow_supervision")
DEFAULT_PYTHON = Path("/Users/ahmedelmorshedy/.local/bin/oracle-btc-python")
DEFAULT_RUNNER = Path("tools/run_crypto_pairs_shadow.py")
STATE_FILE_NAME = "state.json"
SUPERVISOR_LOG_FILE_NAME = "supervisor.log"
STOP_FILE_NAME = "stop.requested"
DEFAULT_RESTART_DELAY_SECONDS = 5
DEFAULT_STATUS_POLL_SECONDS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervise the crypto pairs shadow runner as a detached external process.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start", help="Launch a detached supervisor process.")
    add_common_run_args(start_parser)
    start_parser.add_argument("--supervisor-id", default=None)
    start_parser.add_argument("--force", action="store_true", help="Reuse the requested supervisor id even if stale state exists.")

    status_parser = subparsers.add_parser("status", help="Read the current supervisor state.")
    status_parser.add_argument("--supervisor-id", required=True)
    status_parser.add_argument("--state-root", default=str(DEFAULT_STATE_ROOT))

    stop_parser = subparsers.add_parser("stop", help="Stop a running supervisor and its child runner.")
    stop_parser.add_argument("--supervisor-id", required=True)
    stop_parser.add_argument("--state-root", default=str(DEFAULT_STATE_ROOT))

    supervise_parser = subparsers.add_parser("_supervise", help=argparse.SUPPRESS)
    add_common_run_args(supervise_parser)
    supervise_parser.add_argument("--supervisor-id", required=True)

    return parser.parse_args()


def add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-root", default=str(DEFAULT_STATE_ROOT))
    parser.add_argument("--discovery-report", default=None)
    parser.add_argument("--top-pairs", type=int, default=3)
    parser.add_argument("--pair-key", action="append", default=[], help="Explicit pair key like AAVE/DOGE; can be passed multiple times")
    parser.add_argument("--runtime-seconds", type=int, default=3900)
    parser.add_argument("--restart-delay-seconds", type=int, default=DEFAULT_RESTART_DELAY_SECONDS)
    parser.add_argument("--max-restarts", type=int, default=10)
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--runner-path", default=str(DEFAULT_RUNNER))


def main() -> None:
    args = parse_args()
    if args.command == "start":
        command_start(args)
        return
    if args.command == "status":
        command_status(args)
        return
    if args.command == "stop":
        command_stop(args)
        return
    if args.command == "_supervise":
        command_supervise(args)
        return
    raise SystemExit(f"Unknown command {args.command}")


def command_start(args: argparse.Namespace) -> None:
    supervisor_id = args.supervisor_id or build_supervisor_id()
    supervisor_root = resolve_path(args.state_root) / supervisor_id
    if supervisor_root.exists() and not args.force:
        state = load_state(supervisor_root)
        if state and pid_is_alive(state.get("supervisor_pid")):
            raise SystemExit(f"Supervisor {supervisor_id} is already running")
    supervisor_root.mkdir(parents=True, exist_ok=True)
    state = {
        "supervisor_id": supervisor_id,
        "status": "starting",
        "requested_at": now_iso(),
        "discovery_report": str(resolve_discovery_report_path(args.discovery_report)),
        "top_pairs": args.top_pairs,
        "pair_keys": args.pair_key,
        "runtime_seconds": args.runtime_seconds,
        "restart_delay_seconds": args.restart_delay_seconds,
        "max_restarts": args.max_restarts,
        "attempts": [],
        "restarts": 0,
    }
    write_state(supervisor_root, state)

    supervisor_log = supervisor_root / SUPERVISOR_LOG_FILE_NAME
    with supervisor_log.open("a", encoding="utf-8") as log_handle:
        child = subprocess.Popen(
            [
                str(resolve_path(args.python)),
                str(Path(__file__).resolve()),
                "_supervise",
                "--supervisor-id",
                supervisor_id,
                "--state-root",
                str(resolve_path(args.state_root)),
                "--discovery-report",
                str(resolve_discovery_report_path(args.discovery_report)),
                "--top-pairs",
                str(args.top_pairs),
                *[
                    item
                    for pair_key in args.pair_key
                    for item in ("--pair-key", pair_key)
                ],
                "--runtime-seconds",
                str(args.runtime_seconds),
                "--restart-delay-seconds",
                str(args.restart_delay_seconds),
                "--max-restarts",
                str(args.max_restarts),
                "--python",
                str(resolve_path(args.python)),
                "--runner-path",
                str(resolve_path(args.runner_path)),
            ],
            cwd=str(REPO_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    state["supervisor_pid"] = child.pid
    state["status"] = "launched"
    state["updated_at"] = now_iso()
    write_state(supervisor_root, state)
    print(
        json.dumps(
            {
                "supervisor_id": supervisor_id,
                "supervisor_root": str(supervisor_root),
                "supervisor_pid": child.pid,
                "state_file": str(supervisor_root / STATE_FILE_NAME),
                "supervisor_log": str(supervisor_log),
            },
            indent=2,
        )
    )


def command_status(args: argparse.Namespace) -> None:
    supervisor_root = resolve_path(args.state_root) / args.supervisor_id
    state = load_state(supervisor_root)
    if state is None:
        raise SystemExit(f"No state found for {args.supervisor_id}")
    state["supervisor_alive"] = pid_is_alive(state.get("supervisor_pid"))
    state["child_alive"] = pid_is_alive(state.get("child_pid"))
    state["stop_requested"] = (supervisor_root / STOP_FILE_NAME).exists()
    print(json.dumps(state, indent=2))


def command_stop(args: argparse.Namespace) -> None:
    supervisor_root = resolve_path(args.state_root) / args.supervisor_id
    state = load_state(supervisor_root)
    if state is None:
        raise SystemExit(f"No state found for {args.supervisor_id}")
    stop_file = supervisor_root / STOP_FILE_NAME
    stop_file.write_text(now_iso(), encoding="utf-8")
    child_pid = state.get("child_pid")
    if pid_is_alive(child_pid):
        terminate_process_group(int(child_pid), graceful=True)
    state["status"] = "stop_requested"
    state["updated_at"] = now_iso()
    write_state(supervisor_root, state)
    print(json.dumps({"supervisor_id": args.supervisor_id, "status": "stop_requested"}, indent=2))


def command_supervise(args: argparse.Namespace) -> None:
    supervisor_root = resolve_path(args.state_root) / args.supervisor_id
    supervisor_root.mkdir(parents=True, exist_ok=True)
    stop_file = supervisor_root / STOP_FILE_NAME
    stop_file.unlink(missing_ok=True)
    discovery_report = str(resolve_discovery_report_path(args.discovery_report))
    state = load_state(supervisor_root) or {}
    supervisor_pid = os.getpid()
    state.update(
        {
            "supervisor_id": args.supervisor_id,
            "status": "running",
            "supervisor_pid": supervisor_pid,
            "started_at": state.get("started_at", now_iso()),
            "updated_at": now_iso(),
            "discovery_report": discovery_report,
            "top_pairs": args.top_pairs,
            "pair_keys": args.pair_key,
            "runtime_seconds": args.runtime_seconds,
            "restart_delay_seconds": args.restart_delay_seconds,
            "max_restarts": args.max_restarts,
            "attempts": state.get("attempts", []),
            "restarts": int(state.get("restarts", 0)),
        }
    )
    write_state(supervisor_root, state)

    python_exe = resolve_path(args.python)
    runner_path = resolve_path(args.runner_path)
    start_monotonic = time.monotonic()
    attempt_number = len(state["attempts"]) + 1

    while True:
        elapsed = time.monotonic() - start_monotonic
        remaining = max(0, args.runtime_seconds - int(elapsed))
        if stop_file.exists():
            finalize_state(supervisor_root, status="stopped", child_pid=None)
            return
        if remaining <= 0:
            finalize_state(supervisor_root, status="completed", child_pid=None)
            return

        worker_log = supervisor_root / f"worker_attempt_{attempt_number:03d}.log"
        with worker_log.open("a", encoding="utf-8") as log_handle:
            child = subprocess.Popen(
                [
                    str(python_exe),
                    str(runner_path),
                    "--discovery-report",
                    discovery_report,
                    "--top-pairs",
                    str(args.top_pairs),
                    *[
                        item
                        for pair_key in args.pair_key
                        for item in ("--pair-key", pair_key)
                    ],
                    "--runtime-seconds",
                    str(remaining),
                ],
                cwd=str(REPO_ROOT),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        attempt_state = {
            "attempt": attempt_number,
            "child_pid": child.pid,
            "worker_log": str(worker_log),
            "started_at": now_iso(),
            "requested_runtime_seconds": remaining,
        }
        state = load_state(supervisor_root) or state
        state["child_pid"] = child.pid
        state["child_started_at"] = attempt_state["started_at"]
        state["current_worker_log"] = str(worker_log)
        state["status"] = "running"
        state["updated_at"] = now_iso()
        state.setdefault("attempts", []).append(attempt_state)
        write_state(supervisor_root, state)

        returncode = monitor_child(child=child, supervisor_root=supervisor_root)
        state = load_state(supervisor_root) or state
        state["child_pid"] = None
        state["updated_at"] = now_iso()
        state["last_child_returncode"] = returncode
        state["current_worker_log"] = str(worker_log)
        state["attempts"][-1]["finished_at"] = now_iso()
        state["attempts"][-1]["returncode"] = returncode
        write_state(supervisor_root, state)

        if stop_file.exists():
            finalize_state(supervisor_root, status="stopped", child_pid=None)
            return
        if returncode == 0:
            # Normal exit before the wall clock runtime is treated as completed unless we still have time and want restarts.
            finalize_state(supervisor_root, status="completed", child_pid=None)
            return

        restarts = int(state.get("restarts", 0))
        if restarts >= args.max_restarts:
            finalize_state(supervisor_root, status="failed_max_restarts", child_pid=None, last_child_returncode=returncode)
            return

        state["restarts"] = restarts + 1
        state["status"] = "restarting"
        state["updated_at"] = now_iso()
        write_state(supervisor_root, state)
        time.sleep(args.restart_delay_seconds)
        attempt_number += 1


def monitor_child(*, child: subprocess.Popen[bytes], supervisor_root: Path) -> int:
    stop_file = supervisor_root / STOP_FILE_NAME
    while True:
        returncode = child.poll()
        if returncode is not None:
            return returncode
        if stop_file.exists():
            terminate_process_group(child.pid, graceful=True)
            return child.wait()
        time.sleep(DEFAULT_STATUS_POLL_SECONDS)


def terminate_process_group(pid: int, *, graceful: bool) -> None:
    try:
        process_group_id = os.getpgid(pid)
    except ProcessLookupError:
        return
    sig = signal.SIGTERM if graceful else signal.SIGKILL
    try:
        os.killpg(process_group_id, sig)
    except ProcessLookupError:
        return


def pid_is_alive(pid: object) -> bool:
    if pid is None:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError, TypeError):
        return False
    return True


def finalize_state(supervisor_root: Path, *, status: str, child_pid: int | None, **extra: object) -> None:
    state = load_state(supervisor_root) or {}
    state["status"] = status
    state["child_pid"] = child_pid
    state["updated_at"] = now_iso()
    state["finished_at"] = now_iso()
    for key, value in extra.items():
        state[key] = value
    write_state(supervisor_root, state)


def load_state(supervisor_root: Path) -> dict[str, object] | None:
    path = supervisor_root / STATE_FILE_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_state(supervisor_root: Path, state: dict[str, object]) -> None:
    (supervisor_root / STATE_FILE_NAME).write_text(json.dumps(state, indent=2), encoding="utf-8")


def resolve_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def build_supervisor_id() -> str:
    return f"crypto_pairs_shadow_supervisor_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}_v1"


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


if __name__ == "__main__":
    main()
