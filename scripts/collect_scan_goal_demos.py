#!/usr/bin/env python3
"""Resumable fresh-stack collection of 40 DWB/RPP scan-goal demonstrations."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "generated" / "nav2_portability_params.yaml"
PROFILE = ROOT / "generated" / "nav2_dwb_compiled.yaml"
BRIDGE = ROOT / "nodes" / "candidate_bridge.py"
RECORDER = ROOT / "nodes" / "scan_goal_trace_recorder.py"
TRIAL = ROOT / "adapters" / "scan_goal_expert_trial.py"
REVISION = "scan-goal-demo-matrix-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stop(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=12)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=4)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=4)


def launch(command: list[str], log: Path, env: dict[str, str]):
    handle = log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command, stdout=handle, stderr=subprocess.STDOUT, text=True,
        env=env, start_new_session=True,
    )
    return process, handle


def valid(stem: str, result_dir: Path, expert: str) -> bool:
    trace_path, task_path = result_dir / f"{stem}__trace.json", result_dir / f"{stem}__task.json"
    try:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        task = json.loads(task_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    rows = trace.get("samples", [])
    return bool(
        trace.get("run_id") == stem
        and trace.get("expert") == expert
        and task.get("revision") == "scan-goal-expert-trial-v1"
        and task.get("expert") == expert
        and task.get("goal_result") == "SUCCEEDED"
        and len(rows) >= 40
        and all(row.get("sequence") == index and len(row.get("feature", [])) == 40 for index, row in enumerate(rows))
    )


def archive(stem: str, result_dir: Path) -> None:
    files = [path for path in result_dir.glob(f"{stem}*" ) if path.is_file()]
    if not files:
        return
    target = result_dir / "infra_attempts"
    target.mkdir(exist_ok=True)
    attempt = 1
    while any(target.glob(f"{stem}__attempt{attempt}*")):
        attempt += 1
    for path in files:
        suffix = path.name[len(stem) :].lstrip("_")
        shutil.move(str(path), str(target / f"{stem}__attempt{attempt}__{suffix}"))


def run_one(index: int, expert: str, goal: tuple[float, float], result_dir: Path, domain: int) -> bool:
    stem = f"demo_{index:02d}_{expert.lower()}"
    if valid(stem, result_dir, expert):
        return True
    archive(stem, result_dir)
    env = os.environ.copy()
    env.update(
        {
            "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
            "ROS_LOCALHOST_ONLY": "1",
            "ROS2CLI_NO_DAEMON": "1",
            "ROS_DOMAIN_ID": str(domain),
            "PYTHONPATH": f"{ROOT}:{ROOT / 'adapters'}:{ROOT / 'nodes'}:{env.get('PYTHONPATH', '')}",
        }
    )
    processes, handles = [], []
    try:
        stack, handle = launch(
            [
                "ros2", "launch", "nav2_bringup", "tb3_simulation_launch.py",
                "headless:=True", "use_rviz:=False", "use_composition:=False",
                "autostart:=True", f"params_file:={PARAMS}",
            ],
            result_dir / f"{stem}__stack.log", env,
        )
        processes.append(stack); handles.append(handle)
        time.sleep(7)
        bridge, handle = launch([sys.executable, str(BRIDGE)], result_dir / f"{stem}__bridge.log", env)
        processes.append(bridge); handles.append(handle)
        trace_path = result_dir / f"{stem}__trace.json"
        recorder, handle = launch(
            [
                sys.executable, str(RECORDER), "--run-id", stem, "--expert", expert,
                "--goal-x", str(goal[0]), "--goal-y", str(goal[1]), "--output", str(trace_path),
            ],
            result_dir / f"{stem}__recorder.log", env,
        )
        processes.append(recorder); handles.append(handle)
        time.sleep(1)
        task_path = result_dir / f"{stem}__task.json"
        with (result_dir / f"{stem}__trial.log").open("w", encoding="utf-8") as trial_log:
            subprocess.run(
                [
                    sys.executable, str(TRIAL), "--expert", expert,
                    "--goal-x", str(goal[0]), "--goal-y", str(goal[1]),
                    "--profile", str(PROFILE), "--output", str(task_path),
                ],
                stdout=trial_log, stderr=subprocess.STDOUT, text=True,
                env=env, timeout=220, check=False,
            )
        stop(recorder)
        time.sleep(0.5)
        return valid(stem, result_dir, expert)
    except (OSError, subprocess.TimeoutExpired) as error:
        (result_dir / f"{stem}__runner_error.log").write_text(repr(error) + "\n", encoding="utf-8")
        return False
    finally:
        for process in reversed(processes):
            stop(process)
        for handle in handles:
            handle.close()
        time.sleep(1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=40)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=256)
    parser.add_argument("--domain", type=int, default=215)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    if args.runs % 2:
        raise ValueError("runs must be even for balanced DWB/RPP demonstrations")
    args.result_dir.mkdir(parents=True, exist_ok=True)
    schedule = [(index, "DWB" if index % 2 else "RPP") for index in range(1, args.runs + 1)]
    random.Random(args.seed).shuffle(schedule)
    goal = (2.0, 0.5)
    manifest = {
        "revision": REVISION,
        "created_at": datetime.now().astimezone().isoformat(),
        "seed": args.seed,
        "runs": args.runs,
        "experts": {"DWB": args.runs // 2, "RPP": args.runs // 2},
        "goal": list(goal),
        "rmw_implementation": "rmw_fastrtps_cpp",
        "schedule": schedule,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (Path(__file__), PARAMS, PROFILE, BRIDGE, RECORDER, TRIAL)
        },
    }
    (args.result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    failures = []
    for position, (index, expert) in enumerate(schedule, 1):
        okay = False
        for _attempt in range(args.retries + 1):
            if run_one(index, expert, goal, args.result_dir, args.domain):
                okay = True
                break
        status = "done" if okay else "infra_failure"
        with (args.result_dir / "progress.log").open("a", encoding="utf-8") as stream:
            stream.write(f"{position}/{len(schedule)} demo_{index:02d}_{expert.lower()}: {status}\n")
        print(f"{position}/{len(schedule)} demo_{index:02d}_{expert.lower()}: {status}", flush=True)
        if not okay:
            failures.append([index, expert])
    summary = {
        "revision": REVISION,
        "scheduled_runs": len(schedule),
        "valid_runs": sum(valid(f"demo_{index:02d}_{expert.lower()}", args.result_dir, expert) for index, expert in schedule),
        "failures": failures,
    }
    summary["complete"] = summary["valid_runs"] == summary["scheduled_runs"]
    (args.result_dir / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
