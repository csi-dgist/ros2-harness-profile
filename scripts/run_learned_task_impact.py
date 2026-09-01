#!/usr/bin/env python3
"""Randomized, resumable 320-run causal learned-policy Nav2 matrix."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "generated" / "nav2_public_learned_params.yaml"
PROFILE = ROOT / "generated" / "nav2_dwb_compiled_task.yaml"
BRIDGE = ROOT / "nodes" / "candidate_bridge.py"
LEARNED = ROOT / "nodes" / "scan_goal_learned_controller.py"
TRIAL = ROOT / "adapters" / "nav2_task_impact_trial.py"
CORE = ROOT / "pit_core" / "core.py"
REVISION = "learned-task-impact-matrix-v1"
RMWS = ("rmw_fastrtps_cpp", "rmw_cyclonedds_cpp")
POLICIES = ("reactive_scan_goal_bc", "temporal_scan_goal_bc")
MODES = ("independent", "pit")
SCENES = ("static", "crossing")
CONDITIONS = ("normal", "burst_late")
CAPABILITY_FILES = {
    rmw: ROOT / "capabilities" / f"{rmw}.json" for rmw in RMWS
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_path(policy: str, weights_dir: Path) -> Path:
    return weights_dir / f"{policy}.npz"


def slug(cell: tuple[str, str, str, str, str, int]) -> str:
    rmw, policy, mode, scene, condition, seed = cell
    return f"{rmw}__{policy}__{mode}__{scene}__{condition}__{seed:02d}"


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
        command, stdout=handle, stderr=subprocess.STDOUT,
        text=True, env=env, start_new_session=True,
    )
    return process, handle


def load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def instrumentation_valid(cell: tuple, result_dir: Path, weights_dir: Path) -> bool:
    rmw, policy, mode, scene, condition, seed = cell
    stem = slug(cell)
    row = load(result_dir / f"{stem}.json")
    metrics = load(result_dir / f"{stem}__model.json")
    weights = model_path(policy, weights_dir)
    if row is None or metrics is None or not weights.exists():
        return False
    crossing_ok = scene == "static" or (
        row.get("obstacle_spawned") and row.get("obstacle_activated")
        and len(row.get("clearance_trace", [])) > 0
    )
    return bool(
        row.get("revision") == "learned-task-impact-trial-v7-near-path-obstacle-cross-track"
        and row.get("rmw_implementation") == rmw
        and row.get("policy") == policy
        and row.get("mode") == mode
        and row.get("scene") == scene
        and row.get("condition") == condition
        and row.get("seed") == seed
        and row.get("goal_result") not in {None, "NOT_STARTED", "TIMEOUT_WALL"}
        and row.get("model_weight_sha256") == sha256(weights)
        and row.get("profile_sha256") == sha256(PROFILE)
        and row.get("adapter_sha256") == sha256(TRIAL)
        and row.get("application_source_modified_loc") == 0
        and row.get("controller_source_modified_loc") == 0
        and row.get("active_primary_source") == policy
        and row.get("nav2_shadow_fallback_controller") == "DWB"
        and row.get("prohibited_mppi_command_admissions") == 0
        and row.get("selector_history", [{}])[0].get("controller") == "DWB"
        and row.get("analysis_anchor_wall_ns") is not None
        and isinstance(row.get("raw_events"), list) and len(row["raw_events"]) > 0
        and isinstance(row.get("final_command_trace"), list) and len(row["final_command_trace"]) > 0
        and isinstance(row.get("navigation_feedback_trace"), list) and len(row["navigation_feedback_trace"]) > 0
        and row.get("restricted_navigation_time_s") is not None
        and row.get("progress_loss_auc_5s") is not None
        and row.get("cross_track_error_auc_5s_m_s") is not None
        and row.get("cross_track_error_max_5s_m") is not None
        and isinstance(row.get("global_plan_trace"), list) and len(row["global_plan_trace"]) > 0
        and crossing_ok
        and metrics.get("revision") == "scan-goal-learned-controller-v1"
        and metrics.get("policy") == policy
        and metrics.get("weight_sha256") == sha256(weights)
        and metrics.get("published", 0) > 10
        and len(metrics.get("raw_records", [])) == metrics.get("published")
    )


def mechanism_invariant(row: dict) -> bool:
    if row["condition"] == "normal":
        return row.get("transfer_events") == 0 and row.get("stale_final_commands") == 0
    if row["mode"] == "independent":
        return row.get("transfer_events") == 0 and row.get("stale_final_commands", 0) > 0
    return bool(
        row.get("transfer_events") == 1
        and row.get("stale_final_commands") == 0
        and row.get("fallback_evidence_messages", 0) > 0
        and row.get("first_fallback_command_ms") is not None
    )


def archive(stem: str, result_dir: Path) -> None:
    files = [path for path in result_dir.glob(f"{stem}*") if path.is_file()]
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


def run_one(cell: tuple, result_dir: Path, weights_dir: Path, domain_base: int, wall_watchdog: float) -> bool:
    rmw, policy, mode, scene, condition, seed = cell
    stem = slug(cell)
    if instrumentation_valid(cell, result_dir, weights_dir):
        return True
    archive(stem, result_dir)
    weights = model_path(policy, weights_dir)
    if not weights.exists():
        raise FileNotFoundError(weights)
    env = os.environ.copy()
    env.update(
        {
            "RMW_IMPLEMENTATION": rmw,
            "ROS_LOCALHOST_ONLY": "1",
            "ROS2CLI_NO_DAEMON": "1",
            "ROS_DOMAIN_ID": str(domain_base + (0 if rmw == RMWS[0] else 1)),
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
        metrics = result_dir / f"{stem}__model.json"
        learned, handle = launch(
            [
                sys.executable, str(LEARNED), "--policy", policy,
                "--weights", str(weights), "--goal-x", "2.0", "--goal-y", "0.5",
                "--metrics", str(metrics),
            ],
            result_dir / f"{stem}__learned.log", env,
        )
        processes.append(learned); handles.append(handle)
        time.sleep(1)
        output = result_dir / f"{stem}.json"
        with (result_dir / f"{stem}__trial.log").open("w", encoding="utf-8") as trial_log:
            subprocess.run(
                [
                    sys.executable, str(TRIAL),
                    "--mode", mode, "--condition", condition,
                    "--scene", scene, "--policy", policy, "--seed", str(seed),
                    "--profile", str(PROFILE), "--model-weights", str(weights),
                    "--output", str(output), "--wall-watchdog", str(wall_watchdog),
                ],
                stdout=trial_log, stderr=subprocess.STDOUT, text=True,
                env=env, timeout=wall_watchdog + 60.0, check=False,
            )
        stop(learned)
        time.sleep(0.5)
        return instrumentation_valid(cell, result_dir, weights_dir)
    except (OSError, subprocess.TimeoutExpired) as error:
        (result_dir / f"{stem}__runner_error.log").write_text(repr(error) + "\n", encoding="utf-8")
        return False
    finally:
        for process in reversed(processes):
            stop(process)
        for handle in handles:
            handle.close()
        time.sleep(1)


def bootstrap_mean(values: list[float], seed: int, draws: int = 10000) -> list[float | None]:
    if not values:
        return [None, None]
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    estimates = np.mean(rng.choice(array, size=(draws, len(array)), replace=True), axis=1)
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def paired_trajectory_deviation(
    normal: dict,
    fault: dict,
    horizon_s: float = 5.0,
    sample_period_s: float = 0.05,
) -> dict | None:
    """Compare faulted motion with its own time-aligned no-fault reference."""

    def relative_xy(row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        anchor = row.get("analysis_anchor_monotonic_s")
        if anchor is None:
            return None
        samples = []
        for sample in row.get("navigation_feedback_trace", []):
            values = (sample.get("monotonic_s"), sample.get("x"), sample.get("y"))
            if all(isinstance(value, (int, float)) and np.isfinite(value) for value in values):
                samples.append((float(values[0]) - float(anchor), float(values[1]), float(values[2])))
        samples.sort()
        if len(samples) < 2:
            return None
        # Keep the last pose if duplicate feedback timestamps occur.
        deduplicated = {}
        for relative_t, x, y in samples:
            deduplicated[relative_t] = (x, y)
        times = np.asarray(sorted(deduplicated), dtype=np.float64)
        if len(times) < 2 or times[-1] < 0.0:
            return None
        xs = np.asarray([deduplicated[t][0] for t in times], dtype=np.float64)
        ys = np.asarray([deduplicated[t][1] for t in times], dtype=np.float64)
        return times, xs, ys

    normal_xy = relative_xy(normal)
    fault_xy = relative_xy(fault)
    if normal_xy is None or fault_xy is None:
        return None
    grid = np.arange(0.0, horizon_s + sample_period_s / 2.0, sample_period_s)
    normal_t, normal_x, normal_y = normal_xy
    fault_t, fault_x, fault_y = fault_xy
    # np.interp holds the terminal pose if a run reaches the goal before 5 s.
    normal_grid_x = np.interp(grid, normal_t, normal_x)
    normal_grid_y = np.interp(grid, normal_t, normal_y)
    fault_grid_x = np.interp(grid, fault_t, fault_x)
    fault_grid_y = np.interp(grid, fault_t, fault_y)
    # Translate each trajectory to its observed trigger pose. This measures the
    # post-trigger motion response rather than AMCL sampling offset at entry.
    dx = (fault_grid_x - fault_grid_x[0]) - (normal_grid_x - normal_grid_x[0])
    dy = (fault_grid_y - fault_grid_y[0]) - (normal_grid_y - normal_grid_y[0])
    distance = np.hypot(dx, dy)
    return {
        "horizon_s": horizon_s,
        "sample_period_s": sample_period_s,
        "alignment": "trigger-relative-displacement",
        "auc_m_s": float(np.trapz(distance, grid)),
        "max_m": float(np.max(distance)),
        "trace": [
            {"relative_time_s": float(t), "deviation_m": float(value)}
            for t, value in zip(grid, distance)
        ],
    }


def analyze(schedule: list[tuple], result_dir: Path, weights_dir: Path, analysis_seed: int) -> dict:
    rows = {}
    invalid, mechanism_failures = [], []
    for cell in schedule:
        stem = slug(cell)
        if not instrumentation_valid(cell, result_dir, weights_dir):
            invalid.append(stem)
            continue
        row = load(result_dir / f"{stem}.json")
        rows[cell] = row
        if not mechanism_invariant(row):
            mechanism_failures.append(stem)
    cells: dict[str, dict] = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for cell, row in rows.items():
        grouped["|".join(map(str, cell[:-1]))].append(row)
    for key, selected in sorted(grouped.items()):
        times = [float(row["restricted_navigation_time_s"]) for row in selected]
        aucs = [float(row["progress_loss_auc_5s"]) for row in selected]
        cross_track_aucs = [float(row["cross_track_error_auc_5s_m_s"]) for row in selected]
        clearances = [
            float(row["min_geometric_clearance_m"])
            for row in selected
            if isinstance(row.get("min_geometric_clearance_m"), (int, float))
        ]
        cells[key] = {
            "runs": len(selected),
            "goal_success": sum(row.get("goal_result") == "SUCCEEDED" for row in selected),
            "restricted_navigation_time_mean_s": statistics_mean(times),
            "progress_loss_auc_5s_mean": statistics_mean(aucs),
            "cross_track_error_auc_5s_mean_m_s": statistics_mean(cross_track_aucs),
            "cross_track_error_max_5s_mean_m": statistics_mean(
                [float(row["cross_track_error_max_5s_m"]) for row in selected]
            ),
            "min_geometric_clearance_mean_m": statistics_mean(clearances),
            "path_length_mean_m": statistics_mean([float(row["path_length_m"]) for row in selected]),
            "geometric_overlap_runs": sum(bool(row.get("geometric_overlap_observed")) for row in selected),
            "transfer_runs": sum(row.get("transfer_events") == 1 for row in selected),
            "stale_final_runs": sum(row.get("stale_final_commands", 0) > 0 for row in selected),
        }
    effects = {}
    for rmw in RMWS:
        for policy in POLICIES:
            for scene in SCENES:
                time_did, auc_did, cross_track_did = [], [], []
                clearance_did, overlap_episode_did = [], []
                trajectory_auc_benefit, trajectory_max_benefit = [], []
                independent_trajectory, pit_trajectory = [], []
                complete_seeds = []
                trajectory_complete_seeds = []
                for seed in sorted({cell[-1] for cell in schedule}):
                    key = lambda mode, condition: (rmw, policy, mode, scene, condition, seed)
                    required = [key(mode, condition) for mode in MODES for condition in CONDITIONS]
                    if not all(item in rows for item in required):
                        continue
                    i_normal, i_fault = rows[key("independent", "normal")], rows[key("independent", "burst_late")]
                    p_normal, p_fault = rows[key("pit", "normal")], rows[key("pit", "burst_late")]
                    time_did.append(
                        (i_fault["restricted_navigation_time_s"] - i_normal["restricted_navigation_time_s"])
                        - (p_fault["restricted_navigation_time_s"] - p_normal["restricted_navigation_time_s"])
                    )
                    auc_did.append(
                        (i_fault["progress_loss_auc_5s"] - i_normal["progress_loss_auc_5s"])
                        - (p_fault["progress_loss_auc_5s"] - p_normal["progress_loss_auc_5s"])
                    )
                    cross_track_did.append(
                        (i_fault["cross_track_error_auc_5s_m_s"] - i_normal["cross_track_error_auc_5s_m_s"])
                        - (p_fault["cross_track_error_auc_5s_m_s"] - p_normal["cross_track_error_auc_5s_m_s"])
                    )
                    clearance_values = [
                        row.get("min_geometric_clearance_m")
                        for row in (i_normal, i_fault, p_normal, p_fault)
                    ]
                    if all(isinstance(value, (int, float)) for value in clearance_values):
                        clearance_did.append(
                            (p_fault["min_geometric_clearance_m"] - p_normal["min_geometric_clearance_m"])
                            - (i_fault["min_geometric_clearance_m"] - i_normal["min_geometric_clearance_m"])
                        )
                    overlap_episode_did.append(
                        (i_fault.get("geometric_overlap_episode_count", 0) - i_normal.get("geometric_overlap_episode_count", 0))
                        - (p_fault.get("geometric_overlap_episode_count", 0) - p_normal.get("geometric_overlap_episode_count", 0))
                    )
                    i_deviation = paired_trajectory_deviation(i_normal, i_fault)
                    p_deviation = paired_trajectory_deviation(p_normal, p_fault)
                    if i_deviation is not None and p_deviation is not None:
                        independent_trajectory.append(i_deviation)
                        pit_trajectory.append(p_deviation)
                        trajectory_auc_benefit.append(i_deviation["auc_m_s"] - p_deviation["auc_m_s"])
                        trajectory_max_benefit.append(i_deviation["max_m"] - p_deviation["max_m"])
                        trajectory_complete_seeds.append(seed)
                    complete_seeds.append(seed)
                effect_key = "|".join((rmw, policy, scene))
                effects[effect_key] = {
                    "paired_seeds": complete_seeds,
                    "restricted_time_did_benefit_mean_s": statistics_mean(time_did),
                    "restricted_time_did_bootstrap95_s": bootstrap_mean(time_did, analysis_seed),
                    "progress_loss_auc_did_benefit_mean": statistics_mean(auc_did),
                    "progress_loss_auc_did_bootstrap95": bootstrap_mean(auc_did, analysis_seed + 1),
                    "cross_track_error_auc_did_benefit_mean_m_s": statistics_mean(cross_track_did),
                    "cross_track_error_auc_did_bootstrap95_m_s": bootstrap_mean(
                        cross_track_did, analysis_seed + 2
                    ),
                    "min_clearance_did_benefit_mean_m": statistics_mean(clearance_did),
                    "min_clearance_did_benefit_bootstrap95_m": bootstrap_mean(
                        clearance_did, analysis_seed + 3
                    ),
                    "overlap_episode_did_benefit_mean": statistics_mean(overlap_episode_did),
                    "overlap_episode_did_benefit_bootstrap95": bootstrap_mean(
                        overlap_episode_did, analysis_seed + 4
                    ),
                    "trajectory_paired_seeds": trajectory_complete_seeds,
                    "trajectory_deviation_auc_benefit_mean_m_s": statistics_mean(trajectory_auc_benefit),
                    "trajectory_deviation_auc_benefit_bootstrap95_m_s": bootstrap_mean(
                        trajectory_auc_benefit, analysis_seed + 5
                    ),
                    "trajectory_deviation_max_benefit_mean_m": statistics_mean(trajectory_max_benefit),
                    "trajectory_deviation_max_benefit_bootstrap95_m": bootstrap_mean(
                        trajectory_max_benefit, analysis_seed + 6
                    ),
                    "raw_restricted_time_did": time_did,
                    "raw_progress_loss_auc_did": auc_did,
                    "raw_cross_track_error_auc_did_m_s": cross_track_did,
                    "raw_min_clearance_did_benefit_m": clearance_did,
                    "raw_overlap_episode_did_benefit": overlap_episode_did,
                    "raw_independent_fault_vs_normal_trajectory": independent_trajectory,
                    "raw_pit_fault_vs_normal_trajectory": pit_trajectory,
                    "raw_trajectory_deviation_auc_benefit_m_s": trajectory_auc_benefit,
                    "raw_trajectory_deviation_max_benefit_m": trajectory_max_benefit,
                }
    return {
        "revision": REVISION,
        "scheduled_runs": len(schedule),
        "valid_runs": len(rows),
        "matrix_complete": len(rows) == len(schedule),
        "mechanism_invariants_passed": not mechanism_failures and len(rows) == len(schedule),
        "invalid_or_missing": invalid,
        "mechanism_failures": mechanism_failures,
        "cells": cells,
        "difference_in_differences": effects,
    }


def statistics_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def yaml_load(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--runs-per-cell", type=int, default=10)
    parser.add_argument("--seed", type=int, default=256)
    parser.add_argument("--domain-base", type=int, default=218)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--wall-watchdog", type=float, default=120.0)
    args = parser.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    compiled_profile = yaml_load(PROFILE)
    required_capabilities = compiled_profile["compiler_evidence"]["required_backend_capabilities"]
    for rmw, capability_path in CAPABILITY_FILES.items():
        capability = load(capability_path)
        if capability is None:
            raise RuntimeError(f"missing capability manifest: {capability_path}")
        unsupported = [
            name for name in required_capabilities
            if not capability["capabilities"].get(name, False)
        ]
        if unsupported:
            raise RuntimeError(f"{rmw} lacks required capabilities: {unsupported}")
    for policy in POLICIES:
        if not model_path(policy, args.weights_dir).exists():
            raise FileNotFoundError(model_path(policy, args.weights_dir))
    schedule = [
        (rmw, policy, mode, scene, condition, seed)
        for rmw in RMWS for policy in POLICIES for mode in MODES
        for scene in SCENES for condition in CONDITIONS
        for seed in range(1, args.runs_per_cell + 1)
    ]
    random.Random(args.seed).shuffle(schedule)
    manifest = {
        "revision": REVISION,
        "created_at": datetime.now().astimezone().isoformat(),
        "randomization_seed": args.seed,
        "runs_per_cell": args.runs_per_cell,
        "scheduled_runs": len(schedule),
        "factors": {"rmw": RMWS, "policy": POLICIES, "mode": MODES, "scene": SCENES, "condition": CONDITIONS},
        "schedule": [list(cell) for cell in schedule],
        "model_sha256": {policy: sha256(model_path(policy, args.weights_dir)) for policy in POLICIES},
        "semantic_profile_id": compiled_profile["compiler_evidence"]["semantic_profile_id"],
        "required_backend_capabilities": required_capabilities,
        "capability_manifest_sha256": {
            rmw: sha256(path) for rmw, path in CAPABILITY_FILES.items()
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (Path(__file__), PARAMS, PROFILE, BRIDGE, LEARNED, TRIAL, CORE)
        },
        "application_source_modified_loc": 0,
        "controller_source_modified_loc": 0,
    }
    (args.result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    failures = []
    for position, cell in enumerate(schedule, 1):
        okay = False
        for _attempt in range(args.retries + 1):
            if run_one(cell, args.result_dir, args.weights_dir, args.domain_base, args.wall_watchdog):
                okay = True
                break
        row = load(args.result_dir / f"{slug(cell)}.json") if okay else None
        status = "done" if row is not None and mechanism_invariant(row) else "mechanism_failure" if okay else "infra_failure"
        with (args.result_dir / "progress.log").open("a", encoding="utf-8") as stream:
            stream.write(f"{position}/{len(schedule)} {slug(cell)}: {status}\n")
        print(f"{position}/{len(schedule)} {slug(cell)}: {status}", flush=True)
        if not okay:
            failures.append(slug(cell))
    analysis = analyze(schedule, args.result_dir, args.weights_dir, args.seed)
    (args.result_dir / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(analysis, indent=2))
    return 0 if analysis["matrix_complete"] and analysis["mechanism_invariants_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
