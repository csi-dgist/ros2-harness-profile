#!/usr/bin/env python3
"""Paired DRL/DRL-VO task-impact matrix over Fast DDS and Cyclone DDS."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import run_learned_task_impact as base


ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "generated" / "nav2_public_learned_params.yaml"
PROFILE = ROOT / "generated" / "nav2_public_learned_compiled_task.yaml"
BRIDGE = ROOT / "nodes" / "candidate_bridge.py"
RECORDER = ROOT / "nodes" / "public_learned_trace_recorder.py"
TRIAL = ROOT / "adapters" / "nav2_task_impact_trial.py"
GATE_DEPENDENCY = ROOT / "adapters" / "nav2_trial.py"
CORE = ROOT / "pit_core" / "core.py"
REVISION = "public-learned-task-impact-matrix-v5-post-filter-boundary"
RECORDER_REVISION = "pit-public-learned-recorder-v2-gzip-raw-sidecar"
RMWS = base.RMWS
POLICIES = ("drl", "drl_vo")
MODES = base.MODES
SCENES = base.SCENES
CONDITIONS = base.CONDITIONS
CAPABILITY_FILES = base.CAPABILITY_FILES
DEFAULT_RUNTIME_MODEL = Path(
    "/home/shlee/drl_vo_jazzy_ws/build/nav2py_drl_vo_controller/venv/share/model/drl_vo.zip"
)


def model_path(policy: str, weights_dir: Path) -> Path:
    return weights_dir / f"{policy}.zip"


def parse_subset(raw: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    selected = tuple(item.strip() for item in raw.split(",") if item.strip())
    unknown = sorted(set(selected) - set(allowed))
    if unknown or not selected:
        raise ValueError(f"invalid subset {selected}; allowed={allowed}; unknown={unknown}")
    return selected


def bind_runtime_model(source: Path, runtime_model: Path) -> str:
    runtime_model.parent.mkdir(parents=True, exist_ok=True)
    temporary = runtime_model.with_suffix(runtime_model.suffix + ".binding.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, runtime_model)
    source_hash = base.sha256(source)
    if base.sha256(runtime_model) != source_hash:
        raise RuntimeError("runtime checkpoint binding hash mismatch")
    return source_hash


def instrumentation_valid(cell: tuple, result_dir: Path, weights_dir: Path) -> bool:
    rmw, policy, mode, scene, condition, seed = cell
    stem = base.slug(cell)
    row = base.load(result_dir / f"{stem}.json")
    metrics = base.load(result_dir / f"{stem}__model.json")
    raw_trace = result_dir / f"{stem}__model.raw.json.gz"
    weights = model_path(policy, weights_dir)
    if row is None or metrics is None or not weights.exists():
        return False
    crossing_ok = scene == "static" or (
        row.get("obstacle_spawned") and row.get("obstacle_activated")
        and len(row.get("clearance_trace", [])) > 0
    )
    return bool(
        row.get("revision") == "learned-task-impact-trial-v11-post-filter-boundary"
        and row.get("rmw_implementation") == rmw
        and row.get("policy") == policy
        and row.get("mode") == mode
        and row.get("scene") == scene
        and row.get("condition") == condition
        and row.get("seed") == seed
        and row.get("goal_result") not in {None, "NOT_STARTED", "TIMEOUT_WALL"}
        and row.get("model_weight_sha256") == base.sha256(weights)
        and row.get("profile_sha256") == base.sha256(PROFILE)
        and row.get("adapter_sha256") == base.sha256(TRIAL)
        and row.get("gate_dependency_sha256") == base.sha256(GATE_DEPENDENCY)
        and row.get("application_source_modified_loc") == 0
        and row.get("controller_source_modified_loc") == 0
        and row.get("active_primary_source") == policy
        and row.get("nav2_shadow_fallback_controller") == "DWB"
        and row.get("prohibited_mppi_command_admissions") == 0
        and row.get("middleware_enforcement_boundary") == "post_collision_monitor_pre_actuator"
        and row.get("collision_monitor_preserved") is True
        and row.get("selector_history", [{}])[0].get("controller") == "AIController"
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
        and metrics.get("revision") == RECORDER_REVISION
        and metrics.get("policy") == policy
        and metrics.get("weight_sha256") == base.sha256(weights)
        and metrics.get("model_commit")
        and metrics.get("ros2_adapter_commit")
        and raw_trace.exists()
        and metrics.get("raw_trace_path") == raw_trace.name
        and metrics.get("raw_trace_sha256") == base.sha256(raw_trace)
        and metrics.get("scan_samples_count", 0) > 0
        and metrics.get("scan_history_samples_count", 0) > 0
        and metrics.get("path_samples_count", 0) > 0
        and metrics.get("command_samples_count", 0) >= 8
        and isinstance(metrics.get("observation_to_candidate_ms_samples"), list)
    )


def run_one(
    cell: tuple,
    result_dir: Path,
    weights_dir: Path,
    runtime_model: Path,
    model_commit: str,
    ros2_commit: str,
    domain_base: int,
    wall_watchdog: float,
    goal_x: float,
    goal_y: float,
    obstacle_x: float,
    obstacle_y: float,
    post_trigger_horizon_s: float,
) -> bool:
    rmw, policy, mode, scene, condition, seed = cell
    stem = base.slug(cell)
    if instrumentation_valid(cell, result_dir, weights_dir):
        return True
    base.archive(stem, result_dir)
    weights = model_path(policy, weights_dir)
    if not weights.exists():
        raise FileNotFoundError(weights)
    bind_runtime_model(weights, runtime_model)
    env = os.environ.copy()
    env.update(
        {
            "RMW_IMPLEMENTATION": rmw,
            "ROS_LOCALHOST_ONLY": "1",
            "ROS2CLI_NO_DAEMON": "1",
            "ROS_DOMAIN_ID": str(domain_base + (0 if rmw == RMWS[0] else 1)),
            "PYTHONPATH": (
                f"{ROOT}:{ROOT / 'adapters'}:{ROOT / 'nodes'}:"
                "/opt/ros/jazzy/lib/python3.12/site-packages"
            ),
        }
    )
    processes, handles = [], []
    recorder = None
    try:
        stack, handle = base.launch(
            [
                "ros2", "launch", "nav2_bringup", "tb3_simulation_launch.py",
                "headless:=True", "use_rviz:=False", "use_composition:=False",
                "autostart:=True", f"params_file:={PARAMS}",
            ],
            result_dir / f"{stem}__stack.log", env,
        )
        processes.append(stack); handles.append(handle)
        time.sleep(8)
        fallback_bridge, handle = base.launch(
            [sys.executable, str(BRIDGE), "--output-topic", "/pit/controller_candidate"],
            result_dir / f"{stem}__fallback_bridge.log", env,
        )
        processes.append(fallback_bridge); handles.append(handle)
        primary_bridge, handle = base.launch(
            [
                sys.executable, str(BRIDGE), "--output-topic", "/pit/learned_candidate",
                "--node-name", "pit_public_learned_candidate_bridge",
            ],
            result_dir / f"{stem}__primary_bridge.log", env,
        )
        processes.append(primary_bridge); handles.append(handle)
        metrics = result_dir / f"{stem}__model.json"
        recorder, handle = base.launch(
            [
                sys.executable, str(RECORDER), "--output", str(metrics),
                "--policy", policy, "--model-weights", str(weights),
                "--model-commit", model_commit, "--ros2-commit", ros2_commit,
            ],
            result_dir / f"{stem}__recorder.log", env,
        )
        processes.append(recorder); handles.append(handle)
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
                    "--goal-x", str(goal_x), "--goal-y", str(goal_y),
                    "--obstacle-x", str(obstacle_x), "--obstacle-y", str(obstacle_y),
                    "--post-trigger-horizon-s", str(post_trigger_horizon_s),
                ],
                stdout=trial_log, stderr=subprocess.STDOUT, text=True,
                env=env, timeout=wall_watchdog + 60.0, check=False,
            )
        base.stop(recorder)
        recorder = None
        time.sleep(0.5)
        return instrumentation_valid(cell, result_dir, weights_dir)
    except (OSError, subprocess.TimeoutExpired) as error:
        (result_dir / f"{stem}__runner_error.log").write_text(repr(error) + "\n", encoding="utf-8")
        return False
    finally:
        for process in reversed(processes):
            base.stop(process)
        for handle in handles:
            handle.close()
        time.sleep(1)


def main() -> int:
    global PROFILE
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--runtime-model", type=Path, default=DEFAULT_RUNTIME_MODEL)
    parser.add_argument("--model-commit", required=True)
    parser.add_argument("--ros2-commit", required=True)
    parser.add_argument("--runs-per-cell", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed", type=int, default=256)
    parser.add_argument("--domain-base", type=int, default=218)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--wall-watchdog", type=float, default=150.0)
    parser.add_argument("--goal-x", type=float, default=2.0)
    parser.add_argument("--goal-y", type=float, default=0.5)
    parser.add_argument("--obstacle-x", type=float, default=-1.45)
    parser.add_argument("--obstacle-y", type=float, default=0.15)
    parser.add_argument("--post-trigger-horizon-s", type=float, default=0.0)
    parser.add_argument("--rmws", default=",".join(RMWS))
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--scenes", default=",".join(SCENES))
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    args = parser.parse_args()
    PROFILE = args.profile.resolve()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    if not PARAMS.exists() or not PROFILE.exists():
        raise FileNotFoundError(f"missing public-policy params/Profile: {PARAMS}, {PROFILE}")
    compiled_profile = base.yaml_load(PROFILE)
    required_capabilities = compiled_profile["compiler_evidence"]["required_backend_capabilities"]
    for rmw, capability_path in CAPABILITY_FILES.items():
        capability = base.load(capability_path)
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
    rmws = parse_subset(args.rmws, RMWS)
    policies = parse_subset(args.policies, POLICIES)
    modes = parse_subset(args.modes, MODES)
    scenes = parse_subset(args.scenes, SCENES)
    conditions = parse_subset(args.conditions, CONDITIONS)
    if not 0 <= args.domain_base <= 231:
        raise ValueError(
            f"domain-base must be in [0, 231] so the second DDS backend remains <=232; got {args.domain_base}"
        )
    schedule = [
        (rmw, policy, mode, scene, condition, seed)
        for rmw in rmws for policy in policies for mode in modes
        for scene in scenes for condition in conditions
        for seed in range(args.seed_start, args.seed_start + args.runs_per_cell)
    ]
    random.Random(args.seed).shuffle(schedule)
    manifest = {
        "revision": REVISION,
        "created_at": datetime.now().astimezone().isoformat(),
        "randomization_seed": args.seed,
        "trial_seed_start": args.seed_start,
        "runs_per_cell": args.runs_per_cell,
        "scheduled_runs": len(schedule),
        "factors": {
            "rmw": rmws, "policy": policies, "mode": modes,
            "scene": scenes, "condition": conditions,
        },
        "start_xy": [-2.0, -0.5],
        "goal_xy": [args.goal_x, args.goal_y],
        "obstacle_xy": [args.obstacle_x, args.obstacle_y],
        "obstacle_placement": {
            "rule": (
                "queued-command-predicted-arc"
                if float(compiled_profile["fault"].get("predictive_obstacle_path_m", 0.0)) > 0.0
                else "fixed-world-coordinate"
            ),
            "predictive_obstacle_path_m": compiled_profile["fault"].get("predictive_obstacle_path_m"),
            "queued_command_min_linear_x": compiled_profile["fault"].get("queued_command_min_linear_x"),
            "queued_command_max_abs_angular_z": compiled_profile["fault"].get("queued_command_max_abs_angular_z"),
            "obstacle_active_duration_s": compiled_profile["fault"].get("obstacle_active_duration_s"),
        },
        "middleware_enforcement_boundary": "post_collision_monitor_pre_actuator",
        "independent_nav2_safety_stage": "Collision Monitor preserved upstream",
        "post_trigger_horizon_s": args.post_trigger_horizon_s,
        "schedule": [list(cell) for cell in schedule],
        "model_upstream_commit": args.model_commit,
        "ros2_adapter_commit": args.ros2_commit,
        "model_sha256": {
            policy: base.sha256(model_path(policy, args.weights_dir))
            for policy in POLICIES
        },
        "observation_shape": [19202],
        "action_shape": [2],
        "runtime_model_binding": str(args.runtime_model),
        "semantic_profile_id": compiled_profile["compiler_evidence"]["semantic_profile_id"],
        "profile_sha256": base.sha256(PROFILE),
        "required_backend_capabilities": required_capabilities,
        "capability_manifest_sha256": {
            rmw: base.sha256(path) for rmw, path in CAPABILITY_FILES.items()
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): base.sha256(path)
            for path in (
                Path(__file__), PARAMS, PROFILE, BRIDGE, RECORDER,
                TRIAL, GATE_DEPENDENCY, CORE,
            )
        },
        "application_source_modified_loc": 0,
        "policy_logic_modified_loc": 0,
        "compatibility_scope": (
            "Jazzy build metadata, std_msgs dependency declaration, Nav2 header shim, "
            "and install-tree optimization only; recorded separately from policy logic."
        ),
    }
    (args.result_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    base.PROFILE = PROFILE
    base.POLICIES = policies
    base.REVISION = REVISION
    base.model_path = model_path
    base.instrumentation_valid = instrumentation_valid
    failures = []
    for position, cell in enumerate(schedule, 1):
        okay = False
        for _attempt in range(args.retries + 1):
            if run_one(
                cell, args.result_dir, args.weights_dir, args.runtime_model,
                args.model_commit, args.ros2_commit, args.domain_base,
                args.wall_watchdog, args.goal_x, args.goal_y,
                args.obstacle_x, args.obstacle_y,
                args.post_trigger_horizon_s,
            ):
                okay = True
                break
        row = base.load(args.result_dir / f"{base.slug(cell)}.json") if okay else None
        status = (
            "done" if row is not None and base.mechanism_invariant(row)
            else "mechanism_failure" if okay else "infra_failure"
        )
        with (args.result_dir / "progress.log").open("a", encoding="utf-8") as stream:
            stream.write(f"{position}/{len(schedule)} {base.slug(cell)}: {status}\n")
        print(f"{position}/{len(schedule)} {base.slug(cell)}: {status}", flush=True)
        if not okay:
            failures.append(base.slug(cell))
    analysis = base.analyze(schedule, args.result_dir, args.weights_dir, args.seed)
    (args.result_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(analysis, indent=2))
    return 0 if analysis["matrix_complete"] and analysis["mechanism_invariants_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
