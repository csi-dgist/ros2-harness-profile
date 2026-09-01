#!/usr/bin/env python3
"""Collect one complete DWB or RPP navigation demonstration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
import time

from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
import yaml

from nav2_trial import Nav2Gate, make_pose


REVISION = "scan-goal-expert-trial-v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert", choices=("DWB", "RPP"), required=True)
    parser.add_argument("--goal-x", type=float, default=2.0)
    parser.add_argument("--goal-y", type=float, default=0.5)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=75.0)
    parser.add_argument("--wall-watchdog", type=float, default=180.0)
    args = parser.parse_args()
    raw = yaml.safe_load(args.profile.read_text(encoding="utf-8"))
    rclpy.init()
    gate = Nav2Gate("independent", "normal", "controller", raw)
    executor = SingleThreadedExecutor()
    executor.add_node(gate)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    navigator = BasicNavigator(node_name=f"scan_goal_expert_{args.expert.lower()}")
    navigator.set_parameters([Parameter("use_sim_time", value=True)])
    goal_result, sim_time_s = "NOT_STARTED", None
    feedback_trace: list[dict] = []
    started = time.monotonic()
    try:
        navigator.setInitialPose(make_pose(navigator, -2.0, -0.5, latest=True))
        navigator.waitUntilNav2Active()
        gate.select_controller(args.expert)
        time.sleep(0.5)
        navigator.goToPose(make_pose(navigator, args.goal_x, args.goal_y))
        started = time.monotonic()
        while not navigator.isTaskComplete():
            feedback = navigator.getFeedback()
            if feedback is not None:
                sim_time_s = feedback.navigation_time.sec + feedback.navigation_time.nanosec / 1e9
                pose = getattr(feedback, "current_pose", None)
                feedback_trace.append(
                    {
                        "wall_time_ns": time.time_ns(),
                        "navigation_sim_time_s": sim_time_s,
                        "distance_remaining_m": float(getattr(feedback, "distance_remaining", float("nan"))),
                        "x": None if pose is None else float(pose.pose.position.x),
                        "y": None if pose is None else float(pose.pose.position.y),
                    }
                )
            if sim_time_s is not None and sim_time_s > args.timeout:
                navigator.cancelTask(); goal_result = "TIMEOUT_SIM"; break
            if time.monotonic() - started > args.wall_watchdog:
                navigator.cancelTask(); goal_result = "TIMEOUT_WALL"; break
            time.sleep(0.05)
        if not goal_result.startswith("TIMEOUT"):
            result = navigator.getResult()
            goal_result = {
                TaskResult.SUCCEEDED: "SUCCEEDED",
                TaskResult.CANCELED: "CANCELED",
                TaskResult.FAILED: "FAILED",
            }.get(result, f"UNKNOWN_{result}")
    finally:
        report = gate.report()
        report.update(
            {
                "revision": REVISION,
                "expert": args.expert,
                "goal": [args.goal_x, args.goal_y],
                "goal_result": goal_result,
                "navigation_sim_time_s": sim_time_s,
                "navigation_wall_time_s": time.monotonic() - started,
                "navigation_feedback_trace": feedback_trace,
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        navigator.destroy_node()
        executor.shutdown()
        gate.destroy_node()
        rclpy.shutdown()
        thread.join(timeout=2.0)
    return 0 if goal_result == "SUCCEEDED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
