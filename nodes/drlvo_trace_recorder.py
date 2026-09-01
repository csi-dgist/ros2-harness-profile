#!/usr/bin/env python3
"""Record DRL-VO observations and candidate commands without changing its policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import time

from geometry_msgs.msg import Twist
from nav_msgs.msg import Path as NavPath
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray


REVISION = "pit-portability-drlvo-recorder-v1-raw-observation-trace"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_float(value: float) -> float | str:
    value = float(value)
    if math.isnan(value):
        return "NaN"
    if value == math.inf:
        return "Infinity"
    if value == -math.inf:
        return "-Infinity"
    return value


def policy_float(value: float, range_limit: float = 30.0) -> float:
    value = float(value)
    return value if math.isfinite(value) else range_limit


class Recorder(Node):
    def __init__(self, output: Path, weights: Path, upstream_commit: str) -> None:
        super().__init__("pit_drlvo_raw_trace_recorder")
        self.output = output
        self.weights = weights
        self.upstream_commit = upstream_commit
        self.started_ns = time.time_ns()
        self.last_history_ns: int | None = None
        self.scan_samples: list[dict] = []
        self.scan_history_samples: list[dict] = []
        self.path_samples: list[dict] = []
        self.command_samples: list[dict] = []
        self.observation_to_candidate_ms_samples: list[float] = []
        self.create_subscription(LaserScan, "/scan", self.on_scan, 20)
        self.create_subscription(
            Float32MultiArray, "/scan_history_list", self.on_history, 20
        )
        self.create_subscription(NavPath, "/received_global_plan", self.on_path, 20)
        self.create_subscription(Twist, "/cmd_vel_nav", self.on_command, 20)

    @staticmethod
    def stamp(message) -> dict:
        stamp = getattr(getattr(message, "header", None), "stamp", None)
        return {
            "wall_time_ns": time.time_ns(),
            "ros_sec": int(stamp.sec) if stamp is not None else None,
            "ros_nanosec": int(stamp.nanosec) if stamp is not None else None,
        }

    def on_scan(self, message: LaserScan) -> None:
        self.scan_samples.append(
            {
                **self.stamp(message),
                "angle_min": float(message.angle_min),
                "angle_max": float(message.angle_max),
                "angle_increment": float(message.angle_increment),
                "range_min": float(message.range_min),
                "range_max": float(message.range_max),
                "ranges_ros": [json_float(value) for value in message.ranges],
                "ranges_policy_sanitized": [policy_float(value) for value in message.ranges],
            }
        )

    def on_history(self, message: Float32MultiArray) -> None:
        now_ns = time.time_ns()
        self.last_history_ns = now_ns
        self.scan_history_samples.append(
            {
                "wall_time_ns": now_ns,
                "data_ros": [json_float(value) for value in message.data],
                "data_policy_sanitized": [policy_float(value) for value in message.data],
            }
        )

    def on_path(self, message: NavPath) -> None:
        self.path_samples.append(
            {
                **self.stamp(message),
                "frame_id": message.header.frame_id,
                "poses": [
                    {
                        "x": float(pose.pose.position.x),
                        "y": float(pose.pose.position.y),
                        "qx": float(pose.pose.orientation.x),
                        "qy": float(pose.pose.orientation.y),
                        "qz": float(pose.pose.orientation.z),
                        "qw": float(pose.pose.orientation.w),
                    }
                    for pose in message.poses
                ],
            }
        )

    def on_command(self, message: Twist) -> None:
        now_ns = time.time_ns()
        causal_ms = (
            (now_ns - self.last_history_ns) / 1_000_000.0
            if self.last_history_ns is not None
            else None
        )
        if causal_ms is not None and causal_ms >= 0.0:
            self.observation_to_candidate_ms_samples.append(causal_ms)
        self.command_samples.append(
            {
                "wall_time_ns": now_ns,
                "linear_x": float(message.linear.x),
                "linear_y": float(message.linear.y),
                "angular_z": float(message.angular.z),
                "observation_to_candidate_ms": causal_ms,
            }
        )

    def write(self) -> None:
        payload = {
            "revision": REVISION,
            "trace_semantics": {
                "observation_to_candidate_ms": (
                    "wall-clock gap from the latest externally observed scan_history_list "
                    "message to cmd_vel_nav; this includes scheduling and IPC and is not "
                    "reported as pure neural-network inference latency"
                )
            },
            "started_wall_time_ns": self.started_ns,
            "finished_wall_time_ns": time.time_ns(),
            "upstream_commit": self.upstream_commit,
            "weight_path": str(self.weights),
            "weight_sha256": sha256(self.weights),
            "scan_samples": self.scan_samples,
            "scan_history_samples": self.scan_history_samples,
            "path_samples": self.path_samples,
            "command_samples": self.command_samples,
            "observation_to_candidate_ms_samples": self.observation_to_candidate_ms_samples,
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output.with_suffix(self.output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(self.output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-weights", type=Path, required=True)
    parser.add_argument("--upstream-commit", required=True)
    args = parser.parse_args()
    rclpy.init()
    node = Recorder(args.output, args.model_weights, args.upstream_commit)
    stop_requested = False

    def stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        while rclpy.ok() and not stop_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.write()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
