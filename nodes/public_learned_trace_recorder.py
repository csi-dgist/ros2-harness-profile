#!/usr/bin/env python3
"""Record reconstructable DRL/DRL-VO ROS 2 observations and commands."""

from __future__ import annotations

import argparse
import gzip
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


REVISION = "pit-public-learned-recorder-v2-gzip-raw-sidecar"


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
    def __init__(
        self,
        output: Path,
        policy: str,
        weights: Path,
        model_commit: str,
        ros2_commit: str,
    ) -> None:
        super().__init__(f"pit_{policy}_raw_trace_recorder")
        self.output = output
        self.policy = policy
        self.weights = weights
        self.model_commit = model_commit
        self.ros2_commit = ros2_commit
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
        raw_path = self.output.with_suffix(".raw.json.gz")
        raw_payload = {
            "revision": REVISION,
            "policy": self.policy,
            "model_commit": self.model_commit,
            "ros2_adapter_commit": self.ros2_commit,
            "weight_sha256": sha256(self.weights),
            "scan_samples": self.scan_samples,
            "scan_history_samples": self.scan_history_samples,
            "path_samples": self.path_samples,
            "command_samples": self.command_samples,
        }
        raw_temporary = raw_path.with_suffix(raw_path.suffix + ".tmp")
        with raw_temporary.open("wb") as raw_stream:
            with gzip.GzipFile(fileobj=raw_stream, mode="wb", mtime=0) as compressed:
                compressed.write(
                    (json.dumps(raw_payload, allow_nan=False) + "\n").encode("utf-8")
                )
        raw_temporary.replace(raw_path)
        payload = {
            "revision": REVISION,
            "policy": self.policy,
            "model_commit": self.model_commit,
            "ros2_adapter_commit": self.ros2_commit,
            "weight_path": str(self.weights),
            "weight_sha256": sha256(self.weights),
            "started_wall_time_ns": self.started_ns,
            "finished_wall_time_ns": time.time_ns(),
            "observation_reconstruction": {
                "network_shape": [19202],
                "pedestrian_map": "12800 zeros in the public ROS 2 planner",
                "scan_map": "reconstructed from scan_history_samples with the public planner revision",
                "subgoal": "reconstructed from received_global_plan using the public lookahead binding",
                "timing_scope": (
                    "The observation-to-candidate interval is an external ROS 2 boundary "
                    "measurement including scheduling and IPC, not pure NN inference."
                ),
            },
            "raw_trace_path": raw_path.name,
            "raw_trace_sha256": sha256(raw_path),
            "raw_trace_encoding": "application/json+gzip; deterministic mtime=0",
            "scan_samples_count": len(self.scan_samples),
            "scan_history_samples_count": len(self.scan_history_samples),
            "path_samples_count": len(self.path_samples),
            "command_samples_count": len(self.command_samples),
            "observation_to_candidate_ms_samples": self.observation_to_candidate_ms_samples,
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output.with_suffix(self.output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(self.output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", choices=("drl", "drl_vo"), required=True)
    parser.add_argument("--model-weights", type=Path, required=True)
    parser.add_argument("--model-commit", required=True)
    parser.add_argument("--ros2-commit", required=True)
    args = parser.parse_args()
    rclpy.init()
    node = Recorder(
        args.output, args.policy, args.model_weights,
        args.model_commit, args.ros2_commit,
    )
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
