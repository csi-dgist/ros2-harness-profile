#!/usr/bin/env python3
"""Frozen reactive or temporal scan-goal learned controller at a ROS 2 boundary."""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import signal
import statistics
import time

from geometry_msgs.msg import PoseWithCovarianceStamped, TwistStamped
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from scan_goal_features import (
    FEATURE_REVISION,
    HISTORY,
    FrozenMLP,
    reactive_feature,
    scan_sectors,
    temporal_feature,
    yaw_from_quaternion,
)


REVISION = "scan-goal-learned-controller-v1"


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]


class LearnedController(Node):
    def __init__(
        self,
        model: FrozenMLP,
        policy: str,
        goal_x: float,
        goal_y: float,
        output_topic: str,
        metrics: Path,
        frequency_hz: float,
    ) -> None:
        super().__init__(f"pit_{policy}_controller")
        if model.policy != policy:
            raise ValueError(f"model policy {model.policy!r} != requested {policy!r}")
        self.model, self.policy = model, policy
        self.goal_x, self.goal_y = goal_x, goal_y
        self.metrics_path = metrics
        self.sectors: np.ndarray | None = None
        self.pose: tuple[float, float, float] | None = None
        self.velocity: tuple[float, float] | None = None
        self.history: deque[np.ndarray] = deque(maxlen=HISTORY - 1)
        self.inference_us: list[float] = []
        self.records: list[dict] = []
        self.timer_ticks = 0
        self.unready_ticks = 0
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE
        self.publisher = self.create_publisher(TwistStamped, output_topic, qos)
        self.create_subscription(LaserScan, "/scan", self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/odom", self.on_odom, qos_profile_sensor_data)
        self.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", self.on_pose, 10)
        self.create_timer(1.0 / frequency_hz, self.on_timer)

    def on_scan(self, message: LaserScan) -> None:
        try:
            self.sectors = scan_sectors(message.ranges, message.range_min, message.range_max)
        except ValueError:
            self.sectors = None

    def on_odom(self, message: Odometry) -> None:
        self.velocity = (
            float(message.twist.twist.linear.x),
            float(message.twist.twist.angular.z),
        )

    def on_pose(self, message: PoseWithCovarianceStamped) -> None:
        pose = message.pose.pose
        self.pose = (
            float(pose.position.x), float(pose.position.y),
            yaw_from_quaternion(
                float(pose.orientation.x), float(pose.orientation.y),
                float(pose.orientation.z), float(pose.orientation.w),
            ),
        )

    def on_timer(self) -> None:
        self.timer_ticks += 1
        if self.sectors is None or self.pose is None or self.velocity is None:
            self.unready_ticks += 1
            return
        reactive = reactive_feature(
            self.sectors,
            pose_x=self.pose[0], pose_y=self.pose[1], pose_yaw=self.pose[2],
            goal_x=self.goal_x, goal_y=self.goal_y,
            linear_x=self.velocity[0], angular_z=self.velocity[1],
        )
        feature = reactive if self.policy == "reactive_scan_goal_bc" else temporal_feature(self.history, reactive)
        began = time.perf_counter_ns()
        output = self.model.predict(feature)
        inference_us = (time.perf_counter_ns() - began) / 1000.0
        self.inference_us.append(inference_us)
        message = TwistStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = f"learned:{self.policy}:{self.model.weight_sha256[:12]}"
        message.twist.linear.x = float(output[0])
        message.twist.angular.z = float(output[1])
        self.publisher.publish(message)
        self.records.append(
            {
                "sequence": len(self.records),
                "wall_time_ns": time.time_ns(),
                "feature": feature.tolist(),
                "output": output.tolist(),
                "inference_us": inference_us,
            }
        )
        self.history.append(reactive)

    def write_metrics(self) -> None:
        payload = {
            "revision": REVISION,
            "feature_revision": FEATURE_REVISION,
            "policy": self.policy,
            "weight_path": str(self.model.path),
            "weight_sha256": self.model.weight_sha256,
            "timer_ticks": self.timer_ticks,
            "unready_ticks": self.unready_ticks,
            "published": len(self.records),
            "inference_us_p50": statistics.median(self.inference_us) if self.inference_us else None,
            "inference_us_p99": percentile(self.inference_us, 0.99),
            "inference_us_samples": self.inference_us,
            "raw_records": self.records,
        }
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("reactive_scan_goal_bc", "temporal_scan_goal_bc"), required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--goal-x", type=float, required=True)
    parser.add_argument("--goal-y", type=float, required=True)
    parser.add_argument("--output-topic", default="/pit/learned_candidate")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--frequency-hz", type=float, default=20.0)
    args = parser.parse_args()
    model = FrozenMLP(args.weights)
    rclpy.init()
    node = LearnedController(
        model, args.policy, args.goal_x, args.goal_y,
        args.output_topic, args.metrics, args.frequency_hz,
    )
    signal.signal(signal.SIGTERM, lambda *_: rclpy.shutdown())
    signal.signal(signal.SIGINT, lambda *_: rclpy.shutdown())
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.write_metrics()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
