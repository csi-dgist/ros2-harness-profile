#!/usr/bin/env python3
"""Record perception-to-Twist demonstrations without modifying Nav2 code."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import time

from geometry_msgs.msg import PoseWithCovarianceStamped, TwistStamped
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from scan_goal_features import FEATURE_REVISION, reactive_feature, scan_sectors, yaw_from_quaternion


REVISION = "scan-goal-demonstration-recorder-v1"


class Recorder(Node):
    def __init__(self, run_id: str, expert: str, goal_x: float, goal_y: float, output: Path) -> None:
        super().__init__(f"scan_goal_recorder_{run_id}")
        self.run_id, self.expert = run_id, expert
        self.goal_x, self.goal_y, self.output = goal_x, goal_y, output
        self.sectors: np.ndarray | None = None
        self.pose: tuple[float, float, float] | None = None
        self.velocity: tuple[float, float] | None = None
        self.rows: list[dict] = []
        self.dropped_unready = 0
        self.create_subscription(LaserScan, "/scan", self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/odom", self.on_odom, qos_profile_sensor_data)
        self.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", self.on_pose, 10)
        self.create_subscription(TwistStamped, "/pit/controller_candidate", self.on_command, 50)

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
            float(pose.position.x),
            float(pose.position.y),
            yaw_from_quaternion(
                float(pose.orientation.x), float(pose.orientation.y),
                float(pose.orientation.z), float(pose.orientation.w),
            ),
        )

    def on_command(self, message: TwistStamped) -> None:
        if self.sectors is None or self.pose is None or self.velocity is None:
            self.dropped_unready += 1
            return
        feature = reactive_feature(
            self.sectors,
            pose_x=self.pose[0], pose_y=self.pose[1], pose_yaw=self.pose[2],
            goal_x=self.goal_x, goal_y=self.goal_y,
            linear_x=self.velocity[0], angular_z=self.velocity[1],
        )
        command_stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)
        self.rows.append(
            {
                "revision": REVISION,
                "feature_revision": FEATURE_REVISION,
                "run_id": self.run_id,
                "expert": self.expert,
                "sequence": len(self.rows),
                "received_wall_ns": time.time_ns(),
                "command_stamp_ns": command_stamp_ns,
                "goal": [self.goal_x, self.goal_y],
                "feature": feature.tolist(),
                "target": [float(message.twist.linear.x), float(message.twist.angular.z)],
            }
        )

    def write(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "revision": REVISION,
            "feature_revision": FEATURE_REVISION,
            "run_id": self.run_id,
            "expert": self.expert,
            "goal": [self.goal_x, self.goal_y],
            "dropped_unready": self.dropped_unready,
            "samples": self.rows,
        }
        self.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expert", choices=("DWB", "RPP"), required=True)
    parser.add_argument("--goal-x", type=float, required=True)
    parser.add_argument("--goal-y", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rclpy.init()
    node = Recorder(args.run_id, args.expert, args.goal_x, args.goal_y, args.output)
    signal.signal(signal.SIGTERM, lambda *_: rclpy.shutdown())
    signal.signal(signal.SIGINT, lambda *_: rclpy.shutdown())
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.write()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
