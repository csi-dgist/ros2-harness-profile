#!/usr/bin/env python3
"""Record ordered timestamped expert Twist candidates for behavioral cloning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import time

from geometry_msgs.msg import TwistStamped
import rclpy
from rclpy.node import Node


REVISION = "pit-portability-trace-recorder-v1"


class Recorder(Node):
    def __init__(self, topic: str, run_id: str, output: Path) -> None:
        super().__init__(f"pit_trace_recorder_{run_id}")
        self.run_id, self.output = run_id, output
        self.rows: list[dict] = []
        self.create_subscription(TwistStamped, topic, self.on_message, 50)

    def on_message(self, message: TwistStamped) -> None:
        self.rows.append(
            {
                "revision": REVISION,
                "run_id": self.run_id,
                "sequence": len(self.rows),
                "received_wall_ns": time.time_ns(),
                "source_stamp_ns": message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec,
                "linear_x": float(message.twist.linear.x),
                "angular_z": float(message.twist.angular.z),
            }
        )

    def write(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(self.rows, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/pit/controller_candidate")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rclpy.init()
    node = Recorder(args.topic, args.run_id, args.output)
    signal.signal(signal.SIGTERM, lambda *_: rclpy.shutdown())
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
