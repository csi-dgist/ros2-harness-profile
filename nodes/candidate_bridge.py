#!/usr/bin/env python3
"""Convert Nav2's unstamped Twist output to the typed middleware candidate surface."""

from __future__ import annotations

import argparse

from geometry_msgs.msg import Twist, TwistStamped
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy


REVISION = "pit-portability-nav2-bridge-v1"


class CandidateBridge(Node):
    def __init__(self, input_topic: str, output_topic: str, node_name: str) -> None:
        super().__init__(node_name)
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE
        self.publisher = self.create_publisher(TwistStamped, output_topic, qos)
        self.subscription = self.create_subscription(Twist, input_topic, self.on_twist, qos)
        self.sequence = 0

    def on_twist(self, source: Twist) -> None:
        message = TwistStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = f"controller:{REVISION}:{self.sequence}"
        message.twist = source
        self.sequence += 1
        self.publisher.publish(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-topic", default="/cmd_vel_nav")
    parser.add_argument("--output-topic", default="/pit/controller_candidate")
    parser.add_argument("--node-name", default="pit_nav2_candidate_bridge")
    args = parser.parse_args()
    rclpy.init()
    node = CandidateBridge(args.input_topic, args.output_topic, args.node_name)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
