#!/usr/bin/env python3
"""Frozen learned command producer for the Nav2 portability experiment."""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import signal
import statistics
import time

from geometry_msgs.msg import TwistStamped
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy


REVISION = "pit-portability-learned-producer-v2-raw-trace"


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]


class FrozenModel:
    def __init__(self, kind: str, path: Path) -> None:
        self.kind = kind
        self.path = path
        self.weight_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        data = np.load(path)
        self.mean = data["mean"]
        self.scale = data["scale"]
        if kind == "linear_bc":
            self.w, self.b = data["w"], data["b"]
        elif kind == "mlp_bc":
            self.w1, self.b1 = data["w1"], data["b1"]
            self.w2, self.b2 = data["w2"], data["b2"]
        else:
            raise ValueError(f"unsupported model kind: {kind}")

    def predict(self, vector: np.ndarray) -> np.ndarray:
        normalized = (vector - self.mean) / self.scale
        if self.kind == "linear_bc":
            return normalized @ self.w + self.b
        hidden = np.tanh(normalized @ self.w1 + self.b1)
        return hidden @ self.w2 + self.b2


class LearnedTwistProducer(Node):
    def __init__(
        self,
        model: FrozenModel,
        input_topic: str,
        output_topic: str,
        history: int,
        metrics_path: Path,
    ) -> None:
        super().__init__(f"pit_{model.kind}_producer")
        self.model = model
        self.history_length = history
        self.history: deque[tuple[float, float]] = deque(maxlen=history)
        self.metrics_path = metrics_path
        self.inference_us: list[float] = []
        self.inference_records: list[dict] = []
        self.received = 0
        self.published = 0
        self.warmup_passthrough = 0
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE
        self.publisher = self.create_publisher(TwistStamped, output_topic, qos)
        self.subscription = self.create_subscription(TwistStamped, input_topic, self.on_candidate, qos)

    def on_candidate(self, source: TwistStamped) -> None:
        self.received += 1
        current = (float(source.twist.linear.x), float(source.twist.angular.z))
        history_window = [list(pair) for pair in self.history]
        output = TwistStamped()
        output.header.stamp = self.get_clock().now().to_msg()
        output.header.frame_id = f"learned:{self.model.kind}:{self.model.weight_sha256[:12]}"
        if len(self.history) < self.history_length:
            output.twist = source.twist
            self.warmup_passthrough += 1
            inference_us = None
            kind = "warmup_passthrough"
        else:
            vector = np.asarray([value for pair in self.history for value in pair])
            began = time.perf_counter_ns()
            prediction = self.model.predict(vector)
            inference_us = (time.perf_counter_ns() - began) / 1000.0
            self.inference_us.append(inference_us)
            output.twist.linear.x = float(prediction[0])
            output.twist.angular.z = float(prediction[1])
            kind = "learned"
        self.inference_records.append(
            {
                "sequence": self.received - 1,
                "wall_time_ns": time.time_ns(),
                "source_stamp_ns": int(source.header.stamp.sec) * 1_000_000_000
                + int(source.header.stamp.nanosec),
                "kind": kind,
                "history_window": history_window,
                "current_input": list(current),
                "output": [float(output.twist.linear.x), float(output.twist.angular.z)],
                "inference_us": inference_us,
            }
        )
        self.history.append(current)
        self.publisher.publish(output)
        self.published += 1

    def write_metrics(self) -> None:
        report = {
            "revision": REVISION,
            "model_kind": self.model.kind,
            "weight_path": str(self.model.path),
            "weight_sha256": self.model.weight_sha256,
            "history": self.history_length,
            "received": self.received,
            "published": self.published,
            "warmup_passthrough": self.warmup_passthrough,
            "learned_outputs": len(self.inference_us),
            "inference_us_p50": statistics.median(self.inference_us) if self.inference_us else None,
            "inference_us_p99": percentile(self.inference_us, 0.99),
            "inference_us_samples": self.inference_us,
            "inference_records": self.inference_records,
        }
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-kind", choices=["linear_bc", "mlp_bc"], required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--input-topic", default="/pit/controller_candidate")
    parser.add_argument("--output-topic", default="/pit/learned_candidate")
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--metrics", type=Path, required=True)
    args = parser.parse_args()
    model = FrozenModel(args.model_kind, args.weights)
    rclpy.init()
    node = LearnedTwistProducer(
        model, args.input_topic, args.output_topic, args.history, args.metrics
    )
    signal.signal(signal.SIGTERM, lambda *_: rclpy.shutdown())
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
