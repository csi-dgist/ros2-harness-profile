#!/usr/bin/env python3
"""One Gazebo/Nav2 task using the application-neutral v3 contract core."""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import heapq
import json
import math
from pathlib import Path
import statistics
import threading
import time

from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from nav_msgs.msg import Path as NavPath
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
import yaml

from pit_core import ContractCore, ContractProfile, Sample
from pit_core.core import IMPLEMENTATION_REVISION as CORE_REVISION, file_sha256


ADAPTER_REVISION = "pit-portability-nav2-adapter-v4-profiled-queue-freeze"


def clone_twist(source: Twist) -> Twist:
    target = Twist()
    target.linear.x = source.linear.x
    target.linear.y = source.linear.y
    target.linear.z = source.linear.z
    target.angular.x = source.angular.x
    target.angular.y = source.angular.y
    target.angular.z = source.angular.z
    return target


def signature(message: Twist) -> tuple[float, float]:
    return round(message.linear.x, 6), round(message.angular.z, 6)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def stamp_ns(message: TwistStamped) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


class Nav2Gate(Node):
    def __init__(self, mode: str, scenario: str, source: str, raw: dict) -> None:
        super().__init__(f"pit_v3_gate_{mode}_{scenario}_{source}")
        self.mode, self.scenario, self.source = mode, scenario, source
        self.experiment = raw["experiment"]
        self.fault = raw["fault"]
        self.profile = ContractProfile.from_dict(raw["contract"])
        self.core = ContractCore(self.profile, composed=mode == "pit")
        self.nonzero = 0
        # None preserves the legacy sample-count trigger. Task-impact adapters
        # can set this explicitly from an application-independent spatial event.
        self.external_fault_arm: bool | None = None
        self.fault_started: float | None = None
        self.frozen_fault_command: Twist | None = None
        self.delayed: list[tuple[float, int, int, Twist]] = []
        self.sequence = 0
        self.control_fault_done = False
        self.transfer_started: float | None = None
        self.transfer_settle_until: float | None = None
        self.handoff_latency_ms: float | None = None
        self.first_fallback_command_ms: float | None = None
        self.first_fallback_evidence_ms: float | None = None
        self.fallback_confirmed = False
        self.guard_records: deque[dict] = deque(maxlen=1024)
        self.processing_us: list[float] = []
        self.selector_history: list[dict] = []
        self.raw_events: list[dict] = []
        self.final_command_trace: list[dict] = []
        self.fallback_evidence_trace: list[dict] = []
        self.metrics = {
            "controller_candidates": 0,
            "learned_candidates": 0,
            "selected_primary_candidates": 0,
            "selected_fallback_candidates": 0,
            "guarded_commands": 0,
            "final_commands": 0,
            "stale_guarded_commands": 0,
            "stale_final_commands": 0,
            "projection_replacements": 0,
            "projection_blocks": 0,
            "isolation_violations": 0,
            "transfer_events": 0,
            "transfer_settle_blocks": 0,
            "fallback_evidence_blocks": 0,
            "fallback_evidence_messages": 0,
            "learned_candidates_ignored_after_transfer": 0,
        }

        reliable = QoSProfile(depth=10)
        reliable.reliability = ReliabilityPolicy.RELIABLE
        selector_qos = QoSProfile(depth=1)
        selector_qos.reliability = ReliabilityPolicy.RELIABLE
        selector_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.selector = self.create_publisher(String, self.experiment["selector_topic"], selector_qos)
        self.guarded = self.create_publisher(Twist, self.experiment["guarded_output_topic"], reliable)
        self.create_subscription(
            TwistStamped,
            self.experiment["controller_candidate_topic"],
            self.on_controller_candidate,
            reliable,
        )
        self.create_subscription(
            TwistStamped,
            self.experiment["learned_candidate_topic"],
            self.on_learned_candidate,
            reliable,
        )
        self.create_subscription(Twist, self.experiment["final_output_topic"], self.on_final, reliable)
        self.create_subscription(
            NavPath,
            self.experiment["fallback_evidence_topic"],
            self.on_fallback_evidence,
            reliable,
        )
        self.create_timer(0.001, self.drain_delayed)

    def select_controller(self, name: str) -> None:
        message = String()
        message.data = name
        self.selector.publish(message)
        self.selector_history.append({"controller": name, "wall_time": time.time()})

    def on_controller_candidate(self, message: TwistStamped) -> None:
        self.metrics["controller_candidates"] += 1
        if self.core.transferred:
            self.metrics["selected_fallback_candidates"] += 1
            self.process_candidate(message, is_fallback=True)
        elif self.source in {"controller", "drl_vo"}:
            self.metrics["selected_primary_candidates"] += 1
            self.process_candidate(message, is_fallback=False)

    def on_learned_candidate(self, message: TwistStamped) -> None:
        self.metrics["learned_candidates"] += 1
        if self.core.transferred:
            self.metrics["learned_candidates_ignored_after_transfer"] += 1
            return
        if self.source in {"linear_bc", "mlp_bc"}:
            self.metrics["selected_primary_candidates"] += 1
            self.process_candidate(message, is_fallback=False)

    def process_candidate(self, candidate: TwistStamped, is_fallback: bool) -> None:
        began_ns = time.perf_counter_ns()
        message = clone_twist(candidate.twist)
        if abs(message.linear.x) > 1e-6 or abs(message.angular.z) > 1e-6:
            self.nonzero += 1
        if is_fallback:
            self.evaluate_and_publish(
                message,
                produced_ns=stamp_ns(candidate),
                compute_ms=0.0,
                transport_ms=max(0.0, (time.time_ns() - stamp_ns(candidate)) / 1_000_000.0),
                began_ns=began_ns,
                is_fallback=True,
            )
            return

        armed = (
            self.nonzero >= int(self.fault["arm_after_nonzero_samples"])
            if self.external_fault_arm is None
            else self.external_fault_arm
        )
        if self.scenario == "normal" or not armed:
            self.evaluate_and_publish(
                message, stamp_ns(candidate), 0.0,
                max(0.0, (time.time_ns() - stamp_ns(candidate)) / 1_000_000.0),
                began_ns, False,
            )
            return
        if self.scenario == "control_violation":
            if not self.control_fault_done:
                self.control_fault_done = True
                upper = self.profile.value_bounds["linear_x"][1]
                message.linear.x = upper + 0.5
            self.evaluate_and_publish(message, stamp_ns(candidate), 0.0, 1.0, began_ns, False)
            return
        if self.scenario != "safe_but_late":
            raise RuntimeError(f"unknown scenario {self.scenario}")

        if bool(self.fault.get("freeze_first_command", False)):
            if self.frozen_fault_command is None:
                self.frozen_fault_command = clone_twist(message)
            message = clone_twist(self.frozen_fault_command)

        now = time.monotonic()
        if self.fault_started is None:
            self.fault_started = now
        burst_duration_s = float(self.fault.get("burst_duration_s", 1.0))
        inject = self.mode == "pit" or now - self.fault_started < burst_duration_s
        if not inject:
            self.evaluate_and_publish(message, stamp_ns(candidate), 0.0, 1.0, began_ns, False)
            return
        self.sequence += 1
        compute_ms = float(self.fault["safe_but_late_compute_ms"])
        transport_ms = float(self.fault["transport_ms"])
        release = now + (compute_ms + transport_ms) / 1000.0
        heapq.heappush(self.delayed, (release, self.sequence, time.time_ns(), message))

    def drain_delayed(self) -> None:
        now = time.monotonic()
        while self.delayed and self.delayed[0][0] <= now:
            _, _, produced_ns, message = heapq.heappop(self.delayed)
            self.evaluate_and_publish(
                message,
                produced_ns,
                float(self.fault["safe_but_late_compute_ms"]),
                float(self.fault["transport_ms"]),
                time.perf_counter_ns(),
                False,
            )

    def evaluate_and_publish(
        self,
        message: Twist,
        produced_ns: int,
        compute_ms: float,
        transport_ms: float,
        began_ns: int,
        is_fallback: bool,
    ) -> None:
        observed_ns = time.time_ns()
        input_values = {
            "linear_x": float(message.linear.x),
            "angular_z": float(message.angular.z),
        }
        transferred_before = self.core.transferred
        decision = self.core.evaluate(
            Sample(
                values=input_values,
                produced_ns=produced_ns,
                observed_ns=observed_ns,
                compute_ms=compute_ms,
                transport_ms=transport_ms,
            )
        )
        stale = observed_ns - produced_ns > int(self.profile.max_staleness_ms * 1_000_000)
        if any(item.startswith("isolation_") for item in decision.violations):
            self.metrics["isolation_violations"] += 1
        if decision.transfer:
            self.metrics["transfer_events"] += 1
            self.transfer_started = time.monotonic()
            self.transfer_settle_until = self.transfer_started + float(self.fault["transfer_settle_ms"]) / 1000.0
            self.select_controller(self.experiment["fallback_controller"])
        if decision.replace_with_safe_value:
            message = Twist()
            stale = False
            self.metrics["projection_replacements"] += 1
        if not decision.admit:
            self.metrics["projection_blocks"] += 1
            self.record_raw_event(
                decision, input_values, message, produced_ns, observed_ns,
                compute_ms, transport_ms, is_fallback, stale,
                transferred_before, "projection_block", began_ns,
            )
            return
        if is_fallback and not self.fallback_confirmed:
            self.metrics["fallback_evidence_blocks"] += 1
            self.record_raw_event(
                decision, input_values, message, produced_ns, observed_ns,
                compute_ms, transport_ms, is_fallback, stale,
                transferred_before, "fallback_evidence_block", began_ns,
            )
            return
        if self.transfer_settle_until is not None and time.monotonic() < self.transfer_settle_until:
            self.metrics["transfer_settle_blocks"] += 1
            self.record_raw_event(
                decision, input_values, message, produced_ns, observed_ns,
                compute_ms, transport_ms, is_fallback, stale,
                transferred_before, "transfer_settle_block", began_ns,
            )
            return
        if is_fallback and self.transfer_started is not None and self.first_fallback_command_ms is None:
            self.first_fallback_command_ms = (time.monotonic() - self.transfer_started) * 1000.0
            self.handoff_latency_ms = self.first_fallback_command_ms
        self.guarded.publish(message)
        self.metrics["guarded_commands"] += 1
        if stale:
            self.metrics["stale_guarded_commands"] += 1
        self.guard_records.append({"time": time.monotonic(), "signature": signature(message), "stale": stale})
        self.record_raw_event(
            decision, input_values, message, produced_ns, observed_ns,
            compute_ms, transport_ms, is_fallback, stale,
            transferred_before, "publish", began_ns,
        )

    def record_raw_event(
        self, decision, input_values: dict[str, float], output: Twist,
        produced_ns: int, observed_ns: int, compute_ms: float,
        transport_ms: float, is_fallback: bool, stale: bool,
        transferred_before: bool, action: str, began_ns: int,
    ) -> None:
        processing_us = (time.perf_counter_ns() - began_ns) / 1000.0
        self.processing_us.append(processing_us)
        self.raw_events.append(
            {
                "event_index": len(self.raw_events),
                "produced_ns": produced_ns,
                "observed_ns": observed_ns,
                "age_ms": (observed_ns - produced_ns) / 1_000_000.0,
                "compute_ms": compute_ms,
                "transport_ms": transport_ms,
                "source": "fallback" if is_fallback else self.source,
                "input_values": input_values,
                "output_values": {
                    "linear_x": float(output.linear.x),
                    "angular_z": float(output.angular.z),
                },
                "violations": list(decision.violations),
                "admit": decision.admit,
                "replace_with_safe_value": decision.replace_with_safe_value,
                "transfer": decision.transfer,
                "transferred_before": transferred_before,
                "transferred_after": self.core.transferred,
                "stale": stale,
                "action": action,
                "processing_us": processing_us,
            }
        )

    def on_final(self, message: Twist) -> None:
        self.metrics["final_commands"] += 1
        now = time.monotonic()
        while self.guard_records and now - self.guard_records[0]["time"] > 0.5:
            self.guard_records.popleft()
        target = signature(message)
        matched_stale = None
        for record in list(self.guard_records):
            if record["signature"] == target:
                self.guard_records.remove(record)
                matched_stale = bool(record["stale"])
                if record["stale"]:
                    self.metrics["stale_final_commands"] += 1
                break
        self.final_command_trace.append(
            {
                "wall_time_ns": time.time_ns(),
                "linear_x": float(message.linear.x),
                "angular_z": float(message.angular.z),
                "matched_stale": matched_stale,
            }
        )

    def on_fallback_evidence(self, _message: NavPath) -> None:
        if self.transfer_started is None:
            return
        self.metrics["fallback_evidence_messages"] += 1
        self.fallback_evidence_trace.append(
            {"wall_time_ns": time.time_ns(), "after_transfer_ms":
             (time.monotonic() - self.transfer_started) * 1000.0}
        )
        self.fallback_confirmed = True
        if self.first_fallback_evidence_ms is None:
            self.first_fallback_evidence_ms = (time.monotonic() - self.transfer_started) * 1000.0

    def report(self) -> dict:
        return {
            **self.metrics,
            "core_revision": CORE_REVISION,
            "adapter_revision": ADAPTER_REVISION,
            "mode": self.mode,
            "scenario": self.scenario,
            "source": self.source,
            "fallback": self.experiment["fallback_controller"],
            "handoff_latency_ms": self.handoff_latency_ms,
            "first_fallback_command_ms": self.first_fallback_command_ms,
            "first_fallback_evidence_ms": self.first_fallback_evidence_ms,
            "selector_history": self.selector_history,
            "processing_us_p50": statistics.median(self.processing_us) if self.processing_us else None,
            "processing_us_p99": percentile(self.processing_us, 0.99),
            "processing_us_samples": self.processing_us,
            "raw_events": self.raw_events,
            "final_command_trace": self.final_command_trace,
            "fallback_evidence_trace": self.fallback_evidence_trace,
        }


def make_pose(node: Node, x: float, y: float, latest: bool = False) -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = "map"
    if not latest:
        pose.header.stamp = node.get_clock().now().to_msg()
    pose.pose.position.x, pose.pose.position.y = x, y
    pose.pose.orientation.w = 1.0
    return pose


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["independent", "pit"], required=True)
    parser.add_argument("--scenario", choices=["normal", "control_violation", "safe_but_late"], required=True)
    parser.add_argument("--source", choices=["controller", "linear_bc", "mlp_bc", "drl_vo"], required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--model-weights", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=75.0)
    parser.add_argument("--wall-watchdog", type=float, default=300.0)
    args = parser.parse_args()
    raw = yaml.safe_load(args.profile.read_text(encoding="utf-8"))

    rclpy.init()
    gate = Nav2Gate(args.mode, args.scenario, args.source, raw)
    executor = SingleThreadedExecutor()
    executor.add_node(gate)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    navigator = BasicNavigator(node_name=f"pit_v3_navigator_{args.mode}_{args.source}")
    navigator.set_parameters([Parameter("use_sim_time", value=True)])
    goal_result, sim_time_s, feedback_samples = "NOT_STARTED", None, 0
    feedback_trace: list[dict] = []
    started = time.monotonic()
    try:
        navigator.setInitialPose(make_pose(navigator, -2.0, -0.5, latest=True))
        navigator.waitUntilNav2Active()
        gate.select_controller(raw["experiment"]["primary_controller"])
        time.sleep(0.5)
        navigator.goToPose(make_pose(navigator, 2.0, 0.5))
        started = time.monotonic()
        while not navigator.isTaskComplete():
            feedback = navigator.getFeedback()
            if feedback is not None:
                sim_time_s = feedback.navigation_time.sec + feedback.navigation_time.nanosec / 1e9
                feedback_samples += 1
                current_pose = getattr(feedback, "current_pose", None)
                feedback_trace.append(
                    {
                        "wall_time_ns": time.time_ns(),
                        "navigation_sim_time_s": sim_time_s,
                        "distance_remaining_m": float(getattr(feedback, "distance_remaining", float("nan"))),
                        "number_of_recoveries": int(getattr(feedback, "number_of_recoveries", 0)),
                        "x": None if current_pose is None else float(current_pose.pose.position.x),
                        "y": None if current_pose is None else float(current_pose.pose.position.y),
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
                "goal_result": goal_result,
                "navigation_sim_time_s": sim_time_s,
                "navigation_wall_time_s": time.monotonic() - started,
                "navigation_feedback_samples": feedback_samples,
                "navigation_feedback_trace": feedback_trace,
                "rmw_implementation": __import__("os").environ.get("RMW_IMPLEMENTATION"),
                "profile_path": str(args.profile),
                "profile_sha256": file_sha256(args.profile),
                "core_sha256": file_sha256(Path(__file__).parents[1] / "pit_core" / "core.py"),
                "adapter_sha256": file_sha256(__file__),
                "model_weight_sha256": file_sha256(args.model_weights) if args.model_weights else None,
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
