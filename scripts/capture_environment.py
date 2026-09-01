#!/usr/bin/env python3
"""Capture the execution environment used by the portability matrices."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy


PACKAGES = (
    "ros-jazzy-nav2-bringup",
    "ros-jazzy-navigation2",
    "ros-jazzy-rmw-fastrtps-cpp",
    "ros-jazzy-rmw-cyclonedds-cpp",
    "ros-jazzy-ros2-control",
    "ros-jazzy-ros2-controllers",
)


def command(argv: list[str]) -> str | None:
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def package_version(name: str) -> str | None:
    return command(["dpkg-query", "-W", "-f=${Version}", name])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cpu_raw = command(["lscpu", "-J"])
    try:
        cpu = json.loads(cpu_raw) if cpu_raw else None
    except json.JSONDecodeError:
        cpu = {"raw": cpu_raw}
    report = {
        "schema": "pit-portability-environment-v1",
        "platform": platform.platform(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "python": sys.version,
        "numpy": numpy.__version__,
        "ros_distro": os.environ.get("ROS_DISTRO"),
        "cpu": cpu,
        "packages": {name: package_version(name) for name in PACKAGES},
    }
    missing = [name for name, version in report["packages"].items() if version is None]
    report["complete"] = report["ros_distro"] == "jazzy" and not missing
    report["missing_packages"] = missing
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
