"""Shared perception features and frozen MLP inference for learned Nav2 policies.

This module contains no Nav2 controller code.  ROS-facing nodes provide the
latest LaserScan, localization pose, odometry, and deployment goal; this module
converts them to a fixed 40-value contract and evaluates frozen NumPy weights.
"""

from __future__ import annotations

from collections import deque
import hashlib
import math
from pathlib import Path

import numpy as np


FEATURE_REVISION = "scan-goal-40-v1"
SECTORS = 36
HISTORY = 4
RANGE_CLIP_M = 3.5


def wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)


def scan_sectors(ranges, range_min: float, range_max: float) -> np.ndarray:
    values = np.asarray(ranges, dtype=np.float64)
    if values.size < SECTORS:
        raise ValueError(f"LaserScan has {values.size} rays; need at least {SECTORS}")
    upper = min(float(range_max), RANGE_CLIP_M)
    lower = max(0.0, float(range_min))
    values = np.where(np.isfinite(values), values, upper)
    values = np.clip(values, lower, upper)
    return np.asarray([float(np.min(chunk)) for chunk in np.array_split(values, SECTORS)])


def reactive_feature(
    sectors: np.ndarray,
    *,
    pose_x: float,
    pose_y: float,
    pose_yaw: float,
    goal_x: float,
    goal_y: float,
    linear_x: float,
    angular_z: float,
) -> np.ndarray:
    dx, dy = goal_x - pose_x, goal_y - pose_y
    distance = min(5.0, math.hypot(dx, dy))
    bearing = wrap_angle(math.atan2(dy, dx) - pose_yaw)
    feature = np.concatenate(
        [
            np.asarray(sectors, dtype=np.float64),
            np.asarray([distance, bearing, linear_x, angular_z], dtype=np.float64),
        ]
    )
    if feature.shape != (40,) or not np.all(np.isfinite(feature)):
        raise ValueError("invalid scan-goal feature")
    return feature


def temporal_feature(history: deque[np.ndarray], current: np.ndarray) -> np.ndarray:
    frames = list(history)[-(HISTORY - 1) :] + [np.asarray(current, dtype=np.float64)]
    missing = HISTORY - len(frames)
    padded = [np.zeros(40, dtype=np.float64) for _ in range(missing)] + frames
    mask = [0.0] * missing + [1.0] * len(frames)
    vector = np.concatenate([*padded, np.asarray(mask, dtype=np.float64)])
    if vector.shape != (HISTORY * 40 + HISTORY,):
        raise ValueError(f"invalid temporal feature shape: {vector.shape}")
    return vector


class FrozenMLP:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.weight_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        data = np.load(self.path)
        self.policy = str(data["policy"].item())
        self.feature_revision = str(data["feature_revision"].item())
        self.mean = np.asarray(data["mean"], dtype=np.float64)
        self.scale = np.asarray(data["scale"], dtype=np.float64)
        self.w1 = np.asarray(data["w1"], dtype=np.float64)
        self.b1 = np.asarray(data["b1"], dtype=np.float64)
        self.w2 = np.asarray(data["w2"], dtype=np.float64)
        self.b2 = np.asarray(data["b2"], dtype=np.float64)
        if self.feature_revision != FEATURE_REVISION:
            raise ValueError(f"feature revision mismatch: {self.feature_revision}")
        if self.mean.shape != self.scale.shape or self.w1.shape[0] != self.mean.size:
            raise ValueError("model input dimensions are inconsistent")

    def predict(self, feature: np.ndarray) -> np.ndarray:
        vector = np.asarray(feature, dtype=np.float64)
        if vector.shape != self.mean.shape:
            raise ValueError(f"expected feature shape {self.mean.shape}, got {vector.shape}")
        normalized = (vector - self.mean) / self.scale
        output = np.tanh(normalized @ self.w1 + self.b1) @ self.w2 + self.b2
        return np.asarray(
            [np.clip(output[0], 0.0, 0.5), np.clip(output[1], -1.0, 1.0)],
            dtype=np.float64,
        )
