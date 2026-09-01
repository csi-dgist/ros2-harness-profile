"""Application- and RMW-independent PIT contract evaluator.

Adapters flatten typed ROS messages into named numeric fields and retain all
ROS publication, selector, and fallback-evidence behavior.  This module must
not import rclpy or a ROS message package.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping


IMPLEMENTATION_REVISION = "pit-portability-core-v3"


@dataclass(frozen=True)
class ContractProfile:
    value_bounds: Mapping[str, tuple[float, float]]
    max_staleness_ms: float
    compute_budget_ms: float
    transport_budget_ms: float
    replace_invalid_with_safe_value: bool = True

    @classmethod
    def from_dict(cls, raw: Mapping) -> "ContractProfile":
        bounds = {
            str(name): (float(limits[0]), float(limits[1]))
            for name, limits in raw["value_bounds"].items()
        }
        for name, (lower, upper) in bounds.items():
            if not (math.isfinite(lower) and math.isfinite(upper) and lower <= upper):
                raise ValueError(f"invalid bounds for {name}: {(lower, upper)}")
        profile = cls(
            value_bounds=bounds,
            max_staleness_ms=float(raw["max_staleness_ms"]),
            compute_budget_ms=float(raw["compute_budget_ms"]),
            transport_budget_ms=float(raw["transport_budget_ms"]),
            replace_invalid_with_safe_value=bool(
                raw.get("replace_invalid_with_safe_value", raw.get("replace_invalid_with_zero", True))
            ),
        )
        if min(
            profile.max_staleness_ms,
            profile.compute_budget_ms,
            profile.transport_budget_ms,
        ) <= 0.0:
            raise ValueError("all timing limits must be positive")
        return profile

    def canonical_sha256(self) -> str:
        encoded = json.dumps(
            {
                "value_bounds": {k: list(self.value_bounds[k]) for k in sorted(self.value_bounds)},
                "max_staleness_ms": self.max_staleness_ms,
                "compute_budget_ms": self.compute_budget_ms,
                "transport_budget_ms": self.transport_budget_ms,
                "replace_invalid_with_safe_value": self.replace_invalid_with_safe_value,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Sample:
    values: Mapping[str, float]
    produced_ns: int
    observed_ns: int
    compute_ms: float
    transport_ms: float

    @property
    def age_ms(self) -> float:
        return (self.observed_ns - self.produced_ns) / 1_000_000.0


@dataclass(frozen=True)
class Decision:
    admit: bool
    replace_with_safe_value: bool
    transfer: bool
    violations: tuple[str, ...]
    revision: str = IMPLEMENTATION_REVISION


class ContractCore:
    """Evaluate Projection and Isolation and expose the composed Transfer edge."""

    def __init__(self, profile: ContractProfile, composed: bool) -> None:
        self.profile = profile
        self.composed = composed
        self.transferred = False

    def evaluate(self, sample: Sample) -> Decision:
        violations: list[str] = []
        missing = set(self.profile.value_bounds) - set(sample.values)
        if missing:
            violations.append("projection_missing_field:" + ",".join(sorted(missing)))
        for name, (lower, upper) in self.profile.value_bounds.items():
            if name not in sample.values:
                continue
            value = float(sample.values[name])
            if not math.isfinite(value):
                violations.append(f"projection_nonfinite:{name}")
            elif value < lower or value > upper:
                violations.append(f"projection_value:{name}")
        if sample.compute_ms > self.profile.compute_budget_ms:
            violations.append("isolation_compute")
        if sample.transport_ms > self.profile.transport_budget_ms:
            violations.append("isolation_transport")
        if sample.age_ms > self.profile.max_staleness_ms:
            violations.append("projection_freshness")

        malformed = any(
            item.startswith("projection_missing") or item.startswith("projection_nonfinite")
            for item in violations
        )
        value_violation = any(item.startswith("projection_value") for item in violations)
        timing_violation = any(item.startswith("isolation_") or item == "projection_freshness" for item in violations)
        transfer = self.composed and timing_violation and not self.transferred
        if transfer:
            self.transferred = True

        if value_violation and not malformed and self.profile.replace_invalid_with_safe_value:
            return Decision(
                admit=True,
                replace_with_safe_value=True,
                transfer=transfer,
                violations=tuple(violations),
            )
        admit = not malformed and not (self.composed and timing_violation) and not value_violation
        return Decision(
            admit=admit,
            replace_with_safe_value=False,
            transfer=transfer,
            violations=tuple(violations),
        )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
