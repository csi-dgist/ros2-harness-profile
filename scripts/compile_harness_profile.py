#!/usr/bin/env python3
"""Compile an application-neutral PIT Profile through a deployment binding."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


REVISION = "pit-profile-compiler-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-profile", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--capability", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    semantic = yaml.safe_load(args.semantic_profile.read_text(encoding="utf-8"))
    binding_raw = yaml.safe_load(args.binding.read_text(encoding="utf-8"))["binding"]
    capability = json.loads(args.capability.read_text(encoding="utf-8"))
    profile = semantic["profile"]
    required = profile["required_backend_capabilities"]
    unsupported = [name for name in required if not capability["capabilities"].get(name, False)]
    if unsupported:
        raise RuntimeError(f"backend {capability['backend']} lacks required capabilities: {unsupported}")
    field_map = binding_raw["field_map"]
    missing = sorted(set(profile["command_fields"]) - set(field_map))
    if missing:
        raise RuntimeError(f"binding lacks semantic fields: {missing}")
    experiment = {
        "application": binding_raw["application"],
        "adapter": binding_raw["adapter"],
        **binding_raw["deployment"],
    }
    bounds = {
        field_map[name]: [float(spec["minimum"]), float(spec["maximum"])]
        for name, spec in profile["command_fields"].items()
    }
    compiled = {
        "experiment": experiment,
        "contract": {
            "value_bounds": bounds,
            **profile["timing"],
            "replace_invalid_with_safe_value": profile["response"]["replace_value_violation_with_safe_value"],
        },
        "fault": semantic["fault_calibration"],
        "compiler_evidence": {
            "revision": REVISION,
            "semantic_profile_id": profile["id"],
            "binding_id": binding_raw["id"],
            "composition": profile["composition"],
            "enforcement_tier": binding_raw["enforcement_tier"],
            "required_backend_capabilities": required,
            "semantic_profile_sha256": sha256(args.semantic_profile),
            "binding_sha256": sha256(args.binding),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(compiled, sort_keys=False), encoding="utf-8")
    evidence = {
        "revision": REVISION,
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "backend_validated": capability["backend"],
        "dds_vendor": capability["dds_vendor"],
        "unsupported_required_capabilities": unsupported,
        "semantic_profile_sha256": sha256(args.semantic_profile),
        "binding_sha256": sha256(args.binding),
        "capability_sha256": sha256(args.capability),
    }
    args.output.with_suffix(args.output.suffix + ".evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
