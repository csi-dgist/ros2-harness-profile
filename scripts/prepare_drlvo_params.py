#!/usr/bin/env python3
"""Bind the published DRL-VO Nav2 plugin into the existing PIT experiment stack."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "generated" / "nav2_portability_params.yaml"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = yaml.safe_load(args.source.read_text(encoding="utf-8"))
    controller = raw["controller_server"]["ros__parameters"]
    controller["AIController"] = {
        "plugin": "nav2py_drl_vo_controller::DrlVoController",
        "lookahead_dist": 1.0,
        "transform_tolerance": 0.1,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
