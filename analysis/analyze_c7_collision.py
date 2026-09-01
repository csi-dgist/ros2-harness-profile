#!/usr/bin/env python3
"""Fail-closed analysis for the locked C7 learned-policy collision matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean, median


RMWS = ("rmw_fastrtps_cpp", "rmw_cyclonedds_cpp")
POLICIES = ("drl", "drl_vo")
MODES = ("independent", "pit")
CONDITIONS = ("normal", "burst_late")
SEEDS = tuple(range(30, 40))
BOOTSTRAP_SEED = 20260814
BOOTSTRAP_REPS = 20_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [float("nan"), float("nan")]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def exact_mcnemar(independent: list[int], pit: list[int]) -> dict:
    discordant_independent = sum(i == 1 and p == 0 for i, p in zip(independent, pit))
    discordant_pit = sum(i == 0 and p == 1 for i, p in zip(independent, pit))
    discordant = discordant_independent + discordant_pit
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, k) for k in range(0, min(discordant_independent, discordant_pit) + 1)
        ) / (2 ** discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "independent_collision_pit_safe_pairs": discordant_independent,
        "independent_safe_pit_collision_pairs": discordant_pit,
        "exact_two_sided_p": p_value,
    }


def load_and_audit(result_dir: Path) -> tuple[list[dict], dict]:
    manifest_path = result_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result_files = sorted(
        path for path in result_dir.glob("rmw_*.json")
        if not path.name.endswith("__model.json")
    )
    expected = {
        (rmw, policy, mode, condition, seed)
        for rmw in RMWS for policy in POLICIES for mode in MODES
        for condition in CONDITIONS for seed in SEEDS
    }
    rows = []
    observed = set()
    errors: list[str] = []
    for path in result_files:
        row = json.loads(path.read_text(encoding="utf-8"))
        key = (
            row.get("rmw_implementation"), row.get("policy"), row.get("mode"),
            row.get("condition"), row.get("seed"),
        )
        if key in observed:
            errors.append(f"duplicate result cell: {key}")
        observed.add(key)
        model_path = result_dir / f"{path.stem}__model.json"
        if not model_path.exists():
            errors.append(f"missing model sidecar: {model_path.name}")
            continue
        model = json.loads(model_path.read_text(encoding="utf-8"))
        raw_path = result_dir / str(model.get("raw_trace_path", ""))
        if not raw_path.exists():
            errors.append(f"missing raw trace: {raw_path.name}")
        elif sha256(raw_path) != model.get("raw_trace_sha256"):
            errors.append(f"raw trace hash mismatch: {raw_path.name}")
        if row.get("profile_sha256") != manifest.get("profile_sha256"):
            errors.append(f"Profile hash mismatch: {path.name}")
        source_hashes = manifest.get("source_sha256", {})
        expected_adapter = source_hashes.get("adapters/nav2_task_impact_trial.py")
        expected_gate = source_hashes.get("adapters/nav2_trial.py")
        if row.get("adapter_sha256") != expected_adapter:
            errors.append(f"adapter hash mismatch: {path.name}")
        if row.get("gate_dependency_sha256") != expected_gate:
            errors.append(f"gate hash mismatch: {path.name}")
        if row.get("middleware_enforcement_boundary") != "post_collision_monitor_pre_actuator":
            errors.append(f"wrong enforcement boundary: {path.name}")
        if row.get("collision_monitor_preserved") is not True:
            errors.append(f"Collision Monitor not preserved: {path.name}")
        if row.get("application_source_modified_loc") != 0:
            errors.append(f"application source modification: {path.name}")
        if row.get("controller_source_modified_loc") != 0:
            errors.append(f"controller source modification: {path.name}")
        if row.get("prohibited_mppi_command_admissions") != 0:
            errors.append(f"MPPI admission: {path.name}")
        if not row.get("raw_events") or not row.get("final_command_trace"):
            errors.append(f"missing command trace: {path.name}")
        if not row.get("navigation_feedback_trace") or not row.get("clearance_trace"):
            errors.append(f"missing task trace: {path.name}")
        row["_file"] = path.name
        row["_model_file"] = model_path.name
        row["_raw_file"] = raw_path.name
        rows.append(row)

    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing:
        errors.append(f"missing expected cells: {missing}")
    if extra:
        errors.append(f"unexpected cells: {extra}")
    if len(rows) != 160:
        errors.append(f"expected 160 valid rows, found {len(rows)}")
    progress = result_dir / "progress.log"
    progress_lines = progress.read_text(encoding="utf-8").splitlines() if progress.exists() else []
    if len(progress_lines) != 160 or any(not line.endswith(": done") for line in progress_lines):
        errors.append("progress.log is not 160/160 all-done")
    upstream_analysis = result_dir / "analysis.json"
    if not upstream_analysis.exists():
        errors.append("upstream analysis.json is missing")
    else:
        analysis = json.loads(upstream_analysis.read_text(encoding="utf-8"))
        if not analysis.get("matrix_complete") or not analysis.get("mechanism_invariants_passed"):
            errors.append("upstream matrix/mechanism invariant failed")

    audit = {
        "passed": not errors,
        "errors": errors,
        "expected_rows": 160,
        "observed_rows": len(rows),
        "result_files": len(result_files),
        "profile_sha256": manifest.get("profile_sha256"),
        "manifest_sha256": sha256(manifest_path),
        "analysis_sha256": sha256(upstream_analysis) if upstream_analysis.exists() else None,
    }
    if errors:
        raise RuntimeError("fail-closed audit failed:\n- " + "\n- ".join(errors))
    return rows, audit


def summarize(rows: list[dict], audit: dict) -> dict:
    indexed = {
        (row["rmw_implementation"], row["policy"], row["mode"], row["condition"], row["seed"]): row
        for row in rows
    }
    cells: dict[str, dict] = {}
    strata: dict[str, dict] = {}
    bootstrap_rng = random.Random(BOOTSTRAP_SEED)
    pooled_rd_samples: list[float] = []
    pooled_did_samples: list[float] = []
    pooled_clearance_samples: list[float] = []

    for rmw in RMWS:
        for policy in POLICIES:
            for mode in MODES:
                for condition in CONDITIONS:
                    selected = [indexed[(rmw, policy, mode, condition, seed)] for seed in SEEDS]
                    collisions = sum(bool(row["geometric_overlap_observed"]) for row in selected)
                    key = "|".join((rmw, policy, mode, condition))
                    latencies = [
                        float(row["first_fallback_command_ms"])
                        for row in selected if row.get("first_fallback_command_ms") is not None
                    ]
                    cells[key] = {
                        "n": len(selected),
                        "collisions": collisions,
                        "collision_rate": collisions / len(selected),
                        "collision_rate_wilson_95": wilson(collisions, len(selected)),
                        "goal_or_horizon_complete": sum(
                            row.get("goal_result") in {"SUCCEEDED", "HORIZON_COMPLETE"} for row in selected
                        ),
                        "stale_final_runs": sum(row.get("stale_final_commands", 0) > 0 for row in selected),
                        "stale_final_commands": sum(row.get("stale_final_commands", 0) for row in selected),
                        "transfer_runs": sum(row.get("transfer_events", 0) > 0 for row in selected),
                        "fresh_dwb_runs": sum(
                            row.get("first_fallback_command_ms") is not None
                            and row.get("first_fallback_evidence_ms") is not None
                            for row in selected
                        ),
                        "min_clearance_mean_m": mean(float(row["min_geometric_clearance_m"]) for row in selected),
                        "min_clearance_min_m": min(float(row["min_geometric_clearance_m"]) for row in selected),
                        "handoff_latency_median_ms": median(latencies) if latencies else None,
                        "handoff_latency_p95_ms": quantile(latencies, 0.95) if latencies else None,
                    }

            independent_fault = [
                int(indexed[(rmw, policy, "independent", "burst_late", seed)]["geometric_overlap_observed"])
                for seed in SEEDS
            ]
            pit_fault = [
                int(indexed[(rmw, policy, "pit", "burst_late", seed)]["geometric_overlap_observed"])
                for seed in SEEDS
            ]
            independent_normal = [
                int(indexed[(rmw, policy, "independent", "normal", seed)]["geometric_overlap_observed"])
                for seed in SEEDS
            ]
            pit_normal = [
                int(indexed[(rmw, policy, "pit", "normal", seed)]["geometric_overlap_observed"])
                for seed in SEEDS
            ]
            clearance_differences = [
                float(indexed[(rmw, policy, "pit", "burst_late", seed)]["min_geometric_clearance_m"])
                - float(indexed[(rmw, policy, "independent", "burst_late", seed)]["min_geometric_clearance_m"])
                for seed in SEEDS
            ]
            risk_difference = mean(independent_fault) - mean(pit_fault)
            collision_did = (
                mean(independent_fault) - mean(independent_normal)
                - (mean(pit_fault) - mean(pit_normal))
            )
            rd_boot, did_boot, clearance_boot = [], [], []
            local_rng = random.Random(BOOTSTRAP_SEED + RMWS.index(rmw) * 100 + POLICIES.index(policy))
            for _ in range(BOOTSTRAP_REPS):
                picks = [local_rng.randrange(len(SEEDS)) for _ in SEEDS]
                rd_boot.append(mean(independent_fault[i] - pit_fault[i] for i in picks))
                did_boot.append(mean(
                    (independent_fault[i] - independent_normal[i])
                    - (pit_fault[i] - pit_normal[i]) for i in picks
                ))
                clearance_boot.append(mean(clearance_differences[i] for i in picks))
            key = "|".join((rmw, policy))
            strata[key] = {
                "fault_collision_risk_difference_independent_minus_pit": risk_difference,
                "risk_difference_paired_bootstrap_95": [quantile(rd_boot, 0.025), quantile(rd_boot, 0.975)],
                "collision_difference_in_differences_benefit": collision_did,
                "collision_did_paired_bootstrap_95": [quantile(did_boot, 0.025), quantile(did_boot, 0.975)],
                "fault_clearance_pit_minus_independent_mean_m": mean(clearance_differences),
                "clearance_paired_bootstrap_95_m": [quantile(clearance_boot, 0.025), quantile(clearance_boot, 0.975)],
                "paired_mcnemar": exact_mcnemar(independent_fault, pit_fault),
            }

    stratum_keys = [(rmw, policy) for rmw in RMWS for policy in POLICIES]
    for _ in range(BOOTSTRAP_REPS):
        rd_values, did_values, clearance_values = [], [], []
        for rmw, policy in stratum_keys:
            picks = [bootstrap_rng.choice(SEEDS) for _ in SEEDS]
            ind_fault = [int(indexed[(rmw, policy, "independent", "burst_late", s)]["geometric_overlap_observed"]) for s in picks]
            pit_fault = [int(indexed[(rmw, policy, "pit", "burst_late", s)]["geometric_overlap_observed"]) for s in picks]
            ind_normal = [int(indexed[(rmw, policy, "independent", "normal", s)]["geometric_overlap_observed"]) for s in picks]
            pit_normal = [int(indexed[(rmw, policy, "pit", "normal", s)]["geometric_overlap_observed"]) for s in picks]
            rd_values.append(mean(ind_fault) - mean(pit_fault))
            did_values.append(mean(ind_fault) - mean(ind_normal) - (mean(pit_fault) - mean(pit_normal)))
            clearance_values.append(mean(
                float(indexed[(rmw, policy, "pit", "burst_late", s)]["min_geometric_clearance_m"])
                - float(indexed[(rmw, policy, "independent", "burst_late", s)]["min_geometric_clearance_m"])
                for s in picks
            ))
        pooled_rd_samples.append(mean(rd_values))
        pooled_did_samples.append(mean(did_values))
        pooled_clearance_samples.append(mean(clearance_values))

    pooled_risk_difference = mean(
        strata["|".join(key)]["fault_collision_risk_difference_independent_minus_pit"]
        for key in stratum_keys
    )
    pooled_did = mean(
        strata["|".join(key)]["collision_difference_in_differences_benefit"]
        for key in stratum_keys
    )
    pooled_clearance = mean(
        strata["|".join(key)]["fault_clearance_pit_minus_independent_mean_m"]
        for key in stratum_keys
    )
    all_pit_fault = [
        row for row in rows if row["mode"] == "pit" and row["condition"] == "burst_late"
    ]
    handoff = [float(row["first_fallback_command_ms"]) for row in all_pit_fault]
    containment = {
        "independent_fault_runs": sum(row["mode"] == "independent" and row["condition"] == "burst_late" for row in rows),
        "independent_fault_stale_runs": sum(row["mode"] == "independent" and row["condition"] == "burst_late" and row["stale_final_commands"] > 0 for row in rows),
        "independent_fault_stale_commands": sum(row["stale_final_commands"] for row in rows if row["mode"] == "independent" and row["condition"] == "burst_late"),
        "pit_fault_runs": len(all_pit_fault),
        "pit_fault_stale_runs": sum(row["stale_final_commands"] > 0 for row in all_pit_fault),
        "pit_fault_stale_commands": sum(row["stale_final_commands"] for row in all_pit_fault),
        "pit_fault_transfer_runs": sum(row["transfer_events"] > 0 for row in all_pit_fault),
        "pit_fault_fresh_dwb_runs": sum(row["first_fallback_command_ms"] is not None and row["first_fallback_evidence_ms"] is not None for row in all_pit_fault),
        "handoff_latency_median_ms": median(handoff),
        "handoff_latency_p95_ms": quantile(handoff, 0.95),
        "normal_runs": sum(row["condition"] == "normal" for row in rows),
        "normal_false_transfer_runs": sum(row["condition"] == "normal" and row["transfer_events"] > 0 for row in rows),
        "normal_stale_runs": sum(row["condition"] == "normal" and row["stale_final_commands"] > 0 for row in rows),
    }
    return {
        "revision": "c7-locked-collision-analysis-v1",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_repetitions": BOOTSTRAP_REPS,
        "audit": audit,
        "cells": cells,
        "strata": strata,
        "pooled_equal_strata": {
            "fault_collision_risk_difference_independent_minus_pit": pooled_risk_difference,
            "risk_difference_stratified_bootstrap_95": [quantile(pooled_rd_samples, 0.025), quantile(pooled_rd_samples, 0.975)],
            "collision_difference_in_differences_benefit": pooled_did,
            "collision_did_stratified_bootstrap_95": [quantile(pooled_did_samples, 0.025), quantile(pooled_did_samples, 0.975)],
            "fault_clearance_pit_minus_independent_mean_m": pooled_clearance,
            "clearance_stratified_bootstrap_95_m": [quantile(pooled_clearance_samples, 0.025), quantile(pooled_clearance_samples, 0.975)],
        },
        "containment_and_transfer": containment,
    }


def write_outputs(result_dir: Path, rows: list[dict], result: dict) -> None:
    json_path = result_dir / "c7_locked_statistics.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    csv_path = result_dir / "c7_locked_rows.csv"
    columns = [
        "rmw_implementation", "policy", "mode", "condition", "seed", "goal_result",
        "geometric_overlap_observed", "geometric_overlap_episode_count",
        "min_geometric_clearance_m", "stale_final_commands", "transfer_events",
        "first_fallback_command_ms", "first_fallback_evidence_ms", "profile_sha256",
        "adapter_sha256", "gate_dependency_sha256", "_file", "_model_file", "_raw_file",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# C7 locked held-out collision analysis", "",
        f"- Audit: **{'PASS' if result['audit']['passed'] else 'FAIL'}** ({result['audit']['observed_rows']}/160 rows)",
        f"- Profile SHA-256: `{result['audit']['profile_sha256']}`", "",
        "## Fault collision and containment", "",
        "| DDS | Policy | Independent collisions | PIT collisions | Risk difference | Paired bootstrap 95% CI | McNemar p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for rmw in RMWS:
        for policy in POLICIES:
            independent = result["cells"]["|".join((rmw, policy, "independent", "burst_late"))]
            pit = result["cells"]["|".join((rmw, policy, "pit", "burst_late"))]
            stratum = result["strata"]["|".join((rmw, policy))]
            ci = stratum["risk_difference_paired_bootstrap_95"]
            lines.append(
                f"| {rmw} | {policy} | {independent['collisions']}/10 | {pit['collisions']}/10 | "
                f"{stratum['fault_collision_risk_difference_independent_minus_pit']:.3f} | "
                f"[{ci[0]:.3f}, {ci[1]:.3f}] | {stratum['paired_mcnemar']['exact_two_sided_p']:.4f} |"
            )
    pooled = result["pooled_equal_strata"]
    containment = result["containment_and_transfer"]
    lines.extend([
        "", "## Equal-strata pooled effect", "",
        f"- Fault collision risk reduction: **{pooled['fault_collision_risk_difference_independent_minus_pit']:.3f}** "
        f"(95% stratified bootstrap CI {pooled['risk_difference_stratified_bootstrap_95'][0]:.3f}–{pooled['risk_difference_stratified_bootstrap_95'][1]:.3f}).",
        f"- Collision difference-in-differences benefit: **{pooled['collision_difference_in_differences_benefit']:.3f}** "
        f"(95% CI {pooled['collision_did_stratified_bootstrap_95'][0]:.3f}–{pooled['collision_did_stratified_bootstrap_95'][1]:.3f}).",
        f"- Fault clearance benefit: **{pooled['fault_clearance_pit_minus_independent_mean_m']:.4f} m** "
        f"(95% CI {pooled['clearance_stratified_bootstrap_95_m'][0]:.4f}–{pooled['clearance_stratified_bootstrap_95_m'][1]:.4f} m).",
        "", "## Mechanism", "",
        f"- Independent/Fault stale admission: {containment['independent_fault_stale_runs']}/{containment['independent_fault_runs']} runs, {containment['independent_fault_stale_commands']} commands.",
        f"- PIT/Fault: stale {containment['pit_fault_stale_runs']}/{containment['pit_fault_runs']}, Transfer {containment['pit_fault_transfer_runs']}/{containment['pit_fault_runs']}, fresh DWB {containment['pit_fault_fresh_dwb_runs']}/{containment['pit_fault_runs']}.",
        f"- Handoff latency: median {containment['handoff_latency_median_ms']:.2f} ms, p95 {containment['handoff_latency_p95_ms']:.2f} ms.",
        f"- Normal false Transfer: {containment['normal_false_transfer_runs']}/{containment['normal_runs']}; Normal stale admission: {containment['normal_stale_runs']}/{containment['normal_runs']}.",
        "",
    ])
    (result_dir / "c7_locked_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    rows, audit = load_and_audit(args.result_dir)
    result = summarize(rows, audit)
    write_outputs(args.result_dir, rows, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
