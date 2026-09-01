#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-derive every reported number from the locked run records.

quick_verify.py checks the per-cell table. This checks every value the paper prints,
and it exits non-zero on any disagreement. Rows labelled "beyond the paper" are derived
from the same records for context, and the paper does not print them.
Percentiles are linear interpolations between order statistics, and the per-sample cost
is the median over runs of each run's own percentile, as the paper reports it.

  python3 analysis/derive_paper_numbers.py [result_dir]
"""

from __future__ import annotations

import argparse
import json
from math import comb
import statistics as stat
import sys
from pathlib import Path

DEFAULT_RESULTS = (Path(__file__).resolve().parents[1] / "results"
                   / "public_policy_collision_c7_locked_heldout_n10_v1")
PROFILE_SHA256 = "625b6649e1cf28e673dd4bf8df6af71bbdde55ebfe96457c033d9b96fd87078d"
GOAL_STATES = ("SUCCEEDED", "HORIZON_COMPLETE")


def percentile(values, q):
    """Linear interpolation between order statistics, as in numpy."""
    xs = sorted(values)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def wilson_upper(successes, n, z=1.959963984540054):
    """Upper bound of the Wilson score interval."""
    p = successes / n
    centre = p + z * z / (2 * n)
    spread = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (centre + spread) / (1 + z * z / n)


def mcnemar_exact_two_sided(b, c):
    """Exact two-sided sign test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def load(result_dir: Path):
    runs = []
    for path in sorted(result_dir.glob("*__*.json")):
        if path.name.endswith("__model.json"):
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        if "rmw_implementation" in row:
            runs.append(row)
    stats = json.loads((result_dir / "c7_locked_statistics.json").read_text(encoding="utf-8"))
    return runs, stats


def select(runs, **kw):
    return [r for r in runs if all(r.get(k) == v for k, v in kw.items())]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir", nargs="?", type=Path, default=DEFAULT_RESULTS)
    a = ap.parse_args()
    if not a.result_dir.exists():
        print(f"result directory not found: {a.result_dir}")
        return 2

    runs, stats = load(a.result_dir)
    pooled = stats["pooled_equal_strata"]
    contain = stats["containment_and_transfer"]
    strata = stats["strata"]

    off_fault = select(runs, mode="independent", condition="burst_late")
    pit_fault = select(runs, mode="pit", condition="burst_late")
    normal = select(runs, condition="normal")

    def collisions(rows):
        return sum(1 for r in rows if r["geometric_overlap_observed"])

    def stale_runs(rows):
        return sum(1 for r in rows if r["stale_final_commands"] > 0)

    def stale_cmds(rows):
        return sum(r["stale_final_commands"] for r in rows)

    def transfers(rows):
        return sum(1 for r in rows if r.get("transfer_events"))

    def run_p50(rows):
        return stat.median(r["processing_us_p50"] for r in rows)

    def run_p99(rows):
        return stat.median(r["processing_us_p99"] for r in rows)

    strat_keys = sorted({(r["rmw_implementation"], r["policy"]) for r in runs})
    strat_coll = sorted(collisions([r for r in off_fault
                                    if (r["rmw_implementation"], r["policy"]) == k])
                        for k in strat_keys)
    strat_stale = sorted(stale_cmds([r for r in off_fault
                                     if (r["rmw_implementation"], r["policy"]) == k])
                         for k in strat_keys)
    handoff = [r["handoff_latency_ms"] for r in pit_fault]
    evidence = [r["first_fallback_evidence_ms"] for r in pit_fault]
    handoff_cell = {k: stat.median([r["handoff_latency_ms"] for r in pit_fault
                                    if (r["rmw_implementation"], r["policy"]) == k])
                    for k in strat_keys}
    cost_cell = {k: run_p50([r for r in pit_fault
                             if (r["rmw_implementation"], r["policy"]) == k])
                 for k in strat_keys}
    fast = [v for k, v in cost_cell.items() if k[0] == "rmw_fastrtps_cpp"]
    cyclone = [v for k, v in cost_cell.items() if k[0] == "rmw_cyclonedds_cpp"]
    mcnemar = {k: v["paired_mcnemar"] for k, v in strata.items()}

    # (paper location, printed value, recomputed value, agreement)
    checks = [
        ("§5 matrix", "160 runs", len(runs), len(runs) == 160),
        ("§5 matrix", "16 cells of 10", sorted({len(select(runs, rmw_implementation=k[0], policy=k[1],
                                                           mode=m, condition=c))
                                                for k in strat_keys
                                                for m in ("independent", "pit")
                                                for c in ("normal", "burst_late")}),
         sorted({len(select(runs, rmw_implementation=k[0], policy=k[1], mode=m, condition=c))
                 for k in strat_keys for m in ("independent", "pit")
                 for c in ("normal", "burst_late")}) == [10]),
        ("§5 portability", f"one Profile SHA-256 {PROFILE_SHA256[:8]}…",
         sorted({r["profile_sha256"] for r in runs}),
         sorted({r["profile_sha256"] for r in runs}) == [PROFILE_SHA256]),
        ("§5 portability", "application 0, controller 0",
         (sorted({r["application_source_modified_loc"] for r in runs}),
          sorted({r["controller_source_modified_loc"] for r in runs})),
         sorted({r["application_source_modified_loc"] for r in runs}) == [0]
         and sorted({r["controller_source_modified_loc"] for r in runs}) == [0]),
        ("beyond the paper", "all 160 reached the goal or completed the horizon",
         sorted({r["goal_result"] for r in runs}),
         all(r["goal_result"] in GOAL_STATES for r in runs)),

        ("§5 containment", "Profile-off 40/40", f"{stale_runs(off_fault)}/40",
         stale_runs(off_fault) == 40),
        ("§5 containment", "PIT 0/40", f"{stale_runs(pit_fault)}/40", stale_runs(pit_fault) == 0),
        ("Table 3 stale commands", "Profile-off 447", stale_cmds(off_fault), stale_cmds(off_fault) == 447),
        ("Table 3 stale commands", "PIT 0", stale_cmds(pit_fault), stale_cmds(pit_fault) == 0),
        ("Table 3 stale commands per cell", "78, 145, 80, 144", strat_stale,
         strat_stale == [78, 80, 144, 145]),
        ("Table 3 collisions", "Profile-off 36/40", f"{collisions(off_fault)}/40",
         collisions(off_fault) == 36),
        ("Table 3 collisions", "PIT 0/40", f"{collisions(pit_fault)}/40", collisions(pit_fault) == 0),
        ("Table 3 collisions per cell", "8, 10, 9, 9 of 10", strat_coll, strat_coll == [8, 9, 9, 10]),
        ("§5 transfer", "Profile-off 0/40", f"{transfers(off_fault)}/40", transfers(off_fault) == 0),
        ("§5 transfer", "PIT 40/40", f"{transfers(pit_fault)}/40", transfers(pit_fault) == 40),
        ("§5 transfer", "PIT fresh DWB command 40/40",
         f"{sum(1 for r in pit_fault if r['fallback_evidence_messages'] > 0)}/40",
         sum(1 for r in pit_fault if r["fallback_evidence_messages"] > 0) == 40
         and contain["pit_fault_fresh_dwb_runs"] == 40),
        ("§5 transfer", "0 of 40 normal runs, both arms",
         f"{transfers(normal)}/80 over both arms", transfers(normal) == 0),

        ("§5 decision cost", "Profile-off p50 155 µs", round(run_p50(off_fault), 1),
         round(run_p50(off_fault)) == 155),
        ("beyond the paper", "Profile-off p99 219 µs", round(run_p99(off_fault), 1),
         round(run_p99(off_fault)) == 219),
        ("§5 decision cost", "PIT p50 146 µs", round(run_p50(pit_fault), 1),
         round(run_p50(pit_fault)) == 146),
        ("beyond the paper", "PIT p99 242 µs", round(run_p99(pit_fault), 1),
         round(run_p99(pit_fault)) == 242),
        ("beyond the paper", "Fast DDS 168 to 170 µs", [round(v, 1) for v in sorted(fast)],
         all(168 <= round(v) <= 170 for v in fast)),
        ("beyond the paper", "Cyclone DDS 109 to 110 µs",
         [round(v, 1) for v in sorted(cyclone)], all(109 <= round(v) <= 110 for v in cyclone)),

        ("§5 hand-off", "median 302 ms", round(stat.median(handoff), 2),
         round(stat.median(handoff)) == 302),
        ("beyond the paper", "p95 371 ms", round(percentile(handoff, 0.95), 2),
         round(percentile(handoff, 0.95)) == 371),
        ("beyond the paper", "DWB first local plan, median 276 ms", round(stat.median(evidence), 2),
         round(stat.median(evidence)) == 276),
        ("beyond the paper", "cell medians 294 to 331 ms",
         {"|".join(k): round(v, 1) for k, v in sorted(handoff_cell.items())},
         all(294 <= round(v) <= 331 for v in handoff_cell.values())),

        ("beyond the paper", "risk reduction 0.900",
         round(pooled["fault_collision_risk_difference_independent_minus_pit"], 3),
         abs(pooled["fault_collision_risk_difference_independent_minus_pit"] - 0.9) < 5e-4),
        ("beyond the paper", "95% CI 0.800 to 0.975",
         [round(v, 3) for v in pooled["risk_difference_stratified_bootstrap_95"]],
         [round(v, 3) for v in pooled["risk_difference_stratified_bootstrap_95"]] == [0.8, 0.975]),
        ("beyond the paper", "clearance +0.33 m", round(pooled["fault_clearance_pit_minus_independent_mean_m"], 4),
         round(pooled["fault_clearance_pit_minus_independent_mean_m"], 2) == 0.33),
        ("beyond the paper", "clearance CI 0.29 to 0.36 m",
         [round(v, 4) for v in pooled["clearance_stratified_bootstrap_95_m"]],
         [round(v, 2) for v in pooled["clearance_stratified_bootstrap_95_m"]] == [0.29, 0.36]),
        ("beyond the paper", "McNemar exact two-sided p \u2264 0.008 per cell",
         {k: v["exact_two_sided_p"] for k, v in sorted(mcnemar.items())},
         all(v["exact_two_sided_p"] <= 0.008 for v in mcnemar.values())),
        ("beyond the paper", "McNemar p from the discordant pairs",
         {k: round(mcnemar_exact_two_sided(v["independent_collision_pit_safe_pairs"],
                                           v["independent_safe_pit_collision_pairs"]), 8)
          for k, v in sorted(mcnemar.items())},
         all(abs(mcnemar_exact_two_sided(v["independent_collision_pit_safe_pairs"],
                                         v["independent_safe_pit_collision_pairs"])
                 - v["exact_two_sided_p"]) < 1e-9 for v in mcnemar.values())),
        ("beyond the paper", "Wilson 95% upper bound 27.8% for 0 of 10",
         round(100 * wilson_upper(0, 10), 1), round(100 * wilson_upper(0, 10), 1) == 27.8),
    ]

    width = max(len(w) for w, _, _, _ in checks)
    bad = 0
    print(f"{len(runs)} run records read from {a.result_dir}\n")
    print(f"{'paper':<{width}}  {'printed':<44}  recomputed")
    print("-" * (width + 92))
    for where, printed, got, ok in checks:
        if not ok:
            bad += 1
        print(f"{where:<{width}}  {str(printed):<44}  {got}  {'OK' if ok else 'MISMATCH'}")
    print()
    if bad:
        print(f"{bad} of {len(checks)} checks disagree with the records.")
        return 1
    print(f"{len(checks)} checks agree with the {len(runs)} run records.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
