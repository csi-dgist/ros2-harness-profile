#!/usr/bin/env python3
"""Create audited CSVs, figures, and EN/KR rebuttal text from task-impact raw data."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def bootstrap_mean(values: list[float], seed: int = 256, draws: int = 20000) -> list[float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.mean(rng.choice(array, size=(draws, len(array)), replace=True), axis=1)
    return np.quantile(samples, [0.025, 0.975]).tolist()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    analysis = load(args.result_dir / "analysis.json")
    if not analysis.get("matrix_complete") or not analysis.get("mechanism_invariants_passed"):
        raise RuntimeError("refusing to report an incomplete or mechanism-invalid matrix")
    figure_dir = args.result_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    task_rows, feedback_rows, command_rows, event_rows, inference_rows = [], [], [], [], []
    for path in sorted(args.result_dir.glob("rmw_*__*.json")):
        if path.name.endswith("__model.json"):
            continue
        row = load(path)
        run_id = path.stem
        task_rows.append(
            {
                "run_id": run_id,
                **{key: row.get(key) for key in (
                    "rmw_implementation", "policy", "mode", "scene", "condition", "seed",
                    "goal_result", "restricted_navigation_time_s", "navigation_sim_time_s",
                    "progress_loss_auc_5s", "recovery_time_s", "path_length_m", "linear_jerk_rms",
                    "min_center_distance_m", "geometric_overlap_observed", "transfer_events",
                    "handoff_latency_ms", "stale_final_commands", "fallback_evidence_messages",
                    "processing_us_p50", "processing_us_p99", "model_weight_sha256",
                    "profile_sha256", "adapter_sha256",
                )},
            }
        )
        common = {key: row.get(key) for key in ("policy", "mode", "scene", "condition", "seed")}
        for sample in row.get("navigation_feedback_trace", []):
            feedback_rows.append({"run_id": run_id, **common, **sample})
        for sample in row.get("final_command_trace", []):
            command_rows.append({"run_id": run_id, **common, **sample})
        for sample in row.get("raw_events", []):
            event_rows.append(
                {
                    "run_id": run_id, **common,
                    **{key: sample.get(key) for key in (
                        "event_index", "produced_ns", "observed_ns", "age_ms", "compute_ms",
                        "transport_ms", "source", "violations", "admit", "replace_with_safe_value",
                        "transfer", "stale", "action", "processing_us",
                    )},
                }
            )
        model = load(args.result_dir / f"{run_id}__model.json")
        for sample in model.get("raw_records", []):
            inference_rows.append(
                {
                    "run_id": run_id, **common,
                    "sequence": sample.get("sequence"),
                    "wall_time_ns": sample.get("wall_time_ns"),
                    "linear_x": sample.get("output", [None, None])[0],
                    "angular_z": sample.get("output", [None, None])[1],
                    "inference_us": sample.get("inference_us"),
                }
            )
    write_csv(args.result_dir / "task_runs.csv", list(task_rows[0]), task_rows)
    write_csv(args.result_dir / "navigation_feedback.csv", list(feedback_rows[0]), feedback_rows)
    write_csv(args.result_dir / "command_trace.csv", list(command_rows[0]), command_rows)
    write_csv(args.result_dir / "middleware_events.csv", list(event_rows[0]), event_rows)
    write_csv(args.result_dir / "inference_samples.csv", list(inference_rows[0]), inference_rows)

    effect_items = sorted(analysis["difference_in_differences"].items())
    labels = ["\n".join(key.replace("rmw_", "").replace("_scan_goal_bc", "").split("|")) for key, _ in effect_items]
    time_means = [value["restricted_time_did_benefit_mean_s"] for _, value in effect_items]
    time_low = [value["restricted_time_did_bootstrap95_s"][0] for _, value in effect_items]
    time_high = [value["restricted_time_did_bootstrap95_s"][1] for _, value in effect_items]
    auc_means = [value["progress_loss_auc_did_benefit_mean"] for _, value in effect_items]
    auc_low = [value["progress_loss_auc_did_bootstrap95"][0] for _, value in effect_items]
    auc_high = [value["progress_loss_auc_did_bootstrap95"][1] for _, value in effect_items]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    axes[0].bar(x, time_means, color="#377eb8")
    axes[0].errorbar(x, time_means, yerr=[np.asarray(time_means) - time_low, np.asarray(time_high) - time_means], fmt="none", color="black", capsize=3)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("PIT benefit in restricted time (s)")
    axes[0].set_title("Counterfactual difference-in-differences (positive favors PIT)")
    axes[1].bar(x, auc_means, color="#4daf4a")
    axes[1].errorbar(x, auc_means, yerr=[np.asarray(auc_means) - auc_low, np.asarray(auc_high) - auc_means], fmt="none", color="black", capsize=3)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("PIT benefit in progress-loss AUC")
    axes[1].set_xticks(x, labels, rotation=30, ha="right", fontsize=8)
    fig.savefig(figure_dir / "did_task_impact.png", dpi=220)
    fig.savefig(figure_dir / "did_task_impact.pdf")
    plt.close(fig)

    groups, group_labels = [], []
    for policy in ("reactive_scan_goal_bc", "temporal_scan_goal_bc"):
        for mode in ("independent", "pit"):
            for condition in ("normal", "burst_late"):
                groups.append([
                    float(row["restricted_navigation_time_s"])
                    for row in task_rows
                    if row["policy"] == policy and row["mode"] == mode and row["condition"] == condition
                ])
                group_labels.append(f"{policy.split('_')[0]}\n{mode}\n{condition}")
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.boxplot(groups, labels=group_labels, showmeans=True)
    ax.set_ylabel("Restricted navigation time (s; failures=75 s)")
    ax.set_title("PIT changes task outcomes under the same learned-policy fault")
    fig.savefig(figure_dir / "restricted_time_boxplot.png", dpi=220)
    fig.savefig(figure_dir / "restricted_time_boxplot.pdf")
    plt.close(fig)

    all_time_did = [value for _, effect in effect_items for value in effect["raw_restricted_time_did"]]
    all_auc_did = [value for _, effect in effect_items for value in effect["raw_progress_loss_auc_did"]]
    independent_fault = [row for row in task_rows if row["mode"] == "independent" and row["condition"] == "burst_late"]
    pit_fault = [row for row in task_rows if row["mode"] == "pit" and row["condition"] == "burst_late"]
    inference = [float(row["inference_us"]) for row in inference_rows if row["inference_us"] is not None]
    processing = [float(row["processing_us"]) for row in event_rows if row["processing_us"] is not None]
    summary = {
        "revision": "learned-task-impact-report-v1",
        "runs": len(task_rows),
        "independent_fault_runs": len(independent_fault),
        "independent_stale_runs": sum(int(row["stale_final_commands"] or 0) > 0 for row in independent_fault),
        "pit_fault_runs": len(pit_fault),
        "pit_transfer_runs": sum(int(row["transfer_events"] or 0) == 1 for row in pit_fault),
        "pit_stale_runs": sum(int(row["stale_final_commands"] or 0) > 0 for row in pit_fault),
        "restricted_time_did_benefit_mean_s": float(np.mean(all_time_did)),
        "restricted_time_did_benefit_bootstrap95_s": bootstrap_mean(all_time_did),
        "progress_loss_auc_did_benefit_mean": float(np.mean(all_auc_did)),
        "progress_loss_auc_did_benefit_bootstrap95": bootstrap_mean(all_auc_did, 257),
        "inference_us_p50": float(np.quantile(inference, 0.5)),
        "inference_us_p99": float(np.quantile(inference, 0.99)),
        "pit_processing_us_p50": float(np.quantile(processing, 0.5)),
        "pit_processing_us_p99": float(np.quantile(processing, 0.99)),
    }
    (args.result_dir / "REPORT.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    low, high = summary["restricted_time_did_benefit_bootstrap95_s"]
    auc_low_value, auc_high_value = summary["progress_loss_auc_did_benefit_bootstrap95"]
    en = (
        f"Across 320 Gazebo/Nav2 tasks spanning two learned scan-goal controllers and two DDS implementations, "
        f"PIT blocked stale actuator-facing commands in all {summary['pit_fault_runs']} injected runs "
        f"({summary['pit_stale_runs']} stale admissions) and transferred to DWB in {summary['pit_transfer_runs']}. "
        f"Counterfactual difference-in-differences showed a {summary['restricted_time_did_benefit_mean_s']:.2f}-s "
        f"reduction in fault-attributable restricted task time (bootstrap 95% CI {low:.2f} to {high:.2f}) and "
        f"a {summary['progress_loss_auc_did_benefit_mean']:.3f} reduction in 5-s progress-loss AUC "
        f"(95% CI {auc_low_value:.3f} to {auc_high_value:.3f})."
    )
    kr = (
        f"두 학습형 scan-goal controller와 두 DDS 구현을 포괄한 320회의 Gazebo/Nav2 태스크에서, "
        f"PIT는 지연을 주입한 {summary['pit_fault_runs']}회 모두에서 actuator-facing stale command를 차단했고 "
        f"(stale admission {summary['pit_stale_runs']}회), {summary['pit_transfer_runs']}회 DWB로 권한을 이전했다. "
        f"반사실적 차이의 차이 분석에서 fault-attributable restricted task time은 "
        f"{summary['restricted_time_did_benefit_mean_s']:.2f}초 감소했고(bootstrap 95% CI {low:.2f}–{high:.2f}), "
        f"5초 progress-loss AUC는 {summary['progress_loss_auc_did_benefit_mean']:.3f} 감소했다"
        f"(95% CI {auc_low_value:.3f}–{auc_high_value:.3f})."
    )
    (args.result_dir / "REBUTTAL_RESULT_EN-KR.md").write_text(
        "# Rebuttal-ready result\n\n## EN\n\n" + en + "\n\n## KR\n\n" + kr + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
