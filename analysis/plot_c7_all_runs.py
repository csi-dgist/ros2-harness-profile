#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C7 locked held-out 의 fault 조건 40쌍을 장애물 기준 좌표계에 모두 겹쳐 그린다.

세계 좌표로 40런을 겹치면 장애물 40개가 서로 포개져 아무것도 읽히지 않는다.
각 런을 자기 장애물이 원점에 오도록 평행이동하면 장애물이 원 하나로 모이고,
어떤 궤적이 그 원에 들어갔는지가 바로 보인다.

궤적은 두 발자국이 맞닿는 순간에서 끊는다. 장애물은 실체가 있는 원기둥이므로
로봇은 거기서 멈춘다. 그 뒤의 pose 는 동결 명령의 추측항법이다.

정본 데이터는 잠긴 결과이며 이 스크립트는 읽기만 한다.

실행.
  python plot_c7_all_runs.py --out fig_all
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R_ROBOT, R_OBST = 0.22, 0.25
R_SUM = R_ROBOT + R_OBST              # 감사 판정 반경. 여유를 포함한다.
# 실제 차체가 닿는 거리. TurtleBot3 Burger 178 x 138 mm 의 반대각선.
R_CONTACT = R_OBST + math.hypot(0.178 / 2, 0.138 / 2)
RESULTS = (Path(__file__).resolve().parents[1] / "results"
           / "public_policy_collision_c7_locked_heldout_n10_v1")

RMWS = ("rmw_fastrtps_cpp", "rmw_cyclonedds_cpp")
POLICIES = ("drl", "drl_vo")
SEEDS = range(30, 40)

C_HIT = "#c0392b"
C_OK = "#0f8a5a"
C_OBST = "#7a6a58"


def load(mode):
    out = []
    for rmw in RMWS:
        for pol in POLICIES:
            for seed in SEEDS:
                f = RESULTS / f"{rmw}__{pol}__{mode}__crossing__burst_late__{seed}.json"
                out.append(json.loads(f.read_text(encoding="utf-8")))
    return out


def relative_track(d):
    """장애물을 원점으로 옮긴 궤적. 차체가 닿는 순간에서 끊는다."""
    ox, oy = d["obstacle_x"], d["obstacle_y"]
    fb = d["navigation_feedback_trace"]
    xs, ys = [], []
    for q in fb:
        rx, ry = q["x"] - ox, q["y"] - oy
        xs.append(rx); ys.append(ry)
        if math.hypot(rx, ry) <= R_CONTACT:
            break                     # 차체가 닿는 순간이 이 런의 끝이다
    return np.array(xs), np.array(ys)


def panel(ax, runs, title):
    hits = sum(1 for d in runs if d["geometric_overlap_observed"])

    ax.add_patch(plt.Circle((0, 0), R_OBST, facecolor=C_OBST, alpha=.30,
                            edgecolor=C_OBST, lw=1.4, zorder=3))
    ax.add_patch(plt.Circle((0, 0), R_CONTACT, facecolor="none", edgecolor=C_OBST,
                            lw=1.4, ls=(0, (5, 4)), zorder=3))

    for d in runs:
        xs, ys = relative_track(d)
        hit = d["geometric_overlap_observed"]
        ax.plot(xs, ys, "-", lw=1.6 if not hit else 1.0,
                alpha=.9 if not hit else .55,
                color=C_HIT if hit else C_OK, zorder=5 if not hit else 4)
        ax.plot(xs[-1], ys[-1], "o", ms=3.4, color=C_HIT if hit else C_OK,
                zorder=6, markeredgecolor="none")

    ax.set_title(f"{title}   ·   {hits} / {len(runs)} collisions",
                 fontsize=11.5, weight="bold",
                 color=C_HIT if hits else C_OK, pad=8)
    ax.set_aspect("equal")
    ax.grid(True, color="#e9edf2", lw=.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="c7_all_runs")
    a = ap.parse_args()

    off, pit = load("independent"), load("pit")

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.9), dpi=200, sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    panel(axes[0], off, "Profile-off")
    panel(axes[1], pit, "PIT")
    axes[0].set_ylabel("y relative to this run's obstacle [m]", fontsize=8.8)
    fig.text(.5, .13, "x and y relative to this run's obstacle [m]", ha="center", fontsize=8.8)

    tracks = [relative_track(d) for d in off + pit]
    allx = np.concatenate([t[0] for t in tracks])
    ally = np.concatenate([t[1] for t in tracks])
    m = .18
    axes[0].set_xlim(allx.min() - m, allx.max() + m)
    axes[0].set_ylim(ally.min() - m, ally.max() + m)

    fig.suptitle("All 40 fault-condition runs per arm, each shifted so its own obstacle sits at the origin",
                 fontsize=10.8, y=.975)

    L = plt.Line2D
    fig.legend(handles=[
        L([], [], color=C_HIT, lw=1.6, label="run that reached contact"),
        L([], [], color=C_OK, lw=1.8, label="run that stopped clear"),
        L([], [], marker="o", color=C_OBST, ls="", ms=9, alpha=.5,
          label="obstacle, r = 0.25 m"),
        L([], [], color=C_OBST, lw=1.4, ls=(0, (5, 4)),
          label="chassis contact, 0.36 m"),
    ], loc="lower center", ncol=2, frameon=False, fontsize=8.3, bbox_to_anchor=(.5, .005))



    fig.tight_layout(rect=(0, .175, 1, .95))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix("." + ext), facecolor="white", bbox_inches="tight")
    print(f"saved {out}.png / .pdf")

    for name, runs in (("Profile-off", off), ("PIT", pit)):
        hits = sum(1 for d in runs if d["geometric_overlap_observed"])
        mins = [min(math.hypot(q["x"] - d["obstacle_x"], q["y"] - d["obstacle_y"])
                    for q in d["navigation_feedback_trace"]) for d in runs]
        touched = sum(1 for m in mins if m <= R_CONTACT)
        print(f"  {name:12s} audited {hits}/{len(runs)}   "
              f"chassis contact {touched}/{len(runs)}   closest {min(mins):.3f} m")


if __name__ == "__main__":
    raise SystemExit(main())
