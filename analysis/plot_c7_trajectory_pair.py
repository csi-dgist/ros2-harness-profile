#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C7 locked held-out 짝 궤적 그림 (2D 평면도 + 3D 장면도).

같은 seed·정책·DDS·장면에서 Profile-off (independent) 와 PIT 를 나란히 그린다.
Middleware 2026 #256 프로젝트 페이지와 카메라레디 running case 용.

## 장애물 배치 규칙 (그림에서 반드시 드러내야 하는 것)

장애물은 고정 좌표에 놓이지 않는다. **trigger 시점에 큐에 있던 명령을 예측 호(arc)
로 적분해, 고정 arc-length 약 1.04 m 지점에 스폰**한다. 설계 문서가 이를
"predicted arc-length" 로 부르며 byte-identical 파일로 고정한다.

그래서 두 run 의 장애물 좌표가 다르다. 큐 명령의 angular_z 가 다르기 때문이다
(이 짝에서 -0.112 rad/s 대 +0.102 rad/s → 방위 -13.0° 대 +11.7°).
range 는 열 seed 전부에서 1.0356 ~ 1.0448 m 로 일정하다.

이 설계는 의도된 것이다. 고정 좌표를 쓰면 충돌 노출이 "큐 마지막에 우연히
어떤 명령이 있었는가" 에 좌우된다는 문제가 앞선 회차에서 관측됐다. 예측 호에
놓으면 두 arm 모두 자기 진행 경로에 동일하게 적대적인 장애물을 만난다.

정본 데이터는 잠긴 held-out 결과이며 이 스크립트는 읽기만 한다.

실행.
  python plot_c7_trajectory_pair.py --out fig            # 2D + 3D 둘 다
  python plot_c7_trajectory_pair.py --view 2d --out fig
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# 설계 문서 기준. Gazebo robot radius 0.22 m, obstacle radius 0.25 m.
R_ROBOT = 0.22
R_OBST = 0.25
R_COMBINED = R_ROBOT + R_OBST          # 0.47 m. 이 아래로 내려가면 geometric overlap

RESULTS = (Path(__file__).resolve().parents[1] / "results"
           / "public_policy_collision_c7_locked_heldout_n10_v1")

C_PLAN = "#a7b0bc"
C_OFF = "#c0392b"
C_PIT = "#0f8a5a"
C_TRIG = "#b5740a"
C_OBST = "#7a6a58"
C_ROBOT = "#2b3a4a"
C_GRID = "#e9edf2"
C_DIM = "#46535f"          # 치수선. 궤적 색과 절대 겹치지 않게 둔다.


def load(rmw, policy, mode, seed):
    f = RESULTS / f"{rmw}__{policy}__{mode}__crossing__burst_late__{seed}.json"
    return json.loads(f.read_text(encoding="utf-8"))


def xy(d):
    fb = d["navigation_feedback_trace"]
    return np.array([p["x"] for p in fb]), np.array([p["y"] for p in fb])


def transfer_index(d):
    sw = next((h for h in (d.get("selector_history") or [])
               if h.get("controller") == "DWB"), None)
    if not sw:
        return None
    t = sw["wall_time"] * 1e9
    fb = d["navigation_feedback_trace"]
    return min(range(len(fb)), key=lambda i: abs(fb[i]["wall_time_ns"] - t))


def closest_index(d):
    ox, oy = d["obstacle_x"], d["obstacle_y"]
    xs, ys = xy(d)
    return int(np.argmin((xs - ox) ** 2 + (ys - oy) ** 2))


def predicted_arc(d, n=60):
    """trigger pose 에서 큐 명령을 적분한 예측 호. 끝점이 장애물 스폰 지점이다."""
    tp = d["trigger_pose"]
    q = tp.get("queued_command") or {}
    v, w = q.get("linear_x", 0.0), q.get("angular_z", 0.0)
    ox, oy = d["obstacle_x"], d["obstacle_y"]
    arc = math.hypot(ox - tp["x"], oy - tp["y"])
    if abs(v) < 1e-6:
        return np.array([tp["x"], ox]), np.array([tp["y"], oy]), arc
    # 호 길이가 arc 가 되도록 시간을 잡는다.
    T = arc / abs(v) if abs(w) < 1e-9 else None
    if T is None:
        # 곡선일 때는 현(chord) 길이가 arc 와 맞는 시간을 수치적으로 찾는다.
        ts = np.linspace(.01, 12, 3000)
        th = tp["yaw"] + w * ts
        px = tp["x"] + (v / w) * (np.sin(th) - np.sin(tp["yaw"]))
        py = tp["y"] - (v / w) * (np.cos(th) - np.cos(tp["yaw"]))
        T = ts[int(np.argmin(np.abs(np.hypot(px - tp["x"], py - tp["y"]) - arc)))]
    ts = np.linspace(0, T, n)
    th = tp["yaw"] + w * ts
    if abs(w) < 1e-9:
        return tp["x"] + v * ts * math.cos(tp["yaw"]), tp["y"] + v * ts * math.sin(tp["yaw"]), arc
    px = tp["x"] + (v / w) * (np.sin(th) - np.sin(tp["yaw"]))
    py = tp["y"] - (v / w) * (np.cos(th) - np.cos(tp["yaw"]))
    return px, py, arc


# ------------------------------------------------------------------ 2D
def panel2d(ax, d, color, title, subtitle, ann):
    xs, ys = xy(d)
    ox, oy = d["obstacle_x"], d["obstacle_y"]
    overlap = d["geometric_overlap_observed"]
    ci = closest_index(d)

    ax.plot(*d["global_plan_trace"][0]["points"] and
            ([p[0] for p in d["global_plan_trace"][0]["points"]],
             [p[1] for p in d["global_plan_trace"][0]["points"]]),
            ls="--", color=C_PLAN, lw=1.1, zorder=1)

    # 예측 호. 장애물이 왜 여기에 놓였는지를 보여준다.
    ax_, ay_, arc = predicted_arc(d)
    ax.plot(ax_, ay_, ls=(0, (2, 2)), color=C_TRIG, lw=1.4, alpha=.9, zorder=3)

    # 장애물 실체 (반경 0.25 m)
    ax.add_patch(Circle((ox, oy), R_OBST, facecolor=C_OBST, alpha=.5,
                        edgecolor=C_OBST, lw=1.4, zorder=5))
    # 합산 footprint. 로봇 중심이 이 안에 들어오면 겹침이다.
    ax.add_patch(Circle((ox, oy), R_COMBINED, facecolor="none",
                        edgecolor=C_OFF if overlap else C_OBST,
                        ls=(0, (5, 4)), lw=1.2, alpha=.8, zorder=4))

    # 최소 clearance 시점의 로봇 실체 (반경 0.22 m)
    ax.add_patch(Circle((xs[ci], ys[ci]), R_ROBOT,
                        facecolor=C_OFF if overlap else C_PIT, alpha=.28,
                        edgecolor=color, lw=1.5, zorder=6))

    ax.plot(xs, ys, "-", color=color, lw=2.6, solid_capstyle="round", zorder=7)
    ax.plot(xs[0], ys[0], "o", color="#2f7d4f", ms=8, mew=0, zorder=9)
    ax.plot(*d["goal"], "*", color="#1f5fd0", ms=15, mew=0, zorder=9)

    tp = d["trigger_pose"]
    ax.plot(tp["x"], tp["y"], "v", color=C_TRIG, ms=9, mew=0, zorder=9)
    ax.annotate("delay injected", (tp["x"], tp["y"]), textcoords="offset points",
                xytext=(-4, 13), ha="center", fontsize=8.2, color=C_TRIG)

    ti = transfer_index(d)
    if ti is not None:
        ax.plot(xs[ti], ys[ti], "s", color=C_PIT, ms=8, mew=1.5, mec="white", zorder=10)
        ax.annotate(f"transfer to DWB, {d['handoff_latency_ms']:.0f} ms",
                    (xs[ti], ys[ti]), textcoords="offset points", xytext=ann,
                    fontsize=8.2, color=C_PIT, weight="bold",
                    arrowprops=dict(arrowstyle="-", color=C_PIT, lw=.9, shrinkA=0, shrinkB=3))

    clr = d["min_geometric_clearance_m"]
    ax.text(.035, .055, ("overlap  " if overlap else "clear  ") + f"{clr:+.2f} m",
            transform=ax.transAxes, fontsize=10.5, weight="bold",
            color=C_OFF if overlap else C_PIT)
    ax.text(.035, .935, subtitle, transform=ax.transAxes, fontsize=8.2, color="#5b6673")
    ax.text(.97, .055, f"predicted arc {arc:.2f} m", transform=ax.transAxes,
            fontsize=7.8, color=C_TRIG, ha="right")

    ax.set_title(title, fontsize=12.5, weight="bold", color=color, pad=8)
    ax.set_xlabel("x [m]", fontsize=9, labelpad=2)
    ax.set_aspect("equal")
    ax.grid(True, lw=.5, color=C_GRID, zorder=0)
    ax.tick_params(labelsize=8.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# ------------------------------------------------------------------ 3D
def cylinder(ax, cx, cy, r, h, color, alpha, z0=0.0, n=48, top=True):
    t = np.linspace(0, 2 * np.pi, n)
    X, Y = cx + r * np.cos(t), cy + r * np.sin(t)
    ax.plot_surface(np.vstack([X, X]), np.vstack([Y, Y]),
                    np.vstack([np.full(n, z0), np.full(n, z0 + h)]),
                    color=color, alpha=alpha, linewidth=0, shade=True)
    if top:
        tt, rr = np.meshgrid(t, np.linspace(0, r, 6))
        ax.plot_surface(cx + rr * np.cos(tt), cy + rr * np.sin(tt),
                        np.full_like(tt, z0 + h), color=color,
                        alpha=min(1, alpha + .15), linewidth=0, shade=True)


def _rot(px, py, yaw, cx, cy):
    c, sn = math.cos(yaw), math.sin(yaw)
    return cx + px * c - py * sn, cy + px * sn + py * c


def _round_rect(L, W, r, n=9):
    """모서리를 둥글린 직사각형 윤곽. TurtleBot3 waffle plate 근사."""
    hx, hy = L / 2 - r, W / 2 - r
    pts = []
    for cx_, cy_, a0 in ((hx, hy, 0), (-hx, hy, np.pi / 2),
                         (-hx, -hy, np.pi), (hx, -hy, 3 * np.pi / 2)):
        t = np.linspace(a0, a0 + np.pi / 2, n)
        pts += list(zip(cx_ + r * np.cos(t), cy_ + r * np.sin(t)))
    pts.append(pts[0])
    return np.array(pts)


def _prism(ax, outline, z0, h, yaw, cx, cy, color, alpha):
    X, Y = _rot(outline[:, 0], outline[:, 1], yaw, cx, cy)
    n = len(X)
    ax.plot_surface(np.vstack([X, X]), np.vstack([Y, Y]),
                    np.vstack([np.full(n, z0), np.full(n, z0 + h)]),
                    color=color, alpha=alpha, linewidth=0, shade=True)
    for z in (z0, z0 + h):
        ax.plot_surface(np.vstack([X, np.full(n, cx)]), np.vstack([Y, np.full(n, cy)]),
                        np.full((2, n), z), color=color, alpha=alpha, linewidth=0, shade=True)


def _wheel(ax, cx, cy, yaw, side, color, alpha, n=26):
    """차축이 수평인 바퀴. side = +1 왼쪽, -1 오른쪽."""
    R, Wd, off = .033, .018, .080
    t = np.linspace(0, 2 * np.pi, n)
    for u in (-Wd / 2, Wd / 2):
        px = np.zeros(n) + 0.0
        py = np.full(n, side * off + u)
        pz = R + R * np.sin(t)
        qx = R * np.cos(t)
        X, Y = _rot(qx, py, yaw, cx, cy)
        ax.plot(X, Y, pz, color=color, lw=1.0, alpha=alpha)
    for k in range(n - 1):
        qx = np.array([R * np.cos(t[k]), R * np.cos(t[k + 1])])
        pz = np.array([R + R * np.sin(t[k]), R + R * np.sin(t[k + 1])])
        Y0 = np.full(2, side * off - Wd / 2)
        Y1 = np.full(2, side * off + Wd / 2)
        X0, YY0 = _rot(qx, Y0, yaw, cx, cy)
        X1, YY1 = _rot(qx, Y1, yaw, cx, cy)
        ax.plot_surface(np.vstack([X0, X1]), np.vstack([YY0, YY1]),
                        np.vstack([pz, pz]), color=color, alpha=alpha,
                        linewidth=0, shade=True)


def turtlebot(ax, cx, cy, yaw, color, alpha=.95, dark="#1b2733"):
    """TurtleBot3 Burger 근사. waffle plate 3장 · 스탠드오프 · 바퀴 2개 · LDS.

    실측 치수 기준. 발자국 178 x 138 mm, 높이 192 mm, 바퀴 지름 66 mm.
    충돌 반경 0.22 m 는 이 몸체를 감싸는 원이다.
    """
    plate = _round_rect(.178, .138, .022)
    for z in (.052, .105, .150):
        _prism(ax, plate, z, .006, yaw, cx, cy, color, alpha)
    # 플레이트 사이 스탠드오프
    for sx, sy in ((.062, .048), (.062, -.048), (-.062, .048), (-.062, -.048)):
        X, Y = _rot(np.array([sx]), np.array([sy]), yaw, cx, cy)
        ax.plot([X[0], X[0]], [Y[0], Y[0]], [.058, .150], color=dark, lw=1.6, alpha=alpha)
    _wheel(ax, cx, cy, yaw, +1, dark, alpha)
    _wheel(ax, cx, cy, yaw, -1, dark, alpha)
    # LDS-01 라이다
    cylinder(ax, cx, cy, .0375, .040, dark, alpha, z0=.156)


def ground_disc(ax, cx, cy, r, z, color, alpha, n=64):
    """지면에 깔리는 원판. 접촉 판정이 원통 몸통이 아니라 바닥에서 읽히게 한다."""
    t = np.linspace(0, 2 * np.pi, n)
    rr, tt = np.meshgrid(np.linspace(0, r, 2), t)
    ax.plot_surface(cx + rr * np.cos(tt), cy + rr * np.sin(tt),
                    np.full_like(rr, z), color=color, alpha=alpha,
                    linewidth=0, shade=False)


def ground_ring(ax, cx, cy, r, z, color, lw, ls="-", alpha=1.0):
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(cx + r * np.cos(t), cy + r * np.sin(t), np.full_like(t, z),
            color=color, lw=lw, ls=ls, alpha=alpha, zorder=5)


def panel3d(ax, d, color, title):
    xs, ys = xy(d)
    ox, oy = d["obstacle_x"], d["obstacle_y"]
    overlap = d["geometric_overlap_observed"]
    ci = closest_index(d)

    gx, gy = np.linspace(-2.15, -0.3, 2), np.linspace(-1.35, 0.05, 2)
    GX, GY = np.meshgrid(gx, gy)
    # matplotlib 3D 는 z-buffer 가 없다. 가림은 오직 그리는 순서로 정해진다.
    # 그래서 판정 기하(발자국 원)를 맨 마지막에 그려 장애물 몸통이 지우지 못하게 한다.
    ax.plot_surface(GX, GY, np.full_like(GX, -0.004), color="#eef1f5", alpha=.5,
                    linewidth=0, shade=False)

    ground_disc(ax, ox, oy, R_OBST, .002, C_OBST, .26)
    ground_disc(ax, xs[ci], ys[ci], R_ROBOT, .004, color, .20)

    # global plan 과 예측 호는 여기 그리지 않는다. 둘 다 장애물을 관통해서 지나가
    # 로봇이 장애물 안으로 들어간 것처럼 읽힌다. 그 둘의 서사는 2D 평면도가 맡는다.
    ax.plot(xs, ys, .010, "-", color=color, lw=2.8)

    # 장애물 몸통. 높이는 예시일 뿐이고 판정에는 반지름만 들어간다.
    cylinder(ax, ox, oy, R_OBST, .17, C_OBST, .17)

    yaw_at = lambda i: math.atan2(ys[min(i + 1, len(ys) - 1)] - ys[max(i - 1, 0)],
                                  xs[min(i + 1, len(xs) - 1)] - xs[max(i - 1, 0)])
    turtlebot(ax, xs[0], ys[0], yaw_at(0), "#aab5c1", .40)
    turtlebot(ax, xs[ci], ys[ci], yaw_at(ci), C_ROBOT, .92)

    # 판정 기하는 항상 맨 위에 남는다. 두 발자국 원이 겹치면 collision 이다.
    ground_ring(ax, ox, oy, R_OBST, .016, C_OBST, 1.4)
    ground_ring(ax, xs[ci], ys[ci], R_ROBOT, .016, color, 1.6)

    # 떨어져 있을 때는 두 원 사이의 실제 공기 틈을 재어 보인다.
    gap = d["min_geometric_clearance_m"]
    if gap > 0:
        vx, vy = ox - xs[ci], oy - ys[ci]
        L = math.hypot(vx, vy)
        ux, uy = vx / L, vy / L
        p0 = (xs[ci] + ux * R_ROBOT, ys[ci] + uy * R_ROBOT)
        p1 = (ox - ux * R_OBST, oy - uy * R_OBST)
        # 치수선은 궤적과 색·굵기를 반드시 달리한다. 같게 두면 로봇이 계속 달려가
        # 장애물에 닿는 것처럼 읽힌다.
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ox_, oy_ = -uy * .13, ux * .13          # 중심선에서 살짝 띄운다
        q0 = (p0[0] + ox_, p0[1] + oy_)
        q1 = (p1[0] + ox_, p1[1] + oy_)
        ax.plot([q0[0], q1[0]], [q0[1], q1[1]], [.018, .018],
                color=C_DIM, lw=1.2)
        for q, base in ((q0, p0), (q1, p1)):
            ax.plot([base[0], q[0] + ox_ * .35], [base[1], q[1] + oy_ * .35],
                    [.018, .018], color=C_DIM, lw=.8, alpha=.8)
            ax.plot([q[0] - uy * .038, q[0] + uy * .038],
                    [q[1] + ux * .038, q[1] - ux * .038], [.018, .018],
                    color=C_DIM, lw=1.4)
        ax.text(mx + ox_ * 2.1, my + oy_ * 2.1, .02, f"{gap:+.2f} m",
                color=C_DIM, fontsize=9.5, weight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=.22", fc="white", ec="none", alpha=.85))

    verdict = ("footprints overlap  " if overlap else "footprints clear  ")         + f"{d['min_geometric_clearance_m']:+.2f} m"
    ax.set_title(f"{title}   ·   {verdict}", fontsize=11.5, weight="bold",
                 color=C_OFF if overlap else C_PIT, y=.94)
    ax.set_xlim(-2.15, -0.3); ax.set_ylim(-1.35, 0.05); ax.set_zlim(0, .30)
    ax.set_box_aspect((1.8, 1.35, .30), zoom=1.34)
    ax.view_init(elev=30, azim=-74)
    ax.set_xlabel("x [m]", fontsize=8, labelpad=-6)
    ax.set_ylabel("y [m]", fontsize=8, labelpad=-6)
    ax.set_zticks([])
    ax.tick_params(labelsize=7, pad=-3)
    ax.grid(False)
    ax.zaxis.line.set_lw(0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(False)


def sub(d, mode):
    if mode == "independent":
        return f"{d['stale_final_commands']} stale commands admitted, no transfer"
    return (f"{d['isolation_violations']} isolation violations, "
            f"{d['stale_final_commands']} stale admitted")


def legend_2d(fig):
    L = plt.Line2D
    h = [L([], [], ls="--", color=C_PLAN, lw=1.2, label="global plan"),
         L([], [], ls=(0, (2, 2)), color=C_TRIG, lw=1.4,
           label="predicted arc from the queued command"),
         L([], [], marker="o", color="#2f7d4f", ls="", ms=7, label="start"),
         L([], [], marker="*", color="#1f5fd0", ls="", ms=12, label="goal"),
         L([], [], marker="v", color=C_TRIG, ls="", ms=8, label="fault injected, +121 ms hold"),
         L([], [], marker="o", color=C_OBST, ls="", ms=9, alpha=.6,
           label=f"obstacle, r = {R_OBST} m"),
         L([], [], marker="o", color=C_ROBOT, ls="", ms=8, alpha=.5,
           label=f"robot at min clearance, r = {R_ROBOT} m"),
         L([], [], ls=(0, (5, 4)), color=C_OBST, lw=1.2,
           label=f"combined footprint, {R_COMBINED} m")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False,
               fontsize=8.2, bbox_to_anchor=(.5, .004))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rmw", default="rmw_fastrtps_cpp")
    ap.add_argument("--policy", default="drl_vo")
    ap.add_argument("--seed", type=int, default=30)
    ap.add_argument("--view", default="both", choices=["2d", "3d", "both"])
    ap.add_argument("--out", default="c7_trajectory_pair")
    a = ap.parse_args()

    off = load(a.rmw, a.policy, "independent", a.seed)
    pit = load(a.rmw, a.policy, "pit", a.seed)
    label = {"drl": "DRL", "drl_vo": "DRL-VO"}[a.policy]
    backend = {"rmw_fastrtps_cpp": "Fast DDS", "rmw_cyclonedds_cpp": "Cyclone DDS"}[a.rmw]
    head = (f"Same seed, same policy, same fault, same spawn rule  —  "
            f"{label} on {backend}, held-out seed {a.seed}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if a.view in ("2d", "both"):
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.3), dpi=200, sharex=True, sharey=True)
        fig.patch.set_facecolor("white")
        panel2d(axes[0], off, C_OFF, "Profile-off", sub(off, "independent"), (0, 0))
        panel2d(axes[1], pit, C_PIT, "PIT", sub(pit, "pit"), (10, -32))
        axes[0].set_ylabel("y [m]", fontsize=9)
        for ax in axes:
            ax.set_xlim(-2.2, -0.2); ax.set_ylim(-1.5, 0.2)
        fig.suptitle(head, fontsize=10.8, y=.985)
        legend_2d(fig)
        fig.tight_layout(rect=(0, .175, 1, .955))
        for ext in ("png", "pdf"):
            fig.savefig(out.with_suffix("." + ext), facecolor="white", bbox_inches="tight")
        print(f"saved {out}.png / .pdf")

    if a.view in ("3d", "both"):
        f3 = plt.figure(figsize=(10.6, 4.0), dpi=200)
        f3.patch.set_facecolor("white")
        a1 = f3.add_subplot(121, projection="3d")
        a2 = f3.add_subplot(122, projection="3d")
        panel3d(a1, off, C_OFF, "Profile-off")
        panel3d(a2, pit, C_PIT, "PIT")
        f3.suptitle(head, fontsize=10.5, y=.985)
        f3.text(.5, .056, "TurtleBot3 Burger to scale (178 x 138 x 192 mm, 0.22 m collision radius). "
                          "Rings on the floor are the two footprints, r = 0.22 m and 0.25 m.",
                ha="center", fontsize=8.2, color="#5b6673")
        f3.text(.5, .020, "A run counts as a collision when the two rings overlap, that is when the centres "
                          "close to within 0.47 m. Obstacle height is illustrative; only the radius counts.",
                ha="center", fontsize=8.2, color="#5b6673")
        for k, axk in enumerate((a1, a2)):
            axk.set_position([-.015 + .505 * k, .015, .525, .94])
        o3 = out.with_name(out.name + "_3d")
        for ext in ("png", "pdf"):
            f3.savefig(o3.with_suffix("." + ext), facecolor="white", bbox_inches="tight")
        print(f"saved {o3}.png / .pdf")

    for m, d in (("Profile-off", off), ("PIT", pit)):
        _, _, arc = predicted_arc(d)
        q = d["trigger_pose"]["queued_command"]
        brg = math.degrees(math.atan2(d["obstacle_y"] - d["trigger_pose"]["y"],
                                      d["obstacle_x"] - d["trigger_pose"]["x"])
                           - d["trigger_pose"]["yaw"])
        print(f"  {m:12s} clearance {d['min_geometric_clearance_m']:+.3f} m  "
              f"overlap={d['geometric_overlap_observed']}  stale={d['stale_final_commands']}  "
              f"arc={arc:.3f} m  queued w={q['angular_z']:+.4f}  bearing={brg:+.1f}deg")


if __name__ == "__main__":
    main()
