#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C7 locked held-out 짝의 기록을 웹 재생용 JSON 으로 뽑는다.

프로젝트 페이지의 replay 위젯이 이 payload 를 그대로 읽어 궤적과 이벤트를 재생한다.
시뮬레이터를 다시 돌리지 않는다. **잠긴 결과 파일을 읽기만 한다.**

각 arm 의 t=0 은 자기 navigation_feedback_trace 의 첫 표본이다.
좌표·시각·판정은 어떤 것도 보정하지 않는다. 그림과 같은 원본을 쓴다.

실행.
  python make_replay_payload.py --out replay.json
"""

import argparse
import json
import math
from pathlib import Path

R_ROBOT = 0.22
R_OBST = 0.25

RESULTS = (Path(__file__).resolve().parents[1] / "results"
           / "public_policy_collision_c7_locked_heldout_n10_v1")


def load(rmw, policy, mode, seed):
    f = RESULTS / f"{rmw}__{policy}__{mode}__crossing__burst_late__{seed}.json"
    return json.loads(f.read_text(encoding="utf-8"))


def build_arm(d, label, kind):
    fb = d["navigation_feedback_trace"]
    t0 = fb[0]["wall_time_ns"]
    rel = lambda ns: round((ns - t0) / 1e9, 3)

    path = [[rel(p["wall_time_ns"]), round(p["x"], 4), round(p["y"], 4)] for p in fb]
    ox, oy = d["obstacle_x"], d["obstacle_y"]

    # 장애물 등장 시각. clearance 계산이 시작되는 첫 표본이 활성화 시점이다.
    ct = d.get("clearance_trace") or []
    t_obst = rel(ct[0]["wall_time_ns"]) if ct else None

    # fault 주입 위치. trigger_pose 에 시각이 없으므로 가장 가까운 궤적 표본의 시각을 쓴다.
    tp = d["trigger_pose"]
    ti = min(range(len(fb)), key=lambda i: (fb[i]["x"] - tp["x"]) ** 2 + (fb[i]["y"] - tp["y"]) ** 2)
    t_fault = rel(fb[ti]["wall_time_ns"])

    events = []
    for e in d.get("raw_events") or []:
        t = rel(e["observed_ns"])
        if e.get("stale") and e.get("admit"):
            events.append({"t": t, "k": "stale_admitted"})
        elif e.get("stale") and not e.get("admit"):
            events.append({"t": t, "k": "stale_blocked"})
        if e.get("transfer"):
            events.append({"t": t, "k": "transfer"})
    events.sort(key=lambda e: e["t"])

    clearance = [[rel(c["wall_time_ns"]), round(c["center_distance_m"] - R_ROBOT - R_OBST, 4)]
                 for c in ct]

    # 최소 접근 시각.
    ci = min(range(len(fb)), key=lambda i: (fb[i]["x"] - ox) ** 2 + (fb[i]["y"] - oy) ** 2)
    t_closest = rel(fb[ci]["wall_time_ns"])

    # 접촉 시각. 두 발자국이 처음 맞닿는 순간이다. 장애물은 실체가 있는 원기둥이므로
    # 로봇은 여기서 멈춘다. 이후의 pose 는 동결 명령의 추측항법이라 재생하지 않는다.
    t_contact = None
    for i, q in enumerate(fb):
        t = rel(q["wall_time_ns"])
        if t_obst is None or t < t_obst:
            continue
        if math.hypot(q["x"] - ox, q["y"] - oy) <= R_ROBOT + R_OBST:
            t_contact = t
            break

    gp = (d.get("global_plan_trace") or [{}])[0].get("points") or []
    goal = [round(gp[-1][0], 4), round(gp[-1][1], 4)] if gp else None
    plan = [[round(x, 3), round(y, 3)] for x, y in gp[::3]] if gp else []

    return {
        "id": kind,
        "plan": plan,
        "label": label,
        "obstacle": {"x": round(ox, 4), "y": round(oy, 4), "t": t_obst},
        "trigger": {"x": round(tp["x"], 4), "y": round(tp["y"], 4), "t": t_fault},
        "goal": goal,
        "path": path,
        "events": events,
        "clearance": clearance,
        "tClosest": t_closest,
        "tContact": t_contact,
        "duration": path[-1][0],
        "verdict": {
            "overlap": bool(d["geometric_overlap_observed"]),
            "clearance": round(d["min_geometric_clearance_m"], 4),
            "staleAdmitted": int(d["stale_final_commands"]),
            "transfer": bool(d.get("transfer_events")),
            "handoffMs": round(d["handoff_latency_ms"], 1) if d.get("handoff_latency_ms") else None,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rmw", default="rmw_fastrtps_cpp")
    ap.add_argument("--policy", default="drl_vo")
    ap.add_argument("--seed", type=int, default=30)
    ap.add_argument("--out", default="replay.json")
    a = ap.parse_args()

    off = load(a.rmw, a.policy, "independent", a.seed)
    pit = load(a.rmw, a.policy, "pit", a.seed)

    payload = {
        "meta": {
            "policy": {"drl": "DRL", "drl_vo": "DRL-VO"}[a.policy],
            "dds": {"rmw_fastrtps_cpp": "Fast DDS",
                    "rmw_cyclonedds_cpp": "Cyclone DDS"}[a.rmw],
            "seed": a.seed,
            "rRobot": R_ROBOT,
            "rObst": R_OBST,
        },
        "arms": [build_arm(off, "Profile-off", "off"),
                 build_arm(pit, "PIT", "pit")],
    }

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {out}  {out.stat().st_size / 1024:.1f} KB")
    for arm in payload["arms"]:
        v = arm["verdict"]
        n_stale = sum(1 for e in arm["events"] if e["k"] == "stale_admitted")
        n_block = sum(1 for e in arm["events"] if e["k"] == "stale_blocked")
        n_xfer = sum(1 for e in arm["events"] if e["k"] == "transfer")
        print(f"  {arm['label']:12s} dur={arm['duration']:.2f}s  obst_t={arm['obstacle']['t']}  "
              f"fault_t={arm['trigger']['t']}  closest_t={arm['tClosest']}")
        print(f"               events stale_admitted={n_stale} blocked={n_block} transfer={n_xfer}  "
              f"clearance={v['clearance']:+.3f} overlap={v['overlap']}")


if __name__ == "__main__":
    raise SystemExit(main())
