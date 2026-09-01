#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""잠긴 결과의 headline 수치를 원본 레코드에서 다시 계산해 대조한다.

`analyze_c7_collision.py` 는 fail-closed 전수 감사이고 raw observation trace
(160개, 750 MB) 의 SHA-256 까지 확인한다. 그 아카이브를 받지 않은 사람도
저장소만으로 수치를 확인할 수 있게 하는 것이 이 스크립트의 목적이다.

검사하는 것은 하나뿐이다. 공개된 `c7_locked_statistics.json` 의 셀별 수치가
같은 폴더의 160개 결과 레코드에서 실제로 재현되는가. raw trace 무결성, 소스 해시,
progress.log 완결성은 `analyze_c7_collision.py` 소관이며 여기서 검사하지 않는다.

실행.
  python3 analysis/quick_verify.py results/public_policy_collision_c7_locked_heldout_n10_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

FIELDS = ("n", "collisions", "stale_final_runs", "stale_final_commands", "transfer_runs")


def recompute(result_dir: Path) -> dict[str, dict]:
    cells: dict[str, list[dict]] = {}
    for path in sorted(result_dir.glob("*__*.json")):
        if path.name.endswith("__model.json"):
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        if "rmw_implementation" not in row:
            continue
        key = "|".join((row["rmw_implementation"], row["policy"], row["mode"], row["condition"]))
        cells.setdefault(key, []).append(row)

    out = {}
    for key, rows in cells.items():
        clear = [r["min_geometric_clearance_m"] for r in rows]
        out[key] = {
            "n": len(rows),
            "collisions": sum(1 for r in rows if r["geometric_overlap_observed"]),
            "stale_final_runs": sum(1 for r in rows if r["stale_final_commands"] > 0),
            "stale_final_commands": sum(r["stale_final_commands"] for r in rows),
            "transfer_runs": sum(1 for r in rows if r.get("transfer_events")),
            "min_clearance_mean_m": mean(clear),
            "min_clearance_min_m": min(clear),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir", type=Path)
    a = ap.parse_args()

    stats_path = a.result_dir / "c7_locked_statistics.json"
    if not stats_path.exists():
        print(f"published statistics not found: {stats_path}")
        return 2
    published = json.loads(stats_path.read_text(encoding="utf-8"))["cells"]
    mine = recompute(a.result_dir)

    print(f"recomputed {sum(c['n'] for c in mine.values())} runs "
          f"across {len(mine)} cells from the result records\n")

    bad = []
    for key in sorted(published):
        p, m = published[key], mine.get(key)
        if m is None:
            bad.append(f"{key}: missing from the result records")
            continue
        for f in FIELDS:
            if p[f] != m[f]:
                bad.append(f"{key}.{f}: published {p[f]} but recomputed {m[f]}")
        for f in ("min_clearance_mean_m", "min_clearance_min_m"):
            if abs(p[f] - m[f]) > 1e-9:
                bad.append(f"{key}.{f}: published {p[f]:.6f} but recomputed {m[f]:.6f}")

    hdr = f"{'cell':52s} {'n':>3} {'coll':>5} {'staleRuns':>10} {'staleCmds':>10} {'xfer':>5}"
    print(hdr)
    print("-" * len(hdr))
    for key in sorted(mine):
        c = mine[key]
        print(f"{key:52s} {c['n']:3d} {c['collisions']:5d} "
              f"{c['stale_final_runs']:10d} {c['stale_final_commands']:10d} {c['transfer_runs']:5d}")

    fault = {k: v for k, v in mine.items() if k.endswith("|burst_late")}
    off = sum(v["collisions"] for k, v in fault.items() if "|independent|" in k)
    pit = sum(v["collisions"] for k, v in fault.items() if "|pit|" in k)
    n_off = sum(v["n"] for k, v in fault.items() if "|independent|" in k)
    n_pit = sum(v["n"] for k, v in fault.items() if "|pit|" in k)
    print(f"\nfault condition collisions   Profile-off {off}/{n_off}   PIT {pit}/{n_pit}")

    print()
    if bad:
        print("MISMATCH against the published statistics:")
        for b in bad:
            print("  -", b)
        return 1
    print("OK. Every published cell value is reproduced by the result records.")
    print("This is a value check. The full integrity audit also verifies the raw")
    print("observation traces, which are 750 MB and are not distributed. See the")
    print("Raw observation traces section of the README.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
