#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRLVO_WS="${DRLVO_WS:-$HOME/drl_vo_jazzy_ws}"
cd "$ROOT"
source /opt/ros/jazzy/setup.bash
source "$DRLVO_WS/install/setup.bash"
set -u
result_dir="$ROOT"/results/public_policy_collision_c7_locked_heldout_n10_v1
mkdir -p "${result_dir}"

exec python3 scripts/run_public_learned_task_impact.py \
  --result-dir "${result_dir}" \
  --weights-dir "$ROOT"/results/public_drl_models_v1 \
  --profile "$ROOT"/generated/nav2_public_learned_collision_predictive_freeze.yaml \
  --model-commit 6d734b6e0df77fd4c4faa4649ca0fcb3e69cf835 \
  --ros2-commit 93b771a8c365781af2df479557d45f9a76f68756 \
  --runs-per-cell 10 --seed-start 30 --seed 7129 \
  --domain-base 218 --retries 2 --wall-watchdog 150 \
  --goal-x -0.6 --goal-y -0.15 --post-trigger-horizon-s 5.0 \
  --rmws rmw_fastrtps_cpp,rmw_cyclonedds_cpp --policies drl,drl_vo \
  --modes independent,pit --scenes crossing --conditions normal,burst_late \
  > "${result_dir}/runner.log" 2>&1
