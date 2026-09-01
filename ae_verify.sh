#!/usr/bin/env bash
# Verification path of the artifact. No ROS 2 and no simulator.
#
#   bash ae_verify.sh              # creates .venv, installs, runs every check
#   SKIP_VENV=1 bash ae_verify.sh  # current interpreter, as the Docker image does

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
RESULTS="results/public_policy_collision_c7_locked_heldout_n10_v1"
PROFILE="generated/nav2_public_learned_collision_predictive_freeze.yaml"
PROFILE_SHA256="625b6649e1cf28e673dd4bf8df6af71bbdde55ebfe96457c033d9b96fd87078d"

say() { printf '\n== %s\n' "$1"; }

if [ "${SKIP_VENV:-0}" != "1" ]; then
  say "environment"
  python3 -m venv .venv
  # shellcheck disable=SC1091
  . .venv/bin/activate
  python3 -m pip install --quiet --upgrade pip
  python3 -m pip install --quiet -r requirements-analysis.txt
fi
python3 --version

say "1. the deployed Profile, against the hash the paper prints"
echo "${PROFILE_SHA256}  ${PROFILE}" | sha256sum --check -

say "2. per-cell table, from the 160 run records"
python3 analysis/quick_verify.py "$RESULTS"

say "3. every reported number, from the same records"
python3 analysis/derive_paper_numbers.py "$RESULTS"

say "4. the Profile compiler, without ROS 2"
python3 scripts/compile_harness_profile.py \
  --semantic-profile profiles/semantic_nav_command.yaml \
  --binding bindings/nav2_twist_public_learned_dwb.yaml \
  --capability capabilities/rmw_fastrtps_cpp.json \
  --output out/compiled_fast.yaml >/dev/null
diff -q out/compiled_fast.yaml generated/nav2_public_learned_compiled_task.yaml \
  && echo "compiler output matches generated/nav2_public_learned_compiled_task.yaml"

say "5. figures and the replay payload"
python3 analysis/plot_c7_trajectory_pair.py --out out/fig >/dev/null
python3 analysis/plot_c7_all_runs.py --out out/allruns >/dev/null
python3 analysis/make_replay_payload.py --out out/replay.json >/dev/null
ls -1 out/

say "6. unit tests"
python3 -m unittest discover -s tests

say "7. environment record"
# Fail-closed for the simulation path, so it returns 2 without a sourced ROS 2 Jazzy.
if python3 scripts/capture_environment.py --output out/ENVIRONMENT.json >/dev/null; then
  echo "wrote out/ENVIRONMENT.json, ROS 2 Jazzy complete"
else
  echo "wrote out/ENVIRONMENT.json, complete=false without a sourced ROS 2"
fi

printf '\nALL CHECKS PASSED. Outputs are in out/.\n'
