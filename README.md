# ROS 2 Harness Profile

### ▶ [Read the project page](https://csi-dgist.github.io/ros2-harness-profile/)
### ▶ Artifact Evaluation reviewers: [ARTIFACT.md](ARTIFACT.md), or `bash ae_verify.sh` for everything at once

Trajectories, the replay of a matched pair, the Harness Profile, and every reported number in one place.

---

Middleware-boundary enforcement of **Projection, Isolation, and Transfer (PIT)** for learned
controllers in ROS 2, evaluated on Nav2 with two public learned navigation policies across two DDS
backends.

Companion artifact for *Harness Engineering for Physical AI: Robot Middleware Is the Harness Layer*,
ACM Middleware 2026 (Big Ideas). Paper: [arXiv:2606.09416](https://arxiv.org/abs/2606.09416) ·
Project page: [csi-dgist.github.io/ros2-harness-profile](https://csi-dgist.github.io/ros2-harness-profile/)

---

## What this is

A learned policy's output crosses control, computing, and communication at once. A **Harness Profile**
declares the output region, the inference budget, and the operating regime in one file. The PIT core
enforces that declaration at the ROS 2 interface, between the application graph and the DDS or Zenoh
transport.

The headline result. Under a *safe-but-late* fault, where the Collision Monitor has already approved a
command by value but it arrives stale at the actuator:

| | Profile-off | PIT |
|---|---:|---:|
| Collisions, fault condition | **36 / 40** | **0 / 40** |
| Runs admitting a stale final command | 40 / 40 | 0 / 40 |
| Stale commands admitted | 447 | 0 |
| Transfer to the verified fallback | 0 / 40 | 40 / 40 |
| False transfers, normal condition | 0 / 80 | 0 / 80 |
| Application and controller source changed | — | **0 LOC** |

Equal-weighted collision-risk reduction **0.900**, stratified bootstrap 95% CI **[0.800, 0.975]**.

### Where the enforcement runs

The gate is a graph participant on the typed ROS 2 path, after the Nav2 Collision Monitor and before
the actuator endpoint. It reads typed samples, source timestamps, QoS state, and lifecycle state, and
it admits, replaces, or withholds each sample. These are the surfaces the paper identifies as the
middleware's mediated abstractions, so an enforcement point placed on them sits at the middleware
layer by construction.

---

## Three ways to use this repository

Pick the one that matches how much time you have.

| | What it does | Needs | Time |
|---|---|---|---|
| **1. Verify** | Re-derive every published number, figure, and animation from the shipped run records | Python 3.10+ | minutes |
| **2. Smoke** | Run one matched pair end to end in Gazebo | ROS 2 Jazzy, Nav2, DRL-VO workspace | one pair |
| **3. Reproduce** | Re-run the full 160-trial locked matrix | as above, plus a long wall clock | hours |

---

## 1. Verify the published numbers

No ROS 2 and no simulator. The 160 audited run records ship with the repository.

```bash
git clone https://github.com/csi-dgist/ros2-harness-profile.git
cd ros2-harness-profile
python3 -m pip install -r requirements-analysis.txt
```

Recompute every published cell value from the 160 result records:

```bash
python3 analysis/quick_verify.py \
  results/public_policy_collision_c7_locked_heldout_n10_v1
```

It rebuilds the per-cell table from the records alone and diffs it against the committed
`c7_locked_statistics.json`, then prints the headline `36/40` against `0/40`. Any disagreement is
listed and the exit status is non-zero.

`analysis/analyze_c7_collision.py` is the analyser that produced the committed statistics, and its
SHA-256 is published in the paper. It ships here for inspection, and it runs once the raw observation
traces are in place. See **Raw observation traces** below.

Redraw the figures from the same records:

```bash
python3 analysis/plot_c7_trajectory_pair.py --out out/fig
```

`--rmw`, `--policy`, and `--seed` accept any cell, so every one of the 40 fault pairs can be redrawn.
`--view 2d` or `--view 3d` selects one panel set. `analysis/plot_c7_all_runs.py` draws all 40 fault-condition
runs per arm in one figure, each run shifted so its own obstacle sits at the origin.

Rebuild the project page animation payload:

```bash
python3 analysis/make_replay_payload.py --out out/replay.json
```

### Integrity

The run manifest carries SHA-256 for the trial runner, the parameter generator, the Harness Profile,
and the Nav2 parameter file.

```
Profile   625b6649e1cf28e673dd4bf8df6af71bbdde55ebfe96457c033d9b96fd87078d
Manifest  402c2d363e0625880025308db9dee0c7eed1daebdd317343e434ded86bbc9dc7
Analyser  1d318db8256c21209d26e1ce29ec38bd374fa78c10bfea8ff9f1ed6f73178292
```

Every run record carries `profile_sha256`, `adapter_sha256`, `gate_dependency_sha256`, and
`model_weight_sha256`. `manifest.json` carries the matrix revision, the factor grid, the scene
geometry, and the enforcement boundary.

### Raw observation traces

The 160 gzip raw observation traces are 750 MB and stay with the authors. They are available on
request. Every per-sample decision the mechanism rests on is already in each run's `raw_events` array
inside its result record, so `analysis/quick_verify.py` reproduces every reported value from this
checkout alone. Each model sidecar carries the `raw_trace_path` and `raw_trace_sha256` of its trace,
so a copy received later can be verified against this repository.

---

## 2. Run one matched pair

### Prerequisites

- Ubuntu 24.04 with **ROS 2 Jazzy**
- Nav2 and TurtleBot3 Gazebo packages
- Working GPU-less headless Gazebo is sufficient

### Set up the learned policies

The DRL and DRL-VO checkpoints are **not redistributed here**. They come from
[TempleRAIL/drl_vo_nav](https://github.com/TempleRAIL/drl_vo_nav), pinned to branch `humble` at commit
`93b771a8c365781af2df479557d45f9a76f68756`. The setup script clones that commit, applies the Jazzy
compatibility patch in `patches/drlvo_jazzy_std_msgs.patch`, and builds a workspace at
`~/drl_vo_jazzy_ws`.

```bash
bash scripts/setup_drlvo_jazzy.sh
```

It fails closed on a commit mismatch and records the build log under `results/drlvo_setup_v1/`. Set
`DRLVO_WS` if you put the workspace somewhere other than `~/drl_vo_jazzy_ws`. The trial runner carries an
absolute default for the built model inside that workspace, left as it was when the matrix ran so its
SHA-256 still matches the manifest. The documented command passes `--weights-dir` explicitly, so that
default is never used.

### Stage the checkpoints

The two frozen checkpoints live in a second upstream repository,
[TempleRAIL/drl_vo_nav_models](https://github.com/TempleRAIL/drl_vo_nav_models). The runner reads them
from `results/public_drl_models_v1/`, so copy them there and verify the hashes before running anything.

```bash
mkdir -p results/public_drl_models_v1
git clone --depth 1 https://github.com/TempleRAIL/drl_vo_nav_models.git upstream/drl_vo_nav_models
cp upstream/drl_vo_nav_models/drl_vo/src/model/drl.zip     results/public_drl_models_v1/
cp upstream/drl_vo_nav_models/drl_vo/src/model/drl_vo.zip  results/public_drl_models_v1/
sha256sum results/public_drl_models_v1/*.zip
```

The evaluated checkpoints are pinned by content:

```
89abf66f4f45239a63812f84f11078c9e6b2da5ca8ada533ca37a99f3ecb995c  drl.zip
78a6bc9918b092e2bd26782664248d9ce06a5498e7cd086260dbfe6548f99645  drl_vo.zip
```

Every run record repeats the checkpoint hash in `model_weight_sha256`, so a mismatch is visible in the
audit as well as at setup time. `drl.zip` is the policy trained without the velocity-obstacle
heading term and `drl_vo.zip` is the one trained with it. The official ROS 2 branch of `drl_vo_nav`
packages only the DRL-VO checkpoint, which is why both are taken from the models repository instead.

### Compile the Profile

`profiles/semantic_nav_command.yaml` states the meaning of the contract and compiles to a hash-pinned deployment file. A capability the chosen
backend does not declare fails here, at compile time.

```bash
python3 scripts/compile_harness_profile.py \
  --semantic-profile profiles/semantic_nav_command.yaml \
  --binding bindings/nav2_twist_public_learned_dwb.yaml \
  --capability capabilities/rmw_fastrtps_cpp.json \
  --output generated/nav2_public_learned_compiled.yaml
```

### Run the pair

```bash
source /opt/ros/jazzy/setup.bash
source ~/drl_vo_jazzy_ws/install/setup.bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python3 scripts/run_public_learned_task_impact.py \
  --result-dir out/smoke \
  --weights-dir results/public_drl_models_v1 \
  --profile generated/nav2_public_learned_collision_predictive_freeze.yaml \
  --runs-per-cell 1 --seed-start 30 --seed 7129 \
  --domain-base 218 --retries 2 --wall-watchdog 150 \
  --goal-x -0.6 --goal-y -0.15 --post-trigger-horizon-s 5.0 \
  --rmws rmw_fastrtps_cpp --policies drl_vo \
  --modes independent,pit --scenes crossing --conditions burst_late
```

That is the locked configuration restricted to one cell. It produces two records,
`rmw_fastrtps_cpp__drl_vo__independent__crossing__burst_late__30.json` and the `pit` counterpart, which
are the exact pair the project page animates.

Expect the Profile-off run to admit stale commands and overlap the obstacle, and the PIT run to block
them and transfer to DWB.

---

## 3. Reproduce the full matrix

```bash
bash scripts/launch_c7_locked_heldout_n10.sh
```

`2 DDS × 2 policies × 2 modes × 2 conditions × 10 held-out seeds = 160 trials`, seeds 30 to 39, fixed
after the scene was frozen. Each trial launches a full Gazebo and Nav2 stack, so budget accordingly and
run it under a terminal multiplexer. The runner writes `runner.log` and retries a trial at most twice
under a 150 s wall watchdog.

Then run the analyser from step 1 against your own result directory.

---

## Layout

```
pit_core/        vendor-neutral decision core. Imports neither rclpy nor any ROS message type.
adapters/        per-application adapters. Flatten a message into named numeric fields.
nodes/           candidate bridge and trace recorders.
fallbacks/       verified fallback behaviours.
profiles/        semantic Harness Profiles. Meaning only, no node or topic names.
bindings/        deployment bindings. Concrete topics, types, and endpoints.
capabilities/    per-RMW capability descriptors.
generated/       compiled, hash-pinned deployment files.
scripts/         setup, profile compilation, matrix runners.
analysis/        audit, statistics, figures, replay payload.
tests/           unit tests for the decision core and the trace validators.
results/         the locked held-out matrix, 160 audited run records.
patches/         upstream compatibility patches.
docs/            experiment design and reporting rules.
```

### The four layers

```
Semantic Harness Profile      declares meaning only
        ↓ compile
Vendor-neutral Contract IR    ObservedSample · ObservedStatus · Decision · Evidence
        ↓ backend capability check
Backend adapter, per RMW      normalise · enforce · return evidence
        ↓ deployment binding
ROS 2 application graph       topics, types, field paths, endpoints
```

The decision core reads only the Contract IR and returns the same decision for the same input. The
audit requires the same core SHA-256 across every result in the matrix, and fails otherwise.

---

## Tests

```bash
python3 -m unittest discover -s tests -v
python3 scripts/capture_environment.py --output out/ENVIRONMENT.json
```

The core tests need no ROS 2 installation. Two modules, `test_rebuttal_report.py` and
`test_trace_validation.py`, cover tooling from the experiment tree that is not part of this
release, and they skip with a printed reason.

---

## Scope and naming

- `AIController` is the Nav2 controller-plugin slot that carries the learned policy. In this matrix it
  is bound to `nav2py_drl_vo_controller::DrlVoController`, so a `selector_history` entry naming
  `AIController` means the learned policy held authority at that moment.
- **DRL** and **DRL-VO** are two public learned navigation policies sharing an observation and command
  interface. They share an architecture and differ in one reward term.
- **DWB** is the verified classical fallback, and it appears in `selector_history` exactly when Transfer
  fires. Regulated Pure Pursuit is registered in the parameter file and never selected here.
- Collision means **geometric footprint overlap** at a combined radius of 0.47 m, from a 0.22 m robot
  footprint and a 0.25 m obstacle.
- The reported minimum clearance comes from the clearance monitor. The pose trace in the navigation
  feedback stream samples separately at about 6 Hz, so the two streams can differ on the exact depth of
  an overlap while agreeing on the verdict.

---

## Citation

```bibtex
@inproceedings{lee2026harness,
  title     = {Harness Engineering for Physical {AI}:
               Robot Middleware Is the Harness Layer},
  author    = {Lee, Sanghoon and Chae, Jiyeong and Park, Kyung-Joon},
  booktitle = {Proceedings of the ACM/IFIP International Middleware
               Conference (Big Ideas Track)},
  year      = {2026},
  note      = {arXiv:2606.09416}
}
```

## License

**MIT** for the code in this repository. See [LICENSE](LICENSE).

The upstream `drl_vo_nav` and `drl_vo_nav_models` repositories are **GPL-3.0**. Neither their source
nor their checkpoints are redistributed here. `scripts/setup_drlvo_jazzy.sh` fetches the pinned commit
and applies `patches/drlvo_jazzy_std_msgs.patch`, which is a diff against that GPL-3.0 source and is
offered under the same terms. Anyone redistributing a build that includes the upstream source has to
meet GPL-3.0 obligations for that part.
