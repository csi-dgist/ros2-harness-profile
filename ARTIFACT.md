# Artifact Evaluation guide

Artifact for **Harness Engineering for Physical AI: Robot Middleware Is the Harness Layer**,
ACM/IFIP Middleware 2026, Big Ideas track, paper #256.

Repository <https://github.com/csi-dgist/ros2-harness-profile> · project page
<https://csi-dgist.github.io/ros2-harness-profile/>

## Requested badges

**Artifacts Available** and **Artifacts Functional**.

Everything below runs on a clean machine with Python 3.10 or newer, with no ROS 2 installation and
no simulator. The 160 audited run records of the case study in Section 5 ship here, and one command
re-derives from them every value the paper prints, together with a few the paper leaves out.
Re-running the trials themselves needs a ROS 2 Jazzy machine and is documented in
[README](README.md) sections 2 and 3.

## One command

```bash
bash ae_verify.sh
```

It creates a virtual environment, installs the two analysis dependencies, and runs the seven checks
below. The last line is `ALL CHECKS PASSED` and the exit status is 0. It takes about a minute after
the two packages download.

In a container instead, which is how these instructions were tested on a clean image:

```bash
docker build -t harness-profile-ae .
docker run --rm harness-profile-ae
```

| # | Check | What it shows |
|---|---|---|
| 1 | `sha256sum` of the deployed Profile | the Profile this repository ships is the one every run used, `625b6649…78d`, the hash the paper prints |
| 2 | `analysis/quick_verify.py` | the 16 per-cell values in the committed statistics are recomputed from the 160 run records |
| 3 | `analysis/derive_paper_numbers.py` | every value the paper prints, each beside the value recomputed from the records, plus the context values listed further down |
| 4 | `scripts/compile_harness_profile.py` | the Profile compiler runs without ROS 2 and reproduces a committed compiled file byte for byte |
| 5 | the three plotting and payload scripts | the trajectory pair, the 40-run overlay, and the project-page animation payload |
| 6 | `python3 -m unittest discover -s tests` | the core gate and training unit tests |
| 7 | `scripts/capture_environment.py` | the environment record, for a report. It is fail-closed for the simulation path, so without a sourced ROS 2 Jazzy it reports `complete: false`, which does not fail the run |

## Claims and where to check them

The run records call the arm without a Profile `independent` and the arm with one `pit`, after the
three mechanisms the paper names.

| Paper | Claim | Command | Expected |
|---|---|---|---|
| Table 3, per cell | with the Profile off, collisions 8/10, 10/10, 9/10 and 9/10 and stale commands 78, 145, 80 and 144 across the four backend and policy cells, and 0 for both measures with it on | `python3 analysis/quick_verify.py results/public_policy_collision_c7_locked_heldout_n10_v1` | one line per cell, `coll` and `staleCmds` matching the table |
| Table 3, All row | collisions 36/40 → 0/40, stale commands 447 → 0 | same command | `fault condition collisions   Profile-off 36/40   PIT 0/40` |
| §5 transfer | every fault run with a Profile transferred to DWB and received a fresh DWB command, and no normal run transferred in either arm | `python3 analysis/derive_paper_numbers.py` | `PIT 40/40`, `PIT fresh DWB command 40/40`, and `0/80 over both arms` |
| §5 decision cost | a median of 146 µs per sample against the 50 ms control period, no slower than the 155 µs the same node spends forwarding a command without enforcement | same command | `146.4` and `155.4` |
| §5 hand-off | the hand-off to DWB took a median of 302 ms | same command | `302.39` |
| §5 portability | one compiled Profile, a single SHA-256 across all 160 runs, carried the contract across both policies and both backends, and no application or controller source changed | `sha256sum generated/nav2_public_learned_collision_predictive_freeze.yaml` and the same script | `625b6649…78d`, one hash in the set, and `([0], [0])` for the two source-change fields |
| Listing 1 | the Harness Profile of the case study | `cat profiles/semantic_nav_command.yaml` | the fields of the listing. The paper abridges it, and this file adds the transport budget, the safe-value replacement, and the backend capabilities the contract requires |
| §4 compile-time refusal | a capability the backend does not declare fails at compile time | `python3 scripts/compile_harness_profile.py --semantic-profile profiles/semantic_nav_command.yaml --binding bindings/nav2_twist_public_learned_dwb.yaml --capability capabilities/rmw_cyclonedds_cpp.json --output out/compiled_cyclone.yaml` | a compiled file whose `unsupported_required_capabilities` is empty for both shipped backends |
| §2 running example | the late burst, the obstacle strike with the Profile off, the clean pass with it on. The paper prints no figure of it, the project page shows it | `python3 analysis/plot_c7_trajectory_pair.py --out out/fig` | `Profile-off  clearance -0.308 m  overlap=True  stale=15` and `PIT  clearance +0.220 m  overlap=False  stale=0` |
| project page animation | the replay the page plays | `python3 analysis/make_replay_payload.py --out out/replay.json` | `stale_admitted=0 blocked=2 transfer=1` for the arm with the Profile on |

## Values beyond the paper

`derive_paper_numbers.py` also derives these from the same 160 records and labels them `beyond the
paper`, because the paper does not print them. They are here for a reviewer who wants more than the
medians.

- p99 of the per-sample decision, 242 µs with the Profile on against 219 µs with it off.
- the decision cost by backend, 168 to 170 µs on Fast DDS against 109 to 110 µs on Cyclone DDS.
- DWB producing its first local plan after activation, a median of 276 ms, which is most of the
  302 ms hand-off, and the hand-off p95 of 371 ms with per-cell medians between 294 and 331 ms.
- the equal-weighted collision-risk reduction, 0.900 with a stratified bootstrap 95% CI of 0.800 to
  0.975, and the minimum-clearance gain of 0.33 m (0.29 to 0.36), both read from the committed
  `c7_locked_statistics.json`.
- a paired McNemar exact two-sided p of at most 0.008 in every cell, recomputed from the discordant
  pairs.
- the Wilson 95% upper bound of 27.8% for the 0 of 10 collisions each cell observed with the Profile
  on, which is the bound that keeps a per-cell zero from being read as a guarantee.
- all 160 runs ended with the goal reached or the horizon completed.

## Requirements

- Python 3.10 or newer. Tested on Python 3.12 with numpy 1.26 and 2.5 and with matplotlib 3.6
  and 3.11, and inside `python:3.12-slim`.
- About 60 MB of disk for the checkout, of which 18 MB is the run records, plus the two Python
  packages.
- No GPU, no network access after the clone, no ROS 2, no simulator.
- Sections 2 and 3 of the README, which re-run trials, need Ubuntu 24.04 with ROS 2 Jazzy, Nav2,
  TurtleBot3 Gazebo, and the two upstream checkpoints. Those checkpoints are GPL-3.0 upstream and
  are fetched by `scripts/setup_drlvo_jazzy.sh` rather than redistributed here.

## What is in the repository

| Path | Contents |
|---|---|
| `pit_core/`, `nodes/`, `adapters/`, `fallbacks/` | the gate, the candidate bridge, the Nav2 adapter, and the fallback selection |
| `profiles/`, `bindings/`, `capabilities/`, `generated/` | the semantic Profile of Listing 1, the per-deployment binding, the backend capability files, and the compiled Profiles, including the one all 160 runs used |
| `scripts/` | the Profile compiler, the trial runners, the locked matrix launcher, the environment capture |
| `analysis/` | `quick_verify.py`, `derive_paper_numbers.py`, the analyser that produced the committed statistics, the figure scripts, the replay payload builder |
| `results/public_policy_collision_c7_locked_heldout_n10_v1/` | the 160 audited run records, the manifest, the committed statistics, the locked report |
| `tests/` | unit tests that need no ROS 2 |
| `docs/` | the project page |

## What is not in the repository

- The 160 gzip raw observation traces, 750 MB, stay with the authors and are available on request.
  Each run record already carries the per-sample decisions in its `raw_events` array, which is what
  the verification path reads, and each model sidecar carries the trace's SHA-256 so a copy received
  later can be checked against this repository.
- The DRL and DRL-VO checkpoints, which are GPL-3.0 upstream. Their SHA-256 values are pinned in the
  README and repeated in every run record.
- Two test modules, `tests/test_rebuttal_report.py` and `tests/test_trace_validation.py`, cover
  tooling from the authors' experiment tree that is not part of this release. They skip with a
  printed reason instead of failing.

## Notes

- **Normal-condition collisions.** Table 3 reports the fault runs. In the 40 normal runs without a
  Profile there is one collision, in the Fast DDS and DRL-VO cell, and none in the 40 normal runs
  with one. `quick_verify.py` prints every cell, so this is visible there. The paper's
  normal-condition claim is about transfers, which are 0 in both arms.
- **Where enforcement sits.** Every run record carries
  `middleware_enforcement_boundary = post_collision_monitor_pre_actuator`, and
  `collision_monitor_preserved = true`, so the harness is added to the path the paper describes
  rather than replacing the existing safety check.
- **Percentiles.** `derive_paper_numbers.py` uses linear interpolation between order statistics, and
  the per-sample cost is the median over runs of each run's own percentile, as the paper reports it.
- **Questions.** The authors answer HotCRP questions through the evaluation period.
