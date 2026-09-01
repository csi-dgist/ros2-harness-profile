# C7 locked held-out collision analysis

- Audit: **PASS** (160/160 rows)
- Profile SHA-256: `625b6649e1cf28e673dd4bf8df6af71bbdde55ebfe96457c033d9b96fd87078d`

## Fault collision and containment

| DDS | Policy | Independent collisions | PIT collisions | Risk difference | Paired bootstrap 95% CI | McNemar p |
|---|---|---:|---:|---:|---:|---:|
| rmw_fastrtps_cpp | drl | 8/10 | 0/10 | 0.800 | [0.500, 1.000] | 0.0078 |
| rmw_fastrtps_cpp | drl_vo | 10/10 | 0/10 | 1.000 | [1.000, 1.000] | 0.0020 |
| rmw_cyclonedds_cpp | drl | 9/10 | 0/10 | 0.900 | [0.700, 1.000] | 0.0039 |
| rmw_cyclonedds_cpp | drl_vo | 9/10 | 0/10 | 0.900 | [0.700, 1.000] | 0.0039 |

## Equal-strata pooled effect

- Fault collision risk reduction: **0.900** (95% stratified bootstrap CI 0.800–0.975).
- Collision difference-in-differences benefit: **0.875** (95% CI 0.775–0.975).
- Fault clearance benefit: **0.3255 m** (95% CI 0.2920–0.3575 m).

## Mechanism

- Independent/Fault stale admission: 40/40 runs, 447 commands.
- PIT/Fault: stale 0/40, Transfer 40/40, fresh DWB 40/40.
- Handoff latency: median 302.39 ms, p95 371.46 ms.
- Normal false Transfer: 0/80; Normal stale admission: 0/80.
