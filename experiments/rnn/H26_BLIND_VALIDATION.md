# Frozen h26 blind validation

## Protocol fixed before training seeds 1 and 2

Train seed 1 original MLP, seed 1 h26 with Actor learning-rate decay, seed 2 original MLP, then seed 2 h26 with decay, serially. Each run starts from scratch for 300,001 interactions. Use the exact seed-0 executable for each method, verified by SHA256. The GRU has 146 inputs, hidden size 26, and the same Adam and training settings as the completed seed-0 decay run. Its only scheduled change is Actor alpha from 0.001 to 0.0001 after the second curriculum update finishes at step 200,000. Keep the independent weight-decay term fixed.

The MLP executable is the original 146 → 64 → 64 → 4 RLtools baseline already used for seed 0. Its SHA256 is `42e6625e067f7e3992a1d93f4bbc9a8acf4ac2e37b48e0a640c0dd98158bc80d`. The GRU executable SHA256 is `b92f4a14a29875bbfb64f81f94a509f7ff1fa2132641a2242600d1cdc1d698b7`.

Evaluate all 31 saved checkpoints, from 0 through 300,000 in increments of 10,000, in fresh processes with ten episodes each, no action exploration noise, and the same `--seed` as training. The compiled evaluation RNG mapping stays fixed. Evaluate each job's checkpoints after its trainer exits and before starting the next trainer, keeping trainer resource use serial. Save each run's configuration, executable and hash, commands, process status, full logs, Actor and target checkpoints, trajectories, per-episode metrics and per-checkpoint metrics in its own directory. Stop the entire queue on a failed run, retain artifacts and report its cause. No retries or parameter changes are automatic.

## Comparison rule

Rank each seed at 300k by completion rate first. If completion is tied, use lower all-episode position cost, then higher mean return if position cost is tied. Completed-only position cost is reported alongside the unconditional cost, since it conditions on survival. All comparisons use the saved numerical values without selecting favorable rounding or tolerances after observing results.

A claim of exceeding MLP across these seeds requires no lower final completion in any seed, a nonlosing rank under this rule in every seed and at least one strict win. Any seed failure or tradeoff must be listed explicitly. Report post-200k worst and mean completion separately and disclose any deterioration even if the final ranking favors GRU. The original seed 0 was used to select this schedule. Seeds 1 and 2 are the validation seeds, so present both their individual results and the three-seed summary.

For all six method/seed combinations, report final completion, mean return, all-episode and completed-only position costs, first checkpoint with ten of ten episodes reaching 500 steps, and worst/mean completion across 210k through 300k. Export the individual seed values and the mean and sample standard deviation across the three seed-level values, with denominator n−1. If a run never reaches 10/10, keep its first-completion step missing and report the attainment count. If no episodes complete at 300k, keep completed-only position cost missing and report the count. Do not replace missing values with the training budget or zero.

Position cost retains the seed-0 definition. Average `20 * (x² + y² + z²)` from logged prior states per episode and then across the indicated group. Exclude the first transition, whose prior state is absent, and terminal transitions, whose ordinary reward is replaced. Full trajectory and other episode metrics remain available.

The runner freezes this protocol, binary hashes and source snapshots before starting the first trainer. Run `run_h26_blind_validation.py` with `--output`, `--mlp-exe`, `--gru-exe`, `--seed0-mlp` and `--seed0-gru`. Output must be a fresh directory. Checkpoints contain weights and cannot restore full optimizer, replay and RNG state. Any later restart must remain a separate run.

## Interrupted on 2026-11-11

Protocol and runner were committed as `c20f4e5` before the first new job. Frozen protocol SHA256 is `bf05f55f3af9ee10783d652bd58e5c8620d2bcdb5a07c12d55974b067e973c38`. Seed-1 MLP, seed-1 GRU and seed-2 MLP completed in order with exit code 0. Their training times were 580.16, 1729.12 and 542.83 seconds. All three have 31 evaluated checkpoints with ten episodes each and retain raw trajectories and checkpoint hashes.

| Seed | Method | Final completion | Mean return | All position cost | Completed position cost | First 10/10 step | Worst / mean completion after 200k |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | MLP | 90% | +138.20 | 1.4319 | 1.1597 | 240000 | 70% / 88% |
| 0 | GRU decay | 100% | +291.50 | 1.1291 | 1.1291 | 180000 | 80% / 98% |
| 1 | MLP | 100% | +9.40 | 1.1763 | 1.1763 | 240000 | 60% / 85% |
| 1 | GRU decay | 100% | -685.23 | 3.1486 | 3.1486 | 300000 | 0% / 48% |
| 2 | MLP | 90% | -76.38 | 1.9817 | 1.8543 | Not reached | 0% / 61% |

The seed-2 GRU trainer and serial runner were interrupted by a Windows power-off. System event 1074 records a power-off request at 12:52:16 Asia/Seoul, event 6006 records the event-log service stopping at 12:52:42, and the next system boot was at 13:04:25. Both process IDs are absent. The last logged Actor update is at 112,000 and the latest saved Actor/target checkpoint pair is at 110,000. The exact stop step and process exit codes were not persisted and are unavailable. This is a system interruption, with no logged numerical exception. The stale `running` status was corrected after process and system-event verification.

Artifacts are in `build2026/h26-blind-validation-20261111/results/`. `INTERRUPTED.md`, `interruption.json`, `partial_per_seed_metrics.csv` and `partial_all_checkpoints.csv` preserve the verified partial outcome. The parent directory contains `system-shutdown-evidence.json`, runner logs and the source commit. The protocol, program copies and hashes, source snapshots, configurations and all produced logs, trajectories and checkpoints remain retained. There is no completed seed-2 GRU result. Do not compute a complete three-seed GRU mean or standard deviation from the two available seeds.

The seed-1 MLP wins under the frozen rule because final completion is tied and its position cost is lower. Its return and post-200k completion metrics also favor MLP. Therefore the observed seed-0 advantage has already failed the requirement of a nonlosing rank in every seed. Completing seed 2 is still required for the requested descriptive three-seed statistics.

No retry was launched. A requested retry must train seed-2 GRU from step 0 in a new directory with the same executable and protocol, retaining this interrupted archive. The 110k weight pair lacks Adam state, replay and RNG state and cannot restore a continuous run. The three successfully completed new jobs can be reused unchanged. `summarize_h26_blind_validation.py` audits all six complete method/seed runs, computes raw seed metrics and sample SDs, applies the frozen ranking, and writes reports and figures. It requires complete status and deliberately refuses this interrupted experiment.

## Completed seed-2 recovery

After the interruption report, the user requested completion using only a fresh seed-2 GRU restart. The same executable and frozen protocol were used from step 0, in build2026/h26-blind-validation-seed2-retry-20261111/results/. Training completed 300,001 interactions with exit code 0 in 1668.10 seconds. All 24 Actor/target files through 110,000 match the interrupted attempt exactly. The original interrupted directory is retained. Its partial curve was not concatenated or used as the seed-2 result.

| Seed | Method | Final completion | Mean return | All position cost | Completed position cost | First 10/10 step | Worst / mean completion after 200k |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | mlp-original | 90% | +138.20 | 1.4319 | 1.1597 | 240000 | 70% / 88% |
| 0 | gru-full-decay | 100% | +291.50 | 1.1291 | 1.1291 | 180000 | 80% / 98% |
| 1 | mlp-original | 100% | +9.40 | 1.1763 | 1.1763 | 240000 | 60% / 85% |
| 1 | gru-full-decay | 100% | -685.23 | 3.1486 | 3.1486 | 300000 | 0% / 48% |
| 2 | mlp-original | 90% | -76.38 | 1.9817 | 1.8543 | Not reached | 0% / 61% |
| 2 | gru-full-decay | 60% | -588.06 | 4.1511 | 3.3656 | Not reached | 0% / 32% |

| Method | Metric | Mean | Sample SD | Observed seeds |
| --- | --- | ---: | ---: | ---: |
| mlp-original | final_completion | 0.9333 | 0.0577 | 3 |
| mlp-original | final_mean_length | 469.5333 | 26.3917 | 3 |
| mlp-original | final_return | 23.7414 | 108.0036 | 3 |
| mlp-original | final_all_position_cost | 1.5300 | 0.4116 | 3 |
| mlp-original | final_completed_position_cost | 1.3968 | 0.3963 | 3 |
| mlp-original | first_ten_of_ten_step | 240000.0000 | 0.0000 | 2 |
| mlp-original | post200k_worst_completion | 0.4333 | 0.3786 | 3 |
| mlp-original | post200k_mean_completion | 0.7800 | 0.1480 | 3 |
| gru-full-decay | final_completion | 0.8667 | 0.2309 | 3 |
| gru-full-decay | final_mean_length | 439.0000 | 105.6551 | 3 |
| gru-full-decay | final_return | -327.2630 | 538.0655 | 3 |
| gru-full-decay | final_all_position_cost | 2.8096 | 1.5393 | 3 |
| gru-full-decay | final_completed_position_cost | 2.5477 | 1.2334 | 3 |
| gru-full-decay | first_ten_of_ten_step | 240000.0000 | 84852.8137 | 2 |
| gru-full-decay | post200k_worst_completion | 0.2667 | 0.4619 | 3 |
| gru-full-decay | post200k_mean_completion | 0.5933 | 0.3443 | 3 |

Sample SD uses n−1. Completion fractions are on a 0-to-1 scale. First-10/10 averages include only the two attaining seeds for each method. Seed 2 never reaches 10/10 for either method and remains missing, rather than being replaced by the budget. Seed 0 selected the schedule. Seeds 1 and 2 were blind validation.

The frozen superiority criterion fails. GRU wins in seed 0 on completion. MLP wins in seed 1 on position cost after tied final completion, and also has higher return and better post-200k completion. MLP wins in seed 2 on final completion (90% versus 60%), and also has lower position cost and higher return. Thus the seed-0 advantage did not reproduce in either validation seed. The data do not isolate whether the fixed decay schedule, architecture or other configuration differences account for the poor validation performance.

All six selected method/seed runs have 31 evaluated checkpoints, for 186 summary rows and 1,860 episode rows. The four selected new runs exited normally. The fixed alpha change and exact archived executable hashes were verified. Full curves and audit outputs are in build2026/h26-blind-validation-20261111/results/REPORT.md, comparison.png, returns.png, summary.json, per_seed_metrics.csv, across_seed_metrics.csv, all_checkpoints.csv, all_episodes.csv and trajectory_provenance.json. validation_status.json marks the recovered comparison complete and lists all selected directories. The original status.json and INTERRUPTED.md preserve the historical interruption.

Reproduce the full audit with the following command after all selected runs are complete.

```powershell
python -B experiments/rnn/summarize_h26_blind_validation.py --results build2026/h26-blind-validation-20261111/results --seed2-gru-retry build2026/h26-blind-validation-seed2-retry-20261111/results/seed-2/gru-full-decay
```
