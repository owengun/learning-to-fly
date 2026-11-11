# Actor learning rate after the second curriculum update

## Archived h26 diagnosis

The 146-input, hidden-size-26 GRU uses seed 0 and ten deterministic evaluation episodes per checkpoint. Completion means reaching the 500-step time limit. Position cost is the per-episode mean of `20 * (x² + y² + z²)` from the prior state, excluding the first transition whose prior state was unlogged and termination transitions that replace the ordinary reward. The two columns average that quantity across all or only completed episodes. Signed z is the mean in meters over the last 100 steps of completed episodes. Adam update L2 is the logged magnitude of the actual Actor parameter change at the indicated training step, after the checkpoint at that step.

| Step | Complete / 10 | Mean length | Mean return | All position cost | Completed position cost | Signed last-100 z (m) | Actor update L2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 180000 | 10 | 500.0 | -1007.58 | 3.7504 | 3.7504 | +0.19909 | 0.0233535 |
| 190000 | 10 | 500.0 | -374.26 | 2.4951 | 2.4951 | +0.27104 | 0.0372160 |
| 200000 | 9 | 456.1 | +73.01 | 2.0060 | 1.6061 | +0.07343 | 0.0237256 |
| 210000 | 5 | 286.8 | -465.84 | 3.0849 | 3.2010 | +0.14226 | 0.0328257 |
| 220000 | 5 | 279.6 | -520.48 | 3.6433 | 3.2314 | +0.24162 | 0.0402968 |
| 230000 | 10 | 500.0 | -1068.47 | 3.3334 | 3.3334 | -0.29309 | 0.0422232 |
| 240000 | 8 | 411.1 | -289.94 | 3.0268 | 2.3452 | -0.18886 | 0.0329165 |
| 250000 | 7 | 372.0 | -456.97 | 2.9134 | 2.7925 | -0.15357 | 0.0311128 |
| 260000 | 8 | 412.5 | -212.48 | 2.6593 | 2.1845 | -0.22442 | 0.0330252 |
| 270000 | 10 | 500.0 | +501.80 | 0.7908 | 0.7908 | -0.09152 | 0.0465546 |
| 280000 | 10 | 500.0 | -55.81 | 1.8792 | 1.8792 | -0.16667 | 0.0377379 |
| 290000 | 10 | 500.0 | +11.39 | 1.6874 | 1.6874 | -0.19427 | 0.0281163 |
| 300000 | 9 | 456.4 | -400.69 | 2.7508 | 2.5775 | -0.27718 | 0.0312020 |

The archived training log saves the 200,000 snapshot, then records the curriculum recomputing 200,000 replay rewards and Actor update 8,500 at that same step. Completion falls to 5/10 at 210,000 and 220,000 while the logged Actor update L2 rises. This temporal pattern motivates testing a smaller step size. The later 270,000 snapshot pairs the best return in this window with a comparatively large logged update, so the archived association alone does not establish causality.

## Protocol

Restart from seed 0 and train for 300,001 interactions. Use the same full-observation h26 Actor, Critic, target update, replay, curriculum and Adam state. Change Actor Adam alpha from `1e-3` to `1e-4` when the second curriculum update finishes at step 200,000, before that step's Actor update. The existing independent weight-decay term remains fixed. Score every 10,000 interactions, including step 0, with ten no-exploration episodes. Verify both Actor and target hashes and all ten returns against archived h26 at every checkpoint through 200,000.

Compare the full curves against h26 without decay and the original MLP. Measure worst completion after 200,000, first recovery to the shared h26 reference of 9/10 after the first observed drop, mean return and both position costs at 280,000 and 300,000, and final completion. Use 10,000-step resolution for recovery. A curve with no checkpoint below 9/10 has no observed drop. The archived h26 first recovers at 230,000, 30,000 interactions after the boundary.

Build the `training_gru_h26_decay`, `test_gru_h26_decay` and `test_actor_adam` Release targets. Run the tests in a scratch directory because the boundary test writes checkpoints. Launch from the repository root with the archived seed-0 reference directories.

```powershell
python -B experiments/rnn/run_gru_full_budget.py `
    --decay-exe build2026/src/Release/training_gru_h26_decay.exe `
    --h26 build2026/gru-full-budget-seed0/run `
    --mlp build2026/curriculum-pilot-seed0/seed-0/mlp-original/attempt-0001 `
    --output build2026/gru-h26-decay-seed0/run
```

## Completed restart on 2026-11-11

The previous run was stopped at the user's request after its 150,000 checkpoint and is archived separately. The completed restart used the same verified executable from that archive, with SHA256 `b92f4a14a29875bbfb64f81f94a509f7ff1fa2132641a2242600d1cdc1d698b7`, from source commit `2f21aea95bc61b8f12155f2aae13d26317516de7`. No previous trainer was running when this restart began. The restart trained from seed 0 to 300,001 interactions and exited normally. The trainer reports 1,817.5 seconds. All 31 checkpoints were evaluated with ten no-exploration episodes. Both Actor and target hashes and all ten returns match the archived h26 at all 21 checkpoints through 200,000. The logged order is checkpoint, curriculum update, alpha reduction, then Actor update at step 200,000. All subsequent logged alpha values are 0.0001.

| Step | Actor | Complete / 10 | Mean length | Mean return | All position cost | Completed position cost |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 280000 | MLP | 10 | 500.0 | -144.72 | 1.8030 | 1.8030 |
| 280000 | h26 without decay | 10 | 500.0 | -55.81 | 1.8792 | 1.8792 |
| 280000 | h26 with Actor LR decay | 10 | 500.0 | +238.21 | 1.2400 | 1.2400 |
| 300000 | MLP | 9 | 454.9 | +138.20 | 1.4319 | 1.1597 |
| 300000 | h26 without decay | 9 | 456.4 | -400.69 | 2.7508 | 2.5775 |
| 300000 | h26 with Actor LR decay | 10 | 500.0 | +291.50 | 1.1291 | 1.1291 |

| Actor | Worst completion after 200k | Mean completion after 200k | Points below 90% | First drop | First recovery | Steps since 200k | Steps since first drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP | 70% | 88% | 3 | 210000 | 220000 | 20000 | 10000 |
| h26 without decay | 50% | 82% | 5 | 210000 | 230000 | 30000 | 20000 |
| h26 with Actor LR decay | 80% | 98% | 1 | 270000 | 280000 | 80000 | 10000 |

The smaller alpha improved the worst observed post-curriculum completion and the 280k/300k return and position costs in this seed. The decay run completes all ten episodes at every post-200k checkpoint except 270k, where it completes eight. At 270k the original h26 is better, with ten completions and return +501.80 versus +83.71 for decay. Recovery to at least nine completions takes one 10k evaluation interval after the decay run's first drop. The preregistered time from the curriculum boundary is 80k, because that first drop occurs much later, compared with 30k for the original h26. It would be misleading to describe the 80k and 30k values as outage durations.

Across the ten post-200k checkpoints, mean return is +251.80 for decay, -295.75 for original h26 and -570.80 for MLP. Mean all-episode position cost is 1.3234, 2.5769 and 2.6473 respectively. These checkpoint averages describe this one seed and reuse the same ten evaluation episodes. Performance across training seeds remains untested. The independent weight-decay term remains fixed, so total update L2 need not be one tenth of the original run.

Artifacts are in `build2026/gru-h26-decay-seed0-restart/run/`. `REPORT.md` contains the comparison, `comparison.png` and `comparison.pdf` show complete curves, `comparison_summary.json` and `post_curriculum_summary.csv` retain the derived metrics, `flight_by_checkpoint.csv` contains 93 comparison rows, and `all_episodes.csv` contains 930 episode rows across the three methods. Checkpoints, raw trajectories, individual evaluation logs, training logs and actual Adam update diagnostics are retained. The parent directory contains runner logs, the training source commit and `analyze_completed_run.py` for reproducing the audit and report.

For a future fresh reproduction, use the launch command above with `--decay-exe build2026/gru-h26-decay-seed0/run/training_gru_h26_decay.exe` and a new output directory. The saved checkpoint pairs contain Actor and target weights only. They omit the optimizer, replay and RNG state and cannot resume an exact continuous run. The paused archive and this completed restart must remain separate. No further parameter setting or additional training seed was run.
