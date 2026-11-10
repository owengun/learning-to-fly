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
