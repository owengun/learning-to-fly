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
