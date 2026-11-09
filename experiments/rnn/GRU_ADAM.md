# GRU Actor Adam ablation

This ablation starts from experiment/gru-action-burn-in. The target
training_gru_adam uses the same 22-input actor, 32 hidden units, 32-step
burn-in, up to eight learning steps, replay sampling, TD3 critics, and
curriculum as training_gru_history. Seed 0 starts from the same weights.
The only training update change is the Actor optimizer.

The optimizer reads the original MLP parameter configuration from RLtools.
Alpha is 0.001, beta1 is 0.9, beta2 is 0.999, epsilon is 1e-8, and bias
learning rate factor is 1. Each weight group uses decay 1e-4, matching the
CPU operation that subtracts weight times decay divided by two outside
alpha. Biases do not decay. One update counter drives bias correction.
The target network contains no optimizer state. Actor and target weights
share the SGD checkpoint format and can be evaluated or warm-started.

Logs record batch gradient, actual update and parameter L2 norms, Q scale,
GRU gate means, proposed and applied actions, and replay vertical motor
thrust normalized by vehicle weight. The thrust estimate excludes external
forces. Non-finite checks on the online state, actions, observed critic
values and optimizer updates stop training and preserve the last valid
Adam state and Actor/target weights under failure/. Optimizer diagnostics
are not a full resumable training checkpoint. No gradient clipping is used.

Build training_gru_adam and test_actor_adam in Release. Run the numerical
test from a scratch directory. Launch the seed-0 experiment from the root:

    python -B experiments/rnn/run_gru_adam.py --exe build2026/src/Release/training_gru_adam.exe --reference build2026/gru-history-seed0/screen/seed-0/gru-history/attempt-0001 --output build2026/gru-adam-seed0/run

At the 100,000-transition checkpoint, the trainer pauses for the same ten
no-exploration evaluation episodes as SGD. It continues in the same
process to 300,001 transitions if at least two episodes reach 500 steps,
or if mean length is at least 65 and at least 20 steps above SGD with at
least two fewer lower exits. Otherwise it exits with the screen artifacts.
The checkpoint numbered 100000 precedes its numbered interaction.

## Seed 0 result

The two Actor variants have identical step-zero checkpoint hashes and
evaluation returns. The Adam run completed 100,001 interactions, with
evaluation of 11 checkpoints and 110 episodes. The 100,000 checkpoint
uses the same protocol as the existing SGD reference.

| Actor optimizer | Mean episode length | Lower boundary exits | 500-step completions | Mean return |
| --- | ---: | ---: | ---: | ---: |
| SGD | 46.1 | 8/10 | 0/10 | -119.02 |
| Adam | 60.2 | 6/10 | 0/10 | -160.19 |

Adam passed neither continuation criterion. It reduced downward exits by
two, but did not reach the length threshold of 66.1 steps for this reference
or complete any episode. The longest final Adam episode lasted 99 steps.
The comparison covers one training seed.

On episodes that exited downward, Adam's first ten steps supplied an
estimated 44% of vehicle weight in vertical motor thrust. The last five
steps averaged 114% and -1.33 m/s vertical speed, compared with 33% and
-2.73 m/s for SGD. Four of six Adam downward exits crossed the estimated
weight threshold, with the first crossing between steps 26 and 55.
The other two never crossed it. The higher late thrust did not cancel
the downward velocity before the boundary.

The last logged update had raw gradient L2 0.8764, applied parameter
update L2 0.02681 and parameter L2 15.7989. Mean Q was 34.14 with
maximum absolute Q 56.22. Update/reset gate means were 0.260 and 0.597.
Mean proposed and applied batch actions were 0.324 and 0.032.
Mean recorded replay thrust was 0.406 of vehicle weight. All logged
numeric diagnostics were finite through 3,500 Actor updates.

Run manifests, exact commands, optimizer diagnostics, checkpoint and
trajectory hashes, and raw episode CSVs are saved under
build2026/gru-adam-seed0/run. The run stopped at the screening gate.
