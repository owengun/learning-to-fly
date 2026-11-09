# GRU action-history and burn-in ablation

Run this experiment locally on the experiment/gru-action-burn-in branch.
training_gru is the 18-feature SGD reference. training_gru_history keeps
32 hidden units and SGD 1e-4, adds the preceding applied motor command
to the 18 observations, and keeps the original initialization for shared
parameters. The new input weights start at zero.

The off-policy runner applies exploration and clips the motor command
before retaining it for the next online input and writing it to replay.
At a genuine episode boundary the previous command and hidden state reset.
Replay uses 32 burn-in transitions and up to eight learning transitions.
For a shorter prefix it starts at the genuine episode boundary. It skips
a history with a missing predecessor after circular-buffer overwrite.
The target network extends each aligned history with the stored next
observation and that transition's applied action. TD3 retains its existing
terminal masking and time-limit bootstrap. The BPTT loss uses all valid
learning transitions and detaches the state reconstructed by burn-in.

The text checkpoints for both the original MLP and recurrent targets
are saved before their numbered transition. A label of 100000 therefore
contains exactly 100,000 completed environment interactions. Running 100001
iterations emits that checkpoint without a new checkpoint convention.

Build training_gru, training_gru_history and test_action_history in Release.
Run test_action_history and test_gated_actor. Then run:

    python -B experiments/rnn/run_gru_history_screen.py --bin build2026/src/Release --output build2026/gru-history-seed0/screen

The screen uses seed 0 and ten no-exploration episodes per checkpoint.
The predeclared threshold for running to 300,001 steps is either two
500-step completions, or a mean of at least 65 steps and at least 20 more
than the re-run GRU reference. The primary diagnostics are episode length,
completion, lower-boundary exits, terminal vertical speed and tilt,
motor outputs, recurrent hidden norms and gate saturation. Training logs
include mean Q, maximum absolute Q and the actor gradient L2 norm.

## Seed 0 screen result

Both runs completed 100,001 environment interactions and produced aligned
100,000-transition checkpoints. The step-zero policies reproduced exactly
the same ten returns. The audit verified 22 checkpoints and 220 evaluation
episodes, including actor, target and trajectory file hashes.

| Method | Mean flight steps | 500-step completion | Lower exit | Mean return | 3D position RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reference GRU | 42.4 | 0/10 | 9/10 | -171.25 | 0.425 m |
| Applied-action GRU plus burn-in | 46.1 | 0/10 | 8/10 | -119.02 | 0.343 m |

In the earlier 300,000-transition pilot, the eight downward-exiting GRU
episodes averaged -2.33 m/s vertical speed, 27.3 degrees of tilt and
11,151 RPM in the last five steps. The projected motor thrust was 51.6%
of vehicle weight, excluding external disturbances. The nine completed
MLP episodes were near zero vertical speed and 99.7% of vehicle weight.
The old GRU motor commands and sigmoid gates were unsaturated in those
failure windows. The checkpoint replay reproduced all ten original returns.

At 100,000 transitions, the reference and modified GRU's downward exits
averaged -2.63 and -2.73 m/s over the last five steps. Projected vertical
motor thrust averaged 35.3% and 32.5% of vehicle weight respectively.
The update and reset gates remained near 0.5 and unsaturated. The actor
gradient L2 norm at the logged update was 1.35 and 3.22. Batch mean Q
was 7.23 and 6.81, with maximum absolute Q of 49.64 and 47.44.
The history run's sampled prefixes averaged 12.75 burn-in steps at
that update because many windows began near the true episode start.

The screen did not pass the rule for extending to 300,001 transitions.
This joint ablation changed both input and replay training. One seed
does not identify the independent effect of either change. Adam and
hidden-width variants were not run in this screen.

The local artifacts are under build2026/gru-history-seed0. The screen
contains archived executables, exact commands, update diagnostics,
training logs, checkpoints, trajectories and per-checkpoint evaluations.
