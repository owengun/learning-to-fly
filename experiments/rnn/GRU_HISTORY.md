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
