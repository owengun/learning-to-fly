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
