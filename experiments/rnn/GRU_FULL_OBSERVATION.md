# Full-observation GRU Actor replacement

training_gru_full consumes exactly the 146 simulator observation columns
used by the original MLP. The first 18 are current observations and the
remaining 128 are 32 applied motor commands ordered oldest to newest.
There is no extra action feature. At reset the simulator initializes the
history from the episode's initial rotor speeds, matching the MLP.
Evaluation resets that simulator state and the GRU hidden state together.

The Actor has 26 hidden units. Its parameter count is
3 * 26 * (146 + 26 + 1) + 4 * (26 + 1) = 13,602, against 13,828
for the original MLP. It uses the original MLP Adam configuration and the
same Critic architecture, curriculum, noise, TD3 update schedule and
checkpoint evaluation protocol. Sequence replay is specific to the GRU.
The learning window spans up to eight transitions with 32 burn-in steps
when available in the same episode. Burn-in reconstructs the recurrent
context in addition to the explicit 32-action history in each observation.
Target actions extend the aligned recurrent state with the stored next
observation. The previous 18- and 22-input experiments removed the
explicit history and are separate recurrent-memory ablations.

Build training_gru_full and test_full_observation in Release. The test
checks all input columns, history shifting and reset, recurrent online
and replay agreement, parameter counts and checkpoint roundtrip.
Run the seed-0 screen from the repository root:

    python -B experiments/rnn/run_gru_adam.py --full-observation --exe build2026/src/Release/training_gru_full.exe --reference build2026/curriculum-pilot-seed0/seed-0/mlp-original/attempt-0001 --output build2026/gru-full-seed0/run

The screening threshold is 75.2 mean steps (15 more than the 22-input
Adam screen's 60.2), or at least two of ten 500-step completions. The
trainer resumes the same process to 300,001 interactions if it passes.
Each reported checkpoint is scored on ten noise-free episodes with seed 0.
The original MLP reference is the stored seed-0 checkpoint with the same
transition count. The procedure is a one-seed screening experiment.
