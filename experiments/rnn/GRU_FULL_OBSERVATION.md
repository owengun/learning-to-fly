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

## Seed 0 result

Both policies were scored at their saved 100,000-interaction checkpoints.
The GRU training process exited after 100,001 interactions. The audit
verified 11 checkpoints, 110 evaluation episodes, and the hashes for each
Actor, target and trajectory file. All logged optimizer diagnostics were
finite. The source used for training is 5ffcbcb.

| Actor | Mean flight steps | 500-step completion | Downward exits | Mean return | First 10 steps vertical thrust / weight |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original 146-input MLP | 61.6 | 0/10 | 1/10 | -193.81 | 0.569 |
| 146-input GRU, hidden 26 | 66.7 | 0/10 | 7/10 | -149.60 | 0.489 |
| 22-input Adam GRU, history-removal ablation | 60.2 | 0/10 | 6/10 | -160.19 | 0.439 |

Projected vertical thrust is computed from post-transition rotor speeds
and attitude using the simulator's thrust curve. The first-ten-step
averages use all ten episodes. In the downward-exit subset, the final five
steps averaged 0.887 vehicle weights and -1.53 m/s for the full GRU (seven
episodes). The original MLP had one downward exit, with 1.300 vehicle
weights and -1.35 m/s in its last five steps. These conditional means have
different sample counts and exclude external disturbance forces.

The full GRU gained 6.5 mean steps relative to the 22-input Adam ablation
and 5.1 relative to MLP. It had no 500-step completion, and its longest
episode was 99 steps. The predeclared 75.2-step or 2/10-completion
criterion did not pass, so this run stopped after screening. These data
do not isolate the effect of history from width when comparing the two
GRUs. They provide one seed for the parameter-matched replacement.

The local archive build2026/gru-full-seed0 contains Release test logs,
the run manifest, checkpoint hashes, optimizer diagnostics, the screening
decision, raw flight CSVs and the thrust comparison.
