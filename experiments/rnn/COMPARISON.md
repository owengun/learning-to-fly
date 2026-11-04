# Matched TD3 comparison

training_mlp preserves the repository's original 146-input baseline and its
original optimizer and evaluation entry points. For an architecture comparison,
training_mlp_matched uses the same two 64-unit hidden layers but takes the
same first 18 observations as the recurrent policies. It shares their simulator,
two MLP critics, sequence endpoint sampling, critic and target smoothing,
actor SGD, update schedules, 32-bit precision, initialization range and
noise-free evaluation protocol. Its actor uses only the current observation.
Its tanh implementation and initialization differ from the original MLP
executable. The earlier 50,001-step recurrent pilot remains a separate run.

| Actor | Trainable actor parameters |
| --- | ---: |
| Matched MLP | 5636 |
| Vanilla RNN | 1764 |
| LSTM | 6660 |
| GRU | 5028 |

Build training_mlp_matched, training_headless and test_matched_mlp_actor.
Each coursework branch's training_headless selects its corresponding actor.
In a fresh CMake 4 build with the vendored JSON dependency, pass the quoted
argument '-DCMAKE_POLICY_VERSION_MINIMUM=3.5'.

Run the benchmark serially. Replace each executable path with the Release
binary from its own build tree:

    python experiments/rnn/run_comparison.py --output RESULTS --steps 50001 --episodes 10 --seeds 0 1 2 --exe mlp-matched=MLP_EXE --exe vanilla=VANILLA_EXE --exe lstm=LSTM_EXE --exe gru=GRU_EXE

The runner writes each actor and seed into a separate attempt directory.
After training, it evaluates every 10,000-step actor checkpoint and the final
checkpoint with fixed seeds and ten episodes. Final and best mean return,
best checkpoint step, final episode standard deviation, step-normalized
trapezoidal return AUC, wall time and parameter count are exported to CSV.
Across-seed standard deviation uses completed seed means. Evaluation logs
retain individual episode returns. Only a small fraction of the configured
training budget is covered, so the results cannot establish convergence.

A successful training run with incomplete evaluation can resume evaluation
by repeating the command. A failed training attempt is preserved and a
retry uses a new attempt directory. Actor and target weights are saved
separately at each checkpoint. The training executable can warm-start actor
weights, with fresh replay, critics, optimizer, simulator and random state.
Exact mid-run training continuation is outside this checkpoint format.

The completed supplemental original-MLP comparison is documented in
ORIGINAL_BASELINE_AUDIT.md. It uses training_mlp with 146 inputs and the
original training algorithm, with the same standalone evaluation protocol.
Run run_original_mlp.py to reproduce the supplemental three-seed experiment.
