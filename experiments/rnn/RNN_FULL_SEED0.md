# Fixed RNN h64 seed-0 experiment

This branch records only the vanilla tanh RNN experiment built from shared commit 6823695. Use training_rnn_full with compiler selection LEARNING_TO_FLY_ACTOR_RNN_FULL. The full Actor input is 146, hidden size is 64 and trainable parameter count is 13,764. Cell initialization and forward equations match the existing vanilla RNN.

The frozen protocol, audit and tests are documented in RECURRENT_FULL_AUDIT.md. Train seed 0 from scratch for 300,001 interactions with Actor Adam alpha 0.001, reduced permanently to 0.0001 after the second curriculum update. Retain all other shared settings and the original MLP seed-0 evaluation baseline. Evaluate 31 checkpoints with ten episodes each. The LSTM experiment is recorded on a separate branch from the same shared parent.

## Completed result

The fixed run completed 300,001 interactions normally. At 300k it completes 8/10 episodes with mean length 411 and return -503.52. All/completed position costs are 3.2326/2.7936, compared with MLP 1.4319/1.1597. It first reaches 10/10 at 280k. Its worst and mean completion after 200k are 10% and 50%, compared with MLP 70% and 88%.

Both final evaluation failures cross the positive y position boundary. Episode 2 terminates at step 71 with y=0.60087049 m, and episode 10 at step 39 with y=0.606497645 m. Their z positions are -0.2643 and -0.3541 m. The completed episodes also have elevated position cost, so poor tracking remains in the surviving episodes.

The logged mean Actor gradient L2 is 0.6549 over 180k-199k, 0.8524 over 200k-219k and 1.1159 over 280k-300k. All diagnostics are finite. Actual update L2 falls from 0.03498 to 0.00474 after the prescribed alpha change. Mean replay hidden saturation (absolute tanh activation greater than 0.95) is 20.1%, 20.3% and 13.7% in these windows. The two failed final episodes have mean hidden saturation 8.34% and 10.14%. These observations retain nonzero training signal and show position-boundary failure. They do not identify a unique causal explanation for the tracking error.

The run fails the requested seed-0 promising-result condition because completion is lower than MLP and both return and position costs are worse. No further parameter setting was selected. The planned LSTM seed-0 run follows separately. Raw trajectories, all 31 checkpoint pairs, actual Adam diagnostics, hidden telemetry, plots and manifests remain in this directory.

Training source commit is 17d9e9414d0b79476a1b77f951686b3210be7d13, executable SHA256 is `9e9771e38c73f6c0f469a61ab404a96a4892244fe1e5bfa53c64d139b0a62d12`. Results are in `build2026/recurrent-full-seed0/rnn/run/`.
