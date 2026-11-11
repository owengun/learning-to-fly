# Fixed LSTM h21 seed-0 experiment

This branch records only the LSTM experiment built from shared commit 6823695. Use training_lstm_full with compiler selection LEARNING_TO_FLY_ACTOR_LSTM_FULL. The full Actor input is 146, hidden size is 21 and trainable parameter count is 14,200. Four gates each use one bias. The formula is 4*h*(147+h)+4*(h+1). h21 is 372 parameters from MLP 13,828, compared with a difference of 384 for h20. Keep the existing forget-gate bias of one.

The frozen protocol, audit and tests are documented in RECURRENT_FULL_AUDIT.md. Train seed 0 from scratch for 300,001 interactions with Actor Adam alpha 0.001, reduced permanently to 0.0001 after the second curriculum update. Retain all other shared settings and the original MLP seed-0 evaluation baseline. Evaluate 31 checkpoints with ten episodes each, after the RNN seed-0 run completes. The RNN experiment is recorded on a separate branch from the same shared parent.

## Completed result

The fixed run completed 300,001 interactions normally. At 300k it completes 8/10 episodes with mean length 414.5 and return -535.41. All/completed position costs are 3.1478/3.0383, compared with MLP 1.4319/1.1597. It never reaches 10/10 in the saved checkpoints. Its worst and mean completion after 200k are 0% and 42%, compared with MLP 70% and 88%.

Both final evaluation failures cross the positive x position boundary. Episode 2 terminates at step 97 with x=0.602072895 m, and episode 6 at step 48 with x=0.605241358 m. Their z positions are +0.0075 and -0.1990 m. Completed episodes also have high position cost, so survival alone does not account for the poor return.

All logged gradient/update/state metrics are finite. Mean gradient L2 is 0.6747 in 180k-199k, 0.7948 in 200k-219k and 1.3867 in 280k-300k. Actual update L2 falls from 0.03461 to 0.00381 at the prescribed alpha change. Replay cell-state L2 averages 15.52, 17.75 and 13.13 in those windows. Final evaluation has mean hidden L2 1.692, cell L2 14.936 and hidden saturation 0.215%. Mean input/forget/output gate activations are 0.721/0.681/0.762. These averaged gate values alone cannot establish individual-unit gate saturation. The trajectory shows lateral boundary failure with poor position tracking, while the logs retain finite nonzero gradients.

This run fails the requested seed-0 promising-result condition because completion is lower than MLP and both return and position costs are worse. The cell equations passed the finite-difference checks, and the shared Adam/LR boundary tests passed. This single fixed run does not isolate a unique cause of the performance gap.

The monitoring host connection was re-established during this run. The original trainer PID 30880 and runner PID 31932 were still alive, and trainer CPU usage advanced from 1526.20 to 1531.56 seconds over five seconds while logging resumed at 260k. The same continuous training process reached 300,001. See ../system-process-check.json. No training restart or curve concatenation occurred.

Training source commit is ffcdbc028848fbd9922e726ad3584e84b3dafcf0, executable SHA256 is `bcc8cd3fa2124e2e4f0eaea4fb4883522b4aae445bec864ba96df7bc5a2ff95c`. Results are in `build2026/recurrent-full-seed0/lstm/run/`.
