# Fixed LSTM h21 seed-0 experiment

This branch records only the LSTM experiment built from shared commit 6823695. Use training_lstm_full with compiler selection LEARNING_TO_FLY_ACTOR_LSTM_FULL. The full Actor input is 146, hidden size is 21 and trainable parameter count is 14,200. Four gates each use one bias. The formula is 4*h*(147+h)+4*(h+1). h21 is 372 parameters from MLP 13,828, compared with a difference of 384 for h20. Keep the existing forget-gate bias of one.

The frozen protocol, audit and tests are documented in RECURRENT_FULL_AUDIT.md. Train seed 0 from scratch for 300,001 interactions with Actor Adam alpha 0.001, reduced permanently to 0.0001 after the second curriculum update. Retain all other shared settings and the original MLP seed-0 evaluation baseline. Evaluate 31 checkpoints with ten episodes each, after the RNN seed-0 run completes. The RNN experiment is recorded on a separate branch from the same shared parent.
