# Fixed RNN h64 seed-0 experiment

This branch records only the vanilla tanh RNN experiment built from shared commit 6823695. Use training_rnn_full with compiler selection LEARNING_TO_FLY_ACTOR_RNN_FULL. The full Actor input is 146, hidden size is 64 and trainable parameter count is 13,764. Cell initialization and forward equations match the existing vanilla RNN.

The frozen protocol, audit and tests are documented in RECURRENT_FULL_AUDIT.md. Train seed 0 from scratch for 300,001 interactions with Actor Adam alpha 0.001, reduced permanently to 0.0001 after the second curriculum update. Retain all other shared settings and the original MLP seed-0 evaluation baseline. Evaluate 31 checkpoints with ten episodes each. The LSTM experiment is recorded on a separate branch from the same shared parent.
