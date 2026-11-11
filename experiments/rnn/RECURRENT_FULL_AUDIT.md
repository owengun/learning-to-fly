# Full-observation RNN and LSTM audit

## Read-only source review

Reviewed `feature/rnn-td3` at `9272cec`, `feature/lstm-td3` at `5de00aa`, and the full-observation GRU/Adam experiments through `104c94b`. The older RNN/LSTM paths select 18 observations, sample eight-transition sequences from zero recurrent state, and apply SGD. They preserve online hidden state around training and already reject sequences crossing terminal/truncated boundaries. The full-observation GRU experiment provides the 146-column adapter, 32-step burn-in, up-to-eight-step learning windows, aligned next-observation target action, and Actor Adam needed for the requested comparison.

The LSTM has four affine gates, each with one input matrix, one recurrent matrix and one bias vector. Its parameter count is `4*h*(146+h+1) + 4*(h+1)`. This gives 13,444 for h20, 384 below the MLP's 13,828, and 14,200 for h21, 372 above. Therefore use h21, retaining the existing input/forget/output/candidate gate order and forget bias of one. The vanilla tanh RNN has one affine recurrent layer and a tanh output head. Its count is `h*(146+h+1) + 4*(h+1) = h²+151h+4`, giving 13,764 for h64, 64 below MLP.

The review identified protocol gaps in the older branches. It did not identify a mathematical error in their cell equations. The new tests exhaust finite differences on small cells and sample full-size parameters across all tensors, explicitly including action-history column 145 in every gate. The vanilla implementation also matches the original tanh RNN initialization and forward outputs exactly. Sequence training is functional with respect to live online state.

## Shared implementation and isolated experiment branches

The shared implementation is committed on `experiment/recurrent-full-adam-common`, with independent compiler selections for `RNN_FULL` and `LSTM_FULL`. The tanh RNN uses one affine gate in the common cell representation. LSTM retains four gates. The legacy GRU compiler flags map to the same existing sequence path. The GRU experiment branch and archived reports remain at their existing commits.

Create `experiment/rnn-full-h64-adam-decay` and `experiment/lstm-full-h21-adam-decay` from this shared commit. Record each experiment only on its own branch. Compile each model to its own executable. Run RNN seed 0 first and LSTM seed 0 second, with 300,001 interactions and all 31 checkpoints evaluated in fresh seed-0 processes for ten episodes with no action exploration. The MLP comparison reuses its archived exact executable/protocol data. No MLP retraining is required.

Actor alpha starts at 0.001. At step 200,000, save the pre-change checkpoint, finish the second curriculum update and populated-replay reward recalculation, set alpha to 0.0001, and apply that alpha on the next Actor update in the same training step. Preserve Adam moments, independent weight decay, Critic settings, target updates and all remaining training settings. The existing 32-step burn-in approximation reconstructs the recurrent state from a finite prefix. It is shared with GRU.

## Verification

Release builds and tests for both model selections cover all 146 input columns, oldest-to-newest action history, actual online episode entry and evaluation hidden/cell reset, online/offline forward equality over the same initial prefix, terminal/truncated and circular replay boundaries, target next-observation alignment, target Polyak state preservation, multi-step gradients from a nonzero held-fixed state, Adam numerics/rollback/weight decay, curriculum/LR timing with persistent moments, and checkpoint roundtrip/truncation rejection. Ten test executables pass, including the existing GRU/LSTM cell test and the GRU LR-boundary regression test. Logs are in `build2026/recurrent-full-seed0/tests/`.

The runner retains executable/source hashes, configuration/source snapshots, training and evaluation commands/status, all weights and trajectories, per-episode and per-checkpoint tables, full curves, actual Adam update diagnostics and hidden/cell/gate telemetry. It performs one descriptive trajectory/gradient/state diagnosis per model. Compare final completion, mean length, return, all/completed position costs, first 10/10 checkpoint, and worst/mean completion across 210k through 300k. A promising seed-0 result requires completion at least equal to MLP, with return and position advantages assessed from the raw margins. Further seed validation requires a later instruction.

```powershell
python -B experiments/rnn/run_recurrent_full_budget.py --actor rnn-full --exe build2026/src/Release/training_rnn_full.exe --mlp build2026/curriculum-pilot-seed0/seed-0/mlp-original/attempt-0001 --output build2026/recurrent-full-seed0/rnn/run
python -B experiments/rnn/run_recurrent_full_budget.py --actor lstm-full --exe build2026/src/Release/training_lstm_full.exe --mlp build2026/curriculum-pilot-seed0/seed-0/mlp-original/attempt-0001 --output build2026/recurrent-full-seed0/lstm/run
```
