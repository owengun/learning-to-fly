# Vanilla RNN TD3 Actor experiment

The repository implements the paper *Learning to Fly in Seconds*. The MLP
configuration in src/config uses TD3, a 146 -> 64 -> 64 -> 4 actor, and two
64-unit, two-hidden-layer critics. Each critic consumes 28 privileged
observations and four actions. The default run takes 300001 environment
steps. The earlier experiment note recorded mean return 346.84 from a single
MLP run without enough metadata to reproduce or compare that number.

The recurrent variants use the first 18 policy observation features, a
32-unit recurrent state, and four tanh outputs. The LSTM also has a cell
state. These variants share the same TD3 loop and the same MLP critics.
The vendored RLtools revision supplies the TD3 and critic operations. The
recurrent cells and BPTT are implemented locally.

All recurrent variants reset rollout state on episode boundaries. Replay
windows contain eight observations and reject earlier termination or
truncation within the window. A target window shifts the sampled observations
and appends the stored next observation. Each training window starts from
zero state. TD3 masks the bootstrap on termination and allows it at a
time-limit truncation using the transition's stored next observation.
The actor applies one SGD update to the accumulated batch gradient. The
MLP baseline uses the original RLtools feed-forward TD3 loop.

## Build and run

From the project root with an existing CMake build directory on Windows:

    cmake --build build2026 --config Release --target training_mlp training_headless

Use a separate working directory for each run to isolate checkpoint and log
files. From those directories, invoke the corresponding executable by its
absolute path:

    training_mlp.exe --seed 0 --steps 30041
    training_headless.exe --seed 0 --steps 30041

training_headless selects Vanilla RNN. The smoke budget crosses the default
15000-step critic warmup and 30000-step actor warmup. Omit --steps to use the
full configured budget. The MLP and recurrent variants also differ in input
features, hidden widths and actor optimizers. Comparisons among the recurrent
variants control the input and hidden widths, with different gate parameter
counts. Training episode returns include exploration noise.

## Evaluation and checkpoints

A recurrent run saves policy weights every 10000 steps and on the final
requested step, under checkpoints/vanilla.
To load and evaluate a policy, use its matching executable, for example:

    training_headless.exe --seed 0 --evaluate checkpoints/vanilla/actor_step_30040.txt --episodes 3

Evaluation uses the configured evaluation environment without action
exploration noise and resets recurrent state for each episode. Vanilla also
accepts the previous 18/32/4 text checkpoint. The original MLP uses its
existing validation and checkpoint path. The recurrent text checkpoints
contain actor weights for evaluation and warm-starting. For example:

    training_headless.exe --seed 0 --warm-start path/to/actor_step_30040.txt --steps 30041

Warm-starting initializes both actors from the checkpoint and starts a new
training run with fresh critics, replay, RNG and simulator state. Use a new
working directory to preserve the source run. Exact training continuation
would require replay, optimizer, RNG and simulator state.
