# Final adaptive-curriculum experiment (v5)

This directory is the self-contained final implementation and evaluation package.
It does not include source files from `adaptive_curriculum_v1` through `v4`.

## What is implemented

- TD3 training for 400,000 interactions with 25,000-step checkpoints
- performance-gated force, mass, and reward curriculum
- nominal and severe-stress validation gates
- rollback after a failed retention check
- precision-constrained checkpoint selection without reading held-out test data
- paired, five-seed evaluation on a 16-condition held-out force/mass grid
- diagnostic force, mass, and reward ablations

`adaptive_curriculum_core.cpp` and `stress_evaluation_core.cpp` are included
locally so this directory no longer depends on the older experiment folders.

## Source layout

- `adaptive_curriculum_v5.cpp`: final training entry point
- `adaptive_curriculum_core.cpp`: adaptive training implementation
- `v5_precision_config.h`: final precision thresholds
- `heldout_stress_evaluation.cpp`: held-out evaluation entry point and grid
- `stress_evaluation_core.cpp`: common evaluator implementation
- `ablation_*.cpp`: diagnostic ablation entry points
- `select_checkpoints.py`: validation-only checkpoint selection
- `analyze_training.py`: curriculum-exposure audit
- `analyze_heldout.py`: paired five-seed held-out analysis
- `analyze_ablation.py`: seed-2 diagnostic ablation analysis
- `analysis/`: compact derived result tables and JSON summaries

Generated build trees, TensorBoard logs, checkpoints, and HDF5 learning curves
are excluded by `.gitignore`. The checked-in CSV/JSON files are small result
artifacts needed to audit the reported numbers.

## Build

Run from the repository root inside the project Docker image:

```bash
cmake -S experiments/adaptive_curriculum_v5 \
      -B experiments/adaptive_curriculum_v5/build \
      -DCMAKE_BUILD_TYPE=Release
cmake --build experiments/adaptive_curriculum_v5/build -j
```

The main executables are:

```text
experiments/adaptive_curriculum_v5/build/adaptive_curriculum_v5_training
experiments/adaptive_curriculum_v5/build/heldout_stress_evaluation
```

## Train

The optional argument is the seed offset. Five-seed training uses seeds 0--4:

```bash
mkdir -p experiments/adaptive_curriculum_v5/run
cd experiments/adaptive_curriculum_v5/run
../build/adaptive_curriculum_v5_training 0
```

Repeat with `1`, `2`, `3`, and `4`. Then select checkpoints and audit
whether the selected actors were exposed to a promoted curriculum stage:

```bash
cd ../../..
python experiments/adaptive_curriculum_v5/select_checkpoints.py
python experiments/adaptive_curriculum_v5/analyze_training.py
```

## Held-out protocol

The final grid uses force factors `1.25, 1.75, 2.5, 3.5` and mass factors
`1.025, 1.075, 1.15, 1.25`. These values are absent from both the training
stages and the fixed checkpoint-validation task. Each model/condition pair is
evaluated for 1,000 episodes with paired random seeds.

After producing the raw evaluator CSV files, regenerate the summaries with:

```bash
python experiments/adaptive_curriculum_v5/analyze_heldout.py
python experiments/adaptive_curriculum_v5/analyze_ablation.py
```

## Verified result

Five 400k-interaction runs were completed. The selector chose checkpoints at
250k, 350k, 350k, 350k, and 325k interactions. Compared with the five-seed
baseline over all 16 held-out conditions, the selected actors achieved:

- mean episode length: `+36.518` steps (`+9.71%`), paired bootstrap 95% CI
  `[14.913, 58.122]`, improved in 5/5 seeds
- success rate: `+0.09819` (`+9.82` percentage points), paired bootstrap 95%
  CI `[0.03495, 0.16211]`, improved in 5/5 seeds
- mean position error: `-0.007264 m` (`-2.41%`), CI crosses zero and only 1/5
  seeds improved
- return per step: `+0.29423`, CI crosses zero and only 2/5 seeds improved

Thus the robust-survival improvement is reproducible across these five seeds;
the precision and reward improvements are not statistically consistent.

## Interpretation limit

The training loop saves a checkpoint before running the gate at the same step.
All five selected checkpoints precede actual promoted-stage exposure. The
verified result therefore supports:

1. removing the original fixed curriculum, and
2. precision/stress-aware validation checkpoint selection.

It does **not** establish a causal benefit from the force/mass adaptive
curriculum. A causal curriculum claim requires a new controlled, multi-seed
fine-tuning experiment from identical starting checkpoints. The seed-2
ablation included here is diagnostic only.

