# Performance-adaptive wind/mass curriculum

This experiment is isolated from the original `src` implementation. It reuses
the original TD3 model, simulator, checkpoint writer, and evaluation loop, but
does not call the fixed 100k-step curriculum.

## Curriculum

Validation runs every 25k environment interactions on the unchanged nominal
validation environments. A stage advances after two consecutive successful
checks and at least 50k interactions in the current stage.

| Stage | Wind-equivalent force std | Episode mass factor | Reward promotions |
|---|---:|---:|---:|
| 0 | baseline x1.0 | 1.00 | 0 |
| 1 | baseline x1.5 | Uniform(1.00, 1.05) | 1 |
| 2 | baseline x2.0 | Uniform(1.00, 1.10) | 2 |
| 3 | baseline x3.0 | Uniform(1.00, 1.20) | 2 |

The simulator's existing random external force is used as a wind-equivalent
disturbance. It is a constant force sampled once per episode, not an
aerodynamic wind-speed model. Mass is sampled only when the training runner is
about to reset an episode. Evaluation and gate environments remain nominal.

The gate thresholds are pilot values derived from the three-seed reproduction:

| Promotion | Mean episode length | Mean position error |
|---|---:|---:|
| 0 to 1 | at least 100 | at most 0.45 m |
| 1 to 2 | at least 300 | at most 0.35 m |
| 2 to 3 | at least 450 | at most 0.25 m |

## Docker build

From the repository root in PowerShell:

```powershell
docker run --rm `
  --entrypoint /usr/bin/cmake `
  -v "${PWD}:/workspace" `
  -w /workspace/experiments/adaptive_curriculum `
  arpllab/learning_to_fly:latest `
  -S . -B build -DCMAKE_BUILD_TYPE=Release `
  -DRL_TOOLS_BACKEND_ENABLE_MKL:BOOL=ON `
  -DRL_TOOLS_DISABLE_CPU_SPECIFIC_OPTIMIZATIONS:BOOL=ON

docker run --rm `
  --entrypoint /usr/bin/cmake `
  -v "${PWD}:/workspace" `
  -w /workspace/experiments/adaptive_curriculum `
  arpllab/learning_to_fly:latest `
  --build build --target adaptive_curriculum_training -j 4
```

## One training run

```powershell
New-Item -ItemType Directory -Force experiments/adaptive_curriculum/run | Out-Null

docker run --rm `
  --entrypoint /workspace/experiments/adaptive_curriculum/build/adaptive_curriculum_training `
  -v "${PWD}:/workspace" `
  -w /workspace/experiments/adaptive_curriculum/run `
  arpllab/learning_to_fly:latest 0
```

The run directory receives TensorBoard logs, checkpoints, learning curves, and
run-specific `curriculum_transitions_<run_name>.csv` and
`run_summary_<run_name>.csv` files. The run-specific names prevent later seeds
from overwriting earlier summaries.

## Compare against the baseline

After seeds 0, 1, and 2 have completed, run:

```powershell
python experiments/adaptive_curriculum/analyze_comparison.py
```

The generated CSV, JSON, and SVG files are written to
`experiments/adaptive_curriculum/analysis/`.

## Stress-grid evaluation

The `stress_evaluation` target performs a headless paired evaluation over
force factors `1, 1.5, 2, 3` and mass factors `1, 1.05, 1.10, 1.20`. Its CLI is:

```text
stress_evaluation OUTPUT.csv LABEL SEED CHECKPOINT [LABEL SEED CHECKPOINT ...]
```

Each checkpoint-condition pair uses 1,000 episodes. For a given seed, every
policy and grid condition reuses the same initial-state and standardized
disturbance samples. Aggregate a completed CSV with:

```powershell
python experiments/adaptive_curriculum/analyze_stress.py
```

