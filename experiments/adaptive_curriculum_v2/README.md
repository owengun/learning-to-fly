# Adaptive curriculum v2

This experiment preserves v1 and changes only the curriculum controller.

## Changes from v1

- Stage promotion requires three consecutive 25k validations.
- The first promotion requires episode length >= 250, position error <= 0.40 m,
  and return per step >= -1.0.
- Force and mass changes are split across stages.
- Two consecutive retention failures trigger rollback to the previous stage.
- Reward weights are rebuilt from their initial values on every transition, so
  rollback also restores the correct reward stage.

| Stage | Force std | Mass factor | Reward promotions |
|---|---:|---:|---:|
| 0 | x1.0 | 1.00 | 0 |
| 1 | x1.5 | 1.00 | 1 |
| 2 | x1.5 | Uniform(1.00, 1.05) | 1 |
| 3 | x2.0 | Uniform(1.00, 1.10) | 2 |
| 4 | x3.0 | Uniform(1.00, 1.20) | 2 |

Build target: `adaptive_curriculum_v2_training`.

The v2 run lasts 400k interactions. A 300k checkpoint is retained for the
budget-matched baseline comparison; the additional 100k interactions expose
the policy to the post-promotion stage selected by the stricter gate.

