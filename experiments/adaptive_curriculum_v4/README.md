# Adaptive curriculum v4

V4 keeps the v3 nominal plus severe-stress validation and adds:

- immediate stage rollback after one retention failure;
- stress retention length thresholds of 350, 375, 400, and 425;
- actor checkpoints every 25k interactions;
- post-training checkpoint selection subject to both nominal precision and
  severe-stress survival constraints.

The denser checkpoints are required because changing the curriculum stage does
not restore policy weights. Selecting the best feasible pre-collapse actor is
therefore separate from rolling the training environment back.

Build target: `adaptive_curriculum_v4_training`.

## Reproduced result

Three seeds were trained to 400k interactions. Seeds 0 and 1 promoted to stage
1 and immediately rolled back when stress retention failed; seed 2 remained at
stage 0. Run checkpoint selection with:

```powershell
& "C:\Users\oweng\anaconda3\envs\virtualgf-lora\python.exe" select_checkpoints.py
```

The validation-only selector chose steps 300k, 275k, and 300k for seeds 0, 1,
and 2. The selected ensemble improved all four aggregate metrics relative to
the fixed 300k adaptive checkpoint, but it remains a robustness/precision
trade-off relative to the original baseline. See `analysis/` and the main
report under `reports/` for exact results.

