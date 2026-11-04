# Audit of the original MLP baseline

The previously saved original training_mlp run contained only seed 0 and
30,041 steps, with the stock validation path. It did not cover the three
seeds, 50,001-step budget and fixed evaluation protocol used in the matched
experiment. Three supplemental original-MLP runs have now completed.

Each supplemental run uses the original 146 -> 64 -> 64 -> 4 RLtools actor,
FAST_TANH, Adam and the stock TD3 training loop. The new code serializes
actor and target weights without consuming training RNG and adds standalone
evaluation. The seed-0 stock validation trace through step 30,000 exactly
matches the earlier original run.

Both experiment sets use training seeds 0, 1, 2 and 50,001 environment steps.
Each stored actor at steps 0, 10000, 20000, 30000, 40000 and 50000 is evaluated
in a fresh process with its training seed, the same evaluation environment,
ten episodes, a 500-step limit, true-termination stopping and no action
exploration noise. Evaluation RNG starts from the fresh init state in both
sets. The audit verified all 180 supplemental episode returns and 36
actor/target checkpoint hashes. All training and evaluation exit codes are zero.

## Original repository MLP baseline

| Method | Actor parameters | Final mean | Between-seed SD | Mean best checkpoint return | Mean normalized AUC | Mean training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original MLP, 146 inputs | 13828 | -202.31 | 19.01 | -143.34 | -163.76 | 62.81 |

| Seed | Final mean | Final episode SD | Best mean | Best step | Normalized AUC | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | -223.84 | 60.99 | -125.27 | 0 | -155.50 | 62.65 |
| 1 | -195.25 | 42.22 | -145.95 | 0 | -161.54 | 62.53 |
| 2 | -187.83 | 44.76 | -158.82 | 0 | -174.23 | 63.24 |

## Controlled 18-input comparison

| Method | Actor parameters | Final mean | Between-seed SD | Mean best checkpoint return | Mean normalized AUC | Mean training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Controlled MLP, 18 inputs | 5636 | -143.57 | 6.20 | -134.34 | -149.74 | 36.71 |
| Vanilla RNN | 1764 | -149.22 | 6.62 | -139.66 | -142.80 | 52.20 |
| LSTM | 6660 | -143.52 | 14.97 | -130.77 | -146.01 | 93.54 |
| GRU | 5028 | -141.52 | 15.46 | -122.78 | -132.65 | 73.65 |

Final mean averages the three seed-level evaluation means. Between-seed SD
uses those three means. Best checkpoint return is selected among six means
within each seed, then averaged. AUC integrates the six checkpoint means
and divides by the 50,000-step span. Higher return and AUC are better.

The original MLP final mean is -202.31. The three recurrent actors
have higher observed final means in this short experiment. The earlier
four-row table supported comparisons within the controlled 18-input protocol.
The supplemental results now cover the original method under the same
evaluation budget. Differences from the original method include observation
history, actor optimizer, architecture and replay sampling, so this comparison
measures the whole configuration change. Three seeds and 50,001 steps do not
establish convergence or a general advantage for recurrent architectures.

Reproduce the supplemental runs from the repository root after building
training_mlp and test_original_mlp_checkpoint in Release:

    python -B experiments/rnn/run_original_mlp.py --exe build2026/src/Release/training_mlp.exe --output build2026/original-mlp-audit-750e30ef/results --steps 50001 --episodes 10 --seeds 0 1 2

Raw data are in build2026/original-mlp-audit-750e30ef/results/comparison.csv.
Each per-seed attempt contains run.json, exact commands, episode returns,
exit codes, checkpoint hashes and elapsed time. The persistence test covers
bit-exact weight roundtrip and rejection of a truncated file without mutation.
Existing recurrent branch tips remain 9272cec, 5de00aa and d9c1ec4.
