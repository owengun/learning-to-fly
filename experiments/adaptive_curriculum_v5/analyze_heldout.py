#!/usr/bin/env python3
"""Analyze paired five-seed held-out baseline versus v5 results."""

from __future__ import annotations

import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW_FILES = [
    ROOT / "experiments" / "adaptive_curriculum_v5" / "heldout_stress_v5_raw.csv",
    ROOT / "experiments" / "adaptive_curriculum_v5" / "heldout_stress_v5_400_raw.csv",
]
OUTPUT = ROOT / "experiments" / "adaptive_curriculum_v5" / "analysis"
RAW_METRICS = ["return_mean", "episode_length_mean", "mean_position_error", "success_rate"]
METRICS = [*RAW_METRICS, "return_per_step"]
MODELS = ["baseline5", "v5_selected", "v5_400"]


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_ci(differences: list[float], samples: int = 100_000) -> list[float]:
    rng = random.Random(20261026)
    boot = [statistics.mean(rng.choice(differences) for _ in differences) for _ in range(samples)]
    return [percentile(boot, 0.025), percentile(boot, 0.975)]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in RAW_FILES:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for source in csv.DictReader(handle):
                row = {
                    "model": source["model"],
                    "seed": int(source["seed"]),
                    "force_factor": float(source["force_factor"]),
                    "mass_factor": float(source["mass_factor"]),
                    **{metric: float(source[metric]) for metric in RAW_METRICS},
                }
                row["return_per_step"] = row["return_mean"] / row["episode_length_mean"]
                rows.append(row)
    if len(rows) != 240:
        raise RuntimeError(f"Expected 240 rows, found {len(rows)}")

    seed_means = defaultdict(dict)
    for model in MODELS:
        for seed in range(5):
            subset = [row for row in rows if row["model"] == model and row["seed"] == seed]
            if len(subset) != 16:
                raise RuntimeError(f"Expected 16 rows for {model} seed {seed}; found {len(subset)}")
            seed_means[model][seed] = {
                metric: statistics.mean(row[metric] for row in subset) for metric in METRICS
            }

    result = {
        "seeds": 5,
        "heldout_conditions": 16,
        "episodes_per_checkpoint_condition": 1000,
        "selection_used_heldout_grid": False,
        "aggregate": {},
        "paired_v5_minus_baseline": {},
    }
    for model in MODELS:
        result["aggregate"][model] = {}
        for metric in METRICS:
            values = [seed_means[model][seed][metric] for seed in range(5)]
            result["aggregate"][model][metric] = {
                "mean": statistics.mean(values),
                "sample_std": statistics.stdev(values),
            }

    paired_rows = []
    for seed in range(5):
        output = {"seed": seed}
        for metric in METRICS:
            baseline = seed_means["baseline5"][seed][metric]
            v5 = seed_means["v5_selected"][seed][metric]
            output[f"baseline_{metric}"] = baseline
            output[f"v5_{metric}"] = v5
            output[f"difference_{metric}"] = v5 - baseline
        paired_rows.append(output)

    for metric in METRICS:
        differences = [row[f"difference_{metric}"] for row in paired_rows]
        baseline = result["aggregate"]["baseline5"][metric]["mean"]
        result["paired_v5_minus_baseline"][metric] = {
            "mean_difference": statistics.mean(differences),
            "bootstrap_95_percent_ci": bootstrap_ci(differences),
            "percent_change": 100 * statistics.mean(differences) / abs(baseline),
            "seed_differences": differences,
            "seeds_improved": sum(
                difference < 0 if metric == "mean_position_error" else difference > 0
                for difference in differences
            ),
        }

    by_condition = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_condition[(row["force_factor"], row["mass_factor"])][row["model"]].append(row)
    condition_rows = []
    for (force, mass), models in sorted(by_condition.items()):
        output = {"force_factor": force, "mass_factor": mass}
        for model in MODELS:
            for metric in METRICS:
                output[f"{model}_{metric}"] = statistics.mean(row[metric] for row in models[model])
        condition_rows.append(output)
    result["condition_wins"] = {
        "episode_length": sum(row["v5_selected_episode_length_mean"] > row["baseline5_episode_length_mean"] for row in condition_rows),
        "success_rate": sum(row["v5_selected_success_rate"] > row["baseline5_success_rate"] for row in condition_rows),
        "position_error": sum(row["v5_selected_mean_position_error"] < row["baseline5_mean_position_error"] for row in condition_rows),
        "return_per_step": sum(row["v5_selected_return_per_step"] > row["baseline5_return_per_step"] for row in condition_rows),
    }
    result["v5_400_condition_wins_vs_baseline"] = {
        "episode_length": sum(row["v5_400_episode_length_mean"] > row["baseline5_episode_length_mean"] for row in condition_rows),
        "success_rate": sum(row["v5_400_success_rate"] > row["baseline5_success_rate"] for row in condition_rows),
        "position_error": sum(row["v5_400_mean_position_error"] < row["baseline5_mean_position_error"] for row in condition_rows),
        "return_per_step": sum(row["v5_400_return_per_step"] > row["baseline5_return_per_step"] for row in condition_rows),
    }
    result["selected_minus_v5_400"] = {}
    for metric in METRICS:
        differences = [
            seed_means["v5_selected"][seed][metric] - seed_means["v5_400"][seed][metric]
            for seed in range(5)
        ]
        result["selected_minus_v5_400"][metric] = {
            "mean_difference": statistics.mean(differences),
            "seed_differences": differences,
        }
    result["worst_condition_force3p5_mass1p25"] = next(
        row for row in condition_rows if row["force_factor"] == 3.5 and row["mass_factor"] == 1.25
    )

    with (OUTPUT / "v5_heldout_paired_by_seed.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    with (OUTPUT / "v5_heldout_by_condition.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(condition_rows[0]))
        writer.writeheader()
        writer.writerows(condition_rows)
    (OUTPUT / "v5_heldout_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
