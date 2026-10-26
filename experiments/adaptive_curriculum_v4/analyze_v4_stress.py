#!/usr/bin/env python3
"""Aggregate all stress-grid experiments through adaptive curriculum v4."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "experiments" / "adaptive_curriculum_v4" / "analysis"
INPUTS = [
    ROOT / "experiments" / "adaptive_curriculum" / "analysis" / "stress_grid_raw.csv",
    ROOT / "experiments" / "adaptive_curriculum_v2" / "stress_grid_v2_raw.csv",
    ROOT / "experiments" / "adaptive_curriculum_v3" / "stress_grid_v3_raw.csv",
    ROOT / "experiments" / "adaptive_curriculum_v4" / "stress_grid_v4_raw.csv",
]
MODELS = ["baseline", "v1", "v2_300", "v2_400", "v3_300", "v3_400", "v4_300", "v4_selected"]
RAW_METRICS = ["return_mean", "episode_length_mean", "mean_position_error", "success_rate"]
METRICS = [*RAW_METRICS, "return_per_step"]


def load() -> list[dict]:
    rows = []
    for path in INPUTS:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for source in csv.DictReader(handle):
                model = "v1" if source["model"] == "adaptive" else source["model"]
                row = {
                    "model": model,
                    "seed": int(source["seed"]),
                    "force_factor": round(float(source["force_factor"]), 2),
                    "mass_factor": round(float(source["mass_factor"]), 2),
                    **{metric: float(source[metric]) for metric in RAW_METRICS},
                }
                row["return_per_step"] = row["return_mean"] / row["episode_length_mean"]
                rows.append(row)
    return rows


def mean_std(values: list[float]) -> dict:
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values)}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = load()
    non_nominal = [row for row in rows if (row["force_factor"], row["mass_factor"]) != (1.0, 1.0)]
    result = {"seeds": 3, "conditions": 16, "episodes_per_checkpoint_condition": 1000, "non_nominal": {}}
    for model in MODELS:
        result["non_nominal"][model] = {}
        for metric in METRICS:
            seed_means = [
                statistics.mean(row[metric] for row in non_nominal if row["model"] == model and row["seed"] == seed)
                for seed in range(3)
            ]
            result["non_nominal"][model][metric] = mean_std(seed_means)

    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[(row["force_factor"], row["mass_factor"])][row["model"]].append(row)
    summary_rows = []
    for (force, mass), by_model in sorted(grouped.items()):
        output = {"force_factor": force, "mass_factor": mass}
        for model in MODELS:
            for metric in METRICS:
                output[f"{model}_{metric}"] = statistics.mean(item[metric] for item in by_model[model])
        summary_rows.append(output)

    for model in MODELS[1:]:
        result[f"{model}_vs_baseline_condition_wins"] = {
            "episode_length": sum(row[f"{model}_episode_length_mean"] > row["baseline_episode_length_mean"] for row in summary_rows),
            "success_rate": sum(row[f"{model}_success_rate"] > row["baseline_success_rate"] for row in summary_rows),
            "position_error": sum(row[f"{model}_mean_position_error"] < row["baseline_mean_position_error"] for row in summary_rows),
            "return_per_step": sum(row[f"{model}_return_per_step"] > row["baseline_return_per_step"] for row in summary_rows),
        }

    result["v4_selected_vs_baseline_percent_change"] = {}
    for metric in METRICS:
        baseline = result["non_nominal"]["baseline"][metric]["mean"]
        selected = result["non_nominal"]["v4_selected"][metric]["mean"]
        result["v4_selected_vs_baseline_percent_change"][metric] = 100 * (selected - baseline) / abs(baseline)
    result["v4_selected_vs_v4_300_percent_change"] = {}
    for metric in METRICS:
        fixed = result["non_nominal"]["v4_300"][metric]["mean"]
        selected = result["non_nominal"]["v4_selected"][metric]["mean"]
        result["v4_selected_vs_v4_300_percent_change"][metric] = 100 * (selected - fixed) / abs(fixed)

    result["worst_condition_force3_mass1p2"] = next(
        row for row in summary_rows if (row["force_factor"], row["mass_factor"]) == (3.0, 1.2)
    )

    with (OUTPUT / "v4_stress_comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    (OUTPUT / "v4_stress_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
