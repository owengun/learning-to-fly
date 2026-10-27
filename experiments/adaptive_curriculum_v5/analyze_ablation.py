#!/usr/bin/env python3
"""Summarize the seed-2 post-promotion v5 ablation pilot."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
V5 = ROOT / "experiments" / "adaptive_curriculum_v5"
OUTPUT = V5 / "analysis"
METRICS = ["return_mean", "episode_length_mean", "mean_position_error", "success_rate"]


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def main() -> None:
    rows = read(V5 / "heldout_ablation_seed2_raw.csv")
    full = [row for row in read(V5 / "heldout_stress_v5_400_raw.csv") if row["seed"] == "2"]
    for row in full:
        row["model"] = "full_v5_400"
    rows.extend(full)
    models = ["mass_only_400", "force_only_400", "force_mass_fixed_reward_400", "full_v5_400"]
    result = {
        "scope": "single-seed diagnostic; not an inferential multi-seed ablation",
        "seed": 2,
        "post_promotion_training_steps": 50000,
        "stage_1_mass_factor": 1.0,
        "aggregate": {},
    }
    for model in models:
        subset = [row for row in rows if row["model"] == model]
        result["aggregate"][model] = {}
        for metric in METRICS:
            result["aggregate"][model][metric] = statistics.mean(float(row[metric]) for row in subset)
        result["aggregate"][model]["return_per_step"] = statistics.mean(
            float(row["return_mean"]) / float(row["episode_length_mean"]) for row in subset
        )
    force_rows = [row for row in rows if row["model"] == "force_only_400"]
    force_mass_rows = [row for row in rows if row["model"] == "force_mass_fixed_reward_400"]
    result["force_only_equals_force_mass_fixed_reward"] = all(
        all(a[metric] == b[metric] for metric in METRICS)
        for a, b in zip(force_rows, force_mass_rows)
    )
    result["interpretation"] = (
        "Mass is unchanged in stage 1, so force-only and force+mass/fixed-reward are identical. "
        "Full v5 differs through its reward change; mass effects remain unidentifiable until stage 2 is reached."
    )
    (OUTPUT / "v5_ablation_seed2_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
