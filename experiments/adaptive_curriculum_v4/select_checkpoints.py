#!/usr/bin/env python3
"""Select one v4 actor per seed using only in-training validation metrics.

The stress-grid test set is deliberately not read here.  Selection therefore
does not leak any of the final 16-condition evaluation into model choice.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[2]
V4 = ROOT / "experiments" / "adaptive_curriculum_v4"
LOGS = V4 / "run" / "logs"
CHECKPOINTS = V4 / "run" / "checkpoints" / "multirotor_td3"
OUTPUT = V4 / "analysis"

TAGS = {
    "nominal_length": "adaptive_curriculum/gate_episode_length",
    "nominal_position_error": "adaptive_curriculum/gate_mean_position_error",
    "nominal_return_per_step": "adaptive_curriculum/gate_return_per_step",
    "stress_length": "adaptive_curriculum/stress_episode_length",
    "stress_position_error": "adaptive_curriculum/stress_mean_position_error",
    "stress_return_per_step": "adaptive_curriculum/stress_return_per_step",
}


def latest_runs() -> dict[int, Path]:
    selected: dict[int, Path] = {}
    for event_file in LOGS.glob("*_adaptive_v4/data.tfevents"):
        match = re.search(r"_(\d{3})_adaptive_v4$", event_file.parent.name)
        if match:
            seed = int(match.group(1))
            if seed not in selected or event_file.stat().st_mtime > selected[seed].stat().st_mtime:
                selected[seed] = event_file
    return selected


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    runs = latest_runs()
    if set(runs) != {0, 1, 2}:
        raise RuntimeError(f"Expected seeds 0, 1, 2; found {sorted(runs)}")

    candidates: list[dict] = []
    selected_rows: list[dict] = []
    for seed, event_file in sorted(runs.items()):
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        values = {
            name: {item.step: float(item.value) for item in accumulator.Scalars(tag)}
            for name, tag in TAGS.items()
        }
        common_steps = set.intersection(*(set(metric) for metric in values.values()))
        for step in sorted(common_steps):
            # Exclude immature policies and require minimally useful nominal and
            # severe-stress behavior before ranking the remaining checkpoints.
            if step < 200_000:
                continue
            row = {
                "seed": seed,
                "run_name": event_file.parent.name,
                "step": step,
                **{name: metric[step] for name, metric in values.items()},
            }
            row["feasible"] = (
                row["nominal_length"] >= 250
                and row["nominal_position_error"] <= 0.40
                and row["stress_length"] >= 350
                and row["stress_position_error"] <= 0.50
            )
            # Conservative scalarization: reward the weaker of nominal/stress
            # survival and penalize error in both domains.
            row["selection_score"] = min(row["nominal_length"], row["stress_length"]) - 100 * (
                row["nominal_position_error"] + row["stress_position_error"]
            )
            candidates.append(row)

        seed_rows = [row for row in candidates if row["seed"] == seed]
        feasible = [row for row in seed_rows if row["feasible"]]
        pool = feasible or seed_rows
        winner = max(pool, key=lambda row: (row["selection_score"], -row["step"]))
        checkpoint = CHECKPOINTS / winner["run_name"] / f"actor_{winner['step']:015d}.h5"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        selected_rows.append({**winner, "checkpoint": checkpoint.relative_to(ROOT).as_posix()})

    with (OUTPUT / "v4_checkpoint_candidates.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    with (OUTPUT / "v4_selected_checkpoints.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected_rows[0]))
        writer.writeheader()
        writer.writerows(selected_rows)
    (OUTPUT / "v4_selected_checkpoints.json").write_text(
        json.dumps(
            {
                "selection_data": "in-training nominal and force x3 / mass x1.2 validation only",
                "test_grid_used_for_selection": False,
                "eligibility": {
                    "minimum_step": 200000,
                    "nominal_length_min": 250,
                    "nominal_position_error_max": 0.40,
                    "stress_length_min": 350,
                    "stress_position_error_max": 0.50,
                },
                "score": "min(nominal_length, stress_length) - 100*(nominal_position_error + stress_position_error)",
                "selected": selected_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(selected_rows, indent=2))


if __name__ == "__main__":
    main()
