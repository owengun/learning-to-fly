#!/usr/bin/env python3
"""Select precision-feasible v5 checkpoints without reading test-grid data."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[2]
V5 = ROOT / "experiments" / "adaptive_curriculum_v5"
LOGS = V5 / "run" / "logs"
CHECKPOINTS = V5 / "run" / "checkpoints" / "multirotor_td3"
OUTPUT = V5 / "analysis"
TAGS = {
    "nominal_length": "adaptive_curriculum/gate_episode_length",
    "nominal_position_error": "adaptive_curriculum/gate_mean_position_error",
    "nominal_return_per_step": "adaptive_curriculum/gate_return_per_step",
    "stress_length": "adaptive_curriculum/stress_episode_length",
    "stress_position_error": "adaptive_curriculum/stress_mean_position_error",
    "stress_return_per_step": "adaptive_curriculum/stress_return_per_step",
}


def latest_runs() -> dict[int, Path]:
    selected = {}
    for event_file in LOGS.glob("*_adaptive_v5/data.tfevents"):
        match = re.search(r"_(\d{3})_adaptive_v5$", event_file.parent.name)
        if match:
            seed = int(match.group(1))
            final_checkpoint = CHECKPOINTS / event_file.parent.name / "actor_000000000400000.h5"
            if not final_checkpoint.exists():
                continue
            if seed not in selected or event_file.stat().st_mtime > selected[seed].stat().st_mtime:
                selected[seed] = event_file
    return selected


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    runs = latest_runs()
    if set(runs) != set(range(5)):
        raise RuntimeError(f"Expected seeds 0..4; found {sorted(runs)}")

    candidates, selected_rows, infeasible_seeds = [], [], []
    for seed, event_file in sorted(runs.items()):
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        values = {
            name: {item.step: float(item.value) for item in accumulator.Scalars(tag)}
            for name, tag in TAGS.items()
        }
        common_steps = set.intersection(*(set(metric) for metric in values.values()))
        seed_rows = []
        for step in sorted(common_steps):
            if step < 200_000:
                continue
            row = {
                "seed": seed,
                "run_name": event_file.parent.name,
                "step": step,
                **{name: metric[step] for name, metric in values.items()},
            }
            row["precision_feasible"] = (
                row["nominal_position_error"] <= 0.305
                and row["nominal_length"] >= 250
                and row["stress_length"] >= 325
                and row["stress_position_error"] <= 0.45
            )
            row["tie_break_score"] = min(row["nominal_length"], row["stress_length"]) - 100 * (
                row["nominal_position_error"] + row["stress_position_error"]
            )
            candidates.append(row)
            seed_rows.append(row)

        feasible = [row for row in seed_rows if row["precision_feasible"]]
        if not feasible:
            closest = min(
                seed_rows,
                key=lambda row: (
                    max(0.0, row["nominal_position_error"] - 0.305)
                    + max(0.0, 325 - row["stress_length"]) / 500,
                    -row["stress_length"],
                ),
            )
            infeasible_seeds.append({"seed": seed, "closest_candidate": closest})
            continue
        # Hard precision feasibility first, then maximum severe-stress survival.
        winner = max(feasible, key=lambda row: (row["stress_length"], row["tie_break_score"], -row["step"]))
        checkpoint = CHECKPOINTS / winner["run_name"] / f"actor_{winner['step']:015d}.h5"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        selected_rows.append({**winner, "checkpoint": checkpoint.relative_to(ROOT).as_posix()})

    with (OUTPUT / "v5_checkpoint_candidates.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    if infeasible_seeds:
        diagnostic = {
            "selection_succeeded": False,
            "reason": "At least one seed has no checkpoint satisfying every hard constraint",
            "constraints_were_not_relaxed": True,
            "infeasible_seeds": infeasible_seeds,
        }
        (OUTPUT / "v5_selection_failure.json").write_text(
            json.dumps(diagnostic, indent=2), encoding="utf-8"
        )
        print(json.dumps(diagnostic, indent=2))
        raise RuntimeError(
            f"No precision-feasible checkpoint for seeds {[item['seed'] for item in infeasible_seeds]}"
        )
    with (OUTPUT / "v5_selected_checkpoints.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected_rows[0]))
        writer.writeheader()
        writer.writerows(selected_rows)
    (OUTPUT / "v5_selected_checkpoints.json").write_text(
        json.dumps(
            {
                "selection_uses_test_grid": False,
                "hard_constraints": {
                    "nominal_position_error_max": 0.305,
                    "nominal_length_min": 250,
                    "stress_length_min": 325,
                    "stress_position_error_max": 0.45,
                },
                "ranking": "maximum stress_length; robustness score and earlier step as tie-breakers",
                "selected": selected_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(selected_rows, indent=2))


if __name__ == "__main__":
    main()
