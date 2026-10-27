#!/usr/bin/env python3
"""Summarize v5 stage transitions and curriculum exposure of selected actors."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
V5 = ROOT / "experiments" / "adaptive_curriculum_v5"
OUTPUT = V5 / "analysis"


def main() -> None:
    selected = list(csv.DictReader((OUTPUT / "v5_selected_checkpoints.csv").open(encoding="utf-8-sig")))
    runs = []
    for row in selected:
        run_name = row["run_name"]
        transition_path = V5 / "run" / f"curriculum_transitions_{run_name}.csv"
        transitions = []
        if transition_path.exists():
            with transition_path.open(encoding="utf-8-sig", newline="") as handle:
                transitions = list(csv.DictReader(handle))
        selected_step = int(row["step"])
        first_transition = min((int(item["step"]) for item in transitions), default=None)
        # Checkpoint saving occurs before adaptive_validation_and_gate at the same step.
        selected_was_curriculum_exposed = first_transition is not None and first_transition < selected_step
        runs.append(
            {
                "seed": int(row["seed"]),
                "run_name": run_name,
                "selected_step": selected_step,
                "first_transition_step": first_transition,
                "selected_was_curriculum_exposed": selected_was_curriculum_exposed,
                "transitions": transitions,
            }
        )
    result = {
        "checkpoint_order": "checkpoint is saved before validation/gate at the same step",
        "selected_models_with_curriculum_exposure": sum(run["selected_was_curriculum_exposed"] for run in runs),
        "selected_models_total": len(runs),
        "causal_interpretation": "Selected actors test precision-constrained validation selection and removal of the fixed curriculum, not force/mass/reward curriculum exposure.",
        "runs": runs,
    }
    (OUTPUT / "v5_training_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
