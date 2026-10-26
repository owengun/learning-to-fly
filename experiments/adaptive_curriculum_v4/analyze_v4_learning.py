#!/usr/bin/env python3
"""Summarize v4 learning curves and curriculum transitions."""

from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "experiments" / "adaptive_curriculum_v4" / "analysis"


def latest_runs() -> dict[int, Path]:
    selected = {}
    logs = ROOT / "experiments" / "adaptive_curriculum_v4" / "run" / "logs"
    for path in logs.glob("*_adaptive_v4/data.tfevents"):
        match = re.search(r"_(\d{3})_adaptive_v4$", path.parent.name)
        if match:
            seed = int(match.group(1))
            if seed not in selected or path.stat().st_mtime > selected[seed].stat().st_mtime:
                selected[seed] = path
    return selected


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed, path in sorted(latest_runs().items()):
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
        accumulator.Reload()
        scalar = lambda tag: {item.step: float(item.value) for item in accumulator.Scalars(tag)}
        returns = scalar("evaluation/returns_mean")
        lengths = scalar("evaluation/episode_length_mean")
        stages = scalar("adaptive_curriculum/stage")
        transitions = [
            {"step": step, "from": int(stages[previous]), "to": int(stages[step])}
            for previous, step in zip(sorted(stages)[:-1], sorted(stages)[1:])
            if stages[previous] != stages[step]
        ]
        rows.append(
            {
                "seed": seed,
                "run_name": path.parent.name,
                "return_290k": returns[290000],
                "length_290k": lengths[290000],
                "return_390k": returns[390000],
                "length_390k": lengths[390000],
                "max_stage": int(max(stages.values())),
                "final_stage": int(stages[max(stages)]),
                "transitions": transitions,
            }
        )

    def summary(field: str) -> dict:
        values = [row[field] for row in rows]
        return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values)}

    result = {
        "seeds": 3,
        "v4_290k": {"return": summary("return_290k"), "episode_length": summary("length_290k")},
        "v4_390k": {"return": summary("return_390k"), "episode_length": summary("length_390k")},
        "runs": rows,
    }
    csv_rows = [{**{k: v for k, v in row.items() if k != "transitions"}, "transitions": json.dumps(row["transitions"])} for row in rows]
    with (OUTPUT / "v4_learning_comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    (OUTPUT / "v4_learning_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
