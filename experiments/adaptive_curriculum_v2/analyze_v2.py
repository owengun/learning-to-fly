#!/usr/bin/env python3
"""Compare adaptive curriculum v2 with baseline and v1."""

from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "experiments" / "adaptive_curriculum_v2" / "analysis"


def mean_std(values: list[float]) -> dict:
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values)}


def load_baseline() -> dict[int, dict]:
    runs = {}
    path = ROOT / "reproduction" / "analysis" / "generated" / "evaluation_curves.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["configuration"] != "Baseline":
                continue
            run = runs.setdefault(int(row["seed"]), {"return": {}, "length": {}})
            step = int(row["step"])
            run["return"][step] = float(row["return_mean"])
            run["length"][step] = float(row["episode_length_mean"])
    return runs


def load_v1() -> dict[int, dict]:
    runs = {}
    path = ROOT / "experiments" / "adaptive_curriculum" / "analysis" / "adaptive_evaluation_curves.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            run = runs.setdefault(int(row["seed"]), {"return": {}, "length": {}})
            step = int(row["step"])
            run["return"][step] = float(row["return_mean"])
            run["length"][step] = float(row["episode_length_mean"])
    return runs


def load_v2() -> dict[int, dict]:
    selected = {}
    log_dir = ROOT / "experiments" / "adaptive_curriculum_v2" / "run" / "logs"
    for event_file in log_dir.glob("*_adaptive_v2/data.tfevents"):
        match = re.search(r"_(\d{3})_adaptive_v2$", event_file.parent.name)
        if not match:
            continue
        seed = int(match.group(1))
        if seed not in selected or event_file.stat().st_mtime > selected[seed].stat().st_mtime:
            selected[seed] = event_file
    runs = {}
    for seed, event_file in selected.items():
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        def scalar(tag: str) -> dict[int, float]:
            return {item.step: float(item.value) for item in accumulator.Scalars(tag)}
        runs[seed] = {
            "run_name": event_file.parent.name,
            "return": scalar("evaluation/returns_mean"),
            "length": scalar("evaluation/episode_length_mean"),
            "stage": scalar("adaptive_curriculum/stage"),
        }
    return runs


def aggregate(runs: dict[int, dict], step: int, metric: str) -> dict:
    return mean_std([runs[seed][metric][step] for seed in range(3)])


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    baseline, v1, v2 = load_baseline(), load_v1(), load_v2()
    if set(v2) != {0, 1, 2}:
        raise RuntimeError(f"Expected latest v2 seeds 0,1,2; got {sorted(v2)}")

    rows = []
    for seed in range(3):
        row = {"seed": seed}
        for label, runs in (("baseline", baseline), ("v1", v1), ("v2", v2)):
            row[f"{label}_return_290k"] = runs[seed]["return"][290000]
            row[f"{label}_length_290k"] = runs[seed]["length"][290000]
        row["v2_return_390k"] = v2[seed]["return"][390000]
        row["v2_length_390k"] = v2[seed]["length"][390000]
        row["v2_final_stage"] = int(max(v2[seed]["stage"].values()))
        row["v2_first_stage1"] = next((step for step in sorted(v2[seed]["stage"]) if v2[seed]["stage"][step] >= 1), None)
        row["v2_first_stage2"] = next((step for step in sorted(v2[seed]["stage"]) if v2[seed]["stage"][step] >= 2), None)
        rows.append(row)

    with (OUTPUT / "v2_learning_comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {"seeds": 3, "budget_matched_step": 290000, "extended_v2_step": 390000}
    for label, runs, step in (
        ("baseline_290k", baseline, 290000),
        ("v1_290k", v1, 290000),
        ("v2_290k", v2, 290000),
        ("v2_390k", v2, 390000),
    ):
        summary[label] = {
            "return": aggregate(runs, step, "return"),
            "episode_length": aggregate(runs, step, "length"),
        }
    summary["v2_transitions"] = [
        {"seed": row["seed"], "first_stage1": row["v2_first_stage1"], "first_stage2": row["v2_first_stage2"], "final_stage": row["v2_final_stage"]}
        for row in rows
    ]
    (OUTPUT / "v2_learning_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
