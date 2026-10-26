#!/usr/bin/env python3
"""Compare adaptive-curriculum runs with the three-seed baseline."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[2]
ADAPTIVE_LOGS = ROOT / "experiments" / "adaptive_curriculum" / "run" / "logs"
BASELINE_CURVES = ROOT / "reproduction" / "analysis" / "generated" / "evaluation_curves.csv"
OUTPUT = ROOT / "experiments" / "adaptive_curriculum" / "analysis"


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def first_threshold(series: dict[int, float], threshold: float) -> int | None:
    return next((step for step in sorted(series) if series[step] >= threshold), None)


def convergence(series: dict[int, float], threshold: float = 475.0, consecutive: int = 2) -> int | None:
    streak: list[int] = []
    for step in sorted(series):
        if series[step] >= threshold:
            streak.append(step)
            if len(streak) == consecutive:
                return streak[0]
        else:
            streak = []
    return None


def normalized_auc(series: dict[int, float], end_step: int) -> float:
    points = [(step, value) for step, value in sorted(series.items()) if step <= end_step]
    area = sum((b_step - a_step) * (a_value + b_value) / 2 for (a_step, a_value), (b_step, b_value) in zip(points, points[1:]))
    return area / end_step


def load_adaptive() -> dict[int, dict]:
    runs: dict[int, dict] = {}
    for event_file in sorted(ADAPTIVE_LOGS.glob("*_adaptive_wind_mass/data.tfevents")):
        match = re.search(r"_(\d{3})_adaptive_wind_mass$", event_file.parent.name)
        if not match:
            continue
        seed = int(match.group(1))
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        tags = set(accumulator.Tags()["scalars"])

        def scalar(tag: str) -> dict[int, float]:
            return {event.step: float(event.value) for event in accumulator.Scalars(tag)} if tag in tags else {}

        runs[seed] = {
            "run_name": event_file.parent.name,
            "return": scalar("evaluation/returns_mean"),
            "length": scalar("evaluation/episode_length_mean"),
            "stage": scalar("adaptive_curriculum/stage"),
        }
    return runs


def load_baseline() -> dict[int, dict]:
    runs: dict[int, dict] = {}
    with BASELINE_CURVES.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["configuration"] != "Baseline":
                continue
            seed, step = int(row["seed"]), int(row["step"])
            run = runs.setdefault(seed, {"return": {}, "length": {}})
            run["return"][step] = float(row["return_mean"])
            run["length"][step] = float(row["episode_length_mean"])
    return runs


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_svg(path: Path, baseline: dict[int, dict], adaptive: dict[int, dict], end_step: int) -> None:
    width, height = 1120, 480
    panels = [("Evaluation return", "return", -1300.0, 500.0), ("Episode length", "length", 0.0, 500.0)]
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>', '<style>text{font-family:Arial,sans-serif;fill:#222}.grid{stroke:#ddd}.axis{stroke:#444}.line{fill:none;stroke-width:3}</style>']
    for panel_index, (title, metric, y_min, y_max) in enumerate(panels):
        x0, y0, pw, ph = 65 + panel_index * 550, 55, 465, 340
        svg.append(f'<text x="{x0 + pw/2}" y="30" text-anchor="middle" font-size="17" font-weight="bold">{title}</text>')
        for i in range(6):
            y = y0 + ph - ph * i / 5
            value = y_min + (y_max - y_min) * i / 5
            svg.append(f'<line class="grid" x1="{x0}" y1="{y}" x2="{x0+pw}" y2="{y}"/><text x="{x0-8}" y="{y+4}" text-anchor="end" font-size="11">{value:.0f}</text>')
        for i in range(7):
            x = x0 + pw * i / 6
            step = end_step * i / 6
            svg.append(f'<text x="{x}" y="{y0+ph+20}" text-anchor="middle" font-size="11">{step/1000:.0f}k</text>')
        svg.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+ph}"/><line class="axis" x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0+ph}"/>')
        for label, runs, color in [("Baseline", baseline, "#1f77b4"), ("Adaptive", adaptive, "#d62728")]:
            common_steps = sorted(set.intersection(*(set(run[metric]) for run in runs.values())))
            coords = []
            for step in common_steps:
                if step > end_step:
                    continue
                value = statistics.mean(run[metric][step] for run in runs.values())
                x = x0 + pw * step / end_step
                y = y0 + ph - ph * (value - y_min) / (y_max - y_min)
                coords.append(f"{x:.1f},{y:.1f}")
            svg.append(f'<polyline class="line" stroke="{color}" points="{" ".join(coords)}"/>')
    svg.extend(['<line x1="415" y1="447" x2="445" y2="447" stroke="#1f77b4" stroke-width="3"/><text x="452" y="451" font-size="13">Baseline</text>', '<line x1="565" y1="447" x2="595" y2="447" stroke="#d62728" stroke-width="3"/><text x="602" y="451" font-size="13">Adaptive wind/mass curriculum</text>', '</svg>'])
    path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    adaptive, baseline = load_adaptive(), load_baseline()
    if set(adaptive) != {0, 1, 2} or set(baseline) != {0, 1, 2}:
        raise RuntimeError(f"Expected seeds 0,1,2; adaptive={sorted(adaptive)}, baseline={sorted(baseline)}")
    common_steps = set.intersection(*(set(run["length"]) for run in [*adaptive.values(), *baseline.values()]))
    comparison_step = max(common_steps)

    curve_rows = []
    for seed, run in adaptive.items():
        for step in sorted(run["length"]):
            curve_rows.append({"seed": seed, "step": step, "return_mean": run["return"].get(step, math.nan), "episode_length_mean": run["length"][step], "stage": run["stage"].get(step, math.nan)})
    write_csv(OUTPUT / "adaptive_evaluation_curves.csv", curve_rows, ["seed", "step", "return_mean", "episode_length_mean", "stage"])

    rows = []
    for seed in range(3):
        b, a = baseline[seed], adaptive[seed]
        first_stage_1 = next((step for step in sorted(a["stage"]) if a["stage"][step] >= 1), None)
        first_stage_2 = next((step for step in sorted(a["stage"]) if a["stage"][step] >= 2), None)
        rows.append({
            "seed": seed,
            "comparison_step": comparison_step,
            "baseline_return": b["return"][comparison_step],
            "adaptive_return": a["return"][comparison_step],
            "return_delta": a["return"][comparison_step] - b["return"][comparison_step],
            "baseline_length": b["length"][comparison_step],
            "adaptive_length": a["length"][comparison_step],
            "length_delta": a["length"][comparison_step] - b["length"][comparison_step],
            "baseline_length_auc": normalized_auc(b["length"], comparison_step),
            "adaptive_length_auc": normalized_auc(a["length"], comparison_step),
            "baseline_first_400": first_threshold(b["length"], 400),
            "adaptive_first_400": first_threshold(a["length"], 400),
            "baseline_first_450": first_threshold(b["length"], 450),
            "adaptive_first_450": first_threshold(a["length"], 450),
            "baseline_converged_475x2": convergence(b["length"]),
            "adaptive_converged_475x2": convergence(a["length"]),
            "adaptive_first_stage_1": first_stage_1,
            "adaptive_first_stage_2": first_stage_2,
            "adaptive_final_stage": int(max(a["stage"].values())),
        })
    fields = list(rows[0])
    write_csv(OUTPUT / "baseline_comparison.csv", rows, fields)

    summary = {"comparison_step": comparison_step, "seeds": 3}
    for prefix, field in [("baseline_return", "baseline_return"), ("adaptive_return", "adaptive_return"), ("return_delta", "return_delta"), ("baseline_length", "baseline_length"), ("adaptive_length", "adaptive_length"), ("length_delta", "length_delta"), ("baseline_length_auc", "baseline_length_auc"), ("adaptive_length_auc", "adaptive_length_auc")]:
        mean, std = mean_std([float(row[field]) for row in rows])
        summary[prefix] = {"mean": mean, "sample_std": std}
    summary["return_improved_seed_count"] = sum(row["return_delta"] > 0 for row in rows)
    summary["length_improved_seed_count"] = sum(row["length_delta"] > 0 for row in rows)
    summary["baseline_converged_seed_count"] = sum(row["baseline_converged_475x2"] is not None for row in rows)
    summary["adaptive_converged_seed_count"] = sum(row["adaptive_converged_475x2"] is not None for row in rows)
    (OUTPUT / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_svg(OUTPUT / "baseline_vs_adaptive.svg", baseline, adaptive, comparison_step)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
