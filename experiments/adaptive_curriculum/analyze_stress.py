#!/usr/bin/env python3
"""Aggregate the paired baseline/adaptive stress-grid evaluation."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "experiments" / "adaptive_curriculum" / "analysis"
INPUT = ANALYSIS / "stress_grid_raw.csv"
OUTPUT_CSV = ANALYSIS / "stress_grid_summary.csv"
OUTPUT_JSON = ANALYSIS / "stress_summary.json"
OUTPUT_SVG = ANALYSIS / "stress_length_delta_heatmap.svg"
RAW_METRICS = ["return_mean", "episode_length_mean", "mean_position_error", "success_rate"]
METRICS = [*RAW_METRICS, "return_per_step"]


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    rows = []
    with INPUT.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed = {
                "model": row["model"],
                "seed": int(row["seed"]),
                "force_factor": round(float(row["force_factor"]), 2),
                "mass_factor": round(float(row["mass_factor"]), 2),
                **{metric: float(row[metric]) for metric in RAW_METRICS},
            }
            parsed["return_per_step"] = parsed["return_mean"] / parsed["episode_length_mean"]
            rows.append(parsed)

    grouped: dict[tuple[float, float], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[(row["force_factor"], row["mass_factor"])][row["model"]].append(row)

    summary_rows = []
    for (force, mass), models in sorted(grouped.items()):
        output = {"force_factor": force, "mass_factor": mass}
        for metric in METRICS:
            for model in ("baseline", "adaptive"):
                values = [row[metric] for row in models[model]]
                mean, std = mean_std(values)
                output[f"{model}_{metric}_mean"] = mean
                output[f"{model}_{metric}_std"] = std
            output[f"delta_{metric}"] = output[f"adaptive_{metric}_mean"] - output[f"baseline_{metric}_mean"]
        summary_rows.append(output)

    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    non_nominal = [row for row in rows if not (row["force_factor"] == 1.0 and row["mass_factor"] == 1.0)]
    per_seed_non_nominal = defaultdict(lambda: defaultdict(list))
    for row in non_nominal:
        for metric in METRICS:
            per_seed_non_nominal[(row["model"], row["seed"])][metric].append(row[metric])

    aggregate = {}
    for metric in METRICS:
        for model in ("baseline", "adaptive"):
            seed_means = [statistics.mean(per_seed_non_nominal[(model, seed)][metric]) for seed in range(3)]
            mean, std = mean_std(seed_means)
            aggregate[f"{model}_{metric}"] = {"mean": mean, "sample_std": std}
        aggregate[f"delta_{metric}"] = (
            aggregate[f"adaptive_{metric}"]["mean"] - aggregate[f"baseline_{metric}"]["mean"]
        )

    nominal = next(row for row in summary_rows if row["force_factor"] == 1.0 and row["mass_factor"] == 1.0)
    worst = next(row for row in summary_rows if row["force_factor"] == 3.0 and row["mass_factor"] == 1.2)
    result = {
        "episodes_per_checkpoint_condition": 1000,
        "seeds": 3,
        "conditions": len(summary_rows),
        "non_nominal_average": aggregate,
        "conditions_with_higher_adaptive_length": sum(row["delta_episode_length_mean"] > 0 for row in summary_rows),
        "conditions_with_higher_adaptive_success": sum(row["delta_success_rate"] > 0 for row in summary_rows),
        "conditions_with_lower_adaptive_position_error": sum(row["delta_mean_position_error"] < 0 for row in summary_rows),
        "conditions_with_higher_adaptive_return": sum(row["delta_return_mean"] > 0 for row in summary_rows),
        "conditions_with_higher_adaptive_return_per_step": sum(row["delta_return_per_step"] > 0 for row in summary_rows),
        "nominal": nominal,
        "worst_condition_force3_mass1p2": worst,
    }
    OUTPUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")

    forces = [1.0, 1.5, 2.0, 3.0]
    masses = [1.0, 1.05, 1.1, 1.2]
    values = [row["delta_episode_length_mean"] for row in summary_rows]
    max_abs = max(abs(value) for value in values) or 1
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="720" height="500">', '<rect width="100%" height="100%" fill="white"/>', '<style>text{font-family:Arial,sans-serif;fill:#222}</style>', '<text x="360" y="32" text-anchor="middle" font-size="19" font-weight="bold">Adaptive - baseline episode length (3-seed mean)</text>']
    x0, y0, cw, ch = 150, 75, 125, 80
    for col, mass in enumerate(masses):
        svg.append(f'<text x="{x0 + col*cw + cw/2}" y="60" text-anchor="middle" font-size="13">mass x{mass:g}</text>')
    for row_index, force in enumerate(forces):
        svg.append(f'<text x="{x0-12}" y="{y0 + row_index*ch + ch/2 + 5}" text-anchor="end" font-size="13">force x{force:g}</text>')
        for col, mass in enumerate(masses):
            item = next(row for row in summary_rows if row["force_factor"] == force and row["mass_factor"] == mass)
            value = item["delta_episode_length_mean"]
            intensity = min(abs(value) / max_abs, 1)
            if value >= 0:
                color = f"rgb({int(235-120*intensity)},{int(248-55*intensity)},{int(235-120*intensity)})"
            else:
                color = f"rgb({int(248-35*intensity)},{int(235-130*intensity)},{int(235-130*intensity)})"
            x, y = x0 + col*cw, y0 + row_index*ch
            svg.append(f'<rect x="{x}" y="{y}" width="{cw-4}" height="{ch-4}" rx="5" fill="{color}" stroke="#aaa"/>')
            svg.append(f'<text x="{x+(cw-4)/2}" y="{y+(ch-4)/2+6}" text-anchor="middle" font-size="17" font-weight="bold">{value:+.1f}</text>')
    svg.extend(['<text x="400" y="430" text-anchor="middle" font-size="12">Positive values mean longer adaptive-policy survival.</text>', '</svg>'])
    OUTPUT_SVG.write_text("\n".join(svg), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
