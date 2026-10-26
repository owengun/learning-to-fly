#!/usr/bin/env python3
"""Aggregate the reproduction TensorBoard logs across random seeds.

The script keeps the newest run for each (configuration, seed) pair. This
deliberately removes the extra baseline seed-0 trial in the results folder.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


CONFIG_NAMES = {
    "d+o+a+r+h+c+f+w+e+": "Baseline",
    "d+o+a+r+h+c-f+w+e+": "Without curriculum",
    "d+o+a+r+h-c+f+w+e+": "Without action history",
    "d+o+a-r+h+c+f+w+e+": "Without asymmetric critic",
}

SCALARS = {
    "return_mean": "evaluation/returns_mean",
    "return_std": "evaluation/returns_std",
    "episode_length_mean": "evaluation/episode_length_mean",
    "episode_length_std": "evaluation/episode_length_std",
    "validation_return_mean": "validation/ReturnMean",
    "validation_episode_length_mean": "validation/EpisodeLengthMean",
    "settling_fraction_200mm": "validation/SettlingFractionPosition(200mm)",
    "max_position_error_after_200": "validation/MaxErrorMean(Position, after 200 steps)",
}


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def parse_run_name(run_name: str) -> tuple[str, int]:
    match = re.search(r"_(d[+-]o[+-]a[+-]r[+-]h[+-]c[+-]f[+-]w[+-]e[+-])_(\d{3})$", run_name)
    if not match:
        raise ValueError(f"Unrecognized run name: {run_name}")
    return match.group(1), int(match.group(2))


def load_run(event_file: Path) -> dict:
    run_name = event_file.parent.name
    config, seed = parse_run_name(run_name)
    accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags()["scalars"])
    series = {}
    for output_name, tensorboard_name in SCALARS.items():
        if tensorboard_name in available:
            series[output_name] = {
                event.step: float(event.value)
                for event in accumulator.Scalars(tensorboard_name)
            }
        else:
            series[output_name] = {}
    return {
        "run_name": run_name,
        "config": config,
        "config_name": CONFIG_NAMES.get(config, config),
        "seed": seed,
        "mtime": event_file.stat().st_mtime,
        "event_file": str(event_file),
        "series": series,
    }


def select_runs(log_dir: Path) -> tuple[list[dict], list[dict]]:
    loaded = [load_run(path) for path in sorted(log_dir.glob("*/data.tfevents"))]
    selected_by_key: dict[tuple[str, int], dict] = {}
    excluded = []
    for run in loaded:
        key = (run["config"], run["seed"])
        previous = selected_by_key.get(key)
        if previous is None or run["mtime"] > previous["mtime"]:
            if previous is not None:
                excluded.append(previous)
            selected_by_key[key] = run
        else:
            excluded.append(run)
    selected = sorted(selected_by_key.values(), key=lambda item: (item["config_name"], item["seed"]))
    return selected, excluded


def convergence_step(series: dict[int, float], threshold: float, consecutive: int) -> int | None:
    steps = sorted(series)
    streak = 0
    for step in steps:
        if series[step] >= threshold:
            streak += 1
            if streak >= consecutive:
                return steps[steps.index(step) - consecutive + 1]
        else:
            streak = 0
    return None


def write_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_svg_chart(path: Path, selected: list[dict]) -> None:
    width, height = 1200, 500
    margin_left, margin_top, panel_width, panel_height = 75, 55, 480, 360
    panel_gap = 105
    colors = {
        "Baseline": "#1f77b4",
        "Without curriculum": "#ff7f0e",
        "Without action history": "#2ca02c",
        "Without asymmetric critic": "#d62728",
    }

    def aggregate(config_name: str, metric: str) -> list[tuple[int, float]]:
        runs = [run for run in selected if run["config_name"] == config_name]
        if not runs:
            return []
        steps = sorted(set.intersection(*[set(run["series"][metric]) for run in runs]))
        return [(step, statistics.mean(run["series"][metric][step] for run in runs)) for step in steps]

    return_values = [value for run in selected for value in run["series"]["return_mean"].values()]
    return_min = min(return_values)
    return_max = max(return_values)
    return_pad = max((return_max - return_min) * 0.08, 1.0)
    panels = [
        ("Evaluation return", "return_mean", return_min - return_pad, return_max + return_pad),
        ("Episode length", "episode_length_mean", 0.0, 500.0),
    ]

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.axis{stroke:#444;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.series{fill:none;stroke-width:2.5}</style>',
        '<text x="600" y="28" text-anchor="middle" font-size="20" font-weight="bold">Learning to Fly reproduction: mean across 3 seeds</text>',
    ]
    for panel_index, (title, metric, y_min, y_max) in enumerate(panels):
        x0 = margin_left + panel_index * (panel_width + panel_gap)
        y0 = margin_top
        svg.append(f'<text x="{x0 + panel_width / 2}" y="{y0 - 12}" text-anchor="middle" font-size="16" font-weight="bold">{html.escape(title)}</text>')
        for tick in range(6):
            y = y0 + panel_height - panel_height * tick / 5
            value = y_min + (y_max - y_min) * tick / 5
            svg.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x0 + panel_width}" y2="{y:.1f}"/>')
            svg.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11">{value:.0f}</text>')
        for tick in range(7):
            x = x0 + panel_width * tick / 6
            value = 300000 * tick / 6
            svg.append(f'<text x="{x:.1f}" y="{y0 + panel_height + 20}" text-anchor="middle" font-size="11">{value / 1000:.0f}k</text>')
        svg.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + panel_height}"/>')
        svg.append(f'<line class="axis" x1="{x0}" y1="{y0 + panel_height}" x2="{x0 + panel_width}" y2="{y0 + panel_height}"/>')
        for config_name, color in colors.items():
            points = aggregate(config_name, metric)
            if not points:
                continue
            coords = []
            for step, value in points:
                x = x0 + panel_width * step / 300000
                y = y0 + panel_height - panel_height * (value - y_min) / (y_max - y_min)
                coords.append(f"{x:.1f},{y:.1f}")
            svg.append(f'<polyline class="series" stroke="{color}" points="{" ".join(coords)}"/>')
        svg.append(f'<text x="{x0 + panel_width / 2}" y="{y0 + panel_height + 42}" text-anchor="middle" font-size="12">Environment interactions</text>')

    legend_y = 482
    legend_x = 120
    for index, (config_name, color) in enumerate(colors.items()):
        x = legend_x + index * 260
        svg.append(f'<line x1="{x}" y1="{legend_y - 4}" x2="{x + 28}" y2="{legend_y - 4}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="{x + 36}" y="{legend_y}" font-size="12">{html.escape(config_name)}</text>')
    svg.append('</svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "generated")
    parser.add_argument("--length-threshold", type=float, default=475.0)
    parser.add_argument("--consecutive", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    selected, excluded = select_runs(args.results / "logs")
    expected = {(config, seed) for config in CONFIG_NAMES for seed in range(3)}
    actual = {(run["config"], run["seed"]) for run in selected}

    selected_rows = [
        {
            "configuration": run["config_name"],
            "config_code": run["config"],
            "seed": run["seed"],
            "run_name": run["run_name"],
            "event_file": run["event_file"],
        }
        for run in selected
    ]
    write_rows(
        args.output / "selected_runs.csv",
        ["configuration", "config_code", "seed", "run_name", "event_file"],
        selected_rows,
    )

    evaluation_rows = []
    validation_rows = []
    per_run_rows = []
    for run in selected:
        evaluation_steps = sorted(run["series"]["episode_length_mean"])
        for step in evaluation_steps:
            evaluation_rows.append(
                {
                    "configuration": run["config_name"],
                    "config_code": run["config"],
                    "seed": run["seed"],
                    "step": step,
                    "return_mean": run["series"]["return_mean"].get(step, math.nan),
                    "return_std_within_evaluation": run["series"]["return_std"].get(step, math.nan),
                    "episode_length_mean": run["series"]["episode_length_mean"].get(step, math.nan),
                    "episode_length_std_within_evaluation": run["series"]["episode_length_std"].get(step, math.nan),
                }
            )
        validation_steps = sorted(run["series"]["validation_episode_length_mean"])
        for step in validation_steps:
            validation_rows.append(
                {
                    "configuration": run["config_name"],
                    "config_code": run["config"],
                    "seed": run["seed"],
                    "step": step,
                    "validation_return_mean": run["series"]["validation_return_mean"].get(step, math.nan),
                    "validation_episode_length_mean": run["series"]["validation_episode_length_mean"].get(step, math.nan),
                    "settling_fraction_200mm": run["series"]["settling_fraction_200mm"].get(step, math.nan),
                    "max_position_error_after_200": run["series"]["max_position_error_after_200"].get(step, math.nan),
                }
            )
        final_step = max(evaluation_steps)
        per_run_rows.append(
            {
                "configuration": run["config_name"],
                "config_code": run["config"],
                "seed": run["seed"],
                "final_step": final_step,
                "final_return_mean": run["series"]["return_mean"][final_step],
                "final_episode_length_mean": run["series"]["episode_length_mean"][final_step],
                "convergence_step": convergence_step(
                    run["series"]["episode_length_mean"],
                    args.length_threshold,
                    args.consecutive,
                ),
            }
        )

    write_rows(
        args.output / "evaluation_curves.csv",
        ["configuration", "config_code", "seed", "step", "return_mean", "return_std_within_evaluation", "episode_length_mean", "episode_length_std_within_evaluation"],
        evaluation_rows,
    )
    write_rows(
        args.output / "validation_curves.csv",
        ["configuration", "config_code", "seed", "step", "validation_return_mean", "validation_episode_length_mean", "settling_fraction_200mm", "max_position_error_after_200"],
        validation_rows,
    )
    write_rows(
        args.output / "per_run_summary.csv",
        ["configuration", "config_code", "seed", "final_step", "final_return_mean", "final_episode_length_mean", "convergence_step"],
        per_run_rows,
    )

    timing_by_config = defaultdict(list)
    timing_path = args.results / "timing.csv"
    if timing_path.exists():
        with timing_path.open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                job_id = int(row["JobId"])
                ablation_id = job_id // 50
                config = {0: "d+o+a+r+h+c+f+w+e+", 1: "d+o+a+r+h+c-f+w+e+", 3: "d+o+a+r+h-c+f+w+e+", 5: "d+o+a-r+h+c+f+w+e+"}.get(ablation_id)
                if config:
                    timing_by_config[config].append(float(row["ElapsedSeconds"]))

    grouped = defaultdict(list)
    for row in per_run_rows:
        grouped[row["config_code"]].append(row)
    summary_rows = []
    for config, rows in grouped.items():
        returns = [float(row["final_return_mean"]) for row in rows]
        lengths = [float(row["final_episode_length_mean"]) for row in rows]
        converged = [int(row["convergence_step"]) for row in rows if row["convergence_step"] is not None]
        times = timing_by_config.get(config, [])
        return_mean, return_std = mean_std(returns)
        length_mean, length_std = mean_std(lengths)
        time_mean, time_std = mean_std(times)
        conv_mean, conv_std = mean_std([float(value) for value in converged])
        summary_rows.append(
            {
                "configuration": CONFIG_NAMES.get(config, config),
                "config_code": config,
                "seeds": len(rows),
                "final_return_mean_across_seeds": return_mean,
                "final_return_std_across_seeds": return_std,
                "final_episode_length_mean_across_seeds": length_mean,
                "final_episode_length_std_across_seeds": length_std,
                "converged_seeds": len(converged),
                "convergence_step_mean": conv_mean,
                "convergence_step_std": conv_std,
                "wall_time_mean_seconds": time_mean,
                "wall_time_std_seconds": time_std,
            }
        )
    summary_rows.sort(key=lambda row: row["configuration"])
    write_rows(args.output / "summary.csv", list(summary_rows[0]), summary_rows)

    try:
        import matplotlib.pyplot as plt

        colors = {
            "Baseline": "#1f77b4",
            "Without curriculum": "#ff7f0e",
            "Without action history": "#2ca02c",
            "Without asymmetric critic": "#d62728",
        }
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        for config_name in colors:
            config_runs = [run for run in selected if run["config_name"] == config_name]
            if not config_runs:
                continue
            common_steps = sorted(set.intersection(*[set(run["series"]["return_mean"]) for run in config_runs]))
            for axis, metric, ylabel in [
                (axes[0], "return_mean", "Evaluation return"),
                (axes[1], "episode_length_mean", "Episode length"),
            ]:
                means = []
                stds = []
                for step in common_steps:
                    values = [run["series"][metric][step] for run in config_runs]
                    mean, std = mean_std(values)
                    means.append(mean)
                    stds.append(std)
                axis.plot(common_steps, means, label=config_name, color=colors[config_name])
                axis.fill_between(common_steps, [m - s for m, s in zip(means, stds)], [m + s for m, s in zip(means, stds)], color=colors[config_name], alpha=0.15)
                axis.set_xlabel("Environment interactions")
                axis.set_ylabel(ylabel)
                axis.grid(alpha=0.25)
        axes[1].axhline(args.length_threshold, color="#555555", linestyle="--", linewidth=1, label="95% length threshold")
        axes[0].legend(fontsize=8)
        axes[1].legend(fontsize=8)
        fig.suptitle("Learning to Fly reproduction: mean ± std across 3 seeds")
        fig.savefig(args.output / "learning_curves.png", dpi=180)
        plt.close(fig)
    except ImportError:
        pass
    write_svg_chart(args.output / "learning_curves.svg", selected)

    analysis = {
        "selected_run_count": len(selected),
        "excluded_duplicate_runs": [run["run_name"] for run in excluded],
        "missing_expected_runs": sorted(f"{config}/seed-{seed}" for config, seed in expected - actual),
        "convergence_definition": {
            "metric": "evaluation/episode_length_mean",
            "threshold": args.length_threshold,
            "consecutive_evaluations": args.consecutive,
        },
        "summary": summary_rows,
    }
    (args.output / "analysis.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(analysis, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
