"""Audit and summarize the frozen seed-0/1/2 MLP and h26 validation."""

import argparse
import json
import os
import statistics as stats
from pathlib import Path

from replay_evaluation import sha256, write_csv
from run_comparison import write_json
from run_gru_full_budget import summarize
from run_h26_blind_validation import ACTORS, EXPECTED_HASHES, ORDER, STEPS, load


METRICS = {
    "final_completion": "300k completion fraction",
    "final_mean_length": "300k mean episode length (steps)",
    "final_return": "300k mean return",
    "final_all_position_cost": "300k all-episode position cost",
    "final_completed_position_cost": "300k completed-episode position cost",
    "first_ten_of_ten_step": "First 10/10 checkpoint (interactions)",
    "post200k_worst_completion": "Worst completion after 200k",
    "post200k_mean_completion": "Mean completion after 200k",
}
NAMES = {"mlp-original": "Original MLP", "gru-full-decay": "h26 with Actor LR decay"}


def final_metrics(rows):
    final = rows[-1]
    post = [row["mean_time_limit"] for row in rows if row["checkpoint_step"] > 200000]
    first_full = next((row["checkpoint_step"] for row in rows if row["mean_time_limit"] == 1), None)
    return dict(actor=final["actor"], seed=final["seed"], final_completion=final["mean_time_limit"],
                final_mean_length=final["mean_length"], final_return=final["mean_return"],
                final_all_position_cost=final["all_episodes_position_cost"],
                final_completed_position_cost=final["completed_position_cost"],
                first_ten_of_ten_step=first_full, post200k_worst_completion=min(post),
                post200k_mean_completion=stats.mean(post))


def compare(mlp, gru):
    criteria = [("final_completion", 1), ("final_all_position_cost", -1), ("final_return", 1)]
    for metric, direction in criteria:
        difference = (gru[metric] - mlp[metric]) * direction
        if difference:
            return ("GRU wins" if difference > 0 else "MLP wins"), metric
    return "Tie", "all ranking metrics equal"


def show(value, digits=3):
    return "not reached / unavailable" if value is None else f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--seed2-gru-retry", type=Path)
    args = parser.parse_args()
    root = args.results.resolve()
    status, protocol = load(root / "status.json"), load(root / "protocol.json")
    retry = args.seed2_gru_retry.resolve() if args.seed2_gru_retry else None
    if retry:
        assert status["status"] == "interrupted_by_system_shutdown"
        assert status["completed"] == [dict(seed=seed, actor=actor) for seed, actor in ORDER[:-1]]
        assert load(retry / "run.json")["status"] == "complete"
        assert load(retry.parent.parent / "status.json")["status"] == "complete"
        assert sha256(retry.parent.parent / "protocol.json") == sha256(root / "protocol.json")
    else:
        assert status["status"] == "complete"
        assert status["completed"] == [dict(seed=seed, actor=actor) for seed, actor in ORDER]
    assert status["protocol_sha256"] == sha256(root / "protocol.json")
    for relative, digest in protocol["source_hashes"].items():
        assert sha256(root / "source_snapshot" / relative) == digest
    points, episodes, per_seed, provenance = [], [], [], []
    selected_runs = []
    for seed in (0, 1, 2):
        for actor in ACTORS:
            if seed == 0:
                directory = Path(protocol["seed0"]["mlp" if actor == ACTORS[0] else "gru"])
            else:
                directory = root / f"seed-{seed}" / actor
            if seed == 2 and actor == ACTORS[1] and retry:
                directory = retry
            manifest = load(directory / "run.json")
            assert manifest["training"]["exit"] == 0
            assert (manifest["seed"], manifest["steps"], manifest["episodes"]) == (seed, STEPS, 10)
            assert manifest["executable_sha256"] == EXPECTED_HASHES[actor]
            selected_runs.append(dict(seed=seed, actor=actor, directory=str(directory),
                                      training_exit=manifest["training"]["exit"]))
            binary = root / "executables" / protocol["executables"][actor]["name"]
            assert sha256(binary) == EXPECTED_HASHES[actor]
            if seed:
                assert sha256(directory / binary.name) == EXPECTED_HASHES[actor]
                assert manifest["status"] == "complete"
                assert manifest["protocol_sha256"] == status["protocol_sha256"]
            checkpoint_actor = "mlp-original" if actor == ACTORS[0] else "gru-full"
            run_points = []
            assert sorted(map(int, manifest["evaluations"])) == list(range(0, STEPS, 10000))
            for step in range(0, STEPS, 10000):
                evaluation = manifest["evaluations"][str(step)]
                assert evaluation["exit"] == 0
                for role in ("actor", "target"):
                    checkpoint = directory / "checkpoints" / checkpoint_actor / f"{role}_step_{step}.txt"
                    expected = evaluation.get(f"{role}_sha256", evaluation.get("checkpoint_sha256") if role == "actor" else None)
                    assert sha256(checkpoint) == expected
                trajectory = (directory / f"flight-{step}.csv" if seed == 0 and actor == ACTORS[0]
                              else directory / "evaluation" / f"step-{step}" / "steps.csv")
                trajectory_hash = sha256(trajectory)
                if "trajectory_sha256" in evaluation:
                    assert trajectory_hash == evaluation["trajectory_sha256"]
                batch, summary = summarize(trajectory, evaluation["returns"], actor, step, seed=seed)
                assert len(batch) == 10
                run_points.append(summary)
                episodes.extend(batch)
                provenance.append(dict(seed=seed, actor=actor, checkpoint_step=step,
                                       trajectory=str(trajectory), trajectory_sha256=trajectory_hash,
                                       directory=str(directory)))
            points.extend(run_points)
            per_seed.append(final_metrics(run_points))
    assert len(points) == 186 and len(episodes) == 1860
    write_csv(root / "all_checkpoints.csv", points)
    write_csv(root / "all_episodes.csv", episodes)
    write_csv(root / "per_seed_metrics.csv", per_seed)
    write_json(root / "trajectory_provenance.json", provenance)
    aggregate = []
    for actor in ACTORS:
        rows = [row for row in per_seed if row["actor"] == actor]
        for metric in METRICS:
            values = [row[metric] for row in rows if row[metric] is not None]
            aggregate.append(dict(actor=actor, metric=metric, seed0=rows[0][metric], seed1=rows[1][metric],
                                  seed2=rows[2][metric], observed_seed_count=len(values),
                                  mean=stats.mean(values) if values else None,
                                  sample_std=stats.stdev(values) if len(values) > 1 else None))
    write_csv(root / "across_seed_metrics.csv", aggregate)
    comparisons = []
    for seed in (0, 1, 2):
        mlp, gru = [next(row for row in per_seed if row["seed"] == seed and row["actor"] == actor) for actor in ACTORS]
        winner, decisive_metric = compare(mlp, gru)
        comparisons.append(dict(seed=seed, winner=winner, decisive_metric=decisive_metric,
                                final_completion_difference=gru["final_completion"] - mlp["final_completion"],
                                all_position_cost_difference=gru["final_all_position_cost"] - mlp["final_all_position_cost"],
                                mean_return_difference=gru["final_return"] - mlp["final_return"],
                                post200k_worst_completion_difference=gru["post200k_worst_completion"] - mlp["post200k_worst_completion"],
                                post200k_mean_completion_difference=gru["post200k_mean_completion"] - mlp["post200k_mean_completion"]))
    exceeds = all(row["winner"] != "MLP wins" and row["final_completion_difference"] >= 0 for row in comparisons) and any(row["winner"] == "GRU wins" for row in comparisons)
    blind_exceeds = all(row["winner"] != "MLP wins" and row["final_completion_difference"] >= 0 for row in comparisons[1:]) and any(row["winner"] == "GRU wins" for row in comparisons[1:])
    result = dict(protocol_sha256=status["protocol_sha256"], per_seed=per_seed, across_seed=aggregate,
                  comparisons=comparisons, exceeds_mlp_under_frozen_rule=exceeds,
                  blind_seeds_exceed_mlp_under_frozen_rule=blind_exceeds,
                  audit=dict(training_runs=6, new_training_runs=4, checkpoints=186, episode_rows=1860,
                             new_training_exits=[row["training_exit"] for row in selected_runs if row["seed"] != 0]),
                  selected_runs=selected_runs,
                  recovery=dict(seed2_gru_retry=str(retry), interrupted_attempt_retained=True) if retry else None)
    write_json(root / "summary.json", result)
    lines = ["# Frozen MLP versus h26 blind validation", "",
             "Seed 0 selected the learning-rate schedule. Seeds 1 and 2 were trained after the protocol and "
             "ranking rule were frozen, using the exact archived seed-0 executables. All four new jobs "
             "completed 300,001 interactions in the prescribed serial order with exit code 0. Each method "
             "and seed has 31 checkpoints and ten no-exploration episodes per checkpoint. Evaluation uses "
             "a fresh process with the training seed and the unchanged compiled RNG mapping. "
             "Only the fixed h26 Actor alpha reduction follows its second curriculum update.", "",
             "| Seed | Method | 300k completion | Mean length | Mean return | All position cost | Completed position cost | First 10/10 step | Worst completion after 200k | Mean completion after 200k |",
             "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in per_seed:
        lines.append(f"| {row['seed']} | {NAMES[row['actor']]} | {row['final_completion']:.0%} | "
                     f"{row['final_mean_length']:.1f} | {row['final_return']:+.2f} | "
                     f"{row['final_all_position_cost']:.4f} | {show(row['final_completed_position_cost'], 4)} | "
                     f"{show(row['first_ten_of_ten_step'], 0)} | {row['post200k_worst_completion']:.0%} | "
                     f"{row['post200k_mean_completion']:.0%} |")
    lines += ["", "## Across-seed descriptive statistics", "",
              "Sample SD uses n−1 across seed-level metrics. Fractions remain on a 0-to-1 scale in this table. "
              "Missing attainment or completed-only cost values remain missing. Means for such metrics "
              "refer only to observed seeds and their count is reported. First 10/10 refers to the first "
              "saved checkpoint, with 10,000-interaction resolution. Post-200k covers 210k through 300k.", "",
              "| Method | Metric | Seed 0 | Seed 1 | Seed 2 | Mean | Sample SD | Observed seeds |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in aggregate:
        lines.append(f"| {NAMES[row['actor']]} | {METRICS[row['metric']]} | {show(row['seed0'])} | "
                     f"{show(row['seed1'])} | {show(row['seed2'])} | {show(row['mean'])} | "
                     f"{show(row['sample_std'])} | {row['observed_seed_count']} |")
    lines += ["", "## Decision under the frozen rule", "",
              "Rank each seed by 300k completion, then lower all-episode position cost if tied, then "
              "higher return if still tied. Completed-only cost is supplemental. Cross-seed superiority "
              "requires no lower final completion or losing rank in any seed, and at least one strict win.", ""]
    for row in comparisons:
        lines.append(f"- Seed {row['seed']}. {row['winner']} on {row['decisive_metric']}. "
                     f"GRU minus MLP differences are {row['final_completion_difference']:+.1%} final completion, "
                     f"{row['all_position_cost_difference']:+.4f} all-position cost, "
                     f"{row['mean_return_difference']:+.2f} return, "
                     f"{row['post200k_worst_completion_difference']:+.1%} worst post-200k completion, "
                     f"and {row['post200k_mean_completion_difference']:+.1%} mean post-200k completion.")
    lines += ["", f"The three-seed result meets the frozen superiority rule: {exceeds}. "
              f"The blind seed-1/2 subset meets that rule: {blind_exceeds}.", "",
              "These are descriptive results from three training seeds, one of which selected the schedule. "
              "The ten episodes and repeated checkpoints are not additional independent training seeds. "
              "Any losing seed is retained above and limits claims even when the aggregate mean favors GRU.", "",
              "Position cost is the per-episode mean of 20 times squared position, then averaged over "
              "all or completed episodes. The first unlogged-prior transition and termination transitions "
              "are excluded. Conditioning on completion can favor a method that fails more often, "
              "so ranking uses all-episode cost after comparing completion.", "",
              "All configurations, program hashes, source snapshots, commands, trainer/evaluation exit codes, "
              "logs, trajectory CSVs and Actor/target checkpoints remain saved. Checkpoints are weight-only "
              "and cannot restore optimizer, replay and RNG state. The earlier paused seed-0 run is a "
              "separate archive. Machine-readable data include per_seed_metrics.csv, across_seed_metrics.csv, "
              "all_checkpoints.csv, all_episodes.csv, summary.json and trajectory_provenance.json. "
              "The protocol and full curves are in protocol.json and comparison.png/comparison.pdf.", ""]
    if retry:
        lines += ["The original seed-2 GRU attempt was interrupted by system power-off after the "
                  "112k log record. It remains archived with its 110k weight pair. Following the interruption "
                  "report, only seed-2 GRU was restarted from step 0 in a separate directory with the exact "
                  "same executable and frozen protocol. The selected full run is " + str(retry) + ". "
                  "Its data form the seed-2 GRU result above. No partial curve was concatenated, and no "
                  "other training run was repeated. The original status.json retains the interruption "
                  "history. validation_status.json identifies the complete recovered comparison.", ""]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(root / "validation_status.json", dict(status="complete", protocol_sha256=status["protocol_sha256"],
                                                    selected_runs=selected_runs, recovered_system_interruption=bool(retry)))
    os.environ["MPLCONFIGDIR"] = str(root / "plot-config")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, layout="constrained")
    for seed in (0, 1, 2):
        for actor in ACTORS:
            rows = [row for row in points if row["seed"] == seed and row["actor"] == actor]
            x = [row["checkpoint_step"] / 1000 for row in rows]
            axes[0, seed].plot(x, [100 * row["mean_time_limit"] for row in rows], label=NAMES[actor])
            axes[1, seed].plot(x, [row["all_episodes_position_cost"] for row in rows], label=NAMES[actor])
        axes[0, seed].set_title(f"Seed {seed}" + (" (schedule selection)" if seed == 0 else " (blind validation)"))
        axes[0, seed].set_ylim(-3, 103)
        axes[1, seed].set_xlabel("Training interactions (thousands)")
        for axis in axes[:, seed]:
            axis.axvline(200, color="black", linestyle="--", linewidth=1)
            axis.grid(alpha=0.25)
            axis.set_xlim(0, 300)
    axes[0, 0].set_ylabel("500-step completion (%)")
    axes[1, 0].set_ylabel("All-episode mean position cost")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Frozen h26 validation, ten evaluation episodes per checkpoint")
    fig.savefig(root / "comparison.png", dpi=180)
    fig.savefig(root / "comparison.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, layout="constrained")
    for seed, axis in enumerate(axes):
        for actor in ACTORS:
            rows = [row for row in points if row["seed"] == seed and row["actor"] == actor]
            axis.plot([row["checkpoint_step"] / 1000 for row in rows],
                      [row["mean_return"] for row in rows], label=NAMES[actor])
        axis.set_title(f"Seed {seed}")
        axis.axvline(200, color="black", linestyle="--", linewidth=1)
        axis.set_xlabel("Training interactions (thousands)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Mean episode return")
    axes[0].legend(fontsize=8)
    fig.suptitle("Frozen validation return curves")
    fig.savefig(root / "returns.png", dpi=180)
    fig.savefig(root / "returns.pdf")
    print(json.dumps(dict(per_seed=per_seed, across_seed=aggregate, comparisons=comparisons,
                          exceeds_mlp=exceeds, blind_exceeds_mlp=blind_exceeds), indent=2))


if __name__ == "__main__":
    main()
