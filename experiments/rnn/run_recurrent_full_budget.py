"""Run one fixed parameter-matched recurrent seed-0 experiment and audit its curve."""

import argparse
import csv
import itertools
import json
import math
import os
import shutil
import statistics as stats
import subprocess
import time
from pathlib import Path

from replay_evaluation import sha256, write_csv
from run_comparison import RETURN, write_json
from run_gru_full_budget import CURRICULUM, summarize

STEPS = 300001
EPISODES = 10
MLP_HASH = "42e6625e067f7e3992a1d93f4bbc9a8acf4ac2e37b48e0a640c0dd98158bc80d"
DIAGNOSTIC_FIELDS = ("step,updates,q_mean,q_abs_max,grad_l2,update_l2,parameter_l2,"
                     "gate0_mean,gate1_mean,gate2_mean,gate3_mean,hidden_l2,cell_l2,hidden_saturated,"
                     "proposed_action_mean,applied_action_mean,vertical_thrust_over_weight,mean_burn,actor_alpha")


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def metrics(rows):
    final = rows[-1]
    after = [row["mean_time_limit"] for row in rows if row["checkpoint_step"] > 200000]
    return dict(actor=final["actor"], completion=final["mean_time_limit"], mean_length=final["mean_length"],
                mean_return=final["mean_return"], all_position_cost=final["all_episodes_position_cost"],
                completed_position_cost=final["completed_position_cost"],
                first_10_of_10=next((row["checkpoint_step"] for row in rows if row["mean_time_limit"] == 1), None),
                worst_completion_after_200k=min(after), mean_completion_after_200k=stats.mean(after))


def write_report(directory, points, note):
    names = list(dict.fromkeys(row["actor"] for row in points))
    groups = {name: sorted((row for row in points if row["actor"] == name), key=lambda row: row["checkpoint_step"])
              for name in names}
    final = [metrics(rows) for rows in groups.values()]
    write_csv(directory / "summary.csv", final)
    lines = ["# Full-observation recurrent seed-0 comparison", "", note, "",
             "Each checkpoint is evaluated in a fresh process with seed 0, ten episodes, a 500-step "
             "cap and no action exploration noise. The fixed observation, simulator and evaluation RNG "
             "mapping are retained. Position cost averages 20 times squared position per episode, "
             "then across the indicated group, excluding first and terminal transitions.", "",
             "| Model | 300k completion | Mean length | Mean return | All position cost | Completed position cost | First 10/10 | Worst completion after 200k | Mean completion after 200k |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in final:
        cost = "unavailable" if row["completed_position_cost"] is None else f"{row['completed_position_cost']:.4f}"
        first = row["first_10_of_10"] if row["first_10_of_10"] is not None else "not reached"
        lines.append(f"| {row['actor']} | {row['completion']:.0%} | {row['mean_length']:.1f} | "
                     f"{row['mean_return']:+.2f} | {row['all_position_cost']:.4f} | {cost} | {first} | "
                     f"{row['worst_completion_after_200k']:.0%} | {row['mean_completion_after_200k']:.0%} |")
    lines += ["", "First 10/10 uses 10k checkpoint resolution. The post-200k window is 210k through 300k. "
              "Seed 0 alone does not establish performance across training seeds. This experiment "
              "retains a fixed 32-step burn-in approximation rather than reconstructing the whole "
              "episode for every replay update.", ""]
    (directory / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    write_csv(directory / "flight_by_checkpoint.csv", points)
    os.environ["MPLCONFIGDIR"] = str(directory / "plot-config")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, layout="constrained")
    panels = [("mean_time_limit", "500-step completion (%)", 100),
              ("mean_length", "Mean episode length (steps)", 1),
              ("mean_return", "Mean episode return", 1),
              ("all_episodes_position_cost", "All-episode mean position cost", 1)]
    for axis, (key, label, scale) in zip(axes.flat, panels):
        for name, rows in groups.items():
            axis.plot([row["checkpoint_step"] / 1000 for row in rows],
                      [row[key] * scale for row in rows], label=name)
        axis.axvline(200, color="black", linestyle="--", linewidth=1)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.set_xlim(0, 300)
    for axis in axes[-1]:
        axis.set_xlabel("Training interactions (thousands)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Seed 0, full 146-input actors, ten no-exploration episodes per checkpoint")
    fig.savefig(directory / "comparison.png", dpi=180)
    fig.savefig(directory / "comparison.pdf")
    plt.close(fig)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", choices=("rnn-full", "lstm-full"), required=True)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--mlp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory, baseline = args.output.resolve(), args.mlp.resolve()
    reference = load(baseline / "run.json")
    assert reference["executable_sha256"] == MLP_HASH and reference["training"]["exit"] == 0
    assert (reference["seed"], reference["steps"], reference["episodes"]) == (0, STEPS, EPISODES)
    directory.mkdir(parents=True)
    executable = directory / args.exe.name
    shutil.copy2(args.exe.resolve(), executable)
    repository = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repository, text=True).strip()
    config = dict(input=146, hidden=64 if args.actor == "rnn-full" else 21,
                  parameters=13764 if args.actor == "rnn-full" else 14200, burn_in=32, learn_max=8,
                  alpha=0.001, decay_step=200000, decayed_alpha=0.0001,
                  beta1=0.9, beta2=0.999, epsilon=1e-8, independent_weight_decay=0.0001,
                  seed=0, interactions=STEPS, evaluation_episodes=EPISODES)
    manifest = dict(actor=args.actor, seed=0, steps=STEPS, episodes=EPISODES, config=config,
                    branch=branch, source_commit=commit, executable_sha256=sha256(executable),
                    mlp_reference=str(baseline), mlp_executable_sha256=MLP_HASH, evaluations={}, status="training")
    snapshot = directory / "source_snapshot"
    source_files = [*repository.glob("src/config/*.h"), *repository.glob("experiments/rnn/*.h"),
                    repository / "src/training.h", repository / "src/training_state.h", repository / "src/training.cpp",
                    repository / "src/steps/curriculum.h", repository / "src/CMakeLists.txt", Path(__file__).resolve()]
    manifest["source_hashes"] = {}
    for path in source_files:
        relative = path.relative_to(repository)
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        manifest["source_hashes"][str(relative)] = sha256(path)
    write_json(directory / "config.json", config)
    path = directory / "run.json"
    write_json(path, manifest)
    command = [str(executable), "--seed", "0", "--steps", str(STEPS)]
    started = time.monotonic()
    try:
        with (directory / "training.log").open("wb") as stdout, (directory / "training-error.log").open("wb") as stderr:
            with subprocess.Popen(command, cwd=directory, stdout=stdout, stderr=stderr) as process:
                manifest["active_process"] = dict(pid=process.pid, command=command)
                write_json(path, manifest)
                code = process.wait()
        manifest.pop("active_process")
        manifest["training"] = dict(exit=code, seconds=time.monotonic() - started, command=command)
        write_json(path, manifest)
        if code != 0:
            raise RuntimeError(f"Training exited with {code}")
        log = (directory / "training.log").read_text(encoding="utf-8")
        assert "Training finished at step 300001" in log
        assert CURRICULUM.findall(log) == [("100000", "100000"), ("200000", "200000")]
        markers = ["[RNN] Weights saved at step 200000", "[Curriculum] step 200000",
                   "[Actor Adam LR] step 200000, alpha 0.0001", "[Recurrent Adam Diagnostics] 200000,"]
        indices = [log.index(marker) for marker in markers]
        assert indices == sorted(indices) and log.count(markers[2]) == 1
        diagnostic_lines = [line.split("] ", 1)[1] for line in log.splitlines()
                            if line.startswith("[Recurrent Adam Diagnostics] ")]
        diagnostics = list(csv.DictReader([DIAGNOSTIC_FIELDS, *diagnostic_lines]))
        assert diagnostics and all(len(line.split(",")) == 19 for line in diagnostic_lines)
        assert all(all(math.isfinite(float(value)) for value in row.values()) for row in diagnostics)
        assert all(float(row["actor_alpha"]) == (0.001 if int(row["step"]) < 200000 else 0.0001)
                   for row in diagnostics)
        write_csv(directory / "actor_update_diagnostics.csv", diagnostics)
        manifest["status"] = "evaluating"
        write_json(path, manifest)
        points, all_episodes, telemetry = [], [], []
        for step in range(0, STEPS, 10000):
            folder = directory / "evaluation" / f"step-{step}"
            folder.mkdir(parents=True)
            checkpoint = directory / "checkpoints" / args.actor / f"actor_step_{step}.txt"
            target = checkpoint.with_name(f"target_step_{step}.txt")
            trajectory = folder / "steps.csv"
            command = [str(executable), "--seed", "0", "--episodes", "10", "--evaluate", str(checkpoint),
                       "--trajectory-csv", str(trajectory)]
            with (folder / "evaluation.log").open("wb") as stdout, (folder / "evaluation-error.log").open("wb") as stderr:
                result = subprocess.run(command, cwd=folder, stdout=stdout, stderr=stderr)
            if result.returncode != 0:
                raise RuntimeError(f"Evaluation failed at {step}")
            returns = [float(value) for value in RETURN.findall((folder / "evaluation.log").read_text())]
            episodes, row = summarize(trajectory, returns, args.actor, step)
            original = reference["evaluations"][str(step)]
            for role in ("actor", "target"):
                expected = original["checkpoint_sha256" if role == "actor" else "target_sha256"]
                assert sha256(baseline / "checkpoints/mlp-original" / f"{role}_step_{step}.txt") == expected
            baseline_trajectory = baseline / f"flight-{step}.csv"
            old_episodes, old_row = summarize(baseline_trajectory, original["returns"], "mlp-original", step)
            points.extend([old_row, row])
            all_episodes.extend(old_episodes + episodes)
            with trajectory.open(newline="", encoding="utf-8") as stream:
                for episode, group in itertools.groupby(csv.DictReader(stream), key=lambda value: int(value["episode"])):
                    samples = list(group)
                    values = {name: stats.mean(float(item[name]) for item in samples)
                              for name in ("hidden_l2", "cell_l2", "hidden_saturated", "gate0_mean", "gate1_mean", "gate2_mean", "gate3_mean")}
                    telemetry.append(dict(checkpoint_step=step, episode=episode, **values,
                                          final_z=float(samples[-1]["z_m"]), terminated=int(samples[-1]["terminated"])))
            write_csv(folder / "episodes.csv", old_episodes + episodes)
            manifest["evaluations"][str(step)] = dict(exit=0, returns=returns, command=command,
                actor_sha256=sha256(checkpoint), target_sha256=sha256(target), trajectory_sha256=sha256(trajectory),
                mlp_trajectory=str(baseline_trajectory), mlp_trajectory_sha256=sha256(baseline_trajectory), summaries=[old_row, row])
            write_json(path, manifest)
        write_csv(directory / "all_episodes.csv", all_episodes)
        write_csv(directory / "episode_hidden_diagnostics.csv", telemetry)
        final = write_report(directory, points, f"{args.actor}, {config['parameters']} Actor parameters, from {branch} at {commit}.")
        stages = {}
        for first, last in ((180000,199999), (200000,219999), (280000,300000)):
            rows = [row for row in diagnostics if first <= int(row["step"]) <= last]
            stages[f"{first}-{last}"] = {name: stats.mean(float(row[name]) for row in rows)
                                        for name in ("grad_l2", "update_l2", "q_abs_max", "hidden_l2", "cell_l2", "hidden_saturated")}
        write_json(directory / "diagnosis.json", dict(final=final, training_windows=stages,
            final_hidden=[row for row in telemetry if row["checkpoint_step"] == 300000],
            note="Descriptive trajectory/gradient/state diagnosis from this single fixed run. No causal attribution or parameter search."))
        manifest["status"] = "complete"
        write_json(path, manifest)
        write_json(directory / "status.json", dict(status="complete", steps=STEPS, checkpoints=31, summary=final))
        print(json.dumps(dict(status="complete", summary=final)), flush=True)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failure"] = str(error)
        write_json(path, manifest)
        write_json(directory / "status.json", dict(status="failed", reason=str(error)))
        raise


if __name__ == "__main__":
    main()
