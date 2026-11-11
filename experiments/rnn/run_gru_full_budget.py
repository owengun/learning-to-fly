"""Replay the screened full-observation GRU from seed zero to the full budget."""

import argparse
import csv
import itertools
import json
import re
import shutil
import statistics as stats
import subprocess
import time
from pathlib import Path

import run_comparison as comparison
from replay_evaluation import episode_metrics, sha256, write_csv


SEED = 0
EPISODES = 10
STEPS = 300001
SAVED = re.compile(r"^\[RNN\] Weights saved at step (\d+)$", re.MULTILINE)
CURRICULUM = re.compile(r"^\[Curriculum\] step (\d+), recalculated (\d+) replay rewards$", re.MULTILINE)
MILESTONES = set(range(180000, 300001, 10000))


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def thrust_ratio(row):
    # Simulator rotor-thrust coefficient, tilt projection and vehicle weight.
    qx, qy = float(row["qx"]), float(row["qy"])
    rotor_sq = sum(float(row[f"rpm_{i}"]) ** 2 for i in range(4))
    return 3.16e-10 * rotor_sq * (1 - 2 * (qx * qx + qy * qy)) / (0.027 * 9.81)


def summarize(trajectory, returns, actor, step, seed=SEED):
    key = {"actor": actor, "seed": seed, "checkpoint_step": step}
    episodes = list(episode_metrics(trajectory, returns, key))
    with trajectory.open(newline="", encoding="utf-8") as stream:
        groups = itertools.groupby(csv.DictReader(stream), key=lambda row: int(row["episode"]))
        for episode, (index, group) in zip(episodes, groups, strict=True):
            assert episode["episode"] == index
            rows = list(group)
            last = rows[-1]
            episode["lower_exit"] = int(last["terminated"] == "1" and float(last["z_m"]) < -0.6)
            episode["first10_thrust_over_weight"] = stats.mean(map(thrust_ratio, rows[:10]))
            episode["last5_thrust_over_weight"] = stats.mean(map(thrust_ratio, rows[-5:]))
            costs = [20 * sum(float(prev[k]) ** 2 for k in ("x_m", "y_m", "z_m"))
                     for prev, now in zip(rows, rows[1:]) if now["terminated"] == "0"]
            episode["mean_position_cost"] = stats.mean(costs) if costs else None
            episode["last100_z_m"] = stats.mean(float(row["z_m"]) for row in rows[-100:])
    assert len(episodes) == EPISODES
    columns = ("length", "time_limit", "lower_exit", "return", "first10_thrust_over_weight",
               "last5_thrust_over_weight", "position_rmse_m")
    completed_costs = [row["mean_position_cost"] for row in episodes if row["time_limit"]]
    return episodes, dict(key, **{
        f"mean_{name}": stats.mean(row[name] for row in episodes) for name in columns
    }, completed_position_cost=stats.mean(completed_costs) if completed_costs else None,
        all_episodes_position_cost=stats.mean(row["mean_position_cost"] for row in episodes
                                             if row["mean_position_cost"] is not None),
        completed_last100_z_m=stats.mean(row["last100_z_m"] for row in episodes
                                        if row["time_limit"]) if completed_costs else None)


def checkpoint_hashes(directory, actor, step):
    folder = directory / "checkpoints" / actor
    return {role: sha256(folder / f"{role}_step_{step}.txt") for role in ("actor", "target")}


def evaluate(directory, executable, step, screening, baseline, manifest, width_reference=None):
    key = str(step)
    actor = manifest["actor"]
    checkpoint_actor = manifest.get("checkpoint_actor", actor)
    hashes = checkpoint_hashes(directory, checkpoint_actor, step)
    old = screening["evaluations"].get(key) if screening else None
    if width_reference and step <= 200000:
        old = load(width_reference / "run.json")["evaluations"][key]
    if old is not None:
        assert hashes == {"actor": old.get("actor_sha256", old.get("checkpoint_sha256")),
                          "target": old["target_sha256"]}, f"Seed-0 checkpoint prefix differs at {step}"
    folder = directory / "evaluation" / f"step-{step}"
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "checkpoints" / checkpoint_actor / f"actor_step_{step}.txt"
    trajectory = folder / "steps.csv"
    command = [str(executable), "--seed", str(SEED), "--episodes", str(EPISODES),
               "--evaluate", str(checkpoint), "--trajectory-csv", str(trajectory)]
    result = comparison.invoke(command, folder, "evaluation")
    if result["exit"] != 0:
        raise RuntimeError(f"Evaluation failed at {step}. See {folder}")
    returns = [float(value) for value in comparison.RETURN.findall(
        (folder / "evaluation.log").read_text(encoding="utf-8"))]
    assert len(returns) == EPISODES
    if old is not None:
        assert returns == old["returns"], f"Seed-0 return prefix differs at {step}"
    episodes, gru = summarize(trajectory, returns, actor, step)
    original = load(baseline / "run.json")["evaluations"][key]
    baseline_hashes = checkpoint_hashes(baseline, "mlp-original", step)
    assert baseline_hashes == {"actor": original["checkpoint_sha256"],
                              "target": original["target_sha256"]}
    baseline_csv = baseline / f"flight-{step}.csv"
    baseline_episodes, mlp = summarize(baseline_csv, original["returns"], "mlp-original", step)
    summaries = [mlp]
    all_episodes = baseline_episodes
    if width_reference:
        narrow_manifest = load(width_reference / "run.json")
        narrow = narrow_manifest["evaluations"][key]
        narrow_hashes = checkpoint_hashes(width_reference, "gru-full", step)
        assert narrow_hashes == {"actor": narrow["actor_sha256"], "target": narrow["target_sha256"]}
        narrow_csv = width_reference / "evaluation" / f"step-{step}" / "steps.csv"
        assert sha256(narrow_csv) == narrow["trajectory_sha256"]
        narrow_episodes, narrow_summary = summarize(narrow_csv, narrow["returns"], "gru-full", step)
        all_episodes += narrow_episodes
        summaries.append(narrow_summary)
        result["h26_trajectory"] = str(narrow_csv)
        result["h26_trajectory_sha256"] = narrow["trajectory_sha256"]
    summaries.append(gru)
    write_csv(folder / "episodes.csv", all_episodes + episodes)
    result.update(returns=returns, actor_sha256=hashes["actor"], target_sha256=hashes["target"],
                  trajectory_sha256=sha256(trajectory), prefix_verified=old is not None,
                  original_mlp_trajectory=str(baseline_csv),
                  original_mlp_trajectory_sha256=sha256(baseline_csv), summaries=summaries)
    manifest["evaluations"][key] = result
    comparison.write_json(directory / "run.json", manifest)
    write_csv(directory / "flight_by_checkpoint.csv", [
        summary for evaluation in manifest["evaluations"].values()
        for summary in evaluation["summaries"]
    ])
    print(f"{step}: GRU {gru['mean_length']:.1f} steps, {gru['mean_time_limit']:.0%} completed; "
          f"MLP {mlp['mean_length']:.1f} steps, {mlp['mean_time_limit']:.0%} completed", flush=True)


def write_report(directory, manifest):
    if manifest["actor"] == "gru-full-decay":
        intro = (
            "The only training change is an Actor Adam alpha of 0.0001 beginning with "
            "the first Actor update after the second curriculum change, at step 200,000. "
            "All checkpoints through 200,000 match the unmodified h26 reference exactly. "
            "The comparison uses the archived h26 and MLP at the same interaction counts."
        )
    else:
        intro = (
            "The prior 100,001-interaction process had exited with weight-only checkpoints. "
            "This run restarted from seed 0 with the exact screened executable and no training gate. "
            "The Actor and target hashes and all ten evaluation returns match the screened run "
            "at every checkpoint through 100,000 interactions."
        )
    lines = [
        "# Full-budget full-observation GRU", "",
        intro, "",
        "Ten no-exploration episodes per checkpoint. Position cost is averaged per episode "
        "and then across the indicated group. The first transition has no logged prior state. "
        "The terminal transition uses the termination reward. Both are excluded.", "",
        "| Step | Actor | Mean length | 500-step completion | Downward exits | Mean return | First 10 thrust / weight | Last 5 thrust / weight | All-episode position cost | Completed position cost |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for step, evaluation in manifest["evaluations"].items():
        if int(step) not in MILESTONES:
            continue
        for row in evaluation["summaries"]:
            cost = row["completed_position_cost"]
            cost_text = "n/a" if cost is None else f"{cost:.3f}"
            lines.append(f"| {step} | {row['actor']} | {row['mean_length']:.1f} | "
                         f"{row['mean_time_limit']:.0%} | {row['mean_lower_exit']:.0%} | "
                         f"{row['mean_return']:.2f} | {row['mean_first10_thrust_over_weight']:.3f} | "
                         f"{row['mean_last5_thrust_over_weight']:.3f} | "
                         f"{row['all_episodes_position_cost']:.3f} | {cost_text} |")
    lines += [
        "",
        "The 100,000 and 200,000 snapshots precede their curriculum update. The adjacent "
        "90,000/110,000 and 190,000/210,000 rows show the change around each update. "
        "The motor thrust estimates use measured rotor speed and attitude, excluding external forces.",
        "",
        "The comparison is a single seed. Per-episode data, trajectories, commands and hashes "
        "are retained in evaluation/, flight_by_checkpoint.csv and run.json.",
        "",
    ]
    (directory / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path)
    parser.add_argument("--decay-exe", type=Path)
    parser.add_argument("--h26", type=Path)
    parser.add_argument("--mlp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.screen is None) == (args.decay_exe is None) or bool(args.h26) != bool(args.decay_exe):
        parser.error("Supply either --screen, or --decay-exe and --h26")
    baseline = args.mlp.resolve()
    directory = args.output.resolve()
    original = load(baseline / "run.json")
    assert (original["actor"], original["seed"], original["steps"], original["episodes"]) == (
        "mlp-original", SEED, STEPS, EPISODES)
    width_reference = args.h26.resolve() if args.h26 else None
    if args.screen:
        screen = args.screen.resolve()
        screening = load(screen / "run.json")
        assert (screening["actor"], screening["seed"], screening["steps"], screening["episodes"]) == (
            "gru-full", SEED, 100001, EPISODES)
        source = screen / "training_gru_full.exe"
        assert sha256(source) == screening["executable_sha256"]
        actor, config = "gru-full", screening["config"]
    else:
        screen, screening = None, None
        source = args.decay_exe.resolve()
        narrow = load(width_reference / "run.json")
        assert (narrow["actor"], narrow["seed"], narrow["steps"], narrow["episodes"]) == (
            "gru-full", SEED, STEPS, EPISODES)
        assert narrow["config"]["input"] == 146 and narrow["config"]["hidden"] == 26
        actor, config = "gru-full-decay", dict(narrow["config"], actor_lr_decay_step=200000,
                                               actor_lr_decay_factor=0.1)
    directory.mkdir(parents=True)
    executable = directory / source.name
    shutil.copy2(source, executable)
    command = [str(executable), "--seed", str(SEED), "--steps", str(STEPS)]
    manifest = {"actor": actor, "seed": SEED, "steps": STEPS, "episodes": EPISODES,
                "command": command, "executable_sha256": sha256(executable),
                "config": config, "screen": str(screen) if screen else None,
                "reference": str(baseline), "evaluations": {}}
    if width_reference:
        manifest["h26_reference"] = str(width_reference)
        manifest["checkpoint_actor"] = "gru-full"
    comparison.write_json(directory / "run.json", manifest)
    log_path = directory / "training.log"
    started = time.monotonic()
    with log_path.open("wb") as log, (directory / "training-error.log").open("wb") as error:
        with subprocess.Popen(command, cwd=directory, stdout=log, stderr=error) as process:
            while True:
                exit_code = process.poll()
                text = log_path.read_text(encoding="utf-8")
                for label in SAVED.findall(text):
                    if label not in manifest["evaluations"]:
                        try:
                            evaluate(directory, executable, int(label), screening, baseline,
                                     manifest, width_reference)
                        except Exception as exc:
                            manifest["evaluation_error"] = {"step": int(label), "message": str(exc)}
                            if process.poll() is None:
                                process.terminate()
                            manifest["training"] = {"exit": process.wait(),
                                                    "seconds": time.monotonic() - started}
                            comparison.write_json(directory / "run.json", manifest)
                            raise
                if exit_code is not None:
                    break
                time.sleep(1)
    manifest["training"] = {"exit": exit_code, "seconds": time.monotonic() - started}
    comparison.write_json(directory / "run.json", manifest)
    if exit_code != 0:
        raise RuntimeError(f"Training failed with exit code {exit_code}. See {directory}")
    finished = manifest["steps"]
    assert set(map(int, manifest["evaluations"])) == set(range(0, finished, 10000))
    assert CURRICULUM.findall(text) == [(str(step), str(step))
                                        for step in range(100000, finished, 100000)
                                        if step < STEPS - 1]
    assert f"Training finished at step {finished}" in text
    if width_reference:
        marker = "[Actor Adam LR] step 200000, alpha 0.0001"
        assert text.count(marker) == 1
        assert text.index("[RNN] Weights saved at step 200000") < text.index(
            "[Curriculum] step 200000") < text.index(marker) < text.index("[GRU Adam Diagnostics] 200000,")
        lines = [line.split("] ", 1)[1] for line in text.splitlines()
                 if line.startswith("[GRU Adam Diagnostics] ")]
        assert all(len(line.split(",")) == 14 for line in lines)
        assert all(float(line.split(",")[-1]) == (0.0001 if int(line.split(",")[0]) >= 200000 else 0.001)
                   for line in lines)
        (directory / "actor_update_diagnostics.csv").write_text(
            "step,updates,q_mean,q_abs_max,grad_l2,update_l2,parameter_l2,update_gate_mean,"
            "reset_gate_mean,proposed_action_mean,applied_action_mean,vertical_thrust_over_weight,"
            "mean_burn,actor_alpha\n" + "\n".join(lines) + "\n", encoding="utf-8")
    write_report(directory, manifest)
    comparison.write_json(directory / "status.json", {
        "status": "complete", "steps": finished, "checkpoints": len(manifest["evaluations"]),
        "prefix_verified": sum(e["prefix_verified"] for e in manifest["evaluations"].values()),
    })


if __name__ == "__main__":
    main()
