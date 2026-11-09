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
MILESTONES = {90000, 100000, 110000, 160000, 190000, 200000, 210000, 240000, 300000}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def thrust_ratio(row):
    # Simulator rotor-thrust coefficient, tilt projection and vehicle weight.
    qx, qy = float(row["qx"]), float(row["qy"])
    rotor_sq = sum(float(row[f"rpm_{i}"]) ** 2 for i in range(4))
    return 3.16e-10 * rotor_sq * (1 - 2 * (qx * qx + qy * qy)) / (0.027 * 9.81)


def summarize(trajectory, returns, actor, step):
    key = {"actor": actor, "seed": SEED, "checkpoint_step": step}
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
    assert len(episodes) == EPISODES
    columns = ("length", "time_limit", "lower_exit", "return", "first10_thrust_over_weight",
               "last5_thrust_over_weight", "position_rmse_m")
    return episodes, dict(key, **{
        f"mean_{name}": stats.mean(row[name] for row in episodes) for name in columns
    })


def checkpoint_hashes(directory, actor, step):
    folder = directory / "checkpoints" / actor
    return {role: sha256(folder / f"{role}_step_{step}.txt") for role in ("actor", "target")}


def evaluate(directory, executable, step, screening, baseline, manifest):
    key = str(step)
    hashes = checkpoint_hashes(directory, "gru-full", step)
    old = screening["evaluations"].get(key)
    if old is not None:
        assert hashes == {"actor": old["checkpoint_sha256"], "target": old["target_sha256"]}, (
            f"Seed-0 checkpoint prefix differs at {step}")
    folder = directory / "evaluation" / f"step-{step}"
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "checkpoints" / "gru-full" / f"actor_step_{step}.txt"
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
    episodes, gru = summarize(trajectory, returns, "gru-full", step)
    original = load(baseline / "run.json")["evaluations"][key]
    baseline_hashes = checkpoint_hashes(baseline, "mlp-original", step)
    assert baseline_hashes == {"actor": original["checkpoint_sha256"],
                              "target": original["target_sha256"]}
    baseline_csv = baseline / f"flight-{step}.csv"
    baseline_episodes, mlp = summarize(baseline_csv, original["returns"], "mlp-original", step)
    write_csv(folder / "episodes.csv", episodes + baseline_episodes)
    result.update(returns=returns, actor_sha256=hashes["actor"], target_sha256=hashes["target"],
                  trajectory_sha256=sha256(trajectory), prefix_verified=old is not None,
                  original_mlp_trajectory=str(baseline_csv),
                  original_mlp_trajectory_sha256=sha256(baseline_csv), summaries=[mlp, gru])
    manifest["evaluations"][key] = result
    comparison.write_json(directory / "run.json", manifest)
    write_csv(directory / "flight_by_checkpoint.csv", [
        summary for evaluation in manifest["evaluations"].values()
        for summary in evaluation["summaries"]
    ])
    print(f"{step}: GRU {gru['mean_length']:.1f} steps, {gru['mean_time_limit']:.0%} completed; "
          f"MLP {mlp['mean_length']:.1f} steps, {mlp['mean_time_limit']:.0%} completed", flush=True)


def write_report(directory, manifest):
    lines = [
        "# Full-budget full-observation GRU", "",
        "The prior 100,001-interaction process had exited with weight-only checkpoints. "
        "This run restarted from seed 0 with the exact screened executable and no training gate. "
        "The Actor and target hashes and all ten evaluation returns match the screened run "
        "at every checkpoint through 100,000 interactions.", "",
        "The same 146 observation columns and 26 hidden units, Adam configuration and curriculum "
        "are retained. The original MLP rows are its already saved seed-0 snapshots at the same "
        "interaction counts. All rows use ten episodes per checkpoint without exploration action noise.",
        "",
        "| Step | Actor | Mean length | 500-step completion | Downward exits | Mean return | First 10 thrust / weight | Last 5 thrust / weight |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for step, evaluation in manifest["evaluations"].items():
        if int(step) not in MILESTONES:
            continue
        for row in evaluation["summaries"]:
            lines.append(f"| {step} | {row['actor']} | {row['mean_length']:.1f} | "
                         f"{row['mean_time_limit']:.0%} | {row['mean_lower_exit']:.0%} | "
                         f"{row['mean_return']:.2f} | {row['mean_first10_thrust_over_weight']:.3f} | "
                         f"{row['mean_last5_thrust_over_weight']:.3f} |")
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
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--mlp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    screen = args.screen.resolve()
    baseline = args.mlp.resolve()
    directory = args.output.resolve()
    screening = load(screen / "run.json")
    original = load(baseline / "run.json")
    assert (screening["actor"], screening["seed"], screening["steps"], screening["episodes"]) == (
        "gru-full", SEED, 100001, EPISODES)
    assert (original["actor"], original["seed"], original["steps"], original["episodes"]) == (
        "mlp-original", SEED, STEPS, EPISODES)
    assert screening["config"]["input"] == 146 and screening["config"]["hidden"] == 26
    source = screen / "training_gru_full.exe"
    assert sha256(source) == screening["executable_sha256"]
    directory.mkdir(parents=True)
    executable = directory / source.name
    shutil.copy2(source, executable)
    command = [str(executable), "--seed", str(SEED), "--steps", str(STEPS)]
    manifest = {"actor": "gru-full", "seed": SEED, "steps": STEPS, "episodes": EPISODES,
                "command": command, "executable_sha256": sha256(executable),
                "config": screening["config"], "screen": str(screen),
                "reference": str(baseline), "evaluations": {}}
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
                        evaluate(directory, executable, int(label), screening, baseline, manifest)
                if exit_code is not None:
                    break
                time.sleep(1)
    manifest["training"] = {"exit": exit_code, "seconds": time.monotonic() - started}
    comparison.write_json(directory / "run.json", manifest)
    if exit_code != 0:
        raise RuntimeError(f"Training failed with exit code {exit_code}. See {directory}")
    assert set(map(int, manifest["evaluations"])) == set(range(0, STEPS, 10000))
    assert CURRICULUM.findall(text) == [("100000", "100000"), ("200000", "200000")]
    assert "Training finished at step 300001" in text
    write_report(directory, manifest)
    comparison.write_json(directory / "status.json", {"status": "complete", "steps": STEPS,
                                                      "checkpoints": 31, "prefix_verified": 11})


if __name__ == "__main__":
    main()
