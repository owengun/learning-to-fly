"""Re-evaluate the coursework checkpoints and save step and episode telemetry."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import shutil
import statistics as stats
import subprocess
from pathlib import Path

from run_comparison import RETURN


BINARIES = {
    "mlp-original": "training_mlp",
    "mlp-matched": "training_mlp_matched",
    "vanilla": "training_headless",
    "lstm": "training_lstm",
    "gru": "training_gru",
}
EPISODE_METRICS = (
    "return", "length", "flight_seconds", "terminated", "time_limit",
    "settled_20cm", "position_rmse_m", "horizontal_rmse_m", "final_distance_m",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def episode_metrics(path, returns, key):
    with path.open(newline="", encoding="utf-8") as stream:
        groups = itertools.groupby(csv.DictReader(stream), key=lambda row: int(row["episode"]))
        for episode, group in groups:
            rows = list(group)
            length = len(rows)
            assert [int(row["step"]) for row in rows] == list(range(1, length + 1))
            last = rows[-1]
            terminated, time_limit = int(last["terminated"]), int(last["time_limit"])
            assert terminated + time_limit == 1
            assert all(row["terminated"] == row["time_limit"] == "0" for row in rows[:-1])
            assert not time_limit or length == 500
            xy = [sum(float(row[axis]) ** 2 for axis in ("x_m", "y_m")) for row in rows]
            xyz = [horizontal + float(row["z_m"]) ** 2 for horizontal, row in zip(xy, rows)]
            total_reward = math.fsum(float(row["reward"]) for row in rows)
            assert math.isclose(total_reward, returns[episode - 1], rel_tol=5.1e-6, abs_tol=0.00005)
            yield dict(key, episode=episode, **{"return": total_reward}, length=length,
                       flight_seconds=math.fsum(float(row["dt_s"]) for row in rows),
                       terminated=terminated, time_limit=time_limit,
                       settled_20cm=int(time_limit and xyz[-1] < 0.2**2),
                       position_rmse_m=math.sqrt(stats.mean(xyz)),
                       horizontal_rmse_m=math.sqrt(stats.mean(xy)),
                       final_distance_m=math.sqrt(xyz[-1]))


def replay(source, output, binaries):
    episodes, checkpoints, provenance, training_episodes, training_sources = [], [], [], [], []
    manifests = sorted(path for root in source for path in root.glob("seed-*/*/attempt-*/run.json"))
    assert len(manifests) == 15
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        actor, seed = manifest["actor"], manifest["seed"]
        assert manifest["steps"] == 50001 and manifest["training"]["exit"] == 0
        assert len(manifest["evaluations"]) == 6
        directory = path.parent / "checkpoints" / actor
        training_csv = directory / "training_episodes.csv"
        if training_csv.exists():
            with training_csv.open(newline="", encoding="utf-8") as stream:
                training_episodes.extend(dict(actor=actor, seed=seed, **row)
                                         for row in csv.DictReader(stream))
        training_sources.append({
            "actor": actor, "seed": seed, "manifest": str(path),
            "training_log": str(path.parent / "training.log"),
            "per_episode_csv": str(training_csv) if training_csv.exists() else None,
            "seconds": manifest["training"]["seconds"],
        })
        for step, previous in manifest["evaluations"].items():
            print(f"{actor} seed={seed} checkpoint={step}", flush=True)
            actor_file = directory / f"actor_step_{step}.txt"
            target_file = directory / f"target_step_{step}.txt"
            assert sha256(actor_file) == previous["checkpoint_sha256"]
            assert sha256(target_file) == previous["target_sha256"]
            location = output / "trajectories" / actor / f"seed-{seed}" / f"step-{step}"
            location.mkdir(parents=True)
            trajectory = location / "steps.csv"
            command = [str(binaries[actor]), "--seed", str(seed), "--episodes", "10",
                       "--evaluate", str(actor_file), "--trajectory-csv", str(trajectory)]
            log = location / "evaluation.log"
            with log.open("wb") as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
            returns = [float(value) for value in RETURN.findall(log.read_text(encoding="utf-8"))]
            assert len(returns) == 10 and returns == previous["returns"], (path, step, returns)
            key = {"actor": actor, "seed": seed, "checkpoint_step": int(step)}
            batch = list(episode_metrics(trajectory, returns, key))
            assert [row["episode"] for row in batch] == list(range(1, 11))
            episodes.extend(batch)
            checkpoints.append(dict(key, **{
                f"mean_{metric}": stats.mean(row[metric] for row in batch)
                for metric in EPISODE_METRICS
            }, length_sd=stats.stdev(row["length"] for row in batch)))
            provenance.append(dict(key, command=command, actor_sha256=sha256(actor_file),
                                   target_sha256=sha256(target_file), original_manifest=str(path),
                                   trajectory_sha256=sha256(trajectory), returns_reproduced=True))
    assert len(checkpoints) == 90 and len(episodes) == 900
    write_csv(output / "evaluation_episodes.csv", episodes)
    write_csv(output / "evaluation_checkpoints.csv", checkpoints)
    if training_episodes:
        write_csv(output / "training_episodes.csv", training_episodes)
    (output / "provenance.json").write_text(json.dumps({
        "evaluators": {actor: {"path": str(path), "sha256": sha256(path)}
                       for actor, path in binaries.items()},
        "training": training_sources, "evaluations": provenance,
    }, indent=2) + "\n", encoding="utf-8")

    seed_summary = []
    for (actor, seed), group in itertools.groupby(checkpoints, key=lambda row: (row["actor"], row["seed"])):
        points = sorted(group, key=lambda row: row["checkpoint_step"])
        end = points[-1]
        area = sum((right["checkpoint_step"] - left["checkpoint_step"])
                   * (left["mean_length"] + right["mean_length"]) / 2
                   for left, right in zip(points, points[1:]))
        seed_summary.append(dict(end, episode_length_auc_steps=area / 50000))
    write_csv(output / "final_by_seed.csv", seed_summary)
    methods = []
    for actor in BINARIES:
        rows = [row for row in seed_summary if row["actor"] == actor]
        assert len(rows) == 3
        methods.append({
            "actor": actor,
            **{column: stats.mean(row[column] for row in rows)
               for column in rows[0] if column not in ("actor", "seed", "checkpoint_step", "length_sd")},
            "length_between_seed_sd": stats.stdev(row["mean_length"] for row in rows),
        })
    write_csv(output / "final_by_method.csv", methods)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched", required=True, type=Path)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--bin", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True)
    archived = output / "evaluators"
    archived.mkdir()
    binaries = {}
    for actor, name in BINARIES.items():
        binaries[actor] = archived / f"{name}.exe"
        shutil.copy2(args.bin / f"{name}.exe", binaries[actor])
    replay([args.matched.resolve(), args.original.resolve()], output, binaries)


if __name__ == "__main__":
    main()
