"""Run the original MLP and curriculum-enabled GRU serially on one shared seed."""

import argparse
import csv
import json
import shutil
import statistics as stats
from pathlib import Path

import run_comparison as comparison
from replay_evaluation import episode_metrics, sha256, write_csv


ACTORS = {"mlp-original": "training_mlp", "gru": "training_gru"}
SEED = 0
STEPS = 300001
EPISODES = 10


def collect(directory, binary):
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    actor = run["actor"]
    checkpoint_dir = directory / "checkpoints" / actor
    episodes, summary, provenance = [], [], []
    for step, evaluation in sorted(run["evaluations"].items(), key=lambda pair: int(pair[0])):
        checkpoint = checkpoint_dir / f"actor_step_{step}.txt"
        assert sha256(checkpoint) == evaluation["checkpoint_sha256"]
        target = checkpoint_dir / f"target_step_{step}.txt"
        assert sha256(target) == evaluation["target_sha256"]
        label = f"flight-{step}"
        trajectory = directory / f"{label}.csv"
        command = [str(binary), "--seed", str(SEED), "--episodes", str(EPISODES),
                   "--evaluate", str(checkpoint), "--trajectory-csv", str(trajectory)]
        status = comparison.invoke(command, directory, label)
        if status["exit"] != 0:
            raise RuntimeError(f"Flight telemetry failed: {directory / (label + '-error.log')}")
        log = (directory / f"{label}.log").read_text(encoding="utf-8")
        returns = [float(value) for value in comparison.RETURN.findall(log)]
        assert returns == evaluation["returns"]
        key = {"actor": actor, "seed": SEED, "checkpoint_step": int(step)}
        batch = list(episode_metrics(trajectory, returns, key))
        assert len(batch) == EPISODES
        with trajectory.open(newline="", encoding="utf-8") as stream:
            endings = [row for row in csv.DictReader(stream) if row["terminated"] == "1"]
        lower = {int(row["episode"]) for row in endings if float(row["z_m"]) < -0.6}
        for row in batch:
            row["lower_boundary_exit"] = int(row["episode"] in lower)
        episodes.extend(batch)
        summary.append(dict(key, **{
            f"mean_{field}": stats.mean(row[field] for row in batch)
            for field in ("return", "length", "flight_seconds", "time_limit",
                          "lower_boundary_exit", "settled_20cm", "position_rmse_m")
        }))
        provenance.append(dict(key, command=command, actor_sha256=evaluation["checkpoint_sha256"],
                               target_sha256=evaluation["target_sha256"],
                               trajectory_sha256=sha256(trajectory)))
    write_csv(directory / "flight_episodes.csv", episodes)
    write_csv(directory / "flight_by_checkpoint.csv", summary)
    comparison.write_json(directory / "flight_provenance.json", provenance)
    return summary


def report(root, points):
    write_csv(root / "flight_by_checkpoint.csv", points)
    lines = [
        "# Original MLP and GRU curriculum pilot", "",
        "Seed 0, 300,001 simulator transitions per method. Both use the stock curriculum "
        "schedule and recalculate rewards only for populated replay entries. "
        "Training runs serially. Each saved checkpoint is evaluated in a fresh process on "
        "ten episodes with the same seed, a 500-step cap, and no exploration action noise.", "",
        "| Method | Checkpoint | Mean return | Mean length | Completion | Lower boundary exit |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for point in points:
        step = point["checkpoint_step"]
        if step % 50000 == 0:
            lines.append(f"| {point['actor']} | {step} | {point['mean_return']:.2f} | "
                         f"{point['mean_length']:.2f} | {point['mean_time_limit']:.0%} | "
                         f"{point['mean_lower_boundary_exit']:.0%} |")
    lines += ["", "This is a one-seed pilot. The original MLP still uses 146 inputs and Adam. "
              "GRU uses 18 inputs and SGD. The comparison measures each configuration as a whole.",
              "At a given checkpoint label, the original MLP is saved before that transition and "
              "the GRU is saved after it. Both training jobs complete 300,001 transitions.", ""]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    binaries = root / "executables"
    binaries.mkdir(parents=True, exist_ok=True)
    comparison.PARAMETERS["mlp-original"] = 13828
    points, results = [], []
    for actor, name in ACTORS.items():
        binary = binaries / f"{name}.exe"
        if not binary.exists():
            shutil.copy2(args.bin / binary.name, binary)
        results.append(comparison.run(root, binary, actor, SEED, STEPS, EPISODES))
        parent = root / f"seed-{SEED}" / actor
        for path in sorted(parent.glob("attempt-*/run.json"), reverse=True):
            run = json.loads(path.read_text(encoding="utf-8"))
            if run.get("summary") == results[-1] and run["executable_sha256"] == sha256(binary):
                log = (path.parent / "training.log").read_text(encoding="utf-8")
                for step in (100000, 200000):
                    assert f"[Curriculum] step {step}, recalculated {step} replay rewards" in log
                points.extend(collect(path.parent, binary))
                break
        else:
            raise RuntimeError(f"No completed training attempt found for {actor}")
        report(root, points)
        comparison.export(root, results)
    comparison.write_json(root / "status.json", {"status": "complete", "actors": list(ACTORS)})


if __name__ == "__main__":
    main()
