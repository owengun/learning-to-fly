"""Run and summarize the observation-matched TD3 comparison serially."""

import argparse
import csv
import hashlib
import json
import re
import statistics
import subprocess
import time
from pathlib import Path


PARAMETERS = {"mlp-matched": 5636, "vanilla": 1764, "lstm": 6660, "gru": 5028}
RETURN = re.compile(r"^Evaluation episode \d+ return (\S+)$", re.MULTILINE)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def invoke(command, cwd, name):
    started = time.monotonic()
    with (cwd / (name + ".log")).open("wb") as stdout, (
        cwd / (name + "-error.log")
    ).open("wb") as stderr:
        process = subprocess.run(command, cwd=cwd, stdout=stdout, stderr=stderr)
    return {"command": command, "exit": process.returncode,
            "seconds": time.monotonic() - started}


def evaluate(directory, manifest, executable, actor, seed, steps, episodes):
    checkpoints = directory / "checkpoints" / actor
    evaluated = manifest.setdefault("evaluations", {})
    for step in sorted(set(range(0, steps, 10000)) | {steps - 1}):
        key = str(step)
        previous = evaluated.get(key, {})
        if previous.get("exit") == 0 and len(previous.get("returns", [])) == episodes:
            continue
        checkpoint = checkpoints / f"actor_step_{step}.txt"
        target = checkpoints / f"target_step_{step}.txt"
        attempt = 1 + max(
            (int(path.stem.rsplit("-", 1)[1]) for path in directory.glob(f"eval-{step}-attempt-*.log")
             if not path.stem.endswith("-error")),
            default=0,
        )
        label = f"eval-{step}-attempt-{attempt}"
        command = [str(executable), "--seed", str(seed), "--episodes", str(episodes),
                   "--evaluate", str(checkpoint)]
        result = invoke(command, directory, label)
        result.update(attempt=attempt,
                      checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      target_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        log = (directory / (label + ".log")).read_text(encoding="utf-8")
        result["returns"] = [float(value) for value in RETURN.findall(log)]
        evaluated[key] = result
        write_json(directory / "run.json", manifest)
        if result["exit"] != 0 or len(result["returns"]) != episodes:
            raise RuntimeError(f"Incomplete evaluation, see {directory / (label + '.log')}")
    points = sorted((int(step), statistics.mean(data["returns"]))
                    for step, data in evaluated.items())
    returns = evaluated[str(steps - 1)]["returns"]
    area = sum((right - left) * (a + b) / 2
               for (left, a), (right, b) in zip(points, points[1:]))
    manifest["summary"] = {
        "actor": actor, "seed": seed, "steps": steps, "parameters": PARAMETERS[actor],
        "final_mean": statistics.mean(returns), "final_episode_std": statistics.stdev(returns),
        "best_mean": max(mean for _, mean in points),
        "best_step": max(points, key=lambda point: point[1])[0],
        "step_normalized_auc": area / (steps - 1),
        "training_seconds": manifest["training"]["seconds"],
    }
    write_json(directory / "run.json", manifest)


def run(root, executable, actor, seed, steps, episodes):
    parent = root / f"seed-{seed}" / actor
    parent.mkdir(parents=True, exist_ok=True)
    prior = sorted(parent.glob("attempt-*"))
    executable_hash = hashlib.sha256(executable.read_bytes()).hexdigest()
    for directory in reversed(prior):
        path = directory / "run.json"
        if not path.exists():
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        same_protocol = (
            manifest.get("steps") == steps and manifest.get("episodes") == episodes
            and manifest.get("executable_sha256") == executable_hash
        )
        if same_protocol and manifest.get("training", {}).get("exit") == 0:
            evaluate(directory, manifest, executable, actor, seed, steps, episodes)
            return manifest["summary"]
    directory = parent / f"attempt-{len(prior) + 1:04}"
    directory.mkdir()
    command = [str(executable), "--seed", str(seed), "--steps", str(steps)]
    manifest = {"actor": actor, "seed": seed, "steps": steps, "episodes": episodes,
                "executable_sha256": executable_hash,
                "command": command}
    write_json(directory / "run.json", manifest)
    print(f"Training {actor}, seed {seed}. Logs in {directory}", flush=True)
    manifest["training"] = invoke(command, directory, "training")
    write_json(directory / "run.json", manifest)
    if manifest["training"]["exit"] != 0:
        raise RuntimeError(f"Training failed, see {directory / 'training-error.log'}")
    evaluate(directory, manifest, executable, actor, seed, steps, episodes)
    return manifest["summary"]


def export(root, rows):
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped = {}
    for actor in PARAMETERS:
        values = [row["final_mean"] for row in rows if row["actor"] == actor]
        if values:
            grouped[actor] = {"seed_count": len(values), "final_mean": statistics.mean(values),
                              "final_seed_std": statistics.stdev(values) if len(values) > 1 else None}
    write_json(root / "across-seeds.json", grouped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exe", action="append", required=True, metavar="ACTOR=EXE")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--steps", type=int, default=50001)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    executables = {actor: Path(path).resolve()
                   for actor, path in (entry.split("=", 1) for entry in args.exe)}
    if (set(executables) != set(PARAMETERS) or args.steps < 2 or args.episodes < 2
            or len(set(args.seeds)) != len(args.seeds)):
        parser.error("Require four actors, unique seeds, at least two steps and two episodes")
    root = args.output.resolve()
    rows = []
    failed = False
    for seed in args.seeds:
        for actor in PARAMETERS:
            try:
                rows.append(run(root, executables[actor], actor, seed, args.steps, args.episodes))
                export(root, rows)
            except (OSError, RuntimeError) as error:
                failed = True
                print(error, flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
