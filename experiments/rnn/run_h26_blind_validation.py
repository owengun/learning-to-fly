"""Run the four preregistered blind-validation jobs sequentially, without retries."""

import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from replay_evaluation import sha256, write_csv
from run_comparison import RETURN, write_json
from run_gru_full_budget import CURRICULUM, summarize


STEPS = 300001
EPISODES = 10
ACTORS = ("mlp-original", "gru-full-decay")
ORDER = [(seed, actor) for seed in (1, 2) for actor in ACTORS]
EXPECTED_HASHES = {
    "mlp-original": "42e6625e067f7e3992a1d93f4bbc9a8acf4ac2e37b48e0a640c0dd98158bc80d",
    "gru-full-decay": "b92f4a14a29875bbfb64f81f94a509f7ff1fa2132641a2242600d1cdc1d698b7",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def event(root, kind, **details):
    with (root / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(time=now(), event=kind, **details)) + "\n")


def invoke(command, directory, label, manifest, manifest_path):
    started = time.monotonic()
    with (directory / f"{label}.log").open("wb") as stdout, (directory / f"{label}-error.log").open("wb") as stderr:
        with subprocess.Popen(command, cwd=directory, stdout=stdout, stderr=stderr) as process:
            manifest["active_process"] = dict(pid=process.pid, command=command, started=now())
            write_json(manifest_path, manifest)
            code = process.wait()
    manifest.pop("active_process")
    return dict(command=command, exit=code, seconds=time.monotonic() - started)


def run_one(root, seed, actor, protocol):
    directory = root / f"seed-{seed}" / actor
    directory.mkdir(parents=True)
    source = root / "executables" / protocol["executables"][actor]["name"]
    assert sha256(source) == EXPECTED_HASHES[actor]
    executable = directory / source.name
    shutil.copy2(source, executable)
    checkpoint_actor = "mlp-original" if actor == "mlp-original" else "gru-full"
    config = protocol["config"][actor]
    manifest = dict(actor=actor, seed=seed, steps=STEPS, episodes=EPISODES,
                    checkpoint_actor=checkpoint_actor, config=config,
                    executable_sha256=sha256(executable), protocol_sha256=sha256(root / "protocol.json"),
                    source_commit=protocol["source_commit"], evaluations={}, status="training")
    path = directory / "run.json"
    write_json(path, manifest)
    event(root, "run_started", seed=seed, actor=actor, directory=str(directory))
    command = [str(executable), "--seed", str(seed), "--steps", str(STEPS)]
    manifest["training"] = invoke(command, directory, "training", manifest, path)
    manifest["status"] = "validating_training"
    write_json(path, manifest)
    if manifest["training"]["exit"] != 0:
        raise RuntimeError(f"Training failed for {actor}, seed {seed}, exit {manifest['training']['exit']}. See {directory}")
    log = (directory / "training.log").read_text(encoding="utf-8")
    assert CURRICULUM.findall(log) == [("100000", "100000"), ("200000", "200000")]
    assert "Training took:" in log
    if actor == "gru-full-decay":
        assert "Training finished at step 300001" in log
        marker = "[Actor Adam LR] step 200000, alpha 0.0001"
        assert log.count(marker) == 1
        order = ["[RNN] Weights saved at step 200000", "[Curriculum] step 200000", marker,
                 "[GRU Adam Diagnostics] 200000,"]
        indices = [log.index(value) for value in order]
        assert indices == sorted(indices)
        lines = [line.split("] ", 1)[1] for line in log.splitlines()
                 if line.startswith("[GRU Adam Diagnostics] ")]
        assert lines and all(len(line.split(",")) == 14 for line in lines)
        assert all(float(line.split(",")[-1]) == (0.001 if int(line.split(",")[0]) < 200000 else 0.0001)
                   for line in lines)
        header = ("step,updates,q_mean,q_abs_max,grad_l2,update_l2,parameter_l2,update_gate_mean,"
                  "reset_gate_mean,proposed_action_mean,applied_action_mean,vertical_thrust_over_weight,mean_burn,actor_alpha")
        (directory / "actor_update_diagnostics.csv").write_text(header + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    manifest["status"] = "evaluating"
    write_json(path, manifest)
    summaries, all_episodes = [], []
    for step in range(0, STEPS, 10000):
        checkpoint = directory / "checkpoints" / checkpoint_actor / f"actor_step_{step}.txt"
        target = checkpoint.with_name(f"target_step_{step}.txt")
        folder = directory / "evaluation" / f"step-{step}"
        folder.mkdir(parents=True)
        trajectory = folder / "steps.csv"
        command = [str(executable), "--seed", str(seed), "--episodes", str(EPISODES),
                   "--evaluate", str(checkpoint), "--trajectory-csv", str(trajectory)]
        result = invoke(command, folder, "evaluation", manifest, path)
        if result["exit"] != 0:
            raise RuntimeError(f"Evaluation failed for {actor}, seed {seed}, step {step}. See {folder}")
        result["returns"] = [float(value) for value in RETURN.findall((folder / "evaluation.log").read_text(encoding="utf-8"))]
        assert len(result["returns"]) == EPISODES
        episodes, summary = summarize(trajectory, result["returns"], actor, step, seed=seed)
        write_csv(folder / "episodes.csv", episodes)
        summaries.append(summary)
        all_episodes.extend(episodes)
        result.update(actor_sha256=sha256(checkpoint), target_sha256=sha256(target),
                      trajectory_sha256=sha256(trajectory), summary=summary)
        manifest["evaluations"][str(step)] = result
        write_json(path, manifest)
    assert len(summaries) == 31 and len(all_episodes) == 310
    write_csv(directory / "flight_by_checkpoint.csv", summaries)
    write_csv(directory / "all_episodes.csv", all_episodes)
    manifest["status"] = "complete"
    write_json(path, manifest)
    write_json(directory / "config.json", config)
    event(root, "run_complete", seed=seed, actor=actor, training=manifest["training"], final=summaries[-1])
    print(json.dumps(dict(event="run_complete", seed=seed, actor=actor, final=summaries[-1])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mlp-exe", type=Path, required=True)
    parser.add_argument("--gru-exe", type=Path, required=True)
    parser.add_argument("--seed0-gru", type=Path, required=True)
    parser.add_argument("--seed0-mlp", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    sources = dict(zip(ACTORS, (args.mlp_exe.resolve(), args.gru_exe.resolve())))
    for actor, source in sources.items():
        assert sha256(source) == EXPECTED_HASHES[actor], f"Frozen executable mismatch for {actor}"
    seed0_gru, seed0_mlp = args.seed0_gru.resolve(), args.seed0_mlp.resolve()
    reference = load(seed0_gru / "run.json")
    assert reference["training"]["exit"] == 0 and reference["executable_sha256"] == EXPECTED_HASHES[ACTORS[1]]
    assert load(seed0_mlp / "run.json")["executable_sha256"] == EXPECTED_HASHES[ACTORS[0]]
    root.mkdir(parents=True)
    (root / "executables").mkdir()
    repository = Path(__file__).resolve().parents[2]
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    snapshot = root / "source_snapshot"
    snapshot.mkdir()
    paths = [*repository.glob("src/config/*.h"), repository / "src/training.cpp", repository / "src/training.h",
             repository / "src/steps/curriculum.h", repository / "experiments/rnn/actor_adam.h",
             repository / "experiments/rnn/action_history_update.h", Path(__file__).resolve(),
             repository / "experiments/rnn/run_gru_full_budget.py", repository / "experiments/rnn/replay_evaluation.py",
             repository / "experiments/rnn/run_comparison.py", repository / "experiments/rnn/H26_BLIND_VALIDATION.md"]
    source_hashes = {}
    for path in paths:
        relative = path.relative_to(repository)
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        source_hashes[str(relative)] = sha256(path)
    for actor, source in sources.items():
        shutil.copy2(source, root / "executables" / source.name)
    protocol = dict(created=now(), source_commit=source_commit, source_hashes=source_hashes,
                    order=ORDER, steps=STEPS, episodes=EPISODES, checkpoint_interval=10000,
                    evaluation_seed_mapping="Use the training seed as --seed in a fresh evaluation process, unchanged executable RNG mapping",
                    evaluation_timing="Evaluate each 10k saved checkpoint after its entire training job exits, before starting the next trainer",
                    ranking="Per seed, lexicographic 300k completion descending, all-episode position cost ascending, then return descending; completed-only cost is supplemental",
                    superiority="Require no lower 300k completion in any seed and a nonlosing lexicographic rank in every seed, with at least one strict win; report any instability separately",
                    seed0=dict(gru=str(seed0_gru), mlp=str(seed0_mlp), tuning_seed=True),
                    executables={actor: dict(name=source.name, sha256=EXPECTED_HASHES[actor], original_path=str(source))
                                 for actor, source in sources.items()},
                    config={ACTORS[0]: dict(input=146, hidden=[64, 64], actor="original RLtools MLP", optimizer="original Adam", training="original repository TD3 and curriculum", seed0_executable_hash=EXPECTED_HASHES[ACTORS[0]]),
                            ACTORS[1]: reference["config"]},
                    aggregate="Sample standard deviation across seed-level metrics (ddof=1). First-10/10 not reached and no completed episodes remain missing with counts, never imputed.")
    write_json(root / "protocol.json", protocol)
    frozen_hash = sha256(root / "protocol.json")
    write_json(root / "status.json", dict(status="running", completed=[], protocol_sha256=frozen_hash))
    completed = []
    try:
        for seed, actor in ORDER:
            assert sha256(root / "protocol.json") == frozen_hash
            run_one(root, seed, actor, protocol)
            completed.append(dict(seed=seed, actor=actor))
            write_json(root / "status.json", dict(status="running", completed=completed, protocol_sha256=frozen_hash))
    except Exception as exc:
        write_json(root / "status.json", dict(status="failed", completed=completed, failed_seed=seed,
                                              failed_actor=actor, reason=str(exc), protocol_sha256=frozen_hash))
        event(root, "failure", seed=seed, actor=actor, reason=str(exc))
        raise
    write_json(root / "status.json", dict(status="complete", completed=completed, protocol_sha256=frozen_hash))


if __name__ == "__main__":
    main()
