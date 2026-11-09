"""Run the seed-0 GRU Adam ablation, resuming the live trainer if it passes screening."""

import argparse
import csv
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import run_comparison as comparison
from replay_evaluation import sha256, write_csv
from run_curriculum_pilot import collect


SEED = 0
EPISODES = 10
SCREEN = 100000
FINISH = 300000
DIAGNOSTIC_COLUMNS = (
    "step,updates,q_mean,q_abs_max,grad_l2,update_l2,parameter_l2,update_gate_mean,"
    "reset_gate_mean,proposed_action_mean,applied_action_mean,vertical_thrust_over_weight,mean_burn"
)


def send_decision(gate, decision):
    temporary = gate / "decision.tmp"
    temporary.write_text(decision + "\n", encoding="ascii")
    temporary.replace(gate / "decision")


def wait_until_ready(process, gate):
    while not (gate / "ready").exists():
        if process.poll() is not None:
            raise RuntimeError(f"Trainer exited before screening with code {process.returncode}")
        time.sleep(0.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True, help="Reference run's attempt directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-observation", action="store_true")
    args = parser.parse_args()
    actor = "gru-full" if args.full_observation else "gru-history"
    inputs, hidden = (146, 26) if args.full_observation else (22, 32)
    reference = args.reference.resolve()
    reference_manifest = json.loads((reference / "run.json").read_text(encoding="utf-8"))
    assert reference_manifest["seed"] == SEED and reference_manifest["episodes"] == EPISODES
    assert reference_manifest["actor"] == ("mlp-original" if args.full_observation else "gru-history")
    with (reference / "flight_by_checkpoint.csv").open(newline="", encoding="utf-8") as stream:
        baseline = next(row for row in csv.DictReader(stream) if int(row["checkpoint_step"]) == SCREEN)
    directory = args.output.resolve()
    directory.mkdir(parents=True)
    binary = directory / ("training_gru_full.exe" if args.full_observation else "training_gru_adam.exe")
    shutil.copy2(args.exe, binary)
    gate = directory / "screen_gate"
    gate.mkdir()
    command = [str(binary), "--seed", str(SEED), "--steps", str(FINISH + 1),
               "--screen-gate", str(gate)]
    path = directory / "run.json"
    manifest = {
        "actor": actor, "optimizer": "adam", "seed": SEED, "episodes": EPISODES,
        "steps": FINISH + 1, "planned_steps": FINISH + 1, "command": command,
        "executable_sha256": sha256(binary), "reference": str(reference),
        "config": {"input": inputs, "hidden": hidden, "burn_in": 32, "learn": 8,
                   "alpha": 0.001, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8,
                   "weight_decay": 0.0001, "bias_lr_factor": 1},
    }
    comparison.PARAMETERS[actor] = 3 * hidden * (inputs + hidden + 1) + 4 * (hidden + 1)
    comparison.write_json(path, manifest)
    started = time.monotonic()
    paused_seconds = 0
    with (directory / "training.log").open("wb") as log, (
        directory / "training-error.log"
    ).open("wb") as error:
        process = subprocess.Popen(command, cwd=directory, stdout=log, stderr=error)
        print(f"Training seed {SEED} in {directory}", flush=True)
        try:
            wait_until_ready(process, gate)
            paused = time.monotonic()
            manifest["training"] = {"command": command, "exit": None, "seconds": paused - started}
            comparison.evaluate(directory, manifest, binary, actor, SEED, SCREEN + 1, EPISODES)
            if not args.full_observation:
                assert manifest["evaluations"]["0"]["returns"] == reference_manifest["evaluations"]["0"]["returns"]
            screen = collect(directory, binary)[-1]
            assert screen["checkpoint_step"] == SCREEN
            if args.full_observation:
                advance = screen["mean_time_limit"] >= 0.2 or screen["mean_length"] >= 75.2
                rule = "at least 2/10 completions, or mean length at least 75.2 steps"
            else:
                advance = (screen["mean_time_limit"] >= 0.2 or
                           (screen["mean_length"] >= max(65, float(baseline["mean_length"]) + 20) and
                            screen["mean_lower_boundary_exit"] <= float(baseline["mean_lower_boundary_exit"]) - 0.2 + 1e-9))
                rule = "at least 2/10 completions, or >=65 steps, >=SGD+20 and >=2 fewer downward exits"
            decision = {"advance_to_300k": advance, "rule": rule, "reference": baseline, "adam": screen}
            comparison.write_json(directory / "screen.json", decision)
            send_decision(gate, "continue" if advance else "stop")
            paused_seconds = time.monotonic() - paused
            print(f"100k screen advance={advance}: {screen}", flush=True)
            exit_code = process.wait()
        except BaseException:
            if process.poll() is None:
                if not (gate / "decision").exists():
                    send_decision(gate, "stop")
                process.wait()
            manifest["training"] = {"command": command, "exit": process.returncode,
                                    "seconds": time.monotonic() - started}
            comparison.write_json(path, manifest)
            raise

    manifest["training"] = {"command": command, "exit": exit_code,
                            "seconds": time.monotonic() - started - paused_seconds,
                            "paused_seconds": paused_seconds}
    end = FINISH if advance else SCREEN
    manifest["steps"] = end + 1
    comparison.write_json(path, manifest)
    if exit_code != 0:
        raise RuntimeError(f"Training failed with code {exit_code}. See training-error.log and failure/.")
    log = (directory / "training.log").read_text(encoding="utf-8")
    assert re.search(rf"^Training finished at step {end + 1}$", log, re.M)
    rows = [line.removeprefix("[GRU Adam Diagnostics] ") for line in log.splitlines()
            if line.startswith("[GRU Adam Diagnostics] ")]
    assert rows
    (directory / "actor_update_diagnostics.csv").write_text(
        DIAGNOSTIC_COLUMNS + "\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )
    comparison.evaluate(directory, manifest, binary, actor, SEED, end + 1, EPISODES)
    points = collect(directory, binary)
    write_csv(directory / "final.csv", [points[-1]])
    comparison.write_json(directory / "status.json", {
        "status": "complete", "checkpoint": end, "advance_to_300k": advance,
    })


if __name__ == "__main__":
    main()
