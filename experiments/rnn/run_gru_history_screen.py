"""Screen the reference and action-conditioned GRU serially at 100,000 transitions."""

import argparse
import json
import shutil
from pathlib import Path

import run_comparison as comparison
from run_curriculum_pilot import collect
from replay_evaluation import sha256, write_csv


TARGETS = {"gru": "training_gru", "gru-history": "training_gru_history"}
SEED = 0
CHECKPOINT = 100000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    binaries = root / "executables"
    binaries.mkdir(parents=True, exist_ok=True)
    comparison.PARAMETERS["gru-history"] = 5412
    points = []
    for actor, name in TARGETS.items():
        binary = binaries / f"{name}.exe"
        if not binary.exists():
            shutil.copy2(args.bin / binary.name, binary)
        result = comparison.run(root, binary, actor, SEED, CHECKPOINT + 1, 10)
        directory = root / f"seed-{SEED}" / actor
        for path in sorted(directory.glob("attempt-*/run.json"), reverse=True):
            run = json.loads(path.read_text(encoding="utf-8"))
            if run.get("summary") != result or run["executable_sha256"] != sha256(binary):
                continue
            rows = collect(path.parent, binary)
            assert len(rows) == 11 and rows[-1]["checkpoint_step"] == CHECKPOINT
            points.extend(rows)
            prefix = "[GRU Diagnostics] step,q_mean,q_abs_max,grad_l2,mean_burn "
            log = (path.parent / "training.log").read_text(encoding="utf-8")
            diagnostic_rows = [line[len(prefix):] for line in log.splitlines() if line.startswith(prefix)]
            assert diagnostic_rows
            (path.parent / "actor_update_diagnostics.csv").write_text(
                "step,q_mean,q_abs_max,grad_l2,mean_burn\n" + "\n".join(diagnostic_rows) + "\n",
                encoding="utf-8",
            )
            break
        else:
            raise RuntimeError(f"No completed run found for {actor}")
        write_csv(root / "flight_by_checkpoint.csv", points)
    final = [row for row in points if row["checkpoint_step"] == CHECKPOINT]
    write_csv(root / "final.csv", final)
    reference, history = final
    advance = (history["mean_time_limit"] >= 0.2 or
               history["mean_length"] >= max(65, reference["mean_length"] + 20))
    comparison.write_json(root / "status.json", {
        "status": "complete", "checkpoint": CHECKPOINT, "advance_to_300k": advance,
        "rule": "at least 2/10 full episodes, or mean length >=65 and >= reference+20",
    })


if __name__ == "__main__":
    main()
