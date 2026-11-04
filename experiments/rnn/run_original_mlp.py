"""Supplement the matched experiment with the original 146-input RLtools MLP."""

import argparse
from pathlib import Path

import run_comparison as comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=50001)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    if args.steps < 2 or args.episodes < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Require unique seeds, at least two steps and two episodes")
    comparison.PARAMETERS["mlp-original"] = 13828
    rows = []
    failed = False
    for seed in args.seeds:
        try:
            rows.append(comparison.run(args.output.resolve(), args.exe.resolve(),
                                       "mlp-original", seed, args.steps, args.episodes))
            comparison.export(args.output.resolve(), rows)
        except (OSError, RuntimeError) as error:
            failed = True
            print(error, flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
