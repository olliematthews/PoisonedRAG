"""Initialise a new experiment folder."""

import argparse
from pathlib import Path
import json


def parse_args():
    parser = argparse.ArgumentParser()

    # Retriever and BEIR datasets
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        help="GPT model to use - can be gpt3.5 or gpt4",
        required=True,
    )
    parser.add_argument(
        "--eval_dataset", type=str, default="nq", help="BEIR dataset to evaluate"
    )
    parser.add_argument("--split", type=str, default="test")

    args = parser.parse_args()
    print(args)

    return args


def main():
    args = parse_args()

    EXPERIMENT_DIR = Path("./results/experiments")
    results_dir = EXPERIMENT_DIR / args.experiment_name
    results_dir.mkdir(exist_ok=True, parents=True)
    dataset_split_string = f"{args.eval_dataset}-{args.split}"

    config = {
        "dataset": dataset_split_string,
        "context_configs": {"standard": [0.5, 5, 0.65]},
        "model": args.model,
        "experiments": [],
        "do_no_context": True,
    }

    with open(results_dir / "config.json", "w") as fd:
        json.dump(config, fd, indent=2)


if __name__ == "__main__":
    main()
