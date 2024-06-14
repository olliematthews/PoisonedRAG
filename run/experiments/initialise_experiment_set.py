"""Initialise a new experiment folder."""

import argparse
import json
import sys
from pathlib import Path

main_dir_path = str(Path(__file__).parent.parent.parent)
if main_dir_path not in sys.path:
    sys.path.append(main_dir_path)

from poisoned_rag_defence.logger import logger
from run.experiments.experiment import Experiment


def parse_args():
    parser = argparse.ArgumentParser("Initialise experiment")

    # Retriever and BEIR datasets
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument(
        "--eval_dataset", type=str, default="nq", help="BEIR dataset to evaluate"
    )
    parser.add_argument(
        "--split", type=str, default="test", help="The split on the BEIR dataset to use"
    )

    args = parser.parse_args()

    if args.eval_dataset == "msmarco":
        assert args.split == "train", "PoisonedRAG requires train split on msmarco"
    print(args)

    return args


def main():
    args = parse_args()
    dataset_split_string = f"{args.eval_dataset}-{args.split}"

    config = {
        "dataset": dataset_split_string,
        "retriever_configs": {
            "standard": [
                "openai",
                {"n_contexts": 5, "accept_thresh": 0.5, "similarity_rej_thresh": 0.65},
            ],
        },
        "model": "gpt4",
        "experiments": [["standard", "cot"]],
        "do_no_context": True,
        "n_questions": None,
    }

    experiment = Experiment(args.experiment_name)

    experiment.save_config(config)
    logger.info(
        f"Saved default config for experiment {args.experiment_name}: {json.dumps(config, indent=2)}"
    )
    logger.info(
        f"Please edit the config at {experiment.results_dir / 'config.json'} as desired"
    )


if __name__ == "__main__":
    main()
