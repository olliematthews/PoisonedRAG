import pickle
from pathlib import Path
import pandas as pd
from src.prompts.prompts import wrap_prompt, get_prompts
import asyncio
from src.models import create_model
import tqdm
import json

CACHE_DIR = Path("./.cache")
EXPERIMENT_DIR = Path("./results/experiments")

experiment_name = "varying_n"
experiment_config = {
    "context_configs": {
        "n=2": (2, 1, True, True),
        "n=3": (3, 1, True, True),
    },
    "model": "gpt3.5",
    "prompt_types": ["cot"],
    "do_no_context": False,
}

results_dir = EXPERIMENT_DIR / experiment_name
results_dir.mkdir(exist_ok=True, parents=True)

with open(results_dir / "config.json", "w") as fd:
    json.dump(experiment_config, fd, indent=2)

with open(CACHE_DIR / "RETRIEVAL_DUMP-nq-test.p", "rb") as fd:
    test_cases, corpus = pickle.load(fd)


def get_context(row, n_contexts, n_adv, poison, input_gt=False):
    if n_contexts is None:
        return None
    contexts = []

    ds_contexts = sorted(
        row["dataset_contexts"].items(), key=lambda x: x[1], reverse=True
    )
    contexts = [
        {"context": corpus[key]["text"], "score": score}
        for key, score in ds_contexts[:n_contexts]
    ]

    if poison:
        contexts += row["attack_contexts"][:n_adv]
        contexts = sorted(contexts, key=lambda x: x["score"], reverse=True)[:n_contexts]

    if input_gt:
        contexts = [
            {"context": corpus[key]["text"], "score": None} for key in row["gt_ids"]
        ] + contexts
        contexts = contexts[:n_contexts]

    return [c["context"] for c in contexts]


questions_df = pd.DataFrame(test_cases)
questions_df.set_index("qid", inplace=True)

questions_df.to_pickle(results_dir / "questions.p")

context_df = pd.concat(
    {
        col: questions_df.apply(get_context, args=args_, axis=1)
        for col, args_ in experiment_config["context_configs"].items()
    },
    axis=1,
)

context_df.to_pickle(results_dir / "context.p")
