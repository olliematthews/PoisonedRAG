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

results_dir = EXPERIMENT_DIR / experiment_name
results_dir.mkdir(exist_ok=True, parents=True)

with open(results_dir / "config.json", "r") as fd:
    experiment_config = json.load(fd)

with open(CACHE_DIR / "RETRIEVAL_DUMP-nq-test.p", "rb") as fd:
    test_cases, corpus = pickle.load(fd)


def get_context(row, n):
    contexts = sorted(
        row["attack_contexts"][:n], key=lambda x: x["score"], reverse=True
    )[:n]

    contexts = [
        {"context": corpus[key]["text"], "score": None} for key in row["gt_ids"]
    ] + contexts

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
