import pickle
from pathlib import Path
import pandas as pd
from src.prompts.prompts import wrap_prompt, get_prompts
import asyncio
from src.models import create_model
import tqdm
import json
import numpy as np
from src.models.embedding import get_embeddings

CACHE_DIR = Path("./.cache")
EXPERIMENT_DIR = Path("./results/experiments")

experiment_name = "varying_n_4_red"

results_dir = EXPERIMENT_DIR / experiment_name
results_dir.mkdir(exist_ok=True, parents=True)

with open(results_dir / "config.json", "r") as fd:
    experiment_config = json.load(fd)

with open(CACHE_DIR / "RETRIEVAL_DUMP-nq-test.p", "rb") as fd:
    test_cases, corpus = pickle.load(fd)

emb_df = pd.read_pickle(CACHE_DIR / "GPT_EMBEDDING_DUMP-nq-test.p")
qemb_df = pd.read_pickle(CACHE_DIR / "GPT_QEMBEDDING_DUMP-nq-test.p")
extended_corpus_df = pd.read_pickle(CACHE_DIR / "EXTENDED_CORPUS-nq-test.p")


def get_context(row, n, similarity_rej_thresh):
    contexts = sorted(
        row["attack_contexts"][:n], key=lambda x: x["score"], reverse=True
    )

    context_list = [
        {"context": corpus[key]["text"], "score": None} for key in row["gt_ids"]
    ] + contexts

    qemb = np.array(qemb_df.loc[row.name]["gpt_embedding"])
    qemb_norm = qemb / np.linalg.norm(qemb)

    if similarity_rej_thresh:
        contexts = []
        embeddings = []
        for c in context_list:
            emb = np.array(get_embeddings([c["context"]])[0])
            emb_rel = emb - qemb
            emb_rel_norm = emb_rel / np.linalg.norm(emb_rel)
            if not any(
                [np.dot(emb_rel_norm, e) > similarity_rej_thresh for e in embeddings]
            ):
                contexts.append(c)
                embeddings.append(emb_rel_norm)
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
