import pandas as pd
from pathlib import Path
import numpy as np
import pickle
import json

experiment_name = "final_4"


CACHE_DIR = Path("./.cache")
EXPERIMENT_DIR = Path("./results/experiments")

results_dir = EXPERIMENT_DIR / experiment_name
with open(results_dir / "config.json", "r") as fd:
    experiment_config = json.load(fd)

emb_df = pd.read_pickle(CACHE_DIR / "GPT_EMBEDDING_DUMP-nq-test.p")
qemb_df = pd.read_pickle(CACHE_DIR / "GPT_QEMBEDDING_DUMP-nq-test.p")
extended_corpus_df = pd.read_pickle(CACHE_DIR / "EXTENDED_CORPUS-nq-test.p")

with open(CACHE_DIR / "RETRIEVAL_DUMP-nq-test.p", "rb") as fd:
    test_cases, _ = pickle.load(fd)

questions_df = pd.DataFrame(test_cases)
questions_df.set_index("qid", inplace=True)

questions_df.to_pickle(results_dir / "questions.p")

all_context_embeddings = np.array(emb_df["gpt_embedding"].to_list())
all_context_embeddings_norm = (
    all_context_embeddings / np.linalg.norm(all_context_embeddings, axis=1)[:, None]
)


def get_context(row, accept_thresh, n, similarity_rej_thresh=None):
    qemb = np.array(qemb_df.loc[row.name]["gpt_embedding"])
    qemb_norm = qemb / np.linalg.norm(qemb)

    cos_sim = np.dot(all_context_embeddings_norm, qemb_norm)

    relevant = emb_df[cos_sim > accept_thresh]
    relevant["relevance"] = cos_sim[cos_sim > accept_thresh]

    relevant = relevant.sort_values(by="relevance", ascending=False)

    if similarity_rej_thresh:
        contexts = []
        embeddings = []
        for qid, c in relevant.iterrows():
            emb = np.array(c["gpt_embedding"])
            emb_rel = emb - qemb
            emb_rel_norm = emb_rel / np.linalg.norm(emb_rel)
            if not any(
                [np.dot(emb_rel_norm, e) > similarity_rej_thresh for e in embeddings]
            ):
                contexts.append(qid)
                embeddings.append(emb_rel_norm)

            if len(contexts) >= n:
                break
    else:
        contexts = relevant.index[:n]
    return extended_corpus_df.loc[contexts]["text"].to_list()


context_df = pd.concat(
    {
        col: questions_df.apply(get_context, args=args_, axis=1)
        for col, args_ in experiment_config["context_configs"].items()
    },
    axis=1,
)

context_df.to_pickle(results_dir / "context.p")
