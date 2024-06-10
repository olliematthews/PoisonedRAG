import pickle
from pathlib import Path
import pandas as pd
from src.prompts.prompts import wrap_prompt, get_prompts
import asyncio
from src.models import create_model
from src.models.embedding import get_embeddings
import tqdm
import json
import numpy as np
import tqdm

CACHE_DIR = Path("./.cache")

with open(CACHE_DIR / "RETRIEVAL_DUMP-nq-test.p", "rb") as fd:
    test_cases, corpus = pickle.load(fd)

index = 0


def to_context(row):
    global index
    ret = {}
    for attack in row["attack_contexts"]:
        ret[f"attack_{index}"] = {
            "title": row["question"],
            "text": attack["context"][len(row["question"]) + 1 :],
        }
        index += 1

    return ret


test_case_df = pd.DataFrame(test_cases)

for item in test_case_df.apply(to_context, axis=1).to_list():
    corpus.update(item)

corpus_df = pd.DataFrame(corpus.values())
corpus_df.index = corpus.keys()
corpus_df.to_pickle(CACHE_DIR / "EXTENDED_CORPUS-nq-test.p")

n_split = 10

splits = np.array_split(list(corpus.items()), n_split)

print("Getting corpus embeddings")
emb_sub_dfs = []
for split in tqdm.tqdm(splits):
    all_texts = [f"{t[1]['title']}: {t[1]['text']}" for t in split]
    embs = [{"gpt_embedding": e} for e in get_embeddings(all_texts)]
    df = pd.DataFrame(embs)
    df.index = [t[0] for t in split]
    emb_sub_dfs.append(df)

emb_df = pd.concat(emb_sub_dfs)


emb_df.to_pickle(CACHE_DIR / "GPT_EMBEDDING_DUMP-nq-test.p")

questions = test_case_df["question"].to_list()

embs = get_embeddings(questions)

qemb_df = pd.DataFrame([{"gpt_embedding": e} for e in embs])
qemb_df.index = test_case_df["qid"]

qemb_df.to_pickle(CACHE_DIR / "GPT_QEMBEDDING_DUMP-nq-test.p")
