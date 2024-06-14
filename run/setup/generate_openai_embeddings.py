"""Generate the GPT embeddings for all of the corpus provided. These are cached for use in other files."""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm

main_dir_path = str(Path(__file__).parent.parent.parent)
if main_dir_path not in sys.path:
    sys.path.append(main_dir_path)

from poisoned_rag_defense.models.embedding import get_embeddings


def parse_args():
    parser = argparse.ArgumentParser()

    # Retriever and BEIR datasets
    parser.add_argument(
        "--eval_dataset", type=str, default="nq", help="BEIR dataset to evaluate"
    )
    parser.add_argument("--split", type=str, default="test")

    args = parser.parse_args()
    print(args)
    return args


def main():
    args = parse_args()

    CACHE_DIR = Path("./.cache")

    dataset_split_string = f"{args.eval_dataset}-{args.split}"
    with open(CACHE_DIR / f"poisonedrag_cache-{dataset_split_string}.p", "rb") as fd:
        test_cases, corpus = pickle.load(fd)

    index = 0

    def to_context(row):
        nonlocal index
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
    corpus_df.to_pickle(CACHE_DIR / f"extended_corpus-{dataset_split_string}.p")

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

    emb_df.to_pickle(
        CACHE_DIR / f"openai_corpus_embedding_cache-{dataset_split_string}.p"
    )

    questions = test_case_df["question"].to_list()

    embs = get_embeddings(questions)

    qemb_df = pd.DataFrame([{"gpt_embedding": e} for e in embs])
    qemb_df.index = test_case_df["qid"]

    qemb_df.to_pickle(
        CACHE_DIR / f"openai_question_embedding_cache-{dataset_split_string}.p"
    )


if __name__ == "__main__":
    main()
