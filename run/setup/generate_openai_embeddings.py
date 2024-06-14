"""Generate the GPT embeddings for all of the corpus provided. These are cached for use in other files."""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm

from poisoned_rag.models.embedding import get_openai_embeddings
from poisoned_rag_defence.logger import logger


def parse_args():
    parser = argparse.ArgumentParser()

    # BEIR dataset
    parser.add_argument(
        "--eval_dataset", type=str, default="nq", help="BEIR dataset to evaluate"
    )
    parser.add_argument(
        "--split", type=str, default="test", help="The split on the dataset to use"
    )

    args = parser.parse_args()
    if args.eval_dataset == "msmarco":
        assert args.split == "train", "PoisonedRAG requires train split on msmarco"
    logger.debug(str(args))
    return args


def main():
    args = parse_args()

    CACHE_DIR = Path("./.cache")

    dataset_split_string = f"{args.eval_dataset}-{args.split}"
    try:
        with open(
            CACHE_DIR / f"poisonedrag_cache-{dataset_split_string}.p", "rb"
        ) as fd:
            test_cases, corpus = pickle.load(fd)
    except FileNotFoundError as e:
        raise Exception(
            "Not able to find cache file. Have you run 'generate_poisonedrag_cache.py'?"
        ) from e

    index = 0

    def attacks_to_corpus(row):
        """Returns a corpus entry for each attack in the test case."""
        nonlocal index
        ret = {}
        for attack in row["attack_contexts"]:
            ret[f"attack_{index}"] = {
                "title": row["question"],
                "text": attack["context"][len(row["question"]) + 1 :],
            }
            index += 1

        return ret

    # Build up an extended corpus which has the attacks added in with the relevant corpus items
    test_case_df = pd.DataFrame(test_cases)
    for item in test_case_df.apply(attacks_to_corpus, axis=1).to_list():
        corpus.update(item)

    corpus_df = pd.DataFrame(corpus.values())
    corpus_df.index = corpus.keys()
    corpus_df.to_pickle(CACHE_DIR / f"extended_corpus-{dataset_split_string}.p")

    # Get the embeddings for the extended corpus
    # Split it up due to openai limits on number of concurrent items
    n_split = 10
    splits = np.array_split(list(corpus.items()), n_split)

    logger.info("Getting corpus embeddings")
    emb_sub_dfs = []
    for split in tqdm.tqdm(splits):
        all_texts = [f"{t[1]['title']}: {t[1]['text']}" for t in split]
        embeddings = [{"gpt_embedding": e} for e in get_openai_embeddings(all_texts)]
        df = pd.DataFrame(embeddings)
        df.index = [t[0] for t in split]
        emb_sub_dfs.append(df)

    corpus_embedding_df = pd.concat(emb_sub_dfs)

    save_file = CACHE_DIR / f"openai_corpus_embedding_cache-{dataset_split_string}.p"
    corpus_embedding_df.to_pickle(save_file)
    logger.info(f"Corpus embeddings saved to {save_file}")

    logger.info("Getting question embeddings")
    questions = test_case_df["question"].to_list()
    embeddings = get_openai_embeddings(questions)

    # Save the question embeddings into a df
    question_embedding_df = pd.DataFrame([{"gpt_embedding": e} for e in embeddings])
    question_embedding_df.index = test_case_df["qid"]

    save_file = CACHE_DIR / f"openai_question_embedding_cache-{dataset_split_string}.p"
    question_embedding_df.to_pickle(save_file)
    logger.info(f"Corpus embeddings saved to {save_file}")


if __name__ == "__main__":
    main()
