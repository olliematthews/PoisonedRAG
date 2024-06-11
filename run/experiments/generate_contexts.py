"""Generate a dataset of questions, and related contexts. This is the retrieval stage of the RAG pipeline."""

import pandas as pd
from pathlib import Path
import numpy as np
import pickle
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # Retriever and BEIR datasets
    parser.add_argument("--experiment_name", type=str, required=True)

    args = parser.parse_args()
    print(args)

    return args


def main():
    args = parse_args()

    CACHE_DIR = Path("./.cache")
    EXPERIMENT_DIR = Path("./results/experiments")

    results_dir = EXPERIMENT_DIR / args.experiment_name

    try:
        with open(results_dir / "config.json", "r") as fd:
            experiment_config = json.load(fd)
    except FileNotFoundError as e:
        raise Exception(
            f"Unable to find config for experiment {args.experiment_name}. Have you run 'initialise_experiment.py' for that experiment?"
        ) from e

    try:
        emb_df = pd.read_pickle(
            CACHE_DIR / f"gpt_corpus_embedding_cache-{experiment_config['dataset']}.p"
        )
        qemb_df = pd.read_pickle(
            CACHE_DIR
            / f"openai_question_embedding_cache-{experiment_config['dataset']}.p"
        )
        extended_corpus_df = pd.read_pickle(
            CACHE_DIR / f"extended_corpus-{experiment_config['dataset']}.p"
        )
        with open(
            CACHE_DIR / f"poisonedrag_cache-{experiment_config['dataset']}.p", "rb"
        ) as fd:
            test_cases, _ = pickle.load(fd)
    except FileNotFoundError as e:
        raise Exception(
            "Not able to find cache file. Have you run both 'generate_datasets.py' and 'generate_openai_embeddings.py'?"
        ) from e

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
                    [
                        np.dot(emb_rel_norm, e) > similarity_rej_thresh
                        for e in embeddings
                    ]
                ):
                    contexts.append(qid)
                    embeddings.append(emb_rel_norm)

                if len(contexts) >= n:
                    break
        else:
            contexts = relevant.index[:n]
        return extended_corpus_df.loc[contexts]["text"].to_list()

    print(
        f"Generating context dataframe for contexts: {list(experiment_config['context_configs'].keys())}"
    )
    context_df = pd.concat(
        {
            col: questions_df.apply(get_context, args=args_, axis=1)
            for col, args_ in experiment_config["context_configs"].items()
        },
        axis=1,
    )

    context_df.to_pickle(results_dir / "context.p")


if __name__ == "__main__":
    main()
