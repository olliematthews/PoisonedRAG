"""Generate a dataset of questions, and related contexts. This is the retrieval stage of the RAG pipeline."""

import pandas as pd
from pathlib import Path
import numpy as np
import pickle
import json
import argparse
from typing import Optional

CACHE_DIR = Path("./.cache")
EXPERIMENT_DIR = Path("./results/experiments")


def parse_args():
    parser = argparse.ArgumentParser()

    # Retriever and BEIR datasets
    parser.add_argument("--experiment_name", type=str, required=True)

    args = parser.parse_args()
    print(args)

    return args


def get_context_openai_factory(experiment_config):
    """Get context with openai embeddings."""
    try:
        corpus_embedding_df = pd.read_pickle(
            CACHE_DIR
            / f"openai_corpus_embedding_cache-{experiment_config['dataset']}.p"
        )
        question_embedding_df = pd.read_pickle(
            CACHE_DIR
            / f"openai_question_embedding_cache-{experiment_config['dataset']}.p"
        )
        extended_corpus_df = pd.read_pickle(
            CACHE_DIR / f"extended_corpus-{experiment_config['dataset']}.p"
        )
    except FileNotFoundError as e:
        raise Exception(
            "Not able to find cache file. Have you run 'generate_openai_embeddings.py'?"
        ) from e

    all_context_embeddings = np.array(corpus_embedding_df["gpt_embedding"].to_list())
    all_context_embeddings_norm = (
        all_context_embeddings / np.linalg.norm(all_context_embeddings, axis=1)[:, None]
    )

    def get_context(
        row: pd.Series,
        n_contexts: Optional[int],
        accept_thresh: float,
        similarity_rej_thresh: Optional[float] = None,
        n_poison: Optional[int] = None,
    ):
        if n_contexts is None:
            return None
        question_embedding = np.array(
            question_embedding_df.loc[row.name]["gpt_embedding"]
        )
        question_embedding_norm = question_embedding / np.linalg.norm(
            question_embedding
        )

        cos_sim = np.dot(all_context_embeddings_norm, question_embedding_norm)

        relevant = corpus_embedding_df[cos_sim > accept_thresh]
        relevant["relevance"] = cos_sim[cos_sim > accept_thresh]

        relevant = relevant.sort_values(by="relevance", ascending=False)

        contexts = []
        embeddings = []
        n_attacks = 0
        for qid, c in relevant.iterrows():
            if n_poison is not None and qid.startswith("attack_"):
                if n_attacks >= n_poison:
                    continue
                n_poison += 1
            if similarity_rej_thresh is not None:
                emb = np.array(c["gpt_embedding"])
                emb_rel = emb - question_embedding
                emb_rel_norm = emb_rel / np.linalg.norm(emb_rel)
                if not any(
                    [
                        np.dot(emb_rel_norm, e) > similarity_rej_thresh
                        for e in embeddings
                    ]
                ):
                    contexts.append(qid)
                    embeddings.append(emb_rel_norm)
            else:
                contexts.append(qid)
            if len(contexts) >= n_contexts:
                break
        return extended_corpus_df.loc[contexts]["text"].to_list()

    return get_context


def get_context_contriever_factory(experiment_config):
    """Get context with contriever embeddings."""
    try:
        with open(
            CACHE_DIR / f"poisonedrag_cache-{experiment_config['dataset']}.p", "rb"
        ) as fd:
            corpus, _ = pickle.load(fd)
    except FileNotFoundError as e:
        raise Exception(
            "Not able to find cache file. Have you run 'generate_datasets.py'?"
        ) from e

    def get_context(
        row: pd.Series,
        n_contexts: Optional[int],
        n_poison: Optional[int] = None,
    ):
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

        if n_poison is not None:
            contexts += row["attack_contexts"][:n_poison]
        else:
            contexts += row["attack_contexts"]
        contexts = sorted(contexts, key=lambda x: x["score"], reverse=True)[:n_contexts]

        return [c["context"] for c in contexts]

    return get_context


def get_context_gt_factory(experiment_config):
    """Get context with contriever embeddings."""
    try:
        corpus_embedding_df = pd.read_pickle(
            CACHE_DIR
            / f"openai_corpus_embedding_cache-{experiment_config['dataset']}.p"
        )
        question_embedding_df = pd.read_pickle(
            CACHE_DIR
            / f"openai_question_embedding_cache-{experiment_config['dataset']}.p"
        )
        extended_corpus_df = pd.read_pickle(
            CACHE_DIR / f"extended_corpus-{experiment_config['dataset']}.p"
        )
    except FileNotFoundError as e:
        raise Exception(
            "Not able to find cache file. Have you run 'generate_openai_embeddings.py'?"
        ) from e

    attack_corpus_embedding_df = corpus_embedding_df[
        corpus_embedding_df.apply(lambda row: row.name.startswith("attack_"), axis=1)
    ]
    all_attack_context_embeddings = np.array(
        attack_corpus_embedding_df["gpt_embedding"].to_list()
    )
    all_attack_context_embeddings_norm = (
        all_attack_context_embeddings
        / np.linalg.norm(all_attack_context_embeddings, axis=1)[:, None]
    )

    def get_context(
        row: pd.Series,
        n_poison: Optional[int],
        similarity_rej_thresh: Optional[float] = None,
    ):
        question_embedding = np.array(
            question_embedding_df.loc[row.name]["gpt_embedding"]
        )
        question_embedding_norm = question_embedding / np.linalg.norm(
            question_embedding
        )

        cos_sim = np.dot(all_attack_context_embeddings_norm, question_embedding_norm)

        cos_sim_series = pd.Series(cos_sim, index=attack_corpus_embedding_df.index)
        cos_sim_series = cos_sim_series.sort_values(ascending=False)[:n_poison]

        if similarity_rej_thresh is not None:
            attack_contexts = []
            embeddings = []
            for cid in cos_sim_series.index:
                emb = attack_corpus_embedding_df.loc[cid]["gpt_embedding"]
                emb_rel = emb - question_embedding
                emb_rel_norm = emb_rel / np.linalg.norm(emb_rel)
                if not any(
                    [
                        np.dot(emb_rel_norm, e) > similarity_rej_thresh
                        for e in embeddings
                    ]
                ):
                    attack_contexts.append(cid)
                    embeddings.append(emb_rel_norm)
        else:
            attack_contexts = cos_sim_series.index.to_list()

        return extended_corpus_df.loc[row["gt_ids"] + attack_contexts]["text"].to_list()

    return get_context


def main():
    args = parse_args()

    results_dir = EXPERIMENT_DIR / args.experiment_name

    try:
        with open(results_dir / "config.json", "r") as fd:
            experiment_config = json.load(fd)
    except FileNotFoundError as e:
        raise Exception(
            f"Unable to find config for experiment {args.experiment_name}. Have you run 'initialise_experiment_set.py' for that experiment?"
        ) from e

    try:
        with open(
            CACHE_DIR / f"poisonedrag_cache-{experiment_config['dataset']}.p", "rb"
        ) as fd:
            test_cases, _ = pickle.load(fd)
    except FileNotFoundError as e:
        raise Exception(
            "Not able to find cache file. Have you run 'generate_datasets.py'?"
        ) from e

    if (n_questions := experiment_config.get("n_questions")) is not None:
        test_cases = test_cases[:n_questions]

    questions_df = pd.DataFrame(test_cases)
    questions_df.set_index("qid", inplace=True)

    questions_df.to_pickle(results_dir / "questions.p")

    def get_context_getter(retriever_type):
        match retriever_type:
            case "openai":
                return get_context_openai_factory(experiment_config)
            case "contriever":
                return get_context_contriever_factory(experiment_config)
            case "gt":
                return get_context_gt_factory(experiment_config)

    context_getters = {
        retriever_type: get_context_getter(retriever_type)
        for retriever_type in set(
            [config[0] for config in experiment_config["retriever_configs"].values()]
        )
    }

    print(
        f"Generating context dataframe for contexts: {list(experiment_config['retriever_configs'].keys())}"
    )
    context_df = pd.concat(
        {
            col: questions_df.apply(
                context_getters[retriever_type], axis=1, **retriever_args
            )
            for col, (retriever_type, retriever_args) in experiment_config[
                "retriever_configs"
            ].items()
        },
        axis=1,
    )

    context_df.to_pickle(results_dir / "context.p")


if __name__ == "__main__":
    main()
