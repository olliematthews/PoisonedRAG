"""Generate a dataset of questions, and related contexts. This is the retrieval stage of the RAG pipeline."""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

main_dir_path = str(Path(__file__).parent.parent.parent)
if main_dir_path not in sys.path:
    sys.path.append(main_dir_path)

from poisoned_rag_defense.logger import logger
from run.experiments.experiment import Experiment
from run.experiments.utils import experiment_name_parse_args


def get_context_openai_factory(experiment: Experiment):
    """Get context with openai embeddings."""
    all_context_embeddings = np.array(
        experiment.corpus_embedding_df["gpt_embedding"].to_list()
    )
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
            experiment.question_embedding_df.loc[row.name]["gpt_embedding"]
        )
        question_embedding_norm = question_embedding / np.linalg.norm(
            question_embedding
        )

        # Get cosine similarity for the question to the corpus items
        cos_sim = np.dot(all_context_embeddings_norm, question_embedding_norm)

        # Get relevant contexts
        relevant = experiment.corpus_embedding_df[cos_sim > accept_thresh]
        relevant["relevance"] = cos_sim[cos_sim > accept_thresh]
        relevant = relevant.sort_values(by="relevance", ascending=False)

        contexts = []
        embeddings = []
        n_attacks = 0
        for qid, c in relevant.iterrows():
            # Check if we have reached the limit of poisoned items to add in
            if n_poison is not None and qid.startswith("attack_"):
                if n_attacks >= n_poison:
                    continue
                n_poison += 1

            if similarity_rej_thresh is not None:
                # Do context variance encouragement
                emb = np.array(c["gpt_embedding"])
                # Get the context embedding relative to the question
                emb_rel = emb - question_embedding
                emb_rel_norm = emb_rel / np.linalg.norm(emb_rel)
                # Check the relative embedding is not too close to existing contexts
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
        return experiment.extended_corpus_df.loc[contexts]["text"].to_list()

    return get_context


def get_context_contriever_factory(experiment):
    """Get context with contriever embeddings."""

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
            {"context": experiment.corpus[key]["text"], "score": score}
            for key, score in ds_contexts[:n_contexts]
        ]

        if n_poison is not None:
            contexts += row["attack_contexts"][:n_poison]
        else:
            contexts += row["attack_contexts"]
        contexts = sorted(contexts, key=lambda x: x["score"], reverse=True)[:n_contexts]

        return [c["context"] for c in contexts]

    return get_context


def get_context_gt_factory(experiment):
    """Get context with gt.

    Behaves slightly differently to the others - this will add n_poison poisoned items to the gt context.
    """
    attack_corpus_embedding_df = experiment.corpus_embedding_df[
        experiment.corpus_embedding_df.apply(
            lambda row: row.name.startswith("attack_"), axis=1
        )
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
            experiment.question_embedding_df.loc[row.name]["gpt_embedding"]
        )
        question_embedding_norm = question_embedding / np.linalg.norm(
            question_embedding
        )

        cos_sim = np.dot(all_attack_context_embeddings_norm, question_embedding_norm)

        cos_sim_series = pd.Series(cos_sim, index=attack_corpus_embedding_df.index)
        cos_sim_series = cos_sim_series.sort_values(ascending=False)[:n_poison]

        if similarity_rej_thresh is not None:
            # Do context variance encouragement
            attack_contexts = []
            embeddings = []
            for cid in cos_sim_series.index:
                emb = attack_corpus_embedding_df.loc[cid]["gpt_embedding"]
                # Get the context embedding relative to the question
                emb_rel = emb - question_embedding
                emb_rel_norm = emb_rel / np.linalg.norm(emb_rel)
                # Check the relative embedding is not too close to existing contexts
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

        return experiment.extended_corpus_df.loc[row["gt_ids"] + attack_contexts][
            "text"
        ].to_list()

    return get_context


def main():
    args = experiment_name_parse_args()

    experiment = Experiment(args.experiment_name)

    if (n_questions := experiment.config.get("n_questions")) is not None:
        experiment.test_cases = experiment.test_cases[:n_questions]

    question_df = pd.DataFrame(experiment.test_cases)
    question_df.set_index("qid", inplace=True)

    experiment.save_df(question_df, "questions")

    def get_context_getter(retriever_type):
        match retriever_type:
            case "openai":
                return get_context_openai_factory(experiment)
            case "contriever":
                return get_context_contriever_factory(experiment)
            case "gt":
                return get_context_gt_factory(experiment)

    # Generate the get_context functions lazily, so that you don't need e.g. openai embeddings for contriever contexts
    context_getters = {
        retriever_type: get_context_getter(retriever_type)
        for retriever_type in set(
            [config[0] for config in experiment.config["retriever_configs"].values()]
        )
    }

    logger.info(
        f"Generating context dataframe for contexts: {list(experiment.config['retriever_configs'].keys())}"
    )
    context_df = pd.concat(
        {
            col: question_df.apply(
                context_getters[retriever_type], axis=1, **retriever_args
            )
            for col, (retriever_type, retriever_args) in experiment.config[
                "retriever_configs"
            ].items()
        },
        axis=1,
    )

    experiment.save_df(context_df, "context")


if __name__ == "__main__":
    main()
