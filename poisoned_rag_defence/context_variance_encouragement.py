import numpy as np
import pandas as pd


def context_variance_encouragement(
    embeddings: pd.Series,
    question_embedding: np.array,
    similarity_rej_thresh: float,
    n_contexts: int,
) -> list[str]:
    """Do context variance encouragement to encourage variance in the context.

    Takes embeddings sorted by proximity to the question.
    Runs through the embeddings one by one:
    * Calculates the embedding relative to the question
    * Adds the context to the context as long as the cosine similarity between
      the relative embedding and any of the relative embeddings of the existing
      context items are not too close.

    Args:
        embeddings (pd.Series): a series of openai embeddings for context
          items, where the index is the context item id
        question_embedding (np.array): the openai embedding for the question
        similarity_rej_thresh (float): the similarity threshold, if an embedding
          has cosine similarity > similarity_rej_thresh to the items in the
          context, it will not be added
        n_contexts (int): the limit of the number of items to be included in
          the context

    Returns:
        list[str]: the contexts
    """
    # Do context variance encouragement
    contexts = []
    context_rel_emb_norms = []
    for cid, emb in zip(embeddings.index, embeddings):
        # Get the context embedding relative to the question
        emb_rel = np.array(emb) - question_embedding
        emb_rel_norm = emb_rel / np.linalg.norm(emb_rel)
        # Check the relative embedding is not too close to existing contexts
        if not any(
            [
                np.dot(emb_rel_norm, e) > similarity_rej_thresh
                for e in context_rel_emb_norms
            ]
        ):
            contexts.append(cid)
            context_rel_emb_norms.append(emb_rel_norm)
        if len(contexts) >= n_contexts:
            break
    return contexts
