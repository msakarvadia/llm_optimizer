"""General purpose utility functions for experiments."""

from __future__ import annotations

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer


def semantic_similarity(strings: list[str]) -> dict[str, float]:
    """Report group statistics about semantic similarity.

    return: dict of mean/std_dev/min/max

    usage:

    list_of_strs = ["I am a cat", "I am a dog", "I am a rhino"]
    semantic_similarity(list_of_strs)
    """
    # Handle edge case for fewer than 2
    if len(strings) < 2:  # noqa
        return {'mean': 0.0, 'std_dev': 0.0, 'min': 0.0, 'max': 0.0}

    # embed strs
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(strings, convert_to_numpy=True)

    # normalize embeddings
    # (Note: sentence-transformers models typically output normalized vectors,
    # but explicit normalization ensures safety for any model output)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / np.where(norms == 0, 1, norms)

    # compute pairwise semantic sim and exclude diagonal
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

    # Create a boolean mask to filter out the diagonal (self-similarity)
    mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
    pairwise_similarities = similarity_matrix[mask]

    # compute group stats: mean/std dev/min/max
    semantic_sim_stats: dict[str, float] = {
        'mean': float(np.mean(pairwise_similarities)),
        'std_dev': float(np.std(pairwise_similarities)),
        'min': float(np.min(pairwise_similarities)),
        'max': float(np.max(pairwise_similarities)),
    }

    return semantic_sim_stats


def prompt_lm(client: OpenAI, prompt: str) -> str:
    """Standard LLM api inference call."""
    response = client.chat.completions.create(
        model='gemini-3.5-flash',
        messages=[
            {
                'role': 'system',
                'content': 'You are a helpful assistant.',
            },
            {
                'role': 'user',
                'content': prompt,
            },
        ],
    )
    raw_output = response.choices[0].message.content
    if raw_output is not None:
        return raw_output
    else:
        raise ValueError(
            'LM api call returned None instead of a valid string.',
        )
