"""General purpose utility functions for experiments."""

from __future__ import annotations


def semantic_similarity(strings: list[str]) -> dict[str, float]:
    """Report group statistics about semantic similarity.

    return: dict of mean/std_dev/min/max
    """
    # embed strs
    # normalize embeddings
    # compute pairwise semantic sim and exclude diagonal
    # compute group stats: mean/std dev/min/max
    semantic_sim_stats: dict[str, float] = {}

    return semantic_sim_stats
