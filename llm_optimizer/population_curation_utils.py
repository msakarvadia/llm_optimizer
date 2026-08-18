"""Curate an initial population from pooled solution banks."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def _pool_candidates_within_budget(
    solution_banks: list[dict[str, Any]],
    iteration_budget: int,
) -> tuple[list[dict[str, Any]], int]:
    """Pool deduped (solution, score) pairs within budget, plus token cost."""
    seen_solutions: set[str] = set()
    candidates: list[dict[str, Any]] = []
    total_cost_tokens = 0
    for bank in solution_banks:
        in_budget_keys = [k for k in bank if int(k) <= iteration_budget]
        if in_budget_keys:
            last_key = max(in_budget_keys, key=int)
            total_cost_tokens += (
                bank[last_key].get('cumulative_tokens_spent', 0) or 0
            )
        for iter_key in in_budget_keys:
            entry = bank[iter_key]
            solution = entry['solution']
            if solution in seen_solutions:
                continue
            seen_solutions.add(solution)
            candidates.append({'solution': solution, 'score': entry['score']})
    return candidates, total_cost_tokens


def _tournament_sample(
    candidates: list[dict[str, Any]],
    pop_size: int,
    p: float,
    seed: int,
) -> list[dict[str, Any]]:
    """No-resampling tournament rounds; K = p * active pool size."""
    rng = random.Random(seed)
    active = list(candidates)
    population: list[dict[str, Any]] = []
    while active and len(population) < pop_size:
        k_eff = min(max(1, round(p * len(active))), len(active))
        contestants = rng.sample(active, k_eff)
        winner = max(contestants, key=lambda item: item['score'])
        population.append(winner)
        # Identity-based removal (not list.remove, which matches by == and
        # could drop the wrong record if two distinct candidates happen to
        # share a score) -- exact regardless of score collisions.
        contestant_ids = {id(item) for item in contestants}
        active = [item for item in active if id(item) not in contestant_ids]
    return population


def _diversity_rejection_sample(
    candidates: list[dict[str, Any]],
    embeddings: np.ndarray[Any, Any],
    pop_size: int,
    threshold: float,
) -> list[dict[str, Any]]:
    """Accept diverse candidates by score; backfill if too few pass."""
    order = sorted(
        range(len(candidates)),
        key=lambda i: candidates[i]['score'],
        reverse=True,
    )
    population: list[dict[str, Any]] = []
    pop_embeddings: list[np.ndarray[Any, Any]] = []
    accepted_idx: set[int] = set()
    for i in order:
        if len(population) >= pop_size:
            break
        candidate, embedding = candidates[i], embeddings[i]
        if not population:
            population.append(candidate)
            pop_embeddings.append(embedding)
            accepted_idx.add(i)
            continue
        sims = cosine_similarity([embedding], pop_embeddings)[0]
        if sims.max() <= threshold:
            population.append(candidate)
            pop_embeddings.append(embedding)
            accepted_idx.add(i)

    # Greedy backfill: not enough sufficiently-diverse candidates were
    # found -- fill remaining slots with the next-best scorers
    if len(population) < pop_size:
        for i in order:
            if len(population) >= pop_size:
                break
            if i in accepted_idx:
                continue
            population.append(candidates[i])

    return population


def curate_population(  # noqa: PLR0913
    solution_banks: list[dict[str, Any]],
    iteration_budget: int,
    curation_type: str,
    max_population_size: int = 15,
    p: float = 0.9,
    embed_model_name: str = 'all-MiniLM-L6-v2',
) -> list[dict[str, Any]]:
    """Curate a population via tournament or diversity-rejection sampling."""
    if curation_type not in ('tournament', 'diversity'):
        raise ValueError(
            f"curation_type must be 'tournament' or 'diversity', "
            f'got {curation_type!r}',
        )

    candidates, cost_tokens = _pool_candidates_within_budget(
        solution_banks,
        iteration_budget,
    )
    total_candidates = len(candidates)
    if total_candidates == 0:
        return []

    # Dynamic sizing: cap at max_population_size, otherwise use every
    # available candidate.
    curated_size = min(max_population_size, total_candidates)

    if curation_type == 'tournament':
        population = _tournament_sample(
            candidates,
            pop_size=curated_size,
            p=p,
            seed=42,
        )
    else:
        # curation_type == 'diversity'
        solutions = [c['solution'] for c in candidates]
        if len(solutions) < 2:  # noqa: PLR2004
            # No pairwise similarities to derive a threshold from
            population = candidates[:curated_size]
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model_kwargs = (
                {'torch_dtype': torch.float16} if device == 'cuda' else {}
            )
            model = SentenceTransformer(
                embed_model_name,
                device=device,
                model_kwargs=model_kwargs,
            )
            embeddings = model.encode(solutions, batch_size=8)

            sims = cosine_similarity(embeddings)
            triu_idx = np.triu_indices_from(sims, k=1)
            threshold = float(np.percentile(sims[triu_idx], p))

            population = _diversity_rejection_sample(
                candidates,
                embeddings,
                pop_size=curated_size,
                threshold=threshold,
            )

    # Stamp the pooled token cost onto the last (highest-index) entry only,
    # matching save_population_json's / SolutionBank.read_from_checkpoint's
    # convention, so a downstream OPRO run seeded from this population
    if population:
        population[-1]['cumulative_tokens_spent'] = cost_tokens

    return population
