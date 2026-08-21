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
    dedup: str = 'max',
) -> tuple[list[dict[str, Any]], int]:
    """Pool (solution, score) pairs within budget, plus token cost.

    A solution's score can vary across occurrences (e.g. a task whose
    target-generation step isn't temperature-pinned, so the same
    candidate text gets independently re-sampled and re-judged each time
    it's evaluated). `dedup='max'` (default) keeps the HIGHEST observed
    score for a repeated solution; `dedup='min'` keeps the LOWEST --
    provided as a compare-and-contrast lever against the 'max' default,
    not because 'min' is a recommended policy. `dedup='none'` skips
    deduplication entirely: every occurrence becomes its own candidate,
    so a solution repeated N times across the pooled banks contributes N
    (possibly identical) entries -- a worst-case baseline showing what
    happens if a duplicated high scorer is left free to crowd out the
    rest of a curated population, not a recommended policy either.
    """
    if dedup not in ('max', 'min', 'none'):
        raise ValueError(
            f"dedup must be 'max', 'min', or 'none', got {dedup!r}",
        )

    total_cost_tokens = 0

    if dedup == 'none':
        candidates: list[dict[str, Any]] = []
        for bank in solution_banks:
            in_budget_keys = [k for k in bank if int(k) <= iteration_budget]
            if in_budget_keys:
                last_key = max(in_budget_keys, key=int)
                total_cost_tokens += (
                    bank[last_key].get('cumulative_tokens_spent', 0) or 0
                )
            for iter_key in in_budget_keys:
                entry = bank[iter_key]
                candidates.append(
                    {'solution': entry['solution'], 'score': entry['score']},
                )
        return candidates, total_cost_tokens

    better = (lambda a, b: a > b) if dedup == 'max' else (lambda a, b: a < b)
    best_by_solution: dict[str, float] = {}
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
            if solution not in best_by_solution or better(
                entry['score'],
                best_by_solution[solution],
            ):
                best_by_solution[solution] = entry['score']
    candidates = [
        {'solution': solution, 'score': score}
        for solution, score in best_by_solution.items()
    ]
    return candidates, total_cost_tokens


def _tournament_sample(
    candidates: list[dict[str, Any]],
    pop_size: int,
    p: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Tournament rounds; K = p * active pool size.

    Only the winner is removed each round
    -- losers stay eligible for later rounds.
    """
    rng = random.Random(seed)
    active = list(candidates)
    population: list[dict[str, Any]] = []
    while active and len(population) < pop_size:
        k_eff = min(max(1, round(p * len(active))), len(active))
        contestants = rng.sample(active, k_eff)
        winner = max(contestants, key=lambda item: item['score'])
        population.append(winner)
        # Identity-based removal (exact regardless of score collisions);
        # only the winner leaves `active`, losers can be resampled later.
        active = [item for item in active if id(item) != id(winner)]
    return population


def _greedy_sample(
    candidates: list[dict[str, Any]],
    pop_size: int,
) -> list[dict[str, Any]]:
    """Top `pop_size` candidates by score, no diversity filtering at all."""
    ordered = sorted(candidates, key=lambda item: item['score'], reverse=True)
    return ordered[:pop_size]


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
    dedup: str = 'max',
    population_size: int | None = None,
) -> list[dict[str, Any]]:
    """Curate a population via tournament, diversity-rejection, or greedy.

    (top-N by score, no diversity filtering, `p` unused) sampling. `dedup`
    is passed straight to `_pool_candidates_within_budget` -- see its
    docstring for the 'max' (default) vs. 'min' distinction.

    `population_size`, when given, overrides the adaptive
    `max_population_size` scheme with an exact target instead (still
    capped at however many candidates are actually available -- this
    can't manufacture candidates that don't exist). Leave it `None` (the
    default) to keep the adaptive `min(max_population_size,
    total_candidates)` behavior.
    """
    if curation_type not in ('tournament', 'diversity', 'greedy'):
        raise ValueError(
            f"curation_type must be 'tournament', 'diversity', or 'greedy', "
            f'got {curation_type!r}',
        )

    candidates, cost_tokens = _pool_candidates_within_budget(
        solution_banks,
        iteration_budget,
        dedup=dedup,
    )
    total_candidates = len(candidates)
    if total_candidates == 0:
        return []

    # Dynamic sizing: an explicit population_size overrides the adaptive
    # max_population_size scheme; either way, cap at however many
    # candidates actually exist.
    target_size = (
        population_size if population_size is not None else max_population_size
    )
    curated_size = min(target_size, total_candidates)

    if curation_type == 'tournament':
        population = _tournament_sample(
            candidates,
            pop_size=curated_size,
            p=p,
            seed=42,
        )
    elif curation_type == 'greedy':
        population = _greedy_sample(candidates, pop_size=curated_size)
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
                # nomic-ai/CodeRankEmbed (the code-task embed_model, see
                # generate_experiment_args.py's get_embed_model) ships
                # custom modeling code and needs this to load; harmless
                # no-op for all-MiniLM-L6-v2's stock architecture. Matches
                # opro.py's own SentenceTransformer call.
                trust_remote_code=True,
            )
            embeddings = model.encode(solutions, batch_size=8)

            sims = cosine_similarity(embeddings)
            triu_idx = np.triu_indices_from(sims, k=1)
            # np.percentile wants 0-100, but p is a 0-1 fraction (shared
            # convention with the tournament branch's p) -- rescale.
            threshold = float(np.percentile(sims[triu_idx], p * 100))

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
