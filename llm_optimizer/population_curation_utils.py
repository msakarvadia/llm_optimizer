"""Curate an initial population from pooled solution banks."""

from __future__ import annotations

import glob
import json
import os
import random
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# The two embed models this codebase ever passes as embed_model_name (see
# generate_experiment_args.py's get_embed_model) -- cached globally and
# lazily so repeated curate_population calls don't reload from disk/GPU.
_MINILM_MODEL: SentenceTransformer | None = None
_CODERANK_MODEL: SentenceTransformer | None = None


def _load_embed_model(embed_model_name: str) -> SentenceTransformer:
    """Lazily load + cache the given embed model (one of the two above)."""
    global _MINILM_MODEL, _CODERANK_MODEL  # noqa: PLW0603
    is_minilm = embed_model_name == 'all-MiniLM-L6-v2'
    cached = _MINILM_MODEL if is_minilm else _CODERANK_MODEL
    if cached is not None:
        return cached
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_kwargs = {'torch_dtype': torch.float16} if device == 'cuda' else {}
    model = SentenceTransformer(
        embed_model_name,
        device=device,
        model_kwargs=model_kwargs,
        # nomic-ai/CodeRankEmbed ships custom modeling code and needs this
        # to load; harmless no-op for all-MiniLM-L6-v2's stock architecture.
        trust_remote_code=True,
    )
    if is_minilm:
        _MINILM_MODEL = model
    else:
        _CODERANK_MODEL = model
    return model


def load_solution_banks_by_glob(
    glob_pattern: str,
    bank_filename: str = 'long_running_solution_bank.json',
    args_filename: str = 'experiment_args.json',
) -> list[dict[str, Any]]:
    """Glob for experiment dirs and load their solution banks, in seed order.

    Orders by the `seed` field in each dir's `args_filename` (resolved run
    args dumped by main.py) rather than glob/readdir order (arbitrary) or
    mtime (unreliable under parallel scheduling, no true creation time on
    Linux). Returns one bank dict per matched dir, ascending seed, unmerged
    -- pass to `merge_solution_banks_in_order` before `curate_population`.

    Dirs missing `args_filename`/`bank_filename` (a run that never
    actually executed) are skipped with a printed warning rather than
    raising.
    """
    # Run dirs nest both files one level down, under a literal
    # `placeholder_path/` subdir, not flat at the run dir's root.
    all_dirs = [
        os.path.join(d, 'placeholder_path')
        for d in glob.glob(glob_pattern)
        if os.path.isdir(d)
    ]

    def seed_of(directory: str) -> int:
        with open(
            os.path.join(directory, args_filename),
            encoding='utf-8',
        ) as f:
            return int(json.load(f)['seed'])

    dirs = []
    for directory in all_dirs:
        if not os.path.exists(os.path.join(directory, args_filename)):
            print(
                f'skipping incomplete run dir (no {args_filename}): '
                f'{directory}',
            )
            continue
        dirs.append(directory)
    dirs.sort(key=seed_of)

    banks = []
    for directory in dirs:
        bank_path = os.path.join(directory, bank_filename)
        if not os.path.exists(bank_path):
            print(
                f'skipping incomplete run dir (no {bank_filename}): '
                f'{directory}',
            )
            continue
        with open(bank_path, encoding='utf-8') as f:
            banks.append(json.load(f))
    return banks


def merge_solution_banks_in_order(
    banks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Flatten ordered banks into one bank; keys re-sequenced, tokens summed.

    Each bank (e.g. one seed's solution bank) has its own from-zero running
    `cumulative_tokens_spent`; entries are re-keyed into one increasing
    sequence in the given bank order while a running token offset (each
    bank's own last-entry cost) carries forward, so the merged bank's
    highest key still holds the true total-so-far, matching the convention
    `_pool_candidates_within_budget` relies on.
    """
    merged: dict[str, Any] = {}
    token_offset = 0
    next_key = 0
    for bank in banks:
        for _, entry in sorted(bank.items(), key=lambda kv: int(kv[0])):
            entry_copy = dict(entry)  # don't mutate caller's data
            entry_copy['cumulative_tokens_spent'] = token_offset + (
                entry_copy.get('cumulative_tokens_spent', 0) or 0
            )
            merged[str(next_key)] = entry_copy
            next_key += 1
        if bank:
            last_key = max(bank, key=int)
            token_offset += (
                bank[last_key].get('cumulative_tokens_spent', 0) or 0
            )
    return merged


def _pool_candidates_within_budget(  # noqa: C901
    solution_banks: list[dict[str, Any]],
    iteration_budget: int | None = None,
    dedup: str = 'max',
    token_budget: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Pool (solution, score) pairs within budget, plus token cost.

    `dedup` picks max/min/none handling of repeated solutions.
    Exactly one of `iteration_budget`/`token_budget` is required;
    `token_budget` wins if both are given, else `ValueError`.
    """
    if dedup not in ('max', 'min', 'none'):
        raise ValueError(
            f"dedup must be 'max', 'min', or 'none', got {dedup!r}",
        )
    if iteration_budget is None and token_budget is None:
        raise ValueError(
            'must supply iteration_budget or token_budget (or both -- '
            'token_budget takes priority when both are given)',
        )

    def _in_budget(bank: dict[str, Any], key: str) -> bool:
        if token_budget is not None:
            return (bank[key].get('cumulative_tokens_spent', 0) or 0) <= (
                token_budget
            )
        # Guaranteed non-None here: the check above requires at least one
        # of iteration_budget/token_budget, and token_budget is None.
        assert iteration_budget is not None
        return int(key) <= iteration_budget

    total_cost_tokens = 0

    if dedup == 'none':
        candidates: list[dict[str, Any]] = []
        for bank in solution_banks:
            in_budget_keys = [k for k in bank if _in_budget(bank, k)]
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
        in_budget_keys = [k for k in bank if _in_budget(bank, k)]
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


def _random_sample(
    candidates: list[dict[str, Any]],
    pop_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Uniform-random pick of `pop_size` candidates, score/diversity blind."""
    rng = random.Random(seed)
    return rng.sample(candidates, pop_size)


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
    curation_type: str,
    iteration_budget: int | None = None,
    max_population_size: int = 15,
    p: float = 0.9,
    embed_model_name: str = 'all-MiniLM-L6-v2',
    dedup: str = 'max',
    population_size: int | None = None,
    token_budget: int | None = None,
) -> list[dict[str, Any]]:
    """Curate a population.

    via tournament, diversity-rejection, greedy, or
    random sampling: greedy is top-N by score with no diversity filtering,
    and random is a uniform pick, score/diversity blind -- both baselines
    against which `p` is unused. `dedup` is passed straight to
    `_pool_candidates_within_budget` -- see its docstring for the 'max'
    (default) vs. 'min' distinction.

    `population_size`, when given, overrides the adaptive
    `max_population_size` scheme with an exact target instead (still
    capped at however many candidates are actually available -- this
    can't manufacture candidates that don't exist). Leave it `None` (the
    default) to keep the adaptive `min(max_population_size,
    total_candidates)` behavior.

    Exactly one of `iteration_budget`/`token_budget` is required to decide
    which pooled entries are in-budget; `token_budget` wins if both are
    given (see `_pool_candidates_within_budget`'s docstring), else
    `ValueError`.
    """
    if curation_type not in ('tournament', 'diversity', 'greedy', 'random'):
        raise ValueError(
            f"curation_type must be 'tournament', 'diversity', 'greedy', or "
            f"'random', got {curation_type!r}",
        )

    candidates, cost_tokens = _pool_candidates_within_budget(
        solution_banks,
        iteration_budget,
        dedup=dedup,
        token_budget=token_budget,
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
    elif curation_type == 'random':
        population = _random_sample(candidates, pop_size=curated_size, seed=42)
    else:
        # curation_type == 'diversity'
        solutions = [c['solution'] for c in candidates]
        if len(solutions) < 2:  # noqa: PLR2004
            # No pairwise similarities to derive a threshold from
            population = candidates[:curated_size]
        else:
            model = _load_embed_model(embed_model_name)
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
