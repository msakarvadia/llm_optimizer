"""Currate per-task populations across varying population sizes.

Unlike curate_populations.py (single population_size=15, one pool per
task/mutator/optimizer_llm, budget as a fraction of the pool), this script
partitions each task's raw parallel_zeroshot pool into NUM_SEED_PARTITIONS
independent, non-overlapping seed blocks -- so repeatability can be assessed
across seeds -- and curates each block at one or more population sizes.

Each seed partition is sized so it has at least TASK_TOKEN_BUDGET[task]
tokens of its own (a *full*-scale pool, not an arbitrary 1/N-th slice);
partitioning stops greedily consuming seeds once that floor is cleared.  A
partition that can't clear the floor even after consuming every remaining
seed is skipped entirely (not written with a diminished token count).

The actual curated population is then drawn from INIT_BUDGET_FRACTION of
that floor (the initialization budget), not the full floor itself.

Scope: kincontext mutator only. Covers 3 optimizer_llms per task, across
all 5 tasks (tsp, cloudcast, cantbelate, circlepacking, prompt_gsm8k) --
see _RAW_TASK_OPTIMIZER_LLMS, which lists exactly the optimizer_llms
empirically confirmed (via results.csv) to have been used in that task's
original general_rollout/population_dynamics experiments. NOTE:
curate_populations.py's own _RAW_TASK_OPTIMIZER_LLMS is NOT used as the
source of truth here -- it's known to be stale in places (e.g. it lists
gemini-3.7-flash for prompt_gsm8k, which never actually appears in the
real data).

Of each task's 3 optimizer_llms, exactly one -- see FULL_COVERAGE_PAIR --
already has full multi-seed coverage from an earlier batch: the full
population-size ablation (m=15/30/45) and all 3 init conditions
(Uninitialized, Greedy, Random). The other two optimizer_llms per task
are new additions, restricted to a single m=15 population and only the
Greedy curation config (no Random) -- see CURATION_CONFIGS filtering
below.
"""

from __future__ import annotations

import json
import os
from typing import Any

from curate_populations import build_glob_str
from curate_populations import sanitize_arg
from curate_populations import TASK_SPECS
from population_curation_utils import curate_population
from population_curation_utils import load_solution_banks_by_glob
from population_curation_utils import merge_solution_banks_in_order

POPULATIONS_DIR = 'populations_vary_pop_size'

MUTATOR = 'kincontext'

# The 3 optimizer_llms per task empirically confirmed (via results.csv) to
# have been used in that task's original general_rollout/population_dynamics
# experiments. Raw (unsanitized) form -- sanitized via sanitize_arg below,
# matching main.py's config_dir naming convention.
_RAW_TASK_OPTIMIZER_LLMS: dict[str, list[str]] = {
    'prompt_gsm8k': [
        'gemini-2.5-flash',
        'gemini-3.5-flash',
        'meta-llama/Llama-3.1-8B-Instruct',
    ],
    'tsp': [
        'gemini-3.5-flash',
        'gemini-3.7-flash',
        'gpt-oss-120b',
    ],
    'cloudcast': [
        'deepseek/deepseek-v4-flash',
        'gemini-3.5-flash',
        'gemini-3.7-flash',
    ],
    'cantbelate': [
        'deepseek/deepseek-v4-flash',
        'gemini-3.5-flash',
        'gemini-3.7-flash',
    ],
    'circlepacking': [
        'Qwen3_6-35B-A3B',
        'gemini-2.5-flash',
        'gpt-oss-120b',
    ],
}

# Per task, the one raw optimizer_llm (must match an entry in
# _RAW_TASK_OPTIMIZER_LLMS[task] exactly) that already has full multi-seed
# coverage from an earlier batch -- gets the full POPULATION_SIZES sweep
# and all 3 init conditions. Every other optimizer_llm for that task is a
# new addition: single m=15, Greedy + Uninitialized only (no Random).
FULL_COVERAGE_PAIR: dict[str, str] = {
    'prompt_gsm8k': 'meta-llama/Llama-3.1-8B-Instruct',
    'tsp': 'gemini-3.5-flash',
}

# Mirrors experiments/generate_experiment_args.py's TASK_TOKEN_BUDGETS
# (keyed there by real task_name 'tsp'/'prompt'/etc., not the
# prompt_gsm8k/prompt_drop split). Duplicated here rather than imported --
# experiments/ is a separate top-level package from llm_optimizer/, and
# curate_populations.py already establishes the precedent of duplicating
# these same per-task constants locally instead of importing across that
# boundary.
TASK_TOKEN_BUDGET: dict[str, int] = {
    'tsp': 1_000_000,
    'cloudcast': 2_000_000,
    'cantbelate': 1_000_000,
    'circlepacking': 200_000,
    'prompt_gsm8k': 150_000,
}

NUM_SEED_PARTITIONS = 3
POPULATION_SIZES = [15, 30, 45]
INIT_BUDGET_FRACTION = 0.25

# 'greedy_dedup w/ 0.98 rejection threshold' -> diversity curation (score-
# ranked, greedily accepts if dissimilar enough, dedup='max' collapses
# duplicate solutions to their best score) at p=0.98, matching
# curate_populations.py's diversity_dedup-max_p0.98 config. 'random
# sampling, no filtering' -> uniform random pick, dedup='none' keeps every
# pooled candidate (incl. duplicates) in play.
CURATION_CONFIGS: list[dict[str, Any]] = [
    {
        'label': 'diversity_dedup-max_p0.98',
        'curation_type': 'diversity',
        'dedup': 'max',
        'p': 0.98,
    },
    {
        'label': 'random_dedup-none',
        'curation_type': 'random',
        'dedup': 'none',
    },
]


def _bank_token_cost(bank: dict[str, Any]) -> int:
    """A single raw (pre-merge) bank's own token cost.

    Mirrors _pool_candidates_within_budget's convention: each raw
    parallel_zeroshot seed bank is its own from-zero mini-bank, so its
    total cost is whatever its highest-index entry's cumulative_tokens_spent
    says.
    """
    if not bank:
        return 0
    last_key = max(bank, key=int)
    return bank[last_key].get('cumulative_tokens_spent', 0) or 0


def partition_banks_by_token_floor(
    banks: list[dict[str, Any]],
    num_partitions: int,
    token_floor: int,
) -> list[dict[str, Any]]:
    """Greedily split seed-ordered banks into `num_partitions` blocks.

    Each block accumulates seeds (continuing where the previous block left
    off) until its own cumulative token cost reaches `token_floor`. Returns
    one dict per partition: {'banks': [...], 'tokens': int,
    'reached_floor': bool}. A partition with reached_floor=False consumed
    every remaining bank without reaching token_floor -- the caller should
    skip writing curated output for it rather than use a diminished pool.
    """
    results: list[dict[str, Any]] = []
    idx = 0
    n = len(banks)
    for _ in range(num_partitions):
        partition: list[dict[str, Any]] = []
        cum_tokens = 0
        while idx < n and cum_tokens < token_floor:
            bank = banks[idx]
            partition.append(bank)
            cum_tokens += _bank_token_cost(bank)
            idx += 1
        results.append(
            {
                'banks': partition,
                'tokens': cum_tokens,
                'reached_floor': cum_tokens >= token_floor,
            },
        )
    leftover = n - idx
    if leftover > 0:
        print(f'  {leftover} leftover seed(s) unused after filling partitions')
    return results


if __name__ == '__main__':
    for task, raw_llms in _RAW_TASK_OPTIMIZER_LLMS.items():
        token_floor = TASK_TOKEN_BUDGET[task]
        init_budget = round(token_floor * INIT_BUDGET_FRACTION)
        full_coverage_raw_llm = FULL_COVERAGE_PAIR.get(task)

        for raw_llm in raw_llms:
            # TEMPORARY: deepseek's API key is expired (see parallel_zeroshot
            # top-up run), so skip it here for now -- remove this once the
            # key is renewed and cantbelate/deepseek's pool has been topped
            # up past the 3x-budget floor.
            if 'deepseek' in raw_llm:
                print(f'skipping {task}/{raw_llm}: deepseek API key expired')
                continue

            optimizer_llm = sanitize_arg(raw_llm)
            is_full_coverage = raw_llm == full_coverage_raw_llm
            pop_sizes = POPULATION_SIZES if is_full_coverage else [15]
            curation_configs = (
                CURATION_CONFIGS
                if is_full_coverage
                else [
                    cfg
                    for cfg in CURATION_CONFIGS
                    if cfg['label'] != 'random_dedup-none'
                ]
            )

            glob_str = build_glob_str(optimizer_llm, MUTATOR, task)
            banks = load_solution_banks_by_glob(glob_str)
            if not banks:
                print(f'no dirs matched, skipping: {glob_str}')
                continue
            total_tokens = sum(_bank_token_cost(b) for b in banks)
            print(
                f'{task}/{optimizer_llm} '
                f'(full_coverage={is_full_coverage}): {len(banks)} seed '
                f'banks, {total_tokens:,} total tokens '
                f'(need >= {NUM_SEED_PARTITIONS * token_floor:,} for '
                f'{NUM_SEED_PARTITIONS} full-budget partitions)',
            )

            partitions = partition_banks_by_token_floor(
                banks,
                NUM_SEED_PARTITIONS,
                token_floor,
            )

            for seed_idx, part in enumerate(partitions):
                if not part['reached_floor']:
                    print(
                        f'  seed{seed_idx}: FAILED to reach {token_floor:,} '
                        f'token floor (only {part["tokens"]:,} tokens from '
                        f'{len(part["banks"])} remaining seeds) -- skipping '
                        f'this partition entirely',
                    )
                    continue

                merged_bank = merge_solution_banks_in_order(part['banks'])
                print(
                    f'  seed{seed_idx}: {len(part["banks"])} banks, '
                    f'{part["tokens"]:,} tokens (floor {token_floor:,}, '
                    f'curating at {init_budget:,})',
                )

                for pop_size in pop_sizes:
                    for cfg in curation_configs:
                        population = curate_population(
                            [merged_bank],
                            curation_type=cfg['curation_type'],
                            token_budget=init_budget,
                            population_size=pop_size,
                            dedup=cfg['dedup'],
                            embed_model_name=TASK_SPECS[task]['embed_model'],
                            failed_score=TASK_SPECS[task].get(
                                'failed_score',
                            ),
                            **({'p': cfg['p']} if 'p' in cfg else {}),
                        )
                        if not population:
                            print(
                                f'    empty population, skipping: '
                                f'seed{seed_idx} m={pop_size} {cfg["label"]}',
                            )
                            continue

                        out_dir = os.path.join(
                            POPULATIONS_DIR,
                            task,
                            MUTATOR,
                            optimizer_llm,
                            f'seed{seed_idx}',
                        )
                        os.makedirs(out_dir, exist_ok=True)

                        filename = f'{cfg["label"]}_m{pop_size}.json'
                        out_path = os.path.join(out_dir, filename)
                        with open(out_path, 'w', encoding='utf-8') as f:
                            json.dump(population, f, indent=2)

    print('Done.')
