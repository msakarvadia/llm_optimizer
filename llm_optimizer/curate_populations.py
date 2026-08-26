"""In this file we will currate per-task populations."""

from __future__ import annotations

import itertools
import json
import os

from population_curation_utils import curate_population
from population_curation_utils import load_solution_banks_by_glob
from population_curation_utils import merge_solution_banks_in_order

POPULATIONS_DIR = 'populations'

PARALLEL_ZEROSHOT_DIR = '../parallel_zeroshot'


def sanitize_arg(val: str) -> str:
    """Match main.py's config_dir sanitization for a CLI arg value."""
    return str(val).replace('.', '').replace('/', '')


# Every non-prompt task ran against a single fixed target model and a single
# (functionally unused) dataset suffix ('drop'). `prompt` ran against two
# target models -- we only want oldmo-sft -- and two real datasets, which we
# split into separate task keys (`prompt_drop`/`prompt_gsm8k`) rather than
# pooling them, matching the split already used in
# curated_initial_populations_by_iter_budget/. `embed_model` mirrors
# generate_experiment_args.py's get_embed_model: code tasks get
# CodeRankEmbed, everything else gets all-MiniLM-L6-v2 -- used for the
# 'diversity' curation configs below.
TASK_SPECS = {
    'tsp': {
        'dir_task': 'tsp',
        'target_model': 'googlegemma-4-E4B-it',
        'dataset': 'drop',
        'embed_model': 'all-MiniLM-L6-v2',
    },
    'cloudcast': {
        'dir_task': 'cloudcast',
        'target_model': 'googlegemma-4-E4B-it',
        'dataset': 'drop',
        'embed_model': 'nomic-ai/CodeRankEmbed',
    },
    'cantbelate': {
        'dir_task': 'cantbelate',
        'target_model': 'googlegemma-4-E4B-it',
        'dataset': 'drop',
        'embed_model': 'nomic-ai/CodeRankEmbed',
    },
    'circlepacking': {
        'dir_task': 'circlepacking',
        'target_model': 'googlegemma-4-E4B-it',
        'dataset': 'drop',
        'embed_model': 'nomic-ai/CodeRankEmbed',
    },
    'prompt_drop': {
        'dir_task': 'prompt',
        'target_model': 'allenaiOLMo-2-0425-1B-SFT',
        'dataset': 'drop',
        'embed_model': 'all-MiniLM-L6-v2',
    },
    'prompt_gsm8k': {
        'dir_task': 'prompt',
        'target_model': 'allenaiOLMo-2-0425-1B-SFT',
        'dataset': 'gsm8k',
        'embed_model': 'all-MiniLM-L6-v2',
    },
}

# The (task, optimizer_llm) pairs actually used by general_rollout, i.e.
# generate_experiment_args.py's get_default_optimizer_llms(task_name) --
# copied verbatim (dots/slashes intact) and sanitized below the same way
# main.py's config_dir builds run-dir names, rather than hand-transcribing
# the already-sanitized form. parallel_zeroshot/ also has other
# optimizer_llms on disk per task (e.g. Kimi-K25, gpt-oss-120b,
# meta-llamaLlama-31-8B-Instruct for cloudcast/cantbelate) left over from
# other sweeps -- intentionally excluded here.
_RAW_TASK_OPTIMIZER_LLMS = {
    'tsp': ['gpt-oss-120b', 'gemini-3.7-flash', 'gemini-3.5-flash'],
    'cloudcast': [
        'gemini-3.7-flash',
        'gemini-3.5-flash',
        'deepseek/deepseek-v4-flash',
    ],
    'cantbelate': [
        'gemini-3.7-flash',
        'gemini-3.5-flash',
        'deepseek/deepseek-v4-flash',
    ],
    'circlepacking': [
        'gemini-3.5-flash',
        'gemini-3.7-flash',
        'gemini-2.5-flash',
        'gpt-oss-120b',
        'Qwen3_6-35B-A3B',
    ],
    'prompt_drop': [
        'gemini-2.5-flash',
        'gemini-3.5-flash',
        'gemini-3.7-flash',
        'meta-llama/Llama-3.1-8B-Instruct',
    ],
    'prompt_gsm8k': [
        'gemini-2.5-flash',
        'gemini-3.5-flash',
        'gemini-3.7-flash',
        'meta-llama/Llama-3.1-8B-Instruct',
    ],
}
TASK_OPTIMIZER_LLMS = {
    task: [sanitize_arg(m) for m in models]
    for task, models in _RAW_TASK_OPTIMIZER_LLMS.items()
}
MUTATORS = ['GEPA', 'kincontext']

# define task budgets in a dict
task_budgets = {
    'tsp': [100_000, 250_000, 500_000, 750_000, 900_000],
    'cloudcast': [200_000, 500_000, 1_000_000, 1_500_000, 1_800_000],
    'cantbelate': [100_000, 250_000, 500_000, 750_000, 900_000],
    'prompt_drop': [25_000, 50_000, 75_000, 100_000, 125_000],
    'prompt_gsm8k': [25_000, 50_000, 75_000, 100_000, 125_000],
    'circlepacking': [20_000, 30_000, 40_000, 50_000, 60_000],
}

population_size = 15

population_curation_args = [
    {
        'curation_type': 'random',
        'population_size': population_size,
        'dedup': 'none',
    },
    {
        'curation_type': 'greedy',
        'population_size': population_size,
        'dedup': 'none',
    },
    {
        'curation_type': 'diversity',
        'population_size': population_size,
        'dedup': 'max',
        'p': 0.95,
    },
    {
        'curation_type': 'diversity',
        'population_size': population_size,
        'dedup': 'max',
        'p': 0.9,
    },
]


def build_glob_str(optimizer_llm: str, mutator: str, task: str) -> str:
    """Glob matching every seed for one (optimizer_llm, mutator, task).

    Dir names are laid out
    opro_<optimizer_llm>_<mutator>_..._<seed>_<dir_task>_<target_model>_..._<dataset>
    -- optimizer_llm and mutator both come BEFORE task, not after, so the
    wildcards must follow that same left-to-right order or nothing matches.
    `dataset` is pinned at the end (rather than left as a trailing wildcard)
    so `prompt_drop`/`prompt_gsm8k` don't pool each other's runs.
    """
    spec = TASK_SPECS[task]
    return os.path.join(
        PARALLEL_ZEROSHOT_DIR,
        f'opro_{optimizer_llm}_{mutator}_*_{spec["dir_task"]}_'
        f'{spec["target_model"]}_*_{spec["dataset"]}',
    )


# Paired with (optimizer_llm, mutator, task) since the loop below needs
# all three. optimizer_llm comes from TASK_OPTIMIZER_LLMS[task] (varies
# per task), not one flat list crossed with every task.
glob_strs = [
    (
        optimizer_llm,
        mutator,
        task,
        build_glob_str(optimizer_llm, mutator, task),
    )
    for task in task_budgets
    for optimizer_llm, mutator in itertools.product(
        TASK_OPTIMIZER_LLMS[task],
        MUTATORS,
    )
]

if __name__ == '__main__':
    for optimizer_llm, mutator, task, glob_str in glob_strs:
        banks = load_solution_banks_by_glob(glob_str)
        if not banks:
            # Many (optimizer_llm, mutator, task) combos were never run.
            print(f'no dirs matched, skipping: {glob_str}')
            continue

        # Merged once per combo, not per budget -- neither the merge nor
        # the glob/load depends on budget, only curation below does.
        merged_bank = merge_solution_banks_in_order(banks)

        for budget in task_budgets[task]:
            for cfg in population_curation_args:
                population = curate_population(
                    [merged_bank],
                    curation_type=cfg['curation_type'],
                    token_budget=budget,
                    population_size=cfg['population_size'],
                    dedup=cfg['dedup'],
                    embed_model_name=TASK_SPECS[task]['embed_model'],
                    **({'p': cfg['p']} if 'p' in cfg else {}),
                )
                if not population:
                    print(
                        f'empty population, skipping: {optimizer_llm} '
                        f'{mutator} {task} budget={budget} {cfg}',
                    )
                    continue

                out_dir = os.path.join(
                    POPULATIONS_DIR,
                    task,
                    mutator,
                    optimizer_llm,
                    f'budget_{budget}',
                )
                os.makedirs(out_dir, exist_ok=True)

                p_suffix = f'_p{cfg["p"]}' if 'p' in cfg else ''
                filename = (
                    f'{cfg["curation_type"]}_dedup-{cfg["dedup"]}'
                    f'{p_suffix}.json'
                )
                out_path = os.path.join(out_dir, filename)
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(population, f, indent=2)
