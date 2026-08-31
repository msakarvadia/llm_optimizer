# Read all results dir and compile results into single csv
from __future__ import annotations

import sys
from pathlib import Path

# Force flush ensures this appears instantly
current_dir = Path(__file__).resolve().parent
print(f'📍 Current Dir: {current_dir}', flush=True)

project_root = current_dir.parent.parent
sys.path.append(str(project_root) + '/experiments/')

print('📦 Starting heavy library imports...', flush=True)
import glob
import itertools
from typing import Any
import json
import pandas as pd

# Local Ray-scheduling resources for the task subprocess itself -- independent
# of the shared VLLMServerActor GPUs, which experiments.py provisions
# separately per unique model. Most tasks route all inference/optimizer/
# classifier calls through a shared server (or an API-hosted model) and need
# zero local GPU; kernelbench compiles and executes the candidate kernel on
# real hardware and needs a dedicated GPU per task.
TASK_DEVICE_PROFILES: dict[str, dict[str, int | float]] = {
    'kernelbench': {'num_gpus': 1, 'num_cpus': 1},
    'harmbench': {'num_gpus': 0, 'num_cpus': 0.5},
    # circlepacking's evaluate() runs numpy/scipy (LP solve) in a
    # subprocess; at the default 0.25 share Ray oversubscribes the
    # node badly enough that concurrent BLAS threads contend for
    # cores and candidates that finish in ~1s standalone blow past
    # the 600s eval timeout (or segfault under load).
    'circlepacking': {'num_gpus': 0, 'num_cpus': 1},
}
DEFAULT_DEVICE_PROFILE: dict[str, int | float] = {
    'num_gpus': 0,
    # Higher cpu share 0.1 hurts concurrency
    # was letting these OOM-kill the job's
    'num_cpus': 0.25,
}

# Cumulative mutator+diversity-judge token budget per task_name (see
# OPROOptimizer.optimize's max_tokens; opro-only -- gepa/open_evolve ignore
# it, see main.py). Tasks not listed here get no budget, i.e. num_iter
# remains the sole stopping criterion.
TASK_TOKEN_BUDGETS: dict[str, int] = {
    'harmbench': 70_000,
    'prompt': 150_000,
    'cloudcast': 2_000_000,
    'cantbelate': 1_000_000,
    'tsp': 1_000_000,
    'circlepacking': 70_000,
}

# OPRO-only diversity-check sweep (gepa/open_evolve ignore sim_thresh, see
# opro.py). 'llm_judge' omitted (not None) so experiments.py's str(val)
# CLI serialization doesn't emit a literal "None" main.py can't parse.
# NOTE(MS): diversity filtering / llm-judge sweep disabled for now for
# general_rollout -- only the no-filtering variant runs. Uncomment the
# rest to re-enable the full sweep.
DIVERSITY_CHECK_VARIANTS: list[dict[str, Any]] = [
    {'sim_thresh': 0.0, 'llm_judge': None},  # no filtering (current default)
    {'sim_thresh': -1, 'llm_judge': None},  # exact-dedup baseline
    {'sim_thresh': 0.95, 'llm_judge': None},  # fuzzy, no judge
    {'sim_thresh': 0.95, 'llm_judge': 'gemini-3.5-flash'},  # fuzzy, judged
    {'sim_thresh': 0.8, 'llm_judge': None},  # fuzzy, no judge
    {'sim_thresh': 0.8, 'llm_judge': 'gemini-3.5-flash'},  # fuzzy, judged
]


def get_device_profile(task_name: str) -> dict[str, int | float]:
    """Resolve local Ray-scheduling resources for a task subprocess."""
    return TASK_DEVICE_PROFILES.get(task_name, DEFAULT_DEVICE_PROFILE)


def get_default_optimizer_llms(task_name: str) -> list[str]:
    """Resolve the default optimizer_llm(s) for a task_name.

    Only used when the caller doesn't explicitly pass `optimizer_llms`
    """
    if task_name == 'harmbench':
        return ['mlabonne/NeuralDaredevil-8B-abliterated']
    if task_name == 'prompt':
        # maybe llama
        # maybe mistral (smaller)
        return [
            #'gemini-2.5-flash',
            'gemini-3.5-flash',
            #'gemini-3.7-flash',
            'meta-llama/Llama-3.1-8B-Instruct',
        ]
    if task_name == 'circlepacking':
        # oss-120b
        # weaker code model
        return [
            #'gemini-3.5-flash',
            #'gemini-3.7-flash',
            'gemini-2.5-flash',
            'gpt-oss-120b',
            'Qwen3_6-35B-A3B',
        ]  # I ran w/ gemini-3.7 (but fails for parallel)
    if task_name == 'tsp':
        return [
            'gpt-oss-120b',
            'gemini-3.7-flash',
            'gemini-3.5-flash',
        ]
    if task_name in ['cloudcast', 'cantbelate']:
        # deepseek
        return [
            'gemini-3.7-flash',
            'gemini-3.5-flash',
            #'deepseek/deepseek-v4-flash',
            #'Kimi-K2.5',
        ]
    # return ['gemini-3.1-pro-preview']
    return ['gemini-3.5-flash']


def get_default_inference_model_names(task_name: str) -> list[str]:
    """Resolve the default inference_model_name(s) for a task_name.

    Only used when the caller doesn't explicitly pass `inference_model_names`
    """
    if task_name == 'harmbench':
        return ['allenai/OLMo-2-0425-1B-DPO']
    if task_name == 'prompt':
        return [
            'allenai/OLMo-2-0425-1B-SFT',
            #'meta-llama/Llama-3.2-1B-Instruct',
        ]
    return ['google/gemma-4-E4B-it']


def get_kincontext_n(task_name: str) -> int:
    """Kincontext mutator's in-context history length, per task."""
    if task_name in (
        'kernelbench',
        'cloudcast',
        'cantbelate',
        'circlepacking',
    ):
        return 3
    return 5


def get_embed_model(task_name: str) -> str:
    """Diversity-check embedding model, per task."""
    if task_name in (
        'kernelbench',
        'cloudcast',
        'cantbelate',
        'circlepacking',
    ):
        return 'nomic-ai/CodeRankEmbed'
    return 'all-MiniLM-L6-v2'


def get_args_for_roll_outs(
    task_name: str,
    num_iter: int,
    inference_model_names: list[str] | None = None,
    optimizer_llms: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Generic roll outs experiment.

    'opro' gets the full sampling-strategy x mutator x noise sub-sweep
    (48 combos, noise is OPRO-only) per (benchmark, optimizer_llm,
    inference_model_name) triple. 'gepa' and 'open_evolve' each
    contribute one fixed-noise=0 config on top of that.
    """
    # --- Define Hyperparameter Parameter Search Space
    pruning_strategy = 'lowest_scoring'
    max_population_size = 15
    sampling_strategies = [
        'highest_scoring',
        'tournament',
        'wheel',
    ]
    mutators = ['kincontext', 'DE', 'GA', 'GEPA']
    # OPRO-only sweep -- gepa/open_evolve keep noise fixed at 0 (see
    # base_config below), so they aren't inflated by this axis.
    opro_noise_values = [0.0]
    # opro_noise_values = [0, 0.05, 0.1, 0.25]

    optimizer_llms = optimizer_llms or get_default_optimizer_llms(task_name)
    inference_model_names = (
        inference_model_names or get_default_inference_model_names(task_name)
    )
    device_profile = get_device_profile(task_name)

    # Handle multiple benchmarks for prompt optimization
    benchmarks = ['drop', 'gsm8k'] if task_name == 'prompt' else ['drop']

    experiments_to_run = []

    for benchmark, optimizer_llm, inference_model_name in itertools.product(
        benchmarks,
        optimizer_llms,
        inference_model_names,
    ):
        base_config: dict[str, Any] = {
            'task_name': task_name,
            'optimizer_llm': optimizer_llm,
            'pruning_strategy': pruning_strategy,
            'max_population_size': max_population_size,
            'noise': 0,
            'num_iter': num_iter,
            'benchmark': benchmark,
            'inference_model_name': inference_model_name,
            **device_profile,
        }
        if task_name == 'kernelbench':
            base_config['backend'] = 'cuda'
            base_config['problem_id'] = 1
            base_config['level'] = 1
        if task_name == 'tsp':
            base_config['num_points'] = 100  # 200, 80
        if task_name in TASK_TOKEN_BUDGETS:
            base_config['max_tokens'] = TASK_TOKEN_BUDGETS[task_name]

        # OPRO: full sampling-strategy x mutator x noise x diversity-check
        # sub-sweep. Kincontext is context-length-bound (n); every other
        # mutator uses a fixed n.
        for strategy, mutator, noise, diversity_variant in itertools.product(
            sampling_strategies,
            mutators,
            opro_noise_values,
            DIVERSITY_CHECK_VARIANTS,
        ):
            n_values = (
                [get_kincontext_n(task_name)]
                if mutator == 'kincontext'
                else [5]
            )
            for n in n_values:
                experiments_to_run.append(
                    {
                        **base_config,
                        'optimizer_name': 'opro',
                        'sampling_strategy_name': strategy,
                        'mutator': mutator,
                        'n': n,
                        'noise': noise,
                        'embed_model': get_embed_model(task_name),
                        'prune_noise': 0.0,
                        **diversity_variant,
                    },
                )

        # GEPA / OpenEvolve: neither reads mutator/sampling_strategy_name,
        # so each contributes exactly one config here.
        for optimizer_name in ('gepa', 'open_evolve'):
            experiments_to_run.append(
                {
                    **base_config,
                    'optimizer_name': optimizer_name,
                },
            )

    return experiments_to_run


print('🎉 Finished all imports!', flush=True)

######################################################

experiment_dir = '../../general_rollout'
output_csv = 'compiled_results.csv'

# get all args
tasks = ['cantbelate', 'cloudcast', 'tsp', 'circlepacking', 'prompt']


desired_order = [
    'optimizer_name',
    'optimizer_llm',
    'mutator',
    'sampling_strategy_name',
    'sampling_prob',
    'n',
    'max_population_size',
    'pruning_strategy',
    'noise',
    'embed_model',
    'llm_judge',
    'sim_thresh',
    'task_name',
    'inference_model_name',
    'num_points',
    'benchmark',
]
all_rows = []
for task in tasks:
    args = get_args_for_roll_outs(task, 500)
    print(f"{task=}, {len(args)=}")
    total_files = 0
    for arg_set in args:
        clean_dict = {
            key: str(val).replace('.', '').replace('/', '')
            for key, val in arg_set.items()
        }
        # Clean values, but apply a specific rule for the 'noise' key
        # NOTE(MS): exact matches noise/prune_noise
        clean_dict = {
            key: (
                f'{str(val).replace(".", "").replace("/", "")}_00'
                if key == 'noise'
                else str(val).replace('.', '').replace('/', '')
            )
            for key, val in arg_set.items()
        }
        ordered_dict = {
            key: clean_dict[key] for key in desired_order if key in clean_dict
        }

        folder_pattern = '*'.join(ordered_dict.values())

        json_files = glob.glob(
            f'{experiment_dir}/*{folder_pattern}*/**/long*.json',
            recursive=True,
        )
        total_files += len(json_files)
        if len(json_files) > 1:
            print(arg_set)
            print(folder_pattern)
            for j in json_files:
                print(j)
            print('-----------')

        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    json_data = json.load(f)

                # 3. Create an independent row for each integer step
                if isinstance(json_data, dict):
                    for step, nested_dict in json_data.items():
                        if isinstance(nested_dict, dict):
                            # Combine base arguments + step info + step metrics
                            row_entry = {
                                **arg_set,
                                'step': int(step),  # Keep track of the step index
                                    **nested_dict,
                                'source_file': file_path
                            }
                            all_rows.append(row_entry)

            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    print(f"Total matched experimetns: ", total_files)


df = pd.DataFrame(all_rows)
print(f"Successfully processed {len(df)} rows.")
print(df.head())

# calculate total memory usage in bytes, sum it, and convert to GB
df_size_gb = df.memory_usage(deep=True).sum() / (1024 ** 3)
print(f"DataFrame size in memory: {df_size_gb:.4f} GB")


df.to_csv("results.csv", index=False)
