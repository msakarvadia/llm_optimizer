"""Script to generate the experimental argument configs."""

from __future__ import annotations

import itertools
import os
from typing import Any


def get_args_for_roll_outs() -> list[dict[str, Any]]:
    """Generic roll outs experiment."""
    # --- Define Hyperparameter Parameter Search Space
    num_iter = 50
    tasks = [
        'harmbench',
        'prompt',
        'cantbelate',
        'cloudcast',
        'kernelbench',
        #'tweet',
    ]
    pruning_strategies = ['lowest_scoring']  # 'oldest'
    max_population_sizes = [5, 10, 20, 50]
    sampling_strategies = [
        'highest_scoring',
        'tournament',
        'wheel',
        'random',
        'most_recent',
    ]
    mutators = ['kincontext', 'DE', 'GA', 'GEPA']
    noises = [0, 0.1, 0.5]

    # --- Dynamic Combination Generation
    experiments_to_run = []

    # Perform cross-product combinations using itertools
    for task, pruning, pop_size, strategy, mutator, noise in itertools.product(
        tasks,
        pruning_strategies,
        max_population_sizes,
        sampling_strategies,
        mutators,
        noises,
    ):
        # Handle Kincontext Mutator Dependency
        n_values = [3, 20, 50] if mutator == 'kincontext' else [5]
        # make sure context lengths don't exceed pop_size
        if len(n_values) > 1:
            n_values = [val for val in n_values if val <= pop_size]

        # Handle HarmBench llm optimizer
        optimizer_llm = (
            'mlabonne/NeuralDaredevil-8B-abliterated'
            if task == 'harmbench'
            else 'gemini-3.5-flash'
        )

        # Handle multiple benchmarks for prompt optimization
        benchmarks = ['drop', 'gsm8k'] if task == 'prompt' else ['drop']

        # Num GPUs
        num_cpus = 8
        if task in ['cloudcast', 'cantbelate']:
            num_gpus = 0
        if task in ['prompt', 'kernelbench']:
            num_gpus = 1
        if task in ['tweet']:
            num_gpus = 1
        if task in ['harmbench']:
            num_gpus = 2

        for benchmark in benchmarks:
            for n in n_values:
                # Build your clean parameter dict
                config = {
                    'optimizer_name': 'opro',
                    'optimizer_llm': optimizer_llm,
                    'task_name': task,
                    'pruning_strategy': pruning,
                    'max_population_size': pop_size,
                    'sampling_strategy_name': strategy,
                    'mutator': mutator,
                    'noise': noise,
                    'num_iter': num_iter,
                    'num_gpus': num_gpus,
                    'num_cpus': num_cpus,
                    'benchmark': benchmark,
                    'n': n,
                }

                experiments_to_run.append(config)

    return experiments_to_run


def get_args_for_long_run_cloud() -> list[dict[str, Any]]:
    """Generic roll outs experiment."""
    # --- Define Hyperparameter Parameter Search Space
    num_iter_by_task = {
        'cantbelate': 1500,
        'cloudcast': 1500,
    }
    tasks = [
        'cantbelate',
        'cloudcast',
    ]
    pruning_strategies = ['lowest_scoring']  # 'oldest'
    max_population_sizes = [20, 50]
    sampling_strategies = [
        'highest_scoring',
        'tournament',
        'wheel',
        #'most_recent',
    ]
    mutators = ['kincontext', 'DE', 'GA', 'GEPA']
    noises = [0]

    # --- Dynamic Combination Generation
    experiments_to_run = []

    # Perform cross-product combinations using itertools
    for task, pruning, pop_size, strategy, mutator, noise in itertools.product(
        tasks,
        pruning_strategies,
        max_population_sizes,
        sampling_strategies,
        mutators,
        noises,
    ):
        # Handle Kincontext Mutator Dependency
        n_values = [3] if mutator == 'kincontext' else [5]
        # make sure context lengths don't exceed pop_size
        if len(n_values) > 1:
            n_values = [val for val in n_values if val <= pop_size]

        # Handle HarmBench llm optimizer
        optimizer_llm = 'gemini-3.1-pro-preview'

        # Num GPUs
        num_cpus = 4
        num_gpus = 0

        for n in n_values:
            # Build your clean parameter dict
            config = {
                'optimizer_name': 'opro',
                'optimizer_llm': optimizer_llm,
                'task_name': task,
                'pruning_strategy': pruning,
                'max_population_size': pop_size,
                'sampling_strategy_name': strategy,
                'mutator': mutator,
                'noise': noise,
                'num_iter': num_iter_by_task[task],
                'num_gpus': num_gpus,
                'num_cpus': num_cpus,
                'n': n,
            }

            experiments_to_run.append(config)

    return experiments_to_run


def get_args_for_pop_dynamics() -> list[dict[str, Any]]:
    """Experiments to understand population dynamics."""
    num_iter = 50
    task = 'tweet'
    pruning_strategy = 'lowest_scoring'
    max_population_sizes = [10, 15]
    sampling_strategy = 'wheel'
    mutators = ['kincontext', 'DE', 'GA', 'GEPA']
    noise = 0
    sampling_prob = 0.5
    n = 3

    optimizer_llms = [
        'gemini-3.5-flash',
        'gemini-2.5-flash',
        'openai/gpt-oss-120b',
    ]

    population_dir = (
        '/scratch/mansisak/llm_optimizer/curated_initial_populations'
    )
    init_population_files = sorted(
        [
            os.path.join(population_dir, f)
            for f in os.listdir(population_dir)
            if f.endswith('.json')
        ],
    )

    experiments_to_run = []

    for pop_size, mutator, llm, pop_path in itertools.product(
        max_population_sizes,
        mutators,
        optimizer_llms,
        init_population_files,
    ):
        benchmark = 'drop'
        num_cpus = 8
        num_gpus = 1

        config = {
            'optimizer_name': 'opro',
            'optimizer_llm': llm,
            'task_name': task,
            'pruning_strategy': pruning_strategy,
            'max_population_size': pop_size,
            'sampling_strategy_name': sampling_strategy,
            'mutator': mutator,
            'noise': noise,
            'num_iter': num_iter,
            'num_gpus': num_gpus,
            'num_cpus': num_cpus,
            'benchmark': benchmark,
            'sampling_prob': sampling_prob,
            'n': n,
            'init_population_path': pop_path,
        }
        experiments_to_run.append(config)

    return experiments_to_run
