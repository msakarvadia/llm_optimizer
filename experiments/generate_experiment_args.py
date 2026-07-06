"""Script to generate the experimental argument configs."""

from __future__ import annotations

import itertools
from typing import Any


def get_args_for_roll_outs() -> list[dict[str, Any]]:
    """Generic roll outs experiment."""
    # --- Define Hyperparameter Parameter Search Space
    num_iter = 50
    tasks = [
        'tweet',
        'kernelbench',
        'harmbench',
        'prompt',
        'cloudcast',
        'cantbelate',
    ]
    pruning_strategies = ['lowest_scoring']  # 'oldest'
    max_population_sizes = [5, 10, 20, 50]
    sampling_strategies = [
        'random',
        'most_recent',
        'highest_scoring',
        'tournament',
        'wheel',
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
        # Handle Tournament Sampling Dependency
        sampling_probs = [0.5, 0.9] if strategy == 'tournament' else [0.5]

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
        if task in ['prompt']:
            num_gpus = 1
        if task in ['tweet']:
            num_gpus = 1
        if task in ['kernelbench', 'harmbench']:
            num_gpus = 3

        for benchmark in benchmarks:
            for prob in sampling_probs:
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
                        'sampling_prob': prob,
                        'n': n,
                    }

                    experiments_to_run.append(config)

    return experiments_to_run
