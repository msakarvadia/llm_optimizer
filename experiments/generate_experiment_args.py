"""Script to generate the experimental argument configs."""

from __future__ import annotations

import itertools
from typing import Any


def get_args_for_roll_outs() -> list[dict[str, Any]]:
    """Generic roll outs experiment."""
    # --- Define Hyperparameter Parameter Search Space
    num_iter = 50
    tasks = ['tweet', 'kernelbench']
    pruning_strategies = ['oldest', 'lowest_scoring']
    max_population_sizes = [3, 5, 10, 20, 50]
    sampling_strategies = [
        'random',
        'most_recent',
        'highest_scoring',
        'tournament',
        'wheel',
    ]
    mutators = ['kincontext', 'DE', 'GA', 'GEPA']
    noises = [0, 0.1, 0.5, 1.0]

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
        sampling_probs = (
            [0.5, 0.7, 0.9] if strategy == 'tournament' else [None]
        )

        # Handle Kincontext Mutator Dependency
        n_values = [3, 20, 50] if mutator == 'kincontext' else [None]

        # Num GPUs
        num_gpus = 1 if task == 'tweet' else 3

        for prob in sampling_probs:
            for n in n_values:
                # Build your clean parameter dict
                config = {
                    'optimizer_name': 'opro',
                    'task_name': task,
                    'pruning_strategy': pruning,
                    'max_population_size': pop_size,
                    'sampling_strategy_name': strategy,
                    'mutator': mutator,
                    'noise': noise,
                    'num_iter': num_iter,
                    'num_gpus': num_gpus,
                }

                # Conditionally append optional parameters if they are not None
                if prob is not None:
                    config['sampling_prob'] = prob
                if n is not None:
                    config['n'] = n

                experiments_to_run.append(config)

    return experiments_to_run
