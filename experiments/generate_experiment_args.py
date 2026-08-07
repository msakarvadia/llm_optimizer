"""Script to generate the experimental argument configs."""

from __future__ import annotations

import itertools
import json
import os
import pathlib
import re
from typing import Any

import yaml
from openai import OpenAI

from llm_optimizer.utils import prompt_lm


def get_args_for_roll_outs(
    task_name: str,
    num_gpus: int,
    num_cpus: int,
    num_iter: int,
    inference_model_name: str | None = None,
) -> list[dict[str, Any]]:
    """Generic roll outs experiment."""
    # --- Define Hyperparameter Parameter Search Space
    pruning_strategies = ['lowest_scoring']  # 'oldest'
    max_population_sizes = [5, 10, 20, 50]
    sampling_strategies = [
        'highest_scoring',
        'tournament',
        'wheel',
        #'random',
        'most_recent',
    ]
    mutators = ['kincontext', 'DE', 'GA', 'GEPA']
    noises = [0]
    # noises = [0, 0.1, 0.5]

    # --- Dynamic Combination Generation
    experiments_to_run = []

    # Perform cross-product combinations using itertools
    for pruning, pop_size, strategy, mutator, noise in itertools.product(
        pruning_strategies,
        max_population_sizes,
        sampling_strategies,
        mutators,
        noises,
    ):
        # Handle Kincontext Mutator Dependency
        n_values = [3, 10] if mutator == 'kincontext' else [5]
        # make sure context lengths don't exceed pop_size
        if len(n_values) > 1:
            n_values = [val for val in n_values if val <= pop_size]

        # Handle HarmBench llm optimizer
        optimizer_llm = (
            'mlabonne/NeuralDaredevil-8B-abliterated'
            if task_name == 'harmbench'
            else 'gemini-3.5-flash'
        )

        # Handle multiple benchmarks for prompt optimization
        benchmarks = ['drop', 'gsm8k'] if task_name == 'prompt' else ['drop']

        for benchmark in benchmarks:
            for n in n_values:
                # Build your clean parameter dict
                config = {
                    'optimizer_name': 'opro',
                    'optimizer_llm': optimizer_llm,
                    'task_name': task_name,
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
                if task_name == 'harmbench' and inference_model_name:
                    config['inference_model_name'] = inference_model_name

                experiments_to_run.append(config)

    return experiments_to_run


def get_args_for_long_run_cloud(
    task_name: str,
    num_gpus: int,
    num_cpus: int,
    num_iter: int,
) -> list[dict[str, Any]]:
    """Generic roll outs experiment."""
    # --- Define Hyperparameter Parameter Search Space
    pruning_strategies = ['lowest_scoring']  # 'oldest'
    max_population_sizes = [20]  # 20, 50
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
    for pruning, pop_size, strategy, mutator, noise in itertools.product(
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

        for n in n_values:
            # Build your clean parameter dict
            config = {
                'optimizer_name': 'opro',
                'optimizer_llm': optimizer_llm,
                'task_name': task_name,
                'pruning_strategy': pruning,
                'max_population_size': pop_size,
                'sampling_strategy_name': strategy,
                'mutator': mutator,
                'noise': noise,
                'num_iter': num_iter,
                'num_gpus': num_gpus,
                'num_cpus': num_cpus,
                'n': n,
            }

            experiments_to_run.append(config)

    return experiments_to_run


def run_random_number_bias_experiment(
    n_samples: int = 1000,
    max_retries: int = 3,
) -> dict[str, list[int]]:
    """Zero-shot probe for per-model bias in generated random numbers.

    Standalone: bypasses the Task/optimizer machinery entirely, since
    there's no solution to optimize here, just repeated independent
    samples to check for non-uniformity against a fair prior.
    """
    models = ['gemini-3.5-flash', 'gemini-2.5-flash', 'openai/gpt-oss-120b']

    project_root = pathlib.Path(__file__).parent.parent
    with open(project_root / 'config.yaml') as f:
        model_config = yaml.safe_load(f)

    output_dir = project_root / 'experiments' / 'random_number_bias_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = (
        'Give me a random number. Respond with only the number, no other text.'
    )

    results: dict[str, list[int]] = {}

    for model_name in models:
        base_url = model_config[model_name]['base_url']
        key_env_name = model_config[model_name]['key_env_name']
        client = OpenAI(api_key=os.getenv(key_env_name), base_url=base_url)

        samples: list[int] = []
        for i in range(n_samples):
            raw_output, _ = prompt_lm(
                client=client,
                prompt=prompt,
                model_name=model_name,
                max_retries=max_retries,
            )
            match = re.search(r'-?\d+', raw_output)
            if match is None:
                print(
                    f'[{model_name}] sample {i}: no number found in '
                    f'{raw_output!r}, skipping',
                )
                continue
            samples.append(int(match.group()))

        results[model_name] = samples
        print(f'{model_name}: collected {len(samples)}/{n_samples} samples')

        safe_model_name = model_name.replace('/', '_')
        safe_model_name = safe_model_name.replace('.', '')
        output_path = output_dir / f'{safe_model_name}.json'
        with open(output_path, 'w') as f:
            json.dump(samples, f, indent=2)

    return results


def get_args_for_pop_dynamics(
    population_dir: str,
    task_name: str,
    num_gpus: int,
    num_iter: int,
    num_cpus: int,
) -> list[dict[str, Any]]:
    """Experiments to understand population dynamics.

    `population_dir` is listed
    non-recursively, so this only ever produces configs for whatever single
    task's population files happen to be sitting directly in that directory.
    """
    pruning_strategy = 'lowest_scoring'
    max_population_sizes = [15]
    sampling_strategies = ['wheel', 'tournament']
    mutators = ['kincontext', 'DE', 'GA', 'GEPA']
    noises = [0, 1]
    sampling_prob = 0.5
    n = 3

    optimizer_llms = [
        'openai/gpt-oss-120b',
        'gemini-3.5-flash',
        'gemini-2.5-flash',
    ]

    init_population_files = sorted(
        [
            os.path.join(population_dir, f)
            for f in os.listdir(population_dir)
            if f.endswith('.json')
        ],
    )

    experiments_to_run = []

    for pop_size, strategy, mutator, noise, pop_path, llm in itertools.product(
        max_population_sizes,
        sampling_strategies,
        mutators,
        noises,
        init_population_files,
        optimizer_llms,
    ):
        benchmark = 'drop'

        config = {
            'optimizer_name': 'opro',
            'optimizer_llm': llm,
            'task_name': task_name,
            'pruning_strategy': pruning_strategy,
            'max_population_size': pop_size,
            'sampling_strategy_name': strategy,
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
