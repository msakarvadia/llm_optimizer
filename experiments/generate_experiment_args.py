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

# Local Ray-scheduling resources for the task subprocess itself -- independent
# of the shared VLLMServerActor GPUs, which experiments.py provisions
# separately per unique model. Most tasks route all inference/optimizer/
# classifier calls through a shared server (or an API-hosted model) and need
# zero local GPU; kernelbench compiles and executes the candidate kernel on
# real hardware and needs a dedicated GPU per task.
TASK_DEVICE_PROFILES: dict[str, dict[str, int]] = {
    'kernelbench': {'num_gpus': 1, 'num_cpus': 4},
}
DEFAULT_DEVICE_PROFILE: dict[str, int] = {'num_gpus': 0, 'num_cpus': 4}


def get_device_profile(task_name: str) -> dict[str, int]:
    """Resolve local Ray-scheduling resources for a task subprocess."""
    return TASK_DEVICE_PROFILES.get(task_name, DEFAULT_DEVICE_PROFILE)


def get_default_optimizer_llms(task_name: str) -> list[str]:
    """Resolve the default optimizer_llm(s) for a task_name.

    Only used when the caller doesn't explicitly pass `optimizer_llms`
    """
    if task_name == 'harmbench':
        return ['mlabonne/NeuralDaredevil-8B-abliterated']
    if task_name == 'prompt':
        return ['gemini-3.5-flash']
    return ['gemini-3.1-pro-preview']


def get_kincontext_n(task_name: str) -> int:
    """Kincontext mutator's in-context history length, per task."""
    if task_name in ('kernelbench', 'cloudcast', 'cantbelate'):
        return 3
    return 5


def get_args_for_roll_outs(
    task_name: str,
    num_iter: int,
    inference_model_name: str | None = None,
    optimizer_llms: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Generic roll outs experiment.

    Grid merged in from the former get_args_for_long_run_cloud (the two
    were redundant): fixed pruning strategy/pop size/noise, 3 sampling
    strategies, and all 3 optimizer frameworks -- 'opro' gets the full
    sampling-strategy x mutator sub-sweep (12 combos), 'gepa' and
    'open_evolve' each contribute exactly one config apiece since neither
    takes a mutator/sampling_strategy_name, for 14 combos total per
    (benchmark, optimizer_llm) pair.
    """
    # --- Define Hyperparameter Parameter Search Space
    pruning_strategy = 'lowest_scoring'
    max_population_size = 20
    sampling_strategies = [
        'highest_scoring',
        'tournament',
        'wheel',
    ]
    mutators = ['kincontext', 'DE', 'GA', 'GEPA']
    noise = 0

    optimizer_llms = optimizer_llms or get_default_optimizer_llms(task_name)
    device_profile = get_device_profile(task_name)

    # Handle multiple benchmarks for prompt optimization
    benchmarks = ['drop', 'gsm8k'] if task_name == 'prompt' else ['drop']

    experiments_to_run = []

    for benchmark, optimizer_llm in itertools.product(
        benchmarks,
        optimizer_llms,
    ):
        base_config: dict[str, Any] = {
            'task_name': task_name,
            'optimizer_llm': optimizer_llm,
            'pruning_strategy': pruning_strategy,
            'max_population_size': max_population_size,
            'noise': noise,
            'num_iter': num_iter,
            'benchmark': benchmark,
            **device_profile,
        }
        if task_name == 'harmbench' and inference_model_name:
            base_config['inference_model_name'] = inference_model_name
        if task_name == 'kernelbench':
            base_config['backend'] = 'cuda'
            base_config['problem_id'] = 1
            base_config['level'] = 1

        # OPRO: full sampling-strategy x mutator sub-sweep. Kincontext is
        # context-length-bound (n); every other mutator uses a fixed n.
        for strategy, mutator in itertools.product(
            sampling_strategies,
            mutators,
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
    num_iter: int,
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

    device_profile = get_device_profile(task_name)

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
            'benchmark': benchmark,
            'sampling_prob': sampling_prob,
            'n': n,
            'init_population_path': pop_path,
            **device_profile,
        }
        experiments_to_run.append(config)

    return experiments_to_run
