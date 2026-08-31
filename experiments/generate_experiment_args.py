"""Script to generate the experimental argument configs."""

from __future__ import annotations

import glob
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
    {'sim_thresh': 0.0, 'llm_judge':None},  # no filtering (current default)
    {'sim_thresh': -1,'llm_judge':None},  # exact-dedup baseline
    {'sim_thresh': 0.95, 'llm_judge':None},  # fuzzy, no judge
    {'sim_thresh': 0.95, 'llm_judge': 'gemini-3.5-flash'},  # fuzzy, judged
    {'sim_thresh': 0.8, 'llm_judge':None},  # fuzzy, no judge
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


# Parallel zero-shot discovery: instead of one sequential trajectory rolled
# out to a token/iteration budget, roll a single trajectory out for exactly
# 1 step (num_iter=1) and repeat that over N seeds in parallel -- budget is
# spent as breadth (N parallel 1-step attempts) instead of depth (one long
# sequential rollout). N per task is chosen to roughly match how many
# sequential steps a normal general_rollout run gets through before
# exhausting that task's TASK_TOKEN_BUDGETS entry (see
# get_args_for_parallel_zeroshot's docstring).
PARALLEL_ZEROSHOT_NUM_SEEDS: dict[str, int] = {
    'harmbench': 150,
    'prompt': 150,
    'cloudcast': 600,
    'cantbelate': 300,
    'tsp': 200,
    'circlepacking': 150,
}

# Only kincontext and GEPA are swept for parallel_zeroshot -- at bank size 1
# (exactly what num_iter=1 leaves the bank at), DE and GA both explicitly
# fall back to delegating into KInContextMutator (see their
# `len(past_solutions) < 3` guards in differential_evolution.py /
# genetic_algorithm.py), so sweeping them here would just be kincontext
# again under a different name. GEPA has no such guard and remains a
# genuinely distinct single-parent mutator at this bank size.
PARALLEL_ZEROSHOT_MUTATORS: list[str] = ['kincontext', 'GEPA']


def get_args_for_parallel_zeroshot(
    task_name: str,
    mutators: list[str] | None = None,
    optimizer_llms: list[str] | None = None,
    inference_model_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Parallel zero-shot discovery: N independent 1-step OPRO rollouts.

    For each (mutator, optimizer_llm, inference_model_name) in `mutators` x
    `optimizer_llms` x `inference_model_names` (defaults
    PARALLEL_ZEROSHOT_MUTATORS -- kincontext and GEPA only, see module
    docstring above -- get_default_optimizer_llms(task_name), and
    get_default_inference_model_names(task_name)), emits one config per seed
    in range(PARALLEL_ZEROSHOT_NUM_SEEDS[task_name]) with num_iter=1 and that
    seed. Everything else (device profile, benchmark(s) for 'prompt') mirrors
    get_args_for_roll_outs's defaults so results stay comparable.
    """
    mutators = mutators or PARALLEL_ZEROSHOT_MUTATORS
    optimizer_llms = optimizer_llms or get_default_optimizer_llms(task_name)
    inference_model_names = (
        inference_model_names or get_default_inference_model_names(task_name)
    )
    num_seeds = PARALLEL_ZEROSHOT_NUM_SEEDS[task_name]

    pruning_strategy = 'lowest_scoring'
    max_population_size = 15
    sampling_strategy = 'highest_scoring'

    device_profile = get_device_profile(task_name)

    benchmarks = ['drop', 'gsm8k'] if task_name == 'prompt' else ['drop']

    experiments_to_run = []

    for (
        benchmark,
        optimizer_llm,
        inference_model_name,
        mutator,
        seed,
    ) in itertools.product(
        benchmarks,
        optimizer_llms,
        inference_model_names,
        mutators,
        range(num_seeds),
    ):
        n = get_kincontext_n(task_name) if mutator == 'kincontext' else 5
        experiments_to_run.append(
            {
                'optimizer_name': 'opro',
                'optimizer_llm': optimizer_llm,
                'task_name': task_name,
                'pruning_strategy': pruning_strategy,
                'max_population_size': max_population_size,
                'sampling_strategy_name': sampling_strategy,
                'mutator': mutator,
                'noise': 0,
                'num_iter': 1,
                'benchmark': benchmark,
                'n': n,
                'seed': seed,
                'inference_model_name': inference_model_name,
                **device_profile,
            },
        )

    return experiments_to_run


# population_dynamics's curated-population directories use these bookkeeping
# names (see the iteration-budget curation notebook) rather than main.py's
# raw --task_name choices, since 'prompt' splits into 'prompt_drop'/
# 'prompt_gsm8k' directories but is a single main.py task_name distinguished
# by --benchmark instead. Any task_name not listed here (i.e. every existing
# caller) falls back to today's behavior via the .get(..., (task_name,
# 'drop')) default below: benchmark='drop', task_name passed through as-is.
POP_DYNAMICS_TASK_MAP: dict[str, tuple[str, str]] = {
    'prompt_drop': ('prompt', 'drop'),
    'prompt_gsm8k': ('prompt', 'gsm8k'),
}


def _sanitize_model_dirname(model_name: str) -> str:
    """Map a model name (e.g. an optimizer_llm) to its populations/ dirname.

    populations/<task_name>/<seed_mutator>/<model>/ names its <model> level
    by stripping '.' and '/' from the model name, e.g. 'gemini-3.5-flash' ->
    'gemini-35-flash', 'meta-llama/Llama-3.1-8B-Instruct' ->
    'meta-llamaLlama-31-8B-Instruct'.
    """
    return model_name.replace('.', '').replace('/', '')


def get_args_for_pop_dynamics(
    population_dir: str,
    task_name: str,
    num_iter: int,
) -> list[dict[str, Any]]:
    """Experiments to understand population dynamics.

    Runs the same 12-config OPRO sub-sweep (3 sampling_strategies x 4
    mutators, noise=0) per (optimizer_llm, inference_model_name) pair from
    get_default_optimizer_llms/get_default_inference_model_names -- see
    get_args_for_roll_outs) against each curated initial population file, so
    population_dynamics results are directly comparable to general_rollout's:
    the only thing that varies is where OPRO starts from.

    Directory discovery, in order: `<population_dir>/<task_name>/budget_*/`
    or `<population_dir>/<task_name>/` or `population_dir` itself, all shared
    across every optimizer_llm; or, for the mutator/model-nested layout
    (`<population_dir>/<task_name>/<seed_mutator>/<model>/budget_*/`, e.g.
    populations/), each optimizer_llm is paired ONLY with populations whose
    <model> dir matches its sanitized name (`_sanitize_model_dirname`) --
    a mismatch prints a warning instead of silently yielding 0 experiments.
    """
    pruning_strategy = 'lowest_scoring'
    max_population_size = 15
    sampling_strategies = ['highest_scoring', 'tournament', 'wheel']
    mutators = ['kincontext', 'DE', 'GA', 'GEPA']
    noise = 0
    sampling_prob = 0.5

    real_task_name, benchmark = POP_DYNAMICS_TASK_MAP.get(
        task_name,
        (task_name, 'drop'),
    )
    optimizer_llms = get_default_optimizer_llms(real_task_name)
    inference_model_names = get_default_inference_model_names(real_task_name)
    device_profile = get_device_profile(real_task_name)

    def _json_files(dirs: list[str]) -> list[str]:
        return sorted(
            os.path.join(d, f)
            for d in dirs
            for f in os.listdir(d)
            if f.endswith('.json')
        )

    task_dir = os.path.join(population_dir, task_name)
    top_level_budget_dirs = sorted(
        glob.glob(os.path.join(task_dir, 'budget_*/')),
    )
    nested_budget_dirs = sorted(
        glob.glob(os.path.join(task_dir, '*', '*', 'budget_*/')),
    )

    # Non-None only for the mutator/model-nested layout, in which case
    # shared_init_population_files is unused (each optimizer_llm gets its
    # own file list instead of one list shared by all of them).
    files_by_optimizer_llm: dict[str, list[str]] | None = None
    shared_init_population_files: list[str] = []
    if top_level_budget_dirs:
        shared_init_population_files = _json_files(top_level_budget_dirs)
    elif nested_budget_dirs:
        files_by_optimizer_llm = {}
        for optimizer_llm in optimizer_llms:
            model_dirname = _sanitize_model_dirname(optimizer_llm)
            model_budget_dirs = sorted(
                glob.glob(
                    os.path.join(task_dir, '*', model_dirname, 'budget_*/'),
                ),
            )
            if not model_budget_dirs:
                print(
                    f'get_args_for_pop_dynamics: no population dir matching '
                    f'optimizer_llm={optimizer_llm!r} (looked for dirname '
                    f'{model_dirname!r}) under {task_dir!r} -- skipping '
                    f'this optimizer_llm for task_name={task_name!r}',
                )
            files_by_optimizer_llm[optimizer_llm] = _json_files(
                model_budget_dirs,
            )
    elif os.path.isdir(task_dir):
        shared_init_population_files = _json_files([task_dir])
    else:
        shared_init_population_files = _json_files([population_dir])

    max_tokens = TASK_TOKEN_BUDGETS.get(real_task_name)

    experiments_to_run = []

    for (
        strategy,
        mutator,
        optimizer_llm,
        inference_model_name,
    ) in itertools.product(
        sampling_strategies,
        mutators,
        optimizer_llms,
        inference_model_names,
    ):
        n = get_kincontext_n(real_task_name) if mutator == 'kincontext' else 5
        pop_paths = (
            files_by_optimizer_llm[optimizer_llm]
            if files_by_optimizer_llm is not None
            else shared_init_population_files
        )

        for pop_path in pop_paths:
            config = {
                'optimizer_name': 'opro',
                'optimizer_llm': optimizer_llm,
                'task_name': real_task_name,
                'pruning_strategy': pruning_strategy,
                'max_population_size': max_population_size,
                'sampling_strategy_name': strategy,
                'mutator': mutator,
                'noise': noise,
                'num_iter': num_iter,
                'benchmark': benchmark,
                'sampling_prob': sampling_prob,
                'n': n,
                'init_population_path': pop_path,
                'inference_model_name': inference_model_name,
                # exact-dedup (see opro.py's sim_thresh == -1)
                'sim_thresh': -1,
                **device_profile,
            }
            if max_tokens is not None:
                config['max_tokens'] = max_tokens
            experiments_to_run.append(config)

    return experiments_to_run
