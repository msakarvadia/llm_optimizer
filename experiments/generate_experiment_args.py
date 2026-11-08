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
    # prompt's lm-eval-harness GSM8K scoring is CPU-heavy per candidate
    # (unlike parallel_zeroshot's single-shot num_iter=1 calls); at the
    # default 0.25 share and long-running (num_iter=550) trajectories, the
    # resulting oversubscription (~4x the real core count) starves client
    # HTTP connections to the shared vLLM servers of CPU time long enough
    # to exhaust prompt_lm's retry budget, raising a fatal
    # openai.APIConnectionError partway through an otherwise-successful
    # trajectory. num_cpus=1 caps concurrency at the real core count.
    'prompt': {'num_gpus': 0, 'num_cpus': 1},
}
DEFAULT_DEVICE_PROFILE: dict[str, int | float] = {
    'num_gpus': 0,
    # Higher cpu share 0.1 hurts concurrency
    # was letting these OOM-kill the job's
    'num_cpus': 0.25,  # is typically 0.25
}

# Cumulative mutator+diversity-judge token budget per task_name -- respected
# by all four optimizers' optimize()'s max_tokens (opro, gepa, shinka_evolve,
# open_evolve; see main.py). Tasks not listed here get no budget, i.e.
# num_iter remains the sole stopping criterion.
TASK_TOKEN_BUDGETS: dict[str, int] = {
    'harmbench': 70_000,
    'prompt': 150_000,
    'cloudcast': 2_000_000,
    'cantbelate': 1_000_000,
    'tsp': 1_000_000,
    'circlepacking': 200_000,  # 70_000,
}

# OPRO-only diversity-check sweep (gepa/open_evolve ignore sim_thresh, see
# opro.py). 'llm_judge' omitted (not None) so experiments.py's str(val)
# CLI serialization doesn't emit a literal "None" main.py can't parse.
# NOTE(MS): diversity filtering / llm-judge sweep disabled for now for
# general_rollout -- only the no-filtering variant runs. Uncomment the
# rest to re-enable the full sweep.
DIVERSITY_CHECK_VARIANTS: list[dict[str, Any]] = [
    {'sim_thresh': 0.0, 'llm_judge': None},  # no filtering (current default)
    # {'sim_thresh': -1, 'llm_judge': None},  # exact-dedup baseline
    # {'sim_thresh': 0.95, 'llm_judge': None},  # fuzzy, no judge
    # {'sim_thresh': 0.95, 'llm_judge': 'gemini-3.5-flash'},  # fuzzy, judged
    # {'sim_thresh': 0.8, 'llm_judge': None},  # fuzzy, no judge
    # {'sim_thresh': 0.8, 'llm_judge': 'gemini-3.5-flash'},  # fuzzy, judged
]

# shinka_evolve's own fixed diversity-check config, always on.
SHINKA_DIVERSITY_CONFIG: dict[str, Any] = {
    'sim_thresh': 0.95,
    'llm_judge': 'gemini-3.5-flash',
}


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
            #'gemini-3.5-flash',
            #'gemini-3.7-flash',
            'meta-llama/Llama-3.1-8B-Instruct',
        ]
    if task_name == 'circlepacking':
        # oss-120b
        # weaker code model
        return [
            'Qwen3_6-35B-A3B',
            'gemini-2.5-flash',
            'gpt-oss-120b',
        ]  # I ran w/ gemini-3.7 (but fails for parallel)
    if task_name == 'tsp':
        return [
            #'gpt-oss-120b',
            'gemini-3.7-flash',
            #'gemini-3.5-flash',
        ]
    if task_name in ['cloudcast', 'cantbelate']:
        # deepseek
        return [
            'gemini-3.7-flash',
            'gemini-3.5-flash',
            'deepseek/deepseek-v4-flash',
        ]
        #'Kimi-K2.5',
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
    **kwargs: Any,
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
        for optimizer_name in ('gepa', 'open_evolve', 'shinka_evolve'):
            # shinka_evolve always runs with its judge on (sim_thresh=0.95,
            # matching its own internal default), independent of OPRO's
            # DIVERSITY_CHECK_VARIANTS sweep above.
            extra = (
                {
                    **SHINKA_DIVERSITY_CONFIG,
                    'embed_model': get_embed_model(task_name),
                }
                if optimizer_name == 'shinka_evolve'
                else {}
            )
            experiments_to_run.append(
                {
                    **base_config,
                    'optimizer_name': optimizer_name,
                    **extra,
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
    'prompt': 2200,
    'cloudcast': 550,
    'cantbelate': 335,
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
    benchmarks: list[str] | None = None,
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

    benchmarks = benchmarks or (
        ['drop', 'gsm8k'] if task_name == 'prompt' else ['drop']
    )

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

# Placeholder: per task_name, optimizer_llm + init_population_path (relative
# to population_dir, mirroring get_args_for_pop_dynamics) for
# gepa/open_evolve/shinka_evolve.
POP_DYNAMICS_PROD_GRADE_TRIPLES: dict[str, dict[str, str]] = {
    'cantbelate': {
        'optimizer_llm': 'gemini-3.7-flash',
        'init_population_path': 'cantbelate/kincontext/gemini-37-flash/budget_500000/greedy_dedup-none.json',  # noqa: E501
    },
    'cloudcast': {
        'optimizer_llm': 'gemini-3.5-flash',
        'init_population_path': 'cloudcast/kincontext/gemini-35-flash/budget_200000/greedy_dedup-none.json',  # noqa: E501
    },
    'tsp': {
        'optimizer_llm': 'gemini-3.5-flash',
        'init_population_path': 'tsp/kincontext/gemini-35-flash/budget_100000/greedy_dedup-none.json',  # noqa: E501
    },
    'circlepacking': {
        'optimizer_llm': 'gpt-oss-120b',
        'init_population_path': 'circlepacking/kincontext/gpt-oss-120b/budget_20000/greedy_dedup-none.json',  # noqa: E501
    },
    'prompt_gsm8k': {
        'optimizer_llm': 'meta-llama/Llama-3.1-8B-Instruct',
        'init_population_path': 'prompt_gsm8k/kincontext/meta-llamaLlama-31-8B-Instruct/budget_50000/greedy_dedup-none.json',  # noqa: E501
    },
}


def _sanitize_model_dirname(model_name: str) -> str:
    """Map a model name (e.g. an optimizer_llm) to its populations/ dirname.

    populations/<task_name>/<seed_mutator>/<model>/ names its <model> level
    by stripping '.' and '/' from the model name, e.g. 'gemini-3.5-flash' ->
    'gemini-35-flash', 'meta-llama/Llama-3.1-8B-Instruct' ->
    'meta-llamaLlama-31-8B-Instruct'.
    """
    return model_name.replace('.', '').replace('/', '')


def get_args_for_pop_dynamics(  # noqa: C901
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

    # TEMP: only run kincontext-mutator populations -- GEPA blocked for now.
    if files_by_optimizer_llm is not None:
        files_by_optimizer_llm = {
            k: [p for p in v if f'{os.sep}kincontext{os.sep}' in p]
            for k, v in files_by_optimizer_llm.items()
        }

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
                'sim_thresh': 0,
                #'sim_thresh': -1,
                **device_profile,
            }
            if max_tokens is not None:
                config['max_tokens'] = max_tokens
            experiments_to_run.append(config)

    return experiments_to_run


def get_args_for_pop_dynamics_prod_grade(
    population_dir: str,
    task_name: str,
    num_iter: int,
) -> list[dict[str, Any]]:
    """Hardcoded-pairing pop-dynamics configs for gepa/open_evolve/shinka.

    Looks up one (optimizer_llm, init_population_path) pair per task_name
    from POP_DYNAMICS_PROD_GRADE_TRIPLES; other args mirror
    get_args_for_pop_dynamics.
    """
    pruning_strategy = 'lowest_scoring'
    max_population_size = 15
    noise = 0

    triple = POP_DYNAMICS_PROD_GRADE_TRIPLES[task_name]
    optimizer_llm = triple['optimizer_llm']
    template_path = os.path.join(
        population_dir,
        triple['init_population_path'],
    )
    # Sweep every budget_* dir next to the template's, same greedy filename.
    init_population_paths = sorted(
        glob.glob(
            os.path.join(
                os.path.dirname(os.path.dirname(template_path)),
                'budget_*',
                os.path.basename(template_path),
            ),
        ),
    )

    real_task_name, benchmark = POP_DYNAMICS_TASK_MAP.get(
        task_name,
        (task_name, 'drop'),
    )
    inference_model_names = get_default_inference_model_names(real_task_name)
    device_profile = get_device_profile(real_task_name)
    max_tokens = TASK_TOKEN_BUDGETS.get(real_task_name)

    experiments_to_run = []
    for (
        init_population_path,
        inference_model_name,
        optimizer_name,
    ) in itertools.product(
        init_population_paths,
        inference_model_names,
        ('gepa', 'open_evolve', 'shinka_evolve'),
    ):
        extra = (
            {
                **SHINKA_DIVERSITY_CONFIG,
                'embed_model': get_embed_model(real_task_name),
            }
            if optimizer_name == 'shinka_evolve'
            else {}
        )
        config = {
            'optimizer_name': optimizer_name,
            'optimizer_llm': optimizer_llm,
            'task_name': real_task_name,
            'pruning_strategy': pruning_strategy,
            'max_population_size': max_population_size,
            'noise': noise,
            'num_iter': num_iter,
            'benchmark': benchmark,
            'init_population_path': init_population_path,
            'inference_model_name': inference_model_name,
            **device_profile,
            **extra,
        }
        if max_tokens is not None:
            config['max_tokens'] = max_tokens
        experiments_to_run.append(config)

    return experiments_to_run


# Seeds/population sizes/curation labels shared by the two
# *_vary_pop_size generators below -- both read curated pools written by
# llm_optimizer/curate_populations_vary_pop_size.py, laid out as
# <population_dir>/<task_key>/kincontext/<sanitized_optimizer_llm>/
# seed<seed>/<curation_label>_m<pop_size>.json (seed is 0-indexed, matching
# the harness's own --seed, not curate_populations.py's 1-seed-pool
# convention).
VARY_POP_SIZE_SEEDS: list[int] = [0, 1, 2]
VARY_POP_SIZES: list[int] = [15, 30, 45]
VARY_POP_SIZE_CURATION_LABELS: dict[str, str] = {
    'greedy': 'diversity_dedup-max_p0.98',
    'random': 'random_dedup-none',
}

# Batch 1 scope: prompt_gsm8k/Llama-3.1-8B-Instruct only.
MODULAR_VARY_POP_SIZE_TASK = 'prompt_gsm8k'
MODULAR_VARY_POP_SIZE_OPTIMIZER_LLM = 'meta-llama/Llama-3.1-8B-Instruct'

# Batch 2 scope: SOTA harnesses, both prompt_gsm8k and tsp, fixed m=15.
SOTA_VARY_SEED_TASKS: dict[str, str] = {
    'prompt_gsm8k': 'meta-llama/Llama-3.1-8B-Instruct',
    'tsp': 'gemini-3.5-flash',
}
SOTA_VARY_SEED_M = 15


def _vary_pop_size_population_path(  # noqa: PLR0913
    population_dir: str,
    task_key: str,
    optimizer_llm: str,
    seed: int,
    label: str,
    pop_size: int,
) -> str:
    """Path to one curated pool from curate_populations_vary_pop_size.py."""
    return os.path.join(
        population_dir,
        task_key,
        'kincontext',
        _sanitize_model_dirname(optimizer_llm),
        f'seed{seed}',
        f'{label}_m{pop_size}.json',
    )


def get_args_for_modular_vary_pop_size(
    population_dir: str,
    num_iter: int,
) -> list[dict[str, Any]]:
    """Batch 1: 12 Modular (OPRO) harnesses x 3 seeds x 3 pop sizes x 3 init.

    For prompt_gsm8k/Llama-3.1-8B-Instruct only. Per (sampling_strategy,
    mutator, seed, pop_size) -- 3 x 4 x 3 x 3 = 108 base configs -- emits
    three init-condition variants: uninitialized (no init_population_path,
    max_population_size=pop_size), greedy (diversity_dedup-max_p0.98,
    p=0.98, 25% budget), and random (random_dedup-none) -- both initialized
    variants seeded from the curated pool matching pop_size exactly, and
    max_population_size matched to that same pop_size. 108 x 3 = 324 total.
    """
    sampling_strategies = ['highest_scoring', 'tournament', 'wheel']
    mutators = ['kincontext', 'DE', 'GA', 'GEPA']

    task_key = MODULAR_VARY_POP_SIZE_TASK
    optimizer_llm = MODULAR_VARY_POP_SIZE_OPTIMIZER_LLM
    real_task_name, benchmark = POP_DYNAMICS_TASK_MAP.get(
        task_key,
        (task_key, 'drop'),
    )
    inference_model_name = get_default_inference_model_names(real_task_name)[0]
    device_profile = get_device_profile(real_task_name)
    max_tokens = TASK_TOKEN_BUDGETS.get(real_task_name)

    experiments_to_run = []
    for strategy, mutator, seed, pop_size in itertools.product(
        sampling_strategies,
        mutators,
        VARY_POP_SIZE_SEEDS,
        VARY_POP_SIZES,
    ):
        n = get_kincontext_n(real_task_name) if mutator == 'kincontext' else 5
        base_config: dict[str, Any] = {
            'optimizer_name': 'opro',
            'optimizer_llm': optimizer_llm,
            'task_name': real_task_name,
            'pruning_strategy': 'lowest_scoring',
            'sampling_strategy_name': strategy,
            'mutator': mutator,
            'noise': 0,
            'num_iter': num_iter,
            'benchmark': benchmark,
            'sampling_prob': 0.5,
            'n': n,
            'seed': seed,
            'inference_model_name': inference_model_name,
            'max_population_size': pop_size,
            'sim_thresh': 0,
            **device_profile,
        }
        if max_tokens is not None:
            base_config['max_tokens'] = max_tokens

        # 1) uninitialized -- no init_population_path, same pop_size cap.
        experiments_to_run.append(dict(base_config))

        # 2)+3) initialized (greedy, random), pool size matched to pop_size.
        for label in VARY_POP_SIZE_CURATION_LABELS.values():
            pop_path = _vary_pop_size_population_path(
                population_dir,
                task_key,
                optimizer_llm,
                seed,
                label,
                pop_size,
            )
            if not os.path.exists(pop_path):
                print(f'missing curated population, skipping: {pop_path}')
                continue
            experiments_to_run.append(
                {**base_config, 'init_population_path': pop_path},
            )

    return experiments_to_run


def get_args_for_sota_vary_seed(
    population_dir: str,
    num_iter: int,
) -> list[dict[str, Any]]:
    """Batch 2: OA/OE/SE harnesses x 2 tasks x 3 seeds x 3 init, m=15 fixed.

    Per (task, seed, optimizer_name) -- 2 x 3 x 3 = 18 base configs -- emits
    three init-condition variants (uninitialized, greedy
    diversity_dedup-max_p0.98, random random_dedup-none), all at the fixed
    m=15 curated pool size. 18 x 3 = 54 total.
    """
    experiments_to_run = []
    for task_key, optimizer_llm in SOTA_VARY_SEED_TASKS.items():
        real_task_name, benchmark = POP_DYNAMICS_TASK_MAP.get(
            task_key,
            (task_key, 'drop'),
        )
        inference_model_name = get_default_inference_model_names(
            real_task_name,
        )[0]
        device_profile = get_device_profile(real_task_name)
        max_tokens = TASK_TOKEN_BUDGETS.get(real_task_name)
        embed_model = get_embed_model(real_task_name)

        for seed in VARY_POP_SIZE_SEEDS:
            base_config: dict[str, Any] = {
                'optimizer_llm': optimizer_llm,
                'task_name': real_task_name,
                'pruning_strategy': 'lowest_scoring',
                'noise': 0,
                'num_iter': num_iter,
                'benchmark': benchmark,
                'seed': seed,
                'inference_model_name': inference_model_name,
                'max_population_size': SOTA_VARY_SEED_M,
                **device_profile,
            }
            if max_tokens is not None:
                base_config['max_tokens'] = max_tokens

            for optimizer_name in ('gepa', 'open_evolve', 'shinka_evolve'):
                extra = (
                    {**SHINKA_DIVERSITY_CONFIG, 'embed_model': embed_model}
                    if optimizer_name == 'shinka_evolve'
                    else {}
                )

                # 1) uninitialized
                experiments_to_run.append(
                    {**base_config, 'optimizer_name': optimizer_name, **extra},
                )

                # 2)+3) initialized (greedy, random), fixed m=15.
                for label in VARY_POP_SIZE_CURATION_LABELS.values():
                    pop_path = _vary_pop_size_population_path(
                        population_dir,
                        task_key,
                        optimizer_llm,
                        seed,
                        label,
                        SOTA_VARY_SEED_M,
                    )
                    if not os.path.exists(pop_path):
                        print(
                            f'missing curated population, skipping: '
                            f'{pop_path}',
                        )
                        continue
                    experiments_to_run.append(
                        {
                            **base_config,
                            'optimizer_name': optimizer_name,
                            **extra,
                            'init_population_path': pop_path,
                        },
                    )

    return experiments_to_run
