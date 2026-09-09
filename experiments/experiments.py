"""Submitting experiment in parallel to ray cluster."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import subprocess
import sys
from typing import Any

import ray
import yaml
from generate_experiment_args import get_args_for_parallel_zeroshot
from generate_experiment_args import get_args_for_pop_dynamics
from generate_experiment_args import get_args_for_roll_outs
from generate_experiment_args import run_random_number_bias_experiment

from llm_optimizer.utils import get_shared_vllm_requirements
from llm_optimizer.utils import VLLMServerActor

# Starting port for per-model shared vllm servers. Distinct from the
# fixed 8700/8701/8702/8703 ports main.py uses for its own local
# (non-shared) servers -- here N different models can be scheduled onto
# the same Ray node at once, so ports are allocated per unique model
# rather than per role.
SHARED_VLLM_BASE_PORT = 8800

script_dir = pathlib.Path(__file__).parent.resolve()
project_root = str(script_dir.parent)
print(f'Targeting Shared Project Directory: {project_root=}')
# Extract the exact path to your existing prebuilt .venv Python interpreter
# E.g., "<path_to>/llm_optimizer/.venv/bin/python"
prebuilt_python_exe = os.path.abspath(sys.executable)
print(f'{prebuilt_python_exe=}')

runtime_env = {
    #'working_dir': project_root,
    'py_executable': prebuilt_python_exe,
    'excludes': [
        'pyproject.toml',
        'uv.lock',
        'requirements.txt',
        '.venv',
        '.git',
    ],
    'env_vars': {
        # Ray to bypass internal uv environment generation hook entirely
        'RAY_ENABLE_UV_RUN_RUNTIME_ENV': '0',
        'VIRTUAL_ENV': '/scratch/mansisak/llm_optimizer/.venv',
        'PYTHONPATH': project_root,
    },
}


@ray.remote
def run_experiment(
    # python_path: str,
    true_root: str,
    args_dict: dict[str, Any],
) -> str:
    """Constructs the CLI arguments dynamically.

    Runs experiments.
    """
    python_path = sys.executable
    worker_project_root = os.getcwd()
    print(f'{worker_project_root=}')
    absolute_main_path = os.path.join(true_root, 'llm_optimizer/main.py')
    print(f'{absolute_main_path=}')

    # Initialize the base command array
    # NOTE(MS): -u forces unbuffered stdout so main.py's print()
    # calls show up promptly in the Ray logs instead of sitting in
    # a stdio buffer (stdout isn't a TTY here, so it defaults to
    # fully block-buffered instead of line-buffered).
    cmd = [
        python_path,
        '-u',
        absolute_main_path,
    ]

    args_dict.pop('num_gpus', None)
    args_dict.pop('num_cpus', None)
    # Dynamically unpack all dictionary keys
    # and values into CLI argument strings
    for key, val in args_dict.items():
        cmd.append(f'--{key}')
        cmd.append(str(val))

    print(f'Executing: {" ".join(cmd)}')
    try:
        result = subprocess.run(
            cmd,
            cwd=true_root,
            capture_output=False,
            check=True,
        )
        print(result)
        return (
            f'Task Completed Successfully: {args_dict.get("task_name")} | '
            f'{args_dict.get("mutator")}'
        )

    except subprocess.CalledProcessError as e:
        print(
            f'ERROR: main.py crashed on config '
            f'{args_dict} with exit code {e.returncode}',
        )
        print(f'Command executed: {e.cmd}')
        # NOTE(MS): capture_output=False above means the child's
        # stdout/stderr are inherited (streamed live into the Ray
        # logs), so e.stdout/e.stderr are always None here -- the
        # actual error trace is already visible inline above.
        raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='general_rollout',
        choices=[
            'general_rollout',
            'population_dynamics',
            'perturb',
            'random_number_bias',
            'parallel_zeroshot',
        ],
        help='Name of experiment.',
    )
    parser.add_argument(
        '--task_name',
        type=str,
        nargs='+',
        default=['tweet'],
        help=(
            'Task(s) to run. Accepts multiple values -- all task_names '
            'are batched into one Ray cluster / experiments list, so '
            'shared vLLM servers get deduped across tasks too (e.g. '
            'harmbench and prompt sharing the same inference model).'
        ),
    )
    parser.add_argument(
        '--population_dir',
        type=str,
        default=None,
        help=(
            'Population dir for population-init experiments '
            '(population_dynamics/perturb); '
            'defaults per experiment_name if unset.'
        ),
    )
    parser.add_argument(
        '--num_iter',
        type=int,
        default=50,
        help='Iterations per run.',
    )
    parser.add_argument(
        '--optimizer_llm',
        type=str,
        nargs='+',
        default=None,
        help=(
            'general_rollout only: optimizer LLM(s) to sweep. If unset, '
            'defaults per task_name (harmbench -> abliterated model, '
            'prompt -> gemini-3.5-flash, else -> gemini-3.1-pro-preview).'
        ),
    )
    parser.add_argument(
        '--inference_model_name',
        type=str,
        nargs='+',
        default=None,
        help=(
            'general_rollout/harmbench+prompt only: the target/inference '
            'model(s). If unset, defaults per task_name (see '
            'get_default_inference_model_names): harmbench -> OLMo-2-DPO, '
            'prompt -> Llama-3.2-1B-Instruct, else -> gemma-4-E4B-it.'
        ),
    )
    args = parser.parse_args()

    if args.experiment_name == 'random_number_bias':
        # Standalone probe: direct LLM calls, no Task/optimizer/main.py,
        # so it skips the Ray cluster + subprocess launch path entirely.
        run_random_number_bias_experiment()
        sys.exit(0)

    if not ray.is_initialized():
        try:
            # Ray automatically reads the RAY_ADDRESS="auto"
            ray.init(runtime_env=runtime_env)
            print('Connected to multi-node Ray Cluster successfully.')
        except Exception as e:
            print(
                f'Failed to connect to cluster: {e}.',
                'Falling back to fresh local instance.',
            )
            if 'RAY_ADDRESS' in os.environ:
                del os.environ['RAY_ADDRESS']
            ray.init(
                address='local',
                runtime_env=runtime_env,
                object_store_memory=2 * 1024 * 1024 * 1024,
            )

    total_resources = ray.cluster_resources()
    print('--- Total Cluster Resources ---')
    print(json.dumps(total_resources, indent=4))

    free_resources = ray.available_resources()
    print('\n--- Available (Free) Resources ---')
    print(json.dumps(free_resources, indent=4))

    print('\n--- Launching Experiments ---')

    experiments: list[dict[str, Any]] = []
    for task_name in args.task_name:
        if args.experiment_name == 'general_rollout':
            experiments.extend(
                get_args_for_roll_outs(
                    task_name=task_name,
                    num_iter=args.num_iter,
                    inference_model_names=args.inference_model_name,
                    optimizer_llms=args.optimizer_llm,
                ),
            )
        elif args.experiment_name == 'population_dynamics':
            population_dir = args.population_dir or (
                '/scratch/mansisak/llm_optimizer/curated_initial_populations_v2'
            )
            experiments.extend(
                get_args_for_pop_dynamics(
                    population_dir,
                    task_name=task_name,
                    num_iter=args.num_iter,
                ),
            )
        elif args.experiment_name == 'perturb':
            population_dir = args.population_dir or (
                '/scratch/mansisak/llm_optimizer/curated_perturbation_populations'
            )
            experiments.extend(
                get_args_for_pop_dynamics(
                    population_dir,
                    task_name=task_name,
                    num_iter=args.num_iter,
                ),
            )
        elif args.experiment_name == 'parallel_zeroshot':
            experiments.extend(
                get_args_for_parallel_zeroshot(
                    task_name=task_name,
                    optimizer_llms=args.optimizer_llm,
                ),
            )
    print(f'{len(experiments)=}')

    # Shuffle submission order so task_names interleave (Ray otherwise
    # schedules in submission order, letting one task_name monopolize slots).
    random.Random(42).shuffle(experiments)
    print(
        'shuffled submission order, first 10 task_names: '
        f'{[exp["task_name"] for exp in experiments[:10]]}',
    )

    # Define the experiments to run along with their resource requirements
    # experiments = [
    #    # {'task_name': 'tweet', 'num_gpus': 1, 'num_cpus':16},
    #    # {'task_name': 'prompt', 'num_gpus': 3, 'num_cpus':16},
    #    {
    #        'task_name': 'harmbench',
    #        'num_gpus': 2,
    #        'num_cpus': 16,
    #        'optimizer_llm': 'mlabonne/NeuralDaredevil-8B-abliterated',
    #    },
    # ]

    print('\n--- Resolving Shared vLLM Servers ---')

    with open(
        os.path.join(project_root, 'config.yaml'),
        encoding='utf-8',
    ) as file:
        vllm_config = yaml.safe_load(file)

    # (model_name, override_key) pairs needed per experiment, computed up
    # front so both the dedup below and the per-experiment injection loop
    # reuse the same result instead of recomputing it twice.
    exp_requirements = [
        get_shared_vllm_requirements(exp, vllm_config) for exp in experiments
    ]

    # Dedup by model_name across the *whole* batch, not per experiment --
    # that cross-experiment sharing is the entire point (e.g. the same
    # optimizer model launching ~80 times in the harmbench sweep collapses
    # to one shared actor here).
    unique_models = sorted(
        {model_name for reqs in exp_requirements for model_name, _ in reqs},
    )
    print(f'{unique_models=}')

    # Get-or-create one VLLMServerActor per unique model, each on its own
    # port -- N different models can land on the same Ray node at once,
    # so ports must be allocated per model rather than reusing main.py's
    # fixed per-role ports.
    actors = {
        # known Ray/mypy limitation, not a real attribute-defined bug.
        model_name: VLLMServerActor.remote(  # type: ignore[attr-defined]
            model_name,
            SHARED_VLLM_BASE_PORT + i,
            gpu_id=0,
        )
        for i, model_name in enumerate(unique_models)
    }

    # Blocks until each actor's underlying vllm server is healthy, since
    # get_base_url is queued behind __init__'s health-check loop.
    base_urls = {
        model_name: ray.get(actor.get_base_url.remote())
        for model_name, actor in actors.items()
    }
    print(f'{base_urls=}')

    for exp, reqs in zip(experiments, exp_requirements, strict=True):
        for model_name, override_key in reqs:
            exp[override_key] = base_urls[model_name]

    try:
        # Launch loop: Trigger all tasks asynchronously and gather their
        # futures
        futures = []
        for exp in experiments:
            # add experiment_name as the meta_dir
            # to store the experiments in
            exp['experiment_dir'] = args.experiment_name
            obj_ref = run_experiment.options(
                num_gpus=exp['num_gpus'],
                num_cpus=exp['num_cpus'],
            ).remote(
                # prebuilt_python_exe,
                project_root,
                exp,
            )
            futures.append(obj_ref)

        print('\n--- Worker Return Results ---')

        # Wait loop: iterate through the futures list and block on them
        # one by one
        for obj_ref in futures:
            try:
                res = ray.get(obj_ref)
                print(res)
            except Exception as e:
                # Prevent the script from crashing; log the specific
                # failure and move to the next task
                print(f'Experiment failed with error: {e}')
    finally:
        print('\n--- Shutting Down Shared vLLM Servers ---')
        for model_name, actor in actors.items():
            try:
                ray.get(actor.shutdown.remote())
            except Exception as e:
                print(
                    f'Error shutting down shared server for {model_name}: {e}',
                )
            ray.kill(actor)
