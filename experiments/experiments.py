"""Submitting experiment in parallel to ray cluster."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from typing import Any

import ray
from generate_experiment_args import get_args_for_pop_dynamics
from generate_experiment_args import get_args_for_roll_outs

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
    cmd = [
        python_path,
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
            'ERROR: main.py crashed on config ',
            '{args_dict} with exit code {e.returncode}',
        )
        print(f'Command executed: {e.cmd}')
        print(f'Stderr error trace:\n{e.stderr}')
        print(f'Exit code: {e.returncode}')
        print(f'Stdout logs:\n{e.stdout}')
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
        ],
        help='Name of experiment.',
    )
    args = parser.parse_args()

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
            # Clear env variable explicitly to ensure local fallback succeeds
            if 'RAY_ADDRESS' in os.environ:
                del os.environ['RAY_ADDRESS']
            ray.init(runtime_env=runtime_env)

    total_resources = ray.cluster_resources()
    print('--- Total Cluster Resources ---')
    print(json.dumps(total_resources, indent=4))

    free_resources = ray.available_resources()
    print('\n--- Available (Free) Resources ---')
    print(json.dumps(free_resources, indent=4))

    print('\n--- Launching Experiments ---')

    if args.experiment_name == 'general_rollout':
        experiments = get_args_for_roll_outs()
    if args.experiment_name == 'population_dynamics':
        experiments = get_args_for_pop_dynamics()
    print(f'{len(experiments)=}')

    # Define the experiments to run along with their resource requirements
    # experiments = [
    #    {'task_name': 'tweet', 'num_gpus': 1},
    #    #    {'task_name': 'prompt', 'num_gpus': 3},
    # ]

    # Launch loop: Trigger all tasks asynchronously and gather their futures
    futures = []
    for exp in experiments:
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

    # Wait loop: Iterate through the futures list and block on them one by one
    for obj_ref in futures:
        try:
            res = ray.get(obj_ref)
            print(res)
        except Exception as e:
            # Prevent the script from crashing; log the specific failure
            # and move to the next task
            print(f'Experiment failed with error: {e}')
