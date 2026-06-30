"""Submitting experiment in parallel to ray cluster."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import ray

# project_root = os.path.abspath("../")
# project_root = "/scratch/mansisak/llm_optimizer"
script_dir = pathlib.Path(__file__).parent.resolve()
project_root = str(script_dir.parent)
print(f'Targeting Shared Project Directory: {project_root}')
# Extract the exact path to your existing prebuilt .venv Python interpreter
# E.g., "<path_to>/llm_optimizer/.venv/bin/python"
prebuilt_python_exe = os.path.abspath(sys.executable)

runtime_env = {
    'working_dir': project_root,
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
        'VIRTUAL_ENV': '',
    },
}


if not ray.is_initialized():
    try:
        # Ray automatically reads the RAY_ADDRESS="auto" environment variable
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


@ray.remote
def run_experiment(python_path: str, task_name: str, true_root: str) -> str:
    """Print out allocated GPU ids for this specific task worker."""
    # Command array updated exactly to: uv run python main.py
    worker_project_root = os.getcwd()
    print(f'{worker_project_root=}')
    absolute_main_path = os.path.join(true_root, 'llm_optimizer/main.py')
    print(f'{absolute_main_path=}')
    cmd = [
        python_path,
        absolute_main_path,
        #'llm_optimizer/main.py',
        '--num_iter',
        '5',
        '--task_name',
        task_name,
    ]

    print(f'Executing: {" ".join(cmd)}')
    try:
        result = subprocess.run(
            cmd,
            cwd=true_root,
            capture_output=False,
            check=True,
        )
        print(result)
        return 'Task Completed Successfully'

    except subprocess.CalledProcessError as e:
        print(f'ERROR: main.py crashed with exit code {e.returncode}')
        raise e


total_resources = ray.cluster_resources()
print('--- Total Cluster Resources ---')
print(json.dumps(total_resources, indent=4))

free_resources = ray.available_resources()
print('\n--- Available (Free) Resources ---')
print(json.dumps(free_resources, indent=4))

print('\n--- Launching Experiments ---')

# Define the experiments to run along with their resource requirements
experiments = [
    {'task_name': 'tweet', 'num_gpus': 1},
    {'task_name': 'prompt', 'num_gpus': 3},
]

# Launch loop: Trigger all tasks asynchronously and gather their futures
futures = []
for exp in experiments:
    obj_ref = run_experiment.options(num_gpus=exp['num_gpus']).remote(
        prebuilt_python_exe,
        exp['task_name'],
        project_root,
    )
    futures.append(obj_ref)

print('\n--- Worker Return Results ---')

# Wait loop: Iterate through the futures list and block on them one by one
for obj_ref in futures:
    res = ray.get(obj_ref)
    print(res)
