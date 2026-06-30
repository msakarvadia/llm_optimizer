"""Submitting experiment in parallel to ray cluster."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import ray

project_root = os.path.abspath('../')
# Extract the exact path to your existing prebuilt .venv Python interpreter
# E.g., "<path_to>/llm_optimizer/.venv/bin/python"
prebuilt_python_exe = sys.executable

runtime_env = {
    'working_dir': project_root,
    'excludes': [
        'pyproject.toml',
        'uv.lock',
        'requirements.txt',
        '.venv',
        '.git',
    ],
}

if not ray.is_initialized():
    # If running inside a Slurm allocation where a cluster was already started,
    # address="auto" hooks into it. If no cluster is found,
    # it falls back to a clean local instance.
    try:
        print('Connected to EXISTING Ray Cluster successfully.')
        ray.init(address='auto', runtime_env=runtime_env)
    except ConnectionError:
        print('No active cluster found. Started a fresh LOCAL Ray instance.')
        ray.init(runtime_env=runtime_env)


@ray.remote
def run_experiment(python_path: str, task_name: str) -> str:
    """Print out allocated GPU ids for this specific task worker."""
    # os.chdir('../')
    # Command array updated exactly to: uv run python main.py
    cmd = [
        python_path,
        'llm_optimizer/main.py',
        '--num_iter',
        '5',
        '--task_name',
        task_name,
    ]

    print(f'Executing: {" ".join(cmd)}')
    try:
        result = subprocess.run(
            cmd,
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

print('\n--- Launching Fake Experiments ---')

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
    )
    futures.append(obj_ref)

print('\n--- Worker Return Results ---')

# Wait loop: Iterate through the futures list and block on them one by one
for obj_ref in futures:
    res = ray.get(obj_ref)
    print(res)
