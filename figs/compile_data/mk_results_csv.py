# Read all results dir and compile results into single csv
from __future__ import annotations

import sys
from pathlib import Path

# Force flush ensures this appears instantly
current_dir = Path(__file__).resolve().parent
print(f'📍 Current Dir: {current_dir}', flush=True)

project_root = current_dir.parent.parent
sys.path.append(str(project_root) + '/experiments/')

print('📦 Starting heavy library imports...', flush=True)
import glob
import itertools
from typing import Any
import json
import pandas as pd

from generate_experiment_args import get_args_for_pop_dynamics
from generate_experiment_args import get_args_for_roll_outs

print('🎉 Finished all imports!', flush=True)

######################################################

experiment_dir = '../../general_rollout'
output_csv = 'compiled_results.csv'

# get all args
tasks = ['cantbelate', 'cloudcast', 'tsp', 'circlepacking', 'prompt']


desired_order = [
    'optimizer_name',
    'optimizer_llm',
    'mutator',
    'sampling_strategy_name',
    'sampling_prob',
    'n',
    'max_population_size',
    'pruning_strategy',
    'noise',
    'embed_model',
    'llm_judge',
    'sim_thresh',
    'task_name',
    'inference_model_name',
    'num_points',
    'benchmark',
]
all_rows = []
for task in tasks:
    args = get_args_for_roll_outs(task, 500)
    print(f"{task=}, {len(args)=}")
    total_files = 0
    for arg_set in args:
        clean_dict = {
            key: str(val).replace('.', '').replace('/', '')
            for key, val in arg_set.items()
        }
        # Clean values, but apply a specific rule for the 'noise' key
        # NOTE(MS): exact matches noise/prune_noise
        clean_dict = {
            key: (
                f'{str(val).replace(".", "").replace("/", "")}_00'
                if key == 'noise'
                else str(val).replace('.', '').replace('/', '')
            )
            for key, val in arg_set.items()
        }
        ordered_dict = {
            key: clean_dict[key] for key in desired_order if key in clean_dict
        }

        folder_pattern = '*'.join(ordered_dict.values())

        json_files = glob.glob(
            f'{experiment_dir}/*{folder_pattern}*/**/long*.json',
            recursive=True,
        )
        total_files += len(json_files)
        if len(json_files) > 1:
            print(arg_set)
            print(folder_pattern)
            for j in json_files:
                print(j)
            print('-----------')

        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    json_data = json.load(f)

                # 3. Create an independent row for each integer step
                if isinstance(json_data, dict):
                    for step, nested_dict in json_data.items():
                        if isinstance(nested_dict, dict):
                            # Combine base arguments + step info + step metrics
                            row_entry = {
                                **arg_set,
                                'step': int(step),  # Keep track of the step index
                                    **nested_dict,
                                'source_file': file_path
                            }
                            all_rows.append(row_entry)

            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    print(f"Total matched experimetns: ", total_files)


df = pd.DataFrame(all_rows)
print(f"Successfully processed {len(df)} rows.")
print(df.head())

# calculate total memory usage in bytes, sum it, and convert to GB
df_size_gb = df.memory_usage(deep=True).sum() / (1024 ** 3)
print(f"DataFrame size in memory: {df_size_gb:.4f} GB")


df.to_csv("results.csv", index=False)
