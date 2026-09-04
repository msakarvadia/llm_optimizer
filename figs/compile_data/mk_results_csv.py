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
import json
import os

import pandas as pd
from generate_experiment_args import get_args_for_pop_dynamics
from generate_experiment_args import get_args_for_roll_outs

print('🎉 Finished all imports!', flush=True)

######################################################

output_csv = 'compiled_results.csv'

# get all args

experiments = [
    {
        'experiment_dir': '../../population_dynamics',
        'experiment_arg_function': get_args_for_pop_dynamics,
        'task_list': [
            'cantbelate',
            'cloudcast',
            'tsp',
            'circlepacking',
            'prompt_gsm8k',
            'prompt_drop',
        ],
        'population_dir': '../../llm_optimizer/populations/',
    },
    {
        'experiment_dir': '../../general_rollout',
        'experiment_arg_function': get_args_for_roll_outs,
        'task_list': [
            'cantbelate',
            'cloudcast',
            'tsp',
            'circlepacking',
            'prompt',
        ],
        'population_dir': '..',
    },
]

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
for exp_dict in experiments:
    tasks = exp_dict['task_list']
    experiment_dir = exp_dict['experiment_dir']
    population_dir = exp_dict['population_dir']
    exp_arg_func = exp_dict['experiment_arg_function']
    for task in tasks:
        args = exp_arg_func(
            population_dir=population_dir,
            task_name=task,
            num_iter=500,
        )
        print(f'{task=}, {len(args)=}')
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
                key: clean_dict[key]
                for key in desired_order
                if key in clean_dict
            }

            folder_pattern = '*'.join(ordered_dict.values())

            population_pattern = '**'
            init_population_path = arg_set.get('init_population_path')
            if init_population_path:
                pop_path_parts = init_population_path.split(os.sep)
                if 'llm_optimizer' in pop_path_parts:
                    last_llm_optimizer_idx = len(
                        pop_path_parts,
                    ) - pop_path_parts[::-1].index('llm_optimizer')
                    pop_path_parts = pop_path_parts[last_llm_optimizer_idx:]
                pop_path_parts = [part for part in pop_path_parts if part]
                if pop_path_parts:
                    pop_path_parts[-1] = os.path.splitext(pop_path_parts[-1])[
                        0
                    ]
                population_pattern = '/'.join(pop_path_parts)
                arg_set['init_mutator'] = pop_path_parts[-4]
                arg_set['init_budget'] = pop_path_parts[-2]
                arg_set['init_scheme'] = pop_path_parts[-1]

            json_files = glob.glob(
                f'{experiment_dir}/*{folder_pattern}*/**{population_pattern}/long*.json',
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
                    with open(file_path) as f:
                        json_data = json.load(f)

                    # 3. Create an independent row for each integer step
                    if isinstance(json_data, dict):
                        # Find the lowest cumulative_tokens_spent across all steps in this file,
                        # to use as a fallback for steps missing that field
                        known_token_values = [
                            nested_dict['cumulative_tokens_spent']
                            for nested_dict in json_data.values()
                            if isinstance(nested_dict, dict)
                            and 'cumulative_tokens_spent' in nested_dict
                            and nested_dict['cumulative_tokens_spent']
                            is not None
                        ]
                        fallback_tokens = (
                            min(known_token_values)
                            if known_token_values
                            else None
                        )

                        for step, nested_dict in json_data.items():
                            if isinstance(nested_dict, dict):
                                # Backfill missing/None cumulative_tokens_spent with the fallback
                                if (
                                    nested_dict.get('cumulative_tokens_spent')
                                    is None
                                ):
                                    nested_dict = {
                                        **nested_dict,
                                        'cumulative_tokens_spent': fallback_tokens,
                                    }

                                # Combine base arguments + step info + step metrics
                                row_entry = {
                                    **arg_set,
                                    'step': int(
                                        step,
                                    ),  # Keep track of the step index
                                    **nested_dict,
                                    'source_file': file_path,
                                }
                                all_rows.append(row_entry)

                except Exception as e:
                    print(f'Error reading {file_path}: {e}')
        print('Total matched experimetns: ', total_files)


df = pd.DataFrame(all_rows)
print(f'Successfully processed {len(df)} rows.')
print(df.head())

# calculate total memory usage in bytes, sum it, and convert to GB
df_size_gb = df.memory_usage(deep=True).sum() / (1024**3)
print(f'DataFrame size in memory: {df_size_gb:.4f} GB')


df.to_csv('results.csv', index=False)
