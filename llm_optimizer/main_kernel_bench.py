"""Optimization loop with an llm to search for an optimal task solution."""

from __future__ import annotations

import os
import re
import typing
import subprocess
import pandas as pd
from detoxify import Detoxify
from openai import OpenAI

# Dict of solution reward pairs
solution_bank: list[tuple[typing.Any, typing.Any]] = []

level = 1 # @param {type:"slider", min:1, max:3, step:1}
problem_id = 1 # @param {type:"slider", min:1, max:100, step:1}
backend = "cuda"

# Evaluator
# NOTE(MS): rn using a fake "toxicity" score as a proxy for engagement
# am aware its problematic...just for quick prototyping
def evaluator(solution: str) -> pd.DataFrame:
    """Evaluate LLM optimized solution."""
    # get current dir
    current_dir = os.getcwd()

    # move into Kernel bench
    #os.chdir("/scratch/mansisak/KernelBench")

    # evaluate the current solution vs. baseline!!
    command = [
    "uv",
    "run",
    "python",
    "../KernelBench/scripts/run_and_check.py", 
    f"backend={backend}",
    "ref_origin=kernelbench",
    f"level={level}",
    f"problem_id={problem_id}",
    "ref_arch_src_path=/scratch/mansisak/llm_optimizer/llm_optimizer/ref_arch.py", # TODO(MS): change path
    "kernel_src_path=/scratch/mansisak/llm_optimizer/llm_optimizer/generated_arch.py", # TODO(MS): change path
    "eval_mode=local"
]

    # Execute the command, capture standard output/error, and parse as text
    subprocess.run("module load cuda/12.6",shell=True, executable="/bin/bash")
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        result = result.stdout
    except subprocess.CalledProcessError as e:
        result = (
            f"\n{'='*50}\n"
            f"CRITICAL: 'run_and_check.py' failed with exit code {e.returncode}\n"
            f"{'='*50}\n"
            f"STANDARD OUTPUT (stdout):\n"
            f"{e.stdout}\n\n"
            f"ERROR OUTPUT (stderr):\n"
            f"{e.stderr}\n"
            f"{'='*50}\n"
        )

    # move back into the current dir
    os.chdir(current_dir)
    print(result)

    return result

def extract_first_code(output_string: str, code_language_types: list[str]) -> str:
    """
    Extract first code block from model output, specified by code_language_type
    """
    if output_string is None:
        return None
    
    trimmed = output_string.strip()

    # Extracting the first occurrence of content between backticks
    code_match = re.search(r"```(.*?)```", trimmed, re.DOTALL)

    if code_match:
        # Strip leading and trailing whitespace from the extracted code
        code = code_match.group(1).strip()

        # depends on code_language_type: cpp, python, etc.
        # sometimes the block of code is ```cpp ... ``` instead of ``` ... ```
        # in this case strip the cpp out
        for code_type in code_language_types:
            if code.startswith(code_type):
                code = code[len(code_type) :].strip()

        return code

    return None

# Prompt fed to LLM optimizer

from kernelbench.prompt_constructor_toml import get_prompt_for_backend

# Here is an example of a constructed context with hardware context
# This injects specs (memory bandwidth, cache size) from kernelbench/prompts/hardware/gpu_specs.py
# helping the LLM optimize tile sizes for your specific GPU.

from kernelbench.dataset import construct_kernelbench_dataset


# Unified interface - same code for huggingface and local!
dataset = construct_kernelbench_dataset(
    level=level,
    source="huggingface",
    dataset_name="ScalingIntelligence/KernelBench",
)

print(f"{problem_id=}")
problem = dataset.get_problem_by_id(problem_id)
target_kernel_reference = dataset.get_problem_by_id(problem_id).code
target_kernel_name = dataset.get_problem_by_id(problem_id).name

print(target_kernel_name)
print(target_kernel_reference)
print(f"A"*40)

task_description = get_prompt_for_backend(
    ref_arch_src=target_kernel_reference,
    backend=backend,   # You can also try "triton" or "tilelang" or "cuda" here!
    option="one_shot", # <--- show example of generation format via minimum example
    #include_hardware=True, # <--- Enable hardware specific context
    #gpu_name="T4"          # <--- Specify your GPU (matches keys in gpu_specs.py)
)
direction = 'maximize'
metric = 'speedup'
task_prompt = f"""{task_description} Your goal is to preserve correctness and {direction} {metric}. Do not use wrappers. Do not write dummy kernels to past staic checkers!!! Nothing like: __global__ void dummy_kernel_to_pass_static_checks(). Write custom kernels."""

print(f'{task_prompt}')

# get LLM

api_key = os.getenv('GEMINI_API_KEY')

if api_key is None:
    raise ValueError(
        'API key not found. Please set the MY_API_KEY environment variable.',
    )

client = OpenAI(
    api_key=api_key,
    base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
)

model_prompt = task_prompt
num_iter = 10
for _i in range(num_iter):
    response = client.chat.completions.create(
        model='gemini-3.5-flash',
        # TODO(MS): maybe pass in a pydantic model for the type of
        # response we want...particularly for code
        # response_format={"type": "json_object"},
        messages=[
            {
                'role': 'system',
                'content': 'You are a helpful assistant.',
            },
            {
                'role': 'user',
                'content': model_prompt,
            },
        ],
    )

    # print(response.choices[0].message)

    # Assuming 'response' is your completed OpenAI API call
    raw_output = response.choices[0].message.content
    #print(raw_output)
    custom_kernel = extract_first_code(raw_output, ["python", "cpp"])
    with open("generated_arch.py", "w") as f:
        f.write(custom_kernel)

    if custom_kernel is not None:
        evaluation_score = evaluator(custom_kernel)

        solution_bank.append((custom_kernel, evaluation_score))
    else:
        # Handle the error state appropriately for your pipeline
        raise ValueError(
            'Model returned None instead of a valid string.',
        )

    if len(solution_bank) > 0:
        example_str: str = ''
        for solution, score in solution_bank:
            example_str += f'Example: {solution}\n Score: {score}\n'

        model_prompt = (
            task_prompt
            + f'\nHere are some past examples and the {metric} score they '
            f'received where the goal is to {direction} the metric:'
            f'\n{example_str}\n'
            f'Generate a solution that has as high a score as possible.'
        )

        print(model_prompt)
