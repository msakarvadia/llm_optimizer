"""Instantiate a single kernel bench task."""

from __future__ import annotations

import ast
import re
import subprocess
from typing import Any

from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.prompt_constructor_toml import get_prompt_for_backend

from llm_optimizer.tasks.base_task import Task


def extract_first_code(
    output_string: str,
    code_language_types: list[str],
) -> str | None:
    """Extract first code block from model output.

    Specified by code_language_type.
    """
    if output_string is None:
        return None

    trimmed = output_string.strip()

    # print(' trimmed vv' * 40)
    # print(trimmed)
    # print(' trimmed ^^' * 40)

    # Extracting the first occurrence of content between backticks
    code_match = re.search(r'```(.*?)```', trimmed, re.DOTALL)

    # print(' code match vv' * 40)
    # print(code_match)
    # print(' code match ^^' * 40)

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

    # Fallback: If 'python' is expected, validate the raw string directly
    if 'python' in code_language_types:
        try:
            ast.parse(trimmed)  # Validates syntax without executing code
            # print(' DIRECT PYTHON MATCH**' * 40)
            return trimmed
        except SyntaxError:
            pass  # Not valid Python, handle below or return None

    return None


class KernelBench(Task):
    """KernelBench 'task'."""

    def __init__(
        self,
        metric: str = 'speedup',
        direction: str = 'maximize',
        **kwargs: Any,
    ) -> None:
        """Initialize task."""
        self.problem_id = kwargs['problem_id']
        self.level = kwargs['level']
        self.backend = kwargs['backend']  # cuda/triton/tilelang

        dataset = construct_kernelbench_dataset(
            level=self.level,
            source='huggingface',
            dataset_name='ScalingIntelligence/KernelBench',
        )

        problem = dataset.get_problem_by_id(self.problem_id)
        target_kernel_reference = problem.code
        # target_kernel_name = problem.name
        self.task_description = get_prompt_for_backend(
            ref_arch_src=target_kernel_reference,
            backend=self.backend,  # You can also try "triton" or "tilelang" or "cuda" here! # noqa
            option='one_shot',  # <--- show example of generation format via minimum example # noqa
            # include_hardware=True, # <--- Enable hardware specific context
            # gpu_name="T4"          # <--- Specify your GPU (matches keys in gpu_specs.py) # noqa
        )

        # To prevent hacking
        extra_instructions = (
            ' Do not use wrappers. '
            'Do not write dummy kernels to past staic checkers!!! '
            'Nothing like: __global__ void dummy_kernel_to_pass_static_checks()'  # noqa
            ' Write custom kernels.'
        )
        self.task_description += extra_instructions

        self.solution_description = 'kernel'
        self.metric = metric
        self.direction = direction
        # self.seed_candidate = example_add_model_generation
        # self.seed_candidate = "# follow the system prompt and evolve this into python code w/ custom kernel" # noqa
        self.seed_candidate = (
            'Write a CUDA kernel to replace '
            'the given PyTorch model for better performance. '
            'Include all imports. Output Python code with '
            ' ModelNew using load_inline inside a markdown code block.'
        )

    def evaluate(self, solution: str) -> tuple[float, dict[str, Any]]:
        """Evaluate LLM optimized solution."""
        print('SOLUTION vv  ' * 40)
        print(solution)
        print('SOLUTION ^^ ' * 40)
        custom_kernel = extract_first_code(solution, ['python', 'cpp'])
        print(' Custom Kernel vv' * 40)
        print(custom_kernel)
        print(' Custom Kernel ^^' * 40)
        if not custom_kernel:
            # custom_kernel = (
            #    example_add_model_generation  # there is no code yet
            # )
            speedup = -1.0
            error_dict = {
                'error': (
                    'This is a placeholder comment, '
                    'replace with python code w/ custom cuda kernel'
                    'and model definition and wrap that code inside'
                    ' a markdown python code block.'
                ),
            }
            return speedup, error_dict
        # print('EXTRACTED CODE ' * 40)
        with open('tmp_generated_kernel.py', 'w') as f:
            f.write(custom_kernel)
        command = [
            'uv',
            'run',
            'python',
            '../KernelBench/scripts/run_and_check.py',
            f'backend={self.backend}',
            'ref_origin=kernelbench',
            f'level={self.level}',
            f'problem_id={self.problem_id}',
            'kernel_src_path=./tmp_generated_kernel.py',
            'eval_mode=local',
        ]

        # Execute the command, capture standard output/error, and parse as text
        subprocess.run(
            'module load cuda/12.6',
            shell=True,
            executable='/bin/bash',
            check=False,
        )
        try:
            result_output = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
            )
            result: str = result_output.stdout
            # TODO(MS): Need to parse result to grab speedup and any errors
            match = re.search(r'Speedup over eager:\s*([\d.]+)', result)
            speedup = float(match.group(1)) if match else 0.0
            error_dict = {'meta_data': result}
        except subprocess.CalledProcessError as e:
            result = (
                f'\n{"=" * 50}\n'
                f"CRITICAL: 'run_and_check.py' failed with exit code\n"
                f'{e.returncode}\n'
                f'{"=" * 50}\n'
                # f"STANDARD OUTPUT (stdout):\n"
                # f"{e.stdout}\n\n"
                f'ERROR OUTPUT (stderr):\n'
                f'{e.stderr}\n'
                f'{"=" * 50}\n'
            )
            speedup = 0.0
            error_dict = {'error': e.stderr}

        print(result)

        return speedup, error_dict
