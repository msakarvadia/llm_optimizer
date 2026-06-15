"""Instantiate a single kernel bench task."""

from __future__ import annotations

import re
import subprocess
from typing import Any

from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.prompt_constructor_toml import get_prompt_for_backend
from kernelbench.utils import extract_first_code

from llm_optimizer.tasks.base_task import Task


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
        #self.seed_candidate = example_add_model_generation
        self.seed_candidate = "# follow the system problem and evolve this into python code w/ custom kernel" # noqa

    def evaluate(self, solution: str) -> tuple[float, dict[str, Any]]:
        """Evaluate LLM optimized solution."""
        custom_kernel = extract_first_code(solution, ['python', 'cpp'])
        if not custom_kernel:
            custom_kernel = (
                example_add_model_generation  # there is no code yet
            )
        # print(f"3"*90)
        # print(solution)
        # print(f"3"*90)
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
            speedup = float(match.group(1)) if match else 0
            error_dict = {'meta_data': result}
        except subprocess.CalledProcessError as e:
            result = (
                f'\n{"=" * 50}\n'
                f"CRITICAL: 'run_and_check.py' failed with exit code {e.returncode}\n"
                f'{"=" * 50}\n'
                # f"STANDARD OUTPUT (stdout):\n"
                # f"{e.stdout}\n\n"
                f'ERROR OUTPUT (stderr):\n'
                f'{e.stderr}\n'
                f'{"=" * 50}\n'
            )
            speedup = 0
            error_dict = {'error': e.stderr}

        print(result)

        return speedup, error_dict


example_add_model_generation = '''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# THIS IS A RANDOM CODE; replace for your specific problem
# Define the custom CUDA kernel for element-wise addition
elementwise_add_source = """
#include
#include

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_add_kernel<<>>(a.data_ptr(), b.data_ptr(), out.data_ptr(), size);

    return out;
}
"""

elementwise_add_cpp_source = (
    "torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for element-wise addition
elementwise_add = load_inline(
    name="elementwise_add",
    cpp_sources=elementwise_add_cpp_source,
    cuda_sources=elementwise_add_source,
    functions=["elementwise_add_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.elementwise_add = elementwise_add

    def forward(self, a, b):
        return self.elementwise_add.elementwise_add_cuda(a, b)
'''
