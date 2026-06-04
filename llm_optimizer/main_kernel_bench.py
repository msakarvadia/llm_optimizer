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


# Evaluator
# NOTE(MS): rn using a fake "toxicity" score as a proxy for engagement
# am aware its problematic...just for quick prototyping
def evaluator(solution: str) -> pd.DataFrame:
    """Evaluate LLM optimized solution."""
    # get current dir
    current_dir = os.getcwd()

    # move into Kernel bench
    os.chdir("/scratch/mansisak/KernelBench")

    # evaluate the current solution vs. baseline!!
    command = [
    "uv",
    "run",
    "python",
    "scripts/run_and_check.py", 
    "ref_origin=local",
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
task_description = 'You write custom CUDA operators to replace the pytorch operators in the given architecture to get speedups.\n\nYou have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom CUDA operators and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n\nHere\'s an example to show you the syntax of inline embedding custom CUDA operators in PyTorch:\n\nExample:\n\nInput architecture:\n\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n\nclass Model(nn.Module):\n    def __init__(self) -> None:\n        super().__init__()\n\n    def forward(self, a, b):\n        return a + b\n\n\ndef get_inputs():\n    # randomly generate input tensors based on the model architecture\n    a = torch.randn(1, 128).cuda()\n    b = torch.randn(1, 128).cuda()\n    return [a, b]\n\n\ndef get_init_inputs():\n    # randomly generate tensors required for initialization based on the model architecture\n    return []\n\n\nOptimized with CUDA operators:\n\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom torch.utils.cpp_extension import load_inline\n\n# Define the custom CUDA kernel for element-wise addition\nelementwise_add_source = """\n#include <torch/extension.h>\n#include <cuda_runtime.h>\n\n__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {\n    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n    if (idx < size) {\n        out[idx] = a[idx] + b[idx];\n    }\n}\n\ntorch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {\n    auto size = a.numel();\n    auto out = torch::zeros_like(a);\n\n    const int block_size = 256;\n    const int num_blocks = (size + block_size - 1) / block_size;\n\n    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);\n\n    return out;\n}\n"""\n\nelementwise_add_cpp_source = (\n    "torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);"\n)\n\n# Compile the inline CUDA code for element-wise addition\nelementwise_add = load_inline(\n    name="elementwise_add",\n    cpp_sources=elementwise_add_cpp_source,\n    cuda_sources=elementwise_add_source,\n    functions=["elementwise_add_cuda"],\n    verbose=True,\n    extra_cflags=[""],\n    extra_ldflags=[""],\n)\n\n\nclass ModelNew(nn.Module):\n    def __init__(self) -> None:\n        super().__init__()\n        self.elementwise_add = elementwise_add\n\n    def forward(self, a, b):\n        return self.elementwise_add.elementwise_add_cuda(a, b)\n\nYou are given the following architecture:\n\n\nimport torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n    """\n    Simple model that performs a single square matrix multiplication (C = A * B)\n    """\n    def __init__(self):\n        super(Model, self).__init__()\n    \n    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:\n        """\n        Performs the matrix multiplication.\n\n        Args:\n            A (torch.Tensor): Input matrix A of shape (N, N).\n            B (torch.Tensor): Input matrix B of shape (N, N).\n\n        Returns:\n            torch.Tensor: Output matrix C of shape (N, N).\n        """\n        return torch.matmul(A, B)\n\nN = 2048 * 2\n\ndef get_inputs():\n    A = torch.rand(N, N)\n    B = torch.rand(N, N)\n    return [A, B]\n\ndef get_init_inputs():\n    return []  # No special initialization inputs needed\n\nNote: The kernels should be optimized for FP32 (32-bit floating point) precision.\n\nOptimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! The output of the model will be stored in generated_arch.py.\n'
direction = 'maximize'
metric = 'speedup'
task_prompt = f"""{task_description} Your goal is to preserve correctness and {direction} {metric}."""

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
num_iter = 5
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

        model_prompt += (
            task_prompt
            + f'\nHere are some past examples and the {metric} score they '
            f'received where the goal is to {direction} the metric:'
            f'\n{example_str}\n'
            f'Generate a solution that has as high a score as possible.'
        )

        print(model_prompt)
