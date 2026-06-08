"""Optimization loop with an llm to search for an optimal task solution."""

from __future__ import annotations

import os
import typing
import time
import pandas as pd
from detoxify import Detoxify
from openai import OpenAI

# Dict of solution reward pairs
solution_bank: list[tuple[typing.Any, typing.Any]] = []


# Evaluator
def evaluator(x:int|str)->int:
    try:
        x = float(x)
        coeff = 0.5
        x = x - 0.78
        vertical_shift = 2
        return x**4 - 3*x**2 + coeff*x + 2
    except Exception as error:
        return str(error)

# Prompt fed to LLM optimizer
x_val = 3
task_description = f"""The y value of a hidden 2d function is {evaluator(x_val)} at x={x_val} where the independant variable is x and the dependant variable is y. 
Predict x such that you minimize the function."""
task_description = f"""There is a hidden 2d function is where the independant variable is x and the dependant variable is y. 
Predict x such that you minimize the function. Don't commit too early to a point."""
direction = 'minimize'
metric = 'function value'
task_prompt = f"""{task_description} Your goal is to {direction} {metric}.
Output only the bare minimum text to reach the objective goal; only output a single number."""
print(f'{task_prompt=}')

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
num_iter = 50
timing_per_solution = []
tokens_per_solution = []
solutions = []
for _i in range(num_iter):
    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model='gemini-2.5-flash',
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
    end_time = time.perf_counter()
    # print(response.choices[0].message)

    # Assuming 'response' is your completed OpenAI API call
    raw_output = response.choices[0].message.content
    solutions.append(raw_output)
    timing_per_solution.append(end_time - start_time)
    tokens_per_solution.append(response.usage.completion_tokens)
    # print(raw_output)

    if raw_output is not None:
        evaluation_score = evaluator(raw_output)

        solution_bank.append((raw_output, evaluation_score))
    else:
        # Handle the error state appropriately for your pipeline
        raise ValueError(
            'Model returned None instead of a valid string.',
        )

    print(f"{len(set(solution_bank))=}, {len(solution_bank)=}")
    if len(solution_bank) > 0:
        example_str: str = ''
        for solution, score in solution_bank:
        #for solution, score in set(solution_bank):
            example_str += f'Example: {solution}, Score: {score} \n'

        example_str = "\n".join(dict.fromkeys(example_str.splitlines()))
        model_prompt  = (
            task_prompt
            + f'\nHere are some past examples and the {metric} score they '
            f'received where the goal is to {direction} the metric:'
            f'\n{example_str}\n'
            f'Generate a solution that has as high a score as possible.'
        )

        print("-"* 40)
        print(model_prompt)
        print("-"* 40)
print(f"{timing_per_solution=}")
print(f"{tokens_per_solution=}")
print(f"{solutions=}")
