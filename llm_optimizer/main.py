"""Optimization loop with an llm to search for an optimal task solution."""

from __future__ import annotations

import os
import typing

from openai import OpenAI

# Dict of solution reward pairs
solution_bank: dict[typing.Any, typing.Any] = {}

# Evaluator

# Prompt fed to LLM optimizer
tweet = """@CNN: 'House averts government shutdown'
@user: 'I wish people cheered when I do my job'"""
task_description = f"""The below thread was posted on Twitter.
{tweet}\n Craft a response tweet."""
direction = 'maximize'
metric = 'engagement'
task_prompt = f"""{task_description} Your goal is to {direction} {metric}.
Output only the bare minimum text to reach the objective goal."""
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

num_iter = 1
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
                'content': task_prompt,
            },
        ],
    )

    # print(response.choices[0].message)

    # Assuming 'response' is your completed OpenAI API call
    raw_output = response.choices[0].message.content
    print(raw_output)
