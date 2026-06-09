"""Logic to abstract away LLM-driven optimization."""

from __future__ import annotations

import os

from openai import OpenAI

from llm_optimizer.solution_bank import SolutionBank
from llm_optimizer.tasks.base_task import Task


class LLMOptimizer:
    """LLM optimizer.

    state include: task, solution_score pairs, stopping criteria
    """

    def __init__(self, task: Task, solution_bank: SolutionBank) -> None:
        """Init optimizer."""
        self.task = task
        self.solution_bank = solution_bank

        api_key = os.getenv('GEMINI_API_KEY')
        if api_key is None:
            raise ValueError(
                'API key not found. Set the MY_API_KEY environment variable.',
            )

        # TODO(MS): make generalizable to other base_urls
        self.client = OpenAI(
            api_key=api_key,
            base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
        )

    def get_meta_prompt(self) -> str:
        """Setup meta prompt for optimizer.."""
        task_prompt = (
            f'{self.task.task_description} '
            f'Your goal is to {self.task.direction} {self.task.metric}. '
            f'Output only the bare minimum text to reach the objective goal.'
        )

        meta_prompt = task_prompt
        solution_bank = self.solution_bank.get_solutions()
        if len(self.solution_bank) > 0:
            example_blocks = []
            for solution, score in solution_bank:
                block = (
                    f'### Past Example\n'
                    f'Solution:\n{solution.strip()}\n'
                    f'{self.task.metric}: {score}'
                )
                example_blocks.append(block)

            example_str = '\n\n'.join(example_blocks)

            meta_prompt = (
                task_prompt
                + f'\nHere are some past examples and the {self.task.metric}'
                f'score they received where the goal is to:'
                f'{self.task.direction} the metric\n\n{example_str}\n\n'
                f'Generate a solution that has as high a score as possible.'
            )

        return meta_prompt

    def prompt_lm(self, prompt: str) -> str:
        """Standard LLM api inference call."""
        response = self.client.chat.completions.create(
            model='gemini-3.5-flash',
            # TODO(MS): maybe pass in a pydantic model for the type of
            # response we want...particularly for code
            # response_format={"type": "json_object"},
            # TODO(MS): maybe do somehitng about system prompt??
            messages=[
                {
                    'role': 'system',
                    'content': 'You are a helpful assistant.',
                },
                {
                    'role': 'user',
                    'content': prompt,
                },
            ],
        )
        # Assuming 'response' is your completed OpenAI API call
        raw_output = response.choices[0].message.content
        if raw_output is not None:
            return raw_output
        else:
            # Handle the error state
            raise ValueError(
                'LM api call returned None instead of a valid string.',
            )
        return raw_output

    def optimize(self) -> None:
        """Optimization loop for task."""
        # TODO(MS): impl convergence criteria

        num_iter = 5
        for _i in range(num_iter):
            meta_prompt = self.get_meta_prompt()
            solution = self.prompt_lm(meta_prompt)
            score = self.task.evaluate(solution)
            self.solution_bank.add_solution_score_pair(solution, score)
            print(meta_prompt)
