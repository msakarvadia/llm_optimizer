"""Base LLM Mutator Class."""

from __future__ import annotations

import warnings
from typing import Any

from llm_optimizer.optimizers.llm_mutator_library.base_llm_mutator import (
    Mutator,
)
from llm_optimizer.optimizers.llm_mutator_library.k_in_context import (
    KInContextMutator,
)
from llm_optimizer.tasks.base_task import Task
from llm_optimizer.utils import prompt_lm


class GAMutator(Mutator):
    """Genetic Algorithm mutator.

    # https://github.com/beeevita/EvoPrompt/blob/main/data/template_ga.py#L2
    """

    def mutate(
        self,
        past_solutions: list[tuple[Any, Any, Any]],
        task: Task,
    ) -> str:
        """Single LLM-based Mutation of parent solutions."""
        if len(past_solutions) < 3:
            warnings.warn(
                'Past solution bank has less than 3 solutions, falling back on kincontext mutator.',
            )
            mutator = KInContextMutator(
                api_key=self.api_key,
                base_url=self.base_url,
                model_name=self.model_name,
            )
            meta_prompt = mutator.get_meta_prompt(past_solutions, task)
            solution = prompt_lm(self.client, meta_prompt, self.model_name)
            return solution

        if len(past_solutions) > 3:
            warnings.warn(
                'Past solution bank has more than 3 solutions, will be truncated to last 3.',
            )

        first_prompt = self.get_first_prompt(past_solutions, task)
        first_response = prompt_lm(self.client, first_prompt, self.model_name)
        second_prompt = self.get_second_prompt(
            past_solutions,
            task,
            first_response,
        )
        second_response = prompt_lm(
            self.client,
            second_prompt,
            self.model_name,
        )
        return second_response

    def get_second_prompt(
        self,
        past_solutions: list[tuple[Any, Any, Any]],
        task: Task,
        step_one_output: str,
    ) -> str:
        """Prompt to guide the LLM mutation step."""
        task_prompt = (
            f'{task.task_description} '
            f'Your goal is to {task.direction} {task.metric}. '
            f'Output only the bare minimum text to reach the objective goal.'
        )

        de_instructions = f"""\n
In the previous step, you were instructed to:
1. Identify the different parts between Solution 1 and Solution 2:
Solution 1:
{past_solutions[-3][0]}
Solution 2:
{past_solutions[-2][0]}
2. Randomly mutate the different parts
3. Crossover the different parts with the following Solution 3 and generate a final solution (Do not write anything except the final request solution):
Solution 3:
{past_solutions[-1][0]}

Here is what you wrote:
{step_one_output}

Extract the final solution that was derived after step three in the past step from in between <solution> and </solution>. Output only that final solution.
"""
        meta_prompt = task_prompt + de_instructions

        print(meta_prompt)
        return meta_prompt

    def get_first_prompt(
        self,
        past_solutions: list[tuple[Any, Any, Any]],
        task: Task,
    ) -> str:
        """Prompt to guide the LLM mutation step."""
        task_prompt = (
            f'{task.task_description} '
            f'Your goal is to {task.direction} {task.metric}. '
            f'Output only the bare minimum text to reach the objective goal.'
            f''
        )

        ga_instructions = f"""\n
First we provide an example of evolving a prompt, this may be different from the task you are supposed to complete. Regardless, follow the template:
Please follow the instruction step-by-step to generate a better prompt.
1. Crossover the following prompts and generate a new prompt:
Prompt 1: Rewrite the input text into simpler text.
Prompt 2: Rewrite my complex sentence in simpler terms, but keep the meaning.
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <solution> and </solution>.

1. Crossover Prompt: Rewrite the complex text into simpler text while keeping its meaning.
2. <solution>Transform the provided text into simpler language, maintaining its essence.</solution>

Now it is your turn. Please follow the instruction step-by-step to generate a better solution.
Please follow the instruction step-by-step to generate a better solution.
1. Crossover the following solution and generate a new solution:
Solution 1:
{past_solutions[-2][0]}
Solution 1's meta data:
{past_solutions[-2][2]}
Solution 2:
{past_solutions[-1][0]}
Solution 2's meta data:
{past_solutions[-1][2]}
2. Mutate the solution generated in Step 1 and generate a final solution bracketed with <solution> and </solution>.

1.
"""
        meta_prompt = task_prompt + ga_instructions

        # print(meta_prompt)
        return meta_prompt
