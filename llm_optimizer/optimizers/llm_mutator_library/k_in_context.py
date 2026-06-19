"""Base LLM Mutator Class."""

from __future__ import annotations

from typing import Any

from llm_optimizer.optimizers.llm_mutator_library.base_llm_mutator import (
    Mutator,
)
from llm_optimizer.tasks.base_task import Task


class KInContextMutator(Mutator):
    """Provide K in-context examples to mutate."""

    def get_meta_prompt(
        self,
        past_solutions: list[tuple[Any, Any, Any]],
        task: Task,
    ) -> str:
        """Prompt to guide the LLM mutation step."""
        task_prompt = (
            f'{task.task_description} '
            f'Your goal is to {task.direction} {task.metric}. '
            f'Output only the bare minimum text to reach the objective goal.'
        )

        meta_prompt = task_prompt

        if len(past_solutions) > 0:
            example_blocks = []
            for solution, score, extra_info in past_solutions:
                block = (
                    f'### Past Example\n'
                    f'Solution:\n{solution.strip()}\n'
                    f'{task.metric}: {score}\n'
                    f'Additional meta-data: {extra_info}'
                )
                example_blocks.append(block)

            example_str = '\n\n'.join(example_blocks)

            meta_prompt = (
                task_prompt
                + f'\nHere are some past examples and the {task.metric}'
                f'score they received where the goal is to '
                f'{task.direction} the metric\n\n{example_str}\n\n'
                f'Generate a new {task.solution_description} that is'
                f' different from the old ones to '
                f'{task.direction} the {task.metric}'
                f' as much as possible.'
            )

        return meta_prompt
