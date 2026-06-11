"""Abstraction to define a task (for which a solution will be produced)."""

from __future__ import annotations

import math
from typing import Any

from llm_optimizer.tasks.base_task import Task


class MaximizeFunction(Task):
    """Maximize function task'."""

    def __init__(
        self,
        metric: str = 'function value',
        direction: str = 'maximize',
        **kwargs: Any,
    ) -> None:
        """Initialize task."""
        self.task_description = (
            'There is a hidden 2d function is where'
            'the independent variable is x and the dependent variable is y.'
            ' Predict x such that you maximize the function.'
            " Don't commit too early to a point."
        )

        self.solution_description = 'x value'
        self.metric = metric
        self.direction = direction

    def evaluate(self, x: float | str) -> float | str:
        """Evaluate LLM optimized solution."""
        try:
            x = float(x)
            coeff = 0.5
            x = x - 0.78
            vertical_shift = 2
            solution = -1 * (x**4 - 3 * x**2 + coeff * x + vertical_shift)
            print('-' * 40)
            print(f'{solution=}')
            print('-' * 40)
            return solution
        except Exception:
            # TODO(MS): turn this into side channel info for opro??
            # return str(error)
            return -math.inf
