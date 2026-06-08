"""Abstraction to define a task (for which a solution will be produced)."""

from __future__ import annotations

from typing import Any

from llm_optimizer.tasks.base_task import Task


class MinimizeFunction(Task):
    """Tweet engagement 'task'."""

    def __init__(
        self,
        metric: str = 'function value',
        direction: str = 'minimize',
        **kwargs: Any,
    ) -> None:
        """Initialize task."""
        self.task_description = (
            'There is a hidden 2d function is where'
            'the independent variable is x and the dependent variable is y.'
            ' Predict x such that you minimize the function.'
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
            return x**4 - 3 * x**2 + coeff * x + vertical_shift
        except Exception as error:
            return str(error)
