"""Abstraction to define an llm-driven optimizer."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any

from llm_optimizer.tasks.base_task import Task


class Optimizer(ABC):
    """Base class for 'optimizer'.

    Specialized child tasks should inherit from this base class.
    """

    @abstractmethod
    def __init__(
        self,
        task: Task,
        noise: bool = False,
        num_past_sol: int = 5,
        num_parallel_search: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize optimizer.

        task: task to optimize solution for
            task comes with its custom evaluation logic
        noise: whether or not to noise the reward
            proxy for noise from the world
        num_past_sol: # of past generated solutions to keep
            in context/pereto-front/population
            proxy for adaptive history
        num_parallel_search: # of solutions to generate in parallel
            proxy for gradients (mini-batches) populate past solutions
            which new solutions are conditioned on
        """
        self.noise = noise
        self.num_past_sol = num_past_sol
        self.num_parallel_search = num_parallel_search
        self.task = task

    @abstractmethod
    def optimize(self, solution: str) -> Any:
        """Custom optimization logic to search for best soluiton.

        Should save history of solution/score pairs
        """
        pass
