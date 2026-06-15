"""Abstraction to define a task (for which a solution will be produced)."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any


class Task(ABC):
    """Base class for 'task'.

    Specialized child tasks should inherit from this base class.
    """

    @abstractmethod
    def __init__(self, metric: str, direction: str, **kwargs: Any) -> None:
        """Initialize task."""
        self.task_description: str | None = None
        self.solution_description: str | None = None
        self.metric = metric
        self.direction = direction
        self.seed_candidate = '<placeholder for generated solution>'

    @abstractmethod
    def evaluate(self, solution: str) -> tuple[float, dict[str, Any]]:
        """Custom evaluation logic to 'score' solution for task.

        must return the primary score being optimized, and
        a dict with extra info (such as compilation errors etc.)
        """
        pass
