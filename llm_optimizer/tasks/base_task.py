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
        self.task_description = None
        self.solution_description = None
        self.metric = metric
        self.direction = direction

    @abstractmethod
    def evaluate(self, solution: str) -> Any:
        """Custom evaluation logic to 'score' solution for task."""
        pass
