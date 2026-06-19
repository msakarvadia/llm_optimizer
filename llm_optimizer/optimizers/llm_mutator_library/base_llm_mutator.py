"""Base LLM Mutator Class."""

from __future__ import annotations

import os
from abc import ABC
from abc import abstractmethod
from typing import Any

from openai import OpenAI

from llm_optimizer.tasks.base_task import Task
from llm_optimizer.utils import prompt_lm


class Mutator(ABC):
    """Base class for LLM mutator.

    specialized mutators should inherit from this
    only change the get_meta_prompt method
    """

    def __init__(self) -> None:
        """Initialize the LLM."""
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

    @abstractmethod
    def get_meta_prompt(
        self,
        past_solutions: list[tuple[Any, Any, Any]],
        task: Task,
    ) -> str:
        """Prompt to guide the LLM mutation step."""
        pass

    def mutate(
        self,
        past_solutions: list[tuple[Any, Any, Any]],
        task: Task,
    ) -> str:
        """Single LLM-based Mutation of parent solutions."""
        meta_prompt = self.get_meta_prompt(past_solutions, task)
        print(meta_prompt)
        solution = prompt_lm(self.client, meta_prompt)
        return solution
