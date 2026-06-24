"""Base LLM Mutator Class."""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from llm_optimizer.tasks.base_task import Task
from llm_optimizer.utils import prompt_lm


class Mutator:
    """Base class for LLM mutator.

    specialized mutators should inherit from this
    only change the get_meta_prompt method
    """

    def __init__(self, api_key: str, base_url: str, model_name: str) -> None:
        """Initialize the LLM."""
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        if api_key is None:
            raise ValueError(
                'API key not found. Set the MY_API_KEY environment variable.',
            )

        # TODO(MS): make generalizable to other base_urls
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def get_meta_prompt(
        self,
        past_solutions: list[tuple[Any, Any, Any]],
        task: Task,
    ) -> str:
        """Prompt to guide the LLM mutation step."""
        return 'Place holder prompt'

    def mutate(
        self,
        past_solutions: list[tuple[Any, Any, Any]],
        task: Task,
    ) -> str:
        """Single LLM-based Mutation of parent solutions."""
        meta_prompt = self.get_meta_prompt(past_solutions, task)
        print(meta_prompt)
        solution = prompt_lm(
            self.client,
            meta_prompt,
            model_name=self.model_name,
        )
        return solution
