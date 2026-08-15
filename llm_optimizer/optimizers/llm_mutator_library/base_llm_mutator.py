"""Base LLM Mutator Class."""

from __future__ import annotations

from typing import Any

from llm_optimizer.tasks.base_task import Task
from llm_optimizer.utils import build_openai_client
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
        self.client = build_openai_client(api_key, base_url)

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
    ) -> tuple[str, dict[str, int]]:
        """Single LLM-based Mutation of parent solutions."""
        meta_prompt = self.get_meta_prompt(past_solutions, task)
        print(meta_prompt)
        solution, token_usage = prompt_lm(
            self.client,
            meta_prompt,
            model_name=self.model_name,
        )
        return solution, token_usage


def sum_token_usage(*usages: dict[str, int]) -> dict[str, int]:
    """Sum multiple prompt_lm token usage dicts (same keys) into one.

    Used by multi-call mutators (DE/GA/GEPA) to report total tokens
    spent across all of the LLM calls that went into a single mutation.
    """
    total: dict[str, int] = {}
    for usage in usages:
        for key, value in usage.items():
            total[key] = total.get(key, 0) + value
    return total
