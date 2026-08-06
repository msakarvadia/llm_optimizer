"""Base LLM Mutator Class."""

from __future__ import annotations

from typing import Any

import httpx
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

        # NOTE(MS): work around global proxies
        # specifically for compute nodes
        # Determine if this specific base_url targets your local hardware
        is_local = any(
            addr in base_url
            for addr in ['localhost', '127.0.0.1', '0.0.0.0', 'alcf.anl.gov']
        )
        print(f'DEBUG: base_url={base_url} | is_local={is_local}')

        if is_local:
            # FORCE bypass: Tell httpx to ignore ALL system
            # proxy variables entirely
            custom_http_client = httpx.Client(trust_env=False)
            print(
                '--> Local routing: '
                'Cluster environment proxy bypassed successfully.',
            )
        else:
            # FORCE use: Tell httpx to respect the system
            # proxy so it can reach ANL / Google
            custom_http_client = httpx.Client(trust_env=True)
            print(
                '--> Remote routing: '
                'Utilizing global cluster proxy for external connection.',
            )

        # TODO(MS): make generalizable to other base_urls
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=custom_http_client,  # WORK AROUND FOR GLOBAL PROXIES
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
    ) -> tuple[str, dict[str, int]]:
        """Single LLM-based Mutation of parent solutions."""
        meta_prompt = self.get_meta_prompt(past_solutions, task)
        print(meta_prompt)
        solution, token_usage = prompt_lm(
            self.client,
            meta_prompt,
            model_name=self.model_name,
            return_usage=True,
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
