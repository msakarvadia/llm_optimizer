"""Base LLM Mutator Class."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from llm_optimizer.optimizers.llm_mutator_library.base_llm_mutator import (
    Mutator,
)


class KInContext(Mutator):
    """Provide K in-context examples to mutate."""

    @abstractmethod
    def get_meta_prompt(
        self,
        past_solutions: list[tuple[Any, Any, Any]],
    ) -> str:
        """Prompt to guide the LLM mutation step."""
        pass
