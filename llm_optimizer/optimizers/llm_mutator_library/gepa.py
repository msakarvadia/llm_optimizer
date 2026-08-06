"""Base LLM Mutator Class."""

from __future__ import annotations

from typing import Any

from llm_optimizer.optimizers.llm_mutator_library.base_llm_mutator import (
    Mutator,
)
from llm_optimizer.optimizers.llm_mutator_library.base_llm_mutator import (
    sum_token_usage,
)
from llm_optimizer.tasks.base_task import Task
from llm_optimizer.utils import prompt_lm


class GEPAMutator(Mutator):
    """GEPA mutator.

    # High-level design principles from: https://github.com/gepa-ai/gepa

    # Notably: GEPA has a "summarize why this solution works" step
    # here, any child program that beats its parents is "analyzed"
    # for trends, and those trends are appended to a long running
    # list of "lessons" learned. Here we omit that since evaluation
    # is distinct from mutation
    """

    def mutate(
        self,
        past_solutions: list[tuple[Any, Any, Any]],
        task: Task,
    ) -> tuple[str, dict[str, int]]:
        """Single LLM-based Mutation of parent solutions."""
        first_prompt = self.get_first_prompt(past_solutions, task)
        first_response, first_usage = prompt_lm(
            self.client,
            first_prompt,
            self.model_name,
            return_usage=True,
        )
        second_prompt = self.get_second_prompt(
            past_solutions,
            task,
            first_response,
        )
        second_response, second_usage = prompt_lm(
            self.client,
            second_prompt,
            self.model_name,
            return_usage=True,
        )
        return second_response, sum_token_usage(first_usage, second_usage)

    def get_second_prompt(
        self,
        past_solutions: list[tuple[Any, Any, Any]],
        task: Task,
        step_one_output: str,
    ) -> str:
        """Prompt to guide the LLM mutation step."""
        task_prompt = (
            f'{task.task_description} '
            f'Your goal is to {task.direction} {task.metric}. '
            f'Output only the bare minimum text to reach the objective goal.'
        )

        gepa_instructions = f"""\n
        You are an elite engineering and optimization algorithm. Your goal is to mutate an existing solutions to maximize its performance across a series of target objectives.

You must improve upon the ancestor solution by applying high-level lessons learned from failure data. If the ancestor is a placeholder, replace it with a relevent solution.

[CURRENT ANCESTOR PROMPT]
{past_solutions[-1][0]}

[NEW SYSTEM DIAGNOSIS]
{step_one_output}

CRITICAL RULES FOR MUTATION:
1. Synthesize the new system diagnosis with the accumulated historical lessons.
2. Mutate the instructions to directly prevent the errors noted in the diagnosis while preserving the parts of the solution that successfully handled past tasks.
3. Optimize for clarity, conciseness, and structural robustness.
4. Output ONLY the final mutated solution text. Do not include introductory conversational filler or concluding explanations.
"""
        meta_prompt = task_prompt + gepa_instructions

        print(meta_prompt)
        return meta_prompt

    def get_first_prompt(
        self,
        past_solutions: list[tuple[Any, Any, Any]],
        task: Task,
    ) -> str:
        """Prompt to guide the LLM mutation step."""
        task_prompt = (
            f'{task.task_description} '
            f'Your goal is to {task.direction} {task.metric}. '
            f'Output only the bare minimum text to reach the objective goal.'
            f''
        )

        gepa_instructions = f"""\n

You are an expert AI system diagnostics agent. Your task is to analyze the behavior of an LLM agent and determine exactly why its current instructions are causing it to fail.

Below is the current solution being utilized by the agent:
[CURRENT PROMPT]
{past_solutions[-1][0]}

Below is a minibatch of execution traces where the agent may have failed to achieve the target evaluation metric. Sometimes there are no failed trajectories to review. Review the step-by-step reasoning, tool calls, and outputs:
[(Potential) FAILED TRAJECTORIES]
{past_solutions[-1][2]}

INSTRUCTIONS:
1. Conduct a rigorous error analysis. Identify patterns or assumptions in the [CURRENT PROMPT] that misled the agent.
2. Pinpoint exactly which instructions caused the wrong reasoning steps or incorrect tool invocations.
3. Provide a clear, high-level natural language diagnosis detailing what needs to be changed, added, or removed from the solution to fix these specific edge cases. Do not rewrite the solution yet; only provide the structural diagnosis.
"""
        meta_prompt = task_prompt + gepa_instructions

        print(meta_prompt)
        return meta_prompt
