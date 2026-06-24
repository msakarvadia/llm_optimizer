"""Logic to abstract away LLM-driven optimization."""

from __future__ import annotations

import random
from typing import Any

from gepa.optimize_anything import EngineConfig
from gepa.optimize_anything import GEPAConfig
from gepa.optimize_anything import optimize_anything
from gepa.optimize_anything import ReflectionConfig

from llm_optimizer.optimizers.base_optimizer import Optimizer
from llm_optimizer.tasks.base_task import Task


class GEPAOptimizer(Optimizer):
    """LLM optimizer.

    # https://arxiv.org/abs/2605.19633
    # https://github.com/gepa-ai/gepa
    state include: task, solution_score pairs, stopping criteria
    """

    def __init__(
        self,
        task: Task,
        noise: bool = False,
        num_past_sol: int = 5,
        num_parallel_search: int = 1,
        **kwargs: Any,
    ) -> None:
        """Init optimizer."""
        self.task = task

        # NOTE(MS): for litellm, need to add the 'gemini/' prefix
        self.LLM_MODEL = 'gemini/gemini-3.5-flash'

        # NOTE(MS): variables to manage in-context examples/rewards
        self.n = num_past_sol
        self.noise = noise
        self.num_parallel_search = num_parallel_search
        self.max_population_size = kwargs['max_population_size']

    def optimize(self, num_iter: int = 5) -> None:
        """Optimization loop for task."""
        # TODO(MS): impl convergence criteria

        from gepa.strategies.candidate_selector import (
            TopKParetoCandidateSelector,
        )

        # rng = np.random.default_rng(seed=42)
        rng = random.Random(42)

        self.config = GEPAConfig(
            engine=EngineConfig(
                # run_dir=log_dir,
                max_metric_calls=num_iter,  # TODO: check if this is valid
                # parallel=True,
                # max_workers=64,
                # cache_evaluation=True,
                # track_best_outputs=True,
                # NOTE(K) defaults to 5...need to change this?
                # https://github.com/gepa-ai/gepa/releases
                # https://github.com/gepa-ai/gepa/pull/246
                candidate_selection_strategy=TopKParetoCandidateSelector(
                    k=self.max_population_size,
                    rng=rng,
                ),
            ),
            reflection=ReflectionConfig(
                reflection_lm=self.LLM_MODEL,
            ),
        )

        task_prompt = (
            f'{self.task.task_description} '
            f'Your goal is to {self.task.direction} {self.task.metric}. '
            f'Output only the bare minimum text to reach the objective goal.'
        )

        # NOTE (THIS RETURNS A RESULT)
        optimize_anything(
            # TODO(MS): give a seed candidate to the task definition!!
            seed_candidate=self.task.seed_candidate,
            evaluator=self.task.evaluate,
            objective=task_prompt,
            config=self.config,
        )

        # TODO(MS): temporarily save solution bank
        # experiment_dir = 'temp_results'
        # self.solution_bank.save_to_json(experiment_dir)
