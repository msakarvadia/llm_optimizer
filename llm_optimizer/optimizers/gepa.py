"""Logic to abstract away LLM-driven optimization."""

from __future__ import annotations

import os
import random
from typing import Any

from gepa.optimize_anything import EngineConfig
from gepa.optimize_anything import GEPAConfig
from gepa.optimize_anything import optimize_anything
from gepa.optimize_anything import ReflectionConfig
from gepa.strategies.candidate_selector import TopKParetoCandidateSelector

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

        # build experiment path (also used by gepa as its checkpoint dir --
        # see optimize(): EngineConfig(run_dir=...). gepa auto-resumes from
        # here if a gepa_state.bin checkpoint already exists.
        self.experiment_dir = f'{kwargs["experiment_dir"]}/'
        print(f'{self.experiment_dir=}')
        os.makedirs(self.experiment_dir, exist_ok=True)

        # NOTE(MS): for litellm, need to add the 'gemini/' prefix
        os.environ['OPENAI_API_BASE'] = kwargs['base_url']
        os.environ['OPENAI_API_KEY'] = kwargs['api_key']
        self.LLM_MODEL = f'openai/{kwargs["model_name"]}'

        # NOTE(MS): variables to manage in-context examples/rewards
        self.n = num_past_sol
        self.noise = noise
        self.num_parallel_search = num_parallel_search
        self.max_population_size = kwargs['max_population_size']

    def _gepa_evaluator(self, candidate: str) -> tuple[float, dict[str, Any]]:
        """Adapt Task.evaluate's 3-tuple return to GEPA's (score, side_info).

        Task.evaluate returns (score, extra_info, val_score), but GEPA's
        evaluator protocol expects either a bare score or a
        (score, side_info) pair. Unpacking the raw 3-tuple as a 2-tuple
        inside gepa's EvaluatorWrapper raises a ValueError, so we adapt
        here instead of handing task.evaluate to gepa directly.

        NOTE(MS): val_score is intentionally NOT included in side_info.
        It's the held-out/test score, not training feedback -- exposing
        it to the reflection LM would let optimization "see" the test
        signal it's meant to generalize to, i.e. test-set hacking.
        """
        score, extra_info, _val_score = self.task.evaluate(candidate)
        return score, dict(extra_info)

    def optimize(self, num_iter: int = 5) -> None:
        """Optimization loop for task."""
        # TODO(MS): impl convergence criteria

        # rng = np.random.default_rng(seed=42)
        rng = random.Random(42)

        self.config = GEPAConfig(
            engine=EngineConfig(
                # NOTE(MS): setting run_dir gives us checkpoint/resume for
                # free -- gepa auto-loads it on the next call if it's
                # already present.
                run_dir=self.experiment_dir,
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
            evaluator=self._gepa_evaluator,
            objective=task_prompt,
            config=self.config,
        )

        # TODO(MS): temporarily save solution bank
        # experiment_dir = 'temp_results'
        # self.solution_bank.save_to_json(experiment_dir)
